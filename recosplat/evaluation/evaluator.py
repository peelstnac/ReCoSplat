"""Streaming evaluation for image and camera-pose metrics."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import DataLoader

from ..data.chunk_dataset import ChunkDataset, ChunkDatasetCfg
from ..data.view_sampler_evaluation import ViewSamplerEvaluation, ViewSamplerEvaluationCfg
from ..model.geometry import calculate_camera_poses_scale
from .metrics import compute_lpips, compute_psnr, compute_ssim
from .pose_metrics import (
    accuracy_at,
    compute_ate,
    compute_rra_rta_errors,
    scene_max_pose_error,
)
from .pose_opt import PoseOptCfg, optimize_poses

if TYPE_CHECKING:
    from ..model.encoder import ReCoSplatEncoder
    from ..model.rendering.splatting_cuda import DecoderSplattingCUDA
    from ..model.rendering.splatting_gsplat import DecoderSplattingGSPlat

log = logging.getLogger(__name__)

PoseMode = Literal["gt", "pred_optimized"]


@dataclass
class EvalCfg:
    tag: str
    view_sampler: ViewSamplerEvaluationCfg
    pose_modes: list[PoseMode]
    pose_opt: PoseOptCfg
    render_chunk_size: int
    compute_image_metrics: bool
    compute_pose_metrics: bool
    num_scenes: int | None
    num_workers: int


@dataclass
class SceneResult:
    scene: str
    metrics: dict[str, float]


class Evaluator:
    """Run one evaluation configuration over a test split."""

    def __init__(
        self,
        cfg: EvalCfg,
        dataset_cfg: ChunkDatasetCfg,
        encoder: "ReCoSplatEncoder",
        decoder: "DecoderSplattingCUDA | DecoderSplattingGSPlat",
    ) -> None:
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.encoder = encoder
        self.decoder = decoder

    def _build_view_sampler(self) -> ViewSamplerEvaluation:
        return ViewSamplerEvaluation(self.cfg.view_sampler)

    def _dataloader(self) -> DataLoader:
        dataset = ChunkDataset(self.dataset_cfg, self._build_view_sampler())
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    @staticmethod
    def _to_device(views: dict, device: torch.device) -> dict:
        return {k: v.to(device) if isinstance(v, Tensor) else v for k, v in views.items()}

    def _render_targets(self, gaussians, target: dict, extrinsics: Tensor) -> Tensor:
        """Render target views in bounded-size chunks."""
        _, v_tgt, _, h, w = target["image"].shape
        colors = []
        for start in range(0, v_tgt, self.cfg.render_chunk_size):
            end = min(start + self.cfg.render_chunk_size, v_tgt)
            output = self.decoder.forward(
                gaussians,
                extrinsics[:, start:end],
                target["intrinsics"][:, start:end],
                target["near"][:, start:end],
                target["far"][:, start:end],
                (h, w),
            )
            colors.append(output.color)
        return torch.cat(colors, dim=1)

    def _target_extrinsics_for_mode(
        self,
        mode: PoseMode,
        batch: dict,
        gaussians,
        scale_ratio: Tensor,
    ) -> Tensor:
        """Return scaled target poses, with optional photometric refinement."""
        scaled = batch["target"]["extrinsics"].clone()
        scaled[:, :, :3, 3] *= scale_ratio[:, None, None]
        if mode == "gt":
            return scaled
        return optimize_poses(
            self.decoder,
            gaussians,
            batch["target"],
            scaled,
            self.cfg.pose_opt,
        )

    def evaluate_scene(self, batch: dict) -> SceneResult:
        params = batch["params"]
        device = next(self.encoder.parameters()).device
        batch = {
            "context": self._to_device(batch["context"], device),
            "target": self._to_device(batch["target"], device),
            "scene": batch["scene"],
            "params": params,
        }

        with torch.no_grad():
            encoder_output = self.encoder.forward_streaming(
                batch["context"],
                0,
                chunk_size_f=params["chunk_size_f"],
                chunk_size_s=params["chunk_size_s"],
            )

            if encoder_output.use_gt_pose:
                scale_ratio = encoder_output.gt_scale_factor
            else:
                scale_ratio = calculate_camera_poses_scale(
                    encoder_output.pred_camera_poses
                ) / calculate_camera_poses_scale(batch["context"]["extrinsics"])

        gaussians = encoder_output.gaussians
        metrics: dict[str, float] = {}

        rgb_gt = batch["target"]["image"]
        if self.cfg.compute_image_metrics:
            for i, mode in enumerate(self.cfg.pose_modes):
                extrinsics = self._target_extrinsics_for_mode(
                    mode, batch, gaussians, scale_ratio
                )
                with torch.no_grad():
                    color = self._render_targets(gaussians, batch["target"], extrinsics)
                flat_gt = rearrange(rgb_gt, "b v c h w -> (b v) c h w")
                flat_pred = rearrange(color, "b v c h w -> (b v) c h w")
                mode_metrics = {
                    f"psnr_{mode}": compute_psnr(flat_gt, flat_pred).mean().item(),
                    f"ssim_{mode}": compute_ssim(flat_gt, flat_pred).mean().item(),
                    f"lpips_{mode}": compute_lpips(flat_gt, flat_pred).mean().item(),
                }
                metrics.update(mode_metrics)
                if i == 0:
                    for key, value in mode_metrics.items():
                        metrics[key.removesuffix(f"_{mode}")] = value

        if self.cfg.compute_pose_metrics:
            err_r, err_t = compute_rra_rta_errors(
                encoder_output.pred_camera_poses[0],
                batch["context"]["extrinsics"][0],
            )
            metrics["rra@5"] = accuracy_at(err_r, 5.0)
            metrics["rta@5"] = accuracy_at(err_t, 5.0)
            metrics["rra_err_deg"] = err_r.mean().item()
            metrics["rta_err_deg"] = err_t.mean().item()
            metrics["ate"] = compute_ate(
                encoder_output.pred_camera_poses[0],
                batch["context"]["extrinsics"][0],
            )
            metrics["pose_err_deg"] = scene_max_pose_error(
                encoder_output.pred_camera_poses[0],
                batch["context"]["extrinsics"][0],
            )

        return SceneResult(scene=batch["scene"][0], metrics=metrics)

    def run(self) -> tuple[dict[str, float], list[SceneResult]]:
        """Evaluate every selected scene and return aggregate and per-scene metrics."""
        self.encoder.eval()
        self.decoder.eval()

        results: list[SceneResult] = []
        for batch in self._dataloader():
            if self.cfg.num_scenes is not None and len(results) >= self.cfg.num_scenes:
                break
            result = self.evaluate_scene(batch)
            results.append(result)
            headline = (
                f"psnr {result.metrics['psnr']:.2f}"
                if "psnr" in result.metrics
                else f"pose error {result.metrics['pose_err_deg']:.2f} deg"
            )
            log.info(
                "[eval %s] scene %s (%d): %s",
                self.cfg.tag,
                result.scene[:16],
                len(results),
                headline,
            )
            torch.cuda.empty_cache()

        if not results:
            raise RuntimeError(
                f"eval {self.cfg.tag}: no indexed scene found in the test split "
                f"({self.cfg.view_sampler.index_path})"
            )

        from .report import aggregate_metrics

        return aggregate_metrics(results), results
