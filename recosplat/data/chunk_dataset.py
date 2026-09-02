"""Finite evaluation loader for encoded scene chunks."""

import re
from dataclasses import dataclass
from pathlib import Path

import torch
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from torch import Tensor
from torch.utils.data import IterableDataset
from torchvision.io import decode_image

from .camera_utils import camera_normalization, get_fov
from .collate import collate_examples
from .shims.crop_shim import apply_crop_shim
from .types import BatchedExample, UnbatchedExample
from .view_sampler_evaluation import ViewSamplerEvaluation

_HEX64 = re.compile(r"[0-9a-f]{64}")


def scene_hash(key: str) -> str:
    """Normalize dataset-specific prefixes around a 64-character scene hash."""
    match = _HEX64.search(key)
    return match.group(0) if match is not None else key


def convert_poses(
    poses: Float[Tensor, "view 18"],
) -> tuple[Float[Tensor, "view 4 4"], Float[Tensor, "view 3 3"]]:
    """Decode normalized intrinsics and OpenCV c2w extrinsics."""
    num_views = poses.shape[0]
    intrinsics = repeat(torch.eye(3, dtype=torch.float32), "h w -> v h w", v=num_views).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> v h w", v=num_views).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "v (h w) -> v h w", h=3, w=4)
    return w2c.inverse(), intrinsics


@dataclass
class ChunkDatasetCfg:
    name: str
    roots: list[Path]
    image_shape: tuple[int, int]
    original_image_shape: tuple[int, int]
    relative_pose: bool
    skip_bad_shape: bool
    pose_norm_method: str
    baseline_min: float
    baseline_max: float
    max_fov: float
    near: float = 0.1
    far: float = 100.0


class ChunkDataset(IterableDataset):
    """One deterministic B=1 batch per scene present in an evaluation index."""

    def __init__(self, cfg: ChunkDatasetCfg, sampler: ViewSamplerEvaluation) -> None:
        super().__init__()
        if cfg.pose_norm_method not in {"max_1", "none"}:
            raise ValueError(f"unsupported pose_norm_method: {cfg.pose_norm_method}")
        self.cfg = cfg
        self.sampler = sampler
        self.chunks: list[Path] = []
        for root in cfg.roots:
            test_root = root / "test"
            if not test_root.is_dir():
                raise FileNotFoundError(
                    f"dataset root must contain test/*.torch chunks: {test_root}"
                )
            self.chunks.extend(sorted(test_root.glob("*.torch")))
        if not self.chunks:
            raise FileNotFoundError(f"no test chunks found below: {cfg.roots}")

    def _my_chunks(self) -> list[Path]:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        else:
            rank, world = 0, 1
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        shard_id = rank * num_workers + worker_id
        chunks = self.chunks[shard_id :: world * num_workers]
        if not chunks:
            raise RuntimeError(
                f"empty chunk shard {shard_id}/{world * num_workers}; "
                f"only {len(self.chunks)} chunks are available"
            )
        return chunks

    def __iter__(self):
        for chunk_path in self._my_chunks():
            chunk = torch.load(chunk_path, weights_only=True, mmap=True)
            for raw in chunk:
                indices = self.sampler.sample(raw["key"])
                if indices is None:
                    continue
                context_indices, target_indices = indices
                example = self._build_example(raw, context_indices, target_indices)
                if example is None:
                    continue
                batch: BatchedExample = collate_examples([example])
                batch["params"] = {
                    "num_context_views": len(context_indices),
                    "chunk_size_f": self.sampler.cfg.chunk_size_f,
                    "chunk_size_s": self.sampler.cfg.chunk_size_s,
                    "batch_size": 1,
                    "step": 0,
                    "height": self.cfg.image_shape[0],
                    "width": self.cfg.image_shape[1],
                    "num_target_views": len(target_indices),
                }
                yield batch

    @staticmethod
    def _convert_images(images: list[UInt8[Tensor, " _"]]) -> Float[Tensor, "view 3 h w"]:
        return torch.stack([decode_image(image) for image in images]).float() / 255.0

    def _build_example(
        self,
        raw: dict,
        context_indices: Tensor,
        target_indices: Tensor,
    ) -> UnbatchedExample | None:
        extrinsics, intrinsics = convert_poses(raw["cameras"])
        if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
            return None
        try:
            context_images = self._convert_images(
                [raw["images"][int(index)] for index in context_indices]
            )
            target_images = self._convert_images(
                [raw["images"][int(index)] for index in target_indices]
            )
        except (IndexError, OSError, RuntimeError):
            return None

        expected = (3, *self.cfg.original_image_shape)
        if self.cfg.skip_bad_shape and (
            context_images.shape[1:] != expected or target_images.shape[1:] != expected
        ):
            return None

        context_extrinsics = extrinsics[context_indices]
        used_extrinsics = torch.cat(
            [context_extrinsics, extrinsics[target_indices]], dim=0
        )
        if self.cfg.pose_norm_method == "max_1":
            scale = torch.pdist(context_extrinsics[:, :3, 3]).max()
        else:
            scale = torch.tensor(1.0)
        if not torch.isfinite(scale) or scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
            return None
        used_extrinsics[:, :3, 3] /= scale
        if self.cfg.relative_pose:
            used_extrinsics = camera_normalization(used_extrinsics[:1], used_extrinsics)

        num_context = len(context_indices)
        near = torch.full((num_context,), self.cfg.near / scale, dtype=torch.float32)
        far = torch.full((num_context,), self.cfg.far / scale, dtype=torch.float32)
        target_near = torch.full(
            (len(target_indices),), self.cfg.near / scale, dtype=torch.float32
        )
        target_far = torch.full(
            (len(target_indices),), self.cfg.far / scale, dtype=torch.float32
        )
        example: UnbatchedExample = {
            "context": {
                "extrinsics": used_extrinsics[:num_context],
                "intrinsics": intrinsics[context_indices],
                "image": context_images,
                "near": near,
                "far": far,
                "index": context_indices,
            },
            "target": {
                "extrinsics": used_extrinsics[num_context:],
                "intrinsics": intrinsics[target_indices],
                "image": target_images,
                "near": target_near,
                "far": target_far,
                "index": target_indices,
            },
            "scene": raw["key"],
        }
        return apply_crop_shim(example, self.cfg.image_shape)
