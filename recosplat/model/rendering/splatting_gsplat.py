import logging

import torch
from gsplat import rasterization
from jaxtyping import Float
from torch import Tensor, nn

from ..adapter.types import Gaussians
from .types import DecoderOutput, DecoderSplattingSplatCfg, DepthRenderingMode

__all__ = [
    "DecoderOutput",
    "DecoderSplattingSplatCfg",
    "DecoderSplattingGSPlat",
    "DepthRenderingMode",
    "prune_gaussians",
]


def prune_gaussians(gaussians: Gaussians, opacity_threshold: float) -> Gaussians:
    if opacity_threshold <= 0:
        return gaussians
    if gaussians.means.shape[0] != 1:
        raise ValueError("opacity pruning requires batch size 1")
    mask = gaussians.opacities > opacity_threshold
    num_gaussians = gaussians.means.shape[1]
    num_pruned = num_gaussians - int(mask.sum())
    logging.getLogger(__name__).debug(
        "Pruned %d of %d Gaussians", num_pruned, num_gaussians
    )

    def trim(value: Tensor | None) -> Tensor | None:
        return None if value is None else value[mask].unsqueeze(0)

    return Gaussians(
        means=trim(gaussians.means),
        covariances=trim(gaussians.covariances),
        harmonics=trim(gaussians.harmonics),
        opacities=trim(gaussians.opacities),
        rotations=trim(gaussians.rotations),
        scales=trim(gaussians.scales),
        features=trim(gaussians.features),
    )


class DecoderSplattingGSPlat(nn.Module):
    def __init__(self, cfg: DecoderSplattingSplatCfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.make_scale_invariant = cfg.make_scale_invariant
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ) -> DecoderOutput:
        del near, far, depth_mode, cam_rot_delta, cam_trans_delta
        gaussians = prune_gaussians(gaussians, self.cfg.prune_opacity_threshold)
        batch, views = intrinsics.shape[:2]
        height, width = image_shape
        features = gaussians.harmonics.permute(0, 1, 3, 2).contiguous()
        w2c = extrinsics.float().inverse()
        intrinsics_denorm = intrinsics.clone()
        intrinsics_denorm[:, :, 0] *= width
        intrinsics_denorm[:, :, 1] *= height
        backgrounds = self.background_color.view(1, 1, 3).repeat(batch, views, 1)

        rendering, _, _ = rasterization(
            gaussians.means,
            gaussians.rotations,
            gaussians.scales,
            gaussians.opacities,
            features,
            w2c,
            intrinsics_denorm,
            width,
            height,
            sh_degree=0,
            render_mode="RGB+D",
            packed=False,
            backgrounds=backgrounds,
            radius_clip=0.1,
            covars=gaussians.covariances,
            rasterize_mode="classic",
        )
        color, depth = torch.split(rendering, [3, 1], dim=-1)
        return DecoderOutput(
            color.clamp(0.0, 1.0).permute(0, 1, 4, 2, 3),
            depth.squeeze(-1),
        )
