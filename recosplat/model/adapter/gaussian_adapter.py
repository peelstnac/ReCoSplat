from dataclasses import dataclass

import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from torch import Tensor, nn

from ..geometry import homogenize_points
from .gaussians import build_covariance


@dataclass
class AdapterOutput:
    means: Float[Tensor, "*batch 3"]
    covariances: Float[Tensor, "*batch 3 3"]
    scales: Float[Tensor, "*batch 3"]
    rotations: Float[Tensor, "*batch 4"]
    harmonics: Float[Tensor, "*batch 3 1"]
    opacities: Float[Tensor, " *batch"]


@dataclass
class GaussianAdapterCfg:
    gaussian_scale_min: float
    gaussian_scale_max: float


class UnifiedGaussianAdapter(nn.Module):
    d_opacity_in = 1
    d_in = 10

    def __init__(self, cfg: GaussianAdapterCfg) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        means: Float[Tensor, "*#batch 3"],
        depths: Float[Tensor, "*#batch"],
        opacities: Float[Tensor, "*#batch"],
        raw_gaussians: Float[Tensor, "*#batch 10"],
        eps: float = 1e-8,
        extrinsics: Float[Tensor, "*#batch 4 4"] | None = None,
    ) -> AdapterOutput:
        del depths
        scales, rotations, colors = raw_gaussians.split((3, 4, 3), dim=-1)
        scales = (0.001 * F.softplus(scales)).clamp_max(0.3)
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + eps)
        harmonics = rearrange(colors, "... color -> ... color 1", color=3)
        harmonics = harmonics.broadcast_to((*opacities.shape, 3, 1))
        covariances = build_covariance(scales, rotations)

        if extrinsics is not None:
            rotation = extrinsics[..., :3, :3]
            covariances = rotation @ covariances @ rotation.transpose(-1, -2)
            means = einsum(
                extrinsics,
                homogenize_points(means),
                "... i j, ... j -> ... i",
            )[..., :3]

        return AdapterOutput(
            means=means,
            covariances=covariances,
            scales=scales,
            rotations=rotations.broadcast_to((*scales.shape[:-1], 4)),
            harmonics=harmonics,
            opacities=opacities,
        )
