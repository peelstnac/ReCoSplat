import torch
from einops import rearrange
from gsplat import rasterization
from jaxtyping import Float
from torch import Tensor, nn

from ..adapter.types import AccumulatingGaussians

SH_C0 = 0.2820947917738781


class GuidanceRenderer:
    def __init__(
        self,
        feature_dim: int,
        first_chunk_guidance_embed: nn.Parameter | None,
        opacity_prune_threshold: float = 0.005,
    ) -> None:
        self.feature_dim = feature_dim
        self.first_chunk_guidance_embed = first_chunk_guidance_embed
        self.opacity_prune_threshold = opacity_prune_threshold

    def render(
        self,
        acc_gs: AccumulatingGaussians,
        guidance_poses: Float[Tensor, "b v 4 4"],
        intrinsics: Float[Tensor, "b v 3 3"],
        image_shape: tuple[int, int],
    ) -> Float[Tensor, "b v c h w"]:
        batch, views = guidance_poses.shape[:2]
        height, width = image_shape
        channels = 3 + self.feature_dim
        if acc_gs.end == 0:
            if self.first_chunk_guidance_embed is None:
                raise RuntimeError("missing first-chunk guidance embedding")
            return self.first_chunk_guidance_embed.view(1, 1, channels, 1, 1).expand(
                batch, views, channels, height, width
            ).contiguous()
        if batch != 1:
            raise ValueError("guidance rendering requires batch size 1")

        gaussians = acc_gs.gaussians
        harmonics = gaussians.harmonics.detach()
        extra_features = acc_gs.guidance_features
        if extra_features is None:
            colors = rearrange(harmonics, "b n c k -> b n k c")
            sh_degree = 0
        else:
            rgb = (harmonics[..., 0] * SH_C0 + 0.5).clamp_min(0.0)
            colors = torch.cat([rgb, extra_features], dim=-1)
            sh_degree = None

        intrinsics_denorm = intrinsics.detach().clone()
        intrinsics_denorm[:, :, 0] *= width
        intrinsics_denorm[:, :, 1] *= height
        opacities = gaussians.opacities.detach()
        keep = (opacities > self.opacity_prune_threshold).squeeze(0)
        rendering, _, _ = rasterization(
            gaussians.means.detach()[:, keep],
            None,
            None,
            opacities[:, keep],
            colors[:, keep],
            guidance_poses.float().inverse().detach(),
            intrinsics_denorm,
            width,
            height,
            sh_degree=sh_degree,
            render_mode="RGB",
            backgrounds=None,
            radius_clip=0.1,
            covars=gaussians.covariances.detach()[:, keep],
            rasterize_mode="classic",
        )
        return rearrange(rendering, "b v h w c -> b v c h w").contiguous()
