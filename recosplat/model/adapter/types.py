from dataclasses import dataclass

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


@dataclass
class Gaussians:
    means: Float[Tensor, "batch gaussian 3"]
    covariances: Float[Tensor, "batch gaussian 3 3"]
    harmonics: Float[Tensor, "batch gaussian 3 1"]
    opacities: Float[Tensor, "batch gaussian"]
    rotations: Float[Tensor, "batch gaussian 4"]
    scales: Float[Tensor, "batch gaussian 3"]
    features: Float[Tensor, "batch gaussian feature"] | None = None


class AccumulatingGaussians(nn.Module):
    def __init__(
        self,
        b: int,
        v: int,
        h: int,
        w: int,
        feature_dim: int = 0,
        device: str | torch.device = "cuda",
    ) -> None:
        super().__init__()
        self.b, self.v, self.h, self.w = b, v, h, w
        self.end = 0
        count = v * h * w
        device = torch.device(device)
        self.means = torch.empty((b, count, 3), device=device)
        self.covariances = torch.empty((b, count, 3, 3), device=device)
        self.harmonics = torch.empty((b, count, 3, 1), device=device)
        self.opacities = torch.empty((b, count), device=device)
        self.rotations = torch.zeros((b, 1, 4), device=device)
        self.scales = torch.zeros((b, 1, 3), device=device)
        self.features = (
            torch.empty((b, count, feature_dim), device=device)
            if feature_dim > 0
            else None
        )

    def add_adapter_output(self, gaussians, features: Tensor | None = None) -> None:
        num_views = gaussians.means.size(1)
        end = self.end + num_views * self.h * self.w
        means = rearrange(
            gaussians.means, "b v r srf spp xyz -> b (v r srf spp) xyz"
        )
        if self.end + means.size(1) != end:
            raise ValueError("Gaussian grid size does not match the accumulator")
        self.means[:, self.end : end] = means
        self.covariances[:, self.end : end] = rearrange(
            gaussians.covariances,
            "b v r srf spp i j -> b (v r srf spp) i j",
        )
        self.harmonics[:, self.end : end] = rearrange(
            gaussians.harmonics,
            "b v r srf spp c d -> b (v r srf spp) c d",
        )
        self.opacities[:, self.end : end] = rearrange(
            gaussians.opacities, "b v r srf spp -> b (v r srf spp)"
        )
        if features is not None:
            if self.features is None:
                raise ValueError("feature storage was not initialized")
            self.features[:, self.end : end] = rearrange(
                features, "b v r srf spp f -> b (v r srf spp) f"
            )
        self.end = end

    def scale_gaussians(self, scale_factor: Float[Tensor, " b"]) -> None:
        if self.end:
            self.means *= scale_factor[..., None, None]
            self.covariances *= (scale_factor**2)[..., None, None, None]

    @property
    def gaussians(self) -> Gaussians:
        return Gaussians(
            means=self.means[:, : self.end],
            covariances=self.covariances[:, : self.end],
            harmonics=self.harmonics[:, : self.end],
            opacities=self.opacities[:, : self.end],
            rotations=self.rotations.expand(-1, self.end, -1),
            scales=self.scales.expand(-1, self.end, -1),
        )

    @property
    def guidance_features(self) -> Tensor | None:
        return None if self.features is None else self.features[:, : self.end]
