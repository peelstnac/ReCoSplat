"""Renderer cfg + output types, split from splatting_gsplat.py so that losses and the
config schema can import them without pulling in gsplat (CUDA). The renderer module
re-exports them for back-compat."""

from dataclasses import dataclass
from typing import Literal

from jaxtyping import Float
from torch import Tensor

DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]


@dataclass
class DecoderSplattingSplatCfg:
    name: Literal["splatting_cuda", "splatting_gsplat"]
    background_color: list[float]
    make_scale_invariant: bool
    prune_opacity_threshold: float = 0.005


@dataclass
class DecoderOutput:
    color: Float[Tensor, "batch view 3 height width"]
    depth: Float[Tensor, "batch view height width"] | None
