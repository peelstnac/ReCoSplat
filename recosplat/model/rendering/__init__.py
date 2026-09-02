"""Gaussian rendering components."""

from .splatting_gsplat import (
    DecoderOutput,
    DecoderSplattingGSPlat,
    DecoderSplattingSplatCfg,
    prune_gaussians,
)

__all__ = [
    "DecoderOutput",
    "DecoderSplattingGSPlat",
    "DecoderSplattingSplatCfg",
    "prune_gaussians",
]
