"""Convolutional tokenizer for rendered guidance maps."""

import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class LayerNorm2d(nn.Module):
    """Wrapper around nn.LayerNorm for 2D feature maps shaped [B, C, H, W]."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:

        x_permuted = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x_permuted)
        return x_norm.permute(0, 3, 1, 2)


class ToTokens(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
    ):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=patch_size,
                padding=0,
                stride=patch_size,
            ),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            LayerNorm2d(out_channels, elementwise_affine=True, eps=1.0e-6),
        )

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(self, x: Float[Tensor, "bv c h w"]) -> Float[Tensor, "bv d hp wp"]:
        return self.conv_block(x)
