"""Transformer decoder and dense point-prediction head."""

from functools import partial
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..backbone.dinov2.layers import Mlp
from .attention import FlashAttentionRope, FlashCrossAttentionRope
from .block import BlockRope


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
        guidance_cross_attn: bool = False,
        guidance_num_heads: int | None = None,
        guidance_zero_init: bool = True,
    ):
        super().__init__()

        self.depth = depth
        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()

        self.blocks = nn.ModuleList(
            [
                BlockRope(
                    dim=dec_embed_dim,
                    num_heads=dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    drop_path=0.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    ffn_layer=Mlp,
                    init_values=None,
                    qk_norm=False,
                    attn_class=FlashAttentionRope,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

        self.guidance_norms = None
        self.guidance_cross_attn_layers = None
        if guidance_cross_attn:
            heads = guidance_num_heads if guidance_num_heads is not None else dec_num_heads
            self.guidance_cross_attn_layers = nn.ModuleList(
                [
                    FlashCrossAttentionRope(
                        dim=dec_embed_dim,
                        num_heads=heads,
                        qkv_bias=True,
                        proj_bias=True,
                        qk_norm=True,
                        rope=rope,
                    )
                    for _ in range(depth)
                ]
            )
            if guidance_zero_init and self.guidance_cross_attn_layers is not None:
                for attn in self.guidance_cross_attn_layers:
                    nn.init.constant_(attn.proj.weight, 0)
                    if attn.proj.bias is not None:
                        nn.init.constant_(attn.proj.bias, 0)
            self.guidance_norms = nn.ModuleList(
                [nn.LayerNorm(dec_embed_dim, eps=1e-6) for _ in range(depth)]
            )

    def forward(
        self,
        hidden,
        xpos=None,
        guidance_tokens: Tensor | None = None,
        guidance_pos: Tensor | None = None,
        return_intermediates: bool = False,
    ):
        use_guidance = self.guidance_cross_attn_layers is not None
        if use_guidance:
            assert guidance_tokens is not None
            assert guidance_pos is not None
        hidden = self.projects(hidden)
        intermediates = [] if return_intermediates else None
        if return_intermediates:
            intermediates.append(hidden)
        for i, blk in enumerate(self.blocks):
            guidance_attn = self.guidance_cross_attn_layers[i] if use_guidance else None
            guidance_tokens_i = guidance_tokens if use_guidance else None
            guidance_pos_i = guidance_pos if use_guidance else None

            block_kwargs = {
                "xpos": xpos,
                "guidance_attn": guidance_attn,
                "guidance_tokens": guidance_tokens_i,
                "guidance_pos": guidance_pos_i,
            }
            hidden = blk(hidden, **block_kwargs)
            if return_intermediates:
                intermediates.append(hidden)
        out = self.linear_out(hidden)
        if return_intermediates:
            return out, intermediates
        return out


class LinearPts3d(nn.Module):
    """
    Linear head
    Each token outputs: (self.patch_size // self.downsample_ratio)² points
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3, downsample_ratio=1, points_per_axis=None):
        super().__init__()
        self.patch_size = patch_size
        self.downsample_ratio = downsample_ratio

        points_per_token = (
            (self.patch_size // downsample_ratio) ** 2 if points_per_axis is None else points_per_axis**2
        )
        self.points_per_axis = (
            self.patch_size // downsample_ratio if points_per_axis is None else points_per_axis
        )
        self.proj = nn.Linear(dec_embed_dim, output_dim * points_per_token)

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        upsample_factor = self.points_per_axis

        feat = self.proj(tokens)

        H_patches = int(H // self.patch_size)
        W_patches = int(W // self.patch_size)
        feat = feat.view(B, H_patches, W_patches, -1)
        feat = feat.permute(0, 3, 1, 2)

        feat = F.pixel_shuffle(feat, upsample_factor)

        return feat.permute(0, 2, 3, 1)
