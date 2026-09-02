"""Transformer block with rotary attention and optional guidance attention."""

from typing import Callable, Literal, Optional

from torch import Tensor, nn

from ..backbone.dinov2.layers.drop_path import DropPath
from ..backbone.dinov2.layers.layer_scale import LayerScale
from ..backbone.dinov2.layers.mlp import Mlp
from ..cache.kv_cache import KVCacheManager
from .attention import FlashAttentionRope

BlockLayerType = Literal[
    "default",
    "ignore_kv_cache",
    "global_prev_first_view_plus_current",
    "global_prev_last_view_plus_current",
]


class BlockRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = FlashAttentionRope,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        qk_norm: bool = False,
        rope=None,
        block_layer_type: BlockLayerType = "default",
    ) -> None:
        super().__init__()
        self.block_layer_type = block_layer_type

        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            qk_norm=qk_norm,
            rope=rope,
        )

        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.sample_drop_ratio = drop_path

        self.norm_guidance = norm_layer(dim)

    def forward(
        self,
        x: Tensor,
        xpos=None,
        N: Optional[int] = None,
        kv_cache_manager: Optional[KVCacheManager] = None,
        chunk_first_view_tokens: int | None = None,
        is_first_chunk: bool = False,
        guidance_attn: Optional[nn.Module] = None,
        guidance_tokens: Optional[Tensor] = None,
        guidance_pos: Optional[Tensor] = None,
    ) -> Tensor:
        if self.block_layer_type == "default":
            return self.forward_default(
                x,
                xpos=xpos,
                N=N,
                kv_cache_manager=kv_cache_manager,
                guidance_attn=guidance_attn,
                guidance_tokens=guidance_tokens,
                guidance_pos=guidance_pos,
            )
        elif self.block_layer_type == "ignore_kv_cache":
            return self.forward_default(
                x,
                xpos=xpos,
                N=N,
                kv_cache_manager=None,
                guidance_attn=guidance_attn,
                guidance_tokens=guidance_tokens,
                guidance_pos=guidance_pos,
            )
        elif self.block_layer_type == "global_prev_first_view_plus_current":
            return self.forward_default(
                x,
                xpos=xpos,
                N=N,
                kv_cache_manager=kv_cache_manager,
                kv_cache_mode="prev_first_view_plus_current",
                chunk_first_view_tokens=chunk_first_view_tokens,
                is_first_chunk=is_first_chunk,
                guidance_attn=guidance_attn,
                guidance_tokens=guidance_tokens,
                guidance_pos=guidance_pos,
            )
        elif self.block_layer_type == "global_prev_last_view_plus_current":
            return self.forward_default(
                x,
                xpos=xpos,
                N=N,
                kv_cache_manager=kv_cache_manager,
                kv_cache_mode="prev_last_view_plus_current",
                chunk_first_view_tokens=chunk_first_view_tokens,
                is_first_chunk=is_first_chunk,
                guidance_attn=guidance_attn,
                guidance_tokens=guidance_tokens,
                guidance_pos=guidance_pos,
            )
        else:
            raise ValueError(f"self.block_layer_type {self.block_layer_type} not supported")

    def forward_default(
        self,
        x: Tensor,
        xpos=None,
        N: int | None = None,
        kv_cache_manager: KVCacheManager | None = None,
        kv_cache_mode: str | None = None,
        chunk_first_view_tokens: int | None = None,
        is_first_chunk: bool = False,
        guidance_attn: Optional[nn.Module] = None,
        guidance_tokens: Optional[Tensor] = None,
        guidance_pos: Optional[Tensor] = None,
    ) -> Tensor:
        assert self.sample_drop_ratio == 0

        def attn_residual_func(
            x: Tensor, kv_cache_manager: KVCacheManager | None = None
        ) -> Tensor:
            attn_kwargs = dict(xpos=xpos, kv_cache_manager=kv_cache_manager)
            if kv_cache_mode is not None:
                attn_kwargs["kv_cache_mode"] = kv_cache_mode
                attn_kwargs["chunk_first_view_tokens"] = chunk_first_view_tokens
                attn_kwargs["is_first_chunk"] = is_first_chunk
            return self.ls1(self.attn(self.norm1(x), **attn_kwargs))

        def ffn_residual_func(x: Tensor) -> Tensor:
            return self.ls2(self.mlp(self.norm2(x)))

        x = x + attn_residual_func(x, kv_cache_manager=kv_cache_manager)
        if guidance_attn is not None and guidance_tokens is not None:
            guidance_out = guidance_attn(
                self.norm_guidance(x),
                guidance_tokens,
                guidance_tokens,
                qpos=xpos,
                kpos=guidance_pos,
            )
            x = x + guidance_out
        x = x + ffn_residual_func(x)

        return x
