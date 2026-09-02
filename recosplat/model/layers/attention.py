"""Rotary-position attention layers with selective key/value caching."""

from __future__ import annotations

from typing import Callable, Literal, Optional

import torch
from torch import Tensor, nn
from torch.nn.attention import SDPBackend
from torch.nn.functional import scaled_dot_product_attention

from ..cache.kv_cache import KVCacheManager


def _sdpa(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Use flash attention for bfloat16 and safe fallbacks otherwise."""
    if q.dtype == torch.bfloat16:
        with nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return scaled_dot_product_attention(q, k, v)
    with nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        return scaled_dot_product_attention(q, k, v)


class CrossAttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        rope=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = rope

    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        """
        Args:
            query: Tensor of shape (B, N, C), input query
            key: Tensor of shape (B, M, C), input key
            value: Tensor of shape (B, M, C), input value
            attn_bias: Optional tensor for attention bias
        Returns:
            Tensor of shape (B, N, C), output of cross-attention
        """
        B, N, C = query.shape
        _, M, _ = key.shape

        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        q = q * self.scale

        attn = q @ k.transpose(-2, -1)
        if attn_bias is not None:
            attn = attn + attn_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FlashCrossAttentionRope(CrossAttentionRope):
    def forward(self, query: Tensor, key: Tensor, value: Tensor, attn_bias=None, qpos=None, kpos=None) -> Tensor:
        """
        Args:
            query: Tensor of shape (B, N, C), input query
            key: Tensor of shape (B, M, C), input key
            value: Tensor of shape (B, M, C), input value
            attn_bias: Optional tensor for attention bias
        Returns:
            Tensor of shape (B, N, C), output of cross-attention
        """
        B, N, C = query.shape
        _, M, _ = key.shape

        q = self.q_proj(query).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = self.k_proj(key).reshape(B, M, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = self.v_proj(value).reshape(B, M, self.num_heads, C // self.num_heads).transpose(1, 2)

        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, qpos)
            k = self.rope(k, kpos)

        x = _sdpa(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionRope(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        qk_norm: bool = False,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        rope=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()

        self.rope = rope

    def forward(self, x: Tensor, attn_bias=None, xpos=None, **_: object) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


KVCacheMode = Literal["prev_first_view_plus_current", "prev_last_view_plus_current", None]


class FlashAttentionRope(AttentionRope):
    def forward(
        self,
        x: Tensor,
        attn_bias=None,
        xpos=None,
        kv_cache_manager: Optional[KVCacheManager] = None,
        kv_cache_mode: KVCacheMode = None,
        chunk_first_view_tokens: int | None = None,
        is_first_chunk: bool = False,
    ) -> Tensor:
        """
        kv_cache_mode:
            - None (default): standard behavior (all queries/keys/values are cached)
            - "prev_first_view_plus_current": queries are the current chunk; keys/values are
              previous-chunk first-view tokens (from cache) concatenated with all current-chunk
              tokens, but only the first-view tokens of the current chunk are written to cache.
            - "prev_last_view_plus_current": same, but the LAST view of the current chunk is
              written to cache instead of the first.
        chunk_first_view_tokens: number of tokens that correspond to one view of the chunk
            (the last-view mode slices the same count from the chunk tail)
        """
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).transpose(1, 3)
        q, k, v = [qkv[:, :, i] for i in range(3)]
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        if self.rope is not None:
            q = self.rope(q, xpos)
            k = self.rope(k, xpos)

        def _get_cached_prefix():
            if kv_cache_manager is None:
                return None, None
            return kv_cache_manager.get()

        if kv_cache_mode in ("prev_first_view_plus_current", "prev_last_view_plus_current"):
            assert kv_cache_manager is not None
            if is_first_chunk:
                kv_cache_mode = None
            else:
                assert chunk_first_view_tokens is not None, (
                    "chunk_first_view_tokens is required for selective KV cache modes"
                )

                fv_tokens = chunk_first_view_tokens
                cached_k, cached_v = _get_cached_prefix()

                k_attn = k if cached_k is None else torch.cat([cached_k, k], dim=2)
                v_attn = v if cached_v is None else torch.cat([cached_v, v], dim=2)

                x_out = _sdpa(q, k_attn, v_attn)
                x_out = x_out.transpose(1, 2).reshape([B, N, C])

                if kv_cache_mode == "prev_first_view_plus_current":
                    kv_cache_manager.update(
                        new_k=k[:, :, :fv_tokens, :],
                        new_v=v[:, :, :fv_tokens, :],
                    )
                else:
                    kv_cache_manager.update(
                        new_k=k[:, :, -fv_tokens:, :],
                        new_v=v[:, :, -fv_tokens:, :],
                    )

                x_out = self.proj(x_out)
                x_out = self.proj_drop(x_out)
                return x_out
        elif kv_cache_mode is None:
            pass
        else:
            raise NotImplementedError(f"unsupported kv_cache_mode: {kv_cache_mode!r}")

        if kv_cache_manager is not None:
            kv_cache_manager.update(
                new_k=k,
                new_v=v,
            )
            k, v = kv_cache_manager.get()

        x = _sdpa(q, k, v)

        x = x.transpose(1, 2).reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
