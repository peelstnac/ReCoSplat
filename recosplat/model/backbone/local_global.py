"""Image encoder and alternating local/global transformer backbone."""

import math
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

torch.backends.cuda.matmul.allow_tf32 = True

from ..cache.kv_cache import KVCacheManager
from ..geometry import get_intrinsic_embedding_new
from ..layers.attention import FlashAttentionRope
from ..layers.block import BlockLayerType, BlockRope
from ..layers.pos_embed import PositionGetter, RoPE2D
from .dinov2.hub.backbones import dinov2_vitl14_reg
from .dinov2.layers import Mlp, PatchEmbed


@dataclass
class ARCfg:
    is_enabled: bool = False
    kv_cache_strategy: Literal["accumulate"] = "accumulate"


RegisterTokensMode = Literal[
    "default",
    "first_chunk_and_first_view_of_next_chunks_is_special",
    "first_chunk_and_last_view_of_next_chunks_is_special",
]


@dataclass
class BackboneLocalGlobalCfg:
    name: Literal["local_global"]
    intrinsics_embed_degree: int = 0
    intrinsics_embed_type: Literal["pixelwise", "none"] = "none"
    predict_intrinsics: bool = False
    use_pred_intrinsics_for_embed: bool = False
    override_decoder_block_layer_type: Dict[int, BlockLayerType] = field(default_factory=dict)
    ar: ARCfg = field(default_factory=ARCfg)
    register_tokens: RegisterTokensMode = "default"
    rope_anchor_grid: list[int] | None = None


class BackboneLocalGlobal(nn.Module):
    def __init__(self, cfg: BackboneLocalGlobalCfg):
        super().__init__()

        self.cfg = cfg
        if cfg.use_pred_intrinsics_for_embed and not cfg.predict_intrinsics:
            raise ValueError(
                "use_pred_intrinsics_for_embed requires predict_intrinsics=true"
            )
        self.use_pred_intrinsics_for_embed = cfg.use_pred_intrinsics_for_embed
        self.rope_anchor_grid: tuple[int, int] | None = (
            tuple(cfg.rope_anchor_grid) if cfg.rope_anchor_grid is not None else None
        )

        self.register_tokens_mode = cfg.register_tokens
        self.use_chunk_first_view_register_tokens = (
            self.register_tokens_mode == "first_chunk_and_first_view_of_next_chunks_is_special"
        )
        self.use_chunk_last_view_register_tokens = (
            self.register_tokens_mode == "first_chunk_and_last_view_of_next_chunks_is_special"
        )

        self.patch_size = 14
        self.encoder = dinov2_vitl14_reg(pretrained=False, patch_size=self.patch_size)
        del self.encoder.mask_token

        self.predict_intrinsics = cfg.predict_intrinsics
        if cfg.predict_intrinsics:
            self.intrinsic_head = Mlp(1024, hidden_features=1024, out_features=2)

        self.rope = RoPE2D(freq=100.0)
        self.position_getter = PositionGetter()
        dec_embed_dim = 1024
        dec_num_heads = 16
        mlp_ratio = 4
        dec_depth = 36
        self.decoder = nn.ModuleList(
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
                    init_values=0.01,
                    qk_norm=True,
                    attn_class=FlashAttentionRope,
                    rope=self.rope,
                    block_layer_type=(
                        "default"
                        if layer_idx not in cfg.override_decoder_block_layer_type
                        else cfg.override_decoder_block_layer_type[layer_idx]
                    ),
                )
                for layer_idx in range(dec_depth)
            ]
        )
        self.dec_embed_dim = dec_embed_dim

        num_register_tokens = 5
        self.num_register_tokens = num_register_tokens
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        self.register_token_first_view = None
        self.register_token_last_view = None
        self.register_token_first_chunk = None
        if self.use_chunk_first_view_register_tokens:
            self.register_token_first_chunk = nn.Parameter(
                torch.randn(1, 1, num_register_tokens, self.dec_embed_dim)
            )
            self.register_token_first_view = nn.Parameter(
                torch.randn(1, 1, num_register_tokens, self.dec_embed_dim)
            )
            nn.init.normal_(self.register_token_first_chunk, std=1e-6)
            nn.init.normal_(self.register_token_first_view, std=1e-6)
        elif self.use_chunk_last_view_register_tokens:
            self.register_token_first_chunk = nn.Parameter(
                torch.randn(1, 1, num_register_tokens, self.dec_embed_dim)
            )
            self.register_token_last_view = nn.Parameter(
                torch.randn(1, 1, num_register_tokens, self.dec_embed_dim)
            )
            nn.init.normal_(self.register_token_first_chunk, std=1e-6)
            nn.init.normal_(self.register_token_last_view, std=1e-6)

        self.intrinsics_embed_degree = cfg.intrinsics_embed_degree
        self.intrinsics_embed_type = cfg.intrinsics_embed_type
        if self.intrinsics_embed_type == "pixelwise":
            self.intrinsics_embed_decoder_dim = (
                (self.intrinsics_embed_degree + 1) ** 2 if self.intrinsics_embed_degree > 0 else 3
            )
            self.intrinsics_embed_layer = PatchEmbed(
                patch_size=self.patch_size,
                in_chans=self.intrinsics_embed_decoder_dim,
                embed_dim=dec_embed_dim,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )

            nn.init.constant_(self.intrinsics_embed_layer.proj.weight, 0)
            nn.init.constant_(self.intrinsics_embed_layer.proj.bias, 0)

        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        self.ar_cfg = cfg.ar


    @staticmethod
    def _kv_cache_max_len(
        block_layer_type: str,
        N: int,
        hw: int,
        chunk_size_f: int,
        chunk_size_s: int,
    ) -> int:
        """Return the required cache capacity for one global layer."""
        if block_layer_type == "ignore_kv_cache":
            return 0

        total_tokens = N * hw

        if block_layer_type in {
            "global_prev_first_view_plus_current",
            "global_prev_last_view_plus_current",
        }:
            first_chunk_views = min(chunk_size_f, N)
            remaining_views = max(0, N - first_chunk_views)
            extra_chunks = math.ceil(remaining_views / max(chunk_size_s, 1))
            return first_chunk_views * hw + extra_chunks * hw

        return total_tokens

    def decode_inference(
        self,
        hidden,
        kv_cache: List[KVCacheManager | None],
        chunk_idx: int,
        N,
        H,
        W,
    ):
        """Decode one input chunk while updating the persistent caches."""
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []

        hidden = hidden.reshape(B * N, hw, -1)

        if self.register_tokens_mode == "default":
            register_token = self.register_token.repeat(1, BN, 1, 1).squeeze(0)
        elif self.register_tokens_mode == "first_chunk_and_first_view_of_next_chunks_is_special":
            if chunk_idx == 0:
                register_token = self.register_token_first_chunk.repeat(1, BN, 1, 1).squeeze(0)
            else:
                register_token = torch.cat(
                    [
                        self.register_token_first_view.repeat(B, 1, 1, 1),
                        self.register_token.repeat(B, N - 1, 1, 1),
                    ],
                    dim=1,
                ).reshape(B * N, *self.register_token.shape[-2:])
        elif self.register_tokens_mode == "first_chunk_and_last_view_of_next_chunks_is_special":
            if chunk_idx == 0:
                register_token = self.register_token_first_chunk.repeat(1, BN, 1, 1).squeeze(0)
            else:
                register_token = torch.cat(
                    [
                        self.register_token.repeat(B, N - 1, 1, 1),
                        self.register_token_last_view.repeat(B, 1, 1, 1),
                    ],
                    dim=1,
                ).reshape(B * N, *self.register_token.shape[-2:])
        else:
            raise NotImplementedError(self.register_tokens_mode)

        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        pos = self.position_getter(
            B * N,
            H // self.patch_size,
            W // self.patch_size,
            hidden.device,
            anchor_hw=self.rope_anchor_grid,
        )

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        for i, blk in enumerate(self.decoder):
            if i % 2 == 0:
                pos = pos.view(B * N, hw, -1)
                hidden = hidden.view(B * N, hw, -1)
                hidden = blk(hidden, xpos=pos, N=N)
            else:
                pos = pos.view(B, N * hw, -1)
                hidden = hidden.view(B, N * hw, -1)

                uses_cache = blk.block_layer_type != "ignore_kv_cache"
                cache_mgr = kv_cache[(i - 1) // 2]
                kv_cache_handle = cache_mgr if (uses_cache and cache_mgr is not None) else None

                hidden = blk(
                    hidden,
                    xpos=pos,
                    N=N,
                    kv_cache_manager=kv_cache_handle,
                    chunk_first_view_tokens=hw,
                    is_first_chunk=chunk_idx == 0,
                )

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B * N, hw, -1)

    def forward_inference(
        self,
        imgs,
        kv_cache: List[KVCacheManager | None],
        chunk_idx: int,
        intrinsics=None,
    ):
        """Encode and decode one input chunk."""
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, C, H, W = imgs.shape

        imgs = imgs.reshape(B * N, C, H, W)
        hidden = self.encoder.forward_features(imgs)

        intrinsic_pred = None
        if self.predict_intrinsics:
            x_norm_clstoken = hidden["x_norm_clstoken"]
            intrinsic_pred = self.intrinsic_head(x_norm_clstoken)
            intrinsic_pred = F.relu(intrinsic_pred)

            if self.use_pred_intrinsics_for_embed:
                focal_pred = rearrange(intrinsic_pred, "(b v) d -> b v d", b=B, v=N)
                intrinsics[:, :, 0, 0] = focal_pred[:, :, 0]
                intrinsics[:, :, 1, 1] = focal_pred[:, :, 1]

        hidden = hidden["x_norm_patchtokens"]

        if self.intrinsics_embed_type == "pixelwise":
            intrinsic_emb = get_intrinsic_embedding_new(
                intrinsics, imgs, degree=self.intrinsics_embed_degree
            )
            intrinsic_emb = self.intrinsics_embed_layer(intrinsic_emb)
            hidden = hidden + intrinsic_emb

        hidden, pos = self.decode_inference(hidden, kv_cache, chunk_idx, N, H, W)
        return hidden, pos, self.patch_start_idx, intrinsic_pred

    def make_streaming_kv_caches(
        self,
        batch_size: int,
        num_views: int,
        num_patch_tokens: int,
        chunk_size_f: int,
        chunk_size_s: int,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> List[KVCacheManager | None]:
        """Allocate the global-layer caches for a streaming pass."""
        hw = num_patch_tokens + self.patch_start_idx
        caches: List[KVCacheManager | None] = []
        for layer_idx in range(1, len(self.decoder), 2):
            max_seq_len = self._kv_cache_max_len(
                self.decoder[layer_idx].block_layer_type,
                num_views,
                hw,
                chunk_size_f,
                chunk_size_s,
            )
            if max_seq_len > 0:
                caches.append(
                    KVCacheManager(
                        batch_size=batch_size,
                        num_heads=self.decoder[0].attn.num_heads,
                        max_seq_len=max_seq_len,
                        head_dim=self.dec_embed_dim // self.decoder[0].attn.num_heads,
                        strategy=self.ar_cfg.kv_cache_strategy,
                        chunk_size=[chunk_size_f * hw, chunk_size_s * hw],
                        dtype=dtype,
                        device=device,
                    )
                )
            else:
                caches.append(None)
        return caches
