"""Per-chunk camera, point, Gaussian, and guidance-feature prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

if TYPE_CHECKING:
    from ..encoder import ReCoSplatEncoder


@dataclass
class ChunkPrediction:
    """Adapter output for one chunk, in the shapes `AccumulatingGaussians` expects."""

    gaussians: object
    guidance_features: Tensor | None
    local_pts: Tensor
    depths: Tensor


class ChunkPredictor:
    def __init__(self, encoder: "ReCoSplatEncoder") -> None:
        self.encoder = encoder

    def decode_cameras(
        self,
        hidden: Tensor,
        pos: Tensor,
        patch_start_idx: int,
        b: int,
        v: int,
        patch_h: int,
        patch_w: int,
    ) -> Tensor:
        """Raw (un-normalized) c2w poses (b, v, 4, 4), fp32."""
        enc = self.encoder
        with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            camera_hidden = enc.camera_decoder(hidden, xpos=pos)
        with torch.amp.autocast("cuda", enabled=False):
            camera_hidden = camera_hidden.float()
            camera_poses_raw = enc.camera_head(
                camera_hidden[:, patch_start_idx:], patch_h, patch_w
            ).reshape(b, v, 4, 4)
        return camera_poses_raw

    def predict(
        self,
        hidden_chunk: Tensor,
        pos_chunk: Tensor,
        patch_start_idx: int,
        rgb_feat_chunk: Tensor,
        guidance_tokens: Tensor | None,
        guidance_pos: Tensor | None,
        b: int,
        v_chunk: int,
        h: int,
        w: int,
        patch_h: int,
        patch_w: int,
        global_step: int,
        adapter_poses: Tensor,
        c2w_chunk: Tensor | None,
    ) -> ChunkPrediction:
        enc = self.encoder
        device = hidden_chunk.device

        with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            if enc.upscale_token_ratio > 1:
                pos_aux = pos_chunk[:, :patch_start_idx]
                pos_img = enc.position_getter(
                    b * v_chunk,
                    patch_h * enc.upscale_token_ratio,
                    patch_w * enc.upscale_token_ratio,
                    device=device,
                    anchor_hw=enc._upsampled_rope_anchor(),
                )
                pos_img = pos_img + 1 if patch_start_idx > 0 else pos_img
                pos_upsampled = torch.cat([pos_aux, pos_img], dim=1)
            else:
                pos_upsampled = pos_chunk

            if enc.upscale_token_ratio > 1:
                hidden_aux_token = hidden_chunk[:, :patch_start_idx, :]
                hidden_img_token = hidden_chunk[:, patch_start_idx:, :]
                hidden_img_token = rearrange(
                    hidden_img_token, "b (h w) c -> b c h w", h=patch_h, w=patch_w
                )
                hidden_img_token = F.interpolate(
                    hidden_img_token,
                    scale_factor=enc.upscale_token_ratio,
                    mode="bilinear",
                    align_corners=False,
                )
                hidden_img_token = rearrange(hidden_img_token, "b c h w -> b (h w) c")
                hidden_chunk_upsampled = torch.cat([hidden_aux_token, hidden_img_token], dim=1)
            else:
                hidden_chunk_upsampled = hidden_chunk
            hidden_gaussian_chunk = hidden_chunk_upsampled.clone()

        hidden_gaussian_chunk[:, patch_start_idx:, :] = (
            hidden_gaussian_chunk[:, patch_start_idx:, :] + rgb_feat_chunk
        )
        with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            need_point_intermediates = enc.guidance_feature_head is not None
            point_hidden_chunk_out = enc.point_decoder(
                hidden_chunk_upsampled,
                xpos=pos_upsampled,
                guidance_tokens=guidance_tokens.to(hidden_chunk_upsampled.dtype)
                if guidance_tokens is not None
                else None,
                guidance_pos=guidance_pos.long() if guidance_pos is not None else None,
                return_intermediates=need_point_intermediates,
            )
            gaussian_hidden_chunk = enc.gaussian_decoder(
                hidden_gaussian_chunk,
                xpos=pos_upsampled,
                guidance_tokens=guidance_tokens.to(hidden_chunk_upsampled.dtype)
                if guidance_tokens is not None
                else None,
                guidance_pos=guidance_pos.long() if guidance_pos is not None else None,
            )
            if need_point_intermediates:
                point_hidden_chunk, point_intermediates = point_hidden_chunk_out
            else:
                point_hidden_chunk = point_hidden_chunk_out
                point_intermediates = None

            guidance_features_chunk = None
            if enc.guidance_feature_head is not None:
                assert point_intermediates is not None and len(point_intermediates) == 6
                tokens_for_dpt = [
                    point_intermediates[0][:, enc.backbone.patch_start_idx:].detach(),
                    point_intermediates[2][:, enc.backbone.patch_start_idx:].detach(),
                    point_intermediates[4][:, enc.backbone.patch_start_idx:].detach(),
                    point_intermediates[5][:, enc.backbone.patch_start_idx:].detach(),
                ]
                guidance_features_chunk = enc.guidance_feature_head(
                    tokens_for_dpt, image_size=(h, w)
                )
                guidance_features_chunk = F.interpolate(
                    guidance_features_chunk,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
                guidance_features_chunk = rearrange(
                    guidance_features_chunk,
                    "(b chunk) f hh ww -> b chunk hh ww f",
                    b=b,
                    chunk=v_chunk,
                )

        with torch.amp.autocast("cuda", enabled=False):
            point_hidden_chunk = point_hidden_chunk.float()
            gaussian_hidden_chunk = gaussian_hidden_chunk.float()

            out_h = patch_h * enc.gaussians_per_axis
            out_w = patch_w * enc.gaussians_per_axis
            ret = enc.point_head([point_hidden_chunk[:, patch_start_idx:]], (h, w)).reshape(
                b, v_chunk, out_h, out_w, -1
            )
            gaussian_params = enc.gaussian_head(
                [gaussian_hidden_chunk[:, patch_start_idx:]], (h, w)
            ).reshape(b, v_chunk, out_h, out_w, -1)

        gaussians_chunk, _, local_pts_chunk, depths_chunk = enc._gaussian_adapter_helper(
            b,
            v_chunk,
            h,
            w,
            global_step,
            ret,
            gaussian_params,
            adapter_poses,
            c2w_chunk,
        )
        if guidance_features_chunk is not None:
            guidance_features_chunk = rearrange(
                guidance_features_chunk, "b v h w f -> b v (h w) 1 1 f"
            )
            assert enc.cfg.num_surfaces == 1

        return ChunkPrediction(
            gaussians=gaussians_chunk,
            guidance_features=guidance_features_chunk,
            local_pts=local_pts_chunk,
            depths=depths_chunk,
        )
