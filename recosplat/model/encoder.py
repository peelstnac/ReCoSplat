"""Streaming ReCoSplat encoder."""

from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .adapter.gaussian_adapter import GaussianAdapterCfg, UnifiedGaussianAdapter
from .adapter.types import AccumulatingGaussians, Gaussians
from .ar.chunk_schedule import ChunkScheduler
from .ar.predictor import ChunkPredictor
from .backbone import BACKBONES
from .backbone.dinov2.layers import PatchEmbed
from .backbone.local_global import BackboneLocalGlobalCfg
from .geometry import calculate_camera_poses_scale, se3_inverse
from .heads.camera_head import CameraHead
from .heads.dpt_guidance import DPTOutputAdapter_fix
from .layers.transformer_decoder import LinearPts3d, TransformerDecoder
from .reco.tokenizer import ToTokens


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class PoseFeatureGuidanceCfg:
    enabled: bool = False
    rendered_feature_dim: int = 0
    zero_init: bool = True


@dataclass
class ScaleCfg:
    gs_scale_matching: Literal["none", "first_chunk"] = "first_chunk"


@dataclass
class EncoderCfg:
    backbone: BackboneLocalGlobalCfg
    gaussian_adapter: GaussianAdapterCfg
    opacity_mapping: OpacityMappingCfg
    num_surfaces: int
    gaussians_per_axis: int
    pose_free: bool = True
    gaussian_downsample_ratio: int = 1
    upscale_token_ratio: int = 1

    pose_feature_guidance: PoseFeatureGuidanceCfg = field(default_factory=PoseFeatureGuidanceCfg)
    scale: ScaleCfg = field(default_factory=ScaleCfg)


@dataclass
class EncoderOutput:
    gaussians: Gaussians
    pred_camera_poses: Float[Tensor, "batch view 4 4"]
    pred_intrinsics: Float[Tensor, "batch view 2"] | None
    depths: Float[Tensor, "batch view point srf spp"]
    use_gt_pose: bool
    gt_scale_factor: Float[Tensor, " batch"] | None = None
    pred_pose_max_1_scale: Float[Tensor, " batch"] | None = None


class ReCoSplatEncoder(nn.Module):
    backbone: nn.Module
    gaussian_adapter: UnifiedGaussianAdapter

    def __init__(self, cfg: EncoderCfg) -> None:
        super().__init__()
        self.cfg = cfg

        backbone_cls = BACKBONES.get(cfg.backbone.name)
        self.backbone = backbone_cls(cfg.backbone)

        self.pose_free = cfg.pose_free
        self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.patch_size
        self.raw_gs_dim = self.gaussian_adapter.d_opacity_in + self.gaussian_adapter.d_in

        self.gaussian_downsample_ratio = cfg.gaussian_downsample_ratio
        self.gaussians_per_axis = min(
            cfg.gaussians_per_axis, self.patch_size // cfg.gaussian_downsample_ratio
        )

        self.upscale_token_ratio = cfg.upscale_token_ratio
        self.position_getter = self.backbone.position_getter

        self.dec_embed_dim = 1024

        self.pose_feature_guidance_enabled = cfg.pose_feature_guidance.enabled
        self.guidance_feature_dim = (
            cfg.pose_feature_guidance.rendered_feature_dim
            if self.pose_feature_guidance_enabled
            else 0
        )

        self.point_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.backbone.rope,
            guidance_cross_attn=self.pose_feature_guidance_enabled,
            guidance_zero_init=cfg.pose_feature_guidance.zero_init,
        )
        self.point_head = LinearPts3d(
            patch_size=self.patch_size / self.upscale_token_ratio,
            dec_embed_dim=1024,
            output_dim=3,
            downsample_ratio=self.gaussian_downsample_ratio,
            points_per_axis=self.gaussians_per_axis // self.upscale_token_ratio,
        )

        self.gaussian_decoder = deepcopy(self.point_decoder)
        self.gaussian_head = LinearPts3d(
            patch_size=self.patch_size / self.upscale_token_ratio,
            dec_embed_dim=1024,
            output_dim=self.raw_gs_dim,
            downsample_ratio=self.gaussian_downsample_ratio,
            points_per_axis=self.gaussians_per_axis // self.upscale_token_ratio,
        )

        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            rope=self.backbone.rope,
        )
        self.camera_head = CameraHead(dim=512)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.rgb_embed = PatchEmbed(
            patch_size=self.patch_size // self.upscale_token_ratio,
            in_chans=3,
            embed_dim=2048,
            norm_layer=norm_layer,
        )
        nn.init.constant_(self.rgb_embed.proj.weight, 0)
        nn.init.constant_(self.rgb_embed.proj.bias, 0)

        self.guidance_feature_head = None
        self.guidance_to_tokens = None
        self.first_chunk_guidance_embed = None

        if self.pose_feature_guidance_enabled:
            token_channels = 6 + self.guidance_feature_dim

            patch_size_tokens = self.patch_size // self.upscale_token_ratio
            self.guidance_to_tokens = ToTokens(
                in_channels=token_channels,
                out_channels=self.dec_embed_dim,
                patch_size=patch_size_tokens,
            )

            render_channels = 3 + self.guidance_feature_dim
            self.first_chunk_guidance_embed = nn.Parameter(torch.zeros(render_channels, 1, 1))
            nn.init.normal_(self.first_chunk_guidance_embed, std=0.02)

            if self.guidance_feature_dim > 0:
                self.guidance_feature_head = DPTOutputAdapter_fix(
                    num_channels=self.guidance_feature_dim,
                    stride_level=1,
                    patch_size=patch_size_tokens,
                    hooks=[0, 1, 2, 3],
                    layer_dims=[self.dec_embed_dim] * 4,
                    feature_dim=256,
                    last_dim=64,
                    dim_tokens_enc=self.dec_embed_dim,
                    head_type="regression_8x",
                )

        self.chunk_predictor = ChunkPredictor(self)
        self._guidance_renderer = None

    @property
    def guidance_renderer(self):
        """Build the guidance renderer on first use."""
        if self._guidance_renderer is None:
            from .reco.guidance_renderer import GuidanceRenderer

            self._guidance_renderer = GuidanceRenderer(
                feature_dim=self.guidance_feature_dim,
                first_chunk_guidance_embed=self.first_chunk_guidance_embed,
            )
        return self._guidance_renderer

    def _upsampled_rope_anchor(self) -> tuple[int, int] | None:
        """Return the position-interpolation anchor for the upsampled token grid."""
        anchor = self.backbone.rope_anchor_grid
        if anchor is None:
            return None
        return (anchor[0] * self.upscale_token_ratio, anchor[1] * self.upscale_token_ratio)

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _gaussian_adapter_helper(
        self,
        b,
        v,
        h,
        w,
        global_step,
        ret,
        gaussian_params,
        camera_poses,
        c2w,
    ):
        xy, z = ret.split([2, 1], dim=-1)
        z = torch.exp(z)
        local_points = torch.cat([xy * z, z], dim=-1)
        pts_all = rearrange(local_points, "b v h w xyz -> (b v) xyz h w").contiguous()
        if pts_all.dim() == 4:
            pts_all = rearrange(pts_all, "(b v) d h w -> b v (h w) d", b=b, v=v)
        else:
            pts_all = rearrange(pts_all, "(b v) d l -> b v l d", b=b, v=v)
        local_pts = pts_all.clone()
        pts_all = pts_all.unsqueeze(-2)
        depths = pts_all[..., -1].unsqueeze(-1)

        gaussian_params = rearrange(gaussian_params, "b v h w d -> (b v) d h w").contiguous()
        if gaussian_params.dim() == 4:
            gaussians = rearrange(gaussian_params, "(b v) d h w -> b v (h w) d", b=b, v=v)
        else:
            gaussians = rearrange(gaussian_params, "(b v) d l -> b v l d", b=b, v=v)
        gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)

        densities = gaussians[..., 0].sigmoid().unsqueeze(-1)
        opacities = self.map_pdf_to_opacity(densities, global_step)

        if c2w is None:
            c2w = camera_poses

        gaussians = self.gaussian_adapter.forward(
            pts_all.unsqueeze(-2),
            depths,
            opacities,
            rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
            extrinsics=rearrange(c2w, "b v i j -> b v () () () i j"),
        )

        return gaussians, c2w, local_pts, depths

    def get_c2w(self, context: dict):
        if self.pose_free:
            return None
        return context["extrinsics"].clone()

    calculate_camera_poses_scale = staticmethod(calculate_camera_poses_scale)

    @torch.amp.autocast("cuda", enabled=False)
    def calculate_gt_scale_factor(
        self,
        pred_pose: Float[Tensor, "b view_pred 4 4"],
        gt_pose: Float[Tensor, "b view_gt 4 4"],
        chunk_size_f: int,
    ) -> tuple[Float[Tensor, " b"], Float[Tensor, " b"]]:
        """Return the predicted-to-reference scale ratio and predicted scale."""
        pred_pose = pred_pose.float()

        method = self.cfg.scale.gs_scale_matching

        if method == "none":
            pred_pose_max_1_scale = torch.ones(
                pred_pose.size(0), dtype=pred_pose.dtype, device=pred_pose.device
            )
            gt_pose_max_1_scale = torch.ones_like(pred_pose_max_1_scale)
        elif method == "first_chunk":
            pred_pose_max_1_scale = self.calculate_camera_poses_scale(pred_pose[:, :chunk_size_f])
            gt_pose_max_1_scale = self.calculate_camera_poses_scale(gt_pose[:, :chunk_size_f])
        else:
            raise NotImplementedError(method)

        return pred_pose_max_1_scale / gt_pose_max_1_scale, pred_pose_max_1_scale

    def _build_guidance_tokens(
        self,
        rendered: Float[Tensor, "b v c h w"],
        ctx_img: Float[Tensor, "b v 3 h w"],
    ) -> tuple[Float[Tensor, "bv n d"], Tensor]:
        """Combine rendered guidance with input images and tokenize the result."""
        combined = torch.cat([rendered, ctx_img], dim=2)

        b, v, _, _, _ = combined.shape
        combined = rearrange(combined, "b v c h w -> (b v) c h w")
        token_maps = self.guidance_to_tokens(combined)
        hp, wp = token_maps.shape[-2], token_maps.shape[-1]
        tokens = rearrange(token_maps, "bv d hp wp -> bv (hp wp) d")
        pos = self.position_getter(
            b * v,
            hp,
            wp,
            device=combined.device,
            anchor_hw=self._upsampled_rope_anchor(),
        )
        return tokens, pos

    def _prepare_guidance_for_chunk(
        self,
        acc_gs: AccumulatingGaussians,
        guidance_pose_chunk: Float[Tensor, "b vc 4 4"],
        intrinsics_chunk: Float[Tensor, "b vc 3 3"],
        ctx_img_chunk: Float[Tensor, "b vc 3 h w"],
    ) -> tuple[Tensor | None, Tensor | None]:
        """Render accumulated Gaussians and tokenize the resulting guidance."""
        if not self.pose_feature_guidance_enabled:
            return None, None
        rendered = self.guidance_renderer.render(
            acc_gs,
            guidance_pose_chunk.detach(),
            intrinsics_chunk.detach(),
            (ctx_img_chunk.shape[-2], ctx_img_chunk.shape[-1]),
        )
        with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
            tokens, pos = self._build_guidance_tokens(rendered, ctx_img_chunk)
        return tokens, pos

    def forward_streaming(
        self,
        context: dict,
        global_step: int = 0,
        *,
        chunk_size_f: int,
        chunk_size_s: int,
    ) -> EncoderOutput:
        """Run chunked inference with persistent per-layer key/value caches."""
        assert not self.training, "forward_streaming is the inference path"

        device = context["image"].device
        b, v, _, h, w = context["image"].shape
        patch_h, patch_w = h // self.patch_size, w // self.patch_size

        ctx_img = context["image"]
        ctx_int = context["intrinsics"]
        ctx_ext = context["extrinsics"]

        c2w = self.get_c2w(context)
        use_gt_pose = c2w is not None

        acc_gs = AccumulatingGaussians(
            b,
            v,
            h,
            w,
            feature_dim=self.guidance_feature_dim,
            device=device,
        )
        depths_per_chunk: list[Tensor] = []
        all_camera_poses: list[Tensor] = []
        all_intrinsic_pred: list[Tensor] = []

        kv_cache = self.backbone.make_streaming_kv_caches(
            batch_size=b,
            num_views=v,
            num_patch_tokens=patch_h * patch_w,
            chunk_size_f=chunk_size_f,
            chunk_size_s=chunk_size_s,
            device=device,
            dtype=torch.bfloat16,
        )

        first_pred_pose_raw: Tensor | None = None
        gt_scale_factor: Tensor | None = None
        pred_pose_max_1_scale: Tensor | None = None

        for chunk_idx, (start, end) in enumerate(
            ChunkScheduler(chunk_size_f, chunk_size_s).intervals(v)
        ):
            with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                hidden, pos, patch_start_idx, intrinsic_pred = self.backbone.forward_inference(
                    ctx_img[:, start:end],
                    kv_cache,
                    chunk_idx,
                    intrinsics=ctx_int[:, start:end].clone(),
                )
            if intrinsic_pred is not None:
                all_intrinsic_pred.append(
                    rearrange(intrinsic_pred, "(b vc) d -> b vc d", b=b)
                )

            assert self.cfg.scale.gs_scale_matching in ["none", "first_chunk"]
            camera_poses_raw = self.chunk_predictor.decode_cameras(
                hidden,
                pos,
                patch_start_idx,
                b,
                end - start,
                patch_h,
                patch_w,
            )
            with torch.amp.autocast("cuda", enabled=False):
                if start == 0:
                    assert first_pred_pose_raw is None
                    first_pred_pose_raw = camera_poses_raw[:, 0].clone()
                camera_poses = torch.einsum(
                    "bij, bnjk -> bnik", se3_inverse(first_pred_pose_raw), camera_poses_raw
                )

                if start == 0:
                    assert camera_poses.size(1) == chunk_size_f
                    gt_scale_factor, pred_pose_max_1_scale = self.calculate_gt_scale_factor(
                        camera_poses,
                        ctx_ext,
                        chunk_size_f,
                    )
                    if use_gt_pose:
                        c2w = c2w.clone()
                        c2w[:, :, :3, 3] *= gt_scale_factor[..., None, None]

                all_camera_poses.append(camera_poses)

            with torch.amp.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                rgb = rearrange(ctx_img[:, start:end], "b v c h w -> (b v) c h w")
                rgb_feat = self.rgb_embed(rgb)

            guidance_tokens, guidance_pos = None, None
            if self.pose_feature_guidance_enabled:
                guidance_pose_chunk = camera_poses if not use_gt_pose else c2w[:, start:end]
                intrinsics_chunk = ctx_int[:, start:end].clone()
                if self.backbone.use_pred_intrinsics_for_embed:
                    assert intrinsic_pred is not None
                    focal_pred = rearrange(
                        intrinsic_pred,
                        "(b vc) d -> b vc d",
                        b=b,
                        vc=end - start,
                    )
                    intrinsics_chunk[:, :, 0, 0] = focal_pred[:, :, 0]
                    intrinsics_chunk[:, :, 1, 1] = focal_pred[:, :, 1]
                guidance_tokens, guidance_pos = self._prepare_guidance_for_chunk(
                    acc_gs=acc_gs,
                    guidance_pose_chunk=guidance_pose_chunk,
                    intrinsics_chunk=intrinsics_chunk,
                    ctx_img_chunk=ctx_img[:, start:end],
                )

            prediction = self.chunk_predictor.predict(
                hidden_chunk=hidden,
                pos_chunk=pos,
                patch_start_idx=patch_start_idx,
                rgb_feat_chunk=rgb_feat,
                guidance_tokens=guidance_tokens,
                guidance_pos=guidance_pos,
                b=b,
                v_chunk=end - start,
                h=h,
                w=w,
                patch_h=patch_h,
                patch_w=patch_w,
                global_step=global_step,
                adapter_poses=camera_poses,
                c2w_chunk=None if c2w is None else c2w[:, start:end],
            )
            acc_gs.add_adapter_output(prediction.gaussians, prediction.guidance_features)
            depths_per_chunk.append(prediction.depths)

        gaussians = acc_gs.gaussians
        gaussians.features = acc_gs.guidance_features

        pred_intrinsics = torch.cat(all_intrinsic_pred, dim=1) if all_intrinsic_pred else None

        return EncoderOutput(
            gaussians=gaussians,
            pred_camera_poses=torch.cat(all_camera_poses, dim=1).contiguous(),
            pred_intrinsics=pred_intrinsics,
            depths=torch.cat(depths_per_chunk, dim=1),
            use_gt_pose=use_gt_pose,
            gt_scale_factor=gt_scale_factor,
            pred_pose_max_1_scale=pred_pose_max_1_scale,
        )
