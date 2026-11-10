"""Fine-tune an RGB-only ReCo model from the public encoder checkpoint."""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import math
import sys
from functools import partial
from pathlib import Path

import hydra
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recosplat.config.loading import load_typed_config
from recosplat.data.chunk_dataset import convert_poses
from recosplat.data.shims.crop_shim import rescale_and_crop
from recosplat.model.adapter.types import AccumulatingGaussians
from recosplat.model.ar.chunk_schedule import ChunkScheduler
from recosplat.model.encoder import EncoderCfg, EncoderOutput, ReCoSplatEncoder
from recosplat.model.geometry import get_intrinsic_embedding_new, se3_inverse
from recosplat.model.layers.transformer_decoder import TransformerDecoder
from recosplat.model.rendering.splatting_gsplat import DecoderSplattingGSPlat
from recosplat.model.rendering.types import DecoderSplattingSplatCfg


def load_rgb_initialization(encoder: ReCoSplatEncoder, path: Path) -> None:
    """Load an RGB checkpoint or explicitly remove the released nine features.

    Tokenizer channels are [rendered RGB, nine features, observed RGB]. Keeping
    the first six channels would silently discard the observed input images.
    All unrelated keys/shapes must match strictly.
    """
    state = load_file(path, device="cpu")
    key = "guidance_to_tokens.conv_block.0.weight"
    channels = state[key].shape[1]
    if channels == 15:
        state[key] = state[key][:, [0, 1, 2, 12, 13, 14]].contiguous()
        state["first_chunk_guidance_embed"] = state["first_chunk_guidance_embed"][:3].clone()
        state = {k: v for k, v in state.items() if not k.startswith("guidance_feature_head.")}
    elif channels != 6:
        raise ValueError(f"expected 6 RGB-only or 15 released tokenizer channels, got {channels}")
    encoder.load_state_dict(state, strict=True)


class DifferentiableKVCache:
    """Out-of-place cache: preserve gradients to earlier chunks in this layer."""

    def __init__(self):
        self.k = self.v = None

    def get(self):
        return self.k, self.v

    def update(self, new_k, new_v):
        self.k = new_k if self.k is None else torch.cat([self.k, new_k], dim=2)
        self.v = new_v if self.v is None else torch.cat([self.v, new_v], dim=2)


def causal_layer(block, hidden, pos, intervals, tokens_per_view):
    """Run one global layer; a fresh cache also makes checkpoint replay safe."""
    cache = None if block.block_layer_type == "ignore_kv_cache" else DifferentiableKVCache()
    outputs = []
    for chunk, (start, end) in enumerate(intervals):
        lo, hi = start * tokens_per_view, end * tokens_per_view
        outputs.append(
            block(
                hidden[:, lo:hi],
                xpos=pos[:, lo:hi],
                N=end - start,
                kv_cache_manager=cache,
                chunk_first_view_tokens=tokens_per_view,
                is_first_chunk=chunk == 0,
            )
        )
    return torch.cat(outputs, dim=1)


def training_backbone(backbone, images, intrinsics, chunk_size):
    """Freeze DINO; train alternating attention with causal chunk history."""
    b, v, _, h, w = images.shape
    intervals = ChunkScheduler(chunk_size, chunk_size).intervals(v)
    images = rearrange(
        (images - backbone.image_mean) / backbone.image_std, "b v c h w -> (b v) c h w"
    )
    with torch.no_grad():
        features = backbone.encoder.forward_features(images)
    focal = F.relu(backbone.intrinsic_head(features["x_norm_clstoken"]))
    hidden = features["x_norm_patchtokens"]
    embedding = get_intrinsic_embedding_new(
        intrinsics, images, degree=backbone.intrinsics_embed_degree
    )
    hidden = hidden + backbone.intrinsics_embed_layer(embedding)

    registers = []
    for chunk, (start, end) in enumerate(intervals):
        register = backbone.register_token.expand(b, end - start, -1, -1)
        if backbone.register_tokens_mode != "default":
            if chunk == 0:
                register = backbone.register_token_first_chunk.expand(b, end - start, -1, -1)
            elif backbone.use_chunk_last_view_register_tokens:
                register = torch.cat(
                    [register[:, :-1], backbone.register_token_last_view.expand(b, 1, -1, -1)],
                    dim=1,
                )
            else:
                register = torch.cat(
                    [backbone.register_token_first_view.expand(b, 1, -1, -1), register[:, 1:]],
                    dim=1,
                )
        registers.append(register)
    registers = rearrange(torch.cat(registers, dim=1), "b v p d -> (b v) p d")
    hidden = torch.cat([registers, hidden], dim=1)
    tokens = hidden.shape[1]
    pos = backbone.position_getter(
        b * v, h // 14, w // 14, images.device, anchor_hw=backbone.rope_anchor_grid
    )
    pos = torch.cat([pos.new_zeros(b * v, backbone.patch_start_idx, 2), pos + 1], dim=1)
    outputs = []
    for i, block in enumerate(backbone.decoder):
        if i % 2 == 0:
            hidden = checkpoint(
                block, hidden.reshape(b * v, tokens, -1), xpos=pos, N=v, use_reentrant=False
            )
        else:
            hidden = checkpoint(
                partial(causal_layer, block),
                hidden.reshape(b, v * tokens, -1),
                pos.reshape(b, v * tokens, 2),
                intervals,
                tokens,
                use_reentrant=False,
            )
        if i >= len(backbone.decoder) - 2:
            outputs.append(hidden.reshape(b * v, tokens, -1))
    return torch.cat(outputs, dim=-1), pos, focal.reshape(b, v, 2)


class RGBTrainingEncoder(ReCoSplatEncoder):
    """Assemble each chunk with GT poses and RGB render-and-compare guidance."""

    def forward(self, context, chunk_size, global_step=0):
        if self.guidance_feature_dim != 0 or not self.pose_feature_guidance_enabled:
            raise ValueError("RGB training requires enabled ReCo with rendered_feature_dim=0")
        images, intrinsics, poses = (context[k] for k in ("image", "intrinsics", "extrinsics"))
        b, v, _, h, w = images.shape
        if b != 1 or chunk_size < 2 or v < 2:
            raise ValueError(
                "training requires batch size 1 and at least two views in the first chunk"
            )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden, pos, focal = training_backbone(self.backbone, images, intrinsics, chunk_size)
            rgb_features = self.rgb_embed(images.flatten(0, 1))
        predicted = self.chunk_predictor.decode_cameras(
            hidden, pos, self.backbone.patch_start_idx, b, v, h // 14, w // 14
        )
        predicted = se3_inverse(predicted[:, :1]) @ predicted
        scale, pred_scale = self.calculate_gt_scale_factor(predicted, poses, min(chunk_size, v))
        assembly = poses.clone()
        assembly[:, :, :3, 3] = poses[:, :, :3, 3] * scale[:, None, None]
        acc = AccumulatingGaussians(b, v, h, w, feature_dim=0, device=images.device)
        # No pruning in training guidance; preserve faint Gaussians for comparison.
        self.guidance_renderer.opacity_prune_threshold = 0.0
        depths = []

        def select(tensor, start, end):
            return tensor.reshape(b, v, *tensor.shape[1:])[:, start:end].flatten(0, 1)

        for start, end in ChunkScheduler(chunk_size, chunk_size).intervals(v):
            tokens, positions = self._prepare_guidance_for_chunk(
                acc, assembly[:, start:end], intrinsics[:, start:end], images[:, start:end]
            )
            prediction = self.chunk_predictor.predict(
                hidden_chunk=select(hidden, start, end),
                pos_chunk=select(pos, start, end),
                patch_start_idx=self.backbone.patch_start_idx,
                rgb_feat_chunk=select(rgb_features, start, end),
                guidance_tokens=tokens,
                guidance_pos=positions,
                b=b,
                v_chunk=end - start,
                h=h,
                w=w,
                patch_h=h // 14,
                patch_w=w // 14,
                global_step=global_step,
                adapter_poses=assembly[:, start:end],
                c2w_chunk=assembly[:, start:end],
            )
            acc.add_adapter_output(prediction.gaussians)
            depths.append(prediction.depths)
        return EncoderOutput(
            acc.gaussians, predicted, focal, torch.cat(depths, dim=1), True, scale, pred_scale
        )


def pose_loss(predicted, reference):
    """All-pairs relative pose loss with trajectory-diameter scale alignment."""
    scale = ReCoSplatEncoder.calculate_camera_poses_scale
    aligned = predicted.clone()
    aligned[:, :, :3, 3] = (
        predicted[:, :, :3, 3] * (scale(reference) / scale(predicted))[:, None, None]
    )
    pred_rel = se3_inverse(aligned[:, :, None]) @ aligned[:, None, :]
    gt_rel = se3_inverse(reference[:, :, None]) @ reference[:, None, :]
    mask = ~torch.eye(predicted.shape[1], dtype=torch.bool, device=predicted.device)
    pred_rel, gt_rel = pred_rel[:, mask], gt_rel[:, mask]
    residual = pred_rel[..., :3, :3].transpose(-1, -2) @ gt_rel[..., :3, :3]
    cosine = (residual.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    rotation = cosine.clamp(-1 + 1e-6, 1 - 1e-6).acos().mean()
    translation = F.huber_loss(pred_rel[..., :3, 3], gt_rel[..., :3, 3], delta=0.1)
    return rotation + 100 * translation


def sample_indices(num_frames, context_views, target_views, max_span, generator):
    total = context_views + target_views
    if num_frames < total:
        return None
    span_max = min(num_frames, max_span)
    if span_max < total:
        return None
    span = int(torch.randint(total, span_max + 1, (), generator=generator))
    start = int(torch.randint(num_frames - span + 1, (), generator=generator))
    # Keep the interval endpoints as context, targets strictly held out inside it.
    middle = torch.randperm(span - 2, generator=generator)[: total - 2] + start + 1
    context = (
        torch.cat([torch.tensor([start, start + span - 1]), middle[: context_views - 2]])
        .sort()
        .values
    )
    target = middle[context_views - 2 :].sort().values
    return context, target


def training_batches(root, context_views, target_views, max_span, seed):
    """Shuffle train/*.torch files each epoch; never fall back to test data."""
    from torchvision.io import decode_image

    paths = sorted((Path(root) / "train").glob("*.torch"))
    if not paths:
        raise ValueError(f"no train/*.torch files below {root}")
    generator = torch.Generator().manual_seed(seed)
    while True:
        yielded = False
        for i in torch.randperm(len(paths), generator=generator).tolist():
            scenes = torch.load(paths[i], map_location="cpu", weights_only=True, mmap=True)
            for j in torch.randperm(len(scenes), generator=generator).tolist():
                scene = scenes[j]
                indices = sample_indices(
                    len(scene["images"]), context_views, target_views, max_span, generator
                )
                if indices is None:
                    continue
                context, target = indices
                selected = torch.cat([context, target])
                poses, intrinsics = convert_poses(scene["cameras"][selected])
                diameter = torch.pdist(poses[:context_views, :3, 3]).max()
                if not torch.isfinite(diameter) or diameter < 1e-3:
                    continue
                poses[:, :3, 3] /= diameter
                poses = se3_inverse(poses[:1]) @ poses
                images = (
                    torch.stack([decode_image(scene["images"][int(k)]) for k in selected]).float()
                    / 255
                )
                images, intrinsics = rescale_and_crop(images, intrinsics, (224, 224))
                batch = {}
                for name, sl in (
                    ("context", slice(0, context_views)),
                    ("target", slice(context_views, None)),
                ):
                    batch[name] = {
                        "image": images[None, sl],
                        "intrinsics": intrinsics[None, sl],
                        "extrinsics": poses[None, sl],
                    }
                yielded = True
                yield batch
        if not yielded:
            raise ValueError("no usable training scenes for the requested view counts/frame gap")


@hydra.main(version_base=None, config_path="../configs", config_name="train_rgb")
def main(cfg: DictConfig):
    if not torch.cuda.is_available():
        raise ValueError("training requires CUDA")
    if cfg.data_root is None:
        raise ValueError("set TRAIN_ROOT or data_root=/path/to/dataset containing train/*.torch")
    if cfg.context_views < 2 or cfg.target_views < 1 or cfg.chunk_size < 2:
        raise ValueError("context_views/chunk_size must be >=2 and target_views >=1")
    if min(cfg.steps, cfg.save_every, cfg.max_span, cfg.warmup_steps) < 1:
        raise ValueError("steps, save_every, max_span, and warmup_steps must be positive")
    if cfg.input_mode.name != "posed_calibrated":
        raise ValueError(
            "this training recipe uses GT poses and intrinsics; keep input_mode=posed_calibrated"
        )
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(4)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.jsonl").exists():
        raise ValueError("output_dir already contains a run; select a fresh output directory")
    # Validate and prepare a batch before allocating model/optimizer memory.
    batches = training_batches(
        cfg.data_root, cfg.context_views, cfg.target_views, cfg.max_span, cfg.seed
    )
    batch = next(batches)
    encoder = RGBTrainingEncoder(load_typed_config(cfg.model.encoder, EncoderCfg))
    load_rgb_initialization(encoder, Path(cfg.checkpoint))
    encoder.backbone.encoder.requires_grad_(False)
    for module in encoder.modules():
        if isinstance(module, TransformerDecoder):
            module.use_checkpoint = True
    encoder = encoder.cuda().train()
    encoder.backbone.encoder.eval()
    # Rendering supervision needs gradients even for low-opacity Gaussians.
    decoder_cfg = load_typed_config(cfg.model.decoder, DecoderSplattingSplatCfg)
    decoder_cfg.prune_opacity_threshold = 0.0
    decoder = DecoderSplattingGSPlat(decoder_cfg).cuda()
    perceptual = None
    if cfg.loss.lpips:
        import lpips

        perceptual = lpips.LPIPS(net="vgg").cuda().eval().requires_grad_(False)
    parameters = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay, foreach=False
    )
    OmegaConf.save(cfg, output / "config.yaml", resolve=True)
    print(
        f"Training {sum(p.numel() for p in parameters):,} parameters; DINO encoder frozen",
        flush=True,
    )

    for step in range(1, cfg.steps + 1):
        if step > 1:
            batch = next(batches)
        batch = {part: {k: v.cuda() for k, v in views.items()} for part, views in batch.items()}
        warmup = min(step / cfg.warmup_steps, 1.0)
        progress = max(0, step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
        lr = cfg.lr * warmup * (0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        result = encoder(batch["context"], cfg.chunk_size, step)
        target = batch["target"]
        poses = target["extrinsics"].clone()
        poses[:, :, :3, 3] *= result.gt_scale_factor[:, None, None]
        bounds = torch.ones(poses.shape[:2], device="cuda")
        rendered = decoder(
            result.gaussians, poses, target["intrinsics"], bounds * 0.1, bounds * 100, (224, 224)
        ).color
        losses = {
            "mse": F.mse_loss(rendered, target["image"]),
            "intrinsic": F.mse_loss(
                result.pred_intrinsics.float(), batch["context"]["intrinsics"][..., [0, 1], [0, 1]]
            ),
            "pose": pose_loss(result.pred_camera_poses, batch["context"]["extrinsics"]),
            "opacity": result.gaussians.opacities.abs().mean(),
        }
        if perceptual is not None:
            losses["lpips"] = checkpoint(
                perceptual,
                rendered.flatten(0, 1) * 2 - 1,
                target["image"].flatten(0, 1) * 2 - 1,
                use_reentrant=False,
            ).mean()
        loss = sum(cfg.loss[k] * value for k, value in losses.items())
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, cfg.grad_clip, error_if_nonfinite=True
        )
        optimizer.step()
        record = {
            "step": step,
            "loss": loss.item(),
            "lr": lr,
            "grad_norm": grad_norm.item(),
            **{k: v.item() for k, v in losses.items()},
        }
        print(json.dumps(record), flush=True)
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if step % cfg.save_every == 0 or step == cfg.steps:
            path = output / f"rgb-{step:06d}.safetensors"
            temporary = path.with_suffix(".tmp")
            save_file(
                {k: v.detach().cpu().contiguous() for k, v in encoder.state_dict().items()},
                temporary,
            )
            temporary.replace(path)
        del result, rendered, loss, losses


if __name__ == "__main__":
    main()
