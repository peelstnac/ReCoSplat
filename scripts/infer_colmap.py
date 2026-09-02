#!/usr/bin/env python3
"""Run ReCoSplat on an undistorted COLMAP scene and export a 3DGS model."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import torch
from omegaconf import OmegaConf

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from recosplat.checkpoints import load_encoder_weights
from recosplat.config.loading import load_typed_config
from recosplat.config.root import ModelCfg
from recosplat.data.colmap import (
    load_colmap_model,
    prepare_views,
    save_viewer_cameras,
    select_images,
)
from recosplat.gaussian_export import (
    gaussian_ply_attributes,
    write_gaussian_ply,
    write_input_ply,
)
from recosplat.model.encoder import ReCoSplatEncoder


def model_config() -> ModelCfg:
    mode = OmegaConf.load(REPOSITORY_ROOT / "configs/input_mode/posed_calibrated.yaml")
    model = OmegaConf.load(REPOSITORY_ROOT / "configs/model/recosplat.yaml")
    resolved = OmegaConf.to_container(
        OmegaConf.create({"input_mode": mode, "model": model}), resolve=True
    )
    return load_typed_config(OmegaConf.create(resolved["model"]), ModelCfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="COLMAP scene containing images/ and sparse/")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/colmap"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("weights/recosplat.safetensors")
    )
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--image-list", type=Path, help="ordered image names, one per line")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-views", type=int)
    parser.add_argument("--chunk-size-first", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--opacity-threshold", type=float, default=0.005)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ReCoSplat inference requires a CUDA GPU")
    if not args.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")
    if args.chunk_size_first < 1 or args.chunk_size < 1:
        raise SystemExit("chunk sizes must be positive")

    cameras, images = load_colmap_model(args.scene)
    selected = select_images(images, args.image_list, args.stride, args.max_views)
    if len(selected) < args.chunk_size_first:
        raise SystemExit(
            f"selected {len(selected)} views; at least {args.chunk_size_first} are required"
        )
    views = prepare_views(args.scene, args.images_dir, selected, cameras)
    context = {
        "image": torch.stack([view.image for view in views]).unsqueeze(0).cuda(),
        "intrinsics": torch.stack([view.intrinsics for view in views]).unsqueeze(0).cuda(),
        "extrinsics": torch.stack([view.c2w for view in views]).unsqueeze(0).cuda(),
    }

    torch.manual_seed(111123)
    cfg = model_config()
    encoder = ReCoSplatEncoder(cfg.encoder)
    load_encoder_weights(encoder, args.checkpoint)
    encoder = encoder.cuda().eval()
    with torch.inference_mode():
        output = encoder.forward_streaming(
            context,
            chunk_size_f=args.chunk_size_first,
            chunk_size_s=args.chunk_size,
        )

    gaussians = output.gaussians
    if output.gt_scale_factor is None:
        raise RuntimeError("posed inference did not return its camera scale")
    camera_scale = output.gt_scale_factor.detach().cpu()
    camera_poses = context["extrinsics"][0].detach().cpu()
    del output, encoder, context
    torch.cuda.empty_cache()

    attributes = gaussian_ply_attributes(gaussians, args.opacity_threshold)
    model_path = args.output_dir / args.scene.name
    ply_path = model_path / "point_cloud/iteration_0/point_cloud.ply"
    write_gaussian_ply(ply_path, attributes)
    write_input_ply(model_path / "input.ply", attributes)

    camera_poses[:, :3, 3] *= camera_scale
    save_viewer_cameras(model_path, views, camera_poses)
    print(f"exported {len(attributes):,} Gaussians from {len(views)} views to {model_path}")


if __name__ == "__main__":
    main()
