#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

# Setup: uv sync --frozen
# Checkpoint: uvx --from huggingface_hub hf download chfrn/ReCoSplat recosplat.safetensors --local-dir weights
# Train: TRAIN_ROOT=/path/to/chunks CUDA_VISIBLE_DEVICES=0 bash scripts/train_rgb.sh
# Overrides: bash scripts/train_rgb.sh context_views=8 chunk_size=4 output_dir=outputs/rgb_small
#
# Evaluate an export (set DL3DV_ROOT to the evaluation dataset):
# uv run python scripts/evaluate.py checkpoint=outputs/train_rgb/rgb-150000.safetensors \
#   model.encoder.pose_feature_guidance.rendered_feature_dim=0 \
#   dataset=dl3dv eval=dl3dv_128 input_mode=posed_calibrated
# Use input_mode=unposed_calibrated to evaluate without supplied poses.

exec uv run python scripts/train_rgb.py "$@"
