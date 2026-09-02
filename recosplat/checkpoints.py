"""Strict loading of encoder weights."""

from pathlib import Path

from safetensors.torch import load_file
from torch import nn
def load_encoder_weights(encoder: nn.Module, path: Path) -> None:
    if path.suffix != ".safetensors":
        raise ValueError("checkpoint must be an encoder-only safetensors file")
    state_dict = load_file(path, device="cpu")
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint/model mismatch: {len(missing)} missing keys {missing}; "
            f"{len(unexpected)} unexpected keys {unexpected}"
        )
