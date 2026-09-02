"""Fixed-index evaluation view sampler."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from jaxtyping import Int64
from torch import Tensor


@dataclass
class IndexEntry:
    context: tuple[int, ...]
    target: tuple[int, ...]


@dataclass
class ViewSamplerEvaluationCfg:
    name: Literal["evaluation"]
    index_path: Path
    chunk_size_f: int
    chunk_size_s: int


class ViewSamplerEvaluation:
    index: dict[str, IndexEntry]

    def __init__(self, cfg: ViewSamplerEvaluationCfg) -> None:
        from .chunk_dataset import scene_hash

        self.cfg = cfg
        with Path(cfg.index_path).open("r") as f:
            raw = json.load(f)

        self.index = {}
        for key, entry in raw.items():
            if entry is None:
                continue
            self.index[scene_hash(key)] = IndexEntry(
                context=tuple(int(i) for i in entry["context"]),
                target=tuple(int(i) for i in entry["target"]),
            )
        if not self.index:
            raise ValueError(f"evaluation index {cfg.index_path} contains no usable entries")

    def sample(
        self, scene: str
    ) -> tuple[Int64[Tensor, " context_view"], Int64[Tensor, " target_view"]] | None:
        """Fixed context/target indices for the scene, or None if it is not in the
        index (the dataset skips it)."""
        from .chunk_dataset import scene_hash

        entry = self.index.get(scene_hash(scene))
        if entry is None:
            return None
        return (
            torch.tensor(entry.context, dtype=torch.int64),
            torch.tensor(entry.target, dtype=torch.int64),
        )
