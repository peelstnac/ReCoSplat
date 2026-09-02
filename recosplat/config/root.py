from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from recosplat.data.chunk_dataset import ChunkDatasetCfg
from recosplat.evaluation.evaluator import EvalCfg
from recosplat.model.encoder import EncoderCfg
from recosplat.model.rendering.types import DecoderSplattingSplatCfg


@dataclass
class ModelCfg:
    encoder: EncoderCfg
    decoder: DecoderSplattingSplatCfg


@dataclass
class InputModeCfg:
    name: Literal[
        "unposed_uncalibrated", "unposed_calibrated", "posed_calibrated"
    ]
    pose_free: bool
    use_pred_intrinsics_for_embed: bool
    target_pose_mode: Literal["gt", "pred_optimized"]


@dataclass
class RootCfg:
    seed: int
    checkpoint: Path
    output_dir: Path
    input_mode: InputModeCfg
    dataset: ChunkDatasetCfg
    model: ModelCfg
    eval: EvalCfg
