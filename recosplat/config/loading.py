"""Strict YAML-to-dataclass configuration loading."""

from pathlib import Path
from typing import TypeVar

from dacite import Config, from_dict
from omegaconf import DictConfig, OmegaConf

from .root import RootCfg

T = TypeVar("T")


def load_typed_config(cfg: DictConfig, data_class: type[T]) -> T:
    return from_dict(
        data_class,
        OmegaConf.to_container(cfg, resolve=True),
        config=Config(strict=True, cast=[Path, tuple]),
    )


def load_typed_root_config(cfg: DictConfig) -> RootCfg:
    typed = load_typed_config(cfg, RootCfg)
    mode = typed.input_mode
    if typed.model.encoder.pose_free != mode.pose_free:
        raise ValueError("input mode and model.encoder.pose_free disagree")
    if (
        typed.model.encoder.backbone.use_pred_intrinsics_for_embed
        != mode.use_pred_intrinsics_for_embed
    ):
        raise ValueError("input mode and intrinsic-embedding behavior disagree")
    if typed.eval.pose_modes[0] != mode.target_pose_mode:
        raise ValueError("input mode and primary target pose mode disagree")
    return typed
