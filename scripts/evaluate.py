"""Evaluate ReCoSplat on one protocol."""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")
os.environ.setdefault("HYDRA_FULL_ERROR", "1")

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from recosplat.checkpoints import load_encoder_weights
from recosplat.config import load_typed_root_config
from recosplat.evaluation.evaluator import Evaluator
from recosplat.evaluation.report import write_report
from recosplat.model.encoder import ReCoSplatEncoder
from recosplat.model.rendering.splatting_cuda import DecoderSplattingCUDA
from recosplat.model.rendering.splatting_gsplat import DecoderSplattingGSPlat


@hydra.main(version_base=None, config_path="../configs", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    cfg = load_typed_root_config(cfg_dict)
    if not torch.cuda.is_available():
        raise SystemExit("ReCoSplat evaluation requires a CUDA GPU")
    if not cfg.checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {cfg.checkpoint}")

    torch.manual_seed(cfg.seed)
    encoder = ReCoSplatEncoder(cfg.model.encoder)
    load_encoder_weights(encoder, cfg.checkpoint)
    encoder = encoder.cuda().eval()
    decoder_cls = (
        DecoderSplattingCUDA
        if cfg.model.decoder.name == "splatting_cuda"
        else DecoderSplattingGSPlat
    )
    decoder = decoder_cls(cfg.model.decoder).cuda().eval()

    evaluator = Evaluator(cfg.eval, cfg.dataset, encoder, decoder)
    run_dir = cfg.output_dir / cfg.checkpoint.stem / cfg.input_mode.name
    aggregate, results = evaluator.run()
    write_report(run_dir, cfg.eval.tag, aggregate, results)


if __name__ == "__main__":
    main()
