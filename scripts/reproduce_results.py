#!/usr/bin/env python3
"""Run the quality or camera-pose evaluation matrix."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Evaluation:
    dataset: str
    protocol: str
    mode: str


QUALITY = [
    *(Evaluation("dl3dv", f"dl3dv_{views}", mode)
      for views in (32, 64, 128, 256)
      for mode in ("unposed_uncalibrated", "unposed_calibrated", "posed_calibrated")),
    *(Evaluation("scannet", "scannet_32", mode)
      for mode in ("unposed_calibrated", "posed_calibrated")),
    *(Evaluation("re10k", f"re10k_{views}", mode)
      for views in (64, 128)
      for mode in ("unposed_calibrated", "posed_calibrated")),
    *(Evaluation("scannetpp", f"scannetpp_{views}", mode)
      for views in (256, 512)
      for mode in ("unposed_calibrated", "posed_calibrated")),
]

POSE = [
    *(Evaluation("acid", f"acid_{views}", "unposed_calibrated")
      for views in (32, 64)),
    *(Evaluation("re10k", f"re10k_{views}", "unposed_calibrated")
      for views in (64, 128)),
    *(Evaluation("dl3dv", f"dl3dv_{views}", "unposed_calibrated")
      for views in (128, 256)),
]

ROOT_ENV = {
    "acid": "ACID_ROOT",
    "dl3dv": "DL3DV_ROOT",
    "re10k": "RE10K_ROOT",
    "scannet": "SCANNET_ROOT",
    "scannetpp": "SCANNETPP_ROOT",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("suite", choices=("quality", "pose", "all"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("weights/recosplat.safetensors")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    evaluations = []
    if args.suite in {"quality", "all"}:
        evaluations.extend((item, True) for item in QUALITY)
    if args.suite in {"pose", "all"}:
        evaluations.extend((item, False) for item in POSE)

    required = sorted({ROOT_ENV[item.dataset] for item, _ in evaluations})
    missing = [name for name in required if not os.environ.get(name)]
    if missing and not args.dry_run:
        raise SystemExit(f"missing dataset environment variables: {', '.join(missing)}")
    if not args.checkpoint.is_file() and not args.dry_run:
        raise SystemExit(f"checkpoint not found: {args.checkpoint}")

    evaluate_script = Path(__file__).with_name("evaluate.py")
    for item, image_metrics in evaluations:
        command = [
            sys.executable,
            str(evaluate_script),
            f"checkpoint={args.checkpoint}",
            f"output_dir={args.output_dir}",
            f"dataset={item.dataset}",
            f"eval={item.protocol}",
            f"input_mode={item.mode}",
            f"eval.compute_image_metrics={str(image_metrics).lower()}",
            f"eval.compute_pose_metrics={str(not image_metrics).lower()}",
        ]
        if not image_metrics:
            command.append(f"eval.tag=pose_{item.protocol}")
        print(" ".join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
