#!/usr/bin/env python3
"""Compare completed evaluations with the reference values."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TOLERANCE = {"psnr": 0.10, "ssim": 0.005, "lpips": 0.005}
POSE_TOLERANCE = 0.005


def rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def check(
    run_root: Path,
    expected_path: Path,
    pose: bool,
    datasets: set[str] | None = None,
) -> list[str]:
    failures = []
    for expected in rows(expected_path):
        protocol = expected["protocol"]
        if datasets is not None and protocol.split("_", 1)[0] not in datasets:
            continue
        tag = f"pose_{protocol}" if pose else protocol
        summary_path = run_root / expected["input_mode"] / tag / "summary.json"
        if not summary_path.is_file():
            failures.append(f"missing {summary_path}")
            continue
        actual = json.loads(summary_path.read_text())
        if int(actual["num_scenes"]) != int(expected["num_scenes"]):
            failures.append(
                f"{tag}: expected {expected['num_scenes']} scenes, "
                f"found {int(actual['num_scenes'])}"
            )
        metrics = ("pose_auc@5", "pose_auc@10", "pose_auc@20") if pose else TOLERANCE
        for metric in metrics:
            tolerance = POSE_TOLERANCE if pose else TOLERANCE[metric]
            delta = abs(float(actual[metric]) - float(expected[metric]))
            if delta > tolerance:
                failures.append(
                    f"{tag}/{expected['input_mode']} {metric}: "
                    f"expected {expected[metric]}, found {actual[metric]:.6f}, "
                    f"delta {delta:.6f} > {tolerance}"
                )
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "evaluation_root",
        type=Path,
        help="directory containing <checkpoint-stem>/<input-mode>/...",
    )
    parser.add_argument("--checkpoint-stem", default="recosplat")
    parser.add_argument("--suite", choices=("quality", "pose", "all"), default="all")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("acid", "dl3dv", "re10k", "scannet", "scannetpp"),
    )
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    run_root = args.evaluation_root / args.checkpoint_stem
    datasets = set(args.datasets) if args.datasets is not None else None
    failures = []
    if args.suite in {"quality", "all"}:
        failures.extend(
            check(
                run_root,
                repository / "results/expected_quality.csv",
                pose=False,
                datasets=datasets,
            )
        )
    if args.suite in {"pose", "all"}:
        failures.extend(
            check(
                run_root,
                repository / "results/expected_pose.csv",
                pose=True,
                datasets=datasets,
            )
        )
    if failures:
        raise SystemExit("verification failed:\n- " + "\n- ".join(failures))
    print("verification passed")


if __name__ == "__main__":
    main()
