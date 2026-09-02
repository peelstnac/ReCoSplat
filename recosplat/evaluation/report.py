"""Aggregate and write evaluation results."""

import csv
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluator import SceneResult

log = logging.getLogger(__name__)


def aggregate_metrics(results: list["SceneResult"]) -> dict[str, float]:
    """Return aggregate numeric metrics and pose-error AUCs."""
    keys = sorted({k for r in results for k in r.metrics})
    aggregate = {
        key: sum(r.metrics[key] for r in results if key in r.metrics)
        / max(1, sum(1 for r in results if key in r.metrics))
        for key in keys
    }
    aggregate["num_scenes"] = float(len(results))

    from .pose_metrics import pose_auc

    for name, per_scene_key in (
        ("rra", "rra_err_deg"),
        ("rta", "rta_err_deg"),
        ("pose", "pose_err_deg"),
    ):
        if all(per_scene_key in r.metrics for r in results):
            errors = [r.metrics[per_scene_key] for r in results]
            for threshold, auc in zip((5, 10, 20), pose_auc(errors, [5, 10, 20])):
                aggregate[f"{name}_auc@{threshold}"] = float(auc)

    return aggregate


def _format_table(aggregate: dict[str, float]) -> str:
    width = max(len(k) for k in aggregate)
    lines = [f"{key:<{width}}  {value:.4f}" for key, value in sorted(aggregate.items())]
    return "\n".join(lines)


def write_report(
    output_dir: Path,
    tag: str,
    aggregate: dict[str, float],
    results: list["SceneResult"],
) -> None:
    """Write per-scene and aggregate metrics under ``output_dir/tag``."""
    tag_dir = output_dir / tag
    tag_dir.mkdir(parents=True, exist_ok=True)

    with (tag_dir / "per_scene.json").open("w") as f:
        json.dump([{"scene": r.scene, **r.metrics} for r in results], f, indent=2)
    with (tag_dir / "summary.json").open("w") as f:
        json.dump(aggregate, f, indent=2)
    csv_path = output_dir / "summary.csv"
    fieldnames = ["tag", *sorted(aggregate)]
    rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open() as f:
            rows = [row for row in csv.DictReader(f) if row.get("tag") != tag]
        for row in rows:
            fieldnames.extend(k for k in row if k not in fieldnames)
    rows.append({"tag": tag, **{k: f"{v:.6f}" for k, v in aggregate.items()}})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info("[eval %s] results -> %s\n%s", tag, tag_dir, _format_table(aggregate))
