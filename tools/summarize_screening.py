"""Collect augmentation-screening runs into one comparison table.

Each screening run writes its own ``history.json`` and ``run_metadata.json``
under an ignored artifacts directory. This tool reads those files and renders a
single ranked comparison so the selection can be checked against the recorded
evidence rather than taken on trust.

It deliberately reads nothing but training-run outputs. No test-set or
external-set number is available to it, which is the point: the screening
decision must be reproducible from source-only evidence.

Usage::

    python tools/summarize_screening.py artifacts/screening --output docs/results/screening.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from asl_recognition.robustness import CORRUPTION_NAMES, INDEPENDENT_CORRUPTIONS


def _load(run_dir: Path) -> dict[str, Any] | None:
    history_path = run_dir / "history.json"
    metadata_path = run_dir / "run_metadata.json"
    if not history_path.is_file() or not metadata_path.is_file():
        return None
    history = json.loads(history_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    best_epoch = int(history["best_epoch"])
    best = next(
        (record for record in history["epochs"] if int(record["epoch"]) == best_epoch),
        None,
    )
    if best is None:
        return None
    breakdown = best.get("stress_per_corruption", {})
    independent = [
        breakdown[name]
        for name in sorted(INDEPENDENT_CORRUPTIONS)
        if name in breakdown and breakdown[name]["total"]
    ]
    independent_accuracy = (
        sum(item["correct"] for item in independent) / sum(item["total"] for item in independent)
        if independent
        else None
    )
    return {
        "run": run_dir.name,
        "profile": metadata["configuration"]["augmentation_profile"],
        "select_on": metadata["configuration"]["select_on"],
        "seed": metadata["configuration"]["seed"],
        "limit_per_class": metadata["configuration"]["limit_per_class"],
        "epochs_completed": history["epochs_completed"],
        "best_epoch": best_epoch,
        "train_accuracy": best["train_accuracy"],
        "validation_accuracy": best["validation_accuracy"],
        "stress_accuracy": best["stress_accuracy"],
        "independent_stress_accuracy": independent_accuracy,
        "stress_per_corruption": {
            name: breakdown[name]["accuracy"] for name in CORRUPTION_NAMES if name in breakdown
        },
        "duration_seconds": history["duration_seconds"],
        "peak_resident_bytes": history.get("peak_resident_bytes"),
        "stress_benchmark_version": history["stress_benchmark"]["version"],
        "stress_row_digest": history["stress_benchmark"].get("row_digest"),
        "stress_row_count": history["stress_benchmark"].get("row_count")
        or metadata.get("samples_used", {}).get("stress"),
        "checkpoint_sha256": history["checkpoint_sha256"],
    }


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def render(runs: list[dict[str, Any]]) -> str:
    ranked = sorted(runs, key=lambda item: item["stress_accuracy"], reverse=True)
    lines = [
        "| Profile | Select on | Best epoch | Clean validation | Stress benchmark "
        "| Independent corruptions |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for run in ranked:
        lines.append(
            f"| `{run['profile']}` | {run['select_on']} | {run['best_epoch']} | "
            f"{_percent(run['validation_accuracy'])} | {_percent(run['stress_accuracy'])} | "
            f"{_percent(run['independent_stress_accuracy'])} |"
        )
    lines.append("")
    lines.append("Per-corruption accuracy at each run's selected epoch:")
    lines.append("")
    header = " | ".join(f"`{name}`" for name in CORRUPTION_NAMES)
    lines.append(f"| Profile | {header} |")
    lines.append("| --- |" + " ---: |" * len(CORRUPTION_NAMES))
    for run in ranked:
        cells = " | ".join(
            _percent(run["stress_per_corruption"].get(name)) for name in CORRUPTION_NAMES
        )
        lines.append(f"| `{run['profile']}` | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Directory holding one subdirectory per run.")
    parser.add_argument("--output", type=Path, help="Write the rendered table here.")
    parser.add_argument("--json-output", type=Path, help="Write the collected records here.")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"screening root not found: {root}")
    runs = [
        record for child in sorted(root.iterdir()) if child.is_dir() if (record := _load(child))
    ]
    if not runs:
        raise SystemExit(f"no completed training runs found under {root}")

    versions = {run["stress_benchmark_version"] for run in runs}
    if len(versions) > 1:
        raise SystemExit(
            f"runs used different stress-benchmark versions {sorted(versions)}; "
            "their scores are not comparable"
        )

    # A shared version string is not enough. The benchmark scores whatever
    # validation rows a run consumed, so a subset-limited run and a full-split
    # run share a version while measuring completely different image sets.
    # Comparability key: the row digest when available, otherwise the number of
    # rows scored. Runs predating digest recording still carry a stress sample
    # count, and a differing count is already proof of a differing row set --
    # which is the exact mistake this guard exists to prevent.
    keys = {(run["stress_row_digest"], run["stress_row_count"]) for run in runs}
    if len(keys) > 1:
        counts = sorted(key[1] for key in keys)
        raise SystemExit(
            f"runs scored different stress row sets (row counts {counts}); their stress "
            "numbers are not comparable. Summarise each set of runs separately."
        )
    if all(run["stress_row_digest"] is None for run in runs):
        print(
            "warning: these runs predate row-digest recording; comparability is "
            "inferred from row count alone",
            file=sys.stderr,
        )

    selections = {run["select_on"] for run in runs}
    if len(selections) > 1:
        raise SystemExit(
            f"runs used different selection metrics {sorted(selections)}; a run that "
            "selected its epoch by stress reports a maximum over epochs, so ranking it "
            "against one that did not is unfair"
        )

    table = render(runs)
    print(table)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table, encoding="utf-8")
        print(f"wrote {args.output}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(runs, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {args.json_output}")


if __name__ == "__main__":
    main()
