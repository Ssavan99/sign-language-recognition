"""Split the external set into a dev half and a final half, once and reproducibly.

Why this exists
---------------

The 780-image external capture set is the project's only unbiased measurement of
generalisation, and it has already been scored twice. Every further experiment
run against the whole set erodes it: a blind test that gets consulted repeatedly
quietly becomes a validation set, and any number produced from it afterwards
overstates what the model would do on genuinely unseen data.

So the set is divided once. Experiments iterate on the **dev** half. The
**final** half is scored at most once per published model, at the end.

The split is deterministic from a fixed seed and stratified by class, so it can
be regenerated exactly and cannot be quietly re-drawn to flatter a result. Its
row digests are printed so a reported number can be tied to the half it came
from.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asl_recognition.constants import MANIFEST_FIELDS  # noqa: E402
from asl_recognition.data import read_manifest  # noqa: E402

SPLIT_SEED = 20260820


def _digest(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f"{row['path']}\x00{row['sha256']}\n".encode())
    return digest.hexdigest()


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/external.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/manifests-external-split"))
    parser.add_argument("--record", type=Path, default=Path("docs/results/external-split.json"))
    args = parser.parse_args()

    rows = read_manifest(args.manifest)
    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    dev: list[dict[str, str]] = []
    final: list[dict[str, str]] = []
    for label in sorted(by_label):
        # Sort before shuffling so the input order of the manifest cannot change
        # which images land in which half.
        group = sorted(by_label[label], key=lambda item: item["path"])
        random.Random(f"{SPLIT_SEED}:{label}").shuffle(group)
        half = len(group) // 2
        dev.extend(group[:half])
        final.extend(group[half:])

    dev.sort(key=lambda item: item["path"])
    final.sort(key=lambda item: item["path"])

    _write(args.output_dir / "dev.csv", dev)
    _write(args.output_dir / "final.csv", final)

    record = {
        "_comment": (
            "Fixed division of the external capture set. Experiments iterate on the dev "
            "half; the final half is scored at most once per published model. Regenerate "
            "with tools/split_external_holdout.py -- the seed is fixed, so the halves "
            "cannot be re-drawn to flatter a result."
        ),
        "seed": SPLIT_SEED,
        "source_manifest": str(args.manifest),
        "dev": {"count": len(dev), "row_digest": _digest(dev)},
        "final": {"count": len(final), "row_digest": _digest(final)},
    }
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    overlap = {row["sha256"] for row in dev} & {row["sha256"] for row in final}
    assert not overlap, f"dev and final share {len(overlap)} images"
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
