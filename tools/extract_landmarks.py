"""Extract normalised hand-landmark features for a manifest, and cache them.

Detection is the expensive part of the landmark pipeline and it never changes for
a fixed image, so it runs once here and training reads the cache. Each output
records the manifest and feature contract it was built from, so a stale cache
cannot be silently mixed with fresh code.

Requires MediaPipe, a development-only dependency::

    python -m pip install mediapipe
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asl_recognition.constants import CLASS_NAMES  # noqa: E402
from asl_recognition.data import read_manifest  # noqa: E402
from asl_recognition.landmarks import (  # noqa: E402
    FEATURE_DIMENSION,
    HAND_LANDMARKER_TASK_URL,
    create_detector,
    detect_features,
)

FEATURE_CONTRACT = "landmark-v1"


def ensure_task_model(path: Path) -> Path:
    if path.is_file():
        return path
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {HAND_LANDMARKER_TASK_URL}")
    urllib.request.urlretrieve(HAND_LANDMARKER_TASK_URL, path)  # noqa: S310
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--limit-per-class",
        type=int,
        help="Take this many evenly spaced images per class. Landmark models need "
        "far fewer examples than pixel models, so a subset is usually enough.",
    )
    parser.add_argument("--task-model", type=Path, default=Path("artifacts/hand_landmarker.task"))
    args = parser.parse_args()

    from PIL import Image

    rows = read_manifest(args.manifest)
    if args.limit_per_class:
        from asl_recognition.training import _stratified_indices

        rows = [rows[index] for index in _stratified_indices(rows, args.limit_per_class)]

    detector = create_detector(ensure_task_model(args.task_model))
    features: list[list[float]] = []
    labels: list[int] = []
    paths: list[str] = []
    missed: dict[str, int] = {label: 0 for label in CLASS_NAMES}

    started = time.perf_counter()
    for index, row in enumerate(rows, start=1):
        image_path = args.source_root / Path(*row["path"].split("/"))
        with Image.open(image_path) as opened:
            vector = detect_features(detector, opened)
        if vector is None:
            missed[row["label"]] += 1
            continue
        features.append(vector)
        labels.append(CLASS_NAMES.index(row["label"]))
        paths.append(row["path"])
        if index % 1000 == 0:
            rate = index / (time.perf_counter() - started)
            remaining = (len(rows) - index) / rate
            print(
                f"  {index}/{len(rows)}  {rate:.1f} img/s  ~{remaining / 60:.1f} min left",
                flush=True,
            )
    detector.close()

    import numpy as np

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        paths=np.asarray(paths),
    )
    summary = {
        "contract": FEATURE_CONTRACT,
        "feature_dimension": FEATURE_DIMENSION,
        "manifest": str(args.manifest),
        "source_root": str(args.source_root),
        "images_considered": len(rows),
        "features_extracted": len(features),
        # A miss is not a neutral omission: in a camera pipeline it is a failure
        # to answer, so the detection rate belongs beside any accuracy claim.
        "detection_rate": len(features) / len(rows) if rows else 0.0,
        "missed_per_class": {label: count for label, count in missed.items() if count},
        "duration_seconds": time.perf_counter() - started,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
