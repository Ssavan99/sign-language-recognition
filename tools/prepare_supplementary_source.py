"""Normalise a supplementary A-Z image source into the layout `prepare` expects.

The maintained pipeline discovers class roots by looking for uppercase A-Z
directories. A second corpus may use lowercase names, carry non-letter classes,
or ship a duplicated nested copy of itself. Rather than loosening discovery --
which would make the primary path more permissive for everyone -- this tool
writes a clean normalised tree that the unchanged `prepare` command can consume.

It refuses to include an image whose bytes already appear in any existing
manifest. Adding training data that overlaps the held-out test or the blind
external set would silently inflate the very scores the addition is meant to
improve.

Usage::

    python tools/prepare_supplementary_source.py \\
        --source data/raw/kagglehub/datasets/ayuraj/asl-dataset/versions/1/asl_dataset \\
        --output data/raw/supplement \\
        --manifest-dir data/manifests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asl_recognition.constants import CLASS_NAMES, IMAGE_EXTENSIONS  # noqa: E402
from asl_recognition.data import read_manifest  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _class_directory(source: Path, label: str) -> Path | None:
    """Find this label's directory without relying on filesystem case rules."""

    for child in sorted(source.iterdir()):
        if child.is_dir() and child.name.casefold() == label.casefold():
            return child
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Copy images even if they duplicate existing manifest rows (not recommended).",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"source not found: {source}")

    existing: dict[str, str] = {}
    for split in ("train", "validation", "test", "external"):
        manifest = args.manifest_dir / f"{split}.csv"
        if manifest.is_file():
            for row in read_manifest(manifest):
                existing[row["sha256"]] = split
    print(f"loaded {len(existing)} existing manifest checksums")

    if output.exists():
        shutil.rmtree(output)

    seen: dict[str, Path] = {}
    counts: dict[str, int] = {}
    collisions: dict[str, int] = defaultdict(int)
    internal_duplicates = 0

    for label in CLASS_NAMES:
        directory = _class_directory(source, label)
        if directory is None:
            raise SystemExit(f"no directory for class {label!r} under {source}")
        destination = output / label
        destination.mkdir(parents=True, exist_ok=True)
        kept = 0
        # Sorted for determinism; only this level, so a nested duplicate copy of
        # the whole corpus is never walked into.
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.casefold() not in IMAGE_EXTENSIONS:
                continue
            digest = _sha256(path)
            if digest in existing:
                collisions[existing[digest]] += 1
                if not args.allow_overlap:
                    continue
            if digest in seen:
                internal_duplicates += 1
                continue
            seen[digest] = path
            shutil.copy2(path, destination / f"{label}_{kept:04d}{path.suffix.lower()}")
            kept += 1
        counts[label] = kept
        if kept == 0:
            raise SystemExit(f"class {label!r} kept no images; refusing to write an empty class")

    summary = {
        "source": str(source),
        "output": str(output),
        "classes": len(counts),
        "total_images": sum(counts.values()),
        "per_class": counts,
        "internal_duplicates_skipped": internal_duplicates,
        "overlap_with_existing_splits": dict(collisions),
    }
    (output / "_supplement_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if collisions and not args.allow_overlap:
        print(f"\nskipped {sum(collisions.values())} images already present in existing splits")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
