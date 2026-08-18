"""Deterministic data discovery, manifest creation, and image loading.

Heavy machine-learning dependencies are intentionally imported inside the
functions that need them. This keeps package and CLI discovery usable before
PyTorch and torchvision are installed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .constants import (
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    IMAGE_MEAN,
    IMAGE_STD,
    MANIFEST_FIELDS,
)


class DatasetLayoutError(ValueError):
    """Raised when a dataset cannot be mapped unambiguously to A-Z classes."""


@dataclass(frozen=True)
class _ClassRoot:
    path: Path
    role: str | None


@dataclass(frozen=True)
class _ImageRecord:
    path: Path
    relative_path: str
    label: str
    sha256: str
    dhash: str


def _require_directory(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise DatasetLayoutError(f"{description} does not exist: {resolved}")
    if not resolved.is_dir():
        raise DatasetLayoutError(f"{description} is not a directory: {resolved}")
    return resolved


def _role_for_root(path: Path, source_root: Path) -> str | None:
    """Infer a conventional split role from the candidate's short ancestry."""

    relative_parts = [part.casefold() for part in path.relative_to(source_root).parts]
    for part in reversed(relative_parts):
        normalized = part.replace("-", "_")
        if normalized in {"validation", "valid", "val", "asl_alphabet_validation"}:
            return "validation"
        if normalized in {"test", "testing", "asl_alphabet_test"}:
            return "test"
        if normalized in {"train", "training", "asl_alphabet_train"}:
            return "train"
    return None


def _has_all_class_directories(path: Path) -> bool:
    return all((path / label).is_dir() for label in CLASS_NAMES)


def _discover_class_roots(source_root: Path) -> list[_ClassRoot]:
    """Find A-Z class roots through a small, bounded wrapper hierarchy.

    Three wrapper levels cover both ``asl_alphabet_train/A`` and the common
    extracted ``asl_alphabet_train/asl_alphabet_train/A`` layout, as well as a
    dataset wrapper around conventional ``train/A`` and ``test/A`` roots. The
    bound also accepts this repository's historical double-wrapper extraction.
    """

    candidates: list[Path] = []
    frontier = [source_root]
    for depth in range(4):
        next_frontier: list[Path] = []
        for directory in sorted(
            frontier, key=lambda item: (item.as_posix().casefold(), item.as_posix())
        ):
            if _has_all_class_directories(directory):
                candidates.append(directory)
                continue
            if depth < 3:
                try:
                    next_frontier.extend(child for child in directory.iterdir() if child.is_dir())
                except OSError as exc:
                    raise DatasetLayoutError(
                        f"Cannot inspect dataset directory {directory}: {exc}"
                    ) from exc
        frontier = next_frontier

    unique = sorted(set(candidates), key=lambda item: (item.as_posix().casefold(), item.as_posix()))
    if not unique:
        expected = ", ".join(CLASS_NAMES)
        raise DatasetLayoutError(
            f"No directory containing all A-Z class folders was found within "
            f"three levels of {source_root}. Expected immediate folders: {expected}."
        )
    return [_ClassRoot(path, _role_for_root(path, source_root)) for path in unique]


def _select_internal_roots(roots: Sequence[_ClassRoot]) -> dict[str, _ClassRoot]:
    """Resolve one pooled root or an explicit train/validation/test layout."""

    if len(roots) == 1:
        return {"pool": roots[0]}

    by_role: dict[str, list[_ClassRoot]] = defaultdict(list)
    ambiguous: list[_ClassRoot] = []
    for root in roots:
        if root.role is None:
            ambiguous.append(root)
        else:
            by_role[root.role].append(root)

    duplicates = {role: values for role, values in by_role.items() if len(values) > 1}
    if duplicates or ambiguous or "train" not in by_role:
        rendered = ", ".join(f"{root.path} ({root.role or 'unclassified'})" for root in roots)
        raise DatasetLayoutError(
            "Dataset contains multiple possible A-Z roots that cannot be selected "
            f"unambiguously: {rendered}"
        )
    if "test" not in by_role:
        raise DatasetLayoutError(
            "A multi-root source layout must include both train/A-Z and test/A-Z "
            "partitions. Use one pooled A-Z root to generate all three splits."
        )
    return {role: values[0] for role, values in by_role.items()}


def _iter_class_images(class_root: Path, source_root: Path) -> Iterator[tuple[Path, str, str]]:
    for label in CLASS_NAMES:
        class_directory = class_root / label
        images = sorted(
            (
                path
                for path in class_directory.rglob("*")
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
            ),
            key=lambda item: (
                item.relative_to(source_root).as_posix().casefold(),
                item.relative_to(source_root).as_posix(),
            ),
        )
        if not images:
            raise DatasetLayoutError(
                f"Class {label!r} contains no JPG, JPEG, or PNG images: {class_directory}"
            )
        for image_path in images:
            try:
                image_path.resolve().relative_to(source_root)
            except ValueError as exc:
                raise DatasetLayoutError(
                    f"Image path resolves outside the dataset source root: {image_path}"
                ) from exc
            yield image_path, image_path.relative_to(source_root).as_posix(), label


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetLayoutError(f"Cannot read image file {path}: {exc}") from exc
    return digest.hexdigest()


def _dhash_file(path: Path) -> str:
    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "Image manifest preparation requires Pillow. Install the project data "
            "dependencies (for example: pip install Pillow)."
        ) from exc

    try:
        with Image.open(path) as image:
            grayscale = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            flattened = getattr(grayscale, "get_flattened_data", None)
            pixels = list(flattened() if flattened is not None else grayscale.getdata())
    except (OSError, UnidentifiedImageError) as exc:
        raise DatasetLayoutError(f"Image is unreadable or invalid: {path} ({exc})") from exc

    value = 0
    for row in range(8):
        offset = row * 9
        for column in range(8):
            value = (value << 1) | int(pixels[offset + column] > pixels[offset + column + 1])
    return f"{value:016x}"


def _load_records(class_root: Path, source_root: Path) -> list[_ImageRecord]:
    records: list[_ImageRecord] = []
    for path, relative_path, label in _iter_class_images(class_root, source_root):
        records.append(
            _ImageRecord(
                path=path,
                relative_path=relative_path,
                label=label,
                sha256=_sha256_file(path),
                dhash=_dhash_file(path),
            )
        )
    return records


def _validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
        or value >= 1
        for value in ratios
    ):
        raise ValueError("train, validation, and test ratios must each be between 0 and 1")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("train, validation, and test ratios must sum to 1.0")


def _apportion(total: int, ratios: Sequence[float]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [int(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(range(len(ratios)), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remaining]:
        counts[index] += 1

    # A stratified split should retain every class in every requested partition
    # whenever that is mathematically possible. Largest-remainder rounding alone
    # can otherwise produce zero validation/test samples in small smoke fixtures.
    if total >= len(ratios):
        for empty_index in (index for index, count in enumerate(counts) if count == 0):
            donors = [index for index, count in enumerate(counts) if count > 1]
            donor = max(
                donors,
                key=lambda index: (counts[index] - raw[index], counts[index], -index),
            )
            counts[donor] -= 1
            counts[empty_index] += 1
    return counts


def _assign_groups(
    records: Sequence[_ImageRecord],
    splits: Sequence[str],
    ratios: Sequence[float],
    seed: int,
) -> dict[str, list[_ImageRecord]]:
    """Assign exact-content groups deterministically while tracking target sizes."""

    by_hash: dict[str, list[_ImageRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)

    result: dict[str, list[_ImageRecord]] = {split: [] for split in splits}
    for label in CLASS_NAMES:
        groups = [group for group in by_hash.values() if group[0].label == label]
        if len(groups) < len(splits):
            raise DatasetLayoutError(
                f"Class {label!r} has only {len(groups)} distinct exact-content image "
                f"group(s), but {len(splits)} non-empty splits were requested. Add "
                "distinct images or change the source partitioning."
            )
        groups.sort(
            key=lambda group: hashlib.sha256(
                f"{seed}:{label}:{group[0].sha256}".encode()
            ).hexdigest()
        )
        total = sum(len(group) for group in groups)
        targets = _apportion(total, ratios)
        assigned = [0] * len(splits)

        for group_index, group in enumerate(groups):
            size = len(group)
            empty_splits = [index for index, count in enumerate(assigned) if count == 0]
            remaining_groups = len(groups) - group_index
            candidates = (
                empty_splits if remaining_groups == len(empty_splits) else range(len(splits))
            )
            scored_candidates = []
            for index in candidates:
                hypothetical = assigned.copy()
                hypothetical[index] += size
                error = sum(
                    (hypothetical[item] - targets[item]) ** 2 for item in range(len(splits))
                )
                remaining = targets[index] - assigned[index]
                scored_candidates.append(((error, -remaining, index), index))
            chosen = min(scored_candidates)[1]
            result[splits[chosen]].extend(group)
            assigned[chosen] += size

    for values in result.values():
        values.sort(key=lambda record: (record.label, record.relative_path.casefold()))
    return result


def _validate_hash_labels(records: Iterable[_ImageRecord]) -> None:
    labels_by_hash: dict[str, set[str]] = defaultdict(set)
    paths_by_hash: dict[str, list[str]] = defaultdict(list)
    for record in records:
        labels_by_hash[record.sha256].add(record.label)
        paths_by_hash[record.sha256].append(record.relative_path)
    conflicts = [digest for digest, labels in labels_by_hash.items() if len(labels) > 1]
    if conflicts:
        digest = sorted(conflicts)[0]
        raise DatasetLayoutError(
            "Identical image bytes occur under different class labels; refusing to "
            f"create a mislabeled split. SHA-256 {digest}: {paths_by_hash[digest]}"
        )


def _validate_no_source_split_duplicates(split_records: dict[str, list[_ImageRecord]]) -> None:
    locations: dict[str, set[str]] = defaultdict(set)
    paths: dict[str, list[str]] = defaultdict(list)
    for split, records in split_records.items():
        for record in records:
            locations[record.sha256].add(split)
            paths[record.sha256].append(record.relative_path)
    conflicts = [digest for digest, splits in locations.items() if len(splits) > 1]
    if conflicts:
        digest = sorted(conflicts)[0]
        raise DatasetLayoutError(
            "An exact duplicate crosses source-provided partitions. The test contract "
            "will not be mutated automatically; remove the duplicate or provide a "
            f"single unsplit class root. SHA-256 {digest}: {paths[digest]}"
        )


class _BKNode:
    __slots__ = ("value", "indices", "children")

    def __init__(self, value: int, index: int) -> None:
        self.value = value
        self.indices = [index]
        self.children: dict[int, _BKNode] = {}


class _BKTree:
    """Small integer BK-tree used to avoid quadratic perceptual comparisons."""

    def __init__(self, values: Sequence[int]) -> None:
        self.root: _BKNode | None = None
        for index, value in enumerate(values):
            self.add(value, index)

    @staticmethod
    def _distance(left: int, right: int) -> int:
        return (left ^ right).bit_count()

    def add(self, value: int, index: int) -> None:
        if self.root is None:
            self.root = _BKNode(value, index)
            return
        node = self.root
        while True:
            distance = self._distance(value, node.value)
            if distance == 0:
                node.indices.append(index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(value, index)
                return
            node = child

    def query(self, value: int, threshold: int) -> Iterator[tuple[int, int]]:
        if self.root is None:
            return
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = self._distance(value, node.value)
            if distance <= threshold:
                for index in node.indices:
                    yield index, distance
            low, high = distance - threshold, distance + threshold
            pending.extend(child for edge, child in node.children.items() if low <= edge <= high)


def _duplicate_report(
    split_records: dict[str, list[_ImageRecord]], threshold: int, sample_limit: int = 100
) -> dict[str, Any]:
    split_names = [name for name in ("train", "validation", "test") if name in split_records]
    exact_locations: dict[str, set[str]] = defaultdict(set)
    exact_counts: Counter[str] = Counter()
    for split in split_names:
        for record in split_records[split]:
            exact_locations[record.sha256].add(split)
            exact_counts[record.sha256] += 1
    exact_cross_split = sum(1 for values in exact_locations.values() if len(values) > 1)
    if exact_cross_split:
        raise RuntimeError("Internal error: exact duplicate images crossed generated splits")

    candidate_count = 0
    samples: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(split_names):
        left = split_records[left_name]
        for right_name in split_names[left_index + 1 :]:
            right = split_records[right_name]
            left_by_dhash: dict[int, list[_ImageRecord]] = defaultdict(list)
            right_by_dhash: dict[int, list[_ImageRecord]] = defaultdict(list)
            for record in left:
                left_by_dhash[int(record.dhash, 16)].append(record)
            for record in right:
                right_by_dhash[int(record.dhash, 16)].append(record)
            right_hashes = sorted(right_by_dhash)
            tree = _BKTree(right_hashes)
            for left_hash in sorted(left_by_dhash):
                left_records = left_by_dhash[left_hash]
                for index, distance in tree.query(left_hash, threshold):
                    right_records = right_by_dhash[right_hashes[index]]
                    candidate_count += len(left_records) * len(right_records)
                    remaining_samples = sample_limit - len(samples)
                    if remaining_samples <= 0:
                        continue
                    for left_record in left_records:
                        for right_record in right_records:
                            samples.append(
                                {
                                    "left_path": left_record.relative_path,
                                    "left_split": left_name,
                                    "left_label": left_record.label,
                                    "right_path": right_record.relative_path,
                                    "right_split": right_name,
                                    "right_label": right_record.label,
                                    "hamming_distance": distance,
                                }
                            )
                            if len(samples) >= sample_limit:
                                break
                        if len(samples) >= sample_limit:
                            break

    return {
        "exact_cross_split_duplicates": 0,
        "exact_duplicate_group_count": sum(1 for count in exact_counts.values() if count > 1),
        "exact_duplicate_extra_image_count": sum(
            count - 1 for count in exact_counts.values() if count > 1
        ),
        "near_duplicate_threshold": threshold,
        "perceptual_cross_split_candidate_count": candidate_count,
        "perceptual_cross_split_candidates_are_review_candidates": True,
        "perceptual_cross_split_candidate_sample_limit": sample_limit,
        "perceptual_cross_split_candidate_samples": samples,
        "method": "64-bit difference hash (dHash) indexed with a BK-tree",
        "limitations": (
            "The source images do not provide signer or session identifiers, so "
            "signer-disjoint or session-disjoint grouping cannot be verified."
        ),
    }


def _row(record: _ImageRecord, split: str) -> dict[str, str]:
    return {
        "path": record.relative_path,
        "label": record.label,
        "split": split,
        "sha256": record.sha256,
        "dhash": record.dhash,
    }


def _atomic_write_csv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _file_digest(path: Path) -> str:
    return _sha256_file(path)


def prepare_manifests(
    source_root: Path,
    output_dir: Path,
    seed: int = 42,
    train_ratio: float = 0.70,
    validation_ratio: float = 0.15,
    test_ratio: float = 0.15,
    near_duplicate_threshold: int = 5,
) -> dict[str, Any]:
    """Discover A-Z images and write deterministic train/validation/test manifests.

    When the source contains explicit train and test roots, the test images stay in
    that upstream partition and only the train root is divided into train and
    validation data. With one pooled A-Z root, all three splits are generated.
    """

    _validate_ratios(train_ratio, validation_ratio, test_ratio)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if (
        isinstance(near_duplicate_threshold, bool)
        or not isinstance(near_duplicate_threshold, int)
        or not 0 <= near_duplicate_threshold <= 5
    ):
        raise ValueError("near_duplicate_threshold must be an integer from 0 through 5")

    source = _require_directory(Path(source_root), "Dataset source root")
    destination = Path(output_dir).expanduser().resolve()
    roots = _select_internal_roots(_discover_class_roots(source))

    if "pool" in roots:
        records = _load_records(roots["pool"].path, source)
        _validate_hash_labels(records)
        split_records = _assign_groups(
            records,
            ("train", "validation", "test"),
            (train_ratio, validation_ratio, test_ratio),
            seed,
        )
        layout = "pooled"
    else:
        source_records = {role: _load_records(root.path, source) for role, root in roots.items()}
        all_records = [record for values in source_records.values() for record in values]
        _validate_hash_labels(all_records)
        _validate_no_source_split_duplicates(source_records)
        if "validation" in source_records:
            split_records = {
                "train": source_records["train"],
                "validation": source_records["validation"],
                "test": source_records.get("test", []),
            }
        else:
            internal_total = train_ratio + validation_ratio
            generated = _assign_groups(
                source_records["train"],
                ("train", "validation"),
                (train_ratio / internal_total, validation_ratio / internal_total),
                seed,
            )
            split_records = {
                **generated,
                "test": source_records.get("test", []),
            }
        if not split_records["test"]:
            raise DatasetLayoutError(
                "A multi-root source layout must include a non-empty test/A-Z root. "
                "For generated train/validation/test splits, provide one pooled A-Z root."
            )
        for values in split_records.values():
            values.sort(key=lambda record: (record.label, record.relative_path.casefold()))
        layout = "source_partitions"

    report = _duplicate_report(split_records, near_duplicate_threshold)
    manifest_paths: dict[str, Path] = {}
    manifest_hashes: dict[str, str] = {}
    for split in ("train", "validation", "test"):
        path = destination / f"{split}.csv"
        _atomic_write_csv(path, [_row(record, split) for record in split_records[split]])
        manifest_paths[split] = path
        manifest_hashes[split] = _file_digest(path)

    report_path = destination / "duplicate_report.json"
    _atomic_write_json(report_path, report)

    per_split = {split: len(values) for split, values in split_records.items()}
    total_count = sum(per_split.values())
    per_class = {
        split: dict(Counter(record.label for record in values))
        for split, values in split_records.items()
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "source_root": str(source),
        "layout": layout,
        "class_names": list(CLASS_NAMES),
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": test_ratio,
        },
        "counts": {
            "total": total_count,
            "per_split": per_split,
            "per_class_by_split": per_class,
        },
        "actual_ratios": {split: count / total_count for split, count in per_split.items()},
        "manifest_sha256": manifest_hashes,
        "duplicate_report_sha256": _file_digest(report_path),
        "source_class_roots": {
            role: root.path.relative_to(source).as_posix() or "." for role, root in roots.items()
        },
    }
    metadata_path = destination / "metadata.json"
    _atomic_write_json(metadata_path, metadata)

    return {
        **metadata,
        "manifest_paths": {split: str(path) for split, path in manifest_paths.items()},
        "metadata_path": str(metadata_path),
        "duplicate_report_path": str(report_path),
    }


def _select_external_root(roots: Sequence[_ClassRoot]) -> _ClassRoot:
    if len(roots) == 1:
        return roots[0]
    test_roots = [root for root in roots if root.role == "test"]
    if len(test_roots) == 1:
        return test_roots[0]
    rendered = ", ".join(str(root.path) for root in roots)
    raise DatasetLayoutError(f"External dataset has ambiguous A-Z class roots: {rendered}")


def prepare_external_manifest(source_root: Path, output_file: Path) -> dict[str, Any]:
    """Create a checksummed manifest for a separate, class-aligned A-Z dataset."""

    source = _require_directory(Path(source_root), "External dataset source root")
    selected = _select_external_root(_discover_class_roots(source))
    records = _load_records(selected.path, source)
    _validate_hash_labels(records)
    records.sort(key=lambda record: (record.label, record.relative_path.casefold()))

    output = Path(output_file).expanduser().resolve()
    _atomic_write_csv(output, [_row(record, "external") for record in records])
    per_class = dict(Counter(record.label for record in records))
    return {
        "source_root": str(source),
        "class_root": selected.path.relative_to(source).as_posix() or ".",
        "manifest_path": str(output),
        "manifest_sha256": _file_digest(output),
        "count": len(records),
        "per_class": per_class,
        "class_names": list(CLASS_NAMES),
        "split": "external",
    }


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Read and strictly validate a manifest produced by this module."""

    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(MANIFEST_FIELDS):
            raise ValueError(
                f"Manifest columns must be {list(MANIFEST_FIELDS)}, got {reader.fieldnames}"
            )
        rows = list(reader)

    allowed_splits = {"train", "validation", "test", "external"}
    for line_number, row in enumerate(rows, start=2):
        if any(row.get(field) in {None, ""} for field in MANIFEST_FIELDS):
            raise ValueError(f"Manifest line {line_number} has a missing or empty field")
        relative = PurePosixPath(row["path"])
        if relative.is_absolute() or ".." in relative.parts or "\\" in row["path"]:
            raise ValueError(f"Unsafe path on manifest line {line_number}")
        if relative.as_posix() != row["path"]:
            raise ValueError(f"Non-canonical path on manifest line {line_number}: {row['path']}")
        if row["label"] not in CLASS_NAMES:
            raise ValueError(f"Unsupported label on manifest line {line_number}: {row['label']!r}")
        if row["split"] not in allowed_splits:
            raise ValueError(f"Unsupported split on manifest line {line_number}: {row['split']!r}")
        if len(row["sha256"]) != 64 or any(
            char not in "0123456789abcdef" for char in row["sha256"]
        ):
            raise ValueError(f"Invalid SHA-256 on manifest line {line_number}")
        if len(row["dhash"]) != 16 or any(char not in "0123456789abcdef" for char in row["dhash"]):
            raise ValueError(f"Invalid dHash on manifest line {line_number}")
    return rows


def _resolve_manifest_image(source_root: Path, relative_path: str) -> Path:
    candidate = (source_root / Path(*PurePosixPath(relative_path).parts)).resolve()
    try:
        candidate.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes source root: {relative_path}") from exc
    return candidate


def verify_manifest_files(
    manifest_path: Path,
    source_root: Path,
    expected_split: str | None = None,
) -> list[dict[str, str]]:
    """Verify manifest identity against source files before a model consumes it.

    Verification is intentionally performed once before a training or evaluation
    run rather than on every batch. This keeps data loading inexpensive while
    ensuring the recorded manifest still describes the bytes on disk.
    """

    root = _require_directory(Path(source_root), "Dataset source root")
    rows = read_manifest(Path(manifest_path))
    seen_paths: set[Path] = set()
    for line_number, row in enumerate(rows, start=2):
        candidate = _resolve_manifest_image(root, row["path"])
        if candidate in seen_paths:
            raise ValueError(f"Duplicate path on manifest line {line_number}: {row['path']}")
        seen_paths.add(candidate)
        if expected_split is not None and row["split"] != expected_split:
            raise ValueError(
                f"Expected split {expected_split!r} on manifest line {line_number}, "
                f"got {row['split']!r}"
            )
        if not candidate.is_file():
            raise FileNotFoundError(f"Manifest image does not exist: {candidate}")
        actual_sha256 = _file_digest(candidate)
        if actual_sha256 != row["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for manifest image {row['path']}: "
                f"expected {row['sha256']}, got {actual_sha256}"
            )
    return rows


def build_transforms(image_size: int, training: bool):
    """Build augmentation-only training or deterministic evaluation transforms."""

    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise ValueError("image_size must be a positive integer")
    if not isinstance(training, bool):
        raise TypeError("training must be a boolean")
    try:
        from torchvision import transforms
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "Image transforms require torchvision. Install the project's ML dependencies."
        ) from exc

    if training:
        spatial = [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        ]
    else:
        spatial = [transforms.Resize((image_size, image_size))]
    return transforms.Compose(
        [
            *spatial,
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ]
    )


class ManifestImageDataset:
    """PyTorch DataLoader-compatible dataset backed by a checksummed manifest.

    The class intentionally follows the Dataset protocol without importing torch
    at module import time. PyTorch's DataLoader accepts this map-style protocol.
    """

    def __init__(self, manifest_path: Path, source_root: Path, transform: Any) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.source_root = _require_directory(Path(source_root), "Dataset source root")
        self.transform = transform
        self.rows = read_manifest(self.manifest_path)
        if not self.rows:
            raise ValueError(f"Manifest contains no samples: {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        try:
            from PIL import Image, UnidentifiedImageError
        except ImportError as exc:  # pragma: no cover - depends on optional environment
            raise RuntimeError("Loading images requires Pillow.") from exc

        row = self.rows[index]
        candidate = _resolve_manifest_image(self.source_root, row["path"])
        if not candidate.is_file():
            raise FileNotFoundError(f"Manifest image does not exist: {candidate}")
        try:
            with Image.open(candidate) as opened:
                image = opened.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise DatasetLayoutError(f"Cannot decode manifest image {candidate}: {exc}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return image, CLASS_NAMES.index(row["label"]), row["path"]


_PUBLIC_DATASETS = {
    "primary": "grassknoted/asl-alphabet",
    "grassknoted/asl-alphabet": "grassknoted/asl-alphabet",
    "external": "danrasband/asl-alphabet-test",
    "danrasband/asl-alphabet-test": "danrasband/asl-alphabet-test",
}


def download_public_dataset(kind: str, output_root: Path) -> Path:
    """Download a supported public Kaggle dataset without starting an auth flow."""

    normalized = kind.strip().casefold()
    if normalized not in _PUBLIC_DATASETS:
        choices = ", ".join(sorted({"primary", "external"}))
        raise ValueError(f"Unknown dataset kind {kind!r}; choose one of: {choices}")
    try:
        import kagglehub
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError(
            "Public dataset download requires the optional 'kagglehub' package. "
            "Install the project's data dependencies. No Kaggle account or API token "
            "is required for these public datasets."
        ) from exc

    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    handle = _PUBLIC_DATASETS[normalized]
    destination = root / ("primary" if handle.startswith("grassknoted/") else "external")
    destination.mkdir(parents=True, exist_ok=True)
    try:
        downloaded = kagglehub.dataset_download(handle, output_dir=str(destination))
    except TypeError as exc:
        raise RuntimeError(
            "The installed kagglehub version does not support an explicit output "
            "directory. Upgrade kagglehub and retry; authentication is not required."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Anonymous download of public dataset {handle!r} failed: {exc}. "
            "Check network access and dataset availability. If authentication is "
            "requested, stop and use an existing local directory with the prepare "
            "command instead; do not enter credentials."
        ) from exc

    result = Path(downloaded).expanduser().resolve()
    if not result.exists():
        raise RuntimeError(f"kagglehub reported a missing download path: {result}")
    return result
