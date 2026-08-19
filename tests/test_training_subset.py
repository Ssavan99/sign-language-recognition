"""Tests for the memory-lean training data path and stratified screening subsets."""

from __future__ import annotations

from pathlib import Path

import pytest

from asl_recognition.constants import CLASS_NAMES
from asl_recognition.data import ManifestImageDataset, build_transforms, verify_manifest_files
from asl_recognition.training import _stratified_indices


def _rows(label_counts: dict[str, int]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for label, count in label_counts.items():
        for index in range(count):
            rows.append(
                {
                    "path": f"{label}/{label}{index}.jpg",
                    "label": label,
                    "split": "train",
                    "sha256": f"{index:064x}",
                    "dhash": f"{index:016x}",
                }
            )
    return rows


def test_stratified_subset_is_balanced_and_deterministic() -> None:
    rows = _rows({label: 40 for label in CLASS_NAMES})
    first = _stratified_indices(rows, 5)
    second = _stratified_indices(rows, 5)

    assert first == second
    assert len(first) == 5 * len(CLASS_NAMES)
    counts: dict[str, int] = {}
    for index in first:
        label = rows[index]["label"]
        counts[label] = counts.get(label, 0) + 1
    assert set(counts) == set(CLASS_NAMES)
    assert set(counts.values()) == {5}


def test_stratified_subset_spreads_across_a_class_instead_of_taking_a_prefix() -> None:
    rows = _rows({"A": 100})
    selected = [
        int(rows[index]["path"].removeprefix("A/A").removesuffix(".jpg"))
        for index in _stratified_indices(rows, 4)
    ]

    assert selected == sorted(selected)
    # A leading slice would be [0, 1, 2, 3]; even spacing must reach the tail.
    assert selected[0] == 0
    assert selected[-1] >= 75


def test_stratified_subset_handles_classes_smaller_than_the_request() -> None:
    rows = _rows({"A": 2, "B": 6})
    selected = _stratified_indices(rows, 4)
    labels = [rows[index]["label"] for index in selected]

    assert labels.count("A") == 2
    assert labels.count("B") == 4


def test_stratified_subset_rejects_a_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="limit_per_class must be positive"):
        _stratified_indices(_rows({"A": 3}), 0)


def test_dataset_accepts_verified_rows_without_reparsing(
    prepared_data: tuple[Path, Path, dict],
) -> None:
    import torch

    source, manifests, _ = prepared_data
    manifest = manifests / "train.csv"
    rows = verify_manifest_files(manifest, source, "train")
    transform = build_transforms(32, training=False)

    reparsed = ManifestImageDataset(manifest, source, transform)
    supplied = ManifestImageDataset(manifest, source, transform, rows=rows)

    assert len(supplied) == len(reparsed)
    assert supplied.rows == reparsed.rows
    supplied_image, supplied_label, supplied_path = supplied[0]
    reparsed_image, reparsed_label, reparsed_path = reparsed[0]
    assert torch.equal(supplied_image, reparsed_image)
    assert supplied_label == reparsed_label
    assert supplied_path == reparsed_path


def test_dataset_accepts_a_subset_of_verified_rows(
    prepared_data: tuple[Path, Path, dict],
) -> None:
    source, manifests, _ = prepared_data
    manifest = manifests / "train.csv"
    rows = verify_manifest_files(manifest, source, "train")
    subset = [rows[index] for index in _stratified_indices(rows, 1)]

    dataset = ManifestImageDataset(
        manifest, source, build_transforms(32, training=False), rows=subset
    )

    assert len(dataset) == len(subset)
    assert len(dataset) < len(rows)


def test_dataset_rejects_supplied_rows_that_are_not_manifest_rows(
    prepared_data: tuple[Path, Path, dict],
) -> None:
    source, manifests, _ = prepared_data
    manifest = manifests / "train.csv"
    transform = build_transforms(32, training=False)

    with pytest.raises(ValueError, match="is missing"):
        ManifestImageDataset(manifest, source, transform, rows=[{"path": "A/A0.jpg"}])

    bad_label = dict(verify_manifest_files(manifest, source, "train")[0])
    bad_label["label"] = "not-a-letter"
    with pytest.raises(ValueError, match="has label"):
        ManifestImageDataset(manifest, source, transform, rows=[bad_label])
