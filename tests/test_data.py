from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import pytest

from asl_recognition.constants import CLASS_NAMES, MANIFEST_FIELDS
from asl_recognition.data import (
    DatasetLayoutError,
    ManifestImageDataset,
    build_transforms,
    prepare_manifests,
    read_manifest,
    verify_manifest_files,
)
from asl_recognition.smoke import generate_fixture


def _write_manifest(path: Path, row: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)


def test_manifest_preparation_is_deterministic(
    tmp_path: Path,
    prepared_data: tuple[Path, Path, dict],
) -> None:
    source, manifests, first = prepared_data
    second_dir = tmp_path / "second"
    second = prepare_manifests(
        source,
        second_dir,
        seed=17,
        train_ratio=0.50,
        validation_ratio=0.25,
        test_ratio=0.25,
        near_duplicate_threshold=0,
    )

    assert first["counts"]["per_split"] == {"train": 52, "validation": 26, "test": 26}
    assert first["manifest_sha256"] == second["manifest_sha256"]
    for split in ("train", "validation", "test"):
        assert (manifests / f"{split}.csv").read_bytes() == (
            second_dir / f"{split}.csv"
        ).read_bytes()


def test_manifest_dataset_and_eval_transform_contract(
    prepared_data: tuple[Path, Path, dict],
) -> None:
    import torch

    source, manifests, _ = prepared_data
    rows = verify_manifest_files(manifests / "train.csv", source, "train")
    transform = build_transforms(32, training=False)
    dataset = ManifestImageDataset(manifests / "train.csv", source, transform)
    image, class_index, relative_path = dataset[0]

    assert len(rows) == len(dataset) == 52
    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert CLASS_NAMES[class_index] == rows[0]["label"]
    assert relative_path == rows[0]["path"]


def test_eval_transform_is_repeatable() -> None:
    import torch
    from PIL import Image

    image = Image.new("RGB", (48, 72), color=(40, 120, 220))
    transform = build_transforms(32, training=False)
    assert torch.equal(transform(image), transform(image))


@pytest.mark.parametrize("path", ["../escape.png", "A\\image.png", "A/./image.png"])
def test_manifest_rejects_unsafe_or_noncanonical_paths(tmp_path: Path, path: str) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        {
            "path": path,
            "label": "A",
            "split": "train",
            "sha256": "0" * 64,
            "dhash": "0" * 16,
        },
    )
    with pytest.raises(ValueError, match="path|Unsafe|Non-canonical"):
        read_manifest(manifest)


def test_verification_detects_changed_source_bytes(tmp_path: Path) -> None:
    source = generate_fixture(tmp_path, images_per_class=3, image_size=48, seed=4)
    manifests = tmp_path / "manifests"
    prepare_manifests(source, manifests, near_duplicate_threshold=0)
    row = read_manifest(manifests / "train.csv")[0]
    (source / Path(*Path(row["path"]).parts)).write_bytes(b"changed after manifest creation")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_manifest_files(manifests / "train.csv", source, "train")


def test_identical_bytes_under_different_labels_are_rejected(tmp_path: Path) -> None:
    source = generate_fixture(tmp_path, images_per_class=3, image_size=48, seed=9)
    shutil.copyfile(source / "A" / "a_00.png", source / "B" / "b_00.png")

    with pytest.raises(DatasetLayoutError, match="different class labels"):
        prepare_manifests(source, tmp_path / "manifests", near_duplicate_threshold=0)


def _source_partition_fixture(tmp_path: Path) -> Path:
    from PIL import Image

    source = tmp_path / "source"
    train_fixture = generate_fixture(
        tmp_path / "generated-train", images_per_class=3, image_size=48, seed=31
    )
    test_fixture = generate_fixture(
        tmp_path / "generated-test", images_per_class=1, image_size=48, seed=73
    )
    shutil.copytree(train_fixture, source / "wrapper" / "train")
    shutil.copytree(test_fixture, source / "wrapper" / "test")
    for path in (source / "wrapper" / "test").glob("*/*.png"):
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        image.putpixel((0, 0), (1, 2, 3))
        image.save(path)
    return source


def test_nested_source_test_partition_is_preserved(tmp_path: Path) -> None:
    source = _source_partition_fixture(tmp_path)
    manifests = tmp_path / "manifests"
    result = prepare_manifests(
        source,
        manifests,
        seed=11,
        train_ratio=0.60,
        validation_ratio=0.20,
        test_ratio=0.20,
        near_duplicate_threshold=0,
    )
    test_rows = read_manifest(manifests / "test.csv")

    assert result["layout"] == "source_partitions"
    assert len(test_rows) == 26
    assert all(row["path"].startswith("wrapper/test/") for row in test_rows)
    assert {row["label"] for row in test_rows} == set(CLASS_NAMES)


def test_exact_duplicate_crossing_source_partitions_is_rejected(tmp_path: Path) -> None:
    source = _source_partition_fixture(tmp_path)
    shutil.copyfile(
        source / "wrapper" / "train" / "A" / "a_00.png",
        source / "wrapper" / "test" / "A" / "a_00.png",
    )

    with pytest.raises(DatasetLayoutError, match="crosses source-provided partitions"):
        prepare_manifests(source, tmp_path / "manifests", near_duplicate_threshold=0)


def test_exact_duplicate_group_stays_in_one_generated_split(tmp_path: Path) -> None:
    source = generate_fixture(tmp_path, images_per_class=4, image_size=48, seed=19)
    shutil.copyfile(source / "A" / "a_00.png", source / "A" / "a_duplicate.png")
    result = prepare_manifests(source, tmp_path / "manifests", near_duplicate_threshold=0)
    report = json.loads(Path(result["duplicate_report_path"]).read_text(encoding="utf-8"))

    assert report["exact_cross_split_duplicates"] == 0
    assert report["exact_duplicate_group_count"] == 1
    assert report["exact_duplicate_extra_image_count"] == 1


@pytest.mark.parametrize(
    ("ratios", "message"),
    [((0.7, 0.2, 0.2), "sum"), ((0.0, 0.5, 0.5), "between")],
)
def test_invalid_split_ratios_are_rejected(
    tmp_path: Path,
    ratios: tuple[float, float, float],
    message: str,
) -> None:
    source = generate_fixture(tmp_path, images_per_class=3, image_size=48, seed=2)
    with pytest.raises(ValueError, match=message):
        prepare_manifests(
            source,
            tmp_path / "manifests",
            train_ratio=ratios[0],
            validation_ratio=ratios[1],
            test_ratio=ratios[2],
        )


def test_read_manifest_rejects_a_duplicate_path_anywhere_in_the_file(
    tmp_path: Path,
    prepared_data: tuple[Path, Path, dict],
) -> None:
    source, manifests, _ = prepared_data
    rows = read_manifest(manifests / "train.csv")
    duplicated = tmp_path / "duplicated.csv"
    with duplicated.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(rows[0])

    # The check must live at parse time: a caller that verifies only a subset of
    # rows would otherwise never see a duplicate outside that subset.
    with pytest.raises(ValueError, match="Duplicate path on manifest line"):
        read_manifest(duplicated)
