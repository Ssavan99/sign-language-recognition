"""Tests for augmentation profiles and the frozen stress benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from asl_recognition.constants import CLASS_NAMES, IMAGE_MEAN, IMAGE_STD
from asl_recognition.data import (
    AUGMENTATION_PROFILES,
    DEFAULT_AUGMENTATION_PROFILE,
    build_transforms,
    verify_manifest_files,
)
from asl_recognition.robustness import (
    CORRUPTION_NAMES,
    INDEPENDENT_CORRUPTIONS,
    STRESS_BENCHMARK_VERSION,
    StressDataset,
    apply_corruption,
    benchmark_definition,
    corruption_for_index,
)


def _image(seed: int = 0, size: int = 96) -> Image.Image:
    array = (np.random.default_rng(seed).random((size, size, 3)) * 255).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def test_default_profile_is_the_released_recipe() -> None:
    assert DEFAULT_AUGMENTATION_PROFILE == "baseline"
    assert "baseline" in AUGMENTATION_PROFILES


def test_every_profile_builds_and_produces_the_contract_tensor_shape() -> None:
    import torch

    for profile in AUGMENTATION_PROFILES:
        transform = build_transforms(64, True, profile)
        tensor = transform(_image(1))
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 64, 64)


def test_inference_preprocessing_is_identical_under_every_profile() -> None:
    import torch

    reference = build_transforms(64, False)
    expected = reference(_image(2))
    for profile in AUGMENTATION_PROFILES:
        # The profile argument must not reach the evaluation path at all: the
        # released model, CLI, demo, and browser export all share one contract.
        assert torch.equal(build_transforms(64, False, profile)(_image(2)), expected)


def test_evaluation_transform_keeps_the_published_normalisation() -> None:
    stages = [type(stage).__name__ for stage in build_transforms(64, False).transforms]
    assert stages == ["Resize", "ToTensor", "Normalize"]
    normalize = build_transforms(64, False).transforms[-1]
    assert tuple(normalize.mean) == IMAGE_MEAN
    assert tuple(normalize.std) == IMAGE_STD


def test_unknown_profile_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown augmentation profile"):
        build_transforms(64, True, "does-not-exist")


def test_corruption_assignment_cycles_over_the_frozen_order() -> None:
    assigned = [corruption_for_index(index) for index in range(len(CORRUPTION_NAMES) * 2)]
    assert assigned[: len(CORRUPTION_NAMES)] == list(CORRUPTION_NAMES)
    assert assigned[len(CORRUPTION_NAMES) :] == list(CORRUPTION_NAMES)
    with pytest.raises(ValueError, match="non-negative"):
        corruption_for_index(-1)


def test_every_corruption_is_deterministic_and_shape_preserving() -> None:
    source = _image(3)
    for name in CORRUPTION_NAMES:
        first = apply_corruption(source, name, 11)
        second = apply_corruption(source, name, 11)
        assert first.size == source.size
        assert first.mode == "RGB"
        assert np.array_equal(np.asarray(first), np.asarray(second)), name


def test_every_corruption_actually_changes_the_image() -> None:
    source = _image(4)
    for name in CORRUPTION_NAMES:
        corrupted = np.asarray(apply_corruption(source, name, 5))
        assert not np.array_equal(corrupted, np.asarray(source)), name


def test_noise_corruption_respects_its_seed() -> None:
    source = _image(6)
    same = np.asarray(apply_corruption(source, "gaussian_noise", 1))
    other = np.asarray(apply_corruption(source, "gaussian_noise", 2))
    assert not np.array_equal(same, other)


def test_unknown_corruption_is_rejected() -> None:
    with pytest.raises(KeyError, match="unknown corruption"):
        apply_corruption(_image(7), "not-a-corruption")


def test_benchmark_definition_is_stable_and_self_describing() -> None:
    definition = benchmark_definition()
    assert definition["version"] == STRESS_BENCHMARK_VERSION
    assert definition["corruptions"] == list(CORRUPTION_NAMES)
    assert definition["source"] == "source validation split only"
    assert set(definition["independent_corruptions"]) == INDEPENDENT_CORRUPTIONS
    assert INDEPENDENT_CORRUPTIONS.issubset(set(CORRUPTION_NAMES))


def test_stress_dataset_is_deterministic_and_reuses_validation_rows(
    prepared_data: tuple[Path, Path, dict],
) -> None:
    import torch

    source, manifests, _ = prepared_data
    rows = verify_manifest_files(manifests / "validation.csv", source, "validation")
    transform = build_transforms(32, training=False)

    first = StressDataset(rows, source, transform)
    second = StressDataset(rows, source, transform)

    assert len(first) == len(rows)
    assert first.corruptions == second.corruptions
    image_a, label_a, name_a = first[0]
    image_b, label_b, name_b = second[0]
    assert torch.equal(image_a, image_b)
    assert label_a == label_b == CLASS_NAMES.index(rows[0]["label"])
    assert name_a == name_b == CORRUPTION_NAMES[0]


def test_stress_dataset_differs_from_the_clean_validation_image(
    prepared_data: tuple[Path, Path, dict],
) -> None:
    import torch

    from asl_recognition.data import ManifestImageDataset

    source, manifests, _ = prepared_data
    manifest = manifests / "validation.csv"
    rows = verify_manifest_files(manifest, source, "validation")
    transform = build_transforms(32, training=False)

    clean = ManifestImageDataset(manifest, source, transform, rows=rows)
    stressed = StressDataset(rows, source, transform)

    assert not torch.equal(clean[0][0], stressed[0][0])


def test_stress_dataset_requires_rows(prepared_data: tuple[Path, Path, dict]) -> None:
    source, _, _ = prepared_data
    with pytest.raises(ValueError, match="at least one validation row"):
        StressDataset([], source, build_transforms(32, training=False))
