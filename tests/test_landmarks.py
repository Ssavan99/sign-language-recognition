"""Tests for the landmark feature contract and the classifier that consumes it.

The invariances asserted here are the entire justification for the landmark
model. If translation, scale, rotation, or handedness leak into the features,
then the representation carries capture conditions after all and the approach has
no advantage over the pixel classifier it was built to beat.
"""

from __future__ import annotations

import math
import random

import pytest

from asl_recognition.constants import CLASS_NAMES
from asl_recognition.landmark_model import (
    ARCHITECTURE_NAME,
    FEATURE_CONTRACT,
    LandmarkMLP,
    build_landmark_model,
    count_parameters,
    load_landmark_checkpoint,
    preprocessing_contract,
    save_landmark_checkpoint,
)
from asl_recognition.landmarks import (
    FEATURE_DIMENSION,
    LANDMARK_COUNT,
    MIDDLE_MCP,
    normalize_landmarks,
)

TOLERANCE = 1e-7


def _hand(seed: int = 0) -> list[list[float]]:
    rng = random.Random(seed)
    return [
        [rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9), rng.uniform(-0.1, 0.1)]
        for _ in range(LANDMARK_COUNT)
    ]


def _max_difference(left: list[float], right: list[float]) -> float:
    return max(abs(a - b) for a, b in zip(left, right, strict=True))


def test_feature_dimension_is_consistent() -> None:
    features = normalize_landmarks(_hand())
    assert len(features) == FEATURE_DIMENSION
    assert FEATURE_DIMENSION == LANDMARK_COUNT * 3 + 16


def test_features_are_translation_invariant() -> None:
    hand = _hand(1)
    moved = [[x + 0.31, y - 0.22, z] for x, y, z in hand]
    assert _max_difference(normalize_landmarks(hand), normalize_landmarks(moved)) < TOLERANCE


def test_features_are_scale_invariant() -> None:
    hand = _hand(2)
    for factor in (0.25, 2.5, 10.0):
        scaled = [[x * factor, y * factor, z * factor] for x, y, z in hand]
        assert _max_difference(normalize_landmarks(hand), normalize_landmarks(scaled)) < TOLERANCE


def test_features_are_rotation_invariant_in_the_image_plane() -> None:
    hand = _hand(3)
    reference = normalize_landmarks(hand)
    for angle in (0.2, 1.0, 2.4, -1.7, math.pi):
        cosine, sine = math.cos(angle), math.sin(angle)
        rotated = [[x * cosine - y * sine, x * sine + y * cosine, z] for x, y, z in hand]
        assert _max_difference(reference, normalize_landmarks(rotated)) < TOLERANCE, angle


def test_left_hands_are_mirrored_onto_right_hand_form() -> None:
    hand = _hand(4)
    mirrored = [[-x, y, z] for x, y, z in hand]
    # The same sign made with the other hand must land on the same features,
    # otherwise the classifier has to learn every letter twice.
    assert (
        _max_difference(
            normalize_landmarks(hand, "Right"),
            normalize_landmarks(mirrored, "Left"),
        )
        < TOLERANCE
    )


def test_canonical_rotation_puts_the_middle_knuckle_upright() -> None:
    features = normalize_landmarks(_hand(5))
    x = features[MIDDLE_MCP * 3]
    y = features[MIDDLE_MCP * 3 + 1]
    assert abs(x) < TOLERANCE
    assert y < 0  # negative y is "up" in image coordinates


def test_scale_normalisation_puts_the_furthest_point_at_unit_distance() -> None:
    features = normalize_landmarks(_hand(6))
    distances = [
        math.sqrt(sum(features[index * 3 + axis] ** 2 for axis in range(3)))
        for index in range(LANDMARK_COUNT)
    ]
    assert max(distances) == pytest.approx(1.0, abs=1e-6)


def test_objects_with_xyz_attributes_are_accepted() -> None:
    class Point:
        def __init__(self, x: float, y: float, z: float) -> None:
            self.x, self.y, self.z = x, y, z

    hand = _hand(7)
    assert (
        _max_difference(
            normalize_landmarks(hand),
            normalize_landmarks([Point(*point) for point in hand]),
        )
        < TOLERANCE
    )


def test_wrong_landmark_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected 21 landmarks"):
        normalize_landmarks(_hand()[:10])


def test_degenerate_hand_is_rejected() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        normalize_landmarks([[0.5, 0.5, 0.0]] * LANDMARK_COUNT)


def test_model_shape_and_size() -> None:
    import torch

    model = build_landmark_model()
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(4, FEATURE_DIMENSION))
    assert output.shape == (4, len(CLASS_NAMES))
    # Small on purpose: 79 normalised numbers do not justify a large network.
    assert count_parameters(model) < 100_000


def test_model_rejects_bad_configuration() -> None:
    with pytest.raises(ValueError, match="num_classes"):
        LandmarkMLP(num_classes=1)
    with pytest.raises(ValueError, match="dropout"):
        LandmarkMLP(dropout=1.0)
    with pytest.raises(ValueError, match="hidden"):
        LandmarkMLP(hidden=())


def test_checkpoint_round_trip_preserves_predictions(tmp_path) -> None:
    import torch

    model = build_landmark_model()
    model.eval()
    batch = torch.randn(8, FEATURE_DIMENSION, generator=torch.Generator().manual_seed(0))
    with torch.inference_mode():
        expected = model(batch)

    path = save_landmark_checkpoint(
        tmp_path / "landmark.pt", model, seed=42, best_epoch=3, best_validation_loss=0.1
    )
    restored, metadata = load_landmark_checkpoint(path)
    with torch.inference_mode():
        assert torch.allclose(restored(batch), expected, atol=1e-6)
    assert metadata["feature_contract"] == FEATURE_CONTRACT
    assert metadata["architecture"]["name"] == ARCHITECTURE_NAME
    assert metadata["class_names"] == list(CLASS_NAMES)


def test_checkpoint_from_a_different_feature_contract_is_refused(tmp_path) -> None:
    import torch

    model = build_landmark_model()
    path = tmp_path / "stale.pt"
    save_landmark_checkpoint(path, model, seed=1, best_epoch=1, best_validation_loss=0.5)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["feature_contract"] = "landmark-v0"
    torch.save(payload, path)

    # Features are only meaningful under the normalisation that produced them, so
    # loading across contracts must fail loudly rather than predict nonsense.
    with pytest.raises(ValueError, match="feature contract"):
        load_landmark_checkpoint(path)


def test_preprocessing_contract_is_self_describing() -> None:
    contract = preprocessing_contract()
    assert contract["feature_contract"] == FEATURE_CONTRACT
    assert contract["feature_dimension"] == FEATURE_DIMENSION
    assert contract["augmentation_at_inference"] is False
    assert len(contract["normalisation"]) == 4


def test_browser_landmark_assets_match_the_released_checkpoint() -> None:
    """The exported weights must be the checkpoint, not a stale copy of one."""

    import hashlib
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "models" / "asl_landmark_mlp_seed42.pt"
    manifest = json.loads(
        (root / "site" / "assets" / "landmark-model-manifest.json").read_text(encoding="utf-8")
    )
    weights = (root / "site" / "assets" / "asl-landmark-mlp-v1.f32").read_bytes()
    model, metadata = load_landmark_checkpoint(checkpoint, map_location="cpu")

    chunks: list[bytes] = []
    expected_tensors: list[dict[str, object]] = []
    offset = 0
    for name, value in model.state_dict().items():
        if name.endswith("num_batches_tracked"):
            continue
        array = value.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        expected_tensors.append(
            {"name": name, "shape": list(array.shape), "offset": offset, "length": int(array.size)}
        )
        chunks.append(array.tobytes(order="C"))
        offset += int(array.size)

    assert manifest["format"] == "asl-landmark-mlp-v1"
    assert (
        manifest["source_checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert manifest["architecture"] == metadata["architecture"]
    assert manifest["feature_contract"] == FEATURE_CONTRACT
    assert manifest["feature_dimension"] == FEATURE_DIMENSION
    assert manifest["class_names"] == list(CLASS_NAMES)
    assert manifest["tensors"] == expected_tensors
    assert manifest["float_count"] == offset
    assert manifest["weight_sha256"] == hashlib.sha256(weights).hexdigest()
    assert weights == b"".join(chunks)


def test_browser_landmark_indices_match_python() -> None:
    """The JS normaliser reads its indices from the manifest; they must be right."""

    import json
    from pathlib import Path

    from asl_recognition.landmarks import FINGERTIPS, INDEX_MCP, PINKY_MCP, RING_MCP, WRIST

    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "site" / "assets" / "landmark-model-manifest.json").read_text(encoding="utf-8")
    )
    indices = manifest["landmark_indices"]
    assert indices["wrist"] == WRIST
    assert indices["index_mcp"] == INDEX_MCP
    assert indices["middle_mcp"] == MIDDLE_MCP
    assert indices["ring_mcp"] == RING_MCP
    assert indices["pinky_mcp"] == PINKY_MCP
    assert indices["fingertips"] == list(FINGERTIPS)
