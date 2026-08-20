from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from asl_recognition.constants import CLASS_NAMES
from asl_recognition.demo import create_demo, format_demo_prediction, launch_demo
from asl_recognition.inference import Predictor
from asl_recognition.model import (
    build_model,
    count_parameters,
    load_checkpoint,
    preprocessing_contract,
    save_checkpoint,
)


@pytest.fixture()
def checkpoint(tmp_path: Path) -> Path:
    torch.manual_seed(5)
    model = build_model()
    path = tmp_path / "model.pt"
    save_checkpoint(
        path,
        model,
        image_size=32,
        seed=5,
        best_epoch=1,
        best_validation_loss=0.5,
    )
    return path


def test_compact_model_and_checkpoint_contract(checkpoint: Path) -> None:
    model, metadata = load_checkpoint(checkpoint)
    logits = model(torch.zeros(2, 3, 32, 32))

    assert logits.shape == (2, 26)
    assert count_parameters(model) == 164_546
    assert metadata["class_names"] == list(CLASS_NAMES)
    assert metadata["preprocessing"] == preprocessing_contract(32)
    assert metadata["parameter_count"] == count_parameters(model)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_classes": 1}, "greater than one"),
        ({"channels": (8, 16)}, "three positive"),
        ({"dropout": 1.0}, "dropout"),
    ],
)
def test_invalid_model_configuration_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_model(**kwargs)


def test_predictor_returns_ordered_a_to_z_contract(checkpoint: Path) -> None:
    predictor = Predictor(checkpoint, device="cpu", confidence_threshold=1.0)
    image = Image.new("RGB", (54, 70), color=(20, 100, 210))
    result = predictor.predict(image, top_k=3)
    probabilities = [item["probability"] for item in result["top_k"]]

    assert result["predicted_class"] in CLASS_NAMES
    assert probabilities == sorted(probabilities, reverse=True)
    assert all(item["class"] in CLASS_NAMES for item in result["top_k"])
    assert result["low_confidence"] is True
    assert result["warning"]


def test_predictor_rejects_unsupported_class_order(checkpoint: Path, tmp_path: Path) -> None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    payload["class_names"] = list(reversed(CLASS_NAMES))
    changed = tmp_path / "changed.pt"
    torch.save(payload, changed)

    with pytest.raises(ValueError, match="A-Z order"):
        Predictor(changed, device="cpu")


class _FakePredictor:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence

    def predict(self, _image: Image.Image, *, top_k: int) -> dict:
        assert top_k == 3
        return {
            "predicted_class": "A",
            "confidence": self.confidence,
            "low_confidence": self.confidence < 0.60,
            "top_k": [
                {"class": "A", "probability": self.confidence},
                {"class": "B", "probability": (1 - self.confidence) * 0.75},
                {"class": "C", "probability": (1 - self.confidence) * 0.25},
            ],
        }


def _usable_image() -> Image.Image:
    image = Image.new("RGB", (64, 64), "white")
    for x in range(32):
        for y in range(64):
            image.putpixel((x, y), (20, 40, 80))
    return image


def test_demo_handles_absent_and_unusable_images() -> None:
    status, scores, detail = format_demo_prediction(_FakePredictor(0.9), None)
    assert "Unable to classify" in status
    assert scores == {}
    assert "31.67%" in detail

    status, _, _ = format_demo_prediction(_FakePredictor(0.9), Image.new("RGB", (64, 64), "white"))
    assert "No usable hand image" in status


def test_demo_labels_low_and_high_confidence_without_overclaiming() -> None:
    low_status, _, low_detail = format_demo_prediction(_FakePredictor(0.40), _usable_image())
    high_status, _, high_detail = format_demo_prediction(_FakePredictor(0.90), _usable_image())

    assert "Low-confidence result" in low_status
    assert "Treat this as uncertain" in low_status
    assert "Predicted letter" in high_status
    assert "not a guarantee" in high_status
    assert "31.67%" in low_detail
    assert low_detail == high_detail


def test_demo_builds_and_rejects_nonlocal_binding(checkpoint: Path) -> None:
    demo = create_demo(checkpoint, device="cpu")
    assert type(demo).__name__ == "Blocks"
    with pytest.raises(ValueError, match="127.0.0.1 or localhost"):
        launch_demo(checkpoint, server_name="0.0.0.0")
    with pytest.raises(ValueError, match="between 1 and 65535"):
        launch_demo(checkpoint, port=0)


def test_demo_reports_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Use --checkpoint"):
        create_demo(tmp_path / "missing.pt")
