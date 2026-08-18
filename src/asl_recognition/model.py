"""Compact convolutional model and self-describing checkpoint helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    from .constants import CLASS_NAMES, IMAGE_MEAN, IMAGE_STD
except ImportError:  # pragma: no cover - allows isolated use during development
    CLASS_NAMES = tuple(chr(code) for code in range(ord("A"), ord("Z") + 1))
    IMAGE_MEAN = (0.485, 0.456, 0.406)
    IMAGE_STD = (0.229, 0.224, 0.225)


ARCHITECTURE_NAME = "compact_asl_cnn_v1"


class ConvBlock(nn.Sequential):
    """Two inexpensive convolution stages followed by spatial downsampling."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )


class CompactASLCNN(nn.Module):
    """A small image classifier intended for 26 isolated ASL letter classes."""

    def __init__(
        self,
        num_classes: int = 26,
        channels: Sequence[int] = (24, 48, 96),
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if len(channels) != 3 or any(int(value) <= 0 for value in channels):
            raise ValueError("channels must contain three positive integers")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        c1, c2, c3 = (int(value) for value in channels)
        self.num_classes = int(num_classes)
        self.channels = (c1, c2, c3)
        self.dropout = float(dropout)
        self.features = nn.Sequential(
            ConvBlock(3, c1),
            ConvBlock(c1, c2),
            ConvBlock(c2, c3),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.dropout),
            nn.Linear(c3, self.num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(inputs)))

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": ARCHITECTURE_NAME,
            "num_classes": self.num_classes,
            "channels": list(self.channels),
            "dropout": self.dropout,
        }


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable and non-trainable scalar parameters."""

    return sum(parameter.numel() for parameter in model.parameters())


def preprocessing_contract(image_size: int) -> dict[str, Any]:
    """Describe the image preprocessing required by a saved model."""

    if image_size <= 0:
        raise ValueError("image_size must be positive")
    return {
        "color_mode": "RGB",
        "resize": [int(image_size), int(image_size)],
        "tensor_layout": "CHW",
        "value_range_before_normalization": [0.0, 1.0],
        "normalization": {
            "mean": [float(value) for value in IMAGE_MEAN],
            "std": [float(value) for value in IMAGE_STD],
        },
        "augmentation_at_inference": False,
    }


def build_model(
    *,
    num_classes: int = 26,
    channels: Sequence[int] = (24, 48, 96),
    dropout: float = 0.20,
) -> CompactASLCNN:
    """Build the supported compact architecture."""

    model = CompactASLCNN(
        num_classes=num_classes,
        channels=channels,
        dropout=dropout,
    )
    parameter_count = count_parameters(model)
    if parameter_count >= 1_000_000:
        raise ValueError(f"compact model exceeds the 1M parameter contract: {parameter_count:,}")
    return model


def save_checkpoint(
    path: Path,
    model: CompactASLCNN,
    *,
    image_size: int,
    seed: int,
    best_epoch: int,
    best_validation_loss: float,
    class_names: Sequence[str] = CLASS_NAMES,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Save weights plus all information required for independent inference."""

    names = [str(name) for name in class_names]
    if len(names) != model.num_classes or len(set(names)) != len(names):
        raise ValueError("class_names must uniquely match the model output count")

    payload: dict[str, Any] = {
        "format_version": 1,
        "architecture": model.architecture_config(),
        "state_dict": model.state_dict(),
        "class_names": names,
        "image_size": int(image_size),
        "preprocessing": preprocessing_contract(image_size),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "parameter_count": count_parameters(model),
    }
    if extra_metadata:
        reserved = set(payload)
        collisions = reserved.intersection(extra_metadata)
        if collisions:
            joined = ", ".join(sorted(collisions))
            raise ValueError(f"extra metadata may not replace checkpoint fields: {joined}")
        payload.update(dict(extra_metadata))

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)
    return payload


def load_checkpoint(
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[CompactASLCNN, dict[str, Any]]:
    """Load and validate a supported checkpoint."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint not found: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=True)
    except TypeError:  # PyTorch before the weights_only argument
        payload = torch.load(source, map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a dictionary")

    required = {
        "architecture",
        "state_dict",
        "class_names",
        "image_size",
        "preprocessing",
        "seed",
        "best_epoch",
        "best_validation_loss",
        "parameter_count",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"checkpoint is missing fields: {', '.join(sorted(missing))}")

    architecture = payload["architecture"]
    if not isinstance(architecture, dict) or architecture.get("name") != ARCHITECTURE_NAME:
        raise ValueError(f"unsupported checkpoint architecture: {architecture!r}")
    model = build_model(
        num_classes=int(architecture["num_classes"]),
        channels=tuple(int(value) for value in architecture["channels"]),
        dropout=float(architecture["dropout"]),
    )
    class_names = payload["class_names"]
    if not isinstance(class_names, (list, tuple)) or len(class_names) != model.num_classes:
        raise ValueError("checkpoint class map does not match its architecture")
    model.load_state_dict(payload["state_dict"], strict=True)
    if int(payload["parameter_count"]) != count_parameters(model):
        raise ValueError("checkpoint parameter count does not match its architecture")
    return model, payload


__all__ = [
    "ARCHITECTURE_NAME",
    "CompactASLCNN",
    "build_model",
    "count_parameters",
    "load_checkpoint",
    "preprocessing_contract",
    "save_checkpoint",
]
