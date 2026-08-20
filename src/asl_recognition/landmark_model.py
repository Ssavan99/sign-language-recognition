"""A small classifier over hand-landmark features, and its checkpoint contract.

This is a second model, not a replacement for the convolutional one. They answer
different questions: the CNN reads pixels and is therefore tied to how an image
was captured, while this reads geometry and is not. Both stay in the repository
because the comparison between them is the interesting result.

The network is deliberately tiny. Seventy-nine normalised numbers describing hand
shape carry far less nuisance variation than four thousand pixels, so a wide
network would mostly find ways to overfit the corpora rather than the alphabet.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .constants import CLASS_NAMES
from .landmarks import FEATURE_DIMENSION

ARCHITECTURE_NAME = "landmark_mlp_v1"
FEATURE_CONTRACT = "landmark-v1"


class LandmarkMLP(nn.Module):
    """A three-layer classifier over normalised landmark features."""

    def __init__(
        self,
        num_classes: int = len(CLASS_NAMES),
        hidden: Sequence[int] = (256, 128),
        dropout: float = 0.3,
        input_dimension: int = FEATURE_DIMENSION,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if len(hidden) < 1 or any(int(width) <= 0 for width in hidden):
            raise ValueError("hidden must contain at least one positive width")

        self.num_classes = int(num_classes)
        self.hidden = tuple(int(width) for width in hidden)
        self.dropout = float(dropout)
        self.input_dimension = int(input_dimension)

        layers: list[nn.Module] = []
        previous = self.input_dimension
        for width in self.hidden:
            layers.extend(
                [
                    nn.Linear(previous, width),
                    nn.BatchNorm1d(width),
                    nn.ReLU(inplace=True),
                    nn.Dropout(self.dropout),
                ]
            )
            previous = width
        layers.append(nn.Linear(previous, self.num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)

    def architecture_config(self) -> dict[str, Any]:
        return {
            "name": ARCHITECTURE_NAME,
            "num_classes": self.num_classes,
            "hidden": list(self.hidden),
            "dropout": self.dropout,
            "input_dimension": self.input_dimension,
        }


def build_landmark_model(
    *,
    num_classes: int = len(CLASS_NAMES),
    hidden: Sequence[int] = (256, 128),
    dropout: float = 0.3,
) -> LandmarkMLP:
    return LandmarkMLP(num_classes=num_classes, hidden=hidden, dropout=dropout)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def save_landmark_checkpoint(
    path: Path,
    model: LandmarkMLP,
    *,
    seed: int,
    best_epoch: int,
    best_validation_loss: float,
    class_names: Sequence[str] = CLASS_NAMES,
    extra_metadata: Mapping[str, Any] | None = None,
) -> Path:
    """Write a self-describing checkpoint.

    The feature contract is recorded alongside the weights. Landmark features are
    only meaningful under the exact normalisation that produced them, so a
    checkpoint that does not name its contract is not loadable with confidence.
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "architecture": model.architecture_config(),
        "feature_contract": FEATURE_CONTRACT,
        "feature_dimension": model.input_dimension,
        "class_names": list(class_names),
        "seed": int(seed),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_validation_loss),
        "parameter_count": count_parameters(model),
        "state_dict": model.state_dict(),
    }
    if extra_metadata:
        payload["metadata"] = dict(extra_metadata)
    torch.save(payload, path)
    return path


def load_landmark_checkpoint(
    path: Path, map_location: str = "cpu"
) -> tuple[LandmarkMLP, dict[str, Any]]:
    """Load a checkpoint and refuse one built under a different feature contract."""

    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    for key in ("architecture", "feature_contract", "class_names", "state_dict"):
        if key not in payload:
            raise ValueError(f"landmark checkpoint is missing {key!r}")
    if payload["feature_contract"] != FEATURE_CONTRACT:
        raise ValueError(
            f"checkpoint was built for feature contract {payload['feature_contract']!r}, "
            f"but this code produces {FEATURE_CONTRACT!r}"
        )
    architecture = payload["architecture"]
    if architecture.get("name") != ARCHITECTURE_NAME:
        raise ValueError(f"unsupported landmark architecture: {architecture.get('name')!r}")
    model = LandmarkMLP(
        num_classes=int(architecture["num_classes"]),
        hidden=architecture["hidden"],
        dropout=float(architecture["dropout"]),
        input_dimension=int(architecture["input_dimension"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    metadata = {key: value for key, value in payload.items() if key != "state_dict"}
    return model, metadata


def checkpoint_digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preprocessing_contract() -> dict[str, Any]:
    """Describe what a caller must do before this model sees anything."""

    return {
        "detector": "MediaPipe HandLandmarker, single hand",
        "feature_contract": FEATURE_CONTRACT,
        "feature_dimension": FEATURE_DIMENSION,
        "normalisation": [
            "mirror left hands to right-hand form",
            "translate so the wrist is the origin",
            "scale so the furthest landmark from the wrist is at distance 1",
            "rotate in XY so the middle-finger knuckle points up",
        ],
        "appended_distances": (
            "fingertip-to-wrist, fingertip-to-thumb, adjacent fingertips, fingertip-to-knuckle"
        ),
        "augmentation_at_inference": False,
    }


def contract_json() -> str:
    return json.dumps(preprocessing_contract(), indent=2, sort_keys=True)


__all__ = [
    "ARCHITECTURE_NAME",
    "FEATURE_CONTRACT",
    "LandmarkMLP",
    "build_landmark_model",
    "checkpoint_digest",
    "count_parameters",
    "load_landmark_checkpoint",
    "preprocessing_contract",
    "save_landmark_checkpoint",
]
