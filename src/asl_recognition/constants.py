"""Shared constants for the isolated ASL alphabet classifier."""

from __future__ import annotations

from string import ascii_uppercase

# The supported task is deliberately limited to the 26 static alphabet labels.
# Dataset-specific pseudo-classes such as ``del``, ``nothing``, and ``space`` are
# not part of the model contract.
CLASS_NAMES: tuple[str, ...] = tuple(ascii_uppercase)

IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})

# The compact classifier uses ImageNet normalization for both training and
# inference. Keeping these values here makes the preprocessing contract explicit.
IMAGE_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGE_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

MANIFEST_FIELDS: tuple[str, ...] = (
    "path",
    "label",
    "split",
    "sha256",
    "dhash",
)
