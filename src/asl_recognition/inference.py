"""Shared, checkpoint-driven inference for files, PIL images, and the demo."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from .constants import CLASS_NAMES
from .data import build_transforms
from .model import load_checkpoint, preprocessing_contract
from .training import resolve_device

LOW_CONFIDENCE_MESSAGE = (
    "Low-confidence prediction. The image may be outside the controlled, "
    "single-hand, isolated-letter training domain."
)


class Predictor:
    """Load a model once and reuse it for consistent top-k predictions."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str = "auto",
        confidence_threshold: float = 0.60,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.checkpoint_path = Path(checkpoint_path)
        self.device = resolve_device(device)
        self.model, self.metadata = load_checkpoint(self.checkpoint_path, map_location=self.device)
        self.class_names = tuple(str(name) for name in self.metadata["class_names"])
        if self.class_names != tuple(CLASS_NAMES):
            raise ValueError("checkpoint class map does not match the supported A-Z order")
        self.image_size = int(self.metadata["image_size"])
        if self.metadata["preprocessing"] != preprocessing_contract(self.image_size):
            raise ValueError("checkpoint preprocessing contract is unsupported or inconsistent")
        self.transform = build_transforms(self.image_size, training=False)
        self.confidence_threshold = float(confidence_threshold)
        self.model.to(self.device).eval()

    @staticmethod
    def _load_image(image: Image.Image | str | Path) -> Image.Image:
        if isinstance(image, Image.Image):
            try:
                loaded = image.copy()
            except (OSError, ValueError) as error:
                raise ValueError("PIL image is closed or unreadable") from error
        elif isinstance(image, (str, Path)):
            source = Path(image)
            if not source.is_file():
                raise FileNotFoundError(f"image not found: {source}")
            try:
                with Image.open(source) as opened:
                    opened.load()
                    loaded = opened.copy()
            except (UnidentifiedImageError, OSError, ValueError) as error:
                raise ValueError(f"file is not a readable image: {source}") from error
        else:
            raise TypeError("image must be a PIL Image or filesystem path")

        if loaded.width <= 0 or loaded.height <= 0:
            raise ValueError("image has invalid dimensions")
        return ImageOps.exif_transpose(loaded).convert("RGB")

    def predict(
        self,
        image: Image.Image | str | Path,
        *,
        top_k: int = 3,
        confidence_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Predict one image and return ordered class probabilities."""

        if not 1 <= top_k <= len(self.class_names):
            raise ValueError(f"top_k must be between 1 and {len(self.class_names)}")
        threshold = (
            self.confidence_threshold
            if confidence_threshold is None
            else float(confidence_threshold)
        )
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        prepared_image = self._load_image(image)
        tensor = self.transform(prepared_image)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
            raise ValueError("preprocessing did not produce a CHW image tensor")
        with torch.inference_mode():
            logits = self.model(tensor.unsqueeze(0).to(self.device))
            if logits.shape != (1, len(self.class_names)):
                raise RuntimeError("model output shape does not match the checkpoint class map")
            probabilities = torch.softmax(logits, dim=1)
            values, indices = probabilities.topk(top_k, dim=1)

        ranked = [
            {
                "class": self.class_names[int(index)],
                "probability": float(value),
            }
            for value, index in zip(
                values[0].detach().cpu().tolist(),
                indices[0].detach().cpu().tolist(),
                strict=True,
            )
        ]
        confidence = float(ranked[0]["probability"])
        low_confidence = confidence < threshold
        return {
            "predicted_class": ranked[0]["class"],
            "confidence": confidence,
            "top_k": ranked,
            "low_confidence": low_confidence,
            "confidence_threshold": threshold,
            "warning": LOW_CONFIDENCE_MESSAGE if low_confidence else None,
        }

    def __call__(
        self,
        image: Image.Image | str | Path,
        *,
        top_k: int = 3,
        confidence_threshold: float | None = None,
    ) -> dict[str, Any]:
        return self.predict(
            image,
            top_k=top_k,
            confidence_threshold=confidence_threshold,
        )


__all__ = ["LOW_CONFIDENCE_MESSAGE", "Predictor"]
