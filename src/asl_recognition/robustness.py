"""A fixed corruption benchmark used to rank candidates without spending the blind test.

Why this exists
---------------

The primary corpus is captured under tightly controlled conditions. Its held-out
validation split is saturated -- around 99.85% accuracy -- so it cannot tell two
augmentation recipes apart, while the separate external capture source is the
project's only unbiased generalisation measurement and must be scored once, at
the very end, on one already-selected model.

This module supplies the missing middle: a deterministic set of corruptions
applied to *source validation images only*, defined once and frozen. It consumes
no test data and no external data.

What it is not
--------------

It is a **proxy for capture-domain shift, not a measurement of it**. Some of its
photometric corruptions overlap in kind with the photometric augmentation used by
the stronger training profiles, so a profile trained on colour jitter has a
structural advantage on the gamma and hue corruptions. Per-corruption accuracy is
therefore always reported alongside the aggregate, and the blind external set
remains the only arbiter of whether a model actually transfers.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable, Sequence
from typing import Any

# Bump this whenever the corruption family or its parameters change. Recorded in
# run metadata so a reported stress score can never be silently compared against
# a score produced by a different benchmark.
STRESS_BENCHMARK_VERSION = "stress-v1"

# A neutral mid-grey chosen so letterboxing neither brightens nor darkens the
# frame on average, and is clearly distinct from the corpus's own backdrop.
_LETTERBOX_FILL = (118, 118, 118)


def _require_pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on optional environment
        raise RuntimeError("The stress benchmark requires Pillow.") from exc
    return Image


def _jpeg_quality_25(image: Any, _seed: int) -> Any:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=25)
    buffer.seek(0)
    reloaded = _require_pillow().open(buffer)
    return reloaded.convert("RGB")


def _gaussian_noise(image: Any, seed: int) -> Any:
    import numpy as np

    generator = np.random.default_rng(seed)
    array = np.asarray(image, dtype=np.float32) / 255.0
    noisy = array + generator.normal(0.0, 0.08, size=array.shape).astype(np.float32)
    clipped = np.clip(noisy, 0.0, 1.0) * 255.0
    return _require_pillow().fromarray(clipped.astype("uint8"), mode="RGB")


def _gamma(image: Any, value: float) -> Any:
    import numpy as np

    array = np.asarray(image, dtype=np.float32) / 255.0
    adjusted = np.power(array, value) * 255.0
    return _require_pillow().fromarray(adjusted.astype("uint8"), mode="RGB")


def _gamma_low(image: Any, _seed: int) -> Any:
    return _gamma(image, 0.45)


def _gamma_high(image: Any, _seed: int) -> Any:
    return _gamma(image, 2.0)


def _hue_shift(image: Any, _seed: int) -> Any:
    import numpy as np

    hsv = np.asarray(image.convert("HSV"), dtype=np.int16)
    # Pillow stores hue in 0-255 rather than degrees; 38 is about a 54-degree turn.
    hsv[..., 0] = (hsv[..., 0] + 38) % 256
    shifted = _require_pillow().fromarray(hsv.astype("uint8"), mode="HSV")
    return shifted.convert("RGB")


def _contrast_crush(image: Any, _seed: int) -> Any:
    from PIL import ImageEnhance

    faded = ImageEnhance.Contrast(image).enhance(0.45)
    return ImageEnhance.Brightness(faded).enhance(1.25)


def _letterbox(image: Any, _seed: int) -> Any:
    pillow = _require_pillow()
    width, height = image.size
    inner = image.resize((max(1, int(width * 0.6)), max(1, int(height * 0.6))), pillow.BILINEAR)
    canvas = pillow.new("RGB", (width, height), _LETTERBOX_FILL)
    canvas.paste(inner, ((width - inner.size[0]) // 2, (height - inner.size[1]) // 2))
    return canvas


def _downscale(image: Any, _seed: int) -> Any:
    pillow = _require_pillow()
    width, height = image.size
    small = image.resize((20, 20), pillow.BILINEAR)
    return small.resize((width, height), pillow.BILINEAR)


# Order is part of the benchmark definition: rows are assigned corruptions by
# position, so reordering this tuple would change every recorded score.
STRESS_CORRUPTIONS: tuple[tuple[str, Callable[[Any, int], Any]], ...] = (
    ("jpeg_q25", _jpeg_quality_25),
    ("gaussian_noise", _gaussian_noise),
    ("gamma_low", _gamma_low),
    ("gamma_high", _gamma_high),
    ("hue_shift", _hue_shift),
    ("contrast_crush", _contrast_crush),
    ("letterbox", _letterbox),
    ("downscale", _downscale),
)

CORRUPTION_NAMES: tuple[str, ...] = tuple(name for name, _ in STRESS_CORRUPTIONS)

# Corruptions with no counterpart in any training-augmentation profile. Reported
# separately so a photometric profile cannot look robust purely by having been
# trained on the same kind of photometric change.
INDEPENDENT_CORRUPTIONS: frozenset[str] = frozenset(
    {"jpeg_q25", "gaussian_noise", "letterbox", "downscale"}
)


def corruption_for_index(index: int) -> str:
    """Return the corruption assigned to a row position."""

    if index < 0:
        raise ValueError("index must be non-negative")
    return CORRUPTION_NAMES[index % len(CORRUPTION_NAMES)]


def apply_corruption(image: Any, name: str, seed: int = 0) -> Any:
    """Apply one named corruption to an RGB image."""

    for candidate, function in STRESS_CORRUPTIONS:
        if candidate == name:
            return function(image.convert("RGB"), seed)
    raise KeyError(f"unknown corruption: {name!r}; expected one of {list(CORRUPTION_NAMES)}")


def benchmark_definition(dataset: StressDataset | None = None) -> dict[str, Any]:
    """Describe the frozen benchmark for run metadata.

    The version string alone does not identify a measurement. The benchmark is
    built from whatever validation rows a run consumes, so a subset-limited
    screening run and a full-split run share a version yet score entirely
    different image sets. Passing the dataset records a digest over the exact
    scored rows, which is what makes two stress numbers comparable or not.
    """

    definition: dict[str, Any] = {
        "version": STRESS_BENCHMARK_VERSION,
        "corruptions": list(CORRUPTION_NAMES),
        "independent_corruptions": sorted(INDEPENDENT_CORRUPTIONS),
        "assignment": "row position modulo corruption count",
        "source": "source validation split only",
    }
    if dataset is not None:
        definition["row_count"] = len(dataset)
        definition["row_digest"] = dataset.row_digest()
    return definition


class StressDataset:
    """Source validation rows, each carrying one deterministically assigned corruption.

    The dataset follows the same map-style protocol as
    :class:`~asl_recognition.data.ManifestImageDataset` and reuses its rows, so the
    stress pass reads exactly the images that were checksum-verified for
    validation. Per-image noise is seeded from the row's recorded SHA-256, which
    makes the benchmark independent of batch order, worker count, and run order.
    """

    def __init__(
        self,
        rows: Sequence[dict[str, str]],
        source_root: Any,
        transform: Any,
    ) -> None:
        from .data import _require_directory, validate_manifest_rows

        self.rows = validate_manifest_rows(rows)
        if not self.rows:
            raise ValueError("stress benchmark requires at least one validation row")
        if len(self.rows) < len(CORRUPTION_NAMES):
            # Corruptions are assigned by position, so a row set smaller than the
            # corruption count leaves some corruptions unscored and makes the
            # per-corruption breakdown incomparable between runs.
            raise ValueError(
                f"stress benchmark needs at least {len(CORRUPTION_NAMES)} rows to cover "
                f"every corruption, got {len(self.rows)}"
            )
        self.source_root = _require_directory(source_root, "Dataset source root")
        self.transform = transform
        self.corruptions = [corruption_for_index(index) for index in range(len(self.rows))]

    def row_digest(self) -> str:
        """Digest the exact scored rows, so two stress scores can be compared safely."""

        digest = hashlib.sha256()
        for row, corruption in zip(self.rows, self.corruptions, strict=True):
            digest.update(f"{row['path']}\x00{row['sha256']}\x00{corruption}\n".encode())
        return digest.hexdigest()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from .constants import CLASS_NAMES
        from .data import _resolve_manifest_image

        pillow = _require_pillow()
        row = self.rows[index]
        candidate = _resolve_manifest_image(self.source_root, row["path"])
        with pillow.open(candidate) as opened:
            image = opened.convert("RGB")
        seed = int(row["sha256"][:8], 16)
        corrupted = apply_corruption(image, self.corruptions[index], seed)
        if self.transform is not None:
            corrupted = self.transform(corrupted)
        return corrupted, CLASS_NAMES.index(row["label"]), self.corruptions[index]


__all__ = [
    "CORRUPTION_NAMES",
    "INDEPENDENT_CORRUPTIONS",
    "STRESS_BENCHMARK_VERSION",
    "StressDataset",
    "apply_corruption",
    "benchmark_definition",
    "corruption_for_index",
]
