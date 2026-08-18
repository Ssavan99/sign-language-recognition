"""Integrity checks for the dependency-free browser inference release."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from asl_recognition.constants import CLASS_NAMES, IMAGE_MEAN, IMAGE_STD
from asl_recognition.model import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


def test_browser_weights_exactly_match_released_checkpoint() -> None:
    checkpoint = ROOT / "models" / "asl_alphabet_cnn_seed42.pt"
    manifest_path = ROOT / "site" / "assets" / "browser-model-manifest.json"
    weights_path = ROOT / "site" / "assets" / "asl-alphabet-cnn-v1.f32"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    weights = weights_path.read_bytes()
    model, metadata = load_checkpoint(checkpoint, map_location="cpu")

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

    assert manifest["format"] == "asl-browser-cnn-v1"
    assert (
        manifest["source_checkpoint_sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )
    assert manifest["architecture"] == metadata["architecture"]
    assert manifest["class_names"] == list(CLASS_NAMES)
    assert manifest["image_size"] == 64
    assert manifest["normalization"] == {"mean": list(IMAGE_MEAN), "std": list(IMAGE_STD)}
    assert manifest["tensors"] == expected_tensors
    assert manifest["float_count"] == offset
    assert manifest["weight_sha256"] == hashlib.sha256(weights).hexdigest()
    assert weights == b"".join(chunks)
