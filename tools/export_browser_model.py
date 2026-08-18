"""Export the released PyTorch checkpoint for the dependency-free Pages runtime."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from asl_recognition.constants import CLASS_NAMES, IMAGE_MEAN, IMAGE_STD
from asl_recognition.model import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "models" / "asl_alphabet_cnn_seed42.pt"
ASSET_DIR = ROOT / "site" / "assets"
MANIFEST_PATH = ASSET_DIR / "browser-model-manifest.json"
WEIGHTS_PATH = ASSET_DIR / "asl-alphabet-cnn-v1.f32"
PARITY_PATH = ROOT / "tools" / "browser_model_parity.json"


def main() -> None:
    model, metadata = load_checkpoint(CHECKPOINT, map_location="cpu")
    model.eval()
    state = model.state_dict()

    tensors: list[dict[str, object]] = []
    chunks: list[bytes] = []
    offset = 0
    for name, value in state.items():
        if name.endswith("num_batches_tracked"):
            continue
        array = value.detach().cpu().contiguous().numpy().astype("<f4", copy=False)
        float_count = int(array.size)
        tensors.append(
            {
                "name": name,
                "shape": list(array.shape),
                "offset": offset,
                "length": float_count,
            }
        )
        chunks.append(array.tobytes(order="C"))
        offset += float_count

    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    weights = b"".join(chunks)
    WEIGHTS_PATH.write_bytes(weights)
    manifest = {
        "format": "asl-browser-cnn-v1",
        "source_checkpoint_sha256": hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest(),
        "architecture": metadata["architecture"],
        "class_names": list(CLASS_NAMES),
        "image_size": int(metadata["image_size"]),
        "normalization": {"mean": list(IMAGE_MEAN), "std": list(IMAGE_STD)},
        "float_count": offset,
        "weight_sha256": hashlib.sha256(weights).hexdigest(),
        "tensors": tensors,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    state = 42
    values: list[float] = []
    for _ in range(3 * 64 * 64):
        state = (state * 1_664_525 + 1_013_904_223) & 0xFFFFFFFF
        values.append((state / 4_294_967_296) * 2.0 - 1.0)
    with torch.inference_mode():
        expected = torch.softmax(
            model(torch.tensor(values, dtype=torch.float32).reshape(1, 3, 64, 64)), dim=1
        )[0].tolist()
    PARITY_PATH.write_text(
        json.dumps({"seed": 42, "expected": expected}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {WEIGHTS_PATH.relative_to(ROOT)} ({len(weights):,} bytes)")
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"wrote {PARITY_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
