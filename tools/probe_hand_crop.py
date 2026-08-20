"""Measure whether detect-then-crop closes part of the capture-domain gap.

The released classifier is trained on tightly framed hands filling the frame. A
camera frame, and the external capture set, put a smaller hand somewhere inside a
cluttered scene. That is a framing mismatch, not only a robustness problem, and
no amount of augmentation fixes being asked a different question than the one
trained for.

This probe tests the cheapest possible correction: detect the hand, crop to it,
classify the crop with the **unchanged** released model. Nothing is retrained.

It runs on the dev half of the external split only. The final half is reserved so
that a published number still means something after this iteration.

Requires MediaPipe, a development-only dependency::

    python -m pip install mediapipe

It downloads the same public HandLandmarker task file the website already loads,
so the probe measures the detector the browser would actually use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asl_recognition.constants import CLASS_NAMES  # noqa: E402
from asl_recognition.data import read_manifest  # noqa: E402
from asl_recognition.inference import Predictor  # noqa: E402

# The same public task file the website loads, so the probe measures the detector
# the browser would actually use. Free, no account, no key.
TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def _crop_to_hand(image, landmarks, padding: float):
    """Crop to the hand bounding box, expanded by `padding` and squared off."""

    width, height = image.size
    xs = [point.x * width for point in landmarks]
    ys = [point.y * height for point in landmarks]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)

    # Square the box before padding so the crop does not distort the hand when it
    # is resized to the model's square input.
    side = max(right - left, bottom - top) * (1.0 + padding)
    centre_x = (left + right) / 2.0
    centre_y = (top + bottom) / 2.0
    half = side / 2.0
    box = (
        max(0, int(centre_x - half)),
        max(0, int(centre_y - half)),
        min(width, int(centre_x + half)),
        min(height, int(centre_y + half)),
    )
    if box[2] - box[0] < 8 or box[3] - box[1] < 8:
        return None
    return image.crop(box)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/manifests-external-split/dev.csv")
    )
    parser.add_argument("--source-root", type=Path, default=Path("data/raw/external"))
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/asl_alphabet_cnn_robust_seed42.pt")
    )
    parser.add_argument(
        "--padding",
        type=float,
        nargs="+",
        default=[0.1, 0.3, 0.6],
        help="Bounding-box expansion factors to compare.",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/probe-hand-crop.json"))
    parser.add_argument(
        "--task-model",
        type=Path,
        default=Path("artifacts/hand_landmarker.task"),
        help="Local path for the public HandLandmarker task file; downloaded if absent.",
    )
    args = parser.parse_args()

    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        from PIL import Image
    except ImportError:
        print("This probe needs mediapipe and Pillow.", file=sys.stderr)
        return 2

    if not args.task_model.is_file():
        import urllib.request

        args.task_model.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {TASK_URL}")
        urllib.request.urlretrieve(TASK_URL, args.task_model)  # noqa: S310

    rows = read_manifest(args.manifest)
    predictor = Predictor(args.checkpoint, device="cpu")
    print(f"{len(rows)} dev images, checkpoint {args.checkpoint.name}")

    detector = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(args.task_model)),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.3,
        )
    )

    import numpy as np

    detected = 0
    baseline_correct = 0
    crop_correct = {padding: 0 for padding in args.padding}
    crop_scored = {padding: 0 for padding in args.padding}
    per_class_gain: dict[str, list[int]] = {label: [0, 0] for label in CLASS_NAMES}

    for index, row in enumerate(rows, start=1):
        path = args.source_root / Path(*row["path"].split("/"))
        with Image.open(path) as opened:
            image = opened.convert("RGB")
        truth = row["label"]

        if predictor.predict(path, top_k=1)["predicted_class"] == truth:
            baseline_correct += 1
            per_class_gain[truth][0] += 1

        frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image))
        result = detector.detect(frame)
        if not result.hand_landmarks:
            continue
        detected += 1
        landmarks = result.hand_landmarks[0]

        for padding in args.padding:
            cropped = _crop_to_hand(image, landmarks, padding)
            if cropped is None:
                continue
            crop_scored[padding] += 1
            if predictor.predict(cropped, top_k=1)["predicted_class"] == truth:
                crop_correct[padding] += 1
                if padding == args.padding[0]:
                    per_class_gain[truth][1] += 1
        if index % 50 == 0:
            print(f"  {index}/{len(rows)} processed", flush=True)

    detector.close()

    total = len(rows)
    summary = {
        "manifest": str(args.manifest),
        "checkpoint": str(args.checkpoint),
        "images": total,
        "hand_detected": detected,
        "hand_detection_rate": detected / total,
        "baseline_accuracy_full_frame": baseline_correct / total,
        "baseline_correct": baseline_correct,
        "crops": {
            str(padding): {
                "scored": crop_scored[padding],
                "correct": crop_correct[padding],
                # Accuracy over every dev image: an undetected hand counts as a
                # miss, because in a real camera pipeline it is one.
                "accuracy_over_all_images": crop_correct[padding] / total,
                # Accuracy over only the images where a hand was found, which is
                # the ceiling this approach could reach with perfect detection.
                "accuracy_where_detected": (
                    crop_correct[padding] / crop_scored[padding] if crop_scored[padding] else None
                ),
            }
            for padding in args.padding
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
