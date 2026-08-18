"""Self-contained smoke workflow that requires no downloaded dataset or account."""

from __future__ import annotations

import json
import random
from pathlib import Path


def generate_fixture(
    output_root: Path,
    *,
    images_per_class: int = 8,
    image_size: int = 80,
    seed: int = 42,
) -> Path:
    """Generate a deterministic A-Z image fixture for pipeline validation."""
    from PIL import Image, ImageDraw

    from .constants import CLASS_NAMES

    rng = random.Random(seed)
    fixture_root = output_root / "fixture"
    for class_index, label in enumerate(CLASS_NAMES):
        class_dir = fixture_root / label
        class_dir.mkdir(parents=True, exist_ok=True)
        base_color = (
            30 + (class_index * 47) % 190,
            30 + (class_index * 79) % 190,
            30 + (class_index * 113) % 190,
        )
        for image_index in range(images_per_class):
            image = Image.new("RGB", (image_size, image_size), color=(245, 245, 245))
            draw = ImageDraw.Draw(image)
            jitter = rng.randint(-3, 3)
            margin = 8 + image_index % 4
            draw.rounded_rectangle(
                (margin, margin, image_size - margin, image_size - margin),
                radius=8,
                fill=base_color,
                outline=(20, 20, 20),
                width=2,
            )
            stripe_x = 12 + (class_index % 7) * 8 + jitter
            draw.line((stripe_x, 10, stripe_x, image_size - 10), fill=(255, 255, 255), width=4)
            draw.text((image_size // 2 - 3, image_size // 2 - 6), label, fill=(0, 0, 0))
            image.save(class_dir / f"{label.lower()}_{image_index:02d}.png")
    return fixture_root


def run_smoke_workflow(output_dir: Path, *, seed: int = 42, device: str = "cpu") -> dict:
    """Exercise prepare, train, evaluate, and predict on generated data."""
    from .data import prepare_manifests, read_manifest
    from .evaluation import evaluate_model
    from .inference import Predictor
    from .training import train_model

    output_dir.mkdir(parents=True, exist_ok=True)
    source_root = generate_fixture(output_dir, seed=seed)
    manifest_dir = output_dir / "manifests"
    preparation = prepare_manifests(
        source_root,
        manifest_dir,
        seed=seed,
        train_ratio=0.625,
        validation_ratio=0.125,
        test_ratio=0.25,
        near_duplicate_threshold=0,
    )
    training = train_model(
        manifest_dir,
        source_root,
        output_dir / "training",
        epochs=1,
        batch_size=32,
        learning_rate=1e-3,
        image_size=64,
        seed=seed,
        num_workers=0,
        device=device,
        patience=1,
    )
    checkpoint = Path(training["checkpoint_path"])
    evaluation = evaluate_model(
        checkpoint,
        manifest_dir / "test.csv",
        source_root,
        output_dir / "evaluation",
        batch_size=32,
        device=device,
        scope="generated smoke fixture",
    )
    test_rows = read_manifest(manifest_dir / "test.csv")
    predictor = Predictor(checkpoint, device=device)
    prediction = predictor.predict(source_root / test_rows[0]["path"], top_k=3)
    summary = {
        "status": "ok",
        "source_root": str(source_root),
        "preparation": preparation,
        "training": training,
        "evaluation": evaluation,
        "sample_prediction": prediction,
    }
    summary_path = output_dir / "smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
