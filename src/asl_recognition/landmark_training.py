"""Train and evaluate the landmark classifier from cached feature files.

Detection has already happened by the time this runs, so training is cheap --
minutes on a CPU rather than hours. That changes what is affordable: the honest
thing is to keep the same discipline as the pixel model rather than to relax it
because the run is short.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from .constants import CLASS_NAMES
from .landmark_model import (
    FEATURE_CONTRACT,
    build_landmark_model,
    checkpoint_digest,
    count_parameters,
    save_landmark_checkpoint,
)
from .landmarks import FEATURE_DIMENSION
from .training import seed_everything


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a cached feature file and check it matches the current contract."""

    path = Path(path)
    payload = np.load(path, allow_pickle=False)
    features = payload["features"].astype(np.float32)
    labels = payload["labels"].astype(np.int64)
    if features.ndim != 2 or features.shape[1] != FEATURE_DIMENSION:
        raise ValueError(
            f"{path.name} holds {features.shape[1] if features.ndim == 2 else '?'} features "
            f"but this code expects {FEATURE_DIMENSION}; re-extract with the current contract"
        )
    if len(features) != len(labels):
        raise ValueError(f"{path.name} has {len(features)} features and {len(labels)} labels")
    if len(features) == 0:
        raise ValueError(f"{path.name} contains no samples")
    return features, labels


def _concatenate(paths: Sequence[Path]) -> tuple[np.ndarray, np.ndarray]:
    blocks = [load_features(path) for path in paths]
    return (
        np.concatenate([block[0] for block in blocks]),
        np.concatenate([block[1] for block in blocks]),
    )


def _jitter(batch: torch.Tensor, scale: float, generator: torch.Generator) -> torch.Tensor:
    """Add small Gaussian noise to landmark coordinates.

    The detector's own output wobbles between frames of the same pose, so a model
    trained on pristine coordinates is trained on a precision it will not get at
    inference. This is the landmark equivalent of image augmentation, and it is
    the only augmentation applied.
    """

    if scale <= 0:
        return batch
    noise = torch.randn(batch.shape, generator=generator, device=batch.device) * scale
    return batch + noise


def _evaluate(
    model: nn.Module,
    loader: DataLoader[Any],
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float, dict[str, dict[str, float]]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    per_class: dict[int, list[int]] = {index: [0, 0] for index in range(len(CLASS_NAMES))}
    with torch.inference_mode():
        for batch, labels in loader:
            batch = batch.to(device)
            labels = labels.to(device)
            logits = model(batch)
            loss = loss_function(logits, labels)
            predictions = logits.argmax(dim=1)
            total_loss += float(loss.detach()) * len(labels)
            correct += int((predictions == labels).sum().item())
            total += len(labels)
            for label, prediction in zip(labels.tolist(), predictions.tolist(), strict=True):
                per_class[label][1] += 1
                if label == prediction:
                    per_class[label][0] += 1
    breakdown = {
        CLASS_NAMES[index]: {
            "recall": hits / count if count else 0.0,
            "correct": hits,
            "support": count,
        }
        for index, (hits, count) in sorted(per_class.items())
        if count
    }
    return total_loss / total, correct / total, breakdown


def train_landmark_model(
    train_features: Sequence[Path],
    validation_features: Sequence[Path],
    output_dir: Path,
    *,
    epochs: int = 120,
    batch_size: int = 256,
    learning_rate: float = 2e-3,
    weight_decay: float = 1e-4,
    dropout: float = 0.3,
    hidden: Sequence[int] = (256, 128),
    jitter: float = 0.01,
    seed: int = 42,
    patience: int = 15,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train the landmark classifier, selecting the best epoch by validation loss."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    selected_device = torch.device(device)

    train_x, train_y = _concatenate([Path(path) for path in train_features])
    validation_x, validation_y = _concatenate([Path(path) for path in validation_features])

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    validation_loader = DataLoader(
        TensorDataset(torch.from_numpy(validation_x), torch.from_numpy(validation_y)),
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_landmark_model(hidden=hidden, dropout=dropout).to(selected_device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
    jitter_generator = torch.Generator(device=selected_device).manual_seed(seed)

    checkpoint_path = output_dir / "best_model.pt"
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch, labels in train_loader:
            batch = _jitter(batch.to(selected_device), jitter, jitter_generator)
            labels = labels.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch), labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(labels)
            seen += len(labels)
        scheduler.step()

        validation_loss, validation_accuracy, _ = _evaluate(
            model, validation_loader, loss_function, selected_device
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / seen,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            save_landmark_checkpoint(
                checkpoint_path,
                model,
                seed=seed,
                best_epoch=best_epoch,
                best_validation_loss=best_loss,
                extra_metadata={
                    "feature_contract": FEATURE_CONTRACT,
                    "train_features": [str(path) for path in train_features],
                    "validation_features": [str(path) for path in validation_features],
                },
            )
        else:
            stale += 1
            if stale >= patience:
                break
        if epoch % 10 == 0 or epoch == 1:
            print(
                f"epoch {epoch}/{epochs} train_loss={running / seen:.4f} "
                f"val_loss={validation_loss:.4f} val_acc={validation_accuracy:.4f}",
                file=sys.stderr,
                flush=True,
            )

    duration = time.perf_counter() - started
    metadata = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "configuration": {
            "epochs_requested": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "hidden": list(hidden),
            "jitter": jitter,
            "seed": seed,
            "patience": patience,
            "device": device,
        },
        "feature_contract": FEATURE_CONTRACT,
        "feature_dimension": FEATURE_DIMENSION,
        "train_samples": int(len(train_x)),
        "validation_samples": int(len(validation_x)),
        "train_features": [str(path) for path in train_features],
        "validation_features": [str(path) for path in validation_features],
        "parameter_count": count_parameters(model),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_digest(checkpoint_path),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    for name, payload in (("history.json", {"epochs": history}), ("run_metadata.json", metadata)):
        (output_dir / name).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return metadata


def evaluate_landmark_model(
    checkpoint: Path,
    feature_file: Path,
    output_dir: Path,
    *,
    scope: str,
    considered_images: int | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """Score a cached feature file and record detection-aware accuracy.

    ``considered_images`` is the number of images the detector was *offered*. When
    given, the result also reports accuracy over all of them, counting a failed
    detection as a miss -- which is what it is in a camera pipeline.
    """

    from .landmark_model import load_landmark_checkpoint

    model, checkpoint_metadata = load_landmark_checkpoint(Path(checkpoint), map_location=device)
    features, labels = load_features(Path(feature_file))
    selected_device = torch.device(device)
    model = model.to(selected_device)

    loader = DataLoader(
        TensorDataset(torch.from_numpy(features), torch.from_numpy(labels)),
        batch_size=256,
        shuffle=False,
    )
    loss, accuracy, per_class = _evaluate(model, loader, nn.CrossEntropyLoss(), selected_device)
    macro_f1_terms = []
    predictions_by_class: dict[str, int] = {}
    with torch.inference_mode():
        all_predictions = []
        for batch, _ in loader:
            all_predictions.extend(model(batch.to(selected_device)).argmax(dim=1).tolist())
    for index, name in enumerate(CLASS_NAMES):
        predictions_by_class[name] = sum(1 for value in all_predictions if value == index)
    for name in CLASS_NAMES:
        if name not in per_class:
            continue
        true_positive = per_class[name]["correct"]
        precision = (
            true_positive / predictions_by_class[name] if predictions_by_class[name] else 0.0
        )
        recall = per_class[name]["recall"]
        macro_f1_terms.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )

    result: dict[str, Any] = {
        "scope": scope,
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_digest(Path(checkpoint)),
        "feature_file": str(Path(feature_file).resolve()),
        "feature_contract": checkpoint_metadata["feature_contract"],
        "samples_with_landmarks": int(len(labels)),
        "loss": loss,
        "accuracy_where_detected": accuracy,
        "macro_f1_where_detected": sum(macro_f1_terms) / len(macro_f1_terms),
        "per_class": per_class,
    }
    if considered_images:
        result["images_considered"] = int(considered_images)
        result["detection_rate"] = len(labels) / considered_images
        # The number that matters for a real pipeline: an undetected hand is a
        # wrong answer, not an excused one.
        result["accuracy_over_all_images"] = accuracy * len(labels) / considered_images
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


__all__ = ["evaluate_landmark_model", "load_features", "train_landmark_model"]
