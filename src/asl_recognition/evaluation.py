"""Bounded-memory evaluation and artifact generation."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib.figure import Figure
from torch.utils.data import DataLoader

from .constants import CLASS_NAMES
from .data import ManifestImageDataset, build_transforms, read_manifest
from .model import count_parameters, load_checkpoint, preprocessing_contract
from .training import resolve_device


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("dataset batches must contain image tensors and labels")
    return batch[0], batch[1]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _classification_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    class_names: list[str],
) -> tuple[dict[str, Any], np.ndarray]:
    class_count = len(class_names)
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    np.add.at(matrix, (truth, predictions), 1)
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_count = matrix.sum(axis=0).astype(np.float64)
    support = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support != 0,
    )
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator != 0,
    )
    total = int(matrix.sum())
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(class_names)
    }
    metrics = {
        "sample_count": total,
        "accuracy": float(true_positive.sum() / total) if total else 0.0,
        "macro_f1": float(f1.mean()),
        "per_class": per_class,
    }
    return metrics, matrix


def _save_confusion_matrix(
    path: Path,
    matrix: np.ndarray,
    class_names: list[str],
) -> None:
    figure = Figure(figsize=(12, 10), constrained_layout=True)
    axis = figure.subplots()
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    indices = np.arange(len(class_names))
    axis.set(
        xticks=indices,
        yticks=indices,
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion matrix",
    )
    axis.tick_params(axis="x", labelrotation=90)
    threshold = float(matrix.max()) / 2 if matrix.size else 0.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix[row, column])
            if value:
                axis.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white" if value > threshold else "black",
                )
    figure.savefig(path, dpi=160)


def _measure_single_image_latency(
    model: torch.nn.Module,
    dataset: ManifestImageDataset,
    device: torch.device,
    maximum_samples: int = 200,
) -> dict[str, Any]:
    sample_count = min(len(dataset), maximum_samples)
    if sample_count == 0:
        raise ValueError("cannot benchmark an empty dataset")
    warmup_count = min(5, sample_count)
    model.eval()
    with torch.inference_mode():
        for index in range(warmup_count):
            image = dataset[index][0].unsqueeze(0).to(device)
            model(image)
        _synchronize(device)

        measurements: list[float] = []
        for index in range(sample_count):
            image = dataset[index][0].unsqueeze(0).to(device)
            _synchronize(device)
            started = time.perf_counter()
            model(image)
            _synchronize(device)
            measurements.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(measurements, dtype=np.float64)
    return {
        "definition": "single-image model forward pass; preprocessing and transfer excluded",
        "device": str(device),
        "sample_count": sample_count,
        "warmup_count": warmup_count,
        "p50_ms_per_image": float(np.percentile(values, 50)),
        "p95_ms_per_image": float(np.percentile(values, 95)),
    }


def evaluate_model(
    checkpoint_path: Path,
    manifest_path: Path,
    source_root: Path,
    output_dir: Path,
    batch_size: int = 64,
    device: str = "auto",
    scope: str = "same-corpus image holdout",
) -> dict[str, Any]:
    """Evaluate a checkpoint and write JSON plus a confusion-matrix PNG."""

    checkpoint_path = Path(checkpoint_path)
    manifest_path = Path(manifest_path)
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root not found: {source_root}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not scope.strip():
        raise ValueError("scope must be a non-empty label")

    selected_device = resolve_device(device)
    model, checkpoint = load_checkpoint(checkpoint_path, map_location=selected_device)
    class_names = [str(name) for name in checkpoint["class_names"]]
    if class_names != list(CLASS_NAMES):
        raise ValueError("checkpoint class map does not match the supported A-Z class order")
    image_size = int(checkpoint["image_size"])
    if checkpoint["preprocessing"] != preprocessing_contract(image_size):
        raise ValueError("checkpoint preprocessing contract is unsupported or inconsistent")
    records = read_manifest(manifest_path)
    if not records:
        raise ValueError("evaluation manifest contains no rows")
    unknown_labels = {str(row["label"]) for row in records}.difference(class_names)
    if unknown_labels:
        raise ValueError(f"manifest contains unknown labels: {sorted(unknown_labels)}")

    dataset = ManifestImageDataset(
        manifest_path,
        source_root,
        build_transforms(image_size, training=False),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.to(selected_device).eval()
    true_batches: list[np.ndarray] = []
    predicted_batches: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for raw_batch in loader:
            images, labels = _unpack_batch(raw_batch)
            logits = model(images.to(selected_device, non_blocking=True))
            predictions = logits.argmax(dim=1).cpu().numpy()
            true_batches.append(labels.cpu().numpy())
            predicted_batches.append(predictions)
    _synchronize(selected_device)
    evaluation_duration = time.perf_counter() - started
    truth = np.concatenate(true_batches).astype(np.int64, copy=False)
    predictions = np.concatenate(predicted_batches).astype(np.int64, copy=False)
    metrics, matrix = _classification_metrics(truth, predictions, class_names)
    latency = _measure_single_image_latency(model, dataset, selected_device)

    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "confusion_matrix.png"
    _save_confusion_matrix(matrix_path, matrix, class_names)
    result = {
        "scope": scope.strip(),
        "metrics": metrics,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "per_class_metrics": metrics["per_class"],
        "confusion_matrix": matrix.tolist(),
        "class_names": class_names,
        "latency": latency,
        "evaluation_duration_seconds": evaluation_duration,
        "device": str(selected_device),
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "parameter_count": count_parameters(model),
            "best_epoch": int(checkpoint["best_epoch"]),
            "best_validation_loss": float(checkpoint["best_validation_loss"]),
        },
        "manifest": {
            "path": str(manifest_path.resolve()),
            "sha256": _sha256(manifest_path),
            "row_count": len(records),
            "split_values": sorted({str(row.get("split", "")) for row in records}),
        },
        "artifacts": {
            "confusion_matrix_png": str(matrix_path.resolve()),
            "evaluation_json": str((output_dir / "evaluation.json").resolve()),
        },
    }
    _write_json(output_dir / "evaluation.json", result)
    return result


__all__ = ["evaluate_model"]
