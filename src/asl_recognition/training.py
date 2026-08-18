"""Deterministic training for the compact isolated-letter classifier."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Subset

from .constants import CLASS_NAMES
from .data import ManifestImageDataset, build_transforms, verify_manifest_files
from .model import build_model, count_parameters, save_checkpoint


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _resolve_manifest(manifest_dir: Path, split: str) -> Path:
    candidates = (
        manifest_dir / f"{split}.csv",
        manifest_dir / f"manifest_{split}.csv",
        manifest_dir / f"{split}_manifest.csv",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    expected = ", ".join(path.name for path in candidates)
    raise FileNotFoundError(f"no {split} manifest in {manifest_dir}; expected {expected}")


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit CPU/CUDA request without silently changing it."""

    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        device = torch.device(normalized)
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device does not exist: {requested}")
        return device
    raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch and request deterministic kernels."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _limited(dataset: Dataset[Any], limit: int | None) -> Dataset[Any]:
    if limit is None:
        return dataset
    if limit <= 0:
        raise ValueError("limit_per_split must be positive")
    target = min(limit, len(dataset))
    rows = getattr(dataset, "rows", None)
    if isinstance(rows, list) and all(isinstance(row, dict) for row in rows):
        by_label = {
            label: [index for index, row in enumerate(rows) if row.get("label") == label]
            for label in CLASS_NAMES
        }
        indices: list[int] = []
        offset = 0
        while len(indices) < target:
            added = False
            for label in CLASS_NAMES:
                if offset < len(by_label[label]) and len(indices) < target:
                    indices.append(by_label[label][offset])
                    added = True
            if not added:
                break
            offset += 1
        return Subset(dataset, indices)
    return Subset(dataset, range(target))


def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise ValueError("dataset batches must contain image tensors and labels")
    return batch[0], batch[1]


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    loss_function: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for raw_batch in loader:
            images, labels = _unpack_batch(raw_batch)
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach()) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("manifest produced an empty dataset")
    return total_loss / total_samples, total_correct / total_samples


def _environment(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if device.type == "cuda":
        result["cuda_device_name"] = torch.cuda.get_device_name(device)
    return result


def _validate_manifest(path: Path, source_root: Path, expected_split: str) -> list[dict[str, str]]:
    records = verify_manifest_files(path, source_root, expected_split)
    if not records:
        raise ValueError(f"manifest contains no rows: {path}")
    labels = {str(record["label"]) for record in records}
    unknown = labels.difference(CLASS_NAMES)
    if unknown:
        raise ValueError(f"manifest contains unknown labels: {sorted(unknown)}")
    return records


def train_model(
    manifest_dir: Path,
    source_root: Path,
    output_dir: Path,
    epochs: int = 12,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    image_size: int = 96,
    seed: int = 42,
    num_workers: int = 0,
    device: str = "auto",
    patience: int = 3,
    limit_per_split: int | None = None,
) -> dict[str, Any]:
    """Train a compact model and write a best checkpoint plus run evidence."""

    manifest_dir = Path(manifest_dir)
    source_root = Path(source_root)
    output_dir = Path(output_dir)
    if not manifest_dir.is_dir():
        raise FileNotFoundError(f"manifest directory not found: {manifest_dir}")
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root not found: {source_root}")
    if not 1 <= epochs <= 100:
        raise ValueError("epochs must be between 1 and 100")
    if batch_size <= 0 or learning_rate <= 0 or image_size <= 0:
        raise ValueError("batch_size, learning_rate, and image_size must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative")

    train_manifest = _resolve_manifest(manifest_dir, "train")
    validation_manifest = _resolve_manifest(manifest_dir, "validation")
    train_records = _validate_manifest(train_manifest, source_root, "train")
    validation_records = _validate_manifest(validation_manifest, source_root, "validation")
    train_hashes = {record["sha256"] for record in train_records}
    validation_hashes = {record["sha256"] for record in validation_records}
    overlap = train_hashes.intersection(validation_hashes)
    if overlap:
        raise ValueError(
            "train and validation manifests contain exact duplicate image bytes "
            f"({len(overlap)} shared SHA-256 values)"
        )
    raw_counts = {
        "train": len(train_records),
        "validation": len(validation_records),
    }
    selected_device = resolve_device(device)
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = _limited(
        ManifestImageDataset(
            train_manifest,
            source_root,
            build_transforms(image_size, training=True),
        ),
        limit_per_split,
    )
    validation_dataset = _limited(
        ManifestImageDataset(
            validation_manifest,
            source_root,
            build_transforms(image_size, training=False),
        ),
        limit_per_split,
    )
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": selected_device.type == "cuda",
        "worker_init_fn": _seed_worker if num_workers else None,
        "generator": generator,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model = build_model(num_classes=len(CLASS_NAMES)).to(selected_device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    manifest_hashes = {
        "train": _sha256(train_manifest),
        "validation": _sha256(validation_manifest),
    }
    configuration = {
        "epochs_requested": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "image_size": image_size,
        "seed": seed,
        "num_workers": num_workers,
        "device_requested": device,
        "patience": patience,
        "limit_per_split": limit_per_split,
    }

    checkpoint_path = output_dir / "best_model.pt"
    epoch_history: list[dict[str, Any]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started_at = datetime.now(timezone.utc)
    run_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = _run_epoch(
            model, train_loader, loss_function, selected_device, optimizer
        )
        validation_loss, validation_accuracy = _run_epoch(
            model, validation_loader, loss_function, selected_device, None
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "duration_seconds": time.perf_counter() - epoch_start,
        }
        epoch_history.append(epoch_record)

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path,
                model,
                image_size=image_size,
                seed=seed,
                best_epoch=best_epoch,
                best_validation_loss=best_validation_loss,
                class_names=CLASS_NAMES,
                extra_metadata={
                    "manifest_sha256": manifest_hashes,
                    "training_configuration": configuration,
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    duration_seconds = time.perf_counter() - run_start
    artifact_hash = _sha256(checkpoint_path)
    finished_at = datetime.now(timezone.utc)
    stopped_early = len(epoch_history) < epochs
    history_document = {
        "epochs": epoch_history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(epoch_history),
        "stopped_early": stopped_early,
        "duration_seconds": duration_seconds,
        "checkpoint_sha256": artifact_hash,
    }
    metadata = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": duration_seconds,
        "configuration": configuration,
        "device": str(selected_device),
        "environment": _environment(selected_device),
        "manifest_paths": {
            "train": str(train_manifest.resolve()),
            "validation": str(validation_manifest.resolve()),
        },
        "manifest_sha256": manifest_hashes,
        "manifest_row_counts": raw_counts,
        "samples_used": {
            "train": len(train_dataset),
            "validation": len(validation_dataset),
        },
        "class_names": list(CLASS_NAMES),
        "parameter_count": count_parameters(model),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": artifact_hash,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(epoch_history),
        "stopped_early": stopped_early,
    }
    _write_json(output_dir / "history.json", history_document)
    _write_json(output_dir / "run_metadata.json", metadata)
    return metadata


__all__ = ["resolve_device", "seed_everything", "train_model"]
