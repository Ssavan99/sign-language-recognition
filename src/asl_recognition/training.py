"""Deterministic training for the compact isolated-letter classifier."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
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
from torch.utils.data import ConcatDataset, DataLoader

from .constants import CLASS_NAMES
from .data import (
    AUGMENTATION_PROFILES,
    DEFAULT_AUGMENTATION_PROFILE,
    ManifestImageDataset,
    build_transforms,
    read_manifest,
    verify_manifest_rows,
)
from .model import build_model, count_parameters, save_checkpoint
from .resources import DEFAULT_MINIMUM_AVAILABLE_BYTES, check_available_memory, memory_report
from .robustness import StressDataset, benchmark_definition


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
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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


def _stratified_indices(rows: Sequence[dict[str, str]], per_class: int) -> list[int]:
    """Pick evenly spaced rows per class so a screening subset stays balanced.

    Even spacing rather than a leading slice matters here: images inside one
    class directory are numbered in capture order, so the first N would sample a
    single stretch of one session instead of the whole class.
    """

    if per_class <= 0:
        raise ValueError("limit_per_class must be positive")
    by_label: dict[str, list[int]] = {label: [] for label in CLASS_NAMES}
    for index, row in enumerate(rows):
        bucket = by_label.get(str(row.get("label")))
        if bucket is not None:
            bucket.append(index)
    selected: list[int] = []
    for label in CLASS_NAMES:
        available = by_label[label]
        if not available:
            continue
        take = min(per_class, len(available))
        step = len(available) / take
        selected.extend(
            available[min(int(position * step), len(available) - 1)] for position in range(take)
        )
    return sorted(set(selected))


def _limited_rows(rows: Sequence[dict[str, str]], limit: int | None) -> list[dict[str, str]]:
    """Cap a split at a total row count, round-robin across classes.

    Applied to the records rather than to a wrapped dataset so that every
    consumer -- training, validation, and the stress benchmark -- sees the same
    subset, and so the rows are known before checksum verification runs.
    """

    if limit is None:
        return list(rows)
    if limit <= 0:
        raise ValueError("limit_per_split must be positive")
    target = min(limit, len(rows))
    by_label: dict[str, list[int]] = {label: [] for label in CLASS_NAMES}
    for index, row in enumerate(rows):
        bucket = by_label.get(str(row.get("label")))
        if bucket is not None:
            bucket.append(index)
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
    return [rows[index] for index in sorted(indices)]


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


def _run_stress_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float, dict[str, dict[str, float]]]:
    """Score the frozen corruption benchmark, keeping a per-corruption breakdown.

    The breakdown is not decoration. Some corruptions resemble the photometric
    augmentation used by the stronger profiles, so an aggregate number alone
    would hide whether a profile is genuinely more invariant or merely trained on
    the same kind of change.
    """

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    per_corruption: dict[str, list[int]] = {}
    with torch.inference_mode():
        for raw_batch in loader:
            images, labels = _unpack_batch(raw_batch)
            names = raw_batch[2] if len(raw_batch) > 2 else ["unknown"] * int(labels.shape[0])
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_function(logits, labels)
            correct = (logits.argmax(dim=1) == labels).tolist()
            batch_size = int(labels.shape[0])
            total_loss += float(loss.detach()) * batch_size
            total_correct += int(sum(correct))
            total_samples += batch_size
            for name, is_correct in zip(names, correct, strict=True):
                bucket = per_corruption.setdefault(str(name), [0, 0])
                bucket[0] += int(is_correct)
                bucket[1] += 1
    if total_samples == 0:
        raise ValueError("stress benchmark produced an empty dataset")
    breakdown = {
        name: {"accuracy": hits / count, "correct": hits, "total": count}
        for name, (hits, count) in sorted(per_corruption.items())
    }
    return total_loss / total_samples, total_correct / total_samples, breakdown


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


def _read_manifest_rows(path: Path, expected_split: str) -> list[dict[str, str]]:
    """Parse and structurally validate a manifest without touching image files."""

    records = read_manifest(path)
    if not records:
        raise ValueError(f"manifest contains no rows: {path}")
    labels = {str(record["label"]) for record in records}
    unknown = labels.difference(CLASS_NAMES)
    if unknown:
        raise ValueError(f"manifest contains unknown labels: {sorted(unknown)}")
    wrong_split = {str(record["split"]) for record in records}.difference({expected_split})
    if wrong_split:
        raise ValueError(f"manifest {path} contains rows for splits {sorted(wrong_split)}")
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
    limit_per_class: int | None = None,
    minimum_available_bytes: int = DEFAULT_MINIMUM_AVAILABLE_BYTES,
    allow_low_memory: bool = False,
    augmentation_profile: str = DEFAULT_AUGMENTATION_PROFILE,
    select_on: str = "validation",
    extra_manifest_dir: Path | None = None,
    extra_source_root: Path | None = None,
    extra_repeat: int = 1,
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
    if limit_per_class is not None and limit_per_class <= 0:
        raise ValueError("limit_per_class must be positive")
    if limit_per_class is not None and limit_per_split is not None:
        raise ValueError("use either limit_per_class or limit_per_split, not both")
    if augmentation_profile not in AUGMENTATION_PROFILES:
        raise ValueError(
            f"unknown augmentation profile {augmentation_profile!r}; "
            f"expected one of {list(AUGMENTATION_PROFILES)}"
        )
    if select_on not in {"validation", "stress"}:
        raise ValueError("select_on must be 'validation' or 'stress'")
    if extra_repeat < 1:
        raise ValueError("extra_repeat must be at least 1")
    if (extra_manifest_dir is None) != (extra_source_root is None):
        raise ValueError("extra_manifest_dir and extra_source_root must be given together")

    # Check head-room before the expensive manifest verification, so an
    # overloaded host fails in seconds instead of after a full checksum sweep.
    memory_preflight = check_available_memory(
        minimum_available_bytes, allow_low_memory=allow_low_memory
    )

    train_manifest = _resolve_manifest(manifest_dir, "train")
    validation_manifest = _resolve_manifest(manifest_dir, "validation")
    train_records = _read_manifest_rows(train_manifest, "train")
    validation_records = _read_manifest_rows(validation_manifest, "validation")
    train_hashes = {record["sha256"] for record in train_records}
    validation_hashes = {record["sha256"] for record in validation_records}
    overlap = train_hashes.intersection(validation_hashes)
    if overlap:
        raise ValueError(
            "train and validation manifests contain exact duplicate image bytes "
            f"({len(overlap)} shared SHA-256 values)"
        )
    # The comparison sets have served their only purpose. On the full split they
    # hold about 125,000 hex digests, which is worth releasing before the model
    # and its batches start competing for the same memory.
    primary_sha = train_hashes | validation_hashes
    del train_hashes, validation_hashes, overlap
    raw_counts = {
        "train": len(train_records),
        "validation": len(validation_records),
    }
    selected_device = resolve_device(device)
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    if limit_per_class is not None:
        train_records = [
            train_records[index] for index in _stratified_indices(train_records, limit_per_class)
        ]
        validation_records = [
            validation_records[index]
            for index in _stratified_indices(validation_records, limit_per_class)
        ]
    elif limit_per_split is not None:
        train_records = _limited_rows(train_records, limit_per_split)
        validation_records = _limited_rows(validation_records, limit_per_split)

    # Checksum-verify the rows this run will actually read. The cross-split
    # overlap check above already used the complete manifests, so narrowing the
    # file hashing to the consumed subset costs no safety and saves a screening
    # run from hashing 62,400 images to look at a few hundred.
    train_records = verify_manifest_rows(train_records, source_root, "train")
    validation_records = verify_manifest_rows(validation_records, source_root, "validation")

    # The records were parsed and checksum-verified above; handing them straight
    # to the datasets avoids a second parse of the same two files.
    train_dataset = ManifestImageDataset(
        train_manifest,
        source_root,
        build_transforms(image_size, training=True, profile=augmentation_profile),
        rows=train_records,
    )
    validation_dataset = ManifestImageDataset(
        validation_manifest,
        source_root,
        build_transforms(image_size, training=False),
        rows=validation_records,
    )
    # The stress benchmark reuses the validation rows that were just verified, so
    # it never reads an image the run has not already checksummed.
    stress_dataset = StressDataset(
        validation_records,
        source_root,
        build_transforms(image_size, training=False),
    )
    # Each dataset took its own shallow copy, so this releases the list objects
    # rather than the row dicts. The set deletion above is the one that actually
    # reclaims memory.
    # A supplementary corpus contributes training data only. It never joins the
    # validation split or the stress benchmark, so the selection signal keeps
    # measuring the same thing it did before the pool was enlarged.
    extra_summary: dict[str, Any] | None = None
    extra_diagnostic_dataset = None
    if extra_manifest_dir is not None:
        extra_manifest_dir = Path(extra_manifest_dir)
        extra_source_root = Path(extra_source_root)
        extra_train_manifest = _resolve_manifest(extra_manifest_dir, "train")
        extra_train_records = _read_manifest_rows(extra_train_manifest, "train")
        # Apply the same subsetting to both corpora. Limiting only the primary
        # would silently invert their ratio, so a screening run would train on a
        # pool that looks nothing like the full-scale one it is meant to predict.
        if limit_per_class is not None:
            extra_train_records = [
                extra_train_records[index]
                for index in _stratified_indices(extra_train_records, limit_per_class)
            ]
        elif limit_per_split is not None:
            extra_train_records = _limited_rows(extra_train_records, limit_per_split)
        extra_train_records = verify_manifest_rows(extra_train_records, extra_source_root, "train")
        extra_sha = {record["sha256"] for record in extra_train_records}
        # Also guard the untouched test partition. Training on it would inflate
        # exactly the score the enlargement is meant to be judged against.
        guarded = set(primary_sha)
        try:
            test_manifest = _resolve_manifest(manifest_dir, "test")
        except FileNotFoundError:
            test_manifest = None
        if test_manifest is not None:
            guarded.update(row["sha256"] for row in read_manifest(test_manifest))
        contaminated = extra_sha.intersection(guarded)
        if contaminated:
            raise ValueError(
                "supplementary training manifest shares "
                f"{len(contaminated)} exact images with the primary train, validation, "
                "or test splits"
            )
        del guarded
        extra_dataset = ManifestImageDataset(
            extra_train_manifest,
            extra_source_root,
            build_transforms(image_size, training=True, profile=augmentation_profile),
            rows=extra_train_records,
        )
        train_dataset = ConcatDataset([train_dataset] + [extra_dataset] * extra_repeat)
        extra_summary = {
            "manifest_dir": str(extra_manifest_dir.resolve()),
            "source_root": str(extra_source_root.resolve()),
            "manifest_sha256": _sha256(extra_train_manifest),
            "unique_images": len(extra_dataset),
            "repeat": extra_repeat,
            "contributed_samples": len(extra_dataset) * extra_repeat,
            "share_of_training_pool": (len(extra_dataset) * extra_repeat) / len(train_dataset),
        }

        # A held-out slice of the supplement is a second-domain diagnostic. It is
        # reported, never selected on: the pre-registered selector is stress-v1.
        for split in ("test", "validation"):
            try:
                diagnostic_manifest = _resolve_manifest(extra_manifest_dir, split)
            except FileNotFoundError:
                continue
            diagnostic_rows = verify_manifest_rows(
                _read_manifest_rows(diagnostic_manifest, split), extra_source_root, split
            )
            extra_diagnostic_dataset = ManifestImageDataset(
                diagnostic_manifest,
                extra_source_root,
                build_transforms(image_size, training=False),
                rows=diagnostic_rows,
            )
            extra_summary["diagnostic_split"] = split
            extra_summary["diagnostic_samples"] = len(extra_diagnostic_dataset)
            break
        del extra_train_records, extra_sha

    del train_records, validation_records
    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        # Pinned host memory is optional and can destabilize older Windows WDDM
        # CUDA drivers. The portable default favors reliable transfers.
        "pin_memory": False,
        "worker_init_fn": _seed_worker if num_workers else None,
        "generator": generator,
        "persistent_workers": num_workers > 0,
    }
    # Only the training loader gets worker processes. The two inference passes are
    # comparatively cheap, and three persistent worker pools would spend memory
    # the preflight above already promised was available.
    eval_loader_options = {
        **loader_options,
        "num_workers": 0,
        "worker_init_fn": None,
        "persistent_workers": False,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **eval_loader_options)
    stress_loader = DataLoader(stress_dataset, shuffle=False, **eval_loader_options)
    extra_loader = (
        DataLoader(extra_diagnostic_dataset, shuffle=False, **eval_loader_options)
        if extra_diagnostic_dataset is not None
        else None
    )

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
        "limit_per_class": limit_per_class,
        "minimum_available_bytes": int(minimum_available_bytes),
        "allow_low_memory": bool(allow_low_memory),
        "augmentation_profile": augmentation_profile,
        "select_on": select_on,
    }

    checkpoint_path = output_dir / "best_model.pt"
    epoch_history: list[dict[str, Any]] = []
    best_selection_loss = float("inf")
    best_validation_loss = float("inf")
    best_stress_loss = float("inf")
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
        stress_loss, stress_accuracy, stress_breakdown = _run_stress_epoch(
            model, stress_loader, loss_function, selected_device
        )
        extra_accuracy = None
        if extra_loader is not None:
            _, extra_accuracy = _run_epoch(
                model, extra_loader, loss_function, selected_device, None
            )
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "stress_loss": stress_loss,
            "stress_accuracy": stress_accuracy,
            "stress_per_corruption": stress_breakdown,
            "extra_domain_accuracy": extra_accuracy,
            "duration_seconds": time.perf_counter() - epoch_start,
            "memory": memory_report(),
        }
        epoch_history.append(epoch_record)
        # A full-split run takes hours. Without a heartbeat there is no way to
        # tell a slow epoch from a wedged process, so progress goes to stderr
        # where it stays out of the JSON that stdout carries.
        resident = epoch_record["memory"].get("resident_bytes")
        heartbeat = [
            f"epoch {epoch}/{epochs}",
            f"train_loss={train_loss:.4f}",
            f"train_acc={train_accuracy:.4f}",
            f"val_loss={validation_loss:.4f}",
            f"val_acc={validation_accuracy:.4f}",
            f"stress_acc={stress_accuracy:.4f}",
        ]
        if extra_accuracy is not None:
            heartbeat.append(f"extra_acc={extra_accuracy:.4f}")
        heartbeat.append(f"{epoch_record['duration_seconds']:.1f}s")
        if resident:
            heartbeat.append(f"rss={resident / 1024**3:.2f}GiB")
        print(" ".join(heartbeat), file=sys.stderr, flush=True)

        selection_loss = stress_loss if select_on == "stress" else validation_loss
        if selection_loss < best_selection_loss:
            best_selection_loss = selection_loss
            best_validation_loss = validation_loss
            best_stress_loss = stress_loss
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
                    "selection": {
                        "select_on": select_on,
                        "selection_loss": best_selection_loss,
                        "stress_loss": best_stress_loss,
                        "stress_accuracy": stress_accuracy,
                        "stress_benchmark": benchmark_definition(stress_dataset),
                    },
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
        "best_stress_loss": best_stress_loss,
        "select_on": select_on,
        "augmentation_profile": augmentation_profile,
        "stress_benchmark": benchmark_definition(stress_dataset),
        "epochs_completed": len(epoch_history),
        "stopped_early": stopped_early,
        "duration_seconds": duration_seconds,
        "checkpoint_sha256": artifact_hash,
        "memory_preflight": memory_preflight,
        "memory_final": memory_report(),
        "peak_resident_bytes": max(
            (
                int(record["memory"]["peak_resident_bytes"])
                for record in epoch_history
                if "peak_resident_bytes" in record["memory"]
            ),
            default=None,
        ),
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
            "stress": len(stress_dataset),
        },
        "augmentation_profile": augmentation_profile,
        "select_on": select_on,
        "stress_benchmark": benchmark_definition(stress_dataset),
        "supplementary_source": extra_summary,
        "class_names": list(CLASS_NAMES),
        "parameter_count": count_parameters(model),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": artifact_hash,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_stress_loss": best_stress_loss,
        "best_epoch_metrics": next(
            (record for record in epoch_history if record["epoch"] == best_epoch), None
        ),
        "epochs_completed": len(epoch_history),
        "stopped_early": stopped_early,
        "memory_preflight": memory_preflight,
        "memory_final": memory_report(),
        "peak_resident_bytes": history_document["peak_resident_bytes"],
    }
    _write_json(output_dir / "history.json", history_document)
    _write_json(output_dir / "run_metadata.json", metadata)
    return metadata


__all__ = ["resolve_device", "seed_everything", "train_model"]
