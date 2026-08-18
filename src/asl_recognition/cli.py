"""Command-line interface for the maintained pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _print_result(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=str))


def _add_device_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Execution device. 'auto' uses CUDA when available.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="asl-recognition",
        description="Reproducible isolated-image ASL alphabet classification.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Report runtime and optional dependency status.")
    doctor.set_defaults(handler=_doctor)

    download = subparsers.add_parser(
        "download", help="Download a public dataset anonymously with kagglehub."
    )
    download.add_argument("kind", choices=("primary", "external"))
    download.add_argument("--output-root", type=_path, default=_path("data/raw"))
    download.set_defaults(handler=_download)

    prepare = subparsers.add_parser(
        "prepare", help="Create deterministic A-Z train/validation/test manifests."
    )
    prepare.add_argument("--source-root", type=_path, required=True)
    prepare.add_argument("--output-dir", type=_path, default=_path("data/manifests"))
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--train-ratio", type=float, default=0.70)
    prepare.add_argument("--validation-ratio", type=float, default=0.15)
    prepare.add_argument("--test-ratio", type=float, default=0.15)
    prepare.add_argument("--near-duplicate-threshold", type=int, default=5)
    prepare.set_defaults(handler=_prepare)

    external = subparsers.add_parser(
        "prepare-external", help="Create an A-Z manifest for a separate-source dataset."
    )
    external.add_argument("--source-root", type=_path, required=True)
    external.add_argument("--output-file", type=_path, default=_path("data/manifests/external.csv"))
    external.set_defaults(handler=_prepare_external)

    train = subparsers.add_parser("train", help="Train and export the compact classifier.")
    train.add_argument("--manifest-dir", type=_path, default=_path("data/manifests"))
    train.add_argument("--source-root", type=_path, required=True)
    train.add_argument("--output-dir", type=_path, default=_path("artifacts/training"))
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--image-size", type=int, default=96)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--patience", type=int, default=3)
    train.add_argument("--limit-per-split", type=int)
    _add_device_argument(train)
    train.set_defaults(handler=_train)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate an exported checkpoint.")
    evaluate.add_argument("--checkpoint", type=_path, required=True)
    evaluate.add_argument("--manifest", type=_path, required=True)
    evaluate.add_argument("--source-root", type=_path, required=True)
    evaluate.add_argument("--output-dir", type=_path, default=_path("artifacts/evaluation"))
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.add_argument("--scope", default="same-corpus image holdout")
    _add_device_argument(evaluate)
    evaluate.set_defaults(handler=_evaluate)

    predict = subparsers.add_parser("predict", help="Classify one image with an exported model.")
    predict.add_argument("image", type=_path)
    predict.add_argument("--checkpoint", type=_path, required=True)
    predict.add_argument("--top-k", type=int, default=3)
    predict.add_argument("--confidence-threshold", type=float, default=0.55)
    _add_device_argument(predict)
    predict.set_defaults(handler=_predict)

    smoke = subparsers.add_parser(
        "smoke", help="Run a no-download prepare/train/evaluate/predict workflow."
    )
    smoke.add_argument("--output-dir", type=_path, default=_path("artifacts/smoke"))
    smoke.add_argument("--seed", type=int, default=42)
    _add_device_argument(smoke)
    smoke.set_defaults(handler=_smoke)
    return parser


def _doctor(_: argparse.Namespace) -> dict:
    import importlib.metadata
    import importlib.util
    import platform

    packages = {}
    for name in ("torch", "torchvision", "numpy", "Pillow", "scikit-learn", "kagglehub", "gradio"):
        module_name = {"Pillow": "PIL", "scikit-learn": "sklearn"}.get(name, name)
        present = importlib.util.find_spec(module_name) is not None
        version = None
        if present:
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        packages[name] = {"available": present, "version": version}
    cuda = {"available": False, "device": None}
    if packages["torch"]["available"]:
        import torch

        cuda["available"] = torch.cuda.is_available()
        if cuda["available"]:
            cuda["device"] = torch.cuda.get_device_name(0)
    result = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
        "cuda": cuda,
    }
    _print_result(result)
    return result


def _download(args: argparse.Namespace) -> dict:
    from .data import download_public_dataset

    path = download_public_dataset(args.kind, args.output_root)
    result = {"kind": args.kind, "path": str(path)}
    _print_result(result)
    return result


def _prepare(args: argparse.Namespace) -> dict:
    from .data import prepare_manifests

    result = prepare_manifests(
        args.source_root,
        args.output_dir,
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    _print_result(result)
    return result


def _prepare_external(args: argparse.Namespace) -> dict:
    from .data import prepare_external_manifest

    result = prepare_external_manifest(args.source_root, args.output_file)
    _print_result(result)
    return result


def _train(args: argparse.Namespace) -> dict:
    from .training import train_model

    result = train_model(
        args.manifest_dir,
        args.source_root,
        args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=args.num_workers,
        device=args.device,
        patience=args.patience,
        limit_per_split=args.limit_per_split,
    )
    _print_result(result)
    return result


def _evaluate(args: argparse.Namespace) -> dict:
    from .evaluation import evaluate_model

    result = evaluate_model(
        args.checkpoint,
        args.manifest,
        args.source_root,
        args.output_dir,
        batch_size=args.batch_size,
        device=args.device,
        scope=args.scope,
    )
    _print_result(result)
    return result


def _predict(args: argparse.Namespace) -> dict:
    from .inference import Predictor

    predictor = Predictor(
        args.checkpoint,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )
    result = predictor.predict(args.image, top_k=args.top_k)
    _print_result(result)
    return result


def _smoke(args: argparse.Namespace) -> dict:
    from .smoke import run_smoke_workflow

    result = run_smoke_workflow(args.output_dir, seed=args.seed, device=args.device)
    _print_result(result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 0
