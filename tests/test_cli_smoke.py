from __future__ import annotations

import json
from pathlib import Path

import pytest

from asl_recognition import smoke
from asl_recognition.cli import build_parser, main


def test_cli_exposes_supported_commands() -> None:
    parser = build_parser()
    for command in (
        "doctor",
        "download",
        "prepare",
        "prepare-external",
        "train",
        "evaluate",
        "predict",
        "demo",
        "smoke",
    ):
        assert command in parser.format_help()


def test_doctor_outputs_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert {"python", "platform", "packages", "cuda"} <= report.keys()
    assert report["packages"]["torch"]["available"] is True


def test_synthetic_end_to_end_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = smoke.generate_fixture

    def tiny_fixture(output_root: Path, *, seed: int) -> Path:
        return original(output_root, images_per_class=3, image_size=48, seed=seed)

    monkeypatch.setattr(smoke, "generate_fixture", tiny_fixture)
    result = smoke.run_smoke_workflow(tmp_path / "smoke", seed=23, device="cpu")

    assert result["status"] == "ok"
    assert result["preparation"]["counts"]["per_split"] == {
        "train": 26,
        "validation": 26,
        "test": 26,
    }
    assert result["evaluation"]["metrics"]["sample_count"] == 26
    assert result["sample_prediction"]["predicted_class"]
    assert Path(result["training"]["checkpoint_path"]).is_file()
    assert Path(result["summary_path"]).is_file()
