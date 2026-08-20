"""Tests for host and process memory reporting and the training preflight."""

from __future__ import annotations

import pytest

from asl_recognition import resources
from asl_recognition.resources import (
    InsufficientMemoryError,
    check_available_memory,
    memory_report,
    process_memory,
)


def test_process_memory_values_are_plausible_or_absent() -> None:
    report = process_memory()
    assert isinstance(report, dict)
    for key, value in report.items():
        assert key.endswith("_bytes")
        assert isinstance(value, int)
        # A live interpreter always holds more than a megabyte and, on any
        # machine this project supports, far less than a terabyte.
        assert 1024**2 < value < 1024**4


def test_memory_report_is_json_serialisable_and_consistent() -> None:
    report = memory_report()
    assert set(report).issubset(
        {"resident_bytes", "peak_resident_bytes", "commit_bytes", "host_available_bytes"}
    )
    if "resident_bytes" in report and "peak_resident_bytes" in report:
        assert report["peak_resident_bytes"] >= report["resident_bytes"]


def test_preflight_passes_when_the_floor_is_trivial() -> None:
    report = check_available_memory(1)
    assert report["preflight"] in {"ok", "unknown"}
    assert report["minimum_available_bytes"] == 1
    assert report["allow_low_memory"] is False


def test_preflight_refuses_an_impossible_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resources, "available_memory", lambda: 512 * 1024**2)
    with pytest.raises(InsufficientMemoryError) as excinfo:
        check_available_memory(4 * 1024**3)
    assert "--allow-low-memory" in str(excinfo.value)


def test_preflight_override_records_that_it_was_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resources, "available_memory", lambda: 512 * 1024**2)
    report = check_available_memory(4 * 1024**3, allow_low_memory=True)
    assert report["preflight"] == "overridden"
    assert report["allow_low_memory"] is True


def test_preflight_reports_unknown_rather_than_guessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(resources, "available_memory", lambda: None)
    report = check_available_memory(4 * 1024**3)
    assert report["preflight"] == "unknown"
    assert "host_available_bytes" not in report


def test_negative_floor_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        check_available_memory(-1)
