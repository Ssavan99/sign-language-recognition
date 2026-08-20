"""Assert that the numbers quoted in prose match the released model.

The model replacement that introduced this file updated every figure that changed
*value* but missed several that were *derived* from those values: raw correct
counts, the checkpoint's file size, its best epoch, and the per-class extremes in
two figure captions. Each was individually small; together they left the public
website denying a real regression.

Values live in `docs/results/current/metrics.json`, generated from the evaluation
artifacts. These tests read the prose and fail when it disagrees, so the next
model swap cannot quietly leave a stale number behind.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "docs/results/current/metrics.json").read_text(encoding="utf-8"))

PROSE_FILES = (
    "README.md",
    "models/README.md",
    "docs/results/current/README.md",
    "docs/results/robustness.md",
    "docs/demo/README.md",
    "site/index.html",
    "src/asl_recognition/demo.py",
)


def _text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def _flat(relative: str) -> str:
    """Prose with runs of whitespace collapsed, so a line wrap is not a failure."""

    return " ".join(_text(relative).split())


def _percent(value: float) -> str:
    return f"{value * 100:.2f}"


def test_metrics_file_matches_the_released_checkpoint() -> None:
    checkpoint = ROOT / METRICS["checkpoint"]["path"]
    assert checkpoint.is_file()
    assert checkpoint.stat().st_size == METRICS["checkpoint"]["size_bytes"]

    import hashlib

    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == METRICS["checkpoint"]["sha256"]


def test_accuracy_and_macro_f1_are_quoted_consistently() -> None:
    for split in ("internal", "external"):
        accuracy = _percent(METRICS[split]["accuracy"])
        macro_f1 = _percent(METRICS[split]["macro_f1"])
        assert f"{accuracy}%" in _text("docs/results/current/README.md")
        assert f"{macro_f1}%" in _text("docs/results/current/README.md")
        assert f"{accuracy}%" in _text("README.md")
        assert f"{accuracy}%" in _text("site/index.html")


def test_raw_correct_counts_are_quoted_consistently() -> None:
    # The counts, not just the percentages: a stale numerator silently implies a
    # different accuracy than the table beside it.
    internal = METRICS["internal"]
    external = METRICS["external"]
    readme = _flat("README.md")
    results = _flat("docs/results/current/README.md")
    site = _flat("site/index.html")

    assert f"{internal['correct']:,} of {internal['sample_count']:,}" in results
    assert f"{internal['correct']:,} of {internal['sample_count']:,}" in readme
    assert f"{internal['correct']:,} / {internal['sample_count']:,}" in site
    assert f"{external['correct']} of {external['sample_count']}" in results
    assert f"{external['correct']} of {external['sample_count']}" in readme
    assert f"{external['correct']} / {external['sample_count']}" in site


def test_external_zero_recall_classes_are_reported_accurately() -> None:
    zero = METRICS["external"]["zero_recall_classes"]
    assert zero, "expected at least one zero-recall class to describe"
    site = _text("site/index.html")
    results = _text("docs/results/current/README.md")
    # Naming a class that is no longer zero-recall would overstate the failure;
    # omitting one that is would understate it.
    for label, recall in METRICS["external"]["per_class_recall"].items():
        if recall == 0:
            assert f"0% for {label}" in site or f"0% for {label}" in results


def test_worst_internal_class_is_not_hidden() -> None:
    recalls = METRICS["internal"]["per_class_recall"]
    worst_label = min(recalls, key=lambda label: recalls[label])
    worst = _percent(recalls[worst_label])
    # This is the cost of the augmentation change on the same-corpus test. It must
    # appear beside the internal result, not only in the raw artifacts.
    assert f"{worst_label} " in _text("docs/results/current/README.md")
    assert f"{worst}%" in _text("docs/results/current/README.md")
    assert f"{worst}%" in _text("site/index.html")


def test_best_external_class_is_reported_accurately() -> None:
    recalls = METRICS["external"]["per_class_recall"]
    best_label = max(recalls, key=lambda label: recalls[label])
    best = _percent(recalls[best_label])
    assert f"{best}% for {best_label}" in _text("site/index.html")
    assert f"{best}%" in _text("docs/results/current/README.md")


def test_checkpoint_identity_is_quoted_consistently() -> None:
    checkpoint = METRICS["checkpoint"]
    name = Path(checkpoint["path"]).name
    assert checkpoint["sha256"] in _text("README.md")
    assert checkpoint["sha256"] in _text("models/README.md")
    assert checkpoint["sha256"] in _text("docs/results/current/README.md")
    assert f"{checkpoint['size_bytes']:,} bytes" in _text("models/README.md")
    assert f"{checkpoint['best_epoch']} of {checkpoint['epochs_completed']}" in _text(
        "models/README.md"
    )
    assert checkpoint["augmentation_profile"] in _text("models/README.md")
    assert checkpoint["augmentation_profile"] in _text("docs/results/current/README.md")
    for relative in ("README.md", "docs/demo/README.md", "models/README.md"):
        assert name in _text(relative)


def test_no_reference_to_the_superseded_model_survives_unlabelled() -> None:
    previous = METRICS["previous_released_model"]
    old_hash = previous["sha256"]
    old_external = _percent(previous["external_accuracy"])
    for relative in PROSE_FILES:
        assert old_hash not in _text(relative), relative

    # The old external score may still appear, but only where it is explicitly
    # framed as historical. An unframed occurrence is a stale current claim.
    framing = (
        "previous",
        "Previous",
        "up from",
        "Down from",
        "down from",
        "nearly doubled",
        "was trained with",
        "scored",
        # The pre-registered publication rule names the score it had to beat.
        "improves by",
    )
    needle = f"{old_external}%"
    for relative in PROSE_FILES:
        flat = _flat(relative)
        start = 0
        while (index := flat.find(needle, start)) != -1:
            window = flat[max(0, index - 220) : index + 220]
            assert any(word in window for word in framing), (
                f"{relative} quotes the superseded {needle} without framing it as historical: "
                f"...{window[:160]}..."
            )
            start = index + len(needle)


@pytest.mark.parametrize("relative", PROSE_FILES)
def test_current_external_accuracy_appears_wherever_the_limit_is_stated(relative: str) -> None:
    text = _text(relative)
    current = _percent(METRICS["external"]["accuracy"])
    if "External-domain accuracy" in text or "external-domain accuracy" in text:
        assert f"{current}%" in text
