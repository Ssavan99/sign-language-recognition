"""Capture the README demo screenshot from the running local interface.

The screenshot in `docs/demo/` is evidence, not decoration: it shows the released
model classifying a held-out external image, and it has to keep matching whatever
model is actually released. Regenerating it by hand invites drift, so this script
drives the real Gradio app and captures what it renders.

It is a development tool. Playwright is not a project dependency; install it only
when regenerating the image::

    python -m pip install playwright
    python -m playwright install chromium

Then, with the demo already running on the given port::

    asl-recognition demo --device cpu --port 7861
    python tools/capture_demo_screenshot.py --port 7861

The demo defaults to the released checkpoint, so no path is needed.

The capture deliberately keeps whatever the model predicts, including a wrong
answer. A curated success case would misrepresent the model's external-domain
behaviour.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "demo" / "demo-screenshot.png"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1520)
    parser.add_argument("--height", type=int, default=1180)
    args = parser.parse_args()

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright is not installed. Run:\n"
            "  python -m pip install playwright\n"
            "  python -m playwright install chromium",
            file=sys.stderr,
        )
        return 2

    url = f"http://127.0.0.1:{args.port}"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.goto(url, wait_until="networkidle")

        # Load the committed external sample through the app's own example, so the
        # captured prediction is the same one the CLI reproduces.
        page.get_by_text("External-domain sample", exact=False).click()
        page.wait_for_timeout(1500)
        page.get_by_role("button", name="Classify image").click()

        # Wait on every terminal state the interface can reach, not just the
        # confident one. A model that is merely uncertain on the sample is
        # arguably the more honest evidence, and gating only on "Predicted
        # letter" would time out rather than capture it -- leaving the committed
        # screenshot silently stale.
        terminal = ("Predicted letter", "Treat this as uncertain", "Unable to classify")
        page.wait_for_function(
            "(states) => states.some((state) => document.body.innerText.includes(state))",
            arg=list(terminal),
            timeout=30_000,
        )
        page.wait_for_timeout(800)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(args.output))
        text = page.inner_text("body")
        browser.close()

    resolved = args.output.resolve()
    try:
        shown = resolved.relative_to(ROOT)
    except ValueError:
        # A path outside the repository, or a relative one, is still valid output.
        shown = resolved
    print(f"wrote {shown}")
    reported = [
        line.strip()
        for line in text.splitlines()
        if any(state in line for state in terminal) or "Model confidence" in line
    ]
    for line in reported:
        print(f"  {line}")
    if not reported:
        print("  warning: no prediction text found; check the capture", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
