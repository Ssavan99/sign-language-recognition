"""Capture the README screenshot from the project website.

The README image is evidence, not decoration: it has to show what the published
site actually does with the currently released models. Capturing it by hand
invites drift, so this serves `site/` locally and drives the real page.

It is a development tool. Playwright is not a project dependency; install it only
when regenerating the image::

    python -m pip install playwright
    python -m playwright install chromium
    python tools/capture_site_screenshot.py

The capture waits for both classifiers to report ready, so the image can never
show a half-loaded page or a stale prediction from the previous frame.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import socket
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DEFAULT_OUTPUT = ROOT / "docs" / "demo" / "site-screenshot.png"


@contextlib.contextmanager
def serve(directory: Path):
    """Serve a directory on a free port for the lifetime of the block."""

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/index.html"
    finally:
        server.shutdown()
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--full-page",
        action="store_true",
        help="Capture the whole document rather than the input and prediction panels.",
    )
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

    with serve(SITE) as url, sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(
            viewport={"width": args.width, "height": args.height},
            device_scale_factor=2,
        )
        page.goto(url, wait_until="networkidle")

        # Both models must be loaded and a prediction rendered, otherwise the
        # image would show a loading state or the previous frame's answer.
        page.wait_for_function(
            "() => { const s = document.getElementById('model-state');"
            " const t = document.getElementById('prediction-title');"
            " return s && t && /ready/i.test(s.textContent) && t.textContent.trim() !== '—'; }",
            timeout=45_000,
        )
        page.wait_for_timeout(1200)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.full_page:
            page.screenshot(path=str(args.output), full_page=True)
        else:
            # Frame the input and prediction panels only. The hero text, the
            # reference chart, and the sample grid are all worth having on the
            # page and none of them are what the image is evidence of.
            clip = page.evaluate(
                """() => {
                    const boxes = ['.input-panel', '.prediction-panel']
                        .map((selector) => document.querySelector(selector))
                        .filter(Boolean)
                        .map((node) => node.getBoundingClientRect());
                    if (!boxes.length) return null;
                    const scroll = { x: window.scrollX, y: window.scrollY };
                    const left = Math.min(...boxes.map((box) => box.left));
                    const top = Math.min(...boxes.map((box) => box.top));
                    const right = Math.max(...boxes.map((box) => box.right));
                    const bottom = Math.max(...boxes.map((box) => box.bottom));
                    const pad = 18;
                    return {
                        x: left + scroll.x - pad,
                        y: top + scroll.y - pad,
                        width: right - left + pad * 2,
                        height: bottom - top + pad * 2,
                    };
                }"""
            )
            page.screenshot(path=str(args.output), full_page=True, clip=clip)

        engine = page.eval_on_selector("input[name=engine]:checked", "node => node.value")
        letter = page.inner_text("#prediction-title").strip()
        confidence = page.inner_text("#prediction-confidence").strip()
        browser.close()

    resolved = args.output.resolve()
    try:
        shown = resolved.relative_to(ROOT)
    except ValueError:
        shown = resolved
    print(f"wrote {shown}")
    print(f"  engine: {engine}")
    print(f"  prediction: {letter} — {confidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
