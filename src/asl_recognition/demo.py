"""Optional local Gradio interface over the shared inference module."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from .inference import Predictor

DOMAIN_NOTICE = (
    "This model is a controlled-source experiment, not a hand detector or "
    "accessibility system. External-domain accuracy is 31.67%, and a high "
    "softmax score can still be wrong."
)

DEMO_CSS = """
:root {
  --asl-background: #fff7ed;
  --asl-surface: #ffffff;
  --asl-foreground: #0f172a;
  --asl-muted: #475569;
  --asl-primary: #c2410c;
  --asl-primary-hover: #9a3412;
  --asl-border: #fed7aa;
  --asl-warning-bg: #fffbeb;
  --asl-warning-border: #b45309;
}

.gradio-container {
  background: var(--asl-background) !important;
  color: var(--asl-foreground) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif !important;
  margin-inline: auto !important;
  max-width: 1120px !important;
  padding: 20px 24px 36px !important;
}

.asl-hero {
  margin: 0 auto 16px;
  max-width: 760px;
  text-align: center;
}

.asl-hero h1 {
  color: var(--asl-foreground);
  font-size: clamp(2rem, 5vw, 3rem);
  letter-spacing: -0.035em;
  line-height: 1.05;
  margin-bottom: 12px;
}

.asl-hero p {
  color: var(--asl-muted);
  font-size: 1.05rem;
  line-height: 1.45;
  margin-inline: auto;
  max-width: 68ch;
}

.asl-warning {
  background: var(--asl-warning-bg);
  border: 2px solid var(--asl-warning-border);
  border-radius: 12px;
  color: #78350f;
  line-height: 1.55;
  margin: 0 0 16px;
  padding: 12px 16px;
}

.asl-warning,
.asl-warning * {
  color: #78350f !important;
}

.asl-card {
  background: var(--asl-surface);
  border: 1px solid var(--asl-border);
  border-radius: 16px;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.08);
  padding: 16px;
}

.asl-result {
  min-height: 104px;
}

.asl-result h3 {
  color: var(--asl-foreground);
  font-size: 1.5rem;
  margin-bottom: 8px;
}

.asl-result,
.asl-result *,
.asl-detail,
.asl-detail * {
  color: var(--asl-muted);
  line-height: 1.55;
}

.asl-result h3 {
  color: var(--asl-foreground) !important;
}

.gallery .caption-label,
.gallery .caption-label * {
  color: var(--asl-foreground) !important;
}

.primary {
  background: var(--asl-primary) !important;
  border-color: var(--asl-primary) !important;
  min-height: 48px !important;
}

.primary:hover {
  background: var(--asl-primary-hover) !important;
}

button:focus-visible,
input:focus-visible,
textarea:focus-visible {
  outline: 3px solid #2563eb !important;
  outline-offset: 2px !important;
}

@media (max-width: 640px) {
  .gradio-container {
    padding: 20px 16px 32px !important;
  }

  .asl-card {
    padding: 16px;
  }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    scroll-behavior: auto !important;
    transition-duration: 0.01ms !important;
  }
}
"""


def _validate_demo_image(image: Image.Image | None) -> Image.Image:
    if image is None:
        raise ValueError("Add an image from a file or webcam, then try again.")
    if not isinstance(image, Image.Image):
        raise ValueError(
            "The selected input is not a readable image. Choose a JPG, PNG, or WebP file."
        )
    try:
        prepared = image.copy().convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError(
            "The selected input is not a readable image. Choose another file."
        ) from error
    if min(prepared.size) < 48:
        raise ValueError("The image is too small. Use an image at least 48 pixels on each side.")
    if prepared.width * prepared.height > 40_000_000:
        raise ValueError("The image is too large. Resize it below 40 megapixels and try again.")

    grayscale = prepared.convert("L")
    statistics = ImageStat.Stat(grayscale)
    minimum, maximum = grayscale.getextrema()
    deviation = float(statistics.stddev[0])
    if maximum - minimum < 12 or deviation < 5.0:
        raise ValueError(
            "No usable hand image was found by the basic quality checks. "
            "Use a clear, well-lit image with the hand visible against the background."
        )
    return prepared


def format_demo_prediction(
    predictor: Predictor,
    image: Image.Image | None,
) -> tuple[str, dict[str, float], str]:
    """Validate and classify one demo image without leaking exceptions to the UI."""

    try:
        prepared = _validate_demo_image(image)
        result = predictor.predict(prepared, top_k=3)
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
        status = f"### Unable to classify\n\n{error}"
        detail = f"**Model boundary:** {DOMAIN_NOTICE}"
        return status, {}, detail

    ranked = result.get("top_k")
    if not isinstance(ranked, list):
        raise RuntimeError("Prediction output did not include ranked classes")
    try:
        scores = {
            str(item["class"]): float(item["probability"])
            for item in ranked
            if isinstance(item, Mapping) and "class" in item and "probability" in item
        }
        predicted_class = str(result["predicted_class"])
        confidence = float(result["confidence"])
        low_confidence = bool(result["low_confidence"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("Prediction output was malformed") from error
    if not scores:
        raise RuntimeError("Prediction output did not include usable ranked classes")
    if low_confidence:
        status = (
            f"### Low-confidence result: {predicted_class}\n\n"
            f"Model confidence: **{confidence:.1%}**. Treat this as uncertain and try a "
            "clearer, centered image."
        )
    else:
        status = (
            f"### Predicted letter: {predicted_class}\n\n"
            f"Model confidence: **{confidence:.1%}**. Confidence is not a guarantee of "
            "correctness outside the training domain."
        )
    detail = (
        f"**Always apply this limitation:** {DOMAIN_NOTICE}\n\n"
        "J and Z involve motion in real signing; a still image cannot represent that motion."
    )
    return status, scores, detail


def create_demo(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
    confidence_threshold: float = 0.60,
):
    """Build the local interface and eagerly validate its checkpoint."""

    try:
        import gradio as gr
    except ImportError as error:  # pragma: no cover - optional dependency boundary
        raise RuntimeError(
            "The local demo requires Gradio. Install the optional demo dependencies with "
            "`python -m pip install -e '.[demo]'`."
        ) from error

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Demo checkpoint not found: {checkpoint}. Use --checkpoint with a released model path."
        )
    predictor = Predictor(
        checkpoint,
        device=device,
        confidence_threshold=confidence_threshold,
    )
    example_path = Path.cwd() / "docs" / "demo" / "sample_external_a.jpg"

    with gr.Blocks(
        title="ASL Alphabet Classifier",
        analytics_enabled=False,
        fill_width=True,
    ) as demo:
        gr.Markdown(
            """
            # ASL Alphabet Classifier

            Explore a compact A-Z image classifier with an uploaded photo or webcam frame.
            The model runs locally; this page does not upload images to a hosted project service.
            """,
            elem_classes="asl-hero",
        )
        gr.Markdown(f"**Important limitation:** {DOMAIN_NOTICE}", elem_classes="asl-warning")

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_classes="asl-card"):
                image_input = gr.Image(
                    label="Hand image",
                    sources=["upload", "webcam"],
                    type="pil",
                    image_mode="RGB",
                    height=280,
                    buttons=["fullscreen"],
                    placeholder="Upload a clear, well-lit single-hand image or use the webcam.",
                )
                with gr.Row():
                    classify = gr.Button("Classify image", variant="primary", size="lg")
                    clear = gr.ClearButton(value="Clear", variant="secondary", size="lg")

            with gr.Column(scale=5, elem_classes="asl-card"):
                status = gr.Markdown(
                    "### Ready\n\nAdd an image, then choose **Classify image**.",
                    elem_classes="asl-result",
                )
                scores = gr.Label(label="Top predictions", num_top_classes=3)
                detail = gr.Markdown(
                    f"**Model boundary:** {DOMAIN_NOTICE}",
                    elem_classes="asl-detail",
                )

        if example_path.is_file():
            gr.Examples(
                examples=[[str(example_path)]],
                example_labels=["External-domain sample — true label A"],
                inputs=image_input,
                outputs=[status, scores, detail],
                fn=lambda image: format_demo_prediction(predictor, image),
                cache_examples=False,
                examples_per_page=1,
                label="Reproduce the domain-shift example",
                run_on_click=True,
                api_visibility="private",
            )

        classify.click(
            fn=lambda image: format_demo_prediction(predictor, image),
            inputs=image_input,
            outputs=[status, scores, detail],
            api_name=False,
            api_visibility="private",
            show_progress="full",
            concurrency_limit=1,
        )
        clear.add([image_input, status, scores, detail])
        demo.queue(default_concurrency_limit=1, max_size=8, api_open=False)
    return demo


def launch_demo(
    checkpoint_path: Path,
    *,
    device: str = "cpu",
    confidence_threshold: float = 0.60,
    server_name: str = "127.0.0.1",
    port: int = 7860,
    inbrowser: bool = False,
) -> dict[str, Any]:
    """Launch a local-only demo without public sharing or account requirements."""

    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if server_name not in {"127.0.0.1", "localhost"}:
        raise ValueError("server-name must be 127.0.0.1 or localhost for the local-only demo")
    demo = create_demo(
        checkpoint_path,
        device=device,
        confidence_threshold=confidence_threshold,
    )
    import gradio as gr

    _, local_url, _ = demo.launch(
        server_name=server_name,
        server_port=port,
        inbrowser=inbrowser,
        share=False,
        show_error=False,
        quiet=False,
        footer_links=[],
        enable_monitoring=False,
        strict_cors=True,
        max_file_size="20mb",
        css=DEMO_CSS,
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
    )
    return {"local_url": local_url, "share": False, "checkpoint": str(Path(checkpoint_path))}


__all__ = [
    "DEMO_CSS",
    "DOMAIN_NOTICE",
    "create_demo",
    "format_demo_prediction",
    "launch_demo",
]
