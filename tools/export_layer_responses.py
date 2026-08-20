"""Render representative CNN feature maps for the static project website."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from asl_recognition.data import build_transforms
from asl_recognition.model import load_checkpoint


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/asl_alphabet_cnn_robust_seed42.pt")
    )
    parser.add_argument("--image", type=Path, default=Path("docs/demo/sample_external_a.jpg"))
    parser.add_argument(
        "--output", type=Path, default=Path("site/assets/model-views/layer-response-montage.png")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model, metadata = load_checkpoint(args.checkpoint)
    model.eval()
    image_size = int(metadata["image_size"])
    with Image.open(args.image) as image:
        source = image.convert("RGB").resize((image_size, image_size))
    tensor = build_transforms(image_size, training=False)(source).unsqueeze(0)
    activations: list[torch.Tensor] = []
    handles = [
        block.register_forward_hook(lambda _, __, output: activations.append(output.detach()))
        for block in model.features
    ]
    with torch.inference_mode():
        model(tensor)
    for handle in handles:
        handle.remove()

    figure, axes = plt.subplots(4, 8, figsize=(16, 8), facecolor="#09130f")
    titles = (
        "Input · 64 × 64",
        "Block 1 · 24 channels",
        "Block 2 · 48 channels",
        "Block 3 · 96 channels",
    )
    axes[0, 0].imshow(source)
    for axis in axes[0, 1:]:
        axis.axis("off")
    for row, activation in enumerate(activations, start=1):
        channels = activation[0]
        indices = torch.linspace(0, channels.shape[0] - 1, 8).round().to(torch.int64).tolist()
        for column, index in enumerate(indices):
            view = channels[index].cpu().numpy()
            axes[row, column].imshow(view, cmap="viridis", interpolation="nearest")
            axes[row, column].set_title(f"ch {index + 1}", color="#a7b6a8", fontsize=8)
    for row, title in enumerate(titles):
        axes[row, 0].set_ylabel(
            title, color="#d9ed53", fontsize=10, fontweight="bold", rotation=90, labelpad=30
        )
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#30453a")
    figure.tight_layout(pad=1.4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == "__main__":
    main()
