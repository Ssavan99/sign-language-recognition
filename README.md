# ASL Alphabet Image Classifier

A reproducible computer-vision pipeline for classifying isolated A-Z American Sign Language
alphabet images, with checksummed data manifests, a compact PyTorch model, independent-domain
evaluation, command-line inference, and an optional local demo.

![Local classifier showing a confident domain-shift failure](docs/demo/demo-screenshot.png)

The screenshot shows the released model predicting **N** at 75.9% confidence for an external
sample whose true label is **A**. This failure is intentional evidence: the model reaches 99.82%
accuracy on its same-corpus source test partition but only 17.56% on a separate capture source.
The project is therefore an isolated-image experiment, not a real-world signing or accessibility
system.

## What the project includes

- Deterministic discovery of A-Z image folders, including pooled and source-provided train/test
  layouts.
- Checksummed CSV manifests with exact-duplicate grouping, cross-split refusal, and bounded dHash
  review reporting.
- A 164,546-parameter compact CNN with separate training and deterministic evaluation transforms.
- Self-describing checkpoints containing architecture, class order, preprocessing, seed, manifest
  hashes, and selection metadata.
- Bounded-memory evaluation with per-class metrics, confusion matrices, artifact hashes, and
  single-image latency.
- Shared inference for the CLI and optional local Gradio upload/webcam interface.
- A generated-data smoke workflow and 26-test suite that require no dataset download.
- Preserved historical notebooks, figures, and report under [`archive/`](archive/README.md).

## Results

| Experiment | Evaluation scope | Images | Accuracy | Macro F1 |
| --- | --- | ---: | ---: | ---: |
| Historical reported CNN | Same-corpus image holdout; unseeded historical workflow | Not retained in a manifest | 94.27% | Not reported |
| Maintained compact CNN | Untouched test partition from the training corpus | 15,600 | 99.82% | 99.82% |
| Maintained compact CNN | Separate external capture source | 780 | 17.56% | 16.82% |

The maintained model classified 15,572 of 15,600 internal test images correctly, but only 137 of
780 external images. The 82.26 percentage-point gap is evidence of severe capture-domain bias.
Internal performance must not be interpreted as signer-independent or real-world accuracy.

The released checkpoint uses 64 x 64 RGB inputs, three convolutional blocks, AdamW, seed 42, and
12 CPU training epochs. Its SHA-256 is
`8b3d071082615de4a21a6303c8e9ca5496a747c6f5707064f26b3d6a9f6c40a7`.
See the [current result report](docs/results/current/README.md), [model card](models/README.md), and
[historical archive](archive/README.md) for the full evidence and comparability boundaries. The
broken two-class transfer-learning notebook's near-perfect metric is excluded from supported
26-class results.

## How it works

1. A data root is discovered through a bounded wrapper hierarchy and must contain every A-Z class.
2. Each image is recorded with its relative path, label, split, SHA-256, and 64-bit dHash.
3. Exact-content groups stay in one generated split; exact duplicates across source-provided
   partitions are rejected rather than silently moving test data.
4. Training uses augmentation, while validation, evaluation, and inference use deterministic
   resizing and ImageNet normalization.
5. Every manifest row is rechecked against the current file bytes before training or evaluation.
6. The best-validation-loss checkpoint drives CLI prediction, evaluation, and the local demo.

## Install

Python 3.10-3.12 is supported; Python 3.10.11 is the fully verified environment. The following
PowerShell commands install the free CPU build and every optional data, demo, and development
dependency:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints.txt -e ".[all]"
```

No account, API token, hosted notebook, paid API, subscription, or cloud service is required.
Platform-specific PyTorch builds may be substituted from the official PyTorch index.

Verify the environment:

```powershell
asl-recognition doctor
asl-recognition --help
ruff check --no-cache src tests
ruff format --check src tests
pytest --cov=asl_recognition --cov-report=term-missing --cov-fail-under=75
```

## Quick smoke run

This command generates a small synthetic A-Z dataset and runs preparation, one training epoch,
checkpoint export, evaluation, and prediction on CPU:

```powershell
asl-recognition smoke --output-dir artifacts/smoke --seed 42 --device cpu
```

The smoke result verifies integration only and is not a reported model score.

## Data

The maintained workflow uses two public Kaggle datasets:

| Purpose | Source | Publisher license | Maintained subset |
| --- | --- | --- | --- |
| Training and internal evaluation | [`grassknoted/asl-alphabet`](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) | GPL-2.0 | A-Z images |
| External-domain evaluation | [`danrasband/asl-alphabet-test`](https://www.kaggle.com/datasets/danrasband/asl-alphabet-test) | CC0 | A-Z images |

The official `kagglehub` client downloaded both public sources anonymously during verification.
Acquire and prepare them from the repository root:

```powershell
asl-recognition download primary --output-root data/raw
asl-recognition prepare --source-root data/raw/primary --output-dir data/manifests --seed 42

asl-recognition download external --output-root data/raw
asl-recognition prepare-external --source-root data/raw/external --output-file data/manifests/external.csv
```

If a source later requires authentication or consent, stop the download and pass an existing local
directory to `--source-root`; do not enter credentials. Dataset files, manifests, and run outputs
remain ignored by Git. See [the data contract](docs/data.md) for accepted layouts and provenance.

## Train and evaluate

Train the maintained 64 x 64 configuration:

```powershell
asl-recognition train `
  --manifest-dir data/manifests `
  --source-root data/raw/primary `
  --output-dir artifacts/training `
  --epochs 12 `
  --batch-size 64 `
  --image-size 64 `
  --seed 42 `
  --device cpu
```

Evaluate the selected checkpoint on the untouched source test partition and the external source:

```powershell
asl-recognition evaluate `
  --checkpoint artifacts/training/best_model.pt `
  --manifest data/manifests/test.csv `
  --source-root data/raw/primary `
  --output-dir artifacts/evaluation/internal `
  --scope "same-corpus source test" `
  --device cpu

asl-recognition evaluate `
  --checkpoint artifacts/training/best_model.pt `
  --manifest data/manifests/external.csv `
  --source-root data/raw/external `
  --output-dir artifacts/evaluation/external `
  --scope "separate external capture source" `
  --device cpu
```

Training the recorded full-data CPU run took about two hours. CUDA is optional; an explicit
unavailable device request fails instead of silently changing the run.

## Predict one image

The repository includes the evaluated compact checkpoint and a CC0 external sample:

```powershell
asl-recognition predict docs/demo/sample_external_a.jpg `
  --checkpoint models/asl_alphabet_cnn_seed42.pt `
  --top-k 3 `
  --device cpu
```

The deterministic sample result is N at approximately 75.9% confidence, despite its true A label.

## Local demo

Launch the optional upload/webcam interface on the loopback address:

```powershell
asl-recognition demo --checkpoint models/asl_alphabet_cnn_seed42.pt --device cpu
```

Open the printed local URL. Public sharing and public prediction APIs are disabled. The built-in
external-domain example reproduces the screenshot and keeps the measured limitation visible even
when model confidence is high. See the [demo guide](docs/demo/README.md).

## Repository structure

```text
src/asl_recognition/       Maintained data, model, training, evaluation, inference, and demo code
tests/                     Generated-data unit and integration tests
models/                    Released compact checkpoint and model card
docs/results/current/      Maintained evaluation report and confusion matrices
docs/demo/                 Local demo guide, CC0 sample, and real screenshot
archive/notebooks/         Preserved exploratory notebooks; not the supported run path
.github/workflows/ci.yml   Free generated-data CI checks
```

## Limitations

- The task is isolated A-Z still-image classification, not continuous signing or sentence
  translation.
- The model is not a hand detector; the demo performs only basic image-quality checks.
- The primary dataset lacks signer and capture-session identifiers, so signer-disjoint and
  session-disjoint evaluation cannot be established.
- Same-corpus images share controlled backgrounds and capture conditions, producing optimistic
  internal results and many coarse perceptual-similarity review candidates.
- J and Z depend on motion in real signing; one frame cannot represent that motion fully.
- Softmax confidence is not calibrated proof of correctness. The external sample demonstrates a
  confident wrong prediction.
- The project has not been validated for real-world deployment or as an accessibility system.

## Licensing

Dataset files remain subject to their publishers' licenses listed above. No project code license is
currently granted because contributor licensing for the group-authored code has not been resolved.
