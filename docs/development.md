# Development and verification environment

The maintained package supports Python 3.10 through 3.12. Python 3.10.11 is the fully verified environment for the recorded local run.

## Create an environment

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Install one official PyTorch build. The CPU build works on any supported Windows machine:

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
```

The local verification machine uses an NVIDIA GTX 1660 Ti and the CUDA 11.8 build:

```powershell
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
```

Install the project and all optional development, anonymous-data, and local-demo dependencies using the verified direct-version constraints:

```powershell
python -m pip install -c constraints.txt -e ".[all]"
```

No account, API token, hosted notebook, or paid service is needed.

## Verify the environment

```powershell
asl-recognition doctor
asl-recognition --help
ruff check --no-cache src tests
ruff format --check src tests
pytest --cov=asl_recognition --cov-report=term-missing --cov-fail-under=75
```

The test suite uses generated images and does not download either full dataset. The doctor command reports exact package versions and whether CUDA is available. Training accepts `--device auto`, `--device cpu`, or `--device cuda`; an explicit unavailable CUDA request fails rather than silently switching devices.

## Browser model release

The GitHub Pages site runs a dependency-free JavaScript implementation over an exact float32
export of the released checkpoint. Regenerate the browser assets after intentionally replacing the
checkpoint, then verify parity with Node.js 16 or later:

```powershell
python tools/export_browser_model.py
node tools/check_browser_model.mjs
```

The Python test suite checks that the published browser weights and manifest match the checkpoint
byte-for-byte. The website uses the user's local camera or uploaded image only in the browser.

`tools/export_layer_responses.py` renders the published feature-map montage from the released
checkpoint and the CC0 external A sample. Regenerate it only when either input changes:

```powershell
python tools/export_layer_responses.py
```

## No-download smoke workflow

```powershell
asl-recognition smoke --output-dir artifacts/smoke --device cpu
```

This command generates a deterministic synthetic A-Z fixture and exercises manifest preparation, one-epoch training, checkpoint export, evaluation, and prediction. It validates integration only; its accuracy is not a project result.

## Memory-guarded training runs

Full-split training is a multi-hour CPU job. `train` runs a preflight before it
does any work and refuses to start when the host has less available memory than
the floor:

```powershell
asl-recognition train --source-root Datasets --output-dir artifacts/training/run --device cpu
```

- `--minimum-available-gib` moves the floor. The default is measured, not
  guessed: a full CPU run peaks well under it.
- `--allow-low-memory` starts anyway and records that the floor was overridden.
- `--limit-per-class N` trains on N evenly spaced images per class. Screening
  runs use this so a candidate comparison takes minutes instead of hours.

Every run records per-epoch resident and peak memory in `history.json`, plus the
preflight result. A run that was started under an overridden floor says so in its
own metadata.

Checksum verification is never skipped. When `--limit-per-class` is set, the
cross-split duplicate check still uses the complete manifests, and file hashing
covers exactly the rows the run consumes.

## Augmentation profiles and the stress benchmark

```powershell
asl-recognition train --source-root Datasets --output-dir artifacts/screening/robust `
  --augmentation-profile robust --select-on stress --limit-per-class 200 --device cpu
```

- `--augmentation-profile {baseline,robust,trivialaugment}` selects a
  training-augmentation recipe. It does not change inference preprocessing:
  validation, test, external, and released-model preprocessing are identical
  under every profile.
- `--select-on {validation,stress}` chooses which metric picks the best epoch.
  Both are always computed and recorded, whichever is selected on.

The stress benchmark is a frozen corruption family applied only to source
validation images. Its design, its limits, and the pre-registered decision rule
it feeds are documented in [results/robustness.md](results/robustness.md).
Summarise a set of screening runs with:

```powershell
python tools/summarize_screening.py artifacts/screening
```

## Regenerating the demo screenshot

`docs/demo/demo-screenshot.png` is evidence, not decoration: it must show the
currently released model. Regenerate it whenever the released checkpoint changes.

```powershell
python -m pip install playwright
python -m playwright install chromium
asl-recognition demo --device cpu --port 7861
python tools/capture_demo_screenshot.py --port 7861
```

Playwright is a development tool and is deliberately not a project dependency;
neither the test suite nor CI needs it. The script drives the real interface and
keeps whatever the model predicts, including a wrong answer, because a curated
success case would misrepresent external-domain behaviour.

## Adding a supplementary training corpus

A second labelled A-Z corpus can be folded into training without touching the
held-out sets:

```powershell
python tools/prepare_supplementary_source.py `
  --source <downloaded-corpus-root> --output data/raw/supplement
asl-recognition prepare --source-root data/raw/supplement --output-dir data/manifests-supplement
asl-recognition train --source-root Datasets --output-dir artifacts/training/run `
  --augmentation-profile robust_noflip --select-on stress `
  --extra-manifest-dir data/manifests-supplement --extra-source-root data/raw/supplement `
  --extra-repeat 7 --device cpu
```

- The normaliser skips any image whose bytes already appear in an existing
  manifest, matches class directories case-insensitively, and never walks a
  nested duplicate copy of the corpus.
- Training refuses to start if the supplement shares an image with the primary
  train, validation, or untouched test split.
- The supplement contributes training data only. It never joins the validation
  split or the stress benchmark, so the selection signal is unchanged.
- `--extra-repeat` weights a small second domain against a much larger primary
  one. A held-out slice of the supplement is scored each epoch as a
  second-domain diagnostic; it is reported, never selected on.

This path was used for the negative result recorded in
[results/robustness.md](results/robustness.md): a second *studio* corpus was
learned almost perfectly without improving external transfer at all.

## Reproducibility controls

- Python, NumPy, and PyTorch seeds are set from the run configuration.
- CuDNN benchmarking is disabled and deterministic algorithms are requested.
- Exact-content duplicates are grouped into one split.
- Train, validation, and test CSV files contain content hashes and perceptual hashes.
- Training augmentation is constructed separately from deterministic validation, test, and inference preprocessing, and the selected profile is recorded in the run metadata and the checkpoint.
- The stress benchmark is deterministic: corruptions are assigned by row position and seeded from each row's recorded SHA-256, so scores do not depend on batch order or worker count.
- Checkpoints contain architecture, A-Z class order, preprocessing, seed, manifest hashes, best epoch, validation loss, and parameter count.
- Evaluation records checkpoint and manifest hashes, scope, per-class metrics, confusion matrix, model size, and single-image latency.

The source dataset has no signer/session identifiers. These controls make image-level experiments repeatable; they do not establish signer-independent or session-independent generalization.
