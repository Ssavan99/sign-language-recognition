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

## No-download smoke workflow

```powershell
asl-recognition smoke --output-dir artifacts/smoke --device cpu
```

This command generates a deterministic synthetic A-Z fixture and exercises manifest preparation, one-epoch training, checkpoint export, evaluation, and prediction. It validates integration only; its accuracy is not a project result.

## Reproducibility controls

- Python, NumPy, and PyTorch seeds are set from the run configuration.
- CuDNN benchmarking is disabled and deterministic algorithms are requested.
- Exact-content duplicates are grouped into one split.
- Train, validation, and test CSV files contain content hashes and perceptual hashes.
- Training augmentation is constructed separately from deterministic validation, test, and inference preprocessing.
- Checkpoints contain architecture, A-Z class order, preprocessing, seed, manifest hashes, best epoch, validation loss, and parameter count.
- Evaluation records checkpoint and manifest hashes, scope, per-class metrics, confusion matrix, model size, and single-image latency.

The source dataset has no signer/session identifiers. These controls make image-level experiments repeatable; they do not establish signer-independent or session-independent generalization.
