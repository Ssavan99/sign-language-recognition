# ASL Alphabet Image Classification

An isolated-image computer-vision project for classifying American Sign Language alphabet gestures.

The repository now includes a reproducible data, training, evaluation, and command-line inference workflow. Historical notebooks and their recorded outputs remain available under [`archive/`](archive/README.md), but they are not the supported execution path and should not be treated as independent-domain validation. Full-data validation and the local demo are still being completed on this branch.

## Current scope

- A-Z isolated alphabet images
- Historical landmark, custom CNN, transfer-learning, and feature-extraction experiments
- Same-corpus historical results with documented limitations
- A maintained deterministic pipeline with a no-download smoke workflow
- Full-data validation and a local demo currently under implementation on this branch

The project does not perform continuous signing, sentence translation, or validated real-world accessibility deployment. Motion-dependent letters such as J and Z are a known limitation of still-image classification.

See [`PLAN.md`](PLAN.md) for the active implementation plan, [`SYNC.md`](SYNC.md) for preservation and audit evidence, [`docs/data.md`](docs/data.md) for the canonical data contract, and [`docs/development.md`](docs/development.md) for the verified environment and smoke command. A complete results and usage guide will replace this transition README after full-data validation and demo verification.
