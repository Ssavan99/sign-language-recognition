# ASL Alphabet Image Classification

An isolated-image computer-vision project for classifying American Sign Language alphabet gestures.

The repository is being organized around a reproducible data, training, evaluation, and local-inference workflow. Historical notebooks and their recorded outputs remain available under [`archive/`](archive/README.md), but they are not the supported execution path and should not be treated as independent-domain validation.

## Current scope

- A-Z isolated alphabet images
- Historical landmark, custom CNN, transfer-learning, and feature-extraction experiments
- Same-corpus historical results with documented limitations
- A maintained deterministic pipeline and local demo currently under implementation on this branch

The project does not perform continuous signing, sentence translation, or validated real-world accessibility deployment. Motion-dependent letters such as J and Z are a known limitation of still-image classification.

See [`PLAN.md`](PLAN.md) for the active implementation plan, [`SYNC.md`](SYNC.md) for preservation and audit evidence, and [`docs/data.md`](docs/data.md) for the planned canonical data contract. A complete installation, usage, results, and limitations guide will replace this transition README after the maintained workflow is verified.
