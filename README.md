# ASL Alphabet Image Classification

An isolated-image computer-vision project for classifying American Sign Language alphabet gestures.

The repository now includes a reproducible data, training, evaluation, and command-line inference workflow plus a released compact model and optional local demo. Historical notebooks and their recorded outputs remain available under [`archive/`](archive/README.md), but they are not the supported execution path and should not be treated as independent-domain validation.

## Current scope

- A-Z isolated alphabet images
- Historical landmark, custom CNN, transfer-learning, and feature-extraction experiments
- Same-corpus historical results with documented limitations
- A maintained deterministic pipeline with a no-download smoke workflow
- A released 164,546-parameter model with full-data internal and external evaluation
- An optional local upload/webcam demo with visible reliability limitations

The project does not perform continuous signing, sentence translation, or validated real-world accessibility deployment. Motion-dependent letters such as J and Z are a known limitation of still-image classification.

The model reaches 99.82% accuracy on the same-corpus source test partition but only 17.56% on a separate external capture source. The gap is documented as a domain-generalization failure, not hidden behind the internal score. See the [current result report](docs/results/current/README.md), [model card](models/README.md), and [local demo guide](docs/demo/README.md) for the current evaluation and interface contract.

See [`PLAN.md`](PLAN.md) for the active implementation plan, [`SYNC.md`](SYNC.md) for preservation and audit evidence, [`docs/data.md`](docs/data.md) for the canonical data contract, and [`docs/development.md`](docs/development.md) for the verified environment and smoke command. A complete results and usage guide will replace this transition README after repository verification.
