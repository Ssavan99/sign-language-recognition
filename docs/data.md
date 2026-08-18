# Data layout and provenance

Dataset files are downloaded or supplied locally and are never committed to Git.

## Canonical layout

```text
data/
  raw/
    asl-alphabet/
      asl_alphabet_train/
        A/
        ...
        Z/
    external-asl-alphabet-test/
      A/
      ...
      Z/
  manifests/
    train.csv
    validation.csv
    test.csv
  cache/
```

The planned maintained CLI will accept an explicit data root and validate discovered class directories before it prepares manifests or trains a model. The historical `Datasets/` directory remains ignored and may be used as a local migration source, but it is not the planned maintained layout.

## Sources

- Primary source: Kaggle `grassknoted/asl-alphabet`, licensed GPL-2.0 by the dataset publisher. It contains 87,000 training images across 29 directories; the maintained project uses the A-Z directories for continuity with the original experiments.
- Optional external-domain check: Kaggle `danrasband/asl-alphabet-test`, published under CC0. It contains 870 images with varied backgrounds. Only aligned A-Z classes are evaluated.

The official `kagglehub` client can download ordinary public datasets without authentication. The planned CLI will stop with a clear message and accept a user-supplied local directory if a source later requires consent or authentication; it will never prompt for an account or token.

## Evaluation boundary

The primary source does not provide signer or capture-session identifiers. Deterministic image-level manifests will improve repeatability but cannot establish signer-independent or session-independent generalization. Exact and perceptual duplicate checks are therefore required across prepared splits, and results must be labeled as same-corpus image holdouts.

The optional external dataset provides a separate-source stress test, not proof of production readiness or continuous sign-language understanding.
