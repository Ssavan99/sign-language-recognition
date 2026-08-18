# Reproducible compact-CNN results

These results were produced by the maintained package on 2026-08-18. They are
reported separately because the internal and external evaluations answer very
different questions.

## Summary

| Evaluation | Images | Accuracy | Macro F1 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Internal source test | 15,600 | 99.82% | 99.82% | Same-corpus image holdout; optimistic |
| External source test | 780 | 17.56% | 16.82% | Separate capture source; poor domain transfer |

The external result is the best evidence of practical behavior. The 82.26
percentage-point accuracy gap shows that the model is strongly dependent on the
primary dataset's capture conditions.

## Training contract

- Model: 164,546-parameter compact CNN.
- Input size: 64 x 64 RGB.
- Seed: 42.
- Optimizer: AdamW, learning rate 0.001, weight decay 0.0001.
- Batch size: 64.
- Epochs: 12; best epoch 12.
- Training samples: 51,376.
- Validation samples: 11,024.
- Best validation loss: 0.0067888907.
- Best validation accuracy: 99.85%.
- Runtime: 7,236.95 seconds (2 h 0 min 37 s), CPU.
- Environment: Python 3.10.11, PyTorch 2.7.1+cu118, NumPy 2.2.6,
  Windows 10 build 26200.

The source provided explicit train and test directories. The source test
partition was kept untouched. The source train partition was deterministically
split into train and validation groups, producing overall proportions of 65.87%
train, 14.13% validation, and 20.00% test.

Each manifest row records a relative path, label, split, SHA-256, and 64-bit
dHash. Model runs re-verified every source file against its SHA-256 before use.
The prepared splits contained zero exact cross-split duplicates. A dHash scan at
Hamming distance <= 5 produced 1,754,615 cross-split review candidates; this is
a coarse perceptual signal, not proof that every pair is a duplicate. The
source does not include signer or capture-session identifiers, so signer- and
session-independent grouping cannot be established.

Manifest hashes:

- Train:
  `f1e741d4c32ac356397af9c582b045362bcb687f6c3b0f1968d57a5bf243f23b`
- Validation:
  `dbc71fee51e0a1276901f8d7b2313fb2c46fdc6bfd1d4b9f99f23e7d3374fd56`
- Internal test:
  `4e6b56bc232f621df30cd3fceb43defc634f15bfe50a69ed5409128e0c744957`
- External test:
  `c1b79ff019139c695015a39c51cab817bc0c1f42878ea3cb4b51db0b4caffbc9`
- Duplicate report:
  `5fec8bb0ecb8e59720f6817993124c28630d46dfac197187e1147db2f1fcdc92`

## Internal source test

The primary source test partition contains 600 images per class. The model
classified 15,572 of 15,600 images correctly. Class E had the lowest recall at
98.00%; every other class had at least 99.17% recall.

Median single-image CPU model latency was 4.38 ms and p95 was 6.68 ms over 200
timed forward passes. This timing excludes image decoding, preprocessing, and
application overhead.

![Internal same-corpus confusion matrix](internal_confusion_matrix.png)

This score must not be interpreted as signer-independent or real-world
accuracy. Images come from the same controlled corpus as training data, and the
perceptual-candidate count indicates substantial visual similarity across the
image-level split.

## External source test

The external source contains 30 images per A-Z class from a separate capture
set. The model classified 137 of 780 images correctly. Macro F1 was 16.82%.
Class recall ranged from 0% for A, E, F, T, U, and Z to 73.33% for P. This is a
failure to generalize reliably, not a deployment-ready result.

![External-domain confusion matrix](external_confusion_matrix.png)

The external data was downloaded anonymously from the public
`danrasband/asl-alphabet-test` dataset and was not used for training or model
selection. The A-Z subset is licensed CC0 according to the source listing.

## Artifacts

- Model checkpoint:
  [`models/asl_alphabet_cnn_seed42.pt`](../../../models/asl_alphabet_cnn_seed42.pt)
- Per-epoch history: [`training_history.json`](training_history.json)
- Checkpoint SHA-256:
  `8b3d071082615de4a21a6303c8e9ca5496a747c6f5707064f26b3d6a9f6c40a7`
- Training-history SHA-256:
  `ee21a98d5140fce8b0b863bdc36301ac63692dbcf841285807ecc15fb7cbf843`
- Internal matrix SHA-256:
  `a13cde9f6d0f854af4555ef6f57fdd82b28e40e3344edca123a1de36c1437bc2`
- External matrix SHA-256:
  `dc4704828df86fdcc613522e4146ee2dba0b6d367f4bb2ed0098e281ab2727f1`

The full datasets, generated manifests, raw evaluation JSON, and local run
directories remain ignored because they are either large, machine-specific, or
reproducible from the documented commands.
