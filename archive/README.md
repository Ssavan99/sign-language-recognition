# Historical experiments

This directory preserves the original exploratory notebooks and their recorded outputs. They are retained for historical context and are not the supported execution path.

The maintained pipeline lives under `src/` and exposes deterministic command-line workflows for data preparation, training, evaluation, and prediction. Historical metrics are not directly comparable with maintained-pipeline metrics unless their data split and evaluation scope match.

## Notebook index

| Notebook | Historical purpose | Status |
| --- | --- | --- |
| `notebooks/dataset_download_original.ipynb` | Download and reorganize the original image dataset | Machine-specific Kaggle CLI path; split is unseeded and not actually shuffled |
| `notebooks/data_exploration_original.ipynb` | Dataset counts, dimensions, samples, and duplicate exploration | Recorded analysis only; checks exact duplicates within training data, not across splits |
| `notebooks/landmark_baselines_original.ipynb` | MediaPipe landmarks with Random Forest and SVM classifiers | Records 83.85% and 62.69% same-source holdout accuracy; one stored inference cell fails on a path error |
| `notebooks/cnn_no_augmentation_original.ipynb` | Custom CNN without training augmentation | Records 82.35% same-corpus holdout accuracy |
| `notebooks/cnn_augmented_original.ipynb` | Custom CNN with augmentation | Records 92.91% same-corpus holdout accuracy; validation preprocessing is not cleanly separated from training augmentation |
| `notebooks/transfer_learning_broken_original.ipynb` | VGG16, InceptionV3, and MobileNetV2 exploration | Unsupported: discovers only two directories as classes and contains a stored `NameError`; its near-perfect metric is not a valid 26-class result |
| `notebooks/feature_extraction_original.ipynb` | MediaPipe-derived hand geometry features | Exploratory, memory-heavy batch workflow with no maintained artifact contract |
| `notebooks/cnn_128_colab_incomplete.ipynb` | Larger 128x128 CNN draft | Unsupported and incomplete; depends on Google Drive/Colab, has no completed result, and was preserved from previously untracked work |

## Preserved report and figures

- The original report is stored at `docs/report/asl-sign-language-analysis.pdf`.
- Historical result figures are stored under `docs/results/historical/`.

These artifacts preserve the original analysis and contributor context. Current project scope and revival status are documented in the root README; maintained full-data evaluation outputs are reported separately under [`docs/results/current/`](../docs/results/current/README.md).
