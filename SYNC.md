# Repository Sync and Preservation Audit

Audit date: 2026-08-18

## Remote isolation

- The former `origin` was `git@github.com:Ssavan99/SignLanguageRecognition.git`.
- That remote was removed before the project audit.
- `git remote -v` was empty immediately after removal.
- A new standalone public repository was created at `https://github.com/Ssavan99/sign-language-recognition`.
- The only configured remote is now:
  - fetch: `https://github.com/Ssavan99/sign-language-recognition.git`
  - push: `https://github.com/Ssavan99/sign-language-recognition.git`
- GitHub reports the new repository as `isFork: false` with `parent: null`.
- No active Git configuration points to the former shared repository. The old URL is present only in inert local reflog history.
- No writes or settings changes have been made to the former repository.

## Original repository visibility snapshot

- `Ssavan99/SignLanguageRecognition` was public when checked on 2026-08-18.
- It reported zero forks at that check.
- Because it was already public, the names and email addresses in preserved commits are not a new disclosure.
- Any visibility change is deferred until the new repository is complete and requires explicit approval.

## Current branch and new origin

- Checked-out branch: `revive/portfolio-ready-pipeline`
- `main`: `06816f5bc03cc7dafa116525e1951280219f4747`
- `origin/main`: `06816f5bc03cc7dafa116525e1951280219f4747`
- Preserved feature-branch base: `5db182147520c73209019b668a61344430a6dd06`
- `origin/revive/portfolio-ready-pipeline`: `5db182147520c73209019b668a61344430a6dd06`
- The local feature branch contains new audit and curation commits that have not yet been pushed; the remote branch remains the verified preservation anchor.
- No tags or stashes exist.

## Locally preserved history missing from `main`

Two inert remote-tracking refs remain locally; no corresponding remote is configured:

- `refs/remotes/ghcheck/main` -> `5db182147520c73209019b668a61344430a6dd06`
- `refs/remotes/ghcheck/objective-1` -> `e186f49d0cee91fb03d453880900a9b979b6accd`

`ghcheck/main` is a direct descendant of `main` and is nine commits ahead. `main` has no commits absent from it, so the full lineage can be preserved by branching from or fast-forwarding to `ghcheck/main`; cherry-picking, squashing, rebasing, or force-pushing is neither necessary nor permitted.

The nine commits are:

1. `367894e` - objective 1
2. `e186f49` - Add SVM classification
3. `e7797e8` - Add files via upload
4. `0227db2` - Merge pull request #1 from Ssavan99/objective-1
5. `7ed39d1` - final objective 2 and 3
6. `b90c5b8` - renamed objective 3
7. `36d4d39` - renamed objective 4
8. `437a7bd` - readme
9. `5db1821` - readme

`ghcheck/objective-1` is already merged into `ghcheck/main`. The later lineage adds the report PDF, landmark baselines, augmented and non-augmented CNN experiments, transfer-learning work, feature extraction, six result images, and README changes. The net difference from current `main` is 19 files, 3,799 insertions, and 681 deletions.

The commits are now anchored unchanged on `revive/portfolio-ready-pipeline` at the new origin. The inert refs remain locally available during execution as an additional preservation check.

## Uncommitted and ignored work

- Preserved from formerly untracked work: `archive/notebooks/cnn_128_colab_incomplete.ipynb` (original blob hash `f6b251675be6288756d7cb2617461d01d2aea049`).
  - It is not identical to the later committed augmented-CNN notebook.
  - It contains an incomplete 128x128 Colab/Google Drive experiment with no completed training result.
  - It is now tracked and explicitly labeled unsupported in `archive/README.md`.
- Ignored: `Datasets/` with 156,060 files totaling about 2.03 GB.
  - This includes 156,000 JPEGs and a roughly 1.02 GB archive.
  - It must remain untracked.
  - The extracted directory currently has an extra nested level that does not match notebook paths.
- No tracked working-tree modifications existed before `SYNC.md` and `PLAN.md` were created.

## Git object and content health

- `git fsck --full --lost-found --no-reflogs` reported no unreachable or corrupt objects.
- There are no stashes, tags, or lost commits beyond the explicitly preserved refs above.
- No API keys, passwords, tokens, credential files, or private-key markers were found in reachable notebook/text history or the untracked notebook.
- Largest reachable blobs are acceptable for ordinary Git, but notebook outputs are bulky:
  - `Objective2/model_no_augmentation.ipynb`: about 4.86 MB
  - `ASL Sign Language Analysis.pdf`: about 4.41 MB
  - data exploration notebook: about 1.88 MB
  - largest result PNG: about 0.79 MB
- The two tracked `.DS_Store` files were removed from the final tree in commit `8bdf05e` and remain recoverable in history. Root-anchored ignore coverage prevents their return.

## Preserved contributor metadata

Reachable history includes commits under these identities:

- Savan Patel `<ssavan99@Savans-MacBook-Pro.local>`
- Savan Patel `<ssavan99@savans-mbp.unl.edu>`
- Savan Patel `<58150774+Ssavan99@users.noreply.github.com>`
- Michael Payne `<10254938+michaelpayne02@users.noreply.github.com>`
- maisayed1 `<msayed2@huskers.unl.edu>`

This history will be preserved without rewriting authors, commit emails, merge topology, or timestamps.

## Historical reproducibility and validity findings

- The preserved historical README contained only a title before the revival work.
- The historical project had no dependency manifest, package, CLI, test suite, CI workflow, saved model, class map, or fresh-machine run path.
- Dataset acquisition hard-codes a local Kaggle CLI path and the incomplete notebook mounts Google Drive.
- The data split is described as shuffled but is not shuffled, seeded, or recorded in a manifest.
- Validation is mistakenly created from the augmented training generator.
- The recorded 94.27% CNN test accuracy is a same-corpus image-level holdout, not an independent signer/session/domain test.
- The later augmented CNN records 92.91% test accuracy and the non-augmented CNN 82.35%, also on same-corpus holdouts.
- The landmark baselines record 83.85% for Random Forest and 62.69% for SVM.
- The transfer-learning notebook reports only two discovered classes, later raises `NameError`, and its approximately 99.97% validation result is invalid for a 26-class ASL claim.
- The 128x128 untracked experiment has about 67.5 million parameters, incomplete training, and an evaluation path that would materialize roughly 2.9 GiB of image tensors.

## Sync conclusion

The new origin is safely isolated. The checked-out feature branch begins exactly at the complete locally preserved lineage, retains all nine later commits without rewriting, and now includes local audit/curation commits above that base. The formerly untracked notebook is committed in the archive and remains recoverable. Execution must keep the former repository untouched unless a separately approved visibility-only action occurs at the end.

## Execution progress

- Phase 1 passed independent preservation review. The full lineage is anchored at the new origin and no configured route to the former repository exists.
- Phase 2 curation is complete after review fixes: all seven committed notebooks, the PDF, and six PNGs are exact renames; the incomplete local notebook is newly preserved under `archive/`; only `.DS_Store` metadata was removed; and both `/Datasets/` and `/data/` remain ignored.
- Phase 3 passed independent review after two corrective commits. The maintained package now provides anonymous acquisition, deterministic/checksummed manifests, exact and perceptual duplicate checks, a 164,546-parameter CNN, training/checkpointing, bounded-memory evaluation, reusable inference, and a six-command CLI plus a no-download CPU smoke workflow.
- Training and evaluation verify every manifest row against the current source-file SHA-256 before consuming it, reject aliased/duplicate paths, and reject exact train/validation overlap. The bounded discovery logic accepts the existing ignored nested dataset without moving or committing it.
- A clean Python 3.10 environment using pinned PyTorch 2.7.1/torchvision 0.22.1 CUDA 11.8 dependencies passed `pip check`, Ruff lint/format, compilation, deterministic manifest checks, and the full synthetic prepare/train/evaluate/predict smoke workflow. The smoke checkpoint is only test evidence and is not a reported model result.
- Phase 4 implementation trained all 51,376 training samples for 12 epochs on CPU and selected epoch 12 on 11,024 validation samples. The 674,997-byte checkpoint has 164,546 parameters and SHA-256 `8b3d071082615de4a21a6303c8e9ca5496a747c6f5707064f26b3d6a9f6c40a7`.
- The untouched 15,600-image same-corpus source test produced 99.82% accuracy and 99.82% macro F1. The anonymously downloaded, CC0, 780-image external A-Z set produced only 17.56% accuracy and 16.82% macro F1. Current documentation treats the 82.26-point gap as evidence of severe capture-domain bias and does not present the model as deployment-ready.
- The real manifest audit found zero exact cross-split duplicates and 1,754,615 coarse dHash review candidates at Hamming distance <= 5. Signer/session separation remains unverifiable because the source lacks those identifiers.
- Phase 4 passed independent review with no findings. The reviewer recomputed both metric sets, matched every published artifact and manifest hash, verified the untouched test inventory and zero SHA overlap across all four manifests, safely loaded the checkpoint, and confirmed that no dataset, secret, or machine-local path is published.
- Phase 5 added an optional Gradio 6 upload/webcam interface over the shared `Predictor`. It binds only to loopback, disables public sharing and public prediction APIs, and requires no account or hosted service.
- The interface guards missing checkpoints, absent/non-image/tiny/flat inputs, and low-confidence predictions. Its permanent warning reports the measured 17.56% external-domain accuracy, rejects accessibility-system framing, and calls out the still-image limitation for motion-dependent J/Z.
- A committed CC0 external sample provides a deterministic UI check: the true-A image is incorrectly predicted as N at 75.9% confidence. The real browser capture in `docs/demo/demo-screenshot.png` preserves that failure and its visible warning instead of presenting a curated success case.
- Phase 5 passed independent frontend/code review at `9259128` with no findings. The reviewer confirmed the local-only boundary, shared preprocessing/class mapping, optional dependency isolation, Gradio 6 configuration, failure paths, sample provenance/hash, screenshot authenticity, responsive layout, accessibility styling, and clean worktree.
- Phase 6 added 26 tests covering configuration, deterministic manifests, canonical and escaping paths, checksums, pooled and nested source-partition layouts, exact-duplicate grouping/refusal, deterministic preprocessing, checkpoint/class-map contracts, inference, demo failure states, CLI JSON wiring, and a generated-data prepare/train/evaluate/predict workflow.
- The CI-equivalent local run passed all 26 tests with 82.46% statement coverage, above the enforced 75% floor. Ruff lint/format, compilation, `pip check`, and `git diff --check` also passed without warnings.
- `.github/workflows/ci.yml` uses a free `ubuntu-latest` runner, current official `actions/checkout@v7` and `actions/setup-python@v7`, read-only contents permission, a 20-minute timeout, pinned constraints, generated data only, and no secrets, account, full dataset download, or full model training.
- The final tree audit found no tracked `/Datasets`, `/data`, `/artifacts`, environment, cache, or credential files; no secret-pattern hits; no unreachable/corrupt Git objects; and no blob above 4.87 MB. The 675 KB released model retains its verified SHA-256. Output-heavy notebooks remain unchanged and confined to the explicitly preserved historical archive.
- Phase 6 passed independent review at corrective commit `deeb0cb`. The hosted CI run remains deliberately deferred until the feature branch is pushed to the new repository during the final publication phase.
- The final README remains for the next phase.
