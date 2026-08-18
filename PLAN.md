# Sign Language Recognition Revival Plan

Status: approved; execution in progress  
Created: 2026-08-18

## Completion criteria

This plan is complete when the full local commit history is safely anchored in the new standalone repository; the project has a reproducible, tested pipeline for isolated ASL alphabet image classification; a validated compact model and local demo work without paid services or account signups; the README accurately documents setup, results, and limitations in neutral project voice; all checks pass; a feature-branch pull request is open against the new repository; and the separately gated original-repository visibility decision has been resolved without any unapproved action.

## Rules and guardrails

- **$0 and no signup:** Use only the local machine, ordinary public GitHub features, public data that can be downloaded anonymously, and free/open-source tools. Do not require Colab, Kaggle login/API tokens, cloud accounts, paid APIs, subscriptions, or hosted services.
- **New repository only:** The only working remote is `https://github.com/Ssavan99/sign-language-recognition.git`. Never push to, branch on, open PRs/issues in, archive, delete, alter collaborators on, or change settings in the former repository.
- **One deferred exception:** A public-to-private visibility change on the former repository may be considered only after the new repository is live and good, only after a fresh fork/visibility check, and only after explicit user approval. Never suggest or perform deletion.
- **Preserve history:** Do not rebase, squash, filter, amend published commits, rewrite authors/emails, force-push, or replace the merge topology. Recover the nine later commits by starting the feature branch at the locally preserved descendant `ghcheck/main`.
- **Feature branch:** Never implement on `main`. Use a feature branch, small focused commits, and clear commit messages. Do not merge the final pull request.
- **Preserve untracked work:** Do not silently delete the incomplete local notebook. Move it into a clearly labeled historical/archive location unless the user explicitly authorizes deletion.
- **Neutral README:** Describe what the system does, how it works, how to run it, results, and limitations. Use project voice, omit authorship narration, do not claim sole authorship, and do not frame it as a class project.
- **Honest claims:** Frame the project as isolated ASL alphabet image classification, not continuous sign-language translation, full ASL recognition, or demonstrated real-world accessibility performance.
- **Licensing restraint:** Attribute datasets and their licenses. Do not add a new license for the group-authored code without explicit authorization that contributor licensing is settled.
- **Approval boundaries:** Ask before expanding scope, spending money, requiring an account, deleting work, rewriting Git history, changing public claims materially, or changing the former repository's visibility.
- **Resumability:** Keep `PLAN.md` checkboxes and `SYNC.md` current after every phase. Review each phase's diff for correctness, and use a reviewer subagent before starting the next phase.

## Frontend decision

A frontend genuinely helps after the model and evaluation path are trustworthy. Build a small local image-upload/webcam demo that displays the predicted letter, confidence, and static-image/domain limitation. It is a thin inference wrapper, not the technical center of the project, and it must require no hosted account. If a validated exportable model cannot be produced, do not ship a hollow frontend.

## $0 compute and data plan

- Reuse the existing ignored local dataset for development; never commit its 2.03 GB of files.
- Replace the token-based Kaggle CLI workflow with the official `kagglehub` client, which supports unauthenticated downloads for ordinary public datasets.
- Keep the original `grassknoted/asl-alphabet` source and its GPL-2.0 attribution for continuity.
- Use the separate public `danrasband/asl-alphabet-test` dataset (CC0) as an external-domain check if anonymous download and class alignment validate successfully.
- Provide a tiny committed/generated fixture and a CPU smoke mode so setup/tests do not download the full dataset or train a full model.
- Run full training only on the local machine with bounded epochs, early stopping, and a compact architecture. Use the local GPU only if the selected framework supports it safely; otherwise use CPU. No paid compute fallback.
- CI runs formatting/static checks and smoke tests only; it never downloads the full dataset or trains the production model.
- Keep the exported model below GitHub's ordinary file limit. Do not introduce Git LFS, paid storage, or a hosted model service.

## Phase 1 - Anchor and recover the complete lineage

- [x] Reconfirm that `origin` is the new standalone repository and no configured remote points to the former repository.
- [x] Create `revive/portfolio-ready-pipeline` directly from local `ghcheck/main`.
- [x] Confirm `main` is an ancestor and all nine later commits retain their hashes, authors, emails, and merge topology.
- [x] Carry `SYNC.md`, `PLAN.md`, and the untracked notebook safely onto the feature branch.
- [x] Push only the feature branch to the new origin so the complete lineage is anchored remotely.
- [x] Keep the inert `ghcheck/*` refs until remote anchoring is verified.

Acceptance criteria:

- The feature branch on the new origin reaches `5db1821` and contains all prior commits without rewriting.
- `git remote -v` lists only the new repository.
- The former repository has not been contacted or changed.
- The untracked notebook remains recoverable.

## Phase 2 - Curate the repository without erasing history

- [x] Remove tracked `.DS_Store` files from the final tree and add explicit ignore coverage.
- [x] Organize the historical report, result images, and original notebooks under clear `docs/`, `notebooks/`, and/or `archive/` paths.
- [x] Preserve the incomplete 128x128/Colab notebook under `archive/` with an explicit unsupported/incomplete label.
- [x] Keep substantive historical experiments accessible while excluding the broken transfer-learning result from the primary supported path.
- [x] Document the canonical data directory and ensure archives/extracted data remain ignored.
- [x] Update `SYNC.md` and review the phase diff with a reviewer subagent.

Acceptance criteria:

- The root has a clear entry point and no platform metadata.
- No substantive historical artifact is silently lost.
- Broken/incomplete experiments cannot be mistaken for supported results.
- The 2.03 GB dataset remains untracked.

## Phase 3 - Build a reproducible data and model pipeline

- [x] Add a Python project/dependency manifest with supported-version guidance and deterministic commands.
- [x] Implement importable modules and CLI commands for data acquisition, preparation, training, evaluation, and prediction.
- [x] Implement anonymous public-dataset acquisition with a local-data override and checksum/layout validation.
- [x] Create deterministic stratified train/validation manifests with saved seed/configuration and an untouched test contract.
- [x] Add exact and perceptual cross-split duplicate checks; report what cannot be grouped by signer/session because source metadata is absent.
- [x] Ensure validation/test preprocessing is unaugmented and all relevant random seeds are set.
- [x] Replace the 17M/67M-parameter notebook models with a compact exportable classifier and class/preprocessing metadata.
- [x] Add early stopping/checkpointing and bounded-memory evaluation.
- [x] Add a fast CPU smoke mode that uses a tiny fixture and finishes without the full dataset.
- [x] Update `SYNC.md` and review the phase diff with a reviewer subagent.

Acceptance criteria:

- A fresh environment can install the project and run `--help` and smoke commands without an account or full dataset.
- Repeated preparation with the same seed produces identical manifest hashes.
- Validation and test samples are never augmented.
- Cross-split duplicate checks run automatically.
- Training exports a compact model, class map, preprocessing contract, and run metadata.

## Phase 4 - Validate and record defensible results

- [ ] Verify the existing local dataset layout and correct the extra nesting without committing data.
- [ ] Run smoke tests before full training.
- [ ] Train the compact model locally with recorded environment, seed, configuration, duration, and artifact hash.
- [ ] Evaluate the internal same-corpus holdout with accuracy, macro F1, per-class metrics, confusion matrix, model size, and latency.
- [ ] Evaluate on the separate external-domain dataset if its anonymous access, license, and class mapping pass validation.
- [ ] Clearly separate historical notebook metrics from new reproducible metrics and label same-corpus versus external-domain results.
- [ ] Remove or correct any claim unsupported by the recorded artifacts, especially the invalid two-class transfer-learning result.
- [ ] Update `SYNC.md` and review results/code with an AI/code-review subagent.

Acceptance criteria:

- Every headline metric has a committed config/manifest hash and generated evaluation artifact.
- The invalid approximately 99.97% transfer-learning metric is not presented as a 26-class result.
- The historical 94.27% result is labeled as a same-corpus image holdout.
- Limitations explicitly cover dataset bias, missing signer/session metadata, static-image scope, and motion-dependent J/Z.
- If external evaluation cannot run, the reason is documented and no external-generalization claim is made.

## Phase 5 - Add the local inference demo

- [ ] Implement a small local Gradio image-upload/webcam interface over the shared inference module.
- [ ] Show predicted letter, confidence, and concise static-image/domain limitations in the UI.
- [ ] Handle missing model, invalid image, no detected/usable hand image, and low-confidence cases without crashing or overclaiming.
- [ ] Add a deterministic sample invocation and capture a real screenshot for the README.
- [ ] Keep the demo optional so core tests and package use do not require launching a server.
- [ ] Update `SYNC.md` and review the phase diff with a frontend/code-review subagent.

Acceptance criteria:

- The demo runs locally with one documented command and no signup.
- It uses the same preprocessing and class mapping as CLI inference.
- The screenshot is generated from the working application, not a mockup.
- The interface makes the project's limited scope visible.

## Phase 6 - Tests, CI, and quality controls

- [ ] Add tests for configuration, manifests, path/layout validation, duplicate detection, preprocessing, inference shape/class mapping, and failure cases.
- [ ] Add a CLI/inference integration smoke test using the tiny fixture or deterministic synthetic data.
- [ ] Add formatting/lint/type checks only where they improve maintainability.
- [ ] Add a public GitHub Actions workflow that uses free standard runners and never downloads full datasets or trains the full model.
- [ ] Run the complete local test suite and verify CI configuration.
- [ ] Perform a final secret, large-file, ignored-data, and notebook-output audit.
- [ ] Update `SYNC.md` and review the phase diff with a reviewer subagent.

Acceptance criteria:

- Local tests pass from the documented environment.
- CI passes on the feature branch using no secrets or paid services.
- No data dump, credential, model over the ordinary GitHub limit, or accidental environment artifact is tracked.

## Phase 7 - Write the portfolio-quality README and supporting docs

- [ ] Replace the README with a precise title and one-line isolated-ASL-image-classification pitch.
- [ ] Include the real demo screenshot or representative generated result near the top.
- [ ] Explain scope, pipeline, repository structure, dataset sources/licenses, model/evaluation design, and exact run commands.
- [ ] Include a results table that distinguishes historical same-corpus results, new same-corpus results, and any external-domain result.
- [ ] Document install, anonymous data acquisition, smoke mode, training, evaluation, CLI prediction, and local demo commands literally.
- [ ] Document honest limitations: isolated still images, controlled-source bias, absent signer/session metadata, J/Z motion, no continuous signing/translation, and no validated accessibility deployment claim.
- [ ] Use neutral project voice; omit personal/class-project narrative and authorship claims.
- [ ] State that no project code license is provided unless contributor licensing is explicitly resolved.
- [ ] Use no filler badges.
- [ ] Follow the README exactly on a clean/simulated-fresh path and correct any mismatch.
- [ ] Update `SYNC.md` and review the documentation diff with a reviewer subagent.

Acceptance criteria:

- A reviewer can understand the project, strongest result, and limitation in 30 seconds.
- Every documented command is tested or explicitly marked as a full-training command with expected resource cost.
- The README contains no unsupported claim or sole-authorship implication.
- A real visual result is present and legible.

## Phase 8 - Final review and ship to the new repository

- [ ] Review the full feature-branch diff, commit history, generated artifacts, and `PLAN.md` completion state.
- [ ] Run all tests/checks one final time.
- [ ] Confirm the only remote is the new standalone repository and GitHub still reports it is not a fork.
- [ ] Make small focused commits with the configured user identity; never amend published historical commits.
- [ ] Push the feature branch to the new origin.
- [ ] Open a pull request against `main` in `Ssavan99/sign-language-recognition` with an accurate summary and verification evidence.
- [ ] Leave the pull request unmerged for the user.

Acceptance criteria:

- The new repository contains the preserved historical lineage plus the reviewed improvement commits.
- All checks are green and the PR is open against the new repository.
- No action has been taken on the former repository.

## Phase 9 - Separately gated original-repository visibility decision

- [ ] Only after Phase 8, perform one fresh read-only check of the former repository's visibility and fork count.
- [ ] Reconfirm GitHub's current documented effects of public-to-private conversion, including fork detachment and collaborator access behavior.
- [ ] Assess whether the new repository's final implementation and framing are genuinely distinct enough that two public repositories would no longer be confusing.
- [ ] Present the evidence and ask for explicit approval before any visibility change.
- [ ] If approved, change only the former repository's visibility to private; do not alter collaborators, branches, issues, settings, archive state, or content.
- [ ] If approved and changed, verify visibility and existing collaborator access without modifying it.
- [ ] If not approved, leave the former repository exactly as-is and record that decision.

Acceptance criteria:

- The former repository is never deleted.
- Its collaborators are not removed or edited.
- Any visibility change occurs only after explicit approval and is the sole write action made to it.
- The final handoff states whether distinct naming/framing alone would have been sufficient and why.

## Blockers

None at the approval gate. Full-model training and external-domain evaluation remain resource/runtime risks, but each has a smoke-mode fallback that preserves truthful documentation without introducing paid compute or account requirements.
