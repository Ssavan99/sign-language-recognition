# Domain-robustness experiment

The classifier this experiment was run against scored 99.82% on a same-corpus
image holdout and 17.56% on a separate capture source. That 82.26-point gap was
the project's central weakness, and this document records the experiment run
against it: what was tried, how a candidate was chosen, and what the blind test
said afterwards. It ends with a replaced released model and a gap of 67.25
points.

## The problem the experiment has to solve first

Ranking augmentation recipes needs a metric that can tell them apart.

- **Source validation cannot.** It sits at 99.85%. Every candidate looks the
  same, because validation images share the training corpus's capture setup.
- **The external set must not.** The 780-image CC0 external set is the only
  unbiased generalisation measurement this project owns. Using it to choose
  between candidates would convert it from evidence into a tuning signal, and
  there is no second blind set to replace it.

So neither existing metric can drive the choice.

## The stress benchmark

A fixed corruption benchmark, applied only to source **validation** images,
supplies the missing selection signal. It is defined in
[`src/asl_recognition/robustness.py`](../../src/asl_recognition/robustness.py)
and versioned as `stress-v1`. Every recorded score names the version it was
produced under, so scores from different benchmark definitions can never be
silently compared.

Each validation row is assigned one corruption by position (`index % 8`), giving
a single deterministic pass the same size as the validation split with balanced
coverage. Per-image noise is seeded from the row's recorded SHA-256, so the
benchmark does not depend on batch order, worker count, or run order.

| Corruption | What it simulates | Independent of training augmentation |
| --- | --- | :---: |
| `jpeg_q25` | Heavier compression than the source corpus | yes |
| `gaussian_noise` | Sensor noise, σ = 0.08 | yes |
| `gamma_low` | Underexposure, γ = 0.45 | no |
| `gamma_high` | Overexposure, γ = 2.0 | no |
| `hue_shift` | White-balance error, ~54° turn | no |
| `contrast_crush` | Flat, washed-out capture | no |
| `letterbox` | Different framing and backdrop | yes |
| `downscale` | Distance and resolution loss | yes |

### What this benchmark is not

It is a **proxy for capture-domain shift, not a measurement of it.** Four of the
eight corruptions are photometric, and the stronger training profiles augment
photometrically, so those profiles hold a structural advantage on exactly those
four. That overlap is why the table above marks independence explicitly, why
per-corruption accuracy is always reported next to the aggregate, and why the
blind external set — not this benchmark — decides whether anything gets
published.

## Pre-registered protocol

Fixed before any candidate was trained:

1. **Screen** every augmentation profile on an identical stratified subset of the
   source training split, with one shared seed and epoch budget.
2. **Select** the candidate with the highest stress-benchmark accuracy at its
   best epoch, where the best epoch is chosen by stress-benchmark loss. Ties
   break toward lower parameter count, then lower seed.
3. **Train** the single winner on the full training split.
4. **Evaluate once** on the untouched 15,600-image source test partition and once
   on the blind 780-image external set.
5. **Publish** a replacement only if external accuracy improves by at least
   **5 percentage points absolute** over 17.56% *and* source test accuracy stays
   at or above **95%**.

No re-selection is permitted after an external score is seen. If the selected
candidate fails the rule, that is recorded as a failure; a different candidate is
not then tried against the external set.

## Augmentation profiles

All four share one inference contract — RGB, resized to 64x64, scaled to
[0, 1], ImageNet-normalised, no augmentation — so any of them produces a model
that drops into the existing CLI, demo, and browser export unchanged.

| Profile | Recipe |
| --- | --- |
| `baseline` | The released model's recipe, kept byte-for-byte as a control: mild resized crop, horizontal flip, 10° rotation, mild colour jitter. |
| `robust` | Wider crops, horizontal flip, affine jitter (20°, translate, scale, shear), strong colour jitter with hue, 30% random grayscale, and random erasing. Aimed at the colour, framing, and backdrop shortcuts the corpus makes available. |
| `robust_noflip` | Identical to `robust` with horizontal flip removed. Flipping is inherited from the released recipe but is questionable for fingerspelling: it maps a sign to its opposite-handed form, and for asymmetric letters that is a different shape rather than the same shape seen again. |
| `trivialaugment` | Resized crop, horizontal flip, torchvision's parameter-free `TrivialAugmentWide`, and random erasing. Included so the comparison is not purely between recipes hand-tuned by the same author. |

## Results

### Screening

Two screens were run. The first used 150 images per class for 10 epochs, and its
curves showed every candidate still climbing steeply at the final epoch. That
budget measured convergence speed rather than generalisation, and it
systematically favoured the weakest augmentation, so it was rerun with the
epoch budget doubled: 100 images per class for 20 epochs, no early stopping,
seed 42, identical for every candidate.

**Screen 2 (100 per class, 20 epochs, 2,600-row stress set):**

| Profile | Select on | Best epoch | Clean validation | Stress benchmark | Independent corruptions |
| --- | --- | ---: | ---: | ---: | ---: |
| `robust_noflip` | stress | 19 | 72.69% | **62.35%** | 58.00% |
| `robust` | stress | 20 | 64.12% | 51.23% | 47.62% |
| `baseline` | stress | 19 | 88.54% | 49.27% | **61.62%** |
| `trivialaugment` | stress | 18 | 66.04% | 39.62% | 51.46% |

`robust_noflip` won under the pre-registered rule, and it also led the first
screen, so the winner was stable across both budgets. Removing the horizontal
flip was worth 11.1 stress points over the otherwise identical `robust`, which
supports the suspicion that flipping a fingerspelled letter produces a different
shape rather than the same shape seen again.

**The aggregate was not a clean win.** On the four corruptions independent of
any training augmentation, `baseline` still led, 61.62% to 58.00%. The
challenger's advantage came mostly from the photometric half of the benchmark,
which its own colour and grayscale augmentation trains for. That gap did narrow
as it converged (5.4 points in screen 1, 3.6 in screen 2), and it took the lead
on `letterbox`, the corruption closest to real capture-domain shift. Going into
the blind evaluation, this looked more likely to fail the publication rule than
to pass it.

Screening stress scores are **not** comparable with the final run's: the
benchmark scores whatever validation rows a run consumes, and these screens used
2,600 rows against the final run's 11,024. Recorded row digests make that
mismatch a hard error in `tools/summarize_screening.py` rather than a footnote.

### Final result

`robust_noflip` was trained on the full 51,376-image training split for 12
epochs, selecting epoch 10 by stress-benchmark loss, then scored exactly once on
each held-out set.

| Evaluation | Previous model | `robust_noflip` | Change |
| --- | ---: | ---: | ---: |
| Internal source test (15,600) | 99.82% | 98.92% | -0.90 pts |
| External source test (780) | 17.56% | **31.67%** | **+14.10 pts** |
| Internal-to-external gap | 82.26 pts | 67.25 pts | -15.01 pts |

**The pre-registered rule passed on both criteria** (external gain >= 5 points,
internal >= 95%), so the released checkpoint was replaced. On the external set
18 of 26 classes improved, 6 regressed, and 2 were unchanged; zero-recall
classes fell from six to one.

The blind result contradicted the pessimistic reading of the screening
breakdown. The independent-corruption sub-score had favoured `baseline`, yet
`baseline` is the recipe the previous 17.56% model was trained with, and the
challenger nearly doubled external accuracy. Treating a synthetic corruption
suite as a stand-in for real capture-domain shift is exactly the inference this
document warned against, and here the proxy under-predicted rather than
over-predicted the outcome. It selected a genuinely better model while being
wrong about the margin.

Stress accuracy at the selected epoch was 89.97% on the full 11,024-row
validation set, with `letterbox` (60.7%) and `downscale` (67.6%) far below every
other corruption. Framing and resolution remain the weakest axes, and they are
the two that most resemble why the external set is hard.

### What this does not establish

- 31.67% is a large relative improvement and still a poor absolute result. Two
  of every three external images are misclassified. Nothing here is deployable.
- Only one external capture source has ever been used. Improving on it is not
  evidence of improvement on a third source.
- The source corpus still has no signer or session identifiers, so no
  signer-independent claim is possible at any accuracy.
- The gain came from augmentation on a single-source corpus. It is a cheaper
  substitute for diverse training data, not a replacement for it.

## Follow-up: does adding a second corpus help? (negative result)

The augmentation result above raised an obvious question: if the model is
limited by having seen one capture setup, would a second corpus help more than
augmentation did? This section records that experiment and its answer, which was
no.

### Source

`ayuraj/asl-dataset`, CC0, 1,815 A-Z images at 400x400, downloaded anonymously
with no account or key. Two other candidates were rejected on licensing before
download: `lexset/synthetic-asl-alphabet` grants no reuse, and
`ammarnassanalhajali/american-sign-language-letters` states no licence.

Against all four existing manifests it had **zero exact and zero perceptual
overlap**, so the blind external set stayed blind and the internal test
partition could not be inflated.

It is honest to describe what it actually adds: a different signer, a black
backdrop, brighter even lighting, and a tighter crop. That is a **second
controlled studio condition**, not the natural cluttered scenes the external set
contains. It took the training pool from one domain to two; it did not make the
pool naturally diverse.

### Method

The winning profile was held fixed at `robust_noflip` so the only change was the
data. The supplement was repeated 7 times to reach 14.8% of a 60,266-image pool
-- enough to matter against 51,376 primary images, not so much that 1,270 unique
images would simply be memorised. Selection stayed on the frozen `stress-v1`
benchmark, built only from primary validation images. The external set was
scored once, after training finished.

### Result

| Evaluation | Released model | Enlarged pool | Change |
| --- | ---: | ---: | ---: |
| Internal source test (15,600) | 98.92% | 97.33% | -1.59 pts |
| **External blind test (780)** | **31.67%** | **30.64%** | **-1.03 pts** |
| Supplement held-out test (260) | not trained on it | 94.23% | -- |

**The pre-registered rule failed, so the released model was retained.**

The -1.03 point external change is 0.62 standard errors on 780 samples, so the
correct reading is *no measurable improvement*, not *made it worse*. The
1.59-point internal drop is a real cost. Zero-recall external classes rose from
one to four.

### Why this is worth recording

The model learned the second corpus almost perfectly -- **94.23%** on a held-out
slice it never trained on -- while transfer to the external set did not move at
all. That is the informative part.

Adding a second corpus taught the model that corpus. It did not teach it to
generalise. **Domain count is not the lever; domain kind is.** Both training
corpora are studio captures against a uniform backdrop, and the external set is
neither. Doubling the number of uniform-backdrop domains does nothing for images
with real backgrounds, cluttered scenes, and natural light.

This also bounds the augmentation result above more sharply. Augmentation moved
external accuracy 14.10 points; a whole additional corpus of the wrong kind moved
it by nothing. The remaining gap is unlikely to close without training data that
actually resembles the deployment condition -- varied real backgrounds, lighting,
distances, and cameras -- rather than more of the same kind of data.

The supplementary-corpus pipeline is retained in the repository because the
experiment is reproducible from it, and because a genuinely diverse source could
be dropped into the same path unchanged.

## A note on browser versus CLI confidence

The published page and the Python CLI report the same predicted letter for the
same image but slightly different confidences -- about 79.6% in the browser
against 76.2% from the CLI on the held-out A sample.

This is not a weight mismatch. `tools/check_browser_model.mjs` compares the
browser forward pass against PyTorch on a synthetic tensor and agrees to within
6e-8, one float32 ulp. What differs is everything *before* the tensor: the
browser resizes with a canvas `drawImage`, the CLI resizes with Pillow, and the
two resamplers do not produce identical 64x64 pixels from a 400x400 source.

It is a useful reminder that a model contract is not only its weights. Two
faithful implementations of the same network disagree measurably when their
preprocessing differs, and this model is sensitive enough to capture conditions
that a small resampling difference moves its confidence by three points.
