# Domain-robustness experiment

The released classifier scores 99.82% on a same-corpus image holdout and 17.56%
on a separate capture source. That 82.26-point gap is the project's central
weakness, and this document records the experiment run against it: what was
tried, how a candidate was chosen, and what the blind test said afterwards.

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

Screening and final results are recorded in this section once the runs complete.

<!-- RESULTS: screening -->

<!-- RESULTS: final -->
