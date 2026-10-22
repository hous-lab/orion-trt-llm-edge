# Orion-Lite model variants: latency vs accuracy

All Lite numbers below are end-to-end Orin latency at MAXN power mode and plan_L2 on
the 173-frame Bench2Drive val subset (tiny173; 133 frames with valid future
trajectory), measured through the full real pipeline (real images + real cross-frame
memory evolution). plan_L2 is in meters; lower is better. One caveat up front: the
Lite tiny173 metric is not directly comparable to the 7B official metric (0.690)
quoted in the README - different harness and frame count, so read the two tables
independently.

## The variant matrix

| Variant | Orin e2e | plan_L2 @1s | @2s | @3s |
|---|---|---|---|---|
| 24L@640 + original student (undistilled starting point) | 678 ms | 0.814 | - | - |
| 24L@512 + original student (baseline) | 404 ms | 0.868 | 1.652 | 2.326 |
| 12L@512 (backbone distilled 24 to 12 layers) | 236 ms | 0.694 | 1.320 | 2.084 |
| 6L@448 (Route B decision version) | 121 ms (154 ms at 50W) | 0.776 | 1.754 | 2.878 |
| 6L@384 + original student (final) | 85.2 ms (vision 69.7 + student 15.5) | 0.704 | 1.340 | 2.033 |

Three observations from the matrix:

1. Resolution is the dominant training-free lever: 640 to 512 alone takes the
   undistilled 24L tower from 678 ms to 404 ms end-to-end. The vision tower runs at
   the input resolution, so every pixel cut shrinks every vision kernel.
2. The 12L distilled tower is not just faster - it scores better than the baseline
   (0.694 vs 0.868 @1s). With the student and heads fixed, compressing the backbone
   does not necessarily hurt planning; see the E-note discussion below and the
   feature-fidelity vs task-alignment caveat at the end.
3. 384 input only became usable after the backbone-only distillation: the 24-layer
   tower at 384 collapses (vision-embedding cos 0.866), while the 6-layer distilled
   tower at 384 holds cos 0.926.

## How the final variant came to be (the 6L story)

An earlier round of 6-layer experiments went through a three-tier online
distillation in which the student (Transformer decoder + VAE) was retrained
together with the backbone. It produced a 14.82 m plan_L2 - an 18x regression that
looked like "6 layers does not work". A cross-check (feeding the 6L and 24L towers
to the same untouched 8L student, differing by only 0.34 m) located the real culprit:
the retrained student had been bent by the small distillation set, not the 6-layer
tower. The student is extremely sensitive to its input distribution; retraining it on
the distillation data set bends it, and long online distillation runs diverge numerically
(feat=nan around 18k-25k steps on the transformer-decoder path).

The fix was to stop touching the student: distill the backbone only (feature mimicry,
the same recipe that worked for 12L), keep the original student and heads frozen, and
the 6L@384 variant scores 0.704 m - on par with 12L and 19% better than the 24L@512
baseline - at 85.2 ms, under the 100 ms real-time bar, with no student retraining.

For the pos-embed handling that makes 384 work at all: the EVA-ViT positional
embeddings are bicubic-interpolated to the new resolution, reusing the upstream
interpolation weights (the same mechanism the training repo uses for variable
resolution).

## Feature fidelity vs task alignment

Two acceptance rules fall out of this project's measurement history:

- Intermediate alignment metrics (cos / MSE) are training monitors, not acceptance
  gates. A 12L tower with healthy ve/det/map cos (0.88/0.89/0.85) still shifts
  detection score calibration (high-confidence recall ~60%), while planning actually
  improves - the dimensions decouple. Acceptance runs the downstream task metric
  through the real pipeline.
- Proxy evaluations (zeroed cross-frame memory) degrade every model systematically
  (24L: 2.34 m proxy vs 0.868 m real) and can even invert relative rankings. Any
  cross-variant comparison in this repo uses the real pipeline with real memory
  evolution.

Per-variant eval JSONs (committed under lite/results/) back the table; the harness
recipe and the exclusion rules for contaminated frames are in docs/lessons.md.
