# Phase 4 — Attention Saliency

Looking inside the attention model: which frames does it weight, per action. **Status:
pending run.**

## Goal

Read the attention model's per-frame weights on the test clips and visualise them, so we
can see *where in each clip* the model looks — both averaged per class and on individual
example clips with their actual frames.

## Why / concepts

The accuracy table said attention is competitive, not dominant. The attention weights
say *why*. This is the interpretability payoff of the architecture: unlike mean or
last-state pooling, attention produces a weight per frame that we can read directly as a
temporal saliency map. Two questions:

- **Per class:** does each action concentrate attention on a particular part of the clip
  (a key moment), or spread it evenly across all 16 frames?
- **Per clip:** on a single example, which specific frames did the model lean on?

There is also a direct link back to Phase 3. If the attention turns out to be roughly
**uniform** across frames, that *explains* why mean pooling tied it: when every frame is
about equally informative, a learned weighting has little to improve on, and averaging is
already near-optimal. A flat saliency map is not a failure — it is the mechanism behind
the Phase 3 result.

## Approach (what the cells do)

1. Run the trained attention model over the test set with `return_weights=True` to get a
   (N_test, 16) matrix of per-frame attention weights.
2. **Per-class heatmap:** average the weights within each class and plot a 9 x 16 heatmap
   (class x frame position), so concentration or uniformity is visible at a glance.
3. **Per-clip strips:** rebuild the test clips' file paths in the same order as the
   cached features, re-read a few clips' 16 frames, and display them in a row annotated
   with their attention weights (highlighting the most-attended frame).

## Key decisions

- Uses the attention model already trained in Phase 3 (`trained_models["attention"]`);
  no new training. Interpretability only.

## What we measured

<!-- filled after running -->
- Per-class attention pattern: _TBD (concentrated vs uniform)_.
- Example clips: _TBD_.

## What broke

<!-- filled after running -->

## Interview questions

<!-- filled when the phase closes -->
