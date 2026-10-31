# Phase 6 — Write-up & Demo

Assembling the final deliverables and the reusable inference demo. **Status: done.**

## Goal

Turn the five completed phases into a finished, shareable project: a reusable
`predict_action` function, a consolidated results report, a model card, and a polished
public README.

## Why / concepts

A model that only exists as scattered cells and per-phase notes is not a finished project.
This phase produces the three things a reader or user actually needs: a way to *run* it
(the demo function), a single place that tells the *whole* results story (`RESULTS.md`),
and an honest deployment summary that travels with the model (`MODEL_CARD.md`). The README
becomes the front door. None of this is new modelling — it is making the work usable and
legible, which is the difference between an experiment and a portfolio piece.

## Code walkthrough

**`predict_action(video_path)`** composes the whole pipeline into one call:

```python
frames = read_clip_frames(video_path, NUM_FRAMES)          # (16,3,H,W) uint8
preprocessed = preprocess_frames(frames)                   # (16,3,224,224)
clip_features = feature_extractor(preprocessed)            # (16,512)
logits, frame_attention = model(clip_features.unsqueeze(0), return_weights=True)
probabilities = F.softmax(logits, dim=1)[0]
```

It reuses the exact Phase 2 functions (`read_clip_frames`, `preprocess_frames`,
`feature_extractor`) and the Phase 3 attention model, so inference matches training with no
duplicated logic. `clip_features.unsqueeze(0)` adds the batch dimension the model expects.
It returns the predicted action, its confidence, the full ranked class probabilities, and
the per-frame attention — so the function answers both "what is this" and "where did the
model look". A demo cell runs it on a test clip and prints the top-3 prediction.

## Deliverables produced

- **`predict_action` demo** (notebook): video in -> action, confidence, probabilities,
  per-frame attention.
- **`docs/RESULTS.md`:** the consolidated results — pooling comparison, attention saliency,
  and the full responsible-AI audit, with the honest reading of each.
- **`docs/MODEL_CARD.md`:** intended/out-of-scope use, aggregate and per-class performance,
  calibration, OOD behaviour, and limitations.
- **`README.md`:** completed front page with the real results and run instructions.

## Key decisions

- One self-contained notebook for the whole pipeline, with a demo cell at the end (rather
  than a separate inference script), so the project reads top-to-bottom and runs on Colab
  without setup.

## What we measured

- Demo: `predict_action` runs end-to-end on a test clip and returns a correct, confident
  prediction with its per-frame attention.

## What broke

Nothing — this phase is assembly. The honest findings it reports (calibration fix not
transferring, per-class failures, OOD weakness) are Phase 5's, surfaced here rather than
hidden.

## Interview questions

**Q1. The project's accuracy (~0.85) is decent but not remarkable. If you were presenting
FrameSight, what would you argue is its actual strength?**

The strength is not the accuracy number but the discipline around it. The project shows a
complete, honest pipeline — transfer-learned features, a temporal model, attention, and an
inference demo — and then audits the result instead of stopping at the headline: it shows
*where* the model fails and *why* (appearance-similar actions), reports a calibration fix
that did not work rather than hiding it, and quantifies an OOD signal while being clear it
is not a deployable detector. Every claim has a number and a reason. For a portfolio, that
judgement — knowing what a model can and cannot be trusted to do, and saying so — is worth
more than a higher accuracy with no audit, because it is exactly what real deployment
decisions require.

**Q2. The demo function reuses the training-time preprocessing and feature extractor
directly. Why does that matter, and what bug class does it prevent?**

It prevents train/serve skew — the bug where inference preprocesses inputs even slightly
differently from training (a different resize, normalisation, or frame-sampling rule), so
the model sees out-of-distribution inputs at serving time and silently loses accuracy for
no visible reason. By calling the same `read_clip_frames`, `preprocess_frames`, and frozen
`feature_extractor` used to build the cached training features, the demo guarantees a clip
is turned into exactly the kind of input the model was trained on. Sharing the code path,
rather than reimplementing it, is the cheapest way to keep training and serving in lockstep.
