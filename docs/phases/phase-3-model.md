# Phase 3 — The Model

Training the temporal model on the cached frame features, and measuring what attention
pooling buys. **Status: pending run.**

## Goal

Train a classifier that reads the (16, 512) sequence of frame features for a clip and
predicts its action, and compare three ways of pooling the sequence into one vector —
mean, last-state, and attention — under identical conditions.

## Why / concepts

- **The sequence model recovers motion (Group 3).** The frozen CNN described each frame
  in isolation; it saw no motion. A **bidirectional LSTM** reads the sequence of 16
  frame features in both directions, so its per-step outputs encode how the clip
  evolves over time — the dynamics the CNN could not see.
- **Pooling turns a sequence into a decision (Group 4).** The LSTM produces one vector
  per frame; the classifier needs one vector per clip. How we collapse 16 vectors into
  1 is the experiment:
  - **mean** — average every frame equally (a position-blind baseline),
  - **last** — use only the final frame's state (what a plain RNN classifier does),
  - **attention** — learn a weight per frame and take a weighted sum, so the model can
    emphasise the discriminative moments (and the weights are an interpretable saliency,
    used in Phase 4).
- **Small data demands regularisation.** With 270 training clips, the model overfits
  easily, so it uses dropout, weight decay, and — crucially — **selects the epoch with
  the best validation accuracy** rather than the last epoch, and reports that model's
  test accuracy. The tiny val/test sets (90 / 97 clips) also mean results are noisy, so
  differences of a point or two are not meaningful.

## Approach (what the cells do)

1. Load the cached `features/{train,val,test}.pt` tensors from Drive.
2. Define `AttentionPooling` (additive attention over time) and `VideoActionClassifier`
   (BiLSTM -> chosen pooling -> dropout -> linear), from scratch.
3. A training function that trains with Adam + weight decay, tracks validation accuracy
   each epoch, keeps the best-validation weights, and returns that model's test accuracy.
4. Train all three pooling strategies under the same seed and settings, and print a
   comparison table.

## Key decisions

- **D-6** — BiLSTM + attention pooling, with mean and last-state pooling kept as explicit
  baselines (rejected: mean-only, last-only, a Transformer encoder). See `DECISIONS.md`.

## What we measured

<!-- filled after running -->
- Pooling comparison (best-val / test accuracy): mean _TBD_, last _TBD_, attention _TBD_.
- Chance level: 1/9 = 0.111.

## What broke

<!-- filled after running: overfitting, instability, val/test noise -->

## Interview questions

<!-- filled when the phase closes -->
