# Phase 2 — Feature Extraction

Turning the 457 raw video clips into cached frame-feature tensors. **Status: pending
run.**

## Goal

Convert each clip into a fixed-size sequence of frame features — shape (16, 512) — and
cache the result for every split to Drive as `features/{train,val,test}.pt`, so Phase 3
trains on tensors, not video.

## Why / concepts

- **Transfer learning (Group 2).** With only 270 training clips, training a vision
  model from scratch would overfit instantly. Instead a **ResNet18 pretrained on
  ImageNet, frozen**, turns each frame into a 512-dimensional feature vector. Its fully
  connected classifier head is replaced with an identity, so the output is the pooled
  512-d feature. No gradients flow into it.
- **Cache once (persistence).** Decoding video and running a CNN is the slow part of
  the whole project. Doing it once and saving the features means every later
  experiment — every pooling strategy, every training run — reads a small tensor
  instead of re-decoding video. It also makes the work resumable after a Colab timeout.
- **Fixed-length frame sampling.** Clips vary in length, but the sequence model wants a
  fixed number of steps, so each clip is reduced to **16 evenly-spaced frames** across
  its full duration.

## Approach (what the cells do)

1. Build the frozen ResNet18 feature extractor (replace `fc` with identity; `eval`;
   `requires_grad = False`), and a preprocessing function (resize to 224x224, scale to
   [0,1], normalise with ImageNet statistics).
2. For each clip: read all frames with OpenCV, pick 16 evenly-spaced ones, convert
   BGR to RGB, preprocess, and run them through the backbone to get a (16, 512) tensor.
3. Loop over each split, stack the per-clip features into (N, 16, 512) with an integer
   label per clip, and save `{features, labels, classes}` to Drive. Skip a split whose
   cache file already exists.
4. Verify by reloading the cached tensors and printing their shapes.

## Key decisions

- **D-3** — frozen pretrained ResNet18 (rejected: train a CNN from scratch; fine-tune
  end-to-end).
- **D-4** — extract once and cache to Drive (rejected: decode + CNN every epoch).
- **D-5** — 16 evenly-spaced frames per clip (rejected: all frames; 8 frames).

See `DECISIONS.md`.

## What we measured

<!-- filled after running -->
- Cached shapes: train _TBD_ (expect ~(270, 16, 512)), val _TBD_ (~(90, ...)),
  test _TBD_ (~(97, ...)).
- Any clips skipped as unreadable: _TBD_.
- Rough extraction time: _TBD_.

## What broke

<!-- filled after running: unreadable clips, codec issues, memory, timing -->

## Interview questions

<!-- filled when the phase closes -->
