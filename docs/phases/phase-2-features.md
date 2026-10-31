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

- **Cached shapes:** train (270, 16, 512), val (90, 16, 512), test (97, 16, 512) —
  one 512-d feature per frame, 16 frames per clip, saved to
  `MyDrive/framesight/features/{train,val,test}.pt`.
- **Clips skipped:** 0 — all 457 clips decoded and extracted cleanly.
- **Timing:** a few minutes on the T4, dominated by video decoding. Runs once; re-runs
  print "already cached" and skip.

## What broke

Nothing in Phase 2 itself. Every `.avi` was readable by OpenCV, and the frozen-ResNet
pass produced the expected shapes on the first run. (The data-source trouble all
happened in Phase 1.)

## Interview questions

**Q1. Why freeze the ResNet and extract features once, instead of fine-tuning it
end-to-end with the temporal model?**

Three reasons, and they compound. First, data: with 270 training clips, fine-tuning an
11-million-parameter CNN would overfit almost immediately — there is nowhere near
enough signal to re-learn good visual features, so a frozen ImageNet backbone that
already has them is the safer choice. Second, cost: decoding video and running the CNN
is the expensive part of the pipeline, and freezing lets us do it exactly once and
cache the result, after which every experiment trains in seconds on small tensors;
end-to-end training would re-run the CNN on every frame every epoch. Third,
separation of concerns: freezing cleanly splits "what each frame looks like" (the CNN's
job, solved by transfer) from "how the frames evolve over time" (the sequence model's
job, which is what we actually want to study). The cost is a ceiling on accuracy —
features tuned for ImageNet object classification are not perfectly matched to these
actions — but on this little data that ceiling is far above what fine-tuning could
reach without overfitting.

**Q2. ResNet18 was trained on ImageNet, which is single-image object classification.
Why are its features useful for *video action* recognition, and what do they
fundamentally not capture?**

They are useful because most of what distinguishes these actions is still *appearance*:
a bench and a barbell, a swimming pool, a pull-up bar, a person inverted versus
upright. ImageNet features encode exactly that kind of object-and-scene information, and
it transfers directly — a frame of BenchPress and a frame of Diving look very
different, and the CNN already knows how to describe that difference. What a per-frame
CNN fundamentally cannot capture is **motion**: the features describe each frame in
isolation, with no notion of how one frame follows another, so direction, speed, and
temporal order are invisible to them. That missing information — the dynamics — is
precisely what the BiLSTM in Phase 3 is there to recover from the *sequence* of
features. The division is clean: the CNN sees the stills, the sequence model sees the
motion.

**Q3. Each clip was reduced to 16 evenly-spaced frames regardless of its length. What
is the risk of fixed-count uniform sampling, and when would it hurt?**

Uniform sampling assumes the action is spread roughly evenly across the clip, so 16
samples are a fair summary. The risk is temporal aliasing: a brief but decisive moment
(the instant a diver leaves the board, the top of a single jumping-jack) can fall
between two sampled frames and be missed entirely, and a fast, high-frequency action
sampled at only 16 points can look ambiguous. It hurts most for long clips (16 frames
cover a smaller fraction of the motion) and for actions defined by a short key event
rather than a sustained pattern. For this subset the actions are mostly periodic and
sustained (strokes, reps), so uniform 16-frame sampling is a reasonable fit — but it is
a modelling assumption worth stating, and denser or event-aware sampling would be the
first thing to try if a particular class underperformed.
