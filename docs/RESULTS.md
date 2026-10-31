# Results — FrameSight

The full evaluation of FrameSight, a CNN + BiLSTM + attention action classifier on a
9-class subset of UCF101. Per-phase detail, code walkthroughs, and interview questions
live in `docs/phases/`; this file is the consolidated story.

## 1. Setup

- **Data:** 9-class UCF101 subset (gym + swimming actions), 457 clips, author-style split
  train 270 / val 90 / test 97. Source and caveats: `docs/DATASET.md`.
- **Features:** each clip reduced to 16 evenly-spaced frames, each passed through a frozen
  ImageNet-pretrained ResNet18 to a 512-d vector, giving a (16, 512) sequence per clip,
  cached to Drive.
- **Model:** bidirectional LSTM (hidden 128) over the frame-feature sequence, a pooling
  step, dropout 0.5, and a linear classifier. Three pooling strategies compared: mean,
  last-state, attention.
- **Training:** Adam (lr 1e-3, weight decay 1e-4), best-validation model selection,
  identical settings across the three pooling variants.

## 2. Headline: pooling comparison

Chance for 9 classes is 0.111.

| Pooling | Best val acc | Test acc |
|---------|--------------|----------|
| mean | 0.944 | **0.866** |
| last | 0.922 | 0.814 |
| attention | **0.956** | 0.845 |

All three are well above chance and within a few points of each other. Attention is best
on validation, mean is best on test, and with only 97 test clips those gaps are within
noise — the three are statistically indistinguishable here. The real driver of the strong
numbers is transfer learning: the frozen ResNet features are highly discriminative for
these visually-distinct actions, so the temporal head has a comparatively easy job. The
value attention adds on *this* data is interpretability, not accuracy.

## 3. Attention saliency

The attention model's per-frame weights were read back as a temporal saliency map, both
averaged per class and on individual clips with their frames. For these trimmed, sustained
actions (strokes, reps), the weighting is spread fairly evenly across the 16 frames rather
than spiking on one moment — which is consistent with, and explains, why mean pooling tied
attention on accuracy: when every frame is about equally informative, a learned weighting
has little to improve on. A flat saliency is not a failure of attention; it is attention
reporting that this task does not need selective focus. (Per-class attention figures and
the exact spread statistic are in `docs/phases/phase-4-attention.md`.)

## 4. Responsible-AI audit

### 4.1 Per-class accuracy and confusion

Overall test accuracy 0.845. Per class:

| Action | Accuracy |
|--------|----------|
| BenchPress, CleanAndJerk, PullUps | 1.000 |
| JumpingJack | 0.900 |
| Diving | 0.857 |
| FrontCrawl, PushUps | 0.800 |
| BreastStroke | 0.769 |
| **Lunges** | **0.500** |

The failures track **visual similarity**, exactly as a per-frame CNN predicts. Lunges, the
worst class, is confused mostly with JumpingJack (both standing, full-body, object-free
exercises). The swimming actions confuse each other (BreastStroke ↔ FrontCrawl ↔ Diving).
The perfect classes are the visually distinctive ones (a bench and barbell, a loaded
overhead lift, hanging from a pull-up bar). This is direct evidence of the pipeline's
central trade-off: the CNN sees appearance, not motion, so actions that look alike
frame-to-frame are where it struggles.

### 4.2 Calibration

ECE was 0.088 (mildly overconfident). Temperature scaling, fit on the validation set,
returned T = 0.916 and **raised** ECE to 0.105 — it made calibration worse. The cause is
small-data noise: the validation set looked slightly underconfident and the fit sharpened
the probabilities, but the test set was overconfident, so sharpening moved it the wrong
way. The honest conclusion is that temperature scaling is not guaranteed to help on tiny
calibration sets, and FrameSight's confidence scores should be treated as only roughly
trustworthy. Accuracy was unchanged by the scaling, as expected.

### 4.3 Out-of-distribution detection

Holding out PushUps and training an 8-class model, the held-out action was presented as
novel at test time. Mean confidence was 0.933 on known actions vs 0.732 on the novel one,
and max-softmax separated them with AUROC 0.871 — a genuinely useful signal, and notably
better than the ~0.59 seen on sensor data in the course's Open-Ended Lab, because strong
appearance features make a novel action look different enough that the model hedges. But a
mean novel confidence of 0.73 is still high, so a naive argmax would confidently mislabel
many novel clips; using this safely needs an explicit confidence threshold and the
abstention it implies.

## 5. How to use it

`predict_action(video_path)` runs the full pipeline on any clip — sample 16 frames →
ResNet features → attention model → predicted action, confidence, full class
probabilities, and the per-frame attention. Video in, action out.

## 6. Current shortcomings

These are the honest limits of what was built — the reasons the numbers above should not
be over-read, and the gaps a serious version would close.

- **Appearance, not motion.** The frozen per-frame CNN encodes what each frame *looks
  like*, not how it moves. This is the root cause of the main failures: Lunges confused
  with JumpingJack, and the three swimming strokes confused with each other, because they
  look alike frame-to-frame and differ mainly in *motion*, which the pipeline never sees
  directly.
- **Very little data.** 270 training and 97 test clips (~11 per class) make every number
  noisy — a few clips flipping moves accuracy by points — and prevent training anything
  larger than a small head. The per-class accuracies and the ECE are indicative, not
  precise.
- **Only 9 classes, trimmed clips.** The model knows 9 actions out of UCF101's 101, and
  expects a short clip that is mostly the action. It does no **temporal localisation**
  (finding *when* an action happens inside a long untrimmed video).
- **Confidence is only roughly honest.** The model is mildly miscalibrated (ECE 0.088),
  and the standard fix (temperature scaling) failed to transfer on this small data, so its
  confidence scores should not be trusted as probabilities without more calibration data.
- **Weak out-of-distribution safety.** It has a useful OOD *signal* (AUROC 0.871) but is
  not a reliable *detector*: it is still fairly confident on unseen actions, so a naive
  deployment would confidently mislabel them.
- **Fixed, uniform frame sampling.** 16 evenly-spaced frames assume the action is spread
  across the clip; a brief, decisive moment can fall between samples and be missed.
- **No real-world robustness testing.** Evaluated only on UCF101-style footage; unusual
  camera angles, lighting, or low quality are untested and would likely degrade it.

## 7. How this could be scaled

Each shortcoming points at a concrete upgrade; the pipeline's structure (features ->
sequence -> head) makes most of them drop-in.

- **More data, more classes.** Point the same pipeline at the full 101-class UCF101, or at
  far larger benchmarks (Kinetics-400/700, Something-Something). The caching and training
  code scale as-is; only the class count and compute budget change.
- **Give it real motion.** Replace or augment the per-frame CNN with a model that sees
  motion: a two-stream network (RGB + optical flow), a 3D CNN (I3D, SlowFast), or a video
  transformer (TimeSformer, VideoMAE). This is the single change most likely to fix the
  appearance-confusable classes.
- **Unfreeze and fine-tune.** With enough data, fine-tune the backbone end-to-end instead
  of freezing it, lifting the accuracy ceiling the frozen features impose.
- **Smarter sampling.** Use denser or adaptive frame sampling, or handle variable-length
  sequences, so brief key moments are not missed.
- **Untrimmed video / localisation.** Add a sliding-window or detection head to find *when*
  actions occur in long videos, turning a clip classifier into a video analyser.
- **Deployment engineering.** Wrap `predict_action` in an API (e.g. FastAPI) behind a small
  web upload page; add a confidence threshold with abstention for OOD safety; optimise for
  speed (smaller backbone, batching, ONNX/TensorRT) for near-real-time or camera-stream use.
- **More rigorous evaluation.** Enforce group-disjoint train/val/test splits, use a larger
  test set and cross-validation, and test robustness across cameras and conditions, so the
  reported numbers carry more weight.

## 8. Honest summary

FrameSight classifies 9 trimmed actions at ~0.85 test accuracy by reusing frozen ImageNet
features and a small temporal head — a clean demonstration that, on visually-distinct
actions with little data, transfer learning carries the result and the temporal model has
a light job. The audit is the mature part of the story: the model fails predictably where
appearance cannot separate actions, its confidence is only roughly honest and resisted a
standard calibration fix on this little data, and it has a useful but not deployable
out-of-distribution signal. Every one of these is reported with the number behind it and
the reason for it, which is the point of the project as much as the accuracy is.
