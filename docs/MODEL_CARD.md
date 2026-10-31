# Model Card — FrameSight

A short, standardised summary of what FrameSight is, how it performs, and where it should
and should not be used. Full results: `docs/RESULTS.md`.

## Model

A video action classifier built from three parts: a frozen, ImageNet-pretrained
**ResNet18** that turns each sampled frame into a 512-d feature; a **bidirectional LSTM**
(hidden 128) that reads the sequence of 16 frame features; and an **attention pooling**
layer plus a linear classifier that produces the action label. Only the LSTM, attention,
and classifier are trained (~hundreds of thousands of parameters); the CNN is frozen.

## Intended use

Educational / research classification of short, pre-trimmed video clips into one of 9
actions: BenchPress, BreastStroke, CleanAndJerk, Diving, FrontCrawl, JumpingJack, Lunges,
PullUps, PushUps. A portfolio demonstration of a CNN + sequence + attention pipeline and
of a responsible-AI audit.

## Out-of-scope use

Any consequential decision (safety, health, surveillance). Actions outside the 9 trained
classes (the model will confidently pick one of the 9 — see OOD below). Untrimmed or
long-form video, or temporal localisation (finding *when* an action occurs). Populations,
camera conditions, or video quality unlike UCF101's YouTube clips.

## Training & evaluation data

9-class subset of UCF101, 457 clips, split train 270 / val 90 / test 97, from the Hugging
Face dataset `ambarishnarayan/UCF_101_Subset`. Clips are sampled to 16 frames and encoded
by the frozen ResNet18. Known limitations (narrow subset, tiny noisy test set, visually
similar classes, possible group leakage across splits) are in `docs/DATASET.md`.

## Performance

**Aggregate.** Test accuracy 0.845 (attention model); mean-pooling variant 0.866. Chance
is 0.111.

**Per-class.** Perfect on the visually-distinctive actions (BenchPress, CleanAndJerk,
PullUps = 1.00); weakest on **Lunges = 0.50**, confused mainly with JumpingJack. The
swimming actions confuse one another. Failures track appearance similarity, because the
frozen CNN encodes appearance, not motion. Per-class numbers rest on ~11 test clips each,
so they are indicative, not precise.

## Calibration

Expected Calibration Error 0.088 (mildly overconfident). Temperature scaling, fit on the
90-clip validation set, **did not improve** calibration on test (ECE rose to 0.105),
because the fix did not transfer across the small, noisy val/test split. Confidence scores
should therefore be treated as only roughly trustworthy; do not threshold on them without
re-validating on more data.

## Out-of-distribution behaviour

On a held-out action (PushUps) treated as novel, the model was less confident than on known
actions (mean 0.73 vs 0.93), and max-softmax separated novel from known with AUROC 0.871 —
a useful signal (helped by strong appearance features), but not a reliable detector: mean
novel confidence is still high, so a plain argmax will confidently mislabel many novel
clips. Safe use requires an explicit confidence threshold and abstention.

## Known limitations

- Fails on visually-similar actions (Lunges/JumpingJack; the swimming trio) — no motion
  features, only per-frame appearance.
- Confidence is only roughly calibrated and resisted a standard fix on this little data.
- Recognises only the 9 trained actions; flags novel ones only weakly.
- Validated on a small, trimmed, narrow subset; numbers are noisy and will not transfer to
  full UCF101 or real-world video without re-training.

## Ethical & safety considerations

Not suitable for surveillance, monitoring, or any decision affecting people. The per-class
disparity (Lunges at 0.50) and the OOD weakness mean naive deployment would fail silently on
confusable or unfamiliar actions. The model is a learning artifact; treat its outputs as
illustrative, not authoritative.
