# Project Proposal — FrameSight (Human Action Recognition on Video)

## 1. Problem

Recognising *what a person is doing* from a short video clip — walking, diving,
playing an instrument — is a core computer-vision task with applications in
surveillance, sports analytics, health monitoring, and video search. Unlike image
classification, action recognition is inherently **temporal**: a single frame is
often ambiguous (a frozen frame of "running" and "walking" can look identical), and
the signal lives in how appearance changes across frames.

## 2. Motivation

This project is the capstone of a four-group deep-learning curriculum. Each group
built one piece of the machinery this task needs, on simpler data:

- **Groups 1-2 (CNNs, transfer learning):** extracting strong visual features, and
  reusing a pretrained network when data is limited.
- **Group 3 (RNN / LSTM / GRU):** modelling a sequence over time.
- **Group 4 (attention):** letting a model focus on the informative parts of a
  sequence, with an interpretable weight per position.
- **Open-Ended Lab (responsible AI):** evaluating a model beyond a single accuracy
  number — per-class behaviour, calibration, honest limitations.

Human action recognition on video is the natural problem that requires *all* of these
at once, which is why it is the capstone.

## 3. Objectives

1. Build an end-to-end video action classifier in PyTorch that combines a CNN
   feature extractor, a recurrent temporal model, and an attention pooling layer.
2. Compare temporal-pooling strategies (mean, last-state, attention) under identical
   conditions, to measure what attention actually buys.
3. Produce an interpretable attention saliency over the frames of a clip.
4. Audit the final model responsibly (per-class accuracy, calibration) rather than
   reporting one number.

## 4. Scope

- **In scope:** a 10-class subset of UCF101 (~500 clips), a frozen pretrained CNN
  backbone, a BiLSTM + attention classifier trained on cached frame features, and the
  comparisons and audit above.
- **Out of scope:** training the CNN backbone from scratch; full 101-class UCF101;
  optical-flow / two-stream architectures; real-time inference; deployment.

## 5. Methodology (pipeline)

```
video clip
  -> sample 16 evenly-spaced frames, resize to 224x224
  -> frozen pretrained ResNet18 -> 512-dim feature per frame
  -> sequence of frame features (16 x 512), cached to Drive
  -> BiLSTM over the sequence
  -> attention pooling over time (learned weight per frame)
  -> linear classifier -> action label
```

Frame features are extracted **once** and cached, so training the temporal model is
fast and resumable — the key efficiency decision (see `DECISIONS.md`).

## 6. Deliverables

- The trained model and the full pipeline (code repo).
- A rich public `README.md` explaining the project end to end.
- Supporting docs in `docs/`: this proposal, `DECISIONS.md`, `DATASET.md`, a results /
  evaluation report, and a model card.
- An attention-saliency visualisation and a pooling-strategy comparison table.

## 7. Plan (by phase, not date)

| Phase | Work |
|-------|------|
| 1 | Data: stream the UCF101 10-class subset; verify splits |
| 2 | Feature extraction: frozen ResNet18 over sampled frames; cache to Drive |
| 3 | Model: BiLSTM + attention pooling; train; compare mean / last / attention |
| 4 | Interpretability: attention saliency over frames |
| 5 | Responsible-AI audit: per-class accuracy, calibration |
| 6 | Write-up: results report, model card, finalise README |

## 8. Success criteria

A working end-to-end classifier whose attention pooling is at least competitive with
the pooling baselines, an interpretable per-frame attention map, and an honest audit —
with every significant choice recorded, with its rejected alternatives, in
`DECISIONS.md`.
