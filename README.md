# FrameSight — Human Action Recognition on Video (CNN + BiLSTM + Attention)

Recognising human actions from short video clips, in PyTorch, by combining a
pretrained CNN with a recurrent temporal model and an attention layer. The name is the
idea: FrameSight learns *where to look across the frames* of a clip. The capstone of a
four-group deep-learning curriculum — it uses every piece those groups built.

> Status: in progress. The pipeline is designed and verified; results are filled in as
> each phase completes. See `docs/PROJECT_PROPOSAL.md` for the full plan and
> `docs/DECISIONS.md` for the reasoning behind every choice.

---

## 1. The problem

Action recognition is image classification with a time axis. A single frame is often
ambiguous — a frozen frame of running and of walking can look identical — so the signal
lives in how appearance changes across frames. The task is to label a short clip with
the action it shows (e.g. diving, playing guitar, biking).

## 2. The approach

A single frame tells you *what is in the scene*; the sequence tells you *what is
happening*. The pipeline separates those two jobs:

```
video clip
  -> sample 16 evenly-spaced frames (224x224)
  -> frozen pretrained ResNet18  ->  one 512-d feature per frame   [appearance]
  -> BiLSTM over the 16 features                                   [temporal model]
  -> attention pooling over time (a learned weight per frame)      [focus]
  -> linear classifier -> action label
```

- **Appearance (CNN, transfer learning).** A ResNet18 pretrained on ImageNet, frozen,
  turns each frame into a strong feature vector — no CNN trained from scratch, because
  the dataset is small (the Group 2 transfer-learning lesson).
- **Time (BiLSTM).** A bidirectional LSTM reads the sequence of frame features in both
  directions (the Group 3 sequence-modelling lesson).
- **Focus (attention).** An attention layer learns a weight per frame and pools them,
  so the model emphasises the discriminative moments — and the weights are an
  interpretable saliency over the clip (the Group 4 lesson).

## 3. Dataset

A 10-class subset of **UCF101** (~500 clips), streamed efficiently so the full 6.5 GB
archive is never downloaded. Frame features are extracted once and cached, so training
the temporal model is fast and resumable. Full datasheet: `docs/DATASET.md`.

## 4. Results

> To be completed after Phase 3-5.

- Pooling comparison (mean vs last-state vs attention): _table to follow_
- Attention saliency over frames: _figure to follow_
- Responsible-AI audit (per-class accuracy, calibration): _to follow_

## 5. How it fits the curriculum

| Curriculum piece | Used here as |
|------------------|--------------|
| CNNs + transfer learning (Groups 1-2) | frozen ResNet18 frame-feature extractor |
| RNN / LSTM / GRU (Group 3) | BiLSTM temporal model over frame features |
| Attention (Group 4) | attention pooling over frames + saliency map |
| Responsible-AI audit (Open-Ended Lab) | per-class accuracy and calibration check |

## 6. Repository layout

```
framesight/
  README.md                      <- this file
  <notebook or src>              <- the pipeline (download, features, model, train)
  docs/
    PROJECT_PROPOSAL.md          <- problem, objectives, scope, plan
    DECISIONS.md                 <- every choice + rejected alternatives
    DATASET.md                   <- UCF101 subset datasheet
    RESULTS.md                   <- results / evaluation report (added after training)
    MODEL_CARD.md                <- model card (added after the audit)
```

## 7. Running it

> Full run instructions added once the notebook is finalised. In short: open the
> notebook in Google Colab, set a GPU runtime, mount Drive, and run top to bottom —
> the data streams in, frame features are cached to Drive, and the model trains on
> them.

## 8. Tech

PyTorch · Google Colab (T4) · torchvision (ResNet18) · OpenCV (frame decoding) ·
`remotezip` (streaming the dataset subset).
