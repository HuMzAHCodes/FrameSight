# FrameSight — Human Action Recognition on Video (CNN + BiLSTM + Attention)

Recognising human actions from short video clips, in PyTorch, by combining a
pretrained CNN with a recurrent temporal model and an attention layer. The name is the
idea: FrameSight learns *where to look across the frames* of a clip. The capstone of a
four-group deep-learning curriculum — it uses every piece those groups built.

> Status: complete. Built in six phases (data -> features -> model -> attention ->
> audit -> write-up), each with its own record in `docs/phases/`. See
> `docs/PROJECT_PROPOSAL.md` for the plan, `docs/DECISIONS.md` for the reasoning behind
> every choice, `docs/RESULTS.md` for the full results, and `docs/MODEL_CARD.md` for the
> deployment summary.

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

A **9-class subset of UCF101** (457 clips: 270 train / 90 val / 97 test), pulled from a
small pre-made Hugging Face copy so the full 6.5 GB archive is never downloaded. The
classes are gym and swimming actions — BenchPress, BreastStroke, CleanAndJerk, Diving,
FrontCrawl, JumpingJack, Lunges, PullUps, PushUps — and several are visually similar,
which makes the task non-trivial. Frame features are extracted once and cached, so
training the temporal model is fast and resumable. Full datasheet: `docs/DATASET.md`.

## 4. Results

Full detail in `docs/RESULTS.md`; the headlines:

**Pooling comparison** (9-way, chance 0.111):

| Pooling | Best val acc | Test acc |
|---------|--------------|----------|
| mean | 0.944 | 0.866 |
| last | 0.922 | 0.814 |
| attention | 0.956 | 0.845 |

All three are strong and within noise of each other on this small test set (97 clips) —
the frozen ResNet features do most of the work, and attention's value here is
interpretability rather than a decisive accuracy gain.

**Responsible-AI audit** (`docs/MODEL_CARD.md`):
- **Per-class:** overall 0.845; strongest on visually-distinctive actions (BenchPress,
  CleanAndJerk, PullUps all 1.00), weakest on Lunges (0.50), which is confused with
  JumpingJack. The swimming actions confuse each other — failures track appearance
  similarity, exactly as expected from a per-frame CNN.
- **Calibration:** ECE 0.088; temperature scaling *did not help* here (0.105) — tuned on
  a tiny validation set, it failed to transfer, which is reported honestly rather than
  hidden.
- **Out-of-distribution:** on a held-out action (PushUps), max-softmax separated novel
  from known with AUROC 0.871 — a useful signal, though not a reliable detector without a
  confidence threshold.

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
  framesight.ipynb               <- the full pipeline (data -> features -> model -> attention -> audit -> demo)
  docs/
    PROJECT_PROPOSAL.md          <- problem, objectives, scope, plan
    DECISIONS.md                 <- every choice + rejected alternatives
    DATASET.md                   <- UCF101 subset datasheet
    RESULTS.md                   <- full results / evaluation report
    MODEL_CARD.md                <- model card (intended use, performance, limitations)
    phases/                      <- one record per phase, with code walkthroughs + interview questions
      phase-1-data.md ... phase-6-writeup.md
```

## 7. Running it

Open `framesight.ipynb` in Google Colab, set a GPU runtime (T4), and run top to bottom.
The notebook downloads the dataset subset from Hugging Face, extracts 16-frame features
with a frozen ResNet18 and caches them to Google Drive, trains the BiLSTM + attention
model, and runs the attention visualisation and the responsible-AI audit. Because
features are cached, re-runs are fast and survive session timeouts. The final
`predict_action(video_path)` cell classifies any clip you give it.

## 8. Tech

PyTorch · Google Colab (T4) · torchvision (ResNet18) · OpenCV (frame decoding) ·
`huggingface_hub` (dataset download) · scikit-learn (audit metrics).
