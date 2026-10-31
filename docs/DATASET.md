# Dataset Datasheet — UCF101 (9-class subset)

A short datasheet for the data this project uses, in the spirit of "Datasheets for
Datasets" (Gebru et al.). It records where the data came from, how it is used here,
and its known limitations, so anyone reusing the project understands the data behind
the numbers.

## Source

- **Dataset:** UCF101 — a widely-used human action recognition benchmark of realistic
  action videos collected from YouTube (Soomro, Zamir & Shah, 2012, UCF Center for
  Research in Computer Vision).
- **Full dataset:** 13,320 clips across 101 action classes, ~6.5 GB.
- **This project uses:** a pre-made **9-class subset** (457 clips) downloaded from the
  Hugging Face Hub dataset `ambarishnarayan/UCF_101_Subset` — a 214 MB `tar.gz`,
  public, no token required, already split into train/val/test.
- **Note on the source:** the project originally planned to stream a subset from the
  Google Cloud mirror used by the TensorFlow video tutorial
  (`storage.googleapis.com/thumos14_files/UCF101_videos.zip`), but that mirror now
  returns 403 Forbidden. The Hugging Face subset replaces it (see `DECISIONS.md`, D-7).

## What is used and how

- **Classes (9):** BenchPress, BreastStroke, CleanAndJerk, Diving, FrontCrawl,
  JumpingJack, Lunges, PullUps, PushUps.
- **Counts:** train 270 (30 per class), val 90 (10 per class), test 97 (~11 per class).
  Total 457 clips.
- **Preprocessing:** each clip is sampled to 16 evenly-spaced frames, each resized to
  224x224 and normalised with ImageNet statistics (required by the pretrained ResNet).
- **Derived artifact:** per-clip frame features of shape (16, 512) from a frozen
  ResNet18, cached to Drive. Training runs on these cached features, not raw video.

## Format

- Raw clips: `.avi`, variable resolution and length, laid out as
  `UCF101_subset/{train,val,test}/<class>/<video>.avi`.
- Delivered packed as `UCF101_subset.tar.gz`; extracted once on the Colab VM.

## Known limitations and biases

- **Narrow subset.** 9 of 101 classes, 30 training clips per class. Results describe
  this subset, not UCF101 as a whole, and will not transfer to the full benchmark
  without re-training.
- **Small and imbalanced-ish test set.** ~11 test clips per class means the per-class
  and overall accuracies are noisy — a few clips flipping changes the percentage
  noticeably. Numbers are indicative, not precise.
- **Visually similar classes.** Several classes share appearance (the swimming trio
  BreastStroke / FrontCrawl / Diving; the gym actions BenchPress / PushUps / PullUps),
  so some confusion between them is expected and worth inspecting.
- **Source bias.** UCF101 is drawn from YouTube and skews toward clearly-filmed,
  well-centred actions; performance on real-world, cluttered, or low-quality footage
  would be worse and is untested here.
- **Group structure.** UCF101 clips of one action are recorded in groups (`gXX`)
  sharing a background/actor, so clips within a group are not fully independent. This
  pre-made split is not guaranteed to be group-disjoint across train/val/test, so some
  optimism in the val/test numbers is possible. Noted as a limitation in the report.
- **Short, pre-segmented clips.** Each clip already contains one trimmed action; the
  project does not do temporal localisation (finding *when* an action happens in a
  longer video).

## Licensing / use

UCF101 is released for research and educational use by UCF CRCV. This project uses it
for a non-commercial educational portfolio piece, consistent with that intent. The
Hugging Face dataset is used only as a reliable delivery mechanism; the data is the
original UCF101.
