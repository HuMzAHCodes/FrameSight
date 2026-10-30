# Dataset Datasheet — UCF101 (10-class subset)

A short datasheet for the data this project uses, in the spirit of "Datasheets for
Datasets" (Gebru et al.). It records where the data came from, how it is used here,
and its known limitations, so anyone reusing the project understands the data behind
the numbers.

## Source

- **Dataset:** UCF101 — a widely-used human action recognition benchmark of realistic
  action videos collected from YouTube (Soomro, Zamir & Shah, 2012, UCF Center for
  Research in Computer Vision).
- **Full dataset:** 13,320 clips across 101 action classes, ~6.5 GB.
- **This project uses:** a **10-class subset** (~500 clips), streamed with `remotezip`
  from the Google Cloud mirror used by the TensorFlow "load video data" tutorial:
  `https://storage.googleapis.com/thumos14_files/UCF101_videos.zip`.
  Only the needed clips are extracted; the full archive is never downloaded.

## What is used and how

- **Clips per class:** 50, split train / val / test = 30 / 10 / 10. Total ~500 clips
  (~300 train, ~100 val, ~100 test).
- **Classes:** the first 10 classes in the archive after shuffling (the exact list is
  printed by the download cell and recorded in the results report once fixed).
- **Preprocessing:** each clip is sampled to 16 evenly-spaced frames, each resized to
  224x224 and normalised with ImageNet statistics (required by the pretrained ResNet).
- **Derived artifact:** per-clip frame features of shape (16, 512) from a frozen
  ResNet18, cached to Drive. Training runs on these cached features, not raw video.

## Format

- Raw clips: `.avi`, variable resolution and length, laid out as
  `UCF101_subset/{train,val,test}/<class>/<video>.avi`.
- The filename encodes the class as the third-from-last `_`-separated token
  (`v_ClassName_gXX_cYY.avi`).

## Known limitations and biases

- **Narrow subset.** 10 of 101 classes and 50 of up to ~130 clips per class. Results
  describe this subset, not UCF101 as a whole, and will not transfer to the full
  benchmark without re-training.
- **Source bias.** UCF101 is drawn from YouTube and skews toward clearly-filmed,
  well-centred actions; performance on real-world, cluttered, or low-quality footage
  would be worse and is untested here.
- **Group structure.** Clips of the same action are recorded in groups (`gXX`); clips
  from one group share a background/actor and are not fully independent. The subset
  download shuffles within class but does not enforce group-disjoint splits, so some
  optimism in the val/test numbers is possible. Noted as a limitation in the report.
- **Short, pre-segmented clips.** Each clip already contains one trimmed action;
  the project does not do temporal localisation (finding *when* an action happens in a
  longer video).

## Licensing / use

UCF101 is released for research and educational use by UCF CRCV. This project uses it
for a non-commercial educational portfolio piece, consistent with that intent.

## Provenance note

The Google Cloud mirror is the one distributed with the TensorFlow video tutorial; it
is used here only as a reliable delivery mechanism for the standard UCF101 clips. The
data itself is the original UCF101.
