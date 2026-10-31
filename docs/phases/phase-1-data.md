# Phase 1 — Data

Getting a small UCF101 action-video subset onto Colab, and verifying it. **Status:
done.**

## Goal

Get a few hundred labelled action-video clips onto the Colab machine, laid out as
`train/val/test/<class>/*.avi` so the next phase can turn them into features — without
downloading the full 6.5 GB UCF101 archive.

## Why / concepts

Action recognition needs video, and video is heavy, so the first real engineering
problem of the project is data access, not modelling. The guiding idea is to pull a
small, pre-split subset from a reliable host rather than the whole dataset. The plan
also had to survive a dead source mid-phase (see "What broke"), which is itself the
lesson: external data sources fail, and the pipeline has to adapt.

## Approach (what the cells do)

1. Install `huggingface_hub`, import libraries, mount Drive, set the project directory.
2. `snapshot_download` a pre-made UCF101 subset from the Hugging Face Hub
   (`ambarishnarayan/UCF_101_Subset`, ~214 MB, public, no token).
3. Inspect what actually arrived (it is a packed archive, not loose files), then
   extract `UCF101_subset.tar.gz` to `/content/UCF101_subset` on the local VM.
4. Verify: per-split counts and the class list.

Videos land on the **local VM** (temporary) — fine, because Phase 2 converts them to
features once and caches those to Drive. Raw video does not need to persist.

## Key decisions

- **D-2 (superseded by D-7)** — originally: stream a 10-class subset with `remotezip`
  from the Google Cloud mirror. That mirror now returns 403.
- **D-7** — use the Hugging Face pre-made subset instead. See `DECISIONS.md`.

## What we measured

- **Source:** `ambarishnarayan/UCF_101_Subset` on Hugging Face, a 214 MB `tar.gz`
  containing 457 `.avi` clips in `UCF101_subset/{train,val,test}/<class>/`.
- **Classes (9):** BenchPress, BreastStroke, CleanAndJerk, Diving, FrontCrawl,
  JumpingJack, Lunges, PullUps, PushUps. A coherent gym + swimming set; several are
  visually similar (the three swimming actions; BenchPress / PushUps / PullUps), so
  the task is non-trivial.
- **Counts:** train 270 (30 per class), val 90 (10 per class), test 97 (~11 per class).

## What broke

- **The Google Cloud mirror went 403 Forbidden.** The `storage.googleapis.com/
  thumos14_files/UCF101_videos.zip` URL (used for years by the TensorFlow video
  tutorial) is now locked down, so the original `remotezip` streaming plan failed at
  the first request. Fix: switched to the Hugging Face subset (D-7).
- **The Hugging Face dataset was a webdataset, not loose files.** The initial inspect
  found only a `.gz` and a `.gitattributes`, and no `train/test` folders — the clips
  were packed in `UCF101_subset.tar.gz`. Fix: look inside the archive (457 `.avi`
  members in the expected tree), then extract it. Lesson: inspect a downloaded dataset
  before assuming its layout.

## Interview questions

**Q1. The original plan streamed only the needed clips with `remotezip`; the final
plan downloads a whole 214 MB archive and extracts it. Is that a downgrade?**

In principle `remotezip` is more elegant — it fetches only the members you ask for, so
you never pay for the rest of a large archive. But it depends on the host supporting
HTTP range requests and allowing anonymous access, and the Google mirror stopped doing
so. The Hugging Face archive is only 214 MB (it is already a subset), so downloading
it whole costs little, and in return it is reliable, versioned, and comes pre-split.
The right lesson is that elegance at the mercy of an external host is worth less than a
dependable path; for a subset this small, the full download is the better engineering
choice.

**Q2. The subset has 9 classes with 30 training clips each. What does that scarcity
imply for the modelling choices in later phases?**

With only 270 training clips, training anything large from scratch would overfit
immediately, which is exactly why the design freezes a pretrained CNN and trains only a
small temporal head on top (transfer learning). It also means the validation and test
numbers will be noisy — a few clips moving between right and wrong shifts the percentage
a lot — so results should be read with that uncertainty in mind, and per-class accuracy
on ~11 test clips per class is indicative, not precise. Scarcity is the single biggest
constraint shaping the rest of the project.
