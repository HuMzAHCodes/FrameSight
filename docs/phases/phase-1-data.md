# Phase 1 — Data

Getting a small UCF101 action-video subset onto Colab, and verifying it. **Status:
done.**

## Goal

Get a few hundred labelled action-video clips onto the Colab machine, laid out as
`train/val/test/<class>/*.avi`, so the next phase can turn them into features — without
downloading the full 6.5 GB UCF101 archive.

## Why / concepts

Action recognition needs video, and video is heavy, so the first real engineering
problem of the project is **data access**, not modelling. A full dataset download would
waste time and storage on a Colab T4, so the strategy is to pull a small, pre-split
subset from a reliable host. This phase also had to survive its original source dying
mid-way (see "What broke"), which is itself the lesson: external data sources fail, and
a pipeline has to adapt rather than assume.

A second idea runs underneath the whole phase: **inspect before you trust.** A download
"succeeding" does not mean the data is in the shape you expect. We verify the layout,
file types, and counts explicitly before any later phase builds on them.

## Code walkthrough

**Setup cell.** Installs `huggingface_hub`, imports the libraries, mounts Google Drive,
and defines three things reused everywhere: `PROJECT_DIR` (a folder on Drive where
anything that must persist is saved), `device` (`"cuda"` on the T4, else CPU), and
`seed_all(seed)` which seeds Python, NumPy and PyTorch together so runs are reproducible.

**Download cell.**

```python
download_root = snapshot_download(
    repo_id="ambarishnarayan/UCF_101_Subset",
    repo_type="dataset",
)
```

`snapshot_download` pulls an entire Hugging Face repository to the local machine and
returns the path to it. `repo_type="dataset"` says this is a dataset repo (not a model).
No token is needed for a public dataset — the "unauthenticated" warning it prints is
harmless (it only means slightly slower downloads). The files land in Colab's Hugging
Face cache, and `download_root` is that path.

**Inspection cell.** Before assuming a folder of videos, we look at what actually
arrived: it counts file extensions across the whole download, and defines
`find_subset_root`, which walks the tree for the folder that directly contains both
`train` and `test`:

```python
def find_subset_root(base):
    for current_dir, subdirs, _ in os.walk(base):
        if "train" in subdirs and "test" in subdirs:
            return current_dir
    return None
```

`os.walk` yields `(current_dir, subdirs, files)` for every folder, so this is just "find
the directory whose children include train/ and test/". On the first run it returned
`None` and the extensions were `{'.gz': 1, '': 1}` — the clips were packed, not loose.
That is exactly the surprise the inspection step exists to catch.

**Look-inside cell.** Since the data was an archive, we opened it with `tarfile` to see
its contents without extracting:

```python
with tarfile.open(f, "r:*") as tar:
    members = tar.getnames()
```

`"r:*"` lets `tarfile` auto-detect the compression; `getnames()` lists the archive's
internal paths. This revealed `UCF101_subset.tar.gz` holding 457 `.avi` files in
`UCF101_subset/{train,val,test}/<class>/` — the layout we wanted, just compressed.

**Extract cell.**

```python
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(EXTRACT_ROOT, filter="data")
```

`"r:gz"` opens the gzip-compressed tar; `extractall` unpacks it; `filter="data"` is the
safe extraction mode (Python 3.12+) that refuses members with absolute paths or symlinks
pointing outside the target. We extract to the local VM (`/content`), not Drive, because
the raw videos are temporary — Phase 2 turns them into features and only the features
need to persist.

**Verify cell.** Walks `train/val/test`, counts the `.avi` files per class, and prints
the class list and per-class counts, confirming the real numbers before moving on.

## Key decisions

- **D-2 (superseded by D-7)** — originally: stream a 10-class subset with `remotezip`
  from the Google Cloud mirror. That mirror now returns 403.
- **D-7** — use the Hugging Face pre-made subset instead. See `DECISIONS.md`.

## What we measured

- **Source:** `ambarishnarayan/UCF_101_Subset` on Hugging Face, a 214 MB `tar.gz`
  containing 457 `.avi` clips in `UCF101_subset/{train,val,test}/<class>/`.
- **Classes (9):** BenchPress, BreastStroke, CleanAndJerk, Diving, FrontCrawl,
  JumpingJack, Lunges, PullUps, PushUps.
- **Counts:** train 270 (30 per class), val 90 (10 per class), test 97 (~11 per class).

## What broke

- **The Google Cloud mirror went 403 Forbidden.** The `storage.googleapis.com/
  thumos14_files/UCF101_videos.zip` URL — used for years by the TensorFlow video
  tutorial — is now locked down, so the original `remotezip` streaming plan failed at the
  very first request. Fix: switched to the Hugging Face subset (D-7).
- **The Hugging Face dataset was a packed archive, not loose files.** The first inspect
  found only a `.gz` and a `.gitattributes`, with no `train/test` folders, because the
  clips were inside `UCF101_subset.tar.gz`. Fix: look inside the archive, then extract.
  Lesson: inspect a downloaded dataset before assuming its layout.

## Interview questions

**Q1. The original plan streamed only the needed clips with `remotezip`; the final plan
downloads a whole 214 MB archive and extracts it. Is that a downgrade?**

In principle `remotezip` is more elegant — it reads a remote zip's index and fetches
only the members you ask for, so you never pay for the rest of a large archive. But it
depends on the host supporting HTTP range requests and allowing anonymous access, and the
Google mirror stopped doing both. The Hugging Face archive is only 214 MB (it is already
a subset), so downloading it whole costs little, and in return it is reliable, versioned,
and pre-split. The real lesson is that elegance at the mercy of an external host is worth
less than a dependable path; for a subset this small, the full download is the better
choice.

**Q2. The subset has 9 classes with 30 training clips each. What does that scarcity imply
for the modelling choices in later phases?**

With only 270 training clips, training anything large from scratch would overfit
immediately, which is exactly why the design freezes a pretrained CNN and trains only a
small temporal head on top (transfer learning). It also means validation and test numbers
are noisy — a few clips moving between right and wrong shifts the percentage a lot — so
results are indicative, not precise. Scarcity is the single biggest constraint shaping
the rest of the project.

**Q3. Why extract the videos to the local VM but save features to Drive? What principle
decides where each artifact lives?**

Persist only what is expensive to recreate and needed later. The raw videos are cheap to
re-download and not needed once features exist, so they live on the ephemeral VM and may
vanish when the session ends. The frame features are expensive — they require decoding
every video and running a CNN — and every later phase depends on them, so they go to
Drive, which survives session resets. Matching an artifact's storage to its cost and
lifetime is the persistence discipline used throughout the project.
