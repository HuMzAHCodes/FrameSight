# Phase 5 — Responsible-AI Audit

Looking past the headline accuracy at how the model actually behaves. **Status: done.**

## Goal

Audit the trained attention model on four questions a single accuracy number cannot
answer: which actions does it fail, which does it confuse, is its confidence honest, and
can it tell when it is shown an action it was never trained on. Fix what is cheaply
fixable, and record the rest honestly.

## Why / concepts

The same lesson as the course's Open-Ended Lab, now on the video model: **test accuracy is
a scalar that hides everything that makes a model safe to deploy.** FrameSight scores
0.845 overall, but that average says nothing about per-action behaviour, confidence
honesty, or novel inputs. This phase measures each:

- **Per-class accuracy + confusion.** The average can hide a class the model fails, and a
  confusion matrix shows *which* actions get mistaken for each other. The prediction to
  check: visually-similar classes (the swimming trio BreastStroke / FrontCrawl / Diving;
  the gym actions) should confuse each other more than unrelated ones.
- **Calibration.** A confidence score is only useful if it is truthful. We measure
  Expected Calibration Error (ECE) and try to fix it with temperature scaling.
- **Out-of-distribution (OOD) detection.** A 9-class softmax always picks one of its 9
  classes, even for an action it has never seen. We hold one action out of training and
  check whether the model is at least *less confident* on it.

Everything here is noisy because the test set is tiny (97 clips, ~11 per class), so
per-class numbers and small ECE differences are indicative, not precise — which is itself
part of the honest reporting, and, as it turned out, the cause of the most interesting
finding.

## Code walkthrough

**Audited model.** The audit uses the attention model trained in Phase 3
(`trained_models["attention"]`). Its logits on the val and test sets are computed once up
front.

**Per-class accuracy + confusion matrix.** For each class we take its test clips and
compute the fraction predicted correctly. `sklearn.metrics.confusion_matrix` builds the
9x9 table of true-vs-predicted counts, plotted as an annotated heatmap — off-diagonal
bright cells are the confusions.

**Calibration.** `compute_ece` bins predictions by confidence and sums the gap between
confidence and accuracy in each bin, weighted by bin size. The reliability diagram plots
accuracy against confidence (perfect calibration is the diagonal). **Temperature scaling**
fits one scalar T by minimising cross-entropy on the *validation* logits, then divides the
test logits by T. Dividing all logits by a constant never changes which is largest, so
accuracy is unchanged — only the confidence magnitudes move.

```python
temperature = torch.ones(1, device=device, requires_grad=True)
optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=60)
def closure():
    optimizer.zero_grad()
    loss = F.cross_entropy(val_logits / temperature, y_val)
    loss.backward(); return loss
optimizer.step(closure)
```

**OOD detection.** PushUps is held out of training; an 8-class model is trained on the
rest (labels remapped to 0-7); at test time PushUps clips are the "novel" set. A good
model should be less confident on them, so the maximum softmax probability should separate
known from novel, measured by AUROC (1.0 perfect, 0.5 chance).

## Key decisions

- Audit the attention model (the project's featured, interpretable model), not the
  marginally-higher mean-pooling one — they are within noise on accuracy.
- Fit temperature on validation, measure ECE on test (no leakage).
- Hold out PushUps for OOD as a near-neighbour (gym-exercise) novelty case.

## What we measured

**Per-class accuracy (overall test accuracy 0.845, chance 0.111):**

| Action | Accuracy | Clips |
|--------|----------|-------|
| BenchPress | 1.000 | 10 |
| CleanAndJerk | 1.000 | 10 |
| PullUps | 1.000 | 10 |
| JumpingJack | 0.900 | 10 |
| Diving | 0.857 | 14 |
| FrontCrawl | 0.800 | 10 |
| PushUps | 0.800 | 10 |
| BreastStroke | 0.769 | 13 |
| **Lunges** | **0.500** | 10 |

**Confusion (the prediction held):** the mistakes cluster among look-alikes. The swimming
actions confuse each other — BreastStroke -> FrontCrawl (3), Diving -> FrontCrawl (2),
FrontCrawl -> BreastStroke (2). The worst class, **Lunges (0.500), is confused mostly with
JumpingJack (4 of 10)** — both are standing, full-body exercises that look alike frame to
frame. The strong classes (BenchPress, CleanAndJerk, PullUps at 1.000) are the visually
distinctive ones.

**Calibration:** ECE **0.088 before**, fitted temperature **0.916**, ECE **0.105 after** —
temperature scaling made calibration *worse*, with accuracy unchanged (0.845). See "What
broke".

**OOD (held-out PushUps):** mean confidence **known 0.933 vs novel 0.732**, OOD AUROC
**0.871**. The model is clearly less confident on the novel action, and max-softmax
separates known from novel reasonably well — better than a naive expectation.

## What broke

The notable finding is that **temperature scaling increased ECE (0.088 -> 0.105) instead
of reducing it.** The temperature was fit on the 90-clip validation set and came out below
1 (T = 0.916), i.e. the validation set looked slightly *under*confident and the fit
*sharpened* the probabilities. But on the test set the model was mildly *over*confident, so
sharpening pushed it further from calibrated. The cause is the small-data noise flagged up
front: with ~90 validation and ~97 test clips, the two sets' calibration profiles simply
differ, and a temperature tuned on one does not transfer to the other. The honest
conclusion is that **temperature scaling is not guaranteed to help, and on tiny calibration
sets it can hurt** — the right response here is to report the model's confidence as
only roughly trustworthy rather than to ship a fix that did not generalise.

## Interview questions

**Q1. Temperature scaling is a standard, almost foolproof calibration fix, yet here it made
ECE worse. Explain how that happened and what it teaches.**

Temperature scaling fits a single scalar on a held-out set to best match confidence to
accuracy, and with enough calibration data it reliably helps because one parameter cannot
overfit. The failure here is not the method but the data: the temperature was fit on ~90
validation clips, which looked slightly underconfident, so the fit sharpened the
probabilities (T < 1); but the ~97-clip test set was mildly overconfident, so sharpening
moved it the wrong way and ECE rose. Both sets are too small for their calibration profiles
to agree, so a fix tuned on one does not transfer to the other. The lesson is that
calibration, like accuracy, is estimated from a sample and inherits that sample's noise —
on small data even a well-behaved correction can fail to generalise, and the responsible
move is to measure the fix on held-out data (as we did), notice it did not help, and report
confidence as only roughly trustworthy rather than claim a correction that did not hold.

**Q2. Lunges was the worst class (0.500), confused mostly with JumpingJack, while BenchPress
and PullUps were perfect. What explains the pattern, and what would you try to fix Lunges?**

The pattern tracks visual distinctiveness at the frame level, which is exactly what the
frozen ResNet features encode. BenchPress (a bench and barbell), PullUps (hanging from a
bar), CleanAndJerk (a loaded barbell overhead) have unmistakable objects and poses, so they
are near-perfect. Lunges and JumpingJack are both a standing person doing a full-body
bodyweight movement in an open space, with no distinctive object and similar poses across
many frames, so their ResNet features overlap and the classifier confuses them. The swimming
trio confuses for the same reason — water and a horizontal swimmer look alike regardless of
stroke. To fix Lunges specifically, the most promising levers are the ones that add the
information appearance lacks: sample more frames or use optical-flow / motion features so the
*dynamics* (a lunge's forward-step-and-dip versus a jumping-jack's symmetric jump) become
available, since those actions differ in motion far more than in any single frame. More
training clips for the confusable pair would also help, but the core issue is that per-frame
appearance cannot separate them.

**Q3. OOD detection scored AUROC 0.871 — decent — and the model was less confident on novel
PushUps (0.73) than on known actions (0.93). Does that mean FrameSight can be trusted to
flag actions it was not trained on?**

Not on its own. AUROC 0.871 means the *ranking* is good — novel clips tend to get lower
confidence than known ones — but a mean novel confidence of 0.73 is still high in absolute
terms, so a naive system that just takes the argmax would confidently mislabel many novel
PushUps as some known action. To actually use this, you would set a confidence threshold
(abstain below it), accepting that it will both miss some novel inputs and reject some known
ones, and tune the threshold for the cost you care about. It is also worth noting *why* OOD
worked better here than it did on the sensor data in the course's Open-Ended Lab (AUROC
~0.59 there): strong ImageNet appearance features make a genuinely novel action look
different enough that the model hedges, whereas near-identical sensor signals did not. So the
honest statement is that FrameSight has a useful OOD *signal* but is not a reliable OOD
*detector* without an explicit threshold and the abstention it implies.
