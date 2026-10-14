# RNA fingerprinting in plain terms (and where BRISC fits)

Start here if you have not seen the method before. This page explains what the pipeline does and why
it is so memory-hungry. [`why_brisc_full_atlas.md`](why_brisc_full_atlas.md) gives the case for
BRISC, and [`fit_schematic.md`](fit_schematic.md) has the line-by-line detail.

## The question

You have a transcriptional response you want to explain, for example disease cells compared with
control cells. You want to know **which gene knockouts produce a response that looks like it.**

Perturb-seq gives the reference. Each of about 2,000 genes was knocked out in mouse brain cells, and
each cell was sequenced. From those cells we learn one **fingerprint** per knockout: the direction
in gene-expression space that the knockout pushes cells. We then ask which fingerprints, combined,
best explain the query response.

## Two phases

![Pipeline schematic: Phase A and Phase B](pipeline_schematic.svg)

| Phase | What it does | Data size | Where it runs |
|---|---|---|---|
| **A. Learn the dictionary** (`LearnDictionary` → `fit()`) | One fingerprint per knockout, plus its uncertainty | Whole atlas: ~6.1M QC cells | **Python / BRISC** (this port) |
| **B. Fingerprint a query** (`FingerprintCells` → `classify()`) | Match query cells to the dictionary | ~100k cells | R (unchanged) |

Phase A is the heavy part. Phase B only needs the finished dictionary, a small `.rds` file of
features × knockouts.

## Phase A in five steps

1. **Put every cell on the same scale.** Raw counts depend on sequencing depth and on how highly a
   gene is expressed. Normalization (log1pPF, then a per-gene z-score) makes 1 unit mean the same
   thing for every gene. (R uses SCTransform for this step.)
2. **Learn what normal variation looks like.** Control cells (no knockout, `Negative` guides) still
   vary a lot, mostly by cell type, brain region and cell state. A 30-component PCA on the controls
   captures this baseline variation.
3. **Remove the baseline from each knockout's cells.** Each knocked-out cell is projected onto the
   control PCs, and the part they explain is subtracted. The leftover **residual** is the part of
   that cell that normal variation cannot explain.
4. **Find the main leftover direction.** A rank-1 PCA on one knockout's residuals gives the single
   gene direction that best explains them. That direction is the fingerprint (**Lambda**), and each
   cell's score along it is **ell**. A few cells can dominate this fit, for example doublets or very
   strong responders. Those outlier cells are dropped iteratively (the **MAD** rule) and the fit is
   redone.
5. **Keep only real fingerprints and record their uncertainty.** Wilcoxon tests against control
   residuals do three jobs. They fix the fingerprint's sign, which PCA leaves arbitrary. They choose
   the best candidate among the outlier-removal iterations. They drop knockouts that look no
   different from controls. For each kept knockout we store a mean and SD per gene (**dists**),
   which say how precisely each entry of the fingerprint is known.

## Phase B in three steps

1. Remove the same control baseline from the query cells. This uses the same kind of control PCA as
   Phase A, run on the query object's own controls.
2. Regress each query residual (one cell or one pooled group) on its top 100 most-correlated
   fingerprints with **SuSiE**. SuSiE returns a small set of fingerprints that explain the residual,
   as credible sets with probabilities.
3. **Calibrate.** Draw 50 resampled dictionaries from the `dists` (mean ± SD per gene), then recompute
   the probabilities. A fingerprint whose entries are poorly known gets less confident assignments.

## Where the memory wall is

Step A1 in R (SCTransform) produces `scale.data`, a **dense** matrix with one float per gene per cell.
Single-cell counts are about 90% zeros and are cheap to store sparse, but scaled values are almost
never zero. At 6.1M cells × 5k features in float64 that is about 240 GB before `fit()` copies any of
it, so R can only run on a subsample (about 500k cells).

BRISC keeps the counts **sparse** and stores only a per-gene mean and SD. `fit()` builds a dense,
scaled block only for the cells it needs at that moment. That is the control cells once (about
110 GB float32, centered in place), then one knockout at a time (a few MB each). The full atlas then
fits on one large-memory node, and **every step in Phase A sees every cell.**

## Why the cell count matters

Steps A2–A5 all estimate something from cells. Every one of those estimates gets better with more
cells, and the knockouts are the bottleneck: about 150 cells each in a 500k subsample, about 2,000
on the full atlas. [`why_brisc_full_atlas.md`](why_brisc_full_atlas.md#why-each-step-needs-cells)
goes through each step.

## Glossary

| Term | Meaning |
|---|---|
| Control | Cells with a non-targeting guide (`Negative`). They define "normal". |
| Baseline / control PCs | The 30 main axes of variation among controls. |
| Residual | A cell's expression minus what the control PCs predict for it. |
| Lambda / fingerprint | A knockout's characteristic gene direction (one value per feature). |
| ell | Each cell's score along its knockout's Lambda (response strength). |
| MAD | Median absolute deviation, a robust spread measure used to flag outlier cells. |
| dists | A per-gene mean and SD for each Lambda, used to calibrate Phase B. |
| SuSiE | Sparse regression that picks a few fingerprints and reports how confident it is. |
