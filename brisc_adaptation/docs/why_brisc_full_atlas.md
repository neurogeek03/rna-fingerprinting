# Why BRISC + the full Perturb-seq atlas (vs subsetting for R/Seurat)

New to the method? Read [`pipeline_overview.md`](pipeline_overview.md) first.

Written 2026-10-14. Rough numbers come from the Trillium smoke test on one file (`WB8588_1_1.h5ad`):
62.9k QC cells, 22.4k controls, 1,947 candidate perturbations, about 20 cells per perturbation, 12,306
expression-selected features, and 170 of 1,736 eligible perturbations kept. That file is about 1% of
the 6.1M QC cells. Scaling up gives **about 2,000 cells per perturbation on the full atlas**, versus
**about 150** for a downsample R could handle (about 500k cells).

## Statistical consequences

| Dimension | Downsample (~500k, R) | Full atlas (6.1M, BRISC) | Scaling argument |
|---|---|---|---|
| Cells per perturbation | ~150 | ~2,000 | linear in n |
| Wilcoxon filter (p<0.05, and at least 2 of the top-10 features DE) | Misses modest effects. Only 170 of 1,736 passed at about 20 cells. | Most real effects should pass | z ~ effect·√n_eff. The control reference is fixed at 1,000 cells, so the gain is about 2.3x the z, not 3.6x |
| Rank-1 PCA stability (12k features) | Detection threshold √(features/cells) ≈ 9 | Threshold ≈ 2.5 | Spiked-covariance (BBP) threshold |
| MAD ≥ 10 outlier removal | MAD relative error ~10% (26% at 20 cells). Removals push perturbations toward `min_cells=10`. | ~3% | ~1.17/√n |
| `dists` means/SDs (classify() calibration) | Standard error ~0.08σ | Standard error ~0.02σ | 1/√n |
| Control baseline (30-PC PCA) | ~150k controls. Rare subclasses have only tens of cells. | ~2.2M controls. Every subclass and region is represented. | Smallest gain of all the steps (see below) |
| Cell-type/region coverage | Perturbations in rare types fall below `min_cells` | Full coverage | — |
| Reproducibility | Results depend on which subset was sampled (seed, stratification) | Deterministic over all cells. No sampling choices to justify. | — |

## Why each step needs cells

Each subsection covers what the step does, why `fit()` has it, and how the cell count enters.
n_p is the number of cells for one perturbation: about 20 in the smoke test, 150 in a 500k
downsample and 2,000 on the full atlas. k = 12,306 features. Code refs are to
[`R/fit.R`](../../R/fit.R).

### Wilcoxon tests (fit.R:61-65, 96-99)

**Why they exist.** Rank-1 PCA returns a direction but not its sign: Lambda and −Lambda fit equally
well. PCA also has no notion of significance, so it returns a "fingerprint" even for a perturbation
that does nothing. The Wilcoxon tests compare the perturbation's residuals with 1,000 control
residuals, and they have three jobs:

1. **Orient the sign.** A per-feature rank-sum statistic gives the direction of each gene's shift
   relative to controls. Lambda is flipped so that it is anti-correlated with the control-group
   statistic (fit.R:90-93). With this convention, a positive Lambda entry means the gene goes up in
   perturbed cells.
2. **Choose among outlier iterations.** Each pass of the MAD loop produces a candidate Lambda. The
   one most correlated with the Wilcoxon statistics wins (fit.R:84), because it best matches the
   gene-by-gene differences from controls.
3. **Filter.** Two tests must both pass: the cell scores (ell) differ from controls projected onto
   Lambda (p < 0.05), and at least 2 of the 10 top-weighted genes are individually DE (p < 0.05).
   Without this filter the dictionary fills up with noise fingerprints. Those compete in SuSiE and
   spread the probability away from the true matches.

**How n enters.** A rank-sum z-score grows as effect × √n_eff, where
n_eff = n_p · n_ref / (n_p + n_ref) and n_ref = 1,000 control residuals. The fixed 1,000-cell
reference caps the gain:

| n_p | n_eff | z relative to the 150-cell downsample |
|---|---|---|
| 20 (smoke test) | 20 | 0.39x |
| 150 (downsample) | 130 | 1x |
| 2,000 (full atlas) | 667 | **2.3x** |
| 2,000, with the reference raised to 10,000 | 1,667 | 3.6x |

Only 170 of 1,736 eligible perturbations passed in the smoke test. At 2.3x the z, an effect at the
edge of detection in the downsample (z ≈ 2) reaches z ≈ 4.5. The per-gene DE test in criterion (ii)
gains in the same way, so more of the top-10 genes clear p < 0.05. Raising the 1,000-cell reference
costs very little now that the controls are in memory anyway. It is an easy extra lever, but it
needs an R-parity decision because R hard-codes 1,000.

### Control baseline PCA (fit.R:19-29)

**Why it exists.** Most variation among cells is cell type, region and state, not perturbation. If
this variation were not removed, every perturbation's rank-1 PCA would just find "excitatory vs
inhibitory neuron". The 30 control PCs model this baseline, and subtracting them leaves only what
the perturbation adds.

**How n enters.** This step gains the least from more cells. With 150k controls, k/n_ctrl is already
below 0.1, so the top PCs are estimated well either way. The real gains are smaller:

- `ctrl.mus` and the PC directions represent the true cell-type mix of the atlas, not one sample of it.
- Subclasses with only tens of controls in a subsample hardly affect the PCs. Perturbed cells of
  those types then keep their cell-type signal in the residual, where it can be mistaken for
  perturbation signal.

The binding limit here is the number of PCs (30 for about 300 subclasses), not the number of cells.
See the "one pooled baseline" downside below.

### Rank-1 PCA → Lambda (fit.R:55, 74)

**Why it exists.** It summarizes a perturbation's residuals (n_p cells × k genes) as one gene
direction, the fingerprint. This is the dictionary entry.

**How n enters.** Random-matrix theory (the BBP phase transition) gives a sharp threshold. Take a
signal of strength θ, measured as the squared length of the mean shift in per-gene noise-SD units.
For example, 50 genes shifted by 0.5 SD gives θ ≈ 12.5. The estimated Lambda carries **no
information** about the true direction unless θ > √(k/n_p). Above that threshold, the squared cosine
between the estimate and the truth is (1 − γ/θ²) / (1 + γ/θ), with γ = k/n_p.

| n_p | √(k/n_p), the detection threshold | cos(Lambda, truth) at θ = 12 |
|---|---|---|
| 20 | 24.8 | 0 (below threshold) |
| 150 | 9.1 | 0.23 |
| 2,000 | 2.5 | **0.80** |

A moderate perturbation that is undetectable at 20 cells and mostly noise at 150 becomes a
recognizable fingerprint at 2,000 cells. This is the largest single gain from using the full atlas.
(Selecting fewer features lowers k, which is a second lever, but it discards genes.)

### MAD outlier loop (fit.R:57-81)

**Why it exists.** SVD minimizes squared error, so a few extreme cells can capture the whole rank-1
fit. Doublets, damaged cells and rare strong responders all do this, and the result is a Lambda
that describes those cells rather than the perturbation. Cells whose score sits ≥ 10 MADs from the
median are dropped and the fit is redone. The Wilcoxon correlation then chooses the best iteration.

**How n enters.** In three ways:

- **MAD precision.** With `constant = 1`, the relative standard error of the MAD is about 1.17/√n:
  26% at 20 cells, 10% at 150, 3% at 2,000. At small n an underestimated MAD flags normal cells as
  outliers.
- **Leverage.** With few cells, one cell is a large share of the total variance, so a single cell
  captures the rank-1 fit more often. Heavy-tailed scores trigger more removal rounds.
- **Attrition.** Each removal costs more at small n. At 20 cells, two rounds can push a perturbation
  below `min_cells = 10`, and the loop then stops early with a poor Lambda. At 2,000 cells, removing
  a few cells changes nothing.

### `dists` → classify() calibration (fit.R:103-108; classify.R:167-178)

**Why it exists.** Lambda is an estimate. `dists` stores, for every gene, the mean and SD of that
estimate. `classify()` draws 50 resampled dictionaries from N(mean, SD) and recomputes the SuSiE
probabilities on each. A fingerprint whose entries are poorly known therefore gets less confident
assignments, and correlated fingerprints share the probability more evenly.

**How n enters.** sds = √(σ² / Σell²), and Σell² grows in proportion to n_p, so the SD shrinks as
1/√n_p: about 0.08σ at 150 cells and 0.02σ at 2,000. With wide SDs, the 50 resamples scatter, the
credible sets widen and more query cells end up "unassigned". The calibration is honest about the
uncertainty, but that uncertainty comes from too few cells, not from the biology.

## Engineering and fidelity

| | R/Seurat (subset) | BRISC port (full) |
|---|---|---|
| Memory | Dense SCTransform `scale.data` ≈ 240 GB at 6.1M × 5k features (float64), which forces subsetting | Sparse counts ~196 GB per copy, ~392 GB transient peak. The dense control block in `fit()` is ~110 GB (float32, centered in place). Fits on one Trillium node (~745 GiB). |
| Runtime | `pbmclapply` over cell-level dense data | ~2 min per small file on a 192-core node. Full atlas not yet measured. |
| Normalization | SCTransform Pearson residuals (the reference) | Global z-score of log1pPF. Signed r ≈ 0.87 vs R on the CPA example, with 11–12 of the top-20 genes shared. |

## Downsides of the full-data approach

- **One pooled baseline.** 30 PCs fitted across ~300 subclasses can't capture all within-type
  structure. A perturbation whose cells are enriched in one subclass can partly encode cell-type
  composition in its Lambda. Per-subclass runs would give cleaner baselines, but each would have less power.
- **The filter does not tighten as n grows.** At ~2,000 cells, p<0.05 is easy to pass, so weak or
  confounded signals can get through: DNA-damage response, transduction bias in the `Negative`
  (no-guide) controls. Consider an effect-size or FDR threshold. The Wilcoxon control reference
  stays fixed at 1,000 cells, which caps the power gain (see the Wilcoxon section above).
- **Not SCTransform.** At r ≈ 0.87, fingerprints are close to published R dictionaries but not
  interchangeable with them. Analytic Pearson residuals are deferred work, expected to reach ~0.95
  (see `../future_todos.md`).
- **Still heavy.** It needs a full large-memory node. Survival counts and runtime for the full atlas
  are projections until that run happens.

## Bottom line

Cell count is the binding constraint. The biggest gain is rank-1 PCA: moderate perturbations cross
the detection threshold and become usable fingerprints. Wilcoxon power, MAD robustness and `dists`
precision all improve roughly with √n, and the baseline gains rare-subclass representation. Subsetting cuts n by about 10x. BRISC removes the dense-matrix memory wall, so every
cell can be used. The costs are an approximate normalization and a single pooled baseline, and the
filter thresholds should be re-tuned for the higher power.
