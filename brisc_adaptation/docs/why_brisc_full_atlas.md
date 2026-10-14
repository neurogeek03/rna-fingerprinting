# Why BRISC + the full Perturb-seq atlas (per-subclass dictionaries)

New to the method? Read [`pipeline_overview.md`](pipeline_overview.md) first.

Rewritten 2026-10-14 for per-subclass dictionaries (`../decisions_log.md` §9). The counts are real.
They come from the obs of all 14 h5ads, via
[`tabulate_subclass_counts.py`](../../python/scripts/tabulate_subclass_counts.py). The output is in
[`../subclass_counts/report.txt`](../subclass_counts/report.txt), and class level in
[`../subclass_counts/by_class/`](../subclass_counts/by_class/). Labels are collapsed the way the
pipeline collapses them. An "eligible" pair has ≥ 10 cells (`min_cells`) in a subclass with
≥ 30 controls. k = 12,306 features.

## In short

- **Statistics: the full atlas matters more than before.** The median eligible knockout × subclass
  pair has **20 cells** (q90 70, max 440). That is the low-power regime of the smoke test, where only
  about 10% of perturbations passed the filter. A 10x downsample cuts eligible pairs from **75,116 to
  3,877**.
- **Memory: no longer the reason to use BRISC.** The largest subclass is 495k cells, which is about
  49 GB dense in R. R could run per subclass on full data without downsampling. The case for BRISC is
  now practical (one job, one normalization, speed), not "the only thing that fits".
- **Open design question: subclass may be too fine.** At class level (33 classes), the median pair has
  52 cells and q90 has 229. See [Resolution](#resolution-subclass-vs-class).

## The counts

| | Pooled (one dictionary) | Per class (33) | **Per subclass (320)** |
|---|---|---|---|
| Groups with ≥ 30 controls | — | 29 | **265** (223 after 10x downsample) |
| Controls per group (median) | 2.19M total | 16,607 | **1,768** (35 subclasses have ≥ k) |
| Cells per knockout or pair: median | 1,663 | 52 | **20** |
| Cells per knockout or pair: q90 / max | 2,356 / 3,838 | 229 / 1,105 | **70 / 440** |
| Eligible units | 1,947 knockouts | 33,530 pairs | **75,116 pairs** |
| Eligible after 10x downsample | — | 8,042 (24%) | **3,877 (5%)** |
| Eligible groups per knockout (median) | 1 | 17 | **39** |
| Largest group, dense R `scale.data` (k, float64) | 6.1M cells, ~600 GB | 1.6M cells, 159 GB | **495k cells, 49 GB** |

6.13M QC cells. 5.46M remain after dropping multi-gene combos, of which 2.19M are controls.
There are 1,947 single-gene knockouts.

## Why each step needs cells

n_p is now the number of cells of one knockout *within one subclass*: about 20 at the median, 70 at
q90, 440 at most. A 10x downsample makes the median pair about 2 cells, which is not fittable.
Code refs are to [`R/fit.R`](../../R/fit.R).

### Wilcoxon tests (fit.R:61-65, 96-99)

**Why they exist.** Rank-1 PCA returns a direction but not its sign: Lambda and −Lambda fit equally
well. PCA also has no notion of significance, so it returns a "fingerprint" even for a perturbation
that does nothing. The Wilcoxon tests compare the perturbation's residuals with control residuals,
and they have three jobs:

1. **Orient the sign.** A per-feature rank-sum statistic gives the direction of each gene's shift
   relative to controls. Lambda is flipped to be anti-correlated with the control-group statistic
   (fit.R:90-93), so a positive Lambda entry means the gene goes up in perturbed cells.
2. **Choose among outlier iterations.** Each pass of the MAD loop produces a candidate Lambda. The
   one most correlated with the Wilcoxon statistics wins (fit.R:84).
3. **Filter.** Both tests must pass: the cell scores (ell) differ from controls projected onto Lambda
   (p < 0.05), and at least 2 of the 10 top-weighted genes are individually DE (p < 0.05). Without
   this, noise fingerprints crowd the dictionary and steal SuSiE probability from true matches.

**How n enters.** z grows as effect × √n_eff, where n_eff = n_p · n_ref / (n_p + n_ref) and
n_ref = min(1,000, the subclass's controls). With n_p this small, n_p dominates, and the 1,000-cell
cap rarely matters any more:

| n_p | n_eff (n_ref = 1,000) | z relative to the median pair |
|---|---|---|
| 2 (median pair after 10x downsample) | 2 | not fittable (< `min_cells`) |
| 20 (median pair) | 19.6 | 1x |
| 70 (q90 pair) | 65 | 1.8x |
| 440 (largest pair) | 306 | 4.0x |

The smoke test (one file, pooled, about 20 cells per knockout) kept 170 of 1,736. Most per-subclass
pairs sit at that same n. Expect a low pass rate per pair, and use the smoke test's `summary.tsv` to
measure it.

### Control baseline PCA (fit.R:19-29)

**Why it exists.** Even within one subclass, cells vary by region and state, and that variation is
not caused by the perturbation. The 30 control PCs model this baseline, and subtracting them leaves
only what the perturbation adds. Per-subclass baselines remove cell-type identity from the problem;
30 pooled PCs could not model about 300 subclasses.

**How n enters.** Each subclass's PCA uses only its own controls: a median of 1,768, and only 35 of
320 subclasses have more controls than features (k). Below that, PC directions are noisy. A 10x
downsample drops 42 more subclasses below the 30-control floor (265 → 223) and leaves the median
subclass with about 177 controls.

### Rank-1 PCA → Lambda (fit.R:55, 74)

**Why it exists.** It summarizes one knockout's residuals (n_p cells × k genes) as one gene direction,
the fingerprint.

**How n enters.** Random-matrix theory (the BBP phase transition) sets a hard limit. Let θ be the
signal strength: the squared length of the mean shift in per-gene noise-SD units. For example,
50 genes shifted by 0.5 SD gives θ ≈ 12.5, and 100 genes gives θ ≈ 25. Lambda carries **no
information** about the true direction unless θ > √(k/n_p). Above that, the squared cosine with the
truth is (1 − γ/θ²) / (1 + γ/θ), with γ = k/n_p.

| n_p | Threshold √(k/n_p) | cos(Lambda, truth) at θ = 12 | at θ = 25 |
|---|---|---|---|
| 20 (median subclass pair) | 24.8 | 0 | 0.03 |
| 52 (median class pair) | 15.4 | 0 | 0.24 |
| 70 (q90 subclass pair) | 13.3 | 0 | 0.30 |
| 229 (q90 class pair) | 7.3 | 0.34 | 0.54 |
| 440 (largest subclass pair) | 5.3 | 0.49 | 0.67 |
| 1,663 (median pooled knockout) | 2.7 | 0.77 | 0.87 |

At the median subclass pair, only strong perturbations (θ well above 25) produce a usable
fingerprint. Fewer features is a second lever: at k = 2,000 the threshold at n_p = 20 drops to 10.
But that discards genes, and with them some of the signal.

### MAD outlier loop (fit.R:57-81)

**Why it exists.** SVD minimizes squared error, so a few extreme cells (doublets, damaged cells,
rare strong responders) can capture the whole rank-1 fit. Cells ≥ 10 MADs from the median score are
dropped and the fit is redone.

**How n enters.**
- **Precision.** The MAD's relative error is about 1.17/√n: 26% at 20 cells, 14% at 70, 6% at 440.
- **Leverage.** At 20 cells, one cell is 5% of the data and can capture the fit.
- **Attrition.** At 20 cells, removing a few outliers can push a pair below `min_cells = 10`, and
  the loop then stops early.

### `dists` → classify() calibration (fit.R:103-108; classify.R:167-178)

**Why it exists.** `dists` stores a mean and SD per gene for each Lambda. `classify()` draws 50
dictionaries from N(mean, SD) and recomputes the SuSiE probabilities, so poorly known fingerprints
get less confident matches.

**How n enters.** SD ∝ 1/√n_p: about 0.22σ at 20 cells, 0.12σ at 70 and 0.05σ at 440, compared
with 0.02σ for a pooled knockout. Wide SDs give wider credible sets and more "unassigned" query
cells.

## Engineering: BRISC vs R per subclass

The honest comparison is now **R per subclass on full data vs BRISC per subclass**. Downsampling is
what destroys the statistics, and with per-subclass runs R no longer has to downsample.

| | R/Seurat, per subclass | BRISC port, per subclass |
|---|---|---|
| Memory | Largest subclass 495k cells → dense `scale.data` ≈ 49 GB at k (float64), plus SCTransform's own working memory. Feasible on a large node. | One process: sparse counts ~196 GB per copy, ~400 GB peak during load/normalize. Then one subclass at a time; the largest control block is ~7 GB float32. |
| Workflow | Split the atlas into ~265 objects outside R (a 6.1M-cell Seurat object is itself heavy), then 265 × (SCTransform + `LearnDictionary`) | One job: load and normalize once, loop over subclasses |
| Runtime | 265 SCTransform fits; not measured | Not measured per subclass (pooled: ~2 min per small file on 192 cores) |
| Normalization | SCTransform Pearson residuals, run per subclass (the reference) | log1pPF + per-subclass z-score. Signed r ≈ 0.87 vs R on the CPA example |

At class level the memory argument returns. The largest class is 1.6M cells, ≈ 159 GB dense in R
before SCTransform overhead.

## Resolution: subclass vs class

The preprint splits by "major cell type". Our subclass level is much finer:

| | Class (33) | Subclass (320) |
|---|---|---|
| Median / q90 cells per pair | 52 / 229 | 20 / 70 |
| Eligible pairs | 33,530 | 75,116 |
| Biology | Broad (e.g. "IT-ET Glut") | Specific (e.g. "L4/5 IT CTX Glut") |

Subclass gives specificity, but most pairs fall below the rank-1 detection threshold for moderate
effects. Class has 2.6x more cells per median pair, and its q90 pairs reach the usable range.
`predicted_group` (31) and `neuron_type` (10) are also in obs. Recommendation: run the two-file smoke
test at both levels (`GROUP_COLUMN=predicted_class` vs `predicted_subclass`) and compare pass rates in
`summary.tsv` before the full run.

## Downsides of the current approach

- **Low n per pair at subclass level** (above). The full atlas helps, but it cannot fix this.
- **Shared features and log1pPF target** across subclasses (z-score stats are per subclass). R would
  run SCTransform per subset.
- **Filter thresholds were tuned for one experiment.** p < 0.05 without multiple-testing control over
  75k pairs will let some noise through. Consider FDR across pairs.
- **Not SCTransform.** r ≈ 0.87 vs R. Analytic Pearson residuals are deferred
  (`../future_todos.md`).

## Bottom line

Per subclass, cell count is even more binding: the median pair has 20 cells, and a 10x downsample
removes 95% of eligible pairs. Whatever tool is used, **every cell must be used**. BRISC is no longer
the only way to fit in memory at subclass level. Its advantages are one job instead of about 265, one
consistent normalization, and likely speed. The bigger open question is resolution: at subclass
level most pairs are below the detection threshold for moderate effects, and class level trades
specificity for about 2.6x the cells.
