# CLAUDE.md — rna-fingerprinting (BRISC fork)

## What this is

Fork of [`satijalab/rna-fingerprinting`](https://github.com/satijalab/rna-fingerprinting) — an R package for
mapping transcriptional responses to perturbation dictionaries via SuSiE regression. This fork adds
**BRISC-based Python alternatives** for the memory-bottleneck functions, enabling dictionary learning on
the full ~7M-cell whole-brain Perturb-seq atlas without downsampling.

Upstream repo: `satijalab/rna-fingerprinting`
Fork: `neurogeek03/rna-fingerprinting`

**The Python port is implemented and validated** — see `python/brisc_fingerprinting/` and the
BRISC-adaptation docs under `brisc_adaptation/` (`AGENT_BRIEFING.md` for onboarding, `decisions_log.md`
for every design decision, `future_todos.md` for deferred work). If you are a new agent (or on a new
cluster), **read `brisc_adaptation/AGENT_BRIEFING.md` first.**

### Cluster-portability (paths below are nibi-specific; configure per cluster)
Hardcoded `/scratch/mfafouti/...` paths, the Slurm account, the apptainer module name, and node memory
all vary by cluster. The run scripts take these as env vars (`SIF`, `H5AD_DIR`, `REPO`, `APPTAINER_MODULE`,
account via `sbatch --account`). Per-cluster checklist is in `AGENT_BRIEFING.md`. The container is public:
`docker://marlenfafouti/brisc:bioc-v2-mkl-seurat-sklearn` (pull it on a compute node, see the briefing).

## Why BRISC

The R pipeline hits two memory walls on the 7M-cell dataset:

1. **`SCTransform`** — fits regularized NB model per gene across all cells, produces dense `scale.data`
   (features x cells). At 5M+ cells x 5,000 features = ~200GB dense float64.
2. **`LearnDictionary` → `fit()`** — operates on `scale.data` at cell level (not pseudobulked). PCA on all
   control cells (~1-2M), then per-perturbation residual PCA + Wilcoxon tests.

BRISC (`SingleCell`) handles these at scale: efficient sparse matrix ops, built-in `normalize()`, `pca()`,
`find_markers()`, `pseudobulk()`, `filter_obs()`, `split_by_obs()`, all multi-threaded and memory-efficient.
The BRISC container for the Python port is the public image
`docker://marlenfafouti/brisc:bioc-v2-mkl-seurat-sklearn` (brisc 0.1.5, ryp 0.4.1, numpy/scipy/polars,
plus scikit-learn + joblib used by `fit()`). On nibi it is pulled to
`/scratch/mfafouti/docker/brisc_bioc-v2-mkl-seurat-sklearn.sif`; set `SIF` to its path on other clusters.

## Package architecture (upstream R)

```
R/
  fit.R           # fit() — internal workhorse for LearnDictionary
                  #   1. PCA(30) on control cells' scale.data
                  #   2. Per-perturbation: baseline subtract → PCA(1) on residuals → Lambda
                  #   3. Iterative outlier removal (MAD >= 10)
                  #   4. Wilcoxon filter: keep Lambda if p < 0.05 AND >= 2/10 top features DE
                  #   5. Estimate means + SDs per feature for each kept Lambda
                  #   Output: list(Lambdas = matrix[features x perturbations], dists = list of [features x 2])

  utilities.R     # LearnDictionary() — public wrapper: validates Seurat SCT assay, extracts scale.data,
                  #   relabels control → 'control', calls fit()
                  # ChooseFeatures() — DE-based (FindMarkers per perturbation) or expression-based feature selection
                  # ConvertGenes() — gene ID conversion via org.Hs.eg.db (HUMAN only — not needed for mouse dict)
                  # SummarizeResults() — extracts credible sets from FingerprintCells output, optional tidy format

  classify.R      # classify() — internal workhorse for FingerprintCells
                  #   1. PCA(30) on control cells (same as fit)
                  #   2. Baseline-subtract all non-control cells
                  #   3. If group_level: aggregate residuals per condition (rowSums)
                  #   4. Per cell/group: top-100 correlated Lambdas → susie_plus regression → credible sets
                  #   5. Calibration: resample Lambdas from dists 50x, re-compute PIPs, final credible sets
                  #   6. Optional joint Lambdas for correlated perturbations
                  # FingerprintCells() — public wrapper: validates Seurat SCT, extracts scale.data, calls classify()
                  # get_model_attribute() — helper for visualization (betas, BFs, correlations, residuals)

  susie_plus.R    # susie_plus() — custom SuSiE (Sum of Single Effects) implementation
                  #   IBSS algorithm with optional dummy variables for outlier genes
                  #   Pure linear algebra: X'y, X'X, single-effect regressions, ELBO convergence
                  #   No R-specific dependencies beyond base stats

  visualization.R # LongIslandCityPlot() — visualize fingerprint assignments
                  # ExplainMatch() — identify top genes driving a match (delta-BF analysis)
                  # ProjectFingerprints() — UMAP of projected fingerprint scores
```

## What to rewrite in BRISC (Python)

### Must rewrite (memory bottlenecks)
- **`SCTransform` replacement**: BRISC `normalize()` (log1pPF or similar). The key requirement is producing
  a scaled/centered matrix suitable for PCA. SCTransform's Pearson residuals serve this role; BRISC's
  normalization + manual scaling achieves the same goal.
- **`fit()` → `learn_dictionary()`**: The core algorithm. All operations are linear algebra:
  - PCA on controls: `sc.pca(num_PCs=30)` on control-cell subset
  - Per-perturbation residual computation: matrix ops on sparse/dense arrays
  - Rank-1 PCA on residuals: `scipy.sparse.linalg.svds` or numpy
  - Wilcoxon tests: `scipy.stats.mannwhitneyu` or `scipy.stats.ranksums`
  - Outlier removal loop: numpy
  - Output: `Lambdas` (features x perturbations) + `dists` (means + SDs per feature per perturbation)
- **`ChooseFeatures(method='de')`**: BRISC `find_markers()` can replace Seurat `FindMarkers` per perturbation,
  or use a faster detection-rate approach.

### Keep in R (works fine, no memory issue)
- **`classify()` / `FingerprintCells()`** — operates on the query object (100k cells, not 7M). The dense
  `scale.data` for the query is manageable. Needs the dictionary (`Lambdas` + `dists`) from the Python step.
- **`susie_plus()`** — pure linear algebra on small matrices (top-100 Lambdas x features). Could be ported
  to Python but no need — it's fast and memory-light.
- **`SummarizeResults()`** — metadata extraction, trivial.
- **Visualization** — Seurat plotting, stays in R.

### Dictionary output format (compatibility contract)

The Python `learn_dictionary()` must produce an R-loadable `.rds` with this structure:
```r
dictionary <- structure(
  list(
    fingerprints = list(
      Lambdas = matrix(...),   # features x perturbations, named rows + cols
      dists = list(            # named list, one per perturbation
        "Aak1" = matrix(cbind(means, sds), nrow=n_features, ncol=2),
        "Abca1" = matrix(...),
        ...
      )
    ),
    order = NULL,
    data = NULL
  ),
  class = "dictionary"
)
```
- `Lambdas`: numeric matrix, `rownames` = feature names (mouse gene symbols), `colnames` = perturbation names
  (commas/underscores/semicolons replaced with dashes)
- `dists`: named list matching `colnames(Lambdas)`, each element a 2-column matrix (means, SDs) with
  `rownames` = feature names
- Save via BRISC's R bridge (`ryp`) or `rpy2` to ensure proper R list/matrix structure

## Algorithm details (fit.R, line by line)

1. **Input**: `data` = SCT `scale.data` [features x cells], `perturbation_meta` = character vector (one per
   cell, control relabeled to `"control"`), `features` = selected gene set
2. **Control baseline**: `ctrl.pdev = data[features, control_cells]` → `ctrl.mus = rowMeans(ctrl.pdev)` →
   `PCs = prcomp_irlba(t(ctrl.pdev), n=30)` — 30-PC embedding of control cells
3. **Control residuals** (for Wilcoxon reference): subsample 1,000 control cells, compute
   `residual = observed - (PC_reconstruction + mean)` — what's left after removing the baseline signal
4. **Per-perturbation loop** (parallelized via `pbmclapply`):
   a. Subset to perturbation's cells: `pdev = data[features, perturbation_cells]`
   b. Baseline-subtract: project onto control PCs, reconstruct, subtract → `residual`
   c. Rank-1 PCA on residuals: `PCs2 = prcomp_irlba(t(residual), center=F, n=1)` — the loading vector
      (`PCs2$rotation[,1]`) is the candidate Lambda (fingerprint)
   d. **Outlier removal**: if any cell's score (`PCs2$x[,1]`) has MAD >= 10, remove it, recompute PCA.
      Track correlation between each Lambda iteration and Wilcoxon statistics. Pick the Lambda with
      max |correlation|.
   e. **Sign correction**: if correlation is positive, flip Lambda and scores (convention: Lambda should be
      anti-correlated with the Wilcoxon control-vs-perturbed statistics)
   f. **Quality filter**: keep Lambda if (i) `wilcox.test(ell, t(Lambda) %*% ctrl_residuals)` p < 0.05 AND
      (ii) >= 2 of the top-10 Lambda features are significantly DE (Wilcoxon p < 0.05)
   g. **Estimate distributions**: `means = (residual %*% ell) / sum(ell^2)`,
      `sds = sqrt(sigma^2 / sum(ell^2))` — used downstream for calibration in `classify()`

## Key observations for the port

- `fit()` never pseudobulks — it needs cell-level scaled data. This is the bottleneck.
- The Wilcoxon tests (`wilcoxauc` from presto, `wilcox.test` from base R) are used for two purposes:
  (a) orienting Lambda sign, (b) filtering low-signal perturbations. `scipy.stats` equivalents exist.
- `prcomp_irlba` = truncated SVD. Use `scipy.sparse.linalg.svds` or `sklearn.decomposition.TruncatedSVD`.
- The per-perturbation loop is embarrassingly parallel — use `multiprocessing` or `joblib`.
- SCTransform Pearson residuals vs BRISC normalization: the exact normalization method matters less than
  producing a centered/scaled matrix where baseline variation is captured by PCA. Test empirically.

## Perturb-seq dataset (dictionary source)

- Location (nibi; stage/transfer to other clusters and set `H5AD_DIR`):
  `/scratch/mfafouti/Psych_brain/Perturb-seq/wholebrain_atlas/h5ads/` (14 h5ad files, ~72GB)
- 7.7M cells x 27,658 genes (mouse gene symbols in var index, Ensembl in `gene_ids`)
- `gene_target`: perturbation column (~2,049 targets). Controls: `Negative`, `Non_target`
- `passes_qc`: boolean QC filter (~78% pass)
- `predicted_subclass`: Allen Brain Atlas cell types
- After QC filter: ~6M cells. Use `Negative` as control class.
- Pipeline project consuming this dictionary: `/scratch/mfafouti/RNA-fingerprinting-satija`

## Development workflow

- Edit here, rebuild Docker image, push to Docker Hub, re-pull as SIF on cluster
- Test with BRISC container: `apptainer exec "$SIF" python ...` where `$SIF` is the pulled image
  (`docker://marlenfafouti/brisc:bioc-v2-mkl-seurat-sklearn`)
- Test R functions: an R/Seurat container with the `RNAFingerprinting` package + Seurat/irlba/presto/
  pbmcapply (on nibi: `/scratch/mfafouti/docker/rna-fingerprinting_seurat-v1.sif`)
- Heavy compute (full load/normalize + control-cell PCA) must run on a compute node, not the login node;
  the full atlas needs a large-memory node (~400GB+ RAM). See `brisc_adaptation/AGENT_BRIEFING.md`.
