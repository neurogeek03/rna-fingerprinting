# BRISC port plan — rna-fingerprinting

Goal: replace the two memory-bottleneck R steps (`SCTransform` + `fit()`/`LearnDictionary`)
with a BRISC-based Python implementation that can learn a perturbation dictionary on the full
~6M-cell (QC-passing) whole-brain Perturb-seq atlas without downsampling, and emit an
`.rds` dictionary that is byte-compatible with the existing R `classify()`/`FingerprintCells()`
downstream (which stays in R).

See `CLAUDE.md` for the algorithm spec and compatibility contract.

## Scope of this port

| R function | Status | Python target |
|---|---|---|
| `SCTransform` (normalization + scaling) | rewrite | `scaling.py` — BRISC `normalize()` + global per-gene z-score |
| `fit()` | rewrite | `fit.py::fit()` |
| `LearnDictionary()` | rewrite | `learn_dictionary.py::learn_dictionary()` |
| `ChooseFeatures()` | rewrite | `choose_features.py` (DE via `find_markers`, or expression) |
| dictionary `.rds` writer | new | `rds_io.py::write_dictionary_rds()` (via `ryp`) |
| `classify()` / `FingerprintCells()` | **keep in R** | — |
| `susie_plus()` | **keep in R** | — |
| `SummarizeResults()`, visualization | **keep in R** | — |

## Algorithm mapping (fit.R -> fit.py)

Faithful port. Only two deliberate approximations (CLAUDE.md sanctions both; validate empirically):
1. **scale.data**: SCTransform Pearson residuals -> global per-gene z-score of log1pPF-normalized
   counts. Both give a centered/scaled matrix whose baseline variation PCA captures.
2. **truncated SVD**: `irlba::prcomp_irlba` -> `sklearn.utils.extmath.randomized_svd` (same math,
   different solver; `prcomp` centers columns by default, `center=F` variants do not).

Per-perturbation loop is embarrassingly parallel (`joblib`/`multiprocessing`), as in the R
`pbmclapply`. Wilcoxon: `presto::wilcoxauc` -> vectorized `scipy.stats.ranksums` per feature.

### Memory strategy (the whole point)

Never materialize the full dense features×cells matrix. Instead:
- Load 14 h5ads with BRISC, concat, `filter_obs(passes_qc)`, `normalize()` once.
- Restrict columns to the selected `features` -> sparse CSR `norm_X` (n_cells × k_features).
- Compute global per-gene mean/sd from the sparse matrix (one pass).
- PCA on control cells: densify only the control rows (`norm_X[control_rows]`, ~1-2M × k,
  float32), z-score, `randomized_svd(30)`. Keep loadings (k×30) + control mean (k).
- Per perturbation: densify only that perturbation's rows (small), compute residual, rank-1 SVD,
  Wilcoxon vs a 1,000-cell control-residual reference, outlier loop, filter, estimate dists.

## Compatibility contract (dictionary .rds)

`structure(list(fingerprints=list(Lambdas=<k×P matrix, named>, dists=<named list of k×2>),
order=NULL, data=NULL), class="dictionary")`. Perturbation names: `[,;_]` -> `-`, made unique.
Built in R space via `ryp::to_r` + `r(...)`, then `saveRDS`.

## Package layout

```
python/brisc_fingerprinting/
  __init__.py
  scaling.py            # load/concat/QC/normalize -> norm_X + gene stats
  fit.py                # fit(): core per-perturbation fingerprint learning
  learn_dictionary.py   # public wrapper: orchestrates scaling.py + fit.py
  choose_features.py    # ChooseFeatures(method='de'|'expression')
  rds_io.py             # write_dictionary_rds() via ryp
scripts/run_learn_dictionary.py   # CLI
sbatch/learn_dictionary.sbatch    # cluster job
```

## Validation

Start small (1 h5ad or a few hundred-cell perturbations) on a compute node under the BRISC SIF.
Compare learned Lambdas against the R pipeline on the CPA example dataset (`data/cpa_perturb_seq.rda`)
to confirm sign, feature ordering, and kept-perturbation set agree within tolerance before scaling up.

## Run

```
module load apptainer/1.3.5
apptainer exec /scratch/mfafouti/docker/brisc_bioc-v3-dreamlet.sif \
  python scripts/run_learn_dictionary.py --help
```
Heavy PCA runs on a compute node only (never the login node).
