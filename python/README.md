# BRISC Python port — dictionary learning at atlas scale

Python/BRISC reimplementation of the memory-bottleneck steps of `rna-fingerprinting`
(`SCTransform` + `fit()`/`LearnDictionary()`), so a perturbation dictionary can be learned on the
full QC-passing whole-brain Perturb-seq atlas (~6M cells) without downsampling. Everything
downstream (`classify()`/`FingerprintCells()`, `susie_plus()`, visualization) stays in R and consumes
the `.rds` this produces. See `../BRISC_PORT_PLAN.md` and `../CLAUDE.md` for the full spec.

## Layout

```
brisc_fingerprinting/
  scaling.py          load/concat/QC/normalize (BRISC) -> sparse feature matrix + per-gene stats
  fit.py              core per-perturbation fingerprint learning (port of R/fit.R)
  learn_dictionary.py public wrapper (port of LearnDictionary)
  choose_features.py  ChooseFeatures(method='de'|'expression')
  rds_io.py           write the dictionary .rds via ryp (R-compatible contract)
scripts/run_learn_dictionary.py   CLI
sbatch/learn_dictionary.sbatch    cluster job template
```

## Dependencies

Runs inside the BRISC container, which provides `brisc` 0.1.5, `ryp` 0.4.1, numpy, scipy, polars.
The port additionally requires:

- **`scikit-learn`** — `sklearn.utils.extmath.randomized_svd` is the truncated-SVD solver that stands
  in for R's `irlba::prcomp_irlba`, used for the control-cell PCA and the per-perturbation rank-1 fit.
- **`joblib`** — `joblib.Parallel` runs the embarrassingly-parallel per-perturbation loop (the port
  of R's `pbmcapply::pbmclapply`).

(Container build and Docker Hub image are maintained separately — see the link in the repo root.)

## Usage

```sh
module load apptainer/1.3.5
apptainer exec /scratch/mfafouti/docker/brisc_bioc-v3-dreamlet.sif \
  python python/scripts/run_learn_dictionary.py \
    --h5ad-dir /scratch/mfafouti/Psych_brain/Perturb-seq/wholebrain_atlas/h5ads \
    --features-file features.txt \
    --control-class Negative --exclude-classes Non_target \
    --n-jobs 16 --out dictionary.rds
```

Heavy PCA (densifies ~1-2M control cells) must run on a **compute node**, not the login node — use
`sbatch/learn_dictionary.sbatch` (set your `--account` and `--mem`).

## Deliberate approximations vs the R pipeline

1. **scale.data**: SCTransform Pearson residuals -> global per-gene z-score of log1pPF-normalized
   counts (CLAUDE.md sanctions this; both feed PCA the same way).
2. **truncated SVD**: `irlba::prcomp_irlba` -> numpy randomized SVD (same math).
3. **ChooseFeatures(method='de')**: Seurat `FindMarkers` per-perturbation-vs-control -> BRISC
   `find_markers` one-vs-rest (scales to millions of cells).

Validate against the R pipeline on a small dataset (e.g. `data/cpa_perturb_seq.rda`) before trusting
at scale. `fit()` output is a faithful line-by-line port otherwise.

## Status

- [x] Core algorithm ported and unit-smoke-tested end-to-end on synthetic data (incl. `.rds` round-trip).
- [ ] Validated on real h5ad(s) via `load_and_normalize` / `choose_features` (needs a compute node).
- [ ] Numerical agreement check against the R dictionary on a shared small dataset.
