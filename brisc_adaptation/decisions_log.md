# BRISC adaptation — decisions log

Decisions made while porting the memory-bottleneck steps of `rna-fingerprinting`
(`SCTransform` + `fit()`/`LearnDictionary()`) to a BRISC/Python implementation that scales to the
full QC-passing whole-brain Perturb-seq atlas (~6M cells). Downstream (`classify()`,
`susie_plus()`, visualization) stays in R and consumes the `.rds` dictionary this produces.

Last updated: 2026-10-13.

---

## 1. Container / dependencies

- **Decision:** run the Python port inside `/scratch/mfafouti/docker/brisc_bioc-v2-mkl-seurat-sklearn.sif`
  (brisc 0.1.5, ryp 0.4.1, numpy 2.5.3, scipy 1.18.1, polars 1.44.2, **scikit-learn 1.9.1, joblib 1.6.0**).
- **Why:** the port uses established libraries rather than hand-rolled code — `sklearn.utils.extmath.randomized_svd`
  for truncated SVD (stand-in for R `irlba::prcomp_irlba`) and `joblib.Parallel` for the per-perturbation
  loop (stand-in for R `pbmcapply::pbmclapply`). The original BRISC image lacked sklearn/joblib, so a new
  image was built (by user) and pulled.
- **Ops note:** pull large images on a **compute node** with `APPTAINER_TMPDIR=$SLURM_TMPDIR` (node-local SSD);
  login-node unpack onto networked scratch is pathologically slow (>1h vs ~5min). Keep the blob cache on scratch.
- **Ops note (OOM):** `load_qc` loads only counts + `obs` + `var` (`obsm=False, obsp=False, varm=False,
  varp=False, uns=False`) and QC-filters each file before `concat_obs`. The atlas h5ads carry large
  `obsp` neighbor graphs / `obsm` embeddings; loading all of them across 14 files OOMs a 240G node
  during load. Skipping them is required for the full-atlas run.
- **Memory sizing (measured):** full atlas = 6,127,364 QC cells, 32.6e9 nonzeros, float32 data +
  uint16 indices = 6 bytes/nnz -> **~196 GB per copy** of the QC counts CSR, **~392 GB transient peak**
  (during `concat_obs`, and during build/normalize). nibi CPU nodes have **766 GB** (6 TB largemem
  nodes also exist but unnecessary). The OOM was purely `comp`'s `--mem=240G` cap (< one copy).
  Full-atlas runs request **`--mem=750G`** (sbatch) or an equivalent `salloc`. `normalize_counts` uses
  `inplace=True` to avoid holding raw + normalized at once.

## 2. Normalization / scaling (SCTransform replacement)

- **Decision:** BRISC `normalize(method='log1pPF')`, then a global per-gene z-score computed over all
  QC-passing cells, restricted to the selected features. This is the `scale.data` equivalent fed to PCA.
- **Why:** CLAUDE.md sanctions approximating SCTransform Pearson residuals with a centered/scaled matrix;
  what matters is that PCA can capture baseline variation.
- **Validated & accepted (2026-10-14):** vs the real R `LearnDictionary` on the CPA example the port
  reaches signed Pearson r ≈ 0.87 (see §8). We accept this — captures the same biology and perturbation
  set; a tighter alternative (analytic Pearson residuals) is deferred in `future_todos.md`.
- **Implementation detail:** we do QC by hand (`filter_obs(passes_qc)` drops failing cells), then call
  `skip_qc()` to set `uns['QCed']=True` so `normalize()`'s guard passes, and `normalize(QC_column=None)`
  so every remaining cell is used.

## 3. Truncated SVD & parallelism

- **Decision:** `randomized_svd` (sklearn) for the control-cell 30-PC PCA and the per-perturbation rank-1 fit;
  `joblib.Parallel(backend="threading")` for the per-perturbation loop (shares the read-only sparse matrix,
  numpy/scipy releases the GIL).

## 4. Wilcoxon test (performance)

- **Decision:** drop the exact tie-correction term in the per-feature Mann-Whitney U (use the no-tie variance).
- **Why:** PCA residuals are continuous floats, so ties are essentially nonexistent; the per-feature
  `np.unique` tie loop was the dominant cost (~44 min on a single small file). Negligible accuracy loss.
  *(Status: applied in `fit.py::_wilcox_per_feature`.)*

## 5. Perturbation handling — single-gene dictionary

Real data (`gene_target`, one file) has 14,226 unique labels: 2,047 single-gene, 781 gene+Non_target,
11,395 true multi-gene combos, plus controls. The 2,047 single-gene labels match CLAUDE.md's "~2,049 targets".

- **Decision:** build a **single-gene** dictionary.
  1. Treat as inert control tokens: `Negative`, `Non_target`, and `Safe_target_*`.
  2. For each `|`-split label, strip control tokens to get the "real genes":
     - 0 real genes → control
     - **1 real gene → single-gene perturbation** (collapses e.g. `Abca1`, `Abca1|Non_target`,
       `Abca1|Safe_target_40` → `Abca1`, recovering ~781 gene+NT cells and boosting power)
     - ≥2 real genes → **exclude** (uninterpretable combo)
- **Why:** `fit()` learns one fingerprint per unique label and downstream interprets it as that perturbation's
  signature. Single-gene labels are interpretable and well-powered; combos fragment cells (most < `min_cells`)
  and can't be read as any gene's fingerprint. Matches upstream GWPS single-gene dictionaries.
- **Note:** upstream name-cleaning replaces `[,;_]` with `-`; the pipe `|` is not in that regex (moot once
  combos are excluded and gene+control labels are collapsed to the bare gene name).

## 6. Control definition & baseline pool

Confirmed from the data (QC cells, one file): `Negative` = **cells with no guide detected**
(`guide_call="Negative"`, `num_guides=0`), ~18,662 cells; `Non_target` = guide, non-cutting, ~2,208;
`Safe_target_*` = guide, cutting at a safe locus (100 loci), ~896. This is a **Cas9 knockout** screen
(guide names like `Abat_1`, presence of safe-targeting controls).

- **Decision:** use **`Negative` + `Non_target` + `Safe_target_*` pooled** as the control class (relabeled
  to `'control'`), with `Negative` as the nominal primary per CLAUDE.md.
- **Why:** `fit()`'s baseline is a 30-PC PCA on control cells that must span whole-brain cell-type
  heterogeneity — the binding constraint is cell-type coverage, which favors the large `Negative` pool.
  Pooling the engineered controls (`Non_target`, `Safe_target`) adds matched guide/Cas9 and DSB-response
  signal to the baseline at no cost, so it is strictly better than `Negative`-only.
- **Trade-off noted:** `Negative` (no-guide) is not a matched control and may carry transduction-failure
  bias; the biologically purest baseline would be `Safe_target` (matches the DSB/DDR confound of real
  perturbations), but at ~896 cells/file it cannot span whole-brain cell types. Revisit to engineered-only
  (`Non_target`+`Safe_target`, ~55k cells across the full atlas) if the no-guide pool proves biased.

## 7. Output contract

- **Decision:** write the dictionary as an R `.rds` of class `dictionary` via `ryp`
  (`list(fingerprints=list(Lambdas=<features×P>, dists=<named list of features×2>), order=NULL, data=NULL)`),
  byte-compatible with the existing R `classify()`/`FingerprintCells()`. Verified via round-trip read in R.
- **Fix (2026-10-14, Trillium smoke test):** `ryp.to_r` passes arrays as arrow-backed ALTREP vectors and
  `saveRDS` serialized them as `arrow::array_dbl_vector` ALTVEC objects. `readRDS` in an R session without
  the arrow package (e.g. `rna-fingerprinting_seurat-v1.sif`) warned and returned garbage (~1e-310 values).
  `rds_io.py` now copies every vector into plain R memory before saving. Check any dictionary with
  `Rscript brisc_adaptation/validation/check_dictionary_rds.R <file.rds>` in the R container (errors on
  any readRDS warning; also checks structure + unit-norm Lambdas).

## 8. Validation status

- Synthetic end-to-end smoke test: **passing** (incl. `.rds` round-trip) in the sklearn/joblib container.
- Single real h5ad (`WB8588_1_2.h5ad`, 59,776 QC cells × 27,658 genes), **finalized logic**
  (collapse-to-single-gene + pooled control + Wilcoxon speedup, `n_jobs=32`): **passes end-to-end in 67 s**
  (down from 2,627 s). 21,800 pooled control cells, 1,946 candidate single-gene perturbations across
  54,174 cells (combos dropped), **241 fingerprints** learned; `.rds` round-trips (Lambdas 2000×241).
- `find_markers` returns columns `['cell_type','gene','detection_rate','fold_change']` (no p-value);
  `choose_features(method='de')` ranks by `fold_change` descending accordingly.
- **Numerical agreement vs R** (`brisc_adaptation/validation/`, CPA example, identical counts/labels/896
  features): R's real SCTransform + `LearnDictionary` vs the Python port. Both keep the same
  perturbations with matching sign; **signed Pearson r = +0.866 (CPSF6), +0.864 (PABPN1)**, top-20 gene
  overlap 11–12/20. Gap attributed to z-score(log1pPF) vs SCTransform Pearson residuals; **accepted**.
- **Full atlas, per subclass (Trillium job 2562632, 2026-10-14): passes.** All 14 h5ads went through
  the per-file loader: 6,127,364 QC cells and 11,736 expression-selected features. The run took
  **1 h 40 min**. Peak RSS was 431 GB in the process log and 508 GB in Slurm MaxRSS, on a ~745 GiB
  node. `.err` was empty and the exit code 0.
  - Subclasses: of 321, 265 have ≥ 30 controls, matching `subclass_counts/report.txt`. 148 have no
    knockout with ≥ 10 cells, 56 have < 30 controls, 33 had no fingerprint pass the filter, and
    **84 wrote a dictionary**.
  - Fingerprints: 6,618 kept out of 74,878 candidate knockout × subclass pairs, about 8.8%. That is
    a median of 45 per written subclass (range 1–548). The largest are TH Prkcd Grin2c 548/1,947,
    L4/5 IT CTX 543, Pvalb 408 and L5 ET CTX 357.
  - All 84 `.rds` pass `validation/check_dictionary_rds.R`
    (`validation/check_all_dictionaries.sh`).
  - Output: `/scratch/mfafouti/Perurb-seq_WMB/dict_wholebrain_by_subclass/` (+ `summary.tsv`).

## 9. One dictionary per subclass (2026-10-14)

**Decision.** Learn one dictionary per `predicted_subclass`, not one pooled dictionary.

**Why.** The authors do this. Preprint Methods (Immune Dictionary): "We then estimated fingerprints
separately for each cell type, using the PBS cells as controls, and excluding cell types with fewer than
30 PBS cells available." Queries are then mapped "to the corresponding fingerprints of the same cell
type". The package code never shows this, because `fit()`/`LearnDictionary()` have no cell-type notion:
the split is done by subsetting the object before calling them, and the vignettes use one cell line.
A pooled run blends cell-type-specific effects into one Lambda and leaves cell-type structure in the
residuals (30 PCs cannot model ~300 subclasses).

**How.** `iter_dictionaries_by_group` (`learn_dictionary.py`) loops over subclasses. Each subclass uses
its own control cells (baseline PCA + 1,000-cell Wilcoxon reference) and its own per-gene mean/SD, and
`fit()` is unchanged. Subclasses with < 30 controls are skipped (preprint threshold;
`--min-control-cells`). Only one subclass's rows are copied out of the stacked matrix at a time. Output
is one `<subclass>.rds` per subclass plus `summary.tsv`. CLI `--group-column predicted_subclass`
(default in both run scripts; `GROUP_COLUMN=` restores the pooled run). Test: `python/tests/test_by_group.py`.

**Kept global on purpose.** The feature set and log1pPF target are shared, so every subclass dictionary
has the same rows. (R per-subset SCTransform would be per-subclass; revisit if needed.)

**Cost.** Fewer cells per knockout per subclass, so more knockout × subclass pairs fall below
`min_cells` and the Wilcoxon filter has less power. Check `summary.tsv` from a smoke test before the
full run. Phase B must also be per subclass (in `/scratch/mfafouti/RNA-fingerprinting-satija`).

## 10. Threading in `fit()` (2026-10-14)

**Bug.** The BLAS thread cap for the per-perturbation loop was set with `threadpool_limits` in the
main thread. With MKL's Intel-OpenMP layer that limit is per-thread and does not reach joblib's worker
threads, so they used the startup `OMP_NUM_THREADS`. The run scripts set that to all cores (for the
control PCA), which gave 96–192 workers × 192 BLAS threads and no progress in 20–36 min on the
two-file set. **Fix:** `_fit_one_limited` sets the cap inside each worker (cores // n_jobs).
Mechanism test: `python/tests/check_thread_limits.py`. Write-up:
`docs/thread_scaling_paradox.md` (top correction).

**Wilcoxon.** `_wilcox_per_feature` now uses sorted control keys + one `searchsorted` per feature
chunk instead of re-ranking the fixed 1,000-cell reference every task. U is identical to the rankdata
version (`python/tests/test_wilcox.py`). It is 6.3x faster at n_p=200 and 2.0x at n_p=2000 (single
thread).

## Open / to-do

- [x] §4 Wilcoxon speedup — applied (`fit.py`).
- [x] §5–6 collapse-to-single-gene + pooled control — applied (`perturbations.py`,
      `learn_dictionary(collapse_single_gene=True)`); unit + synthetic integration tests pass.
- [x] Re-run single-file validation with finalized logic at `n_jobs=32` — passes in 67 s, 241 fingerprints.
- [x] Numerical agreement check vs the R dictionary (CPA example) — r ≈ 0.87, accepted.
- [ ] Confirm `Safe_target_*` treatment is acceptable (currently: control, not perturbation).
- [x] §9 per-subclass dictionaries — applied; synthetic test passes.
- [ ] Smoke-test per-subclass on 1–2 files; inspect `summary.tsv` (controls, candidates, kept per subclass).
- [ ] **Resolution: subclass vs class.** Real counts (`subclass_counts/report.txt`): the median
      eligible knockout × subclass pair has 20 cells (q90 70), versus 52 (q90 229) per class. That is
      below the rank-1 detection threshold for moderate effects. Smoke-test both
      `GROUP_COLUMN=predicted_subclass` and `predicted_class`, then decide (see `docs/why_brisc_full_atlas.md`).
- [ ] Full-atlas run across all 14 h5ads once single-file validation looks good.
- [ ] Phase B per subclass in the consuming project (subset query by subclass, matching `.rds`).
- Deferred (see `future_todos.md`): analytic Pearson residuals for tighter R agreement (~0.95+).
