# Trillium test run — BRISC dictionary learning

## Context
Port is implemented/validated on nibi; remaining deliverable is the full 14-file atlas run. Moving to
Trillium (login node `tri-login01`). Data and SIF are now in place:
- `H5AD_DIR=/scratch/mfafouti/Perurb-seq_WMB/data/h5ads` — 14 files, 72G (matches expected; transfer looks complete)
- `SIF=/scratch/mfafouti/docker/brisc_bioc-v2-mkl-seurat-sklearn.sif`

## Trillium-specific concerns found
1. Trillium allocates whole nodes (192 cores, ~755G usable). `--mem=750G` / `--cpus-per-task=48` in
   `python/sbatch/learn_dictionary.sbatch` don't fit that model; `--account=CHANGE_ME` must be `rrg-shreejoy`.
2. Memory is tight: measured ~392G transient peak, plus `build_feature_matrix` (`scaling.py:103`)
   converts the *full* genes matrix to CSC before column-subsetting → another full copy. Risk of OOM on 755G.
3. [DONE: CLI default now 0.05] `--mean-expression` default differs: CLI 0.01 (`run_learn_dictionary.py:61`) vs sbatch/interactive 0.05.
4. No `config.conf` — paths/tunables live in env-var defaults + argparse (violates global rule).

## Recommended steps
1. **[DONE 2026-10-14, job 2560134: 170/1736 perturbations x 12306 features, 2 min, 52G peak; R check passes.
   Found + fixed arrow-ALTREP .rds bug in `rds_io.py`]** **Smoke test (1 file)** on a debug node: `n -h 1`, then
   `SIF=... H5AD_DIR=<dir with symlink to WB8588_1_1.h5ad> OUT=$SCRATCH/... bash brisc_adaptation/run_dictionary_interactive.sh`.
   Confirms container, apptainer module, ryp `.rds` write on Trillium.
2. **[DONE] Memory fix before full run** (`restrict_to_features` in `scaling.py`, called in CLI after normalize): `sc.filter_var` to selected features right after choosing them
   (open item in AGENT_BRIEFING), so `build_feature_matrix` only touches k genes.
3. **[DONE] Update sbatch for Trillium**: account `rrg-shreejoy`, drop `--mem`, `--cpus-per-task=192`, compute partition.
4. **Full run** via sbatch (24h max) or `n -h 4`; monitor with `seff`/peak RSS.
   - [DONE] float32 dense blocks + in-place control centering in `fit.py` (job 2560200: identical
     170 perturbations, Lambda/dist-mean r = 1.000 vs float64). Control block ~110 GB at full scale.
   - [FAILED 2026-10-14, job 2560243] full run died in `concat_obs`: obs columns differ in dtype
     across files. [DONE] `load_qc` loads only `passes_qc` + `gene_target` (cast to String).
   - [DONE] Removed whole-matrix copies after normalize (`filter_obs` -> `__excluded__` label,
     no CSR->CSC->CSR, chunked `gene_mean_sd`): two-file fit peak 124 -> 70 GB, r = 1.000.
   - [DONE] Timestamped/memory logging (`logs.py`), `python -u`, TZ passed to container.
   - [DONE] Threads: 192x1 measured >10x slower than 96x2 -> reverted to cores/2 workers, BLAS
     cores/NJOBS per worker. See `brisc_adaptation/docs/thread_scaling_paradox.md`.
   - [DONE, validating] **Fix 2 — no-concat per-file loader** (`stacked.py`, `--loader per-file`
     default): int32 blocks behind `RowStackedCSR`, single global log1pPF. Expected full-atlas peak
     ~400 GB (was ~780 GB). Unit test `python/tests/test_stacked.py` passes (log1pPF vs BRISC
     max |diff| 2.4e-7). Two-file run: job 2561968.
   - [DONE, validating] **Fix 1 — searchsorted Wilcoxon** (`fit.py`): identical U to rankdata
     (`python/tests/test_wilcox.py`); 6.3x faster at n_p=200, 2.0x at n_p=2000 (single thread).
   - [ ] Two-file run with both fixes -> then thread-count sweep -> then full run.
5. (Optional, separate change) introduce `config.conf` for paths/tunables.

## Verification
- Smoke run exits `DICT_EXIT=0`, `.rds` loads in the R container (`readRDS`, class `dictionary`,
  `Lambdas` dims = features x perturbations, `names(dists) == colnames(Lambdas)`).
- Full run: peak memory < 700G, number of learned perturbations logged.
