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
   - [DONE] Threads: 96 workers (cores/2) x cores/NJOBS BLAS per worker.
   - [DONE] **Thread-limit bug found + fixed.** `threadpool_limits` in the main thread does not
     reach joblib worker threads (MKL Intel-OpenMP: per-thread). Workers therefore kept the startup
     `OMP_NUM_THREADS` = 192. That was the real cause of the 192x1 slowdown (really 192x192) and of
     job 2561968 stalling (really 96x192). The fix is `_fit_one_limited` in `fit.py`, which sets the
     cap inside each worker; `python/tests/check_thread_limits.py` demonstrates the mechanism. The
     correction is at the top of `brisc_adaptation/docs/thread_scaling_paradox.md` and in
     `decisions_log.md` §10.
   - [DONE] **Fix 2 — no-concat per-file loader** (`stacked.py`, `--loader per-file` default):
     int32 blocks behind `RowStackedCSR`, single global log1pPF. Expected full-atlas peak ~400 GB
     (was ~780 GB). `python/tests/test_stacked.py` passes (log1pPF vs BRISC max |diff| 2.4e-7). Not
     yet validated on real data; that happens in the step-5 smoke.
   - [DONE] **Fix 1 — searchsorted Wilcoxon** (`fit.py`): identical U to rankdata
     (`python/tests/test_wilcox.py`); 6.3x faster at n_p=200, 2.0x at n_p=2000 (single thread).
   - [CANCELLED 2026-10-14] Pooled two-file run (job 2561968): stalled in the loop because of the
     thread-limit bug. It also ran the pooled pipeline and the old Wilcoxon, so it is superseded.
     All three test suites pass after the fixes.
   - Thread-count sweep / full pooled run: **superseded by step 5**. Only do them if a pooled
     dictionary is still wanted (set `GROUP_COLUMN=`).
5. **Per-subclass dictionaries** (authors' method; `brisc_adaptation/decisions_log.md` §9,
   `brisc_adaptation/docs/plan_per_subclass_dictionaries.md`).
   **Heads-up for any new submission:** the sbatch and interactive scripts now **default to
   `GROUP_COLUMN=predicted_subclass`**, and `OUT` is then a directory. To reproduce a pooled run, pass
   `GROUP_COLUMN=` (empty).
   - [DONE] Code: `iter_dictionaries_by_group` + CLI `--group-column/--min-control-cells` + per-file
     `obs_columns`. Synthetic test `python/tests/test_by_group.py` passes. `test_stacked` and
     `test_wilcox` still pass. `predicted_subclass` is present in all 14 h5ads (categorical, 294–321
     per file).
   - [DONE 2026-10-14, job 2562495] **Per-subclass smoke, two files** (432k cells, 11,796
     features). Results: exit 0, **2 min 10 s total**, MaxRSS 61 GB, empty `.err`.
     - Subclasses: of 298, 215 have ≥ 30 controls. 202 have no knockout with ≥ 10 cells, 83 have < 30
       controls, 3 had no fingerprint pass the filter, and **10 wrote a dictionary** (1–81 kept;
       largest TH Prkcd Grin2c 81/1,465, Pvalb 36/708).
     - All 10 `.rds` pass `check_dictionary_rds.R`.
     - Loop speed with the thread fix: 1,465 perturbations in 51 s, versus the pooled run that
       stalled for 36 min.
     - The per-file loader holds the raw atlas in 20 GB RSS (concat path: 29 GB). Same 11,796
       features as the concat path.
     - Open: log timestamps are still UTC on compute nodes (the `/etc/localtime` lookup comes back
       empty there). They are unambiguous because the offset is printed.
     Original checklist:
     `H5AD_DIR=/scratch/mfafouti/Perurb-seq_WMB/smoke2_h5ads OUT=$SCRATCH/dict_smoke2_by_subclass sbatch ...`
     (debug QOS / `n -h 1` is fine). Check:
     - `summary.tsv`: how many subclasses reach ≥ 30 controls; candidates and kept per subclass.
     - Wall time: there are ~300 sequential control PCAs, and the small subclasses underuse 96 workers.
     - Peak RSS should be below the pooled run (one subclass's rows + its control block at a time).
     - One `<subclass>.rds` passes `brisc_adaptation/validation/check_dictionary_rds.R`.
   - [ ] Decide from `summary.tsv`: keep `min_cells=10` and `min_control_cells=30`? If the loop is
     slow, run subclasses concurrently (a future change, not needed for correctness).
   - [DONE 2026-10-14, job 2562632] **Full per-subclass run** (14 files, 6,127,364 cells, 11,736
     features): exit 0, **1 h 40 min**, peak 431 GB in the log (508 GB Slurm MaxRSS), empty `.err`.
     **84 of 321 subclasses wrote a dictionary**, with 6,618 fingerprints kept out of 74,878 eligible
     pairs (median 45 per subclass, max 548). All 84 pass the R check. Details in `decisions_log.md`
     §8. Output: `/scratch/mfafouti/Perurb-seq_WMB/dict_wholebrain_by_subclass/`.
6. [ ] Phase B per subclass in `/scratch/mfafouti/RNA-fingerprinting-satija`: subset the query by
   subclass, then run `FingerprintCells` with the matching `<subclass>.rds` and that subclass's query controls.
7. (Optional, separate change) introduce `config.conf` for paths/tunables.

## Verification
- Smoke run exits `DICT_EXIT=0`, `.rds` loads in the R container (`readRDS`, class `dictionary`,
  `Lambdas` dims = features x perturbations, `names(dists) == colnames(Lambdas)`).
- Full run: peak memory < 700G, number of learned perturbations logged.
- Per-subclass run: `summary.tsv` has one row per subclass, and every non-empty `file` entry exists
  and passes the R check.
