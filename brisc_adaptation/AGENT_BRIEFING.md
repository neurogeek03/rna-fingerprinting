# Agent briefing — BRISC port of rna-fingerprinting

Read this first if you're a new agent or setting up on a new cluster. It is the portable summary of
the work (the authoritative detail is in `decisions_log.md`; deferred ideas in `future_todos.md`).
For the method in plain terms, see `docs/pipeline_overview.md`; for why the full atlas matters,
`docs/why_brisc_full_atlas.md`; for `fit()` step by step, `docs/fit_schematic.md`.

## Goal

`rna-fingerprinting` (Satija lab, R) maps transcriptional responses to perturbation dictionaries via
SuSiE. Its `SCTransform` + `LearnDictionary()`/`fit()` path materializes a dense genes×cells
`scale.data` (~200GB at 6M cells) and OOMs. This fork reimplements those two steps in **Python on
BRISC** so a dictionary can be learned on the **full ~6M QC-passing whole-brain Perturb-seq atlas**
without downsampling. Everything downstream (`classify()`/`FingerprintCells()`, `susie_plus()`,
`SummarizeResults()`, visualization) **stays in R** and consumes the `.rds` dictionary this produces.

## Status: implemented & validated

- Port complete: `python/brisc_fingerprinting/` (`scaling.py`, `fit.py`, `learn_dictionary.py`,
  `choose_features.py`, `perturbations.py`, `rds_io.py`), CLI `python/scripts/run_learn_dictionary.py`.
- Validated end-to-end on synthetic data, on one real h5ad, and **numerically vs the real R
  `LearnDictionary`** on the CPA example: signed Pearson r ≈ 0.87 per fingerprint, same perturbations
  kept, matching sign. Accepted (see `decisions_log.md` §2, §8). Tighter agreement (analytic Pearson
  residuals, ~0.95+) is deferred in `future_todos.md`.
- **Remaining deliverable: the full 14-file atlas run** (see "Run it" below).

## How it works (one process)

`run_learn_dictionary.py` does it all in a single process:
load h5ads (counts + obs + var only) → QC-filter → choose features → normalize (log1pPF) →
collapse perturbation labels → `fit()` (control 30-PC PCA, per-perturbation baseline-subtract →
rank-1 PCA → Wilcoxon sign/filter → dist estimation) → write R-compatible `.rds` via `ryp`.

Key design decisions already made (defaults in the code):
- **Scaling** = BRISC `normalize(log1pPF)` + global per-gene z-score (SCTransform-residual stand-in).
- **SVD** = `sklearn.utils.extmath.randomized_svd`; **parallelism** = `joblib` threading.
- **Single-gene dictionary**: `collapse_to_single_gene` strips control tokens (`Negative`,
  `Non_target`, `Safe_target_*`), collapses `Gene|control → Gene`, drops true multi-gene combos.
- **Control class** = `Negative` + `Non_target` + `Safe_target_*` pooled. (`Negative` = no-guide cells.)
- Memory-safety: never materialize the dense full matrix; densify only per cell-subset inside `fit()`.

## Per-cluster config checklist (do this after cloning)

1. **Container** — pull the public image on a **compute node** (login-node unpack onto networked
   scratch is pathologically slow):
   ```
   module load apptainer
   export APPTAINER_CACHEDIR=$SCRATCH/.apptainer/cache APPTAINER_TMPDIR=$SLURM_TMPDIR
   apptainer pull mybrisc.sif docker://marlenfafouti/brisc:bioc-v2-mkl-seurat-sklearn
   ```
   Set `SIF=/path/to/mybrisc.sif`.
2. **Data** — stage the 14 Perturb-seq h5ads to this cluster; set `H5AD_DIR=/path/to/h5ads`.
3. **Slurm account** — set it in `python/sbatch/learn_dictionary.sbatch` (`--account=CHANGE_ME`) or
   pass `sbatch --account=...`.
4. **Module name** — if apptainer's module isn't plain `apptainer`, set `APPTAINER_MODULE=apptainer/x.y`.
5. **Memory** — the full atlas needs a **large-memory node** (see sizing below).

## Run it

Interactive (inside a large-mem allocation):
```
salloc --nodes=1 --ntasks=1 --cpus-per-task=48 --mem=750G --time=02:00:00 --account=<acct> srun --pty bash
SIF=/path/mybrisc.sif H5AD_DIR=/path/h5ads bash brisc_adaptation/run_dictionary_interactive.sh
```
Batch:
```
REPO=$PWD SIF=/path/mybrisc.sif H5AD_DIR=/path/h5ads \
  sbatch --account=<acct> python/sbatch/learn_dictionary.sbatch
```
Quick sanity run: point `H5AD_DIR` at a 1-file subset (needs far less memory).

## Memory sizing (measured on the full atlas)

6,127,364 QC cells, 32.6e9 nonzeros, float32 + uint16 indices = **~196 GB per copy**, **~392 GB
transient peak** (concat / normalize / build). Request ~750 GB (nibi nodes are 766 GB; 6 TB largemem
nodes exist but are unnecessary). **Validate memory like this:** the first OOMs were from (a) BRISC
loading `obsp`/`obsm`/etc — `load_qc` now skips them (`obsm=False, obsp=False, varm=False, varp=False,
uns=False`) and QC-filters each file before concat; (b) a `comp`-style 240 GB cap — too small for even
one copy.

## Gotchas (learned the hard way — see decisions_log.md)

- Pull containers on a compute node with `APPTAINER_TMPDIR=$SLURM_TMPDIR`; don't let apptainer cache to
  `$HOME` (small quota). Don't `pkill -f apptainer` (it matches its own shell and self-kills).
- `normalize()` errors unless `uns['QCed']` is set — we call `skip_qc()` (we QC by hand).
- BRISC `find_markers` returns `['cell_type','gene','detection_rate','fold_change']` — **no p-value**;
  `choose_features(method='de')` ranks by `fold_change`.
- Expression feature selection must run on **raw counts** (before `normalize`); the CLI does this via
  the `load_qc` → choose → `normalize_counts` split.
- `find_markers` rejects `cell_types` and `excluded_cell_types` together.

## Open items

- [ ] Full-atlas run (the deliverable) and a sanity check of the resulting dictionary.
- [ ] Decide feature-selection for the real dictionary (expression threshold vs DE vs curated); the
      expression path has no hard gene cap yet.
- [ ] Confirm `Safe_target_*`-as-control is acceptable.
- [ ] Deferred: analytic Pearson residuals for tighter R agreement (`future_todos.md`).
- [ ] Optional memory win: `filter_var` to the selected features right after choosing them, so the
      normalized/densified matrices only carry k genes (would let the full run fit on smaller nodes).
