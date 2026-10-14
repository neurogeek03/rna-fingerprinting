# Plan: one dictionary per subclass (match the authors' method)

## Context
The preprint learns fingerprints separately per cell type (Immune Dictionary Methods: "estimated
fingerprints separately for each cell type, using the PBS cells as controls, and excluding cell types
with fewer than 30 PBS cells") and maps each query cell type to its own dictionary. The port pooled all
~300 subclasses into one baseline and one fingerprint per knockout. Decision: split by `predicted_subclass`.

## Approach
1. DONE — `stacked.load_qc_blocks`: optional extra obs columns (`obs_columns`) returned as `obs` dict.
2. DONE — `learn_dictionary.py`: factor label processing into `_process_labels`; add generator
   `iter_dictionaries_by_group(Xf, labels, groups, features, min_control_cells=30, ...)`. Per group:
   rows → `Xf[rows]` (one subclass copied at a time) → per-subclass gene mean/SD → unchanged `fit()`.
   Groups with < `min_control_cells` controls are skipped. Yields `(group, stats, result)`.
3. DONE — CLI `run_learn_dictionary.py`: `--group-column`, `--min-control-cells`. With a group column,
   `--out` is a directory: one `<subclass>.rds` per kept subclass + `summary.tsv`. Per-file loader only.
4. DONE — sbatch: `GROUP_COLUMN` (default `predicted_subclass`; empty = pooled), `OUT` is a directory.
5. DONE — test `python/tests/test_by_group.py`: group loop == direct fit on the subset; small groups skipped.
6. DONE — docs: `decisions_log.md` entry, `pipeline_overview.md` + schematic, `why_brisc_full_atlas.md`.

## Kept global (on purpose)
- Feature set and log1pPF target: shared across subclasses so all dictionaries have the same rows.
- `fit()` itself: unchanged.

## Not in scope
- Phase B per subclass (subset query by subclass, `FingerprintCells` with the matching `.rds`) lives in
  the consuming project `/scratch/mfafouti/RNA-fingerprinting-satija`.

## Verification
- Run `test_by_group.py` in the BRISC container.
- Before the full run: check `summary.tsv` from a 1–2 file smoke test for controls / knockouts per subclass.
