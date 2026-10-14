# BRISC adaptation — future TODOs (deferred, not blocking)

Ideas considered and intentionally deferred. None block the current pipeline, which uses BRISC
z-score(log1pPF) scaling (see `decisions_log.md`).

---

## 1. Analytic Pearson residuals as an alternative scaling (deferred)

**Status:** deferred 2026-10-14. Current pipeline keeps z-score(log1pPF); accepted as sufficient.

**Motivation.** The R `fit()` consumes SCTransform's Pearson residuals as `scale.data`. Our port uses
BRISC `normalize(log1pPF)` + global per-gene z-score instead. On the CPA example
(`data/cpa_perturb_seq.rda`), the Python fingerprints correlate with the real R
`LearnDictionary` output at **signed Pearson r ≈ 0.87** (CPSF6 +0.866, PABPN1 +0.864; top-20 gene
overlap 11–12/20), same perturbations kept, same sign. The ~0.13 gap is almost entirely the
scaling choice: SCTransform's mean-variance-based residuals weight genes differently than
z-score(log1pPF). **We accept 0.87** — the dictionary captures the same dominant biology and the
same perturbation set; the goal is a standardized BRISC-native dictionary, not a bit-for-bit R clone.

**If we ever want tighter agreement with R (~0.95+):** implement **analytic Pearson residuals**
(Lause, Kobak & Berens 2021 — the closed form SCTransform approximates), *not* a full SCTransform
reimplementation and *not* `pysctransform` (extra fragile dependency, same dense-output cost, still
not bit-identical to Seurat).

**Design — lazy, per-subset (memory-safe, same profile as the z-score path):**
- Pearson residuals are *dense*, so never materialize the full genes×cells matrix.
- Precompute once from the sparse counts (all small vectors): `cell_total[c]` (per-cell depth over
  all genes), `gene_total[g]`, grand total `N`, and overdispersion `theta` (≈100 or cheaply estimated).
- Compute residuals only per cell-subset that `fit()` requests (control cells for PCA; one
  perturbation at a time):
  `mu = outer(cell_total[idx], gene_total[feat]) / N`;
  `residual = (counts_block - mu) / sqrt(mu + mu**2/theta)`; clip to `±sqrt(n_cells)`.
- Cleanest implementation: a `Scaler` abstraction with `.block(idx) -> dense (len(idx) x k)`;
  `ZScoreScaler` (current) and `PearsonScaler` (new); `fit()` calls `scaler.block(idx)` wherever it
  now densifies+scales, so the core algorithm is untouched. Thread a `scaling='zscore'|'pearson'`
  option through `learn_dictionary`.

**How to validate:** reuse `brisc_adaptation/validation/` as-is (same CPA data, same 896 features,
same control) and just set `scaling='pearson'` in `compare_python.py`; expect r to move 0.87 → ~0.95.
