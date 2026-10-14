# Plan: deepen "why full atlas" doc + add plain-language pipeline overview

## Context
`brisc_adaptation/docs/why_brisc_full_atlas.md` asserts that more cells help Wilcoxon, PCA, MAD and `dists`,
but only in one-line table cells. The user wants the *mechanism* explained (and why each step exists at all),
plus a zoomed-out, plain-language doc of the whole pipeline so the case for BRISC is easy to appreciate.

## Changes

### 1. DONE — New `brisc_adaptation/docs/pipeline_overview.md` (plain language, read first)
- **The question**: given a query response (e.g. disease vs control cells), which gene knockouts does it look like?
- **Two phases**: (A) learn a dictionary from Perturb-seq (`fit()`, needs all 6M cells — the heavy part,
  ported to BRISC); (B) match query cells to it (`classify()` + SuSiE, ~100k cells, stays in R).
- **Phase A in 5 everyday steps**: normalize → learn "normal variation" from controls (PCA) → subtract it from
  each knockout's cells → the main leftover direction = fingerprint (rank-1 PCA, drop outlier cells) →
  keep only fingerprints that really differ from controls (Wilcoxon), record how certain each gene's value is (`dists`).
- **Phase B in 3 steps**: subtract baseline from query → regress on fingerprints (SuSiE) → calibrate with `dists`.
- **Where the memory wall is** (dense SCTransform matrix ≈ 240 GB) and what BRISC changes (sparse, z-score on demand).
- Short glossary (control, residual, Lambda/fingerprint, ell/score, MAD, dists). Links to `fit_schematic.md` and the why doc.

### 2. DONE — Expand `why_brisc_full_atlas.md` with a section "Why each step needs cells"
Keep the existing tables; add per-step subsections, each = *what it does / why it exists / how n enters / numbers*:
- **Wilcoxon** — three jobs: orient Lambda's sign (SVD sign is arbitrary), choose among outlier-loop iterations
  (`track_cors`), filter noise. Power depends on effective n = n_p·1000/(n_p+1000) because the control
  reference is fixed at 1,000 cells: ~20 → 130 → 670 (downsample → full). **Correct the existing "3.6×" claim to
  ~2.3× z gain over the downsample**, and note raising the 1,000-cell reference is a cheap extra lever (future_todos).
- **Control PCA** — rare subclasses need enough controls for their axes to be among the top 30; otherwise
  cell-type signal leaks into every residual and becomes fake fingerprint signal.
- **Rank-1 PCA** — spiked-covariance (BBP) threshold: a fingerprint is only recoverable when its strength exceeds
  ~√(features/cells): √(12k/150) ≈ 9 vs √(12k/2000) ≈ 2.4 → weaker knockouts become detectable at all.
- **MAD loop** — `mad(constant=1)` on 20–150 points is unstable; a few strong-responder cells dominate rank-1 and
  get flagged, removals can push below `min_cells`. Large n → MAD stable, single cells can't dominate.
- **dists** — SE of means ∝ 1/√Σell² ∝ 1/√n; `classify()` resamples Lambdas from N(mean, sd) 50× for calibration,
  so noisy dists → miscalibrated PIPs/credible sets.
- Add a pointer to `pipeline_overview.md` at the top.

### 3. DONE — Housekeeping
- Link `pipeline_overview.md` from `AGENT_BRIEFING.md` (docs list) and from `fit_schematic.md` intro.
- Save this plan to `brisc_adaptation/docs/plan_why_brisc_docs.md` (CLAUDE.md rule), mark steps DONE.

## Verification
- Cross-check every line ref/claim against `R/fit.R`, `R/classify.R`, `python/brisc_fingerprinting/fit.py`
  (1,000-cell ref, `constant=1` MAD, 50 resamples).
- Check relative links resolve (`ls` targets).
