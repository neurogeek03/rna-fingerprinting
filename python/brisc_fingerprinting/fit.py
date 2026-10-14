"""Core fingerprint-learning algorithm: a faithful Python port of ``R/fit.R``.

Orientation note: the R code works with ``scale.data`` as features x cells and calls
``prcomp_irlba(t(...))``. Here the natural orientation coming out of a cells x genes sparse matrix
is cells x features, so we densify each cell-subset as (n_cells x k) -- which is already the
``t(...)`` that ``prcomp_irlba`` consumes. Residuals are transposed back to features x cells only
where the R code needs that layout (Wilcoxon, Lambda bookkeeping, dist estimation).
"""

from __future__ import annotations

import os
import re

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.utils.extmath import randomized_svd
from threadpoolctl import threadpool_limits

from .logs import log


def _mad(x):
    """Median absolute deviation with constant=1 (matches R ``mad(..., constant=1)``)."""
    return np.median(np.abs(x - np.median(x)))


def _mad_scores(ell):
    med = np.median(ell)
    m = _mad(ell)
    if m == 0:
        return np.zeros_like(ell)
    return (ell - med) / m


def _scaled_block(Xf, idx, gene_mean, gene_sd):
    """Densify rows ``idx`` of ``Xf`` and z-score them, in float32 (n x k == t(scale.data[, idx]))."""
    block = Xf[idx].toarray().astype(np.float32, copy=False)
    block -= gene_mean
    block /= gene_sd
    return block


def _rank1(residual_cells_by_feat):
    """Rank-1 SVD of (cells x features), no centering (R: ``prcomp_irlba(t(res), center=F, n=1)``).

    Returns ``(Lam, ell)`` where ``Lam`` is the feature loading (length k) and ``ell`` the per-cell
    score (length n_cells). SVD sign is arbitrary here; it is fixed later by sign correction.
    """
    U, S, Vt = randomized_svd(residual_cells_by_feat, n_components=1, random_state=0)
    Lam = Vt[0]
    ell = U[:, 0] * S[0]
    return Lam, ell


def _row_keys(x_fx, row_offset: int = 0):
    """Exact, order-preserving uint64 sort keys for a features x cells float array.

    High 32 bits = feature (row) index, low 32 bits = the float32 value mapped to a uint32 whose
    unsigned order equals the float order (flip all bits of negatives, set the sign bit of
    non-negatives). Sorting / searching the flattened keys therefore sorts / searches every row
    independently in one vectorized call, with ties preserved exactly.
    """
    x = np.ascontiguousarray(x_fx, dtype=np.float32) + np.float32(0.0)  # -0.0 -> +0.0
    bits = x.view(np.uint32)
    sortable = np.where(bits & np.uint32(0x80000000), ~bits, bits | np.uint32(0x80000000))
    rows = np.arange(row_offset, row_offset + x.shape[0], dtype=np.uint64)[:, None]
    return (rows << np.uint64(32)) | sortable.astype(np.uint64)


def _sorted_control_keys(ctrl_res_fx):
    """Precompute (once per fit) the sorted per-feature keys of the control residual reference."""
    return np.sort(_row_keys(ctrl_res_fx).ravel())


def _wilcox_per_feature(ctrl_keys, n1, pert_res_fx, chunk: int = 1024):
    """Vectorized Mann-Whitney U (control group) per feature over control vs perturbed residuals.

    Mirrors ``presto::wilcoxauc(cbind(ctrl.residual, residual), y=Control/Perturbed)`` restricted to
    ``group=='Control'``: returns the U ``statistic`` for the control group and a two-sided ``pval``
    (normal approximation, as in large-sample ``wilcox.test``).

    Uses U_control = #(control > perturbed pairs) + 0.5 * #(ties), which equals the rank-sum
    ``R1 - n1(n1+1)/2``, but needs no combined matrix and no re-ranking of the (fixed) controls:
    ``ctrl_keys`` (from :func:`_sorted_control_keys`, features x n1 flattened) is sorted once per
    fit, and each perturbed value is located in its feature's controls by binary search. Features
    are processed in chunks to bound temporaries.

    ``pert_res_fx`` is features x n2.
    """
    k, n2 = pert_res_fx.shape
    U1 = np.empty(k, dtype=np.float64)
    for g0 in range(0, k, chunk):
        g1 = min(k, g0 + chunk)
        ctrl = ctrl_keys[g0 * n1:g1 * n1]                    # this chunk's rows, sorted per row
        q = _row_keys(pert_res_fx[g0:g1], row_offset=g0)     # (g1-g0) x n2
        start = ((np.arange(g0, g1, dtype=np.int64) - g0) * n1)[:, None]
        left = np.searchsorted(ctrl, q, side="left") - start   # controls <  p, per row
        right = np.searchsorted(ctrl, q, side="right") - start  # controls <= p, per row
        # per perturbed value p: controls > p  +  0.5 * controls == p
        U1[g0:g1] = (n1 * n2 - right.sum(axis=1, dtype=np.float64)
                     + 0.5 * (right - left).sum(axis=1, dtype=np.float64))

    mu = n1 * n2 / 2.0
    # No-tie variance: residuals are continuous floats, so ties are effectively absent and the exact
    # tie correction is negligible. Dropping it removes a per-feature np.unique loop that dominated
    # runtime. Same value across features, so this is a scalar.
    N = n1 + n2
    var = n1 * n2 * (N + 1) / 12.0
    sd = np.sqrt(var)
    z = (U1 - mu) / sd if sd > 0 else np.zeros_like(U1)
    pval = 2.0 * stats.norm.sf(np.abs(z))
    return U1, pval


def _fit_one_limited(blas_threads, *args):
    """Run :func:`_fit_one` with this worker thread's BLAS limited to ``blas_threads`` threads."""
    with threadpool_limits(limits=blas_threads, user_api="blas"):
        return _fit_one(*args)


def _fit_one(
    name,
    idx,
    Xf,
    gene_mean,
    gene_sd,
    rotation,
    ctrl_mus,
    ctrl_res_fx,
    ctrl_keys,
    features,
    min_cells,
    override_filter,
):
    """Learn (or reject) the fingerprint for a single perturbation. Returns ``(Lam, dists)`` or None."""
    try:
        # Densify this perturbation's cells and z-score -> scaled (n_p x k) == t(pdev).
        scaled = _scaled_block(Xf, idx, gene_mean, gene_sd)
        # Baseline subtraction: project centered data onto control PCs and reconstruct.
        centered = scaled - ctrl_mus
        scores = centered @ rotation                      # n_p x 30
        reconstruct = scores @ rotation.T + ctrl_mus      # n_p x k
        residual = scaled - reconstruct                   # n_p x k  (== t(residual) in R)
        residual_fx = residual.T                          # k x n_p  (== residual in R)

        # Wilcoxon reference (computed once, as in fit.R) for sign/quality.
        statistic, pval = _wilcox_per_feature(ctrl_keys, ctrl_res_fx.shape[1], residual_fx)

        # Initial rank-1 fit + outlier bookkeeping.
        Lam, ell = _rank1(residual)
        mads = _mad_scores(ell)
        outlier = np.max(np.abs(mads)) >= 10

        def _cor(a, b):
            if np.std(a) == 0 or np.std(b) == 0:
                return 0.0
            return float(np.corrcoef(a, b)[0, 1])

        track_cors = [_cor(Lam, statistic)]
        # store: (Lam, ell, positions-into-residual-rows)
        store = [(Lam, ell, np.arange(residual.shape[0]))]

        while outlier:
            pts = np.abs(mads) >= 10
            keep_pos = store[-1][2][~pts]
            if len(keep_pos) < min_cells:
                break
            res_sub = residual[keep_pos]
            Lam, ell = _rank1(res_sub)
            mads = _mad_scores(ell)
            outlier = np.max(np.abs(mads)) >= 10
            track_cors.append(_cor(Lam, statistic))
            store.append((Lam, ell, keep_pos))

        # Pick the Lambda most correlated (in magnitude) with the Wilcoxon statistics.
        pick = int(np.argmax(np.abs(track_cors)))
        Lam, ell, keep_pos = store[pick]
        residual_fx_pick = residual_fx[:, keep_pos]       # k x n_kept

        # Sign convention: Lambda anti-correlated with control-vs-perturbed statistics.
        if track_cors[pick] > 0:
            Lam = -Lam
            ell = -ell

        # Quality filter.
        proj = Lam @ ctrl_res_fx                          # length n1 (control cells)
        score = stats.ranksums(ell, proj).pvalue
        top10 = np.argsort(np.abs(Lam))[::-1][:10]
        count = int(np.sum(pval[top10] < 0.05))
        keep = (score < 0.05) and (count >= 2)

        if not (keep or override_filter):
            return None

        # Estimate per-feature means + sds of the rank-1 model (used by classify() calibration).
        XtX = float(np.sum(ell**2))
        XtY = residual_fx_pick @ ell                      # length k
        means = XtY / XtX
        fitted = np.outer(means, ell)                     # k x n_kept
        sigma_squared = np.sum((residual_fx_pick - fitted) ** 2, axis=1) / (len(ell) - 1)
        sds = np.sqrt((1.0 / XtX) * sigma_squared)
        dists = np.column_stack([means, sds])             # k x 2
        return name, Lam, dists
    except Exception as e:  # mirror R: a failed perturbation is simply dropped
        log(f"[fit] perturbation {name!r} failed: {type(e).__name__}: {e}")
        return None


def fit(
    Xf,
    gene_mean,
    gene_sd,
    perturbation,
    features,
    control_label: str = "control",
    min_cells: int = 10,
    override_filter: bool = False,
    n_jobs: int = 1,
    seed: int | None = 1448145,
    exclude_label: str | None = None,
):
    """Learn fingerprints for every eligible perturbation.

    Parameters
    ----------
    Xf : scipy.sparse.csr_matrix
        Normalized matrix restricted to ``features`` (cells x k), from ``build_feature_matrix``.
    gene_mean, gene_sd : np.ndarray
        Global per-gene mean / sd (length k), from ``gene_mean_sd``.
    perturbation : np.ndarray[str]
        Per-cell perturbation labels; control cells labeled ``control_label``.
    features : list[str]
        Feature names (length k), matching the columns of ``Xf``.
    control_label : str
        Label marking control cells (default ``'control'``).
    min_cells : int
        Minimum cells for a perturbation to be considered (default 10).
    override_filter : bool
        Keep low-signal fingerprints instead of filtering them out (default False).
    n_jobs : int
        Threads for the per-perturbation loop (joblib threading backend; ``Xf`` is shared read-only).
    seed : int or None
        Seed for the 1,000-cell control-residual subsample.
    exclude_label : str or None
        Label marking cells to ignore entirely (neither control nor perturbation). Lets callers drop
        cells by relabeling instead of ``filter_obs``, which would copy the whole matrix.

    Returns
    -------
    dict or None
        ``{'Lambdas': k x P array, 'dists': list of k x 2 arrays, 'names': list[str]}``, or None if
        no perturbation survived.
    """
    features = list(features)
    k = len(features)
    perturbation = np.asarray(perturbation)
    rng = np.random.default_rng(seed)

    ctrl_idx = np.where(perturbation == control_label)[0]
    if ctrl_idx.size == 0:
        raise ValueError("No control cells found.")
    log(f"[fit] densifying {ctrl_idx.size} control cells x {k} features (float32)")

    # Control baseline: 30-PC PCA on z-scored control cells (cells x k). prcomp centers columns,
    # so we subtract the control mean before the SVD and keep that mean as ctrl_mus. This is the one
    # full-control dense block (~2.4M x k at full-atlas scale): float32 and centered in place, so only
    # one copy exists.
    ctrl_centered = _scaled_block(Xf, ctrl_idx, gene_mean, gene_sd)  # n_ctrl x k
    ctrl_mus = ctrl_centered.mean(axis=0, dtype=np.float64).astype(np.float32)  # length k
    ctrl_centered -= ctrl_mus
    n_pc = min(30, min(ctrl_centered.shape) - 1)
    log(f"[fit] control PCA ({n_pc} PCs)")
    U, S, Vt = randomized_svd(ctrl_centered, n_components=n_pc, random_state=0)
    rotation = Vt.T.astype(np.float32)                             # k x n_pc (loadings)
    ctrl_scores = (U * S).astype(np.float32)                       # n_ctrl x n_pc

    # Control residual reference (up to 1,000 cells) after baseline removal -> features x samp.
    if ctrl_idx.size <= 1000:
        samp = np.arange(ctrl_idx.size)
    else:
        samp = rng.choice(ctrl_idx.size, size=1000, replace=False)
    # scaled - (scores @ R^T + mus) == centered - scores @ R^T
    ctrl_res_fx = (ctrl_centered[samp] - ctrl_scores[samp] @ rotation.T).T  # k x samp
    del ctrl_centered, U
    # Sorted per-feature control keys for the Wilcoxon: the 1,000 reference controls are the same for
    # every perturbation, so sort them once here instead of re-ranking them in every task.
    ctrl_keys = _sorted_control_keys(ctrl_res_fx)
    log("[fit] control baseline done")

    # Perturbations with enough cells.
    mask = perturbation != control_label
    if exclude_label is not None:
        mask &= perturbation != exclude_label
    labels, counts = np.unique(perturbation[mask], return_counts=True)
    perturbations = list(labels[counts >= min_cells])
    too_few = list(labels[counts < min_cells])
    if not perturbations:
        log("[fit] No perturbations found with more than minimum number of cells")
        return None
    if too_few:
        log(f"[fit] Too few cells (<{min_cells}) for {len(too_few)} perturbations: "
            f"{', '.join(map(str, too_few))}")

    # Precompute row indices per perturbation.
    idx_by_pert = {p: np.where(perturbation == p)[0] for p in perturbations}

    n_total = len(perturbations)
    log(f"[fit] learning fingerprints for {n_total} perturbations ({n_jobs} threads x "
        f"{max(1, (os.cpu_count() or 1) // max(1, n_jobs))} BLAS)")
    # Per-perturbation loop (embarrassingly parallel, as in R pbmclapply). The threading backend
    # shares the read-only sparse ``Xf`` across workers; the numpy/scipy heavy lifting releases the GIL.
    # ``return_as='generator'`` yields results in input order as they finish, for progress logging.
    # BLAS gets cores // n_jobs threads per worker so workers x BLAS threads == cores. The limit must
    # be set INSIDE each worker thread: with MKL's Intel-OpenMP layer the BLAS thread count is
    # per-thread, so a threadpool_limits() in the main thread does not reach joblib's workers, which
    # keep the startup OMP/MKL_NUM_THREADS (all cores, for the control PCA above). Without this every
    # worker ran 192-thread BLAS (96 x 192 threads on 192 cores) -- see python/tests/check_thread_limits.py
    # and brisc_adaptation/docs/thread_scaling_paradox.md.
    results = []
    step = max(1, n_total // 10)
    blas_per_job = max(1, (os.cpu_count() or 1) // max(1, n_jobs))
    gen = Parallel(n_jobs=n_jobs, backend="threading", return_as="generator")(
        delayed(_fit_one_limited)(
            blas_per_job, p, idx_by_pert[p], Xf, gene_mean, gene_sd, rotation, ctrl_mus,
            ctrl_res_fx, ctrl_keys, features, min_cells, override_filter,
        )
        for p in perturbations
    )
    for i, r in enumerate(gen, 1):
        results.append(r)
        if i % step == 0 or i == n_total:
            n_kept = sum(x is not None for x in results)
            log(f"[fit] {i}/{n_total} perturbations done, {n_kept} kept so far")

    # Perturbation name cleaning (R: replace [,;_] with '-', then make unique).
    raw_names = [r[0] for r in results if r is not None]
    cleaned = [re.sub(r"[,;_]", "-", str(n)) for n in raw_names]
    if cleaned != raw_names:
        log("[fit] Replacing commas, underscores, and semicolons in perturbation names with dashes")
    cleaned = _make_unique(cleaned)

    kept = [r for r in results if r is not None]
    if not kept:
        log("[fit] No perturbations with sufficiently high signal found; set override_filter=True to keep all")
        return None

    Lambdas = np.column_stack([r[1] for r in kept])               # k x P
    dists = [r[2] for r in kept]
    names = cleaned

    log(f"[fit] Learned {Lambdas.shape[1]} of {len(perturbations)} possible perturbations")
    return {"Lambdas": Lambdas, "dists": dists, "names": names, "features": features}


def _make_unique(names):
    """R ``make.unique`` semantics: first occurrence unchanged, later ones get .1, .2, ..."""
    seen = {}
    out = []
    for n in names:
        if n not in seen:
            seen[n] = 0
            out.append(n)
        else:
            seen[n] += 1
            out.append(f"{n}.{seen[n]}")
    return out
