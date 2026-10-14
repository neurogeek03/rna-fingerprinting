"""Core fingerprint-learning algorithm: a faithful Python port of ``R/fit.R``.

Orientation note: the R code works with ``scale.data`` as features x cells and calls
``prcomp_irlba(t(...))``. Here the natural orientation coming out of a cells x genes sparse matrix
is cells x features, so we densify each cell-subset as (n_cells x k) -- which is already the
``t(...)`` that ``prcomp_irlba`` consumes. Residuals are transposed back to features x cells only
where the R code needs that layout (Wilcoxon, Lambda bookkeeping, dist estimation).
"""

from __future__ import annotations

import re

import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from sklearn.utils.extmath import randomized_svd


def _mad(x):
    """Median absolute deviation with constant=1 (matches R ``mad(..., constant=1)``)."""
    return np.median(np.abs(x - np.median(x)))


def _mad_scores(ell):
    med = np.median(ell)
    m = _mad(ell)
    if m == 0:
        return np.zeros_like(ell)
    return (ell - med) / m


def _rank1(residual_cells_by_feat):
    """Rank-1 SVD of (cells x features), no centering (R: ``prcomp_irlba(t(res), center=F, n=1)``).

    Returns ``(Lam, ell)`` where ``Lam`` is the feature loading (length k) and ``ell`` the per-cell
    score (length n_cells). SVD sign is arbitrary here; it is fixed later by sign correction.
    """
    U, S, Vt = randomized_svd(residual_cells_by_feat, n_components=1, random_state=0)
    Lam = Vt[0]
    ell = U[:, 0] * S[0]
    return Lam, ell


def _wilcox_per_feature(ctrl_res_fx, pert_res_fx):
    """Vectorized Mann-Whitney U (control group) per feature over control vs perturbed residuals.

    Mirrors ``presto::wilcoxauc(cbind(ctrl.residual, residual), y=Control/Perturbed)`` restricted to
    ``group=='Control'``: returns the U ``statistic`` for the control group and a two-sided ``pval``
    (normal approximation with tie correction, as in large-sample ``wilcox.test``).

    Both inputs are features x cells.
    """
    n1 = ctrl_res_fx.shape[1]  # control cells
    n2 = pert_res_fx.shape[1]  # perturbed cells
    M = np.concatenate([ctrl_res_fx, pert_res_fx], axis=1)  # features x (n1+n2)
    ranks = stats.rankdata(M, axis=1)
    R1 = ranks[:, :n1].sum(axis=1)
    U1 = R1 - n1 * (n1 + 1) / 2.0  # U statistic for the control group

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


def _fit_one(
    name,
    idx,
    Xf,
    gene_mean,
    gene_sd,
    rotation,
    ctrl_mus,
    ctrl_res_fx,
    features,
    min_cells,
    override_filter,
):
    """Learn (or reject) the fingerprint for a single perturbation. Returns ``(Lam, dists)`` or None."""
    try:
        # Densify this perturbation's cells and z-score -> scaled (n_p x k) == t(pdev).
        scaled = (Xf[idx].toarray() - gene_mean) / gene_sd
        # Baseline subtraction: project centered data onto control PCs and reconstruct.
        centered = scaled - ctrl_mus
        scores = centered @ rotation                      # n_p x 30
        reconstruct = scores @ rotation.T + ctrl_mus      # n_p x k
        residual = scaled - reconstruct                   # n_p x k  (== t(residual) in R)
        residual_fx = residual.T                          # k x n_p  (== residual in R)

        # Wilcoxon reference (computed once, as in fit.R) for sign/quality.
        statistic, pval = _wilcox_per_feature(ctrl_res_fx, residual_fx)

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
        print(f"[fit] perturbation {name!r} failed: {type(e).__name__}: {e}")
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

    print("Processing data...")
    ctrl_idx = np.where(perturbation == control_label)[0]
    if ctrl_idx.size == 0:
        raise ValueError("No control cells found.")

    # Control baseline: 30-PC PCA on z-scored control cells (cells x k). prcomp centers columns,
    # so we subtract the control mean before the SVD and keep that mean as ctrl_mus.
    ctrl_scaled = (Xf[ctrl_idx].toarray() - gene_mean) / gene_sd   # n_ctrl x k
    ctrl_mus = ctrl_scaled.mean(axis=0)                            # length k
    n_pc = min(30, min(ctrl_scaled.shape) - 1)
    U, S, Vt = randomized_svd(ctrl_scaled - ctrl_mus, n_components=n_pc, random_state=0)
    rotation = Vt.T                                                # k x n_pc (loadings)
    ctrl_scores = U * S                                            # n_ctrl x n_pc

    # Control residual reference (up to 1,000 cells) after baseline removal -> features x samp.
    if ctrl_idx.size <= 1000:
        samp = np.arange(ctrl_idx.size)
    else:
        samp = rng.choice(ctrl_idx.size, size=1000, replace=False)
    ctrl_recon = ctrl_scores[samp] @ rotation.T + ctrl_mus        # samp x k
    ctrl_res_fx = (ctrl_scaled[samp] - ctrl_recon).T              # k x samp

    # Perturbations with enough cells.
    mask = perturbation != control_label
    labels, counts = np.unique(perturbation[mask], return_counts=True)
    perturbations = list(labels[counts >= min_cells])
    too_few = list(labels[counts < min_cells])
    if not perturbations:
        print("No perturbations found with more than minimum number of cells")
        return None
    if too_few:
        print(f"Too few cells for perturbations: {', '.join(map(str, too_few))}")

    # Precompute row indices per perturbation.
    idx_by_pert = {p: np.where(perturbation == p)[0] for p in perturbations}

    print("Learning fingerprints...")
    # Per-perturbation loop (embarrassingly parallel, as in R pbmclapply). The threading backend
    # shares the read-only sparse ``Xf`` across workers; the numpy/scipy heavy lifting releases the GIL.
    results = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(_fit_one)(
            p, idx_by_pert[p], Xf, gene_mean, gene_sd, rotation, ctrl_mus,
            ctrl_res_fx, features, min_cells, override_filter,
        )
        for p in perturbations
    )

    # Perturbation name cleaning (R: replace [,;_] with '-', then make unique).
    raw_names = [r[0] for r in results if r is not None]
    cleaned = [re.sub(r"[,;_]", "-", str(n)) for n in raw_names]
    if cleaned != raw_names:
        print("Replacing commas, underscores, and semicolons in perturbation names with dashes")
    cleaned = _make_unique(cleaned)

    kept = [r for r in results if r is not None]
    if not kept:
        print("No perturbations with sufficiently high signal found; set override_filter=True to keep all")
        return None

    Lambdas = np.column_stack([r[1] for r in kept])               # k x P
    dists = [r[2] for r in kept]
    names = cleaned

    print(f"Learned {Lambdas.shape[1]} of {len(perturbations)} possible perturbations")
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
