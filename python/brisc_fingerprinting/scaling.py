"""Normalization + scaling: the BRISC replacement for Seurat ``SCTransform``.

The R pipeline runs ``SCTransform`` and feeds its dense Pearson-residual ``scale.data``
(features x cells) into ``fit()``. That dense matrix is the memory wall (~200GB at 5M+ cells).

Here we instead keep a *sparse* log-normalized matrix and produce the scaled values on demand,
one cell-subset at a time, inside ``fit()``. The scaling we apply is a global per-gene z-score of
the BRISC-normalized counts. Per CLAUDE.md this is a sanctioned approximation of SCTransform's
Pearson residuals: both yield a centered/scaled matrix whose baseline variation PCA can capture.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import sparse

import brisc


def load_and_normalize(
    paths,
    qc_column: str = "passes_qc",
    normalize_method: str = "log1pPF",
    num_threads: int = -1,
):
    """Load one or more h5ad files, concatenate, drop QC-failing cells, and normalize.

    Parameters
    ----------
    paths : sequence of str
        Paths to the h5ad files making up the atlas.
    qc_column : str
        Boolean obs column marking cells that pass QC (default ``'passes_qc'``). These are the
        only cells kept. After filtering, every remaining cell passes, so downstream BRISC calls
        are given ``QC_column=None``.
    normalize_method : {'log1pPF', 'logCP10k', 'PFlog1pPF'}
        BRISC normalization method (default ``'log1pPF'``).
    num_threads : int
        BRISC thread count (``-1`` = all cores).

    Returns
    -------
    brisc.SingleCell
        A normalized SingleCell whose ``X`` is the log-normalized (sparse) matrix, cells x genes.
    """
    paths = list(paths)
    parts = [brisc.SingleCell(p, num_threads=num_threads) for p in paths]
    sc = parts[0] if len(parts) == 1 else brisc.concat_obs(*parts)

    # Physically drop QC-failing cells, then treat all remaining cells as passing. We do QC by hand
    # (dropping non-passing cells) rather than via BRISC's qc(), so mark the dataset QCed to satisfy
    # normalize()'s guard, and pass QC_column=None so every remaining cell is used.
    sc = sc.filter_obs(pl.col(qc_column))
    sc = sc.skip_qc()
    sc = sc.normalize(method=normalize_method, QC_column=None)
    return sc


def build_feature_matrix(sc, features):
    """Restrict a normalized SingleCell to ``features`` and return a sparse cells x k matrix.

    Returns
    -------
    (scipy.sparse.csr_matrix, list[str])
        The CSR matrix (cells x kept_features) in the order of ``kept_features``, and the list of
        features actually present in the object (missing ones are dropped, as in the R wrapper).
    """
    var_names = list(sc.var_names)
    name_to_idx = {n: i for i, n in enumerate(var_names)}
    kept_features = [f for f in features if f in name_to_idx]
    missing = len(features) - len(kept_features)
    if missing:
        print(f"[build_feature_matrix] {missing} of {len(features)} features not in var; dropping them")
    cols = np.array([name_to_idx[f] for f in kept_features], dtype=np.int64)

    X = sc.X
    # scipy sparse: column-fancy-index is cheap on CSC, then store as CSR for fast row subsetting.
    Xf = sparse.csc_matrix(X)[:, cols].tocsr()
    return Xf, kept_features


def gene_mean_sd(Xf):
    """Per-gene mean and (sample) standard deviation of a sparse cells x k matrix.

    Computed in a single pass from column sums and sums-of-squares. Zero-variance genes get sd=1
    so the z-score is well defined (their scaled values are all 0).
    """
    n = Xf.shape[0]
    s = np.asarray(Xf.sum(axis=0)).ravel()
    mean = s / n
    sq = np.asarray(Xf.multiply(Xf).sum(axis=0)).ravel()
    var = (sq / n - mean**2) * (n / (n - 1))
    var[var < 0] = 0.0
    sd = np.sqrt(var)
    sd[sd == 0] = 1.0
    return mean, sd
