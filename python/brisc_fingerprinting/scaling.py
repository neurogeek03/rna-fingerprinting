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


def load_qc(paths, qc_column: str = "passes_qc", num_threads: int = -1):
    """Load one or more h5ads, concatenate, and drop QC-failing cells (counts left RAW).

    Returns a SingleCell whose ``X`` is still raw counts -- use this for raw-count feature selection
    (``choose_features(method='expression')``) before normalizing. We do QC by hand (dropping
    non-passing cells) rather than via BRISC's ``qc()``, so ``skip_qc()`` marks the dataset QCed to
    satisfy ``normalize()``'s guard downstream.
    """
    import gc

    paths = list(paths)
    # Load ONLY the counts matrix + obs + var. The atlas h5ads also carry obsm/obsp/varm/varp/uns
    # (neighbor graphs, embeddings, ...) which are large and irrelevant here -- loading them for all
    # 14 files OOMs the node. ``False`` skips each. Filter each file to QC-passing cells *before*
    # concatenating so ~22% less data flows into (and is copied by) concat_obs.
    parts = []
    for p in paths:
        s = brisc.SingleCell(p, obsm=False, obsp=False, varm=False, varp=False, uns=False,
                             num_threads=num_threads)
        parts.append(s.filter_obs(pl.col(qc_column)))
    sc = parts[0] if len(parts) == 1 else brisc.concat_obs(*parts)
    del parts
    gc.collect()
    return sc.skip_qc()


def normalize_counts(sc, normalize_method: str = "log1pPF"):
    """Normalize a QC-filtered SingleCell in place (every cell treated as passing: ``QC_column=None``).

    ``inplace=True`` avoids holding the raw and normalized matrices simultaneously -- at ~196 GB per
    copy of the full-atlas counts, that saved copy matters.
    """
    return sc.normalize(method=normalize_method, QC_column=None, inplace=True)


def load_and_normalize(
    paths,
    qc_column: str = "passes_qc",
    normalize_method: str = "log1pPF",
    num_threads: int = -1,
):
    """Load one or more h5ad files, concatenate, drop QC-failing cells, and normalize.

    Convenience wrapper over :func:`load_qc` + :func:`normalize_counts`. Returns a normalized
    SingleCell whose ``X`` is the log-normalized (sparse) matrix, cells x genes.

    Parameters
    ----------
    paths : sequence of str
        Paths to the h5ad files making up the atlas.
    qc_column : str
        Boolean obs column marking cells that pass QC (default ``'passes_qc'``).
    normalize_method : {'log1pPF', 'logCP10k', 'PFlog1pPF'}
        BRISC normalization method (default ``'log1pPF'``).
    num_threads : int
        BRISC thread count (``-1`` = all cores).
    """
    return normalize_counts(
        load_qc(paths, qc_column=qc_column, num_threads=num_threads),
        normalize_method=normalize_method,
    )


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
