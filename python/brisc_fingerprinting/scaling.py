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

from .logs import log


def load_qc(paths, qc_column: str = "passes_qc", obs_columns=(), num_threads: int = -1):
    """Load one or more h5ads, concatenate, and drop QC-failing cells (counts left RAW).

    Returns a SingleCell whose ``X`` is still raw counts -- use this for raw-count feature selection
    (``choose_features(method='expression')``) before normalizing. We do QC by hand (dropping
    non-passing cells) rather than via BRISC's ``qc()``, so ``skip_qc()`` marks the dataset QCed to
    satisfy ``normalize()``'s guard downstream.

    ``obs_columns`` lists the obs columns to keep besides ``qc_column`` (e.g. the perturbation
    column). Only those are loaded, and they are cast to ``pl.String``: the atlas files encode the
    same column with different dtypes (int32 vs int64, categoricals with per-file category sets),
    which ``concat_obs`` rejects, and the other ~40 obs columns are dead weight here.
    """
    import gc

    paths = list(paths)
    keep_cols = [qc_column, *[c for c in obs_columns if c != qc_column]]
    string_cols = {c: pl.String for c in keep_cols if c != qc_column}
    # Load ONLY the counts matrix + the needed obs columns + var. The atlas h5ads also carry
    # obsm/obsp/varm/varp/uns (neighbor graphs, embeddings, ...) which are large and irrelevant here
    # -- loading them for all 14 files OOMs the node. ``False`` skips each. Filter each file to
    # QC-passing cells *before* concatenating so ~22% less data flows into (and is copied by) concat_obs.
    parts = []
    for p in paths:
        log(f"[load_qc] loading {p}")
        s = brisc.SingleCell(p, obs_columns=keep_cols, obsm=False, obsp=False, varm=False,
                             varp=False, uns=False, num_threads=num_threads)
        s = s.filter_obs(pl.col(qc_column))
        parts.append(s.cast_obs(string_cols) if string_cols else s)
        del s
        gc.collect()
        log(f"[load_qc] {p}: {parts[-1].shape[0]} QC cells")
    if len(parts) > 1:
        log(f"[load_qc] concatenating {len(parts)} files")
    sc = parts[0] if len(parts) == 1 else brisc.concat_obs(*parts)
    del parts
    gc.collect()
    log(f"[load_qc] {sc.shape[0]} cells x {sc.shape[1]} genes")
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


def restrict_to_features(sc, features):
    """Drop all genes not in ``features`` from a SingleCell (returns a new, smaller object).

    Call this *after* normalization -- log1pPF size factors use per-cell totals over all genes, so
    filtering first would change the normalized values. Rebinding the caller's variable to the result
    frees the full-genes matrix, so later copies (``filter_obs``, ``build_feature_matrix``) only carry
    the selected genes.
    """
    keep = np.isin(np.asarray(list(sc.var_names)), list(features))
    return sc.filter_var(keep)


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
        log(f"[build_feature_matrix] {missing} of {len(features)} features not in var; dropping them")

    X = sc.X
    if len(kept_features) == len(var_names) and sparse.issparse(X) and X.format == "csr":
        # Already restricted (``restrict_to_features``): use X as-is, with features in var order.
        # Reordering columns would copy the whole matrix; R output rows are named, so order is free.
        return X, var_names

    cols = np.array([name_to_idx[f] for f in kept_features], dtype=np.int64)
    # scipy sparse: column-fancy-index is cheap on CSC, then store as CSR for fast row subsetting.
    # NOTE: copies the whole matrix (several times transiently) -- call restrict_to_features first.
    Xf = sparse.csc_matrix(X)[:, cols].tocsr()
    return Xf, kept_features


def gene_mean_sd(Xf, rows=None, chunk_size: int = 100_000):
    """Per-gene mean and (sample) standard deviation of a sparse cells x k matrix.

    Computed from column sums and sums-of-squares, over ``rows`` only if given (default: all rows),
    in row chunks so no full-size temporary (e.g. ``Xf.multiply(Xf)``) is ever allocated.
    Zero-variance genes get sd=1 so the z-score is well defined (their scaled values are all 0).

    Accumulated in float64 for accuracy, returned as float32 so the dense blocks ``fit()`` builds from
    them stay float32 (a float64 mean would silently upcast every block and double its memory).
    """
    rows = np.arange(Xf.shape[0]) if rows is None else np.asarray(rows)
    n = rows.size
    s = np.zeros(Xf.shape[1])
    sq = np.zeros(Xf.shape[1])
    for start in range(0, n, chunk_size):
        block = Xf[rows[start:start + chunk_size]]
        s += np.asarray(block.sum(axis=0, dtype=np.float64)).ravel()
        sq += np.asarray(block.multiply(block).sum(axis=0, dtype=np.float64)).ravel()
    mean = s / n
    var = (sq / n - mean**2) * (n / (n - 1))
    var[var < 0] = 0.0
    sd = np.sqrt(var)
    sd[sd == 0] = 1.0
    return mean.astype(np.float32), sd.astype(np.float32)
