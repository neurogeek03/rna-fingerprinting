"""Per-file ("no-concat") loading and normalization, for atlases too large to concatenate.

``brisc.concat_obs`` builds the combined matrix while the per-file pieces still exist, so the
full-atlas peak is ~2x the data (~780 GB at 32.6e9 nonzeros x 12 B; a Trillium node has ~745 GiB).
Downstream code only ever *reads rows* (``fit()``) and *sums columns over row chunks*
(``gene_mean_sd``), so instead we keep one CSR block per file behind :class:`RowStackedCSR`, store
each block with int32 indices (8 B/nonzero instead of 12), and normalize every block in place with a
single dataset-wide log1pPF target. The combined matrix is never materialized.

See ``brisc_adaptation/decisions_log.md`` for the measurements behind this.
"""

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from scipy import sparse

import brisc

from .logs import log

_INT32_MAX = np.iinfo(np.int32).max


class RowStackedCSR:
    """Read-only vertical stack of CSR blocks that behaves like one cells x genes CSR for row access.

    Supports exactly what ``fit()`` / ``gene_mean_sd`` need: ``shape``, ``dtype``, ``nnz`` and
    ``X[row_indices]``, which returns a ``scipy.sparse.csr_array`` with the rows in the requested
    order. All blocks must have the same columns.
    """

    def __init__(self, blocks):
        self.blocks = list(blocks)
        n_cols = {b.shape[1] for b in self.blocks}
        if len(n_cols) != 1:
            raise ValueError(f"blocks have different numbers of columns: {sorted(n_cols)}")
        self.offsets = np.cumsum([0] + [b.shape[0] for b in self.blocks])
        self.shape = (int(self.offsets[-1]), n_cols.pop())
        self.dtype = self.blocks[0].dtype

    @property
    def nnz(self):
        return sum(b.nnz for b in self.blocks)

    def __getitem__(self, idx):
        idx = np.asarray(idx)
        if idx.ndim != 1 or idx.dtype.kind not in "iu":
            raise TypeError("RowStackedCSR only supports 1-D integer row-index arrays")
        if idx.size == 0:
            return sparse.csr_array((0, self.shape[1]), dtype=self.dtype)
        # Work on sorted indices (callers pass np.where output, already sorted) and restore order.
        order = None
        if np.any(idx[1:] < idx[:-1]):
            order = np.argsort(idx, kind="stable")
            idx = idx[order]
        which = np.searchsorted(self.offsets, idx, side="right") - 1
        bounds = np.searchsorted(which, np.arange(len(self.blocks) + 1))
        pieces = [self.blocks[b][idx[bounds[b]:bounds[b + 1]] - self.offsets[b]]
                  for b in range(len(self.blocks)) if bounds[b + 1] > bounds[b]]
        out = pieces[0] if len(pieces) == 1 else sparse.vstack(pieces, format="csr")
        if order is not None:
            inverse = np.empty_like(order)
            inverse[order] = np.arange(order.size)
            out = out[inverse]
        return out


def _to_int32_csr(X):
    """Rebuild a CSR matrix as a ``csr_array`` with float32 data and (if they fit) int32 indices."""
    data = X.data if X.data.dtype == np.float32 else X.data.astype(np.float32)
    index_dtype = np.int32 if X.nnz <= _INT32_MAX and max(X.shape) <= _INT32_MAX else np.int64
    indices = X.indices.astype(index_dtype, copy=False)
    indptr = X.indptr.astype(index_dtype, copy=False)
    return sparse.csr_array((data, indices, indptr), shape=X.shape, copy=False)


def load_qc_blocks(paths, qc_column: str = "passes_qc", label_column: str = "gene_target",
                   num_threads: int = -1):
    """Load each h5ad's QC-passing RAW counts as its own CSR block (no concatenation).

    Only ``qc_column`` and ``label_column`` are read from obs (the files encode other columns with
    inconsistent dtypes); labels are returned as plain strings.

    Returns
    -------
    dict with keys
        ``blocks`` (list of csr_array, cells x all genes, raw counts), ``labels`` (object array, one per
        cell, concatenated in file order), ``var_names`` (list[str]), ``lib_sizes`` (list of float64
        arrays: per-cell total counts over ALL genes, needed by :func:`normalize_log1ppf`), and
        ``gene_sums`` (float64 array: raw count sum per gene over all cells, for expression-based
        feature selection).
    """
    blocks, labels, lib_sizes = [], [], []
    var_names = None
    gene_sums = None
    for p in paths:
        log(f"[load_qc_blocks] loading {p}")
        s = brisc.SingleCell(p, obs_columns=[qc_column, label_column], obsm=False, obsp=False,
                             varm=False, varp=False, uns=False, num_threads=num_threads)
        s = s.filter_obs(pl.col(qc_column))
        names = list(s.var_names)
        if var_names is None:
            var_names = names
        elif names != var_names:
            raise ValueError(f"{p}: var names differ from {paths[0]}; cannot stack files")
        labels.append(np.asarray(s.obs[label_column].cast(pl.String).to_list(), dtype=object))
        src = f"data={s.X.data.dtype} indices={s.X.indices.dtype} indptr={s.X.indptr.dtype}"
        block = _to_int32_csr(s.X)
        del s
        gc.collect()
        lib_sizes.append(np.asarray(block.sum(axis=1, dtype=np.float64)).ravel())
        col = np.asarray(block.sum(axis=0, dtype=np.float64)).ravel()
        gene_sums = col if gene_sums is None else gene_sums + col
        blocks.append(block)
        log(f"[load_qc_blocks] {p}: {block.shape[0]} QC cells, nnz={block.nnz:,} "
            f"(loaded {src} -> stored indices={block.indices.dtype})")
    return {
        "blocks": blocks,
        "labels": np.concatenate(labels),
        "var_names": var_names,
        "lib_sizes": lib_sizes,
        "gene_sums": gene_sums,
    }


def restrict_blocks(blocks, var_names, features):
    """Keep only ``features`` columns in every block, one block at a time (bounded transient memory).

    Returns ``(blocks, kept_features)``; columns follow the order of ``features`` (missing ones are
    dropped, as in the R wrapper). Replaces the list entries in place so each full block is freed
    as soon as its restricted copy exists.
    """
    name_to_idx = {n: i for i, n in enumerate(var_names)}
    kept = [f for f in features if f in name_to_idx]
    if len(kept) < len(features):
        log(f"[restrict_blocks] {len(features) - len(kept)} of {len(features)} features not in var; dropping them")
    cols = np.array([name_to_idx[f] for f in kept], dtype=np.int64)
    for i in range(len(blocks)):
        blocks[i] = _to_int32_csr(blocks[i][:, cols])
        gc.collect()
    return blocks, kept


def normalize_log1ppf(blocks, lib_sizes, num_threads: int = 1):
    """In-place log1pPF over all blocks, matching ``brisc.SingleCell.normalize(method='log1pPF')``.

    BRISC defines log1pPF as ``log1p(x / (L / mean(L)))`` where ``L`` is each cell's library size
    (total counts over all genes) and the mean is over all cells. Here ``lib_sizes`` must therefore
    be computed from the raw, all-gene blocks (as :func:`load_qc_blocks` does) and the mean is taken
    across *every* block -- normalizing each file with its own mean would shift files relative to
    each other. Because ``L`` is precomputed, this can run after :func:`restrict_blocks`.

    Returns the dataset-wide mean library size.
    """
    mean_l = float(np.concatenate(lib_sizes).mean())

    def _one(block, lib):
        with np.errstate(divide="ignore"):
            scale = np.where(lib > 0, mean_l / lib, 0.0).astype(np.float32)
        block.data *= np.repeat(scale, np.diff(block.indptr))
        np.log1p(block.data, out=block.data)

    # numpy releases the GIL in the elementwise kernels, so files normalize in parallel threads.
    with ThreadPoolExecutor(max_workers=max(1, min(num_threads, len(blocks)))) as ex:
        list(ex.map(_one, blocks, lib_sizes))
    return mean_l
