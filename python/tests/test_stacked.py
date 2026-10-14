#!/usr/bin/env python
"""Equivalence checks for the per-file (no-concat) path in ``brisc_fingerprinting.stacked``.

Small synthetic data; runs in seconds inside the BRISC container::

    apptainer exec "$SIF" python python/tests/test_stacked.py

Checks that RowStackedCSR row access == the concatenated matrix (sorted, unsorted, cross-block,
empty), that gene_mean_sd agrees, and that normalize_log1ppf (global target, applied after column
restriction) == BRISC normalize(method='log1pPF') on the concatenated, unrestricted data.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import polars as pl
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import brisc  # noqa: E402

from brisc_fingerprinting import RowStackedCSR, gene_mean_sd, normalize_log1ppf, restrict_blocks  # noqa: E402
from brisc_fingerprinting.stacked import _to_int32_csr  # noqa: E402


def random_counts(rng, n_cells, n_genes, density=0.1):
    X = sparse.random(n_cells, n_genes, density=density, format="csr", random_state=rng,
                      data_rvs=lambda k: rng.integers(1, 20, size=k)).astype(np.float32)
    return sparse.csr_array(X)


def check(name, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return ok


def main():
    rng = np.random.default_rng(0)
    sizes = [300, 50, 500]
    n_genes = 200
    raw_blocks = [random_counts(rng, n, n_genes) for n in sizes]
    full = sparse.vstack(raw_blocks, format="csr")
    results = []

    # 1. Row access.
    stacked = RowStackedCSR([_to_int32_csr(b) for b in raw_blocks])
    n = full.shape[0]
    cases = {
        "sorted cross-block rows": np.sort(rng.choice(n, 200, replace=False)),
        "unsorted rows": rng.choice(n, 200, replace=False),
        "single block rows": np.arange(310, 340),
        "all rows": np.arange(n),
    }
    for label, idx in cases.items():
        results.append(check(f"row access: {label}",
                             np.array_equal(stacked[idx].toarray(), full[idx].toarray())))
    results.append(check("row access: empty", stacked[np.array([], dtype=np.int64)].shape == (0, n_genes)))
    results.append(check("int32 indices stored", all(b.indices.dtype == np.int32 for b in stacked.blocks)))

    # 2. gene_mean_sd on the wrapper vs the concatenated matrix (with a row subset).
    rows = np.sort(rng.choice(n, 600, replace=False))
    m1, s1 = gene_mean_sd(stacked, rows=rows, chunk_size=97)
    m2, s2 = gene_mean_sd(full, rows=rows)
    results.append(check("gene_mean_sd matches", np.allclose(m1, m2) and np.allclose(s1, s2)))

    # 3. Normalization: per-file global log1pPF after restriction vs BRISC on concat, unrestricted.
    lib_sizes = [np.asarray(b.sum(axis=1, dtype=np.float64)).ravel() for b in raw_blocks]
    features = [f"g{i}" for i in rng.choice(n_genes, 80, replace=False)]
    var_names = [f"g{i}" for i in range(n_genes)]
    blocks, kept = restrict_blocks([_to_int32_csr(b) for b in raw_blocks], var_names, features)
    normalize_log1ppf(blocks, lib_sizes, num_threads=3)
    ours = sparse.vstack(blocks, format="csr").toarray()

    sc = brisc.SingleCell(X=full.copy(),
                          obs=pl.DataFrame({"cell": [f"c{i}" for i in range(n)]}),
                          var=pl.DataFrame({"gene": var_names})).skip_qc()
    ref_sc = sc.normalize(method="log1pPF", QC_column=None)
    cols = [var_names.index(f) for f in kept]
    ref = sparse.csr_array(ref_sc.X)[:, cols].toarray()
    max_abs = float(np.max(np.abs(ours - ref)))
    results.append(check(f"log1pPF matches BRISC (max |diff| = {max_abs:.2e})",
                         np.allclose(ours, ref, rtol=1e-5, atol=1e-6)))

    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
