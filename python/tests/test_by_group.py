#!/usr/bin/env python
"""Checks for per-group dictionaries (``iter_dictionaries_by_group``).

Small synthetic data; runs in seconds inside the BRISC container::

    apptainer exec "$SIF" python python/tests/test_by_group.py

Checks that each group's dictionary equals ``learn_dictionary_from_matrix`` run directly on that
group's rows (own controls, own gene stats), that it works through ``RowStackedCSR``, that groups with
too few controls are skipped, and that a planted group-specific effect is recovered in its group only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy import sparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brisc_fingerprinting import (  # noqa: E402
    RowStackedCSR,
    iter_dictionaries_by_group,
    learn_dictionary_from_matrix,
)
from brisc_fingerprinting.stacked import _to_int32_csr  # noqa: E402


def check(name, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    return ok


def make_group(rng, n_genes, n_ctrl, n_per_pert, effects, baseline):
    """Dense log-ish expression for one group: controls + one block of cells per perturbation."""
    rows, labels = [], []
    for name, eff in [("Negative", None)] + list(effects.items()):
        n = n_ctrl if eff is None else n_per_pert
        X = np.clip(baseline + rng.normal(0, 1, size=(n, n_genes)), 0, None)
        if eff is not None:
            X[:, eff] += 3.0
        rows.append(X)
        labels += [name] * n
    return np.vstack(rows).astype(np.float32), labels


def main():
    rng = np.random.default_rng(0)
    n_genes = 60
    features = [f"g{i}" for i in range(n_genes)]
    # Group A: GeneX moves genes 0-4. Group B: GeneX moves genes 30-34 (cell-type-specific effect).
    # Group C: only 10 controls -> must be skipped at min_control_cells=30.
    XA, lA = make_group(rng, n_genes, 300, 60, {"GeneX": np.arange(0, 5), "GeneY": np.arange(10, 15)},
                        baseline=rng.uniform(0.5, 2, n_genes))
    XB, lB = make_group(rng, n_genes, 300, 60, {"GeneX": np.arange(30, 35)},
                        baseline=rng.uniform(0.5, 2, n_genes))
    XC, lC = make_group(rng, n_genes, 10, 60, {"GeneX": np.arange(0, 5)},
                        baseline=rng.uniform(0.5, 2, n_genes))
    X = np.vstack([XA, XB, XC])
    labels = np.array(lA + lB + lC, dtype=object)
    groups = np.array(["A"] * len(lA) + ["B / sub"] * len(lB) + ["C"] * len(lC), dtype=object)

    # Shuffle rows so groups interleave, and split into blocks like the per-file loader.
    perm = rng.permutation(X.shape[0])
    X, labels, groups = X[perm], labels[perm], groups[perm]
    csr = sparse.csr_array(X)
    cuts = [0, 400, 700, X.shape[0]]
    stacked = RowStackedCSR([_to_int32_csr(csr[cuts[i]:cuts[i + 1]]) for i in range(3)])

    out = {g: (stats, res) for g, stats, res in
           iter_dictionaries_by_group(stacked, labels, groups, features, min_control_cells=30, n_jobs=2)}
    results = [check("one entry per group", sorted(out) == ["A", "B / sub", "C"])]

    stats_c, res_c = out["C"]
    results.append(check("group with <30 controls skipped",
                         res_c is None and stats_c["status"].startswith("skipped")))

    for g in ["A", "B / sub"]:
        rows = np.flatnonzero(groups == g)
        direct = learn_dictionary_from_matrix(csr[rows], labels[rows], features, n_jobs=1)
        stats, res = out[g]
        same = (res is not None and direct is not None and res["names"] == direct["names"]
                and np.allclose(res["fingerprints"]["Lambdas"], direct["fingerprints"]["Lambdas"], atol=1e-5)
                and all(np.allclose(a, b, atol=1e-5)
                        for a, b in zip(res["fingerprints"]["dists"], direct["fingerprints"]["dists"])))
        results.append(check(f"group {g!r} == direct fit on its rows", same))
        results.append(check(f"group {g!r} stats n_kept", stats["n_kept"] == len(res["names"])))

    # Planted effect: GeneX's top genes are 0-4 in A and 30-34 in B.
    def top5(res, name):
        lam = res["fingerprints"]["Lambdas"][:, res["names"].index(name)]
        return set(np.argsort(np.abs(lam))[::-1][:5].tolist())

    results.append(check("GeneX fingerprint group-specific (A: genes 0-4, B: genes 30-34)",
                         top5(out["A"][1], "GeneX") == set(range(5))
                         and top5(out["B / sub"][1], "GeneX") == set(range(30, 35))))

    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
