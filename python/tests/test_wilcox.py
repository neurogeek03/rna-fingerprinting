#!/usr/bin/env python
"""Check the searchsorted Wilcoxon in ``fit.py`` against the original rank-based version, and time both.

    apptainer exec "$SIF" python python/tests/test_wilcox.py [--bench]

The reference below is the pre-2026-10-14 ``_wilcox_per_feature`` (concatenate controls + perturbed,
``scipy.stats.rankdata`` every row, sum control ranks). Both must give identical U and p.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brisc_fingerprinting.fit import _sorted_control_keys, _wilcox_per_feature  # noqa: E402


def wilcox_ranks_reference(ctrl_res_fx, pert_res_fx):
    n1, n2 = ctrl_res_fx.shape[1], pert_res_fx.shape[1]
    M = np.concatenate([ctrl_res_fx, pert_res_fx], axis=1)
    ranks = stats.rankdata(M, axis=1)
    U1 = ranks[:, :n1].sum(axis=1, dtype=np.float64) - n1 * (n1 + 1) / 2.0
    N = n1 + n2
    z = (U1 - n1 * n2 / 2.0) / np.sqrt(n1 * n2 * (N + 1) / 12.0)
    return U1, 2.0 * stats.norm.sf(np.abs(z))


def new(ctrl, pert):
    return _wilcox_per_feature(_sorted_control_keys(ctrl), ctrl.shape[1], pert, chunk=97)


def check(name, ctrl, pert):
    U_ref, p_ref = wilcox_ranks_reference(ctrl, pert)
    U_new, p_new = new(ctrl, pert)
    ok = np.array_equal(U_ref, U_new) and np.allclose(p_ref, p_new, rtol=1e-12, atol=0)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  (max |dU| = {np.max(np.abs(U_ref - U_new)):.3g})")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="also time both at realistic sizes")
    args = ap.parse_args(argv)
    rng = np.random.default_rng(0)
    results = []

    k = 300
    ctrl = rng.normal(size=(k, 1000)).astype(np.float32)
    results.append(check("continuous values", ctrl, rng.normal(0.2, 1, size=(k, 150)).astype(np.float32)))
    # Heavy ties (rounded values, shared between groups), negatives and signed zeros.
    ctrl_t = np.round(rng.normal(size=(k, 1000)), 1).astype(np.float32)
    pert_t = np.round(rng.normal(size=(k, 150)), 1).astype(np.float32)
    pert_t[0, :10] = -0.0
    ctrl_t[0, :10] = 0.0
    results.append(check("heavy ties + signed zeros", ctrl_t, pert_t))
    results.append(check("single perturbed cell", ctrl, rng.normal(size=(k, 1)).astype(np.float32)))
    results.append(check("large magnitudes", ctrl * 1e4, (rng.normal(size=(k, 50)) * 1e4).astype(np.float32)))

    if args.bench:
        k = 11796
        ctrl = rng.normal(size=(k, 1000)).astype(np.float32)
        keys = _sorted_control_keys(ctrl)
        for n2 in (200, 2000):
            pert = rng.normal(size=(k, n2)).astype(np.float32)
            t0 = time.perf_counter(); wilcox_ranks_reference(ctrl, pert); t_ref = time.perf_counter() - t0
            t0 = time.perf_counter(); _wilcox_per_feature(keys, 1000, pert); t_new = time.perf_counter() - t0
            print(f"BENCH k={k} n1=1000 n2={n2}: rankdata {t_ref:.2f}s  searchsorted {t_new:.2f}s  "
                  f"speedup {t_ref / t_new:.1f}x")

    print("ALL PASS" if all(results) else "SOME CHECKS FAILED")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
