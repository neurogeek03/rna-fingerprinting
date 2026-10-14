#!/usr/bin/env python
"""Python side of the numerical comparison: run the port on the exact input R used, then compare
Lambdas against the R reference dictionary. Runs in the sklearn/joblib BRISC container."""

from __future__ import annotations

import sys

sys.path.insert(0, "/scratch/mfafouti/rna-fingerprinting/python")

import numpy as np
import polars as pl
from scipy import sparse, io as sio

import brisc
from brisc_fingerprinting import learn_dictionary, write_dictionary_rds

OUT = "/scratch/mfafouti/rna-fingerprinting/brisc_adaptation/validation/out"


def load_lines(p):
    with open(p) as fh:
        return [ln.rstrip("\n") for ln in fh]


def main():
    genes = load_lines(f"{OUT}/genes.txt")
    barcodes = load_lines(f"{OUT}/barcodes.txt")
    target = load_lines(f"{OUT}/target_gene.txt")
    features = load_lines(f"{OUT}/features.txt")
    print(f"python: {len(genes)} genes, {len(barcodes)} cells, {len(features)} features")

    # counts.mtx is genes x cells (Seurat orientation); SingleCell wants cells x genes.
    counts = sio.mmread(f"{OUT}/counts.mtx").tocsr()
    X = sparse.csr_matrix(counts.T)  # cells x genes
    assert X.shape == (len(barcodes), len(genes)), X.shape

    sc = brisc.SingleCell(
        X=X,
        obs=pl.DataFrame({"cell": barcodes, "target_gene": target, "passes_qc": [True] * len(barcodes)}),
        var=pl.DataFrame({"gene": genes}),
    )

    # Same control label ('control') and feature set as R; no combos here, so collapse is off.
    result = learn_dictionary(
        paths=None, sc=sc, features=features,
        perturbation_column="target_gene", collapse_single_gene=False,
        control_class="control", qc_column="passes_qc",
        min_cells=10, override_filter=False, n_jobs=4, seed=1448145,
    )
    assert result is not None, "python learned no fingerprints"
    write_dictionary_rds(result, f"{OUT}/Py_dict.rds")

    # ---- compare against R reference ----
    from ryp import r, to_py
    r(f'rd <- readRDS("{OUT}/R_dict.rds")')
    r("R_L <- rd$fingerprints$Lambdas; R_feats <- rownames(R_L); R_perts <- colnames(R_L); "
      "R_Lnum <- matrix(as.double(R_L), nrow=nrow(R_L), ncol=ncol(R_L))")
    R_L = np.asarray(to_py("R_Lnum"), dtype=float)
    R_feats = [str(x) for x in np.atleast_1d(to_py("R_feats"))]
    R_perts = [str(x) for x in np.atleast_1d(to_py("R_perts"))]

    Py_L = np.asarray(result["fingerprints"]["Lambdas"])
    Py_feats = list(result["features"])
    Py_perts = list(result["names"])

    print("\n=== comparison ===")
    print(f"R perturbations : {R_perts}")
    print(f"Py perturbations: {Py_perts}")
    print(f"R features: {len(R_feats)}  Py features: {len(Py_feats)}  "
          f"shared: {len(set(R_feats) & set(Py_feats))}")

    shared_feats = [f for f in R_feats if f in set(Py_feats)]
    r_fi = {f: i for i, f in enumerate(R_feats)}
    p_fi = {f: i for i, f in enumerate(Py_feats)}
    for p in R_perts:
        if p not in Py_perts:
            print(f"  {p}: present in R, ABSENT in Python")
            continue
        rv = R_L[[r_fi[f] for f in shared_feats], R_perts.index(p)]
        pv = Py_L[[p_fi[f] for f in shared_feats], Py_perts.index(p)]
        if np.std(rv) == 0 or np.std(pv) == 0:
            print(f"  {p}: degenerate vector")
            continue
        rho = np.corrcoef(rv, pv)[0, 1]
        # top-gene agreement (|loading|)
        kr = set(np.array(shared_feats)[np.argsort(np.abs(rv))[::-1][:20]])
        kp = set(np.array(shared_feats)[np.argsort(np.abs(pv))[::-1][:20]])
        print(f"  {p}: signed Pearson r = {rho:+.3f} | |r| = {abs(rho):.3f} | "
              f"top-20 gene overlap = {len(kr & kp)}/20")
    print("\nDONE - comparison complete")


if __name__ == "__main__":
    main()
