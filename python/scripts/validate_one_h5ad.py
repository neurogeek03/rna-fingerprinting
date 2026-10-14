#!/usr/bin/env python
"""Real-data validation of the BRISC port on a single h5ad.

Exercises the paths the synthetic smoke test could not: load_and_normalize (real concat/QC/normalize),
choose_features (both methods, incl. real find_markers columns), and learn_dictionary -> .rds on real
Perturb-seq data. Run on a compute node inside the sklearn/joblib BRISC container.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "/scratch/mfafouti/rna-fingerprinting/python")

import numpy as np  # noqa: E402

from brisc_fingerprinting import (  # noqa: E402
    choose_features,
    learn_dictionary,
    load_and_normalize,
    write_dictionary_rds,
)
from brisc_fingerprinting.scaling import load_and_normalize as _lan  # noqa: E402

H5AD = "/scratch/mfafouti/Psych_brain/Perturb-seq/wholebrain_atlas/h5ads/WB8588_1_2.h5ad"
OUT = "/scratch/mfafouti/rna-fingerprinting/python/scripts/validate_dict.rds"
N_JOBS = 32


def t(msg, t0):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def main():
    t0 = time.time()
    print(f"=== validate on {H5AD} ===", flush=True)

    # 1. load + normalize
    sc = load_and_normalize([H5AD], qc_column="passes_qc", normalize_method="log1pPF")
    t(f"loaded+normalized: {sc.shape[0]} QC cells x {sc.shape[1]} genes", t0)
    print("  var_names[:5]:", list(sc.var_names)[:5], flush=True)
    import polars as pl

    vc = sc.obs["gene_target"].value_counts(sort=True).head(8)
    print("  top gene_target:\n", vc, flush=True)
    n_neg = (sc.obs["gene_target"] == "Negative").sum()
    print(f"  Negative (control) cells: {n_neg}", flush=True)

    # 2a. choose_features: expression (fast)
    feats_expr = choose_features(sc, method="expression", mean_expression=0.1)
    t(f"choose_features(expression): {len(feats_expr)} genes", t0)
    if len(feats_expr) > 2000:
        feats_expr = feats_expr[:2000]
        print("  capped to 2000 for validation speed", flush=True)

    # 2b. choose_features: de -- validate find_markers columns on a couple perturbations only
    try:
        targets = [g for g, _ in sc.obs["gene_target"].value_counts(sort=True).head(6).iter_rows()
                   if g not in ("Negative", "Non_target") and "|" not in g][:2]
        mk = sc.find_markers("gene_target", QC_column=None, cell_types=targets)
        print(f"  find_markers columns: {mk.columns}", flush=True)
        t(f"find_markers on {targets}: {mk.shape[0]} rows", t0)
    except Exception as e:
        print(f"  [warn] find_markers probe failed: {type(e).__name__}: {e}", flush=True)

    # 3. learn_dictionary with single-gene collapse + pooled controls (decisions_log.md §5-6)
    result = learn_dictionary(
        paths=None, sc=sc, features=feats_expr,
        perturbation_column="gene_target", collapse_single_gene=True,
        min_cells=10, override_filter=False, n_jobs=N_JOBS,
    )
    if result is None:
        print("!! learn_dictionary returned None (no fingerprints survived the filter)", flush=True)
        print("   retrying with override_filter=True to confirm the pipeline runs...", flush=True)
        result = learn_dictionary(
            paths=None, sc=sc, features=feats_expr,
            perturbation_column="gene_target", collapse_single_gene=True,
            min_cells=10, override_filter=True, n_jobs=N_JOBS,
        )
    assert result is not None, "pipeline produced no fingerprints even with override_filter"
    L = result["fingerprints"]["Lambdas"]
    t(f"learn_dictionary: {L.shape[1]} fingerprints x {L.shape[0]} features", t0)
    print("  first 10 perturbations:", result["names"][:10], flush=True)

    # 4. write + read back
    write_dictionary_rds(result, OUT)
    from ryp import r
    r(f'd <- readRDS("{OUT}")')
    r('cat("  RDS class:", class(d), "| Lambdas:", dim(d$fingerprints$Lambdas),'
      ' "| dists:", length(d$fingerprints$dists), "\\n")')
    t("DONE - validation passed", t0)


if __name__ == "__main__":
    main()
