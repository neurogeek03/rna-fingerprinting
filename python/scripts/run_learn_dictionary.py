#!/usr/bin/env python
"""CLI: learn a perturbation dictionary with BRISC and write a compatible .rds.

Run inside the BRISC container, on a compute node::

    module load apptainer/1.3.5
    apptainer exec /scratch/mfafouti/docker/brisc_bioc-v2-mkl-seurat-sklearn.sif \
        python python/scripts/run_learn_dictionary.py \
            --h5ad-dir /scratch/mfafouti/Psych_brain/Perturb-seq/wholebrain_atlas/h5ads \
            --choose-features expression \
            --out dictionary.rds \
            --n-jobs 16
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

# Make the package importable whether or not it's installed.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brisc_fingerprinting import (  # noqa: E402
    choose_features,
    learn_dictionary,
    load_qc,
    normalize_counts,
    write_dictionary_rds,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--h5ad-dir", help="Directory containing *.h5ad files")
    src.add_argument("--h5ad", nargs="+", help="Explicit list of h5ad paths")

    feat = p.add_mutually_exclusive_group(required=True)
    feat.add_argument("--features-file", help="Text file, one feature (gene symbol) per line")
    feat.add_argument("--choose-features", choices=["de", "expression"], help="Select features automatically")

    p.add_argument("--out", required=True, help="Output .rds path")
    p.add_argument("--perturbation-column", default="gene_target")
    p.add_argument("--no-collapse-single-gene", action="store_true",
                   help="Disable single-gene collapse; use legacy control-class/exclude-classes logic")
    p.add_argument("--control-class", default="Negative",
                   help="(legacy, only with --no-collapse-single-gene) control label")
    p.add_argument("--exclude-classes", nargs="*", default=["Non_target"],
                   help="(legacy, only with --no-collapse-single-gene) labels to drop")
    p.add_argument("--qc-column", default="passes_qc")
    p.add_argument("--normalize-method", default="log1pPF", choices=["log1pPF", "logCP10k", "PFlog1pPF"])
    p.add_argument("--min-cells", type=int, default=10)
    p.add_argument("--override-filter", action="store_true")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--num-threads", type=int, default=-1)
    p.add_argument("--seed", type=int, default=1448145)
    p.add_argument("--max-cells", type=int, default=None, help="Subsample to this many cells (testing)")
    p.add_argument("--max-per-perturbation", type=int, default=500, help="For --choose-features de")
    p.add_argument("--mean-expression", type=float, default=0.01, help="For --choose-features expression")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.h5ad_dir:
        paths = sorted(glob.glob(os.path.join(args.h5ad_dir, "*.h5ad")))
        if not paths:
            sys.exit(f"No h5ad files found in {args.h5ad_dir}")
    else:
        paths = args.h5ad

    print(f"[run] {len(paths)} h5ad file(s)")

    # Load QC-filtered RAW counts first: expression-based feature selection must see raw counts
    # (matching R ChooseFeatures), so it runs before normalization.
    sc = load_qc(paths, qc_column=args.qc_column, num_threads=args.num_threads)
    print(f"[run] loaded {sc.shape[0]} QC-passing cells x {sc.shape[1]} genes (raw)")

    features = None
    if args.features_file:
        with open(args.features_file) as fh:
            features = [ln.strip() for ln in fh if ln.strip()]
        print(f"[run] {len(features)} features from {args.features_file}")
    elif args.choose_features == "expression":
        features = choose_features(sc, method="expression", mean_expression=args.mean_expression)
        print(f"[run] chose {len(features)} features via method='expression' (raw counts)")

    # Normalize for fingerprint learning (and DE feature selection, which uses normalized data).
    sc = normalize_counts(sc, normalize_method=args.normalize_method)

    if features is None:  # --choose-features de
        features = choose_features(
            sc,
            method="de",
            perturbation_column=args.perturbation_column,
            control_class=args.control_class,
            max_per_perturbation=args.max_per_perturbation,
        )
        print(f"[run] chose {len(features)} features via method='de' (normalized)")

    result = learn_dictionary(
        paths=None,
        features=features,
        perturbation_column=args.perturbation_column,
        collapse_single_gene=not args.no_collapse_single_gene,
        control_class=args.control_class,
        exclude_classes=tuple(args.exclude_classes),
        min_cells=args.min_cells,
        override_filter=args.override_filter,
        n_jobs=args.n_jobs,
        seed=args.seed,
        max_cells=args.max_cells,
        sc=sc,
    )
    if result is None:
        sys.exit("[run] no fingerprints learned")

    write_dictionary_rds(result, args.out)
    print("[run] done")


if __name__ == "__main__":
    main()
