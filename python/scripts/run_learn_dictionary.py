#!/usr/bin/env python
"""CLI: learn a perturbation dictionary with BRISC and write a compatible .rds.

Run inside the BRISC container, on a compute node (see python/sbatch/learn_dictionary.sbatch)::

    apptainer exec "$SIF" python -u python/scripts/run_learn_dictionary.py \
        --h5ad-dir "$H5AD_DIR" --choose-features expression --out dictionary.rds --n-jobs 96

Per-group dictionaries (the authors learn one dictionary per cell type)::

    ... --group-column predicted_subclass --out dictionaries/

  writes dictionaries/<group>.rds for every group with enough controls, plus dictionaries/summary.tsv.
  Per-file loader only.

Loaders:
  per-file (default) -- one CSR block per h5ad, never concatenated, single global log1pPF target.
                        Needed for the full atlas (concat would double peak memory). No DE selection.
  concat             -- BRISC concat_obs + normalize(); supports --choose-features de.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# Make the package importable whether or not it's installed.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brisc_fingerprinting import (  # noqa: E402
    RowStackedCSR,
    choose_features,
    features_by_mean_expression,
    iter_dictionaries_by_group,
    learn_dictionary,
    learn_dictionary_from_matrix,
    load_qc,
    load_qc_blocks,
    log,
    normalize_counts,
    normalize_log1ppf,
    restrict_blocks,
    restrict_to_features,
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

    p.add_argument("--out", required=True,
                   help="Output .rds path, or an output directory with --group-column")
    p.add_argument("--group-column", default=None,
                   help="obs column to split by (e.g. predicted_subclass): one dictionary per group")
    p.add_argument("--min-control-cells", type=int, default=30,
                   help="With --group-column: skip groups with fewer control cells")
    p.add_argument("--loader", choices=["per-file", "concat"], default="per-file",
                   help="per-file (default; memory-safe for the full atlas) or concat (BRISC concat_obs)")
    p.add_argument("--perturbation-column", default="gene_target")
    p.add_argument("--no-collapse-single-gene", action="store_true",
                   help="Disable single-gene collapse; use legacy control-class/exclude-classes logic")
    p.add_argument("--control-class", default="Negative",
                   help="(legacy, only with --no-collapse-single-gene) control label")
    p.add_argument("--exclude-classes", nargs="*", default=["Non_target"],
                   help="(legacy, only with --no-collapse-single-gene) labels to drop")
    p.add_argument("--qc-column", default="passes_qc")
    p.add_argument("--normalize-method", default="log1pPF", choices=["log1pPF", "logCP10k", "PFlog1pPF"],
                   help="concat loader only; the per-file loader always uses log1pPF")
    p.add_argument("--min-cells", type=int, default=10)
    p.add_argument("--override-filter", action="store_true")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--num-threads", type=int, default=-1)
    p.add_argument("--seed", type=int, default=1448145)
    p.add_argument("--max-cells", type=int, default=None, help="Subsample to this many cells (testing; concat only)")
    p.add_argument("--max-per-perturbation", type=int, default=500, help="For --choose-features de")
    p.add_argument("--mean-expression", type=float, default=0.05, help="For --choose-features expression")
    args = p.parse_args(argv)

    if args.loader == "per-file":
        if args.choose_features == "de":
            p.error("--choose-features de needs --loader concat (BRISC find_markers on one object)")
        if args.normalize_method != "log1pPF":
            p.error("--loader per-file implements log1pPF only")
        if args.max_cells is not None:
            p.error("--max-cells is only supported with --loader concat")
    elif args.group_column:
        p.error("--group-column needs --loader per-file")
    return args


def read_features_file(path):
    with open(path) as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def fit_kwargs(args):
    """learn_dictionary* keyword arguments shared by both loaders."""
    return dict(
        collapse_single_gene=not args.no_collapse_single_gene,
        control_class=args.control_class,
        exclude_classes=tuple(args.exclude_classes),
        min_cells=args.min_cells,
        override_filter=args.override_filter,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )


def run_per_file(args, paths):
    """Per-file loader: load blocks -> choose features -> restrict -> global log1pPF -> fit.

    Returns the dictionary, or with ``--group-column`` a generator of ``(group, stats, dictionary)``.
    """
    group_cols = [args.group_column] if args.group_column else []
    loaded = load_qc_blocks(paths, qc_column=args.qc_column, label_column=args.perturbation_column,
                            obs_columns=group_cols, num_threads=args.num_threads)
    blocks, var_names = loaded["blocks"], loaded["var_names"]
    n_cells = sum(b.shape[0] for b in blocks)
    log(f"[run] loaded {n_cells} QC-passing cells x {len(var_names)} genes (raw) in {len(blocks)} blocks")

    if args.features_file:
        features = read_features_file(args.features_file)
        log(f"[run] {len(features)} features from {args.features_file}")
    else:
        features = features_by_mean_expression(loaded["gene_sums"], n_cells, var_names, args.mean_expression)
        log(f"[run] chose {len(features)} features via method='expression' (raw counts)")

    log(f"[run] restricting {len(blocks)} blocks to {len(features)} genes")
    blocks, kept = restrict_blocks(blocks, var_names, features)
    log("[run] normalizing (log1pPF, one global target across all files)")
    num_threads = os.cpu_count() if args.num_threads == -1 else args.num_threads
    mean_l = normalize_log1ppf(blocks, loaded["lib_sizes"], num_threads=num_threads)
    log(f"[run] normalized (mean library size {mean_l:.1f})")

    if args.group_column:
        return iter_dictionaries_by_group(
            RowStackedCSR(blocks), labels=loaded["labels"], groups=loaded["obs"][args.group_column],
            features=kept, min_control_cells=args.min_control_cells, **fit_kwargs(args),
        )
    return learn_dictionary_from_matrix(RowStackedCSR(blocks), labels=loaded["labels"], features=kept,
                                        **fit_kwargs(args))


def run_concat(args, paths):
    """Concat loader: BRISC concat_obs -> choose features -> normalize() -> restrict -> fit."""
    # Load QC-filtered RAW counts first: expression-based feature selection must see raw counts
    # (matching R ChooseFeatures), so it runs before normalization.
    sc = load_qc(paths, qc_column=args.qc_column, obs_columns=[args.perturbation_column],
                 num_threads=args.num_threads)
    log(f"[run] loaded {sc.shape[0]} QC-passing cells x {sc.shape[1]} genes (raw)")

    features = None
    if args.features_file:
        features = read_features_file(args.features_file)
        log(f"[run] {len(features)} features from {args.features_file}")
    elif args.choose_features == "expression":
        features = choose_features(sc, method="expression", mean_expression=args.mean_expression)
        log(f"[run] chose {len(features)} features via method='expression' (raw counts)")

    # Normalize for fingerprint learning (and DE feature selection, which uses normalized data).
    log(f"[run] normalizing ({args.normalize_method})")
    sc = normalize_counts(sc, normalize_method=args.normalize_method)
    log("[run] normalized")

    if features is None:  # --choose-features de
        features = choose_features(
            sc,
            method="de",
            perturbation_column=args.perturbation_column,
            control_class=args.control_class,
            max_per_perturbation=args.max_per_perturbation,
        )
        log(f"[run] chose {len(features)} features via method='de' (normalized)")

    # Memory: drop non-selected genes now (after normalization, which needs all-gene cell totals) so
    # the full-genes matrix is freed before the feature-matrix step.
    log(f"[run] restricting to {len(features)} selected genes")
    sc = restrict_to_features(sc, features)
    log(f"[run] restricted to {sc.shape[1]} genes")

    return learn_dictionary(
        paths=None,
        features=features,
        perturbation_column=args.perturbation_column,
        max_cells=args.max_cells,
        sc=sc,
        **fit_kwargs(args),
    )


def group_file_name(group, used):
    """Filesystem-safe, unique ``<group>.rds`` name (subclass labels contain spaces and slashes)."""
    stem = re.sub(r"[^A-Za-z0-9.-]+", "_", group).strip("_") or "group"
    name, i = stem, 1
    while name in used:
        name, i = f"{stem}.{i}", i + 1
    used.add(name)
    return f"{name}.rds"


def write_group_dictionaries(groups, out_dir):
    """Write one .rds per group as it is learned, plus summary.tsv (one row per group, flushed as it
    goes so a killed job still leaves a usable summary)."""
    os.makedirs(out_dir, exist_ok=True)
    cols = ["group", "file", "n_cells", "n_control", "n_candidates", "n_kept", "status"]
    summary = os.path.join(out_dir, "summary.tsv")
    used, n_groups, n_written = set(), 0, 0
    with open(summary, "w") as fh:
        fh.write("\t".join(cols) + "\n")
        for group, stats, result in groups:
            fname = ""
            if result is not None:
                fname = group_file_name(group, used)
                write_dictionary_rds(result, os.path.join(out_dir, fname))
                n_written += 1
            row = {"group": group, "file": fname, **stats}
            fh.write("\t".join(str(row[c]) for c in cols) + "\n")
            fh.flush()
            n_groups += 1
    log(f"[run] wrote {n_written} of {n_groups} group dictionaries -> {out_dir} (see {summary})")
    if n_written == 0:
        sys.exit("[run] no fingerprints learned for any group")


def main(argv=None):
    args = parse_args(argv)

    if args.h5ad_dir:
        paths = sorted(glob.glob(os.path.join(args.h5ad_dir, "*.h5ad")))
        if not paths:
            sys.exit(f"No h5ad files found in {args.h5ad_dir}")
    else:
        paths = args.h5ad

    log(f"[run] {len(paths)} h5ad file(s), loader={args.loader}")
    result = (run_per_file if args.loader == "per-file" else run_concat)(args, paths)
    if args.group_column:
        write_group_dictionaries(result, args.out)
        return
    if result is None:
        sys.exit("[run] no fingerprints learned")

    log("[run] fit done; writing .rds")
    write_dictionary_rds(result, args.out)
    log("[run] done")


if __name__ == "__main__":
    main()
