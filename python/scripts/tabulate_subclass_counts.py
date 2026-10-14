#!/usr/bin/env python
"""Tabulate cells per subclass, controls per subclass and cells per knockout x subclass (obs only).

Reads only ``passes_qc``, ``gene_target`` and ``predicted_subclass`` from each h5ad's obs with h5py
(never touches X), so it runs on a login node in about a minute::

    apptainer exec "$SIF" python python/scripts/tabulate_subclass_counts.py \
        --h5ad-dir "$H5AD_DIR" --out-dir brisc_adaptation/subclass_counts

Labels are collapsed exactly as ``learn_dictionary`` does (``collapse_to_single_gene``: control guides
pooled into 'control', multi-gene combos dropped). Writes ``subclass_summary.tsv`` (one row per
subclass), ``pair_counts.tsv`` (one row per knockout x subclass) and ``report.txt``.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import h5py
import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brisc_fingerprinting.perturbations import collapse_to_single_gene  # noqa: E402

CONTROL = "control"


def read_categorical(g):
    """Per-cell strings from an h5ad categorical obs group (missing codes -> None)."""
    cats = np.asarray([c.decode() if isinstance(c, bytes) else str(c) for c in g["categories"][:]],
                      dtype=object)
    codes = g["codes"][:]
    out = np.full(codes.shape[0], None, dtype=object)
    ok = codes >= 0
    out[ok] = cats[codes[ok]]
    return out


def load_obs(path, qc_column, label_column, group_column):
    """QC-passing cells' (collapsed perturbation label, group) for one file."""
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        qc = obs[qc_column][:].astype(bool)
        labels = read_categorical(obs[label_column])[qc]
        groups = read_categorical(obs[group_column])[qc]
    # Collapse distinct labels once, then map per cell.
    uniq, inv = np.unique(labels.astype(str), return_inverse=True)
    collapsed, keep = collapse_to_single_gene(uniq, control_label=CONTROL)
    pert = collapsed[inv]
    keep_cells = keep[inv] & (groups != None)  # noqa: E711 (object array comparison)
    return pert[keep_cells], groups[keep_cells], int(qc.sum())


def quantiles(x, qs=(0.1, 0.25, 0.5, 0.75, 0.9, 1.0)):
    x = np.asarray(x)
    if x.size == 0:
        return "n=0"
    return f"n={x.size:,}  " + "  ".join(f"q{int(q * 100)}={np.quantile(x, q):,.0f}" for q in qs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--qc-column", default="passes_qc")
    p.add_argument("--perturbation-column", default="gene_target")
    p.add_argument("--group-column", default="predicted_subclass")
    p.add_argument("--min-cells", type=int, default=10, help="fit() min cells per knockout")
    p.add_argument("--min-control-cells", type=int, default=30, help="min controls per subclass")
    p.add_argument("--downsample", type=float, default=10.0,
                   help="Fold downsample to compare against (eligibility scaled by this factor)")
    p.add_argument("--n-features", type=int, default=12306, help="k, for dense-memory estimates")
    args = p.parse_args(argv)

    paths = sorted(glob.glob(os.path.join(args.h5ad_dir, "*.h5ad")))
    if not paths:
        sys.exit(f"No h5ad files in {args.h5ad_dir}")

    perts, groups, n_qc = [], [], 0
    for path in paths:
        pt, gr, nq = load_obs(path, args.qc_column, args.perturbation_column, args.group_column)
        perts.append(pt)
        groups.append(gr)
        n_qc += nq
        print(f"{os.path.basename(path)}: {nq:,} QC cells, {pt.size:,} kept", flush=True)

    df = pl.DataFrame({"group": np.concatenate(groups).astype(str),
                       "pert": np.concatenate(perts).astype(str)})
    is_ctrl = pl.col("pert") == CONTROL

    sub = (df.group_by("group")
             .agg(pl.len().alias("n_cells"), is_ctrl.sum().alias("n_control"))
             .sort("n_cells", descending=True))
    pairs = (df.filter(~is_ctrl).group_by(["group", "pert"]).agg(pl.len().alias("n_cells"))
               .join(sub.select("group", "n_control"), on="group"))
    per_ko = df.filter(~is_ctrl).group_by("pert").agg(pl.len().alias("n_cells"))

    mc, mcc, ds = args.min_cells, args.min_control_cells, args.downsample
    elig = pairs.filter((pl.col("n_cells") >= mc) & (pl.col("n_control") >= mcc))
    elig_ds = pairs.filter((pl.col("n_cells") >= mc * ds) & (pl.col("n_control") >= mcc * ds))
    sub = sub.join(elig.group_by("group").agg(pl.len().alias("n_knockouts_eligible"),
                                              pl.col("n_cells").median().alias("median_cells_per_knockout")),
                   on="group", how="left").fill_null(0).sort("n_cells", descending=True)

    os.makedirs(args.out_dir, exist_ok=True)
    sub.write_csv(os.path.join(args.out_dir, "subclass_summary.tsv"), separator="\t")
    pairs.sort(["group", "n_cells"], descending=[False, True]).write_csv(
        os.path.join(args.out_dir, "pair_counts.tsv"), separator="\t")

    n_sub = sub.height
    ctrl = sub["n_control"].to_numpy()
    big = sub.head(5)
    k = args.n_features
    lines = [
        f"files: {len(paths)}   QC cells: {n_qc:,}   kept after label collapse: {df.height:,}",
        f"controls: {int(ctrl.sum()):,}   knockouts: {per_ko.height:,}   subclasses: {n_sub}",
        "",
        "== Pooled (whole atlas)",
        f"cells per knockout: {quantiles(per_ko['n_cells'])}",
        "",
        "== Subclasses",
        f"cells per subclass: {quantiles(sub['n_cells'])}",
        f"controls per subclass: {quantiles(ctrl)}",
        f"subclasses with >= {mcc} controls: {int((ctrl >= mcc).sum())} / {n_sub}"
        f"   (>= 1,000: {int((ctrl >= 1000).sum())};  >= k={k:,}: {int((ctrl >= k).sum())})",
        f"subclasses with >= {mcc} controls after {ds:g}x downsample: {int((ctrl >= mcc * ds).sum())}",
        "largest subclasses (cells, controls, dense R scale.data at k features, float64):",
        *[f"  {r['group']}: {r['n_cells']:,} cells, {r['n_control']:,} controls, "
          f"{r['n_cells'] * k * 8 / 1e9:,.0f} GB" for r in big.iter_rows(named=True)],
        "",
        f"== Knockout x subclass pairs (eligible: >= {mc} cells and subclass >= {mcc} controls)",
        f"all pairs: {pairs.height:,}   eligible: {elig.height:,}   "
        f"eligible after {ds:g}x downsample: {elig_ds.height:,}",
        f"cells per eligible pair: {quantiles(elig['n_cells'])}",
        f"eligible pairs with >= 150 cells: {elig.filter(pl.col('n_cells') >= 150).height:,};  "
        f">= 2,000: {elig.filter(pl.col('n_cells') >= 2000).height:,}",
        f"knockouts with >= 1 eligible subclass: {elig['pert'].n_unique():,} / {per_ko.height:,}   "
        f"(after {ds:g}x downsample: {elig_ds['pert'].n_unique():,})",
        f"eligible subclasses per knockout: "
        f"{quantiles(elig.group_by('pert').agg(pl.len().alias('n'))['n'])}",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(args.out_dir, "report.txt"), "w") as fh:
        fh.write(report + "\n")


if __name__ == "__main__":
    main()
