#!/usr/bin/env python
"""Print the X dtypes/nnz and obs column encodings of h5ad files (read-only, metadata only).

Use to diagnose concat_obs dtype mismatches and to estimate in-memory size before a big load::

    apptainer exec "$SIF" python python/scripts/inspect_h5ad_schema.py <file.h5ad> [...] \
        --columns gene_target passes_qc
"""

from __future__ import annotations

import argparse

import h5py


def describe_obs_column(g):
    """Return a short encoding/dtype string for one obs column (plain dataset or categorical group)."""
    if isinstance(g, h5py.Dataset):
        return f"{g.attrs.get('encoding-type', 'array')}:{g.dtype}"
    enc = g.attrs.get("encoding-type", "group")
    if enc == "categorical":
        cats = g["categories"]
        return f"categorical(n={cats.shape[0]}, codes={g['codes'].dtype}, cats={cats.dtype}, ordered={g.attrs.get('ordered')})"
    return str(enc)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("h5ads", nargs="+")
    p.add_argument("--columns", nargs="*", default=None, help="obs columns to report (default: all)")
    args = p.parse_args(argv)

    for path in args.h5ads:
        with h5py.File(path, "r") as f:
            X = f["X"]
            n_obs = f["obs"][f["obs"].attrs["_index"]].shape[0]
            print(f"== {path}")
            print(f"   n_obs={n_obs} X: data={X['data'].dtype} indices={X['indices'].dtype} "
                  f"indptr={X['indptr'].dtype} nnz={X['data'].shape[0]:,}")
            cols = args.columns or list(f["obs"].attrs["column-order"])
            for c in cols:
                print(f"   obs[{c}]: {describe_obs_column(f['obs'][c]) if c in f['obs'] else 'MISSING'}")


if __name__ == "__main__":
    main()
