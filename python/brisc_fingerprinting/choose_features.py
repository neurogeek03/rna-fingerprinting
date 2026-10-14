"""Feature selection: the BRISC equivalent of R ``ChooseFeatures()``.

Two methods, mirroring the R API:

* ``method='de'``   -- differential-expression-based. The R code runs Seurat ``FindMarkers`` per
  perturbation *against the control class*. BRISC's :meth:`SingleCell.find_markers` is one-vs-rest
  (each perturbation vs all other cells), which scales to millions of cells. This is a deliberate
  approximation; validate against the R feature set on a small dataset before trusting it at scale.
* ``method='expression'`` -- genes whose mean raw count exceeds a threshold.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def choose_features(
    sc,
    method: str,
    perturbation_column: str | None = None,
    control_class: str | None = None,
    max_per_perturbation: int = 500,
    mean_expression: float = 0.01,
    min_detection_rate: float = 0.25,
    min_fold_change: float = 2.0,
    num_threads: int | None = None,
):
    """Select features for fingerprint estimation.

    Parameters
    ----------
    sc : brisc.SingleCell
        For ``method='de'`` this may be normalized or raw (BRISC normalizes internally as needed).
        For ``method='expression'`` this must hold **raw counts** in ``X`` (pass the object *before*
        :func:`load_and_normalize`).
    method : {'de', 'expression'}
    perturbation_column, control_class : str
        Required for ``method='de'``. The control class is excluded from the one-vs-rest comparison.
    max_per_perturbation : int
        Max markers kept per perturbation (default 500).
    mean_expression : float
        Minimum mean raw count for ``method='expression'`` (default 0.01).
    min_detection_rate, min_fold_change : float
        Passed to :meth:`SingleCell.find_markers`.

    Returns
    -------
    list[str]
        Selected feature names.
    """
    if method not in ("de", "expression"):
        raise ValueError("Method must be one of 'de', 'expression'")

    if method == "expression":
        X = sc.X
        gene_sums = np.asarray(X.sum(axis=0, dtype=np.float64)).ravel()
        return features_by_mean_expression(gene_sums, X.shape[0], list(sc.var_names), mean_expression)

    # method == 'de'
    if perturbation_column is None or control_class is None:
        raise ValueError("Please specify perturbation_column and control_class")

    markers = sc.find_markers(
        perturbation_column,
        QC_column=None,
        excluded_cell_types=control_class,
        min_detection_rate=min_detection_rate,
        min_fold_change=min_fold_change,
        all_genes=False,
        num_threads=num_threads,
    )

    # Identify the grouping column, the gene column, and a ranking column from the result schema.
    # BRISC find_markers returns pareto-optimal markers with effect-size columns (fold_change,
    # detection_rate) rather than a p-value, so rank by effect size (descending) when present;
    # fall back to a p-value column (ascending) only if BRISC's schema ever changes.
    cols = markers.columns
    group_col = _pick(cols, [perturbation_column, "cell_type", "cell_types", "group"])
    gene_col = _pick(cols, ["gene", "feature", "gene_name", "name"])
    rank_col = _pick(cols, ["fold_change", "detection_rate"], required=False)
    descending = rank_col is not None
    if rank_col is None:
        rank_col = _pick(cols, ["p", "p_value", "pval", "fdr", "q"], required=False)

    features: list[str] = []
    for _, sub in markers.group_by(group_col):
        if rank_col is not None:
            sub = sub.sort(rank_col, descending=descending)
        features.extend(sub[gene_col].to_list()[:max_per_perturbation])

    # Preserve first-seen order while de-duplicating.
    seen = set()
    out = []
    for f in features:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def features_by_mean_expression(gene_sums, n_cells: int, var_names, mean_expression: float = 0.01):
    """Genes whose mean RAW count (``gene_sums / n_cells``) exceeds ``mean_expression``, in var order.

    Split out so the per-file loader (which only has summed column totals) shares the same rule.
    """
    mean = np.asarray(gene_sums, dtype=np.float64) / n_cells
    return [var_names[i] for i in np.flatnonzero(mean > mean_expression)]


def _pick(columns, candidates, required=True):
    for c in candidates:
        if c in columns:
            return c
    if required:
        raise KeyError(f"None of {candidates} found in find_markers columns: {list(columns)}")
    return None
