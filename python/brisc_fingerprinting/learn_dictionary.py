"""Public wrapper: the BRISC equivalent of R ``LearnDictionary()``.

Orchestrates load -> normalize -> feature matrix -> ``fit()`` and returns a dictionary structure
ready to be written as a compatible ``.rds`` by ``rds_io.write_dictionary_rds``.
"""

from __future__ import annotations

import numpy as np

from .fit import fit
from .perturbations import (
    DEFAULT_CONTROL_PREFIXES,
    DEFAULT_CONTROL_TOKENS,
    collapse_to_single_gene,
)
from .logs import log
from .scaling import build_feature_matrix, gene_mean_sd, load_and_normalize

# Label for cells dropped from the dictionary (multi-gene combos, excluded classes).
EXCLUDED_LABEL = "__excluded__"


def learn_dictionary(
    paths,
    features,
    perturbation_column: str = "gene_target",
    control_class="Negative",
    exclude_classes=(),
    collapse_single_gene: bool = True,
    control_tokens=DEFAULT_CONTROL_TOKENS,
    control_prefixes=DEFAULT_CONTROL_PREFIXES,
    qc_column: str = "passes_qc",
    normalize_method: str = "log1pPF",
    min_cells: int = 10,
    override_filter: bool = False,
    n_jobs: int = 1,
    num_threads: int = -1,
    seed: int | None = 1448145,
    max_cells: int | None = None,
    sc=None,
):
    """Learn a perturbation dictionary from one or more Perturb-seq h5ads.

    Parameters
    ----------
    paths : sequence of str
        h5ad paths. Ignored if ``sc`` is given.
    features : sequence of str
        Feature (gene) names to use for fingerprint estimation.
    perturbation_column : str
        obs column holding perturbation identities (default ``'gene_target'``).
    control_class : str or sequence of str
        Label(s) treated as the control condition; relabeled internally to ``'control'``.
    exclude_classes : sequence of str
        Perturbation labels to drop entirely (e.g. ``('Non_target',)`` to keep it out of the
        learned dictionary without treating it as control).
    qc_column, normalize_method, num_threads
        Passed to :func:`load_and_normalize`.
    min_cells, override_filter, n_jobs, seed
        Passed to :func:`fit`.
    max_cells : int or None
        If set, randomly subsample to this many cells after QC (useful for quick tests).
    sc : brisc.SingleCell or None
        A pre-loaded, already-normalized SingleCell to use instead of loading from ``paths``.

    Returns
    -------
    dict or None
        ``{'fingerprints': {'Lambdas', 'dists'}, 'names', 'features'}`` or None if nothing survived.
    """
    if sc is None:
        sc = load_and_normalize(
            paths, qc_column=qc_column, normalize_method=normalize_method, num_threads=num_threads
        )

    if max_cells is not None and sc.shape[0] > max_cells:
        sc = sc.subsample_obs(n=max_cells, QC_column=None, seed=seed or 0)

    Xf, kept_features = build_feature_matrix(sc, list(features))
    return learn_dictionary_from_matrix(
        Xf,
        labels=np.asarray(sc.obs[perturbation_column].to_list(), dtype=object),
        features=kept_features,
        control_class=control_class,
        exclude_classes=exclude_classes,
        collapse_single_gene=collapse_single_gene,
        control_tokens=control_tokens,
        control_prefixes=control_prefixes,
        min_cells=min_cells,
        override_filter=override_filter,
        n_jobs=n_jobs,
        seed=seed,
    )


def learn_dictionary_from_matrix(
    Xf,
    labels,
    features,
    control_class="Negative",
    exclude_classes=(),
    collapse_single_gene: bool = True,
    control_tokens=DEFAULT_CONTROL_TOKENS,
    control_prefixes=DEFAULT_CONTROL_PREFIXES,
    min_cells: int = 10,
    override_filter: bool = False,
    n_jobs: int = 1,
    seed: int | None = 1448145,
):
    """Label processing + global gene stats + :func:`fit`, on an already-normalized feature matrix.

    Parameters
    ----------
    Xf : csr matrix or :class:`~brisc_fingerprinting.stacked.RowStackedCSR`
        Normalized cells x k matrix whose columns are ``features``.
    labels : array of str
        Raw per-cell perturbation labels (e.g. ``gene_target``), one per row of ``Xf``.
    features : list[str]
        Column names of ``Xf``.

    Other parameters and the return value are as in :func:`learn_dictionary`.
    """
    pert, keep_mask = _process_labels(
        labels, control_class, exclude_classes, collapse_single_gene, control_tokens, control_prefixes,
    )
    keep_rows = np.flatnonzero(keep_mask)

    n_control = int(np.sum(pert == "control"))
    if n_control == 0:
        raise ValueError("No control cells found after label processing.")
    n_perts = len(set(pert[keep_rows].tolist()) - {"control"})
    log(f"[learn_dictionary] {n_control} control cells, {n_perts} candidate perturbations "
        f"across {keep_rows.size} cells ({len(pert) - keep_rows.size} excluded)")

    # Global per-gene stats over kept cells.
    log(f"[learn_dictionary] feature matrix {type(Xf).__name__} {Xf.shape} nnz={Xf.nnz:,} "
        f"data={Xf.dtype}; computing gene mean/sd")
    gene_mean, gene_sd = gene_mean_sd(Xf, rows=keep_rows)

    fingerprints = fit(
        Xf,
        gene_mean,
        gene_sd,
        perturbation=pert,
        features=list(features),
        control_label="control",
        min_cells=min_cells,
        override_filter=override_filter,
        n_jobs=n_jobs,
        seed=seed,
        exclude_label=EXCLUDED_LABEL,
    )
    return _as_dictionary(fingerprints)


def iter_dictionaries_by_group(
    Xf,
    labels,
    groups,
    features,
    min_control_cells: int = 30,
    control_class="Negative",
    exclude_classes=(),
    collapse_single_gene: bool = True,
    control_tokens=DEFAULT_CONTROL_TOKENS,
    control_prefixes=DEFAULT_CONTROL_PREFIXES,
    min_cells: int = 10,
    override_filter: bool = False,
    n_jobs: int = 1,
    seed: int | None = 1448145,
):
    """Learn one dictionary per group (e.g. cell-type subclass), as the authors do per cell type.

    Each group gets its own control baseline (its own control cells), its own per-gene z-score stats
    and its own fingerprints; ``fit()`` is unchanged. Only one group's rows are copied out of ``Xf``
    at a time. Groups with fewer than ``min_control_cells`` controls are skipped (preprint: cell
    types with < 30 control cells were excluded).

    Parameters
    ----------
    groups : array of str
        Per-cell group label (e.g. ``predicted_subclass``), one per row of ``Xf``.
    min_control_cells : int
        Minimum control cells for a group to get a dictionary (default 30).

    Other parameters are as in :func:`learn_dictionary_from_matrix`.

    Yields
    ------
    (group, stats, result)
        ``stats`` is a dict (``n_cells``, ``n_control``, ``n_candidates``, ``n_kept``, ``status``);
        ``result`` is as returned by :func:`learn_dictionary_from_matrix`, or None.
    """
    pert, keep_mask = _process_labels(
        labels, control_class, exclude_classes, collapse_single_gene, control_tokens, control_prefixes,
    )
    # Integer group codes: one pass over the string labels instead of one per group.
    names, codes = np.unique(np.asarray(groups, dtype=object).astype(str), return_inverse=True)
    log(f"[by_group] {len(names)} groups")

    for gi, g in enumerate(names.tolist()):
        rows = np.flatnonzero((codes == gi) & keep_mask)
        pert_g = pert[rows]
        is_ctrl = pert_g == "control"
        n_control = int(is_ctrl.sum())
        labs, counts = np.unique(pert_g[~is_ctrl], return_counts=True)
        stats = {"n_cells": int(rows.size), "n_control": n_control,
                 "n_candidates": int(np.sum(counts >= min_cells)), "n_kept": 0}

        skip = None
        if n_control < min_control_cells:
            skip = f"skipped: <{min_control_cells} controls"
        elif stats["n_candidates"] == 0:
            skip = f"skipped: no perturbation with >={min_cells} cells"
        if skip:
            stats["status"] = skip
            log(f"[by_group] {gi + 1}/{len(names)} {g!r}: {skip} ({rows.size} cells, {n_control} controls)")
            yield g, stats, None
            continue

        log(f"[by_group] {gi + 1}/{len(names)} {g!r}: {rows.size} cells, {n_control} controls, "
            f"{stats['n_candidates']} candidate perturbations")
        Xg = Xf[rows]  # this group's rows only; pert_g has no excluded cells
        gene_mean, gene_sd = gene_mean_sd(Xg)
        fingerprints = fit(
            Xg, gene_mean, gene_sd, perturbation=pert_g, features=list(features),
            control_label="control", min_cells=min_cells, override_filter=override_filter,
            n_jobs=n_jobs, seed=seed,
        )
        del Xg
        result = _as_dictionary(fingerprints)
        stats["n_kept"] = 0 if result is None else len(result["names"])
        stats["status"] = "ok" if result is not None else "no fingerprint passed the filter"
        yield g, stats, result


def _process_labels(labels, control_class, exclude_classes, collapse_single_gene, control_tokens,
                    control_prefixes):
    """Perturbation labels with control relabeled 'control' and dropped cells EXCLUDED_LABEL.

    Returns ``(pert, keep_mask)``.
    """
    pert = np.asarray(labels, dtype=object)

    if collapse_single_gene:
        # Single-gene dictionary: strip control tokens, collapse Gene|control -> Gene, pool all
        # control guides into 'control', drop true multi-gene combos. See decisions_log.md §5-6.
        pert, keep_mask = collapse_to_single_gene(
            pert, control_tokens=control_tokens, control_prefixes=control_prefixes,
            control_label="control",
        )
    else:
        # Legacy behavior: relabel control_class -> 'control', drop exclude_classes verbatim.
        control_set = {control_class} if isinstance(control_class, str) else set(control_class)
        exclude_set = set(exclude_classes)
        keep_mask = (~np.isin(pert, list(exclude_set)) if exclude_set
                     else np.ones(len(pert), dtype=bool))
        pert = np.where(np.isin(pert, list(control_set)), "control", pert)

    # Dropped cells are relabeled (fit() ignores them) rather than removed with filter_obs, which
    # would copy the whole matrix; global gene stats are computed over kept cells only.
    pert = np.where(keep_mask, pert, EXCLUDED_LABEL)
    return pert, keep_mask


def _as_dictionary(fingerprints):
    """``fit()`` output -> the dict :func:`write_dictionary_rds` expects (None stays None)."""
    if fingerprints is None:
        return None
    return {
        "fingerprints": {"Lambdas": fingerprints["Lambdas"], "dists": fingerprints["dists"]},
        "names": fingerprints["names"],
        "features": fingerprints["features"],
    }
