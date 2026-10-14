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
from .scaling import build_feature_matrix, gene_mean_sd, load_and_normalize


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

    # Perturbation labels, with control relabeled and excluded classes dropped.
    pert = np.asarray(sc.obs[perturbation_column].to_list(), dtype=object)

    if collapse_single_gene:
        # Single-gene dictionary: strip control tokens, collapse Gene|control -> Gene, pool all
        # control guides into 'control', drop true multi-gene combos. See decisions_log.md §5-6.
        pert, keep_mask = collapse_to_single_gene(
            pert, control_tokens=control_tokens, control_prefixes=control_prefixes,
            control_label="control",
        )
        if not keep_mask.all():
            sc = sc.filter_obs(keep_mask)
            pert = pert[keep_mask]
    else:
        # Legacy behavior: relabel control_class -> 'control', drop exclude_classes verbatim.
        control_set = {control_class} if isinstance(control_class, str) else set(control_class)
        exclude_set = set(exclude_classes)
        keep_mask = (~np.isin(pert, list(exclude_set)) if exclude_set
                     else np.ones(len(pert), dtype=bool))
        if exclude_set:
            sc = sc.filter_obs(keep_mask)
            pert = pert[keep_mask]
        pert = np.where(np.isin(pert, list(control_set)), "control", pert)

    n_control = int(np.sum(pert == "control"))
    if n_control == 0:
        raise ValueError("No control cells found after label processing.")
    n_perts = len(set(pert.tolist()) - {"control"})
    print(f"[learn_dictionary] {n_control} control cells, {n_perts} candidate perturbations "
          f"across {len(pert)} cells")

    # Sparse feature matrix + global per-gene stats.
    Xf, kept_features = build_feature_matrix(sc, list(features))
    gene_mean, gene_sd = gene_mean_sd(Xf)

    fingerprints = fit(
        Xf,
        gene_mean,
        gene_sd,
        perturbation=pert,
        features=kept_features,
        control_label="control",
        min_cells=min_cells,
        override_filter=override_filter,
        n_jobs=n_jobs,
        seed=seed,
    )
    if fingerprints is None:
        return None

    return {
        "fingerprints": {"Lambdas": fingerprints["Lambdas"], "dists": fingerprints["dists"]},
        "names": fingerprints["names"],
        "features": fingerprints["features"],
    }
