"""Write the learned dictionary as an R ``.rds`` compatible with the existing R downstream.

The contract (see CLAUDE.md) is::

    structure(
      list(
        fingerprints = list(
          Lambdas = <k x P matrix, rownames=features, colnames=perturbations>,
          dists   = <named list of <k x 2 matrix>, names=perturbations, rownames=features>
        ),
        order = NULL,
        data  = NULL
      ),
      class = "dictionary"
    )

We build it in R space through ``ryp`` (BRISC's R bridge) so the list/matrix structure is native R.
"""

from __future__ import annotations

import numpy as np
from ryp import r, to_r

from .logs import log


def write_dictionary_rds(result, path: str):
    """Serialize a :func:`learn_dictionary` result to ``path`` as a ``dictionary``-class ``.rds``.

    Parameters
    ----------
    result : dict
        Output of :func:`brisc_fingerprinting.learn_dictionary` (keys ``fingerprints``, ``names``,
        ``features``).
    path : str
        Output ``.rds`` path.
    """
    Lambdas = np.asarray(result["fingerprints"]["Lambdas"], dtype=float)
    dists = result["fingerprints"]["dists"]
    names = list(result["names"])
    features = list(result["features"])

    # ryp hands numpy arrays/lists to R as arrow-backed ALTREP vectors, which saveRDS serializes as
    # arrow ALTVEC objects: readRDS in a session without the arrow package then returns garbage.
    # Copy every vector into plain R memory before saving (arithmetic / paste0 allocate fresh vectors;
    # `+ 0` keeps the dim attribute).
    r("materialize_num <- function(x) x + 0; materialize_chr <- function(x) paste0(x)")

    to_r(Lambdas, "Lambdas_mat")
    to_r(features, "feat_names")
    to_r(names, "pert_names")
    r("Lambdas_mat <- materialize_num(Lambdas_mat); "
      "feat_names <- materialize_chr(feat_names); pert_names <- materialize_chr(pert_names)")
    r("rownames(Lambdas_mat) <- feat_names; colnames(Lambdas_mat) <- pert_names")

    r('dists_list <- vector("list", length(pert_names)); names(dists_list) <- pert_names')
    for i, d in enumerate(dists):
        to_r(np.asarray(d, dtype=float), "tmp_d")
        r(
            "tmp_d <- materialize_num(tmp_d); rownames(tmp_d) <- feat_names; colnames(tmp_d) <- NULL; "
            f"dists_list[[{i + 1}]] <- tmp_d"
        )

    to_r(path, "out_path")
    r(
        """
        dictionary <- structure(
          list(
            fingerprints = list(Lambdas = Lambdas_mat, dists = dists_list),
            order = NULL,
            data  = NULL
          ),
          class = "dictionary"
        )
        saveRDS(dictionary, out_path)
        """
    )
    log(f"[write_dictionary_rds] wrote {Lambdas.shape[1]} fingerprints x {Lambdas.shape[0]} features -> {path}")
