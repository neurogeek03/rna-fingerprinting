"""Perturbation-label handling for the whole-brain Perturb-seq atlas.

The ``gene_target`` column mixes single-gene perturbations, control guides, and ``|``-delimited
multi-guide combinations (e.g. ``Abat|Brd1|Trpm3``, ``Abca1|Non_target``). For a single-gene
dictionary we collapse each label to its real gene after stripping control tokens:

- 0 real genes -> control
- 1 real gene  -> that gene (so ``Abca1``, ``Abca1|Non_target``, ``Abca1|Safe_target_40`` all map
  to ``Abca1``, recovering the gene+control cells)
- >= 2 real genes -> excluded (uninterpretable combinatorial perturbation)

Control tokens (see ``brisc_adaptation/decisions_log.md``): ``Negative`` (no guide detected),
``Non_target`` (non-cutting guide), and any ``Safe_target_*`` (cutting safe-harbor guide). All three
are pooled into the control class.
"""

from __future__ import annotations

import numpy as np

DEFAULT_CONTROL_TOKENS = ("Negative", "Non_target")
DEFAULT_CONTROL_PREFIXES = ("Safe_target",)


def _is_control_token(tok, control_tokens, control_prefixes):
    return tok in control_tokens or any(tok.startswith(p) for p in control_prefixes)


def collapse_to_single_gene(
    labels,
    control_tokens=DEFAULT_CONTROL_TOKENS,
    control_prefixes=DEFAULT_CONTROL_PREFIXES,
    control_label="control",
    sep="|",
):
    """Collapse raw perturbation labels to single-gene perturbations + a pooled control.

    Parameters
    ----------
    labels : sequence of str
        Raw per-cell ``gene_target`` labels.
    control_tokens : tuple of str
        Exact labels treated as control guides (pooled into ``control_label``).
    control_prefixes : tuple of str
        Prefixes treated as control guides (e.g. ``'Safe_target'`` matches ``Safe_target_40``).
    control_label : str
        Label assigned to control cells (default ``'control'``).
    sep : str
        Delimiter between guides within a label (default ``'|'``).

    Returns
    -------
    (np.ndarray[object], np.ndarray[bool])
        ``new_labels`` (gene name, ``control_label``, or None for excluded cells) and ``keep_mask``
        (False for excluded multi-gene-combo cells).
    """
    labels = np.asarray(labels, dtype=object)
    new = np.empty(len(labels), dtype=object)
    keep = np.ones(len(labels), dtype=bool)

    # Map each distinct raw label once.
    mapping = {}
    for lab in set(labels.tolist()):
        real = [p for p in str(lab).split(sep)
                if not _is_control_token(p, control_tokens, control_prefixes)]
        if len(real) == 0:
            mapping[lab] = control_label
        elif len(real) == 1:
            mapping[lab] = real[0]
        else:
            mapping[lab] = None  # multi-gene combo -> exclude

    for i, lab in enumerate(labels):
        m = mapping[lab]
        if m is None:
            keep[i] = False
            new[i] = None
        else:
            new[i] = m
    return new, keep
