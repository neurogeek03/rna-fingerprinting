"""BRISC-based Python port of the memory-bottleneck steps of rna-fingerprinting.

This package replaces the R ``SCTransform`` + ``fit()``/``LearnDictionary()`` path with a
BRISC-backed implementation that scales to the full QC-passing whole-brain Perturb-seq atlas
(~6M cells) without downsampling. The learned dictionary is written as an ``.rds`` that is
byte-compatible with the existing R ``classify()``/``FingerprintCells()`` downstream.

See ``BRISC_PORT_PLAN.md`` and ``CLAUDE.md`` for the algorithm spec and compatibility contract.
"""

from .scaling import (
    load_and_normalize, load_qc, normalize_counts, restrict_to_features, build_feature_matrix, gene_mean_sd,
)
from .logs import log, rss_gb
from .fit import fit
from .learn_dictionary import iter_dictionaries_by_group, learn_dictionary, learn_dictionary_from_matrix
from .stacked import RowStackedCSR, load_qc_blocks, normalize_log1ppf, restrict_blocks
from .choose_features import choose_features, features_by_mean_expression
from .perturbations import collapse_to_single_gene
from .rds_io import write_dictionary_rds

__all__ = [
    "load_and_normalize",
    "load_qc",
    "normalize_counts",
    "restrict_to_features",
    "build_feature_matrix",
    "gene_mean_sd",
    "log",
    "rss_gb",
    "fit",
    "learn_dictionary",
    "learn_dictionary_from_matrix",
    "iter_dictionaries_by_group",
    "RowStackedCSR",
    "load_qc_blocks",
    "normalize_log1ppf",
    "restrict_blocks",
    "choose_features",
    "features_by_mean_expression",
    "collapse_to_single_gene",
    "write_dictionary_rds",
]
