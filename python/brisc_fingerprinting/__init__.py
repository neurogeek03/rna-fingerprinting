"""BRISC-based Python port of the memory-bottleneck steps of rna-fingerprinting.

This package replaces the R ``SCTransform`` + ``fit()``/``LearnDictionary()`` path with a
BRISC-backed implementation that scales to the full QC-passing whole-brain Perturb-seq atlas
(~6M cells) without downsampling. The learned dictionary is written as an ``.rds`` that is
byte-compatible with the existing R ``classify()``/``FingerprintCells()`` downstream.

See ``BRISC_PORT_PLAN.md`` and ``CLAUDE.md`` for the algorithm spec and compatibility contract.
"""

from .scaling import load_and_normalize, build_feature_matrix, gene_mean_sd
from .fit import fit
from .learn_dictionary import learn_dictionary
from .choose_features import choose_features
from .perturbations import collapse_to_single_gene
from .rds_io import write_dictionary_rds

__all__ = [
    "load_and_normalize",
    "build_feature_matrix",
    "gene_mean_sd",
    "fit",
    "learn_dictionary",
    "choose_features",
    "collapse_to_single_gene",
    "write_dictionary_rds",
]
