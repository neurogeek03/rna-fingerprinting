#!/bin/bash
#SBATCH --job-name=brisc_dict
#SBATCH --account=CHANGE_ME              # <<< EDIT PER CLUSTER (or `sbatch --account=...`)
#SBATCH --time=03:00:00                  # estimate; 14-file load/normalize dominates
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=750G                        # full-atlas QC counts ~196GB/copy; ~392GB transient peak.
                                          # Needs a large-memory node (nibi=766GB). Lower for subsets.
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

# Learn the perturbation dictionary on the full QC-passing whole-brain atlas with BRISC, in one
# process (load 14 h5ads -> concat -> QC -> normalize -> feature select -> collapse -> fit -> .rds).
#
#   REPO=/path/to/rna-fingerprinting SIF=/path/to.sif H5AD_DIR=/path/to/h5ads \
#     sbatch --account=<acct> $REPO/python/sbatch/learn_dictionary.sbatch
#
# See brisc_adaptation/AGENT_BRIEFING.md for the per-cluster config checklist.

set -euo pipefail
module load "${APPTAINER_MODULE:-apptainer}"

# --- cluster-specific: edit here or pass via sbatch --export=ALL,REPO=...,SIF=...,H5AD_DIR=... ---
REPO=${REPO:-$SLURM_SUBMIT_DIR}          # repo root; defaults to the directory you submit from
SIF=${SIF:?set SIF (pull docker://marlenfafouti/brisc:bioc-v2-mkl-seurat-sklearn)}
H5AD_DIR=${H5AD_DIR:?set H5AD_DIR to the Perturb-seq h5ad directory on this cluster}
OUT=${OUT:-$REPO/brisc_adaptation/dictionary_wholebrain.rds}

# Feature selection: "expression" (fast), "de" (find_markers), or "file" (use FEATURES_FILE).
FEATURE_MODE=${FEATURE_MODE:-expression}
FEATURES_FILE=${FEATURES_FILE:-$REPO/features.txt}
MEAN_EXPRESSION=${MEAN_EXPRESSION:-0.05}

# Balance joblib threads (per-perturbation loop) against BLAS threads (one-time control PCA).
NJOBS=${NJOBS:-$(( SLURM_CPUS_PER_TASK / 2 ))}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

case "$FEATURE_MODE" in
  file) FEAT_ARGS=(--features-file "$FEATURES_FILE") ;;
  *)    FEAT_ARGS=(--choose-features "$FEATURE_MODE" --mean-expression "$MEAN_EXPRESSION") ;;
esac

echo "node=$(hostname) cpus=$SLURM_CPUS_PER_TASK njobs=$NJOBS feature_mode=$FEATURE_MODE"
# collapse-to-single-gene + pooled control (Negative/Non_target/Safe_target_*) are the defaults.
apptainer exec "$SIF" python "$REPO/python/scripts/run_learn_dictionary.py" \
    --h5ad-dir "$H5AD_DIR" \
    "${FEAT_ARGS[@]}" \
    --perturbation-column gene_target \
    --qc-column passes_qc \
    --n-jobs "$NJOBS" \
    --out "$OUT"
echo "DICT_EXIT=$?"
