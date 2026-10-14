#!/bin/bash
#SBATCH --job-name=brisc_dict
#SBATCH --account=rrg-shreejoy_cpu       # CPU allocation (also: def-shreejoy_cpu)
#SBATCH --time=06:00:00                  # estimate; the control-cell PCA + 14-file load dominate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=249G                       # peak: densifying ~1-2M control cells for the 30-PC PCA
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out

# Learn the perturbation dictionary on the full QC-passing whole-brain atlas with BRISC, in one
# process (load 14 h5ads -> concat -> QC -> normalize -> feature select -> collapse -> fit -> .rds).
#
#   cd /scratch/mfafouti/rna-fingerprinting && sbatch python/sbatch/learn_dictionary.sbatch
#
# Tunables below: FEATURE_MODE (expression|de|file), OUT, and the resources above.

set -euo pipefail
module load apptainer/1.3.5

REPO=/scratch/mfafouti/rna-fingerprinting
SIF=/scratch/mfafouti/docker/brisc_bioc-v2-mkl-seurat-sklearn.sif
H5AD_DIR=/scratch/mfafouti/Psych_brain/Perturb-seq/wholebrain_atlas/h5ads
OUT="$REPO/brisc_adaptation/dictionary_wholebrain.rds"

# Feature selection: "expression" (fast), "de" (find_markers), or "file" (use FEATURES_FILE).
FEATURE_MODE=expression
FEATURES_FILE="$REPO/features.txt"

# Balance joblib threads (per-perturbation loop) against BLAS threads (one-time control PCA).
NJOBS=$(( SLURM_CPUS_PER_TASK / 2 ))
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2

case "$FEATURE_MODE" in
  file) FEAT_ARGS=(--features-file "$FEATURES_FILE") ;;
  *)    FEAT_ARGS=(--choose-features "$FEATURE_MODE") ;;
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
