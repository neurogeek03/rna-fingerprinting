#!/bin/bash
# Learn the whole-brain perturbation dictionary INTERACTIVELY.
#
# Run this from inside an interactive compute allocation, NOT on the login node.
# MEMORY: the full atlas needs ~400G RAM (QC counts ~196GB/copy, ~392GB transient peak), so request
# a large-memory allocation, e.g. (edit --account for this cluster):
#     salloc --nodes=1 --ntasks=1 --cpus-per-task=48 --mem=750G --time=02:00:00 \
#            --account=<ACCOUNT> srun --pty bash
#     SIF=<path.sif> H5AD_DIR=<h5ad_dir> bash brisc_adaptation/run_dictionary_interactive.sh
# (A single file / small subset needs far less memory.)
#
# Required env: SIF, H5AD_DIR. Optional: OUT, FEATURE_MODE (expression|de|file), MEAN_EXPRESSION,
# NJOBS, OMP_NUM_THREADS, APPTAINER_MODULE. See brisc_adaptation/AGENT_BRIEFING.md.

set -euo pipefail
module load "${APPTAINER_MODULE:-apptainer}" >/dev/null 2>&1 || true

# Repo root derived from this script's location (portable across clusters / clone paths).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"

# --- cluster-specific: override via env vars (see brisc_adaptation/AGENT_BRIEFING.md) ---
SIF=${SIF:?set SIF to the BRISC .sif path (pull docker://marlenfafouti/brisc:bioc-v2-mkl-seurat-sklearn)}
H5AD_DIR=${H5AD_DIR:?set H5AD_DIR to the Perturb-seq h5ad directory on this cluster}
OUT=${OUT:-$REPO/brisc_adaptation/dictionary_wholebrain.rds}

# Feature selection: expression (raw-count mean threshold) | de (find_markers) | file
FEATURE_MODE=${FEATURE_MODE:-expression}
FEATURES_FILE=${FEATURES_FILE:-$REPO/features.txt}
MEAN_EXPRESSION=${MEAN_EXPRESSION:-0.05}

# Cores: use the allocation's cpus; split between joblib (per-perturbation) and BLAS (control PCA).
CORES=${SLURM_CPUS_ON_NODE:-$(nproc)}
NJOBS=${NJOBS:-$(( CORES / 2 > 0 ? CORES / 2 : 1 ))}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} OPENBLAS_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${OMP_NUM_THREADS:-2}

case "$FEATURE_MODE" in
  file) FEAT_ARGS=(--features-file "$FEATURES_FILE") ;;
  *)    FEAT_ARGS=(--choose-features "$FEATURE_MODE" --mean-expression "$MEAN_EXPRESSION") ;;
esac

echo "host=$(hostname) cores=$CORES njobs=$NJOBS omp=$OMP_NUM_THREADS feature_mode=$FEATURE_MODE"
echo "h5ad_dir=$H5AD_DIR"
echo "out=$OUT"

# collapse-to-single-gene + pooled control (Negative/Non_target/Safe_target_*) are the defaults.
time apptainer exec "$SIF" python "$REPO/python/scripts/run_learn_dictionary.py" \
    --h5ad-dir "$H5AD_DIR" \
    "${FEAT_ARGS[@]}" \
    --perturbation-column gene_target \
    --qc-column passes_qc \
    --n-jobs "$NJOBS" \
    --out "$OUT"
echo "DICT_EXIT=$? -> $OUT"
