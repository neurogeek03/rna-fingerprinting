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
# GROUP_COLUMN (default predicted_subclass; empty = pooled), MIN_CONTROL_CELLS, NJOBS, OMP_NUM_THREADS, APPTAINER_MODULE. See brisc_adaptation/AGENT_BRIEFING.md.

set -euo pipefail
module load "${APPTAINER_MODULE:-apptainer}" >/dev/null 2>&1 || true

# Repo root derived from this script's location (portable across clusters / clone paths).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$SCRIPT_DIR")"

# --- cluster-specific: override via env vars (see brisc_adaptation/AGENT_BRIEFING.md) ---
SIF=${SIF:?set SIF to the BRISC .sif path (pull docker://marlenfafouti/brisc:bioc-v2-mkl-seurat-sklearn)}
H5AD_DIR=${H5AD_DIR:?set H5AD_DIR to the Perturb-seq h5ad directory on this cluster}
# One dictionary per subclass (as the authors do per cell type). GROUP_COLUMN= (empty) -> one pooled
# dictionary, and OUT is then a .rds path instead of a directory.
GROUP_COLUMN=${GROUP_COLUMN-predicted_subclass}
MIN_CONTROL_CELLS=${MIN_CONTROL_CELLS:-30}
if [ -n "$GROUP_COLUMN" ]; then
  OUT=${OUT:-$REPO/brisc_adaptation/dictionaries_wholebrain_by_subclass}
  GROUP_ARGS=(--group-column "$GROUP_COLUMN" --min-control-cells "$MIN_CONTROL_CELLS")
else
  OUT=${OUT:-$REPO/brisc_adaptation/dictionary_wholebrain.rds}
  GROUP_ARGS=()
fi

# Feature selection: expression (raw-count mean threshold) | de (find_markers) | file
FEATURE_MODE=${FEATURE_MODE:-expression}
FEATURES_FILE=${FEATURES_FILE:-$REPO/features.txt}
MEAN_EXPRESSION=${MEAN_EXPRESSION:-0.05}

# joblib workers = cores/2 (192 workers measured >10x slower; see docs/thread_scaling_paradox.md).
# BLAS = all cores for the control PCA; fit() gives each worker cores/NJOBS BLAS threads in the loop.
CORES=${SLURM_CPUS_ON_NODE:-$(nproc)}
NJOBS=${NJOBS:-$(( CORES / 2 > 0 ? CORES / 2 : 1 ))}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-$CORES}; export OPENBLAS_NUM_THREADS=$OMP_NUM_THREADS MKL_NUM_THREADS=$OMP_NUM_THREADS

# Log timestamps in the host timezone (the container defaults to UTC).
export TZ=${TZ:-$(readlink -f /etc/localtime 2>/dev/null | sed -n "s|.*/zoneinfo/||p")}
[ -n "$TZ" ] || TZ=$(timedatectl show -p Timezone --value 2>/dev/null)
# Compute nodes may have neither a zoneinfo symlink nor timedatectl; fall back to the cluster's zone
# (SciNet Trillium = Toronto). Override with TZ=... on other clusters.
export TZ=${TZ:-America/Toronto}

case "$FEATURE_MODE" in
  file) FEAT_ARGS=(--features-file "$FEATURES_FILE") ;;
  *)    FEAT_ARGS=(--choose-features "$FEATURE_MODE" --mean-expression "$MEAN_EXPRESSION") ;;
esac

echo "$(date "+%F %T") host=$(hostname) cores=$CORES njobs=$NJOBS omp=$OMP_NUM_THREADS feature_mode=$FEATURE_MODE"
echo "h5ad_dir=$H5AD_DIR"
echo "out=$OUT"

# collapse-to-single-gene + pooled control (Negative/Non_target/Safe_target_*) are the defaults.
time apptainer exec "$SIF" python -u "$REPO/python/scripts/run_learn_dictionary.py" \
    --h5ad-dir "$H5AD_DIR" \
    "${FEAT_ARGS[@]}" \
    "${GROUP_ARGS[@]}" \
    --perturbation-column gene_target \
    --qc-column passes_qc \
    --n-jobs "$NJOBS" \
    --out "$OUT"
echo "$(date "+%F %T") DICT_EXIT=$? -> $OUT"
