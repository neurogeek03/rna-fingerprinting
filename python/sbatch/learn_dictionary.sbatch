#!/bin/bash
#SBATCH --job-name=brisc_dict
#SBATCH --account=rrg-shreejoy           # Trillium; override per cluster with `sbatch --account=...`
#SBATCH --time=06:00:00                  # estimate; 14-file load/normalize dominates
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192              # Trillium = whole node (192 cores, ~745 GiB). No --mem on
                                          # Trillium; on other clusters add --mem=750G (full atlas
                                          # ~196GB/copy, ~392GB transient peak).
#SBATCH --output=logs/%x_%j.out          # relative to the submit dir: `mkdir -p logs` first
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=mariaeleni.fafouti@mail.utoronto.ca
#SBATCH --mail-type=BEGIN,END,FAIL

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

# Feature selection: "expression" (fast), "de" (find_markers), or "file" (use FEATURES_FILE).
FEATURE_MODE=${FEATURE_MODE:-expression}
FEATURES_FILE=${FEATURES_FILE:-$REPO/features.txt}
MEAN_EXPRESSION=${MEAN_EXPRESSION:-0.05}

NJOBS=${NJOBS:-$(( SLURM_CPUS_PER_TASK / 2 ))}
# joblib workers = cores/2 (per-perturbation loop; 192 workers measured >10x slower, see
# brisc_adaptation/docs/thread_scaling_paradox.md). BLAS = all cores for the control PCA; fit()
# gives each worker cores/NJOBS BLAS threads inside the loop, so the two never oversubscribe.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Log timestamps in the host timezone (the container defaults to UTC).
export TZ=${TZ:-$(readlink -f /etc/localtime 2>/dev/null | sed -n "s|.*/zoneinfo/||p")}
[ -n "$TZ" ] || unset TZ

case "$FEATURE_MODE" in
  file) FEAT_ARGS=(--features-file "$FEATURES_FILE") ;;
  *)    FEAT_ARGS=(--choose-features "$FEATURE_MODE" --mean-expression "$MEAN_EXPRESSION") ;;
esac

echo "$(date "+%F %T") node=$(hostname) cpus=$SLURM_CPUS_PER_TASK njobs=$NJOBS feature_mode=$FEATURE_MODE"
# collapse-to-single-gene + pooled control (Negative/Non_target/Safe_target_*) are the defaults.
apptainer exec "$SIF" python -u "$REPO/python/scripts/run_learn_dictionary.py" \
    --h5ad-dir "$H5AD_DIR" \
    "${FEAT_ARGS[@]}" \
    "${GROUP_ARGS[@]}" \
    --perturbation-column gene_target \
    --qc-column passes_qc \
    --n-jobs "$NJOBS" \
    --out "$OUT"
echo "$(date "+%F %T") DICT_EXIT=$?"
