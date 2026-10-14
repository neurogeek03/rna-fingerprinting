#!/bin/bash
# Run check_dictionary_rds.R on every .rds in a per-subclass dictionary directory and summarize.
#
#   R_SIF=/path/to/rna-fingerprinting_seurat.sif bash check_dictionary_dir.sh <dict_dir>
#
# Prints one line per file (OK: <features> x <perturbations>, or the error) and a pass/fail count.
# Exit status is non-zero if any file fails.

set -uo pipefail
DIR=${1:?usage: check_dictionary_dir.sh <dict_dir>}
R_SIF=${R_SIF:?set R_SIF to the R/Seurat container with the RNAFingerprinting package}
CHECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check_dictionary_rds.R"
module load "${APPTAINER_MODULE:-apptainer}" >/dev/null 2>&1 || true

pass=0; fail=0
for f in "$DIR"/*.rds; do
  out=$(apptainer exec "$R_SIF" Rscript "$CHECK" "$f" 2>&1)
  if [ $? -eq 0 ]; then
    pass=$((pass + 1)); printf "%-45s %s\n" "$(basename "$f")" "$(grep -m1 '^OK' <<<"$out")"
  else
    fail=$((fail + 1)); printf "%-45s FAIL: %s\n" "$(basename "$f")" "$(tail -1 <<<"$out")"
  fi
done
echo "pass=$pass fail=$fail"
[ "$fail" -eq 0 ]
