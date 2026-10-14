#!/bin/bash
# Run check_dictionary_rds.R on every .rds in a per-subclass dictionary directory.
#
#   R_SIF=/path/to/rna-fingerprinting_seurat.sif bash check_all_dictionaries.sh <dict_dir>
#
# Prints one line per file (OK: <features> x <perturbations>, or the error) and a final tally;
# exits non-zero if any file fails. R_SIF must be an R container with base R (no extra packages).

set -uo pipefail
DIR=${1:?usage: check_all_dictionaries.sh <dict_dir>}
R_SIF=${R_SIF:?set R_SIF to an R container}
module load "${APPTAINER_MODULE:-apptainer}" >/dev/null 2>&1 || true
CHECK="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/check_dictionary_rds.R"

n_ok=0; n_fail=0
for f in "$DIR"/*.rds; do
  out=$(apptainer exec "$R_SIF" Rscript "$CHECK" "$f" 2>&1)
  if [ $? -eq 0 ]; then
    n_ok=$((n_ok + 1)); printf '%-45s %s\n' "$(basename "$f")" "$(grep -m1 '^OK' <<<"$out")"
  else
    n_fail=$((n_fail + 1)); printf '%-45s FAIL: %s\n' "$(basename "$f")" "$(grep -m1 -i 'error' <<<"$out")"
  fi
done
echo "passed: $n_ok  failed: $n_fail"
[ "$n_fail" -eq 0 ]
