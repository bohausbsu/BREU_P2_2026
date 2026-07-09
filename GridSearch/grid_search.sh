#!/usr/bin bash
#
# Usage:
#   ./grid_search.sh --dataset data.csv --target-col Placement \
#       [--flag-fracs "0.1 0.2 0.3"] [--at-rs "0.01 0.02 0.05"] \
#       [--outdir grid_search_results] [--experiment-py /path/to/experiment.py] \
#       [-- <extra args forwarded verbatim to experiment.py>]
#
# Output:
#   <outdir>/grid_search_summary.csv   - one row per (flag_frac, at_r) combo
#   <outdir>/result_ff<F>_r<R>.csv     - raw experiment.py output per combo
#   <outdir>/log_ff<F>_r<R>.txt        - full stdout/stderr per run
#
set -euo pipefail

# ---- defaults ---------------------------------------------------------------
DATASET="college_student_placement_dataset.csv"
TARGET_COL="Placement"
# FLAG_FRACS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7)
# AT_RS=(0.01 0.02 0.05 0.1 0.15 0.2 0.25)
FLAG_FRACS=(0.1)
AT_RS=(0.01)
OUTDIR="grid_search_results"
EXPERIMENT_PY="$(cd "$(dirname "$0")" && pwd)/experiment.py"
EXTRA_ARGS=()

# ---- arg parsing --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"; shift 2 ;;
    --target-col)
      TARGET_COL="$2"; shift 2 ;;
    --flag-fracs)
      read -r -a FLAG_FRACS <<< "$2"; shift 2 ;;
    --at-rs)
      read -r -a AT_RS <<< "$2"; shift 2 ;;
    --outdir)
      OUTDIR="$2"; shift 2 ;;
    --experiment-py)
      EXPERIMENT_PY="$2"; shift 2 ;;
    --)
      shift; EXTRA_ARGS=("$@"); break ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^#//'; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$DATASET" || -z "$TARGET_COL" ]]; then
  echo "Usage: $0 --dataset <path> --target-col <name> [--flag-fracs \"...\"] [--at-rs \"...\"] [--outdir DIR] [-- extra experiment.py args]" >&2
  exit 1
fi

if [[ ! -f "$EXPERIMENT_PY" ]]; then
  echo "Could not find experiment.py at: $EXPERIMENT_PY" >&2
  exit 1
fi

mkdir -p "$OUTDIR"
SUMMARY_CSV="$OUTDIR/grid_search_summary.csv"
echo "flag_frac,at_r,tp,fp,tn,fn,precision,recall,f1" > "$SUMMARY_CSV"

BEST_F1="-1"
BEST_COMBO=""

echo "Grid search: ${#FLAG_FRACS[@]} flag-frac value(s) x ${#AT_RS[@]} at-r value(s) = $((${#FLAG_FRACS[@]} * ${#AT_RS[@]})) runs"
echo "Results directory: $OUTDIR"
echo

for FLAG_FRAC in "${FLAG_FRACS[@]}"; do
  for AT_R in "${AT_RS[@]}"; do
    TAG="ff${FLAG_FRAC}_r${AT_R}"
    RUN_CSV="$OUTDIR/result_${TAG}.csv"
    RUN_PNG="$OUTDIR/result_${TAG}.png"
    LOG="$OUTDIR/log_${TAG}.txt"

    echo "=== flag-frac=${FLAG_FRAC}  at-r=${AT_R} ==="

    if python "$EXPERIMENT_PY" \
        --dataset "$DATASET" \
        --target-col "$TARGET_COL" \
        --flag-frac "$FLAG_FRAC" \
        --at-r "$AT_R" \
        --out-csv "$RUN_CSV" \
        --out "$RUN_PNG" \
        "${EXTRA_ARGS[@]}" \
        > "$LOG" 2>&1; then
      :
    else
      echo "  run FAILED (see $LOG), skipping"
      continue
    fi

    if [[ ! -f "$RUN_CSV" ]]; then
      echo "  no results csv produced, skipping"
      continue
    fi

    # experiment.py writes a header line + exactly one data row
    ROW=$(sed -n '2p' "$RUN_CSV")
    if [[ -z "$ROW" ]]; then
      echo "  empty results row, skipping"
      continue
    fi

    IFS=',' read -r TP FP TN FN PRECISION RECALL F1 <<< "$ROW"
    echo "${FLAG_FRAC},${AT_R},${TP},${FP},${TN},${FN},${PRECISION},${RECALL},${F1}" >> "$SUMMARY_CSV"
    echo "  Precision=${PRECISION}  Recall=${RECALL}  F1=${F1}"

    IS_BEST=$(awk -v f1="$F1" -v best="$BEST_F1" 'BEGIN { print (f1 > best) ? "1" : "0" }')
    if [[ "$IS_BEST" == "1" ]]; then
      BEST_F1="$F1"
      BEST_COMBO="flag-frac=${FLAG_FRAC} at-r=${AT_R}"
    fi
  done
done

echo
echo "==================================================================="
echo "Grid search complete. Summary: $SUMMARY_CSV"
if [[ -n "$BEST_COMBO" ]]; then
  echo "Best combination: $BEST_COMBO  (F1=${BEST_F1})"
else
  echo "No successful runs — check the log files in $OUTDIR"
fi
