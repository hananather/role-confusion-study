#!/bin/bash
# v3: GPU fitter (fit_probes_mlx.py), all 8 role spaces, both splits, per pass; reference check and
# position null after each pass; keep layers 10, 12, 14. Usage: bash probes/run_full_probes_v3.sh [groups...]
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=runs/full-249; PY=.venv/bin/python; KEEP="10 12 14"
fit_and_check() {
  local g="$1"
  echo "=== $(date) fit layers $g (8 spaces, both splits) ==="
  $PY probes/fit_probes_mlx.py --run "$RUN" --layers "$g" --splits prompt,base 2>&1 | grep --line-buffered -v Warning
  echo "=== $(date) reference check layers $g ==="
  $PY probes/apply_reference_probes.py --run "$RUN" --out "$RUN/reference-check-L${g//,/-}.json" 2>&1 | grep -v "^\[ref\]" | tail -8
  echo "=== $(date) position null layers $g ==="
  for sp in suca sucat uat; do $PY probes/position_null.py --run "$RUN" --space $sp --layers "$g" --out "$RUN/position-null-$sp-L${g//,/-}.csv" 2>&1 | grep "^\[done\]\|^\[skip\]"; done
  for l in ${g//,/ }; do keep=0; for k in $KEEP; do [ "$l" = "$k" ] && keep=1; done; [ $keep = 0 ] && rm -f "$RUN/layer$(printf %02d "$l").npy"; done
  echo "=== $(date) pass $g done; disk: $(du -sh "$RUN" | cut -f1) ==="
}
# layers already on disk from pass 4
fit_and_check "0,1,20,21,22,23"
for g in "$@"; do
  echo "=== $(date) extract layers $g ==="
  $PY probes/extract_activations.py --out "$RUN" --layers "$g" --prompts "$RUN/prompts.parquet" 2>&1 | grep --line-buffered -v "it/s\]$"
  fit_and_check "$g"
done
echo "=== $(date) all done ==="
