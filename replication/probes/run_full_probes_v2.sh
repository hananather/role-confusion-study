#!/bin/bash
# Same as run_full_probes.sh but takes the layer groups as arguments and stops on the first failure.
# Usage: bash probes/run_full_probes_v2.sh "8,9,10,11,12,13" "14,15,16,17,18,19" "2,3,4,5,6,7"
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=runs/full-249
PY=.venv/bin/python
KEEP="10 12 14"
for g in "$@"; do
  echo "=== $(date) extract layers $g ==="
  $PY probes/extract_activations.py --out "$RUN" --layers "$g" --prompts "$RUN/prompts.parquet" 2>&1 | grep -v "it/s\]$"
  echo "=== $(date) fit layers $g (authors' split) ==="
  $PY probes/fit_probes.py --run "$RUN" --role-spaces suca,sucat,uat --layers "$g" --split-by prompt 2>&1 | grep -v "RuntimeWarning\|by_role = "
  mv "$RUN/probes.npz" "$RUN/probes-L${g//,/-}.npz"; mv "$RUN/probes.json" "$RUN/probes-L${g//,/-}.json"
  echo "=== $(date) fit layers $g (base split) ==="
  $PY probes/fit_probes.py --run "$RUN" --role-spaces suca,sucat,uat --layers "$g" --split-by base 2>&1 | grep -v "RuntimeWarning\|by_role = "
  mv "$RUN/probes-basesplit.npz" "$RUN/probes-basesplit-L${g//,/-}.npz"; mv "$RUN/probes-basesplit.json" "$RUN/probes-basesplit-L${g//,/-}.json"
  for l in ${g//,/ }; do
    keep=0; for k in $KEEP; do [ "$l" = "$k" ] && keep=1; done
    [ $keep = 0 ] && rm -f "$RUN/layer$(printf %02d "$l").npy"
  done
  echo "=== $(date) pass $g done; disk: $(du -sh "$RUN" | cut -f1) ==="
done
echo "=== $(date) all requested passes done ==="
