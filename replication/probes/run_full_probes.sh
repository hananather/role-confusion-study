#!/bin/bash
# Paper-scale probe run on the MacBook, in passes of six layers because 24 layers of fp16
# activations (176 GB) do not fit on disk. Each pass: extract the full 249-text prompt set
# for six layers (~65 min), fit probes (three role spaces, both splits), then delete the
# layer files except layers 10, 12, 14, which stay for Appendix E and later role spaces.
# Pass order puts 8-13 first so Appendix E (layer 12) unblocks early.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN=runs/full-249
PY=.venv/bin/python
mkdir -p "$RUN"
GROUPS_LIST=("8,9,10,11,12,13" "14,15,16,17,18,19" "2,3,4,5,6,7" "0,1,20,21,22,23")
KEEP="10 12 14"
for g in "${GROUPS_LIST[@]}"; do
  echo "=== $(date) extract layers $g ==="
  if [ -f "$RUN/prompts.parquet" ]; then
    $PY probes/extract_activations.py --out "$RUN" --layers "$g" --prompts "$RUN/prompts.parquet" 2>&1 | grep -v "it/s\]$"
  else
    $PY probes/extract_activations.py --n-base 250 --out "$RUN" --layers "$g" 2>&1 | grep -v "it/s\]$"
  fi
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
  echo "=== $(date) pass done; disk: $(du -sh "$RUN" | cut -f1) ==="
done
echo "=== $(date) all passes done ==="
