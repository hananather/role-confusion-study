#!/bin/bash
cd "$(dirname "$0")"
PY=../.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
while [ ! -f runs/forgery5-tool_minus_cot-a8/supervisor.json ]; do sleep 20; done
for arm in "random_0 8 tool" "tool_minus_cot 8 all" "tool_minus_cot 16 tool" "tool_minus_cot 0 tool"; do
  set -- $arm
  out="runs/forgery5-$1-a$2"; [ "$3" = "all" ] && out="$out-all"
  [ -d "$out" ] && continue
  echo "=== $(date '+%H:%M:%S') start $out"
  $PY run_steer.py --manifest data/forgery-5.json --direction "$1" --alpha "$2" --mask "$3" --out "$out" --run-seconds 2700
  echo "=== $(date '+%H:%M:%S') exit $? $out"
done
echo "=== $(date '+%H:%M:%S') start runs/doubt-1"
$PY continue_doubt.py --points data/doubt-points.json --out runs/doubt-1 --seeds 1,2,3,4
echo "=== $(date '+%H:%M:%S') exit $? doubt-1"
echo "=== chain done $(date '+%H:%M:%S')"
