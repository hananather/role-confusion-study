#!/bin/bash
# Sequential arms on the five forgery pages. One model at a time. Any cutoff leaves complete arms.
cd "$(dirname "$0")"
PY=../.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
for arm in "tool_minus_cot 8" "random_0 8" "tool_minus_cot 4" "tool_minus_cot 16" "tool_minus_cot 2"; do
  set -- $arm
  out="runs/forgery5-$1-a$2"
  echo "=== $(date '+%H:%M:%S') start $out" 
  $PY run_steer.py --manifest data/forgery-5.json --direction "$1" --alpha "$2" --out "$out" --run-seconds 2700
  echo "=== $(date '+%H:%M:%S') exit $? $out"
done
echo "=== chain done $(date '+%H:%M:%S')"
