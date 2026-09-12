#!/bin/bash
# Waits for the main chain, then runs the identity control (alpha 0) with the same code and seeds.
cd "$(dirname "$0")"
while ! grep -q "chain done" runs/chain.log 2>/dev/null; do sleep 30; done
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
echo "=== $(date '+%H:%M:%S') start runs/forgery5-tool_minus_cot-a0"
../.venv/bin/python run_steer.py --manifest data/forgery-5.json --direction tool_minus_cot --alpha 0 --out runs/forgery5-tool_minus_cot-a0 --run-seconds 2700
echo "=== $(date '+%H:%M:%S') exit $? a0"; echo "=== after done"
