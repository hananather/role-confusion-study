#!/bin/bash
cd "$(dirname "$0")"
PY=../.venv/bin/python
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
while ! grep -q "chain4 done" runs/chain.log 2>/dev/null; do sleep 30; done
echo "=== $(date '+%H:%M:%S') chain5 begin: attribution readout"
$PY attribution_readout.py --out runs/attribution-1
echo "=== $(date '+%H:%M:%S') exit $? attribution-1"
echo "=== $(date '+%H:%M:%S') start patch-1"
$PY patch_paragraph.py --out runs/patch-1
echo "=== $(date '+%H:%M:%S') exit $? patch-1"
echo "=== chain5 done $(date '+%H:%M:%S')"
