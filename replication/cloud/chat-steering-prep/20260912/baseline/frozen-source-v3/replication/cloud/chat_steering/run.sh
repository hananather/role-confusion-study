#!/usr/bin/env bash
# I run my frozen toy-lab batch once; the Mac controller owns provider operations.
set -euo pipefail
export HF_HOME=/workspace/hf/home
export HF_HUB_CACHE=/workspace/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec python3 -u /workspace/replication/cloud/chat_steering/control.py pod-run "$@"
