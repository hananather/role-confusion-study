#!/usr/bin/env bash
# I recreate the September 11 CUDA environment on one approved fresh H100 pod.
# Source recipe: ../probes/setup_pod.sh, verified by the 20260911T030050Z setup receipt.
# Base image: runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04.
# I need 100 GB of pod disk; I fail below 45 GiB free. No cuML, datasets, FA2,
# authors' checkout, model inference, provider credentials, or model API calls.
# I download public model/kernel assets only after the approved pod exists.
# Usage: bash setup_fresh.sh --results /workspace/results/chat-steering/RUN
set -euo pipefail

CHAT_RESULTS=""
CHAT_INSIDE_CAP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --results) CHAT_RESULTS="$2"; shift 2;;
    --inside-cap) CHAT_INSIDE_CAP=1; shift;;
    --help) printf '%s\n' 'Usage: bash setup_fresh.sh --results /workspace/results/chat-steering/RUN'; exit 0;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2;;
  esac
done
[[ -n "$CHAT_RESULTS" && "$CHAT_RESULTS" == /workspace/results/chat-steering/* ]] || {
  printf '%s\n' 'I require the absolute results directory for this bounded chat-steering run.' >&2; exit 2;
}
if [[ "$CHAT_INSIDE_CAP" -eq 0 ]]; then
  command -v timeout >/dev/null
  exec timeout --signal=TERM --kill-after=30s 1800 bash "$0" --inside-cap --results "$CHAT_RESULTS"
fi

CHAT_WS=/workspace
CHAT_VENV=/workspace/venv-probes
CHAT_MODEL_ID=openai/gpt-oss-20b
CHAT_MODEL_REV=6cee5e81ee83917806bbde320786a8fb61efebee
export CHAT_RESULTS CHAT_MODEL_ID CHAT_MODEL_REV
export HF_HOME=/workspace/hf/home HF_HUB_CACHE=/workspace/hf
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DOWNLOAD_TIMEOUT=120
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export UV_CACHE_DIR=/workspace/.uv-cache UV_PYTHON_INSTALL_DIR=/workspace/.uv-python UV_HTTP_TIMEOUT=120
export TRITON_CACHE_DIR=/workspace/.triton
mkdir -p "$CHAT_RESULTS" "$HF_HOME" "$TRITON_CACHE_DIR"
CHAT_HEARTBEAT_PID=""
finish_setup() {
  CHAT_CODE="$1"
  trap - EXIT
  if [[ -n "$CHAT_HEARTBEAT_PID" ]]; then
    kill "$CHAT_HEARTBEAT_PID" 2>/dev/null || true
    wait "$CHAT_HEARTBEAT_PID" 2>/dev/null || true
  fi
  python3 - "$CHAT_RESULTS" "$CHAT_CODE" <<'PY'
import json, sys, time
from pathlib import Path
out = Path(sys.argv[1]); code = int(sys.argv[2])
(out / "setup-exit.json").write_text(json.dumps({"returncode": code, "unix_time": time.time(), "status": "complete" if code == 0 else "failed"}) + "\n")
PY
  exit "$CHAT_CODE"
}
trap 'finish_setup "$?"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
stage() { printf '%s\n' "$1" > "$CHAT_RESULTS/setup-stage.txt"; printf '[fresh setup] %s\n' "$1"; }
stage hardware_and_disk
python3 - "$CHAT_RESULTS" "$$" <<'PY' &
import json, os, sys, time
from pathlib import Path
out = Path(sys.argv[1]); parent = int(sys.argv[2])
while True:
    try: os.kill(parent, 0)
    except ProcessLookupError: break
    now = time.time()
    row = {"stage": "fresh_setup", "setup_stage": (out / "setup-stage.txt").read_text().strip(),
           "progress_unix": now, "unix": now, "pid": parent, "rows": 0}
    temporary = out / "heartbeat.setup.tmp"
    temporary.write_text(json.dumps(row) + "\n")
    temporary.replace(out / "heartbeat.json")
    time.sleep(30)
PY
CHAT_HEARTBEAT_PID=$!
python3 - <<'PY'
import shutil, subprocess
free = shutil.disk_usage('/workspace').free
if free < 45 * 2**30:
    raise SystemExit(f'I need at least 45 GiB free; only {free / 2**30:.1f} GiB is available. Allocate 100 GB disk.')
gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,compute_cap', '--format=csv,noheader'], text=True).strip()
print(gpu, flush=True)
if 'H100' not in gpu or len(gpu.splitlines()) != 1:
    raise SystemExit('I require exactly one H100 for this frozen setup.')
PY

# The runtime's default Hugging Face lookup and the authors' explicit cache path
# must resolve to the same cached bytes without downloading the weights twice.
if [[ ! -e "$HF_HOME/hub" && ! -L "$HF_HOME/hub" ]]; then
  ln -s "$HF_HUB_CACHE" "$HF_HOME/hub"
elif [[ "$(readlink -f "$HF_HOME/hub")" != "$HF_HUB_CACHE" ]]; then
  printf '%s\n' 'Existing HF_HOME/hub has a different target; I refuse to silently move its contents.' >&2
  exit 3
fi

stage python_environment
python3 -m pip install --disable-pip-version-check --no-cache-dir uv==0.12.13
python3 -m uv python install 3.12.14
if [[ -e "$CHAT_VENV" ]]; then
  [[ -x "$CHAT_VENV/bin/python" ]] || { printf '%s\n' 'Existing venv is incomplete.' >&2; exit 3; }
  "$CHAT_VENV/bin/python" -c 'import sys; assert sys.version_info[:3] == (3, 12, 14), sys.version'
else
  python3 -m uv venv "$CHAT_VENV" --python 3.12.14 --seed
fi
CHAT_PY="$CHAT_VENV/bin/python"
stage cuda_packages
python3 -m uv pip install --python "$CHAT_PY" --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1
python3 -m uv pip install --python "$CHAT_PY" \
  transformers==4.57.5 triton==3.5.1 kernels==0.11.5 accelerate==1.12.0 \
  numpy==2.2.6 pandas==2.3.3 pyarrow==19.0.1 pyyaml==6.0.3 safetensors==0.8.0 \
  huggingface_hub==0.36.2 hf_transfer==0.1.9 tokenizers==0.22.2 \
  tiktoken==0.12.0 blobfile==3.1.0
"$CHAT_PY" - <<'PY'
import importlib.metadata as md
import torch
pins = {'torch':'2.9.1+cu128','transformers':'4.57.5','triton':'3.5.1','kernels':'0.11.5',
        'accelerate':'1.12.0','numpy':'2.2.6','pandas':'2.3.3','pyarrow':'19.0.1'}
for name, expected in pins.items():
    actual = md.version(name)
    if actual != expected: raise SystemExit(f'{name}: expected {expected}, got {actual}')
    print(f'{name}=={actual}', flush=True)
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
assert torch.cuda.get_device_capability(0)[0] == 9
PY

stage public_model_download
"$CHAT_PY" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
path = Path(snapshot_download(os.environ['CHAT_MODEL_ID'], revision=os.environ['CHAT_MODEL_REV'],
                              cache_dir=os.environ['HF_HUB_CACHE'], token=False,
                              allow_patterns=['*.json','*.safetensors','*.jinja','*.txt']))
assert path.name == os.environ['CHAT_MODEL_REV']
assert (path / 'config.json').is_file() and (path / 'tokenizer.json').is_file()
assert list(path.glob('*.safetensors')), 'No model weight shards were downloaded'
print('Pinned model snapshot:', path, flush=True)
PY

stage public_kernel_download_and_import
# Verified against the pinned transformers v4.57.5 source: MXFP4 requests
# kernels-community/triton_kernels and FA3 requests vllm-flash-attn3.
# I record the downloaded kernel snapshot revisions because the old recipe used
# their default revisions. These imports load kernels, not language-model weights.
"$CHAT_PY" - <<'PY'
import importlib.metadata as md
import json, os, re, sys, time
from pathlib import Path
from kernels import get_kernel
fa3 = get_kernel('kernels-community/vllm-flash-attn3')
assert callable(getattr(fa3, 'flash_attn_varlen_func', None))
triton = get_kernel('kernels-community/triton_kernels')
assert hasattr(triton.tensor, 'FP4') and hasattr(triton.matmul_ogs, 'PrecisionConfig')
paths = sorted({str(Path(module.__file__).resolve()) for module in tuple(sys.modules.values())
                if getattr(module, '__file__', None) and '/snapshots/' in str(module.__file__)})
report = {'unix_time': time.time(), 'model_id': os.environ['CHAT_MODEL_ID'],
          'model_revision': os.environ['CHAT_MODEL_REV'], 'hf_home': os.environ['HF_HOME'],
          'hf_hub_cache': os.environ['HF_HUB_CACHE'], 'kernel_module_paths': paths,
          'kernel_snapshot_revisions': sorted({m.group(1) for p in paths if (m := re.search(r'/snapshots/([0-9a-f]{40})/', p))}),
          'versions': {name: md.version(name) for name in ('torch','transformers','triton','kernels','numpy','pandas','pyarrow','accelerate','huggingface_hub','tokenizers','safetensors')},
          'model_weights_loaded': False, 'attention': 'kernels-community/vllm-flash-attn3'}
Path(os.environ['CHAT_RESULTS'], 'setup-environment.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2), flush=True)
PY

stage offline_cache_gate
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$CHAT_PY" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoTokenizer
from kernels import get_kernel
model, revision = os.environ['CHAT_MODEL_ID'], os.environ['CHAT_MODEL_REV']
path = Path(snapshot_download(model, revision=revision, local_files_only=True))
assert path.name == revision
config = AutoConfig.from_pretrained(model, revision=revision, local_files_only=True, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, local_files_only=True, trust_remote_code=False,
                                        add_eos_token=False, add_bos_token=False, padding_side='left')
assert config.model_type == 'gpt_oss' and tokenizer.is_fast and tokenizer.pad_token_id is not None
get_kernel('kernels-community/vllm-flash-attn3')
get_kernel('kernels-community/triton_kernels')
print('Offline model configuration, tokenizer, and both required kernel imports passed; no model weights loaded.', flush=True)
PY
stage complete
printf '%s\n' 'Fresh setup complete. The separately frozen runtime owns the five-prompt model gate.'
