#!/usr/bin/env bash
# Pod environment for cloud/probes/train_probes_pod.py. Idempotent; safe to re-run.
#
# Recipe = the authors' setup_python.sh (prompt-injection-as-role-confusion, commit ec333c40) with the
# four fixes the reference H200 run needed (reference/role-confusion-extension/deliverables/REPORT.md s3):
#   RAPIDS 25.9.* does not exist -> cuML/cuDF 25.08 from pypi.nvidia.com; scikit-learn pinned 1.7.2
#   (1.9 breaks the cuML import); zstandard added (Dolma3 is zstd); pandas 2.3.3 (cuDF's pin).
# Target versions: python 3.12, torch 2.9.1+cu128, transformers 4.57.5, triton 3.5.1, kernels 0.11.5,
# cuml 25.08, datasets 5.0.1. The base image's own Python is not used: uv builds a 3.12 venv, as the
# authors' script does, so any runpod/pytorch image with a CUDA 12.8 driver works.
#
# Usage (on the pod):  bash setup_pod.sh [--check-model] [--attn auto|fa3|eager] [--skip-download]
# Env: WS (default /workspace), HF_TOKEN (optional; C4/Dolma/gpt-oss are public).
set -euo pipefail

WS="${WS:-/workspace}"
REPO="$WS/prompt-injection-as-role-confusion"
VENV="$WS/venv-probes"
HF_CACHE="$WS/hf"                       # utils/loader.py hardcodes cache_dir='/workspace/hf'
COMMIT="ec333c40fd43fe991e1ebf66765051b6d7e35784"
MODEL_ID="openai/gpt-oss-20b"
MODEL_REV="6cee5e81ee83917806bbde320786a8fb61efebee"   # snapshot the reference run used
FA_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECK_MODEL=0; ATTN="auto"; SKIP_DOWNLOAD=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-model) CHECK_MODEL=1; shift;;
    --attn) ATTN="$2"; shift 2;;
    --skip-download) SKIP_DOWNLOAD=1; shift;;
    *) echo "unknown argument: $1" >&2; exit 1;;
  esac
done

export UV_CACHE_DIR="$WS/.uv-cache" UV_PYTHON_INSTALL_DIR="$WS/.uv-python" UV_HTTP_TIMEOUT=120
export HF_HOME="$HF_CACHE/home" HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$WS" "$HF_CACHE" "$HF_HOME"
log() { printf '[setup %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

log "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader 2>/dev/null || echo 'nvidia-smi unavailable')"
log "disk: $(df -h "$WS" | tail -1)"

# ---------- 1. uv
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
case ":$PATH:" in *":$HOME/.local/bin:"*) :;; *) export PATH="$HOME/.local/bin:$PATH";; esac

# ---------- 2. venv, Python 3.12
if [[ -x "$VENV/bin/python" ]] && "$VENV/bin/python" -c 'import sys; exit(0 if sys.version_info[:2]==(3,12) else 1)'; then
  log "using existing venv $VENV"
else
  uv python install 3.12 >/dev/null 2>&1 || true
  uv venv "$VENV" --python 3.12 --seed
fi
PY="$VENV/bin/python"
PIP=(uv pip install --python "$PY")

# ---------- 3. packages (authors' pins, then the reference fixes)
"${PIP[@]}" --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1
"${PIP[@]}" \
  transformers==4.57.5 hf_transfer==0.1.9 accelerate==1.12.0 triton==3.5.1 \
  tiktoken==0.12.0 blobfile==3.1.0 kernels==0.11.5 compressed-tensors==0.13.0 \
  "pandas==2.3.3" pyyaml tqdm termcolor python-dotenv \
  "datasets==5.0.1" zstandard "scikit-learn==1.7.2" pyarrow huggingface_hub
# flash-attn 2 wheel: installed by the authors' script and the reference; not on the gpt-oss path
# (the loader uses kernels-community/vllm-flash-attn3 through `kernels`). Best effort.
"${PIP[@]}" "$FA_URL" || log "flash-attn 2.8.3 wheel skipped (not needed for gpt-oss)"
# RAPIDS: the authors ask for 25.9.*, which does not exist; 25.08 matches their neighbouring pins.
"${PIP[@]}" libucx-cu12==1.18.1 ucx-py-cu12==0.45.0
"${PIP[@]}" --extra-index-url https://pypi.nvidia.com "cudf-cu12==25.8.*" "cuml-cu12==25.8.*" "cupy-cuda12x==13.*"
# cuML 25.08 needs scikit-learn < 1.9; re-pin in case the resolver moved it.
"${PIP[@]}" "scikit-learn==1.7.2" "pandas==2.3.3"

# ---------- 4. authors' repo at the pinned commit
if [[ ! -d "$REPO/.git" ]]; then
  git clone https://github.com/role-confusion/prompt-injection-as-role-confusion.git "$REPO"
fi
git -C "$REPO" fetch --quiet origin || true
git -C "$REPO" checkout --quiet "$COMMIT"
[[ "$(git -C "$REPO" rev-parse HEAD)" == "$COMMIT" ]] || { echo "repo is not at $COMMIT" >&2; exit 1; }
mkdir -p "$REPO/experiments/role-analysis/outputs/probes" "$REPO/experiments/role-analysis/outputs/probe-training"
SITE_DIR="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' "$REPO" > "$SITE_DIR/add_path_analysis.pth"   # `utils.*` importable, as the authors' setup does

# ---------- 5. model weights into the loader's hardcoded cache dir
if [[ $SKIP_DOWNLOAD -eq 0 ]]; then
  "$PY" - <<PYEOF
from huggingface_hub import snapshot_download
import os
p = snapshot_download("$MODEL_ID", revision="$MODEL_REV", cache_dir="$HF_CACHE", token=os.environ.get("HF_TOKEN"))
print("snapshot:", p)
PYEOF
  # loader.py resolves 'main' at load time; warn if main has moved past the reference snapshot.
  "$PY" - <<PYEOF || true
from huggingface_hub import HfApi
info = HfApi().model_info("$MODEL_ID")
print("hub main sha:", info.sha, "(reference $MODEL_REV)", "" if info.sha == "$MODEL_REV" else "!! main moved; metadata.json will record the loaded snapshot")
PYEOF
fi

# ---------- 6. checks
log "versions:"
"$PY" - <<'PYEOF'
from importlib.metadata import version, PackageNotFoundError
import traceback
DIST = {"cuml": "cuml-cu12", "cudf": "cudf-cu12", "cupy": "cupy-cuda12x", "scikit-learn": "scikit-learn"}
for n in ["torch","transformers","triton","kernels","flash_attn","cuml","cudf","cupy","scikit-learn","numpy","pandas","datasets","zstandard","accelerate"]:
    try: print(f"  {n:14s} {version(DIST.get(n, n))}")
    except PackageNotFoundError: print(f"  {n:14s} MISSING")
for mod in ("cuml", "cupy", "cudf"):
    try:
        __import__(mod); print(f"  import {mod}: ok")
    except Exception as e:
        print(f"  import {mod}: FAILED {type(e).__name__}: {str(e)[:200]}")
import torch; print("  cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0), "sm", ".".join(map(str, torch.cuda.get_device_capability(0))))
PYEOF
"$PY" "$HERE/train_probes_pod.py" --check-env
CAP="$("$PY" -c 'import torch; print(torch.cuda.get_device_capability(0)[0])')"
if [[ "$CAP" == "9" ]]; then
  log "Hopper (sm 9.x): the authors' attention kernel kernels-community/vllm-flash-attn3 will be used"
else
  log "NOT Hopper (sm ${CAP}.x): FA3 is unavailable; train_probes_pod.py --attn auto falls back to eager (a logged divergence)"
fi
if [[ $CHECK_MODEL -eq 1 ]]; then
  # Loads the model through the authors' loader: prints 'Expert precision: FloatType(bitwidth_exponent=2, bitwidth_mantissa=1, ...)'
  # for MXFP4 and the attention implementation, then verifies the custom forward pass equals HF logits.
  "$PY" "$HERE/train_probes_pod.py" --check-model --attn "$ATTN" --repo "$REPO" --results "$WS/results/check-model" --acts "$WS/acts/check-model" --hours 0.25
fi
log "setup done. venv=$VENV repo=$REPO hf=$HF_CACHE"
