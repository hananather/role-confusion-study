#!/usr/bin/env bash
# One-shot cloud probe job: train the authors' gpt-oss-20b role probes on all 24 layers with two
# splits, sync the probes back, terminate the pod. Two entry points in one file so the pod runs the
# exact bytes the Mac reviewed:
#
#   Mac:  RUNPOD_API_KEY=... cloud/probes/run_probes_job.sh launch \
#           --gpu "NVIDIA H100 PCIe" --cloud SECURE --rate 2.89 --hours 3 --cap 12 [--balance 12] \
#           [--attn auto|fa3|eager] [--pilot 10] [--pilot-only] [--network-volume <id>] [--volume-disk 250] \
#           [--image runpod/pytorch:...] [--ssh-key ~/.ssh/id_ed25519] [--allow-existing-pods] [--dry-run]
#
#   Pod:  bash run_probes_job.sh pod --run-id <id> --hours 3 [--attn auto] [--pilot 10] [--pilot-only] [--skip-setup]
#         (what `launch` starts over ssh; can also be run by hand on any pod that has this directory)
#
# Frozen pod-side sequence: backstop timer -> key check -> setup_pod.sh -> pilot (10 base texts, all
# layers, both splits; abort on failure) -> full run under the remaining --hours -> sha256 -> DONE.
# Mac-side: preflight (key, no stray pods, balance, cap) -> cost-log row -> create pod -> wait ->
# rsync this directory -> start the pod sequence detached under `timeout` -> cloud/watchdog.py
# (rsync the outbox every 90 s, heartbeat, deadline, DONE -> final rsync -> DELETE, cost-log close).
# GPU-PLAYBOOK.md safeguards 2, 3, 4, 5, 8, 10. --dry-run prints every REST, ssh and rsync call.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"; shift || true
[[ "$MODE" == "launch" || "$MODE" == "pod" ]] || { sed -n 2,22p "$0"; exit 1; }

# =============================================================================== pod side
if [[ "$MODE" == "pod" ]]; then
  WS="${WS:-/workspace}"; RUN_ID=""; HOURS=""; ATTN="auto"; PILOT=10; PILOT_ONLY=0; SKIP_SETUP=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --run-id) RUN_ID="$2"; shift 2;;
      --hours) HOURS="$2"; shift 2;;
      --attn) ATTN="$2"; shift 2;;
      --pilot) PILOT="$2"; shift 2;;
      --pilot-only) PILOT_ONLY=1; shift;;
      --skip-setup) SKIP_SETUP=1; shift;;
      *) echo "unknown pod argument: $1" >&2; exit 1;;
    esac
  done
  [[ -n "$RUN_ID" && -n "$HOURS" ]] || { echo "pod mode needs --run-id and --hours" >&2; exit 1; }
  R="$WS/results/$RUN_ID"; ACTS="$WS/acts/$RUN_ID"; PY="$WS/venv-probes/bin/python"
  mkdir -p "$R" "$R/pilot" "$ACTS"
  T0=$(date +%s); date -u +%FT%TZ > "$R/STARTED"
  exec > >(tee -a "$R/job.log") 2>&1
  echo "[pod] run $RUN_ID hours $HOURS attn $ATTN pilot $PILOT pilot_only $PILOT_ONLY"

  # 1. backstop: DELETE this pod at planned hours + 15 min no matter what (COMPUTE-PLAN 4.4 step 1)
  BACKSTOP_S=$(awk -v h="$HOURS" 'BEGIN{printf "%d", h*3600+900}')
  if [[ -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
    nohup sh -c "sleep $BACKSTOP_S; curl -s -X DELETE https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID -H 'Authorization: Bearer $RUNPOD_API_KEY'" > "$R/backstop.log" 2>&1 &
    # 2. key check: this key must see this pod, or self-delete cannot work (the Mac watchdog still can)
    curl -sf "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" > "$R/self.json" \
      || echo "pod key GET failed; self-delete will not work" | tee "$R/KEYFAIL"
  else
    echo "[pod] RUNPOD_POD_ID/RUNPOD_API_KEY not in env: no in-pod backstop (Mac watchdog only)"
  fi

  # 3. shell heartbeat during setup (the Python job writes the same file once it runs)
  hb() { printf '{"time_utc":"%s","phase":"%s"}\n' "$(date -u +%FT%TZ)" "$1" > "$R/heartbeat.tmp" && mv "$R/heartbeat.tmp" "$R/heartbeat"; }
  ( while true; do hb "setup"; sleep 30; done ) & HB_PID=$!
  finish() { local rc=$1; kill "$HB_PID" 2>/dev/null || true; (cd "$R" && find . -type f ! -name 'sha256.txt' ! -name EXIT ! -name DONE | sort | xargs sha256sum > sha256.txt) || true
             echo "$rc" > "$R/EXIT"; [[ $rc -eq 0 ]] && date -u +%FT%TZ > "$R/DONE"; echo "[pod] exit $rc after $(( ($(date +%s)-T0)/60 )) min"; exit "$rc"; }

  # 4. environment (idempotent; ~8 min cold, seconds warm)
  if [[ $SKIP_SETUP -eq 0 ]]; then
    bash "$HERE/setup_pod.sh" --attn "$ATTN" > "$R/setup.log" 2>&1 || { tail -50 "$R/setup.log"; echo "setup failed" > "$R/FAILED"; finish 9; }
  fi
  [[ -x "$PY" ]] || { echo "venv missing at $PY" > "$R/FAILED"; finish 9; }
  kill "$HB_PID" 2>/dev/null || true
  export HF_HOME="$WS/hf/home" HF_HUB_ENABLE_HF_TRANSFER=1 TOKENIZERS_PARALLELISM=false TRITON_CACHE_DIR="$WS/.triton"
  sha256sum "$HERE/train_probes_pod.py" "$HERE/setup_pod.sh" "$HERE/run_probes_job.sh" > "$R/source.sha256"
  cd "$WS/prompt-injection-as-role-confusion"

  # 5. pilot gate: N base texts, all layers, both splits; prints the accuracy table; aborts the job on failure
  remaining_h() { awk -v h="$HOURS" -v e="$(( $(date +%s) - T0 ))" -v m="$1" 'BEGIN{r=h-e/3600-m/60; if (r<0.05) r=0.05; printf "%.2f", r}'; }
  if [[ "$PILOT" -gt 0 ]]; then
    PH=$(remaining_h 5); [[ $(awk -v p="$PH" 'BEGIN{print (p>0.75)}') -eq 1 ]] && PH=0.75
    echo "[pod] pilot: $PILOT base texts, budget $PH h"
    timeout --signal=TERM --kill-after=120 "$(awk -v h="$PH" 'BEGIN{printf "%d", h*3600+300}')s" \
      "$PY" "$HERE/train_probes_pod.py" --pilot "$PILOT" --layers all --splits prompt,base --attn "$ATTN" \
        --results "$R/pilot" --acts "$ACTS-pilot" --heartbeat-file "$R/heartbeat" --hours "$PH" || { echo "pilot failed rc=$?" | tee "$R/FAILED"; finish 10; }
    echo "[pod] pilot accuracy by layer (prompt split):"; cat "$R/pilot/val_acc_by_layer.csv" || true
    echo "[pod] pilot projection:"; cat "$R/pilot/pilot.json" || true
    "$PY" - "$R/pilot/metadata.json" <<'PYEOF' || { echo "pilot gate failed" | tee "$R/FAILED"; finish 11; }
import json, sys
m = json.load(open(sys.argv[1]))
problems = []
if not m.get("custom_forward_verified"): problems.append("custom forward not verified")
if not m.get("role_counts_equal"): problems.append(f"role counts unequal: {m.get('role_counts')}")
if m.get("n_probes") != m.get("n_probes_planned"): problems.append(f"{m.get('n_probes')} of {m.get('n_probes_planned')} probes")
if "partial" in m: problems.append("pilot hit its deadline")
dt = m.get("expert_dtype", "")
if any(t in dt for t in ("bfloat16", "float16", "float32")): problems.append(f"experts not MXFP4: {dt}")
print("pilot gate:", "PASS" if not problems else problems, "| attn", m.get("attn_implementation"), "| experts", dt)
sys.exit(1 if problems else 0)
PYEOF
    [[ $PILOT_ONLY -eq 1 ]] && { echo "[pod] --pilot-only: stopping after the pilot"; finish 0; }
  fi

  # 6. full run under the remaining budget (the Python --hours alarm arms after the model loads)
  FH=$(remaining_h 10)
  echo "[pod] full run: budget $FH h"
  timeout --signal=TERM --kill-after=180 "$(awk -v h="$FH" 'BEGIN{printf "%d", h*3600+600}')s" \
    "$PY" "$HERE/train_probes_pod.py" --layers all --splits prompt,base --attn "$ATTN" \
      --results "$R" --acts "$ACTS" --heartbeat-file "$R/heartbeat" --hours "$FH" || { rc=$?; echo "full run rc=$rc" | tee -a "$R/FAILED"; finish "$rc"; }
  rm -rf "$ACTS-pilot"
  echo "[pod] outbox:"; ls -la "$R"; du -sh "$ACTS" || true
  finish 0
fi

# =============================================================================== Mac side
CLOUD="$(cd "$HERE/.." && pwd)"; PY=python3
GPU=(); CLOUD_TYPE="SECURE"; IMAGE="runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04"
RATE=""; HOURS=""; CAP=""; BALANCE=""; MARGIN=0.2; ATTN="auto"; PILOT=10; PILOT_ONLY=0
NETWORK_VOLUME=""; VOLUME_DISK=250; CONTAINER_DISK=30; SSH_KEY="${HOME}/.ssh/id_ed25519"; DRY=0; ALLOW_EXISTING=0; RUN_ID=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) GPU+=("$2"); shift 2;;
    --cloud) CLOUD_TYPE="$2"; shift 2;;
    --image) IMAGE="$2"; shift 2;;
    --rate) RATE="$2"; shift 2;;
    --hours) HOURS="$2"; shift 2;;
    --cap) CAP="$2"; shift 2;;
    --balance) BALANCE="$2"; shift 2;;
    --margin) MARGIN="$2"; shift 2;;
    --attn) ATTN="$2"; shift 2;;
    --pilot) PILOT="$2"; shift 2;;
    --pilot-only) PILOT_ONLY=1; shift;;
    --network-volume) NETWORK_VOLUME="$2"; shift 2;;
    --volume-disk) VOLUME_DISK="$2"; shift 2;;
    --container-disk) CONTAINER_DISK="$2"; shift 2;;
    --ssh-key) SSH_KEY="$2"; shift 2;;
    --run-id) RUN_ID="$2"; shift 2;;
    --allow-existing-pods) ALLOW_EXISTING=1; shift;;
    --dry-run) DRY=1; shift;;
    -h|--help) sed -n 2,22p "$0"; exit 0;;
    *) echo "unknown argument: $1" >&2; exit 1;;
  esac
done
[[ -n "$RATE" && -n "$HOURS" && -n "$CAP" ]] || { echo "need --rate --hours --cap" >&2; exit 1; }
[[ ${#GPU[@]} -gt 0 ]] || GPU=("NVIDIA H100 PCIe")
[[ "$ATTN" == "auto" || "$ATTN" == "fa3" || "$ATTN" == "eager" ]] || { echo "--attn must be auto, fa3 or eager" >&2; exit 1; }
if [[ $DRY -eq 0 && -z "${RUNPOD_API_KEY:-}" ]]; then echo "RUNPOD_API_KEY is not set" >&2; exit 1; fi
awk -v h="$HOURS" 'BEGIN{ if (h <= 0 || h > 5) { print "--hours must be in (0, 5]" > "/dev/stderr"; exit 1 } }'
for f in runpod_api.py watchdog.py; do [[ -f "$CLOUD/$f" ]] || { echo "missing $CLOUD/$f" >&2; exit 1; }; done
DRYFLAG=(); [[ $DRY -eq 1 ]] && DRYFLAG=(--dry-run)
$PY "$HERE/train_probes_pod.py" --self-test >/dev/null || { echo "train_probes_pod.py self-test failed" >&2; exit 1; }

EXPERIMENT="probes-all24-2splits"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
SRC_HASH=$(cat "$HERE/train_probes_pod.py" "$HERE/setup_pod.sh" "$HERE/run_probes_job.sh" | shasum -a 256 | cut -c1-8)
[[ -n "$RUN_ID" ]] || RUN_ID="${EXPERIMENT}-${STAMP}-${SRC_HASH}"
RUN_DIR="$HERE/runs/$RUN_ID"; mkdir -p "$RUN_DIR"
LOG="$RUN_DIR/launch.log"; COST_LOG="$CLOUD/cost-log.csv"
echo "run_id=$RUN_ID run_dir=$RUN_DIR gpu=${GPU[*]} cloud=$CLOUD_TYPE rate=$RATE hours=$HOURS cap=$CAP attn=$ATTN pilot=$PILOT pilot_only=$PILOT_ONLY" | tee -a "$LOG"

# ---- 1. pre-flight: key, no stray pods, balance, cap
PRE_ARGS=(--rate "$RATE" --hours "$HOURS" --cap "$CAP" --margin "$MARGIN")
[[ -n "$BALANCE" ]] && PRE_ARGS+=(--balance "$BALANCE")
[[ $ALLOW_EXISTING -eq 1 ]] && PRE_ARGS+=(--allow-existing-pods)
if ! $PY "$CLOUD/runpod_api.py" "${DRYFLAG[@]}" --log "$LOG" preflight "${PRE_ARGS[@]}" | tee "$RUN_DIR/preflight.json"; then
  echo "PRE-FLIGHT FAILED; nothing launched" | tee -a "$LOG" >&2; exit 2
fi
EXPECTED=$(awk -v r="$RATE" -v h="$HOURS" -v m="$MARGIN" -v d="$VOLUME_DISK" 'BEGIN{printf "%.2f", r*h*(1+m) + d*0.10/730*h}')

# ---- 2. cost-log row before the pod exists (playbook safeguard 8)
$PY "$CLOUD/runpod_api.py" costlog-append --path "$COST_LOG" --row "$(printf '{"date_utc":"%s","run_id":"%s","experiment":"%s","gpu":"%s","cloud":"%s","rate_usd_h":"%s","planned_hours":"%s","expected_usd":"%s","balance_before":"%s","status":"%s","notes":"pilot=%s pilot_only=%s attn=%s volume_gb=%s dry_run=%s"}' \
  "$STAMP" "$RUN_ID" "$EXPERIMENT" "${GPU[0]}" "$CLOUD_TYPE" "$RATE" "$HOURS" "$EXPECTED" "$BALANCE" "planned" "$PILOT" "$PILOT_ONLY" "$ATTN" "$VOLUME_DISK" "$DRY")" >/dev/null
echo "cost-log row appended: $COST_LOG (expected \$$EXPECTED incl. disk)" | tee -a "$LOG"

# ---- 3. create the pod: volume disk at /workspace holds venv, HF cache, activations, outbox
PUBKEY=""
if [[ -f "${SSH_KEY}.pub" ]]; then PUBKEY="$(cat "${SSH_KEY}.pub")"; elif [[ $DRY -eq 0 ]]; then echo "ssh public key ${SSH_KEY}.pub not found" >&2; exit 1; else PUBKEY="ssh-ed25519 DRYRUN"; fi
CREATE_ARGS=(--name "${RUN_ID}" --cloud "$CLOUD_TYPE" --image "$IMAGE" --container-disk "$CONTAINER_DISK"
             --volume-disk "$VOLUME_DISK" --volume-mount /workspace --pass-api-key
             --env "PUBLIC_KEY=$PUBKEY" --env "RUN_ID=$RUN_ID" --env "PLANNED_HOURS=$HOURS")
for g in "${GPU[@]}"; do CREATE_ARGS+=(--gpu "$g"); done
[[ -n "$NETWORK_VOLUME" ]] && CREATE_ARGS+=(--network-volume "$NETWORK_VOLUME")
[[ -n "${HF_TOKEN:-}" ]] && CREATE_ARGS+=(--env "HF_TOKEN=$HF_TOKEN")
$PY "$CLOUD/runpod_api.py" "${DRYFLAG[@]}" --log "$LOG" create "${CREATE_ARGS[@]}" > "$RUN_DIR/create.json"
POD_ID=$($PY -c "import json; print(json.load(open('$RUN_DIR/create.json'))['pod']['id'])")
echo "pod created: $POD_ID" | tee -a "$LOG"
$PY "$CLOUD/runpod_api.py" costlog-update --path "$COST_LOG" --run-id "$RUN_ID" --updates "{\"pod_id\":\"$POD_ID\",\"status\":\"created\"}" >/dev/null

cleanup() {
  local rc=$?
  if [[ -n "${POD_ID:-}" && "${WATCHDOG_OWNS:-0}" -eq 0 ]]; then
    echo "exiting with rc=$rc before the watchdog took over; terminating pod $POD_ID" | tee -a "$LOG" >&2
    $PY "$CLOUD/runpod_api.py" "${DRYFLAG[@]}" --log "$LOG" delete --pod-id "$POD_ID" || echo "!! DELETE failed; terminate $POD_ID in the console NOW !!" | tee -a "$LOG" >&2
    $PY "$CLOUD/runpod_api.py" costlog-update --path "$COST_LOG" --run-id "$RUN_ID" --updates "{\"status\":\"terminated before job start (rc=$rc)\"}" >/dev/null || true
  fi
}
trap cleanup EXIT

# ---- 4. wait for RUNNING + public IP + SSH port; refuse a rate more than 10% above the quote
$PY "$CLOUD/runpod_api.py" "${DRYFLAG[@]}" --log "$LOG" wait --pod-id "$POD_ID" --timeout 900 > "$RUN_DIR/ready.json"
HOST=$($PY -c "import json; print(json.load(open('$RUN_DIR/ready.json'))['host'])")
PORT=$($PY -c "import json; print(json.load(open('$RUN_DIR/ready.json'))['port'])")
RATE_SEEN=$($PY -c "import json; print(json.load(open('$RUN_DIR/ready.json')).get('costPerHr'))")
echo "pod ready: $HOST:$PORT costPerHr=$RATE_SEEN (quoted $RATE)" | tee -a "$LOG"
if [[ $DRY -eq 0 ]]; then
  awk -v seen="$RATE_SEEN" -v q="$RATE" 'BEGIN{ if (seen+0 > q*1.1) { printf "pod rate %s exceeds quoted %s by more than 10%%\n", seen, q > "/dev/stderr"; exit 1 } }'
fi
SSH=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o BatchMode=yes -i "$SSH_KEY" -p "$PORT" "root@$HOST")
RSYNC_E="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -o BatchMode=yes -i $SSH_KEY -p $PORT"
REMOTE_JOB="/workspace/job/$RUN_ID"; REMOTE_RES="/workspace/results/$RUN_ID"
run_ssh() { if [[ $DRY -eq 1 ]]; then echo "DRY-RUN ssh root@$HOST:$PORT -- $*" | tee -a "$LOG"; else "${SSH[@]}" "$@"; fi; }

# ---- 5. ssh reachable
for i in $(seq 1 30); do
  if [[ $DRY -eq 1 ]]; then run_ssh true; break; fi
  if "${SSH[@]}" true 2>/dev/null; then break; fi
  [[ $i -eq 30 ]] && { echo "ssh never came up" | tee -a "$LOG" >&2; exit 3; }
  sleep 10
done

# ---- 6. ship this directory (the three files) and record their hashes
(cd "$HERE" && shasum -a 256 train_probes_pod.py setup_pod.sh run_probes_job.sh) > "$RUN_DIR/source.sha256"
if [[ $DRY -eq 1 ]]; then
  echo "DRY-RUN rsync -az $HERE/{train_probes_pod.py,setup_pod.sh,run_probes_job.sh} root@$HOST:$REMOTE_JOB/" | tee -a "$LOG"
else
  run_ssh "mkdir -p $REMOTE_JOB $REMOTE_RES; command -v rsync >/dev/null 2>&1 || (apt-get update -qq && apt-get install -y -qq rsync) >/dev/null 2>&1; command -v rsync >/dev/null"
  rsync -rltz --partial -e "$RSYNC_E" "$HERE/train_probes_pod.py" "$HERE/setup_pod.sh" "$HERE/run_probes_job.sh" "root@$HOST:$REMOTE_JOB/"
fi

# ---- 7. start the frozen pod sequence, detached, under timeout = hours + 20 min
TIMEOUT_S=$(awk -v h="$HOURS" 'BEGIN{printf "%d", h*3600+1200}')
POD_ARGS="--run-id $RUN_ID --hours $HOURS --attn $ATTN --pilot $PILOT"; [[ $PILOT_ONLY -eq 1 ]] && POD_ARGS="$POD_ARGS --pilot-only"
REMOTE_CMD="cd $REMOTE_JOB && nohup sh -c 'timeout --signal=TERM --kill-after=300 ${TIMEOUT_S}s bash run_probes_job.sh pod $POD_ARGS > $REMOTE_RES/nohup.log 2>&1; echo \$? > $REMOTE_RES/EXIT' < /dev/null > /dev/null 2>&1 & echo started"
STARTED_AT=$(date -u +%FT%TZ)
run_ssh "mkdir -p $REMOTE_RES && $REMOTE_CMD" | tee -a "$LOG"
$PY "$CLOUD/runpod_api.py" costlog-update --path "$COST_LOG" --run-id "$RUN_ID" --updates "{\"start_utc\":\"$STARTED_AT\",\"status\":\"running\"}" >/dev/null
echo "job started at $STARTED_AT on $POD_ID; outbox syncs to $RUN_DIR/shard-0 every 90 s" | tee -a "$LOG"

# ---- 8. watchdog owns the pod: rsync loop, heartbeat, deadline, DONE -> DELETE, cost-log close
WATCHDOG_OWNS=1
WD_ARGS=(--run-dir "$RUN_DIR" --run-id "$RUN_ID" --pod-id "$POD_ID" --ssh-host "$HOST" --ssh-port "$PORT" --ssh-key "$SSH_KEY"
         --remote-dir "$REMOTE_RES" --shard-index 0 --planned-hours "$HOURS" --grace-minutes 20 --started-at "$STARTED_AT"
         --rate "$RATE" --cost-log "$COST_LOG" --interval 90 --heartbeat-stale 900 --first-heartbeat-by 5)
[[ $DRY -eq 1 ]] && WD_ARGS+=(--dry-run)
if [[ $DRY -eq 0 ]] && command -v caffeinate >/dev/null; then
  caffeinate -i $PY "$CLOUD/watchdog.py" "${WD_ARGS[@]}"
else
  $PY "$CLOUD/watchdog.py" "${WD_ARGS[@]}"
fi
echo "done. verify with: python3 $CLOUD/runpod_api.py list   (must print [])" | tee -a "$LOG"
echo "probes: $RUN_DIR/shard-0/{gptoss-20b.pkl,gptoss-20b-basesplit.pkl,role_probes*.pkl,probes*.npz,probes*.json,acc_by_*.csv,metadata.json}" | tee -a "$LOG"
