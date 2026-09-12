#!/bin/bash
# TOY LAB (MATS 12.0 application experiment). Orchestrates the steering experiments on a multi-GPU pod.
# Prep steps use one GPU; cell runs shard across all GPUs (one process per GPU via CUDA_VISIBLE_DEVICES).
# Nothing here launches a pod or spends money; Codex runs it inside an already-approved pod. set -e so a
# broken stage stops the chain instead of cascading into empty results.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/workspace/venv-probes/bin/python}
PROBES=${PROBES:-/workspace/results/20260911T030050Z/probes-full/probes.npz}
ACTS=${ACTS:-/workspace/acts/20260911T030050Z/probes-full}
PROBES_DIR=${PROBES_DIR:-/workspace/results/20260911T030050Z/probes-full}
N_GPU=${N_GPU:-$(nvidia-smi -L | wc -l)}
PAGES_DIR=${PAGES_DIR:-data/pages-24}
SCREEN_PAGES=${SCREEN_PAGES:-12}
OUT=${OUT:-out}
echo "=== $(date -u +%H:%M:%S) GPUs=$N_GPU pages_dir=$PAGES_DIR screen_pages=$SCREEN_PAGES"

shard_run () {  # $1 cells file, $2 out dir, $3 optional --prefixes path
  local cells="$1" outdir="$2" pref="${3:-}"
  local pids=()
  for g in $(seq 0 $((N_GPU - 1))); do
    CUDA_VISIBLE_DEVICES=$g $PY run_cells.py --cells "$cells" --manifest "$PAGES_DIR/manifest.json" \
      --out "$outdir" --probes "$PROBES" --shard "$g/$N_GPU" $pref > "$outdir/shard-$g.log" 2>&1 &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
}

# 0. page set (fetch on the pod; falls back to the pilot's 5 if the dataset is unavailable)
if [ ! -f "$PAGES_DIR/manifest.json" ]; then
  $PY pages.py --out "$PAGES_DIR" --pages 24 --seed 2026 || \
  $PY pages.py --out "$PAGES_DIR" --from-local ../agent-hijacking/data/pilot-20260911
fi

# 1. directions + geometry (exp 6), one GPU
CUDA_VISIBLE_DEVICES=0 $PY directions.py geometry --probes-dir "$PROBES_DIR" --acts-dir "$ACTS" --out directions
CUDA_VISIBLE_DEVICES=0 $PY directions.py block --probes-dir "$PROBES_DIR" --layers 5,8,11,14,17 --n-prompts 300 --out directions

# 2. prep for exp 2 (doubt prefixes) and exp 7 (decision vector), one GPU each
CUDA_VISIBLE_DEVICES=0 $PY prep.py doubt --manifest "$PAGES_DIR/manifest.json" --pages 24 --out prefixes.json
CUDA_VISIBLE_DEVICES=0 $PY prep.py decision --manifest "$PAGES_DIR/manifest.json" --pages 24 --steer-layer 11 --out directions/decision.npz

# 2b. the destyle program and the declaration direction (2026-09-12 direction change; see README)
DESTYLED=${DESTYLED:-data/destyled.json}   # page_id -> destyled forged paragraph, generated with the authors' destyle prompt
if [ -f "$DESTYLED" ]; then
  CUDA_VISIBLE_DEVICES=0 $PY prep.py destyle --manifest "$PAGES_DIR/manifest.json" --pages 24 --destyled "$DESTYLED" --steer-layer 11 --out directions/destyle.npz
fi
CUDA_VISIBLE_DEVICES=0 $PY prep.py declaration --manifest "$PAGES_DIR/manifest.json" --pages 24 --steer-layer 11 --out directions/declaration.npz

# 3. screen: build cells (small n), run sharded, analyze. Order = most likely to show prevention first.
mkdir -p cells "$OUT"
$PY build_cells.py exp10 --manifest "$PAGES_DIR/manifest.json" --pages "$SCREEN_PAGES" --out cells/exp10.jsonl
if [ -f "$DESTYLED" ]; then
  $PY build_cells.py exp8 --manifest "$PAGES_DIR/manifest.json" --pages "$SCREEN_PAGES" --dir-file directions/destyle.npz --alpha 1 --alpha2 2 --out cells/exp8.jsonl
  $PY build_cells.py exp1 --manifest "$PAGES_DIR/manifest.json" --pages "$SCREEN_PAGES" --patch-source destyled --out cells/exp1d.jsonl --layers 3,5,8,11,14,17,20
fi
$PY build_cells.py exp9 --manifest "$PAGES_DIR/manifest.json" --pages "$SCREEN_PAGES" --dir-file directions/declaration.npz --alpha 4 --alpha2 8 --out cells/exp9.jsonl
for e in exp10 exp8 exp1d exp9; do
  [ -f "cells/$e.jsonl" ] || continue
  mkdir -p "$OUT/$e"; echo "=== $(date -u +%H:%M:%S) screen $e"; shard_run "cells/$e.jsonl" "$OUT/$e" "--destyled $DESTYLED"
  $PY analyze.py --out "$OUT/$e" --manifest "$PAGES_DIR/manifest.json" --cells "cells/$e.jsonl"
done
$PY build_cells.py exp1 --manifest "$PAGES_DIR/manifest.json" --pages "$SCREEN_PAGES" --out cells/exp1.jsonl --layers 3,5,8,11,14,17,20
$PY build_cells.py exp2 --manifest "$PAGES_DIR/manifest.json" --pages "$SCREEN_PAGES" --out cells/exp2.jsonl
$PY build_cells.py exp3 --manifest "$PAGES_DIR/manifest.json" --pages "$SCREEN_PAGES" --out cells/exp3.jsonl
$PY build_cells.py exp4 --manifest "$PAGES_DIR/manifest.json" --pages "$SCREEN_PAGES" --out cells/exp4.jsonl
$PY build_cells.py exp5 --manifest "$PAGES_DIR/manifest.json" --pages "$SCREEN_PAGES" --dir-file directions/block11.npz --out cells/exp5.jsonl
$PY build_cells.py exp7 --manifest "$PAGES_DIR/manifest.json" --pages "$SCREEN_PAGES" --dir-file directions/decision.npz --out cells/exp7.jsonl

for e in exp1 exp3 exp4 exp5; do
  mkdir -p "$OUT/$e"; echo "=== $(date -u +%H:%M:%S) screen $e"; shard_run "cells/$e.jsonl" "$OUT/$e"
  $PY analyze.py --out "$OUT/$e" --manifest "$PAGES_DIR/manifest.json" --cells "cells/$e.jsonl"
done
mkdir -p "$OUT/exp2"; shard_run cells/exp2.jsonl "$OUT/exp2" "--prefixes prefixes.json"; $PY analyze.py --out "$OUT/exp2" --manifest "$PAGES_DIR/manifest.json" --cells cells/exp2.jsonl
mkdir -p "$OUT/exp7"; shard_run cells/exp7.jsonl "$OUT/exp7"; $PY analyze.py --out "$OUT/exp7" --manifest "$PAGES_DIR/manifest.json" --cells cells/exp7.jsonl

# 4. stage B: rerun the positive arms with more seeds on the pages where baseline uploaded
for e in exp10 exp8 exp1d exp9 exp1 exp2 exp3 exp5 exp7; do
  if [ -f "$OUT/$e/stage-b.jsonl" ]; then
    echo "=== $(date -u +%H:%M:%S) stage-B $e"; mkdir -p "$OUT/$e-b"
    pref="--destyled $DESTYLED"; [ "$e" = "exp2" ] && pref="--prefixes prefixes.json"
    shard_run "$OUT/$e/stage-b.jsonl" "$OUT/$e-b" "$pref"
    $PY analyze.py --out "$OUT/$e-b" --manifest "$PAGES_DIR/manifest.json" --cells "$OUT/$e/stage-b.jsonl"
  fi
done
echo "=== $(date -u +%H:%M:%S) done. Sync $OUT, directions/, prefixes.json, cells/ back to the Mac."
