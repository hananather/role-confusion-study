# Cloud probe job: the authors' gpt-oss-20b role probes, all 24 layers, two splits

One pod, one shot. Trains the role probes of Ye, Cui, and Hadfield-Menell (NB02 cells 1 to 20,
commit `ec333c40`) on a cloud GPU with the authors' own utils, extended to all 24 layers and to a
second split grouped by base text. Probes and CSVs rsync back to the Mac every 90 seconds. The pod
is terminated when `DONE` appears or the deadline passes. Activations stay on the pod.

| File | What |
| --- | --- |
| `train_probes_pod.py` | Headless NB02 cells 1 to 20 for `gptoss-20b`. Extensions behind flags: `--layers all`, `--splits prompt,base`, `--attn auto`. Pilot mode `--pilot N`. `--hours` guard armed after the model loads. |
| `setup_pod.sh` | The authors' `setup_python.sh` recipe plus the four fixes the reference H200 run needed. Python 3.12 venv via uv, torch 2.9.1+cu128, transformers 4.57.5, triton 3.5.1, kernels 0.11.5, cuML 25.08, datasets 5.0.1, zstandard, pandas 2.3.3, scikit-learn 1.7.2. Clones the authors' repo at the commit. Downloads the model into `/workspace/hf` (the loader hardcodes that path). |
| `run_probes_job.sh` | `launch` (Mac): preflight, cost-log row, create pod, rsync these three files, start the pod sequence, watchdog. `pod` (pod): backstop, key check, setup, pilot, full run, sha256, `DONE`. |

Reuses `cloud/runpod_api.py` (REST calls, cost log) and `cloud/watchdog.py` (rsync loop,
heartbeat, deadline, DELETE). Nothing here is executed on the Mac except a CUDA-free self-test.

## Commands

Pre-flight on the Mac (nothing is launched):

```bash
cd "/Users/hananather/Desktop/MATS 12.0/replication"
python3 cloud/probes/train_probes_pod.py --self-test
RUNPOD_DRY_RUN=1 cloud/probes/run_probes_job.sh launch --gpu "NVIDIA H100 PCIe" --cloud SECURE \
  --rate 2.89 --hours 3 --cap 12 --dry-run
python3 cloud/runpod_api.py list          # must print []
```

Launch on H100 PCIe, Secure Cloud (the authors' attention kernel, FA3):

```bash
export RUNPOD_API_KEY=...                 # account-scoped key; rotate after the run
cloud/probes/run_probes_job.sh launch --gpu "NVIDIA H100 PCIe" --cloud SECURE \
  --rate 2.89 --hours 3 --cap 12 --balance <console balance> --pilot 10
```

Launch on A100 80 GB, Community Cloud (eager attention fallback, a logged divergence):

```bash
cloud/probes/run_probes_job.sh launch --gpu "NVIDIA A100 80GB PCIe" --cloud COMMUNITY \
  --rate 1.19 --hours 4 --cap 7 --balance <console balance> --pilot 10 --attn eager
```

Useful flags: `--pilot-only` stops after the pilot so a human can read the table before paying
for the full run. `--network-volume <id>` keeps the activations after termination (250 GB at
$0.07/GB/month is $17.50/month; without it they die with the pod). `--gpu` is repeatable in
priority order. `--attn auto` picks FA3 on Hopper and eager elsewhere.

While it runs: the watchdog prints status every 30 minutes. Pilot outputs appear in
`cloud/probes/runs/<run_id>/shard-0/pilot/` within a few minutes of the pilot finishing.
`touch cloud/probes/runs/<run_id>/STOP` terminates after one last rsync.

After it finishes:

```bash
python3 cloud/runpod_api.py list          # must print []
ls cloud/probes/runs/<run_id>/shard-0/    # DONE, gptoss-20b.pkl, gptoss-20b-basesplit.pkl, ...
tail -3 cloud/cost-log.csv
```

## What the pod does, in order

1. Backstop: `sleep hours+15min; DELETE /pods/$RUNPOD_POD_ID` in the background. Key check with
   `GET /pods/$RUNPOD_POD_ID` (a failure is logged as `KEYFAIL`; the Mac watchdog still terminates).
2. `setup_pod.sh` (about 8 minutes cold). Prints the version table and whether the GPU is Hopper.
3. Pilot: `--pilot 10` (2 C4 + 7 Dolma3 documents, 45 prompts), all 24 layers, both splits, all
   8 role spaces (384 tiny fits). Prints the accuracy table and `pilot.json` with measured seconds
   per forward batch and per fit. Gate: custom forward equals HF logits, role counts equal across
   the five roles, experts are MXFP4, all 384 probes present. A failed gate ends the job.
4. Full run: 250 requested documents (249 delivered, as the authors got), 1,245 prompts, about
   705k content tokens, 24 layers x 2 splits x 8 role spaces = 384 probes (192 per split).
5. `sha256.txt`, `EXIT`, `DONE`. The watchdog does a final rsync and DELETEs the pod.

## Outputs (outbox, rsynced to `cloud/probes/runs/<run_id>/shard-0/`)

| File | Format |
| --- | --- |
| `gptoss-20b.pkl`, `gptoss-20b-basesplit.pkl` | The authors' cell 20 pickle: list of dicts with the cuML `Pipeline`, `acc`, `nll`, `acc_by_role`, `acc_by_pos`, `layer_ix`, `role_space`, `roles_map`, `n_inputs`. Notebook order (role space outer, layer inner). Needs cuML to unpickle. |
| `role_probes.pkl`, `role_probes-basesplit.pkl` | Portable sklearn clone (coef and intercept copied), the reference repo's format with its two corrections. |
| `probes.npz`, `probes-basesplit.npz` | Our format: keys `{space}_L{layer:02d}__coef` and `__intercept`, float32, spaces `ua uat uca ucat sua suat suca sucat`. Binary spaces carry one coefficient row (cuML convention). |
| `probes.json`, `probes-basesplit.json` | Our results document: per probe `acc`, `nll`, `acc_by_role`, `n_train`, `n_test`, `n_inputs`, seconds. |
| `acc_by_role_gptoss-20b[-basesplit].csv`, `acc_by_pos_gptoss-20b[-basesplit].csv` | Exactly what cell 20 writes. |
| `val_acc_by_layer[-basesplit].csv` | The cell 20 pivot (layer x role space). |
| `tokens.parquet`, `prompts.parquet` | The labelled token table (with `question_ix` and the notebook's `keep` filter as a column) and the prompt table. `sample_ix` is the row in each `layerNN.npy`. |
| `metadata.json` | Versions, GPU, compute capability, expert dtype string, attention implementation, HF snapshot, document counts, role counts, timings per phase, peak VRAM and RSS, the list of extensions. |
| `train.log`, `job.log`, `setup.log`, `pilot/` | Logs. cuML solver warnings (line-search failures, see the reference report section 6.3) are in `train.log`. |

Activations on the pod at `/workspace/acts/<run_id>/`: `layerNN.npy` float16 `(n_tokens, 2880)`,
`tokens.parquet`, `metadata.json`, the layout of `replication/probes/extract_activations.py`.
About 97 GB for 24 layers. Not synced.

## Fidelity

Unchanged from the notebook: model loader (MXFP4 experts, the authors' custom forward pass
verified equal to HF logits), C4 and Dolma3 streaming with seed 123, sequence rendering, batch
32 with left padding, role labelling, `fit_lr` and `get_probe_result` (cuML L2 logistic
regression, C 5e-3, max_iter 5000, linesearch_max_iter 100, no scaling), the prompt split, the
save cell.

Extensions, each recorded in `metadata.json`:

- Layers 0 to 23 instead of 0, 2, ..., 22.
- A second split on unique `question_ix` with the same seed and test size, so the five renderings
  of one text never straddle train and test. The prompt split is the authors' and is saved first.
- Attention: FA3 on Hopper as the authors. On A100 the loader's kernel cannot run, so the
  metadata tuple is patched to `eager`. Exact attention either way, different kernels.
- Memory: cell 14 holds the whole activation cube in RAM (49 GB for 12 layers, twice that for
  24, plus a float16 copy). This script runs the same forward function and writes each layer's
  states per batch into float16 memmaps, then fits one layer at a time. Same values, same casts,
  same row order. The fit loop is layer outer; the saved lists are re-sorted to the notebook's order.

Known, inherited caveats: the C4 sample is not revision-pinned (nobody can reproduce the authors'
exact 249 documents); many cuML fits stop on line-search failures because the features are
unscaled (the authors' setting). See `reference/README-jamesnelmore.md`.

## Expected runtime and cost

Reference (authors' notebook unchanged, H200, cuML 25.08): forward passes 210 s for 12 layers,
803 s for 96 probes (8.4 s per probe), 1,097 s for the notebook.

Ours, estimated from those numbers. Not measured. The pilot measures them.

| Phase | H100 PCIe (FA3) | A100 80 GB (eager) |
| --- | --- | --- |
| Setup: uv env (about 2 GB of RAPIDS wheels), repo, 14 GB model | 8 to 12 min | 8 to 12 min |
| Pilot (load, generation tests, 45 prompts, 384 tiny fits) | 8 to 12 min | 10 to 15 min |
| Full forward, 39 batches, 24 layers to disk (97 GB) | 5 to 8 min | 10 to 15 min |
| Full fits, 384 probes at 8 to 10 s each, plus 24 layer loads | 55 to 70 min | 80 to 110 min |
| Total | about 1.5 to 1.8 h | about 2 to 2.7 h |

Cost at COMPUTE-PLAN 1.3 prices (billing per second; 250 GB volume disk adds about $0.03 per hour):

| GPU | Rate | Likely billed | Budget (`--hours`) | Cap (`--cap`) |
| --- | --- | --- | --- | --- |
| H100 PCIe, Secure | $2.89/h | $4.50 to $5.50 | 3 h ($8.70) | $12 |
| A100 80 GB PCIe, Community | $1.19/h | $2.50 to $3.50 | 4 h ($4.80) | $7 |

If the pilot's `pilot.json` projects the full run past the budget, the pod-side script still
starts the full run with whatever remains; the `--hours` alarm saves partial probes (marked
`FAILED` with `partial`), and the watchdog terminates. Prefer `--pilot-only` first if unsure.

## Checklist

Before launch (GPU-PLAYBOOK pre-flight):

- [ ] Billing page read; balance near the cap; Auto-Pay off. Pass `--balance`.
- [ ] `python3 cloud/runpod_api.py list` prints `[]`.
- [ ] Account-scoped key in `RUNPOD_API_KEY` (never in a file). Rotate after the run.
- [ ] `~/.ssh/id_ed25519.pub` exists (passed as `PUBLIC_KEY`).
- [ ] Dry run passed. `--self-test` passed.
- [ ] GPU id string confirmed in the console if `create` returns 400 (`NVIDIA H100 PCIe`, `NVIDIA A100 80GB PCIe`).
- [ ] Cost-log row exists in `cloud/cost-log.csv` (the script writes it before the pod exists).

Pilot gate (read `shard-0/pilot/` when it lands):

- [ ] `train.log` shows `Expert precision: FloatType(bitwidth_exponent=2, bitwidth_mantissa=1, ...)` and the attention implementation you expected.
- [ ] `Verified custom forward pass successfully matches original model output!`
- [ ] Role counts equal across system, user, cot, assistant, tool.
- [ ] `pilot.json` projection fits the budget.

After the run:

- [ ] `DONE` present; `EXIT` is `0`; `sha256.txt` matches the synced files.
- [ ] `python3 cloud/runpod_api.py list` prints `[]`. Cost-log row closed with billed USD.
- [ ] `val_acc_by_layer.csv` even layers within a few points of the reference table (`reference/README-jamesnelmore.md` section 2.3; layer 16 five-role about 0.72, binary about 0.97).
- [ ] `metadata.json` records the HF snapshot `6cee5e81...`, cuML 25.08, transformers 4.57.5.
- [ ] Copy `probes*.npz`, `probes*.json`, `role_probes*.pkl`, `gptoss-20b*.pkl`, the CSVs and `metadata.json` to wherever the comparison study reads them; the pod copies are gone.
