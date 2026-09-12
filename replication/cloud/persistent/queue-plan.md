# My persistent H100 experiment queue

## My objective and execution boundary

I use one H100 with a 2 TB persistent volume for the next approved working session. I allow the brief handover overlap needed to replace the old pod, then keep one GPU worker. My first checkpoint remains **Appendix E Experiments 1, 2, and 3, with Figures 7 and 20–22**. I preserve the all-24-layer, eight-role-space, two-split probe objective as its prerequisite. The current priority update in `RUNBOOK-2026-09-10.md` supersedes that file's older multi-pod schedule.

I am in **Understand: controlled replication**. My north star is whether the fixed gardening example reproduces under the authors' implemented recipe. I prioritize fidelity, inspecting the actual measurements, and a verifiable result over adding experiments to fill the GPU session.

This document records commands and readiness; it does not claim that any stage has run. My session controller supplies the actual deployment paths, run IDs, remaining time, and billing deadline. I retain a session-wide deadline and artifact sync. A completed or failed experiment ends that experiment; my persistent worker does not inherit the old per-job pod-deletion behavior.

## My path contract

These commands run **inside the new pod**, after deployment and environment checks. I require my controller to set the following variables from its saved session record:

```bash
REPLICATION_ROOT=/workspace/replication
PROBE_PY=/workspace/venv-probes/bin/python
AUTHORS_REPO=/workspace/prompt-injection-as-role-confusion
: "${SESSION_ID:?I need the controller's recorded session ID}"
: "${PILOT_HOURS:?I need the remaining pilot budget}"
: "${FULL_HOURS:?I need the remaining full-run budget}"
PILOT_RESULTS=/workspace/results/$SESSION_ID/probe-pilot
PROBE_RESULTS=/workspace/results/$SESSION_ID/probes-full
PROBE_ACTS=/workspace/acts/$SESSION_ID/probes-full
APPENDIX_E_RESULTS=/workspace/results/$SESSION_ID/appendix-e-L12
cd "$REPLICATION_ROOT"
```

I preserve the authors' checkout at `ec333c40fd43fe991e1ebf66765051b6d7e35784`. I reuse the model and validated environment from the persistent volume. My controller verifies that `/workspace` is the intended persistent mount before creating outputs there. The direct commands below do not call `cloud/launch.sh`, `cloud/probes/run_probes_job.sh pod`, or the old per-job watchdog: those routes can delete the worker.

## 1. My probe prerequisite

I first complete the existing ten-text pilot with all requested probe combinations:

```bash
"$PROBE_PY" cloud/probes/train_probes_pod.py \
  --repo "$AUTHORS_REPO" --pilot 10 --layers all --splits prompt,base \
  --attn fa3 --results "$PILOT_RESULTS" \
  --acts "/workspace/acts/$SESSION_ID/probe-pilot" \
  --heartbeat-file "$PILOT_RESULTS/heartbeat" --hours "$PILOT_HOURS"
```

I continue only after the pilot exits successfully and its saved metadata confirms the custom forward check, MXFP4 experts, expected attention, equal target-role counts, and all 384 planned probes. I check its measured projection against the time remaining in this session. I do not treat an allocated GPU or a passing package-import check as a completed pilot.

```bash
"$PROBE_PY" cloud/probes/train_probes_pod.py \
  --repo "$AUTHORS_REPO" --layers all --splits prompt,base --attn fa3 \
  --results "$PROBE_RESULTS" --acts "$PROBE_ACTS" \
  --heartbeat-file "$PROBE_RESULTS/heartbeat" --hours "$FULL_HOURS"
```

I require 192 probes per split, 384 total, with `suca_L12__coef` and `suca_L12__intercept` present in `probes.npz`. I retain the native `gptoss-20b.pkl`, both portable probe archives, result JSON, CSVs, token/prompt tables, logs, and metadata. A partial run is recorded as partial. I do not use a tiny pilot probe as the full-corpus gardening probe.

The activation cube stays at `$PROBE_ACTS`; compact artifacts and logs sync back to my Mac. My current corpus has 713,109 tokens, so 24 layers of 2,880 float16 values require about **98.58 GB**. Both splits and all eight role spaces reuse that cube. My 2 TB volume supplies room for later distinct activation sets; it does not change the approved training method.

## 2. My first checkpoint: the three gardening conditions

Once the full-corpus prompt-split `suca` layer-12 probe is valid, I run the dedicated CUDA implementation:

```bash
"$PROBE_PY" cloud/persistent/appendix_e_cuda.py \
  --repo "$AUTHORS_REPO" --probe-run "$PROBE_RESULTS" \
  --out "$APPENDIX_E_RESULTS"
```

This command's interface was supplied by the implementation agent; my controller must confirm that the finished script has been deployed and checked before scheduling it. It uses the probe environment plus matplotlib. I record any renderer or typography differences while preserving the source's probability and displayed-token subset rules.

My inputs and acceptance contract are in `CHECKPOINT-1-APPENDIX-E.md`: the saved `gptoss-20b` transcript in `config/tomato.yaml`; the authors' three renderings; CUDA/Transformers with MXFP4 and FA3; the layer-12 post-attention-layernorm site; and the native prompt-split `suca` probe. The gardening forward length is 2,048. Training length remains 1,024.

I save exact rendered strings and token IDs, source/model/probe identities, raw per-token probabilities and labels, comparison means with denominators, and all four figures in PNG and PDF. I inspect Figures 20–22 as four-row role plots and Figure 7 as the three-condition CoTness overview. I recompute at least one mean from saved rows and compare the new figures against their source references. The source's 82%/83% no-tags CoTness discrepancy stays visible. I report the measured agreement or discrepancy; I do not tune the data or plotting denominator to match the paper.

## 3. My remaining-session rule

After the gardening outputs exist, I finish their verification and preserve the session artifacts. My subsequent request to run the existing experiments sequentially authorizes implementing the CUDA readout paths below. I now have one shared CUDA runner for the prepared Appendix K neutral prompts and RH6 prefills. It has passed static and model-free checks; its first pod runs remain smoke validations. I run **Appendix K first, then RH6** only within the time remaining on my one H100. These retain the existing experiment definitions and do not add model API calls.

### My CUDA Appendix K continuation

My 1,330 saved prompt variants contain 315,973 tokens across seven already-prepared conditions. I preserve their token IDs directly, their stored insertion/tag indices, their existing prompt subset, and the `suca` prompt-split probes at layers 8, 12, and 16. I preserve batch 1 without padding, matching my existing readout stage. The neutral texts also occur in the probe corpus, so these controls do not become a held-out replication simply by changing the execution backend.

I run one base prompt's available conditions as a smoke check in a separate directory:

```bash
"$PROBE_PY" cloud/persistent/cuda_readouts.py appendix-k \
  --repo "$AUTHORS_REPO" --probe-run "$PROBE_RESULTS" \
  --input "$REPLICATION_ROOT/appendix-k/runs/neutral" \
  --out "/workspace/results/$SESSION_ID/appendix-k-smoke" \
  --layers 8,12,16 --limit 1 --hours 0.1
```

I check the source hashes, token identities/counts, finite normalized probabilities, and generated tables before the full input set:

```bash
: "${APPENDIX_K_HOURS:?I need the controller's remaining Appendix K budget}"
"$PROBE_PY" cloud/persistent/cuda_readouts.py appendix-k \
  --repo "$AUTHORS_REPO" --probe-run "$PROBE_RESULTS" \
  --input "$REPLICATION_ROOT/appendix-k/runs/neutral" \
  --out "/workspace/results/$SESSION_ID/appendix-k" \
  --layers 8,12,16 --hours "$APPENDIX_K_HOURS"
```

The runner saves original-compatible `tokens.parquet`, copied input/build provenance, and per-item checkpoints, then calls the original Appendix K analysis without changing its bootstrap or smoothing definitions. Its outputs include `curves.csv`, `block_table.csv`, and the existing SVG figures. My CUDA model follows the authors' verified custom forward; I use float32 projection of the same pre-MLP site and record the backend/probe provenance change from the former MLX implementation.

### My CUDA RH6 continuation

My 1,200 prefills cover 100 existing templates. Offline tokenization of the frozen input gave **3,751,698 total tokens**, with 2,111–4,225 per item. Every required command/user/injection span selects at least one token. The source input has 750 tool-result span ends beyond the rendered string; I retain the existing token-overlap selection rule, which can only select actual tokenizer offsets, and record this inherited issue. I do not rewrite those spans or execute the command text being measured.

```bash
"$PROBE_PY" cloud/persistent/cuda_readouts.py rh6 \
  --repo "$AUTHORS_REPO" --probe-run "$PROBE_RESULTS" \
  --input "$REPLICATION_ROOT/rh/out/rh6-readings" \
  --out "/workspace/results/$SESSION_ID/rh6-smoke" \
  --layers 8,12,16 --limit 12 --hours 0.1
```

The smoke input is the first complete template's 12 existing condition cells. If it passes and the measured rate fits my remaining session, I continue with the unchanged full input:

```bash
: "${RH6_HOURS:?I need the controller's remaining RH6 budget}"
"$PROBE_PY" cloud/persistent/cuda_readouts.py rh6 \
  --repo "$AUTHORS_REPO" --probe-run "$PROBE_RESULTS" \
  --input "$REPLICATION_ROOT/rh/out/rh6-readings" \
  --out "/workspace/results/$SESSION_ID/rh6" \
  --layers 8,12,16 --hours "$RH6_HOURS"
```

I preserve the `sucat` and `uat` prompt-split probabilities and original span/page means, then call the existing RH6 analysis. It retains paired template contrasts, the command-minus-null comparison, and order/overlap summaries. This is a readout experiment; it adds neither behavior generation nor a who-wrote-it judge. An expired budget produces a recorded partial stage with completed item files; I resume only the identical input/probe/source contract. I produce the existing full summaries only after every selected item is extracted.

| Existing stage | What I verified | What prevents automatic scheduling now |
| --- | --- | --- |
| E4 OpenAssistant conversation generation | `cloud/job_conversations.py` is a CUDA implementation; its manifest selects 100 OpenAssistant conversations, two turns, temperature 1.0. | `e4/data/oasst_user_queries.csv` is missing. I need its data preparation and a measured pilot first. Generation alone does not produce Figure 23. |
| Figure 23 / E4 projection | `e4/project_conversations.py` exists. | Its forward path imports MLX. It cannot run unchanged on the H100; the full generated conversation set is also missing. |
| Appendix K neutral controls | Frozen inputs and `cloud/persistent/cuda_readouts.py appendix-k` now pass model-free checks. | The first pod smoke must verify actual CUDA extraction before the full run. The original `run_k.py extract` remains MLX-only. |
| E12b long-context gate | `e12/run_e12b.py` and small smoke artifacts exist. | Both generation and extraction use MLX. The candidate's decision and controls remain as specified in `EXPERIMENTS.md`; spare GPU time does not select it. |
| RH6 readings / who-wrote-it | Frozen inputs and `cloud/persistent/cuda_readouts.py rh6` now pass model-free checks. | CUDA readouts need the pod smoke and enough remaining time. The separate who-wrote-it generation path still uses MLX and is outside this readout implementation. |
| E8 refusal pilot | `cloud/job_steering.py`, its pilot manifest, and `data/strongreject_dataset.csv` exist. | `e8/harmless_prompts.csv` is missing; the positive-control smoke result and judge-label plan are unresolved. |
| E8 selection / confirmation | The CUDA runner exists. | Harmless prompts, selection forgeries, and declaration-contrast files are missing; the refusal and attack gates are unpassed. |
| E12a position-only null | `probes/position_null.py` is model-free. | It expects `probes-basesplit-L*.npz`, not the combined cloud archive; it recreates held-out groups with sklearn while the cloud trainer uses cuML, and it omits the cloud table's `keep` filter. I must resolve these compatibility and held-out-identity checks before calling its output a held-out null. |

### My existing CUDA E4 command, once its prerequisites are met

This is a conditional continuation, not an automatically appended stage. I use OpenAssistant only, consistent with the recorded no-ToxicChat constraint. If the existing session authorization covers the dataset preparation, the existing helper is:

```bash
mkdir -p e4/data
"$PROBE_PY" cloud/job_conversations.py \
  --prepare-oasst 100 --out e4/data/oasst_user_queries.csv
"$PROBE_PY" cloud/job_conversations.py \
  --manifest cloud/manifests/e4-conversations.json --plan
```

The preparation helper fetches the public OpenAssistant dataset; it is not a model API call. I record that download and the resulting input hash. I then use the existing sixteen-conversation pilot before any full batch:

```bash
: "${E4_HOURS:?I need the controller's remaining E4 pilot budget}"
"$PROBE_PY" cloud/job_conversations.py \
  --manifest cloud/manifests/e4-conversations.json \
  --results "/workspace/results/$SESSION_ID/e4-pilot" \
  --cache-dir /workspace/hf --pilot 16 --hours "$E4_HOURS"
```

I verify that `/workspace/hf` is compatible with this loader's cache layout before this optional command; its usual cache is `/workspace/hf-cache`, which could otherwise download a second model copy. The existing E4 manifest selects eager attention; I preserve and report that setting rather than silently changing it to the probe path's FA3. I omit `--terminate-self` so this experiment cannot delete my persistent worker. I check the process exit code and `runtime.json` status: this script currently writes `DONE` even in its failure cleanup, so that marker alone does not establish success.

## My external-call boundary

I do not need OpenAI, OpenRouter, Gemini, or other model APIs for the probe and gardening checkpoint. Existing Codex-account judge authorization is distinct from paid API-key billing. A new paid judge, forgery-generation service, ToxicChat credential, cloud sandbox, or shell-capable agent evaluation requires its own established authorization and inputs; the H100-and-storage approval does not supply those. I keep generated attacks and evaluation text as data, and I do not execute upstream shell-capable notebooks or their public upload examples.

## My handoff at the session boundary

I preserve the pod and volume IDs, actual rate and billing observation, deadline, current process and stage, output/activation paths, pinned environment and source hashes, successful and failed stage records, synchronized figures, and the next eligible command. The session controller enforces the authorized GPU window while retaining the persistent volume. I report whether the probes and gardening checkpoint completed, not whether the pod merely remained online.
