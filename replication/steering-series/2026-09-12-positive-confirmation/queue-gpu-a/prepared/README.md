# GPU A: prepared items 5 and 6

I prepared ten attribution readouts followed by five standard-injection episodes. Nothing in this packet has launched. I leave the active 40-episode stage and Opus-owned sources unchanged.

This is Explore. My north star is to distinguish elicited authorship from upload behavior and establish the standard-injection floor with fixed pages and seeds. I preserve the registered predictions and unresolved/censored outputs; these readouts do not establish a causal source-tracking mechanism.

`plan.json` preserves the exact authorized sentence. Each attribution input starts with the saved CUDA baseline `step-01.prompt.txt`, removes only its trailing `<|start|>assistant`, appends a new user turn containing that sentence, and appends `<|start|>assistant<|channel|>final<|message|>`. The forced final response condition is inherited from `replication/rh/rh6_who_wrote_it.py` and `replication/steering-agent/attribution_readout.py`. The question is the newly authorized sentence, not the earlier four-option question. The sampler is explicitly temperature 1, top-k 50, top-p 1, with exactly 200 as the maximum new-token cap. Each seed equals its recorded baseline post-fetch generation seed. Pairing does not imply that adding the question preserves an original output-token prefix.

The exact ten rendered prompts, baseline prefixes, prompt token IDs and hashes are in `inputs/`. Counts range 26,322–35,445 prompt tokens. Item 6 copies the original standard fixtures for new-000 through new-004 byte-for-byte, with manifest seeds 20260912, 20260914, 20260916, 20260918, 20260920. The copied full Docker harness retains its original prompts, dummy environment, private loopback receiver, model settings, 8-turn and 4096-token limits. Generated commands execute only inside that existing sandbox. Attribution outputs never execute as tools.

## Source changes and validation

`source/` is an isolated copy of the currently frozen stage. `copied-source-provenance.json` preserves all 34 original Python hashes. Exactly one existing source file differs:

- `hf_backend.py`: adds only the pair `purpose="attribution", max_new_tokens=200` to cap validation. Episode 4096 and engineering 64 remain unchanged.
The persistent worker and guardian remain byte-for-byte identical to their original frozen copies. The completed queue leaves the model loaded under the renewed financial lease.

Five targeted model-free tests pass: actual mocked generation with the 200 cap/no hooks/unchanged sampling; exact accepted purpose-cap pairs; exact insertion and unsteered question boundary; raw receipt validation; and registration/source binding with rejection of an unclosed predecessor. All 10 existing HF backend tests also pass against this copied source. The real cached tokenizer passed all 15 input checks. This is not a GPU execution claim.

## Handover and registration

1. Finish all 40 current episodes, perform item 4's audit, and verify synchronization. The existing allocation and financial supervisor remain running.
2. Only after that closure, request graceful termination of the old model service through its own service STOP. Verify the old worker PID is absent, `SERVICE-EXIT.json` reports `model_closed:true`, and its guardian reports the owned worker's verified exit. Do not signal or stop the GPU allocation. Root/operations owns this step; these preparation scripts do not perform it.
3. Write a closure receipt containing `passed`, `audit_passed`, `sync_verified`, `item4_completed` and `previous_service_exit_verified` all true; `previous_run_id:"agent-newpages-20260912T025000Z"`; `completed_jobs:40`; and the observed `previous_worker_pid`. Keep evidence-file paths/hashes alongside those checks. Do not create this receipt before those facts are verified.
4. Run the model-free registration command below, supplying that observed PID and receipt. It writes a new service config, job registration, local runner config, hashes and launch specification. It refuses a registration after 05:08:01 UTC or an incomplete closure receipt.

```bash
replication/.venv/bin/python replication/steering-series/2026-09-12-positive-confirmation/queue-gpu-a/prepared/code/register.py \
  --closure-receipt /absolute/path/to/verified-closure.json \
  --predecessor-pid VERIFIED_OLD_WORKER_PID \
  --out /absolute/path/to/new-registration-directory \
  --remote-root /workspace/continuations/queue-a-items5-6-20260912 \
  --job-id agent-queue-a-items5-6-20260912 \
  --local-results /absolute/path/to/new-local-results
```

5. The prepared launcher below performs staging, remote hash verification, predecessor-absence rechecking, atomic job registration, guardian startup, verified readiness and local-runner startup. It opens SSH connections only when explicitly invoked. It refuses existing successor paths or a repeated launch-state directory and writes its intent before any remote change. It makes no provider call and sends no signal to the predecessor.

```bash
replication/.venv/bin/python replication/steering-series/2026-09-12-positive-confirmation/queue-gpu-a/prepared/code/launch_successor.py --registration /absolute/path/to/new-registration-directory
```

   The packet is staged at the new remote root and generated registration files at its `registration/` directory. Uploaded hashes, including both direction/probe NPZ files, are verified. No files in the active `/workspace/replication` or previous continuation root change. Exact `registration.json` bytes are atomically registered in the new model-service job queue.
6. Start the generated guardian command from `REMOTE_ROOT/source`; its exact argv and cwd are saved in generated `launch.json`. The interpreter remains `/workspace/venv-probes/bin/python`. Preserve offline environment variables `HF_HOME=/workspace/hf/home`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`; clear `HF_HUB_CACHE`, `PYTHONPATH` and `PYTHONHOME` as in the reviewed previous dispatcher. The guardian waits for the already-verified predecessor absence and loads one successor model on the same H100. The source has no provider calls.
7. Before starting the local runner, the launcher verifies guardian/service/job READY records agree on the new guardian and worker identities, service configuration and exact registration hashes, and loaded direction/probe hashes. It then uses the existing reviewed atomic mirror-list helper to append only this job's `/workspace/results/agent-steering/JOB_ID` target, preserving every prior target. The local mirror is the new local-results directory's sibling `remote-results/`; service paths are not added. The exact local runner argv appears in `launch.json`; it invokes `code/run_queue.py --config GENERATED/local-config.json`. The local config hashes the full source/input map and requires the closure receipt. It checks the same renewed financial lease.

Items run in order 5 then 6 through one successor job and one loaded model. No new item starts after 05:08:01 UTC. At 05:18:01 UTC the local runner stops queued work through its remaining-time and per-generation deadlines, then ends its job while retaining the idle model. A pathological CUDA prefill that does not return cannot observe a token-level timeout; the unchanged independent financial-lease guardian remains its external bound. This explicit limitation avoids adding an automatic shutdown of an idle model. A started item may be incomplete at closeout; I retain its outputs and mark it partial. Existing generation/episode caps are shortened only by the explicit remaining closeout time.

The registration command is executable after real closure information exists; the example PID and paths above are placeholders rather than invented observed values. Registration alone opens no connection, starts no worker and spends nothing.

## Outputs and interpretation

`status.json` tracks item and completed counts. `attribution/*.json` preserves full rendered-prompt references, all raw generation fields/token IDs, actual stats, stop/censor state and the same page's finalized unsteered upload/exposure/censor outcome with its episode hash. Authorship labels remain null pending reading; no keyword proxy turns a partial answer into a correct attribution.

Item 6 retains all original per-step prompt/completion/token files, sandbox receipts and an episode `summary.json`; `episode-index.json` records exposure, emitted attempts, receiver-verified uploads and censoring separately. `FINISHED.json` states exactly how many of the ten readouts and five episodes were recorded. Job STOP ends this queue; only the session supervisor owns the GPU allocation.
