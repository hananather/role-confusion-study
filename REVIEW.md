# My code-review route

I want this review to establish whether the code implements the recorded
intervention and whether the saved outcomes support the tables. This is
Distill: my north star is a short path from an experimental choice to its raw
evidence, following the criteria of clarity, baselines and checking agent output.

## Follow one complete episode

| Step | Code or evidence | What I check |
| --- | --- | --- |
| 1. Prompt and tool protocol | [Frozen protocol](replication/cloud/agent_steering/frozen_harness/protocol.py) | Which messages the agent sees, how tool responses are represented and where generation begins |
| 2. Agent loop and receipts | [Episode loop](replication/cloud/agent_steering/episode.py), [sandbox](replication/cloud/agent_steering/frozen_harness/sandbox.py) | Full prompts, generated tokens, tool results, page exposure, receiver evidence and stopping conditions |
| 3. Activation intervention | [CUDA backend](replication/cloud/agent_steering/hf_backend.py) | Tool-content token masks, zero-based block 11 edits, dose scaling, no direct edits to generated tokens and downstream layer-12 probe recording |
| 4. Later direction | [Tool-raising direction](replication/cloud/parallel_h100/tool_raising.py) | Tool-minus-mean(User, CoT), fixed norm, preservation of original arrays and the five-prompt gate |
| 5. Allocation and accounting | [Bridge runner](replication/cloud/agent_steering/runner.py), [new-page runner](replication/cloud/agent_steering/new_page_runner.py), [closeout](replication/cloud/parallel_h100/closeout.py) | Frozen identities/seeds, receiver-verified uploads, censoring, unrun cells and summary-presence heuristics |
| 6. Recorded comparison | [All 110 assignments](replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z/arm-by-page.csv) | Every arm and every assigned page, including adverse and unresolved outcomes |

The first page in the registered bridge execution order is
[001-forgery](replication/cloud/outbox/agent-steering/agent-bridge-20260912T022500Z/local/episodes/001-forgery/).
Its seven arm directories make a complete matched review block. Each includes
`episode.json`, full prompts, raw completions, token IDs, tool evidence and
receiver receipts. I preserve the adverse role-steering result on this page.
The [bridge plan](replication/steering-series/2026-09-12-positive-confirmation/bridge-001/bridge-plan.json)
records the complete allocation and the preselected trajectory-review indices.

## Locate the exact implementation

The directly readable modules above are copied from the archived project code.
The frozen source packets identify the versions actually staged for each run:

- [Historical CUDA bridge](replication/cloud/outbox/agent-steering/agent-bridge-20260912T022500Z/stage/)
- [New-page CUDA batch](replication/cloud/outbox/agent-steering/agent-newpages-20260912T025000Z/stage/)
- [Authorship readouts and standard controls](replication/steering-series/2026-09-12-positive-confirmation/queue-gpu-a/prepared/)
- [Developer sentence and Tool-raising queue](replication/cloud/outbox/parallel-h100/gpu-b-20260912T041801Z/source/)

I preserve historical source snapshots unchanged. The source maps and frozen
launch manifests identify adaptations; a current module's presence alone does
not establish that its exact bytes produced a historical result.

## Keep the measurements separate

The receiver determines whether dummy-file bytes were uploaded. An emitted
command, absent receipt or unfinished episode is a different observation.
The probe is a measurement of classifier output, and the engineering gate is
a manipulation check. Neither alone demonstrates a behavioral mechanism.

The ten authorship answers follow an added user question and forced-final
response format. Their preliminary assistant labels do not measure spontaneous
source recognition. “Candidate summary” is a presence heuristic; factual
adequacy remains unjudged. These distinctions are retained in the
[frozen result report](replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z/README.md).

## Run the model-free checks

```bash
python3 scripts/verify_results.py
python3 scripts/verify_export.py
python3 -m unittest replication.cloud.agent_steering.test_hf_backend replication.cloud.parallel_h100.test_tool_raising replication.cloud.parallel_h100.test_closeout -v
```

The unit tests require NumPy and use fake inference. The two verifiers need
only the standard library. The [verification record](VERIFICATION.md) states
what passed and what was not tested. Launch scripts and historical README
commands are retained for inspection; a complete GPU rerun also requires model
assets, a reviewed sandbox and current execution configuration.
