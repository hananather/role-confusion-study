# Approach overview

[README.md](../README.md) · [experiments.md](experiments.md) · [reproducing.md](reproducing.md)

## Proxy task and attacker control

| File | Contents |
| --- | --- |
| [protocol.py](../replication/cloud/agent_steering/frozen_harness/protocol.py) | Prompts and tool protocol |
| [episode.py](../replication/cloud/agent_steering/episode.py) | Agent loop and recorded actions |
| [sandbox.py](../replication/cloud/agent_steering/frozen_harness/sandbox.py) | Isolated environment and receiver |
| [001-forgery](../replication/cloud/outbox/agent-steering/agent-bridge-20260912T022500Z/local/episodes/001-forgery/) | Matched runs, prompts, completions and receiver records |

## Technical setup: steering and role probes

| File | Contents |
| --- | --- |
| [hf_backend.py](../replication/cloud/agent_steering/hf_backend.py) | Activation edits, token masks and role measurements |
| [directions](../replication/steering-agent/directions/) | Class means and steering directions |

## Matched comparisons

| File | Contents |
| --- | --- |
| [bridge-plan.json](../replication/steering-series/2026-09-12-positive-confirmation/bridge-001/bridge-plan.json) | Assignment and sampling plan |
| [runner.py](../replication/cloud/agent_steering/runner.py) | Bridge runner |
| [new_page_runner.py](../replication/cloud/agent_steering/new_page_runner.py) | New-page runner |
| [arm-by-page.csv](../replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z/arm-by-page.csv) | Recorded, censored and unrun assignments |

| Run | Frozen source |
| --- | --- |
| CUDA bridge | [stage](../replication/cloud/outbox/agent-steering/agent-bridge-20260912T022500Z/stage/) |
| New-page batch | [stage](../replication/cloud/outbox/agent-steering/agent-newpages-20260912T025000Z/stage/) |
| Authorship question and standard controls | [prepared](../replication/steering-series/2026-09-12-positive-confirmation/queue-gpu-a/prepared/) |
| Developer sentence and revised direction | [source](../replication/cloud/outbox/parallel-h100/gpu-b-20260912T041801Z/source/) |

## Refining the direction

| File | Contents |
| --- | --- |
| [tool_raising.py](../replication/cloud/parallel_h100/tool_raising.py) | Tool−mean(User, CoT) and the probe check |
| [closeout records](../replication/cloud/outbox/parallel-h100/gpu-b-20260912T041801Z/closeout-20260912T051046Z/README.md) | Completed and unrun assignments |

## Measuring useful behavior

| File | Contents |
| --- | --- |
| [closeout.py](../replication/cloud/parallel_h100/closeout.py) | Upload records, summary-presence labels and censored runs |
| [saved-results verifier](../scripts/verify_results.py) | Episode and table checks |
| [export verifier](../scripts/verify_export.py) | File hashes |
| [verification.md](verification.md) | Recorded checks |
