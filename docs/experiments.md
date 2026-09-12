# Experiments

[README.md](../README.md) · [reproducing.md](reproducing.md) · [provenance.md](provenance.md)

## Agent runs

| Experiment | Status | Files |
| --- | --- | --- |
| Earlier transfer and permission tasks | Partial recovery | [Source](../earlier-study/experiment/README.md), [data](../earlier-study/data/README.md), [analysis](../earlier-study/README.md) |
| Local Section 3.3 agent benchmark | Pilot complete; censored runs retained | [Harness](../replication/agent-hijacking/README.md), [runs](../replication/agent-hijacking/runs/), [records](../replication/agent-hijacking/reports/) |
| Local steering controls | Partial | [Harness](../replication/steering-agent/README.md), [directions](../replication/steering-agent/directions/), [runs](../replication/steering-agent/runs/) |
| CUDA bridge and new-page steering | Completed; censored runs retained | [Allocation](../replication/steering-series/2026-09-12-positive-confirmation/bridge-001/bridge-plan.json), [bridge audit](../replication/steering-series/2026-09-12-positive-confirmation/bridge-001/final-audit/README.md), [new-page audit](../replication/steering-series/2026-09-12-positive-confirmation/newpage-001/final-audit/README.md), [inputs](../replication/steering-series/2026-09-12-positive-confirmation/new-pages/) |
| Developer provenance sentence | Completed | [Assignments](../replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z/arm-by-page.csv), [run records](../replication/cloud/outbox/parallel-h100/gpu-b-20260912T041801Z/closeout-20260912T051046Z/README.md) |
| Tool−mean(User, CoT) | Partial; new pages unrun | [Implementation](../replication/cloud/parallel_h100/tool_raising.py), [run records](../replication/cloud/outbox/parallel-h100/gpu-b-20260912T041801Z/closeout-20260912T051046Z/README.md) |
| Additional standard-injection controls | Partial; censored and unrun assignments | [Audit](../replication/steering-series/2026-09-12-positive-confirmation/queue-gpu-a/final-audit/README.md), [assignments](../replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z/arm-by-page.csv) |
| Agent authorship question | Diagnostic complete | [Answers](../replication/steering-series/2026-09-12-positive-confirmation/queue-gpu-a/item5-answer-review/README.md), [records](../replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z/README.md) |
| Selective-mitigation instrumentation | Prepared | [Design](../replication/agent-hijacking/docs/selective-mitigation.md), [implementation](../replication/agent-hijacking/agent_hijacking/), [tests](../replication/agent-hijacking/tests/) |
| Role-uptake office-task pilot | Pilot complete; actions unexecuted | [Protocol](../replication/role-uptake/README.md), [results](../replication/role-uptake/runs/pilot-24-20260911/run.json), [transcripts](../replication/role-uptake/runs/pilot-24-20260911/) |

## Probes and role measurements

| Experiment | Status | Files |
| --- | --- | --- |
| MLX feasibility and local probe development | Pilot; partial extraction | [Feasibility](../replication/feasibility/), [probe pipeline](../replication/probes/), [development metadata](../replication/runs/dev-10/metadata.json), [larger-run metadata](../replication/runs/full-249/metadata.json) |
| H100 role probes | Completed; convergence unestablished | [Implementation](../replication/cloud/probes/), [audit](../replication/cloud/persistent/sessions/20260911T030050Z/full-probes-completion-audit-20260911T043828Z/README.md), [arrays and metrics](../replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/) |
| Gardening / Appendix E | Measurements complete | [Code](../replication/appendix-e/), [records](../replication/cloud/persistent/sessions/20260911T030050Z/GARDENING-RESULTS.md), [outputs](../replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-e-L12/) |
| Appendix K neutral-position controls | Measurements complete | [Code and local runs](../replication/appendix-k/), [audit](../replication/cloud/persistent/sessions/20260911T030050Z/appendix-k-completion-audit-20260911T044857Z/README.md), [outputs](../replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-k/) |
| RH6 permission and probe readings | Measurements complete; no behavioral test | [Implementation](../replication/rh/), [records](../replication/cloud/persistent/sessions/20260911T030050Z/RH6-RESULTS.md), [audit](../replication/cloud/persistent/sessions/20260911T030050Z/rh6-integrity-audit-20260911T045805Z/README.md), [outputs](../replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/rh6/) |
| E4 conversation generation | Pilot complete | [Code](../replication/e4/), [metadata](../replication/e4/data/smoke/generation-stats.json), [synthetic fixtures](../replication/e4/data/synthetic/) |
| E12 generated-reasoning and position pilot | Pilot complete | [Implementation](../replication/e12/run_e12b.py), [generation metadata](../replication/e12/runs/smoke/generate-metadata.json), [extraction metadata](../replication/e12/runs/smoke/extract-metadata.json), [analysis](../replication/e12/runs/smoke/analysis/) |

## Chat runs and prepared experiments

| Experiment | Status | Files |
| --- | --- | --- |
| StrongREJECT unsteered chat baseline | Collection complete; judging unfinished | [Collection record](../replication/cloud/outbox/chat-steering/baseline-retry-20260912T004400Z/RESULTS.md), [generations](../replication/cloud/outbox/chat-steering/baseline-retry-20260912T004400Z/results/generations.jsonl), [index](../replication/cloud/outbox/chat-steering/baseline-retry-20260912T004400Z/review/index/README.md), [runtime](../replication/cloud/chat_steering/) |
| Chat forgery and steering | Prepared | [Inputs](../replication/cloud/chat-steering-prep/20260912/), [example scope](../replication/cloud/chat-steering-prep/20260912/examples/SCOPE.md), [E8 steering code](../replication/e8/steer.py) |
| E9 propagated-vector comparison | Deferred; no model measurements | [Source contract](../replication/e9/README.md), [inputs](../replication/e9/prepared-source/), [assets](../replication/e9/assets/), [requirements](../replication/e9/BACKLOG.md) |
| Cloud experimental battery | Prepared; individual runs listed above | [Source and plan](../replication/cloud-steering/README.md) |
| Other registered candidates | Planned | [Historical registry](../replication/EXPERIMENTS.md) |

## Figures and measured examples

| Experiment | Status | Files |
| --- | --- | --- |
| Selected case-002 agent comparison | Completed illustration | [Inputs and trajectories](../replication/chart-library/data/figure8-steering/local-case002-v1/README.md), [renderer](../replication/chart-library/render_forgery_figure.py) |
| Authored conversation illustrations | Measurements complete; no generated continuation | [Inputs and measurements](../replication/chart-library/data/mats-six-passage/README.md), [data](../replication/chart-library/data/mats-six-passage/), [renderer](../replication/chart-library/render_mats_six_passage.py) |
| Generated Hanan–MATS illustration | Completed illustration | [Second attempt](../replication/chart-library/data/mats-hanan-dialogue-v2/README.md), [first attempt](../replication/chart-library/data/mats-hanan-dialogue/), [display](../replication/chart-library/canonical/mats-dialogue-v1/README.md) |
| Offline classifier offsets | Saved-activation analysis | [Figure library](../replication/chart-library/README.md), [data](../replication/chart-library/data/offset-dose-response.csv), [provenance](../replication/chart-library/data/render-provenance.json) |

## Introduction and related work

| Source | Files |
| --- | --- |
| Prompt Injection as Role Confusion | [Paper](../MAIN_PAPER.md), [upstream repository](../prompt-injection-as-role-confusion/) |
| Earlier reference implementations | [Reference projects](../replication/reference/) |
