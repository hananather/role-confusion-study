# My experiment catalogue

I keep the completed experiments, engineering pilots, partial runs and prepared
follow-ups together so a reader can trace each result to its code and saved
records. This catalogue describes the September 12 snapshot. The family READMEs
and execution logs retain their original dates; their older status statements
are historical. The [review route](REVIEW.md) is the shortest path through the
agent-steering code, and [provenance](PROVENANCE.md) records the export boundary.

I am in Distill: my north star is an inspectable account of what I measured,
what changed behavior, and what remains untested. The evidence below comes from
one model family. A role-classifier score, a generated answer, a recorded action
and a receiver-confirmed transfer remain separate measurements.

## Agent behavior and steering

| Experiment | Status and evidence | Code and records |
| --- | --- | --- |
| Earlier selective-transfer study | 909 recovered episodes across two tasks: 760 original transfer episodes and 149 simpler permission episodes. Missing planned episodes remain missing; the eight-case score contrast and the all-21 Toolward comparison have separate endpoints and recovery limits. | [Report and saved-data analysis](earlier-study/README.md), [frozen experiment source](earlier-study/experiment/README.md), [data notes](earlier-study/data/README.md) |
| Local Section 3.3 agent benchmark | One initial engineering episode, a ten-episode timing cohort and a separate forgery resample. The sandbox records dummy-file receipts and retains censored trajectories. MLX, local endpoints and operational caps differ from the authors' setup. | [Harness and source contract](replication/agent-hijacking/README.md), [runs](replication/agent-hijacking/runs/), [reports](replication/agent-hijacking/reports/) |
| Local steering controls | Completed five-page Tool-minus-CoT alpha-8, random alpha-8, all-token alpha-8 and Tool-minus-CoT alpha-16 arms; the zero-dose identity arm completed three pages before stopping. Missing page exposure in the all-token arm prevents interpreting its lack of uploads as selective protection. Queued doubt, attribution and patching work must not be counted as completed. | [Local steering harness](replication/steering-agent/README.md), [directions](replication/steering-agent/directions/), [raw runs](replication/steering-agent/runs/), [analysis](replication/steering-agent/analysis/) |
| CUDA bridge and new-page steering | The fixed bridge has 35 episodes and the new-page extension has 40. Across ten forged pages, the original direction yields 6 uploads, unsteered yields 7 and the best random direction yields 2. All random controls, adverse outcomes and unchanged new-page outcomes remain in the comparison. | [Bridge allocation](replication/steering-series/2026-09-12-positive-confirmation/bridge-001/bridge-plan.json), [bridge audit](replication/steering-series/2026-09-12-positive-confirmation/bridge-001/final-audit/README.md), [new-page audit](replication/steering-series/2026-09-12-positive-confirmation/newpage-001/final-audit/README.md), [77-page input bank](replication/steering-series/2026-09-12-positive-confirmation/new-pages/) |
| Developer provenance sentence | All 15 assigned episodes completed: 4 uploads among ten forged pages and none among five benign pages. Candidate summaries on benign pages are a presence check; factual quality is unjudged. | [Combined assignment table](replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z/arm-by-page.csv), [GPU B closeout](replication/cloud/outbox/parallel-h100/gpu-b-20260912T041801Z/closeout-20260912T051046Z/README.md) |
| Tool-minus-mean(User, CoT) direction | The five-prompt probe gate passed at the original norm. Five historical-page episodes completed with no uploads; matched historical counts are 3 unsteered, 2 under the old direction and 1 under the best random control. The five new-page assignments remain unrun, so the ten-page prediction is untested. | [Direction implementation](replication/cloud/parallel_h100/tool_raising.py), [GPU B closeout](replication/cloud/outbox/parallel-h100/gpu-b-20260912T041801Z/closeout-20260912T051046Z/README.md) |
| Additional standard-injection controls | Five new-page standard controls were attempted: four resolved nonuploads and one censored outcome. Five historical standard controls assigned to GPU B remain unrun. These controls have a different payload from the forgery cohort. | [GPU A audit](replication/steering-series/2026-09-12-positive-confirmation/queue-gpu-a/final-audit/README.md), [all assignments](replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z/arm-by-page.csv) |
| Agent authorship question | Ten separate diagnostic answers follow an added question and forced-final format. The preliminary review finds no answer naming the page/tool. These are not ten more agent episodes or a measure of spontaneous source recognition. | [Answer review](replication/steering-series/2026-09-12-positive-confirmation/queue-gpu-a/item5-answer-review/README.md), [combined closeout](replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z/README.md) |
| Selective-mitigation instrumentation | Release 0 contains permission fixtures, provenance masks, a receiver ledger and a zero-dose source-local hook. The planned 54-episode development grid is not a completed experiment. | [Design and implementation boundary](replication/agent-hijacking/docs/selective-mitigation.md), [implementation](replication/agent-hijacking/agent_hijacking/), [tests](replication/agent-hijacking/tests/) |
| Role-uptake office-task pilot | All 24 cases emitted valid ACTION lines. All eight permit cells returned FOLLOW; all eight ordinary-injection/forgery draft-only cells returned DRAFT. One benign-quotation case returned FOLLOW and is recorded as unauthorized. No action executed. This small protocol does not establish a useful behavioral monitor. | [Pilot protocol](replication/role-uptake/README.md), [24-case results](replication/role-uptake/runs/pilot-24-20260911/run.json), [case transcripts](replication/role-uptake/runs/pilot-24-20260911/) |

The September 12 CUDA closeout accounts for **110 assigned episode slots: 100
recorded attempts and ten unrun assignments**, plus ten separate authorship
readouts. Three recorded attempts are censored. These counts exclude the earlier
study, local baseline and local steering cohorts. The
[frozen closeout](replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z/README.md)
retains each arm and denominator.

## Probe training, role readouts and position controls

| Experiment | Status and evidence | Code and records |
| --- | --- | --- |
| MLX feasibility and local probe development | The local model and hook were exercised; the nine-base-text development corpus has all 24 layers. The 249-text local extraction saved layers 0, 1 and 20–23. Tiny-data scores and reference-probe checks are diagnostic; they are not the final held-out comparison. | [Feasibility code](replication/feasibility/), [local probe pipeline](replication/probes/), [development metadata](replication/runs/dev-10/metadata.json), [larger-run metadata](replication/runs/full-249/metadata.json) |
| Full H100 role probes | 384 probes span 24 layers, eight role spaces and two splits. Full-data partition and coefficient audits passed. Across 96 matched reference probes, overall accuracy differs by at most 0.0791 percentage points; solver convergence remains unestablished. | [Training implementation](replication/cloud/probes/), [completion audit](replication/cloud/persistent/sessions/20260911T030050Z/full-probes-completion-audit-20260911T043828Z/README.md), [fitted arrays and metrics](replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/) |
| Gardening / Appendix E | Three tagging conditions were measured. Role-style signals persist, but the paper's reported numerical probabilities did not reproduce. The numerical discrepancy and the saved-data summary repair are preserved. | [Local projection code](replication/appendix-e/), [result account](replication/cloud/persistent/sessions/20260911T030050Z/GARDENING-RESULTS.md), [full outputs](replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-e-L12/) |
| Appendix K neutral-position controls | A local engineering smoke precedes a completed 1,330-item H100 control. All 200 neutral texts overlap the probe corpus, so this is not a held-out replication of the paper's conversation figure. The position effect depends on layer. | [Implementation and local runs](replication/appendix-k/), [completion audit](replication/cloud/persistent/sessions/20260911T030050Z/appendix-k-completion-audit-20260911T044857Z/README.md), [H100 outputs](replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-k/) |
| RH6 permission and probe readings | 1,200 prefills across 100 template–page pairs were scored. Permission shifts the relative User/Tool reading in the tested comparisons, while absolute Userness depends on the probe. No behavior was tested in this experiment. | [RH6 implementation and local smoke](replication/rh/), [results and limits](replication/cloud/persistent/sessions/20260911T030050Z/RH6-RESULTS.md), [integrity audit](replication/cloud/persistent/sessions/20260911T030050Z/rh6-integrity-audit-20260911T045805Z/README.md), [H100 outputs](replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/rh6/) |
| E4 conversation generation | A two-conversation, three-turn local generation smoke completed. Synthetic condition fixtures are retained separately. This is not a completed Figure 23 replication cohort. | [Generation and projection code](replication/e4/), [smoke metadata](replication/e4/data/smoke/generation-stats.json), [synthetic fixtures](replication/e4/data/synthetic/) |
| E12 generated-reasoning and position pilot | Two prompts were generated and replayed across six conditions, lengths 50 and 100, and probe layers 8 and 12. These are engineering measurements using the tiny development probes, not a long-context behavioral result. | [Implementation](replication/e12/run_e12b.py), [generation metadata](replication/e12/runs/smoke/generate-metadata.json), [extraction metadata](replication/e12/runs/smoke/extract-metadata.json), [saved analysis](replication/e12/runs/smoke/analysis/) |

The [H100 session summary](replication/cloud/persistent/sessions/20260911T030050Z/RESULTS-SUMMARY.md)
connects probe training, gardening, Appendix K and RH6 to their audits. Pilot
failures and repairs remain in the preserved evidence. Counts from these
readout experiments are not additional behavioral episodes.

## Chat baseline and prepared follow-ups

| Experiment | Status and evidence | Code and records |
| --- | --- | --- |
| StrongREJECT unsteered chat baseline | All 313 unique prompts have saved trajectories: 309 final answers and four token-capped outputs. Harmfulness judging is unfinished. No forged or steered comparison was collected, and no prompt has a matched second arm in the saved index. | [Collection report](replication/cloud/outbox/chat-steering/baseline-retry-20260912T004400Z/RESULTS.md), [raw generations](replication/cloud/outbox/chat-steering/baseline-retry-20260912T004400Z/results/generations.jsonl), [index](replication/cloud/outbox/chat-steering/baseline-retry-20260912T004400Z/review/index/README.md), [runtime](replication/cloud/chat_steering/) |
| Chat forgery and steering preparation | The generic forgery corpus was not completed. An option based on eight released author examples was prepared but not executed. Local E8 steering code and frozen chat configurations are preserved as preparation. | [Chat preparation](replication/cloud/chat-steering-prep/20260912/), [example scope](replication/cloud/chat-steering-prep/20260912/examples/SCOPE.md), [local steering implementation](replication/e8/steer.py) |
| E9 propagated-vector comparison | Deferred. There are 200 source candidates, vector assets and synthetic-tested analysis, but no completed E9 model measurements or eligible generated cohort. Offline edits and propagated interventions remain distinct experiments. | [Status and source contract](replication/e9/README.md), [prepared inputs](replication/e9/prepared-source/), [assets](replication/e9/assets/), [future requirements](replication/e9/BACKLOG.md) |
| Cloud experimental battery | Source is prepared for patching, doubt continuations, wrapper changes, attribution, steering controls, geometry, deliberation/adoption vectors, destyling, declaration vectors and text baselines. Its planned cells are not a completed battery; the independently executed CUDA queue is accounted for above. | [Battery source and historical plan](replication/cloud-steering/README.md) |

Other candidate names in the [historical experiment registry](replication/EXPERIMENTS.md),
such as Userness-quintile evaluation and larger confirmation cohorts, are plans
unless this catalogue links a completed run. A prediction, command or prepared
input is not evidence that the experiment ran.

## Measured examples and figure diagnostics

| Experiment | Status and evidence | Code and records |
| --- | --- | --- |
| Selected case-002 agent comparison | Four additional local episodes compare ordinary injection, forged reasoning, steering and one authored destyled variant. Both unsteered forgery variants upload; the steered and ordinary-injection arms do not. This is a historically selected favorable case with no random arm. The destyled variant also changes phrasing and length. These episodes are separate from the CUDA closeout. | [Input freeze, outcomes and raw trajectories](replication/chart-library/data/figure8-steering/local-case002-v1/README.md), [figure implementation](replication/chart-library/render_forgery_figure.py) |
| Authored conversation illustrations | Twelve authored six-passage conversations were measured in three formatting conditions each. No continuation was generated in this set. The chosen example was selected for legibility after viewing figures, not by representative sampling. | [Candidate set and measurement boundary](replication/chart-library/data/mats-six-passage/README.md), [all candidate data](replication/chart-library/data/mats-six-passage/), [renderer](replication/chart-library/render_mats_six_passage.py) |
| Generated Hanan–MATS illustration | Two illustration attempts are retained. In the second, the local model generated two analysis/reply pairs and the full exchange was replayed in three formats. The selected display matches 578 content tokens per format. The first attempt and all forwarded scores remain available; this selected example is not an evaluation cohort. | [Second attempt and audit](replication/chart-library/data/mats-hanan-dialogue-v2/README.md), [first attempt](replication/chart-library/data/mats-hanan-dialogue/), [canonical display](replication/chart-library/canonical/mats-dialogue-v1/README.md) |
| Offline classifier offsets | Dose curves edit saved activations independently before applying the classifier. They illustrate classifier response; they are not propagated model interventions or new behavioral episodes. | [Figure library](replication/chart-library/README.md), [offset data](replication/chart-library/data/offset-dose-response.csv), [provenance](replication/chart-library/data/render-provenance.json) |

## Reading and reproducing the evidence

I distinguish my implementations from the pinned authors' repository and the
independent replication references. Historical scripts sometimes expect the
original workspace layout or external model assets. The
[provenance notes](PROVENANCE.md) identify those dependencies and the bulk-array
archive; inclusion of a script does not establish a fresh-install model rerun.

I preserve frozen prompts, fixtures, raw generations and source packets as
research data. The portable verification commands in [REPRODUCING.md](REPRODUCING.md)
check saved records without model inference. New summaries and navigation are
packaging; they do not alter the underlying experimental evidence.
