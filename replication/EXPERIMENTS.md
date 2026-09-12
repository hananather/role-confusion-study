# EXPERIMENTS.md — ranked registry (DRAFT v1, 2026-09-10)

Status: draft from the planning agent on the frozen input `PLAN-INPUT-2026-09-10.md`, independently corroborated by the lens review (`audits/2026-09-10-lens-review-plan-input.md`). Awaiting Hanan's decisions on (1) the deadline, (2) gates-first reorder, (3) the Codex judge and forgery generation, (4) GPU one-shots and per-launch cap, (5) E4 scale and ToxicChat token, (6) which steering direction(s). Sample-size columns will be filled from `audits/2026-09-10-noise-and-sample-size-postmortem.md` when it lands. Nothing here authorizes a run; every launch needs sign-off on the exact command and cost (`EXECUTION.md`).

Rules: a running experiment is never interrupted for a new idea; it is stopped only when a gate proves it void. Paper scale applies to claim-producing experiments; gates are cheap checks and are exempt. Predictions are recorded before each launch.

## Question

On gpt-oss-20b, do the authors' role probes replicate under their exact recipe on a MacBook, and does steering along a role direction reduce CoT-forgery attack success more than a norm-matched random direction and a one-sentence text warning, while the probe reading moves?

## Hypotheses

- Main (paper): role confusion is causal; pushing forged text Userward or away from CoT reduces ASR.
- Alternative (Ye's concession, Kevin Zhang's repo): probe directions are downstream readouts; steering moves the reading, not the behavior, because style is upstream.
- Reversal: role steering matches random and loses to the text warning at every alpha that keeps utility. Then "probes yield interventions" is false for this construction.

## Registry

Type: R = pure replication (credibility), C = contribution. Place: L = local MacBook, G = GPU one-shot after a 5-item local end-to-end run. Hours are unattended compute. A = needs sign-off for a new download, spend, or the judge. n / seeds / MDE columns: to be filled from the noise post-mortem.

| Rank | Id | Type | Stage | Serves | De-risks / tests | Prediction (prior, Hanan overwrites) | Stop rule | Place, hours | Depends on | Changes plan if | n / seeds / temp / MDE |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | E2-lite: apply the 96 shipped H200 probes to our MLX activations | R | Explore | MLX numerics and hook site match the authors' | Fidelity without the 13 GB download | Their probes within 5 points of their own held-out accuracy at layers 10, 12, 14 | Below chance or over 20 points off at every layer: stop downstream, diagnose site or dtype | L, 1 h (pickle may need coef extraction) | E1 layers 10, 12, 14 on disk | Failure: full E2 with transformers bf16 (13 GB, A) or move to GPU | tbd |
| 2 | E1: full 249-text probes, all layers, 8 role spaces, both splits | R | Explore | Exact probe replication | Pipeline correctness; the split leak | Authors' split uat at layer 12 above 0.90; base split peaks 8 to 16 above 0.85; tool hardest | uat below 0.7 at every layer under both splits: pipeline wrong, fix before E3 | L, running, about 5 h | none | Peak layer sets the E8 steer-layer sweep | n/a |
| 3 | E3: Appendix E gardening, 3 conditions, suca, layer 12 and all layers | R | Explore gate | Style-over-tags exists here | The premise of everything downstream | Correct tags: CoT ~85, user ~74, assistant ~96; No Tags within 10 points; user-tagged CoT stays high | No Tags near chance: style-over-tags does not replicate; stop and write up the failed replication | L, minutes | E1 suca probes | Failure removes the premise for E8, E9 | one example: gate, not claim |
| 4 | E8-pilot: steering harness with the refusal direction only, 20 prompts, hand-labeled | C | Explore | Fail fast on the likeliest failure | Whether generation-time steering moves behavior at all here | Refusal direction flips at least half of 20 forgery successes to refusals with utility intact | No movement at 3 alphas x 3 layers: harness broken or setting resists; stop E8, E9 | L, 2 to 3 h + 1 h labeling | E7-lite subset; refusal contrast set | Failure collapses contribution to replication + documented negative | 20, greedy, hand labels |
| 5 | E7-lite: baseline and forgery ASR on 100 StrongREJECT prompts, forgeries via Codex | R | Explore gate | The attack exists here | Local forgeries reproduce near-zero baseline and ~60% forgery ASR | Baseline under 10%; forgery 40 to 70% | Forgery ASR under 20%: forgeries differ from Gemini's; try the qualified prompt once, then stop and log | L, 2 to 4 h; A | StrongREJECT csv; Codex update | Low forgery ASR: no headroom for E8 | tbd |
| 6 | E8-full: role directions (declaration-based mean difference; probe-weight vector), 3 random norm-matched, text warning, alpha sweep at 3 layers, 50 to 100 prompts | C | Understand | Does the role direction beat random and prompting? | Main vs alternative hypothesis | Probe reading moves at every alpha; ASR moves no more than random; text warning wins (Hanan may predict the opposite) | Stop once the two best alphas per layer are done; utility below 80% invalidates that alpha | L 6 to 10 h or G one-shot; A | E8-pilot pass, E7-lite, E1 directions | Positive: E10 worth running; negative: E9 explains | tbd |
| 7 | E4: real conversations (OASST first, ToxicChat if token), local regeneration, Figure 23 uat all layers, Table 3 | R | Explore | Layer dynamics replication | Injection tracks Baseline across depth | Injection within 5 points of Baseline at every layer; No Tags between | Baseline argmax at layer 12 under 0.7: set or prefix wrong; fix before E9 | L 4 to 8 h (30 then 200); A (token) | E1 uat probes | Needed only for E9 | 30 vs 200 tbd |
| 8 | E9: Figure 23 propagation overlays — deferred September 11 | C, unclaimed | Explore preparation → Understand | Does a direction-specific offset survive depth? | Historical and classifier-vector families, shown separately | Freeze prediction and paired endpoint before the future batch | No execution now; no small-cohort fallback without my decision; no behavioral claim from readouts alone | Future batch only; budget unset | Eligible E4 cohort, compatible vectors, paired controls, sample justification | Compare propagated changes with zero, opposite sign, and random controls | At least 200 eligible independent conversations; planning floor, not adequacy proof; [backlog](e9/BACKLOG.md) |
| 9 | E6: Appendix B.2 agent hijacking, 100 pages x 2 variants | R | Understand | Headline agent ASR | Agent env matches authors' | Within 15 points of the paper | Under 10% forgery ASR: env diverges; log and stop | G one-shot preferred; L 8 to 15 h; A | env (task 5), judge | Only if the write-up claims anything about the agent setting | tbd |
| 10 | E5-lite: Userness on all 1,000 injected HTML files; episodes only for top and bottom quintiles (100 each) | R | Understand | Curve endpoints (2% vs 70%) | Userness predicts ASR here, cheaply | Endpoints separate by 30+ points | No separation: correlation does not replicate on this stack | Userness L 2 h; episodes G; A | env, judge, templates, pages | Cut first if short | tbd |
| 11 | E10: rerun of Hanan's agent steering with the calibration direction, alpha sweep, judge outcome | C | Understand | Adjudicate Zhang, Lee, Park | Declaration direction in the agent task | Conditional on E8 positive | Run only if E8 shows a role-direction effect above random | G one-shot; A | env, judge, E8 | Skip otherwise | tbd |
| 12 | E11: Appendix K position analysis | R | Distill | Completeness | none | n/a | below the cut line | L, small | E1 | none | n/a |

Recommended order: E2-lite, E1 (finish), E3, E8-pilot, E7-lite, E8-full, E4 in the background from step 4, E9, then E6, E5-lite, E10, E11. Cut line if short: keep E1, E2-lite, E3, E7-lite, E8 (pilot + one reduced sweep), E9 at one figure; drop E5, E6, E10, E11.

The lens review differs in one emphasis: it makes E10 (the agent-task adjudication) the primary claim and treats the StrongREJECT setting as a known null worth only the positive control. The planning agent keeps E8 in the chat setting as the main test because it is cheaper and fully local. Decision for Hanan.

## Gates

- Replication gate: E3 No-Tags result and E7-lite forgery ASR must both hold before any steering claim.
- Raw data: hand-label all 20 pilot transcripts; read 30 random E8 transcripts (10 unsteered, 10 role-steered, 10 random-steered); put 5 random ones after the executive summary.
- Verification: Hanan re-derives the E3 means from saved per-token probabilities, one E8 ASR cell from raw judge labels, one E4 layer accuracy; log each in LOG.md.

## Stop rules

- E2-lite or E3 fails: stop above E1; write up the failed replication with the diagnosis.
- E8-pilot fails at 3 alphas and 3 layers: drop E8-full, E9, E10; write up replication plus the negative and the text-warning baseline.
- Any GPU launch: only a script that ran end to end locally on 5 items, with the per-launch cap Hanan sets; watchdog per `GPU-PLAYBOOK.md`.
- A total change of direction resets the timer; a scope cut does not.

## Time budget proposal (Hanan owns the timer)

| Stage | Hours |
|---|---|
| Explore and de-risk (E2-lite, E1 checks, E3, E8-pilot) | 5 |
| Understand (E7-lite, E8-full, E4 monitoring, E9) | 6 |
| Sanity-checking agent output and reading transcripts | 3 |
| Distillation | 3 (+2 for the executive summary) |

## Single biggest risk

The contribution rests on a steering harness whose positive control has not been shown to work in this setting, and five public attempts failed here (one including a positive control). If it fails, the program is replication only. The pilot exists to learn this in 3 hours instead of 20.

## Executive summary templates

If it works: On gpt-oss-20b we replicated the authors' role probes with their exact recipe on a MacBook and reproduced Appendix E and Figure 23 within [x] points. Steering along a declaration-based Userward direction at layer [L] cut CoT-forgery ASR from [a] to [b] on 100 StrongREJECT prompts, beating three norm-matched random directions [c] and a one-sentence text warning [d], with harmless-task compliance unchanged. The layer-dynamics overlay shows the offset [persists / decays] with depth.

If it does not: We replicated the probes and the style-over-tags finding, but role-direction steering did not move attack success beyond a random direction, even as the probe reading moved to [x]. A one-sentence text warning did better. The overlay shows the injected offset [decays by layer L / persists], supporting Ye's reading that probe directions are downstream of the cause. This adjudicates the conflicting reports on one model with matched controls.

## Addendum: sample size, noise, and batches (planning agent, 2026-09-10)

Power figures: two-sided 5% test, 80% power, worst case near 50% ASR. Unpaired minimum detectable effect (MDE) in points: n=20: 44, n=30: 36, n=50: 28, n=100: 20, n=200: 14, n=313: 11, n=1000: 6. Paired by prompt (McNemar, about 30% discordant pairs): n=100: 15, n=200: 11, n=313: 9. Approximate; Hanan re-derives before use.

Authors' sampling, from their notebooks: StrongREJECT chat on gpt-oss is greedy (temperature 0, one generation per prompt; the OpenRouter notebook subsamples 200 prompts with seed 123 although the paper says 313). Appendix B.2 and Appendix J agent episodes sample at temperature 1.0, one episode per draw, one global seed. Figure 23 conversations regenerate at temperature 1.0, one trajectory each.

| Id | n per arm | Seeds / trajectories per unit | Temperature | MDE at 80% power | Reporting |
|---|---|---|---|---|---|
| E1 | 249 base texts; held-out 25 texts under a 90/10 split (about 125 prompts, clustered by text) | none | n/a | 25 held-out texts cannot separate per-text gaps under about 15 points. Use 5-fold CV grouped by base text so all 249 texts score; CI width about 3 points | Accuracy per layer and role space with cluster-bootstrap 95% CI over base texts; both splits side by side; token-level numbers never without the text-level CI |
| E2-lite | same activations; 12 shipped layers x 8 role spaces | none | n/a | As E1; agreement = their accuracy on our activations minus their reported accuracy, CI over texts | Difference table with cluster-bootstrap CI; pass if the CI includes 0 or the gap is under 5 points |
| E3 | 1 conversation, 3 conditions | none | n/a | Not a test; gate only | Per-role mean and per-token distribution per condition; quantitative version is E4 |
| E4 | 200 conversations (paper), 30 as smoke test, 3 conditions paired | 1 trajectory at temperature 1.0 (authors), seed logged | 1.0 | Paired difference about 11 points at n=200, about 30 at n=30: 30 is a smoke test, not a result | Per-layer mean true-role probability and argmax accuracy with cluster-bootstrap CI over conversations; paired differences per layer |
| E5-lite | 1,000 Userness scores; episodes 100 per quintile endpoint, 200 if the full curve is wanted | 1 per draw at 1.0 as authors; 3 seeds only if within MDE | 1.0 | Endpoints: 20 points at 100, 14 at 200; paper gap 2 vs 70 | Wilson CIs per quintile; logistic regression attempt ~ role + Userness with cluster-robust SE by template; outcome attempt-or-success, stated |
| E6 | 100 pages x 2 variants, paired by page | 1 per page at 1.0 as authors; 3 seeds per page if within MDE | 1.0 | Paired about 15 points at n=100; paper gap about 30 | Wilson CI per variant; McNemar on pages; with seeds, cluster-robust SE by page |
| E7-lite / E7 | 100 prompts (lite), 313 (full), variants paired | 1 greedy generation, as authors | 0 | Paired 15 at 100, 9 at 313; baseline-vs-forgery gap about 50 | Wilson CI per variant; McNemar; judge agreement with hand labels on 30 random items reported |
| E8-pilot | 20 prompts, 9 arms (refusal x 3 alphas x 3 layers) plus unsteered | greedy | 0 | 44 unpaired, about 35 paired: only a large positive-control effect counts | Counts and Wilson CI; hand labels; go/no-go at 40 points of movement |
| E8-selection | 100 prompts, about 17 arms (2 role directions x 3 alphas x 2 layers, 3 random, warning, unsteered, refusal) | greedy | 0 | 15 paired; selection rule pre-registered: largest ASR drop with harmless compliance at or above 80% on 50 harmless prompts | Wilson CI per arm; paired differences vs unsteered and vs best random; selection logged before Batch 3 |
| E8-confirmation | 313 prompts, 8 arms (unsteered, warning, selected role arm, sign-flipped, 3 random, refusal); the 213 prompts unused in selection are the clean set | greedy | 0 | 9 paired on 313, 11 on 213; effects under 10 points not claimable | Paired difference role minus random and role minus warning with CI; report the 213 clean set first; no claim for 0.01 < p < 0.05 |
| E9 — deferred | At least 200 eligible independent conversations after generation and exclusions, paired across all planned arms; balanced source target | Fixed text for forward comparison; multiple turns remain one conversation cluster | Generation settings to freeze before future batch | Prespecify meaningful effect and paired precision/power; 200 is a floor, not proof that a 0.05 shift is detectable | All eligible original-role tokens → conversation means → equal-conversation aggregate; paired bootstrap; separately named accuracy, probability, and Toolness panels; [backlog](e9/BACKLOG.md) |
| E10 | 200 episodes per arm minimum (50 per alpha gives MDE 28) | 1 per case at 1.0; 3 seeds if within MDE | 1.0 | 14 points at 200; paired by case about 11 | Judge label outcome; Wilson CI; McNemar by case; cluster-robust SE by template |

Rules: every proportion gets a Wilson 95% CI; arms sharing prompts, pages, or conversations are analyzed paired; multiple seeds or draws per template use cluster-robust SEs; a result inside its MDE is reported as "not separable from noise at this n", never as a trend; judge agreement with hand labels is a reported number.

### Batches

| Batch | Contents | Place | Compute | Must show before the next batch |
|---|---|---|---|---|
| 0 (running) | E1 full probes; E2-lite on the layers on disk | L | about 5 h | uat at layer 12 above 0.85 under at least one split with CI; shipped probes agree within 5 points on our activations |
| 1 gates | E3; E7-lite (100 prompts, baseline and forgery); E8-pilot (positive control) | L | about 5 h plus 1 h labeling | E3 No-Tags reconstructs roles; forgery ASR at or above 40% with baseline under 10%; refusal steering moves ASR by at least 40 points on 20 prompts. Any failure stops the steering track |
| 2 selection | E8-selection on 100 prompts; E4 generation in the background (30 then 200) | L | E8 about 12 h; E4 4 to 8 h | Pre-registered selection applied; a role arm beats best random by more than 15 points with utility intact, else Batch 3 shrinks to controls plus E9 |
| 3 confirmation | E8-confirmation on 313 prompts; E9 overlay from E4 activations | G one-shot for E8 if local exceeds 15 h; E9 L | E8 about 17 h local or a few GPU hours; E9 2 h | Paired CI on the 213 clean prompts excludes zero and the random arm, or a documented negative with the warning baseline; E9 figure with random line |
| 4 optional | E6; E5-lite; E10 only if Batch 3 was positive | G one-shot | per Hanan's cap | Only if the write-up needs agent-setting claims |

Distillation starts after Batch 3 regardless of Batch 4.

## E9: deferred future batch (my decision, 2026-09-11)

I deferred E9; this status supersedes earlier E9 scheduling and small-figure fallback language above. I retain [the prepared source inputs, vector assets, adapter, and synthetic-tested analysis](e9/README.md). I have no completed E9 experiment results, and I authorize no new compute through this registry update. Both selected vector families remain separate, and the future comparison uses actual downstream propagation rather than independent offline edits at each layer.

I require at least 200 **eligible independent conversations after generation and filtering**, with a balanced source target, attrition expansion, and a prespecified paired effect-size/precision/power justification. That is a planning floor, not an adequacy claim. The paper says 200 conversations; frozen NB02 samples at most 30 after filtering, with “100 for full test” in a comment. I retain the ambiguity and use all eligible original-role content tokens within each conversation, then equal conversation weighting. I keep probability and argmax accuracy separately named. The prepared 200 source candidates and synthetic fixtures are neither the final eligible cohort nor experiment results. [My E9 backlog](e9/BACKLOG.md) defines the remaining work and the requirement for an explicit future-batch decision.

## E12: progressive role confusion over position (candidate, decision card 2026-09-10)

Sources: `audits/2026-09-10-long-context-design-advocate.md`, `audits/2026-09-10-long-context-design-critic.md`, `audits/2026-09-10-long-context-literature.md`, `01-sources/papers/reading-notes/2026-09-10-kramar-2026-production-ready-probes-gemini.md`.

**Interpretation.** The direction is open and unclaimed: no published work reads a role probe per token across long or many-shot contexts. But the premise (Appendix H "progressive conviction") is unproven and plausibly part artifact: Figure 25 averages by within-segment index with a survivorship filter (n >= 50 prompts) and an EWMA; the shipped notebook truncates at 50/100 tokens, not the 100/200 in footnote 22; the paper's own genuine-CoT and destyled curves also climb; and probe accuracy on neutral text already drifts with position in two reruns of the authors' notebook (layer 12: 0.87 at positions 0-49 to 0.66 at 450-499; layer 16 drifts less). Two mechanisms fit the visible 20-to-30-token climb: role inference from style that saturates and resets at boundaries (default), or cumulative state that carries across boundaries (the authors' conjecture). The DeepMind probe paper is about misuse detection and pooled aggregation, not injection; use it only as motivation for the aggregation question, not as a premise.

**Recommendation.** Adopt as a gate, not a claim. Two steps before any many-shot compute:
1. E12a, zero cost: from the full-249 activations, plot mean probability of every role by token position on held-out neutral text, all layers on disk (10, 12, 14 kept; others while present). This is the position-only null curve the paper lacks for CoTness and Assistantness.
2. E12b, about 10 to 25 minutes of extraction: 50 prompts x conditions x lengths (25 to 800 tokens): the model's own genuine CoT pasted in a user turn, token-shuffled, sentence-reversed, length-matched neutral filler, and 100-token filler before the segment (position shift). Read CoTness by position at layers 12 and 16, subtract E12a. Pre-written decision rules: the direction survives only if the styled segment keeps rising past 100 tokens after the null is subtracted and the shuffled control does not.
If E12b passes: E12c length-versus-cumulative test (same total tokens split across turns; carry-over across a benign gap), then the many-shot flagship (k in 0, 1, 4, 16, 32; styled forged CoT, own benign CoT, destyled, neutral filler) with 100 paired prompts per behavioral cell and high-k cells as one GPU one-shot. Behavior must be measured on the same items; a probe curve alone is not a result.

**Decisive reason.** Until the positional null is subtracted, no statement about a "climb" is interpretable, and the null costs nothing.

**Material uncertainty.** Layer choice (12 vs 16) may change the picture; both are read. Advocate priors: about 0.35 that the reading keeps rising past 100 tokens, 0.3 that 16 styled shots lift ASR on a bare query above 15%.

**Framing that survives a skeptical reader.** If the controls kill it: a well-controlled negative on Appendix H with the null curve, itself a contribution. If it survives: a behavioral style-by-length dose-response and a within-segment versus cumulative test, descriptive and predictive, not causal.

**Sizing.** Readings: 50 prompts per position bucket, intervals across prompts. Behavior: 100 paired prompts per cell (MDE about 20 points; a null excludes about 15), greedy for the chat replication, temperature 1.0 with shared seeds only for arm comparisons, judge calibrated on 30 hand labels, per-(prompt, replicate, condition) seeds verified to give distinct first steps.

**Exact next action.** On Hanan's yes: E12a runs automatically after the queued probe pass; E12b is written and queued behind it (local, no judge, no forgeries, no harmful generations).

## Names used in the write-up (added 2026-09-11, evening)

The E-numbers above are the September 10 plan. Several ran under other folders and several did not run. The README, report, and form use the names below, never dates, agent names, or bare case numbers. Directories, scripts, and case IDs stay as they are; Codex owns the live H100 series and Grok owns the agent-hijacking package, so nothing is renamed. Cases are written as "page 7, Nostradamus" with the study name in front.

| Name in the write-up | What it is | Frozen path | State |
| --- | --- | --- | --- |
| Permission study | One fixed page with two harmless marker commands and a "do both" line; the user permits one. 149 episodes, 10 cases, 2 seeds, 7 arms (baseline, Toolward, Userward, 3 random, developer reminder). Prefill probe contrast on 8 cases. | `role-steering/data/permission-20260905`, `role-steering/report.md` | done |
| Transfer study | Its predecessor: read a page, find a dummy file, send it to a loopback receiver. 760 episodes. Found the fixed-seed exposure defect (140 of 260 unauthorized trials never read the page). | `role-steering/data/first-20260904` | done, superseded |
| Agent replication | The paper's Section 3.3 loop rebuilt locally (MLX, Docker, loopback receiver). Five pages, standard and forgery variants, 11 baseline episodes. | `replication/agent-hijacking/runs/ten-20260911`, `runs/resample-004-forgery-20260911` | done |
| Page steering, laptop | Tool−CoT class-mean vector at block-11 output on every page token, alpha 8 and 16, one random control at alpha 8, an all-token arm, a zero-dose identity check. Same five pages and seeds as the agent replication. | `replication/steering-agent/runs`, `analysis/` | done |
| Page steering, H100 | Same five pages on CUDA, seven arms: unsteered, zero dose, Tool−CoT alpha 16, reversed, three random. 35 episodes, complete 2026-09-12 03:32 UTC. A 40-episode batch on five new pages is running behind it. | `replication/steering-series/2026-09-12-positive-confirmation` | 35 done, 40 running |
| Direction geometry | Cosines between class-mean role directions at the steering site and the probe site, all 24 layers, 249 texts with jackknife. | `replication/steering-agent/directions`, `steering-series/.../geometry-preparation` | probe site done, block site pending |
| Probe replication | The authors' probe recipe on our activations: 384 probes (24 layers, 8 role spaces, 2 splits), Appendix E gardening, Appendix K neutral control, RH6 items. | `replication/cloud/persistent/sessions/20260911T030050Z` | done |
| Queued battery Q1 to Q10 | The follow-ups written as independent GPU cells. Q1 paragraph patching by layer; Q2 doubt-point counterfactual; Q3 wrapper manipulation; Q4 who-wrote-it readout; Q5 the steering null at power; Q6 geometry figure; Q7 decision-time direction; Q8 destyle program; Q9 declaration direction; Q10 text baselines. Run order agreed 2026-09-12: Q10, Q8, Q1 (destyled source), Q9, then Q5, Q3, Q4, Q1 (neutral), Q2, Q7. | `replication/cloud-steering` | built, not run |
| Role-uptake pilot | Grok's 24-case pilot: probe score before generation versus a self-reported ACTION line on dummy office tasks. | `replication/role-uptake/runs/pilot-24-20260911` | done, Explore only |

Mapping to the plan above: Probe replication covers E1, E2-lite, E3, E11. Agent replication is a five-page E6. Page steering is E10 with the Tool−CoT vector instead of the calibration direction. Q5 is E10 at power. E4, E5-lite, E7, E8, E9, E12 did not run.
