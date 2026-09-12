# Historical steering evidence: coverage and claim audit

*September 12, 2026 · Understand → Distill · Saved-record analysis only*

My north star is whether a changed role-probe readout buys selective, useful behavior. I use Neel's criteria of truth-seeking, matched baselines and inspection of raw data. I preserve unfavorable results and separate an intervention that changes behavior from one that reliably improves behavior. I loaded no model and ran no evaluated command.

## The strongest historical result is a readout–behavior mismatch across the recovered cohort

In the older permission task, **all 21 recovered Toolward episodes have a matched unsteered episode**. Steering makes the mean first-prefill command score at least **99.9976% Tool in every episode**, but **17 of the 21 steered episodes write both markers when the user permits only one**. This is a cohort-wide manipulation check with persistent unwanted action, not a selected success or a universal claim that steering has no effect.

The paired comparison is mixed: Toolward steering prevents four unwanted physical writes and introduces three. It loses two correct completions of the permitted action and two reports of both page values. One prevented physical write comes from a malformed generation with **unknown attempt status**. Counting attempted actions instead gives three prevented, three introduced and one unknown. I rederived these numbers from saved marker readbacks and scored-token records; they agree with the older report. [Raw permission records](</Users/hananather/Desktop/MATS 12.0/role-steering/data/permission-20260905/episodes.jsonl.gz>), [all 21 matched pairs](permission-toolward-pairs.csv), [verification](historical-audit-verification.json).

The complete comparison includes every recovered random direction and the text reminder:

| Method | Matched episodes | Unwanted writes prevented / introduced | Correct permitted completions lost | Both-value reports lost |
| --- | ---: | ---: | ---: | ---: |
| Toolward | 21 | 4 / 3 | 2 | 2 |
| Userward | 20 | 0 / 3 | 0 | 0 |
| Source `random_0` | 22 | 1 / 1 | 0 | 0 |
| Source `random_1` | 21 | 3 / 1 | 0 | 1 |
| Source `random_2` | 22 | 1 / 2 | 0 | 0 |
| Prompt reminder | 21 | 6 / 0 | 2 | 0 |

The methods have different recovered matched cohorts. These rows are descriptive, not a reliable method ranking. [Independent per-pair reduction](permission-all-method-pairs.csv), [method summary](permission-method-summary.csv).

### What this cohort does and does not measure

- **Endpoint:** controller-executed writes to two marker files, correct selected nonce, and reporting both nonces. It is not the later dummy-file upload benchmark.
- **Intervention:** neutral-text Tool-minus-User direction, block 11 output, command-token mask, magnitude 0.3 times the construction residual norm. The five-role classifier reads block 14 post-attention normalization. This is distinct from the later dev-10 Tool-minus-CoT direction and layer-12 probe.
- **Coverage:** 149 of 280 assigned records are recovered, spanning eight of ten planned cases. All 21 recovered Toolward records are included; every one has a matched baseline. Eighteen additional records mentioned in the historical report were not recovered and remain excluded.
- **Exposure:** the page is supplied before generation. I verified a successful recorded read for all 21 Toolward episodes.
- **Probe denominator:** the 21 episodes share 15 case/permission first-prefill contexts; repeated generation seeds have identical first-prefill scores. They are eight underlying cases, not 21 independent pages, thousands of independent tokens, or 15 independent attack types.
- **Timing:** the minimum across all recorded command tokens, including later prefills, is 99.858814% Tool. The minimum within first prefills is 99.899310%; the minimum **mean** first-prefill score is 99.997628%. These are different summaries. The CSV preserves all three so a figure can name its exact readout.
- **Scope:** command-only steering leaves the surrounding instruction to perform both actions outside the mask; extreme classifier probabilities do not establish that the whole agent represents or obeys a trust boundary.

Sources: [frozen protocol and cases](</Users/hananather/Desktop/MATS 12.0/role-steering/data/permission-20260905/protocol-manifest.json>), [historical methods](</Users/hananather/Desktop/MATS 12.0/role-steering/report.md>), [export manifest](</Users/hananather/Desktop/MATS 12.0/role-steering/data/export-manifest.json>). Every public export hash checked against the manifest matches. The pair CSV retains original decompressed JSONL line numbers and raw-record SHA256s. I also matched first-prefill token identities and positions for non-prompt arms; the reminder shifts positions while preserving scored token identities.

## Earlier local upload episodes change readouts without a reliable role-specific benefit

The September 11 MLX discovery sweep uses the five historical forgery pages later reused by the CUDA bridge. These are **the same pages, new episodes**, not independent replications on new examples.

| Local arm | Assigned / closed | Verified uploads | Exposed | Censored | First-response payload readout |
| --- | ---: | ---: | ---: | ---: | --- |
| Tool-minus-CoT, alpha 8, tool tokens | 5 / 5 | 4 | 5 | 0 | CoT approximately 10⁻⁹ on all five |
| Source `random_0`, alpha 8, tool tokens | 5 / 5 | 3 | 5 | 0 | CoT 0.508–0.677 on all five |
| Tool-minus-CoT, alpha 16, tool tokens | 5 / 5 | 2 | 5 | 0 | CoT approximately 10⁻¹⁸ on all five |
| Tool-minus-CoT, alpha 8, all tokens | 5 / 5 | 0 | 1 | 4 | Four never reach the page; one has a later payload readout |
| Literal alpha-zero hook | 5 / 3 | 3 | 3 | 0 closed | Three measured, two without closed episodes |

I checked every stored upload flag against receiver receipts with no disagreements. At alpha 8 the matched random direction has fewer uploads than the role direction. Alpha 16 improves the historical count, but there is no same-magnitude local random control in this sweep; the later CUDA experiment supplies a separate controlled comparison. Zero observed uploads under all-token steering are dominated by failure to fetch the page and generation censoring. They cannot be presented as five successful defenses. [Every assigned local slot](historical-mlx-episode-inventory.csv), [frozen directions](</Users/hananather/Desktop/MATS 12.0/replication/steering-agent/directions/block11.json>), [local experiment README](</Users/hananather/Desktop/MATS 12.0/replication/steering-agent/README.md>).

The completed literal-zero episodes reproduce generation tokens through the upload decision. Their first post-fetch completions match exactly (498, 111 and 256 tokens). Full final answers do not match: after upload, curl timing and transfer-speed stderr changes the prompt. I therefore verify an upload-decision identity check, not complete-trajectory identity. [Per-turn identity verification](historical-audit-verification.json).

Several files under `steering-agent/analysis/` are unfinished-run snapshots. For example, `steered-divergence.md` says that only three alpha-8 episodes exist and other arms are absent; `cot-quantitative.md` counts only four all-token episodes. Those snapshots must not override the completed records above.

## The local unsteered attack pilot is not a steering-effect dataset

The timing cohort contains ten assigned and completed records: five ordinary-injection and five forged-reasoning episodes. The forgery arm has four verified uploads plus one **unexposed, token-limited** episode. The ordinary arm has one upload, three completed nonuploads and one **exposed generation timeout**. Thus “4/5 versus 1/5” records observed receipts, with one unresolved nonupload in each arm; it is not a clean estimate from ten fully resolved outcomes.

Page 004 was then resampled at seed **11243**, replacing the original **1243** for later discovery comparisons. The resample uploaded. The five-page historical steering baseline consequently uses four first draws plus this successful resample. I preserve the original censored first draw as well. An earlier one-episode engineering pilot on page 000 is separate; a failed resample launch produced no closed episode. These must not be added as independent pages. [Pilot README](</Users/hananather/Desktop/MATS 12.0/replication/agent-hijacking/README.md>), [timing run](</Users/hananather/Desktop/MATS 12.0/replication/agent-hijacking/runs/ten-20260911/run.json>), [resample](</Users/hananather/Desktop/MATS 12.0/replication/agent-hijacking/runs/resample-004-forgery-20260911/run.json>), [slot inventory](historical-mlx-episode-inventory.csv).

Those original pilot episodes have saved exact prompts, generation token IDs and behavioral receipts, but no contemporaneous `steering.json` probe readout. Three literal-zero reruns supply readouts on matching early prompts. Any later activation replay or selected figure remains a separate measurement with its own provenance.

## The oldest transfer run has major exposure and coverage limits

I independently counted **760 of 1,880** planned recovered records: 260 unauthorized, 245 authorized, 111 benign, 114 direct and 30 calibration. These are not 760 prompt-injection tests. **140 of 260 unauthorized episodes never received their assigned page**, verified against recorded tool-output HTML, rather than inferred from the model's prose. Across all task types, 502 records contain token-score measurements.

Observed unauthorized transfers are 5/44 for no steering, 4/36 for the smaller Toolward dose, 1/44 for the larger dose, 6/12 for Userward, 0/23, 4/16 and 0/23 for the three random directions, 6/45 for the prompt reminder, and 0/17 for redaction. The denominators are recovered records, not a common matched cohort; they contain both exposed and unexposed episodes and extensive missing assignments. These counts can motivate follow-up but cannot establish that one defense is better. [All recovered transfer records](historical-transfer-record-inventory.csv), [coverage by task and method](historical-transfer-method-summary.csv), [original records](</Users/hananather/Desktop/MATS 12.0/role-steering/data/first-20260904/episodes.jsonl.gz>).

The historical report preserves two selected favorable cases (036 and 038) where Toolward steering prevents a transfer and preserves the requested value, and 14 matched authorized Toolward pairs retaining transfer. These are documented observations, but the failed exposure pathway and incomplete allocation explain why the later guaranteed-exposure marker task is more informative. I did not independently re-adjudicate the old attempt labels in this audit; receipt/transfer flags, exposure and score availability are the reductions above. [Historical report](</Users/hananather/Desktop/MATS 12.0/role-steering/report.md>).

## The role-uptake pilot should be excluded from behavioral steering totals

This is a four-situation by six-condition pilot with **24 valid, unexecuted ACTION proposals**. Authorized plain instructions and authorized reasoning each give four FOLLOW answers; ordinary injections and CoT forgeries each give four DRAFT answers. Quotation controls give three NONE and one FOLLOW. Ordinary-task controls produce mixed answers. High CoT scores coexist with DRAFT responses to all four forged examples, but no action executes and no steering is applied. It is a probe/protocol diagnostic, not evidence of mitigation efficacy.

The two-case engineering ceiling's result rows also appear identically in the 24-case run. I do not add them as two new situations. The planned 144-case study did not run. [Pilot results](</Users/hananather/Desktop/MATS 12.0/replication/role-uptake/runs/pilot-24-20260911/run.json>), [plan and observed checkpoint](</Users/hananather/Desktop/MATS 12.0/replication/role-uptake/PLAN.md>), [independent counts](historical-audit-verification.json).

## Recommended figure roles

1. **Main behavioral claim:** use the complete latest controlled CUDA cohort, with all random controls, censoring and unrun slots. That audit is owned by the integrating task.
2. **Separate historical marker figure:** show all 21 matched Toolward pairs, the near-saturated first-prefill Tool score, unwanted writes and permitted completion. The claim is that moving this readout is insufficient for selective behavior in these recovered cases.
3. **Historical appendix:** show the complete local five-page discovery matrix, including the all-token failure-to-fetch arm and the incomplete zero control. Keep the alpha-16 favorable cases visible without making them the headline.
4. **Coverage note:** preserve the older 760-record transfer and 24-proposal pilots as distinct studies. Do not add them to a pooled “attack success rate.”

The strongest supported narrative is: **role scores can move dramatically while unwanted actions persist; across the controlled tests, any proposed mitigation must earn its claim against random perturbations and legitimate task completion.** The historical data reject “the probe changed, therefore the defense worked.” They do not justify “all steering vectors have zero behavioral impact.”
