# Independent numerical review — reviewer 04

Report: `draft-r1.md`  
Report SHA-256: `022c1f4958500451a8568f2749e7878964f69607b023c692130acac016c7db4b`  
Lens: Numerical accuracy, denominators, rounded claims, and figure/text consistency  
Verdict: **revise** — one bookkeeping clarification; the reported result values and principal denominators check out.

## Strongest accurate contribution

The strongest contribution is a concrete demonstration that the same paired permission comparison can have opposite signs in absolute Userness under separately fitted probes while the adjusted relative User/Tool score remains positive. I independently reconstructed all 24 plotted permission estimates and their interval bounds from the 100 paired per-item measurements. The layer-12 examples, including the sign disagreement under both command orders, match the report. The offline offset example supplies a second, independently checkable measurement lesson: at the small positive Tool offset, mean Userness and Toolness both rise in the original User passages. The report correctly limits these findings to the measured readouts.

## Retelling

A role probe assigns probabilities to internal token states, and a higher score acquires meaning only after specifying the probe, metric, intervention and averaging unit. One gardening conversation retains a reasoning-related pattern under altered tags, although its averages do not reproduce the paper's reported levels. Adding saved classifier directions to those same states gives predictable but direction- and magnitude-dependent probability changes; it does not establish a propagated model effect. Across 100 constructed template–page pairs, a permission cue raises the marker's adjusted relative User/Tool score in all 12 tested configurations, while the absolute Userness sign depends on the probe. Held-out recall varies by role and layer. The report therefore motivates a separately controlled, adequately sized propagation study without treating the saved readouts as behavioral evidence.

## Recommendation

### 04-r1-01 — Count both probability means plotted per group

**Exact passage, “Review status and sources”:**

> The numerical audit independently reconstructed all 30,720 offset probabilities, 30 plotted passage means, 24 permission contrasts and their interval bounds, and 240 recall values from saved records.

**Concern:** The phrase “30 plotted passage means” counts the rows in offset-dose-response.csv rather than the scalar means shown in Figure 4. The file has 30 direction × offset × original-role groups, each containing a mean Userness and a mean Toolness. Figure 4 therefore plots 60 probability means: four panels × three original-role curves × five evaluated offsets. The numerical values themselves reproduce correctly.

**Suggested remedy:** Replace “30 plotted passage means” with “60 plotted probability means across 30 passage groups,” or with “30 passage-group rows, each containing mean Userness and Toolness.” Keep the other reported audit counts.

**Evidence:** [The dose-response source table](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/offset-dose-response.csv>) has exactly 30 rows and two plotted probability columns, `p_user` and `p_tool`. I independently reproduced all 60 means from the token-level offset table; the largest absolute difference was 1.11 × 10⁻¹⁶, with exact token counts. The four visible panels in Figure 4 each contain 15 points. This is a count/wording correction, not a change to any plotted measurement or scientific conclusion.

## Independent verification

I read the entire frozen report and visually inspected all six PNGs. I checked the saved source tables, metadata, split records, coefficient and activation arrays, the original per-item permission table, confusion counts, the frozen notebook sampling cell, the locally preserved paper text, and the recovered historical episodes. I did not read the other current reviewers' verdicts or the narrative card. The earlier numerical audit was used only as a lead; the checks below were recomputed from the underlying records. I ran no model, GPU job or experiment and changed no report, figure or evidence file.

| Check | Independent result |
| --- | --- |
| Gardening displayed subset | Each condition contains 512 rows: 95 User, 177 CoT, 240 Assistant. Displayed CoT means are 64.3294%, 71.7761%, and 71.7160% for correct tags, no tags, and all User tags; all reported one-decimal values are correct. |
| Gardening full content | Original-role token counts are 97 User, 179 CoT, 582 Assistant in every condition. Full CoT means are 64.0258%, 71.0436%, and 70.9750%; the report's 64.0%, 71.0%, and 71.0% are correct. The preserved paper text reports 85%, 82%/83%, and 85% in the cited locations. |
| Offline offset arithmetic | All 30,720 scalar probabilities were finite and reproduced from the saved float16 activations and float32 five-role coefficient arrays promoted to float64. With explicit summed products, the maximum discrepancy was 1.05 × 10⁻¹⁵. The 921-token median norm is 45.24711366617226; offset lengths 0.4524711366617226 and 2.262355683308613 round as stated. |
| Quoted offset means | The small User offset gives 70.1551% → 86.6849% for original User tokens and 6.6483% → 15.7765% for CoT tokens. The large User offset gives 99.9531%, 77.0041%, and 77.1774% for User, CoT and Assistant tokens. The Tool-offset table rounds correctly at all three shown offsets. |
| Permission means and intervals | All 24 plotted means and 48 interval bounds reproduce from the per-item measurements using 2,000 paired percentile-bootstrap resamples and seed 123; maximum discrepancy 1.67 × 10⁻¹⁶. All 12 relative-score lower bounds exceed zero. Layer-12 marker-first absolute effects are −1.3947849 points [−2.0264482, −0.7788938] for SUCAT and +4.4576235 [3.4230701, 5.4743961] for UAT, matching the stated rounding. |
| Permission denominators | 1,200 distinct input IDs, 100 templates, 100 pages, one page per template and 12 input conditions. Exactly 47 of the 100 pairs match both saved standalone replacement-span token counts to their corresponding commands. |
| Recall and split sizes | All 240 recalls and all associated integer numerators/denominators reproduce from the five-role confusion counts. At layer 12 in the prompt split: User 9853/12846 = 76.7009%; Assistant 12416/14793 = 83.9316%; CoT 6360/13765 = 46.2041%; Tool 7577/16909 = 44.8105%. The split audit confirms 124 variants/67,727 tokens and 24 base texts/56,685 tokens. |
| Training total | The live metadata and portable probe records support 249 base texts, 1,245 variants, 24 layers, eight role-class sets and two splits: 384 fitted probes. |
| Supplementary counts | The recovered `tool_03` arm contains 21 episodes across eight cases; 17 wrote both markers. The frozen notebook cell sets `max_samples = 30` with the comment “100 for full test”; the preserved paper describes 200 conversations. The cited Claude source contains 19 TaskCreate calls and its subagent directory contains 35 JSONL logs. |
| Figure/text agreement | The visible six figures match the stated tag conditions, probe class sets, layer choices, signs, axes, passage colors and averaging units. The first three remain the same figures recorded in the earlier hash capture. No plotted-value correction is recommended. |

## Scientific uncertainty already disclosed correctly

The gardening example is one authored conversation, with an unresolved numerical gap and an unknown published averaging denominator. Reused tokens and offset strengths do not create independent conversations. Offline coefficient-vector edits do not show propagation or behavioral control, and individual coefficient rows depend on the saved softmax parameterization. The permission comparisons are exploratory, reuse the same 100 coupled template–page pairs, retain imperfect replacement matching and lexical/position alternatives, and use pointwise intervals without simultaneous coverage. The recall panels use different cohorts and separately fitted probes; they do not isolate the causal effect of text overlap or establish calibration or convergence. The recovered behavioral sample has repeated prompts and only eight cases. These limits are already accurately stated; none requires new science before this report can describe the saved measurements.

## Verdict

**Revise.** Make recommendation 04-r1-01. I found no numerical error that reverses a result, no incorrect quoted rounding, and no denominator inflation in the main scientific claims. This verdict does not request additional model runs or a change of research direction.
