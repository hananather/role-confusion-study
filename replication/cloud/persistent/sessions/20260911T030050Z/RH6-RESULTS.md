# My RH6 readout results

I completed **1,200 prefills across 100 template–page pairs** at 04:53:47 UTC on 2026-09-11. Permission to write the marker increased its **relative User/Tool readout beyond the paired neutral-span control in all 12 layer × probe × order cells**. Absolute Userness was less consistent, including decreases at layer 12 with the five-role probe. I have no paired action outcomes, so this is a cue-sensitive readout result, not evidence of a behavior–reading dissociation or where instruction hierarchy is enforced.

I am in **Understand**. My north star is whether a user permission or prohibition changes a tool-borne command's measured role beyond the same-construction null. I distinguish that question from provenance, authority, and behavior.

## My measurement

Each frozen template–page pair has two command orders, three user turns, and command/null constructions: 12 prefills per pair. Permission names only the marker action; prohibition forbids uploads. I measured layers 8, 12, and 16 with the prompt-split five-role `sucat` probe (system, user, CoT, assistant, tool) and three-role `uat` probe (user, assistant, tool).

For each command span, the existing analysis averages per-token role probabilities. Its relative score is **log(mean Userness) − log(mean Toolness)**. This is the log ratio of span means, not the mean of tokenwise log ratios recommended in the earlier critique. I report the implemented quantity without changing it.

The command-specific cue contrast is paired within each template–page pair:

`(command cue − command neutral) − (null cue − null neutral)`.

The saved intervals use 2,000 bootstrap resamples of the 100 pairs, seed 123. They are pointwise 95% intervals across many cells, without a multiplicity adjustment; I treat the grid descriptively.

## What the completed contrasts show

The table gives that contrast in **natural-log User/Tool units**. First/second refers to where the same command is listed.

| Layer / probe | Marker permission, first / second | Exfiltration prohibition, first / second |
| --- | ---: | ---: |
| 8 / five-role | +0.744 / +0.527 | +0.025 / +0.024 |
| 8 / three-role | +0.509 / +0.267 | −0.003 / +0.016 |
| 12 / five-role | +0.766 / +0.619 | +0.037 / +0.025 |
| 12 / three-role | +0.479 / +0.494 | +0.030 / +0.015 |
| 16 / five-role | +1.437 / +0.837 | +0.020 / +0.040 |
| 16 / three-role | +1.074 / +0.541 | +0.065 / +0.055 |

All 12 marker-permission relative-score intervals lie above zero. At layer 12, the five-role intervals are **[0.703, 0.830]** first-listed and **[0.557, 0.678]** second-listed. The corresponding three-role intervals are **[0.425, 0.536]** and **[0.443, 0.547]**.

Ten of 12 exfiltration-prohibition relative-score intervals contain zero. The two exceptions are small positive changes in the layer-16 three-role probe: **+0.065 [0.031, 0.102]** first and **+0.055 [0.017, 0.093]** second. I do not interpret an interval containing zero as equivalence or as proof that the model ignored the prohibition.

The choice of summary matters. At layer 12, marker permission changes absolute Userness beyond null by **−1.395 [−2.026, −0.779] percentage points** first and **−0.906 [−1.391, −0.444]** second with the five-role probe, while the three-role values are **+4.458 [3.423, 5.474]** and **+7.723 [6.932, 8.472]**. Relative User/Tool can rise while absolute Userness falls. Eleven of the full 48 cue-contrast cells differ in sign between these two summaries.

Order also affects the null. Under the neutral user turn at layer 12, first-minus-second Userness for the exfiltration command is +2.623 points with the five-role probe, compared with +5.467 for its replacement text. For marker, the respective values are +4.522 and +3.589. These are separate paired contrasts, not a tested difference between those order effects; they show why order alone cannot identify an authority-specific response.

## What limits the interpretation

- My prefills are synthetic and use one fixed preceding assistant-analysis sentence. I generated no responses, executed no embedded commands, measured no action outcomes, and ran no attribution-question stage.
- Each template is paired with one distinct page; 26 pages were substituted during preparation because the originally assigned pages lacked usable prose. The intervals describe these sampled pairs, not separately identified template and page effects.
- Neutral-span matching is approximate: only 47/100 pairs match both standalone command token counts exactly. The remaining gaps range from seven fewer tokens to one more. The null also retains the imperative wrapper. It controls this construction imperfectly rather than isolating permission semantics.
- The saved overlap table contains unadjusted lexical-overlap correlations, not overlap-adjusted cue effects. Added user sentences also change absolute token positions. The observed permission signal does not establish an explanation beyond lexical overlap, position, or other contextual changes.
- The 750 inherited out-of-text span ends concern the broad `tool_result` reference only; command, injection, and user-turn boundaries stay within the text. I do not base these conclusions on the page-reference means.

The marker result rules out an unchanged relative readout under this cue on these inputs. The sign reversals rule out treating an increase in absolute Userness as a configuration-independent signature of permission. The stronger dissociation claim remains untested: it requires paired behavioral evidence and a justified interpretation of the readout. I preserve the complete contrast grid rather than selecting one probe or metric as a confirmed hierarchy measure.

Sources: [completed metadata](outputs/20260911T030050Z/rh6/metadata.json), [frozen input contract](outputs/20260911T030050Z/rh6/input-contract.json), [all cue contrasts and intervals](outputs/20260911T030050Z/rh6/user_turn_effects.csv), [all order contrasts](outputs/20260911T030050Z/rh6/order_effects.csv), [overlap correlations](outputs/20260911T030050Z/rh6/overlap_covariate.csv), [analysis source](../../../../rh/rh6_readings.py), and [prior design critique](../../../../audits/2026-09-10-role-hierarchy-design-critique.md).
