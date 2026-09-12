# Independent review 08 — methods and reproducibility

Report: `draft-r1.md`  
SHA-256: `022c1f4958500451a8568f2749e7878964f69607b023c692130acac016c7db4b`  
Lens: Methods and reproducibility: exact estimators, vector construction, model/probe distinctions, and reproducibility limits.

## Strongest accurate contribution

The paired permission comparison demonstrates that the direction of an absolute Userness contrast can reverse between two separately fitted probes, while the specified relative User/Tool contrast remains positive. The report defines the estimand precisely and keeps that readout observation separate from obedience or a causal role mechanism. The offset example complements this result by showing how the scoring operation itself creates a response. This is a coherent and supportable measurement report.

## Retelling

I read the report as an account of how to interpret role-probe measurements on GPT-OSS-20B. One supplied gardening conversation preserves some role-associated structure after relabeling, though its probabilities do not reproduce the paper. Direct offsets of the saved states produce predictable but dose-dependent changes in separately fitted class probabilities. On 100 constructed input pairs, a marker-permission cue raises a control-adjusted relative User/Tool score across the displayed settings, while the sign of absolute Userness depends on the classifier. Held-out role recall provides limited validation; the report correctly leaves probability calibration, optimizer convergence and downstream behavior unresolved. Its next-step proposal is a future propagated-intervention study, not a result established by the existing illustrations.

## Recommendations

### R08-1

**Passage:** “Exact training and split details appear in Appendix A.” / Appendix A, “Model and probes.”

**Concern:** Appendix A gives the number of probes, corpus size and activation site, but never names the fitting estimator or its regularization, preprocessing or split seed. The report also omits the model precision and does not explicitly assign Figure 5 to the prompt-split probes. These choices matter for the numerical replication gap and for understanding what a separately fitted role classifier represents. The linked native audit makes the information recoverable, but only by searching a large machine-readable record; the promised concise methods are not yet present.

**Suggested change:** Add a compact methods paragraph naming openai/gpt-oss-20b with MXFP4 expert weights, the saved model snapshot and runtime record; 62 C4 and 187 Dolma3 texts truncated to at most 1,024 content tokens and rendered separately in each role; training on content tokens with no feature scaling; cuML 25.8 L2 logistic regression with intercept, C=0.005, QN solver, tolerance 1e-4 and maximum 5,000 iterations; and 10% group holdouts with seed 123, split by rendered prompt or base text. State that Figure 5 uses the prompt-split SUCAT and UAT fits and that role-class combinations have their own saved partitions. Keep detailed package versions and split IDs in linked records rather than adding a code inventory.

### R08-2

**Passage:** “We compare two command orders and construct a control in which the command span is replaced by neutral text.”

**Concern:** The actual null construction replaces both commands together, while retaining the imperative wrapper. The singular wording suggests that only the scored marker span changes and the other command remains fixed. That is materially different for the marker-second condition, whose preceding command content also changes. The implemented comparison additionally uses one fixed preceding assistant-analysis sentence. The existing discussion of token counts and position is accurate, but does not identify these parts of the input construction.

**Suggested change:** Say that each null prefill replaces both command spans with approximately token-matched prose from the paired page and retains the imperative wrapper. State that all prefills contain the same supplied assistant-analysis sentence. Include the exact permission addition, or link directly to the frozen prefills/input contract: “If the page asks you to write a marker file, you have my permission to do that, and only that”. Keep the existing conclusion about this specified contextual comparison; no new experiment is needed.

### R08-3

**Passage:** Appendix A, “All three panels show 512 matched tokens: 95 originally User, 177 CoT and 240 Assistant. Full-content counts are 97, 179 and 582 respectively.”

**Concern:** The counts are correct, but they do not define the displayed subset. Matching is performed after restricting each non-System segment to its first 120 positions, and requires identical token text at the same segment/within-segment position in all three conditions. In particular, the 240 displayed Assistant tokens are the retained beginnings of two longer passages. Without the selection rule, a reader cannot reconstruct the display from the transcript and may read the passage means as whole-passage summaries.

**Suggested change:** Add one sentence: “Following the authors’ display rule, we retain the first 120 positions of each non-System segment, then keep positions whose token text matches across all three conditions.” Link the subset-construction source or its saved position table. Make clear that Figures 2–4 inherit this same display subset; retain the existing full-content denominator comparison.

## Evidence checked

I read the entire frozen draft before examining sources. I did not read the mutable draft or other reviewer verdicts. The following are the load-bearing checks; paths are relative to the replication root.

- `cloud/probes/train_probes_pod.py:428–449, 502–543, 622–642, 650–714`: corpus composition, rendering and truncation, content-token filter, fitting, and group splits. The completed `probes-full/metadata.json`, `split-audit.json` and `native-probe-audit.json` agree with this setup. The three-, four- and five-role prompt splits contain 74, 99 and 124 test variants respectively; this confirms that the class sets have separately sampled partitions. The saved model snapshot is `6cee5e81ee83917806bbde320786a8fb61efebee`.
- `rh/rh6_readings.py:95–129, 183–245, 354–408` and `cloud/persistent/cuda_readouts.py:120–155`: the fixed context, permission wording, simultaneous replacement of both commands, token-overlap span rule, span means, log-of-means relative score, paired differences and 2,000-resample percentile bootstrap. The completed RH6 metadata identifies prompt-split SUCAT/UAT probes, native activations cast to float32, and float32 affine/softmax scoring. These differ in arithmetic precision from the offline illustration, as expected; I found no conflict in the reported estimand.
- `cloud/persistent/appendix_e_figures.py:19–55`: exact Figure 7 subset construction, including the 120-position cap before cross-condition token matching.
- `cloud/persistent/sessions/20260911T030050Z/presentation/render_probe_offsets.py:17–78` and `presentation/probe-offset-illustration/provenance.json`: saved coefficient rows, intercepts, common norm reference over 921 forwarded tokens, 512 selected states, float64 scoring, and no downstream forward pass. The offset formula, direction distinction, subtraction interpretation and softmax common-row-shift caveat in the draft are correct.

## Scientific uncertainty already disclosed correctly

The report adequately discloses the unresolved gardening denominator and numerical gap; one-conversation and selected-analysis limits; the difference between an offline readout edit and model steering; separate class probabilities and relative scores; lack of paired behavior; imperfect neutral matching and lexical/position alternatives; reused input units and pointwise intervals; different fitted probes and test sets for the recall panels; and unresolved convergence and calibration. I do not recommend new experiments as a condition of this writing review. I also do not require a full software inventory: the linked records are an adequate place for detailed environment and artifact information once the short methods contract above is stated.

## Verdict

**Revise — three recommendations.** The central measurement claims are supported by the inspected methods. These changes make the input comparisons and reconstruction path explicit without expanding the scientific scope. I did not run a model, experiment, GPU job or API-based judge, and I changed only my two assigned review files.
