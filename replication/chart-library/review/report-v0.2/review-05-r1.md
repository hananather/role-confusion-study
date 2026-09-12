# Independent review 05 — statistical interpretation

- **Frozen report:** `draft-r1.md`
- **SHA-256:** `022c1f4958500451a8568f2749e7878964f69607b023c692130acac016c7db4b`
- **Lens:** Statistical units, uncertainty, paired contrasts, sampling and selection, probe validation splits, and justification of future sample sizes.
- **Stage:** Understand / Distill.
- **North star:** I want the reader to know what was averaged, what was held fixed, and which conclusion the saved evidence supports.
- **Research criteria:** Clarity, skeptical interpretation, source fidelity, adequate statistical units, and a claim strength matched to the evidence.
- **Verdict:** **Revise — three reporting clarifications.** The central descriptive findings survive this review; I found no numerical error in the permission estimates or their saved interval bounds.

## Strongest accurate contribution

The strongest contribution is the demonstrated dependence of the adjusted permission result on the readout: at layer 12, the same paired input comparison has negative mean absolute Userness under the five-role classifier and positive mean absolute Userness under the three-role classifier, while relative User/Tool rises under both. This is a useful empirical warning against treating these summaries as interchangeable. The offline offsets add a separate, concrete illustration that Userness and Toolness can increase together in a multiclass probe. The report correctly treats this as measurement evidence rather than observed instruction following.

## My retelling

I read the report as a study of how to interpret role-probe measurements on GPT-OSS-20B. A single gardening conversation preserves some reasoning-text structure after role tags change, but the numerical replication remains incomplete. Rescoring its saved states after offsets shows a direct response of the classifier, including simultaneous increases in Userness and Toolness at one strength. A larger constructed-input study then shows that a marker-permission contrast changes sign across separately fitted absolute-probability readouts even while all 12 mean relative contrasts are positive. The reported recall curves, split differences and unresolved convergence explain why those scores need validation. Propagation and behavior are deferred questions, and the future 200-conversation floor still requires an effect-size and precision justification.

## Recommendations

### R05-01

**Exact passage:** TLDR: “Across 100 constructed input pairs, the marker command’s permission effect is positive in all 12 tested relative-score settings after subtracting a neutral-text control.” Section 4: “The full dataset reuses the same 100 input pairs across 12 input conditions, yielding 1,200 prefills; those are not 1,200 independent examples.”

**Concern:** The report correctly gives three layers × two probe class sets × two orders for the 12 plotted settings, but never enumerates the different factors that create the 12 input conditions. The study initially describes only permission and neutral user wording, so the third user condition is missing from the construction explanation. “Input pairs” can also mean a treatment/control pair, although the statistical unit is a template paired with a page. Finally, the positive result is an average over the 100 units; individual adjusted contrasts can be negative.

**Suggested remedy:** Call the unit a “template–page unit” consistently and say “mean adjusted permission effect” in the TLDR. In Section 4, give the construction explicitly: “Each of the 100 template–page units has 12 input conditions: two command orders × three user turns (neutral, marker permission, upload prohibition) × two constructions (commands or neutral replacements), producing 1,200 prefills. Figure 5 selects the marker-permission contrast and evaluates it at 12 readout settings: three layers × two probe class sets × two orders.” No new analysis is needed.

### R05-02

**Exact passage:** Section 4: “The interval bars describe uncertainty across pairs for each plotted setting, without a simultaneous guarantee across the grid.” Appendix A: “The intervals use 2,000 paired bootstrap resamples with seed 123.”

**Concern:** The pairing and percentile interval calculation are correct, and the report already discloses exploratory selection and the absence of simultaneous coverage. The remaining inferential boundary is that the bootstrap holds the fitted probes, model, and input-construction procedure fixed. It resamples the observed template–page units; it does not quantify uncertainty from fitting/selecting the readout or choosing the highlighted comparisons. The draft also omits that 26 initially assigned pages were replaced for usable prose, which helps readers identify the population to which any exchangeability assumption would apply.

**Suggested remedy:** Add a compact statement near the Figure 5 interval explanation: “These descriptive intervals resample the observed template–page units with the model and fitted probes fixed. They do not include probe-fitting uncertainty or adjust for selecting the highlighted comparisons after inspection.” In Appendix A, identify the selection procedure: 100 sampled template records, one sampled page per template, and 26 page substitutions for usable prose. Preserve the existing restriction to the constructed inputs; do not imply population coverage for deployed conversations. This is a reporting clarification, not a request for nested bootstrapping or new experiments.

### R05-03

**Exact passage:** Section 5: “These are the five-role probes; they are not the four-role readout used for the gardening overview.” Appendix A: “Figure 1 uses the prompt-split four-role probe; Figures 2–4 use the prompt-split five-role probe at the same layer-12 site.”

**Concern:** The validation paragraph distinguishes the four-role gardening probe but leaves the three-role probe in the central permission comparison unmentioned. The appendix also omits the split used for Figure 5. A reader can therefore overextend Figure 6’s validation to both probes underlying the sign disagreement. The held-out data are role-rendered neutral passages, whereas Figures 1–5 concern full conversation contexts or edited activations; recall on the training task does not itself validate those applications.

**Suggested remedy:** Add “Figure 5 uses the prompt-split five-role and three-role probes at layers 8, 12 and 16” to Appendix A. In Section 5, state that Figure 6 evaluates the five-role classifiers on held-out role-rendered neutral passages; the four-role and three-role classifiers are separately fitted, and mixed conversation contexts and offset activations lie outside this validation set. Retain the existing calibration and convergence caveats. No additional validation experiment is required to keep the present results descriptive.

## Evidence I checked

I read the complete frozen report and inspected the current saved records without reading another reviewer’s verdict or the narrative card.

- I independently recomputed all **24 plotted permission contrasts and their percentile interval bounds** from `per_item_span_means.csv`, using the stated within-unit difference of differences and 2,000 resamples with seed 123. The maximum absolute difference from the plotted values was **2.22 × 10⁻¹⁶** in the stored metric units.
- The frozen inputs contain **100 template–page units, 100 distinct pages and 1,200 prefills**, with exactly 100 rows in each order × construction × user-turn condition. There are 100 template records but 99 distinct template strings. The report does not claim 100 unique template wordings, so this is not a requested correction.
- All **12** relative-score lower interval bounds are positive. These are mean effects: for example, at layer 12, positive unit-level contrasts occur in 96/100 units for five-role marker-second and 98/100 for three-role marker-first.
- The underlying input records confirm **47/100** exact matches for both replacement-span standalone token counts and **26** page substitutions.
- The split audit confirms **124 held-out prompt variants / 67,727 content tokens** and **24 held-out base texts / 56,685 content tokens** for the five-role fits. Train/test group IDs are disjoint under each respective grouping. The displayed layer-12 prompt-split recalls match the saved count/correct ratios.
- The estimator audit explicitly leaves convergence unestablished; the report represents that status accurately.
- The current deferred E9 specification requires at least 200 eligible independent conversations after exclusions, paired conversation-level analysis, source balance, a prechosen endpoint, effect-size/precision justification, a multiplicity rule and reserve candidates. The report’s short description is consistent with that specification.

Primary local records inspected:

- `cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/rh6/{prefills.json,metadata.json,input-contract.json,per_item_span_means.csv,page_substitutions.json}`
- `rh/rh6_readings.py`, especially input construction and the paired estimator.
- `chart-library/data/{rh6-plotted.csv,probe-recall-by-layer.csv}`
- `cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/{metadata.json,split-audit.json,native-probe-audit.json}`
- `cloud/persistent/sessions/20260911T030050Z/RH6-RESULTS.md`
- `e9/BACKLOG.md`

## Scientific uncertainty already disclosed correctly

I would preserve the report’s existing treatment of these issues:

- The gardening result is one authored conversation, with a displayed-token subset and an unresolved published denominator. Token repetitions, strengths and layers provide no new independent conversations.
- The probability changes under saved-state offsets are direct classifier responses. They do not establish propagation, behavior or a unique semantic direction.
- The emphasized comparisons were selected after inspection. Pointwise intervals give no simultaneous grid guarantee, and the report makes no equivalence claim from weaker contrasts.
- Neutral replacements are imperfect; lexical overlap, token position and the template–page coupling remain alternative explanations for the contextual response.
- The prompt-variant and base-text splits use different held-out sets and separately fitted probes. Their difference does not isolate a causal effect of cross-split text overlap.
- Token-weighted recall is a classifier diagnostic, distinct from predicted probabilities and calibration. Optimizer convergence remains unresolved.
- The 200-conversation figure is a planning floor, not a completed sample or a demonstration of adequate power. The draft correctly requires the final sample size to depend on a useful paired effect and desired precision.
- Earlier repeated behavioral episodes cannot support a general steering-effect estimate.

## Final assessment

I recommend the three wording and provenance clarifications above. They preserve the report’s present claims and require no new model work, refitting, uncertainty analysis or experiment. Once they are incorporated, I see no unresolved statistical-reporting objection that would reverse the descriptive interpretation.

