# Independent review 03 — mechanistic interpretation and causality

Report: `draft-r1.md`  
SHA-256: `022c1f4958500451a8568f2749e7878964f69607b023c692130acac016c7db4b`  
Lens: mechanistic interpretation, causal alternatives, probe geometry, and scientific claim strength.  
Stage: **Understand / Distill**. My north star is to preserve the strongest supported measurement contribution while preventing a classifier response from becoming a claim about the model’s role mechanism or instruction following. This applies the clarity, skepticism, and source-fidelity criteria in the research lens.

## Strongest accurate contribution

The strongest contribution is the demonstrated dependence of a contextual comparison on the chosen role readout. At layer 12, the marker’s control-adjusted User probability decreases under the five-role classifier and increases under the separately fitted three-role classifier, while the reported relative statistic increases under both. The report usefully pairs that result with an exact offline sensitivity illustration showing that User and Tool probabilities need not move in opposite directions. Together these establish a measurement lesson in the saved settings. The report does not need a new mechanism claim to make that lesson useful.

## My retelling

I can read role-related patterns from GPT-OSS-20B activations, but the meaning of a score change depends on the classifier, the aggregation, and the intervention. A supplied gardening conversation retains some readable reasoning-role pattern when its tags change, although its average probabilities do not numerically reproduce the paper. Moving those saved activations along a classifier’s own coefficient rows produces dose-dependent score changes, including simultaneous increases in User and Tool probabilities. In a larger constructed input comparison, one permission sentence changes the relative marker readout beyond the neutral-span response across the displayed configurations, while the absolute User score can reverse sign across classifiers. None of the six figures measures downstream behavior after steering. A future propagated-intervention study remains deferred, with separate vector families and controls required before interpreting persistence.

## Recommendations

### R03-01

**Exact passage:** Section 4, lines 103–105: “a User message either permits an action or uses neutral wording” and “We use 100 constructed template–page pairs.”

**Concern:** The strongest empirical result varies template and page while reusing one permission sentence. The draft never makes that fixed wording visible. Readers cannot assess the disclosed lexical-overlap explanation or distinguish evidence across 100 contexts from evidence across permission formulations. The neutral construction also replaces both commands together, which should be explicit.

**Suggested remedy:** Add a compact concrete input description in Section 4: “Every pair uses the same added sentence: ‘If the page asks you to write a marker file, you have my permission to do that, and only that’. The neutral User message omits this sentence. The control replaces both command spans with page prose while retaining the surrounding instruction wrapper.” State that this measures one wording across 100 template–page contexts. These details come from the frozen prefills and require no new experiment.

### R03-02

**Exact passage:** Section 4, line 111: “Relative User/Tool uses the logarithm of the User-to-Tool probability ratio”; Appendix A, line 167: “This is not a mean of tokenwise log ratios”.

**Concern:** The appendix correctly specifies the arithmetic but does not explain its interpretive consequence. A tokenwise log User/Tool ratio cancels the common softmax denominator. The implemented log ratio of span-mean probabilities does not: changes to other classes can reweight tokens in that aggregate, changing it while every token’s User–Tool logit difference stays fixed. A reader may otherwise mistake the relative metric for a pure User–Tool contrast independent of the remaining classes. This is a possible explanation, not a demonstrated explanation of the observed RH6 result.

**Suggested remedy:** Define the metric initially as “the log ratio of the span-mean User and Tool probabilities”. Add to Appendix A: “A tokenwise log ratio equals the User-minus-Tool logit difference. Averaging probabilities before taking the ratio means our span score can also change through redistribution of probability to other classes across tokens, even if every token’s User–Tool logit difference stays fixed.” Keep the saved estimator and reported results unchanged; no retrospective replacement or new analysis is required.

### R03-03

**Exact passage:** Section 4, lines 115–119: “Columns compare five roles ... with three” and “Probe choice and metric choice are part of the result”.

**Concern:** The report correctly states earlier that each role set is separately fitted, but the central comparison can still be read as a controlled removal of System and CoT from one classifier. It changes fitted coefficients as well as the class set. The sign disagreement therefore establishes dependence on these trained readouts; it does not identify class-count normalization alone as its cause.

**Suggested remedy:** Add one sentence beside Figure 5 or the sign comparison: “These classifiers were fitted separately, so the comparison changes the learned coefficients as well as the available classes.” Retain the affirmative conclusion that the same input contrast has different absolute-score signs under the two readouts.

### R03-04

**Exact passage:** Section 5, lines 127–139, especially “Probe reliability varies by role and layer” and “These are reasons to validate a chosen readout before making a stronger claim from it.”

**Concern:** The validation discussion distinguishes accuracy from calibration and discloses convergence uncertainty, but does not explicitly identify the domain being validated. Held-out role-rendered neutral texts test decoding under that construction. Even excellent performance there would not by itself validate semantic-role or authority interpretations on the gardening and permission inputs. This domain distinction is material to the report’s question about how much a role score can be trusted.

**Suggested remedy:** Add one concise scope sentence: “This evaluation tests role-label decoding on held-out neutral texts; it does not validate the probe’s probabilities as estimates of authority on the gardening or permission inputs.” No further experiments are a condition of this editorial correction.

## Evidence checked independently

- I read the entire frozen report before forming this verdict. I did not read the other report-r1 verdicts or the narrative card.
- I inspected the saved offset formula, rendering source, provenance, and layer-12 coefficient rows. The operation is an affine shift followed by the same probe’s softmax. The draft correctly distinguishes that arithmetic from a propagated model intervention, projection removal, or behavioral evidence.
- I checked the actual RH6 input file against the completed run’s contract. Its SHA-256 is `be9254d581e88715478bfd825b56058558fba2babe09f7cb2d204ceccd2468aa`, matching the recorded contract. All 400 permission-condition prefills contain the exact sentence in R03-01; the 1,200 total prefills reuse 100 template–page pairs. I inspected the construction and aggregation in `rh/rh6_readings.py`, including replacement of both commands and averaging probabilities before logs.
- I checked the saved `user_turn_effects.csv` rather than relying only on a previous audit. At layer 12, marker first, the control-adjusted User-probability effects are -1.394785 and +4.457623 percentage points after rounding to six decimals for the five-role and three-role readouts, respectively. The raw marker effects also have the same opposing signs, so the direction of that illustrative statement is not an artifact of silently treating a difference-in-differences as an unadjusted change.
- I checked R03-02 by direct algebra. Two tokens can have `(P_User, P_Tool, P_other)` equal to `(0.45, 0.05, 0.50)` and `(0.05, 0.45, 0.50)`. Changing only the third-class logit of the second token can make the latter `(0.005, 0.045, 0.95)`. Both tokenwise User/Tool ratios remain 9 and 1/9, but the log ratio of span means moves from 0 to about 1.5664. This is a mathematical counterexample to denominator independence, not an additional model result.
- I inspected the current estimator audit, which explicitly leaves convergence unestablished. I checked the E9 backlog and source contract for the separation of vector families, deferred execution, and the difference between prepared candidates and eligible measured conversations.
- I checked the recovered September 5 raw compressed episodes. The `tool_03` arm has 21 episodes across eight cases and 15 distinct initial prompts; 17 episodes write both markers. The minimum initial command-token Tool probability is 99.8993099%. These support the report’s narrow adverse-evidence statement.
- I visually inspected the permission and dose-response figures. Their displayed boundaries agree with the prose. I did not load a model, run an experiment, or alter source evidence.

## Scientific uncertainty already correctly disclosed

The draft explicitly labels its emphasized comparisons exploratory and selected after inspection. It limits the gardening pattern and the offset means to one conversation, preserves the numerical replication gap and unresolved averaging denominator, and names style/content/position alternatives. The score probabilities are correctly distinguished from obedience probabilities. The offset section correctly identifies direct reuse of the classifier’s own weights, fixed normalization, a separately fitted five-role readout, and the absence of subsequent model computation.

The softmax gauge explanation is mathematically correct: a common shift of all coefficient rows leaves predictions unchanged while changing an intervention built from an individual row. A fitting convention or penalty may select a saved representative, but that does not establish unique semantic User and Tool axes. No removal of the current illustration or new gauge-invariant experiment is necessary for this report’s stated claim.

The permission section already discloses approximate span matching, lexical overlap, position changes, template–page coupling, absent action outcomes, reused units, and pointwise rather than simultaneous intervals. The validation section correctly separates recall from probability, prompt overlap from a causal split comparison, and correctness from calibration. The future-work section correctly treats propagated persistence and useful mitigation as different evidence requirements, preserves the two vector families, and refuses to treat the 200 prepared candidates as completed measurements. The historical behavioral result is appropriately restricted to an incomplete recovered cohort.

## Verdict

**Revise — four bounded explanatory changes.** The core measurement contribution and existing causal boundaries are supportable. R03-01 makes the tested cue concrete; R03-02 explains a consequential property of the actual estimator; R03-03 prevents a causal attribution to class count; R03-04 limits the interpretation of validation to its evaluated domain. These are writing repairs within the existing direction. None requires new science, new compute, or a changed estimand before this report can be approved as a review draft.
