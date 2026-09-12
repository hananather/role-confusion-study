# Independent review 10 — frozen report r1

**Lens:** End-to-end coherence, scientific usefulness, adverse evidence, coverage, and next-experiment logic for a reader with no project context

**Report SHA-256:** `022c1f4958500451a8568f2749e7878964f69607b023c692130acac016c7db4b`

**Verdict:** Revise with six bounded clarifications. The report already has a coherent, scientifically useful story. I do not recommend a new experiment, another figure, a change of central claim, or a major reordering as a condition of this writing review.

## Strongest accurate contribution

The strongest contribution is the controlled permission comparison: across the same 100 constructed template–page pairs, the adjusted marker permission contrast is positive in the relative User/Tool score in all 12 measured settings, while layer-12 absolute Userness changes in opposite directions under the five-role and three-role probes. That is a concrete result about these measurement choices. The single-conversation offset analysis makes the lesson legible: because the classifier has several classes, Userness and Toolness need not trade off, and coefficient-directed displacement directly changes the chosen classifier. The report appropriately stops short of a new mechanism or mitigation claim.

## My retelling for a new reader

A classifier can reveal structure in a model’s internal states, but its output must be interpreted together with the classifier, metric, and manipulation. In one gardening conversation, reasoning text retains a visible role signature after message tags change, although local probabilities fall short of the paper’s reported values. Moving saved states along probe coefficient vectors produces magnitude-dependent changes; a small Tool offset can raise both Userness and Toolness, whereas a large one creates a tradeoff. Across 100 constructed inputs, a legitimate permission cue increases the marker’s relative User/Tool reading consistently across the chosen grid, even though absolute Userness can increase or decrease depending on the classifier. Held-out role recall varies, calibration and convergence are not established, and none of the six charts measures a behavioral effect of steering. The useful next step is a deferred, adequately sized test of whether separately specified vector families produce direction-specific downstream changes beyond matched random perturbations; behavioral usefulness remains a separate question.

## What I inspected

I read all of `draft-r1.md` and visually inspected all six PNGs. The small- and large-offset grids have the same arrangement and axes, and the dose plot makes their nonmonotonic example easier to follow. All six figures earn a place: the gardening example supplies concrete states, Figures 2–4 explain the arithmetic response, Figure 5 supplies the larger controlled input comparison, and Figure 6 bounds classifier reliability. I found no need to replace their existing paper-style presentation.

I inspected the current RH6 results and underlying `user_turn_effects.csv`, the dose-response CSV, the saved estimator audit, the E9 source contract and backlog, the historical report and raw recovered episodes, and the actual Figure 23 plotting cell in the authors’ notebook. Read-only checks confirmed 12/12 positive adjusted marker-permission relative intervals, 10/12 upload-prohibition relative intervals containing zero, the stated 0/+1/+5 Tool-direction means, and 17/21 historical Toolward episodes with both authorized and unauthorized writes across eight cases. These checks support the recommendations below; I did not rerun the full numerical audit. I also opened the [v6 primary paper](https://arxiv.org/html/2603.12277v6), and applied the supplied Neel Nanda writing guidance. I did not read other reviewers’ verdicts or a mutable report draft.

## Recommendations

### 10.1

**Exact passage:** TLDR, line 10: “Adding a vector to saved internal states raises both scores for the original User passages; a larger offset makes them move in opposite directions.”

**Concern:** The summary omits the edited direction, aggregation, and single-conversation scope. Those details determine the result: the cited increase is in the mean Userness and Toolness of original User tokens under a positive Tool coefficient offset. The generic wording can sound like a broad property of role vectors, or a tokenwise effect.

**Suggested change:** Replace with: “For the original User passages in one gardening conversation, a +1% Tool coefficient offset raises mean Userness and Toolness; at +5%, Toolness rises while Userness falls.” Retain the statement that this is offline classifier rescoring. The exact norm definition can remain in the body and appendix.

### 10.2

**Exact passage:** Section 4, line 123: “The full table retains the weaker prohibition contrasts as well.”

**Concern:** The weaker companion result is preserved only through a vague sentence and a link. A reader cannot tell whether it is a smaller consistent effect, an opposite effect, or largely unresolved. That matters because the positive permission comparison was selected after inspection and is the report’s strongest aggregate result.

**Suggested change:** Give the result in one sentence: “For the upload-prohibition comparison, 10 of 12 relative-score intervals include zero; the other two are small positive shifts at layer 16 with the three-role probe.” Retain the full-results link and state that intervals including zero do not establish no effect. No new figure or experiment is needed.

### 10.3

**Exact passage:** Section 5, lines 127 and 135: “We therefore check how often the five-role probes correctly classify held-out role-labeled tokens across all 24 layers.” / “These are the five-role probes; they are not the four-role readout used for the gardening overview.”

**Concern:** The section title asks how much to trust the probes, but the scope of this validation is only partly spelled out. This is held-out classification in a corpus of neutral text rendered under role tags. It is not direct validation of semantic-role interpretation under misleading tags, permission cues, or edited activations. Figure 6 also does not show the three-role classifiers used in the headline RH6 comparison.

**Suggested change:** Add a compact boundary: “These recalls test recovery of assigned roles in the neutral, role-rendered corpus. They do not directly validate role interpretations for misleading tags, permission cues, or displaced activations.” Extend the existing classifier-scope sentence to identify both the four-role gardening probe and the three-role permission probe as absent from Figure 6. Keep the six-figure scope; additional experiments are unnecessary for this clarification.

### 10.4

**Exact passage:** Next-experiment section, line 143: “Our next batch would extend the paper’s Figure 23, which compares role readouts across layers, by adding intervention curves for GPT-OSS-20B.”

**Concern:** For the intended reader, the figure number and “across layers” do not identify the baseline experiment. The bridge from one-conversation offset illustrations to a larger propagation experiment is therefore less concrete than the rest of the report. The source’s original-role rows and three formatting conditions explain both continuity with gardening and what the added interventions would contribute.

**Suggested change:** Describe the source comparison before naming the extension: “Figure 23 compares readouts of original User and Assistant content across layers under correct tags, no tags, and Tool tags. We would add intervention curves to this comparison for GPT-OSS-20B.” Preserve the separate probability/accuracy definitions, the deferred status, independent-conversation sample requirement, and separate vector-family analysis. This describes the existing source contract rather than choosing a new method.

### 10.5

**Exact passage:** Appendix B, line 175: “In the recovered Toolward arm, 17 of 21 episodes wrote both markers despite near-saturated initial Tool readings. Those episodes cover only eight cases and repeated prompts.”

**Concern:** The reader has not seen this historical task, so “both markers” does not itself explain what went wrong. The earlier marker/upload setup does not establish that the historical two-marker setup authorizes only one action. “Recovered” also leaves the incomplete allocation implicit.

**Suggested change:** Write that only one marker write was authorized in each episode, making the second write the adverse outcome. Describe the 21 episodes as an incomplete recovered subset spanning eight cases. Keep the different-setup boundary and the statement that this is not a general steering-effect estimate; do not replace it with an aggregate success-rate claim.

### 10.6

**Exact passage:** Review status and sources, line 183: “We are in Understand / Distill: the aim is to make each measurement and its consequence legible, with particular attention to clarity, source fidelity, skeptical interpretation and adequate samples.”

**Concern:** This sentence names an internal workflow and writing aspirations after the scientific story has already concluded. It adds no measurement, evidence boundary, or decision for a reader without project context, and gives the ending the tone of an audit handoff.

**Suggested change:** Move the workflow declaration to the linked verification or project-status record. Keep the useful figure-numbering and source instructions in the report. The scientific conclusion is already supplied by the deferred propagation test and need for separate behavioral evidence, so another summary paragraph is unnecessary.

## Scientific uncertainty already correctly disclosed

- Gardening is one authored conversation, and its numerical discrepancy with the paper remains unresolved. Neither repeated tokens nor three renderings supply independent conversations; style, content, and position are not isolated.
- The offsets are direct rescoring of saved states, not propagated interventions. Their coefficient-row directions depend on the saved parameterization, and the illustrated five strengths do not establish a population response or intermediate-dose behavior.
- The permission contrast is descriptive for 100 constructed template–page pairs. Span matching, lexical overlap, token position, and page/template coupling remain alternative explanations. The intervals are pointwise, and the report discloses selection after inspection.
- Five-role recall varies across roles and layers. Different train/test splits and fitted classifiers prevent a causal interpretation of panel differences. Recall does not establish calibration, and saved estimator checks do not establish convergence.
- Historical adverse behavior comes from a different setup and incomplete repeated-case allocation. It supports caution about equating a saturated score with selective control, not a general method ranking.
- The future experiment is deferred. Both vector families remain separate; the source paper and notebook disagree about sample count and plotted quantity; 200 prepared candidates are not 200 eligible measured conversations; and the 200-conversation planning floor still needs an effect-size and precision justification. The stated random controls, zero check, propagation falsifier, and additional behavioral requirement are appropriate. Final sites, strengths, endpoint, and sampling decisions belong in the future frozen protocol.

## Final verdict

**Revise.** The core contribution and six-figure story are supportable from saved evidence. The changes above make the scope and adverse findings easier to understand without expanding the research program. Preserve the numerical replication gap, weak prohibition result, unresolved validation issues, and earlier adverse steering evidence. No model loads, GPU jobs, or experiment execution are needed to resolve this review.
