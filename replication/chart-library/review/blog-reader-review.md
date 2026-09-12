# My reader review of the six role-signal figures

I inspected all six current PNGs, the library HTML, its README, the evidence audit, and the gardening result on September 11, 2026. This is a **Distill** review: my north star is to make the difference between a readable role signal, a classifier response, and a model intervention clear on the first reading. I apply the Neel lens by giving each figure one question, making the evidence unit visible, and preserving the numerical mismatch and measurement weaknesses.

I reviewed the saved HTML source and the rendered figure images; I did not repeat browser interaction checks. I changed no figures, data, experiments, or methods. The blog draft was not yet present at this first review.

## My recommendation

I would retain the six figures in their current order and place a short explanation of the measurement before the first image. The sequence moves from a recognizable conversation to arithmetic on the same activations, then to independent context and probe checks. Its strongest contribution is an auditable worked example of why a role score needs an operational definition, not a claim that I have already found a reliable steering mechanism.

My three supported claims are:

1. **The CoT-style passages remain visibly distinguishable in this gardening example under three tag conditions.** This is one conversation with 512 matched display tokens per condition, measured with the four-role layer-12 probe. The published numerical probabilities are not reproduced.
2. **Offsets along the saved classifier rows produce direction- and size-dependent changes in the same tokens' scores.** The small, large, and dose figures are three views of one deterministic offline calculation. The selected direction directly changes the classifier's logits; the response does not establish a change to later model computation or behavior.
3. **The interpretation depends on the metric and the fitted probe.** Across RH6's 100 paired template–page inputs, the adjusted marker-permission contrast is positive in all 12 relative User/Tool configurations; absolute Userness has opposite signs between the five- and three-role probes at layer 12. Held-out recall also varies across roles and depth, and recall does not establish probability calibration.

## Definitions I would give before the figures

| Term | Explanation I would use | Misreading it prevents |
| --- | --- | --- |
| Role | The conversation format distinguishes instructions, user messages, assistant reasoning and answers, and tool output. I keep each passage's original label when changing its tags or saved state. | A blue point means the passage came from the User part, not that the probe currently predicts User. |
| CoT | Here, chain of thought is the authors' assistant-analysis passage in the fixed conversation. | A CoT score does not verify the faithfulness of an internal thought process. |
| Token and horizontal position | A token is a piece of text. In the scatter plots, I read the conversation from left to right; dots are text positions. | The x-axis is neither a hidden-state coordinate nor a series of independent trials. |
| Activation | The model represents each token with a vector of numbers at each layer. I saved these vectors at the specified internal location. | An activation is not the Userness score or an individual semantic neuron. |
| Probe | A separately fitted linear classifier reads an activation and assigns probabilities to its available role classes. | The classifier is a measurement tool; its score is not the model's stated belief or an authority level. |
| Userness / Toolness | The probe's probability for the User / Tool class, within its particular set of classes. | These numbers do not measure permission, obedience, or whether a command will run. |
| Offset | I add a vector to a saved activation, then score the edited vector. “1%” and “5%” describe the added vector's length relative to a reference activation length. | These percentages are not changes in model weights, probabilities, or success rates. |
| Layer | One processing stage in the model; the plots use zero-based indices. | Layer 12 in these files should not silently be treated as a universal semantic location or the historical steering site. |

I would define pre-MLP only if the technical detail is needed: the saved point is immediately before a layer's feed-forward subnetwork. The opening does not need GPU type, precision, solver details, or the source commit.

## Six sections with one question each

| Section and figure | Reader question and visual route | Evidence sentence I would keep visible |
| --- | --- | --- |
| 1. A conversation with recognizable role patterns — `00-gardening` | What is the probe reading? Read left to right within a row, then compare the same passages across correct, absent, and all-User tags. Yellow/orange dots remain high for many CoT tokens, but the second CoT passage is visibly less uniform. | I measured one author-provided conversation; the colors track the original passages and the y-axis shows only CoT probability. |
| 2. A small change to the saved state — `01-small-offsets` | What does ±1% do? Explain the top pair of rows as User-direction edits and the bottom pair as Tool-direction edits; within each pair, Userness is above Toolness. Read subtract / unmodified / add across the columns. | I now use a five-role probe on the all-User-tag condition; I edit the same saved 512 tokens and do not continue the model. |
| 3. The same calculation at a larger size — `02-large-offsets` | What changes at ±5%? Compare the same panel locations with the preceding image. The large positive User offset raises Userness across the original passage colors. | This is the same calculation and sample at another illustrative offset size, not a second experiment or an optimized setting. |
| 4. Summarizing the change without losing the comparison — `03-offset-dose-response` | Which changes are systematic within this example? Columns identify the direction; rows identify the readout; colors still identify original passages. Zero is the unchanged reference. | Each dot is a within-role token mean from the same conversation; lines connect only five evaluated offsets. |
| 5. A new contextual test with paired controls — `04-rh6-readout` | Does a permission cue move every role summary in the same way? First define the marker command, paired null span, and two orders; then read absolute Userness above relative User/Tool. | This is a different dataset: 100 template–page pairs, 12 repeated conditions per pair, actual model-forward readouts, and pointwise paired intervals. |
| 6. Checking the classifier that supplies the scores — `05-probe-recall` | How reliably does the probe identify known roles on held-out text? Each curve is the fraction of tokens of that original role classified correctly; compare layers within a panel first. | The two panels use different fitted probes and test cohorts; their difference is descriptive, and the four-role gardening probe is not itself plotted here. |

After these sections, I would end with the already deferred propagation question: does an actual intervention at one layer survive through later layers and change relevant behavior on an adequately justified cohort? I would distinguish that future dataset from the present saved-state illustrations, and retain both approved vector families as separate conditions.

## Material reader issues to resolve in prose

- **The four-to-five-class transition needs its own sentence.** The gardening chart uses System/User/CoT/Assistant; the offsets include Tool as a fifth class. The center panels in the offset charts are a common baseline for those charts, not the same readout as the first figure. RH6's UAT probe has User/Assistant/Tool; SUCAT has all five. I would expand both acronyms where the reader first sees them.
- **The small/large charts need a reading instruction immediately above them.** Their 12-panel layout is readable enlarged but demanding inline. I would give one concrete panel example, such as “the upper-right panel shows Userness after adding the User direction,” and retain PNG/PDF enlargement links. No new layout or recoloring is necessary for the draft.
- **Positive Tool and User changes are compatible.** With five classes, both probabilities can increase while the other three decline. A Tool offset does not mathematically require Userness to fall. I would explain this before calling the dose pattern surprising.
- **“Subtract” is an opposite displacement.** It is not projection removal or erasure of a role. I would place the precise equation after the plain-language description: `h′ = h + s·r·w/||w||`, where `s` is ±0.01 or ±0.05 and `r` is the median saved activation norm in the all-User-tag condition. I would describe `w` as the saved classifier row, not a uniquely identified semantic direction.
- **The RH6 contrast needs a concrete construction.** I would name the marker as the instruction to write a dummy marker file, explain that permission and neutral user turns are paired, and explain that I subtract the same cue change measured on corresponding neutral text. The explicit permission applies only to that marker action; the other command requests an upload. “Marker first” means first among the two tool-page command slots, not first message in the conversation. Zero means no extra cue effect beyond this null control. These commands were present as research text and were not executed in the RH6 readout run.
- **“Absolute” is still a probe probability.** It distinguishes `P_user` from the relative `ln(mean P_user) − ln(mean P_tool)` score; it does not mean an absolute semantic quantity. The ratio uses span means before taking logs. Its units are not percentage points.
- **The sample changes twice.** Figures 1–4 share one conversation; Figure 5 uses 100 paired inputs; Figure 6 is held-out classifier evaluation from 249 base texts. I would state each change in the main text, not only behind HTML details. The 1,200 RH6 prefills and 512 display tokens are not independent units.
- **The adverse numerical evidence belongs in the body.** The gardening display subset averages 64.33%, 71.78%, and 71.72% CoTness in condition order. The paper's corresponding numbers are higher, with a no-tags discrepancy between its figure and prose; I would preserve this mismatch without implying that the recall dip already explains it. A table must specify whether it uses the displayed 177 CoT tokens or all 179.
- **Recall is a different y-axis.** An 80% recall means that 80% of held-out tokens originally labeled that role received the correct top class; it is not mean role probability. Prompt splitting can place other role-rendered versions of a held-out text in training. The grouped text split holds out the base text across roles, but the two panels are not a paired test of the splitting choice.

## Visual and HTML observations

The palette is consistent: blue User, yellow/orange CoT, and green Assistant identify original passages; System gray and Tool purple appear only in the five-role recall chart. RH6's gray lines and two line styles correctly avoid assigning nonexistent passage roles to its series. I would preserve these meanings and add no colors for offset strength.

The current dose title avoids claiming that larger offsets always blur role separation. I would retain that correction: positive Tool offsets increase the range of mean Toolness between passage roles in this example. The plot supports a direction-dependent pattern, not universal monotonic loss of separation.

All six PNGs have legible labels at enlarged size and consistent boxed panels. The two 12-panel scatter figures need their prose captions and enlargement links when embedded in a narrow Markdown column. The first figure's “Experiment 1/2/3” labels are source-style condition labels within one worked example; I would not describe them as three independent replications.

The HTML hides the large-offset and probe-validation charts behind switches. The Markdown draft should embed all six images, so readers do not depend on interactive state or collapsed method details to reach the evidence. The page already links sources and states evidence types well; the blog should bring the definitions and class-set transitions into the reading path.

## Questions I would ask Hanan about the draft

1. Does the opening make the intended contribution clear: an explanation of what these role measurements establish and the next model-level question?
2. Can a reader identify the edited direction, measured probability, and original passage color without interpreting them as the same thing?
3. Is the gardening mismatch visible enough, and does the measured-versus-offline distinction remain clear before the later controls?
4. Does the next-step paragraph preserve the deferred adequately sized batch, rather than turning these illustrations into a new result claim?

I would ask these as review prompts on the completed draft; none requires changing the current research method or launching another experiment.

## My review of draft v0.1

I read the complete [blog draft](../role-signals-blog-draft.md), SHA-256 **`38b6da6972c807cfde23fe51584fa29b92e9ac54dd8d7a8cd6f16242af325fb4`**, against the six previously inspected figures. This review applies to that exact version. I checked the reader's evidence boundary and document structure; separate reviewers own the complete numerical and primary-source audits.

The draft resolves the prerequisites from my first pass: it defines activations and probes, distinguishes passage color from predicted role, explains offset units, makes the four-to-five-class transition explicit, identifies the independent samples, and separates probabilities from recall. It contains a TLDR, exactly six embedded images, captions numbered 1–6, figure-reading instructions, data/PDF links, and a review section. All reference-style citation labels resolve within the Markdown. The two offset images correctly share one narrative section; six charts do not require six separate chart sections.

I recommend two material wording patches before delivery:

### 1. Scope the permission headline to the plotted marker contrast

At line 11, “permission increases the relative User/Tool score across every tested configuration” is broader than the result plotted here. The full RH6 grid also includes the other command and other cue contrasts. Section 4's lead at line 106 repeats the unspecific formulation, although its later explanation correctly narrows the estimand.

I would replace the TLDR's experiment sentence with:

> In a separate experiment with 100 paired template–page inputs, permission increases the marker command's relative User/Tool score beyond its paired neutral-span control in all 12 plotted configurations.

I would replace section 4's bold lead with:

> In the 12 plotted settings, marker permission raises the relative User/Tool contrast beyond the neutral-span control, while absolute Userness can decrease.

This preserves the observed positive result and makes its command, control and configuration scope visible without requiring a reader to infer them from the caption.

### 2. Make the RH6 stimulus and first/second order concrete

At line 108, “One command writes a marker” can initially read as an executed action, and “first or second” has no named counterpart. “Prefill” also appears later without a definition. The existing later statement about missing action outcomes limits the claim, but a cold reader still has to reconstruct what was actually run.

I would replace the RH6 setup paragraph with:

> This section switches to a separate dataset and actual model forward passes. I use **100 template–page pairs**, each containing controlled Tool content with two instructions: one asks for a dummy marker file to be written, and the other asks for an upload. I compare a neutral User turn with a User turn that permits only the marker action, and test the marker listed first or second within that pair of instructions. I process each prepared conversation, or **prefill**, to record activations; I execute neither instruction and generate no continuation.

The remainder of the section can stay as written. In particular, its paired difference-of-differences, exact relative metric, probe-dependent sign reversal, pointwise intervals and approximate matching limits are clear.

I found no other material cold-reader defect in this version. I would keep the remaining prose stable rather than add further caveats, reorganize the six figures, or introduce another experiment.
