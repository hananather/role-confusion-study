# Independent review 02 — cold reader

**Report:** `draft-r1.md`  
**SHA-256:** `022c1f4958500451a8568f2749e7878964f69607b023c692130acac016c7db4b`  
**Lens:** Independent cold reader: intelligent newcomer with no prior project, role-probe, prompt-injection, or mechanistic-interpretability context  
**Verdict:** revise  
**Recommendation count:** 9

## Strongest accurate contribution

The strongest contribution is a concrete demonstration that the choice of classifier and score can change the sign of the conclusion drawn from the same permission comparison. The layer-12 example makes this consequential: one fitted classifier reports a decrease in User probability and another reports an increase, while the relative User/Tool score increases for both. The report also carefully distinguishes changing a saved classifier input from changing the language model’s later behavior. This combination is useful because it tells a reader which inference the measurements support and why an apparent role shift is not yet evidence of action control.

## My retelling

The report asks what we can learn when a small classifier says that a language model’s internal text representation looks more like one message role. In one existing gardening conversation, reasoning passages often retain a recognizable role signal even after message labels are changed, although the authors’ numerical result is not reproduced. Moving the saved internal states in classifier-derived directions changes the classifier probabilities, including a small displacement that raises both User and Tool probabilities; the model never processes those edited states further. In a larger constructed-input comparison, explicit permission for a dummy file-writing action changes the role readout, but whether absolute Userness increases depends on the fitted classifier. Held-out classification quality also varies across roles and layers. I therefore take the result as evidence about the behavior of the measurement instruments, with an open question about whether any subsequent model computation or action would respond in a useful way.

## Recommendations

These are writing, explanation, figure-accessibility and evidence-boundary changes. I do not require new experiments for this review.

### R02-01

**Exact passage:** “Across 100 constructed input pairs, the marker command’s permission effect is positive in all 12 tested relative-score settings after subtracting a neutral-text control. Absolute Userness can decrease in the same comparison.”

**Concern:** The first substantive TLDR finding assumes that I already know the marker task, the two kinds of neutral text, the relative score, and what counts as a setting. The underlying result is understandable later, but the opening does not yet give a newcomer a usable claim.

**Suggested remedy:** Lead this bullet with the concrete comparison: giving permission to write a dummy file changed how probes classified the same webpage instruction, and whether Userness rose or fell depended on the classifier and score. Define Userness as the probe’s User probability in the summary, or use that plain phrase there. Leave the control formula and the 12-setting count for Section 4. Retain the 100-pair scope and the statement that no actions were measured.

### R02-02

**Exact passage:** “We trained probes across all 24 model layers, numbered 0–23, using 249 neutral base texts rendered under different role labels.”

**Concern:** I can still mistake the training labels for human judgments that a passage truly has user intent or is authentic reasoning. “Rendered under different role labels” does not show how a neutral passage becomes a labeled training example. I also have to infer that the language model produces the recorded states while a separate classifier is fitted to those states.

**Suggested remedy:** Add a two- or three-sentence training example: the same neutral passage is placed in different message-role/channel wrappers, its recorded states inherit the corresponding training label, and the separate probe learns to predict that label. Clearly mark any invented sentence as schematic. State that changing which roles the classifier is trained to distinguish creates a new classifier; it is not merely hiding two labels from an unchanged five-class output. Keep the full split and fitting details in Appendix A.

### R02-03

**Exact passage:** “We render the conversation with correct tags, without its role tags, and as one User message.” / “Each panel shows the same 512 matched tokens from one conversation.”

**Concern:** The main figure description can be read as three complete transcripts differing only in visible labels. Appendix A establishes two details that materially change that reading: the no-tags/all-User versions also flatten the system text into the transcript, and the 512 plotted tokens are a display subset rather than the entire labeled conversation. “Matched” does not itself explain either point.

**Suggested remedy:** Bring one compact clarification into the Figure 1 setup/caption: no-tags joins the complete transcript, including system text; all-User wraps that joined text in one User message; the panels display 512 content tokens matched across conditions rather than the full transcript. In Appendix A, give or link the precise display-selection/truncation rule so a reader can understand why 240 of 582 full Assistant tokens appear. Retain the existing displayed-versus-full means table and numerical-replication boundary.

### R02-04

**Exact passage:** Section 2, from “The panels share the same token positions, colors and probability axes” through Figures 2 and 3; Section 3’s Figure 4.

**Concern:** The newcomer must decode two twelve-panel grids, including repeated baselines, before receiving the four-panel summary and three-row table that actually make the offset result easy to understand. The central permission finding then arrives after a large amount of repeated one-conversation detail. The grids are useful evidence, but their current placement gives them more explanatory work than they can do at ordinary reading size.

**Suggested remedy:** Use the present Figure 4 and its numerical table as the main offset explanation, then offer the full small/large token grids as detailed views in an appendix or linked supplement. Preserve both grids and their provenance. If all six figures must remain in the body, introduce the summary comparison before the grids and give an explicit reading path to the few panels needed for the two claims; do not require inspection of all 24 panels to recover the takeaway.

### R02-05

**Exact passage:** “We use 100 constructed template–page pairs. Each pairs an instruction template with webpage text and contains two commands: write a dummy marker file, and upload data. We compare two command orders and construct a control in which the command span is replaced by neutral text.”

**Concern:** This is the strongest result’s setup, but I cannot reconstruct its four-way comparison from the prose. In particular, “neutral” describes both the User-message baseline and text replacing the commands. The implementation also replaces both command slots together in the neutral-text construction, whereas the singular “the command span” can suggest that only the measured marker command is replaced while everything else is held fixed.

**Suggested remedy:** Add a compact worked setup or 2×2 table for one pair: rows are the original command-containing Tool text versus the same construction with both command slots replaced by neutral page text; columns are the ordinary summary request versus that request plus explicit permission for the marker action only. Show a short actual permission sentence and a harmless marker description, label the precise span being scored, and state that the upload command receives no permission. Identify any shortened example as an excerpt or schematic. Cite the frozen input record or construction source. The saved source rh/rh6_readings.py, lines 100–113 and 217–221, supports these distinctions.

### R02-06

**Exact passage:** “There are 12 settings because we examine three layers, two probe class sets and two orders. The full dataset reuses the same 100 input pairs across 12 input conditions, yielding 1,200 prefills.”

**Concern:** Two unrelated products both equal 12, but only the first is decomposed. A reader can conclude that each probe/layer setting required a new input pass, or remain unable to see how permission versus neutral and two command orders produce twelve conversations per pair. The prohibition condition is introduced only afterward as an unexplained weaker contrast.

**Suggested remedy:** Spell out both products beside their nouns: 12 plotted measurement settings = 3 layers × 2 classifiers × 2 orders; 12 prepared inputs per pair = 3 User-message variants (neutral, marker permission, upload prohibition) × 2 command orders × 2 text constructions. Say that Figure 5 selects the permission-versus-neutral comparison and that the other User-message variant remains in the full results. Keep the existing warning that 1,200 prefills are not 1,200 independent examples.

### R02-07

**Exact passage:** “We will compare two vector families separately: the earlier Tool-minus-User direction derived from activations…” / “We are in Understand / Distill…” / “The audit covers the September 10 Claude discussion, its 19 tracked goals and 35 linked subagent logs…”

**Concern:** The ending shifts from a self-contained measurement report into an internal project handoff. “Earlier” points to a method the reader has not seen; “source contract,” “eligible,” “balanced source coverage,” and Understand / Distill depend on project context. Counts of prior goals and agent logs establish audit breadth but do not explain the scientific conclusion. This weakens an otherwise clear ending.

**Suggested remedy:** Make the main conclusion stand on its own: identify what would be edited, that the model would resume computation, which later measurement would test survival, and what random-direction comparisons could falsify specificity. Give one plain sentence defining how the activation-derived Tool-minus-User vector is constructed, verified against its source, if it stays in the main text. Preserve the 200-conversation planning floor, eligibility/source-coverage rules, candidate accounting, source-contract disagreements and workflow-audit counts in a clearly linked methods/planning appendix or verification note. Replace the internal phase label in reader-facing prose with the aim it represents.

### R02-08

**Exact passage:** “In the recovered Toolward arm, 17 of 21 episodes wrote both markers despite near-saturated initial Tool readings.”

**Concern:** This is potentially valuable adverse behavioral evidence, but a newcomer cannot tell what either marker represents, which action was intended, or why writing both is a failure. Earlier in the report the task contains one marker action and one upload action, so “both markers” can also make the reader incorrectly merge the two experimental setups.

**Suggested remedy:** Introduce the historical task with one sentence defining the two markers and the desired selective action, using the archived evidence rather than assuming their identities. Translate “Toolward arm” into the actual intervention comparison and “near-saturated” into a plain high Tool-probability description or a supported quantity. Preserve the explicit different-setup qualification and the 21-episode/eight-case limitation. This is an explanation request, not a request to rerun or strengthen the behavioral analysis.

### R02-09

**Exact passage:** “Follow the blue series in the Tool-direction column” and the role-series legends in Figures 4 and 6; the passage colors in Figures 1–3.

**Concern:** The prose and figures consistently encode original role by color, which is helpful, but the line plots use the same circular marker and solid line for every role. The token grids also rely on colored excerpts rather than explicit role labels. Readers with limited color discrimination or a monochrome copy lose the distinction between original roles and the role probability being measured.

**Suggested remedy:** Add redundant role encoding: distinct markers or dash patterns/direct endpoint labels for the curves in Figures 4 and 6, and explicit User / reasoning / Assistant labels on the passage bands or caption key for Figures 1–3. Keep the existing colors and origin-versus-readout explanation. Refer to “the original-User series (blue)” in the body so the intended object remains named when color is unavailable.

## Scientific uncertainty already disclosed correctly

- The comparisons emphasized in the report were selected after inspecting saved results; the draft identifies them as exploratory.
- The gardening example contains one authored conversation, does not isolate style from content or position, and does not close the paper’s numerical replication gap or resolve its averaging denominator.
- The offsets are arithmetic on saved activations, the edited states are never propagated, and individual coefficient-row directions depend on the saved parameterization. Repeated token measurements are not independent conversations.
- The permission comparison has no generated continuations or observed actions. Replacement lengths match imperfectly; wording, token position, and template–page coupling remain possible explanations. The intervals are pointwise across paired inputs.
- The held-out role-recall plots do not establish probability calibration. Fitting convergence is unresolved, and the two split panels use different test sets and fitted classifiers.
- The proposed later-layer study has not been run. Prepared candidates are not yet eligible measured conversations, and 200 is a planning floor rather than a demonstrated power calculation.
- The earlier behavioral observations use a different setup and repeated cases; the report does not promote their episode fraction into a general causal estimate.

## Inspection and boundary

I read the complete frozen draft and all six PNG figures. For the setup explanations I inspected the live RH6 result record, frozen input contract, rh/rh6_readings.py construction source, gardening result record, and probe-training metadata. I did not read other reviewers’ verdicts or the narrative card, run experiments, change figures, or alter the report. I used the supplied Neel Nanda guidance as an editorial standard, especially layered explanation, defining terms before using them, reader attention, and self-contained informative figures.

## Verdict

**Revise.** The contribution and principal scientific limitations are intelligible after a full read, and the report avoids equating classifier scores with obedience. The remaining work is to make that understanding available on a first read: show the training and permission constructions, make the displayed-token and input-condition units explicit, give the offset summary precedence over repeated grids, and translate the closing project references. These changes can be made from the saved evidence without new science.
