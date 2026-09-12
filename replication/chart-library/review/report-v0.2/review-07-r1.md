# Independent review 07 — figures and scientific visual communication

Report: `draft-r1.md`  
SHA-256: `022c1f4958500451a8568f2749e7878964f69607b023c692130acac016c7db4b`  
Lens: Scientific figure design, caption and axis semantics, faithful paper styling, and accessibility.

## Strongest accurate contribution

I find the strongest contribution in the concrete demonstration that probe choice and score definition are part of the measurement: the permission comparison has positive relative effects across the displayed settings, while absolute Userness can change sign across probe class sets. The saved-state offset example gives a useful second illustration: Userness and Toolness can rise together at a small positive Tool offset, and the larger offset has a different effect. The report consistently keeps those observations separate from downstream model behavior.

## My retelling of the message

This report uses saved GPT-OSS-20B measurements to ask what a role-probe score actually tells us. One gardening conversation retains visible reasoning-role structure under changed tags, but the local probabilities do not numerically reproduce the paper. Direct offsets to those saved states reveal classifier responses whose direction and size depend on the vector, strength, passage group, and readout; they do not show propagation through the model. A separate paired permission comparison makes the measurement dependence systematic across the plotted settings. Held-out role recall varies across roles and layers, and neither recall nor an extreme probability establishes calibration, obedience, or selective action control. The next informative step remains a sufficiently sized propagation experiment under an explicit source contract.

## Inspection and constraints

I read the entire frozen draft and visually inspected all six linked PNGs. I compared them with the preserved renders of the original paper’s page 6 (Figure 7) and page 25 (Figure 23), inspected the current rendering code and saved offset means, and verified that all six exported PDFs embed TeX Gyre Termes. The original notebook defines User `#00a6f4`, CoT `#fd9a00`, and Assistant `#00d492`; the current rendering code preserves this mapping. The boxed panels, light grid, raw points, and serif type already form a coherent continuation of the paper’s style. Figure 5 appropriately uses a neutral style to distinguish command orders, and Figure 6 adds real System and Tool series. I recommend retaining the style and both detailed small/large offset views. I did not execute a renderer, model, notebook, or experiment, and I did not read other review verdicts.

## Recommendations

### R07-01

**Exact passage / figure element:** Figure 1 image panel titles: “Experiment 1: Correct tags”, “Experiment 2: No tags”, “Experiment 3: All in user tags”; caption: “These are our measured activations, using the authors’ Figure 7 setup.”

**Concern:** The exported PNG closely reproduces the original paper’s panel titles and has no embedded model, probe, sample, or local-result identifier. Detached from its Markdown caption, it is easy to mistake it for the paper’s Figure 7. The experiment numbering also makes three renderings of one conversation sound like three independent experiments.

**Suggested remedy:** Keep the paper’s three boxed scatter panels and exact passage palette. Add a compact TeX Gyre Termes subtitle such as “Local GPT-OSS-20B readout · layer 12 · four-role probe · 512 displayed tokens from one conversation”. Label the panels “Correct tags”, “No tags”, and “All text in one User message”, or use A/B/C condition labels. Retain the explicit attribution to the authors’ gardening conversation and the numerical replication gap in the caption/body.

### R07-02

**Exact passage / figure element:** Figure 2 caption: “The 1% label specifies offset length relative to a fixed activation-norm reference”; Figure 3 caption: “with 5% rather than 1% offsets”; Figure 4 x-axis: “Signed offset (% of reference norm)”.

**Concern:** The captions say that the percentages are not probability changes, which is helpful, but the reference population and actual lengths are only defined in Appendix A. A reader inspecting the offset figures cannot recover what makes 1% small or what five times that length means without leaving the figure discussion.

**Suggested remedy:** Add one short shared definition adjacent to Figures 2–3 and in their export footer: “Reference norm = 45.2471, the median across 921 forwarded tokens; 1% and 5% give offset lengths 0.4525 and 2.2624.” State that these are the same fixed lengths for every displayed token and both directions. Give Figure 4’s caption the same reference definition or an explicit nearby cross-reference. Preserve the common 0–100% probability axes and the separate 1% and 5% figures.

### R07-03

**Exact passage / figure element:** Figure 4 image title: “Offset direction and size change the role-score pattern”; Figure 4 caption: “A small Tool offset raises both scores for the original User passages.”

**Concern:** The central result is only a 3.2 percentage-point Userness rise and a 3.3-point Toolness rise at +1%. On the full probability axes, the latter sits near the baseline and the former is a shallow bend. The broad figure title and unlabeled points make the claim much harder to see than the adjacent table. The +5% reversal is much more visually prominent.

**Suggested remedy:** Keep the full axes, raw evaluated points, all three original-role series, and the existing statement that intermediate strengths were not measured. Add a compact callout or direct labels for the blue original-User series in the Tool-direction column: at 0%, +1%, +5%, Userness is 70.2%, 73.4%, 30.4%, and Toolness is 1.5%, 4.9%, 67.7%. A concise callout may instead say “0→+1%: Userness +3.2 pp; Toolness +3.3 pp”. Use a takeaway title such as “A small Tool offset raises both scores; the larger offset lowers Userness”. Keep the current table as the accessible numerical companion. Do not zoom away the common probability baseline or invent measurements between the five strengths.

### R07-04

**Exact passage / figure element:** Figure 1 caption: “Colors identify original passage roles”; Figures 2–3 image footers: “Colors identify the original passages: User, CoT and Assistant”; Figure 4 and Figure 6 legends use the same circular marker for every role.

**Concern:** The exact paper palette is correctly retained, but role identity relies too heavily on color. In the passage figures, the colored text “The user…” is actually CoT, so the excerpt does not independently identify its source role. In the dose and recall plots, overlapping lines have identical marker shapes; the blue User and green Assistant means nearly coincide at some points. This remains difficult in monochrome or when the colors are not distinguishable to a reader.

**Suggested remedy:** Preserve User #00a6f4, CoT #fd9a00, Assistant #00d492, TeX Gyre Termes, boxed panels, and the light grid. Add dark text role labels at the passage boundaries, for example “User 1”, “CoT 1”, “Assistant 1”, followed by the existing excerpt where space permits; repeat consistently for turn 2. For Figures 4 and 6, add a restrained redundant cue, such as distinct marker shapes or direct role labels with leader lines, and show it in the legend. Keep real System and Tool series gray and purple only where those roles are actually measured; retain Figure 5’s neutral order encoding. Check the revised labels at report display width, especially the short first-User span in the 12-panel figures. Do not replace the requested palette with a generic accessible palette.

### R07-05

**Exact passage / figure element:** Figure 5 image title: “Relative role shifts agree; absolute Userness depends on the probe”; y-axes: “Change in Userness (percentage points)” and “Change in log(User/Tool)”; caption: “Bottom: changes in the natural-log User/Tool ratio.”

**Concern:** “Agree” leaves unclear that the agreement is in sign, rather than effect size or layer pattern. The axes omit that the plotted quantity is the command-minus-neutral-span adjusted permission effect. The log label also does not identify the order of averaging and logarithms, although Appendix A correctly distinguishes those estimators. These ambiguities matter because the report’s main lesson is that the readout definition changes the conclusion.

**Suggested remedy:** Use an affirmative title such as “Adjusted relative effects are positive in all 12 settings”. Label the rows “Adjusted Userness effect (percentage points)” and “Adjusted ln(User/Tool) effect”, keeping the definition of the subtraction in the subtitle/caption. Add a short caption sentence: “Within each span, User/Tool is the ratio of mean User probability to mean Tool probability; logarithms precede the paired contrast.” Retain the existing pointwise paired-bootstrap interval definition, common scale within each row, and solid/filled versus dashed/open order encoding. State in the caption that lines join the three measured layers; intervening layers are not displayed.

### R07-06

**Exact passage / figure element:** Figure 6 image title: “Probe quality varies by role, depth and split”; subtitle: “My H100 run”; y-axis: “Correct-role classification”; caption: “Probe reliability varies by role and layer” and “Blue, yellow and green retain their passage roles”.

**Concern:** The body correctly defines recall and separates it from calibration and convergence, but “quality” and “reliability” broaden the figure’s claim beyond the measured quantity. “Retain their passage roles” also carries the gardening-passage language into a different evaluation of true role-labeled neutral texts. “My H100 run” identifies hardware while leaving the model unnamed in the exported image.

**Suggested remedy:** Use “Held-out role recall varies by layer and split” as the image title and caption lead, and “Recall within each true role” as the y-axis. Replace the hardware subtitle with “GPT-OSS-20B · five-role probes · all 24 layers · 249 base texts in the full corpus”. Say that colors identify the true role label of the held-out tokens, with gray System and purple Tool. Preserve the different-set/separately-fitted-probe caveat and the token-weighting statement; this recommendation does not require new calibration analyses or uncertainty intervals.

## Scientific uncertainty already disclosed correctly

- The gardening example is one authored conversation; the displayed subset and full-content summaries differ, and neither resolves the paper’s numerical discrepancy or separates style from content and position.
- The offset figures rescore saved activations using the saved classifier parameterization. They establish neither unique semantic axes nor model propagation or behavior. Five strengths on repeated tokens are not independent conversations; the absence of population confidence intervals here is appropriate.
- The permission intervals are pointwise over 100 reused paired units, not a simultaneous guarantee or 1,200 independent examples. Replacement length, lexical overlap, position, and template–page coupling remain unresolved.
- Recall is not calibration. The optimizer convergence question and the distinct fitting/evaluation sets remain visible.
- The Figure 23 probability-versus-accuracy and sample-count discrepancies remain open, and the proposed extension is clearly deferred.

## Verdict

**Revise.** I recommend six bounded presentation changes. The existing evidence can support the report after these changes; my figure review does not require new science, a replacement palette, or removal of the small/large offset displays.

## Sources inspected for this lens

- The frozen `draft-r1.md` and all six PNG/PDF figure pairs in `chart-library/figures/`.
- `chart-library/render_library.py`, `chart-library/render_rh6.py`, `chart-library/data/offset-dose-response.csv`, and the current saved-state offset method implementation.
- Original paper page renders 6 and 25 under `01-sources/papers/markdown/prompt-injection-as-role-confusion/2026-09-04/assets/pages/`; the source Figure 7/23 text in the preserved Markdown paper.
- The original `experiments/cot-forgery-role-confusion/04-analyze-injection-probe-results.ipynb` palette definitions, inspected statically.
- The supplied Neel Nanda writing guidance, especially informative figures and layered explanations (lines 17–22).
