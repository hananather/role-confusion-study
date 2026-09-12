# Independent review 01: narrative and explanation

Report: `draft-r1.md`  
SHA-256: `022c1f4958500451a8568f2749e7878964f69607b023c692130acac016c7db4b`  
Lens: narrative, strongest supported contribution, motivation, and explanations at several levels of detail  
Stage: Understand / Distill  
North star: a cold reader can state the supported result, name its measurement, and understand what it changes about the interpretation of role scores.  
Criteria: clarity, evidence matched to claims, prioritization, source fidelity, and informative figures.

## Strongest supported contribution

The strongest empirical contribution is the controlled permission comparison on 100 constructed template–page pairs. At layer 12, two separately fitted probe class sets assign opposite signs to the marker’s adjusted absolute-Userness effect, while the relative User/Tool effect remains positive across all 12 tested layer × probe × order settings. The report makes a concrete case that “more User-like” is incomplete unless the probe, score, and comparison are specified. That is useful when researchers choose a measurement for a later steering experiment.

The other analyses give this result context. The saved-state offsets show exactly how a multiclass classifier can raise Userness and Toolness together at one strength and separate them at another. The gardening example checks that a familiar qualitative pattern is present while preserving the numerical replication gap. Neither needs to be sold as a new mechanism to be useful. The report is strongest as an exploratory measurement study with an explicit route toward testing propagated effects.

## My one-paragraph retelling

We can read role information from GPT-OSS-20B’s activations, but the resulting probabilities answer questions about a particular classifier. One gardening conversation retains a visible reasoning-text pattern under changed tags, although the local averages do not reproduce the paper’s probabilities. When we directly displace saved states along probe coefficients, a small Tool offset raises both Userness and Toolness for originally User passages; a larger one raises Toolness while lowering Userness. A more substantial comparison changes the actual user input: across 100 constructed pairs, permission to write a marker changes its relative User/Tool score beyond a neutral-span control in all 12 tested settings. Yet the adjusted absolute-Userness effect can have opposite signs under different probe class sets. These observations make probe and metric choice part of the scientific result. They do not show that the model will follow an instruction after steering. The unresolved next question is whether a specified direction produces a distinctive effect after later model computation; behavior and legitimate-task performance would be needed to support a mitigation claim.

## Recommendations

### R01

**Exact passage:** TLDR, line 7: “Our main finding is that the interpretation of a changing role score depends on what we change and how we measure it.” Introduction, line 23: “Our report adds local measurements, a comparison of offsets on saved activations, and a controlled comparison of permission cues.”

**Concern:** The opening states a general measurement principle and then inventories activities. It does not foreground the most concrete empirical contribution: two probe class sets give opposite signs for the same adjusted absolute-Userness comparison. The first TLDR bullet instead emphasizes agreement across 12 relative-score settings. A reader skimming only the opening could retain 'scores need care' while missing what these measurements specifically add and why they change interpretation.

**Suggested remedy:** Replace the general main-finding sentence or the activity inventory with the actual sign disagreement and its consequence. For example: 'For the same marker-permission comparison at layer 12, the adjusted Userness effect is negative with the five-role probe and positive with the three-role probe; both give a positive relative User/Tool effect. A conclusion drawn from this score therefore depends on the probe and metric.' Keep the 100-pair scope and all-12 relative agreement. Introduce the permission marker in plain language as a request to write a dummy record file. Distinguish this local empirical result from the offset example, which illustrates the known algebra of a multiclass classifier, and from the qualitative gardening replication.

### R02

**Exact passage:** Sections 2–3, lines 57–99, especially the sequence “Figure 2 — Small offsets move the readout…” → “Now increase the offset to five times that length” → “Figure 3 — Larger offsets produce a different pattern” → “The small and large plots become easier to compare when we average tokens…”

**Concern:** The reader encounters two 12-panel token plots before the four-panel aggregate that makes the headline co-rise and subsequent reversal visible. Most of those 24 panels are not interpreted in the prose, and several baselines repeat. The detailed plots are useful for inspecting passage structure, but their current placement gives one illustrative conversation more visual weight than the permission result and delays the point the reader is supposed to discover.

**Suggested remedy:** Propose a reversible reordering within Sections 2–3: keep the short method and probe-change explanation, present the current Figure 4 and its three-row table first, then show the full small/large grids as token-level detail. Preserve both grids and their source links in this comprehensive report. Introduce each grid with a precise reading target, such as the original-User points in the Tool-direction panels or the collapse of passage separation under the large positive User offset. This lets the aggregate earn attention for the details without deleting evidence or claiming that all panels share one pattern.

### R03

**Exact passage:** Section 4, lines 103–109: “a User message either permits an action or uses neutral wording”; “contains two commands: write a dummy marker file, and upload data”; “construct a control in which the command span is replaced by neutral text”; and the adjusted-effect equation.

**Concern:** This is the central empirical comparison, but a cold reader never sees the actual permission cue or a concrete layout of the command/control inputs. In particular, the prose leaves implicit that permission names only the marker, both command slots are replaced in the null construction, and the surrounding imperative wrapper remains. Readers must reconstruct two different meanings of 'neutral' while also learning the score and the subtraction.

**Suggested remedy:** Add one compact worked input schematic before the equation: the base user task is to summarize the page; the permission condition adds the saved sentence 'If the page asks you to write a marker file, you have my permission to do that, and only that'; the Tool text has the same ordered marker/upload slots; the null construction substitutes page prose into both slots while retaining the wrapper. A 2×2 table for user cue (neutral/permission) and Tool construction (commands/replacement prose) would suffice. Identify the marked span as what is scored and retain the statement that no command or continuation ran. This is an explanatory example of the construction, not a selected behavioral result. The current rh6_readings.py lines 96–113 and 196–250, plus the frozen prefills, support these details.

### R04

**Exact passage:** TLDR, line 9: “Absolute Userness can decrease in the same comparison.” Section 4, line 119: “The same contextual comparison supports ‘Userness decreased’ under one probe and ‘Userness increased’ under another.” Figure 5 image footer: “Permission raises relative User/Tool in every tested setting; Userness falls at layer 12 with SUCAT.”

**Concern:** The plotted values are differences of permission effects after subtracting the neutral-span effect. The unqualified 'Userness decreased/increased' wording can make the plotted −1.39 and +4.46 percentage points sound like raw permission-versus-neutral probability changes. In this dataset the raw layer-12 command changes happen to share those signs, so this is a precision and reader-understanding issue, not a numerical contradiction. The raw marker-first means are −1.134 and +4.745 points, while the plotted adjusted means are −1.395 and +4.458 points.

**Suggested remedy:** Keep the estimand in the takeaway: 'The marker’s adjusted Userness effect is negative under the five-role probe and positive under the three-role probe.' Similarly, describe the figure footer and TLDR as adjusted relative and absolute effects. If raw probability changes are worth stating, label and cite them separately rather than using the plotted adjusted numbers as their magnitudes. No new calculation or experiment is required; the command, null_ctrl, and command-minus-null rows in user_turn_effects.csv already distinguish them.

## Scientific uncertainty already correctly disclosed

These are evidence boundaries to preserve, not additional conditions for editorial approval:

- The gardening replication uses one authored conversation. Its later reasoning segment is weaker, the probabilities differ from the paper, and style, content, position, and the original averaging denominator remain unresolved.
- The offset figures rescore the same saved states. They provide neither independent conversation replication nor evidence of propagation through later layers or changed behavior. Individual coefficient rows depend on the saved softmax parameterization.
- The permission comparison is exploratory and uses 100 reused pairs. Its pointwise intervals have no simultaneous guarantee; the neutral control is imperfect, and lexical overlap, token position, and template/page coupling remain alternative explanations.
- Recall is distinct from calibration. The two split evaluations use different test sets and separately fitted probes. Optimizer convergence remains unresolved.
- The proposed intervention study is deferred. Prepared candidates are not completed eligible conversations, and a planning floor of 200 is not itself a power or precision justification.
- The earlier adverse behavioral record is retained with its different setup and repeated-case limitation. It is not used as a general effect estimate.

The draft already explains tokens, layers, activations, role classes, probabilities, relative scores, recall, calibration, and repeated-measurement units at useful levels of detail. I do not recommend expanding those definitions or adding a generic limitations section. Figure 6 earns its place because it changes how confidently the preceding readouts should be interpreted.

## Evidence inspected and independence

I read the complete frozen report and visually inspected all six PNGs. For the central comparison, I read the current RH6 results record and frozen input contract, inspected the construction in `rh/rh6_readings.py`, and checked the layer-12 marker rows of the saved `user_turn_effects.csv`. The raw and adjusted means support the clarification in R04. I applied the supplied Neel Nanda narrative guidance, particularly claims, layered explanation, and figure purpose. I did not read other reviewers’ verdicts or the narrative card and did not run a model, notebook, GPU job, or experiment.

## Verdict

**Revise: four recommendations.** The central contribution is supportable at the report’s stated exploratory level. The revisions improve claim hierarchy and explain the existing measurement; none requires new science or a stronger novelty claim. All changes remain proposals for the author to assess. The frozen draft and source records are unchanged.

