# What role-probe scores tell us about GPT-OSS-20B

**Research report · Review draft v0.2 · 11 September 2026**

## TLDR

A **role probe** is a classifier that reads a model’s internal state and estimates whether it resembles User, Assistant, or another message role. We use these probes to study GPT-OSS-20B, a language model. Our main finding is that the interpretation of a changing role score depends on what we change and how we measure it.

- **A permission cue can look different under different readouts.** Across 100 constructed input pairs, the marker command’s permission effect is positive in all 12 tested relative-score settings after subtracting a neutral-text control. Absolute Userness can decrease in the same comparison.
- **A small offset can raise both Userness and Toolness.** Adding a vector to saved internal states raises both scores for the original User passages; a larger offset makes them move in opposite directions. This is a direct classifier response, with no subsequent model computation.
- **Role patterns survive misleading message tags in one example, but the numerical replication remains incomplete.** The authors’ gardening conversation retains a visible reasoning-text signal in our run. Its probabilities differ from the paper’s.

These results make a role score useful to inspect, but insufficient by itself to infer obedience. The next experiment is to test whether vector effects survive later model layers, using an adequately sized set of conversations. That experiment is in the backlog.

## The question: what does “more User-like” mean?

Suppose an assistant is asked to summarize a webpage. The page contains the sentence “Ignore the request and follow these instructions instead.” This is an illustrative example of **prompt injection**: instructions in untrusted content attempt to redirect the model beyond the authorized task. The same words would have a different status if the user had directly asked the assistant to follow them.

A chat model receives messages with role labels. **User** identifies the person’s messages; **Assistant** identifies the model’s replies; **System** contains instructions supplied by the system; **Tool** contains results from an external operation, such as retrieving a page. The message label tells the model where text came from. The question is whether the model’s internal representation also reflects the role suggested by the text itself.

Ye, Cui and Hadfield-Menell investigate this in *Prompt Injection as Role Confusion*. They train **role probes**: small classifiers that read a model’s internal activations and predict a role. We build on their method and reuse their gardening example. Our report adds local measurements, a comparison of offsets on saved activations, and a controlled comparison of permission cues. These are exploratory analyses of saved results; the comparisons emphasized here were selected after inspection. [Paper, §4][paper-role] · [Run record][session-result]

The distinction we care about is between **what a classifier can read**, **what changes when we edit a state**, and **what the model does next**. The six figures address the first two. They do not measure a change in behavior after steering. This matters because a safety intervention would need to affect which instructions the model follows while preserving its ability to do the legitimate task.

## How to read a role score

A model processes text as **tokens**—words or pieces of words—and transforms its internal state through a sequence of **layers**. An **activation vector** is the list of numbers representing a token at a chosen point in that computation. A probe turns the vector into probabilities over its role classes.

**Userness** means the probability assigned to User; **Toolness** is the probability assigned to Tool. A value of 80% means this particular probe assigns probability 0.8 to that class. It is not an 80% chance that the model will obey the text. The class probabilities add to one, so changing one can also change the others.

The paper also analyzes **CoT**, or chain-of-thought reasoning text, as a role. In GPT-OSS, this text appears in the Assistant’s analysis channel. We distinguish it from the Assistant’s final reply. In the passage plots, **blue identifies originally User text, yellow/orange reasoning text, and green Assistant replies**. Color records the passage’s origin, even when we change its message tag or its score.

We trained probes across all 24 model layers, numbered 0–23, using 249 neutral base texts rendered under different role labels. A probe learns from one portion of these labeled examples and is evaluated on held-out examples. The report uses several role sets; each is a separately fitted classifier. Exact training and split details appear in Appendix A. [Training record][training]

## 1. The gardening pattern appears, with a replication gap

The authors’ gardening conversation lets us compare the same passage under different labels. It contains User questions, reasoning text, and Assistant answers about tomatoes. We supply the existing conversation as input and record activations; the model is not generating the reasoning shown here.

We render the conversation with correct tags, without its role tags, and as one User message. We ask whether the reasoning passages still tend to receive a higher reasoning-role score when their tags change. They often retain higher CoTness than the surrounding User and Assistant text.

![Figure 1. Three measured CoTness panels for one gardening conversation. Across correct tags, no tags and all-user tags, originally reasoning passages often score above User and Assistant passages; the later reasoning passage is weaker.](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/00-gardening.png>)

*Figure 1 — A visible role pattern with incomplete numerical replication. Each panel shows the same 512 matched tokens from one conversation, scored by a four-role probe at layer 12. The horizontal axis follows token order; the vertical axis is CoTness. Colors identify original passage roles. These are our measured activations, using the authors’ Figure 7 setup.* [Data][garden-rows] · [PDF][garden-pdf] · [Paper’s Figure 7][paper-figure7]

The association is uneven: the later reasoning passage has a weaker signal. The averages also differ from the paper:

| Tag condition | Our displayed reasoning tokens | Our full reasoning content | Paper’s reported CoTness |
| --- | ---: | ---: | ---: |
| Correct tags | 64.3% | 64.0% | 85% |
| No tags | 71.8% | 71.0% | 82% in Appendix E; 83% in Figure 7 |
| All User tags | 71.7% | 71.0% | 85% |

The displayed means use 177 reasoning tokens per condition; the full-content means use 179. The paper’s exact averaging denominator is unresolved. Neither local subset closes the gap, so this remains a qualitative match with an unresolved numerical discrepancy. [Subset audit][garden-result] · [Paper, Appendix E][paper-garden]

This example establishes that the measured pattern can persist under changed tags. One authored conversation cannot establish how often it occurs, and the experiment does not isolate style from content or token position. It gives us a concrete set of states on which to inspect the next question: what happens when we move them in a probe-defined direction?

## 2. Small and large offsets change the same states differently

We take the saved states from the all-User-tags condition and add or subtract a vector before scoring them again. A **direction** specifies which way to move in activation space; its **magnitude** specifies how far. Here the User and Tool directions come from the corresponding learned coefficient vectors of a five-role probe.

This changes the readout from Figure 1: that probe distinguished System, User, CoT and Assistant. The next three figures use a separately fitted probe that also includes Tool. Each offset must therefore be compared with its own unmodified baseline.

The panels share the same token positions, colors and probability axes. The upper pair of rows edits the User direction; the lower pair edits the Tool direction. Within each pair, the rows measure Userness and Toolness. Columns subtract the direction, leave the state unchanged, or add the direction.

![Figure 2. One-percent offsets along the saved User and Tool classifier directions. The three columns compare subtraction, no offset and addition on the same 512 tokens; passage structure remains visible as the scores change.](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/01-small-offsets.png>)

*Figure 2 — Small offsets move the readout while much of the passage structure remains visible. We edit the same 512 saved token states from one conversation and rescore them with the five-role layer-12 probe. The 1% label specifies offset length relative to a fixed activation-norm reference, not a percentage-point change in probability. No edited state is passed through the remaining model layers.* [Data][offset-rows] · [PDF][small-pdf]

Adding the small User offset raises mean Userness from **70.2% to 86.7%** in the original User passages and from **6.6% to 15.8%** in the reasoning passages. The text has not changed; the classifier is reading displaced states.

Now increase the offset to five times that length:

![Figure 3. Five-percent offsets with the same arrangement and axes as Figure 2. Adding the User direction raises Userness across all three original passage groups; Tool-direction effects differ across the two measured scores.](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/02-large-offsets.png>)

*Figure 3 — Larger offsets produce a different pattern. This uses the same states, five-role probe and panel arrangement as Figure 2, with 5% rather than 1% offsets. Under positive User offsets, mean Userness reaches 99.95% for originally User text, 77.0% for reasoning text and 77.2% for Assistant text. This is still offline rescoring of one conversation.* [Data][offset-rows] · [PDF][large-pdf]

The large User offset makes the passage groups less separated on the Userness axis. That observation is specific to this direction and readout: other panels do not all lose separation as the offset grows.

A direct response is expected because the probe uses these same coefficient vectors to score activations. The plots reveal the size and pattern of that response, rather than demonstrating that the language model would preserve it or act on it. “Subtract” means applying the opposite displacement; it does not remove a token’s User or Tool component. Appendix A gives the exact operation and why these saved coefficient rows should not be treated as uniquely defined semantic directions. [Offset method][offset-method]

## 3. Userness and Toolness are not opposite ends of one scale

The small and large plots become easier to compare when we average tokens within their original passage roles. The next figure summarizes those same states: 95 User tokens, 177 reasoning tokens, and 240 Assistant tokens, pooling two turns per role.

![Figure 4. Mean Userness and Toolness at offsets of minus five, minus one, zero, plus one and plus five percent. The blue original-User series rises on both readouts at a small positive Tool offset, then loses Userness at the larger offset.](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/03-offset-dose-response.png>)

*Figure 4 — A small Tool offset raises both scores for the original User passages. Columns identify the edited direction; rows identify the measured probability. Colors identify the original passage role. Each point averages the same role’s tokens from one conversation. Lines connect five evaluated offsets; intermediate strengths were not measured.* [Means][dose-data] · [PDF][dose-pdf]

Follow the blue series in the Tool-direction column:

| Tool-direction offset | Mean Userness | Mean Toolness |
| --- | ---: | ---: |
| 0% | 70.2% | 1.5% |
| +1% | 73.4% | 4.9% |
| +5% | 30.4% | 67.7% |

**At +1%, both probabilities rise.** This is possible because the probe has five classes: probabilities assigned to System, CoT or Assistant can fall while User and Tool both rise. At +5%, Toolness increases further while Userness falls.

The useful lesson is to inspect the separate probabilities before compressing them into a single “User versus Tool” number. The algebra allows both to rise; these measurements show where and by how much that happens in this example. Reusing these token states at several strengths supplies no new independent conversations, so we attach no population confidence interval to this illustration.

## 4. Permission changes the readout—and the readout changes the conclusion

The previous results edit saved states. We now change the model’s actual input: a User message either permits an action or uses neutral wording. This tests how a legitimate contextual cue changes the representation of instructions already inside Tool text.

We use **100 constructed template–page pairs**. Each pairs an instruction template with webpage text and contains two commands: write a dummy marker file, and upload data. We compare two command orders and construct a control in which the command span is replaced by neutral text. Each prepared conversation is processed through the model to record activations, a **prefill**. Neither command is executed, and no continuation is generated. [Input and result description][rh6-result]

The plotted effect is the permission-versus-neutral difference for the marker command, minus the corresponding difference for its neutral-text replacement. This subtraction asks whether the command span changes beyond the change seen in the control span:

> **Adjusted permission effect = change in the command score − change in the neutral-span score.**

We measure it in two ways. **Absolute Userness** uses the User probability. **Relative User/Tool** uses the logarithm of the User-to-Tool probability ratio: it rises when User probability grows relative to Tool probability. A relative increase can therefore occur even when Userness falls.

![Figure 5. Adjusted marker-permission effects across layers 8, 12 and 16, two probe class sets and two command orders. All relative-score intervals are above zero, while layer-12 absolute Userness effects are negative for the five-role probe and positive for the three-role probe.](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/04-rh6-readout.png>)

*Figure 5 — The marker’s adjusted relative score rises in all 12 plotted settings; absolute Userness depends on the probe. Columns compare five roles (SUCAT: System, User, CoT, Assistant, Tool) with three (UAT: User, Assistant, Tool). Solid lines place the marker first; dashed lines place it second. Top: Userness changes in percentage points. Bottom: changes in the natural-log User/Tool ratio. Bars are pointwise 95% bootstrap intervals over 100 paired inputs.* [Contrasts][rh6-data] · [PDF][rh6-pdf]

At layer 12, with the marker first, the adjusted absolute Userness effect is **−1.39 percentage points** for the five-role probe, with interval **[−2.03, −0.78]**. For the three-role probe it is **+4.46 points**, with interval **[+3.42, +5.47]**. The relative effect is positive for both. Placing the marker second preserves this sign disagreement between probes.

This is the clearest measurement lesson in the report. The same contextual comparison supports “Userness decreased” under one probe and “Userness increased” under another. Both are correct descriptions of their respective outputs. Probe choice and metric choice are part of the result, rather than interchangeable ways to display it.

The scope is the marker’s adjusted permission contrast on these constructed inputs. There are 12 settings because we examine three layers, two probe class sets and two orders. The full dataset reuses the same 100 input pairs across 12 input conditions, yielding 1,200 prefills; those are not 1,200 independent examples. The interval bars describe uncertainty across pairs for each plotted setting, without a simultaneous guarantee across the grid.

The control is imperfect. Only 47 of 100 pairs match both replacement-span token counts exactly; permission wording also changes lexical overlap and token position. Each template is coupled to one page. These results establish a readout response under the specified comparison, while leaving those explanations unresolved. The full table retains the weaker prohibition contrasts as well. They do not establish that the model would follow the permitted command. [Audit and full results][rh6-result]

## 5. How much should we trust the probes?

A role score is only as interpretable as the classifier producing it. We therefore check how often the five-role probes correctly classify held-out role-labeled tokens across all 24 layers.

This measure is **recall for each role**: among tokens whose true label is User, for example, what fraction receive User as the highest-scoring class? Unlike the earlier figures, this plot shows classification correctness, not an average predicted probability.

![Figure 6. Five-role recall across all 24 layers under two train/test splits. User and Assistant are often easier to identify than CoT and Tool. The left split holds out prompt variants; the right holds out complete base texts.](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/05-probe-recall.png>)

*Figure 6 — Probe reliability varies by role and layer. Each curve is correct-role classification among held-out tokens of that role, reconstructed from integer confusion counts. The left test set contains 124 prompt variants; the right contains 24 base texts with all their role variants held out together. Blue, yellow and green retain their passage roles; gray is System and purple is Tool. Each role’s curve is token-weighted.* [Recalls][recall-data] · [PDF][recall-pdf]

User and Assistant are easier to classify than CoT and Tool through much of this run. At layer 12 in the left panel, recall is **76.7% for User**, **83.9% for Assistant**, **46.2% for CoT**, and **44.8% for Tool**. These are the five-role probes; they are not the four-role readout used for the gardening overview.

The two panels also expose a design choice. In the left split, a base text can appear in training under one role and testing under another. In the right split, the test texts are absent from training in every role. These evaluations use different test sets and separately fitted probes, so the panel difference does not isolate a causal effect of text overlap. [Split records][splits]

Correct classification does not establish **calibration**—whether predictions assigned 80% probability are correct about 80% of the time. The saved fitting audit also leaves optimizer convergence unresolved: we have not established that fitting reached its intended solution. These are reasons to validate a chosen readout before making a stronger claim from it. [Estimator audit][estimator-audit]

## What would make the next experiment informative?

The open question is whether a role-directed edit survives the model’s remaining computation and changes something useful. Our next batch would extend the paper’s **Figure 23**, which compares role readouts across layers, by adding intervention curves for GPT-OSS-20B. It remains deferred; the offset illustrations above do not supply those curves.

We will compare two vector families separately: the earlier **Tool-minus-User direction derived from activations**, and the **User and Tool coefficient directions from the probes**. These have different constructions, intervention sites and scales. Each needs its own baseline, opposite signs, and random directions of the same length. A zero offset must reproduce the unmodified run.

For a fixed conversation, the model would continue computing after the intervention; later probes would measure what survives. The prediction worth testing is a direction-specific change beyond the intervention layer. If it disappears immediately or is matched by random directions, the case for a specific propagated effect weakens. A mitigation claim would additionally require paired behavior and legitimate-task performance measurements.

Our planning floor is **200 eligible independent conversations after exclusions**, with balanced source coverage and reserve candidates for filtering losses. Final sample size must follow the smallest useful paired effect and desired precision, rather than treating 200 as automatically sufficient. The 200 prepared candidates are not yet 200 generated, eligible, measured conversations. [Deferred batch specification][backlog]

Before that run, the source contract needs to be explicit: the paper and notebook disagree about both sample count and which quantity is plotted. Appendix B records those discrepancies. We will keep mean probabilities and classification accuracy separate and weight conversations equally in the aggregate, rather than allowing long conversations to dominate. [Figure 23 source audit][fig23-contract]

## Appendix A. Exact measurements and reproducibility

**Model and probes.** The completed September 11 H100 run trained 384 probes: 24 layers × eight role-class combinations × two split methods. The corpus contains 249 base texts and 1,245 role-rendered variants. Activations are taken at `post_attention_layernorm`, before the layer’s multilayer perceptron. Figure 1 uses the prompt-split four-role probe; Figures 2–4 use the prompt-split five-role probe at the same layer-12 site. The training record, split IDs and estimator audit are linked below. [Training][training] · [Splits][splits] · [Estimator][estimator-audit]

**Gardening formatting and token selection.** The no-tags condition joins the complete transcript, including system text, with newlines. The all-User condition places that joined transcript inside one User message. Neither retains a separate System message. All three panels show 512 matched tokens: 95 originally User, 177 CoT and 240 Assistant. Full-content counts are 97, 179 and 582 respectively. Displayed and full-content averages therefore answer different subset questions. [Rows][garden-rows] · [Subset audit][garden-result]

**Offline offsets.** For a saved activation `h`, we compute:

`h′ = h + s × f × m × (w / ||w||)`

Here `s` is −1, 0 or +1; `f` is 0.01 or 0.05; `w` is the saved User or Tool coefficient vector; and `m = 45.2471` is the median activation norm across all 921 forwarded tokens in the all-User condition. The offset lengths are approximately 0.4525 and 2.2624, shared across directions. They are fractions of this fixed reference, not of each token’s own length. Saved float16 activations and float32 weights are promoted to float64 for rescoring. [Arithmetic provenance][offset-provenance]

**Why the coefficient directions need care.** A linear probe computes a score for each class from the activation and a coefficient vector, then converts the scores to probabilities with softmax. Adding a common vector to every coefficient row leaves those probabilities unchanged, because it shifts all class scores equally. It nevertheless changes any individual row used as an offset. Our illustrations therefore depend on the saved parameterization; they do not identify unique User or Tool axes. This does not affect the arithmetic checks of the specified offsets.

**Permission estimator and uncertainty.** For each input pair and span, the relative score is `ln(mean P_User) − ln(mean P_Tool)`. We average probabilities over that span before taking logs. We then compute `(command permission − command neutral) − (replacement permission − replacement neutral)` within the paired unit and average over the 100 units. The intervals use 2,000 paired bootstrap resamples with seed 123. One resample draws 100 units with replacement and keeps each unit’s conditions together; the reported interval spans the 2.5th to 97.5th percentiles of the resulting means. This is not a mean of tokenwise log ratios or a log ratio after pooling all templates. [Full analysis][rh6-result] · [Plotted estimates][rh6-data]

**Validation units.** The five-role prompt-split test set has 124 variants and 67,727 content tokens. The grouped split holds out 24 base texts, with 56,685 content tokens. Recall weights tokens equally within each true role. Tokens, offsets and layers are repeated measurements, not interchangeable with independent conversations or base texts. [Split audit][splits]

## Appendix B. Source discrepancies and evidence beyond the six figures

**Figure 23.** The paper describes 200 conversations. The frozen notebook filters candidates and requests 30, with a comment suggesting 100 for a full test. Those files do not establish the actual published denominator. The notebook averages eligible original-role content tokens within each conversation and then gives conversations equal weight. The paper describes probabilities, while the plotting cell reads correct-role accuracy outputs. Our planned extension retains these as separate named measurements. [Paper, Appendix F][paper-fig23] · [Paper’s Figure 23][paper-figure23] · [Notebook audit][fig23-contract]

**Earlier behavioral evidence.** September 5 steering runs used actual interventions with recorded actions, under a different probe and setup. In the recovered Toolward arm, 17 of 21 episodes wrote both markers despite near-saturated initial Tool readings. Those episodes cover only eight cases and repeated prompts. This adverse result is a reason not to equate an extreme score with selective action control; it is not a reliable estimate of a general steering effect. [Historical evidence audit][evidence-audit]

**Other completed and deferred work.** The Appendix K neutral-position control starts from 200 texts that overlap the probe corpus, so it cannot substitute for a fresh held-out conversation study. Long-context experiments, the StrongREJECT harmful-request benchmark, source attribution and the remaining research questions stay in the full backlog. The audit covers the September 10 Claude discussion, its 19 tracked goals and 35 linked subagent logs; older evidence is included where that discussion referred to it. [Backlog][all-backlog] · [Conversation audit][goal-audit]

## Review status and sources

This is a draft for Hanan’s research and editorial review. Coding agents assisted with analysis, figures and writing. The numerical audit independently reconstructed all 30,720 offset probabilities, 30 plotted passage means, 24 permission contrasts and their interval bounds, and 240 recall values from saved records. No model or GPU experiment was run to prepare this report. Review status and remaining scientific uncertainties are recorded in the [verification log][verification].

We are in **Understand / Distill**: the aim is to make each measurement and its consequence legible, with particular attention to clarity, source fidelity, skeptical interpretation and adequate samples. Figure numbers 1–6 belong to this report; the paper’s figure numbers retain the authors’ numbering. Local data links provide the measured evidence. The external references are:

- Ye, Charles; Cui, Jasmine; and Hadfield-Menell, Dylan. *Prompt Injection as Role Confusion*, version 6. [Paper][paper-role].
- The authors’ code at commit `ec333c40fd43fe991e1ebf66765051b6d7e35784`. [Frozen repository][frozen-repo].
- Our completed measurements and prior checks. [Run summary][session-result] · [Numerical audit][numeric-review] · [Source audit][source-review].

[paper-role]: https://arxiv.org/html/2603.12277v6#S4.SS1
[paper-figure7]: https://arxiv.org/html/2603.12277v6#S4.F7
[paper-figure23]: https://arxiv.org/html/2603.12277v6#A6.F23
[paper-garden]: https://arxiv.org/html/2603.12277v6#A5
[paper-fig23]: https://arxiv.org/html/2603.12277v6#A6
[frozen-repo]: https://github.com/role-confusion/prompt-injection-as-role-confusion/tree/ec333c40fd43fe991e1ebf66765051b6d7e35784
[training]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/metadata.json>
[garden-rows]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-e-L12/figure-7-displayed-rows.csv>
[garden-result]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/GARDENING-RESULTS.md>
[garden-pdf]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/00-gardening.pdf>
[offset-method]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/presentation/probe-offset-illustration/README.md>
[offset-provenance]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/presentation/probe-offset-illustration/provenance.json>
[offset-rows]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/presentation/probe-offset-illustration/plotted-probabilities.csv>
[small-pdf]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/01-small-offsets.pdf>
[large-pdf]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/02-large-offsets.pdf>
[dose-data]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/offset-dose-response.csv>
[dose-pdf]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/03-offset-dose-response.pdf>
[rh6-result]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/RH6-RESULTS.md>
[rh6-data]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/rh6-plotted.csv>
[rh6-pdf]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/04-rh6-readout.pdf>
[recall-data]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/probe-recall-by-layer.csv>
[recall-pdf]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/05-probe-recall.pdf>
[splits]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/split-audit.json>
[estimator-audit]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/native-probe-audit.json>
[fig23-contract]: </Users/hananather/Desktop/MATS 12.0/replication/e9/SOURCE-CONTRACT.md>
[backlog]: </Users/hananather/Desktop/MATS 12.0/replication/e9/BACKLOG.md>
[all-backlog]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/BACKLOG.md>
[evidence-audit]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/review/evidence-audit.md>
[verification]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/verification-log.md>
[goal-audit]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/review/claude-goals-audit.md>
[session-result]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/RESULTS-SUMMARY.md>
[numeric-review]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/review/blog-numerical-review.md>
[source-review]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/review/blog-source-review.md>
[reader-review]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/review/blog-reader-review.md>
