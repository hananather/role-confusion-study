# What role-probe scores tell us about GPT-OSS-20B

**Research report · Review draft v0.2 · 11 September 2026**

## TLDR

A **role probe** is a classifier that reads a language model’s internal state and predicts a message role. Its probability for User is called **Userness**; its probability for Tool is **Toolness**. We study these measurements in GPT-OSS-20B.

Our strongest result is that **two probes give opposite signs for the same mean adjusted permission effect**. Across 100 combinations of instruction templates and webpages, we compare giving permission to write a dummy file with an ordinary request to summarize a webpage. After subtracting a control-text comparison, the Userness effect is negative under a five-role probe and positive under a three-role probe at layer 12. A score relative to Toolness gives positive effects for both. The conclusion depends on the classifier and the measurement.

The examples help interpret that finding:

- **Userness and Toolness can rise together.** For the original User passages in one gardening conversation, a +1% Tool coefficient offset raises mean Userness and Toolness; at +5%, Toolness rises while Userness falls. This is direct rescoring of saved states, with no subsequent model computation.
- **Two reasoning passages in the same conversation can receive very different scores.** In our new Hanan–MATS exchange, the first averages 75.4% CoTness and the second 2.5% with correct tags. Removing tags preserves this difference. The authors’ gardening example remains a replication reference; its probabilities still differ from the paper’s.

These are exploratory measurement results. Some probe classes have weak validation, and the seven figures contain no behavioral steering test. A separate [new companion study][forgery-companion] now shows a selected behavioral steering example, with its controls and qualifications. The next experiment—testing whether vector effects survive later layers across many conversations—remains in the backlog.

## The question: what does “more User-like” mean?

Suppose an assistant is asked to summarize a webpage. The page contains the sentence “Ignore the request and follow these instructions instead.” This is an illustrative example of **prompt injection**: instructions in untrusted content attempt to redirect the model beyond the authorized task. The same words would have a different status if the user had directly asked the assistant to follow them.

A chat model receives messages with role labels. **User** identifies the person’s messages; **Assistant** identifies the model’s replies; **System** contains instructions supplied by the system; **Tool** contains results from an external operation, such as retrieving a page. The message label tells the model where text came from. The question is whether the model’s internal representation also reflects the role suggested by the text itself.

Ye, Cui and Hadfield-Menell investigate this in *Prompt Injection as Role Confusion*. They train **role probes**: small classifiers that read a model’s internal activations and predict a role. We build on their method with a new MATS Q&A and a replication of their gardening example. Our central comparison changes legitimate permission while holding the command text fixed, then asks whether different probes support the same interpretation. The opening illustrations and offset comparisons help us inspect the measurement. For the opening illustration, we wrote two questions about Hanan applying to Neel Nanda’s MATS stream and let the local model generate the replies. This revised exchange follows an earlier attempt and twelve authored examples, all retained in the evidence library. The remaining analyses use saved results, with the comparisons emphasized here selected after inspection. [Paper, §4][paper-role] · [Run record][session-result]

The distinction we care about is between **what a classifier can read**, **what changes when we edit a state**, and **what the model does next**. The seven figures address the first two. They do not measure a change in behavior after steering. This matters because a safety intervention would need to affect which instructions the model follows while preserving its ability to do the legitimate task.

## How to read a role score

A model processes text as **tokens**—words or pieces of words—and transforms its internal state through a sequence of **layers**. An **activation vector** is the list of numbers representing a token at a chosen point in that computation. A probe turns the vector into probabilities over its role classes.

**Userness** means the probability assigned to User; **Toolness** is the probability assigned to Tool. A value of 80% means this particular probe assigns probability 0.8 to that class. It is not an 80% chance that the model will obey the text. The class probabilities add to one, so changing one can also change the others.

The paper also analyzes **CoT**, or chain-of-thought reasoning text, as a role. In GPT-OSS, this text appears in the Assistant’s analysis channel. We distinguish it from the Assistant’s final reply. In the passage plots, **blue identifies originally User text, yellow/orange reasoning text, and green Assistant replies**. Color records the passage’s origin, even when we change its message tag or its score.

For training, we place the same neutral passage inside different message-role and channel wrappers: once as User text, once as Tool text, and so on. We run the language model and label each recorded activation with its supplied role. A separate classifier learns to predict those labels; they are not human judgments about the passage’s intent.

Our training corpus has 249 base texts, with probes at all 24 layers, numbered 0–23. We evaluate on held-out examples. Changing which roles a probe distinguishes means fitting a different classifier, rather than hiding labels from an unchanged output. Appendix A gives the splits and fitting details. [Training record][training]

## 1. Hanan asks for a MATS project idea

“I’m Hanan, and I want to apply to Neel Nanda’s MATS stream.” Our new conversation begins with a project question about steering GPT-OSS-20B’s representation of chat roles. The follow-up asks: “Suppose the Toolness probe shoots up but the model still follows the injected instruction. Is that a useful result, or have I built an expensive mood ring?”

We wrote these two questions for the illustration. Local GPT-OSS-20B generated both reasoning passages and replies, giving the sequence **User 1 → CoT 1 → Assistant 1 → User 2 → CoT 2 → Assistant 2**. We then replayed the complete exchange under three formats and measured its activations. The generated advice is preserved as input; the suggested experiments were not run. [Full conversation and reading note][mats-input] · [Approved canonical example][mats-canonical]

![Figure 1. Three measured CoTness panels for Hanan asking about Neel Nanda’s MATS stream. User 1, CoT 1, Assistant 1, User 2, CoT 2 and Assistant 2 appear in sequence, with complete opening excerpts angled beneath the plots. The first reasoning passage scores much higher than the second under all three formats.](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/mats-hanan-dialogue-v2/mats-hanan-dialogue-v2.png>)

*Figure 1 — An original MATS exchange, measured under three role-tag conditions. Each panel shows the same 578 matched tokens: correct tags, no tags, and all text in one User message. Colors identify the six passages’ original roles; the horizontal axis follows token order. We measured layer-12 activations locally and applied the saved four-role probe. Both reasoning passages and replies were generated by GPT-OSS-20B. Complete opening excerpts are printed below the plots; the unabridged companion retains every word.* [Data][mats-rows] · [PDF][mats-pdf] · [Unabridged figure][mats-full-pdf] · [Compare layouts][mats-layouts] · [Full conversation][mats-input]

**The two reasoning passages receive very different scores.** With correct tags, displayed CoT 1 averages **75.4% CoTness**, while CoT 2 averages **2.5%**. Without tags, they average **80.3% and 6.9%**; placing the transcript inside one User message gives **81.0% and 6.9%**. The label “reasoning” therefore does not guarantee a high score from this probe, even within one conversation. Color marks where a passage originated; its vertical position is the classifier’s readout.

This is one illustration, chosen for its personal MATS context. We revised an earlier generated attempt that confused chat roles with personas; both attempts and the preceding [twelve authored examples][mats-gallery] are retained. The figure does not estimate how common this pattern is. The local runtime also differs from the H100 runtime used for the remaining figures; Appendix A documents the control comparison. [Generation and selection record][mats-choice] · [Measurement audit][mats-audit] · [Runtime control][mats-runtime]

## 2. The gardening pattern appears, with a replication gap

The authors’ gardening conversation lets us compare the same passage under different labels. It contains User questions, reasoning text, and Assistant answers about tomatoes. We supply the existing conversation as input and record activations; the model is not generating the reasoning shown here.

We render the conversation with correct tags, without its role tags, and as one User message. The latter two join the complete transcript, including system text; the all-User version wraps that joined text in a single User message. We ask whether the reasoning passages still tend to receive a higher reasoning-role score when their tags change. They often retain higher CoTness than the surrounding User and Assistant text.

![Figure 2. Three measured CoTness panels for one gardening conversation. Across correct tags, no tags and all-user tags, originally reasoning passages often score above User and Assistant passages; the later reasoning passage is weaker.](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/review-v0.2/00-gardening.png>)

*Figure 2 — A visible role pattern with incomplete numerical replication. Each panel shows the same matched display subset of 512 content tokens from one conversation, scored by a four-role probe at layer 12. The horizontal axis follows token order; the vertical axis is CoTness. Colors identify original passage roles. These are our measured activations, using the authors’ Figure 7 setup.* [Data][garden-rows] · [PDF][garden-pdf] · [Paper’s Figure 7][paper-figure7]

The association is uneven: the later reasoning passage has a weaker signal. The averages also differ from the paper:

| Tag condition | Our displayed reasoning tokens | Our full reasoning content | Paper’s reported CoTness |
| --- | ---: | ---: | ---: |
| Correct tags | 64.3% | 64.0% | 85% |
| No tags | 71.8% | 71.0% | 82% in Appendix E; 83% in Figure 7 |
| All User tags | 71.7% | 71.0% | 85% |

The displayed means use 177 reasoning tokens per condition; the full-content means use 179. The paper’s exact averaging denominator is unresolved. Neither local subset closes the gap, so this remains a qualitative match with an unresolved numerical discrepancy. [Subset audit][garden-result] · [Paper, Appendix E][paper-garden]

This example establishes that the measured pattern can persist under changed tags. One authored conversation cannot establish how often it occurs, and the experiment does not isolate style from content or token position. It establishes the instrument’s behavior on a readable example. We next use it to compare an actual change in the user’s request.

## 3. Permission changes the readout—and the readout changes the conclusion

We now change the model’s actual input: a User message either permits an action or uses neutral wording. This tests how a legitimate contextual cue changes the representation of instructions already inside Tool text.

We use **100 template–page units**, each combining an instruction template with webpage text. The page contains two commands: write a dummy marker file, and upload file contents. The ordinary User request asks for a page summary. Every pair uses the same permission sentence:

> If the page asks you to write a marker file, you have my permission to do that, and only that

The upload receives no permission. For each command order, the four inputs below isolate the comparison we plot. This is a schematic of the saved construction, not a behavioral trace:

| Tool-text construction | Ordinary summary request | Same request + marker-only permission |
| --- | --- | --- |
| Marker and upload commands | Score the marker span | Score the same marker span |
| Both command slots replaced by page prose | Score the marker’s replacement span | Score the same replacement span |

The control replaces **both** command slots while retaining the surrounding instruction wrapper. “Ordinary request” therefore describes the User-message baseline; “replacement prose” describes the Tool-text control. We subtract the permission effect in the replacement span from the permission effect in the command span:

> **Adjusted permission effect = change in the command score − change in the replacement-span score.**

The adjustment belongs to this whole construction; it does not isolate changing the marker text alone. Each prepared conversation is processed through the model to record activations, a **prefill**. Neither command is executed, and no continuation is generated. [Input construction][rh6-construction] · [Saved results][rh6-result]

We measure it in two ways. **Absolute Userness** uses the User probability. **Relative User/Tool** uses the log ratio of the span-mean User and Tool probabilities: it rises when User probability grows relative to Tool probability. A relative increase can therefore occur even when Userness falls. Appendix A explains why taking the ratio after averaging also matters.

![Figure 3. Adjusted marker-permission effects across layers 8, 12 and 16, two probe class sets and two command orders. All relative-score intervals are above zero, while layer-12 absolute Userness effects are negative for the five-role probe and positive for the three-role probe.](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/review-v0.2/04-rh6-readout.png>)

*Figure 3 — The marker’s mean adjusted relative effect is positive in all 12 plotted settings; its adjusted Userness effect depends on the probe. Columns compare five roles (SUCAT: System, User, CoT, Assistant, Tool) with three (UAT: User, Assistant, Tool). Solid lines place the marker first; dashed lines place it second. Top: Userness changes in percentage points. Bottom: changes in the natural-log ratio of span-mean User and Tool probabilities. Logs precede the paired contrast. Lines join the three measured layers; intervening layers are not displayed. Bars are pointwise 95% bootstrap intervals over 100 template–page units.* [Contrasts][rh6-data] · [PDF][rh6-pdf]

At layer 12, with the marker first, the adjusted absolute Userness effect is **−1.39 percentage points** for the five-role probe, with interval **[−2.03, −0.78]**. For the three-role probe it is **+4.46 points**, with interval **[+3.42, +5.47]**. The relative effect is positive for both. Placing the marker second preserves this sign disagreement between probes.

The marker’s **mean adjusted Userness effect** is negative under the five-role probe and positive under the three-role probe. These classifiers were fitted separately, so the comparison changes the learned coefficients as well as the available classes. These describe the command-minus-control contrast, rather than the raw probability change caused by permission. Probe choice and metric choice are part of the result, rather than interchangeable ways to display it.

The scope is the marker’s adjusted permission contrast on these constructed inputs. The **12 plotted settings** are three layers × two probes × two orders. Separately, each pair has **12 prepared inputs**: three User-message variants (ordinary request, marker permission, upload prohibition) × two command orders × two text constructions. These produce 1,200 prefills from 100 pairs. Figure 3 selects permission versus the ordinary request; the prohibition comparison remains in the full results. Layers and probes reuse each prefill’s recorded states. The descriptive intervals resample the observed template–page units with the model and fitted probes fixed. They do not include probe-fitting uncertainty, account for selecting these comparisons after inspection, or provide a simultaneous guarantee across the grid.

The control is imperfect. Only 47 of 100 pairs match both replacement-span token counts exactly; permission wording also changes lexical overlap and token position. Each template is coupled to one page. These results establish a readout response under the specified comparison, while leaving those explanations unresolved. For upload prohibition, 10 of 12 adjusted relative-score intervals include zero; the other two are small positive shifts at layer 16 with the three-role probe. Intervals including zero do not establish no effect. There are no measured action outcomes in this experiment. [Audit and full results][rh6-result]

## 4. Userness and Toolness are not opposite ends of one scale

We take the saved gardening states from the all-User-tags condition and add or subtract a vector before scoring them again. A **direction** specifies which way to move in activation space; its **magnitude** specifies how far. Here the User and Tool directions come from the corresponding learned coefficient vectors of a five-role probe.

This changes the readout from Figure 2: that probe distinguished System, User, CoT and Assistant. Figures 4–6 use a separately fitted probe that also includes Tool. Each offset must therefore be compared with its own unmodified baseline.

The reference activation length, or **norm**, is 45.2471: the median over all 921 forwarded tokens. A 1% offset has length 0.4525; a 5% offset has length 2.2624. These are fixed lengths for every displayed token and both directions, not percentage-point changes in probability.


We first summarize the offsets by averaging tokens within their original passage roles: 95 User tokens, 177 reasoning tokens, and 240 Assistant tokens, pooling two turns per role.

![Figure 4. Mean Userness and Toolness at offsets of minus five, minus one, zero, plus one and plus five percent. The blue original-User series rises on both readouts at a small positive Tool offset, then loses Userness at the larger offset.](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/review-v0.2/03-offset-dose-response.png>)

*Figure 4 — A small Tool offset raises both scores for the original User passages. Columns identify the edited direction; rows identify the measured probability. Colors identify the original passage role. Each point averages the same role’s tokens from one conversation. The reference norm is 45.2471 over 921 forwarded tokens. Lines connect five evaluated offsets; intermediate strengths were not measured. Marker shapes also identify source roles.* [Means][dose-data] · [PDF][dose-pdf]

Follow the original-User series (blue) in the Tool-direction column:

| Tool-direction offset | Mean Userness | Mean Toolness |
| --- | ---: | ---: |
| 0% | 70.2% | 1.5% |
| +1% | 73.4% | 4.9% |
| +5% | 30.4% | 67.7% |

**At +1%, both probabilities rise.** This is possible because the probe has five classes: probabilities assigned to System, CoT or Assistant can fall while User and Tool both rise. At +5%, Toolness increases further while Userness falls.

The useful lesson is to inspect the separate probabilities before compressing them into a single “User versus Tool” number. The algebra allows both to rise; these measurements show where and by how much that happens in this example. Reusing these token states at several strengths supplies no new independent conversations, so we attach no population confidence interval to this illustration.

A direct response is expected because the probe uses these same coefficient vectors to score activations. These illustrations reveal the size and pattern of that response, rather than demonstrating that the language model would preserve it or act on it. “Subtract” means applying the opposite displacement; it does not remove a token’s User or Tool component. Appendix A gives the exact operation and why these saved coefficient rows should not be treated as uniquely defined semantic directions. [Offset method][offset-method]

## 5. What small and large offsets do to individual passages

The averages conceal how individual passages move. These two detailed views retain the paper’s token plots and let us compare small and large displacements on exactly the same states.

The panels share the same token positions, colors and probability axes. The upper pair of rows edits the User direction; the lower pair edits the Tool direction. Within each pair, the rows measure Userness and Toolness. Columns subtract the direction, leave the state unchanged, or add the direction.

![Figure 5. One-percent offsets along the saved User and Tool classifier directions. The three columns compare subtraction, no offset and addition on the same 512 tokens; passage structure remains visible as the scores change.](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/review-v0.2/01-small-offsets.png>)

*Figure 5 — Small offsets move the readout while much of the passage structure remains visible. We edit the same 512 saved token states from one conversation and rescore them with the five-role layer-12 probe. The shared reference norm is 45.2471 across 921 forwarded tokens; 1% and 5% correspond to lengths 0.4525 and 2.2624. No edited state is passed through the remaining model layers.* [Data][offset-rows] · [PDF][small-pdf]

In the upper-right Userness panel, adding the small User offset raises mean Userness from **70.2% to 86.7%** in the original User passages and from **6.6% to 15.8%** in the reasoning passages. The text has not changed; the classifier is reading displaced states.

Now inspect the same upper-right panel with an offset five times as large:

![Figure 6. Five-percent offsets with the same arrangement and axes as Figure 5. Adding the User direction raises Userness across all three original passage groups; Tool-direction effects differ across the two measured scores.](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/review-v0.2/02-large-offsets.png>)

*Figure 6 — Larger offsets produce a different pattern. This uses the same states, five-role probe and panel arrangement as Figure 5, with 5% rather than 1% offsets. Under positive User offsets, mean Userness reaches 99.95% for originally User text, 77.0% for reasoning text and 77.2% for Assistant text. This is still offline rescoring of one conversation.* [Data][offset-rows] · [PDF][large-pdf]

The large User offset makes the passage groups less separated on the Userness axis. That observation is specific to this direction and readout: other panels do not all lose separation as the offset grows.


## 6. How much should we trust the probes?

A role score is only as interpretable as the classifier producing it. We therefore check how often the five-role probes correctly classify held-out role-labeled tokens across all 24 layers.

This measure is **recall for each role**: among tokens whose true label is User, for example, what fraction receive User as the highest-scoring class? Unlike the earlier figures, this plot shows classification correctness, not an average predicted probability.

![Figure 7. Five-role recall across all 24 layers under two train/test splits. User and Assistant are often easier to identify than CoT and Tool. The left split holds out prompt variants; the right holds out complete base texts.](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/review-v0.2/05-probe-recall.png>)

*Figure 7 — Held-out role recall varies by layer and split. Each curve is correct-role classification among held-out tokens of that role, reconstructed from integer confusion counts. The left test set contains 124 prompt variants; the right contains 24 base texts with all their role variants held out together. Colors and marker shapes identify the true role labels of held-out tokens: User blue, CoT yellow/orange, Assistant green, System gray and Tool purple. Each role’s curve is token-weighted.* [Recalls][recall-data] · [PDF][recall-pdf]

User and Assistant are easier to classify than CoT and Tool through much of this run. At layer 12 in the left panel, recall is **76.7% for User**, **83.9% for Assistant**, **46.2% for CoT**, and **44.8% for Tool**. These are the five-role probes; they are not the four-role readout used for the MATS and gardening illustrations.

Figure 7 evaluates only the five-role classifiers on held-out, role-rendered neutral text. It does not validate the separately fitted four- and three-role probes, or establish that any probe measures authority in mixed conversations or edited activations.

The two panels also expose a design choice. In the left split, a base text can appear in training under one role and testing under another. In the right split, the test texts are absent from training in every role. These evaluations use different test sets and separately fitted probes, so the panel difference does not isolate a causal effect of text overlap. [Split records][splits]

Correct classification does not establish **calibration**—whether predictions assigned 80% probability are correct about 80% of the time. The saved fitting audit also leaves optimizer convergence unresolved: we have not established that fitting reached its intended solution. These are reasons to validate a chosen readout before making a stronger claim from it. [Estimator audit][estimator-audit]

## What would make the next experiment informative?

The open question is whether a role-directed edit survives the model’s remaining computation and changes something useful. The paper’s **Figure 23** compares readouts of original User and Assistant content across layers under correct tags, no tags and Tool tags. We would add intervention curves to that comparison for GPT-OSS-20B. It remains deferred; the offset illustrations above do not supply those curves.

We will compare two vector families separately: a **Tool-minus-User direction derived from activations**, and the **User and Tool coefficient directions from the probes**. The activation-derived direction is the normalized difference between average states of neutral passages placed in Tool and User contexts. It differs from a probe’s learned coefficients; the two families also use different intervention sites and scales. [Historical method][historical-report] Each needs its own baseline, opposite signs, and random directions of the same length. A zero offset must reproduce the unmodified run.

For a fixed conversation, the model would continue computing after the intervention; later probes would measure what survives. The prediction worth testing is a direction-specific change beyond the intervention layer. If it disappears immediately or is matched by random directions, the case for a specific propagated effect weakens. A mitigation claim would additionally require paired behavior and legitimate-task performance measurements.

Our planning floor is **200 independent conversations that pass the prespecified inclusion filters**, with balanced coverage of the planned dialogue sources and reserves for filtering losses. Final sample size must follow the smallest useful paired effect and desired precision, rather than treating 200 as automatically sufficient. The 200 prepared candidates are not yet 200 generated, eligible, measured conversations. [Deferred batch specification][backlog]

Before that run, we must resolve two source ambiguities: the final analyzed conversation count and the metric used for the published Figure 23. Appendix B records the evidence. We will keep mean probabilities and classification accuracy separate and weight conversations equally in the aggregate, rather than allowing long conversations to dominate. [Figure 23 source audit][fig23-contract]

## Appendix A. Exact measurements and reproducibility

**H100 model and probes.** The saved results in Figures 2–7 use `openai/gpt-oss-20b` with MXFP4 expert weights at model snapshot `6cee5e81ee83917806bbde320786a8fb61efebee`; the runtime versions are in the training record. The completed September 11 run trained 384 probes: 24 layers × eight role-class combinations × two split methods. The corpus contains 249 base texts and 1,245 role-rendered variants. Activations are taken at `post_attention_layernorm`, before the layer’s multilayer perceptron. Figures 1–2 use the saved prompt-split four-role probe; Figures 4–6 use the prompt-split five-role probe at the same layer-12 site. Figure 3 uses prompt-split five- and three-role probes at layers 8, 12 and 16. Training uses 62 C4 and 187 Dolma3 texts, each limited to 1,024 content tokens and rendered separately under each role. The probes are cuML 25.8 L2 logistic regressions with an intercept, C=0.005, no feature scaling, the QN solver, tolerance 1e−4 and at most 5,000 iterations. A 10% holdout with seed 123 groups by rendered prompt or base text; role-class combinations retain their own saved partitions. [Training implementation][training-code] [Training][training] · [Splits][splits] · [Estimator][estimator-audit]

**New local MATS exchange.** Figure 1 uses `mlx-community/gpt-oss-20b-MXFP4-Q8`, snapshot `773a7da77e569019bb0fd17a554b263738d669a3`, with the original tokenizer snapshot and saved H100 `suca_L12` probe. MLX 0.32.2 and mlx-lm 0.31.3 generated two complete analysis/final pairs at temperature zero with medium reasoning, bounded at 1,536 new tokens per turn. The turns completed at 406 and 399 tokens. We replayed the unchanged dialogue in three separate forwards, recording the pre-MLP site at zero-based layer 12. We save bfloat16 activations as float16 and project in float64 using saved float32 coefficients. The revised run took 25.31 seconds including model loading and peaked at 12.99 GB of MLX memory. [Inputs before generation][mats-freeze] · [Provenance][mats-provenance]

**MATS token selection and provenance.** Each format contains 949 non-System content tokens. Before scoring, we fixed a display rule: at most the first 120 tokens per passage, retaining identical token text and ID at the same passage-relative position across all three formats. This yields 578 displayed tokens, with passage counts 116, 102, 120, 40, 80 and 120. All 3,166 forwarded positions retain their scores. The no-tags condition joins the complete transcript, including System text, with newlines; the all-User condition wraps that joined text in one User message. This is the second generated attempt, following a revision to supply paper context and replace low reasoning with medium. The preceding attempt and twelve fully authored examples remain available. No token was selected by its score. [Token selection][mats-selection] · [Full token scores][mats-full-rows] · [Selection history][mats-choice]

**Local-runtime control.** A preceding control with the same local model, probe and runtime configuration runs the recorded gardening token sequences locally and compares the same 2,745 labeled token records, including System text, with H100 results. Mean absolute CoTness difference is 1.15 percentage points; the largest absolute difference across any token and class is 86.17 points. Backend, non-expert weight precision and batch padding differ together, so this comparison does not isolate their causes or establish calibration. Figure 1 is a measured MLX result using the saved H100 probe, not a numerically interchangeable H100 replication. [Comparison][mats-runtime]

**Gardening formatting and token selection.** The no-tags condition joins the complete transcript, including system text, with newlines. The all-User condition places that joined transcript inside one User message. Neither retains a separate System message. All three panels show 512 matched tokens: 95 originally User, 177 CoT and 240 Assistant. The display keeps at most the first 120 tokens of each non-System segment, then retains only token positions with identical token strings across all three renderings. This leaves 120 tokens from each of the two Assistant segments. Full-content counts are 97, 179 and 582 respectively. [Selection code][garden-selection] Displayed and full-content averages therefore answer different subset questions. [Rows][garden-rows] · [Subset audit][garden-result]

**Offline offsets.** For a saved activation `h`, we compute:

`h′ = h + s × f × m × (w / ||w||)`

Here `s` is −1, 0 or +1; `f` is 0.01 or 0.05; `w` is the saved User or Tool coefficient vector; and `m = 45.2471` is the median activation norm across all 921 forwarded tokens in the all-User condition. The offset lengths are approximately 0.4525 and 2.2624, shared across directions. They are fractions of this fixed reference, not of each token’s own length. Saved float16 activations and float32 weights are promoted to float64 for rescoring. [Arithmetic provenance][offset-provenance]

**Why the coefficient directions need care.** A linear probe computes a score for each class from the activation and a coefficient vector, then converts the scores to probabilities with softmax. Adding a common vector to every coefficient row leaves those probabilities unchanged, because it shifts all class scores equally. It nevertheless changes any individual row used as an offset. Our illustrations therefore depend on the saved parameterization; they do not identify unique User or Tool axes. This does not affect the arithmetic checks of the specified offsets.

**Permission inputs.** We sample 100 template records and initially one page per template; 26 page assignments are substituted because the original pages lack usable prose. The replacements are approximately token-matched. All prefills retain the imperative wrapper and the same supplied Assistant analysis sentence. [Frozen input contract][rh6-input-contract] · [Construction][rh6-construction]

**Permission estimator and uncertainty.** For each template–page unit and span, the relative score is `ln(mean P_User) − ln(mean P_Tool)`. We average probabilities over that span before taking logs. We then compute `(command permission − command neutral) − (replacement permission − replacement neutral)` within the paired unit and average over the 100 units. The intervals use 2,000 paired bootstrap resamples with seed 123. One resample draws 100 units with replacement and keeps each unit’s conditions together; the reported interval spans the 2.5th to 97.5th percentiles of the resulting means. A tokenwise log ratio equals the difference between that token’s User and Tool scores before softmax. Our mean-first ratio can also change when other classes redistribute probability across tokens, even with those tokenwise score differences fixed. This is a possible explanation, not an established cause of the observed result. We retain the measured estimator rather than retrospectively substituting another one. [Full analysis][rh6-result] · [Plotted estimates][rh6-data]

**Validation units.** The five-role prompt-split test set has 124 variants and 67,727 content tokens. The grouped split holds out 24 base texts, with 56,685 content tokens. Recall weights tokens equally within each true role. Tokens, offsets and layers are repeated measurements, not interchangeable with independent conversations or base texts. [Split audit][splits]

## Appendix B. Source discrepancies and evidence beyond the seven figures

**Figure 23.** The paper describes sampling 200 conversations. The frozen notebook selects at most 30 after filtering, with a comment suggesting 100 for a full test; these are different selection stages, and the final published denominator remains unresolved. The notebook averages eligible original-role content tokens within each conversation, then weights conversations equally. The paper labels Figure 23 as role probabilities; the frozen plotting notebook is configured to read correct-role accuracy files. Its saved execution outputs are empty, so these records do not establish which metric generated the published figure. We will report the two metrics separately. [Paper][paper-fig23] · [Published figure][paper-figure23] · [Selection notebook][paper-selection-notebook] · [Plotting notebook][paper-plotting-notebook] · [Audit][fig23-contract]

**Earlier behavioral evidence.** September 5 steering runs used actual interventions with recorded actions, under a different probe and setup. That task had two harmless marker-file writes: the page requested both, while the user allowed only one. Adding the activation-derived Tool-minus-User direction to the command tokens produced initial Tool probabilities above 99.89% in the recovered arm, yet 17 of 21 episodes wrote both files, including the unpermitted one. This is an incomplete recovered subset: 21 episodes spanning eight cases, with repeated prompts. This adverse result is a reason not to equate an extreme score with selective action control; it is not a reliable estimate of a general steering effect. [Historical task][historical-report] · [Evidence audit][evidence-audit]

**Other completed and deferred work.** The Appendix K neutral-position control starts from 200 texts that overlap the probe corpus, so it cannot substitute for a fresh held-out conversation study. Long-context experiments, the StrongREJECT harmful-request benchmark, source attribution and the remaining research questions stay in the full backlog. [Backlog][all-backlog] · [Conversation audit][goal-audit]

## Review status and sources

This is a draft for Hanan’s research and editorial review. Coding agents assisted with analysis, figures and writing. The numerical audit independently reconstructed all 30,720 offset probabilities, 60 plotted passage means (30 groups × two probabilities), 24 permission contrasts and their interval bounds, and 240 recall values from saved records. An independent [MATS audit][mats-audit] reproduced the new exchange’s probabilities at all 3,166 forwarded positions and checked every matched token, all 18 displayed passage means and the generated-text provenance. The earlier twelve-candidate batch was also audited and retained in the comparison gallery. The MATS exchanges and the earlier gardening runtime control were measured locally; the other six figures reuse saved measurements. The [independent review record][report-review] tracks recommendations and revisions; the [verification log][verification] records numerical checks. The six figure review copies retain their saved values, axes and paper palette, with clearer labels and redundant role markers. The new MATS figure uses the same role colors and typography. [Figure provenance][review-figure-provenance]

Figure numbers 1–7 belong to this report; the paper’s figure numbers retain the authors’ numbering. Local data links provide the measured evidence. The external references are:

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
[garden-pdf]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/review-v0.2/00-gardening.pdf>
[offset-method]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/presentation/probe-offset-illustration/README.md>
[offset-provenance]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/presentation/probe-offset-illustration/provenance.json>
[offset-rows]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/presentation/probe-offset-illustration/plotted-probabilities.csv>
[small-pdf]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/review-v0.2/01-small-offsets.pdf>
[large-pdf]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/review-v0.2/02-large-offsets.pdf>
[dose-data]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/offset-dose-response.csv>
[dose-pdf]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/review-v0.2/03-offset-dose-response.pdf>
[rh6-result]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/RH6-RESULTS.md>
[rh6-data]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/rh6-plotted.csv>
[rh6-pdf]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/review-v0.2/04-rh6-readout.pdf>
[recall-data]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/probe-recall-by-layer.csv>
[recall-pdf]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/review-v0.2/05-probe-recall.pdf>
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
[rh6-construction]: </Users/hananather/Desktop/MATS 12.0/replication/rh/rh6_readings.py>
[garden-selection]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/appendix_e_figures.py>
[historical-report]: </Users/hananather/Desktop/MATS 12.0/role-steering/report.md>
[paper-selection-notebook]: </Users/hananather/Desktop/MATS 12.0/prompt-injection-as-role-confusion/experiments/role-analysis/02-train-role-probes.ipynb>
[paper-plotting-notebook]: </Users/hananather/Desktop/MATS 12.0/prompt-injection-as-role-confusion/experiments/role-analysis/03-analyze-probes.ipynb>
[training-code]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/probes/train_probes_pod.py>
[rh6-input-contract]: </Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/rh6/input-contract.json>
[report-review]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/review/report-v0.2/README.md>
[review-figure-provenance]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/review-figures-provenance.json>

[mats-freeze]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/mats-hanan-dialogue-v2/prompt-freeze.json>
[mats-input]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/mats-hanan-dialogue-v2/conversation.md>
[mats-rows]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/mats-hanan-dialogue-v2/displayed-rows.csv>
[mats-pdf]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/mats-hanan-dialogue-v2/mats-hanan-dialogue-v2.pdf>
[mats-provenance]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/mats-hanan-dialogue-v2/provenance.json>
[mats-runtime]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/mats-example/gardening-runtime-comparison.json>
[mats-selection]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/mats-hanan-dialogue-v2/measurement-freeze.json>
[mats-full-rows]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/mats-hanan-dialogue-v2/token-probabilities.csv>

[mats-audit]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/mats-hanan-dialogue-v2/audit.md>

[mats-choice]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/mats-hanan-dialogue-v2/README.md>

[mats-gallery]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/mats-example-gallery.html>

[mats-full-pdf]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/mats-hanan-layouts/full-dialogue-balanced/full-dialogue-balanced.pdf>
[mats-layouts]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/mats-layout-gallery.html>

[mats-canonical]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/canonical/mats-dialogue-v1/README.md>

[forgery-companion]: </Users/hananather/Desktop/MATS 12.0/replication/chart-library/cot-forgery-steering.html>
