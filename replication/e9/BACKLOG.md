# My future E9 batch

**Status: deferred by my September 11, 2026 decision. No E9 generation, forward measurement, or propagated-steering result is complete.** I preserve the preparation and synthetic tests. I do not rent compute or start a model job from this backlog.

I am in **Explore**, with an **Understand** target: distinguish a persistent, direction-specific perturbation from a local probe response, generic perturbation, or formatting effect. The comparison uses both selected vector families separately. It does not claim a new mechanism or improved instruction following merely because a plotted score moves.

## 1. Freeze a sufficiently large evaluation cohort

- I require **at least 200 eligible independent conversations in every paired analysis**, after target-model generation and all exclusions. I target balanced OASST/ToxicChat representation; at the floor that means 100 eligible conversations from each source. A different source mix requires an explicit decision and disclosure rather than an automatic fallback.
- I retain the existing 200-candidate freeze and create a separately versioned reserve expansion. I choose source IDs and reserve order before inspecting intervention outcomes. I recruit enough candidates to reach the final target after missing reasoning/final responses, message-length limits, substring ambiguity, token-label failures, or other prespecified exclusions. I record counts and reasons by source at every stage.
- I specify the smallest scientifically meaningful paired effect, primary contrast and layer endpoint, desired interval width or power, and multiplicity rule before choosing the final sample. I use the variance of conversation-level paired differences, not a binary-ASR shortcut or token count, to justify the sample. A 200-conversation floor may need to increase.
- I preserve one complete paired block per conversation across all planned arms and layers. Multiple turns or sampled generations are clustered under their source conversation; they do not inflate the independent sample count. If the eligible cohort falls short, I expand under the frozen reserve rule or pause for my decision. I do not turn a small smoke run into the claim-producing evaluation.

## 2. Preserve the Figure 23 source and its ambiguities

I inspected the supplied paper and frozen authors' repository at `ec333c40fd43fe991e1ebf66765051b6d7e35784`:

- The paper's Appendix F says **200 conversations** from OpenAssistant and ToxicChat, with target-model responses regenerated. NB02 cell 25 filters first, then requests `max_samples = 30`, followed by the comment “100 for full test.” It samples `min(max_samples, eligible_count)`. These statements do not establish the denominator or source mix actually used in the published figure.
- NB02 cells 37 and 38 average **all eligible tokens of each original role within each prompt/conversation**, then average those conversation-level values equally. I do not use a 512-token display subset, truncate to a short first segment, pool all tokens across conversations, or give equal weight to turns.
- User-style and final-assistant-style content supply the two plotted rows. CoT remains in the forward context but does not enter those two denominators. I exclude unmatched prefix/tag/separator tokens under the frozen labeling rule and keep cohort exclusions paired across conditions and arms.
- The paper describes probabilities; NB03 `03-analyze-probes.ipynb` cell 8 reads the accuracy outputs from NB02 cell 38. I retain separately named **argmax correct-role accuracy** and **mean original-role probability**. The source uses `uat` classifiers at even layers 0–22; the requested 0–23 traces explicitly extend that coverage.

The [source contract](SOURCE-CONTRACT.md) records exact rendering and filtering. I do not silently resolve paper/code differences by claiming one was the measured publication configuration.

## 3. Freeze the intervention and analysis before execution

I keep the historical Tool-minus-User activation direction separate from the User and Tool classifier-row directions. I record each vector's model, layer, coordinate space, site, normalization, and hash. I do not interchange residual-output and normalized pre-MLP vectors.

The future protocol must specify the actual injection site, token mask, signed strengths, and downstream continuation. I retain zero, opposite-sign, and three seeded norm-matched random controls, fixed text within every intervention comparison, and all-layer readouts. Any small/large dose comparison must be encoded explicitly in the data and protocol; the current analysis CSV contract has one positive and one negative arm per family and does not silently represent a two-dose sweep.

I will verify the zero intervention, unchanged upstream measurements, correct first affected layer, and complete paired token/cohort coverage before interpreting propagation. I average conversations equally and use the same 2,000 conversation-bootstrap draws, seed 123, across cells. I retain full intervals and paired differences against the tool-tagged zero condition. The existing intervals are pointwise, not simultaneous bands.

My falsifier for a direction-specific persistence claim is an effect that disappears downstream or is matched by the norm-matched random controls at the prespecified endpoint. A persistent readout change alone does not establish a role mechanism, successful mitigation, or behavioral dissociation; those require separately justified evidence.

## 4. Keep the visual story and color meanings distinct

1. **Figure 23 comparison:** colors identify formatting condition—Baseline gray `#62748e`, No Tags blue `#00a6f4`, Injection red `#ff6467`. I keep User/Assistant rows and layer on the horizontal axis. Historical and classifier-vector families occupy separate panels; actual probabilities and Toolness have companion figures. Random controls remain available without crowding the main comparison.
2. **Figure 7-style passage view:** blue/yellow/green identify original User/CoT/Assistant passages. I preserve that meaning. I show zero/small/large offsets in separate panels or through an explicitly labeled non-color encoding; those three passage colors do not become strength labels.
3. **Source metrics and extension claims:** the baseline source comparison and new intervention results have distinct titles and legends. I retain the original role when reporting correct-role predictions. I label offline illustrations and synthetic fixtures as such, rather than presenting them as propagated model results. Source-faithful publication formatting and title/layout checks remain a final review step.

## Release condition for the future batch

I resume only after a concrete protocol fixes the final eligible cohort, sample justification, vector/site/strength definitions, controls, outputs, and compute budget, and I explicitly authorize that batch. Preparation and passing synthetic tests are not a launch instruction.
