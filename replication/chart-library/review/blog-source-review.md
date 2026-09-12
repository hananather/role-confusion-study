# Blog source and citation review

I can support a blog about measured role readouts and their sensitivity to context and offline offsets. I cannot present the six-chart collection as a completed steering intervention on the running model, an exact numerical reproduction of the paper, or evidence that role probabilities measure subjective belief.

**Review scope:** Distill; source and measurement fidelity for the six-chart Markdown draft. I checked the live primary paper, the archived Figure 7 and Figure 23 images, the unchanged reference checkout at `ec333c40fd43fe991e1ebf66765051b6d7e35784`, and saved numerical artifacts. No article, figures, source data, or compute were changed.

## Material wording corrections

1. **State which probability is being measured.** A role probe is a fitted classifier of hidden activations. For the multiclass probes, the score is a softmax probability over the included roles. It is not the model's next-token probability, an attack-success probability, or a calibrated estimate of its subjective belief. “The probe assigns higher CoT probability” is supported; “the model believes/trusts this text” is the paper's interpretation and should be attributed if used.
2. **Keep four-role and five-role instruments distinct.** The gardening overview uses the `suca` probe at layer 12. The offline offset/dose views use `sucat`. Changing the classifier's class set changes the probabilities. The views share saved text and activations but not an unchanged readout instrument.
3. **Call the offset experiment offline.** Adding a saved coefficient row to a saved activation and rescoring it measures a classifier response. No transformer layers or new text generation follow that change. Individual softmax coefficient rows also depend on the saved parameterization; they are not unique semantic axes. These figures support a sensitivity demonstration, not an intervention effect on behavior.
4. **Use recall for the validation figure.** `probe-recall-by-layer.csv` reports the fraction of true-role tokens assigned to that role. That is per-role recall, not mean role probability. A prompt split can place variants of the same base text on both sides; the grouped split is a different partition, not a causal isolation of leakage.
5. **Keep Figure 23 deferred.** The current E9 files contain preparation and tests, not measured propagation results. The two real E4 smoke conversations are not a completed 200-conversation cohort. Synthetic fixtures cannot supply missing results.

## Stable public citations and supported claims

The primary citation is Charles Ye, Jasmine Cui, and Dylan Hadfield-Menell, *Prompt Injection as Role Confusion*, **arXiv:2603.12277v6**. I use the versioned URL so later revisions do not silently change the reference.

| Claim to attribute | Stable public citation | What the citation supports |
| --- | --- | --- |
| Role scores are classifier probabilities conditioned on hidden activations. | [Section 4.1](https://arxiv.org/html/2603.12277v6#S4.SS1), [Appendix G.2](https://arxiv.org/html/2603.12277v6#A7.SS2) | The definition of CoTness and the analogous role scores; multinomial logistic probes on content tokens. |
| The gardening example compares the same conversation under correct tags, no tags, and a user wrapper. | [Figure 7](https://arxiv.org/html/2603.12277v6#S4.F7), [Appendix E](https://arxiv.org/html/2603.12277v6#A5) | Experimental conditions and published CoTness values: 85/83/85% in Figure 7; 85/82/85% in Appendix E. |
| Figure 23 motivates a future layerwise comparison. | [Appendix F](https://arxiv.org/html/2603.12277v6#A6), [Figure 23](https://arxiv.org/html/2603.12277v6#A6.F23) | The paper describes 200 conversations and calls the plotted quantity role probability. This does not resolve the frozen-code discrepancy below. |

I should cite the paper for these source claims, then cite my own numerical artifact next to my result. A paper citation alone does not establish my measurements. The paper's stronger causal language does not become my experimental conclusion through citation.

## Frozen training recipe and deliberate differences

The [GPT-OSS-20B configuration](https://github.com/role-confusion/prompt-injection-as-role-confusion/blob/ec333c40fd43fe991e1ebf66765051b6d7e35784/experiments/role-analysis/config/probe.yaml#L3) specifies a nominal 250 texts, content truncation at 1,024 tokens, `C=0.005`, no scaling, and empty training prefixes. The [training notebook](https://github.com/role-confusion/prompt-injection-as-role-confusion/blob/ec333c40fd43fe991e1ebf66765051b6d7e35784/experiments/role-analysis/02-train-role-probes.ipynb#L441) uses cuML L2 logistic regression with intercept, `max_iter=5000`, `linesearch_max_iter=100`, and a 90/10 split of rendered prompt IDs. Integer rounding of the 25% C4 / 75% Dolma sampling produces **62+187=249** base texts, hence **1,245** role-wrapped prompts.

The [author forward implementation](https://github.com/role-confusion/prompt-injection-as-role-confusion/blob/ec333c40fd43fe991e1ebf66765051b6d7e35784/utils/pretrained_models/gptoss.py#L59) saves the normalized post-attention, pre-MLP activation. The source notebook selects even layers for GPT-OSS-20B. My run adds all 24 layers and the base-text-grouped split, uses extraction batch 16 rather than 32, and stores the same activation stream in per-layer files. Thus “I followed the authors' probe recipe with documented execution changes and two explicit extensions” is more accurate than “I ran the exact experiment unchanged.”

My [training metadata](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/metadata.json>) records the actual H100 run, MXFP4 expert dtype, model snapshot, cuML version, corpus counts, and 384 completed probes. The [native probe audit](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/native-probe-audit.json>) preserves estimator checks. Finite coefficients or stopping before 5,000 iterations do not by themselves establish optimizer convergence.

## Gardening numbers and denominators

I independently recomputed displayed CoT means from `prob_unrounded` in the two displayed-row CSVs. They agree with the stored summary at the reported two-decimal precision; serialization differences occur only below that precision.

| Condition | My full-content / Figures 20–22 CoT mean | My Figure 7 displayed CoT mean | Published Appendix E / Figure 7 |
| --- | --- | --- | --- |
| Correct tags | 64.03% | 64.33% | 85% / 85% |
| No tags | 71.04% | 71.78% | 82% / 83% |
| All in user tags | 70.97% | 71.72% | 85% / 85% |

My full-content user/CoT/assistant denominators are **97/179/582 per condition**. Figures 20–22 retain at most 160 tokens per consecutive source-role segment, giving **97/179/320**. Figure 7 retains at most 120 per segment and matches token identity and position across conditions, giving **95/177/240**, or **512 displayed tokens per condition**. These are token counts in one conversation, not independent experimental samples.

The [gardening plotting notebook](https://github.com/role-confusion/prompt-injection-as-role-confusion/blob/ec333c40fd43fe991e1ebf66765051b6d7e35784/experiments/role-analysis/04-tomato-probe-results.ipynb) supplies those display rules: cell 7 for Figures 20–22, cell 14 for Figure 7. Cell 14 computes an unused smoothed series but actually plots raw `prob`; there is no plotted smoothing. The notebook does not establish which denominator produced the paper's prose percentages. I therefore report a numerical gap without claiming an exact like-for-like estimate or blaming a particular implementation choice for it.

Direct local evidence:

- [Full and displayed-subset means](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-e-L12/mean-role-probabilities.csv>) and [published-comparison table](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-e-L12/published-comparison.csv>).
- [Figure 7 displayed rows](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-e-L12/figure-7-displayed-rows.csv>) and [Figures 20–22 displayed rows](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-e-L12/figure-20-22-displayed-rows.csv>).
- [Gardening run provenance](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-e-L12/metadata.json>), including native estimator, template hash, source cells, activation cast, and model snapshot.

## Figure 23: paper versus executable recipe

In the frozen [training notebook, cell 25](https://github.com/role-confusion/prompt-injection-as-role-confusion/blob/ec333c40fd43fe991e1ebf66765051b6d7e35784/experiments/role-analysis/02-train-role-probes.ipynb#L843), conversation filtering is followed by `max_samples=30`; its comment says 100 for a full test. Cells 37 and 38 separately export mean probability and hard argmax accuracy. Both first average eligible tokens of each original role **within each conversation**, then give conversations equal weight.

The [plot loader](https://github.com/role-confusion/prompt-injection-as-role-confusion/blob/ec333c40fd43fe991e1ebf66765051b6d7e35784/experiments/role-analysis/03-analyze-probes.ipynb#L116) reads `all_conv_acc`, and cell 8 plots `uat` with tagged, untagged, and tool-tagged conditions. CoT remains in context but is excluded from the displayed user/final-assistant denominators. There is no gardening-style first-120/160 truncation or common-token intersection. For future work I keep probability and accuracy separately named. I cannot infer the paper's actual run cohort from the frozen notebook default.

## Palette and figure semantics

The [gardening palette](https://github.com/role-confusion/prompt-injection-as-role-confusion/blob/ec333c40fd43fe991e1ebf66765051b6d7e35784/experiments/role-analysis/04-tomato-probe-results.ipynb#L155) assigns **point colors by original token source**: User `#00a6f4`, CoT `#fd9a00`, Assistant `#00d492`, System `#90a1b9`. Text annotations use darker variants. Tool purple `#7e6cff` comes from the [agent-analysis palette](https://github.com/role-confusion/prompt-injection-as-role-confusion/blob/ec333c40fd43fe991e1ebf66765051b6d7e35784/experiments/cot-forgery-role-confusion/05-analyze-agent-probe-result.ipynb#L341).

Figure 23 instead [colors conditions](https://github.com/role-confusion/prompt-injection-as-role-confusion/blob/ec333c40fd43fe991e1ebf66765051b6d7e35784/experiments/role-analysis/03-analyze-probes.ipynb#L221): Baseline `#62748e`, No tags `#00a6f4`, Injection `#ff6467`. The shared [paper theme](https://github.com/role-confusion/prompt-injection-as-role-confusion/blob/ec333c40fd43fe991e1ebf66765051b6d7e35784/r-utils/plots.r#L1) uses TeX Gyre Termes, white background, boxed panels, and light major grid. The archived figure images visually match the relevant raw-point/line recipes. Matching a palette does not make a new chart an original paper figure.

For the six-chart blog, the remaining exact plotted sources are [offset dose values](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/offset-dose-response.csv>), [RH6 plotted values](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/rh6-plotted.csv>) with [RH6 provenance](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/rh6-provenance.json>), and [probe recall](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/probe-recall-by-layer.csv>). The [render provenance](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/render-provenance.json>) links display assets to their inputs. These support my measurements; public paper links provide context and method attribution.

## Final draft citation pass

**Reviewed file:** [role-signals-blog-draft.md](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/role-signals-blog-draft.md>), 27,115 bytes, SHA-256 `38b6da6972c807cfde23fe51584fa29b92e9ac54dd8d7a8cd6f16242af325fb4`. This finding applies to that exact draft, before any subsequent editorial edits. The draft was read in full; it was not edited by this review.

**Verdict:** no fabricated paper claim, role-score definition, or material denominator error found. The draft correctly separates the four-role gardening probe from five-role offset views, marks the offsets as offline arithmetic, distinguishes recall from probability, preserves the 82%/83% source inconsistency, and keeps Figure 23 propagation deferred. Its Figure 23 aggregation and 200/30/100 distinctions match the sources. Numerical/statistical auditing is covered by the separate review rather than asserted complete here.

The remaining narrow precision edits are:

- **Role taxonomy, draft line 20:** CoT is accurately identified as an analytical role. Adding “In this GPT-OSS rendering, it is text in the Assistant analysis channel” would prevent readers inferring a fifth ordinary message role. The saved [correctly tagged prompt](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-e-L12/prompt-proper_tags.txt>) uses Assistant `analysis` and `final` channels, with the author's frozen template.
- **Formatting, line 32:** the current wording is broadly correct. The exact manipulation includes the original system text: no-tags joins every original message with newlines after the beginning-of-sequence token, while all-user places that whole joined text inside a single User message. A short clarification should avoid implying that a separate, correctly tagged System message survives those conditions. The [no-tags prompt](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-e-L12/prompt-basic_no_format.txt>) and [all-user prompt](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-e-L12/prompt-everything_in_user_tags.txt>) establish this directly.
- **Opening definition, line 16:** “influence the model's response” is broader than prompt injection, because legitimate retrieved content should influence a response too. Prefer wording about instructions in untrusted content redirecting the model beyond the authorized task.
- **Citation precision:** narrow `[paper-role]` from `#S4` to `#S4.SS1`. Add `#S4.F7` beside the gardening comparison to directly support the Figure 7-specific 83%. Keep `#A6` for the 200-conversation description; `#A6.F23` is the more precise additional link for Figure 23's probability caption. The current broader anchors are valid, so these changes improve navigation rather than repair broken evidence.

No additional experiment, dataset replacement, or changed estimand is required by this source review.
