# Independent review 06 — source fidelity

Report: `draft-r1.md`  
SHA-256: `022c1f4958500451a8568f2749e7878964f69607b023c692130acac016c7db4b`  
Lens: Primary-source fidelity, attribution, paper/code distinctions, and accurate novelty claims  
Verdict: **revise** — one bounded source-attribution correction.

## Strongest accurate contribution

The strongest contribution is the concrete demonstration that the direction of a contextual effect depends on the chosen role classifier and score: the marker-permission comparison can raise relative User/Tool while lowering absolute Userness, and separately fitted role spaces can give different absolute signs. The report connects this finding to an interpretable worked example and an explicitly arithmetic sensitivity analysis. It attributes the probes and gardening construction to Ye, Cui and Hadfield-Menell, presents the new measurements as exploratory local results, and makes no unsupported claim to a new general mechanism or effective defense.

## My retelling

The report asks how much meaning to assign to a role-probe probability. A local gardening run preserves some of the authors’ visible role pattern, but misses their reported probabilities. Moving saved states along classifier coefficients changes scores without testing subsequent model computation, and small Tool offsets can raise both Toolness and Userness. Across 100 constructed input pairs, permission effects depend on the probe and metric. Held-out recall gives a partial measurement check, while calibration and fitting convergence remain unresolved. The next planned experiment would test whether specified interventions persist through later layers, using independently sampled conversations and separate behavioral validation if mitigation is claimed.

## Recommendation R06-01 — leave the published Figure 23 metric unresolved

**Exact passages**

> Before that run, the source contract needs to be explicit: the paper and notebook disagree about both sample count and which quantity is plotted.

> The paper describes probabilities, while the plotting cell reads correct-role accuracy outputs.

**Concern.** This describes a confirmed disagreement about the published figure, although the source evidence only establishes a difference between the paper's description and the frozen notebook's configuration. The paper says it samples 200 conversations; NB02 cell 25 applies eligibility filters and then samples at most 30, which refers to a later selection stage. NB03 cells 6 and 8 read and plot accuracy files, but their saved execution outputs are empty. The draft correctly leaves the published denominator unresolved and should apply the same explicit uncertainty to the published metric.

**Suggested remedy.** Replace the main-text sentence with: “Before that run, we need to resolve two source ambiguities: the final analyzed conversation count and the metric used for the published Figure 23.” In Appendix B, state that the paper describes sampling 200 conversations, whereas the frozen notebook selects at most 30 after filtering, and add: “The paper labels Figure 23 as role probabilities; the frozen plotting notebook is configured to read correct-role accuracy files. Its saved execution outputs are empty, so these records do not establish which metric generated the published figure.” Cite the frozen NB02 and NB03 files beside that account; retain the plan to report both metrics separately.

**Underlying evidence.** I read the paper’s Appendix F and the notebook cells directly from commit `ec333c40fd43fe991e1ebf66765051b6d7e35784`, rather than accepting the source-contract audit as sufficient evidence. NB02 cell 25 uses `min(max_samples, len(df))` with `max_samples = 30` and the comment “100 for full test.” Cells 37/45 export mean probabilities to projection files; cells 38/46 separately export argmax accuracy. NB03 cell 6 reads the accuracy files, and cell 8 plots their `mean_acc` column for the `uat` role space. These are source-code observations; they do not identify the published image’s execution provenance. [Paper, Appendix F](https://arxiv.org/html/2603.12277v6#A6) · [Frozen training and projection notebook](https://github.com/role-confusion/prompt-injection-as-role-confusion/blob/ec333c40fd43fe991e1ebf66765051b6d7e35784/experiments/role-analysis/02-train-role-probes.ipynb) · [Frozen plotting notebook](https://github.com/role-confusion/prompt-injection-as-role-confusion/blob/ec333c40fd43fe991e1ebf66765051b6d7e35784/experiments/role-analysis/03-analyze-probes.ipynb)

This requires a prose correction, with no experiment or change to the planned method.

## Source checks that support the existing account

- **Figure 7 is distinguished from the local Figure 1.** The paper reports 85% / 83% / 85% in its Figure 7 account; Appendix E reports 82% for no tags. The draft preserves that discrepancy. I independently reduced the displayed local CSV to 177 CoT tokens per condition: 64.3294%, 71.7761%, and 71.7160%, which round to the report’s values. The full-content denominator of 179 and its distinct means are documented in the saved gardening result. The draft does not describe either subset as an exact numerical replication. [Paper’s Figure 7](https://arxiv.org/html/2603.12277v6#S4.F7) · [Appendix E](https://arxiv.org/html/2603.12277v6#A5)
- **The gardening formatting is accurately disclosed.** Frozen NB02 cells 48–49 put the system text into the newline-joined no-tags and all-User strings; the separately tagged System message appears in the correctly tagged condition. Frozen NB04 cell 14 selects the matched token positions after the 120-token-per-segment cap and plots raw probabilities. This supports the local caption and the Appendix A description. I found no basis to call the local formatting an undocumented departure from that frozen notebook.
- **The training metadata supports the stated local scope.** It records 249 base documents, 1,245 rendered prompts, 24 layers, two split methods, 384 probes, and the frozen authors’ commit. The report appropriately identifies the separate four-role, five-role, and three-role classifiers. It does not present the additional layers or grouped split as the authors’ original run.
- **Old behavioral evidence is kept distinct.** I read the recovered September 5 episode archive directly. The `tool_03` arm has 21 episodes across eight cases; 17 wrote both markers. Across 1,241 initial scored command tokens, minimum Tool probability is 0.9989931. The 21 episodes contain 15 distinct initial prompts. The draft’s short historical paragraph is supported, retains the dependence warning, and does not import this result into the six new figures or claim a general steering failure.
- **References are usable.** All 34 defined reference labels resolve; every local reference target exists. The arXiv v6 pages and the frozen GitHub commit link opened successfully. The numerical/source audits are clearly local records rather than the paper itself.

## Scientific uncertainty already disclosed correctly

The draft already states that the comparisons were selected after inspection; gardening is one conversation; the probability mismatch is unresolved; offline offsets have no downstream computation or behavioral outcome; coefficient rows depend on the saved parameterization; the permission control has residual token-position and lexical differences; bootstrap intervals are pointwise over the same 100 paired units; recall does not establish calibration; fitting convergence is unresolved; historical episodes are sparse and repeated; and the future 200-candidate pool is not an eligible measured cohort. These uncertainties do not require new science for writing approval, and I do not recommend turning them into stronger claims or adding an unsupported novelty claim.

## Verdict

**Revise** the Figure 23 source account as specified in R06-01. The report otherwise maintains the primary-source and local-evidence boundary well within my lens. I recommend no numerical correction, replacement of a central claim, or new experiment as a condition of this review.

I read the complete frozen report and used the supplied Neel Nanda writing guidance for claim strength and attribution. I did not read another reviewer’s verdict, the narrative-card, or the mutable report. I changed no source record, figure, or report and ran no model.

