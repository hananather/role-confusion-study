# My adversarial claim review

**Verdict: PASS — no unresolved material recommendations.** I verified that the revised follow-up criterion resolves AC1. The original finding and recommendation remain below as review history. I found no evidence-based reason to replace the report with a blanket null, remove the favorable Tool-raising result, or weaken the observed probe–behavior finding.

*12 September 2026 · Reviewer 6 · Understand → Distill. My north star is whether changing a measured role score supports a claim about useful, selective behavior. I apply Neel's criteria of clear claims, matched baselines, inspection of evidence and preservation of adverse results.*

I reviewed the saved [report](../report.md), [arm counts](../data/arm-counts.csv), [payload scores](../data/forgery-payload-probes.csv), [paired changes](../data/paired-changes.csv), [complete batch audit](full-batch-audit.md), [historical audit](historical-evidence-audit.md) and [statistical review](claims-statistics-review.md). I independently reduced the supplied score and count tables; I rely on the source audits for their separate receipt and hidden-state provenance checks. I loaded no model and changed no report or code.

Original reviewed report SHA-256: `f451f8cb2550708c1f9303b1a28d56befc8654cebe0dfe5b34cb4acd72423672`.

AC1 resolution recheck, 12 September 2026: report SHA-256 `2ec3513b6f29534ea32543d963a1fabe81044b606b3126acb943e8f2f1f52fc0`. I reread the revised Section 4 from the current file; this was a wording recheck and introduced no new empirical claim.

## The strongest supported insight

I can substantially change this role classifier's output without preventing the unwanted action. Under the original Tool−CoT vector, all ten mean payload CoT probabilities lie between 3.49 × 10⁻¹⁹ and 2.09 × 10⁻¹⁸; six complete episodes still upload. Every historical page has a matched zero-dose reference, and each shows a large reduction from a reference score between 52.84% and 67.45%. Thus this is a measured intervention/readout mismatch with persistent attacks, rather than a selected near-zero example.

The behavioral evidence also explains why the report should resist a null about steering generally. Original steering prevents two uploads and introduces one. All five new-page endpoints remain unchanged. The displayed Random 2 (`random_1_a16`) produces two uploads while retaining a mean payload CoT score of 60.80%. Tool-raising produces no uploads on its five historical pages, with five new assignments unrun. These facts support the report's three-part story: a changed readout alone cannot certify prevention; the original construction has not demonstrated a role-specific advantage; a different construction remains an exploratory positive.

The strongest alternative explanation is that the original construction suppresses CoT while leaving a mixture of User and Tool scores, whereas the later construction targets both competing roles. Nonspecific perturbation also remains plausible because the random controls change outcomes. Neither explanation is resolved by the current scores and receipts. The report correctly presents construction choice as an open question rather than an established mechanism.

## Original material clarification — resolved

**AC1 — Keep predictive usefulness distinct from certification of prevention.** At reviewed report line 69, “If the original vector's lower reasoning-role score consistently predicted fewer attacks … the present concern would weaken” blends two claims. Future predictive association could establish usefulness within a specified setting; it would not erase this cohort's counterexamples to treating score suppression alone as evidence that an attack was prevented. It also would not, by itself, identify the role representation as a causal mediator.

I recommend this narrow replacement for that sentence:

> A future matched comparison could establish whether this readout predicts upload risk within a specified intervention and task distribution. The present episodes would still show why a lower score alone cannot certify attack prevention.

This preserves a meaningful future test without silently changing the present claim. I recommend no removals or changes to the central result, intervention, endpoint, or proposed fixed-page follow-up.

**Resolution:** Section 4 now says a future association would support the readout as a predictor “in that setting,” followed by “It would not erase the counterexamples above.” This explicitly separates scoped predictive usefulness from the existing failures of score-only certification. AC1 is resolved; I recommend no further change.

## Adversarial checks that passed

- **Causal claim:** the report distinguishes the measured classifier output from obedience and does not claim that a role mechanism was identified. Showing a near-zero score with an upload is sufficient for its local counterexample; population significance is unnecessary for that existence claim.
- **Generalization and equivalence:** the report names ten pages, five reused templates, one trajectory per arm/page, adaptive historical evaluation and missing new Tool-raising tests. It does not turn unchanged binary outcomes or a failed superiority comparison into proof of no effect.
- **Selection:** all three random directions, reverse steering, the reminder, the favorable Tool-raising branch and the adverse original-vector pair remain visible. Historical and new pages stay separate. The selected four-condition illustration does not determine the conclusion.
- **Censoring:** the two unresolved forgery outcomes belong to `random_2_a16`, displayed as Random 3. Its four recorded uploads could become four to six eventual uploads; “fewer recorded uploads” is a defensible descriptive statement, not an established superiority claim. I verified the one-based display mapping in the renderer.
- **Utility and historical scope:** completed nonupload is not labelled factual task success. The marker study keeps its distinct vector, mask, endpoint, repeated contexts and incomplete recovery boundary. Its corroborating measurement lesson is not presented as a replication of the upload intervention.

With AC1 resolved, I see no unresolved claim-level objection that would reverse the report's recommendation. Additional stylistic tightening would not change the scientific decision.
