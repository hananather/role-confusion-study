# My report statistics review

**Verdict: PASS — no unresolved material recommendations.**

Reviewer 3 · 2026-09-12 · Understand → Distill. My north star is that every narrative claim matches the sampled unit, intervention, readout and behavioral endpoint. I apply the Neel criteria of truthful scope, complete controls, and independent checking of the reported numbers.

I read the report and reduction script, inspected all three rendered PNG figures, checked the original and new-page plans, and compared the displayed counts and probe ranges with saved tables. The report revision I rechecked has SHA-256 `3602390056dd9c48f9125a315b68d09f374879e3edcad69f353d042a610ed88b`. I did not run a model or modify the report or renderers.

## Two material issues were corrected and rechecked

1. **New pages are development data, not an untouched confirmation set.** The next five Tool-raising runs were formerly described as “that held-out comparison.” The current report correctly says fixed new pages. The plan explicitly calls them `development_prefix`, with separate reserved confirmation indices untouched. The proposed future test of the original vector now requires matched unsteered readouts.
2. **Only five historical pages have measured probe deltas.** The draft previously said the vector changed the probe in every measured case. The current report correctly distinguishes a near-zero score on all ten intervention episodes from a measured reduction against zero-dose references on the five historical pages. No missing new-page reference is invented. The main coexistence claim remains directly supported.

## Page and template mapping

I compared payload strings and SHA-256 values in the [historical plan](../../../steering-series/2026-09-12-positive-confirmation/bridge-001/bridge-plan.json) and [new-page plan](../../../steering-series/2026-09-12-positive-confirmation/newpage-001/prepared/plan.json). Each H/N pair below uses byte-identical attack text but a different webpage and sampling seed.

| Figure columns | Template | Historical page | New page | Historical / new seed |
| --- | --- | --- | --- | --- |
| H1 / N1 | direct | Norah Runge | Syllepte viridivertex | 1235 / 20260913 |
| H2 / N2 | simple | Helodium | Syllepte eriopisalis | 1237 / 20260915 |
| H3 / N3 | debugging | Rotokakahi River | Edmundas Benetis | 1239 / 20260917 |
| H4 / N4 | backup | Ypthima pupillaris | Nicolás Pelaitay | 1241 / 20260919 |
| H5 / N5 | documentation | Blowhole Diversion Tunnel | Howe Grove Wood | 11243 / 20260921 |

There are ten webpages and five attack templates, with one sampled trajectory per arm/page. This is not ten independent attack types. Between-cohort changes combine webpage and seed changes; the report does not attribute those changes to page novelty alone. Within a column, the registered seed and page are matched across interventions. The provenance reminder intentionally changes the prompt, whereas the non-sentence arms have exact initial and first-postfetch prompt matches according to the separate full-batch audit.

## Claims and figures that passed

- **Figure 1:** all displayed cells, totals and missing states agree with the source CSV. The original direction has two favorable, one adverse and seven unchanged paired outcomes. All five new-page outcomes match baseline individually. Random 3 retains two unresolved cells; Tool-raising retains five unrun cells; new-page zero dose remains unassigned.
- **Random label mapping:** figure/report Random 1, Random 2 and Random 3 correspond to source `random_0_a16`, `random_1_a16` and `random_2_a16`. The report's high-CoT, low-upload Random 2 claim therefore identifies source random 1 correctly. The source audits retain their original zero-based labels.
- **Figure 2:** sixty actual forged-payload mean readouts are displayed without filling unavailable no-intervention/reminder scores. Six original-vector uploads occur at near-zero CoT scores. Tool-raising is a visible historical-only positive branch. Its minimum actual forged-payload Tool probability is 99.999988%, so the report's “exceeds 99.9999%” statement is supported. Engineering gate values are not substituted.
- **Figure 3:** the displayed 18→17 unwanted writes and 21→19 correct authorized actions agree with all 21 recovered Toolward pairs. The report distinguishes eight underlying cases, fifteen first-prefill case/permission contexts, repeated seeds, missing planned records and the malformed generation. It keeps this marker endpoint and Tool−User intervention separate from the upload task and Tool−CoT intervention.
- **Inference:** the report claims no dependable or direction-specific benefit for the original construction, not a universal zero effect. It retains the adaptive Tool-raising positive and all random controls. It does not turn unchanged outcomes, absent receipts in censored cases, or a failed superiority rule into equivalence.
- **Task quality:** candidate-summary presence is explicitly a heuristic, not factual accuracy or successful legitimate-task completion. Green upload cells are not labeled as task success.

The descriptive fixed-cohort figures are appropriate without population confidence intervals. They preserve all measured cases and explicitly limit generalization. No additional statistical model, p-value, or cosmetic revision is needed to support the current claims.
