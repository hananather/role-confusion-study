# Reviewer 1 — Source fidelity and measured claims

I reviewed `report.md`, `prepare_evidence.py`, `render_figures.py`, and the rendered first two figures against my independently reconstructed batch. This review covers the current upload/probe evidence; the separate historical permission figure has its own source audit.

**Status: PASS. The requested wording correction is resolved; no material source-fidelity recommendations remain in this review’s scope.**

## Initial material correction — resolved

Section 2 says, “The original vector changes the probe in every measured case.” There are no saved no-intervention probe readings for the five new pages. The complete ten-page dataset establishes that the original vector **produces a near-zero CoT score on every measured page**, with six verified uploads. A measured before/after reduction is supported by the five historical zero-dose comparisons. Replace the sentence with this level/contrast distinction; the TLDR’s “drives ... below 0.01%” can likewise be “produces ... below 0.01%.”

This does not change the central conclusion. Low scores coexist with actual uploads in the recorded episodes; the favorable new construction remains visible.

I checked the revised report: the TLDR now says the vector “produces” scores below 0.01%, and Section 2 explicitly distinguishes the measured reduction on the five historical zero-dose comparisons from the near-zero scores observed across all ten pages. The methods retain the missing new-page baseline boundary. This resolves the concern without weakening the supported result.

## Checks that pass

- All **190 first-postfetch span readouts**, including all **60 forged-passage rows**, exactly match the independent reduction for all five class probabilities, token counts, outcomes, prompt hashes and episode hashes.
- All **110 assigned outcomes** match. All **70 recorded intervention/reference paired changes** match after translating the display terms prevented/introduced/unchanged to favorable/adverse/same endpoint.
- Figure 1 includes every forgery arm, the three random controls, the reverse direction, the reminder, historical zero-dose and historical Tool-raising. Its unresolved, unrun and unassigned states remain distinct. Its total counts and adverse original-vector cell are correct.
- Figure 2 uses forged-passage probabilities, not whole-page averages or engineering-gate results. It shows the sixty measurements without adding nonexistent no-hook probe scores. Censored episodes remain crosses. Every plotted probability and outcome agrees with the source.
- Display Random 1/2/3 maps to saved `random_0_a16`/`random_1_a16`/`random_2_a16`. The report’s Random 2 discussion correctly refers to the second displayed control, which retains approximately 61% mean CoT scores and has two uploads.
- Methods identify the steering site and readout site correctly: block 11 output and block 12 post-attention normalization, both zero-based. The text states that saved mean scores were checked against independent copies and that hidden tensors were unavailable for reprojection.
- The report keeps observed no-upload behavior separate from summary accuracy, separates this cohort from the earlier permission study, preserves the positive 0/5 Tool-raising result and its five missing new-page tests, and avoids an equivalence/no-effect claim.
- The displayed PNGs have no visible clipped labels, hidden outcome categories or mismatched legends. Figure 1’s green cells mean no upload, not successful task completion; the surrounding explanation makes that boundary explicit.

## Provenance of this pass

Initial reviewed report SHA-256: `f737f7eee36632135d350384b3d5de432c5f6366f70fccbc5c84538ef79a62f3`.

Resolution-pass report SHA-256: `3602390056dd9c48f9125a315b68d09f374879e3edcad69f353d042a610ed88b`.

The comparison script used the independent `full-batch-audit-first-postfetch-probes.csv`, `full-batch-audit-outcomes.csv` and `full-batch-audit-paired-changes.csv` against the corresponding final `data/` files. No source data, report prose or figure renderer was edited during this review.
