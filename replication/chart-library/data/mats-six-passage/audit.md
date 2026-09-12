# Independent audit: twelve six-passage MATS examples

**Verdict: all twelve saved measurements pass the numerical and display-selection checks below.** I did not load or run a model for this audit. I independently recomputed probabilities from the saved activation arrays and reconstructed the text and passage assignments.

## What was measured

Each candidate is an authored illustration with the ordered passages User 1, CoT 1, Assistant 1, User 2, CoT 2, Assistant 2. The three conditions forward the same text with correct tags, no tags, or all text inside one User message. The System message remains in each full input. These are supplied texts, not generated reasoning or real conversations with Neel.

All twelve candidates and their inputs were frozen before the recorded completion. The source, input, executing-script and probe hashes match the saved provenance. The batch contains 36 forwards and 21,456 token activations; every output is retained. The displayed data contain 18,684 rows in total.

## Independent checks

- Saved token strings reconstruct every rendered prompt exactly, and token IDs agree with the forwarded records. I independently verified every content assignment against text character spans.
- Passage identity uses the message index 1–6, preserving both repetitions of each role. The display uses the first 120 tokens per passage, then retains only positions with the same text and token ID across all three formats. I reconstructed that rule from the full table and obtained precisely the saved display indices.
- Each candidate has the same displayed positions in all three panels and all six passages in order. No score threshold, interpolation, smoothing, or score-dependent token removal enters this display selection.
- All saved activations have 2,880 finite coordinates. Independent float64 softmax projection using the saved H100 SUCA layer-12 coefficients has maximum error 5.55e-16 against the probability arrays and 5.55e-16 against CSV probabilities. The largest simplex row-sum error is 4.44e-16.
- All 216 passage means independently recompute from the display rows. Every displayed probability matches the full probability table at its saved sample index.
- The installed MLX forward places the zero-based layer-12 post_attention_layernorm output directly before the MLP. The projection uses the saved four-class order System, User, CoT, Assistant, consistent with the native SUCA probe metadata and prior local audit.

## Interpretation and visual selection

The user requested multiple examples and selection of the most attractive graph. A featured example chosen after viewing these outcomes must be described as selected for illustration, and the complete twelve-example set should remain accessible. The selection cannot support a claim about typical MATS conversations, independent replication frequency, model behavior, or semantic authority. Repeated tokens and formatting conditions are not independent conversation samples.

The earlier matched gardening control uses this same local model snapshot and probe configuration. Its mean absolute CoT probability difference from H100 was 1.146759 percentage points, while the largest token/class difference was 86.166465 percentage points. It establishes a recorded comparison, not pointwise runtime equivalence or probe calibration. The new graphs remain local MLX measurements using transferred H100-trained probes.

This audit verifies the saved data and plotting input. Final image layout and any featured-example selection are separate review steps.

| Candidate | Displayed tokens per condition |
|---|---:|
| mats-01 | 495 |
| mats-02 | 513 |
| mats-03 | 516 |
| mats-04 | 520 |
| mats-05 | 517 |
| mats-06 | 517 |
| mats-07 | 527 |
| mats-08 | 525 |
| mats-09 | 519 |
| mats-10 | 524 |
| mats-11 | 529 |
| mats-12 | 526 |

Audit completed 2026-09-12T01:42:29.311353+00:00. Full artifact hashes and numerical audit metrics are in [audit-results.json](audit-results.json).

## Featured candidate check: mats-03

The featured candidate is **mats-03, Designing a readable figure**, selected after measurement and visual inspection. All twelve candidates remain available. Its display contains **516 matched tokens per condition**, in the order User 1 (44), CoT 1 (97), Assistant 1 (118), User 2 (41), CoT 2 (100), Assistant 2 (116). These counts and the means below were independently recomputed from its displayed rows.

Mean CoT probe probability, as a percentage:

| Condition | User 1 | CoT 1 | Assistant 1 | User 2 | CoT 2 | Assistant 2 |
|---|---:|---:|---:|---:|---:|---:|
| Correct tags | 0.0928% | 34.5673% | 0.0290% | 7.2847% | 4.5680% | 0.0392% |
| No tags | 2.5132% | 70.8564% | 47.8859% | 71.1500% | 78.8840% | 77.9718% |
| All text in one User message | 1.3677% | 74.3057% | 50.5629% | 69.5830% | 79.2837% | 77.7217% |

The proposed caption numbers are correct: CoT 2 averages 4.568% with correct tags, 78.884% without tags, and 79.284% with all text inside one User message. Assistant 2 averages 0.0392%, 77.972%, and 77.722%, respectively (rounded to the stated precision).

The supported reading is specific to this selected example: removing the original tags or placing the transcript inside a single User message raises mean CoT scores in both authored reasoning and answer passages. The second User passage also rises, from 7.2847% to 71.1500% / 69.5830%. The untagged signal therefore spreads across different original roles; it does not exclusively identify authored reasoning. This is a description of the displayed probabilities, not evidence of a general phenomenon, generated reasoning, or a behavioral change.
