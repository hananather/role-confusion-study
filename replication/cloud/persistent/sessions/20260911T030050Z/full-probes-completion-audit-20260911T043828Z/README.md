# My completed probe artifact audit

Captured 2026-09-11 at 04:38:28 UTC. I found no failed coverage, shape, finiteness, partition, or saved-metric consistency check. The trainer finished **384/384 probes at 04:31:38 UTC**. I am in **Understand**: my north star is trustworthy probe readouts, with verified numbers and claim strength matched to the saved evidence.

I checked the local files without loading native cuML pickles or running model inference. I preserved the pilot, failed attempt, and earlier fit-quality audits.

## Completion and data

- I found all 24 layers × eight role spaces × two splits, with 192 records per split and 384 coefficient/intercept arrays per split. All **768 arrays are finite float32** and have the expected shapes: one row for binary coefficients, three to five rows for multiclass coefficients, 2,880 features, and matching intercept lengths. No probe has an entirely zero coefficient matrix.
- I verified **249 distinct documents, 1,245 prompts, 713,109 token rows, and 705,390 retained content rows**. Every document has five role prompts; each role contributes exactly **141,078 retained tokens**. The trainer reports 187 Dolma3 and 62 C4 documents; the parquet files establish document identity and role coverage but do not independently label corpus membership.
- I independently checked the saved IDs and token masks for **all 16 partitions**. Each has unique IDs, nonempty training and test sets, no overlap, no omitted groups or rows, all eligible roles in both sets, and exact agreement with the recorded counts. Every full-run split records the unchanged requested fraction 0.1, seed 123, and pilot setting 0.
- For every base split, I found **225 training documents and 24 test documents**. The five-role base split has 129,741 training and 11,337 held-out content tokens per role. The five-role prompt split has 1,121 training prompts and 124 test prompts; its held-out role counts vary, as expected from the existing prompt-level split.
- I reconciled all **384 saved overall accuracies and their per-role accuracies** with the integer confusion counts and saved split denominators, within the JSON's four-decimal precision. The validation CSVs also agree at their two-decimal precision. All saved accuracy and negative-log-likelihood values are finite.

## Comparison with the saved H200 reproduction

I computed accuracy directly as correct confusion counts divided by all confusion counts. I compared **96 prompt-split probes: the 12 even layers × eight role spaces**. Every total and true-role held-out denominator matches the reference.

| Comparison | Observed difference |
| --- | ---: |
| Largest absolute overall accuracy difference | 0.079101 percentage point, layer 0 `uca` |
| Mean absolute overall accuracy difference | 0.015545 percentage point |
| Gardening `suca`, layer 12 | −0.005595 percentage point |
| Largest absolute true-role recall difference | 0.614977 percentage point, user at layer 0 `sucat` |

For the gardening probe, I count **37,708 / 53,615 = 0.703310641**, compared with **37,711 / 53,615 = 0.703366595** in the reference. This exact-count result refines my earlier comparison based on the rounded training log. Its largest per-role recall difference is +0.086296 percentage point for system.

The largest true-role difference is visible despite the close overall accuracies: the layer-0 five-role probe correctly classifies 1,938 user tokens here versus 2,017 in the reference, out of 12,846. I retain this difference rather than summarizing all metrics as identical.

## Evidence boundary

I compare with JamesElmore's independent H200 reproduction, not original paper-trained artifacts. Matching historical token denominators does not prove identical historical split IDs. The odd layers and base split extend beyond this comparison. I have no reference negative-log-likelihood counterpart in these confusion counts.

These checks support proceeding with the completed probe exports. They do not establish optimizer convergence, coefficient equality, calibrated role probabilities, or reproduction of the gardening figures. The recorded solver warnings remain relevant; this local audit does not replace the separate native-estimator check. My full-run recorded forward peak is **47.14 GB** at batch size 16; the final reset counter of 13.83 GB is not the run's peak.

- [All checks, counts, formulas, absolute source paths, and hashes](audit.json)
- [All 96 matched overall accuracies](matched-prompt-accuracies.csv)
- [All 336 matched true-role recalls](matched-role-recalls.csv)
