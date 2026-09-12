# My Appendix K neutral-control audit

Captured 2026-09-11 at 04:48:57 UTC. **I found no data-integrity or summary-calculation defect.** The extraction and analysis completed at 04:44:39 UTC in 186.72 seconds. I am in **Understand**: my north star is to distinguish the probe's response to position, text, and tags without claiming more than the saved control establishes.

## What I verified

I found all **1,330 prepared inputs**, covering 200 neutral base texts and seven conditions. The counts are 200 no-insert, 200 system-at-1, 199 system-at-25, 194 system-at-50, 184 system-at-100, 169 system-at-150, and 184 user-at-100. Every omission follows the frozen rule that the base text must extend beyond the insertion index.

I checked all **947,919 token/layer rows**: 315,973 input tokens projected at layers 8, 12, and 16 with the prompt-split `suca` probe. Token IDs, positions, BOS masks, tag masks, and inclusive block boundaries exactly match the saved inputs. The four role columns are system, user, CoT, and assistant; all probabilities are finite and between zero and one. The largest sum-to-one error is **2.38 × 10⁻⁷**.

The source input, runner, probe, and original analysis hashes match the frozen contract. The output prompt parquet has a different byte hash because the runner reserializes its records. I verified exact equality of all 1,330 rows, 22 columns, values, order, and dtypes; I record both hashes in the audit.

I recomputed all **21,000 curve rows and 36 block-table rows** with the unchanged settings: seed 123, 200 prompt bootstraps, 250-token window, and per-prompt EWMA alpha 0.25 with `adjust=True` and `ignore_na=True`. The largest numerical discrepancy is **1.41 × 10⁻⁷**; every per-index sample count matches exactly. All 26 SVG files parse as XML. I did not rerun model inference or modify the experiment outputs.

## What I observe

These are mean Systemness scores on the **57 content tokens inside the inserted canonical block**, excluding its four tag tokens:

| Probe layer | System block at 1, n=200 | System block at 100, n=184 | Same text in user tags at 100, n=184 |
| --- | ---: | ---: | ---: |
| 8 | 22.23% | 23.44% | 11.70% |
| 12 | 12.70% | 13.52% | 8.58% |
| 16 | 20.50% | 10.51% | 2.55% |

At layer 12, I do not observe a large early-versus-late contrast in the canonical block's mean score on this neutral input set. The saved all-token smoothed curves have mean scores across the block indices of 18.73% at position 1 and 20.44% at position 100. The corresponding content scores differ more at layer 16. I treat these as descriptive comparisons because eligibility changes across positions.

The no-insert neutral text itself has high layer-12 scores: the raw curve is **94.72% at index 1, 71.42% at index 100, and 66.74% at index 249**. Its sample count falls from 200 to 184 to 147 at those indices. The canonical block scores below the different neutral text occupying the same indices in the baseline; that comparison changes text and tags together. It does not isolate a position effect. At index 100, the same canonical content scores higher with system tags than with user tags in all three saved layers.

## Claim boundary

I verified that **all 200 neutral texts exactly match texts in the current probe corpus**. The prepared inputs explicitly document this overlap. They do not meet the broader spec's held-out-neutral ideal, and I have not evaluated the original shuffled conversation rows used for Figure 32. I therefore record a completed neutral-text control with a layer-dependent descriptive result, not a successful replication of Figure 32 or evidence that position alone explains role assignment.

Systemness is a probe score, not a demonstrated instruction-following priority or a calibrated probability of a true role. I ran no behavioral generations or instruction-hierarchy regression here. I have no reference probabilities for these exact neutral inputs.

For token-exclusion claims, I use the raw content curves and content block table. The existing content EWMA carries the previous content estimate across masked tag positions; its per-index `n` counts those carried estimates. This behavior matches the saved analysis and is recorded rather than silently changed.

- [Checks, settings, numerical errors, provenance hashes, and limits](audit.json)
- [All content-block summaries](content-block-summary.csv)
- [Layer-12 no-insert curve checkpoints](layer12-neutral-curve-checkpoints.csv)
