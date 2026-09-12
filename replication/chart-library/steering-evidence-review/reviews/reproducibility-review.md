# My reproducibility review

**PASS — no unresolved material recommendations.** Independent reviewer 7; saved-artifact audit, 12 September 2026.

My stage is **Understand → Distill**. My north star is whether another reduction of the frozen records produces the same plotted evidence. I apply the Neel lens through source inspection, independent arithmetic and claims bounded by what was measured.

## What I verified

- I reran `prepare_evidence.py` with only its output directory redirected to a temporary directory. All six CSVs and `source-manifest.json` reproduced byte for byte. The pinned input is the corrected `20260912T051758Z/arm-by-page.csv`, SHA-256 `9b6110ce9b781c8dfd13470eb537717c13def35a542183f6eee7ab316d477093`.
- Every one of the 201 source-manifest entries exists and matches its hash. Every derived-file hash in `data/verification.json` matches. I also checked the current renderer hash, all nine PNG/PDF/SVG hashes and all declared per-figure input hashes against `figures/render-provenance.json`.
- The complete table contains 110 assignments: 100 recorded episodes and ten unrun assignments. The recorded episodes split into 80 forgery, 15 benign and five standard-injection episodes. Three records are censored, including two forgery records. The ten unrun assignments split into five new forgery Tool-raising cases and five historical standard-floor cases. The new zero-dose slots are unassigned, rather than unrun.
- The original vector has six uploads across ten pages, compared with seven for no intervention. Its paired counts are two prevented, one introduced and seven unchanged. The displayed random-direction counts, two unresolved random-control outcomes and historical-only Tool-raising denominator agree with the saved rows.
- The probe table has all sixty available forged-payload measurements. The probability-to-percent conversion is correct. The original vector's maximum payload CoT probability is `2.0865409680836265e-18`, safely below the report's 0.01% bound. The minimum actual-episode Tool probability for Tool-raising is `0.9999998807907104`, above the report's 99.9999% bound after conversion to percent.
- I independently checked the 21 permission pairs against the raw compressed export: each baseline/intervention record hash, each first-prefill mean and each authorized/unwanted marker endpoint agrees. These are all 21 recovered Toolward records, spanning fifteen case/permission contexts and eight cases. The markers give unwanted writes 18 → 17 and correct authorized completion 21 → 19.
- The report accurately distinguishes checking saved aggregate readouts against record copies from recomputing probe projections from hidden activation tensors. Its missing-measurement statements agree with the export. This audit used saved records and model-free reduction.

## Recommendations resolved during review

1. **Figure 1 input provenance.** The renderer reads `data/arm-counts.csv` for displayed totals, but its original per-figure manifest omitted that input. The save call now includes it, and I verified the regenerated provenance contains its correct hash.
2. **Random-direction names across documents.** Figure/report Random 2 denotes `random_1_a16` (60.80% mean CoT; two uploads), whereas the original audit's Random 2 denoted `random_2_a16` (46.99%; four observed uploads and two unresolved). The report now supplies the explicit Random 1/2/3 → source 0/1/2 mapping, and the audit names the source identifiers. I verified both corrections.

The source cutoff, exact denominators, unresolved outcomes and saved-activation limitation are represented consistently in the reviewed report. These checks support reproducibility of the displayed saved-record analysis; they do not add behavioral observations or establish a causal mechanism.
