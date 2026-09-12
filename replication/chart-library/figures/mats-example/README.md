# Original MATS Q&A: measured role-probe readout

[PNG](mats-role-readout.png) · [PDF](mats-role-readout.pdf) · [SVG](mats-role-readout.svg)

**Role-probe readout for an original MATS Q&A.** Each point is the four-role probe’s CoT probability at layer 12 for one of 137 matched content tokens. Panels change message formatting while preserving one authored conversation. Blue, amber and green identify original User, CoT and Assistant passages. Horizontal position follows retained token order. All passages were supplied to the model; no reasoning or answers were generated in this run.

The authored reasoning passage retains a higher average CoT score than the question and answer in all three conditions. Without correct tags, some answer tokens also receive high CoT scores: answer means are 0.03%, 27.92% and 27.42% in panel order. This is one illustrative example, not an aggregate or behavioral effect.

The local MLX run uses saved H100 probe coefficients. A separate gardening control measured a 1.15-point mean absolute CoT difference and an 86.17-point maximum individual class-probability difference from H100; the two runtimes are not pointwise equivalent. See [measurement record](../../data/mats-example/READY.md) and [provenance](../../data/mats-example/provenance.json).

The role colors and TeX Gyre Termes font match the paper’s plotting specification. PDF and SVG are vector exports; PNG is 600 dpi at 6.75 × 3.85 inches. This is a locally measured extension of the paper’s Figure 7 construction, not the authors’ example.
