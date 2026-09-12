# Original MATS illustration — measured locally

**Completed:** six local model forwards, three MATS conditions and three gardening controls. The model was released after measurement; no cloud compute or generation was used.

The original question is: “For a MATS project with Neel, how can I turn a broad idea into one clear research question?” All passages are authored illustration text, not a real conversation with Neel or model-generated reasoning. The input was frozen before inference; all outcomes are retained.

The figure shows 137 matched content tokens per condition (21 User, 61 CoT, 55 Assistant). Full data retain 139 content tokens per condition; two format-dependent boundary tokens are omitted from the matched display. Mean CoT probabilities on displayed reasoning text are 55.10% with correct tags, 72.47% with no tags, and 72.31% with all text in one User message. The corresponding Assistant-answer means are 0.03%, 27.92%, and 27.42%.

We used the cached MLX GPT-OSS-20B model, layer-12 pre-MLP activations and the saved H100 four-role probe. The gardening control gives a mean absolute CoT-probability difference of 1.15 percentage points across 2,745 content-token records; the largest individual role-probability difference is 86.17 points. This is a separate local measurement, not pointwise runtime equivalence or calibration validation.

Outputs include the frozen inputs and hashes, all token labels, saved activation/probability arrays, token-probabilities.csv, displayed-rows.csv, display-selection.json, gardening-runtime-comparison.json, provenance.json and the independent audit. Source results must not be overwritten.

[Figure PNG](../../figures/mats-example/mats-role-readout.png) · [Vector PDF](../../figures/mats-example/mats-role-readout.pdf) · [SVG](../../figures/mats-example/mats-role-readout.svg)

The prepared-state record is preserved in pre-run-status.md. The run log and provenance describe execution; the prior unrelated model-status note was archived before this run after checking the model was free.
