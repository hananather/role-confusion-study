# Independent audit: first generated Hanan MATS dialogue

Status: measurements verified; example retired for content and display suitability. The original files remain unchanged.

## Verified

- Both generations ended with the normal stop reason and an explicit final-channel return token. Turn 1 produced 449 tokens; turn 2 produced 265. Neither was length- or timeout-truncated.
- Raw saved text exactly equals decoded response token IDs, with the deliberately supplied Assistant analysis header. Extracted analysis and final passages exactly match the originals preserved for measurement.
- Both user prompts match the pre-generation freeze. All three measurement prompts preserve the seven complete messages (System plus six displayed passage types).
- Correct-tag prompts contain two User, two Assistant analysis, and two Assistant final messages. The earlier Assistant final uses the authors’ custom template’s end token when replayed as history; the final turn uses its return token. No-tags and all-in-User constructions agree exactly with the frozen template and joined text.
- Independent character-overlap labels agree for all 2,713 forwarded tokens. All token IDs re-encode exactly. Forward lengths are 927, 891 and 895.
- Independent matching reproduces all 1,095 displayed rows: 365 tokens per condition, with passage sizes 74 / 4 / 120 / 40 / 7 / 120. The first-120-per-passage rule is applied before matching; text and token ID must agree in all formats.
- Saved activation shape is 2,713 × 2,880; all entries are finite. Recomputed four-class softmax from the saved activations and SUCA layer-12 weights matches the binary probability file exactly. Maximum CSV round-trip error is 1.12 × 10⁻¹⁶. Script, prompt-freeze, measurement-freeze and probe hashes match provenance.
- No model was loaded or run by this auditor; all checks used the saved data and tokenizer.

## Why this example is unsuitable for the feature figure

The generated analysis passages contain only four and seven words. They satisfy the six-passage structure but make the two CoT regions very narrow. The generated final answers also confuse a surface persona change with internal role steering and infer awareness or an overridden representation from high Toolness. Those claims are unsupported. The first answer proposes an imagined result with X% and p < 0.05; that is a model-generated suggested write-up, not an observed result. Neither answer obeys the requested 100-word maximum (257 and 171 words).

These shortcomings do not invalidate the recorded activations or probabilities. They make the exchange a poor original MATS illustration. Retiring it on content and layout grounds should be disclosed if a revised exchange is featured; it must not be presented as the only generated candidate. A displayed transcript would need to identify the advice as unendorsed model output.

## Evidence boundary

One authored pair of User prompts generated one pair of complete model responses. Scoring the frozen text uses MLX activations and separately fitted CUDA-era probes; this audit verifies the projection and labeling, not cross-runtime probe calibration or semantic authority. The analysis channel is model-generated text, not a verified mechanism explanation.
