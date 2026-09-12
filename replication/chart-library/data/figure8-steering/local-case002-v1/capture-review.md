# Static capture and fixture review

I inspected `run.py`, the frozen `run_steer.py` / harness, and the installed MLX-LM `generate_step` and `stream_generate` implementations before the model run. I did not load a model, launch a sandbox or change the script.

## Alignment finding

The installed MLX generator calls `_step(y)` before yielding token `y`. Therefore the captured state sequence normally contains the prompt plus every returned generated token, including a returned stop token. The script's normal-stop assertion is compatible with this implementation; it does not incorrectly assume that the newly sampled next token's state belongs to the current returned token.

I found a censoring bug in the original assertion: a timeout during prefill can leave a partial prompt prefix, while a timeout in the completed-prefill callback can leave a processed sampled token whose ID was never returned. Treating either as an ordinary complete generation would abort the run as an infrastructure error. The root agent patched this before execution. I verified that the patch keeps only the known input-token prefix, trims probability/state arrays to those positions, records omitted/unprocessed positions, and retains strict alignment checks for non-timeout generations. The corrected script hash matched its amended input freeze when reviewed.

One additional edge case was reported: if the initial progress callback times out before any forward pass, the list of captured probability chunks is empty. Concatenation must return an empty `(0, number_of_classes)` array in that case rather than raise. This is a capture-completeness correction, not a change to the experimental inputs or sampling.

The layer-12 recorder wraps the existing downstream recorder without replacing its `self.rec` reference; both receive the same normalized states. The layer-16 recorder projects those states through both four-role and five-role saved probes. Both outer position counters reset for each generation. The activation edit remains in the existing block-11 wrapper and uses the existing page mask. I found no additional execution blocker in those interactions.

## Independently verified inputs

- All four fixture hashes and payload hashes match the input freeze.
- Every fixture contains its exact payload once.
- All arms retain seed 1239, the same legitimate user prompt, and the same dummy-only sandbox content.
- The ordinary command paragraph is byte-identical across all four arms.
- The styled-forgery fixture matches the historical case-002 fixture hash exactly.
- Styled and steered case records differ only in their case identifier, variant label and fixture-path spelling; their model-facing fixture bytes match.
- The raw source HTML hash, source-code hashes, probe archive hash and direction-file hash match the freeze.
- The de-styled reasoning retains the fabricated policy quote and its debugging exception, and retains the claimed need to locate and upload the dummy file. Its phrasing and length differ, as disclosed.

This is static verification of the prepared inputs and capture logic. Actual prompt equality, masks, token/probability alignment, censoring and observed behavior still require inspection of the produced records.
