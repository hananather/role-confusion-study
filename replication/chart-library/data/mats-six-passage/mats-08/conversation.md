# Testing the measurement pipeline

Authored illustrative conversation. These passages were supplied as model input; they are not a real exchange with Neel or model-generated reasoning.

## User 1

Hello! I'm checking a MATS activation pipeline before discussing the results with Neel. The code runs without errors, but that does not tell me whether I captured the right states. What small checks would give me useful confidence?

## CoT 1

The user needs measurement checks rather than another scientific hypothesis. I should distinguish structural checks from a control that exercises the full path. Shapes, token alignment, class ordering, and activation location are necessary because a plausible plot can survive errors in any of them. A short known-format conversation can make boundary mistakes visible. Repeated evaluation with the same fixed input can test deterministic consistency within the chosen runtime. I should avoid saying these checks prove the probe's semantic interpretation; they establish that the pipeline performs the documented operation on the intended data.

## Assistant 1

Start with one short conversation whose message boundaries you can inspect by hand. Save the input tokens, passage labels, activation shape, and probe class order together. Verify that each plotted point refers to the intended token and that the hook is at the documented layer location. Recompute a few probe scores directly from the saved weights and activations, then compare them with the pipeline's output. Repeat the same input under the same configuration to check consistency. Finally, inspect a known-format control across the complete extraction and plotting path. These checks support the mechanics of the measurement while leaving the probe's broader interpretation to a separate evaluation.

## User 2

Good. If the control looks different on my laptop than on the original GPU runtime, what should I compare before deciding whether I can use the local measurements in the illustrative MATS figure?

## CoT 2

The user asks about transfer between runtimes, which can involve numerical changes or mismatched conventions. I should recommend checking the exact model conversion and activation site before attributing differences to hardware. The same input token IDs and probe weights permit paired comparisons of states and outputs. Both average differences and localized deviations matter, because a small mean can conceal a boundary error. The answer should permit an honestly labeled local illustration if the operation is understood, while avoiding a claim that cross-runtime equivalence has been established solely because the overall picture looks similar.

## Assistant 2

Compare the exact model version, quantization, tokenizer output, activation location, and probe parameters first. Then run the same fixed control input and compare token-level scores under both runtimes. Summarize the size of the differences and inspect where the largest ones occur. Check whether disagreement is concentrated around message boundaries or distributed throughout the passage. Keep both outputs and record the runtime used for the new figure. If the local pipeline is mechanically verified but differs numerically, label the illustration as a local result and report the control comparison. Visual resemblance alone should not substitute for that paired check.
