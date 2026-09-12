# My offline illustration of User and Tool probe offsets

I made these plots to answer how the graph changes when I add or subtract a saved User or Tool probe direction. I use the measured gardening activations from my September 11 H100 session, then edit copies locally and recompute the five-role probe. I do not continue the model, generate a response, or measure behavior.

- [One-percent offset](probe-offsets-1pct.png)
- [Five-percent offset](probe-offsets-5pct.png)
- [Every plotted probability](plotted-probabilities.csv)
- [Mean scores](summary.csv)
- [Source checksums, formula and arithmetic checks](provenance.json)

I keep the 512 Figure 7 display positions from the all-user-tags gardening conversation. My probe is the prompt-split `sucat` probe at layer 12; it includes System, User, CoT, Assistant and Tool. The previous gardening Figure 7 used the four-role `suca` probe, so its probabilities are a separate readout.

For each role I use its raw saved classifier row, normalized to unit length. I add or subtract the same offset magnitude for both roles: 1% or 5% of the median activation norm over all forwarded tokens in this condition. These are illustrative strengths, not validated model intervention doses. I recompute all five logits and their softmax in float64 from the saved float16 activations and float32 weights.

At 1%, mean Userness is 9.9%, 15.5% and 22.5% for subtract-User, unchanged and add-User. Mean Toolness is 1.5%, 4.1% and 10.0% for subtract-Tool, unchanged and add-Tool. Each mean covers the same 512 tokens. Both scores can rise together because other roles can lose probability.

The probe directly scores an offset based on its own weights. Its response is a mathematical consequence of that operation and does not independently validate a role mechanism or behavioral effect. Subtracting a constant vector is also different from removing the activation's projection along that vector.

Individual class rows depend on which equivalent weight representation I choose; adding a common vector to every softmax row leaves the original probabilities unchanged. I preserve the exact saved rows here. The difference `w_user - w_tool` has a cleaner invariant interpretation as the tokenwise relative User/Tool readout direction. It is not the distinct mean Tool-minus-User activation direction used in my September 5 steering experiment.

I checked finite normalized probabilities and the exact affine-logit shift for every direction, sign and strength. I visually inspected both rendered PNGs. The source artifacts are unchanged.
