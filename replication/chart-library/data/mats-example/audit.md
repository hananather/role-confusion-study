# Independent measurement audit: authored MATS example

**Verdict: the saved measurements and matched-token display pass the numerical and method checks below.** This is one prespecified illustrative conversation. The checks establish the recorded local computation, not generalization or a validated measure of semantic authority.

I did not load or run a model for this audit. I recomputed the projections from the saved activation arrays and inspected the measurement code, native H100 probe metadata, and MLX forward implementation.

## Inputs and role mapping

- The frozen inputs predate the run. Their three recorded SHA-256 hashes remain unchanged. The executing harness hash and source probe hash match the run provenance.
- The authored conversation comprises a System message, one User question about a MATS project with Neel, an authored analysis passage, and an authored final answer. It is neither a real conversation with Neel nor model-generated reasoning.
- All three rendered prompts reconstruct exactly from saved token text. Saved token IDs agree with the forwarded records, with lengths 197, 201, and 217. The full inputs include the System message under each formatting condition.
- Native H100 metadata records the SUCA order `System, User, CoT, Assistant`, classifier classes `[0,1,2,3]`, and coefficients of shape `(4,2880)`. The local projection uses precisely this order.
- The recorded hook is zero-based layer 12 `post_attention_layernorm` output, immediately before the MLP in the inspected MLX forward implementation. No autoregressive generation or behavioral test was performed.

## Independent recomputation

- All 615 saved states have 2,880 finite coordinates. Recomputing softmax of `h @ W.T + b` in float64 reproduces the saved local probability array exactly; the CSV differs by at most `1.11e-16` probability from that array.
- Probabilities lie on the four-class simplex; the largest row-sum error is `3.33e-16`.
- Applying those same NPZ coefficients to the saved H100 gardening states reproduces the native cuML predictions to within `1.87e-6` probability. This independently supports coefficient identity, ordering, and scoring convention.
- Every displayed probability equals the corresponding full-table probability. The figure's display selection is based only on input token matching and retains 137 positions per condition: 21 User, 61 CoT, and 55 Assistant tokens. Each position has the same text **and token ID** at the same message-relative position in all three conditions.
- Each full condition contains 139 non-System content tokens. Only the question's final punctuation and the analysis passage's final punctuation are omitted from the figure, because their tokens include a newline in the untagged/all-User conditions. All six omitted token records and their probabilities remain available in the full CSV. No truncation or score-based selection was applied.

## Verified displayed means

Mean CoT probability, expressed as a percentage. Each row is one formatting condition on the same authored text; tokens are repeated measurements within one conversation.

| Condition | Question (21 tokens) | Authored analysis (61 tokens) | Final answer (55 tokens) |
|---|---:|---:|---:|
| Correct tags | 0.012559% | 55.095864% | 0.027228% |
| No tags | 4.896228% | 72.470553% | 27.920475% |
| All text in one User message | 4.645378% | 72.309201% | 27.417227% |

The analysis passage has a higher average CoT score than the question and answer in all three conditions. Removing tags or placing the text inside one User message raises both the analysis passage's and final answer's CoT score relative to correct tags. The pattern is illustrative; these averages are not independent-sample effect estimates.

## Runtime boundary

The local run used cached `mlx-community/gpt-oss-20b-MXFP4-Q8`, with MXFP4 expert weights, 8-bit non-expert weights, MLX kernels, and an unpadded batch of one. The H100 probe was fitted using the native runtime and different non-expert precision. Source and target have the same recorded activation width and measurement site; this does not establish equivalent activations or probe calibration.

The matched gardening control forwards the same 2,791 token IDs and scores 2,745 labeled tokens, including System tokens. Independent recomputation reproduces its saved local probabilities exactly. Across those labeled tokens, mean absolute probability differences from H100 are 0.217449 percentage points for System, 1.046686 for User, 1.146759 for CoT, and 1.103074 for Assistant. The **largest individual token/class difference is 86.166465 percentage points**. Small averages therefore do not establish pointwise runtime equivalence. The figure should remain identified as a new local MLX result, with this runtime distinction explained in its methods or caption context.

## Audit artifacts

- `input-freeze.json`: `1fb0d3da833b9080f55e9dd881a8771e00139cb9f3987a62e18ad695ebfb976c`
- `run_local.py`: `902803388367ea881c9fdeaac7d1cb3090bae9d4608d8187ddd9247d01003b8d`
- `provenance.json`: `a578bdb43ea393e5522468b373aa04c6a1015be592b98ac1af483d23ecabccc7`
- `mats-layer12-float16.npy`: `d202477d7f77977ca3d2cef8d968959ea0b3722efd1319c4b253fa99d3c3ba23`
- `mats-probabilities.npy`: `9d575f839e6a74e7bbc53e50a57a9e788006f7da5db6067df5c219e6745be9aa`
- `token-probabilities.csv`: `c5279678009d882c7a365f4db36117cdc2b1f753090f6ab885352f50095dfb82`
- `displayed-rows.csv`: `21743b0ae5692a996c2ba9e81a3312d471329a2c773d36477f76ca61f4ba5e0e`
- `gardening-runtime-comparison.json`: `0f84304aa56f454d76d8df239db5d2960f38e614bc2bf8871bb81489bdc780b7`

Audit completed 2026-09-12T01:21:52.242527+00:00.
