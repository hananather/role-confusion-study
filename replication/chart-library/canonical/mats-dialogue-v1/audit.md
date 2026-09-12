# Independent audit: generated Hanan MATS dialogue, revised prompt

Status: numerical and transcript integrity verified. Content qualification is required wherever the complete model advice is presented.

## Verified measurement

The local model generated two complete responses, each with an analysis channel followed by a final channel. Turn 1 produced 406 tokens; turn 2 produced 399. Both ended with a normal stop reason and an explicit return token; neither was truncated by length or timeout. The two analysis passages contain 76 and 67 words. The final answers contain 186 and 221 words, so the model did not follow the requested one-paragraph, 100-word limit.

I independently decoded both saved response-token sequences. Each exactly reproduces its saved raw response; extraction exactly reproduces the analysis and final strings in the original conversation. Both User prompts match the pre-generation freeze. Turn 2 includes the complete, unaltered first response as history. All seven source messages (System plus User 1 / CoT 1 / Assistant 1 / User 2 / CoT 2 / Assistant 2) occur exactly once in each measurement format.

The correct-tag prompt contains two User messages, two Assistant analysis messages and two Assistant final messages. The authors’ custom template renders the first final answer with an end token when replayed as history, and the last with a return token. The two other prompts exactly match joined text without tags and the same joined text inside one User message. Every saved prompt token ID re-encodes exactly.

Independent character-overlap labeling agrees for all 3,166 forwarded tokens. The forward lengths are 1,078 / 1,042 / 1,046 for correct tags / no tags / all in User. Independent selection reproduces 1,734 displayed rows: 578 matched tokens per format, with passage counts 116 / 102 / 120 / 40 / 80 / 120. Matching uses the first 120 content-token positions in each passage and requires identical text and token ID across all three formats. No probability-dependent token selection occurs.

Saved activations have shape 3,166 × 2,880 and are finite. Recomputing four-class softmax from these activations and the saved SUCA layer-12 coefficient rows matches the binary probability file exactly. CSV round-trip error is at most 1.12 × 10⁻¹⁶. Every displayed probability, saved passage mean and passage size is reproduced independently. The script, probe, original-message and freeze hashes agree with provenance.

The displayed CoT passages have mean CoT scores of 43.38%, 48.07% and 48.42% under correct tags, no tags and all text in User, respectively. These are 182 matched CoT tokens within one selected illustrative conversation, not 182 independent examples. The user passages also receive nonzero CoT scores; the figure should show their actual points without claiming perfect semantic separation.

## Content and selection qualification

The exchange is clearly authored for this project: Hanan asks about applying to Neel Nanda’s MATS stream, names the paper and GPT-OSS-20B, describes a Tool-minus-User activation intervention, and follows with the “expensive mood ring” question. The local model generated the analysis passages and answers once for this revised prompt. It is not a historical Hanan conversation, a Neel quotation, or a claim about admission outcomes.

The generated research advice is not sound enough to present as our protocol or findings. In particular, Assistant 2 infers that the probe is “not causally linked” to behavior from a hypothetical dissociation; that conclusion is too strong. It also suggests a 100-dimensional random vector although the measured hidden width is 2,880. Assistant 1 gives a 10-prompt suggestion and illustrative threshold/p-value without a power or precision justification. These statements must remain identifiable as unedited model output rather than endorsed methods or observed results. A short transcript note such as “Unedited model output for illustration; the suggested research advice is not our experimental protocol” addresses that distinction.

This is the second generated dialogue attempt. The earlier low-reasoning attempt remains in the sibling `mats-hanan-dialogue` directory and was rejected because it confused chat roles with personas and produced very short analysis passages. The current run’s freeze records that history. The display can be selected for relevance and legibility, but should not be described as a prespecified representative sample or as the only generated candidate. The current run itself retains all outputs and scores without further sampling or token selection based on results.

## Scope

This audit used only saved artifacts and the local tokenizer. I did not load or run the language model. It verifies transcript integrity, prompt construction, labels, score projection and reported summaries; it does not establish cross-runtime probe calibration, generalization to other conversations, behavioral steering or mechanism. The analysis channel is generated model text, not a verified causal explanation.

## Audited artifact identities

- `run_dialogue.py`: `12c66f46a9a3477bb7c105bff6622421928abfba1388f118064afe9a39ccd9ac`
- `prompt-freeze.json`: `4cd43fe8c9ed9300abe080dd51961ccdc95c877782b94af5c8d0a1f167c2c1f3`
- `measurement-freeze.json`: `39351af5bb050775fd3a8da364c0cb793d8da810587996da1bbd10713ba13122`
- `original-messages.json`: `8578b612f9b54da6361b9aa0c549bc911bdb307d0c44fc62621bd5f4ceaed0f4`
- `generated-conversation.json`: `c7529a1d539ccdb6757bc1e3b7e123d8e04cb032dfc54592f345780dee6e1703`
- `layer12-float16.npy`: `15212c2b3d0c0e00f4f500b07f2478bdf17d2c830510ca93d94767440d7e8c86`
- `probabilities.npy`: `37ab33dc88ee426d0f9f960391357d3153c851c6d1f2dd24f7b83c079d8b5952`
- `displayed-rows.csv`: `f8a449751eaf6959126eab8d741b3718cfb355dc092727b34fc715ec34e57f4f`
- `provenance.json`: `f1641cef259d741f66ea57715b9977ef312d1aca503747c946563f61a665b250`
