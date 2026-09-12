# Original Hanan–MATS illustration

I authored two questions about applying to Neel Nanda’s MATS stream and asking whether a high Toolness score without changed behavior amounts to an “expensive mood ring.” The cached local GPT-OSS-20B model generated two analysis and final-answer pairs. I then replayed the complete exchange in three formats and applied the existing layer-12 four-role probe.

This is the second generated attempt. The first, preserved in `../mats-hanan-dialogue/`, confused chat roles with personas and produced very short analysis passages. Before this revision I supplied the missing paper context and changed low reasoning to medium. I retained all responses, all scores and the earlier twelve authored examples in `../mats-six-passage/`. The selected exchange is illustrative, with no claim of random or representative sampling.

Read `conversation.md` for all six passages and the model-advice caveat. `prompt-freeze.json` records the inputs before generation; `measurement-freeze.json` fixes token selection before scoring. `turn-*-raw.txt` and `turn-*-generation.json` retain raw generated text and token IDs. `original-messages.json` stores the exact seven text spans, including System. `rendered-prompts-and-token-ids.json` stores the three complete measurement inputs.

`displayed-rows.csv` has the same 578 matched content tokens in each format. The first 120 tokens per passage are eligible; boundary mismatches are excluded before viewing scores. `token-probabilities.csv` includes all 3,166 forwarded positions. `passage-means.csv` summarizes the displayed passages. `layer12-float16.npy` and `probabilities.npy` retain complete saved activations and probabilities. `provenance.json` identifies the model, probe and runtime. `audit.md` records independent checks.

The revised run took 25.31 seconds including model loading, two generations and three measurement forwards. It used cached local weights, with peak MLX memory of 12.99 GB. No cloud instance was started. The model process exited after saving the results.
