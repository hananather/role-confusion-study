# Original MATS illustration — prepared, measurement not started

I froze one authored MATS Q&A before measuring any activation or probe probability. This is a new illustrative input, not a real conversation with Neel, a quotation, or model-generated reasoning. The question is: “For a MATS project with Neel, how can I turn a broad idea into one clear research question?”

**Current status: blocked by another local model run.** On 12 September 2026 UTC, `MODEL-IN-USE.md` and the live steering-chain process showed that Claude's existing local experiment held the model. I did not start inference, load a second model, interrupt the existing experiment, or schedule a background watcher. No measured figure or results exist for this input yet. Several queued arms have 2,700-second budgets; availability could be hours away and has no verified finish time.

The frozen inputs have 197, 201, and 217 total tokens under no tags, all text in User tags, and proper tags. Each contains 139 non-System content tokens. Matching the exact token string at each message-relative position across all conditions retains 137 per condition; two boundary tokens per condition are omitted from the display. The full table preserves them. The input hashes are in `input-freeze.json`.

Once the current owner has finished, verify no model process remains and resolve its lock before running this one foreground command from `replication/`:

```sh
.venv/bin/python chart-library/data/mats-example/run_local.py
```

The script deliberately refuses to run while `MODEL-IN-USE.md` exists. It uses only the cached MLX model and tokenizer, offline, with batch size one and at most 2,048 tokens. It computes six forwards: three MATS conditions and the three frozen gardening controls. No generation, probe fitting, downloads, cloud calls, or model weight edits occur. It records the observed activation dtype, saves layer-12 pre-MLP states, and applies the saved H100 prompt-split four-role probe. The gardening comparison quantifies probability differences under the two runtimes; it does not establish calibration or interchangeability.

Expected outputs include `token-probabilities.csv`, `displayed-rows.csv`, `display-selection.json`, the MATS and gardening `.npy` states/probabilities, `gardening-runtime-comparison.json`, and `provenance.json`. The displayed table exposes `prompt_key`, `display_token_ix`, `base_message_type`, `token`, `prob` (CoT probability), and `seg_ix` for the renderer. Source inputs are frozen and result files must not be overwritten.

Checks completed: script syntax; input hashes; offline tokenizer construction; equal content counts; exact common-position selection; saved probe dimensions `(4, 2880)`; cached model and tokenizer snapshots. Model execution and numerical outputs remain untested.
