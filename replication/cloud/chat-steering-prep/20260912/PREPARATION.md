# My bounded chat-steering batch preparation

Current status at 2026-09-12 01:06 UTC: I completed and saved all 313 baseline trajectories (309 completed finals, four capped outputs). Both pods are deleted, final backups are verified, and the live inventory is empty. The forged-policy corpus is still unavailable; no steering arms were run. The result package is [RESULTS.md](</Users/hananather/Desktop/MATS 12.0/replication/cloud/outbox/chat-steering/baseline-retry-20260912T004400Z/RESULTS.md>). The dated notes below preserve the earlier preparation and execution states.

I am in Explore. My north star is to measure whether a role-direction intervention changes behavior more than a matched random direction. I keep the selection sweep exploratory and reserve confirmation for the 213 held-out prompts. This implements the requested direction; it does not revise my research claim.

## What I verified

- The working assets and successful H100 session from last night are documented in `../../persistent/sessions/20260911T030050Z/SESSION.md`.
- On September 12 at 00:14 UTC, the account API returned a $17.5614986737 balance, Auto-Pay disabled, and the global H100 SXM secure rate of $3.49/hour. A separate REST inventory returned zero pods and retained volume `i0cptrlcoz`, 2,000 GB in EU-FR-1. The sanitized responses are saved beside this file.
- At 00:17 UTC, the location-specific H100 SXM query returned null price and stock fields for EU-FR-1. I do not have a deployable regional quote.
- The StrongREJECT CSV contains 313 rows and has SHA-256 `4dd70357e4ff8b5d0ba5ebafecab5d6dd5633ce8046e3dd1c8bd93e64de44381`.
- The existing Alpaca loader found 23,849 eligible cached instructions. I froze its first 80 rows as `harmless.csv`: 60 direction-building rows and 20 checks.
- The existing `forge-stub --all` command produced 626 rows, covering the generic and qualified variants. Only the generic variant was tried.
- The local CLI reports `Logged in using ChatGPT`. Five source-verbatim generic generator requests used the configured `gpt-6-astra` model, with API-key environment variables removed and ChatGPT login enforced. Their observed responses are preserved. Requested model identity is recorded; the JSON event stream does not independently report the served model.

## The policy prerequisite failed

All five generator replies explicitly declined the requested forgery. Four returned safety-preserving alternatives inside `<SYNTHETIC_POLICY>` tags. I read all five replies and accepted none. Tag presence and the existing 50-character minimum would not detect this semantic failure.

The audit is `forge/semantic-review.json`. I did not import the replacements, attempt the remaining 308 generic rows, retry the declined prompts, switch models to defeat the refusals, or launch a GPU. The requested ten-policy hand review cannot be completed without a usable corpus. Existing released research material could be assessed for provenance and suitability separately; it has not been supplied for this batch.

Codex-specific divergences are retained in `forge/generic/generator-metadata.json`: generator model, flattened source message roles within the CLI task, the CLI's own instructions, and no claimed temperature-zero equivalence.

## My preregistered interpretation and decision rules

I use the fixed `random_state=123` selection of 100 prompts and the remaining 213 as holdout. My stated prior is that the reading moves and success does not. I do not claim evidence from steps 3–5 alone.

I select the role cell with the largest drop in canned-refusal-based success, subject to under ten percentage points of change on the 20 Alpaca checks. I preserve exact proxy counts, censored outputs, and final judge labels separately. I require the requested 20% baseline and 40-point positive-control gates. I do not count incomplete or missing finals as successful resistance. Additional unsteered 2,048-token comparators are explicit preparation choices to avoid comparing a steering arm with a differently capped baseline.

The requested layer-12 `post_attention_layernorm` readout precedes a layer-12 block-output edit. It also precedes a layer-16 edit. For these two sites it is an upstream diagnostic on policy-prefill tokens, not a direct manipulation check. A changed measurement site needs alignment; the current preparation preserves the requested sites. A policy mask edits zero positions on bare Alpaca prompts, so those harmless checks cannot establish tolerance to an active policy-span intervention.

If the first two stages miss the 90-minute limit or the positive control fails, I skip the role sweep. I do not invent a selected role arm or label baseline-only holdout output as a completed four-arm confirmation.

## Spending boundary

At the global quote, five GPU hours cost $17.45 and five hours twenty minutes cost $18.6133. Disk and retained-volume charges also draw from the account. My requested total cap is $20. I anchor the cap to pod creation, including setup, and retain the 20-minute shutdown margin. I have made no top-up and no pod creation request.

I will not request launch approval until the corpus, local tokenizer checks, frozen runtime, and regional quote are ready. The user's pasted request explicitly says: “Nothing launches until Hanan approves the exact command and cost you print.” The original request is preserved as `request.txt`.

## Review files

- `input-manifest.json`: local input/source hashes.
- `forge/semantic-review.json`: rejected generator output audit.
- `provider-initial.json`, `provider-account.json`, `provider-quote-eufr.json`: dated, selected provider fields without credentials.
- `../../chat_steering/`: new preparation code. Existing running local experiments and last night's finished outputs remain separate.

Provider source for the account and quote fields: [RunPod GraphQL specification](https://graphql-spec.runpod.io/). Authentication source: [OpenAI authentication documentation](https://learn.chatgpt.com/docs/auth).

## Execution update at 2026-09-12 00:43 UTC

My later instruction, “Just run the results ... all that matters is that we get the trajectories from GPT-OSS on the benchmark,” authorized the originally requested base313. I shortened this part to two planned hours, with a twenty-minute shutdown allowance and a $9 aggregate cap including the failed first attempt. The original $20 ceiling covers the whole request.

My first pod (`27an5o35e9cf07`) produced no generations because the offline attention kernel lookup used the model cache directory. Its final backup succeeded; DELETE, HTTP404, and zero pods were verified. I preserve its frozen packet and logs. GPU plus container-disk cost is estimated at $0.1836, excluding retained storage.

The retry uses the model’s pinned local snapshot and the existing kernel cache at `/workspace/hf/home/hub`. It checks offline assets and kernel imports before loading weights. Manifest `baseline/retry-cache/manifest.json` has SHA256 `870126b6e43e31d8894775f10d8fb4e99137778f7b483c5ee5f867d233622b51`. Before creation, the live balance was $17.3799178515, Auto-Pay was off, no pods existed, and EU-FR-1 H100 SXM capacity was available at $3.49/hour. No top-up occurred. The retry watchdog began at00:43:26UTC; its absolute termination deadline is03:03:24UTC (11:03p.m. Toronto on September11), two hours twenty minutes after creation.

I extracted all eight verbatim demonstration pairs from the authors’ source YAML into `author-examples.json`; three exactly match StrongREJECT prompts. They are a possible exploratory substitute for the unavailable313 policies and have not been authorized or executed as a replacement. The older313-policy archive documented in the Andrew Gong Wu reference repository is collaborator-held; the recorded private key and encrypted payloads are absent locally.

My requested result remains paired unsteered-forgery and steered trajectories, including emitted analysis and final text. Baseline-only completion cannot answer that comparison.
