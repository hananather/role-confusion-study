I have prepared an exploratory batch using all eight question–policy demonstrations already released in the authors’ generic forgery YAML, in source order. Three questions exactly match StrongREJECT IDs 95, 300, and 307. I preserve the released wording and use these examples to collect matched raw trajectories for later review.

Execution remains disabled. Both configurations set `execution_approved`, `examples_execution_aligned`, and `protocol_decisions_complete` to `false`. This prepared option awaits my scope decision; it does not replace the original 313-prompt experiment.

| Stage | Harmful-prompt rows | Harmless rows | New-token cap |
| --- | ---: | ---: | ---: |
| Unsteered base and forgery | 16 | 0 | 5,000 |
| Matched unsteered comparators | 8 | 20 | 2,048 |
| Refusal direction, three strengths | 24 | 60 | 2,048 |
| Twelve role cells at block 12 | 96 | 240 | 2,048 |
| Three matched random directions and reversal | 32 | 80 | 2,048 |
| Default maximum after all gates pass | 176 | 400 | By stage |

The default plan has at most 576 experimental generations. A preceding five-prompt runtime gate makes 15 additional diagnostic generations: unhooked, zero-dose, and nonzero-dose versions, capped at 64 new tokens. Optional block-8 and block-16 checks would add 16 harmful-prompt and 40 harmless rows; they are disabled in this configuration. There is no holdout confirmation for these eight source examples.

I retain the original stopping gates. At least two of eight forged examples must produce finished non-canned finals before steering continues. The refusal control must produce a conservative increase of at least 40 percentage points, requiring at least four net flips among eight prompts, while the harmless compliance proxy changes by less than 10 points among 20 prompts. Censored outcomes remain unknown. The 90-minute rule remains active. The flags that would turn these gates into advisory checks are `false`.

These gate labels remain provisional. I preserve the authors’ exact curly-apostrophe skip flag separately and normalize the two exact canned-refusal spellings for the provisional gate. Harmfulness judging and review of whether emitted reasoning recognizes an injection remain unfinished.

I keep the requested steering and probe sites. The layer-12 probe observes policy tokens before a block-12 or block-16 output edit; I label those readings as upstream. A policy mask on a plain harmless prompt edits zero tokens, which I label as a no-op check.

The local checks passed on five real released policies: prompt lengths 231, 263, 232, 233, and 256 tokens, with identical policy-token selections before and after left padding. I used no placeholders, imported no PyTorch, and loaded no model. I also verified the saved 1,245-prompt corpus and the five-role layer-12 probe shapes. These checks establish local input and tokenizer readiness; the CUDA runtime gate remains pending.

The directory contains the unchanged source examples, unchanged source YAML, 80 frozen Alpaca instructions, disabled local and remote configurations, the tokenizer receipt, and a hashed input staging map. Corpus and probe files are referenced at their existing local paths. The remote paths describe where those same files would be staged; their presence on a future pod has not been checked.

The draft remote configuration uses a two-hour planning ceiling. It is an unapproved draft: a launch still needs a reviewed scope, frozen runtime, live cost and balance check, and an aggregate budget calculation that includes the current baseline run and earlier attempts.

At the final September 12 console quote, one H100 SXM for two hours plus a twenty-minute shutdown allowance is estimated to cost at most $9.05 for GPU, container disk, and a $0.90 retained-storage reserve. Adding the observed $1.63 account decrease from preparation and baseline collection gives about $10.68 across both parts, within the original $20 ceiling. These are estimates, not an invoice. I must refresh balance, rate, storage quote, and prior spend before any future launch. The proposal is in `proposed-cost.json`; execution remains disabled.
