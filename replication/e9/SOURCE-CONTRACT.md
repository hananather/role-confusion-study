# Figure 23 source contract

I am in **Understand**: my north star is to reproduce the GPT-OSS-20B layer traces and measure separately how each approved vector family changes them. I keep source observations, local measurements and intervention choices distinct. This follows the research lens of checking the measurement, preserving adverse results and testing the simplest stated comparison.

I freeze the authors' repository at `ec333c40fd43fe991e1ebf66765051b6d7e35784`. My source is Appendix F, Figure 23 on PDF page 25, NB01 `01-get-conversations-data.ipynb`, NB02 `02-train-role-probes.ipynb`, NB03 `03-analyze-probes.ipynb` cell 8, and `r-utils/plots.r`. Cell numbers here are zero-based. Figure 23 contains no activation steering; the two vector-family overlays are my requested extension, defined separately in the execution protocol.

## Candidate inputs, before generation

I read existing local train Arrow files directly. I do not import the E4 generation module: its selection functions call `load_dataset`, and its imports reach model-related modules. I implement only the source selection logic with PyArrow and NumPy.

- **ToxicChat:** `lmsys/toxic-chat`, configuration `toxicchat1123`, train split; shuffle with seed 1234; take the first 100 user inputs whose Python string length is 100–500 characters inclusive. I reproduce `Dataset.shuffle` with `numpy.random.default_rng(1234).permutation`, without creating a dataset cache. Each source conversation contributes one user turn.
- **OASST:** `OpenAssistant/oasst1`, train split; keep English rows with `tree_state == ready_for_export`. I preserve row order, the resulting tree insertion order, the first root and the first child at each branch. I validate every prompter message along the full root-to-leaf path at 100–500 characters before retaining the first two prompter turns. I select the first 100 valid trees without shuffling.
- I concatenate OASST then ToxicChat, assign conversation IDs 0–199, and preserve source conversation/message IDs, raw row positions, text hashes and OASST path IDs. I never substitute synthetic E4 test fixtures. I reject any selected OASST user turn whose source `synthetic` field is not explicitly false.
- I prespecify smoke IDs **0, 1, 2, 100, 101**: three OASST and two ToxicChat candidates, selected without model outcomes. These are integration checks, not a scientific evaluation cohort.

`prepared-source/user_queries.csv` starts with the compatible columns `conv_id,dataset,user_query_ix,user_query`, followed by source IDs and hashes. `conversations.jsonl` retains full provenance; the corresponding `smoke-*` files contain the fixed five candidates. `manifest.json` records raw and eligible candidate counts, source-file hashes, exact selection and artifact hashes. I preserve the outputs by refusing to overwrite a nonempty preparation directory.

My September 11 local freeze contains **200 candidates and 225 user turns**. The cached OASST train split has 84,437 rows, 39,283 English/ready rows and 3,574 resulting trees; 694 full first-child paths meet the user-length rule. The ToxicChat train split has 5,082 rows, including 1,167 length-eligible user inputs. I retain the first 100 eligible selections from each recipe. Cache revision directories are `fdf72ae0827c1cda404aff25b6603abec9e3399b` (OASST) and `29df8e4dba60e1f4af4b4075c0705c5b313548a8` (ToxicChat); exact source-file hashes are in the manifest.

I independently executed only the authors' two selection function definitions against in-memory copies of these cached inputs, replacing their dataset loader with a local-only function. All 100 OASST full path IDs and truncated user-turn sequences match; all 100 ToxicChat selections match the actual `datasets.Dataset.shuffle` implementation in order. `prepared-source/independent-verification.json` records those checks, runtime versions and verified output hashes.

```bash
.venv/bin/python e9/prepare_inputs.py --self-test
.venv/bin/python e9/prepare_inputs.py
.venv/bin/python e9/prepare_inputs.py --verify
```

## Generation and later eligibility

The paper describes **200 conversations**, mixing one and two user turns. The authors regenerate final responses and reasoning with the target model. NB01's GPT-OSS-20B provider recipe is `nebius/fp4`, reasoning medium, temperature 1, top-p 1, top-k 0 and 4,000 output tokens. Later-turn history carries the previous final response, not previous reasoning. I do not run generation in input preparation.

NB02 cell 25 rejects conversations with missing user, assistant or CoT fields; strips these fields; requires every message to have at least 100 characters, total content length below 10,000 characters and at most two user turns; and rejects any full message contained in another. It then samples **at most 30 conversations with seed 123**, with the comment “100 for full test.” Thus the 200 prepared candidates are neither the frozen notebook's final denominator nor guaranteed model-valid conversations. I retain source IDs through later renumbering and record every exclusion.

The authors' local fallback in NB01 cell 13 retains only the final assistant segment before pivoting, leaving the CoT field missing. A literal fallback then fails the later missing-CoT filter. I must extract both reasoning and final content explicitly when using local generation and disclose that repair.

## Rendering, masks and probes

I use the GPT-OSS-20B settings from `config/probe.yaml` and the frozen custom `gptoss.j2` template. All conditions retain the same tagged system prefix. Baseline folds each CoT into the following assistant message before applying the custom template. No Tags joins all original message contents with single spaces. Injection joins the same contents and places them in the authors' standalone `functions. to=assistant` commentary message, without adding a preceding tool call. `user_tagged` and other-model `alt_tagged` are not Figure 23 conditions.

I identify original-message content through exact substring matching with any positive character overlap, preserving the authors' boundary convention. Unmatched template and prefix tokens are excluded. CoT stays in the forward context but is not part of the plotted user/final-assistant denominators. I use the entire eligible content: no first-120/160 truncation and no cross-condition common-token intersection.

The Figure 23 classifier space is **`uat`**: User, Assistant and Tool. I use each layer's classifier at that layer's post-attention normalization output, before its MLP. The original GPT-OSS-20B source probes **0, 2, …, 22**, although the paper says all layers. My requested **0–23** traces are an explicit extension. The prompt-split native classifiers and their class order must be recorded; I do not silently replace them with the September 5 five-role probe or the gardening `suca` probe.

## Both metrics, with separate names

The paper describes mean role probabilities. NB02 cells 37/45 save them as `all_conv_projs_*` and `alt_conv_projs_*`. NB02 cells 38/46 separately save argmax correct-role accuracy as `all_conv_acc_*` and `alt_conv_acc_*`. **NB03 reads the accuracy files for the Figure 23 plot.** The frozen notebooks contain no execution outputs establishing which metric produced the published image. I preserve this discrepancy rather than relabeling accuracy as probability.

For each condition, vector family, steering setting, layer and original role, I export:

1. **Mean role probability:** within each conversation, average the original-role probability over all tokens of that role; then average those conversation means equally.
2. **Argmax role accuracy:** within each conversation, average `argmax(probabilities) == original_role` over those tokens; then average those conversation accuracies equally.

Both use an equal-conversation mean after token averaging, not pooled tokens or equal turns. I export conversation counts, token counts by conversation/role/condition, and the exact cohort. The source rounds projection probabilities to 12 decimals after prediction and 8 in the exported `prob` column; a direct native/unrounded companion remains separately labeled. Tie behavior for source-rounded argmax rows must be recorded rather than inferred from the metric name.

## Figure appearance

Figure 23 colors represent **formatting condition**, unlike Figure 7's token-source colors:

| Condition | Legend | Exact color |
| --- | --- | --- |
| `tagged` | Baseline | `#62748e` |
| `untagged` | No tags | `#00a6f4` |
| `tool_tagged` | Injection (tool tagged) | `#ff6467` |

I preserve two stacked panels: Userness of user-style text, then Assistantness of assistant-style text; the GPT-OSS-20B column title; layer index on x; 0%, 50%, 100% y ticks and 2% y padding. Source curves are solid lines of linewidth 0.5 and alpha 0.7, with circular points of size 1.5, no smoothing or uncertainty bands. I preserve boxed `grey20` panels, `grey85` major grid, bold serif facet labels and a bottom legend. The original four-model canvas is 7 × 3.5 inches at 300 dpi; a single-model extension should disclose its adjusted canvas.

The exact font is **TeX Gyre Termes**. It is available locally in `/usr/local/texlive/2019/texmf-dist/fonts/opentype/public/tex-gyre/texgyretermes-{regular,bold,italic,bolditalic}.otf`. I register those files rather than substitute a custom theme. Steering line types are not defined in the paper; any added encoding must preserve the three condition colors and be documented.

## Intervention boundary

The user chose both vector families, separately. Input preparation does not choose their training data, normalization, coefficient, intervention site, token mask or whether interventions are independent at each layer versus propagated from a fixed earlier site. The execution protocol must freeze these choices before measurement. I do not conflate an offline classifier-weight sensitivity calculation with a propagated model intervention or with the historical neutral Tool-minus-User direction.
