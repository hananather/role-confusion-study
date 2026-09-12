# My next batch of experiments

I deferred fresh model work on September 11. I want a larger, adequately justified batch rather than converting small smoke tests into results. This file routes the open questions recovered from yesterday's conversation; the [complete goal audit](review/claude-goals-audit.md) preserves the original task descriptions and their provenance.

| Question from yesterday | What I can show now | What remains |
| --- | --- | --- |
| Do the probes reproduce the authors' method and readings? | All-layer H100 probes, both splits, source/corpus checks and descriptive validation curves. | Stronger text-level uncertainty, optimizer-quality resolution, and fresh held-out evaluation where required. The 24-document grouped holdout is not a large evaluation cohort. |
| Do role patterns survive absent or misleading tags? | One measured gardening example under three conditions. | An adequately sized real-conversation cohort, with retained source IDs and attrition accounting. |
| Does a steering offset survive through the model? | Local offline offset illustrations; earlier steering records with a different setup. | [E9 / Figure 23](../e9/BACKLOG.md): both vector families separately, all 24 layers, matched random and sign controls, frozen texts, real intervention forwards. |
| Does the intervention change attack behavior while preserving task performance? | Older recovered outcomes are adverse context with small clustered samples. | StrongREJECT and the steering pilot/selection/confirmation program need a complete behavioral dataset, judge validation, utility controls, and prespecified precision. |
| Does apparent progressive role confusion exceed position effects? | Probe accuracy-by-position outputs and the completed Appendix K neutral-position control. | E12's full held-out styled/neutral/shuffled/position-shift comparison; the existing smoke output is insufficient. |
| Do user permissions change a tool command's reading, source attribution, and behavior? | RH6: 100 template–page pairs, 1,200 prefills, paired command/null reading contrasts. | Full source-attribution and action outcomes on the same units, plus improved position and lexical-overlap controls. |
| Does the Appendix L transfer/statistics idea reveal persistent influence? | Yesterday's separately scoped planning and reviews. | Keep the separate workspace and design gates. No result belongs in this gallery before a completed, adequately sized dataset exists. |

## My sample rule for every future experiment

I define the independent unit, primary effect, smallest useful effect, target interval width or power, pairing/clustering, and multiple-comparison plan before fixing the sample. Extra tokens, layers, prompts from the same conversation, or repeated seeds do not automatically add independent units. I preserve adverse results and report uncertainty at the correct unit.

For Figure 23, my planning floor is **200 eligible independent conversations after exclusions**, with a balanced-source target and enough prespecified reserve candidates to cover attrition. This floor is not a proof of adequacy; the variance of paired conversation differences and my desired precision may require more. I do not fall back to a small claim-producing sample without a fresh decision.

The original paper describes 200 conversations, while the frozen notebook requests 30 after filtering and comments 100 for a full test. Its aggregation uses all eligible original-role content tokens within each conversation, then weights conversations equally. I preserve the unresolved publication denominator and probability-versus-accuracy discrepancy in the [source contract](../e9/SOURCE-CONTRACT.md).

I distinguish an engineering smoke test from an inferential experiment. A smoke test may be small to test plumbing; it does not replace the planned sample or supply a scientific effect estimate.

No fresh GPU job, model generation, API judge, or model forward pass is authorized by this backlog. I retain the prepared inputs and analysis for a later explicitly approved batch.
