# Choosing an ablation

Authored illustrative conversation. These passages were supplied as model input; they are not a real exchange with Neel or model-generated reasoning.

## User 1

Hi! I'm planning a MATS study to discuss with Neel. A role probe seems sensitive to both message labels and writing style. What ablation would help me distinguish those explanations while keeping the experiment easy to understand?

## CoT 1

The user is considering two possible sources of a probe signal: formatting and content style. I should turn these into factors that can be varied independently rather than recommending a large collection of loosely related edits. Holding the words fixed while changing tags tests a formatting effect on that input. Holding the tags fixed while rewriting style tests a content effect, although the rewrite necessarily changes tokens and may introduce other differences. A small crossed design can expose interactions between the two. I should explain what each comparison isolates and where residual ambiguity remains.

## Assistant 1

Begin with a crossed comparison. Take the same passage and render it under two role labels; then repeat those labels with a second version that expresses the same substance in a different style. This gives four conditions. Compare labels within each wording, and compare wording within each label. Keep the surrounding conversation fixed, including its order and length wherever possible. If the effect of a label changes across styles, show that interaction rather than reporting only one overall average. Preserve both wordings so readers can judge whether the rewrite introduced a new topic or instruction that could explain the difference.

## User 2

That sounds manageable. How should I decide which parts of the wording to change? I don't want an ablation that changes everything at once, because then a difference in the graph would be hard to interpret.

## CoT 2

The user is correctly concerned that a broad rewrite may conflate several content changes. I should recommend a concrete transformation with an explicit scope. For instance, a passage can move from an internal planning voice to a direct explanatory answer while retaining the factual propositions and their order. That does not make the texts identical in meaning or tokenization, so those limitations should be acknowledged. Multiple independently prepared rewrites would help distinguish a style pattern from one unusual paraphrase, but the initial demonstration can remain small and clearly labeled as such.

## Assistant 2

Specify the transformation before writing the alternative. For example, convert an internal planning passage into a direct explanation while preserving its factual claims, examples, and order. Avoid adding a new instruction, changing the topic, or inserting extra evidence. Keep a short checklist of the properties you intended to preserve, and ask another reader to inspect the pair before measurement. Record token counts because a stylistic rewrite may also change length. Treat the first pair as an illustration of the comparison. To claim a repeatable style effect, prepare several independent pairs using the same rule and evaluate the whole set rather than selecting the clearest pair afterward.
