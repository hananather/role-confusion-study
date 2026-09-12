# Designing a text replacement control

Authored illustrative conversation. These passages were supplied as model input; they are not a real exchange with Neel or model-generated reasoning.

## User 1

Hi! I'm designing a MATS experiment to discuss with Neel. I want to know whether a permission sentence changes how a model represents an instruction inside a webpage. What would make the comparison more informative than one before-and-after plot?

## CoT 1

The user wants to isolate a permission-related change from general effects of adding text to the conversation. I should propose an explicit comparison between instruction-bearing content and a replacement passage, each with and without the same permission sentence. This creates a difference of changes, but it still depends on the quality of the replacement and does not make the wording variation magically isolated. The ordinary task, surrounding webpage, order, and scored span should be specified. I should keep representation and behavior separate, since the proposed measurement only concerns captured activations.

## Assistant 1

Use four conditions: instruction content with and without permission, and replacement content with and without the same permission. Keep the ordinary user request and the rest of the webpage fixed. Choose replacement prose that occupies the intended span without asking for the action, and document any remaining differences in length or wording. Compare the permission-related change in the instruction span with the change in the replacement span. This tests whether the added sentence has a different association with the two constructions. Save the exact texts, and measure any generated behavior separately if you want to connect the representation change to an action.

## User 2

Thanks! How do I explain what remains uncertain after that control? I want the MATS report to make a clear affirmative claim, but I don't want the phrase permission effect to suggest that every alternative explanation has been removed.

## CoT 2

The user is asking for a claim whose wording matches a controlled but imperfect comparison. I should name the exact intervention and population rather than imply an abstract causal mechanism of permission. The same added sentence can change lexical context, attention, and interpretation at once. A replacement control addresses some shared changes while leaving others possible. The answer should state the measured contrast directly, then identify the most consequential unresolved explanation and a follow-up that could distinguish it. This gives the result a clear meaning without surrounding it with an unstructured list of caveats.

## Assistant 2

Name the actual comparison: adding this permission sentence changed the chosen readout differently for the instruction passage and its replacement control. State the model, sampled contexts, scored span, and aggregation rule alongside the result. That is a concrete finding if the measurements support it. Then explain the main remaining uncertainty, such as whether the difference depends on lexical overlap between the permission sentence and the instruction. A follow-up can vary the permission wording while preserving its intended meaning. This lets the reader understand both what the current comparison establishes and which broader interpretation still requires another test.
