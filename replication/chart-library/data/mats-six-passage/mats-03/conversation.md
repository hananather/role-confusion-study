# Designing a readable figure

Authored illustrative conversation. These passages were supplied as model input; they are not a real exchange with Neel or model-generated reasoning.

## User 1

Hi! I'm designing the opening figure for a MATS report that I hope to discuss with Neel. I want it to explain a role probe to someone who has never seen our project. What should the figure show first?

## CoT 1

The user wants a figure that teaches the measurement before asking the reader to interpret a result. I should recommend a small conversation whose structure is visible, because role probabilities are hard to understand when the underlying text is absent. The visual needs aligned token positions, consistent scales, and colors that identify the original passage roles. Different formatting conditions should be separated into panels. The answer should make each element earn its space and leave technical details such as the model hook and probe fit in the caption or methods.

## Assistant 1

Show one short conversation with two questions, two reasoning passages, and two answers. Place the same conversation under three formatting conditions in vertically aligned panels. Use the same vertical scale in every panel, and use passage colors consistently so the reader can follow the text across conditions. Beneath the plot, label the six passages and include a short excerpt from each. The figure should answer one question: how does the measured role signal change when the same text receives different tags? Put the model, layer, probe classes, and treatment of special tokens in the caption. Provide the full transcript beside the downloadable figure.

## User 2

Great. How do I choose which excerpts appear below the plot? Some reasoning passages are much longer than the questions, and I'd like the text to be readable without making the diagram crowded or changing its overall structure.

## CoT 2

The user is asking about a layout problem caused by unequal passage lengths. I should separate the amount of text shown from the number of token measurements plotted. Excerpts can be short while all eligible measurements remain visible under a documented display rule. Taking the opening words of each passage gives an interpretable and reproducible excerpt choice. The six labels should establish the conversation structure even if a narrow passage has room for little text. I should also avoid moving labels in a way that makes their relationship to token boundaries ambiguous.

## Assistant 2

Use the opening phrase of each passage and keep the excerpt rule consistent. A short question might fit in full; a longer passage can end with an ellipsis after the same small number of displayed lines. Put the passage name directly above its excerpt, aligned with the corresponding region of the graph. Keep token boundaries and plotted data unchanged when adjusting the typography. If a passage is too narrow for readable text, give its excerpt a separate aligned column beneath the panels and connect it through the passage label. Include the complete conversation as a companion page so the figure stays compact without withholding the input.
