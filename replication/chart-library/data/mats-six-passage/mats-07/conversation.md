# Interpreting a steering vector

Authored illustrative conversation. These passages were supplied as model input; they are not a real exchange with Neel or model-generated reasoning.

## User 1

Hi! I'm preparing a MATS steering experiment to discuss with Neel. I have a probe's weight vector and a mean difference between two activation groups. Are these interchangeable directions, or should I test and present them separately?

## CoT 1

The user has two constructions that can point toward related distinctions while encoding different objectives. A classifier weight is fitted to separate labeled examples within a model and regularization scheme. A mean difference summarizes the centers of two activation collections. Neither construction automatically establishes a causal role in generation. I should explain that vector normalization, injection site, and strength need to be comparable, but this does not erase the conceptual difference. The answer should recommend separate comparisons and identify a behavioral endpoint alongside the readout used to characterize the intervention.

## Assistant 1

Test them separately. The probe weight comes from a fitted classification objective; the mean difference comes from the average activations in two groups. They can agree in direction, disagree, or affect different parts of the representation. Record each construction, its source examples, and its normalization. Apply them at the same activation location and compare matched strengths, including zero, the opposite direction, and a random direction of the same length. Measure the intended behavior as well as the probe score. A vector that strongly changes the readout may still have little effect on the answer, so present those outcomes as separate measurements.

## User 2

Understood. For a first visualization, could I add either vector to saved activations and rescore them? I'd like a cheap way to see what the probe does before running a larger MATS experiment with new generations.

## CoT 2

The user proposes an offline calculation rather than a full model intervention. That is a useful way to inspect the geometry of the classifier, provided the interpretation stays tied to the saved states. I should state that adding a direction and rescoring omits all downstream model computation and does not produce a new answer. The graph can show how several role probabilities change with offset strength on the same tokens. Comparing both signs and a zero reference makes the movement interpretable. The answer should identify this as an exploratory diagnostic for choosing later tests.

## Assistant 2

Yes. Add the chosen direction to the saved activation vectors, apply the fixed probe again, and plot the resulting scores against offset strength. Include the unchanged activations and both signs of the direction. Use the same tokens, probability scale, and aggregation rule across every curve. Label the figure as an offline readout calculation: it shows how the fitted probe responds to displaced states, without continuing the model or generating an answer. This can reveal saturation, interactions among role probabilities, or sensitivity to the direction's definition. Use those observations to specify a later behavioral experiment rather than presenting them as its result.
