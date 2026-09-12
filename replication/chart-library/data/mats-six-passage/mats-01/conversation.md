# Choosing one question

Authored illustrative conversation. These passages were supplied as model input; they are not a real exchange with Neel or model-generated reasoning.

## User 1

Hi! I'm sketching a MATS project to discuss with Neel. I have ideas about role probes, steering vectors, and agent behavior, but the plan feels scattered. How can I turn these into one useful research question?

## CoT 1

The user wants help narrowing a research plan, rather than a list of additional techniques. I should begin with the decision the project could change. Role classification and behavior are related questions, but a probe score alone cannot settle whether an agent follows a command. A concrete first step is to choose one behavior, identify two explanations, and design a comparison that separates them. I should keep the example small enough to understand while explaining how a later evaluation would test whether it extends beyond the initial case.

## Assistant 1

Start with a question that connects a measurement to a decision: does changing a message's role label change which instruction the assistant follows? Write down one ordinary task, one conflicting instruction, and the behavior you will score. Then compare the same text under different labels while keeping the rest of the conversation fixed. Use the probe to describe the internal change, and inspect the answer separately to measure behavior. Your first deliverable can be a single transparent comparison with two plausible explanations and one control that distinguishes them. Expand the sample after the task and measurement are working.

## User 2

Thanks! How would I explain this choice in the MATS write-up without making a tiny example sound more conclusive than it is? I'd like the reader to understand why this was a sensible place to start.

## CoT 2

The user now needs language that connects a small exploratory comparison to a larger research aim. I should distinguish why the example is informative from how broadly it generalizes. A small case can establish that a pattern exists in that case and expose bugs in the measurement. It cannot estimate the frequency of the pattern across tasks. The answer should therefore put the question and comparison first, describe the observed result only after it is measured, and make the next test follow from the remaining uncertainty.

## Assistant 2

Explain the purpose of the example before describing its outcome. For instance: we begin with one controlled conversation to make the input, measurement, and competing explanations easy to inspect. The example tests whether our pipeline can detect a role-dependent difference under a precise change to the prompt. It does not estimate how often that difference occurs across conversations. Follow the figure with the control you ran and the uncertainty it leaves. Then describe the broader evaluation as the test of generality. This gives the reader a reason to care about the small example without treating it as population evidence.
