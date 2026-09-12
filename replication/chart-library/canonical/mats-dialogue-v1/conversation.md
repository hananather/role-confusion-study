# Hanan asks about Neel Nanda’s MATS stream

This is a new illustrative exchange, created on 11 September 2026 (Toronto time). We authored the two User questions in Hanan’s voice. Local GPT-OSS-20B generated both analysis passages and final replies; those four passages are reproduced unchanged below. This is not a historical conversation or a quotation from Neel.

The figure measures role-probe scores while replaying this exchange under three formats. It does not test the research suggestions made in the replies.

**Reading note.** The model overstates what a score–behavior mismatch establishes. Such a mismatch does not prove that a role representation has no causal connection to behavior. Its proposed sample size, significance thresholds and “100-dimensional” control are also not our protocol: a control vector must match the activation space (2,880 dimensions at this hook). We preserve these imperfections as generated text.

## User 1 · Hanan

I'm Hanan, and I want to apply to Neel Nanda's MATS stream. I've been reading Prompt Injection as Role Confusion and experimenting with GPT-OSS-20B. The paper trains linear probes to read chat roles, such as User and Tool, from hidden activations. I can add a Tool-minus-User steering vector to those activations while the model reads a tool response containing an injected instruction. Can you suggest one small project testing whether this changes which instructions it follows? I'd like one result worth showing Neel, even if it ruins my favourite hypothesis.

## CoT 1 · Model-generated analysis

We need to propose a small project to test whether adding a Tool-minus-User steering vector changes instruction following. Should be a concrete experiment: pick a set of prompts with injected instructions, run baseline, run with steering vector, measure compliance. Use GPT-OSS-20B. Use linear probe to identify role activations, then add vector. Evaluate on a small set of instruction-following tasks. Provide a single result metric: e.g., compliance rate or difference in logit scores. Provide a concise suggestion.

## Assistant 1 · Model-generated reply

**Mini‑experiment: “Steering‑Vector Compliance Test”**

1. **Select a small instruction set** (e.g., 10 prompts that ask the model to follow a specific rule: “Never mention the word *foo*” or “Always answer in two sentences”).  
2. **Baseline run**: Feed each prompt to GPT‑OSS‑20B, record the generated text and compute a compliance score (1 = fully compliant, 0 = non‑compliant).  
3. **Steering run**: For each prompt, identify the hidden activation slice that the linear probe labels as *Tool* (or *User*). Add the pre‑computed *Tool‑minus‑User* steering vector to that slice before decoding. Generate the response and compute the same compliance score.  
4. **Analysis**: Compute the mean compliance difference Δ = (steering – baseline). A statistically significant positive Δ (e.g., Δ > 0.2 with p < 0.05) would show that the steering vector biases the model toward following the injected instruction.  
5. **Result to show Neel**: A single table of compliance scores per prompt, with a clear Δ and a p‑value, plus a short narrative explaining whether the steering vector increased or decreased instruction following. This concrete, reproducible metric will let you demonstrate the effect (or lack thereof) of your hypothesis.

## User 2 · Hanan

Okay, suppose the Toolness probe shoots up but the model still follows the injected instruction. Is that a useful result, or have I built an expensive mood ring? What's the first control you would run?

## CoT 2 · Model-generated analysis

Need to explain that if probe indicates toolness but instruction still followed, indicates probe not capturing causal role. It's still useful to show that probe activation doesn't guarantee behavior. First control: test whether the steering vector actually changes the probe activation (i.e., measure probe output before/after). Also test a random vector to see if any vector changes behavior. Also test a baseline with no steering. Provide suggestion.

## Assistant 2 · Model-generated reply

If the *Toolness* probe spikes yet the model still obeys the injected instruction, you’ve shown that the probe’s signal is **not causally linked** to the behavior you’re trying to steer. That’s a valuable finding: it tells you that the probe is a *diagnostic* marker, not a *control knob*. In other words, the “mood ring” is telling you the model *sees* the tool role, but that perception alone doesn’t alter its policy.

**First control to run**

1. **Probe‑activation sanity check**  
   *Run the same prompts with the steering vector applied, then immediately feed the hidden activations into the probe.*  
   - If the probe’s *Tool* score rises as expected, you’ve confirmed that the vector is indeed moving the activations into the probe’s “tool” subspace.  
   - If the score does not change, the vector is ineffective or the probe is mis‑aligned.

2. **Behavioral baseline with a random vector**  
   *Apply a random 100‑dimensional vector (same norm as your steering vector) to the same hidden slice and generate responses.*  
   - If the random vector produces no systematic change in compliance, you can attribute any effect you see with the engineered vector to its specific direction rather than to a generic perturbation.

These two controls let you separate “probe activation” from “behavioral change” and will strengthen the claim that the steering vector’s effect (or lack thereof) is meaningful.
