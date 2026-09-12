# E9: deferred Figure 23 and propagated-vector comparison

**I deferred E9 to a future batch on September 11, 2026. I have no completed E9 model measurements or experiment results.** I retain the prepared inputs, vector assets, adapter, and analysis code. This deferral supersedes earlier scheduling language for E9; it does not start generation, inference, APIs, or rented compute.

I am in **Explore**, preparing an **Understand** experiment. My north star is whether a vector introduced at its specified model site produces a direction-specific change that persists through later layers. I compare the historical neutral Tool-minus-User activation direction and the saved User/Tool classifier-row directions separately. A propagated intervention and independent offline edits of saved layer activations answer different questions. This future comparison concerns propagation; it does not establish a behavioral benefit by itself.

## What is prepared

| Material | Current evidence |
| --- | --- |
| [Source contract](SOURCE-CONTRACT.md) | Frozen paper/notebook definitions, model, conditions, masks, metrics, colors, and documented discrepancies. |
| [Candidate inputs](prepared-source/manifest.json) | 200 source candidates: 100 OASST and 100 ToxicChat, containing 225 user turns. These are user text before target-model generation and eligibility filtering. |
| [Independent input check](prepared-source/independent-verification.json) | Local source selection matches the inspected author functions. This verifies preparation, not 200 eligible generated conversations. |
| [Vector/probe assets](assets/contract.json) | Prepared identities and hashes for the separate vector families and readout probes. Preparation is not an intervention result. |
| [Forward adapter](forward_interventions.py) | Retained implementation for the future model path; no E9 model-run success is claimed here. |
| [Paired analysis](analyze.py) and [tests](test_analyze.py) | Ten synthetic-only tests passed. I verified equal conversation weighting, shared bootstrap draws, paired deltas, missing-cell rejection, probability checks, and rendering. |

I rendered temporary plots titled **SYNTHETIC VALIDATION DATA** for visual checking. They are test fixtures, not research figures or model observations. The current renderer uses the requested TeX Gyre Termes font, original Figure 23 condition colors, separate accuracy/probability/Toolness figures, and no confidence shading in main plots. Final publication styling remains part of the future batch.

## My future sample requirement

I require **at least 200 eligible, independent conversations after generation, filtering, and paired-cohort checks**, with a source-balanced target. This is a planning floor, not proof of adequate precision or power. The 200 prepared candidates do not meet that requirement yet. I will expand the candidate pool to cover attrition and set the larger required sample using a prespecified effect size and paired precision/power calculation. I will not substitute a small fallback cohort without my explicit decision.

The paper describes 200 conversations, while frozen NB02 cell 25 samples at most 30 after filtering, with the comment “100 for full test.” The actual published denominator is not established by the unexecuted notebook. I preserve this ambiguity rather than treating 30, 100, and 200 as interchangeable.

I use every eligible original-role content token within each conversation, then weight conversations equally. Multiple turns, tokens, layers, and arms do not create additional independent conversations. The paper calls Figure 23 a probability plot; the plotting notebook reads argmax-accuracy files. I keep both metrics separately named.

The remaining cohort, analysis, intervention, and presentation requirements are in [my backlog](BACKLOG.md). No command to launch E9 is authorized by this README.
