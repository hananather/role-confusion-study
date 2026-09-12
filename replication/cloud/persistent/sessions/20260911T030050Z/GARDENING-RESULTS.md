# My gardening replication result

I completed the three fixed gardening conditions and generated Figures 7 and 20–22 from the saved native layer-12 probe probabilities. In this conversation, the probe retains role-style signals under misleading tags. I did not reproduce the paper's reported numerical probabilities, and the cause of that mismatch remains unresolved.

| Condition | Mean CoT probability over all labeled CoT tokens | Figures 20–22 subset | Figure 7 subset | Published comparison |
| --- | ---: | ---: | ---: | ---: |
| Correct tags | 64.03% | 64.03% | 64.33% | 85% |
| No tags | 71.04% | 71.04% | 71.78% | 82%; 83% in Figure 7 |
| All user tags | 70.97% | 70.97% | 71.72% | 85% |

The published prose's precise denominator is not established. None of the three implemented subsets resolves the numerical gap. The full content means use 179 CoT tokens per condition; Figure 7 uses 177. The complete user/CoT/assistant denominators are 97/179/582 for full content, 97/179/320 for Figures 20–22, and 95/177/240 for Figure 7. Its 512 displayed positions match token text across all three conditions.

Under all-user tags, assistant-style content averages 97.94% Assistantness while CoT-style content averages 70.97% CoTness. These are probe measurements on this fixed conversation, not attack success rates or evidence of a confirmed causal mechanism. The second CoT segment is weaker than the first: 43.86%/55.01%/54.15% versus 87.35%/89.58%/90.44%, in condition order. This dispersion is visible in the figures and differs from the paper's more uniform CoT plateau.

I independently reviewed all four PNGs for panel order, colors, message boundaries and labels. I found no material probability/label alignment defect: 2,745 labeled tokens each have four unique target-role rows, finite probabilities and correct matched positions. The corrected serialized-mean check passes exactly over 179 tokens. The analysis-only repair preserved all 11 protected forward/probability/prompt files; it did not rerun the model.

I retain solver convergence and probability calibration as unresolved qualifications. The close H200 agreement for the trained probes does not establish reproduction of this separate gardening result. I continue the already approved Appendix K and RH6 queue without changing methods to improve agreement.

Evidence: [completed metadata](outputs/20260911T030050Z/appendix-e-L12/metadata.json), [published comparison with subsets](outputs/20260911T030050Z/appendix-e-L12/published-comparison.csv), [all subset means](outputs/20260911T030050Z/appendix-e-L12/mean-role-probabilities.csv), [Figure 7](outputs/20260911T030050Z/appendix-e-L12/figure-7-cotness-overview.png), [serialized mean check](outputs/20260911T030050Z/appendix-e-L12/mean-recomputation-check.json), and the saved failure evidence in `failed-attempts/`.
