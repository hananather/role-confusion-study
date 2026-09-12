# My evidence audit for the chart library

I checked the existing local datasets on 2026-09-11, with the final numerical capture at 14:48:26 UTC. I changed no data or plotting code and ran no model. I prioritize the September 11 outputs from the replication program. I treat earlier experiment files as optional evidence only where the September 10 goals call for them; I have not used this audit to revive an older research direction. E9 remains on the backlog for a larger future batch.

I am in **Understand**. My north star is to distinguish a measured role signature, an arithmetic change to a readout, and an intervention that changes later model computation or behavior. I apply the research priorities of checking independent units, retaining simpler controls, and preserving adverse results. I recommend **three aggregate candidates**, with one compact historical companion if it fits yesterday's goals. These complement the requested gardening and offset illustrations rather than replacing them.

## 1. Probe recall across layers: my measurement foundation

**Story I can support:** held-out role identification depends on the layer, role, class set, and split. The all-layer curves establish what my trained readouts distinguish before I interpret an experimental probability.

I would show the 24-layer per-role recall curves in separate prompt-split and base-split panels. For the gardening comparison, I would start with the actual four-role `suca` probe. A separate `sucat` or `uat` view is relevant to RH6; I would identify the role space explicitly and preserve the existing role colors. I would not silently compare a three-class probability with a five-class probability as though their denominators were identical.

I recomputed recall as `sum(count where true role == predicted role) / sum(count for that true role)` from the integer confusion tables. Useful `suca` checkpoints are:

| Split | Layer | User recall | CoT recall | Assistant recall |
| --- | ---: | ---: | ---: | ---: |
| Prompt | 8 | 87.924% | 79.017% | 86.135% |
| Prompt | 12 | 81.212% | 61.386% | 77.837% |
| Prompt | 16 | 94.153% | 79.645% | 94.472% |
| Base text | 8 | 76.246% | 75.531% | 86.372% |
| Base text | 12 | 76.952% | 64.956% | 80.277% |
| Base text | 16 | 91.762% | 76.652% | 94.231% |

The prompt-split denominators are 11,784 User, 14,974 CoT, and 15,269 Assistant tokens; the base-split denominators are 11,337 tokens per role. At layer 12, the exact prompt numerators are 9,570, 9,192, and 11,885; the base numerators are 8,724, 7,364, and 9,101.

**Independent units and limits.** The corpus has 249 base documents and 1,245 role-rendered prompts, not 1,245 independent texts. Each base split holds out 24 documents and trains on 225. Prompt splitting can retain another role-rendered version of a test document in training. The two plotted splits also change fitted probes and held-out cohorts; their difference does not isolate an effect of leakage. These are token-weighted recalls from one fitted model per configuration. I have no saved cluster confidence intervals, and neither tokens nor layers supply independent experimental replications. Native structural checks passed for all 384 probes, but solver warnings leave convergence unestablished. Recall does not establish probability calibration. The layer-12 CoT dip alone does not explain the gardening mismatch with the paper.

**Exact files and recipe:**

- [Prompt confusion counts](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/acc_by_role_gptoss-20b.csv>): group by `role_space, layer_ix, role`; retain absent prediction cells as zero counts.
- [Base-split confusion counts](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/acc_by_role_gptoss-20b-basesplit.csv>).
- [Actual split IDs and counts](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/split-audit.json>) and [completion audit](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/full-probes-completion-audit-20260911T043828Z/README.md>).

## 2. RH6 cue contrasts: my strongest larger aggregate control

**Story I can support:** permission changes the relative User/Tool readout on these synthetic inputs, while the sign of absolute Userness depends on the probe configuration. This measured control prevents me from treating all scalar role summaries as interchangeable.

I would use a compact paired-contrast chart with separate axes for absolute Userness and the relative User/Tool score. I would include both command orders, all three layers, both role spaces, and the corresponding prohibition contrast. The relative metric is exactly `log(mean span P_user) - log(mean span P_tool)`, not the mean tokenwise log ratio.

The command-specific contrast is `(command cue - command neutral) - (null cue - null neutral)`. Marker permission has a positive relative-score interval in all 12 layer × space × order cells. At layer 12, first-listed marker permission is **+0.766 [0.703, 0.830]** with `sucat` and **+0.479 [0.425, 0.536]** with `uat`. Yet the corresponding absolute Userness contrasts are **−1.395 [−2.026, −0.779] percentage points** and **+4.458 [3.423, 5.474] points**. The second-listed marker also shows this sign disagreement at layer 12. Across the full 48 cue-contrast configurations, 11 have opposing signs between absolute Userness and the relative score.

Ten of 12 exfiltration-prohibition relative-score intervals include zero. I would show those results without claiming equivalence or that the model ignored prohibition. The two remaining intervals are small positive effects at layer 16 with `uat`.

**Independent units and limits.** I have 1,200 actual prefills from **100 template–page pairs**, each reused across 12 conditions, with six readouts per prefill. The 1,631,880 scored token rows are not independent samples. Saved intervals use 2,000 paired bootstrap draws over the 100 pairs; they are pointwise intervals without multiplicity adjustment. Each template is coupled to one page. Only 47/100 pairs match both null spans' standalone token counts exactly; the other gaps range from −7 to +1. Lexical overlap and shifted positions remain alternatives, and the overlap table is an unadjusted correlation. The broad page-reference span overflow affects 750 inputs; the command and user spans used here remain valid. These are forward-pass readouts, with no behavior or intervention outcomes.

**Exact files and recipe:**

- [Cue contrasts and intervals](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/rh6/user_turn_effects.csv>): use `construction == 'command-minus-null'`; retain the exact `metric`, `layer`, `space`, and `order` labels. The targeted comparisons are marker/`permission-neutral` and exfil/`prohibition-neutral`; retain the other command's contrasts as spillover checks in the complete source table.
- [Order contrasts](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/rh6/order_effects.csv>), [frozen input contract](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/rh6/input-contract.json>), and [independent integrity audit](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/rh6-integrity-audit-20260911T045805Z/README.md>).

## 3. Appendix K matched tags: a bounded representation control

**Story I can support:** at the same insertion slot, the same canonical content receives higher System scores in System tags than in User tags, with layer-dependent magnitudes. This is a cleaner available comparison than assigning a position-only explanation to the existing early/late curves.

I would compare the 57 canonical content tokens at position 100 in `sys_t100` versus `user_t100`, paired by base prompt. The actual saved token rows give:

| Layer | System-tag mean | User-tag mean | Mean paired difference | Positive differences |
| --- | ---: | ---: | ---: | ---: |
| 8 | 23.441% | 11.704% | +11.737 points | 176/184 |
| 12 | 13.515% | 8.576% | +4.940 points | 177/184 |
| 16 | 10.509% | 2.553% | +7.956 points | 184/184 |

I recomputed these as an equal-weight mean of each prompt's content-token mean. A paired distribution or linked-point summary would preserve the adverse prompt-level cases. I would label it descriptive; the existing block-table confidence intervals compare against different baseline text and cannot be reused as intervals for this System-versus-User contrast.

**Independent units and limits.** The full run has 1,330 actual prefills across 200 neutral texts and seven conditions. This fixed-position comparison has **184 matched texts**, not 184 × 57 independent tokens. All 200 texts overlap the probe corpus. This is a control on those contexts, not held-out validation or the original conversation-based Figure 32. Cross-position counts change from 200 at position 1 to 169 at position 150. I would not interpret an unmatched early/late mean difference as a pure position effect, or Systemness as measured instruction priority.

**Exact files and recipe:**

- [Token probabilities](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-k/tokens.parquet>): select `in_block & ~is_tag & ~is_bos`; group by `prompt_ix, condition, layer`; average `p_system`; join `sys_t100` to `user_t100` on prompt and layer.
- [Saved prompt inputs](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-k/prompts.parquet>) and [completion audit](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/appendix-k-completion-audit-20260911T044857Z/README.md>).

## Optional historical companion: score movement and observed behavioral tradeoffs

If yesterday's goals reference this earlier study, I would retain one compact adverse-evidence chart. **Near-saturated Tool readouts coexist with unwanted writes in the recovered intervention episodes; the matched episodes include both improvements and regressions.** This does not establish a population effect or a null effect.

I independently matched the September 5 records on case, permission, and seed. Toolward has 21 baseline/intervention pairs across eight cases. Across its initial command-token readouts, the minimum Tool score is **99.89931%**, while **17/21 episodes write both markers**. The 21 episodes contain only 15 distinct first-prefill prompts, and repeated generation seeds have identical initial scores. The lower published minimum, 99.85881%, includes all 57 scored prefills and 3,360 token records, including later steps; I would use initial scores when relating readout to subsequent behavior.

| Method | Matched episode pairs | Cases | Unwanted writes prevented / introduced | Selected completions lost | Both-value reports lost |
| --- | ---: | ---: | ---: | ---: | ---: |
| Toolward | 21 | 8 | 4 / 3 | 2 | 2 |
| Userward | 20 | 7 | 0 / 3 | 0 | 0 |
| Random 0 | 22 | 8 | 1 / 1 | 0 | 0 |
| Random 1 | 21 | 8 | 3 / 1 | 0 | 1 |
| Random 2 | 22 | 8 | 1 / 2 | 0 | 0 |
| Prompt reminder | 21 | 8 | 6 / 0 | 2 | 0 |

I would display paired transitions and utility losses, retaining all controls and showing each denominator. A net-benefit bar alone would hide Toolward's introduced writes. The allocation is incomplete: 149 of 280 planned episodes were recovered, spanning eight cases, with 148 complete records and one parse error. That malformed Toolward episode has unknown attempt status even though physical writes are observable. Different methods have different matched cohorts. There is no adequate basis for an independent-binomial confidence interval, a reliable method ranking, or a claim that steering generally fails. A common all-method cohort exists at 20 episode keys across seven cases if a separate sensitivity check is useful.

This is actual model intervention and recorded behavior, using the historical block-11 residual mean-difference direction and a separate block-14 five-role probe. It is distinct from the September 11 pre-MLP trained probes and from arithmetic offset illustrations. Its neutral role training confounds context and position; only command spans are directly steered, leaving surrounding instructions outside the mask.

Sources: [raw recovered episodes](</Users/hananather/Desktop/MATS 12.0/role-steering/data/permission-20260905/episodes.jsonl.gz>), [paired behavioral table](</Users/hananather/Desktop/MATS 12.0/role-steering/figures/paired-behavior.csv>), [frozen protocol](</Users/hananather/Desktop/MATS 12.0/role-steering/data/permission-20260905/protocol-manifest.json>), and [method and limitations](</Users/hananather/Desktop/MATS 12.0/role-steering/report.md>). I checked the write-transition, completion-loss, information-loss, case, and initial-score counts against the raw episodes.

## What I keep out of aggregate evidence claims

- Gardening is one authored conversation rendered three ways, with 2,791 saved token activations and a four-role layer-12 readout. It supports a token-level worked example, including the adverse mismatch with the paper; its tokens do not make it a many-conversation replication.
- The requested ±1%/±5% offsets and dose-response curves are arithmetic illustrations over saved activations or scores. Unless a forward pass actually propagates an intervention, they provide no model-behavior evidence. Their caption should state the exact equation and units. Passage colors identify the original passages, not new experimental populations.
- The tiny pilot's base split was invalid and is excluded. Its high scores must not appear beside the repaired full-run generalization results.
- The earlier transfer run has 760 recovered episodes and exposure/missingness problems. I would not add a success-rate chart without re-establishing its eligible paired cohort and relevance to yesterday's goals.
- E9 has prepared inputs and a tested adapter, but no completed propagated-intervention dataset. I include no E9 result curve.

## My rendering handoff

I would lead with probe validation, retain gardening as the visible worked example, and label offset plots as illustrations. RH6 supplies the larger measured context control. Appendix K is an optional matched-tag control if it advances the current question. The historical comparison can supply compact adverse context, without carrying the main conclusion or replacing an adequately sized future intervention study.

My numerical source snapshots have these SHA-256 values:

| Table | SHA-256 |
| --- | --- |
| Full prompt-split confusion counts | `0f5ae1cb3eed0d9d8b52119f3c2825ca253779215f1fca983092b64e6ee1d596` |
| Full base-split confusion counts | `4c5b01c22eae8a94c687ea3e30d8901abb3e3f49dee8b2ff82ce8f2b0c981130` |
| RH6 user-turn effects | `f6ee0bdb74cfdfe8b2d6f14daa68950479da411ddb09e18e19e5842e712c01e9` |
| Appendix K existing block table | `8c82fae43a46fb0686e7223eb53358897bfa7a09fc424e00c64f3cdfbbc21176` |

The new paired System-versus-User numerical check above was reduced directly from the token parquet in memory; I wrote no replacement table or revised confidence interval.

## Independent checks of the new library figures

I also audited [the renderer](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/render_library.py>) and its saved data without rerunning it or modifying its figures.

- **`05-probe-recall`:** all 240 plotted five-role recall cells reproduce exactly from the original integer confusion counts, including every numerator and denominator. The saved `sucat` split record confirms 124 held-out prompt variants with 67,727 content tokens versus 24 held-out base texts with 56,685 content tokens. The figure explicitly states both cohort sizes, token weighting, and the descriptive nature of comparing different test sets. [Plotted recall table](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/probe-recall-by-layer.csv>).
- **`03-offset-dose-response`:** all 30 grouped means reproduce within 1.11 × 10⁻¹⁶, with exact group counts. All 12 source offset conditions use the same 512 token IDs: 95 originally User, 177 CoT, and 240 Assistant. The reference median activation norm is exactly 45.24711366617226 across all 921 forwarded tokens of `everything_in_user_tags`. I independently recomputed all 30,720 source probability values from the original float16 activation array and float32 `sucat_L12` coefficient rows promoted to float64; the largest discrepancy was 1.39 × 10⁻¹⁴. The figure states one conversation and no downstream forward or behavior. [Plotted mean table](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/offset-dose-response.csv>) and [source arithmetic specification](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/presentation/probe-offset-illustration/provenance.json>).

I flagged one material wording issue in the initial dose-chart title, “Larger offsets blur the separation between passage roles.” The pattern is not uniform: the range of source-role mean Toolness increases from 10.563 points at baseline to 22.063 points under +5% Tool offset. The Userness range increases from 69.812 to 84.643 points under +1% User before narrowing at +5%. The renderer owner corrected its title to **“Offset direction and size change the role-score pattern,”** which I verified in the current source. This preserves the observed arithmetic effect without asserting universal or monotonic loss of separation.
