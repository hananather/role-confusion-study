# My numerical and scientific review of the six blog figures

I reviewed the current six PNGs, their renderer source, source CSVs, saved activation/probe arrays, and provenance records on 2026-09-11. My figure-hash capture was **15:19:12 UTC**. I reran only small local numerical reductions; I loaded no model and used no GPU or API. I found no material numerical defect in the current figures. The draft must preserve the distinctions below to avoid stronger conclusions than these figures support.

## The evidence boundary I would keep in the TLDR

I have measured role-probe responses to existing text and contextual changes, plus arithmetic illustrations made from saved states. **None of these six figures shows an intervention propagated through the language model or a behavioral outcome.** The gardening example is one conversation, regardless of the number of plotted tokens or panels. RH6 supplies 100 paired template–page units. Probe validation uses a separate 249-document corpus.

I can say that role scores vary with content, context, probe configuration, and the illustrated offset. I cannot conclude from these figures that changing a role score changes instruction following, improves security, or identifies an instruction-hierarchy mechanism. E9 is future work. The earlier behavioral study is a different dataset and method; any discussion of it needs its own citation and sample accounting.

## Figure 00: actual gardening forwards, four-role readout

I inspected the current [gardening figure](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/figures/00-gardening.png>). Its three panels are three renderings of **one authors' gardening conversation**, not three independent conversations. The points show CoT probability from the native prompt-split **`suca` layer-12** classifier: System, User, CoT, Assistant. Layer 12 is zero-based. The colors identify the original passage's role, not the probe's predicted class. Existing authored CoT passages are supplied as input; this plot does not show newly generated reasoning or behavioral responses.

Each panel retains 512 matched displayed tokens: 95 originally User, 177 CoT, and 240 Assistant. I checked matching token strings and segment positions across all three conditions. The exact mean on the **177 plotted CoT-source tokens** differs from the summary over all **179 CoT-source tokens**:

| Rendering | Plotted 177-token CoT mean | Full 179-token CoT mean |
| --- | ---: | ---: |
| Correct tags | 64.329429% | 64.025820% |
| No tags | 71.776112% | 71.043557% |
| Everything in User tags | 71.715985% | 70.974987% |

If the prose quotes 64.03%/71.04%/70.97%, I would explicitly call them full-span summaries rather than means of the displayed points. The full native matrix and serialized plotted probabilities differ by at most 7.99 × 10⁻⁸ probability units, consistent with the saved decimal/rounding path; this is immaterial to these percentages.

The visible distinction between the first and second CoT passages is adverse evidence worth retaining. I would not say that removing tags leaves the readout unchanged: it remains visibly associated with the original CoT passages, but its means and individual points change. The quantitative mismatch with the paper remains unresolved. Published CoT percentages should be attributed to the paper, with its no-tag 82%/83% discrepancy and unestablished exact denominator retained if making a numerical comparison.

Sources: [displayed rows](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-e-L12/figure-7-displayed-rows.csv>), [all subset summaries](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-e-L12/mean-role-probabilities.csv>), and [published comparison with denominator caveat](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/appendix-e-L12/published-comparison.csv>).

## Figures 01–03: exact arithmetic, with a different classifier

These figures use only the **everything-in-User-tags** condition and switch to the five-role **`sucat` layer-12** classifier, which includes Tool. They rescore saved float16 states and float32 weights using float64 local arithmetic. The unmodified offset panels are therefore not the same readout as Figure 00: adding a fifth class entails a separately fitted classifier, not merely adding a Tool column to the four-role probabilities.

I independently reproduced all **30,720 probability values** in the 6,144 source rows from the original arrays; the maximum difference was **1.39 × 10⁻¹⁴**. All 12 combinations of edited role, magnitude, and sign retain the identical 512 token IDs. All 30 dose-curve group means match the current output CSV exactly in this recheck. The duplicate zero-offset baselines match before the renderer retains only one zero value per edited role.

The operation is:

`h' = h + sign × fraction × median_reference_norm × w_role / ||w_role||`.

The reference median is **45.24711366617226**, calculated over **all 921 forwarded tokens** of that condition, including tokens outside the displayed subset. The two nonzero offset norms are **0.4524711366617226** and **2.262355683308613**. “1%” and “5%” are fractions of this reference norm, not percentage-point changes in probability, percentages of every individual token's norm, or doses already validated inside the running model. “Subtract” is an opposite constant offset, not projection removal or erasure of a role.

Useful whole-display checkpoints, weighted over the same 512 tokens, are:

| Applied direction / measured score | −5% | −1% | Zero | +1% | +5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| User row / mean Userness | 0.062701% | 9.883664% | 15.476215% | 22.495416% | 81.343460% |
| Tool row / mean Toolness | 0.002231% | 1.508833% | 4.140744% | 10.001244% | 73.783243% |

Figure 03 instead shows a separate token mean for each source-role group, with group sizes **95/177/240**. I would not average its three colored means equally and call that the 512-token mean. Lines join five evaluated strengths; no intermediate strengths were evaluated. The corrected title accurately says the score pattern changes. A claim that larger offsets always reduce separation is false for these panels: under +Tool, the range of source-group mean Toolness rises from 10.563 to 22.063 percentage points at +5%.

**The classifier-row ambiguity matters.** Adding the same vector to every softmax coefficient row leaves unmodified predictions unchanged, but changes an individual raw row used as an offset. I therefore describe these as offsets along the saved classifier rows, not unique or independently validated “User” and “Tool” mechanisms. The difference `w_user − w_tool` is invariant to that common row shift and controls the tokenwise log User/Tool ratio. That identity does not turn these single-row illustrations into propagated model interventions. The response of a probe to its own coefficient directions is partly specified by the linear/softmax calculation itself.

Sources: [arithmetic source and formula](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/presentation/render_probe_offsets.py>), [every illustrated probability](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/presentation/probe-offset-illustration/plotted-probabilities.csv>), [offset provenance](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/presentation/probe-offset-illustration/provenance.json>), and [dose means](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/offset-dose-response.csv>).

## Figure 04: permission contrasts, not behavioral compliance

The plotted subset is **marker permission**, not all RH6 hypotheses: 24 plotted metric × probe × order × layer cells. It contains 12 relative-score settings and the corresponding 12 absolute-Userness settings. All relative-score lower confidence bounds exceed zero. The figure compares the marker command with its replacement span using:

`(command permission − command neutral) − (null permission − null neutral)`.

The relative metric is **`ln(mean span P_user) − ln(mean span P_tool)`**, calculated for each item before taking the paired contrast and averaging across units. It is neither a mean tokenwise log ratio nor a log ratio computed after pooling all templates. The top panel is in **percentage points**; the bottom panel is in **natural-log units**.

I recomputed all 24 plotted point estimates and all their confidence bounds directly from the saved per-item span means, using the actual seed-123, 2,000-resample paired bootstrap. They match exactly. Recomputing the relative-score formula differs by at most 8.88 × 10⁻¹⁶. I separately matched all 72 plotted mean/bound values to their source-table values and scaling exactly.

Layer-12 numerical examples for the draft:

| Probe / marker position | Absolute Userness contrast, points | Relative User/Tool contrast, natural-log units |
| --- | ---: | ---: |
| SUCAT / first | −1.3948 [−2.0264, −0.7789] | +0.7663 [+0.7029, +0.8302] |
| SUCAT / second | −0.9065 [−1.3911, −0.4436] | +0.6187 [+0.5568, +0.6782] |
| UAT / first | +4.4576 [+3.4231, +5.4744] | +0.4788 [+0.4252, +0.5358] |
| UAT / second | +7.7227 [+6.9322, +8.4718] | +0.4941 [+0.4427, +0.5471] |

I verified **1,200 distinct prefills, 100 template–page units, and 12 input conditions**. Reusing the same 100 units across layers, probes, orders, and metrics does not create more independent units. The intervals are pointwise percentile intervals over those units, not simultaneous or multiplicity-adjusted intervals. Each template has one associated page; they are not separately crossed factors. The construction and token positions change with the trusted cue; the lexical-overlap calculation is not an adjusted causal analysis. Null-length matching is approximate (both spans match for 47/100 units). I would say that the relative readout responds to these permission cues, while the absolute Userness sign depends on configuration. I would not infer behavioral compliance, a hierarchy-enforcement layer, or removal of the lexical/position alternatives.

The full source also contains prohibition results, but Figure 04 does not show them. If the draft discusses them, it should say that ten of twelve targeted prohibition-relative intervals include zero, not that prohibition has no effect.

Sources: [plotted grid](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/rh6-plotted.csv>), [plot specification and provenance](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/rh6-provenance.json>), [per-item span means used in my recomputation](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/rh6/per_item_span_means.csv>), and [original analysis implementation](</Users/hananather/Desktop/MATS 12.0/replication/rh/rh6_readings.py:354>).

## Figure 05: five-role recall on two fixed splits

This figure uses **SUCAT**, not gardening's SUCA. Each curve is the fraction of held-out tokens of a given true role classified correctly among five roles. It is **per-role recall**, not a mean predicted probability, precision, or overall accuracy. I independently reproduced all 240 plotted values and every integer numerator and denominator exactly.

The prompt split has **124 held-out role variants** from a corpus of 1,245 variants of 249 base texts, with 67,727 held-out content tokens. The base split holds out **24 base documents**, with 56,685 content tokens. Those are different training/evaluation partitions. The trainer saves exact IDs and raises an error if a space/split partition changes between layers; this full run completed with those checks. The base texts are the independent source units, and the recall values are token-weighted. I would not attach independent-token intervals or treat the difference between panels as an isolated causal effect of leakage.

At layer 12 the actual five-role checkpoints are:

| Role | Prompt correct / total | Prompt recall | Base correct / total | Base recall |
| --- | ---: | ---: | ---: | ---: |
| System | 4,076 / 9,414 | 43.2972% | 6,391 / 11,337 | 56.3729% |
| User | 9,853 / 12,846 | 76.7009% | 8,297 / 11,337 | 73.1851% |
| CoT | 6,360 / 13,765 | 46.2041% | 6,267 / 11,337 | 55.2792% |
| Assistant | 12,416 / 14,793 | 83.9316% | 8,883 / 11,337 | 78.3541% |
| Tool | 7,577 / 16,909 | 44.8105% | 7,252 / 11,337 | 63.9675% |

The earlier four-role layer-12 CoT recall of 61.3864% describes SUCA and must not appear as the value for this figure. The weak-to-moderate five-role recall for several layer-12 classes is a material qualification when interpreting offset and RH6 scores. Native structural checks do not establish probability calibration or solver convergence; line-search warnings remain documented.

Sources: [plotted recall table](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/data/probe-recall-by-layer.csv>), [prompt confusion counts](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/acc_by_role_gptoss-20b.csv>), [base confusion counts](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/acc_by_role_gptoss-20b-basesplit.csv>), and [split audit](</Users/hananather/Desktop/MATS 12.0/replication/cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z/probes-full/split-audit.json>).

## Current file identities and checks

All **19 current source references** in the renderer, RH6, and offset provenance records matched their stored SHA-256 hashes. The first three library PNG/PDF pairs are byte-identical copies of their measured/illustrated presentation originals. I visually inspected all six PNGs: the current dose title is corrected, the denominators and units described above match the visible panels, and I found no material clipping or ordering defect. Figure 00 itself does not state n=1 or the probe class count, so its blog caption must do so.

| Figure PNG | SHA-256 |
| --- | --- |
| `00-gardening.png` | `ea97ced82bb08fccfcdaf4b6a19d037a6dfa0defae81ccf7cfb17b7d4f5be163` |
| `01-small-offsets.png` | `a52e7495b52eec418093e7a1981d35921abf8ad1fa0894af850f01142e5c011e` |
| `02-large-offsets.png` | `8addc84af53fbeded0378612a304e09320191c1f0bbc79735f44dc0258aef83c` |
| `03-offset-dose-response.png` | `dbd66aa09cfd4119592b9e626be945b6fa7542a7cf0baf914f1588cf21fa1a2c` |
| `04-rh6-readout.png` | `52d060c0cb35333c7a65a7ed44aa21af089844ff9fce079498b27cc723d6494b` |
| `05-probe-recall.png` | `7a7d88510c41bc389fe8429f61ff4a1edbf602ac5f6ed31df00d686b72e916d1` |

| Plotted CSV | SHA-256 |
| --- | --- |
| `offset-dose-response.csv` | `583c2a9666641a388c79b3075b855cbdfff59188e4d0e9a61745354a8a6e2523` |
| `probe-recall-by-layer.csv` | `ba9cafc6c2d1e2b420fa4dc8579541076f0a336b7b011284a47de4e12c0acc83` |
| `rh6-plotted.csv` | `5d2c23858b4de37df4862d9319066d31abe7e69c171dcf3b98c9eac24590088f` |

## My review of the actual blog draft

I read [the complete draft](</Users/hananather/Desktop/MATS 12.0/replication/chart-library/role-signals-blog-draft.md>) at SHA-256 **`a3aa84eadf4ea1ca2b88e53b0e5de8663636dce73cab49a9931387cc47e34f17`**. My disposition is **pass on the local numerical and statistical claims, with two precision edits recommended below**. I found no claim-reversing numerical or interpretation defect requiring a change of narrative or method. External paper/source attribution receives its separate source review.

I checked every quoted local number against the current measurements: 384 probes/24 layers/eight spaces/two splits/249 texts; the gardening table and its 177-versus-179 denominators; the 921-token norm reference and 512-token offset cohort; the 95/177/240 source-role groups; the 1% passage-score changes; the +5% passage means; all six percentages in the Tool-offset comparison table; the RH6 100-pair/1,200-prefill/12-condition description, four quoted layer-12 point estimates and interval bounds; the five-role layer-12 recalls; and the historical 17/21 episode/eight-case context. Their displayed rounding agrees with the source tables. The 200-conversation quantity is explicitly a future planning floor, not an achieved sample or an adequacy result.

I also checked the substantive measurement distinctions. The draft states the four-to-five-role classifier change, compares offsets to their own baselines, distinguishes probability from recall, keeps the two splits descriptive, explains the span-mean relative metric before paired contrasts, limits bootstrap interpretation to the sampled pairs, and describes the row-parameterization ambiguity correctly. Its propagation and behavior claims remain future-facing. The main text does not treat the historical small run as an estimated general steering effect.

My two recommended edits are narrow:

1. Replace **“100.0% after rounding”** with **“about 99.95%”** for the original User passages under +5% User offset. The exact mean is approximately 99.95315%, so the current one-decimal rounding is mathematically correct; the replacement avoids suggesting every token reached a probability of exactly one.
2. In the TLDR, specify **“marker permission increases the marker's relative User/Tool score across all 12 plotted settings”**. This makes the plotted target and grid explicit rather than leaving “every tested configuration” to cover unspecified commands or other RH6 contrasts.

I left all draft edits to the root writer. These recommendations do not change the experiment, estimand, or quantitative conclusion.

## Final wording check

I reread the complete revised draft at SHA-256 **`2f66c2d3c66eb2b2d953b703a7f4187c80339eb1367a1203ef87896e0e26989d`**. Both precision edits are incorporated: the large User offset is reported as 99.95%, and the TLDR explicitly identifies the marker command's null-adjusted permission contrast across the 12 plotted settings. The revised gardening-format description and expanded methods note preserve the measured conditions and arithmetic. My final disposition is **pass on the local numerical and statistical claims, with no outstanding numerical correction**. The six figures, their denominators, and the distinction between local rescoring, model prefills, and future propagated interventions remain consistent with the saved evidence. This review does not establish optimizer convergence, probability calibration, or a behavioral intervention effect; the draft correctly retains those boundaries.
