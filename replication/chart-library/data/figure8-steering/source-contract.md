# Figure 8 / Figure 25 source contract

Status: source inspection only; no model run, cloud call, or local measurement in this audit.

I am in **Distill**: my north star is one legible, traceable example connecting the attacker’s text, the role-probe readout, and the observed response. This follows the local Neel lens on clarity, inspecting raw examples, and keeping an illustrative success separate from an overall effectiveness claim. I inspected the actual source images and frozen plotting code; this document does not certify that a local steering example satisfies the four conditions below.

## Source identity

The upstream clone is `/Users/hananather/Desktop/MATS 12.0/prompt-injection-as-role-confusion`, verified at `ec333c40fd43fe991e1ebf66765051b6d7e35784`.

- Figure 8 image: `/Users/hananather/Documents/ChatGPT/MATS 12/research/sources/paper-markdown/assets/figure-08.png`.
- Figure 25 image: same directory, `figure-25.png`.
- Paper text: `/Users/hananather/Documents/ChatGPT/MATS 12/research/sources/paper-markdown/prompt-injection-as-role-confusion.md`, lines 528–587 (Figure 8 / §5.1), 1918–1977 (Figure 25 / Appendix H).
- Plot notebook, relative to upstream clone: `experiments/cot-forgery-role-confusion/04-analyze-injection-probe-results.ipynb` (below: **plot notebook**). References below are physical notebook-file line numbers; zero-based cell numbers are also supplied.
- Theme: `r-utils/plots.r`, lines 1–48.

## What Figure 8 actually displays

Figure 8 is **one StrongREJECT chat example**. Attacker-authored reasoning is inside a User message; it is not the tool-output agent benchmark. The source panels are `(a) No CoT Forgery`, `(b) CoT Forgery`, `(c) Destyled CoT Forgery`. The first has User → CoT → Assistant spans; the latter two insert a pink User (CoT Forgery) span after the User query. Colored span labels sit at their starting x positions. Each arm contains its own generated reasoning and answer, so arm lengths and token identities can differ.

The source’s narrative is that the baseline and destyled arms refuse, while the styled forgery succeeds. These are the **authors’ results and interpretation**, not local reproduction evidence. A local caption must be based on the actual saved outcomes.

The frozen code selects `harmful_question_ix = 33`, zero-based **layer 16**, and a four-role probe (`assistant-cot,assistant-final,system,user`), with `qualifier_type == no_qualifier`. See plot notebook lines 934–956, 1277–1307 (cells 24, 27). This differs from Figure 25’s layer 12 setting; I will identify any local layer change explicitly.

### Exact visual settings

| Element | Figure 8 source setting |
| --- | --- |
| Canvas | 3.75 × 4.5 inches, three vertically stacked plots; PDF via Cairo, PNG 300 dpi |
| Font | TeX Gyre Termes, theme base 9 pt |
| Panel title | Bold 8.5 pt, `#45556c`, 2 pt bottom / 7 pt top margin |
| User points/line; text | `#00a6f4`; `#0084d1` |
| Forged CoT points/line; text | `#ff637e`; `#ff637e` |
| Genuine CoT points/line; text | `#fd9a00`; `#e17100` |
| Assistant points/line; text | `#00d492`; `#009966` |
| Points | ggplot size 0.5, alpha 0.9 |
| Lines | ggplot linewidth 0.5, alpha 0.7, grouped by consecutive content segment |
| Y axis | CoTness; 0% and 100%; 2% expansion on both ends |
| X axis | Sequential displayed content-token positions; no numeric ticks; colored labels at segment starts, 8 pt, horizontal, left-aligned |
| Frame / grid | White panel; `grey20` border linewidth 0.5; vertical `grey85` grid linewidth 0.3 at segment boundaries; no horizontal grid |
| Legend / shading | No legend, no uncertainty band, no colored background spans |

These settings are in plot notebook lines 1260–1435 (cell 27), plus theme lines 9–44. The figure’s pink is the source-forgery color, not a new steering color. Figure 8 has no area shading to reproduce. Figure 25 has a pale `#f4f7fb` legend background, not an uncertainty band. Any added highlight for an intervention window should be separately labeled as a local design extension.

**Smoothing is part of the source figure.** Each plotted point and connecting line uses a normalized trailing exponential average of probe probabilities with weights `1, 0.5, 0.25, …`, reset within each prompt/segment/role space. Only the first 200 tokens of each content segment are displayed; System and role-tag tokens are omitted. The x axis concatenates those retained spans. See lines 1279–1307, 1362–1374. Preserve the unsmoothed complete token table and disclose smoothing/caps in the local methods; do not call these raw points or imply the displayed axis is the uninterrupted full transcript.

## Measurement requirements for the local four-arm illustration

The natural adaptation is four stacked panels: **No forgery; CoT forgery; CoT forgery + steering; Destyled forgery**. This keeps the styled and styled-plus-steering comparison adjacent and retains the paper’s ordering. If space calls for three rows, the middle row can be a clearly labeled pair; four panels make differing generated continuations easier to read.

Before I label this a steering success, I need:

1. A matched example ID with the complete exact prompts, forged and destyled text, rendered role/channel tags, generated analysis/final text, model/tokenizer identity, and decoding settings for all arms. For the central pair, everything except the recorded activation intervention is held fixed up to generation. A new user-authored benign task should be called an adaptation, not a StrongREJECT reproduction.
2. Token IDs, original token positions, source-segment offsets, and content-role masks. The forged span has its **actual wrapper role** and a separate source label. A tool-output adaptation must say Tool (CoT Forgery), use the appropriate Tool-aware probe, and retain legitimate Tool spans; it cannot silently relabel tool tokens as User or reuse four-role CoTness as Toolness.
3. Actual activations under each condition, or corresponding verified per-token role probabilities. The authors capture the output of `post_attention_layernorm` immediately before the MLP (`utils/pretrained_models/gptoss.py`, lines 47–62), with zero-based layer indices. The export notebook replays the full saved generation; its output metadata maps saved layers. The projection notebook applies the saved probe to those activations and drops role-tag tokens. A steered generation replayed **without steering** is not a measurement of its steered hidden states.
4. Steering provenance: vector file/hash and construction, direction/sign, coefficient and normalization, layer/hook, token mask, and whether injection acts during prompt prefill, decoding, or both. Readout hook ordering must be explicit. Direct movement of a probe score at the same edited hook does not by itself establish a changed internal mechanism.
5. An outcome read from the saved action/answer: baseline does the intended task, forgery causes the intended wrong action, steering restores the intended task, and the destyled arm’s actual outcome. Label any unmatched pattern honestly. A refusal alone may fail the legitimate task. Include exact action/answer evidence and selection disclosure, with the larger benchmark result nearby when available.

For a single positive illustration, I can choose the clearest verified success as the user requested, while reporting the candidate pool and selection rule. This establishes what occurred in that example; population-level steering efficacy requires the paired benchmark results.

## Figure 25 is an aggregate and needs a different data contract

The paper says Appendix H covers **313 StrongREJECT attacks** (paper lines 1961 onward). Figure 25 uses the four-role probe at zero-based **layer 12** (plot notebook lines 1547–1548, cell 33). It averages the probability at each source-role-relative token position, separately by attack condition, and retains a position only if at least **50 prompts** contribute (lines 1594–1608). Its y ticks are 0%, 50%, 100%; it has no x labels, a role legend below, and thinner lines (0.4). Nominal export size is 7 × 4 inches (commented save calls in cell 34). Show contributors per position and preserve the user’s adequate-sample-size requirement before making a local aggregate.

Two source-code caveats matter for faithful reimplementation:

- `(token_in_seg_ix <= 50 & base_message_type == user) | token_in_seg_ix <= 100` at line 1627 permits **all roles through 100**, including User. It does not enforce a 50-token User cap.
- EWMA grouping at lines 1633–1634 includes segment and role space but omits condition. In that frozen pipeline it can carry smoothing between conditions. A local aggregate should reset by condition and segment, and document this corrective difference rather than silently claiming exact numerical replication.

A four-arm single-example figure should therefore borrow **Figure 8’s visual design**; it should not be described as Figure 25’s aggregate, and it should not display invented error bands.
