# My curated role-signal figures

Start with my [review draft in Markdown](role-signals-blog-draft.md), which includes a TLDR, seven charts, explanations and citations. Its [reading preview](blog-draft.html) is generated from the Markdown. A [portable review bundle](role-signals-review-bundle.zip) contains the draft, all chart files and directly cited evidence.

My approved reusable conversation is [MATS dialogue v1](canonical/mats-dialogue-v1/README.md). Its frozen text, measured scores, figure files and checksums are preserved together. The [canonical-example register](CANONICAL-EXAMPLES.md) defines how I reuse it in later work.

I use this collection to follow one evidence chain: **read the role pattern → change the offset → inspect the dose response → check the measurement → plan the next model experiment**.

The separate [CoT-forgery steering study](cot-forgery-steering.html) adds a measured behavioral example and the complete attack text.

[Open the browsable library](index.html). All figures have PNG and vector PDF versions; new summaries have plotted CSVs and source checksums.

| Step | Figure | What it contributes | Independent sample / evidence type |
| --- | --- | --- | --- |
| 1 | [Gardening](figures/00-gardening.png) | The same conversation retains visible CoT-style patterns under three tag conditions. | One conversation; measured H100 activations; 512 matched display tokens per condition. |
| 2 | [Small offsets](figures/01-small-offsets.png) and [large offsets](figures/02-large-offsets.png) | The same passage colors make local score changes visible. | The same one conversation; arithmetic on saved activations; no model propagation. |
| 3 | [Dose response](figures/03-offset-dose-response.png) | Each direction changes both readouts. Small and large offsets can have different effects on the passage means. | Five evaluated signed offsets on the same 512 tokens. No independent replication or population interval. |
| 4a | [Context and metric](figures/04-rh6-readout.png) | The relative User/Tool cue effect is positive across configurations; absolute Userness changes sign with probe choice at layer 12. | 100 paired template–page inputs; measured prefills; pointwise intervals; no action outcomes. |
| 4b | [Probe validation](figures/05-probe-recall.png) | Reliability varies across roles and layers, with a dip around layer 12. | 249 base texts in the full corpus; five-role test sets contain 124 prompt variants or 24 grouped base texts. |

I am in **Understand / Distill**. My north star is to separate a visible role signal, a direct classifier response, and evidence of downstream model change. I apply the Neel lens by making every figure answer a specific question, checking the independent units, retaining adverse evidence, and resisting a mechanism claim from a probe plot.

## Scope of my history review

I reviewed the September 10, 2026 Claude conversation in Toronto local time, including its continuations and linked subagents. The primary session is `3288d75d-aecb-4598-8e1e-657b5a5f7a14`; the session spans 16:26–23:18 Toronto and the substantive discussion begins at 16:35. The review covers 34 human inputs, 19 tracked tasks and 35 linked subagent logs. I use older experiment files only where that conversation refers to them.

- [Conversation, goals and provenance](review/claude-goals-audit.md)
- [Independent evidence and figure audit](review/evidence-audit.md)
- [Unfinished questions and sample requirements](BACKLOG.md)
- [Figure 23 deferred batch](../e9/BACKLOG.md)
- [Verification log](verification-log.md)

## My curation decisions

I keep the paper's blue User, yellow/orange CoT and green Assistant passage colors, exact TeX Gyre Termes font, raw points, boxed panels and light grid. I retain gray System and purple Tool only where those roles are actually plotted. The RH6 companion uses the paper's neutral gray and line styles for its two orders; it has no CoT or Assistant passage series to color.

I leave the earlier single-case saturated steering plot in supporting evidence. Its intervention site, probe and source text differ from this gardening comparison. Its flat Tool-only traces are less useful for the present visual question. I do not recolor Tool tokens as three nonexistent passage roles.

I retain earlier adverse steering outcomes and Appendix K's positional controls in the evidence audit. I exclude synthetic fixtures and incomplete smoke outputs from the results gallery. I keep the original report, figures and measurements unchanged.

## Interpretation boundaries

The gardening source is the authors' conversation, scored by my model run; its numerical values do not exactly reproduce the paper. The offset views use the saved five-role probe, while the gardening overview uses the original four-role probe. Raw individual classifier rows depend on the saved softmax representation. Adding or subtracting those rows illustrates classifier geometry; it does not establish a unique semantic direction, erase a role, or measure an effect on model behavior.

I interpret token-level curves descriptively. The 512 tokens are not 512 independent experiments. The future batch must justify its sample using independent conversations or other design-appropriate units, and preserve complete paired controls.
