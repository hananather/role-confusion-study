# My figure-library verification

I verified this library on 2026-09-11 using saved local data. I ran no model, generation job, or cloud computation.

| Check | Result |
| --- | --- |
| History scope | September 10 in Toronto: 34 human inputs, all 19 Claude task states, 35 linked subagent logs. The reviewer distinguishes actual human queued messages from task notifications and teammate reports. Main task snapshot and source timestamps were spot-checked. |
| Offset data | All 12 source conditions contain the same 512 token IDs: 95 User, 177 CoT, 240 Assistant. Both stored zero offsets are identical. |
| Independent offset recomputation | All 30,720 source probabilities were recomputed from the saved activations and classifier weights; maximum discrepancy 1.39e-14. All 30 plotted passage means agree within 1.11e-16. |
| Probe recall | All 240 plotted cells, including integer correct and total counts, match the original confusion tables. The held-out counts are 124 prompt variants or 24 base texts for the five-role classifier. |
| RH6 | The source hash matches the completed 44-check audit. All 24 displayed metric/contrast cells round-trip exactly; pointwise intervals and the 100-pair denominator are preserved. |
| Styling | Source-role colors, exact TeX Gyre Termes font, boxed panels, light grids and un-smoothed values are retained. The scalar RH6 control uses neutral gray. PNG and vector PDF outputs are present for all six figures. |
| Narrative review | I replaced the initial dose-title claim of general separation loss with the narrower observed claim: direction and size change the pattern. Large User offsets reduce Userness separation here; this is not a universal effect of every offset. |
| Visual inspection | The dose, probe and RH6 renders were inspected; the original small/large and gardening figures retain the verified render files. I checked the browsable library and both selection controls in the in-app browser. |
| UI behavior | Selecting large offsets changes the image, caption and download links; selecting probe validation replaces the RH6 panel. Both states were observed. |
| Evidence boundary | Every illustration states its one-conversation sample. Future results remain backlog; offline edits, actual model readouts and historical behavioral records are identified separately. |
| Source preservation | Original figure and numerical source hashes in the render provenance were rechecked. The original report, data and source figures remain unchanged. |

The [history review](review/claude-goals-audit.md) and [independent evidence review](review/evidence-audit.md) retain methods, numerical checkpoints, adverse findings and source paths. These checks do not resolve the outstanding optimizer-convergence question, the paper/code Figure 23 discrepancy, or the need for adequately sized future behavioral experiments.

## Blog review draft v0.1 — 2026-09-11

I prepared [the Markdown review draft](role-signals-blog-draft.md) and its [reading preview](blog-draft.html) from the existing library. The draft is awaiting my review. Its final SHA-256 is `2f66c2d3c66eb2b2d953b703a7f4187c80339eb1367a1203ef87896e0e26989d`.

I used three independent agent reviews: [numerical accuracy](review/blog-numerical-review.md), [paper and source fidelity](review/blog-source-review.md), and [cold-reader clarity](review/blog-reader-review.md). Each review records the exact draft it read. The numerical reviewer reread the final version above and found no outstanding numerical correction. The source and reader reviews evaluated earlier drafts; I incorporated their material corrections in the final version.

The corrections clarify the four-role to five-role probe change, the exact gardening formatting, and the distinction between offset length and probability change. I narrowed the permission result to the marker command's null-adjusted contrast in the 12 plotted settings, explained the two commands and fixed prefills, and reported the large-offset User mean as 99.95%. I also defined prompt injection more precisely and linked the paper's exact figure and appendix sections. These are wording and attribution corrections; I changed no measurements, figures, or experimental methods.

| Final check | Result |
| --- | --- |
| Numerical evidence | The independent review reproduced all 30,720 offset probabilities, 30 dose means, 24 RH6 point estimates and their 48 interval bounds, and 240 recall cells from the saved records. No material numerical defect was found. |
| Figure preservation | All six PNG hashes match the reviewed figures. All 19 numerical/render source references and both paper-style sources match their saved checksums. PNG and PDF files remain present for every chart. |
| Markdown | Six embedded charts with descriptive alt text; all 34 reference definitions resolve, and all 34 local reference/image targets exist. |
| Reading preview | All 63 local link/image occurrences resolve. I inspected the title, TLDR, chart placement and captions in the in-app browser. The preview retains the exact figure files and provides full-resolution image links. |
| Review package | The local ZIP includes the article, all six PNG/PDF chart pairs, and directly cited evidence. Its main article uses checked portable links. Deeper audit links may still refer to the original workspace; this is not a full reproduction environment. |
| Scope | September 10 remains the conversation-audit scope. I ran no model or GPU job while preparing this draft and did not publish or send the package. |

I preserve the unresolved replication gap, optimizer-convergence and calibration questions, and future-only status of propagated steering. A numerical and source review does not establish those scientific claims. The machine-readable final checks are in [blog-integrity.json](data/blog-integrity.json).


## MATS illustration added on 11 September 2026 local time

Six local MLX forwards completed (three MATS formats plus three gardening controls), with no generation or cloud calls. The new Figure 1 retains 137 prespecified matched tokens per format. An independent [audit](data/mats-example/audit.md) recomputed every displayed probability from the saved states and H100 coefficients, verified token identity and role mapping, and retained the runtime discrepancy. The original six figures remain in the report. PNG and vector PDF/SVG were rendered with the paper palette and Termes font; the final image was visually inspected for readable labels and clipping. This update does not complete the report-wide ten-reviewer process.


## Six-passage MATS batch and visual selection

The user requested the original User 1 / CoT 1 / Assistant 1 / User 2 / CoT 2 / Assistant 2 layout and multiple measured candidate examples. I authored and froze 12 conversations, completed all 36 local MLX forwards in 80.7 seconds including loading, retained every result, and selected mats-03 after inspecting all plots for visual clarity. The earlier single-turn result and report snapshot remain preserved. The [batch audit](data/mats-six-passage/audit.md) independently checked 21,456 activation rows, 18,684 plotted points and all 216 passage means. The featured figure contains 516 matched tokens per condition, with original-text excerpts and the paper palette/font. PDF fonts were checked as embedded TrueType. All twelve figures and full conversations appear in the comparison gallery. This selection supports an illustration only; the report makes no representative-sample claim.


## Personal Hanan–MATS exchange, 11 September 2026 Toronto time

I replaced the opening authored illustration with two User questions in my voice about applying to Neel Nanda’s MATS stream and an “expensive mood ring” follow-up. Local GPT-OSS-20B generated both analysis and final-answer pairs. I retained a first attempt that confused chat roles with personas; the featured revision supplies paper context and medium reasoning. All generated text is unchanged, and its methodological errors are qualified beside the full transcript.

The revised run completed two generations and three measurement forwards in 25.31 seconds including loading, with 12.99 GB peak MLX memory. The independent audit reproduced all 3,166 probabilities and 18 displayed passage means, and checked the exact same 578 tokens per panel. A separate visual reviewer verified six genuine text excerpts, the paper palette and embedded Termes fonts. I inspected the full-resolution image and the live report preview with the new figure visible. The six earlier figure images remain unchanged; prior reports, generated attempts and the twelve-example gallery are preserved. This scoped update does not complete the report-wide review process.


## Complete text and professional title trials

I compared six layouts: 35° and 45° wide excerpts, a compact 35° version, full text in six columns, and full text in two rows with equal or balanced widths. The current main figure uses the compact 35° layout at 8.3 × 4.955 inches, with fixed 7.2 pt excerpts and 0.196 inch minimum adjacent-text clearance. It prints complete opening excerpts, including the MATS application sentence and the two-sentence mood-ring question, with no ellipses. The complete dialogue remains in an 8.2 pt unabridged figure and the linked transcript. All versions share the title “Reasoning-role scores across dialogue formats,” the original paper palette, and unchanged 578-point panels. The former short-label figure is archived. No new model run was required for these typography changes.

## 12 September 2026 UTC — canonical MATS example and measured forgery companion

I froze the user-approved MATS conversation as `canonical/mats-dialogue-v1`, including its text, scores, three figure formats and twelve source checksums. The current report links the canonical copy.

For the separate CoT-forgery figure, I selected historical case 002 before running four fixed local conditions. All four finished exposed and uncensored. The ordinary command produced refusal; both forged and destyled reasoning led to verified dummy-file uploads; the identical forged input plus Tool−CoT steering produced a summary without uploading. The steered summary's unsupported final sentence is retained and identified.

The new figure uses actual activation readouts captured during those runs, the paper's palette and Termes font, four stacked panels, a shared token scale, exact input excerpts and a full-text companion. I verified 1,951 plotted token measurements and 247 independent audit checks, including receiver evidence, token alignment, exact central-pair inputs, EWMA and independent probe projection. The four-role readout's missing Tool class is explicit; the five-role diagnostic and matched H100 random successes are reported separately. The H100 adverse case is retained.

I checked the rendered figure and live reading page, fifty direct companion/canonical links, four live HTTP assets, all twelve canonical hashes and 101 scientific measurement hashes. Local inference exited successfully and released its lock and containers. No H100 work was queued or pod lifecycle changed for this illustration. This scoped figure review does not certify completion of the report-wide ten-reviewer goal.

## 12 September 2026 — upload outcomes made explicit

I added a separate outcome diagram with large outlined UPLOADED / NO UPLOAD labels, the paper's pink and green hues, and an explicit success/failure legend. The labels were checked against receiver receipts and all four completed episodes. The baseline refusal and steered answer's unsupported sentence remain visible as task context.

I removed the technical footer from the probe figure and moved its interpretation details from the long caption to the report's methods. The new outcome diagram appears before the curves. I archived the prior presentation, verified all scientific hashes unchanged and all 1,951 plotted measurements preserved, checked PDF/PNG/SVG outputs and page links, and obtained an independent visual/wording pass. No new model measurement was needed.
