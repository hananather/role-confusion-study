# Conversion and fidelity notes

This Markdown derivative was created on 4 September 2026 from the user-supplied *Prompt Injection as Role Confusion*, arXiv:2603.12277v6, dated 27 June 2026. It preserves the full paper rather than summarizing its arguments.

## Source identity

- Supplied source: `/Users/hananather/Desktop/PromptInjectionRoleConfusion.pdf`.
- SHA-256: `609332fdfaae35838e5c82c0ef9a9b7d23aac7ac24000452917c47a56d4f29e0`.
- 33 pages; original PDF metadata names Charles Ye, Jasmine Cui, and Dylan Hadfield-Menell.
- At the initial check, the source was byte-identical to `research/sources/prompt-injection-as-role-confusion-v6.pdf`. That duplicate was absent at final integration. The supplied PDF and the [Desktop archive](</Users/hananather/Desktop/MATS 12.0/01-sources/papers/pdf/ye-2026-prompt-injection-role-confusion.pdf>) still match the recorded hash. Reading links use these verified sources.

## What the derivative preserves

The document contains the full main text, abstract, headings, acknowledgements, impact statement, references, appendices A–L, all 32 figures and their captions, four numbered tables (including both parts of Table 3), and source footnotes 1–24. Each page has an anchor and a link to its rendered original. Figures are reproduced from the source and accompanied by literal, searchable spatial text extractions. Prompt and code examples remain source quotations.

Two-column prose was placed in reading order. Footnotes were moved to labelled blocks at the end of their source page so they do not interrupt sentences spanning columns. Paragraph boundaries were reconstructed from the PDF’s positioned text blocks. Tables also have native Markdown versions. The mathematical expressions on pages 5 and 26 were transcribed into Markdown math. The PDF’s external hyperlink targets were used to restore line-wrapped reference URLs.

## Verification and limits

All 33 page anchors and page images and all 32 figure images are present. A geometric audit found no source word boxes outside the conversion regions within the paper’s content area; see [the page coverage metadata](extraction-metadata.json) for the precise boundary. All 24 numbered footnotes are present. Page contact sheets were inspected, along with representative original page images. An independent review compared all four tables and the principal equations against the source renders.

This is a readable PDF derivative, not a certified character-for-character transcription. PDF extraction can alter spacing, ligatures, superscripts, subscripts, and the ordering of labels inside plots. Line-wrap hyphens and small-cap typography were normalized where identified; the original layout remains in the source text extraction and page images. Figure text extractions retain their spatial layout and can be awkward to read independently of the image. Color, curves, and other graphical information remain in the reproduced figures. Markdown renderers differ in their support for math and expandable text blocks; the page links work independently of those features.

Source claims, numerical results, typos, and speculative statements were preserved as the authors’ material; they were not endorsed or described as local replications. No notebook, model, API call, prompt, or attack example from the paper was executed.

## Reproducibility

- `build_markdown.py`: local text extraction and deterministic layout reconstruction using Poppler, Pillow, and pypdf.
- `verify_conversion.py`: source identity and geometric page-coverage checks.
- `conversion-regions.json`: the source-coordinate regions used for prose, figures, examples, tables, and footnotes.
- `hyphen-normalization.json`: the line-wrap hyphen decisions made during prose cleanup.
- `source-text-by-page.json`: the full Poppler layout extraction, including original page furniture.
- `source-hyperlinks.json`: external PDF annotation targets and recovered Markdown links.

The Markdown and image links are portable together. The original-PDF link intentionally points to the user-supplied absolute source path.
