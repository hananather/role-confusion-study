# My delivery, accessibility and provenance review

**PASS — I found no unresolved delivery defect after checking the rebuilt package.** The live report and portable report pass the direct-reference checks below. I made no changes to the report, figures, package or experiment records.

## Resolved correction

**Resolved P2 — The portable Markdown report previously sent readers to an HTML copy whose figures did not load.** Its “offline offset plots” link in the Supporting illustrations paragraph now points to `../role-signals-blog-draft.md`, matching the portable HTML and reaching the portable earlier article. I verified that the broken relocated HTML copy was removed and the rebuilt ZIP contains no nested ZIP file.

The portable HTML article now matches the rendered portable Markdown exactly. The live HTML article also matches its rendered Markdown exactly.

## Checks that passed

- I checked all 49 HTML references in each main report, including section anchors, image sources, resources, figure downloads and three local font files. Every direct target exists; every section anchor resolves. The package replaces its recursive ZIP link with working package notes.
- I checked all 33 Markdown references in each main report and all local links in its directly cited numerical, historical and statistical audits: 4, 23 and 5 references respectively. All destination files exist. All portable destinations remain inside the portable folder.
- Each HTML report declares English, UTF-8 and a responsive viewport; it has one H1, ordered H2/H3 headings, unique IDs, a keyboard-visible skip link, labelled navigation and three nonempty descriptive image alternatives. The main text uses a 20 px font with 1.55 line height, reducing to a separately specified mobile layout. The three referenced TeX Gyre Termes OpenType font files are present in the live library and portable package.
- All three PNGs decode, have substantial nonuniform content, and have expected large dimensions: 2944 × 2112, 2944 × 2080 and 2944 × 1600. The main report links each to an available full-size image, PDF, SVG and relevant data table.
- All nine figure artifact hashes, their recorded input-table hashes and the figure-renderer hash match `figures/render-provenance.json` in both the live report and the portable package.
- Every one of the 534 manifest-listed files in the checked rebuild has the recorded SHA-256 and byte count. Four package wrapper files are outside this rebuild's `files` list: `README.md`, `index.html`, `manifest.json` and `role-signals-blog-draft.md`. The top-level `draft_sha256` correctly identifies the original live earlier-article source before portable link rewriting; it is not a checksum of the rewritten article.
- The ZIP integrity check reports no corrupt member. All 538 ZIP file contents in the checked rebuild match the portable directory byte-for-byte, with no extra member and no nested ZIP. The review ZIP retains one enclosing `role-signals-review-v0.2` folder. A subsequent package rebuild is needed to include this updated review record and final review-status metadata.

## Scope

I used static parsing, Markdown rendering, file decoding and hash checks. I did not run a model, call an external service, launch experiment code, or claim a screen-reader or browser visual test. I treated preserved deeper external source-document links as outside this delivery check. Other reviewers cover the charts' visual readability and scientific accuracy.
