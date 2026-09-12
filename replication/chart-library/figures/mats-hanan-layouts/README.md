# Text-layout candidates

I rendered these alternatives from the same 578 measured points per panel. I have not replaced the current report figure. Every version uses the title **Reasoning-role scores across dialogue formats**, the paper's role colors, and embedded TeX Gyre Termes fonts.

| Candidate | Page size | Text size | Text included |
|---|---|---|---|
| [35° diagonal](diagonal-35/diagonal-35.pdf) | 11.7 × 8.3 in | 8.5 pt | Complete opening sentences or title; both User 2 sentences through the mood-ring question |
| [45° diagonal](diagonal-45/diagonal-45.pdf) | 11.7 × 8.3 in | 8.5 pt | Same complete excerpts as the 35° version |
| [Six complete columns](six-full-columns/six-full-columns.pdf) | 11.7 × 12 in | 8.3 pt | Every word from all six passages; Markdown emphasis markers removed for presentation |
| [Compact 35° diagonal](diagonal-35-compact/diagonal-35-compact.pdf) | 8.3 × 4.955 in | 7.2 pt | Exactly the same complete excerpts as the wider 35° version |

I find the 35° version easier to scan than the 45° version. Both diagonal versions retain complete sentences without ellipses, but they remain explicitly disclosed excerpts. Six complete columns preserve the entire exchange at the cost of a tall page and narrow text columns. These physical sizes matter: shrinking any version to 6.75 inches wide makes the text too small for a main paper figure.

The compact alternative is intended for native 8.3-inch page width. I optimized each block's wrapping at a fixed 7.2-point font, retained every excerpt word, and cropped 1.045 inches of bottom whitespace using the measured text bounds. Its footer explicitly says that the full conversation is provided separately. Earlier alternatives remain unchanged.

All three exports passed exact scatter-point comparison against the CSV, page-boundary checks, and font-embedding inspection. Full transcript wrapping preserves every word. Adjacent transcript blocks have at least 0.167 inches of horizontal clearance. The individual `layout-provenance.json` files record exact text extents, input and renderer hashes, and artifact hashes.

The assistant passages are unchanged model output apart from displayed Markdown markers. They contain scientific overclaims and are examples to be scored, not accepted methodological conclusions.
