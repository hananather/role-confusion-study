# Figure accessibility review

**Current status: PASS — no unresolved material accessibility finding.** I reinspected the updated Figure 2 and Figure 3 PNGs at full viewing resolution and 800 pixels wide on September 12, 2026. Figure 2 now uses pink triangles for uploads, green circles for no upload, and gray crosses for unresolved episodes. The shapes remain distinguishable at page width, the legend matches, and the report paragraph describes the same encoding. Figure 3 now states in the exported PNG that its 21 pairs cover eight cases and that repeated-context lines overlap. Neither update introduces clipping or text overlap. The original palette and Termes typography are preserved.

I verified these PNG SHA-256 hashes: Figure 2 `6440c181cf0f18eb7de023950ae07698d97e994466984103e96f624a82752e4c`; Figure 3 `3b053896bbc1e16e2ac741f692d3f87c613fe20c17b164642ce8ea4ccac0c875`. The optional denominator and page-order refinements below do not block delivery.

## Original review, retained as history

**Initial status: one material accessibility correction before delivery.** I inspected all three then-current PNGs at full viewing resolution and at 800 pixels wide, then checked `render_figures.py`, the report captions, the plotted CSVs, and the authors' plotting sources. I did not run a model or change a figure or the report. The findings below describe that earlier rendering; the current status above records their resolution.

## Material recommendation

**Figure 2: encode upload outcome with marker shape as well as color.** Uploaded and nonuploaded episodes currently use the same filled circle, so the central comparison depends on distinguishing pink from green. I would keep the exact colors and use a triangle for uploaded, a circle for no upload, and the existing cross for unresolved. The legend and report text should describe both shape and color. This is the only change I consider necessary for accessibility: Figure 1 already duplicates color with `U`, `–`, `?`, `·`, and `/`, and Figure 3 labels its outcomes in separate panels.

## Reading and layout

- **No clipping or title/subtitle overlap is visible in the three inspected PNGs.** At 800 pixels wide, the large title and the highlighted original-vector row identify the main point quickly. Figure 2's subtitle and panel A title are close, but they do not collide; a few more pixels of separation would be a minor refinement.
- Figure 1's matrix remains readable at 800 pixels wide. Its bottom takeaway makes the paired result explicit. All missing-cell types are distinguishable without relying on color. The unresolved count is smaller than the main labels but still readable.
- Figure 2 keeps five vertically separated symbols in every populated row, including scores near zero. The data do not vanish through overplotting. Its empty new-page rows explicitly distinguish `Not assigned` from `Unrun`.
- Figure 3's count labels and separate axes make 18→17 unwanted writes and 21→19 authorized actions immediately visible. The two action panels remain interpretable without their colors.

## Data representation

- Figure 1 represents 90 arm/page positions: **80 recorded episodes, five unrun assignments, and five unassigned zero-dose positions**. The recorded cells contain 35 uploads, 43 no-upload episodes, and two unresolved episodes. I found no omitted arm/page outcome in the renderer's traversal.
- Figure 2 represents **all 60 saved payload readouts**: 24 uploads, 34 no-upload episodes, and two unresolved episodes. Each populated arm/cohort group contains five records, and the renderer gives each record a separate vertical position. Page order is explained in Methods; adding “top to bottom: pages 1–5” to the caption would improve standalone traceability without crowding the panels.
- Figure 3 processes **all 21 matched pairs**, with before/after action counts agreeing with the CSV. Those pairs have **15 distinct exact pairs of plotted probe values**. Repeated contexts therefore overlap exactly in panel A. The report explains this, but the exported PNG does not. If the PNG must stand alone, I recommend adding “21 pairs; repeated contexts overlap” to its subtitle or caption. I would preserve exact score coordinates rather than introduce unexplained jitter.
- Figure 1's `Verified uploads` fractions include unresolved records in the denominator. The nearby `2 unresolved` note and report explanation preserve the uncertainty. “Uploads / recorded” would make the denominator explicit for readers who encounter only the figure.

## Paper palette and typography

I verified the chart's pink `#ff637e`, green `#00d492`, and Tool purple `#7e6cff` against the authors' `experiments/cot-forgery-role-confusion/05-analyze-agent-probe-result.ipynb`. The authors' `r-utils/plots.r` specifies TeX Gyre Termes, and the local renderer registers and selects that font. The actual PNG lettering is consistent with that setting. Figures 1–2 deliberately reuse these colors for outcomes, and their legends identify the changed meaning. The requested shape correction preserves every color value.

The SVG/PDF assets and the caption-linked CSVs provide scalable and numeric alternatives. I did not inspect a browser-rendered page or independently rerender the PDFs in this bounded review; my page-width assessment used the actual PNGs scaled to 800 pixels.
