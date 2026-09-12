# My complete steering evidence review

[Read the report](index.html) · [Markdown](report.md) · [Working goal](GOAL.md) · [Independent reviews](REVIEWS.md)

My three main figures follow one argument:

1. [Every controlled forgery-page outcome](figures/01-complete-outcomes.png): the original vector has mixed effects and leaves the new-page outcomes unchanged.
2. [Every available forged-passage probe readout](figures/02-probe-behavior.png): near-zero reasoning-role scores coexist with successful uploads; the favorable Tool-raising branch remains visible.
3. [All recovered Toolward permission pairs](figures/03-permission-task.png): saturated Tool scores coexist with unwanted writes and lost authorized actions in a different task.

Each figure has PDF and SVG counterparts. The data directory contains the exact plotted rows and their provenance; the reviews directory contains independent reductions and coverage audits.

I freeze the corrected `20260912T051758Z` CUDA snapshot and keep the historical datasets separate. Preparation and rendering use saved files only; no model job or cloud-resource change is part of this work.

For reproducibility, `prepare_evidence.py` validates the source-table checksum, reads all assigned rows and actual episode records, and recomputes outcomes and probe joins. `render_figures.py` consumes those data, uses the paper's palette and Termes fonts, and records its output hashes. `render_report.py` builds the reading page directly from `report.md`.

The [review bundle](review-bundle.zip) is a portable reading package with the report, figure exports, plotted data, audits and source-access notes. It is not a model reproduction environment.
