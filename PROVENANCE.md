# My review export

I assembled the experiment files from the frozen `desktop/` tree in the separate
private [mats-role-confusion-archive](https://github.com/hananather/mats-role-confusion-archive)
repository. I expanded this initial package to cover every experiment family
listed in [the catalogue](EXPERIMENTS.md). I also captured the later saved
steering report, plotted data and figures from the execution workspace; their
source paths and hashes are recorded separately. The
[export manifest](provenance/export-manifest.json) maps each copied file to its
archive path, byte count and SHA-256. All copied source, scientific fixtures and
results retain their source bytes. The root [report](REPORT.md) is a reading copy
of that later report with links adapted for this repository. I retain its
scientific text and the original report.

The scientific cutoff is
`replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z`.
Its original [artifact manifest](replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z/artifact-manifest.json)
indexes 2,224 local evidence files. I also include the exact scientific source
packets, fixture HTML, role-probe/direction arrays, original analysis inputs and
audit receipts. Those are necessary because the raw-evidence manifest itself
contains no executable Python source or NumPy assets.

## Paths and historical files

I preserve the relative `replication/` package layout. Historical absolute paths
remain inside frozen records. My [saved-results verifier](scripts/verify_results.py)
maps the exact original workspace prefix to the clone root and refuses paths
outside that root. It does not fall back to files in the original workspace.
The [export verifier](scripts/verify_export.py) checks the copied and newly
prepared files against this export's separate manifest.

The [historical combined exporter](replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/build_combined_closeout.py)
is retained unchanged for inspection. It reads absolute paths and creates a new
timestamped snapshot, so it is not the portable verification entry point.
Historical READMEs and launch records describe the state at their own dates;
they are not current instructions to allocate compute or continue experiments.
Some links in those unchanged historical documents lead into the full archive.

## Included and omitted work

I include the completed September 11 local baseline and steering evidence, the
September 12 CUDA queue, its controls and unrun assignments, and the separate
[earlier transfer and permission experiments](earlier-study/README.md). I retain the full frozen
77-page carrier bank for input-selection provenance; only the registered prefix
was used in the completed new-page comparison.

The expanded package also includes probes, gardening, Appendix K, RH6, the
313-trajectory unsteered chat baseline, permission-study instrumentation,
the role-uptake pilot, illustration experiments and the prepared experimental
battery. I keep their individual completion states in the catalogue; they are
separate from the September 12 agent cohort. I include the pinned upstream
source and earlier reference implementations with their own attribution.

I leave personal application drafts, general source libraries, the 30 large
activation arrays stored as release assets, duplicate presentation bundles and
routine account/provider state in the private archive. Smaller scientific
arrays, fitted probes, saved trajectories and figure inputs are included. Runtime
environments and language-model weights are absent from both repositories.
I omit operational connection configuration from otherwise scientific source
packets and list those exact exclusions in the export manifest. A few immutable
provider-state receipts remain because the scientific closeout preserved them
as provenance inputs; their historical contents do not establish current
account state.

The new root documents, portable verifiers, verification record and export map
are review packaging. I distinguish them from the frozen experiment files in
the manifest. This package makes saved evidence inspectable; it does not supply
a newly validated, one-command GPU rerun.

The original export used archive commit `cd9c9b30b6f342c7033ebd55d252176a94ef3666`;
the expansion used `3320fb63e0da70025213f248e206868cbf8ff46b`. Neither source
workspace was edited. I record excluded files and the separately captured
review in `expansion` within the export manifest. [Reproduction boundaries](REPRODUCING.md)
explain which checks run from a clone and which workflows require additional assets.
