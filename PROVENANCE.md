# My review export

I assembled this candidate from the frozen `desktop/` tree in the separate
private `mats-role-confusion-archive` repository. I did not export from the
changing execution workspace. The
[export manifest](provenance/export-manifest.json) maps each copied file to its
archive path, byte count and SHA-256. All copied source, scientific fixtures and
results retain their archived bytes.

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
[earlier permission study](earlier-study/README.md). I retain the full frozen
77-page carrier bank for input-selection provenance; only the registered prefix
was used in the completed new-page comparison.

The private archive holds the broader probes/gardening/Appendix K/RH6 work,
313-prompt unsteered chat baseline, evolving permission-study instrumentation,
role-uptake pilot, deferred experimental battery and historical drafts. They
are not represented here as completed parts of the September 12 cohort.

I leave personal application drafts, general source libraries, activation
arrays and routine account/provider state in the private archive. Runtime
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
