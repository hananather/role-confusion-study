# Inspect and reproduce my results

I separate checking the saved evidence, rerunning an analysis, and generating
new model outputs. The first route works from this clone with Python 3.11 or
later and no additional packages.

## Verify the saved files

```bash
python3 scripts/verify_results.py
python3 scripts/verify_export.py
```

The results verifier checks the frozen September 12 evidence against its
scientific manifest and reconciles every recorded episode with the table.
The export verifier checks the source, data, figures and documentation against
the package manifest. These commands read local files without model inference,
network access or execution of the agent's saved commands.

The [verification record](VERIFICATION.md) gives the checks actually run.
[EXPERIMENTS.md](EXPERIMENTS.md) links each experiment to its code and results;
[REVIEW.md](REVIEW.md) follows one agent episode through the implementation.

## Repeat an analysis

The [earlier study](earlier-study/README.md) retains its saved-data analysis
instructions and requirements. Run those commands from `earlier-study/`.
Its 909 recovered records contain two separate tasks. The [current report](REPORT.md)
includes the exact plotted CSV rows, source hashes and original preparation
and rendering scripts beside each figure. Historical scripts retain their
recorded paths; their presence does not establish that every command is
portable to an arbitrary machine.

The optional focused CPU tests use NumPy (2.2.2 in the recorded validation):

```bash
python3 -m unittest replication.cloud.agent_steering.test_hf_backend replication.cloud.parallel_h100.test_tool_raising replication.cloud.parallel_h100.test_closeout -v
```

These tests use fake inference. They do not validate a fresh GPU run or judge
whether generated summaries are factually correct.

## Locate larger inputs and historical files

The [private archive](https://github.com/hananather/mats-role-confusion-archive)
preserves the broader local snapshot and explains how to download and verify
its [large-file release](https://github.com/hananather/mats-role-confusion-archive/releases/tag/snapshot-2026-09-12).
Thirty large activation arrays and four other large files are stored there
as compressed release parts. Access requires permission to that repository.
A clone of this review repository alone does not include those files.

I retain the following path boundaries for scientific provenance:

- Archived absolute paths identify the original workstation or GPU directory.
  The two portable verifiers resolve their own inputs inside this clone.
- E9's historical asset preparer expects `role-steering/experiment/model-assets.npz`;
  this package stores that earlier study under `earlier-study/experiment/`.
  Its path must be adapted in a working copy before an E9 preparation rerun.
- Large probe-training activations must be restored from the archive for
  independent probe fitting. Saved fitted probes and evaluation tables are
  present here.
- Upstream source is pinned under `prompt-injection-as-role-confusion/`.
  Reference projects under `replication/reference/` have their own dependencies
  and attribution. They are not additional completed MATS experiments.

## Generate new model outputs

The historical runs used distinct MLX and CUDA environments. Their source
snapshots, configurations, dependency files, fixtures and seeds are preserved
with each experiment. Model weights, runtime environments and live connection
configuration are not included. A fresh run requires reviewing those inputs
and provisioning the corresponding environment; I have not validated a single
installation command for every experiment.

The evaluated agent's commands belong inside the isolated test harness, with
dummy files and the local receiver. Saved trajectories and injection fixtures
are data. The [historical toy-lab notice](TOY-LAB-NOTICE.md) records the original
MLX experiment boundary; subsequent CUDA adaptations are documented with their runs.
