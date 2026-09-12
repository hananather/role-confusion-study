# Reproduction

## Saved results

From the repository root, with Python 3.11 or later:

```bash
python3 scripts/verify_results.py
python3 scripts/verify_export.py
```

The first command checks the September 12 agent snapshot's saved files and
reconciles its episode records with the assignment table. The second checks
every file in the export manifest. Both use the standard library and read local
files without model inference. The [verification records](verification.md) list
the observed checks.

To rebuild the README from the captured document and image map:

```bash
python3 scripts/export_readme.py
```

## Analysis and tests

The [earlier study](../earlier-study/README.md) includes its analysis commands and
requirements. Run them from `earlier-study/`.

The [README figure files](../replication/chart-library/figure-selection/README.md)
list the data and renderers for Figures 1–4.

The [earlier figure data](../replication/chart-library/steering-evidence-review/data/)
retain the plotted rows and source hashes. The corresponding
[preparation](../replication/chart-library/steering-evidence-review/prepare_evidence.py)
and [rendering](../replication/chart-library/steering-evidence-review/render_figures.py)
scripts retain their original paths and dependencies.

The following CPU tests require NumPy and use fake inference:

```bash
python3 -m unittest replication.cloud.agent_steering.test_hf_backend replication.cloud.parallel_h100.test_tool_raising replication.cloud.parallel_h100.test_closeout -v
```

## Model runs

I have not validated a single installation command for every experiment. The
historical MLX and CUDA runs have separate configurations, dependencies and
source snapshots. Model weights and runtime environments are not included.
The [experiment index](experiments.md) links their recorded methods and inputs.

Thirty large activation arrays and four other large files are in the
[private archive release](https://github.com/hananather/mats-role-confusion-archive/releases/tag/snapshot-2026-09-12).
Independent probe fitting requires restoring those inputs; saved fitted probes
and evaluation tables are included here.

Historical absolute paths identify the original workstation or GPU directory.
E9's asset preparer expects `role-steering/experiment/model-assets.npz`; this
repository stores that study under `earlier-study/experiment/`. Adapt the path
in a working copy before rerunning preparation.

The evaluated agent's commands run inside the isolated harness, with dummy
files and the local receiver. Saved trajectories and injection fixtures are
research data. The [toy-lab notice](../TOY-LAB-NOTICE.md) records the original
MLX boundary; CUDA adaptations are documented with their runs.
