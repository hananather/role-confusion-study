# Role confusion in a tool-using agent

I study whether changing an agent's internal role representations changes how it
responds to instructions embedded in a retrieved webpage. I use GPT-OSS-20B,
role-classifier readouts and activation steering, building on
[Prompt Injection as Role Confusion](https://github.com/role-confusion/prompt-injection-as-role-confusion/tree/ec333c40fd43fe991e1ebf66765051b6d7e35784).
The agent executes commands in an isolated sandbox containing dummy files; a
local receiver records whether the unwanted upload actually occurred.

This is my private initial review version. The [experiment inventory](EXPERIMENTS.md)
covers all experiment families and distinguishes completed measurements,
partial studies and prepared work. The latest agent results are frozen at the
September 12 queue closeout: 100 recorded attempts, three of them censored,
ten separately indexed authorship readouts and ten unrun assignments. These
attempts cover repeated interventions on ten forged pages and their controls.

The original Tool-minus-CoT (chain of thought) direction produced uploads on 6/10 forged pages,
versus 7/10 without intervention and 2/10 under the best random control. A
later Tool-minus-mean(User, CoT) direction produced 0/5 uploads on the historical
pages; its five new-page tests remain unrun. Its margin over the best historical
random control is one upload. I have not established a direction-specific
defense or judged factual summary quality. The
[frozen results](replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z/README.md)
retain the full comparisons and limitations.

The original direction reduced the forged passage's reasoning-role score to
nearly zero on every measured page, including the six that still uploaded.
All five new-page upload outcomes were unchanged. The [report](REPORT.md)
connects these probe measurements to behavior and retains every random control.

![Every assigned intervention across ten forgery pages, including censored episodes and the five unrun new-page tests of the later Tool-raising direction.](replication/chart-library/steering-evidence-review/figures/01-complete-outcomes.png)

*Read down each page to compare the original role direction with no intervention,
random controls and the unfinished Tool-raising branch.*

## Review the work

1. [Read the report](REPORT.md) for the findings, figures and evidence boundaries.
2. [Browse all experiments](EXPERIMENTS.md) for each study's question, status and files.
3. [Follow the code-review route](REVIEW.md) from prompt construction to a complete matched episode.

To check the saved evidence with Python 3.11 or later, from this clone:

```bash
python3 scripts/verify_results.py
python3 scripts/verify_export.py
```

These commands read local files, check hashes and reconcile recorded outcomes.
They do not load a model, execute agent commands or contact a service.
[Verification notes](VERIFICATION.md) distinguish these checks from the
historical GPU runs. [Reproduction instructions](REPRODUCING.md) explain the
saved-data analyses, dependencies and larger archive inputs.

| Location | What I preserve |
| --- | --- |
| [Latest result snapshot](replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z/) | All assigned outcomes, raw-evidence index, separate readouts and audit records |
| [Earlier study](earlier-study/README.md) | 909 recovered records: 760 from the transfer task and 149 from the permission task, with saved-data analysis |
| [Provenance](PROVENANCE.md) | Source snapshots, export map, omitted material and portability boundaries |
| [Attribution](ATTRIBUTION.md) | Upstream sources, my project ownership and coding-agent assistance |

The broader [private archive](https://github.com/hananather/mats-role-confusion-archive)
retains bulk activation arrays and historical material separately. The
provenance notes identify this repository's coverage and portability limits.
I still need to review this initial version personally before presenting it as
my finished Neel submission.
