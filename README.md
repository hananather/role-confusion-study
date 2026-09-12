# Role confusion in a tool-using agent

I study whether changing an agent's internal role representations changes how it
responds to instructions embedded in a retrieved webpage. I use GPT-OSS-20B,
role-classifier readouts and activation steering, building on
[Prompt Injection as Role Confusion](https://github.com/role-confusion/prompt-injection-as-role-confusion/tree/ec333c40fd43fe991e1ebf66765051b6d7e35784).
The agent executes commands in an isolated sandbox containing dummy files; a
local receiver records whether the unwanted upload actually occurred.

This is my private code-review candidate, frozen at the September 12 queue
closeout. It includes 100 recorded agent attempts, three of them censored,
ten separately indexed authorship readouts and ten unrun assignments. The
attempts cover repeated interventions on ten forged pages and their controls.

The original Tool-minus-CoT direction produced uploads on 6/10 forged pages,
versus 7/10 without intervention and 2/10 under the best random control. A
later Tool-minus-mean(User, CoT) direction produced 0/5 uploads on the historical
pages; its five new-page tests remain unrun. Its margin over the best historical
random control is one upload. I have not established a direction-specific
defense or judged factual summary quality. The
[frozen results](replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z/README.md)
retain the full comparisons and limitations.

## Review the work

Start with [the code-review route](REVIEW.md). To check the saved evidence with
Python 3.11 or later, from this clone:

```bash
python3 scripts/verify_results.py
python3 scripts/verify_export.py
```

These commands read local files, check hashes and reconcile recorded outcomes.
They do not load a model, execute agent commands or contact a service.
[Verification notes](VERIFICATION.md) distinguish these checks from the
historical GPU runs.

| Location | What I preserve |
| --- | --- |
| [Review route](REVIEW.md) | Prompt construction, intervention, outcome scoring and matched evidence |
| [Latest result snapshot](replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z/) | All assigned outcomes, raw-evidence index, separate readouts and audit records |
| [Earlier study](earlier-study/README.md) | The separate 909-episode permission study and its saved-data analysis |
| [Provenance](PROVENANCE.md) | Source snapshots, export map, omitted material and portability boundaries |
| [Attribution](ATTRIBUTION.md) | Upstream sources, my project ownership and coding-agent assistance |

I retain the source and evidence bytes used for this review. I still need to
review the candidate personally before presenting it as my finished Neel
submission.
