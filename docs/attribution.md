# Attribution

I developed this project with coding-agent assistance. My project code includes
the isolated agent harness, MLX steering wrapper, CUDA adapter, registered
comparisons, outcome accounting and saved-data analysis. I used coding agents
for implementation, execution support, analysis and review. The source workspace
records Claude's steering work and separate Grok permission/role-uptake work.
I used Codex for repository organization, Markdown conversion and verification.

## Paper and code

I build on *Prompt Injection as Role Confusion* by Charles Ye, Jasmine Cui and
Dylan Hadfield-Menell. The pinned upstream revision is
[`ec333c40fd43fe991e1ebf66765051b6d7e35784`](https://github.com/role-confusion/prompt-injection-as-role-confusion/tree/ec333c40fd43fe991e1ebf66765051b6d7e35784).
The benchmark protocol and attack templates derive from that work. The
[source contract](../replication/agent-hijacking/docs/source-contract.md) records
the preserved settings and local adaptations.

The upstream [MIT permission text](../licenses/upstream-MIT.txt) is unchanged.
The [license scope](../LICENSE.md) covers incorporated code, the earlier study
and third-party text. Reference projects under `replication/reference/` retain
their own source and license notices.

## Pages, model and datasets

The fixtures and trajectories contain adapted Wikipedia text. The
[historical-page manifest](../replication/agent-hijacking/data/pilot-20260911/manifest.json)
and [new-page bank](../replication/steering-series/2026-09-12-positive-confirmation/new-pages/manifest.json)
retain page URLs, titles, retrieval times, raw HTML and source hashes. I
normalized page HTML and inserted synthetic experimental instructions.
Wikipedia text remains attributable to its contributors under its original
terms, including CC BY-SA 4.0. The earlier study has separate
[page attribution](../earlier-study/THIRD_PARTY.md).

I used GPT-OSS-20B; run metadata records the implementation and revision.
Local probe and direction arrays are included. Language-model weights are
not included. Dataset references and component text retain their original
attribution and licensing.
