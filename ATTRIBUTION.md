# My contribution and source attribution

I developed this project with coding-agent assistance. My project code includes
the isolated agent harness, MLX steering wrapper, CUDA adapter, registered
comparisons, outcome accounting and saved-data analysis. I used coding agents
for implementation, execution support, analysis and review. The source workspace
records Claude's steering work and separate Grok permission/role-uptake work;
this export retains the completed evidence within its stated scope.

This candidate's packaging, navigation documents and portable verifiers were
prepared with Codex. I have not represented these files as entirely hand-written
or claimed that I personally reviewed every line. The
[verification record](VERIFICATION.md) identifies the checks actually run.
My personal review of the candidate remains separate from those automated
checks.

## Paper and code

I build on *Prompt Injection as Role Confusion* by Charles Ye, Jasmine Cui and
Dylan Hadfield-Menell. The frozen upstream code reference is
[`ec333c40fd43fe991e1ebf66765051b6d7e35784`](https://github.com/role-confusion/prompt-injection-as-role-confusion/tree/ec333c40fd43fe991e1ebf66765051b6d7e35784).
The benchmark protocol and attack templates derive from that work. The
[source contract](replication/agent-hijacking/docs/source-contract.md) records
the preserved settings and local adaptations.

I preserve the upstream [MIT permission text](licenses/upstream-MIT.txt)
unchanged. The [license scope](LICENSE.md) distinguishes incorporated upstream
code, the earlier study and third-party text. The upstream repository itself
is not presented as my implementation.

## Pages, model and datasets

The fixtures and trajectories contain adapted Wikipedia text. I preserve the
original page URLs, titles, retrieval times, raw HTML and source hashes in the
[historical-page manifest](replication/agent-hijacking/data/pilot-20260911/manifest.json)
and the [new-page bank](replication/steering-series/2026-09-12-positive-confirmation/new-pages/manifest.json).
I normalized page HTML and inserted the synthetic experimental instructions;
the Wikipedia text remains attributable to its contributors under its original
terms, including CC BY-SA 4.0. Available retrieval metadata is preserved; I do
not invent revision identifiers missing from a manifest.

The earlier study has its own
[page-by-page attribution](earlier-study/THIRD_PARTY.md). Its page list describes
that study, not the later agent batches.

I used GPT-OSS-20B, with the exact implementation and revision recorded in the
run metadata. I include local probe and direction arrays, not language-model
weights. Upstream dataset references and component text retain their original
attribution and licensing. The broader replication files, frozen preparation
inputs and fitted probes are included where the archive contains them. Large
activation arrays remain in the private archive release; the
[scope record](PROVENANCE.md) identifies that boundary. Files under
`replication/reference/` are earlier reference projects, not additional
experiments I claim to have run in this study. Their own source and license
notices remain beside them.
