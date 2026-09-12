# cloud-steering/

> **Toy lab.** Six experiments that deepen the role-confusion steering study for Hanan's MATS 12.0 application, sized to run in parallel on a multi-GPU H100 pod. Local cached gpt-oss-20b, dummy secrets, loopback addresses in text only. No command is executed and no sandbox is used: each unit is one generation from a fixed post-fetch prompt, and the outcome is read from the emitted text. Runtime-only activation edits; weights never modified. See [`../../TOY-LAB-NOTICE.md`](../../TOY-LAB-NOTICE.md). `protocol.py` and the fixtures under any `data/` page set are model-facing and must never carry this notice.

## Why this shape

Tonight's laptop run left five questions that a GPU answers cleanly. Each is written as a list of **independent cells** (one JSON line = one generation), so the work shards across GPUs by "take every n-th cell" and resumes by skipping ids already in the output. Sample size is deliberately small: a screen at 12 pages with one seed, then `analyze.py` re-emits only the arms that beat their matched random control, with extra seeds, on the pages where the attack landed. That is the "get ~10 positives, then rerun the positives" loop.

## The experiments

| Exp | Question | Mode | Confident-positive odds |
| --- | --- | --- | --- |
| 1 | Is the probe axis on the causal path? Patch the forged paragraph's activations from a token-aligned clean page, layer by layer | patch | localizes either way |
| 2 | Does the forgery create intent or resolve a doubt? Restart at the model's own doubt sentence with the paragraph present vs replaced | generate (doubt prefix) | ~65% |
| 3 | Is misattribution causal? Same forged page under a bare, a "Tool output:", and a "# The user has responded" wrapper | generate | ~60% |
| 4 | Recognition vs enforcement: does the model say "tool" when asked, on the same pages it acts on? | attribution | ~70% |
| 5 | The dissociation at power: none / Tool−CoT / random at alpha 8 and 16, with request-surfaced as a second endpoint | generate | ~75% the null holds |
| 6 | The geometry as one figure: cosines among role directions, all 24 layers, both sites | directions.py | ~95% |
| 7 | The one steering bet: a "deliberation minus adoption" direction from the model's own reasoning, on generated tokens | generate + prep | ~30% |
| 8 | **The destyle program.** Text destyle as positive control (paper: 60.5% styled vs 9.7% destyled), then the `destyled_minus_forged` direction on the paragraph tokens vs random; pair with exp 1 `--patch-source destyled` (content kept, style removed) | generate + prep | text ~85%, direction ~35% |
| 9 | **The declaration direction.** Same injected command inside the tool result minus in the user's own message, block 11: the attack-derived vector behind the one published positive and the September calibration recipe | generate + prep | ~35% |
| 10 | **Text baselines.** Provenance line and attribute-first scaffold in the developer message, both variants. The bar every vector must clear | generate | ~65% |

**Run order after the 2026-09-12 direction change:** 10 → 8 → 1 (destyled) → 9 → then 5, 3, 4, 1 (neutral), 2, 7. Reason: on the CUDA bridge, alpha-16 Tool−CoT equals baseline and random on three complete page blocks, so more pages of that arm buy precision on a null. Exps 8 to 10 are the arms that could show prevention with a mechanism story. Exp 8 needs `data/destyled.json` (page_id → destyled paragraph from the authors' destyle prompt).

## Files

`common.py` model load (authors' loader, MXFP4, eager or FA3), forward hooks (steer / patch / probe), prompt and span helpers, outcome classifier, RH6 attribution readout, Wilson intervals. `protocol.py` frozen Harmony rendering and parser. `pages.py` builds the page set (24 fresh Wikipedia pages x two variants, or copies the pilot's five). `directions.py` geometry (exp 6) and block-output class-mean directions. `prep.py` the two stateful prep steps: doubt prefixes (exp 2) and the decision vector (exp 7). `build_cells.py` spec generators. `run_cells.py` the sharded runner. `analyze.py` per-arm tables with paired Wilson intervals and the stage-B positive-cell file. `run_pod.sh` the orchestrator.

## Run

Approve a pod and cost first ([`../GPU-PLAYBOOK.md`](../GPU-PLAYBOOK.md), [`../EXECUTION.md`](../EXECUTION.md)). Then, inside the pod, from this directory:

```bash
N_GPU=4 PROBES=/workspace/results/20260911T030050Z/probes-full/probes.npz bash run_pod.sh
```

It builds the page set, directions, and prep once on GPU 0, screens all six experiments sharded across the GPUs, analyzes each, then reruns the positive arms with more seeds. Sync `out/`, `directions/`, `prefixes.json`, and `cells/` back; Claude and Hanan read the tables. Every arm keeps three endpoints separate (upload attempt, request surfaced in reasoning, summary produced) and every steering arm carries a norm-matched random control.
