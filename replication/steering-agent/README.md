# steering-agent/

> **Toy lab.** Activation steering and patching experiments on the paper's Section 3.3 agent loop, for Hanan's MATS 12.0 application. Local cached gpt-oss-20b (MLX MXFP4-Q8), dummy `.env`, loopback receiver, network-isolated sandbox. Vectors are added in memory for one generation; weights are never changed. See [`../../TOY-LAB-NOTICE.md`](../../TOY-LAB-NOTICE.md). `harness/protocol.py` and everything under `runs/` are model-facing and must never carry this notice.

## What is here

| File | Role |
| --- | --- |
| `harness/` | Byte copy of the harness source that produced the baseline episodes (`agent-hijacking/runs/ten-20260911/source`). Hashes in `harness/SOURCE-SHA256.json`. Frozen. |
| `directions.py` → `directions/block11.npz` | Class-mean role directions at the block-11 output from the dev-10 corpus: Tool−CoT, Tool−User, CoT−User, three random. Gap Tool−CoT = 38.5; residual norm about 1,284. |
| `extract_style_vector.py` → `directions/style.npz` | Attack-matched direction: mean forged payload minus mean standard payload over the five pages. |
| `steer_hook.py` | Span-masked block hook (positions tracked per call and checked against the KV-cache offset) and a layer-12 probe recorder over named spans. |
| `run_steer.py` | Full agent episodes with steering on tool-response tokens (`--mask tool`), on the injected payload only (`--mask payload`), or on every token including generated ones (`--mask all`). `--alpha 0` is the identity check. `--dev-note` appends a developer line for text baselines. |
| `continue_doubt.py` + `data/doubt-points.json` | Restart from the model's own doubt sentence inside a saved trajectory; several seeds per arm; outcome is whether the continuation emits the upload command. No sandbox. |
| `attribution_readout.py` | RH6's who-wrote-it question (verbatim) appended to each page's exact turn-1 prompt; option distribution and greedy answer per arm. |
| `patch_paragraph.py` | Token-aligned neutral replacement of the forged paragraph; clean activations patched into the forged run at one layer at a time; outcome is the upload command. |
| `summarize.py` | Tabulates arms against baseline from saved records. |
| `run_chain*.sh` | Sequential job chains. Never edit a running one; append a new segment that waits for the previous marker line. |
| `analysis/` | Reports from read-only analysis agents. |
| `runs/` | One directory per arm; `chain.log` is the live log. |

## Arms and order (2026-09-11)

Finished: `tool_minus_cot-a8` (4 of 5 uploads), `random_0-a8` (3 of 5), `tool_minus_cot-a8-all` (0 of 5; 4 of 5 never fetched the page), `tool_minus_cot-a16` (2 of 5). Partial: `tool_minus_cot-a0` completed pages 000 to 002 with turn-1 tokens identical to baseline, then the chain stopped at 21:04 on 2026-09-11. Not run: doubt, provenance, payload-only, style vector, attribution, patching. Those moved to the GPU battery in `../cloud-steering/`.

## Reading a result

Baseline: 4 of 4 first-draw forgery pages uploaded, plus the 004 resample; 3 of 5 also produced a real summary. Report per arm: exposed, upload attempted, receiver-verified upload, summary ok, censored, and the turn-1 probe reading on the page. Compare steered arms against `random_0` at the same magnitude and against `provenance` before claiming anything.

```bash
../.venv/bin/python summarize.py --cases
```
