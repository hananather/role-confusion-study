# What I learned from the persistent-pod pilot

I completed the small probe pipeline, but the pilot did **not** establish full-run memory safety or a valid grouped held-out evaluation. It produced all 384 requested probe fits in 342.7 seconds. My first full attempt then failed during output-probability extraction, and my inspection found complete train/test overlap in the pilot's grouped split. I preserve both findings before interpreting any accuracy.

This is an operational audit in support of **Understand: controlled replication**. My north star remains the gardening checkpoint under the authors' recipe. I use this pilot to validate execution and discover failure modes; I do not present it as a paper-scale replication result.

## My verified pilot observations

| Check | Observation from saved outputs |
| --- | --- |
| End-to-end completion | The stage receipt records completion at 2026-09-11 03:21:37 UTC. Training metadata spans 03:15:51.820–03:21:34.476 UTC: 342.656 seconds. |
| Corpus | Ten requested documents yielded nine by the existing rounding: two C4 and seven Dolma3 documents; 45 prompts. |
| Token labels | I reread `tokens.parquet`: 18,789 total token rows, 18,510 kept content rows, and **3,702 target-role tokens each** for system, user, CoT, assistant, and tool. This agrees with the prior small-pilot token accounting. |
| Forward fidelity | Metadata records `custom_forward_verified=true`; the captured console also reports exact agreement between the authors' custom forward and the model's ordinary forward. |
| Model and source | The recorded model snapshot is `6cee5e81ee83917806bbde320786a8fb61efebee`; the authors' checkout is `ec333c40fd43fe991e1ebf66765051b6d7e35784`. |
| Precision and attention | Experts have `FloatType(bitwidth_exponent=2, bitwidth_mantissa=1, is_signed=True)`; attention is `kernels-community/vllm-flash-attn3`. The H100 reports compute capability 9.0. |
| Environment | Python 3.12.14; torch 2.9.1+cu128; Transformers 4.57.5; kernels 0.11.5; Triton 3.5.1; cuML/cuDF 25.8.0; CuPy 13.6.0; pandas 2.3.3; NumPy 2.2.6. These are observed package versions, not installation intentions. |
| Output integrity | Each split's JSON has 192 fit records. Each NPZ has 192 coefficient arrays and 192 intercept arrays; I checked that every stored value is finite. |

My pilot supports the conclusion that this environment can load the intended model, extract the chosen site, label the small corpus, fit all requested probe combinations, and save portable results. It does not establish the gardening result, successful full-corpus fitting, or the quality of every fitted optimum.

## My grouped-split finding

Every one of the 192 grouped-split pilot records has `n_train == n_test == n_inputs`. For example, layer-12 `sucat` uses 18,510 training rows and 18,510 test rows; `suca` uses 14,808 in each. These are the complete eligible token counts for those spaces. The near-perfect grouped-split pilot accuracies therefore provide no held-out evidence.

The official **cuML 25.08.00** implementation converts a fractional test size to `int(n * test_size)`, then selects test indices with a suffix slice. Nine groups make that count zero; the suffix slice beginning at negative zero returns every group. Training also receives all nine. My existing fallback checks whether the returned test array is empty, so it never runs in this case. [Versioned cuML source, size calculation and index slicing](https://github.com/rapidsai/cuml/blob/v25.08.00/python/cuml/cuml/model_selection/_split.py#L296).

I reproduced the size-and-slice behavior with a small NumPy stand-in, without running cuML or a GPU job:

| Unique groups and requested test size | Training groups | Test groups | Overlap |
| --- | ---: | ---: | ---: |
| 9, fraction 0.1 | 9 | 9 | 9 |
| 9, integer 1 | 8 | 1 | 0 |
| 45, fraction 0.1 | 41 | 4 | 0 |
| 249, fraction 0.1 | 225 | 24 | 0 |
| 1,245, fraction 0.1 | 1,121 | 124 | 0 |

The stand-in verifies the boundary arithmetic and slicing, not cuML's exact random permutation. The 45/1,245 prompt-ID examples describe the five-role space; narrower role spaces have fewer prompt IDs. The source and saved pilot counts agree on the failure mechanism.

**My proposed mechanical correction:** choose an integer one-group test size *before* calling cuML only for a tiny pilot whose fractional count would be zero. I retain `test_size=0.1`, the same cuML splitter, and seed 123 for full runs. Before fitting under either split, I require nonempty disjoint train/test group sets, their union to equal the input groups, and the resulting row masks to partition the eligible rows. I save the actual group IDs and counts in a split audit. This corrects the pilot boundary without replacing the full split method. I have not edited the trainer in this audit.

## My memory and runtime lessons

The pilot's **forward-pass GPU peak was 62.03 GB**. The final `peak_vram_gb=13.83` field comes after an intervening peak-counter reset; it must not be reported as the whole-run peak. My final process memory figure is also not a substitute for the maximum required workspace.

The pilot measured 35.3 seconds for model load, 11.6 seconds for forward verification, 23.1 seconds for its two extraction batches, and 187.8 seconds for all fits. Its saved forward estimate was 447 seconds for a projected 40 full batches. The actual full corpus has 39 batches at the initial batch size, 713,109 valid tokens, and a maximum sequence length of 1,035, versus the pilot's 749. The pilot's 0.5 seconds per fit cannot be extrapolated unchanged to the full corpus, which has about 38 times as many valid tokens.

The first full attempt reached the 249-document corpus and allocated the 98.6 GB activation layout, then failed after one completed extraction batch. The captured error is a CUDA out-of-memory exception at the authors' `utils/store_outputs.py:85`, in `torch.logsumexp` after converting logits to float32: it requested 24.81 GiB with 15.96 GiB free. The queue recorded `needs_attention`, `probes-full`, process exit 1 at **03:23:14 UTC**. This is GPU workspace pressure; the 2 TB storage allocation does not fix it. A preallocated activation-file shape does not mean the full activation set has been written.

That failed attempt remains a failed attempt even if a later retry succeeds. I leave the mechanical memory correction and current live retry state to the session controller. This note records the saved failure evidence; it does not claim the original full run is still progressing or that full fitting has completed.

## My solver-warning boundary

The captured pilot console contains **51 cuML L-BFGS line-search warnings**: 22 report a failed search stopping at the last valid step, and 29 report no further step advance. This counts warning messages, not necessarily distinct failed fits. All requested fit records and finite coefficient arrays were saved, but that does not prove convergence to each optimum.

These warnings are a known caveat of the inherited unscaled fitting recipe. I preserve them and compare the full-run diagnostics under that recipe. I do not silently change regularization, scaling, or solver settings to make the warning count disappear. The initial CPU-load warning is distinct: subsequent device/forward records confirm GPU execution.

## My next evidence gate

Before I call the full probes ready, I require successful full extraction and fitting, complete output counts, an explicit disjoint-group audit, and confirmed source/precision/attention metadata. I then run and verify the three fixed gardening conditions. The current pilot's grouped accuracies are excluded from scientific claims; the pipeline-completion result and measured runtime remain useful operational evidence.

My local sources are the saved [pilot metadata](outputs/20260911T030050Z/probe-pilot/metadata.json), [pilot grouped results](outputs/20260911T030050Z/probe-pilot/probes-basesplit.json), [pilot console](outputs/20260911T030050Z/stage-logs/probe-pilot.log), [pilot receipt](outputs/20260911T030050Z/stage-receipts/probe-pilot.json), and [first full-attempt console](outputs/20260911T030050Z/stage-logs/probes-full.log). The session controller may archive the last path when preserving a retry; the failure timestamp above identifies the attempt I inspected.
