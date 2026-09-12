# My local inference adapter

I run GPT-OSS-20B natively through MLX and send only prompt text to the inference
adapter. The adapter returns generated text and token IDs; the separate sandbox
owns tool execution. I require an existing local model snapshot, disable remote
tokenizer code, and do not download weights during construction.

## Paper protocol

I transcribed the Harmony helpers, system prompt, developer prompt, and parser
from cells 12 and 16 of
`experiments/cot-forgery-agent-evals/01-run-injections-gpt-oss.ipynb`. The system
date is January 1, 2026, reasoning is high, and the sole tool is `functions.bash`.
The developer warning is unchanged. I use 4,096 generated tokens per call and
temperature 1.0 unless a run explicitly records a different feasibility limit.

I preserve the notebook's reconstructed tool-call terminator `<|end|>` and its
space after `<|constrain|>`. These are deliberate source details. A generation
stops on exactly `<|call|>` or `<|return|>`, excluding the cached tokenizer's
additional `<|endoftext|>` stop. Raw evidence retains the stopping token. The
parser removes it before applying the authors' regular expressions, matching
the notebook's `run_step` trimming.

The authors' generation call leaves `top_k` unspecified. Transformers defaults
to `top_k=50`, and the cached generation configuration does not override it. I
therefore use top-k 50 and top-p 1.0, record them in metadata, and apply no
repetition, frequency, or presence penalty. MLX's default top-k 0 would otherwise
change this detail silently. This is an inference from the upstream call and
configuration, rather than an explicit paper specification.

I record parseable partial tool calls and malformed arguments. The parser can
accept end-of-string as a tool-call terminator, as the notebook does. The runner
must validate the tool name and require an object containing a string `command`;
parser success alone does not make a command executable. I preserve raw text
because reconstructing only parsed messages can discard malformed output.

## Limits and evidence

`MLXBackend.generate` returns a `Generation` with text, token IDs, input/output
token counts, wall-clock duration, prefill duration, peak MLX memory, finish
reason, stopping token, and an observational repetition flag. Output counts
include a generated stopping token. MLX memory excludes the container and other
Mac processes. Prefill duration includes reaching the first generated token.

I reject prompts whose token count plus the requested output allowance exceeds
the configured context cap. I never truncate the prompt, quantize the key/value
cache, or shrink its retained context in this adapter. The default 65,536-token
cap is a resource guard for local feasibility, and I record it separately from
the model's 131,072-position limit. Raising it requires another measured pilot.

I check the generation deadline between prefill chunks and tokens. A GPU kernel
or model loading can block beyond that deadline, so the runner must retain its
outer process watchdog. Progress events contain prefill counts and generation
snapshots at token 1, every 32 tokens, and completion. The runner can persist
these snapshots to recover partial output after a process termination.

I flag four adjacent identical token spans of length 8–128 after generation.
This heuristic can miss paraphrased loops or flag legitimate repetition. I do
not alter the primary experimental condition or stop early on that flag.

I record the local snapshot revision, model repository, configuration and
tokenizer hashes, quantization summary, software versions, context cap, stop
IDs, and effective sampling defaults. Seeded generation is reproducible only to
the extent the underlying runtime permits; an equal seed across MLX and CUDA
does not imply equal token trajectories.

## CUDA migration contract

`InferenceBackend` defines `metadata`, `count_tokens`, `generate`, and `close`.
The protocol and sandbox do not depend on MLX. A future CUDA implementation can
implement this interface, use the authors' `openai/gpt-oss-20b` model and attention
implementation, and leave the environment, inputs, limits, and scoring intact.

I have not implemented or validated a CUDA backend in this local pilot. The
authors' loader uses the `kernels-community/vllm-flash-attn3` attention
implementation and native model loading; I will verify those dependencies on
the selected cloud image before making a fidelity claim. My MLX snapshot uses
MXFP4 experts with 8-bit non-expert weights, which differs from their stack.
