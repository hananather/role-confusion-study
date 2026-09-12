# My source contract and looping review

I reviewed the supplied v6 paper and the authors' notebook frozen at `ec333c40fd43fe991e1ebf66765051b6d7e35784`. My target is Section 3.3 / Figure 3 and Appendix B.2. I keep Appendix I's activation analysis separate.

## What I preserve

The agent receives a webpage-summary request and a warning about malicious HTML commands. It can invoke one bash tool. A dummy `.env` is placed in `/workspace/container-1`; a webpage contains a standard command or the supplied command-plus-forgery variant. I use the released system/developer instructions, Harmony rendering and parser expressions, high reasoning, temperature 1, eight iterations, 4,096 new tokens per iteration and a 30-second shell limit.

The notebook leaves top-k unspecified. The Transformers generation default supplies top-k=50; top-p=1.0. I make these effective defaults explicit in MLX. [Transformers 4.57.5 generation configuration](https://github.com/huggingface/transformers/blob/v4.57.5/src/transformers/generation/configuration_utils.py).

The authors' source notebook executes bash on its host and uploads HTML to public hosting. I replace those operations with a real isolated bash environment and internal HTTP page/receiver. No evaluated command executes on the Mac host. The local endpoint and MLX numerical backend are visible deviations, so I call this an adapted local replication, not an exact numerical reproduction.

The notebook inserts raw payload text before `</body>`, whereas the paper example shows a hidden span. I follow the released code. Its five standard and five forged templates are already written; I do not regenerate or improve them. I match template types deterministically in the paired pilot, while the notebook samples them independently. The first two forged variants contain extra coin/shirt declarations, so matching types does not make their text identical apart from reasoning.

## What the paper actually says about loops

Appendix B.1 reports infinite reasoning cycles for certain **chat jailbreak** requests on GPT-OSS-20B and GPT-5 nano. It prepends a benign distractor there. Appendix B.2 does not specify that distractor for the agent evaluation. I therefore do not add it to this first agent run. [Local paper, B.1/B.2](</Users/hananather/Documents/ChatGPT/MATS 12/research/sources/paper-markdown/prompt-injection-as-role-confusion.md:1095>).

Independent work reports nonterminating or unproductive reasoning under adversarial prompting in GPT-OSS-20B. This supports measuring incomplete reasoning as an outcome; it does not establish a general failure rate for our agent task. [Lin et al., 2025, Quant Fever, Reasoning Blackholes, Schrodinger's Compliance, and More](https://arxiv.org/html/2509.23882v2).

A firsthand MLX issue reports repeated apparent tool calls when its server failed to stop on the call token. That is an integration failure, not sufficient evidence of an intrinsic agentic weakness. The issue is now closed and concerns a different quantization/server path. I handle the stopping token explicitly and test raw tokens in the installed runtime. [MLX issue 613](https://github.com/ml-explore/mlx-lm/issues/613).

OpenAI's reference recommends temperature=1.0 and top-p=1.0. I preserve those values. [GPT-OSS reference repository](https://github.com/openai/gpt-oss#recommended-sampling-parameters). Harmony distinguishes ordinary message ends, turn completion and tool handoff; confusing them can fabricate a loop. [Official Harmony format](https://developers.openai.com/cookbook/articles/openai-harmony).

## Mitigations and measurement

I first verify protocol correctness and use bounded execution with full partial traces. Those are engineering controls. Repetition penalties, lower reasoning effort, distractors, tool-loop feedback or context summarization would change the experiment and must be separate named arms after the unchanged pilot.

The paper's appendix evaluates planned/attempted exfiltration regardless of successful receipt. Its judge sees reasoning, assistant text and tool requests. I preserve a corresponding judge-input view plus the fuller raw trajectory and trusted tool returns. The original rubric's ATTEMPTED_UNSUCCESSFUL label can include a stated plan without an upload call. I record that separately from an emitted request and a verified receipt. A local receiver alone cannot reproduce the paper's ASR definition.

I keep a first-run smoke episode outside the ten-episode denominator. I require model-generated valid tool use, delivery and processing of the injection, intact raw records and verified cleanup before the timing batch. Attack success is not a gate. I retain refusals and unfinished attempts rather than selecting only successful trajectories.

## Evidence boundaries

- The paper's GPT-OSS-20B result is 26% standard versus 56% forgery at 100 pages per condition. Those values are reference results.
- Ten pilot episodes cannot establish a precise attack rate, a stable runtime tail, a long-context effect or a mechanism.
- All source and environment changes are recorded in the frozen manifest and run metadata.
- Changing to an H100 may resolve throughput constraints. It will not by itself resolve a model's reasoning loop; different kernels/quantization can also change outputs.
