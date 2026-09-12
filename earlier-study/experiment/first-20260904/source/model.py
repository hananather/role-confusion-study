"""Native gpt-oss inference, two activation sites, and a five-role linear probe.

The source probe uses C=0.005, L2, an intercept, and no feature scaling.
This implementation uses chunked torch L-BFGS instead of cuML and saves that
deviation. Native tool content is JSON encoded; every neutral role receives the
same encoded passage so escaping cannot define the User-minus-Tool direction.
No source notebook is imported or executed.
"""

import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
from pathlib import Path


ROLES = ("system", "user", "assistant", "cot", "tool")
READ_TOOL = [{"type": "function", "function": {
    "name": "read", "description": "Read the fixed reference passage.",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}},
                   "required": ["path"]}}}]


class Model:
    def __init__(self, config, result_dir):
        import numpy as np
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.np, self.torch, self.config = np, torch, config
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = self.result_dir / "model-metadata.json"
        previous = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        revision = config["model_revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("model_revision must be a full immutable commit SHA")
        if not torch.cuda.is_available():
            raise RuntimeError("This experiment requires the allocated CUDA GPU")
        if transformers.__version__ != "4.57.5":
            raise RuntimeError("The verified model adapter requires transformers==4.57.5")
        # gpt_oss in 4.57.5 explicitly declares _supports_sdpa=False.
        if config.get("attn_implementation", "eager") != "eager":
            raise ValueError("Use the verified eager attention implementation")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_id"], revision=revision, use_fast=True, trust_remote_code=False)
        if not self.tokenizer.is_fast:
            raise RuntimeError("Exact character-to-token masks require a fast tokenizer")
        self.lm = AutoModelForCausalLM.from_pretrained(
            config["model_id"], revision=revision, dtype=torch.bfloat16,
            device_map="cuda", attn_implementation="eager", trust_remote_code=False)
        self.lm.eval().requires_grad_(False)
        self.layers = self.lm.model.layers
        self.steer_layer = int(config.get("steering_layer", 11))
        self.probe_layer = int(config.get("probe_layer", 14))
        if not 0 <= self.steer_layer < self.probe_layer < len(self.layers):
            raise ValueError("Require steering block < probe block < layer count")
        quant = getattr(self.lm.config, "quantization_config", None)
        quant = quant.to_dict() if hasattr(quant, "to_dict") else quant
        experts = self.layers[0].mlp.experts
        if not quant or quant.get("quant_method") != "mxfp4" or quant.get("dequantize", False):
            raise RuntimeError("MXFP4 was not retained; check the Triton/kernels installation")
        self.stop_ids = {}
        for token in ("<|call|>", "<|return|>"):
            ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(ids) != 1 or self.tokenizer.decode(ids) != token:
                raise RuntimeError(f"Unsupported Harmony control token: {token}")
            self.stop_ids[ids[0]] = token[2:-2]
        self.directions, self.scale, self.probe = {}, None, None
        versions = {}
        for name in ("torch", "transformers", "accelerate", "numpy", "kernels", "triton"):
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        kernel_paths = sorted({str(Path(module.__file__).resolve())
                               for name, module in list(sys.modules.items())
                               if "triton_kernels" in name and getattr(module, "__file__", None)})
        self.meta = {**previous, "stage": previous.get("stage", "loaded"), "model_id": config["model_id"],
                     "model_revision": revision, "versions": versions,
                     "gpu": torch.cuda.get_device_name(), "quantization": quant,
                     "expert_module": type(experts).__module__ + "." + type(experts).__name__,
                     "attention": self.lm.config._attn_implementation,
                     "generation_config": self.lm.generation_config.to_dict(),
                     "hf_home": os.environ.get("HF_HOME"),
                     "hf_modules_cache": os.environ.get("HF_MODULES_CACHE"),
                     "triton_cache": os.environ.get("TRITON_CACHE_DIR"),
                     "loaded_kernel_files": kernel_paths,
                     "loaded_kernel_snapshot_commits": sorted({match.group(1) for path in kernel_paths
                         if (match := re.search(r"/snapshots/([0-9a-f]{40})/", path))}),
                     "steering_site": f"model.layers.{self.steer_layer} output residual",
                     "probe_site": f"model.layers.{self.probe_layer}.post_attention_layernorm output",
                     "roles": ROLES, "boundary_rule": "any positive character overlap",
                     "adaptations": ["Native valid role contexts, not bare role tags",
                         "Same JSON-rendered passage in all roles; native context/position remain different",
                         "Fixed C=0.005 torch L-BFGS in place of cuML",
                         "Base-text-grouped 200/25/25 split in each 250-passage pool"]}
        self._save_meta()

    def _save_meta(self):
        target = self.result_dir / "model-metadata.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.meta, indent=2))
        temporary.replace(target)
        print("Model progress:", self.meta.get("stage"),
              "passages=", self.meta.get("completed_passages"),
              "optimizer_calls=", self.meta.get("probe_optimizer_calls"), flush=True)

    def _tokens(self, prompt, spans):
        for start, end in spans:
            if not 0 <= start < end <= len(prompt):
                raise ValueError(f"Invalid character span {(start, end)}")
        encoded = self.tokenizer(prompt, add_special_tokens=False,
                                 return_offsets_mapping=True, return_tensors="pt")
        offsets = encoded.pop("offset_mapping")[0].tolist()
        indices = [i for i, (a, b) in enumerate(offsets)
                   if b > a and any(a < end and b > start for start, end in spans)]
        for start, end in spans:
            if not any(offsets[i][0] < end and offsets[i][1] > start for i in indices):
                raise ValueError("A requested span maps to zero tokens")
        return {k: v.to("cuda") for k, v in encoded.items()}, indices, offsets

    def _capture(self, prompt, span, zero=False):
        torch = self.torch
        inputs, indices, _ = self._tokens(prompt, [span])
        captured = {}

        def residual(module, args, output):
            if not isinstance(output, torch.Tensor):
                raise RuntimeError("Unexpected block output; the verified contract changed")
            captured["residual"] = output[0, indices].detach().float().cpu()
            return output + torch.zeros_like(output) if zero else output

        def probe(module, args, output):
            captured["probe"] = output[0, indices].detach().to("cpu", dtype=torch.float16)

        handles = [self.layers[self.steer_layer].register_forward_hook(residual),
                   self.layers[self.probe_layer].post_attention_layernorm.register_forward_hook(probe)]
        try:
            with torch.inference_mode():
                logits = self.lm(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float().cpu()
        finally:
            for handle in handles:
                handle.remove()
        captured.update(logits=logits, indices=indices,
                        token_ids=inputs["input_ids"][0, indices].tolist())
        return captured

    def _render(self, messages, **kwargs):
        rendered = self.tokenizer.apply_chat_template(
            messages, tools=READ_TOOL, tokenize=False, add_generation_prompt=False,
            reasoning_effort=self.config.get("reasoning_effort", "medium"), **kwargs)
        # The source template calls strftime_now; freeze the date across a long run.
        return re.sub(r"Current date: \d{4}-\d{2}-\d{2}",
                      "Current date: " + self.config.get("date", "2026-09-04"), rendered)

    def _role_contexts(self, text):
        messages = [{"role": "user", "content": "Read the reference passage."},
                    {"role": "assistant", "tool_calls": [{"type": "function", "function": {
                        "name": "read", "arguments": {"path": "reference.txt"}}}]},
                    {"role": "tool", "name": "read", "content": text}]
        tool = self._render(messages)
        header = "<|start|>functions.read to=assistant<|channel|>commentary<|message|>"
        if tool.count(header) != 1:
            raise RuntimeError("Native tool rendering does not match the verified template")
        start = tool.index(header) + len(header)
        end = tool.index("<|end|>", start)
        content = tool[start:end]
        user = self._render([{"role": "user", "content": content}])
        assistant = self._render([{"role": "user", "content": "Continue the passage."},
                                 {"role": "assistant", "content": content}])
        cot = self._render([{"role": "user", "content": "Continue the passage."},
                           {"role": "assistant", "thinking": content, "content": "Done."}])
        system = self._render([{"role": "user", "content": "Continue."}], model_identity=content)
        contexts = {"system": system, "user": user, "assistant": assistant, "cot": cot, "tool": tool}
        result = {}
        for role, prompt in contexts.items():
            if prompt.count(content) != 1:
                raise ValueError("Neutral content does not have one unambiguous rendered location")
            begin = prompt.index(content)
            result[role] = (prompt, (begin, begin + len(content)))
        return result

    def prepare(self, corpus, calibration_pair, calibration_span):
        torch, np = self.torch, self.np
        identity = hashlib.sha256(json.dumps({
            "config": {k: v for k, v in self.config.items() if k != "wall_seconds"},
            "corpus": corpus, "calibration_pair": calibration_pair,
            "calibration_span": calibration_span,
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }, sort_keys=True).encode()).hexdigest()
        assets = self.result_dir / "model-assets.npz"
        if self.meta.get("assets_ready"):
            if self.meta.get("preparation_sha256") != identity or not assets.exists():
                raise ValueError("Existing model assets do not match the exact preparation inputs/source/config")
            with np.load(assets, allow_pickle=False) as saved:
                self.probe = tuple(torch.from_numpy(saved[name].copy()).to("cuda")
                                   for name in ("probe_weight", "probe_bias"))
                self.scale = float(saved["scale"])
                self.directions = {name: torch.from_numpy(saved[name].copy())
                                   for name in ("clean", "calibration", "zero", "random_0", "random_1", "random_2")}
            if not self.scale > 0 or not all(torch.isfinite(v).all() for v in self.directions.values()):
                raise ValueError("Existing assets contain invalid values")
            self.meta.update(stage="ready", resumed_assets=True)
            self._save_meta()
            return
        self.meta["preparation_sha256"] = identity
        if len(corpus) != 500:
            raise ValueError("Expected exactly 250 probe and 250 direction passages")
        texts, hashes = [], set()
        retained = []
        for row in corpus:
            tokens = self.tokenizer.encode(row["text"], add_special_tokens=False)[:1024]
            text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            # JSON escaping can expand content beyond the raw passage's token cap.
            while True:
                rendered, span = self._role_contexts(text)["tool"]
                count = len(self.tokenizer.encode(rendered[span[0]:span[1]], add_special_tokens=False))
                if count <= 1024:
                    break
                tokens = tokens[:max(1, int(len(tokens) * 1022 / count))]
                text = self.tokenizer.decode(tokens, skip_special_tokens=False)
            digest = hashlib.sha256(text.encode()).hexdigest()
            if not text.strip() or digest in hashes:
                raise ValueError("Empty or duplicate truncated base passage")
            if any(s in text for s in self.tokenizer.all_special_tokens):
                raise ValueError("Neutral passage contains native special-token text")
            hashes.add(digest)
            texts.append(text)
            retained.append({"id": row["id"], "retained_sha256": digest,
                             "raw_content_tokens": len(tokens), "encoded_content_tokens": count})
        self.meta["retained_passages"] = retained
        records, positions, example_contexts = [], [], {}
        for i, text in enumerate(texts[:250]):
            contexts = self._role_contexts(text)
            for role_id, role in enumerate(ROLES):
                prompt, span = contexts[role]
                cap = self._capture(prompt, span)
                if i == 0 and role == "user":
                    zero = self._capture(prompt, span, zero=True)
                    difference = (cap["logits"] - zero["logits"]).abs().max().item()
                    self.meta["zero_hook_max_logit_difference"] = difference
                    if not torch.allclose(cap["logits"], zero["logits"], atol=1e-5, rtol=1e-5):
                        raise RuntimeError("Zero-hook logits differ; resolve before proceeding")
                records.append((i, role_id, cap["probe"]))
                positions.append({"pool": "probe", "id": corpus[i]["id"], "role": role,
                                  "first": cap["indices"][0], "last": cap["indices"][-1],
                                  "tokens": len(cap["indices"])})
                if i == 0:
                    example_contexts[role] = {"prompt": prompt, "span": span,
                                              "indices": cap["indices"], "token_ids": cap["token_ids"]}
            if (i + 1) % 25 == 0:
                self.meta.update(stage="probe_extraction", completed_passages=i + 1)
                self._save_meta()
        self._fit_probe(records)
        del records
        deltas, norms, validation = [], [], []
        for i, text in enumerate(texts[250:]):
            contexts = self._role_contexts(text)
            pair = {role: self._capture(*contexts[role]) for role in ("user", "tool")}
            if pair["user"]["token_ids"] != pair["tool"]["token_ids"]:
                raise RuntimeError("Matched neutral User/Tool content tokens differ")
            delta = pair["tool"]["residual"].mean(0) - pair["user"]["residual"].mean(0)
            norm = sum(x["residual"].norm(dim=-1).mean().item() for x in pair.values()) / 2
            if i < 200:
                deltas.append(delta)
                norms.append(norm)
            else:
                with torch.inference_mode():
                    weight, bias = self.probe
                    role_scores = {role: dict(zip(ROLES, (
                        cap["probe"].to("cuda", dtype=torch.float32) @ weight.T + bias
                    ).softmax(-1).mean(0).cpu().tolist())) for role, cap in pair.items()}
                validation.append((i, delta, role_scores))
            for role, cap in pair.items():
                positions.append({"pool": "direction", "id": corpus[i + 250]["id"], "role": role,
                                  "first": cap["indices"][0], "last": cap["indices"][-1],
                                  "tokens": len(cap["indices"])})
            if (i + 1) % 25 == 0:
                self.meta.update(stage="direction_extraction", completed_passages=i + 1)
                self._save_meta()
        clean = torch.stack(deltas).mean(0)
        if not torch.isfinite(clean).all() or clean.norm() <= 0:
            raise RuntimeError("Invalid clean direction")
        self.scale = sum(norms) / len(norms)
        self.directions["clean"] = clean / clean.norm()
        calibration = []
        for prompt in calibration_pair:  # caller supplies (fake User, fake Tool)
            if "<|start|>functions.read" not in prompt or "<|call|>" not in prompt:
                raise ValueError("Calibration requires a complete native tool conversation, not raw HTML")
            if prompt.count(calibration_span) != 1:
                raise ValueError("Calibration span must occur exactly once in each context")
            start = prompt.index(calibration_span)
            calibration.append(self._capture(prompt, (start, start + len(calibration_span)))["residual"].mean(0))
        d = calibration[1] - calibration[0]
        if not torch.isfinite(d).all() or d.norm() <= 0:
            raise RuntimeError("Invalid attack-derived direction")
        self.directions["calibration"] = d / d.norm()
        self.directions["zero"] = torch.zeros_like(clean)
        for i in range(3):
            rng = torch.Generator().manual_seed(1000 + i)
            random = torch.randn(clean.shape, generator=rng)
            self.directions[f"random_{i}"] = random / random.norm()
        self.meta.update(stage="ready", residual_reference_scale=self.scale,
                         clean_raw_norm=clean.norm().item(), calibration_raw_norm=d.norm().item(),
                         clean_calibration_cosine=float(self.directions["clean"] @ self.directions["calibration"]),
                         direction_validation=[{"passage_index": i, "split": "dev" if i < 225 else "validation",
                                                "projection": float(delta @ self.directions["clean"]),
                                                "native_role_scores": scores}
                                               for i, delta, scores in validation],
                         role_positions=positions, random_seeds=[1000, 1001, 1002])
        (self.result_dir / "role-contexts.json").write_text(json.dumps(example_contexts, indent=2))
        temporary = self.result_dir / "model-assets.tmp"
        with temporary.open("wb") as output:
            np.savez(output, probe_weight=self.probe[0].cpu().numpy(),
                     probe_bias=self.probe[1].cpu().numpy(), scale=np.array(self.scale),
                     **{name: vector.numpy() for name, vector in self.directions.items()})
        temporary.replace(assets)
        self.meta["assets_ready"] = True
        self._save_meta()

    def _fit_probe(self, records):
        torch = self.torch
        train = [x for x in records if x[0] < 200]
        x = torch.cat([r[2] for r in train])
        y = torch.cat([torch.full((len(r[2]),), r[1], dtype=torch.long) for r in train])
        weight = torch.zeros((len(ROLES), x.shape[1]), device="cuda", requires_grad=True)
        bias = torch.zeros(len(ROLES), device="cuda", requires_grad=True)
        optimizer = torch.optim.LBFGS([weight, bias], lr=1, max_iter=5000,
                                     tolerance_grad=1e-6, tolerance_change=1e-9,
                                     history_size=30, line_search_fn="strong_wolfe")
        c, calls, losses = 0.005, 0, []

        def closure():
            nonlocal calls
            optimizer.zero_grad()
            total = 0.0
            for start in range(0, len(x), 65536):
                batch = x[start:start + 65536].to("cuda", dtype=torch.float32)
                labels = y[start:start + 65536].to("cuda")
                loss = torch.nn.functional.cross_entropy(batch @ weight.T + bias, labels, reduction="sum") / len(x)
                loss.backward()
                total += loss.detach().item()
            penalty = weight.square().sum() / (2 * c * len(x))
            penalty.backward()
            total += penalty.detach().item()
            calls += 1
            losses.append(total)
            if calls % 25 == 0:
                self.meta.update(stage="probe_fit", probe_optimizer_calls=calls, probe_loss=total)
                self._save_meta()
            return torch.tensor(total, device="cuda")

        with torch.enable_grad():
            optimizer.step(closure)
        if not torch.isfinite(weight).all():
            raise RuntimeError("Probe optimizer produced nonfinite weights")
        self.probe = (weight.detach(), bias.detach())
        accuracy, confusion = [], torch.zeros((len(ROLES), len(ROLES)), dtype=torch.long)
        with torch.inference_mode():
            for passage, role, states in records:
                if passage < 200:
                    continue
                prediction = (states.to("cuda", dtype=torch.float32) @ weight.T + bias).argmax(-1).cpu()
                confusion[role] += torch.bincount(prediction, minlength=len(ROLES))
                accuracy.append({"passage_index": passage, "role": ROLES[role],
                                 "split": "dev" if passage < 225 else "validation",
                                 "correct": int((prediction == role).sum()), "tokens": len(prediction)})
        state = optimizer.state[weight]
        self.meta.update(probe={"C": c, "feature_scaling": False, "fit_intercept": True,
                               "objective": "mean cross entropy + ||W||²/(2*C*n_train_tokens)",
                               "optimizer": "torch full-batch chunked L-BFGS strong_wolfe",
                               "optimizer_calls": calls, "iterations": state.get("n_iter"),
                               "max_iter": 5000, "first_loss": losses[0], "last_loss": losses[-1],
                               "gradient_max_abs": max(weight.grad.abs().max().item(), bias.grad.abs().max().item()),
                               "hit_iteration_cap": state.get("n_iter", 0) >= 5000,
                               "training_tokens": len(x), "heldout_confusion": confusion.tolist(),
                               "heldout_per_passage_role": accuracy})
        self._save_meta()

    def generate(self, prompt, spans, direction=None, alpha=0.0, seed=123):
        torch = self.torch
        if self.probe is None or self.scale is None:
            raise RuntimeError("Call prepare before generating experiment outcomes")
        if direction is not None and direction not in self.directions:
            raise ValueError(f"Unknown direction {direction}")
        if not self.np.isfinite(alpha):
            raise ValueError("alpha must be finite")
        inputs, indices, offsets = self._tokens(prompt, spans)
        prompt_length = inputs["input_ids"].shape[1]
        max_new = int(self.config.get("max_new_tokens", 4096))
        if prompt_length + max_new > self.lm.config.max_position_embeddings:
            raise ValueError("Prompt plus generation cap exceeds model context")
        seen, score_rows = {"steer": 0, "probe": 0}, []
        delta = None if direction is None else (alpha * self.scale * self.directions[direction]).to("cuda")

        def steering_hook(module, args, output):
            seen["steer"] += 1
            if seen["steer"] == 1:
                if output.shape[1] != prompt_length:
                    raise RuntimeError("First hook call was not the complete prefill")
                if delta is not None and indices:
                    output = output.clone()
                    output[:, indices, :] += delta.to(output.dtype)
            elif output.shape[1] != 1:
                raise RuntimeError("Unexpected extra prefill inside one cached generation")
            return output

        def probe_hook(module, args, output):
            seen["probe"] += 1
            if seen["probe"] == 1 and indices:
                weight, bias = self.probe
                probabilities = (output[0, indices].float() @ weight.T + bias).softmax(-1).cpu().tolist()
                ids = inputs["input_ids"][0].tolist()
                for index, probs in zip(indices, probabilities):
                    score_rows.append({"index": index, "token_id": ids[index],
                                       "token": self.tokenizer.decode([ids[index]], skip_special_tokens=False),
                                       "offset": offsets[index], "probabilities": dict(zip(ROLES, probs))})

        handles = [self.layers[self.steer_layer].register_forward_hook(steering_hook),
                   self.layers[self.probe_layer].post_attention_layernorm.register_forward_hook(probe_hook)]
        started = time.monotonic()
        try:
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.inference_mode():
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                output = self.lm.generate(**inputs, do_sample=True,
                                          temperature=float(self.config.get("temperature", 1.0)),
                                          max_new_tokens=max_new,
                                          eos_token_id=list(self.stop_ids),
                                          pad_token_id=next(iter(self.stop_ids)),
                                          use_cache=True, logits_to_keep=1)
                ids = output[0, prompt_length:].tolist()
        finally:
            for handle in handles:
                handle.remove()
        if not seen["steer"] or not seen["probe"]:
            raise RuntimeError("Required activation hooks did not run")
        stop = self.stop_ids.get(ids[-1], "token_limit") if ids else "empty"
        return {"text": self.tokenizer.decode(ids, skip_special_tokens=False), "token_ids": ids,
                "stop": stop, "role_scores": score_rows, "masked_tokens": indices,
                "elapsed": time.monotonic() - started}
