"""I run my prepared Appendix K or RH6 readouts on the persistent CUDA worker.

I preserve the existing input rows, token/span selection, class order, float32
softmax scoring, and analysis functions. I replace the MLX forward pass with
the authors' verified CUDA custom forward at the same pre-MLP measurement site.
I never generate responses, execute the text being measured, or call a judge.

Examples (inside my deployed replication directory):
  python cloud/persistent/cuda_readouts.py appendix-k --repo /workspace/prompt-injection-as-role-confusion --probe-run /workspace/results/R/probes-full --input appendix-k/runs/neutral --out /workspace/results/R/appendix-k --hours 0.5
  python cloud/persistent/cuda_readouts.py rh6 --repo /workspace/prompt-injection-as-role-confusion --probe-run /workspace/results/R/probes-full --input rh/out/rh6-readings --out /workspace/results/R/rh6 --hours 0.5
  python cloud/persistent/cuda_readouts.py --self-test
"""
import argparse
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

AUTHORS_COMMIT = "ec333c40fd43fe991e1ebf66765051b6d7e35784"
MODEL_ID = "openai/gpt-oss-20b"
REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
ATTENTION = "kernels-community/vllm-flash-attn3"
ROOT = Path(__file__).resolve().parents[2]
ROLES = {"s": "system", "u": "user", "c": "cot", "a": "assistant", "t": "tool"}
ROLES5 = ["system", "user", "cot", "assistant", "tool"]


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_parquet(path, frame):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    with tmp.open("rb") as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def load_legacy(mode):
    path = ROOT / ("appendix-k/run_k.py" if mode == "appendix-k" else "rh/rh6_readings.py")
    spec = importlib.util.spec_from_file_location("readouts_legacy_" + mode.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def load_inputs(mode, source, limit=None):
    import pandas as pd
    source = Path(source)
    if mode == "appendix-k":
        frame = pd.read_parquet(source / "prompts.parquet")
        needed = {"prompt_ix", "condition", "ids", "tag_ix", "block_start", "block_end", "n_tokens"}
        if not needed.issubset(frame.columns):
            raise ValueError(f"My Appendix K input lacks {sorted(needed - set(frame.columns))}")
        if frame.duplicated(["prompt_ix", "condition"]).any():
            raise ValueError("My Appendix K prompt identities are duplicated")
        if limit:
            frame = frame[frame.prompt_ix.isin(sorted(frame.prompt_ix.unique())[:limit])]
        records = frame.reset_index(drop=True).to_dict("records")
        for row in records:
            row["ids"] = [int(x) for x in row["ids"]]
            row["tag_ix"] = [int(x) for x in row["tag_ix"]]
            if not row["ids"] or len(row["ids"]) != int(row["n_tokens"]):
                raise ValueError("My stored Appendix K token count does not match its IDs")
            if any(i < 0 or i >= len(row["ids"]) for i in row["tag_ix"]):
                raise ValueError("My Appendix K tag index is outside its prompt")
        files = [source / "prompts.parquet", source / "build_meta.json"]
    else:
        records = json.loads((source / "prefills.json").read_text())
        if limit:
            records = records[:limit]
        if len({row["item_id"] for row in records}) != len(records):
            raise ValueError("My RH6 item identities are duplicated")
        for row in records:
            for name in ("user_turn", "tool_result", "injection", "slot1", "slot2"):
                start, end = row["spans"][name]
                # I retain the legacy overlap rule when a saved tool-result end
                # extends past the string: only real tokenizer offsets can match.
                if not (0 <= start < len(row["prefill"]) and end > start):
                    raise ValueError(f"My RH6 character span is invalid: {row['item_id']} {name}")
        files = [source / "prefills.json"]
        if (source / "page_substitutions.json").exists():
            files.append(source / "page_substitutions.json")
    if not records:
        raise ValueError("My selected input is empty")
    return records, files


def token_spans(tokenizer, prefill, spans):
    """I keep the original RH6 overlap rule, including boundary-crossing tokens."""
    enc = tokenizer(prefill, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc.input_ids, enc.offset_mapping
    indices = {name: [i for i, (ts, te) in enumerate(offsets) if ts < end and te > start and te > ts]
               for name, (start, end) in spans.items()}
    return ids, offsets, indices


def load_probes(path, spaces, layers):
    import numpy as np
    probes = {}
    with np.load(path, allow_pickle=False) as archive:
        for space in spaces:
            roles = [ROLES[c] for c in space]
            for layer in layers:
                prefix = f"{space}_L{layer:02d}"
                coef = archive[prefix + "__coef"].astype(np.float32)
                intercept = archive[prefix + "__intercept"].astype(np.float32)
                if coef.shape != (len(roles), 2880) or intercept.shape != (len(roles),):
                    raise ValueError(f"My probe shape or class count is wrong for {prefix}: {coef.shape}, {intercept.shape}")
                if not np.isfinite(coef).all() or not np.isfinite(intercept).all():
                    raise ValueError(f"My probe has nonfinite coefficients: {prefix}")
                probes[(space, layer)] = coef, intercept, roles
    return probes


def probabilities(hidden, probe):
    import numpy as np
    coef, intercept, _ = probe
    z = np.asarray(hidden, dtype=np.float32) @ coef.T + intercept
    z = z - z.max(-1, keepdims=True)
    values = np.exp(z)
    values = values / values.sum(-1, keepdims=True)
    if not np.isfinite(values).all() or not np.allclose(values.sum(-1), 1, atol=1e-6):
        raise ValueError("My projected probabilities are nonfinite or do not sum to one")
    return values


def appendix_rows(row, tokenizer, projections):
    import numpy as np
    import pandas as pd
    ids = row["ids"]
    n = len(ids)
    tag_set = set(row["tag_ix"])
    block = np.zeros(n, dtype=bool)
    if row["block_start"] is not None and not pd.isna(row["block_start"]):
        block[int(row["block_start"]):int(row["block_end"]) + 1] = True
    base = pd.DataFrame({"prompt_ix": row["prompt_ix"], "condition": row["condition"],
                         "token_in_prompt_ix": np.arange(n), "token_id": ids,
                         "is_bos": np.array(ids) == tokenizer.bos_token_id, "in_block": block,
                         "is_tag": np.array([i in tag_set for i in range(n)])})
    parts = []
    for (space, layer), (p, roles) in projections.items():
        part = base.copy()
        part["layer"] = layer
        for j, role in enumerate(roles):
            part["p_" + role] = p[:, j]
        part["token"] = tokenizer.convert_ids_to_tokens(ids)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def rh6_rows(item, offsets, indices, projections):
    import numpy as np
    import pandas as pd
    injected = set(indices["injection"])
    page = [i for i in indices["tool_result"] if i not in injected]
    slots = set(indices["slot1"]) | set(indices["slot2"])
    keep = {"slot1": indices["slot1"], "slot2": indices["slot2"], "user_turn": indices["user_turn"],
            "wrapper": [i for i in indices["injection"] if i not in slots]}
    tokens, contexts = [], []
    for (space, layer), (p, roles) in projections.items():
        for span, positions in keep.items():
            command = item[f"{span}_cmd"] if span in ("slot1", "slot2") else span
            for j, i in enumerate(positions):
                row = {"item_id": item["item_id"], "layer": layer, "space": space, "span": span,
                       "command": command, "token_ix": i, "pos_in_span": j,
                       "token": item["prefill"][offsets[i][0]:offsets[i][1]]}
                row.update({f"p_{role}": float(p[i, roles.index(role)]) if role in roles else np.nan for role in ROLES5})
                tokens.append(row)
        mean = p[page].mean(0) if page else np.full(len(roles), np.nan)
        contexts.append({"item_id": item["item_id"], "layer": layer, "space": space, "span": "page",
                         "n_tokens": len(page), **{f"p_{role}": float(mean[roles.index(role)]) if role in roles else np.nan for role in ROLES5}})
    if not tokens:
        raise ValueError(f"My RH6 item has no scored span tokens: {item['item_id']}")
    return pd.DataFrame(tokens), pd.DataFrame(contexts)


def merge_parts(paths, destination):
    """I combine completed item files without retaining the entire run in RAM."""
    import pyarrow.parquet as pq
    destination = Path(destination)
    tmp = destination.with_suffix(".parquet.tmp")
    writer = None
    try:
        for path in paths:
            table = pq.read_table(path)
            if writer is None:
                writer = pq.ParquetWriter(tmp, table.schema, compression="snappy")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("I have no completed readout parts to combine")
    tmp.replace(destination)


def model_loader(repo):
    import torch
    sys.path.insert(0, str(repo))
    from utils.loader import load_model_and_tokenizer, load_custom_forward_pass
    if not torch.cuda.is_available():
        raise RuntimeError("I require the CUDA pod for extraction; --plan and --self-test are model-free")
    tokenizer, model, architecture, n_layers = load_model_and_tokenizer("gptoss-20b", device="cuda:0")
    dtype = str(model.model.layers[0].mlp.experts.down_proj.dtype)
    attn = str(model.model.config._attn_implementation)
    if any(t in dtype for t in ("bfloat16", "float16", "float32")):
        raise RuntimeError(f"My experts are dequantized: {dtype}")
    if attn != ATTENTION:
        raise RuntimeError(f"My attention differs from the approved author path: {attn}")
    forward = load_custom_forward_pass(architecture, model, tokenizer)
    from huggingface_hub import snapshot_download
    snapshot = Path(snapshot_download(MODEL_ID, cache_dir="/workspace/hf", local_files_only=True)).name
    meta = {"model_id": MODEL_ID, "snapshot": snapshot, "expected_snapshot": REVISION,
            "expert_dtype": dtype, "attention": attn, "n_layers": n_layers,
            "custom_forward_verified": True, "gpu": torch.cuda.get_device_name(0)}
    if snapshot != REVISION:
        raise RuntimeError(f"My model snapshot changed: {snapshot}; expected {REVISION}")
    return tokenizer, model, forward, meta


def analyze(mode, out, layers):
    legacy, _ = load_legacy(mode)
    if mode == "appendix-k":
        legacy.cmd_analyze(SimpleNamespace(run=str(out), layers=",".join(map(str, layers)),
                                         seed=123, n_boot=200, ewma_alpha=0.25))
    else:
        legacy.analyze(out, lambda message: print(message, flush=True))


def self_test():
    """I check scoring, span boundaries, saved-input contracts, and table assembly without a model."""
    import numpy as np
    import pandas as pd
    k_legacy, _ = load_legacy("appendix-k")
    rh_legacy, _ = load_legacy("rh6")
    rng = np.random.default_rng(123)
    hidden = rng.normal(size=(6, 2880)).astype(np.float32)
    probe = (rng.normal(size=(4, 2880)).astype(np.float32) * .01, np.zeros(4, np.float32), ROLES5[:4])
    p = probabilities(hidden, probe)
    np.testing.assert_array_equal(p, k_legacy.softmax(hidden @ probe[0].T + probe[1]))
    np.testing.assert_array_equal(p, rh_legacy.softmax(hidden @ probe[0].T + probe[1]))

    class FakeTokenizer:
        bos_token_id = 0

        def convert_ids_to_tokens(self, ids):
            return [f"t{i}" for i in ids]

        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[0, 1, 2, 3, 4, 5], offset_mapping=[(0, 0), (0, 2), (2, 4), (4, 6), (6, 8), (8, 10)])

    tokenizer = FakeTokenizer()
    row = {"prompt_ix": 0, "condition": "sys_t1", "ids": list(range(6)), "tag_ix": [1, 3], "block_start": 1, "block_end": 3}
    frame = appendix_rows(row, tokenizer, {("suca", 12): (p, ROLES5[:4])})
    assert frame.in_block.tolist() == [False, True, True, True, False, False]
    assert frame.is_tag.tolist() == [False, True, False, True, False, False]
    np.testing.assert_array_equal(frame[["p_system", "p_user", "p_cot", "p_assistant"]].to_numpy(), p)
    item = {"item_id": "test", "prefill": "abcdefghij", "slot1_cmd": "exfil", "slot2_cmd": "marker",
            "spans": {"user_turn": (0, 2), "tool_result": (2, 10), "injection": (4, 8), "slot1": (4, 6), "slot2": (6, 8)}}
    ids, offsets, indices = token_spans(tokenizer, item["prefill"], item["spans"])
    assert (ids, offsets, indices) == rh_legacy.token_spans(tokenizer, item["prefill"], item["spans"])
    tokens, contexts = rh6_rows(item, offsets, indices, {("suca", 12): (p, ROLES5[:4])})
    assert tokens.token_ix.tolist() == [3, 4, 1]
    assert contexts.n_tokens.tolist() == [2]
    np.testing.assert_allclose(contexts[["p_system", "p_user", "p_cot", "p_assistant"]].to_numpy()[0], p[[2, 5]].mean(0))
    assert tokens.p_tool.isna().all()
    with tempfile.TemporaryDirectory(prefix="cuda-readouts-selftest-") as directory:
        root = Path(directory)
        write_parquet(root / "one.parquet", frame)
        write_parquet(root / "two.parquet", frame)
        merge_parts([root / "one.parquet", root / "two.parquet"], root / "combined.parquet")
        pd.testing.assert_frame_equal(pd.read_parquet(root / "combined.parquet"), pd.concat([frame, frame], ignore_index=True))
    print("I passed the model-free scoring, span-boundary, token-label, and checkpoint-assembly checks.")


def run(args):
    import numpy as np
    import pandas as pd
    layers = [int(value) for value in args.layers.split(",")]
    if len(set(layers)) != len(layers) or any(l < 0 or l >= 24 for l in layers):
        raise ValueError("I require unique layer indices from 0 to 23")
    spaces = ["suca"] if args.mode == "appendix-k" else ["sucat", "uat"]
    records, input_files = load_inputs(args.mode, args.input, args.limit)
    source = Path(args.input).resolve()
    out = Path(args.out).resolve()
    if source == out:
        raise ValueError("I keep my prepared input and new CUDA output directories separate")
    probe_file = Path(args.probe_run) / ("probes-basesplit.npz" if args.probe_split == "base" else "probes.npz")
    _, legacy_path = load_legacy(args.mode)
    description = {"mode": args.mode, "n_items": len(records), "layers": layers, "spaces": spaces,
                   "probe_file": str(probe_file), "probe_present": probe_file.exists(), "probe_split": args.probe_split,
                   "input_sha256": {str(p): sha256(p) for p in input_files},
                   "total_stored_tokens": sum(len(r["ids"]) for r in records) if args.mode == "appendix-k" else None,
                   "n_templates": len({r["template_id"] for r in records}) if args.mode == "rh6" else None,
                   "inherited_span_ends_past_text": sum(end > len(r["prefill"]) for r in records for start, end in r["spans"].values()) if args.mode == "rh6" else 0}
    if args.plan:
        print(json.dumps(description, indent=2))
        return 0
    if args.analyze_only:
        if not (out / "EXTRACTION_DONE").exists():
            raise ValueError("I need complete extracted readouts before running the original summaries")
        analyze(args.mode, out, layers)
        write_json(out / "ANALYSIS_DONE", {"time_utc": utc_now()})
        return 0
    probes = load_probes(probe_file, spaces, layers)
    repo = Path(args.repo).resolve()
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if commit != AUTHORS_COMMIT:
        raise ValueError(f"My authors' checkout is {commit}; I require {AUTHORS_COMMIT}")
    description.update(probe_sha256=sha256(probe_file), source_sha256=sha256(__file__),
                       legacy_analysis_sha256=sha256(legacy_path), authors_commit=commit, limit=args.limit)
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract = out / "input-contract.json"
    if contract.exists() and json.loads(contract.read_text()) != description:
        raise ValueError("My input, probe, source, or selected rows changed; I require a new output directory")
    write_json(contract, description)
    if args.mode == "appendix-k":
        pd.DataFrame(records).to_parquet(out / "prompts.parquet", index=False)
        shutil.copy2(source / "build_meta.json", out / "build_meta.json")
    else:
        write_json(out / "prefills.json", records)
        if (source / "page_substitutions.json").exists():
            shutil.copy2(source / "page_substitutions.json", out / "page_substitutions.json")
    parts = out / "parts"
    parts.mkdir(exist_ok=True)
    completed = {i for i in range(len(records)) if (parts / f"item-{i:05d}.json").exists()}
    for index in completed:
        expected = [parts / f"tokens-{index:05d}.parquet"]
        if args.mode == "rh6":
            expected.append(parts / f"page-{index:05d}.parquet")
        if not all(path.exists() for path in expected):
            raise ValueError(f"My completed item {index} is missing its probability checkpoint")
    metadata = {**description, "status": "running", "started_utc": utc_now(), "backend": "CUDA/Transformers",
                "site": "post_attention_layernorm output (pre-MLP)", "batch_size": 1, "padding": "none",
                "hidden_state_cast": "native author pre-MLP output to float32; no new float16 storage round trip",
                "probabilities": "float32 affine map and stable softmax; original role ordering",
                "n_completed": len(completed), "hours": args.hours,
                "divergences": ["CUDA author loader/custom forward replaces my MLX forward pass",
                                "I reuse cloud cuML-trained probe coefficients through their portable NPZ export",
                                "I preserve the prepared prompts and controls; these include previously documented differences from the paper"]}
    if (out / "metadata.json").exists():
        prior = json.loads((out / "metadata.json").read_text())
        if "model" in prior:
            metadata["model"] = prior["model"]
        metadata["previous_attempts"] = prior.get("previous_attempts", []) + [
            {key: prior[key] for key in ("started_utc", "finished_utc", "status", "error", "n_completed", "elapsed_seconds") if key in prior}]
    metadata["versions"] = {}
    for package in ("numpy", "pandas", "pyarrow", "torch", "transformers", "kernels", "triton"):
        try:
            metadata["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata["versions"][package] = None
    write_json(out / "metadata.json", metadata)
    (out / "DONE").unlink(missing_ok=True)
    (out / "FAILED").unlink(missing_ok=True)

    def heartbeat(phase):
        write_json(out / "heartbeat", {"time_utc": utc_now(), "phase": phase,
                                      "completed": len(completed), "planned": len(records)})

    def interrupted(signum, frame):
        raise TimeoutError("My readout stage reached its time limit or received a stop signal")

    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
    signal.alarm(max(1, int(args.hours * 3600)))
    started = time.monotonic()
    try:
        if len(completed) < len(records):
            heartbeat("loading")
            import torch
            tokenizer, model, forward, model_meta = model_loader(repo)
            metadata["model"] = model_meta
            write_json(out / "metadata.json", metadata)
            for index, item in enumerate(records):
                if index in completed:
                    continue
                if (out / "STOP").exists():
                    raise InterruptedError("I received the stage STOP file")
                heartbeat("forward")
                if args.mode == "appendix-k":
                    ids = item["ids"]
                    if tokenizer.bos_token_id != json.loads((out / "build_meta.json").read_text())["bos_id"]:
                        raise ValueError("My loaded tokenizer BOS differs from the frozen Appendix K input")
                    offsets = indices = None
                else:
                    ids, offsets, indices = token_spans(tokenizer, item["prefill"], item["spans"])
                    if not all(indices[name] for name in ("slot1", "slot2", "user_turn", "injection")):
                        raise ValueError(f"My RH6 item has an empty required token span: {item['item_id']}")
                inputs = torch.tensor([ids], dtype=torch.long, device=model.device)
                attention = torch.ones_like(inputs)
                with torch.no_grad():
                    result = forward(model, inputs, attention, return_hidden_states=True)
                projections = {}
                for layer in layers:
                    hidden = result["all_pre_mlp_hidden_states"][layer].to(torch.float32).numpy()
                    if hidden.shape != (len(ids), 2880):
                        raise ValueError(f"My hidden-state shape differs from the input: {hidden.shape}")
                    for space in spaces:
                        projections[(space, layer)] = probabilities(hidden, probes[(space, layer)]), probes[(space, layer)][2]
                next_token = tokenizer.decode([int(result["logits"][0, -1].argmax().item())])
                if args.mode == "appendix-k":
                    tokens = appendix_rows(item, tokenizer, projections)
                else:
                    tokens, contexts = rh6_rows(item, offsets, indices, projections)
                    write_parquet(parts / f"page-{index:05d}.parquet", contexts)
                write_parquet(parts / f"tokens-{index:05d}.parquet", tokens)
                item_meta = {"index": index, "n_tokens": len(ids), "next_token": next_token,
                             "time_utc": utc_now(), "token_rows": len(tokens)}
                if args.mode == "rh6":
                    item_meta.update(item_id=item["item_id"], ids=ids, offsets=offsets, span_indices=indices)
                else:
                    item_meta.update(prompt_ix=int(item["prompt_ix"]), condition=item["condition"])
                write_json(parts / f"item-{index:05d}.json", item_meta)
                completed.add(index)
                metadata.update(n_completed=len(completed), elapsed_seconds=round(time.monotonic() - started, 2))
                write_json(out / "metadata.json", metadata)
                heartbeat("checkpoint")
                if len(completed) % 10 == 0 or len(completed) == len(records):
                    print(f"I completed {len(completed)}/{len(records)} {args.mode} inputs in {time.monotonic() - started:.1f}s", flush=True)
                del result, inputs, attention, projections, tokens
            del model
            torch.cuda.empty_cache()
        heartbeat("assembling")
        merge_parts([parts / f"tokens-{i:05d}.parquet" for i in range(len(records))], out / "tokens.parquet")
        if args.mode == "rh6":
            merge_parts([parts / f"page-{i:05d}.parquet" for i in range(len(records))], out / "page_means.parquet")
        write_json(out / "EXTRACTION_DONE", {"time_utc": utc_now(), "n_items": len(records)})
        metadata["extraction_complete"] = True
        if not args.no_analyze:
            heartbeat("analysis")
            analyze(args.mode, out, layers)
            write_json(out / "ANALYSIS_DONE", {"time_utc": utc_now()})
            metadata["analysis_complete"] = True
        metadata.update(status="complete", finished_utc=utc_now(), elapsed_seconds=round(time.monotonic() - started, 2))
        write_json(out / "metadata.json", metadata)
        heartbeat("complete")
        write_json(out / "DONE", {"time_utc": utc_now(), "n_items": len(records)})
        (out / "FAILED").unlink(missing_ok=True)
        return 0
    except BaseException as error:
        metadata.update(status="partial" if isinstance(error, (TimeoutError, InterruptedError)) else "failed",
                        error=f"{type(error).__name__}: {error}", n_completed=len(completed),
                        finished_utc=utc_now(), elapsed_seconds=round(time.monotonic() - started, 2))
        write_json(out / "metadata.json", metadata)
        write_json(out / "FAILED", metadata)
        (out / "DONE").unlink(missing_ok=True)
        heartbeat(metadata["status"])
        raise
    finally:
        signal.alarm(0)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=["appendix-k", "rh6"])
    parser.add_argument("--repo", default="/workspace/prompt-injection-as-role-confusion")
    parser.add_argument("--probe-run")
    parser.add_argument("--input")
    parser.add_argument("--out")
    parser.add_argument("--layers", default="8,12,16")
    parser.add_argument("--probe-split", choices=["prompt", "base"], default="prompt")
    parser.add_argument("--hours", type=float, default=0.5)
    parser.add_argument("--limit", type=int, help="I select the first N base prompts for K or the first N prefills for RH6")
    parser.add_argument("--plan", action="store_true", help="I inspect the frozen inputs without loading a model")
    parser.add_argument("--no-analyze", action="store_true", help="I retain readouts and run the original summaries separately")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.mode or not args.probe_run or not args.input or not args.out:
        parser.error("I require mode, --probe-run, --input, and --out")
    if not 1 / 60 <= args.hours <= 6 or (args.limit is not None and args.limit < 1):
        parser.error("I require --hours between 1/60 and 6, and a positive --limit")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
