#!/usr/bin/env python
"""Headless reproduction of the authors' NB02 cells 1-20 for gptoss-20b, on a CUDA pod.

Source notebook: prompt-injection-as-role-confusion/experiments/role-analysis/02-train-role-probes.ipynb
at commit ec333c40fd43fe991e1ebf66765051b6d7e35784. The authors' utils are imported from a clone of
that repo (--repo); nothing in the probe definition is re-implemented: model loader, custom forward
pass, dataset streaming, sequence rendering, role labelling, fit_lr, get_probe_result and the save
cell are the notebook's own code, copied verbatim or imported.

Deliberate extensions, each behind an argument and recorded in metadata.json:
  --layers all      probe every layer 0..23 (authors: range(0, 24, 2)).
  --splits prompt,base
                    'prompt' is the authors' split (cuml.train_test_split on unique prompt_ix,
                    test_size 0.1, random_state 123). 'base' is the same call on unique question_ix
                    (the base text), so the five role renderings of one text never straddle the split.
  --attn auto|fa3|eager
                    the authors load kernels-community/vllm-flash-attn3 (Hopper only). 'auto' keeps it
                    on compute capability 9.x and falls back to eager elsewhere (A100), logged loudly.
  --batch-size 32   I retain the authors' batch size by default; a smaller batch bounds peak GPU
                    memory while preserving prompt order, padding, dataset, splits and fitting.

One memory-driven deviation from the notebook's control flow, not from its numerics: cell 14 keeps
the whole activation cube in host RAM (49 GB for 12 layers, twice that for 24, plus a float16 copy).
This script runs the same forward function batch by batch and writes each layer's pre-MLP states
straight into a float16 .npy memmap (layerNN.npy, rows in sample_ix order, the layout of
replication/probes/extract_activations.py). Fitting then loads one layer at a time. The values fed to
cuML are identical to the notebook's (same tensor, same bf16 -> fp16 -> fp32 casts, same row order);
only the residency changes. Probe order in the saved pickle is restored to the notebook's
(role space outer, layer inner).

Outputs (--results, the outbox that the Mac rsyncs):
  gptoss-20b.pkl, gptoss-20b-basesplit.pkl          authors' pickle (cuML Pipeline per probe)
  role_probes.pkl, role_probes-basesplit.pkl        portable sklearn clone (coef_/intercept_ copied)
  probes.npz/.json, probes-basesplit.npz/.json      our format ({space}_L{layer:02d}__coef/__intercept)
  acc_by_role_gptoss-20b[-basesplit].csv, acc_by_pos_gptoss-20b[-basesplit].csv   as cell 20 writes
  val_acc_by_layer[-basesplit].csv                   the cell 20 pivot
  tokens.parquet, prompts.parquet, metadata.json, heartbeat, progress.json, DONE or FAILED
Activations (--acts, stays on the pod volume): layerNN.npy float16, tokens.parquet, metadata.json.

Usage on the pod (see run_probes_job.sh):
  python train_probes_pod.py --check-env
  python train_probes_pod.py --pilot 10 --results /workspace/results/R/pilot --acts /workspace/acts/R-pilot --hours 0.5
  python train_probes_pod.py --results /workspace/results/R --acts /workspace/acts/R --hours 2.5
Pure-Python parts can be checked anywhere:  python train_probes_pod.py --self-test
"""
import argparse
import gc
import importlib
import json
import os
import pickle
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

MODEL_PREFIX = "gptoss-20b"
MODEL_ID = "openai/gpt-oss-20b"
EXPECTED_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
AUTHORS_COMMIT = "ec333c40fd43fe991e1ebf66765051b6d7e35784"
AUTHORS_ATTN = "kernels-community/vllm-flash-attn3"
SEED = 123                       # NB02 cell 1
MAIN_DEVICE = "cuda:0"
BATCH_SIZE = 32                  # NB02 cell 12
N_LAYERS = 24
ROLE_CHAR = {"system": "s", "user": "u", "cot": "c", "assistant": "a", "tool": "t"}
# NB02 cell 19, the gptoss-20b branch, in the notebook's order.
ALL_ROLE_COMBINATIONS = [
    ("user", "assistant"),
    ("user", "assistant", "tool"),
    ("user", "cot", "assistant"),
    ("user", "cot", "assistant", "tool"),
    ("system", "user", "assistant"),
    ("system", "user", "assistant", "tool"),
    ("system", "user", "cot", "assistant"),
    ("system", "user", "cot", "assistant", "tool"),
]
VERSION_PACKAGES = ["torch", "transformers", "kernels", "triton", "flash_attn", "cuml", "cudf", "cupy",
                    "scikit-learn", "numpy", "pandas", "datasets", "zstandard", "accelerate", "tiktoken",
                    "huggingface_hub", "pyarrow"]


# ----------------------------------------------------------------------------- pure-Python helpers

def positive_int(text):
    """I reject zero, negative and fractional batch sizes before loading a model."""
    try:
        value = int(text)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def parse_layers(text):
    """'all' -> 0..23, 'authors' -> 0,2,..,22 (cell 14 for a 24-layer model), else a comma list."""
    if text == "all":
        return list(range(N_LAYERS))
    if text == "authors":
        return list(range(0, N_LAYERS, 2))
    layers = sorted({int(x) for x in text.split(",") if x.strip()})
    bad = [l for l in layers if not 0 <= l < N_LAYERS]
    if bad:
        raise ValueError(f"layers out of range: {bad}")
    return layers


def parse_splits(text):
    splits = [s.strip() for s in text.split(",") if s.strip()]
    bad = [s for s in splits if s not in ("prompt", "base")]
    if bad:
        raise ValueError(f"unknown splits {bad}; use prompt,base")
    return splits


def split_tag(split):
    return "" if split == "prompt" else "-basesplit"


def space_abbrev(roles):
    return "".join(ROLE_CHAR[r] for r in roles)


def npz_key(roles, layer_ix, part):
    return f"{space_abbrev(roles)}_L{layer_ix:02d}__{part}"


def authors_order_key(probe):
    """Sort key restoring the notebook's list order: role space outer loop, layer inner loop."""
    return (ALL_ROLE_COMBINATIONS.index(tuple(probe["role_space"])), probe["layer_ix"])


def to_numpy(x):
    """cuML exposes coef_/intercept_ as cupy arrays that refuse implicit host conversion."""
    import numpy as np
    if hasattr(x, "get"):
        return np.asarray(x.get())
    if hasattr(x, "to_numpy"):
        return np.asarray(x.to_numpy())
    return np.asarray(x)


def coef_arrays(probe_obj):
    clf = probe_obj.named_steps["clf"] if hasattr(probe_obj, "named_steps") else probe_obj
    return to_numpy(clf.coef_), to_numpy(clf.intercept_)


def make_sklearn_clone(coef, intercept, C):
    """Portable estimator: reference REPORT.md section 4 with its two corrections (cupy .get(),
    binary probes carry one coefficient row for two classes)."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    coef = np.asarray(coef, dtype=np.float64)
    intercept = np.asarray(intercept, dtype=np.float64).reshape(-1)
    if coef.ndim == 1:
        coef = coef[None, :]
    n_rows, n_features = coef.shape
    n_classes = 2 if n_rows == 1 else n_rows
    sk = LogisticRegression(C=C, fit_intercept=True, max_iter=5000)
    sk.classes_ = np.arange(n_classes)
    sk.coef_ = coef
    sk.intercept_ = intercept
    sk.n_features_in_ = n_features
    return sk


def our_format(all_probes, split, C, extra_meta):
    """The npz dict and the json document of replication/probes/fit_probes_mlx.py."""
    import numpy as np
    arrays, results = {}, []
    for p in all_probes:
        coef, intercept = coef_arrays(p["probe"])
        roles = list(p["role_space"])
        arrays[npz_key(roles, p["layer_ix"], "coef")] = coef.astype(np.float32)
        arrays[npz_key(roles, p["layer_ix"], "intercept")] = intercept.astype(np.float32).reshape(-1)
        results.append({
            "role_space": space_abbrev(roles), "layer_ix": int(p["layer_ix"]), "acc": round(float(p["acc"]), 4),
            "nll": round(float(p["nll"]), 6), "acc_by_role": p.get("acc_by_role_dict"),
            "n_train": p.get("n_train"), "n_test": p.get("n_test"), "n_inputs": int(p["n_inputs"]),
            "coef_shape": list(coef.shape), "sec": p.get("sec"),
        })
    doc = {
        "split_by": split, "C": C, "penalty": "l2", "fit_intercept": True, "max_iter": 5000, "linesearch_max_iter": 100,
        "objective": "cuML L2 logistic regression, the authors' fit_lr unchanged (NB02 cell 18)",
        "solver": "cuml.linear_model.LogisticRegression (QN / L-BFGS), inside sklearn.pipeline.Pipeline",
        "split_rule": ("cuml.train_test_split(unique prompt_ix, test_size=0.1, random_state=123) [authors]" if split == "prompt"
                       else "cuml.train_test_split(unique question_ix, test_size=0.1, random_state=123) [extension: grouped by base text]"),
        "binary_convention": "one coefficient row for two classes (cuML/sklearn): logit > 0 means class index 1 of roles_map",
        "role_char": {v: k for k, v in ROLE_CHAR.items()}, "results": results,
        "source": f"NB02 cells 1-20, commit {AUTHORS_COMMIT}", **extra_meta,
    }
    return arrays, doc


def package_versions():
    from importlib.metadata import PackageNotFoundError, version
    out = {"python": platform.python_version()}
    dist = {"cuml": "cuml-cu12", "cudf": "cudf-cu12", "cupy": "cupy-cuda12x"}  # distribution names differ from import names
    for name in VERSION_PACKAGES:
        try:
            out[name] = version(dist.get(name, name))
        except PackageNotFoundError:
            out[name] = None
    return out


def nvidia_smi():
    try:
        text = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        return text or None
    except (OSError, subprocess.SubprocessError):
        return None


def compute_capability():
    try:
        import torch
        major, minor = torch.cuda.get_device_capability(0)
        return f"{major}.{minor}"
    except Exception:  # noqa: BLE001
        return None


def resolve_attn(choice):
    """'auto': the authors' FA3 kernel on Hopper (sm 9.x), eager elsewhere."""
    if choice == "fa3":
        return AUTHORS_ATTN
    if choice == "eager":
        return "eager"
    cap = compute_capability() or ""
    return AUTHORS_ATTN if cap.startswith("9.") else "eager"


def peak_rss_gb():
    try:
        import resource
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(kb / 1024 / 1024, 2)  # Linux reports KB
    except Exception:  # noqa: BLE001
        return None


class Outbox:
    """Heartbeat (the Mac watchdog's format: JSON with time_utc), progress, log, markers."""

    def __init__(self, results_dir, heartbeat_file=None):
        self.dir = Path(results_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.heartbeat_file = Path(heartbeat_file) if heartbeat_file else self.dir / "heartbeat"
        self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.dir / "train.log", "a")
        self.t0 = time.time()

    def log(self, msg):
        line = f"[{time.time() - self.t0:7.1f}s] {msg}"
        print(line, flush=True)
        self.log_file.write(line + "\n")
        self.log_file.flush()

    def heartbeat(self, phase, **extra):
        record = {"time_utc": utc_now(), "pid": os.getpid(), "phase": phase, **extra}
        tmp = self.heartbeat_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(record) + "\n")
        tmp.replace(self.heartbeat_file)

    def progress(self, **fields):
        (self.dir / "progress.json").write_text(json.dumps({"time_utc": utc_now(), **fields}, indent=2))

    def marker(self, name, text=""):
        (self.dir / name).write_text(text + "\n")


class Deadline:
    """--hours guard: armed after the model loads (playbook fix 6). SIGTERM from `timeout` is handled the same way."""

    def __init__(self, hours):
        self.hours = float(hours)

    def arm(self):
        def on_alarm(signum, frame):
            raise TimeoutError(f"--hours {self.hours} reached")

        def on_term(signum, frame):
            raise InterruptedError("SIGTERM")

        signal.signal(signal.SIGALRM, on_alarm)
        signal.signal(signal.SIGTERM, on_term)
        signal.alarm(int(self.hours * 3600))


# ----------------------------------------------------------------------------- the notebook, cell by cell

def run(args):
    import numpy as np
    import pandas as pd
    import torch
    import yaml

    out = Outbox(args.results, args.heartbeat_file)
    acts = Path(args.acts)
    acts.mkdir(parents=True, exist_ok=True)
    layers_to_probe = parse_layers(args.layers)
    splits = parse_splits(args.splits)
    batch_size = args.batch_size
    timings, meta = {}, {"args": vars(args), "started_utc": utc_now(), "versions": package_versions(),
                         "nvidia_smi": nvidia_smi(), "compute_capability": compute_capability(),
                         "authors_commit_expected": AUTHORS_COMMIT, "hostname": platform.node(),
                         "runpod_pod_id": os.environ.get("RUNPOD_POD_ID"),
                         "batch_size": batch_size, "authors_batch_size": BATCH_SIZE}
    out.log(f"versions: {json.dumps(meta['versions'])}")
    out.log(f"gpu: {meta['nvidia_smi']}")
    out.heartbeat("start")

    # --- repo on sys.path (the notebook relies on setup_python.sh's .pth for `utils.*`)
    repo = Path(args.repo).resolve()
    if not (repo / "utils" / "loader.py").exists():
        raise SystemExit(f"authors' repo not found at {repo}")
    sys.path.insert(0, str(repo))
    try:
        meta["authors_commit"] = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                                text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        meta["authors_commit"] = None
    if meta["authors_commit"] and meta["authors_commit"] != AUTHORS_COMMIT and not args.allow_other_commit:
        raise SystemExit(f"repo is at {meta['authors_commit']}, expected {AUTHORS_COMMIT} (pass --allow-other-commit to override)")
    ws = str(repo)  # cell 1: ws = '/workspace/deliberative-alignment-jailbreaks'
    probe_cfg = yaml.safe_load(open(repo / "experiments/role-analysis/config/probe.yaml"))[MODEL_PREFIX]
    meta["probe_yaml"] = {k: probe_cfg[k] for k in ("n_sample_size", "seq_len", "nested_reasoning", "train_prefixes", "train_params")}

    # --- attention implementation (extension: eager fallback off Hopper)
    attn = resolve_attn(args.attn)
    meta["attn_requested"] = attn
    import utils.loader as loader_mod
    if attn != AUTHORS_ATTN:
        out.log(f"!! attention fallback: {attn} (authors: {AUTHORS_ATTN}, Hopper only; this GPU is sm {meta['compute_capability']}) !!")
        original = loader_mod.get_supported_model_metadata

        def patched(model_prefix):
            model_id, arch, _, use_hf, n_layers = original(model_prefix)
            return model_id, arch, attn, use_hf, n_layers
        loader_mod.get_supported_model_metadata = patched

    # ===== cell 1: imports
    import cupy
    import cuml
    import sklearn
    from datasets import load_dataset
    from tqdm import tqdm
    from utils.memory import check_memory, clear_all_cuda_memory
    from utils.loader import load_model_and_tokenizer, load_custom_forward_pass
    from utils.probes import check_max_seq_len
    main_device = MAIN_DEVICE
    seed = SEED
    clear_all_cuda_memory()
    check_memory()

    def display(obj):
        print(obj.to_string() if hasattr(obj, "to_string") and len(getattr(obj, "index", [])) <= 200 else obj, flush=True)

    # ===== cell 3: load model (the loader prints expert precision and attention implementation)
    t = time.time()
    out.heartbeat("load_model")
    model_prefix = MODEL_PREFIX
    tokenizer, model, model_architecture, model_n_layers = load_model_and_tokenizer(model_prefix, device=main_device)
    check_memory()
    timings["load_model_s"] = round(time.time() - t, 1)
    expert_dtype = str(model.model.layers[0].mlp.experts.down_proj.dtype)
    attn_loaded = str(model.model.config._attn_implementation)
    meta.update(expert_dtype=expert_dtype, attn_implementation=attn_loaded, model_n_layers=model_n_layers)
    out.log(f"expert dtype: {expert_dtype}; attention: {attn_loaded}")
    if "bfloat16" in expert_dtype or "float16" in expert_dtype or "float32" in expert_dtype:
        msg = f"experts are {expert_dtype}, not MXFP4 (transformers dequantized the model)"
        if args.require_mxfp4:
            raise SystemExit(msg)
        out.log("!! " + msg)
    if attn_loaded != AUTHORS_ATTN and attn == AUTHORS_ATTN:
        raise SystemExit(f"asked for {AUTHORS_ATTN} but the model loaded {attn_loaded}; on Hopper this means the kernel install is wrong")
    try:
        from huggingface_hub import snapshot_download
        snap = Path(snapshot_download(MODEL_ID, cache_dir="/workspace/hf", local_files_only=True))
        meta["hf_snapshot"] = snap.name
        if snap.name != EXPECTED_REVISION:
            out.log(f"!! HF snapshot {snap.name} differs from the reference {EXPECTED_REVISION}")
    except Exception as error:  # noqa: BLE001
        meta["hf_snapshot"] = f"unresolved: {type(error).__name__}"
    meta["model_n_layers"] = model_n_layers
    if any(l >= model_n_layers for l in layers_to_probe):
        raise SystemExit(f"layers {layers_to_probe} exceed model_n_layers {model_n_layers}")

    Deadline(args.hours).arm()  # armed after the model load
    out.log(f"--hours guard armed: {args.hours} h")

    # ===== cell 4: custom forward pass, verified equal to the HF forward pass (asserts inside)
    t = time.time()
    run_forward_with_hs = load_custom_forward_pass(model_architecture, model, tokenizer)
    timings["verify_forward_s"] = round(time.time() - t, 1)
    meta["custom_forward_verified"] = True

    if args.check_model:
        meta.update(finished_utc=utc_now(), timings=timings, check_model_only=True)
        (out.dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        out.log("check-model done")
        return 0

    # ===== cell 5: test generation is sensible
    if not args.skip_gen_tests:
        def test_generation():
            conv = tokenizer.apply_chat_template([{"role": "user", "content": "Write a haiku about GPUs"}], tokenize=False,
                                                 enable_thinking=True, add_generation_prompt=True)
            inputs = tokenizer(conv, return_tensors="pt")
            gen_ids = model.generate(inputs["input_ids"].to(main_device), max_new_tokens=100, do_sample=False)
            print(tokenizer.batch_decode(gen_ids, skip_special_tokens=False)[0], flush=True)
        test_generation()

    # ===== cell 8: load SFT dataset
    out.heartbeat("load_dataset")
    n_sample_size = int(args.pilot) if args.pilot else probe_cfg["n_sample_size"]

    def load_raw_ds():
        def get_c4():
            return load_dataset("allenai/c4", "en", split="validation", streaming=True).shuffle(seed=seed, buffer_size=50_000)

        def get_dolma3():
            return load_dataset("allenai/dolma3_mix-150B-1025", split="train", revision="3a8349c", streaming=True).shuffle(seed=seed, buffer_size=50_000)

        def get_data(ds, n_samples, data_source):
            raw_data = []
            ds_iter = iter(ds)
            for _ in range(n_samples):
                sample = next(ds_iter, None)
                if sample is None:
                    break
                raw_data.append({"text": sample["text"], "source": data_source})
            return raw_data
        return get_data(get_c4(), int(n_sample_size * .25), "c4") + get_data(get_dolma3(), int(n_sample_size * .75), "dolma3")

    t = time.time()
    raw_data = load_raw_ds()
    timings["load_dataset_s"] = round(time.time() - t, 1)
    meta["n_docs"] = {"total": len(raw_data), **pd.Series([r["source"] for r in raw_data]).value_counts().to_dict()}
    out.log(f"raw docs: {meta['n_docs']} (n_sample_size {n_sample_size})")

    # ===== cell 9: test rendering
    import utils.role_templates
    importlib.reload(utils.role_templates)
    from utils.role_templates import render_single_message, render_mixed_cot

    def test_render():
        print(tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hi! I am a dog and I like to bark"}, {"role": "assistant", "content": "Hello! What a lovely dog you are!"}],
            tokenize=False, padding="max_length", truncation=True, max_length=512, add_generation_prompt=True, enable_thinking=True))
        print("--")
        print(render_single_message(model_prefix, role="user", content="Hi"))
        print("--")
        print(render_mixed_cot(model_prefix, "The user...", "Yes!"))
        return True
    test_render()

    # ===== cell 10: validate forward passes work
    train_prefixes = probe_cfg["train_prefixes"]
    for p in train_prefixes:
        print(p)

    @torch.no_grad()
    def test1():
        conv = tokenizer.apply_chat_template([{"role": "user", "content": "Hi stinky"}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(conv, add_special_tokens=True, return_tensors="pt")
        gen_ids = model.generate(inputs["input_ids"].to(main_device), attention_mask=inputs["attention_mask"].to(main_device), max_new_tokens=500, do_sample=False)
        df = pd.DataFrame({"token_id": gen_ids[0].tolist(), "token": tokenizer.convert_ids_to_tokens(gen_ids[0].tolist())})
        print(tokenizer.decode(df["token_id"].tolist(), skip_special_tokens=False))
        return df

    @torch.no_grad()
    def test2():
        for tr_prefix in train_prefixes:
            conv = tr_prefix + render_single_message(model_prefix, "user", "Where is Atlanta?") + "<|start|>assistant<|channel|>analysis<|message|>"
            inputs = tokenizer(conv, add_special_tokens=False, return_tensors="pt")
            gen_ids = model.generate(inputs["input_ids"].to(main_device), attention_mask=inputs["attention_mask"].to(main_device), max_new_tokens=500, do_sample=False)
            print(tokenizer.decode(gen_ids[0], skip_special_tokens=False))
            print("\n")

    if not args.skip_gen_tests:
        t = time.time()
        test1()
        test2()
        timings["gen_tests_s"] = round(time.time() - t, 1)

    # ===== cell 11: create sample sequences
    SEQLEN = probe_cfg["seq_len"]
    NESTED_REASONING = probe_cfg["nested_reasoning"]
    GENERALIZE_PREFIX = False

    def get_sample_seqs_for_input_seq(probe_text, partner_text, prefix=""):
        seqs = []
        gen_prefix = partner_text if GENERALIZE_PREFIX else ""
        if model_prefix in ["gptoss-20b"]:
            seqs.append({"role": "system", "prompt": prefix + gen_prefix + render_single_message(model_prefix, role="system", content=probe_text)})
        for role in ["user", "tool", "cot"]:
            seqs.append({"role": role, "prompt": prefix + gen_prefix + render_single_message(model_prefix, role=role, content=probe_text)})
        if NESTED_REASONING:
            seqs.append({"role": "assistant", "prompt": prefix + render_mixed_cot(model_prefix, cot=partner_text, assistant=probe_text)})
        else:
            seqs.append({"role": "assistant", "prompt": prefix + gen_prefix + render_single_message(model_prefix, role="assistant", content=probe_text)})
        return seqs

    def build_sample_seqs(train_prefixes):
        truncated_texts = tokenizer.batch_decode(tokenizer([t["text"] for t in raw_data], add_special_tokens=False, padding=False, truncation=True, max_length=SEQLEN).input_ids)
        n_seqs = len(truncated_texts)
        np.random.seed(seed)
        partner_lengths = ((np.random.beta(0.5, 4.0, size=n_seqs) * (SEQLEN / 2 + 1)).astype(int)).tolist()
        partner_texts = [
            tokenizer.decode(tokenizer(text["text"], add_special_tokens=False, padding=False, truncation=True, max_length=int(partner_lengths[i])).input_ids)
            for i, text in enumerate(raw_data)
        ]
        perm = np.random.permutation(n_seqs)
        while np.any(perm == np.arange(n_seqs)):
            perm = np.random.permutation(n_seqs)
        sampled_prefixes = np.random.choice(train_prefixes, size=n_seqs)
        input_list = []
        for base_ix, base_text in enumerate(truncated_texts):
            partner_text = partner_texts[int(perm[base_ix])].strip()
            prefix = sampled_prefixes[base_ix]
            for seq in get_sample_seqs_for_input_seq(base_text, partner_text, prefix):
                row = {"question_ix": base_ix, "question": base_text, **seq}
                input_list.append(row)
        input_df = pd.DataFrame(input_list).assign(prompt_ix=lambda df: list(range(len(df))))
        return input_df

    input_df = build_sample_seqs(train_prefixes=train_prefixes)
    display(input_df)
    for p in [row["prompt"] for row in input_df.pipe(lambda df: df[df["question_ix"] == 2]).to_dict("records")]:
        print(p)
        print("=" * 80)
    input_df.to_parquet(out.dir / "prompts.parquet", index=False)
    input_df.to_parquet(acts / "prompts.parquet", index=False)
    meta["n_prompts"] = int(len(input_df))

    # ===== cell 12: dataloader
    from utils.dataset import ReconstructableTextDataset, stack_collate
    from torch.utils.data import DataLoader
    max_seqlen = check_max_seq_len(tokenizer, input_df["prompt"].tolist())
    train_dl = DataLoader(
        ReconstructableTextDataset(input_df["prompt"].tolist(), tokenizer, max_length=max_seqlen, prompt_ix=input_df["prompt_ix"].tolist()),
        batch_size=batch_size, shuffle=False, collate_fn=stack_collate)
    meta["max_seqlen"] = int(max_seqlen)
    out.log(f"prompts {len(input_df)}, max_seqlen {max_seqlen}, batch_size {batch_size}, batches {len(train_dl)}")

    # ===== cell 14: forward passes. Same function and bookkeeping as utils.probes.run_and_export_states,
    # but each layer's states go to a float16 memmap per batch instead of one cube in RAM.
    from termcolor import colored
    from transformers.loss.loss_utils import ForCausalLMLoss
    from utils.store_outputs import convert_outputs_to_df_fast
    n_tokens_total = int(train_dl.dataset.attention_mask.sum().item())
    hidden_dim = int(model.config.hidden_size)
    layer_files = {l: np.lib.format.open_memmap(acts / f"layer{l:02d}.npy", mode="w+", dtype=np.float16, shape=(n_tokens_total, hidden_dim))
                   for l in layers_to_probe}
    out.log(f"activation memmaps: {len(layer_files)} layers x ({n_tokens_total}, {hidden_dim}) float16 = {len(layer_files) * n_tokens_total * hidden_dim * 2 / 1e9:.1f} GB at {acts}")
    sample_dfs, pos, batch_secs = [], 0, []
    t_fwd = time.time()
    with torch.no_grad():
        for batch_ix, batch in tqdm(enumerate(train_dl), total=len(train_dl)):
            tb = time.time()
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            original_tokens = batch["original_tokens"]
            prompt_indices = batch["prompt_ix"]
            output = run_forward_with_hs(model, input_ids, attention_mask, return_hidden_states=True)
            if batch_ix == 0:
                loss = ForCausalLMLoss(output["logits"], torch.where(input_ids == tokenizer.pad_token_id, torch.tensor(-100), input_ids), output["logits"].size(-1)).detach().cpu().item()
                for i in range(min(20, input_ids.size(0))):
                    decoded_input = tokenizer.decode(input_ids[i, :], skip_special_tokens=False)
                    next_token_id = torch.argmax(output["logits"][i, -1, :]).item()
                    print("---------\n" + decoded_input + colored(tokenizer.decode([next_token_id], skip_special_tokens=False).replace("\n", "<lb>"), "green"))
                meta["ppl_batch0"] = torch.exp(torch.tensor(loss)).item()
                print(f"PPL:", meta["ppl_batch0"], flush=True)
            original_tokens_df = pd.DataFrame([(seq_i, tok_i, tok) for seq_i, tokens in enumerate(original_tokens) for tok_i, tok in enumerate(tokens)],
                                              columns=["sequence_ix", "token_ix", "token"])
            prompt_indices_df = pd.DataFrame([(seq_i, seq_source) for seq_i, seq_source in enumerate(prompt_indices)], columns=["sequence_ix", "prompt_ix"])
            sample_df = (convert_outputs_to_df_fast(input_ids, attention_mask, output["logits"])
                         .merge(original_tokens_df, how="left", on=["token_ix", "sequence_ix"])
                         .merge(prompt_indices_df, how="left", on=["sequence_ix"])
                         .assign(batch_ix=batch_ix))
            sample_dfs.append(sample_df)
            valid_pos = torch.where(attention_mask.cpu().view(-1) == 1)
            hs = torch.stack(output["all_pre_mlp_hidden_states"], dim=1)[valid_pos]  # (T_b, n_layers, D) on CPU, bf16
            n_b = hs.shape[0]
            if n_b != len(sample_df):
                raise RuntimeError(f"batch {batch_ix}: {n_b} valid positions but {len(sample_df)} sample rows")
            hs16 = hs.to(torch.float16)  # the notebook's cast, applied per batch
            for l in layers_to_probe:
                layer_files[l][pos:pos + n_b] = hs16[:, l, :].numpy()
            pos += n_b
            del output, hs, hs16
            batch_secs.append(time.time() - tb)
            out.heartbeat("forward", batch=batch_ix + 1, batches=len(train_dl), tokens=pos)
            out.progress(phase="forward", batch=batch_ix + 1, batches=len(train_dl), tokens=pos, sec_per_batch=round(sum(batch_secs) / len(batch_secs), 2))
    for f in layer_files.values():
        f.flush()
    del layer_files
    if pos != n_tokens_total:
        raise RuntimeError(f"wrote {pos} token rows, expected {n_tokens_total}")
    sample_df_all = pd.concat(sample_dfs, ignore_index=True).drop(columns=["batch_ix", "sequence_ix"])
    del sample_dfs
    timings["forward_s"] = round(time.time() - t_fwd, 1)
    timings["forward_sec_per_batch"] = round(sum(batch_secs) / len(batch_secs), 2)
    meta["peak_vram_gb_forward"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    out.log(f"forward done: {pos} tokens in {timings['forward_s']} s ({timings['forward_sec_per_batch']} s/batch), peak VRAM {meta['peak_vram_gb_forward']} GB")
    clear_all_cuda_memory()
    gc.collect()

    # ===== cell 16: label roles
    import utils.role_assignments
    importlib.reload(utils.role_assignments)
    from utils.role_assignments import label_content_roles
    from utils.substring_assignments import flag_message_types
    probe_sample_df = (
        label_content_roles(model_prefix, sample_df_all)
        .assign(sample_ix=lambda df: range(0, len(df)))
        .merge(input_df[["prompt_ix", "role", "question_ix"]].rename(columns={"role": "target_role"}), how="inner", on="prompt_ix")
        .assign(match_target_role=lambda df: np.where(df["role"] == df["target_role"], True, False))
        .pipe(lambda df: flag_message_types(df, train_prefixes, allow_ambiguous=True))
        .drop(columns="base_message")
    )
    # The notebook's filter, kept as a column so the full token table can be reused on the Mac.
    probe_sample_df = probe_sample_df.assign(
        keep=lambda df: (df["base_message_ix"].isna()) & (df["is_content"] == True) & (df["role"].notna()) & (df["match_target_role"]))  # noqa: E712
    tok_cols = [c for c in probe_sample_df.columns if probe_sample_df[c].dtype != object or c in ("token", "role", "target_role")]
    probe_sample_df[tok_cols].to_parquet(out.dir / "tokens.parquet", index=False)
    probe_sample_df[tok_cols].to_parquet(acts / "tokens.parquet", index=False)
    probe_sample_df = probe_sample_df[probe_sample_df["keep"]]
    role_counts = probe_sample_df.groupby("role", as_index=False).agg(count=("sample_ix", "count"))
    display(role_counts)
    display(probe_sample_df.pipe(lambda df: df[df["prompt_ix"] <= 14]).groupby(["prompt_ix", "seg_ix", "role"], as_index=False)
            .agg(combined_text=("token", "".join)).assign(eot=lambda df: df["combined_text"].str[-30:]))
    meta["role_counts"] = role_counts.set_index("role")["count"].astype(int).to_dict()
    meta["role_counts_equal"] = len(set(meta["role_counts"].values())) == 1 and len(meta["role_counts"]) == 5
    out.log(f"role counts: {meta['role_counts']} equal={meta['role_counts_equal']}")
    if not meta["role_counts_equal"]:
        out.log("!! role counts differ across the five roles; the notebook says they should be exactly equal for gpt-oss")

    # ===== cell 18: fit_lr / get_probe_result (extended with split_by; the grid search stays commented out)
    SKIP_FIRST_N = 32 if NESTED_REASONING else 0

    def fit_lr(x_train, y_train, x_test, y_test, add_scaling=False, **lr_params):
        steps = []
        if add_scaling:
            steps.append(("scaler", cuml.preprocessing.StandardScaler()))
        steps.append(("clf", cuml.linear_model.LogisticRegression(penalty="l2", max_iter=5_000, linesearch_max_iter=100, fit_intercept=True, **lr_params)))
        lr_model = sklearn.pipeline.Pipeline(steps)
        lr_model.fit(x_train, y_train)
        accuracy = lr_model.score(x_test, y_test)
        y_test_pred = lr_model.predict(x_test)
        y_test_prob = lr_model.predict_proba(x_test)
        nll = cuml.metrics.log_loss(y_test, y_test_prob,)
        return lr_model, accuracy, nll, y_test_pred

    split_audit = {}

    def get_probe_result(sample_df, layer_hs, roles_map, split_by="prompt", add_scaling=False, **lr_params):
        # Train/test split. 'prompt' is the notebook's line; 'base' groups by the underlying text (extension).
        key = "prompt_ix" if split_by == "prompt" else "question_ix"
        unique_ids = sample_df[key].unique()
        # I avoid cuML's zero-test-size slicing overlap only in a tiny approved pilot.
        # I retain test_size=.1 unchanged for every full-corpus split.
        test_size = 1 if args.pilot and int(len(unique_ids) * 0.1) == 0 else 0.1
        if test_size == 1:
            print(f"[pilot] {split_by} has {len(unique_ids)} groups; using test_size=1 group before cuML splitting", flush=True)
        ids_train, ids_test = cuml.train_test_split(unique_ids, test_size=test_size, random_state=seed)
        train_ids = to_numpy(ids_train).tolist()
        test_ids = to_numpy(ids_test).tolist()
        original_ids = to_numpy(unique_ids).tolist()
        train_set, test_set, original_set = set(train_ids), set(test_ids), set(original_ids)
        if not train_set or not test_set or train_set & test_set or train_set | test_set != original_set:
            raise RuntimeError(f"Invalid {split_by} group partition: train={len(train_set)}, test={len(test_set)}, overlap={len(train_set & test_set)}, original={len(original_set)}")
        train_mask = sample_df[key].isin(train_ids)
        test_mask = sample_df[key].isin(test_ids)
        if not train_mask.any() or not test_mask.any() or not (train_mask.astype(int) + test_mask.astype(int)).eq(1).all():
            raise RuntimeError(f"Invalid {split_by} token-row partition")
        train_df = sample_df[train_mask]
        test_df = sample_df[test_mask]
        audit_key = f"{split_by}:{space_abbrev(list(roles_map))}"
        audit_record = {"split_by": split_by, "group_column": key, "role_order": list(roles_map),
                        "test_size_requested": test_size, "seed": seed, "pilot": args.pilot,
                        "original_group_ids": sorted(original_ids), "train_group_ids": sorted(train_ids),
                        "test_group_ids": sorted(test_ids), "n_original_groups": len(original_set),
                        "n_train_groups": len(train_set), "n_test_groups": len(test_set),
                        "n_original_rows": len(sample_df), "n_train_rows": len(train_df), "n_test_rows": len(test_df),
                        "groups_nonempty_disjoint_exhaustive": True, "rows_partition_exactly_once": True}
        if audit_key in split_audit and split_audit[audit_key] != audit_record:
            raise RuntimeError(f"Split assignment changed between layer fits for {audit_key}")
        split_audit[audit_key] = audit_record
        # I persist the partition before fitting so failures still leave reviewable split evidence.
        audit_tmp = out.dir / "split-audit.json.tmp"
        audit_tmp.write_text(json.dumps({"updated_utc": utc_now(), "splits": split_audit}, indent=2))
        audit_tmp.replace(out.dir / "split-audit.json")
        role_labels_train_cp = cupy.asarray([roles_map[r] for r in train_df["role"]])
        role_labels_test_cp = cupy.asarray([roles_map[r] for r in test_df["role"]])
        x_train_cp = cupy.asarray(layer_hs[train_df["sample_ix"].tolist(), :].to(torch.float32).detach().cpu())
        x_test_cp = cupy.asarray(layer_hs[test_df["sample_ix"].tolist(), :].to(torch.float32).detach().cpu())
        if (len(train_df) != x_train_cp.shape[0]):
            raise Exception(f"Shape mismatch!")
        uniq_train = np.unique(role_labels_train_cp.get())
        if len(uniq_train) < len(roles_map):
            raise Exception(f"Skipping mapping {roles_map}: missing roles in train", uniq_train)
        lr_model, test_acc, test_nll, y_test_pred = fit_lr(x_train_cp, role_labels_train_cp, x_test_cp, role_labels_test_cp, add_scaling=add_scaling, **lr_params)
        results_df = (test_df.assign(pred=y_test_pred.tolist())
                      .assign(pred=lambda df: df["pred"].map({v: k for k, v in roles_map.items()}))
                      .assign(is_acc=lambda df: df["role"] == df["pred"]))
        acc_by_role = results_df.groupby(["role", "pred"], as_index=False).agg(count=("sample_ix", "count"))
        acc_by_pos = results_df.groupby("token_in_seg_ix", as_index=False).agg(count=("sample_ix", "count"), acc=("is_acc", "mean"))
        per_role = results_df.groupby("role")["is_acc"].mean().round(4).to_dict()
        del x_train_cp, x_test_cp
        return {"probe": lr_model, "acc": float(test_acc), "nll": float(test_nll), "acc_by_role": acc_by_role, "acc_by_pos": acc_by_pos,
                "acc_by_role_dict": per_role, "n_train": int(len(train_df)), "n_test": int(len(test_df))}

    clear_all_cuda_memory()
    gc.collect()

    # ===== cell 19: all role spaces x all probe layers (x both splits). Layer outer here (one layer file in RAM at a time).
    train_params = probe_cfg["train_params"]
    all_role_combinations = [
        {"roles": list(roles), "roles_map": {x: i for i, x in enumerate(roles)},
         "sample_df": probe_sample_df.pipe(lambda df: df[(df["role"].isin(roles)) & (df["token_in_seg_ix"] >= SKIP_FIRST_N)]).reset_index(drop=True)}
        for roles in ALL_ROLE_COMBINATIONS
    ]
    n_probes_total = len(layers_to_probe) * len(splits) * len(all_role_combinations)
    all_probes = {s: [] for s in splits}
    probe_secs, done = [], 0
    t_fit = time.time()
    try:
        for layer_ix in layers_to_probe:
            tl = time.time()
            layer_hs = torch.from_numpy(np.ascontiguousarray(np.load(acts / f"layer{layer_ix:02d}.npy", mmap_mode="r")))
            out.log(f"layer {layer_ix}: loaded {tuple(layer_hs.shape)} float16 in {time.time() - tl:.0f}s")
            for split in splits:
                for roles_dict in all_role_combinations:
                    tp = time.time()
                    probe_res = get_probe_result(sample_df=roles_dict["sample_df"], layer_hs=layer_hs, roles_map=roles_dict["roles_map"],
                                                 split_by=split, add_scaling=train_params["add_scaling"], C=train_params["C"])
                    sec = round(time.time() - tp, 1)
                    probe_secs.append(sec)
                    done += 1
                    all_probes[split].append({**probe_res, "layer_ix": layer_ix, "role_space": roles_dict["roles"], "roles_map": roles_dict["roles_map"],
                                              "n_inputs": len(roles_dict["sample_df"]), "sec": sec})
                    out.log(f"[fit] {split:6s} {space_abbrev(roles_dict['roles']):5s} layer {layer_ix:2d}: acc {probe_res['acc']:.3f} nll {probe_res['nll']:.3f} ({sec}s) [{done}/{n_probes_total}]")
                    out.heartbeat("fit", done=done, total=n_probes_total)
                    out.progress(phase="fit", done=done, total=n_probes_total, sec_per_probe=round(sum(probe_secs) / len(probe_secs), 1), layer=layer_ix)
            del layer_hs
            gc.collect()
    except (TimeoutError, InterruptedError) as error:
        out.log(f"!! stopped early: {error}; saving {done} probes as partial")
        meta["partial"] = str(error)
    timings["fit_s"] = round(time.time() - t_fit, 1)
    timings["fit_sec_per_probe"] = round(sum(probe_secs) / max(1, len(probe_secs)), 1)
    for split in splits:
        all_probes[split].sort(key=authors_order_key)  # the notebook's list order
        print(f"Num probes ({split}): {len(all_probes[split])}")
    print(f"Probe layers:\n  {', '.join(str(x) for x in layers_to_probe)}")

    # ===== cell 20: save probes + metrics, once per split, plus our format and the portable clone
    def validate_accuracy_and_save(all_probes_split, split):
        tag = split_tag(split)
        print(f"Val accuracy by layer ({split} split):")
        pivot = (pd.DataFrame(all_probes_split)[["layer_ix", "role_space", "acc"]]
                 .assign(acc=lambda df: df["acc"].round(2), role_space=lambda df: df["role_space"].apply(lambda x: ",".join([r[0] for r in x])))
                 .pivot(index="layer_ix", columns="role_space", values="acc"))
        display(pivot)
        pivot.to_csv(out.dir / f"val_acc_by_layer{tag}.csv")
        print("Concatenating across layers and saving...")
        acc_by_role = pd.concat([p["acc_by_role"].assign(model=model_prefix, layer_ix=p["layer_ix"], role_space=",".join(p["role_space"])) for p in all_probes_split], ignore_index=True)
        acc_by_pos = (pd.concat([p["acc_by_pos"].assign(model=model_prefix, layer_ix=p["layer_ix"], role_space=",".join(p["role_space"])) for p in all_probes_split], ignore_index=True)
                      .assign(acc=lambda df: df["acc"].round(4)))
        acc_by_role.to_csv(out.dir / f"acc_by_role_{model_prefix}{tag}.csv", index=False)
        acc_by_pos.to_csv(out.dir / f"acc_by_pos_{model_prefix}{tag}.csv", index=False)
        authors_keys = ("probe", "acc", "nll", "acc_by_role", "acc_by_pos", "layer_ix", "role_space", "roles_map", "n_inputs")
        with open(out.dir / f"{model_prefix}{tag}.pkl", "wb") as f:
            pickle.dump([{k: p[k] for k in authors_keys} for p in all_probes_split], f)
        print("Accuracy by role:")
        base_sums = acc_by_role.groupby(["role", "layer_ix", "role_space"], as_index=False).agg(base_sum=("count", "sum"))
        by_role = (acc_by_role.pipe(lambda df: df[df["role"] == df["pred"]])
                   .groupby(["role", "layer_ix", "role_space"], as_index=False).agg(sum=("count", "sum"))
                   .merge(base_sums, on=["layer_ix", "role_space", "role"], how="inner")
                   .assign(acc=lambda df: df["sum"] / df["base_sum"])
                   .pivot(index=["role_space", "layer_ix"], columns="role", values="acc").round(2))
        display(by_role)
        # portable sklearn clone (reference REPORT.md section 4) and our npz/json format
        portable = []
        for p in all_probes_split:
            coef, intercept = coef_arrays(p["probe"])
            portable.append({"probe": make_sklearn_clone(coef, intercept, train_params["C"]), "acc": p["acc"], "nll": p["nll"], "layer_ix": int(p["layer_ix"]),
                             "role_space": list(p["role_space"]), "roles_map": dict(p["roles_map"]), "n_inputs": p.get("n_inputs"), "C": train_params["C"],
                             "feature": "pre_mlp", "split_by": split})
        with open(out.dir / f"role_probes{tag}.pkl", "wb") as f:
            pickle.dump(portable, f)
        arrays, doc = our_format(all_probes_split, split, train_params["C"], {"model_prefix": model_prefix, "layers": layers_to_probe,
                                                                             "attn_implementation": attn_loaded, "expert_dtype": expert_dtype,
                                                                             "n_docs": meta["n_docs"], "role_counts": meta["role_counts"], "pilot": args.pilot})
        np.savez(out.dir / f"probes{tag}.npz", **arrays)
        (out.dir / f"probes{tag}.json").write_text(json.dumps(doc, indent=2))
        return True

    for split in splits:
        if all_probes[split]:
            validate_accuracy_and_save(all_probes[split], split)

    # ===== metadata (ours)
    # I preserve the historical final-counter field and expose the higher pre-reset forward peak.
    peak_vram_gb_final = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    peak_vram_gb_total = max(meta["peak_vram_gb_forward"], peak_vram_gb_final)
    meta.update(finished_utc=utc_now(), timings=timings, layers=layers_to_probe, splits=splits, n_probes=sum(len(v) for v in all_probes.values()),
                n_probes_planned=n_probes_total, peak_vram_gb=peak_vram_gb_final,
                peak_vram_gb_final_counter=peak_vram_gb_final, peak_vram_gb_total=peak_vram_gb_total,
                peak_vram_accounting="I retain the maximum of the saved forward-stage peak and final torch CUDA counter; clear_all_cuda_memory resets counters between stages. These are torch allocator peaks, not total device usage.",
                peak_rss_gb=peak_rss_gb(),
                n_tokens=int(n_tokens_total), acts_dir=str(acts), hidden_dim=hidden_dim,
                split_audit_file="split-audit.json", n_verified_split_partitions=len(split_audit),
                extensions=[f"layers {args.layers} (authors: range(0, 24, 2))", f"splits {splits} (authors: prompt only)",
                            f"attention {attn_loaded} (authors: {AUTHORS_ATTN})",
                            f"I use extraction batch_size {batch_size} (authors: {BATCH_SIZE}); prompt order, padding, corpus, seed and estimator are unchanged",
                            "activations written per layer to float16 memmaps during the forward loop instead of one RAM cube (same values, same row order)",
                            "fit loop order layer > split > role space; saved lists re-sorted to the notebook's order"],
                source=f"NB02 cells 1-20, commit {AUTHORS_COMMIT}")
    (out.dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    (acts / "metadata.json").write_text(json.dumps({"model_prefix": model_prefix, "hf_snapshot": meta.get("hf_snapshot"), "site": "post_attention_layernorm output (pre-MLP)",
                                                    "layers": layers_to_probe, "n_prompts": meta["n_prompts"], "n_base_texts": meta["n_docs"]["total"],
                                                    "n_tokens": int(n_tokens_total), "dtype": "float16", "backend": "transformers", "attn": attn_loaded,
                                                    "expert_dtype": expert_dtype, "batch_size": batch_size, "authors_batch_size": BATCH_SIZE,
                                                    "padding": "left, max_length=max_seqlen (authors)",
                                                    "row_order": "sample_ix (tokens.parquet)", "source": meta["source"]}, indent=2))
    if args.pilot:
        full_n_docs = int(probe_cfg["n_sample_size"] * .25) + int(probe_cfg["n_sample_size"] * .75)
        full_batches = -(-(full_n_docs * 5) // batch_size)
        projection = {"forward_sec_per_batch_pilot": timings["forward_sec_per_batch"], "full_batches": full_batches,
                      "forward_full_projected_s": round(timings["forward_sec_per_batch"] * full_batches),
                      "fit_sec_per_probe_pilot": timings["fit_sec_per_probe"], "full_probes": n_probes_total,
                      "note": "pilot fits use ~25x fewer rows; the reference H200 run took 8.4 s per probe on the full data (803 s / 96)"}
        (out.dir / "pilot.json").write_text(json.dumps(projection, indent=2))
        out.log(f"pilot projection: {json.dumps(projection)}")
    out.log(f"done: {meta['n_probes']} probes in {time.time() - out.t0:.0f}s; peak VRAM across recorded stages {meta['peak_vram_gb_total']} GB (final counter {meta['peak_vram_gb_final_counter']} GB), peak RSS {meta['peak_rss_gb']} GB")
    if "partial" in meta:
        out.marker("FAILED", f"partial: {meta['partial']}")
        return 124
    out.marker("DONE", utc_now())
    return 0


# ----------------------------------------------------------------------------- entry points

def check_env():
    versions = package_versions()
    print(json.dumps(versions, indent=2))
    missing = [k for k in ("torch", "transformers", "kernels", "cuml", "cupy", "scikit-learn", "datasets", "zstandard", "pandas", "pyarrow") if not versions.get(k)]
    problems = list(missing)
    try:
        import torch
        print(f"cuda available: {torch.cuda.is_available()}; device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}; sm {compute_capability()}")
        if not torch.cuda.is_available():
            problems.append("no CUDA device")
    except Exception as error:  # noqa: BLE001
        problems.append(f"torch import failed: {error}")
    try:
        import cuml  # noqa: F401
        import cudf  # noqa: F401
    except Exception as error:  # noqa: BLE001
        problems.append(f"cuml/cudf import failed: {error}")
    print("attention plan:", resolve_attn("auto"))
    print("nvidia-smi:", nvidia_smi())
    if problems:
        print("PROBLEMS:", problems)
        return 2
    print("env ok")
    return 0


def self_test():
    """Pure-Python checks that need no CUDA: parsing, naming, order, and the sklearn clone."""
    import numpy as np
    assert BATCH_SIZE == 32
    assert positive_int("16") == 16 and positive_int("1") == 1
    for invalid in ("0", "-1", "1.5", "abc"):
        try:
            positive_int(invalid)
        except argparse.ArgumentTypeError:
            pass
        else:
            raise AssertionError(f"Invalid batch size accepted: {invalid}")
    assert parse_layers("all") == list(range(24))
    assert parse_layers("authors") == list(range(0, 24, 2))
    assert parse_layers("0,12,23") == [0, 12, 23]
    assert parse_splits("prompt,base") == ["prompt", "base"]
    assert split_tag("prompt") == "" and split_tag("base") == "-basesplit"
    assert space_abbrev(["system", "user", "cot", "assistant", "tool"]) == "sucat"
    assert npz_key(["user", "assistant"], 5, "coef") == "ua_L05__coef"
    probes = [{"role_space": list(r), "layer_ix": l} for l in (2, 0) for r in reversed(ALL_ROLE_COMBINATIONS)]
    probes.sort(key=authors_order_key)
    assert [(p["role_space"][0], p["layer_ix"]) for p in probes[:3]] == [("user", 0), ("user", 2), ("user", 0)]
    assert probes[0]["role_space"] == ["user", "assistant"] and probes[-1]["layer_ix"] == 2
    try:
        import sklearn  # noqa: F401
        binary = make_sklearn_clone(np.ones((1, 4)), np.zeros(1), 5e-3)
        assert list(binary.classes_) == [0, 1] and binary.coef_.shape == (1, 4)
        five = make_sklearn_clone(np.ones((5, 4)), np.zeros(5), 5e-3)
        assert list(five.classes_) == [0, 1, 2, 3, 4]
        x = np.random.default_rng(0).normal(size=(3, 4))
        assert five.predict_proba(x).shape == (3, 5)
        print("sklearn clone ok")
    except ImportError:
        print("sklearn not installed here; clone check skipped")
    print("self-test ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="/workspace/prompt-injection-as-role-confusion", help="authors' clone at commit ec333c40")
    ap.add_argument("--results", default="/workspace/results/probes", help="outbox: probes, CSVs, metadata (rsynced to the Mac)")
    ap.add_argument("--acts", default="/workspace/acts/probes", help="activation memmaps (stay on the pod volume)")
    ap.add_argument("--layers", default="all", help="all | authors | comma list")
    ap.add_argument("--splits", default="prompt,base")
    ap.add_argument("--attn", choices=["auto", "fa3", "eager"], default="auto")
    ap.add_argument("--batch-size", type=positive_int, default=BATCH_SIZE,
                    help="extraction batch size; I retain the authors' 32 by default and can reduce it to bound GPU memory")
    ap.add_argument("--pilot", type=int, default=0, help="N base texts instead of probe.yaml's 250; prints the accuracy table and exits")
    ap.add_argument("--heartbeat-file", default=None, help="where the watchdog reads the heartbeat (default <results>/heartbeat)")
    ap.add_argument("--hours", type=float, default=3.0, help="deadline armed after the model loads; partial probes are saved on expiry")
    ap.add_argument("--skip-gen-tests", action="store_true", help="skip the notebook's generation sanity cells 5 and 10")
    ap.add_argument("--require-mxfp4", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow-other-commit", action="store_true")
    ap.add_argument("--check-env", action="store_true")
    ap.add_argument("--check-model", action="store_true", help="load the model, verify the custom forward pass, print dtype and attention, exit")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if args.check_env:
        return check_env()
    if not 1 / 60 <= args.hours <= 6:
        ap.error("--hours must be between 1/60 and 6")
    try:
        return run(args)
    except (TimeoutError, InterruptedError) as error:
        Path(args.results).mkdir(parents=True, exist_ok=True)
        (Path(args.results) / "FAILED").write_text(f"{utc_now()} {error}\n")
        print(f"FAILED: {error}", flush=True)
        return 124
    except BaseException as error:  # noqa: BLE001
        Path(args.results).mkdir(parents=True, exist_ok=True)
        (Path(args.results) / "FAILED").write_text(f"{utc_now()} {type(error).__name__}: {error}\n")
        raise


if __name__ == "__main__":
    sys.exit(main())
