#!/usr/bin/env python3
"""I reproduce Appendix E's three fixed gardening prompts on the authors' CUDA path.

My stage is Understand: controlled replication. I measure fidelity on the fixed
example; I do not tune probes to the paper's rounded percentages. I import the
authors' loader, custom forward, dataset, collator, substring assignment and
native cuML projection helper from the frozen, locally supplied checkout.

Example (on my already provisioned CUDA pod):
  python appendix_e_cuda.py --repo /workspace/role-confusion \
    --probe-run /workspace/results/full --out /workspace/results/appendix-e-L12

I never execute a notebook or generate new conversation text. My model cache
must already exist under /workspace/hf. I set Hugging Face offline flags before
imports so this stage cannot silently download another model or dataset.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import subprocess
import sys
import traceback

MODEL_PREFIX = "gptoss-20b"
MODEL_ID = "openai/gpt-oss-20b"
AUTHORS_COMMIT = "ec333c40fd43fe991e1ebf66765051b6d7e35784"
AUTHORS_ATTN = "kernels-community/vllm-flash-attn3"
EXPECTED_REVISION = "6cee5e81ee83917806bbde320786a8fb61efebee"
ROLES = ("system", "user", "cot", "assistant")
BASE_ROLES = ("system", "user", "cot", "assistant", "user", "cot", "assistant")
SOURCE_FILES = (
    "experiments/role-analysis/02-train-role-probes.ipynb",
    "experiments/role-analysis/04-tomato-probe-results.ipynb",
    "experiments/role-analysis/config/tomato.yaml",
    "experiments/role-analysis/config/probe.yaml",
    "utils/loader.py", "utils/pretrained_models/gptoss.py", "utils/dataset.py",
    "utils/probes.py", "utils/store_outputs.py", "utils/substring_assignments.py",
    "utils/role_templates.py", "utils/chat_templates/gptoss.j2", "r-utils/plots.r",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def has_mxfp4_expert_format(expert):
    """I accept packed uint8 or Triton's signed E2M1 format on an MXFP4 expert."""
    if "mxfp4" not in type(expert).__name__.lower():
        return False
    dtype = expert.down_proj.dtype
    # I inspect Triton's format fields because FloatType is not a torch.dtype.
    if type(dtype).__name__ == "FloatType":
        return (getattr(dtype, "bitwidth_exponent", None) == 2
                and getattr(dtype, "bitwidth_mantissa", None) == 1
                and getattr(dtype, "is_signed", None) is True)
    return str(dtype) == "torch.uint8" and getattr(dtype, "is_floating_point", None) is False


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def status(out, phase, **extra):
    record = {"phase": phase, "updated_utc": utc_now(), **extra}
    write_json(out / "progress.json", record)
    (out / "heartbeat").write_text(utc_now() + "\n")
    print(json.dumps(record), flush=True)


def source_provenance(repo):
    actual = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if actual != AUTHORS_COMMIT:
        raise ValueError(f"Authors checkout {actual} differs from the checkpoint's {AUTHORS_COMMIT}.")
    sources = {}
    for relative in SOURCE_FILES:
        path = repo / relative
        if not path.exists():
            raise FileNotFoundError(path)
        committed = subprocess.check_output(["git", "-C", str(repo), "show", f"{AUTHORS_COMMIT}:{relative}"])
        if hashlib.sha256(committed).hexdigest() != sha256(path):
            raise ValueError(f"I found a modified source file: {relative}")
        sources[relative] = sha256(path)
    for path in sorted((repo / "utils/chat_templates").glob("*gpt*")):
        if path.is_file():
            relative = str(path.relative_to(repo))
            committed = subprocess.check_output(["git", "-C", str(repo), "show", f"{AUTHORS_COMMIT}:{relative}"])
            if hashlib.sha256(committed).hexdigest() != sha256(path):
                raise ValueError(f"I found a modified chat template: {relative}")
            sources[relative] = sha256(path)
    return {"authors_commit": actual, "source_sha256": sources,
            "notebook_cells": {"NB02": [3, 4, 23, 48, 49, 50, 51, 52], "NB04": [3, 5, 7, 14]}}


def check_probe_metadata(probe_run):
    path = probe_run / "probes.json"
    metadata = json.loads(path.read_text())
    if metadata.get("split_by") != "prompt":
        raise ValueError("My primary projection requires the authors' prompt split.")
    if metadata.get("pilot"):
        raise ValueError("A pilot probe cannot complete my primary checkpoint.")
    if metadata.get("n_docs", {}).get("total") != 249:
        raise ValueError("My primary checkpoint requires the full 249-document probe corpus.")
    if metadata.get("C") != 0.005 or metadata.get("penalty") != "l2" or metadata.get("fit_intercept") is not True:
        raise ValueError("My supplied probe metadata differs from the authors' estimator recipe.")
    for key, expected in (("max_iter", 5000), ("linesearch_max_iter", 100)):
        if metadata.get(key) != expected:
            raise ValueError(f"Probe metadata has {key}={metadata.get(key)!r}, expected {expected}.")
    if metadata.get("attn_implementation") != AUTHORS_ATTN:
        raise ValueError("My training artifact does not record the reference FA3 attention path.")
    matches = [row for row in metadata.get("results", []) if row.get("role_space") == "suca" and row.get("layer_ix") == 12]
    if len(matches) != 1:
        raise ValueError("I require exactly one suca L12 entry in prompt-split metadata.")
    return metadata


def build_prompts(tokenizer, repo):
    import yaml
    from utils.role_templates import load_chat_template
    entries = yaml.safe_load((repo / "experiments/role-analysis/config/tomato.yaml").read_text())[MODEL_PREFIX]
    messages = [entry["content"] for entry in entries]
    if len(messages) != 7 or any(not isinstance(message, str) or not message for message in messages):
        raise ValueError("I require all seven saved nonempty gardening message contents.")
    prefix = tokenizer.bos_token or ""
    # I install NB02 cell 23's custom template before BOTH tagged conditions.
    template = load_chat_template(str(repo / "utils/chat_templates"), MODEL_PREFIX)
    tokenizer.chat_template = template
    joined = "\n".join(messages)
    # I preserve cell 49's insertion order so prompt_ix agrees with the source.
    prompts = {
        "basic_no_format": prefix + joined,
        "everything_in_user_tags": prefix + tokenizer.apply_chat_template(
            [{"role": "user", "content": joined}], tokenize=False, add_generation_prompt=False),
        "proper_tags": prefix + tokenizer.apply_chat_template([
            {"role": "system", "content": messages[0]},
            {"role": "user", "content": messages[1]},
            {"role": "assistant", "content": f"<think>{messages[2]}</think>{messages[3]}"},
            {"role": "user", "content": messages[4]},
            {"role": "assistant", "content": f"<think>{messages[5]}</think>{messages[6]}"},
        ], tokenize=False, add_generation_prompt=False),
    }
    for condition, prompt in prompts.items():
        for index, message in enumerate(messages):
            if message not in prompt:
                raise ValueError(f"My {condition} prompt lost original message {index}.")
    return messages, prompts, template


def package_versions():
    packages = ("torch", "transformers", "kernels", "triton", "cuml-cu12", "cupy-cuda12x", "cudf-cu12",
                "numpy", "pandas", "scikit-learn", "matplotlib", "huggingface_hub", "pyarrow", "PyYAML")
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def run(args):
    repo, probe_run, out = args.repo.resolve(), args.probe_run.resolve(), args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"I preserve existing results; choose an empty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    status(out, "validate_sources_and_probe")
    metadata = {"started_utc": utc_now(), "stage": "Understand: controlled replication",
                "north_star": "I test whether the authors' fixed gardening measurement reproduces.",
                "primary_probe": {"space": "suca", "layer_ix": 12, "split": "prompt", "class_order": list(ROLES)},
                "model_id": MODEL_ID, "measurement_site": "layer 12 post_attention_layernorm output before MLP",
                "batch_size": 16, "max_length": 2048, "shuffle": False,
                "activation_cast": "authors custom-forward dtype -> torch.float16 storage -> cupy.asarray directly in authors run_projections",
                "probability_cast": "native cuML predict_proba -> round(12) -> pandas round(8), as authors run_projections; unrounded values also retained",
                "script_sha256": sha256(__file__), "figure_script_sha256": sha256(Path(__file__).with_name("appendix_e_figures.py")),
                "review_status": "pending inspection of actual results and figures"}
    try:
        metadata.update(source_provenance(repo))
        probe_metadata = check_probe_metadata(probe_run)
        probe_path = probe_run / "gptoss-20b.pkl"
        metadata["training_metadata"] = probe_metadata
        metadata["probe_files"] = {name: sha256(probe_run / name) for name in ("gptoss-20b.pkl", "probes.json")}
        if (probe_run / "metadata.json").exists():
            metadata["training_run_metadata"] = json.loads((probe_run / "metadata.json").read_text())
        metadata["software_versions"] = package_versions()
        metadata["divergences_and_boundaries"] = [
            "I save only gardening layer 12 after the unchanged authors forward computes all layers; this changes retained storage, not the measured site.",
            "I use the trained prompt-split probe from my recorded corpus and software environment; the authors' original training realization is not fully pinned.",
            "I retain the paper's 82% versus 83% no-tags CoTness discrepancy and all subset denominators without tuning a denominator.",
            "I port NB04 plotting to Matplotlib; its font metrics and facet spacing differ from the R ggplot/Cairo renderer.",
        ]
        write_json(out / "metadata.json", metadata)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        sys.path.insert(0, str(repo))
        import cupy
        import numpy as np
        import pandas as pd
        import torch
        from huggingface_hub import snapshot_download
        from torch.utils.data import DataLoader
        from utils.dataset import ReconstructableTextDataset, stack_collate
        from utils.loader import load_model_and_tokenizer, load_custom_forward_pass
        from utils.probes import run_and_export_states, run_projections
        from utils.substring_assignments import flag_message_types
        from appendix_e_figures import export_analysis

        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0)[0] != 9:
            raise ValueError("My reference FA3 checkpoint requires a Hopper CUDA device.")
        metadata["gpu"] = {"name": torch.cuda.get_device_name(0), "compute_capability": torch.cuda.get_device_capability(0)}
        snapshot = Path(snapshot_download(MODEL_ID, cache_dir="/workspace/hf", local_files_only=True))
        metadata["hf_snapshot"] = snapshot.name
        metadata["hf_snapshot_path"] = str(snapshot)
        if snapshot.name != EXPECTED_REVISION:
            metadata["divergences_and_boundaries"].append(f"My actual model snapshot {snapshot.name} differs from prior reference snapshot {EXPECTED_REVISION}.")
        with probe_path.open("rb") as handle:
            # I deserialize only the native artifact from my own supplied training run.
            probes = pickle.load(handle)
        matches = [probe for probe in probes if list(probe["role_space"]) == list(ROLES) and probe["layer_ix"] == 12]
        if len(matches) != 1:
            raise ValueError("Native probe artifact must contain exactly one suca L12 probe.")
        probe = matches[0]
        if probe["roles_map"] != dict(zip(ROLES, range(4))):
            raise ValueError("Native probe class mapping differs from the four-role source order.")
        estimator = probe["probe"]
        if list(estimator.named_steps) != ["clf"]:
            raise ValueError("I require the authors' unscaled one-estimator Pipeline.")
        clf = estimator.named_steps["clf"]
        if not type(clf).__module__.startswith("cuml."):
            raise ValueError("My primary projection requires native cuML, not a portable sklearn clone.")
        classes = cupy.asnumpy(cupy.asarray(clf.classes_)).tolist()
        if classes != list(range(4)):
            raise ValueError(f"Unexpected classifier column order: {classes}")
        metadata["native_estimator"] = {"module": type(clf).__module__, "class": type(clf).__name__, "classes": classes,
                                          "coef_shape": list(clf.coef_.shape), "params": clf.get_params()}
        status(out, "load_cached_model")
        tokenizer, model, architecture, n_layers = load_model_and_tokenizer(MODEL_PREFIX, device="cuda:0")
        expert = model.model.layers[0].mlp.experts
        metadata.update(model_n_layers=n_layers, expert_dtype=str(expert.down_proj.dtype),
                        expert_class=type(expert).__name__, attn_implementation=str(model.model.config._attn_implementation),
                        model_config_commit_hash=getattr(model.config, "_commit_hash", None),
                        tokenizer_commit_hash=tokenizer.init_kwargs.get("_commit_hash"),
                        cache_environment={key: os.environ.get(key) for key in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")})
        if not has_mxfp4_expert_format(expert):
            raise ValueError("My model did not load the required packed MXFP4 expert path.")
        if metadata["attn_implementation"] != AUTHORS_ATTN:
            raise ValueError("My model did not load the reference FA3 attention implementation.")
        status(out, "verify_custom_forward")
        forward = load_custom_forward_pass(architecture, model, tokenizer)
        metadata["custom_forward_exact_logits_check"] = True
        messages, prompts, template = build_prompts(tokenizer, repo)
        (out / "custom-chat-template.jinja").write_text(template)
        metadata["custom_template_sha256"] = hashlib.sha256(template.encode()).hexdigest()
        prompt_frame = pd.DataFrame({"prompt_key": list(prompts), "prompt": list(prompts.values()), "prompt_ix": range(3)})
        dataset = ReconstructableTextDataset(prompt_frame.prompt.tolist(), tokenizer, max_length=2048,
                                             prompt_ix=prompt_frame.prompt_ix.tolist())
        prompt_records = []
        for index, (key, prompt) in enumerate(prompts.items()):
            original_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            mask = dataset.attention_mask[index].bool()
            actual_ids = dataset.input_ids[index][mask].tolist()
            if len(original_ids) > 2048:
                raise ValueError(f"My frozen {key} prompt exceeds 2048 tokens; I will not silently truncate it.")
            if actual_ids != original_ids:
                raise ValueError(f"My {key} IDs changed in the source dataset's truncation/padding step.")
            (out / f"prompt-{key}.txt").write_text(prompt)
            prompt_records.append({"prompt_ix": index, "prompt_key": key, "text": prompt,
                                   "text_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                   "token_ids": actual_ids, "n_tokens": len(actual_ids),
                                   "padded_input_ids": dataset.input_ids[index].tolist(),
                                   "attention_mask": dataset.attention_mask[index].tolist()})
        write_json(out / "rendered-prompts-and-token-ids.json", prompt_records)
        write_json(out / "original-messages.json", [{"base_message_ix": i, "base_message_type": role, "content": content}
                   for i, (role, content) in enumerate(zip(BASE_ROLES, messages))])
        status(out, "gardening_forward", n_prompts=3)
        outputs = run_and_export_states(model, tokenizer, run_model_return_states=forward,
                                       dl=DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=stack_collate),
                                       layers_to_keep_acts=[12])
        states = outputs["all_hs"].to(torch.float16)[:, 0, :]
        np.save(out / "gardening-layer12-float16.npy", states.numpy())
        sample_frame = outputs["sample_df"]
        sorted_frame = sample_frame.sort_values(["prompt_ix", "token_ix"]).reset_index(drop=True)
        if not sample_frame.reset_index(drop=True).equals(sorted_frame):
            raise ValueError("Source sample order changed; substring sorting would break activation alignment.")
        labeled_all = flag_message_types(sample_frame, messages)
        labeled_all["sample_ix"] = range(len(labeled_all))
        labeled_all["token_in_prompt_ix"] = labeled_all.groupby("prompt_ix").cumcount()
        labeled_all = labeled_all.merge(prompt_frame[["prompt_ix", "prompt_key"]], on="prompt_ix", how="left")
        labeled_all["base_message_type"] = labeled_all.base_message_ix.map(dict(enumerate(BASE_ROLES)))
        labeled_all.to_csv(out / "all-tokens-and-labels.csv", index=False)
        labeled = labeled_all[labeled_all.base_message_type.isin(ROLES)].copy()
        counts = labeled.groupby(["prompt_key", "base_message_ix"]).size()
        if len(counts) != 21 or (counts <= 0).any():
            raise ValueError("At least one of the seven original messages has no matched content in a condition.")
        metadata["n_tokens_forward"] = len(labeled_all)
        metadata["n_labeled_tokens"] = len(labeled)
        metadata["label_counts"] = [{"prompt_key": key[0], "base_message_ix": int(key[1]), "n_tokens": int(value)}
                                     for key, value in counts.items()]
        status(out, "native_cuml_projection")
        raw_matrix = cupy.asnumpy(estimator.predict_proba(cupy.asarray(states[labeled.sample_ix.tolist(), :])))
        if raw_matrix.shape != (len(labeled), 4) or not np.isfinite(raw_matrix).all():
            raise ValueError("Invalid native probability matrix.")
        if np.min(raw_matrix) < -1e-6 or np.max(raw_matrix) > 1 + 1e-6 or not np.allclose(raw_matrix.sum(axis=1), 1, atol=1e-5):
            raise ValueError("Native role probabilities are outside the probability simplex.")
        np.save(out / "native-probabilities-unrounded.npy", raw_matrix)
        np.save(out / "native-probability-sample-indices.npy", labeled.sample_ix.to_numpy())
        unrounded = pd.DataFrame(raw_matrix, columns=ROLES).assign(sample_ix=labeled.sample_ix.to_numpy()).melt(
            id_vars="sample_ix", var_name="target_role", value_name="prob_unrounded")
        projections = run_projections(valid_sample_df=labeled, layer_hs=states, probe=probe)
        projections = projections.merge(unrounded, on=["sample_ix", "target_role"], validate="one_to_one")
        if not np.allclose(projections.prob, projections.prob_unrounded, rtol=0, atol=2e-7):
            raise ValueError("Repeated native projection differs beyond the source's rounding tolerance.")
        columns = ["sample_ix", "prompt_ix", "prompt_key", "base_message_ix", "base_message_type", "token_in_prompt_ix", "token", "token_ix"]
        projections = projections.merge(labeled[columns], on="sample_ix", how="inner", validate="many_to_one")
        projections = projections.assign(layer_ix=12, role_space="suca")
        projections.to_csv(out / "tomato-role-projections-gptoss-20b.csv", index=False)
        # I release GPU model memory before the CPU-only figure and comparison stage.
        del outputs, model, forward
        torch.cuda.empty_cache()
        status(out, "export_figures_and_comparisons")
        metadata["analysis"] = export_analysis(projections, out)
        metadata["finished_utc"] = utc_now()
        metadata["execution_status"] = "artifacts_complete_review_pending"
        write_json(out / "metadata.json", metadata)
        artifact_files = [path for path in out.iterdir() if path.is_file() and path.name not in ("artifact-manifest.json", "progress.json", "heartbeat", "DONE")]
        write_json(out / "artifact-manifest.json", {path.name: {"sha256": sha256(path), "bytes": path.stat().st_size} for path in artifact_files})
        status(out, "artifacts_complete_review_pending")
        (out / "DONE").write_text("I generated the artifacts; source comparison interpretation and visual review remain pending.\n")
        return 0
    except Exception as error:
        metadata.update(failed_utc=utc_now(), execution_status="failed", error=f"{type(error).__name__}: {error}")
        write_json(out / "metadata.json", metadata)
        (out / "FAILED").write_text(traceback.format_exc())
        status(out, "failed", error=str(error))
        raise


def resume_analysis(args):
    """I finish figures and summaries from my existing probabilities without CUDA imports."""
    import numpy as np
    import pandas as pd
    from appendix_e_figures import export_analysis

    out = args.out.resolve()
    metadata = json.loads((out / "metadata.json").read_text())
    if (metadata.get("custom_forward_exact_logits_check") is not True
            or metadata.get("attn_implementation") != AUTHORS_ATTN
            or metadata.get("hf_snapshot") != EXPECTED_REVISION
            or metadata.get("primary_probe") != {"space": "suca", "layer_ix": 12, "split": "prompt", "class_order": list(ROLES)}):
        raise ValueError("My saved forward metadata does not identify the approved verified checkpoint.")
    provenance = source_provenance(args.repo.resolve())
    if provenance["source_sha256"] != metadata.get("source_sha256"):
        raise ValueError("My authors sources changed since the saved forward pass.")
    check_probe_metadata(args.probe_run.resolve())
    for filename, digest in metadata["probe_files"].items():
        if sha256(args.probe_run / filename) != digest:
            raise ValueError(f"My trained probe artifact changed: {filename}")
    protected = ["gardening-layer12-float16.npy", "native-probabilities-unrounded.npy",
                 "native-probability-sample-indices.npy", "tomato-role-projections-gptoss-20b.csv",
                 "all-tokens-and-labels.csv", "original-messages.json", "custom-chat-template.jinja",
                 "rendered-prompts-and-token-ids.json"]
    prompt_records = json.loads((out / "rendered-prompts-and-token-ids.json").read_text())
    if len(prompt_records) != 3 or {p["prompt_key"] for p in prompt_records} != {"proper_tags", "basic_no_format", "everything_in_user_tags"}:
        raise ValueError("My saved three-condition input record is incomplete.")
    for prompt in prompt_records:
        filename = f"prompt-{prompt['prompt_key']}.txt"
        if ((out / filename).read_text() != prompt["text"]
                or sha256(out / filename) != prompt["text_sha256"]
                or len(prompt["token_ids"]) != prompt["n_tokens"]):
            raise ValueError("My saved prompt text or token count changed.")
        protected.append(filename)
    before = {filename: sha256(out / filename) for filename in protected}
    activations = np.load(out / "gardening-layer12-float16.npy", mmap_mode="r")
    native = np.load(out / "native-probabilities-unrounded.npy")
    indices = np.load(out / "native-probability-sample-indices.npy")
    raw = pd.read_csv(out / "tomato-role-projections-gptoss-20b.csv", keep_default_na=False,
                      dtype={"prob": "float64"}, float_precision="round_trip")
    if (activations.shape != (metadata["n_tokens_forward"], 2880) or activations.dtype != np.float16
            or native.shape != (metadata["n_labeled_tokens"], 4) or len(indices) != len(native)
            or len(raw) != len(native) * 4 or set(raw.layer_ix) != {12} or set(raw.role_space) != {"suca"}):
        raise ValueError("My saved activation/probability dimensions do not match the completed forward.")
    projected = raw.pivot(index="sample_ix", columns="target_role", values="prob").reindex(index=indices, columns=ROLES).to_numpy()
    if (not np.isfinite(native).all() or not np.isfinite(projected).all()
            or not np.allclose(projected, native, rtol=0, atol=2e-7)
            or not np.allclose(projected.sum(axis=1), 1, rtol=0, atol=1e-5)):
        raise ValueError("My saved projection rows differ from the native probability array.")
    resume = {"started_utc": utc_now(), "previous_error": metadata.get("error"),
              "previous_metadata_sha256": sha256(out / "metadata.json"), "input_sha256": before,
              "analysis_script_sha256": sha256(__file__),
              "figure_script_sha256": sha256(Path(__file__).with_name("appendix_e_figures.py")),
              "model_forward_rerun": False}
    write_json(out / "analysis-resume-inputs.json", resume)
    try:
        status(out, "resume_existing_probability_analysis")
        analysis = export_analysis(raw, out)
        if any(sha256(out / filename) != digest for filename, digest in before.items()):
            raise ValueError("A protected forward artifact changed during analysis-only resume.")
        resume["finished_utc"] = utc_now()
        metadata.update(analysis=analysis, analysis_resume=resume, finished_utc=utc_now(),
                        execution_status="artifacts_complete_review_pending")
        metadata.pop("error", None)
        metadata.pop("failed_utc", None)
        write_json(out / "metadata.json", metadata)
        for marker in ("FAILED", "NEEDS_ATTENTION"):
            (out / marker).unlink(missing_ok=True)
        excluded = {"artifact-manifest.json", "progress.json", "heartbeat", "DONE"}
        write_json(out / "artifact-manifest.json", {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                   for p in out.iterdir() if p.is_file() and p.name not in excluded})
        status(out, "artifacts_complete_review_pending")
        (out / "DONE").write_text("I completed analysis from saved probabilities; visual review remains pending.\n")
        return 0
    except Exception as error:
        metadata.update(execution_status="failed", failed_utc=utc_now(), error=f"{type(error).__name__}: {error}")
        write_json(out / "metadata.json", metadata)
        (out / "FAILED").write_text(traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True, help="My frozen authors checkout, including Git provenance")
    parser.add_argument("--probe-run", type=Path, required=True, help="My native full-corpus prompt-split probe output directory")
    parser.add_argument("--out", type=Path, required=True, help="A new or empty directory for the primary L12 results")
    parser.add_argument("--resume-analysis", action="store_true", help="I reuse and verify saved probabilities; I do not reload or run the model")
    args = parser.parse_args()
    return resume_analysis(args) if args.resume_analysis else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
