"""E4 conversation regeneration on CUDA: the authors' NB01 local fallback, batched and sharded.

Source: prompt-injection-as-role-confusion/experiments/role-analysis/01-get-conversations-data.ipynb,
cells 10 and 11 (generate_local_batch, generate_convs_data_local). Kept as the authors wrote it:
apply_chat_template with add_generation_prompt, left padding, truncation at 12,288 input tokens,
do_sample=True, temperature 1.0, max_new_tokens 4,000, eos = tokenizer.eos_token_id, two rounds
(user_query_ix 0 then 1), a conversation whose round-0 reply is empty is skipped in round 1, and
the raw CSV columns conv_id, dataset, user_query_ix, user_query, assistant, state_text, model,
model_prefix, written as <model_prefix>-raw.csv. The CoT/assistant split (cell 13,
label_content_roles) runs on the Mac afterwards from state_text, exactly as in the notebook.

Additions for the cloud job: sharding by conversation id (contiguous blocks after one fixed
shuffle, so a shard holds complete conversations), a per-batch seed derived from the batch's
identities, JSONL side records with timing and seeds, checkpoint every 20 rows, resume, heartbeat,
STOP file, --pilot N, --hours.

Seed note (assumption, stated): HF generate samples one RNG stream per batch, so a seed cannot be
set per row. The seed is blake2b(experiment|round|first conv_id in batch|replicate) and every row
records it with seed_scope = "batch". Batches are deterministic (conv ids sorted within a shard,
fixed batch size), so a rerun reproduces the draws.

Input: conversations_file (CSV or JSONL) with conv_id, dataset, user_query_ix, user_query, the
authors' user_queries_df (cell 5) exported on the Mac by `python job_conversations.py --prepare-oasst`
(needs `datasets`; approval required for the download).

Usage on the pod:
  python job_conversations.py --manifest manifest.json --results /workspace/results/<run>/shard-0 --hours 1
CUDA-free:
  python job_conversations.py --manifest manifest.json --plan
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import jobcommon as jc

MAX_INPUT_TOKENS = 1024 * 12
MAX_NEW_TOKENS = 4_000
TEMPERATURE = 1.0
BATCH_SIZE = 16
MODEL_PREFIX = "gptoss-20b"
RAW_COLUMNS = ["conv_id", "dataset", "user_query_ix", "user_query", "assistant", "state_text", "model", "model_prefix"]
ROW_KEY = ("conv_id", "user_query_ix")


# ----------------------------------------------------------------------------- data

def load_queries(manifest):
    path = jc.resolve_path(manifest, manifest["conversations_file"])
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    else:
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    queries = []
    for r in rows:
        ix = int(r["user_query_ix"])
        if ix > 1:
            continue  # the authors truncate to two user turns (cell 5)
        queries.append({"conv_id": int(r["conv_id"]), "dataset": r.get("dataset", "oasst"),
                        "user_query_ix": ix, "user_query": r["user_query"]})
    return queries


def plan_shard(manifest, pilot=0):
    queries = load_queries(manifest)
    conv_ids = sorted({q["conv_id"] for q in queries})
    n = manifest.get("n_conversations")
    if n:
        conv_ids = conv_ids[:int(n)]
    mine = jc.shard_items(conv_ids, manifest["shard_index"], manifest["shard_count"], manifest["shuffle_seed"])
    if pilot:
        mine = mine[:pilot]
    mine = sorted(mine)
    keep = set(mine)
    my_queries = [q for q in queries if q["conv_id"] in keep]
    rounds = {}
    for q in my_queries:
        rounds.setdefault(q["user_query_ix"], []).append(q)
    for ix in rounds:
        rounds[ix].sort(key=lambda q: q["conv_id"])
    return rounds, {"conversations_total": len(conv_ids), "conversations_shard": len(mine),
                    "round_sizes": {ix: len(v) for ix, v in sorted(rounds.items())}}


def prepare_oasst(n_samples, out_path):
    """The authors' load_oasst_conversations (cell 4) and cell 5 query table, oasst only. Needs `datasets`."""
    from datasets import load_dataset
    ds = load_dataset("OpenAssistant/oasst1", split="train", streaming=False)
    rows = [{"message_tree_id": ex["message_tree_id"], "message_id": ex["message_id"], "parent_id": ex["parent_id"],
             "role": ex["role"], "text": ex["text"]}
            for ex in ds if ex["lang"] == "en" and ex["tree_state"] == "ready_for_export"]
    trees = {}
    for r in rows:
        trees.setdefault(r["message_tree_id"], []).append(r)
    conversations = []
    for tid in list(trees.keys()):
        msgs = trees[tid]
        id2msg = {m["message_id"]: m for m in msgs}
        children = {}
        for m in msgs:
            if m["parent_id"]:
                children.setdefault(m["parent_id"], []).append(m["message_id"])
        roots = [m for m in msgs if not m["parent_id"]]
        if not roots:
            continue
        root = roots[0]
        path_ids, cur = [root["message_id"]], root["message_id"]
        while cur in children:
            cur = children[cur][0]
            path_ids.append(cur)
        conv = [id2msg[mid] for mid in path_ids]
        if all(100 <= len(m["text"]) <= 500 for m in conv if m["role"] == "prompter"):
            conversations.append(conv)
        if len(conversations) >= n_samples:
            break
    out = []
    for conv_id, conv in enumerate(conversations):
        user_turns = [m["text"] for m in conv if m["role"] == "prompter"][:2]
        for ix, text in enumerate(user_turns):
            out.append({"conv_id": conv_id, "dataset": "oasst", "user_query_ix": ix, "user_query": text})
    with Path(out_path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["conv_id", "dataset", "user_query_ix", "user_query"])
        writer.writeheader()
        writer.writerows(out)
    print(f"wrote {len(out)} user queries from {len(conversations)} conversations to {out_path}")


# ----------------------------------------------------------------------------- generation (NB01 cell 11)

def generate_local_batch(tokenizer, model, message_histories, max_new_tokens):
    import torch
    enc = tokenizer.apply_chat_template(message_histories, tokenize=True, add_generation_prompt=True,
                                        return_tensors="pt", padding=True, truncation=True,
                                        max_length=MAX_INPUT_TOKENS, return_dict=True)
    input_ids = enc["input_ids"].to("cuda")
    attention_mask = enc["attention_mask"].to("cuda")
    prompt_len = input_ids.shape[1]  # padded length; new tokens start here for every row
    t0 = time.time()
    with torch.inference_mode():
        out = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
                             do_sample=True, temperature=TEMPERATURE, pad_token_id=tokenizer.pad_token_id,
                             eos_token_id=tokenizer.eos_token_id, use_cache=True)
    elapsed = time.time() - t0
    new_tokens = out[:, prompt_len:]
    results = []
    for i in range(out.shape[0]):
        prompt_ids = input_ids[i][attention_mask[i].bool()].tolist()
        gen_ids = new_tokens[i].tolist()
        while gen_ids and gen_ids[-1] == tokenizer.pad_token_id:
            gen_ids.pop()
        asst = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        asst = asst if asst else None
        state_text = tokenizer.decode(prompt_ids + gen_ids, skip_special_tokens=False)
        finished = bool(gen_ids) and gen_ids[-1] == tokenizer.eos_token_id
        results.append({"assistant": asst, "state_text": state_text, "n_gen": len(gen_ids),
                        "prompt_tokens": len(prompt_ids), "finished": finished, "batch_s": elapsed})
    return results


def step_generate(manifest, run, tokenizer, model, result_dir, pilot):
    import torch
    rounds, counts = plan_shard(manifest, pilot)
    batch_size = int(manifest.get("batch_size", BATCH_SIZE))
    max_new = int(manifest.get("max_new_tokens", MAX_NEW_TOKENS))
    experiment_id, replicate = manifest["experiment_id"], int(manifest["replicate"])
    side = jc.Jsonl(result_dir / "conversations.jsonl", ROW_KEY, manifest["checkpoint_every"])
    responses = {}  # (conv_id, ix) -> assistant text or None, from disk plus this run
    for line in (result_dir / "conversations.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            responses[(r["conv_id"], r["user_query_ix"])] = r["assistant"]
    query_lookup = {(q["conv_id"], q["user_query_ix"]): q for ix in rounds for q in rounds[ix]}
    print(f"[conv] shard {manifest['shard_index']}/{manifest['shard_count']}: {counts}, resume rows {len(responses)}", flush=True)
    run.attempt.update(counts=counts, resume_rows=len(responses))
    run.save()
    t_start, n_done, tot_gen = time.time(), 0, 0
    for round_ix in sorted(rounds):
        histories, ids, skipped = [], [], 0
        for q in rounds[round_ix]:
            key = (q["conv_id"], round_ix)
            if key in responses:
                continue
            if any(responses.get((q["conv_id"], prev), None) is None for prev in range(round_ix)):
                skipped += 1  # prior round missing or empty: record the skip as the authors do
                row = {"conv_id": q["conv_id"], "dataset": q["dataset"], "user_query_ix": round_ix, "user_query": q["user_query"],
                       "assistant": None, "state_text": None, "model": MODEL_PREFIX, "model_prefix": MODEL_PREFIX,
                       "skipped_prior_failed": True, "seed": None, "seed_scope": "batch", "experiment_id": experiment_id,
                       "shard_index": manifest["shard_index"], "replicate": replicate}
                side.write(row)
                responses[key] = None
                continue
            msgs = []
            for prev in range(round_ix):
                msgs.append({"role": "user", "content": query_lookup[(q["conv_id"], prev)]["user_query"]})
                msgs.append({"role": "assistant", "content": responses[(q["conv_id"], prev)]})
            msgs.append({"role": "user", "content": q["user_query"]})
            histories.append(msgs)
            ids.append(q)
        print(f"[conv] round {round_ix}: {len(histories)} to generate, {skipped} skipped", flush=True)
        for b0 in range(0, len(histories), batch_size):
            if jc.stop_requested(result_dir):
                print("[conv] STOP file present; stopping after the last completed batch", flush=True)
                run.attempt["stopped_by_stop_file"] = True
                side.close()
                return {"done": n_done, "tokens": tot_gen, "stopped": True}
            batch_h, batch_q = histories[b0:b0 + batch_size], ids[b0:b0 + batch_size]
            seed = jc.item_seed(experiment_id, f"round{round_ix}|conv{batch_q[0]['conv_id']}", "sample", replicate)
            torch.manual_seed(jc.torch_seed_from(seed))
            torch.cuda.manual_seed_all(jc.torch_seed_from(seed))
            results = generate_local_batch(tokenizer, model, batch_h, max_new)
            for q, res in zip(batch_q, results):
                row = {"conv_id": q["conv_id"], "dataset": q["dataset"], "user_query_ix": round_ix, "user_query": q["user_query"],
                       "assistant": res["assistant"], "state_text": res["state_text"], "model": MODEL_PREFIX,
                       "model_prefix": MODEL_PREFIX, "n_gen": res["n_gen"], "prompt_tokens": res["prompt_tokens"],
                       "finished": res["finished"], "batch_s": res["batch_s"], "batch_size": len(batch_q),
                       "temperature": TEMPERATURE, "max_new_tokens": max_new, "seed": seed, "seed_scope": "batch",
                       "skipped_prior_failed": False, "experiment_id": experiment_id,
                       "shard_index": manifest["shard_index"], "replicate": replicate, "backend": "cuda-transformers"}
                side.write(row)
                responses[(q["conv_id"], round_ix)] = res["assistant"]
                n_done += 1
                tot_gen += res["n_gen"]
            elapsed = time.time() - t_start
            jc.heartbeat(result_dir, {"stage": f"round{round_ix}", "done": n_done, "elapsed_s": elapsed})
            print(f"[conv] round {round_ix} {min(b0 + batch_size, len(histories))}/{len(histories)} | "
                  f"{results[0]['batch_s']:.0f}s batch | {sum(r['n_gen'] for r in results) / results[0]['batch_s']:.1f} tok/s | "
                  f"finished {sum(r['finished'] for r in results)}/{len(results)} | {elapsed / 60:.1f} min", flush=True)
    side.close()
    write_raw_csv(result_dir)
    total = time.time() - t_start
    if pilot:
        (result_dir / "pilot.json").write_text(json.dumps({
            "pilot_n": pilot, "rows_done": n_done, "elapsed_s": total, "seconds_per_row": total / max(n_done, 1),
            "generated_tokens": tot_gen, "tokens_per_second": tot_gen / max(total, 1e-9),
            "source_sha256": run.source_sha, "manifest_sha256": manifest.get("_sha256"), "time_utc": jc.utc_now()}, indent=2))
    return {"done": n_done, "tokens": tot_gen, "elapsed_s": total}


def write_raw_csv(result_dir):
    """<model_prefix>-raw.csv with the authors' columns, from the JSONL side records."""
    rows = [json.loads(l) for l in (Path(result_dir) / "conversations.jsonl").read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: (r["conv_id"], r["user_query_ix"]))
    out = Path(result_dir) / f"{MODEL_PREFIX}-raw.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in RAW_COLUMNS})
    print(f"[conv] wrote {len(rows)} rows to {out}", flush=True)
    return out


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest")
    parser.add_argument("--results")
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--pilot", type=int, default=0, help="first N conversations of the shard only")
    parser.add_argument("--cache-dir", default="/workspace/hf-cache")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--no-strict-pins", action="store_true")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--csv-only", action="store_true", help="rebuild the raw csv from conversations.jsonl")
    parser.add_argument("--prepare-oasst", type=int, metavar="N", help="Mac: build the query csv from OpenAssistant/oasst1 (download)")
    parser.add_argument("--out", help="output path for --prepare-oasst")
    parser.add_argument("--terminate-self", action="store_true")
    args = parser.parse_args()

    if args.prepare_oasst:
        prepare_oasst(args.prepare_oasst, args.out or "e4/data/oasst_user_queries.csv")
        return 0
    if not args.manifest:
        parser.error("--manifest is required")
    manifest = jc.load_manifest(args.manifest)
    if manifest["job"] != "conversations":
        parser.error("manifest job must be 'conversations'")
    if args.shard_index is not None:
        manifest["shard_index"] = args.shard_index
    if args.shard_count is not None:
        manifest["shard_count"] = args.shard_count
    result_dir = Path(args.results or f"results/{manifest['experiment_id']}/shard-{manifest['shard_index']}").resolve()
    if args.plan:
        rounds, counts = plan_shard(manifest, args.pilot)
        print(json.dumps({"counts": counts, "first_conv_ids": [q["conv_id"] for q in rounds.get(0, [])[:8]],
                          "batches_round0": -(-len(rounds.get(0, [])) // int(manifest.get("batch_size", BATCH_SIZE)))}, indent=2))
        return 0
    if args.csv_only:
        write_raw_csv(result_dir)
        return 0

    jc.set_cache_env(args.cache_dir)
    run = jc.Run(manifest, result_dir, args.hours)
    status, error = "complete", None
    try:
        jc.heartbeat(result_dir, {"stage": "loading"})
        tokenizer, model, meta = jc.load_model(manifest, strict_pins=not args.no_strict_pins)
        run.runtime["model"] = meta
        run.save()
        (result_dir / "model-metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        run.arm_alarm()
        run.runtime["generate"] = step_generate(manifest, run, tokenizer, model, result_dir, args.pilot)
    except BaseException as caught:  # noqa: BLE001
        status, error = "failed", caught
        raise
    finally:
        run.finish(status, error)
        (result_dir / "DONE").write_text(jc.utc_now() + "\n")
        if args.terminate_self:
            jc.terminate_self(result_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
