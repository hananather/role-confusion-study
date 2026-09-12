"""I freeze Figure 23 candidate user turns from existing local Arrow caches only.

I use the authors' NB01 cells 3–5 selection, before response-dependent filtering.
I do not import model code, use load_dataset, or download missing inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
AUTHORS = HERE.parents[1] / "prompt-injection-as-role-confusion"
AUTHOR_COMMIT = "ec333c40fd43fe991e1ebf66765051b6d7e35784"
SEED = 1234
SMOKE_IDS = [0, 1, 2, 100, 101]
SOURCES = {
    "oasst": ("OpenAssistant___oasst1", "oasst1-train.arrow", "oasst1", "default"),
    "toxicchat": ("lmsys___toxic-chat", "toxic-chat-train.arrow", "toxic-chat", "toxicchat1123"),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def locate(cache_root: Path, dataset: str, explicit: str | None) -> Path:
    directory, filename, _, _ = SOURCES[dataset]
    matches = [Path(explicit).expanduser()] if explicit else sorted((cache_root / directory).rglob(filename))
    matches = [p for p in matches if p.is_file()]
    if len(matches) != 1:
        raise ValueError(f"I require one local {dataset} train Arrow file; found {len(matches)}. Supply its explicit path.")
    return matches[0].resolve()


def read_cache(path: Path, dataset: str):
    import pyarrow as pa

    _, _, expected_name, expected_config = SOURCES[dataset]
    info_path = path.parent / "dataset_info.json"
    info = json.loads(info_path.read_text())
    if (info.get("dataset_name"), info.get("config_name")) != (expected_name, expected_config):
        raise ValueError(f"Unexpected dataset identity for {dataset}")
    before = path.stat()
    with pa.memory_map(str(path), "r") as stream:
        table = pa.ipc.open_stream(stream).read_all()
    if table.num_rows != info["splits"]["train"]["num_examples"]:
        raise ValueError(f"The {dataset} cache row count differs from dataset_info.json")
    columns = (["message_id", "parent_id", "message_tree_id", "role", "text", "lang", "tree_state", "synthetic"]
               if dataset == "oasst" else ["conv_id", "user_input", "human_annotation"])
    rows = table.select(columns).to_pylist()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"The {dataset} cache changed while I read it")
    for ix, row in enumerate(rows):
        row["source_row_ix"] = ix
    provenance = {
        "dataset": "OpenAssistant/oasst1" if dataset == "oasst" else "lmsys/toxic-chat",
        "config": expected_config, "split": "train", "raw_train_rows": len(rows),
        "arrow_path": str(path), "arrow_sha256": file_digest(path),
        "dataset_info_path": str(info_path), "dataset_info_sha256": file_digest(info_path),
        "cache_revision_directory": path.parent.name,
        "revision_boundary": "I record the cache directory identity and exact bytes; this is not proof of the paper's original dataset snapshot.",
    }
    return rows, provenance


def select_oasst(rows, limit=100):
    """I preserve source row/tree/child order, validating the full path before truncation."""
    filtered = [r for r in rows if r["lang"] == "en" and r["tree_state"] == "ready_for_export"]
    trees = {}
    for row in filtered:
        trees.setdefault(row["message_tree_id"], []).append(row)
    valid = []
    missing_roots = 0
    for tree_id, messages in trees.items():
        by_id = {m["message_id"]: m for m in messages}
        if len(by_id) != len(messages):
            raise ValueError("Duplicate OASST message ID within a tree")
        children = {}
        for message in messages:
            if message["parent_id"]:
                children.setdefault(message["parent_id"], []).append(message["message_id"])
        roots = [m for m in messages if not m["parent_id"]]
        if not roots:
            missing_roots += 1
            continue
        path = [roots[0]]
        seen = {roots[0]["message_id"]}
        while path[-1]["message_id"] in children:
            child_id = children[path[-1]["message_id"]][0]
            if child_id in seen:
                raise ValueError("Cycle in an OASST first-child path")
            seen.add(child_id)
            path.append(by_id[child_id])
        users = [m for m in path if m["role"] == "prompter"]
        if any(not 100 <= len(m["text"]) <= 500 for m in users):
            continue
        if not users:
            raise ValueError("An eligible OASST path has no user messages")
        valid.append({"dataset": "oasst", "source_conv_id": tree_id, "users": users[:2],
                      "full_user_turn_count": len(users), "path": path})
    return valid[:limit], {
        "raw_train_rows": len(rows), "english_ready_rows": len(filtered),
        "english_ready_trees": len(trees), "trees_without_root": missing_roots,
        "eligible_first_child_conversations_before_limit": len(valid),
        "selected_conversations": min(limit, len(valid)),
    }


def select_toxicchat(rows, limit=100):
    import numpy as np

    # Dataset.shuffle(seed=1234) uses this generator/permutation in the local datasets implementation.
    permutation = np.random.default_rng(SEED).permutation(len(rows)).tolist()
    eligible = [ix for ix in permutation if 100 <= len(rows[ix]["user_input"]) <= 500]
    selected = []
    for ix in eligible[:limit]:
        row = rows[ix]
        selected.append({"dataset": "toxicchat", "source_conv_id": row["conv_id"],
                         "users": [row], "full_user_turn_count": 1,
                         "shuffled_rank": permutation.index(ix)})
    return selected, {"raw_train_rows": len(rows), "eligible_user_inputs_before_limit": len(eligible),
                      "selected_conversations": len(selected), "shuffle_seed": SEED,
                      "permutation_sha256": digest(json.dumps(permutation, separators=(",", ":")).encode())}


def freeze_conversations(selected):
    result = []
    for conv_id, conv in enumerate(selected):
        turns = []
        for ix, row in enumerate(conv["users"]):
            if conv["dataset"] == "oasst" and row["synthetic"] is not False:
                raise ValueError("I cannot label an OASST user turn with synthetic/unknown provenance as human source data")
            text = row["text"] if conv["dataset"] == "oasst" else row["user_input"]
            turns.append({"user_query_ix": ix, "user_query": text,
                          "source_message_id": row.get("message_id"), "source_row_ix": row["source_row_ix"],
                          "source_text_sha256": digest(text.encode()),
                          "oasst_source_synthetic": row.get("synthetic"),
                          "toxicchat_human_annotation": row.get("human_annotation")})
        record = {"conv_id": conv_id, "dataset": conv["dataset"], "source_conv_id": conv["source_conv_id"],
                  "source_kind": "cached_dataset_user_turns", "generated_by_this_preparation": False,
                  "full_source_path_user_turn_count": conv["full_user_turn_count"], "user_turns": turns}
        if conv["dataset"] == "oasst":
            record["source_first_child_path"] = [
                {k: m[k] for k in ["message_id", "parent_id", "role", "source_row_ix"]} for m in conv["path"]
            ]
        else:
            record["source_shuffled_rank"] = conv["shuffled_rank"]
        result.append(record)
    return result


def csv_bytes(conversations):
    fields = ["conv_id", "dataset", "user_query_ix", "user_query", "source_conv_id", "source_message_id", "source_row_ix", "source_text_sha256"]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for conv in conversations:
        for turn in conv["user_turns"]:
            writer.writerow({k: (turn[k] if k in turn else conv[k]) for k in fields})
    return stream.getvalue().encode()


def jsonl_bytes(rows):
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows).encode()


def verify_artifacts(out: Path):
    manifest = json.loads((out / "manifest.json").read_text())
    for name, entry in manifest["artifacts"].items():
        if file_digest(out / name) != entry["sha256"]:
            raise ValueError(f"Prepared artifact checksum differs: {name}")
    conversations = [json.loads(line) for line in (out / "conversations.jsonl").read_text().splitlines()]
    if len(conversations) != 200 or Counter(c["dataset"] for c in conversations) != {"oasst": 100, "toxicchat": 100}:
        raise ValueError("I expected exactly 100 genuine candidates from each dataset")
    for ix, conv in enumerate(conversations):
        if conv["conv_id"] != ix or not 1 <= len(conv["user_turns"]) <= 2:
            raise ValueError("Candidate IDs or two-turn bound differ")
        for turn_ix, turn in enumerate(conv["user_turns"]):
            if turn["user_query_ix"] != turn_ix or not 100 <= len(turn["user_query"]) <= 500:
                raise ValueError("A user turn violates the source length/index rule")
            if digest(turn["user_query"].encode()) != turn["source_text_sha256"]:
                raise ValueError("Source text changed during export")
    smoke = [json.loads(line) for line in (out / "smoke-conversations.jsonl").read_text().splitlines()]
    if smoke != [conversations[ix] for ix in SMOKE_IDS]:
        raise ValueError("Smoke candidates differ from the prespecified IDs")
    if (out / "user_queries.csv").read_bytes() != csv_bytes(conversations):
        raise ValueError("CSV and JSONL user turns differ")
    return {"conversations": 200, "user_turns": sum(len(c["user_turns"]) for c in conversations), "smoke_ids": SMOKE_IDS}


def self_test():
    def message(mid, parent, tree, role="prompter", size=100):
        return {"message_id": mid, "parent_id": parent, "message_tree_id": tree, "role": role,
                "text": mid[0] * size, "lang": "en", "tree_state": "ready_for_export", "synthetic": False}
    # I verify that the first child wins, and that a bad third user rejects the full path before truncation.
    rows = [message("z", None, "first"), message("b", "z", "first", "assistant"),
            message("a", "z", "first", "assistant"), message("c", "b", "first", size=500),
            message("d", "c", "first", "assistant"), message("e", "d", "first", size=100),
            message("f", None, "bad"), message("g", "f", "bad", "assistant"),
            message("h", "g", "bad"), message("i", "h", "bad", "assistant"),
            message("j", "i", "bad", size=99)]
    for ix, row in enumerate(rows): row["source_row_ix"] = ix
    selected, stats = select_oasst(rows)
    assert len(selected) == 1 and stats["eligible_first_child_conversations_before_limit"] == 1
    assert [m["message_id"] for m in selected[0]["path"]] == ["z", "b", "c", "d", "e"]
    assert [m["message_id"] for m in selected[0]["users"]] == ["z", "c"]
    assert freeze_conversations(selected)[0]["full_source_path_user_turn_count"] == 3
    selected[0]["users"][0]["synthetic"] = True
    try: freeze_conversations(selected)
    except ValueError: pass
    else: raise AssertionError("Synthetic source turn was mislabeled")
    print("Passed model-free tests: first-child order, full-path validation before two-turn truncation, length boundaries and synthetic-source rejection.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("~/.cache/huggingface/datasets").expanduser())
    parser.add_argument("--oasst-arrow")
    parser.add_argument("--toxicchat-arrow")
    parser.add_argument("--out", type=Path, default=HERE / "prepared-source")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.verify:
        print(json.dumps(verify_artifacts(args.out), sort_keys=True))
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("I preserve the frozen outputs; use --verify or choose an empty output directory")
    source_rows, sources, selections, counts = {}, {}, [], {}
    for dataset, explicit in [("oasst", args.oasst_arrow), ("toxicchat", args.toxicchat_arrow)]:
        path = locate(args.cache_root, dataset, explicit)
        source_rows[dataset], sources[dataset] = read_cache(path, dataset)
        selected, counts[dataset] = (select_oasst if dataset == "oasst" else select_toxicchat)(source_rows[dataset])
        if len(selected) != 100:
            raise ValueError(f"Only {len(selected)} eligible cached {dataset} conversations; I will not substitute synthetic data")
        selections.extend(selected)
    conversations = freeze_conversations(selections)
    smoke = [conversations[ix] for ix in SMOKE_IDS]
    artifacts = {"conversations.jsonl": jsonl_bytes(conversations), "user_queries.csv": csv_bytes(conversations),
                 "smoke-conversations.jsonl": jsonl_bytes(smoke), "smoke-user_queries.csv": csv_bytes(smoke)}
    notebook = AUTHORS / "experiments/role-analysis/01-get-conversations-data.ipynb"
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "stage": "user_candidates_only_before_model_generation",
                "authors_commit": AUTHOR_COMMIT, "source_notebook": str(notebook), "source_notebook_sha256": file_digest(notebook),
                "source_cells_zero_based": [1, 3, 4, 5], "preparation_script_sha256": file_digest(Path(__file__)),
                "dataset_sources": sources, "candidate_counts": counts,
                "n_candidate_conversations": len(conversations), "n_user_turns": sum(len(c["user_turns"]) for c in conversations),
                "turn_count_distribution": dict(Counter(len(c["user_turns"]) for c in conversations)),
                "smoke_selection": {"conv_ids": SMOKE_IDS, "rule": "first three selected OASST and first two selected ToxicChat; fixed before model outcomes"},
                "boundaries": ["I export only cached source user text, with no model-generated assistant or reasoning text.",
                               "These 200 candidates are not 200 post-filter model-valid conversations or measured Figure 23 results.",
                               "The paper describes 200 conversations; the frozen NB02 later samples at most 30 after response-dependent filters (comment: 100 for full test).",
                               "I preserve full-path OASST checks before taking the first two user turns.",
                               "ToxicChat human_annotation describes its annotation process, not whether this preparation generated the user text.",
                               "I perform no network, model, download, API or compute-rental action."],
                "artifacts": {name: {"sha256": digest(data), "bytes": len(data)} for name, data in artifacts.items()}}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in artifacts.items():
        with (args.out / name).open("xb") as stream: stream.write(data)
    with (args.out / "manifest.json").open("x") as stream: json.dump(manifest, stream, indent=2); stream.write("\n")
    print(json.dumps({**verify_artifacts(args.out), "counts": counts, "output": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
