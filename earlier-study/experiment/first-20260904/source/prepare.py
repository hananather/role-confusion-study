"""Freeze the authors' templates, ten current Wikipedia pages, and neutral text.

Run once with `python prepare.py`; subsequent runs verify and reuse inputs.json.
Downloads public text and metadata only. No model weights or inference.
"""
import csv
import hashlib
import io
import json
import platform
import random
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Comment, NavigableString
from datasets import load_dataset

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
CSV_PATH = DATA / "templates.csv"
SEED, BUFFER = 123, 2_000
HTTP = requests.Session()
HTTP.headers["User-Agent"] = "RoleSteeringResearch/1.0 (small MATS research sample; https://github.com/hananather)"
REJECT = re.compile(r"<\|[^>]+\|>|\[INST\]|ignore (?:all |the )?(?:previous|prior) instructions|"
                    r"you are chatgpt|system prompt|prompt injection|"
                    r"^\s*(?:system|developer|user|assistant|tool)\s*:", re.I | re.M)


def sha(text):
    return hashlib.sha256(text if isinstance(text, bytes) else text.encode()).hexdigest()


def fetch(url):
    response = HTTP.get(url, timeout=40)
    response.raise_for_status()
    return response


def resolve(kind, name, revision="main"):
    return fetch(f"https://huggingface.co/api/{kind}/{name}/revision/{revision}").json()["sha"]


def clean_html(raw):
    # Authors' standard-agent notebook cell 9, unchanged normalization order.
    soup = BeautifulSoup(raw, "lxml")
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    title = soup.head.title.extract()
    soup.head.clear()
    soup.head.append(title)
    for tag in soup.find_all(True):
        tag.attrs = {}
    for node in soup.find_all(string=lambda s: isinstance(s, NavigableString) and not s.strip()):
        node.extract()
    for tag in soup.find_all(["script", "style", "footer", "nav"]):
        tag.decompose()
    return str(soup)


def pages(spec):
    stream = load_dataset(spec["name"], spec["config"], split=spec["split"],
                          revision=spec["revision"], streaming=True).shuffle(seed=SEED, buffer_size=BUFFER)
    selected, rejected = [], []
    for index, row in enumerate(stream):
        if index >= 200:
            raise RuntimeError("No ten qualifying Wikipedia pages within 200 candidates; inspect source access.")
        response = fetch(row["url"])
        raw = response.content.decode("utf-8", errors="ignore")
        html = clean_html(raw)
        size = len(html.encode("utf-8"))
        revision = re.search(r'"wgRevisionId"\s*:\s*(\d+)', raw)
        if not 5 * 1024 <= size <= 10 * 1024:
            rejected.append({"url": row["url"], "bytes": size})
            continue
        if revision is None:
            raise RuntimeError(f"Cannot establish revision from retrieved HTML: {row['url']}")
        selected.append({"id": f"page-{len(selected):02d}", "url": response.url,
                         "title": row["title"], "html": html, "sha256": sha(html),
                         "raw_sha256": sha(response.content), "revision_id": int(revision[1]),
                         "revision_url": f"https://en.wikipedia.org/w/index.php?oldid={revision[1]}",
                         "source_id": row["id"], "stream_index": index, "bytes": size})
        print(f"Wikipedia {len(selected)}/10: {row['title']} ({size} bytes)", flush=True)
        if len(selected) == 10:
            return selected, rejected
    raise RuntimeError("Wikipedia stream exhausted.")


def neutral(spec, count, seen):
    stream = load_dataset(spec["name"], spec.get("config"), split=spec["split"],
                          revision=spec["revision"], streaming=True).shuffle(seed=SEED, buffer_size=BUFFER)
    selected, rejected = [], 0
    for index, row in enumerate(stream):
        raw = row["text"]
        text = raw[:8_000]
        base_hash, text_hash = sha(raw), sha(text)
        if len(text) < 400 or REJECT.search(text) or base_hash in seen or text_hash in seen:
            rejected += 1
            continue
        seen.update([base_hash, text_hash])
        selected.append({"source": spec["source"], "text": text, "sha256": text_hash,
                         "base_sha256": base_hash, "stream_index": index,
                         "source_id": row.get("id") or {"url": row.get("url"), "timestamp": row.get("timestamp")}})
        if len(selected) == count:
            print(f"Neutral {spec['source']}: {count} accepted; {rejected} rejected", flush=True)
            return selected, rejected
    raise RuntimeError(f"Insufficient acceptable text in {spec['name']}")


def fingerprint(bundle):
    copied = dict(bundle)
    copied["provenance"] = {k: v for k, v in bundle["provenance"].items() if k != "bundle_sha256"}
    return sha(json.dumps(copied, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def verify(bundle):
    assert fingerprint(bundle) == bundle["provenance"]["bundle_sha256"], "Input bundle hash mismatch"
    assert sha((DATA / "templates.csv").read_bytes()) == bundle["provenance"]["templates_sha256"]
    assert len(bundle["templates"]) == 211 and len(bundle["pages"]) == 10 and len(bundle["cases"]) == 50
    assert len(bundle["corpus"]) == 500 and len({x["sha256"] for x in bundle["corpus"]}) == 500
    for item in bundle["pages"] + bundle["corpus"]:
        assert sha(item.get("html", item.get("text"))) == item["sha256"], item["id"]
    assert len({(c["template_id"], c["page_id"]) for c in bundle["cases"]}) == 50
    for pool in ("probe", "direction"):
        rows = [x for x in bundle["corpus"] if x["pool"] == pool]
        assert len(rows) == 250 and sum(x["source"] == "c4" for x in rows) == 63


def main():
    DATA.mkdir(exist_ok=True)
    target = DATA / "inputs.json"
    if target.exists():
        bundle = json.loads(target.read_text())
        verify(bundle)
        print(f"Verified existing frozen inputs: {target}")
        return
    raw_csv = CSV_PATH.read_bytes()
    templates = [{"id": f"template-{i:03d}", "source_row": i,
                  "model": r["variant_model"] or None, "role": r["variant_role"] or None,
                  "raw_template": r["variant_template"],
                  "template": re.sub(r"\\n", "\n", r["variant_template"])}
                 for i, r in enumerate(csv.DictReader(io.StringIO(raw_csv.decode()))) ]
    assert len(templates) == 211 and all(t["template"].count("[CONTENT]") == 1 for t in templates)
    specs = [{"source": "c4", "name": "allenai/c4", "config": "en", "split": "validation"},
             {"source": "dolma3", "name": "allenai/dolma3_mix-150B-1025", "split": "train"},
             {"source": "wikipedia", "name": "wikimedia/wikipedia", "config": "20231101.en", "split": "train"}]
    for spec in specs:
        spec["revision"] = resolve("datasets", spec["name"], "3a8349c" if spec["source"] == "dolma3" else "main")
    model_revision = resolve("models", "openai/gpt-oss-20b")
    frozen_pages, rejected_pages = pages(specs[2])
    seen = set()
    c4, c4_rejected = neutral(specs[0], 126, seen)
    dolma, dolma_rejected = neutral(specs[1], 374, seen)
    corpus = []
    for part, pool in enumerate(("probe", "direction")):
        rows = c4[part * 63:(part + 1) * 63] + dolma[part * 187:(part + 1) * 187]
        random.Random(SEED + part).shuffle(rows)
        corpus.extend({"id": f"{pool}-{i:03d}", "pool": pool, **row} for i, row in enumerate(rows))
    reserved = [templates[157], templates[159]]
    assert [t["role"] for t in reserved] == ["user", "tool"]
    excluded = {t["id"] for t in templates if t["template"] in {r["template"] for r in reserved}}
    pairs = [(t["id"], p["id"]) for t in templates if t["id"] not in excluded for p in frozen_pages]
    chosen = random.Random(SEED).sample(pairs, 50)
    cases = [{"id": f"case-{i:03d}", "template_id": t, "page_id": p} for i, (t, p) in enumerate(chosen)]
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": SEED,
                  "upstream_commit": "ec333c40fd43fe991e1ebf66765051b6d7e35784",
                  "templates_sha256": sha(raw_csv), "prepare_sha256": sha(Path(__file__).read_bytes()),
                  "runtime": {"python": platform.python_version(), **{name: version(name) for name in
                              ("datasets", "numpy", "beautifulsoup4", "lxml", "requests")}},
                  "datasets": specs, "shuffle_buffer": BUFFER, "rejected_pages": rejected_pages,
                  "neutral_rejected": {"c4": c4_rejected, "dolma3": dolma_rejected},
                  "excluded_template_ids": sorted(excluded),
                  "local_choices": ["New current Wikipedia sample; author HTML cleanup and 5–10KiB filter.",
                    "Neutral stream buffer 2000 instead of 50000; 63 C4 +187 Dolma per 250 fixes upstream integer rounding.",
                    "Minimum 400 and maximum 8000 characters; reject obvious role/attack text; later tokenizer caps at 1024.",
                    "Deduplicate full and retained text hashes across all 500; fixed shuffle within each 250-text pool.",
                    "50 unique template-page pairs sampled without replacement before outcomes; first20 control subset.",
                    "Calibration rows157/159 are closest preserved fake declarations, not verified exact post strings.",
                    "Template decoding exactly re.sub(r'\\\\n', newline, raw), from authors' notebook cell10."]}
    carrier = next(row for row in corpus if len(row["text"]) >= 2000)
    bundle = {"model_revision": model_revision, "pages": frozen_pages, "templates": templates,
              "cases": cases, "control_case_ids": [c["id"] for c in cases[:20]], "corpus": corpus,
              "calibration": {"user_template_id": reserved[0]["id"], "tool_template_id": reserved[1]["id"],
                              "carrier_text": carrier["text"][:2000], "carrier_source_id": carrier["id"]},
              "provenance": provenance}
    provenance["bundle_sha256"] = fingerprint(bundle)
    (DATA / "templates.csv").write_bytes(raw_csv)
    target.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    verify(bundle)
    print(f"Frozen and verified {target}; bundle SHA256 {provenance['bundle_sha256']}")


if __name__ == "__main__":
    main()
