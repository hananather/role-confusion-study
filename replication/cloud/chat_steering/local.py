"""I prepare local inputs and preserve each account-authenticated generator response."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time


def digest(value):
    return hashlib.sha256(value).hexdigest()


def account_env():
    env = dict(os.environ)
    for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN",
                 "OPENAI_IDENTITY_TOKEN_FILE", "OPENAI_FEDERATION_RULE_ID"):
        env.pop(name, None)
    return env


def forge(args):
    root = args.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in args.stub.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if row["qualifier_type"] == "no_qualifier"]
    if args.limit:
        rows = rows[:args.limit]
    env = account_env()
    auth = subprocess.run(["codex", "login", "status"], env=env, capture_output=True, text=True, check=True)
    if "Logged in using ChatGPT" not in auth.stdout + auth.stderr:
        raise RuntimeError("I require the existing ChatGPT login; API-key fallback is forbidden.")
    metadata = {"generator_model_requested": args.model,
                "auth": "ChatGPT", "api_key_environment_removed": True,
                "stub_sha256": digest(args.stub.read_bytes()),
                "prompt_construction": "e8 forge-stub prompt_text, unchanged UTF-8 bytes",
                "divergences": ["Codex CLI model replaces gemini-2.5-pro",
                    "Original message contents are flattened with role headings by e8 forge-stub; Codex retains its own system instructions",
                    "Temperature 0 is not exposed by codex exec; not asserted equivalent"],
                "rows_requested": len(rows)}
    (root / "generator-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    def one(row):
        target = root / f"row-{row['sr_ix']:03d}"
        target.mkdir(exist_ok=True)
        prompt = row["prompt_text"]
        expected = digest(prompt.encode())
        done = target / "record.json"
        if done.exists():
            old = json.loads(done.read_text())
            if old["prompt_sha256"] != expected or old["model_requested"] != args.model:
                raise RuntimeError("I refuse to reuse a response from another prompt or model.")
            return old
        (target / "prompt.txt").write_text(prompt)
        output = target / "completion.txt"
        argv = ["codex", "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                "-s", "read-only", "-m", args.model, "-c", 'forced_login_method="chatgpt"',
                "-c", 'model_reasoning_effort="low"', "--json", "-o", str(output), "-"]
        start = time.time()
        with tempfile.TemporaryDirectory(prefix="mats-forge-") as cwd:
            try:
                result = subprocess.run(argv, input=prompt, text=True, env=env, cwd=cwd,
                                        capture_output=True, timeout=args.timeout)
                status, stdout, stderr = result.returncode, result.stdout, result.stderr
            except subprocess.TimeoutExpired as error:
                status = "timeout"
                stdout = error.stdout or ""
                stderr = error.stderr or ""
                stdout = stdout.decode() if isinstance(stdout, bytes) else stdout
                stderr = stderr.decode() if isinstance(stderr, bytes) else stderr
        (target / "events.jsonl").write_text(stdout)
        (target / "stderr.log").write_text(stderr)
        response = output.read_text() if output.exists() else ""
        rec = {**row, "output": response, "exit_status": status,
               "model_requested": args.model, "prompt_sha256": expected,
               "elapsed_seconds": time.time() - start,
               "tags_present": "<SYNTHETIC_POLICY>" in response and "</SYNTHETIC_POLICY>" in response}
        done.write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps({"sr_ix": row["sr_ix"], "exit_status": status,
                          "tags_present": rec["tags_present"], "elapsed_seconds": rec["elapsed_seconds"]}), flush=True)
        return rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        records = list(pool.map(one, rows))
    (root / "filled.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    if any(r["exit_status"] != 0 or not r["tags_present"] for r in records):
        raise SystemExit("I preserved incomplete or declined responses. I will not fabricate replacements or retry a declined request.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("forge")
    p.add_argument("--stub", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        parser.error("I allow one to four generator workers.")
    forge(args)


if __name__ == "__main__":
    main()
