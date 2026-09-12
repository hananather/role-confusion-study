"""I exchange data with one H100 worker; model commands never reach host shell code."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

from .frozen_harness.backend import ContextLimitError, Generation


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


class RemoteBackend:
    def __init__(self, connection, remote_results, cache, arms, *, transport=None):
        self.connection = dict(connection)
        self.remote_results = remote_results
        if not re.fullmatch(r"/workspace/results/agent-steering/[A-Za-z0-9_-]+", remote_results):
            raise ValueError("I require one fixed worker output directory")
        self.cache = Path(cache); self.cache.mkdir(parents=True, exist_ok=True)
        self.arms = {a["arm_id"]: a for a in arms}
        self.arm = self.arms["none"]
        self.context = {}
        self._transport = transport or self._ssh
        self.ready = None

    def _ssh(self, command, payload=None):
        c = self.connection
        argv = ["ssh", "-i", c["key"], "-p", str(c["port"]), "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
                "root@" + c["host"], command]
        result = subprocess.run(argv, input=payload, capture_output=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError("My fixed SSH transport failed: " + result.stderr.decode(errors="replace")[:400])
        return result.stdout

    def read_remote(self, relative):
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", relative) or ".." in Path(relative).parts:
            raise ValueError("Invalid worker artifact path")
        target = shlex.quote(self.remote_results + "/" + relative)
        raw = self._transport(f"if test -f {target}; then cat {target}; fi")
        return json.loads(raw) if raw.strip() else None

    def write_remote(self, relative, payload):
        if not re.fullmatch(r"[A-Za-z0-9_./-]+", relative) or ".." in Path(relative).parts:
            raise ValueError("Invalid worker artifact path")
        final = self.remote_results + "/" + relative
        temporary = final + ".uploading"
        parent = str(Path(final).parent)
        command = (f"umask 077; mkdir -p {shlex.quote(parent)}; cat > {shlex.quote(temporary)}"
                   f" && mv {shlex.quote(temporary)} {shlex.quote(final)}")
        self._transport(command, payload)

    def wait_ready(self, timeout_s=900):
        started = time.monotonic()
        while time.monotonic() - started < timeout_s:
            ready = self.read_remote("READY.json")
            if ready:
                self.ready = ready
                (self.cache / "READY.json").write_text(json.dumps(ready, indent=2) + "\n")
                return ready
            exit_record = self.read_remote("EXIT.json") or self.read_remote("FAILED.json")
            if exit_record:
                raise RuntimeError("The H100 worker exited before readiness: " + str(exit_record))
            time.sleep(3)
        raise TimeoutError("The H100 worker did not become ready in its setup window")

    def select(self, arm_id, **context):
        self.arm = self.arms[arm_id]
        self.context = context

    @property
    def metadata(self):
        metadata = dict((self.ready or {}).get("backend", (self.ready or {}).get("metadata", {})))
        metadata["steering"] = {**self.arm, "layer_zero_based": 11, "site": "TransformerBlock output",
                                "probe": "sucat_L12 (prompt split)", "backend": "remote CUDA"}
        return metadata

    def generate_steered(self, prompt, char_spans, *, seed, max_new_tokens=4096, temperature=1.0,
                         timeout_s=300, on_progress=None):
        request = {"schema_version": 1, "arm_id": self.arm["arm_id"], "prompt": prompt,
                   "char_spans": char_spans, "seed": int(seed), "max_new_tokens": int(max_new_tokens),
                   "temperature": float(temperature), "timeout_s": float(timeout_s), **self.context}
        request_id = "r-" + hashlib.sha256(canonical(request)).hexdigest()[:32]
        request["request_id"] = request_id
        raw = canonical(request)
        expected = hashlib.sha256(raw).hexdigest()
        local_request = self.cache / f"{request_id}.request.json"
        if local_request.exists() and local_request.read_bytes() != raw:
            raise ValueError("A request ID was reused with different content")
        local_request.write_bytes(raw)
        response = self.read_remote(f"responses/{request_id}.json")
        if response is None:
            self.write_remote(f"requests/{request_id}.json", raw)
        started, last_progress = time.monotonic(), None
        while response is None:
            if time.monotonic() - started > timeout_s + 120:
                raise TimeoutError("My worker response exceeded the generation and transport deadline")
            time.sleep(2)
            response = self.read_remote(f"responses/{request_id}.json")
            if on_progress:
                update = self.read_remote(f"progress/{request_id}.json")
                if update and update != last_progress:
                    on_progress(update); last_progress = update
            if response is None and (self.read_remote("EXIT.json") or self.read_remote("FAILED.json")):
                raise RuntimeError("The worker exited with an outstanding request; I preserve it as incomplete")
        (self.cache / f"{request_id}.response.json").write_text(json.dumps(response, indent=2) + "\n")
        if response.get("request_sha256") != expected or response.get("request_id") != request_id:
            raise RuntimeError("My worker response does not match the exact request bytes")
        if response.get("status") != "ok":
            error = response.get("error", {})
            if error.get("type") == "ContextLimitError":
                raise ContextLimitError(error.get("message", "Context limit"))
            raise RuntimeError("H100 generation failed: " + str(error))
        return Generation(**response["generation"]), response["stats"]

    def finish(self, outcome):
        self.write_remote("CLIENT-DONE.json", canonical(outcome))
        self.write_remote("STOP", b"The fixed local episode queue has ended.\n")

    def close(self):
        pass
