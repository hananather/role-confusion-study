"""I exercise queue identity, partial progress and shutdown without GPU work."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .frozen_harness.backend import Generation
from .worker import execute_request, read_request, run, write_json


class Backend:
    metadata = {"backend": "test-fixture"}
    def __init__(self, config=None): self.closed = False
    def close(self): self.closed = True
    def generate_steered(self, prompt, spans, **kw):
        kw["on_progress"]({"phase": "generation", "token_ids": [5], "generated_tokens": 1, "text": "test"})
        return Generation("test", [5], 2, 1, .1, .01, 0., "stop", "<|return|>", False), {"edited_positions": 0}


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.out = Path(self.temp.name)
        (self.out / "requests").mkdir()
    def tearDown(self): self.temp.cleanup()
    def request(self, name="request-1"):
        path = self.out / "requests" / f"{name}.json"
        write_json(path, {"schema_version": 1, "request_id": name, "arm_id": "none", "prompt": "ab",
                          "char_spans": {}, "seed": 1235, "purpose": "engineering_pilot", "max_new_tokens": 64})
        return path
    def test_response_hash_progress_and_raw_generation(self):
        path = self.request(); state = {}
        result = execute_request(Backend(), path, self.out, state)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["request_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(result["generation"]["token_ids"], [5])
        self.assertGreater(state["progress_unix"], 0)
        self.assertTrue((self.out / "progress" / path.name).is_file())
        self.assertFalse(list(self.out.rglob("*.tmp")))
    def test_mismatched_request_id_is_not_executed(self):
        path = self.request()
        row = json.loads(path.read_text()); row["request_id"] = "another"
        write_json(path, row)
        result = execute_request(Backend(), path, self.out)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["type"], "ValueError")
        self.assertNotIn("generation", result)
    def test_symlink_request_is_rejected(self):
        source = self.request()
        path = source.with_name("linked.json"); path.symlink_to(source)
        with self.assertRaises(ValueError): read_request(path)
    def test_bounded_idle_worker_closes_and_preserves_completion(self):
        self.request()
        backend = Backend()
        run({"out_dir": str(self.out), "worker_seconds": 3, "idle_timeout_s": .01},
            backend_factory=lambda config: backend)
        self.assertTrue(backend.closed)
        self.assertEqual(json.loads((self.out / "DONE.json").read_text())["completed"], 1)
        self.assertTrue(json.loads((self.out / "EXIT.json").read_text())["model_closed"])
        self.assertGreater(json.loads((self.out / "heartbeat.json").read_text())["progress_unix"], 0)
        with self.assertRaises(FileExistsError):
            run({"out_dir": str(self.out)}, backend_factory=lambda config: backend)


if __name__ == "__main__": unittest.main()
