"""I check bounded leases and two jobs sharing one model with no GPU or API."""
from pathlib import Path
import tempfile
import threading
import time
import unittest

import numpy as np

from .frozen_harness.backend import Generation
from .persistent_worker import read_lease, run, validate_job, validate_lease
from .worker import write_json


def lease(now, seconds=90):
    return {"schema_version": 1, "approved": True, "lease_id": "session-1", "expected_pod_id": "pod1",
            "issued_unix": now, "lease_expires_unix": now+seconds,
            "shutdown_at_unix": now+600, "budget_remaining_usd": 4.0}


class Backend:
    loads = 0
    last = None
    def __init__(self, config):
        Backend.loads += 1; Backend.last = self
        self.closed = False
        v = np.zeros(2880, np.float32); v[0] = 1
        self.directions = {"tool_minus_cot": v, "gap_tool_cot": np.float32(38.507904)}
        self._metadata = {"directions_file_sha256": "directions-hash", "probe_file_sha256": "probe-hash"}
        self.arms = config["arms"]
    @property
    def metadata(self): return dict(self._metadata)
    def generate_steered(self, prompt, spans, **kw):
        kw["on_progress"]({"phase": "generation", "generated_tokens": 1, "token_ids": [9]})
        return Generation("fixture", [9], 2, 1, .1, .01, 0, "stop", "<|return|>", False), {"edited_positions": 0}
    def close(self): self.closed = True


class Tests(unittest.TestCase):
    def test_lease_bounds_and_identity(self):
        now = 1000
        self.assertEqual(validate_lease(lease(now), "pod1", now)["lease_id"], "session-1")
        variants = [dict(lease(now), expected_pod_id="other"), dict(lease(now), approved=False),
                    dict(lease(now), budget_remaining_usd=0), dict(lease(now), budget_remaining_usd=float("nan")),
                    lease(now, 181), dict(lease(now), shutdown_at_unix=now), lease(now-100)]
        for row in variants:
            with self.subTest(lease=row), self.assertRaises(ValueError): validate_lease(row, "pod1", now)

    def setup(self, directory):
        root = Path(directory)
        cfg = {"out_dir": str(root / "service"), "job_results_root": str(root / "results"),
               "execution_approved": True, "persistent_budget_lease_required": True,
               "lease_path": str(root / "lease.json"), "expected_pod_id": "pod1",
               "directions_file": "fixed-directions.npz", "probe_file": "fixed-probe.npz", "arms": []}
        write_json(cfg["lease_path"], lease(time.time()))
        Backend.loads = 0
        return cfg

    def job(self, cfg, identifier):
        return {"schema_version": 1, "job_id": identifier, "execution_approved": True,
                "out_dir": str(Path(cfg["job_results_root"]) / identifier),
                "directions_file": cfg["directions_file"], "probe_file": cfg["probe_file"],
                "directions_file_sha256": "directions-hash", "probe_file_sha256": "probe-hash",
                "arms": [{"arm_id": "zero", "direction": "tool_minus_cot", "alpha": 0,
                          "hooks_enabled": True, "layer": 11, "mask_mode": "tool"}]}

    def wait_for(self, path, timeout=4):
        deadline = time.monotonic() + timeout
        while not path.exists() and time.monotonic() < deadline: time.sleep(.01)
        self.assertTrue(path.exists(), str(path))

    def test_two_jobs_keep_one_model_then_explicit_service_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.setup(directory); service = Path(cfg["out_dir"])
            errors = []
            def target():
                try: run(cfg, backend_factory=Backend)
                except BaseException as error: errors.append(error)
            thread = threading.Thread(target=target); thread.start()
            try:
                self.wait_for(service / "SERVICE-READY.json")
                for identifier in ("cohort1", "cohort2"):
                    job = self.job(cfg, identifier)
                    write_json(service / "jobs" / f"{identifier}.json", job)
                    out = Path(job["out_dir"])
                    self.wait_for(out / "READY.json")
                    write_json(out / "requests" / "r-test.json", {"schema_version": 1, "request_id": "r-test",
                               "arm_id": "zero", "prompt": "ab", "char_spans": {}, "seed": 123,
                               "max_new_tokens": 64, "purpose": "engineering_pilot"})
                    self.wait_for(out / "responses" / "r-test.json")
                    (out / "STOP").touch()
                    self.wait_for(out / "DONE.json")
                    self.assertFalse(Backend.last.closed)
                    self.assertTrue((out / "EXIT.json").exists())
                self.assertEqual(Backend.loads, 1)
            finally:
                (service / "STOP").touch()
                thread.join(timeout=4)
            self.assertFalse(thread.is_alive())
            self.assertFalse(errors)
            self.assertTrue(Backend.last.closed)
            self.assertTrue((service / "SERVICE-EXIT.json").is_file())

    def test_expired_lease_closes_idle_model(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.setup(directory)
            write_json(cfg["lease_path"], lease(time.time(), .15))
            run(cfg, backend_factory=Backend)
            self.assertEqual(Backend.loads, 1)
            self.assertTrue(Backend.last.closed)

    def test_job_cannot_switch_model_assets_layer_or_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = self.setup(directory); backend = Backend(cfg)
            job = self.job(cfg, "cohort")
            validate_job(job, "cohort", cfg, backend, Path(cfg["job_results_root"]))
            for key, value in (("probe_file", "different"), ("directions_file_sha256", "different"), ("execution_approved", False)):
                with self.subTest(field=key), self.assertRaises(ValueError):
                    validate_job({**job, key: value}, "cohort", cfg, backend, Path(cfg["job_results_root"]))
            for field, value in (("layer", 12), ("mask_mode", "all")):
                changed = {**job, "arms": [{**job["arms"][0], field: value}]}
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_job(changed, "cohort", cfg, backend, Path(cfg["job_results_root"]))


if __name__ == "__main__": unittest.main()
