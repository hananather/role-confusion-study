"""I check my queue adapter without loading a model or contacting a provider."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from replication.cloud.parallel_h100 import queue_service as q


class Child:
    pid = 45678
    def __init__(self):
        self.returncode = None
        self.terminated = False
    def poll(self): return self.returncode
    def terminate(self): self.terminated = True; self.returncode = -15
    def wait(self, timeout=None): return self.returncode
    def kill(self): self.returncode = -9


class QueueServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name).resolve()
        self.root = self.base / "queue-test"
        self.root.mkdir()
        self.patch_root = patch.object(q.prime, "BASE_ROOT", self.base)
        self.patch_root.start(); self.addCleanup(self.patch_root.stop)
        self.patch_env = patch.dict(os.environ, {"RUNPOD_POD_ID": "newpod", "RUNPOD_API_KEY": "fake-private"})
        self.patch_env.start(); self.addCleanup(self.patch_env.stop)
        fixed = {"model_id": q.prime.MODEL_ID, "model_revision": q.prime.MODEL_REVISION,
                 "cache_dir": "/workspace/hf", "hf_home": "/workspace/hf/home",
                 "kernel_cache_dir": "/workspace/hf/home/hub", "attn_implementation": q.prime.ATTENTION}
        self.config = {"schema_version": 1, "purpose": "queue_service", "execution_approved": True,
                       "run_id": "queue-test", "expected_pod_id": "newpod", "protected_pod_id": q.FORBIDDEN_POD,
                       "isolated_root": str(self.root), "lease_path": str(self.root / "control/lease.json"),
                       "source_root": str(self.root / "source"), "queue_job_id": "queue-test-queue",
                       "startup_timeout_seconds": 600, "poll_seconds": 1, **fixed}
        self.config["worker_config"] = {"execution_approved": True, "persistent_budget_lease_required": True,
                     "out_dir": str(self.root / "model-service"), "job_results_root": "/workspace/results/agent-steering",
                     "expected_pod_id": "newpod", "lease_path": self.config["lease_path"], **fixed,
                     "directions_file": str(self.root / "directions.npz"), "directions_file_sha256": "0" * 64,
                     "probe_file": str(self.root / "probes.npz"), "probe_file_sha256": "1" * 64,
                     "max_context_tokens": 65536, "arms": [{"arm_id": "none", "direction": "none", "alpha": 0,
                                                        "hooks_enabled": False, "layer": 11, "mask_mode": "tool"}]}
        self.path = self.root / "worker-config.json"
        self.path.write_text(json.dumps(self.config))
        self.child = Child()
        self.spawn = Mock(return_value=self.child)
        self.clock = 0
        self.lease = {"lease_expires_unix": 180, "shutdown_at_unix": 600}

    def guardian(self, reader=None, sleep=None):
        return q.Guardian(self.config, self.path, spawn=self.spawn,
                          lease_reader=reader or (lambda _: self.lease), clock=lambda: self.clock,
                          wall=lambda: 1000 + self.clock, sleep=sleep or self.stop_after_one)

    def stop_after_one(self, seconds):
        self.clock += seconds
        (self.root / "service/STOP").touch()

    def test_loaded_model_ready_explicitly_does_not_claim_engineering_pass(self):
        out = self.root / "model-service"; out.mkdir()
        (out / "SERVICE-READY.json").write_text(json.dumps({"status": "ready", "lease_required": True,
                "config_sha256": q.prime.canonical_sha(self.config["worker_config"])}))
        guardian = self.guardian()
        self.assertEqual(guardian.run(), 0)
        ready = json.loads((self.root / "service/READY.json").read_text())
        self.assertEqual(ready["status"], "queue_model_loaded")
        self.assertFalse(ready["engineering_gate_passed"])
        self.assertTrue(ready["experiment_inbox"])
        self.assertTrue(self.child.terminated)
        env = self.spawn.call_args.kwargs["env"]
        self.assertNotIn("RUNPOD_API_KEY", env)
        self.assertTrue(env["TRITON_CACHE_DIR"].startswith(str(self.root / "cache")))

    def test_expired_lease_stops_child_even_during_blocking_model_load(self):
        reader = Mock(side_effect=[self.lease, ValueError("expired")])
        guardian = self.guardian(reader=reader)
        self.assertEqual(guardian.run(), 0)
        self.assertTrue(self.child.terminated)
        self.assertEqual(guardian.stop_reason, "identity_or_lease_unavailable")
        self.assertFalse((self.root / "service/READY.json").exists())

    def test_startup_timeout_is_finite(self):
        def advance(seconds): self.clock += 601
        guardian = self.guardian(sleep=advance)
        self.assertEqual(guardian.run(), 2)
        self.assertEqual(guardian.stop_reason, "startup_timeout")
        self.assertTrue(self.child.terminated)

    def test_foreign_queue_pod_model_and_results_root_rejected(self):
        for key, value in (("expected_pod_id", q.FORBIDDEN_POD), ("queue_job_id", "old-job"),
                           ("source_root", "/workspace/old-source"), ("purpose", "prime_only")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                q.validate_config({**self.config, key: value}, self.path)
        for key, value in (("job_results_root", "/workspace/old-results"), ("out_dir", "/workspace/model-service"),
                           ("model_revision", "other"), ("persistent_budget_lease_required", False)):
            cfg = {**self.config, "worker_config": {**self.config["worker_config"], key: value}}
            with self.subTest(key=key), self.assertRaises(ValueError):
                q.validate_config(cfg, self.path)

    def test_readiness_with_wrong_frozen_worker_hash_is_not_accepted(self):
        out = self.root / "model-service"; out.mkdir()
        (out / "SERVICE-READY.json").write_text(json.dumps({"status": "ready", "lease_required": True,
                                                           "config_sha256": "wrong"}))
        self.assertFalse(self.guardian().loaded())

    def test_changed_frozen_config_stops_child(self):
        def alter(seconds): self.path.write_text("{}")
        guardian = self.guardian(sleep=alter)
        self.assertEqual(guardian.run(), 2)
        self.assertEqual(guardian.stop_reason, "frozen_config_changed")
        self.assertTrue(self.child.terminated)


if __name__ == "__main__":
    unittest.main()
