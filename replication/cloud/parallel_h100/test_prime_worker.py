"""I check setup ownership and financial shutdown without a GPU or network."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from . import prime_worker as w


class Process:
    pid = 12345
    def __init__(self, stubborn=False):
        self.returncode = None; self.terminated = False; self.killed = False
        self.stubborn = stubborn
    def poll(self): return self.returncode
    def terminate(self):
        self.terminated = True
        if not self.stubborn: self.returncode = 0
    def kill(self): self.killed = True; self.returncode = -9
    def wait(self, timeout):
        if self.returncode is None: raise subprocess.TimeoutExpired("owned-child", timeout)
        return self.returncode


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name).resolve()
        self.root = base / "newrun"; (self.root / "control").mkdir(parents=True)
        self.config = {"schema_version": 1, "purpose": "prime_only", "execution_approved": True,
                       "run_id": "newrun", "expected_pod_id": "newpod", "protected_pod_id": "oldpod",
                       "isolated_root": str(self.root), "lease_path": str(self.root / "control/lease.json"),
                       "poll_seconds": .01, "startup_timeout_seconds": 1}
        self.config_path = self.root / "control/prime-config.json"
        w.write_json(self.config_path, self.config)
        self.patch_base = patch.object(w, "BASE_ROOT", base); self.patch_base.start()
        self.addCleanup(self.patch_base.stop)
        self.patch_env = patch.dict(os.environ, {"RUNPOD_POD_ID": "newpod"}); self.patch_env.start()
        self.addCleanup(self.patch_env.stop)
        self.now = 1000
        self.lease = {"schema_version": 1, "approved": True, "lease_id": "newrun",
                      "run_id": "newrun", "expected_pod_id": "newpod", "issued_unix": 1000,
                      "lease_expires_unix": 1100, "shutdown_at_unix": 1300, "budget_remaining_usd": 3}
        w.write_json(self.config["lease_path"], self.lease)

    def test_configuration_rejects_protected_pod_path_escape_and_experiment_fields(self):
        w.validate_config(self.config, self.config_path)
        variants = [{"expected_pod_id": "oldpod"}, {"purpose": "experiment"}, {"execution_approved": False},
                    {"run_id": "../old"}, {"lease_path": str(self.root.parent / "lease.json")},
                    {"isolated_root": str(self.root / "../newrun")}, {"model_revision": "unfrozen"},
                    {"inbox": "/tmp/jobs"}, {"startup_timeout_seconds": float("nan")}]
        for changes in variants:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                w.validate_config({**self.config, **changes}, self.config_path)
        with self.assertRaises(ValueError): w.validate_config(self.config, self.root.parent / "config.json")

    def test_wrong_or_missing_actual_pod_blocks_loading(self):
        for actual in ("oldpod", ""):
            with patch.dict(os.environ, {"RUNPOD_POD_ID": actual}), self.assertRaises(ValueError):
                w.validate_config(self.config, self.config_path)

    def bind_marker(self):
        marker_dir = self.root.parent / "container-tmp"; marker_dir.mkdir()
        marker_patch = patch.object(w, "MARKER_DIR", marker_dir); marker_patch.start()
        self.addCleanup(marker_patch.stop)
        path = marker_dir / "mats-prime-newrun-identity.json"
        self.config.update(identity_marker_path=str(path), identity_nonce="a" * 64,
                           identity_source="provider_api_ssh_binding")
        marker = {"schema_version": 1, "identity_source": "provider_api_ssh_binding",
                  "expected_pod_id": "newpod", "run_id": "newrun", "identity_nonce": "a" * 64}
        path.write_text(json.dumps(marker)); path.chmod(0o600)
        w.write_json(self.config_path, self.config)
        return path, marker

    def test_provider_bound_marker_works_without_inventing_environment(self):
        self.bind_marker()
        with patch.dict(os.environ, {}, clear=True):
            cfg = w.validate_config(self.config, self.config_path)
            report = w.identity_report(cfg)
            env = w.child_environment(cfg)
        self.assertEqual(report["identity_source"], "provider_api_ssh_binding")
        self.assertFalse(report["provider_environment_present"])
        self.assertNotIn("RUNPOD_POD_ID", env)

    def test_marker_cannot_override_conflicting_actual_environment(self):
        self.bind_marker()
        with patch.dict(os.environ, {"RUNPOD_POD_ID": "oldpod"}), self.assertRaises(ValueError):
            w.validate_config(self.config, self.config_path)

    def test_marker_nonce_pod_run_source_or_shared_path_mismatch_blocks(self):
        path, marker = self.bind_marker()
        for change in ({"identity_nonce": "b" * 64}, {"expected_pod_id": "oldpod"},
                       {"run_id": "another"}, {"identity_source": "unverified"}, {"schema_version": True}):
            path.write_text(json.dumps({**marker, **change}))
            with self.subTest(change=change), self.assertRaises(ValueError):
                w.validate_config(self.config, self.config_path)
        path.write_text(json.dumps(marker))
        for change in ({"identity_marker_path": str(self.root / "marker.json")},
                       {"identity_nonce": "too-short"}, {"identity_source": "unverified"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                w.validate_config({**self.config, **change}, self.config_path)

    def test_private_regular_marker_is_required_and_missing_marker_blocks(self):
        path, marker = self.bind_marker()
        path.chmod(0o644)
        with self.assertRaises(ValueError): w.validate_config(self.config, self.config_path)
        path.unlink()
        with self.assertRaises(OSError): w.validate_config(self.config, self.config_path)
        target = self.root / "shared-marker.json"; target.write_text(json.dumps(marker)); target.chmod(0o600)
        path.symlink_to(target)
        with self.assertRaises(OSError): w.validate_config(self.config, self.config_path)

    def test_changed_identity_marker_stops_only_owned_child(self):
        path, marker = self.bind_marker()
        guardian, child = self.guardian(ready=True)
        def sleep(_): path.write_text(json.dumps({**marker, "identity_nonce": "b" * 64}))
        guardian.sleep = sleep
        self.assertEqual(guardian.run(), 2)
        self.assertTrue(child.terminated)
        self.assertEqual(guardian.stop_reason, "container_identity_unavailable_or_changed")

    def test_lease_rejects_foreign_expired_invalid_or_extended_funding(self):
        w.read_lease(self.config, now=self.now)
        variants = [{"expected_pod_id": "oldpod"}, {"run_id": "different"}, {"approved": False},
                    {"issued_unix": 1002}, {"lease_expires_unix": 1181}, {"lease_expires_unix": 1000},
                    {"shutdown_at_unix": 1000}, {"budget_remaining_usd": 0},
                    {"budget_remaining_usd": float("inf")}, {"issued_unix": True}, {"lease_id": None}]
        for changes in variants:
            Path(self.config["lease_path"]).write_text(json.dumps({**self.lease, **changes}))
            with self.subTest(changes=changes), self.assertRaises(ValueError): w.read_lease(self.config, now=self.now)

    def test_symlink_lease_and_writable_cache_escape_rejected(self):
        path = Path(self.config["lease_path"]); path.unlink()
        target = self.root / "other.json"; w.write_json(target, self.lease); path.symlink_to(target)
        with self.assertRaises(ValueError): w.read_lease(self.config, now=self.now)
        (self.root / "cache").symlink_to(self.root / "control")
        with self.assertRaises(ValueError): w.child_environment(w.validate_config(self.config))

    def test_child_environment_omits_credentials_and_disables_shared_writes(self):
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "dummy", "HF_TOKEN": "dummy", "OPENAI_API_KEY": "dummy",
                                     "PYTHONPATH": "/external", "PYTHONHOME": "/external"}):
            env = w.child_environment(w.validate_config(self.config))
        for key in ("RUNPOD_API_KEY", "HF_TOKEN", "OPENAI_API_KEY", "PYTHONPATH", "PYTHONHOME"):
            self.assertNotIn(key, env)
        for key in ("TRITON_CACHE_DIR", "HF_MODULES_CACHE", "CUDA_CACHE_PATH", "TORCH_HOME", "TMPDIR"):
            self.assertTrue(Path(env[key]).is_relative_to(self.root / "cache"))
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")

    def guardian(self, *, stubborn=False, ready=False):
        child = Process(stubborn); self.spawned = []
        def spawn(argv, **kwargs):
            self.spawned.append((argv, kwargs))
            if ready: w.write_json(self.root / "service/READY.json", {"status": "primed_idle"})
            return child
        ticks = [0]
        def sleep(seconds): ticks[0] += seconds
        guardian = w.Guardian(self.config, self.config_path, spawn=spawn,
                              lease_reader=lambda _: self.lease, clock=lambda: ticks[0], sleep=sleep)
        return guardian, child

    def test_expired_lease_stops_own_child_even_while_loading(self):
        guardian, child = self.guardian()
        reads = [0]
        def lease_reader(_):
            reads[0] += 1
            if reads[0] >= 3: raise ValueError("expired")
            return self.lease
        guardian.lease_reader = lease_reader
        self.assertEqual(guardian.run(), 0)
        self.assertTrue(child.terminated)
        self.assertFalse((self.root / "service/READY.json").exists())
        receipt = json.loads((self.root / "service/EXIT.json").read_text())
        self.assertEqual(receipt["stop_reason"], "lease_unavailable_or_expired")
        self.assertTrue(receipt["worker_exit_verified"])
        argv, kwargs = self.spawned[0]
        self.assertIn("--model-child", argv); self.assertIn("-B", argv); self.assertIn("-s", argv)
        self.assertTrue(kwargs["start_new_session"])

    def test_idle_model_retained_until_explicit_stop_without_inbox(self):
        guardian, child = self.guardian(ready=True)
        def sleep(_): (self.root / "service/STOP").touch()
        guardian.sleep = sleep
        self.assertEqual(guardian.run(), 0)
        self.assertTrue(child.terminated)
        self.assertEqual(len(self.spawned), 1)
        self.assertFalse((self.root / "service/jobs").exists())

    def test_hung_startup_gets_bounded_termination_and_kill(self):
        guardian, child = self.guardian(stubborn=True)
        self.assertEqual(guardian.run(), 2)
        self.assertEqual(guardian.stop_reason, "startup_timeout")
        self.assertTrue(child.terminated); self.assertTrue(child.killed)

    def test_missing_initial_lease_never_spawns(self):
        guardian, _ = self.guardian()
        def absent(_): raise FileNotFoundError("missing")
        guardian.lease_reader = absent
        with self.assertRaises(FileNotFoundError): guardian.run()
        self.assertEqual(len(self.spawned), 0)

    def test_existing_service_claim_blocks_second_writer(self):
        guardian, _ = self.guardian()
        (self.root / "service").mkdir()
        (self.root / "service/guardian.lock").write_text("claimed")
        with self.assertRaises(FileExistsError): guardian.run()
        self.assertEqual(len(self.spawned), 0)

    def test_metadata_rejects_wrong_device_unpinned_model_and_dequantization(self):
        meta = {"model_id": w.MODEL_ID, "model_revision": w.MODEL_REVISION, "gpu": "NVIDIA H100 80GB HBM3",
                "capability": "9.0", "attention": w.ATTENTION, "n_layers": 24, "hidden": 2880,
                "quantization": {"quant_method": "mxfp4"}, "expert_module": "test.Mxfp4GptOssExperts"}
        w.validate_metadata(meta)
        for changes in ({"gpu": "A100"}, {"model_revision": "unfrozen"}, {"attention": "eager"},
                        {"quantization": {"quant_method": "mxfp4", "dequantize": True}}, {"hidden": 1}):
            with self.subTest(changes=changes), self.assertRaises(RuntimeError): w.validate_metadata({**meta, **changes})


if __name__ == "__main__": unittest.main()
