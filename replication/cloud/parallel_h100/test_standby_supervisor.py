"""I test my separate standby supervisor with fake time, transport and provider."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from replication.cloud.parallel_h100 import standby_supervisor as s


class FakeProvider:
    def __init__(self):
        self.balance, self.autopay = 25.0, False
        self.account_fails, self.pod_fails = False, False
        self.deleted, self.delete_failures = False, 0
        self.returned_id = "newownedpod"
        self.calls = []

    def account(self, _):
        if self.account_fails:
            raise RuntimeError("fake account failure")
        return {"balance_usd": self.balance, "autopay_enabled": self.autopay}

    def request(self, method, path, body=None):
        self.calls.append((method, path))
        if path != "/pods/newownedpod":
            raise AssertionError("I may contact only my own pod endpoint")
        if method == "DELETE":
            if self.delete_failures:
                self.delete_failures -= 1
                return 503, None
            self.deleted = True
            return 204, None
        if self.deleted:
            return 404, None
        if self.pod_fails:
            raise RuntimeError("fake pod failure")
        return 200, {"id": self.returned_id, "costPerHr": 3.49}


class StandbyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = {"version": 1, "scope": "parallel_h100_standby", "run_id": "standby-test",
                       "pod_id": "newownedpod", "pod_created_at": "2026-09-12T04:00:00Z",
                       "remote_root": "/workspace/parallel-lanes/standby-test",
                       "connection_file": str(self.root / "connection.json"),
                       "local_mirror": str(self.root / "mirror"), "allowance_usd": 3,
                       "incremental_rate_usd_h": s.INCREMENTAL_RATE, "credit_reserve_usd": 2,
                       "protected_peer_pod_id": s.FORBIDDEN_POD,
                       "protected_peer_deadline_unix": s.PEER_DEADLINE,
                       "protected_peer_rate_usd_h": s.PEER_RATE, "aggregate_cap_usd": 20,
                       "protected_prior_projected_total_usd": 16.3444,
                       "prior_sidecar_spend_usd": .16,
                       "identity_marker_path": "/tmp/mats-prime-standby-test-identity.json",
                       "identity_nonce": "a" * 64, "identity_source": "provider_api_ssh_binding",
                       "account_check": {"query": "query { myself { clientBalance isAutoPayEnabled } }"}}
        self.wall = s.timestamp(self.config["pod_created_at"])
        self.config["startup_deadline_unix"] = self.wall + 600
        self.mono = 0
        self.provider = FakeProvider()
        self.command = Mock(return_value=subprocess.CompletedProcess([], 0))
        self.supervisor = self.make()

    def make(self):
        return s.StandbySupervisor(self.config, self.root / "supervisor", self.provider,
                                   clock=lambda: self.wall, monotonic=lambda: self.mono,
                                   command=self.command, sleep=lambda _: None)

    def advance(self, seconds):
        self.wall += seconds
        self.mono += seconds

    def connect(self, pod_id="newownedpod"):
        s.write_json(self.config["connection_file"], {"pod_id": pod_id, "host": "203.0.113.1",
                     "port": 2222, "key": "/tmp/fake-key"})

    def ready(self):
        s.write_json(Path(self.config["local_mirror"]) / "service/READY.json",
                     {"expected_pod_id": "newownedpod", "status": "primed_idle", "smoke_passed": True,
                      "experiment_inbox": False, "ready_unix": self.wall})

    def deletes(self):
        return [path for method, path in self.provider.calls if method == "DELETE"]

    def test_startup_without_connection_has_immediate_provider_and_financial_coverage(self):
        self.supervisor.tick()
        self.assertEqual(self.provider.calls, [("GET", "/pods/newownedpod")])
        self.assertFalse(self.command.called)
        self.assertFalse(self.deletes())
        self.assertTrue((self.supervisor.directory / "ledger.json").exists())
        self.assertFalse((self.supervisor.directory / "lease.json").exists())

    def test_first_lease_requires_owned_connection_and_live_account(self):
        self.connect()
        self.supervisor.tick()
        lease = s.read_json(self.supervisor.directory / "lease.json")
        self.assertEqual(lease["expected_pod_id"], "newownedpod")
        self.assertEqual(lease["run_id"], lease["lease_id"])
        self.assertEqual(lease["lease_expires_unix"], self.wall + 180)
        self.assertTrue((self.supervisor.directory / "SUPERVISOR_READY.json").exists())
        self.assertTrue(any("--include=/service/***" in call.args[0] for call in self.command.call_args_list))
        self.assertFalse(self.deletes())

    def test_startup_limit_reclaims_only_owned_pod_without_ssh(self):
        self.advance(601)
        self.supervisor.tick()
        self.assertEqual(self.supervisor.reason, "startup_readiness_deadline")
        self.assertEqual(self.deletes(), ["/pods/newownedpod"])
        self.assertTrue(self.supervisor.delete_verified)

    def test_ready_idle_is_retained_beyond_startup_deadline(self):
        self.connect(); self.ready(); self.supervisor.tick()
        for _ in range(21):
            self.advance(30); self.supervisor.tick()
        self.assertFalse(self.deletes())
        self.assertFalse(self.supervisor.ended)

    def test_full_lifetime_cap_includes_time_before_supervisor_and_ignores_topup(self):
        self.connect(); self.ready(); self.supervisor.tick()
        self.advance(90); self.provider.balance += 100
        self.supervisor.tick()
        before = self.supervisor.high_consumed
        self.assertGreater(before, 0)
        self.advance(3 / s.INCREMENTAL_RATE * 3600)
        self.supervisor.tick()
        self.assertEqual(self.supervisor.reason, "initial_lifetime_allowance_exhausted")
        self.assertGreaterEqual(self.supervisor.high_consumed, before)
        self.assertEqual(self.deletes(), ["/pods/newownedpod"])

    def test_peer_future_burn_and_two_dollar_credit_are_reserved(self):
        peer = (s.PEER_DEADLINE - self.wall) * s.PEER_RATE / 3600
        finance = self.supervisor.financial_state({"balance_usd": peer + 2.5, "autopay_enabled": False})
        self.assertAlmostEqual(finance["peer_obligation_usd"], peer)
        self.assertAlmostEqual(finance["budget_remaining_usd"], .5 - s.SHUTDOWN_RESERVE)
        self.assertAlmostEqual(finance["shutdown_at_unix"] - self.wall, (.5 - s.SHUTDOWN_RESERVE) / s.INCREMENTAL_RATE * 3600, delta=1e-6)
        self.assertFalse(finance["shared_volume_charged_here"])

    def test_peer_reserve_floor_terminates_even_with_own_allowance_remaining(self):
        self.provider.balance = 2 + (s.PEER_DEADLINE - self.wall) * s.PEER_RATE / 3600
        self.supervisor.tick()
        self.assertEqual(self.supervisor.reason, "protected_peer_and_credit_reserve_floor")
        self.assertGreater(self.supervisor.high_consumed + 3, 0)
        self.assertEqual(self.deletes(), ["/pods/newownedpod"])

    def test_stale_balance_projects_both_lanes_burn_without_double_charging_peer(self):
        peer = (s.PEER_DEADLINE - self.wall) * s.PEER_RATE / 3600
        self.supervisor.financial_state({"balance_usd": peer + 3, "autopay_enabled": False})
        self.advance(60)
        finance = self.supervisor.financial_state()
        self.assertAlmostEqual(finance["credit_available_for_own_pod_usd"], 1 - 60 * s.INCREMENTAL_RATE / 3600)

    def test_provider_staleness_over_120_seconds_deletes_owned_only(self):
        self.supervisor.tick()
        self.provider.account_fails = True
        self.advance(121)
        self.supervisor.tick()
        self.assertEqual(self.supervisor.reason, "provider_verification_stale")
        self.assertEqual(self.deletes(), ["/pods/newownedpod"])

    def test_wrong_live_pod_never_renews_lease_or_targets_peer(self):
        self.connect(); self.provider.returned_id = s.FORBIDDEN_POD
        self.supervisor.tick()
        self.assertFalse((self.supervisor.directory / "lease.json").exists())
        self.advance(121); self.supervisor.tick()
        self.assertEqual(self.deletes(), ["/pods/newownedpod"])
        self.assertTrue(all(path == "/pods/newownedpod" for _, path in self.provider.calls))

    def test_wrong_connection_id_never_executes_remote_command(self):
        self.connect(s.FORBIDDEN_POD)
        self.supervisor.tick()
        self.assertFalse(self.command.called)
        self.assertEqual(self.supervisor.reason, "owned_connection_invalid")
        self.assertEqual(self.deletes(), ["/pods/newownedpod"])

    def test_old_pod_remote_root_and_budget_extension_are_rejected(self):
        for key, value in (("pod_id", s.FORBIDDEN_POD), ("remote_root", "/workspace"),
                           ("allowance_usd", 4), ("credit_reserve_usd", 1),
                           ("protected_peer_deadline_unix", s.PEER_DEADLINE - 60),
                           ("prior_sidecar_spend_usd", 0), ("identity_nonce", "short"),
                           ("identity_marker_path", "/workspace/identity.json")):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    s.validate_config({**self.config, key: value})

    def test_lease_is_bounded_by_financial_deadline(self):
        self.connect(); self.supervisor.refresh_connection()
        peer = (s.PEER_DEADLINE - self.wall) * s.PEER_RATE / 3600
        finance = self.supervisor.financial_state({"balance_usd": peer + 2.01, "autopay_enabled": False})
        lease = self.supervisor.publish_lease(finance)
        self.assertLess(lease["lease_expires_unix"], self.wall + 180)
        self.assertEqual(lease["lease_expires_unix"], finance["shutdown_at_unix"])

    def test_remote_lease_failure_reclaims_after_stale_window(self):
        self.connect(); self.command.return_value = subprocess.CompletedProcess([], 1)
        self.supervisor.tick()
        self.assertFalse((self.supervisor.directory / "SUPERVISOR_READY.json").exists())
        self.advance(121); self.supervisor.tick()
        self.assertEqual(self.supervisor.reason, "lease_publication_stale")

    def test_signal_and_persistent_stop_sync_before_only_owned_delete(self):
        self.connect(); self.supervisor.tick()
        self.supervisor.signal_seen = True
        self.supervisor.tick()
        self.assertTrue((self.supervisor.directory / "STOP").exists())
        self.assertEqual(self.deletes(), ["/pods/newownedpod"])
        receipt = s.read_json(self.supervisor.directory / "termination.json")
        self.assertTrue(receipt["verified_404"])
        self.assertNotIn("zero_pods_verified", receipt)

    def test_deletion_retains_supervision_and_escalates_after_bounded_fast_retries(self):
        self.provider.delete_failures = 100
        (self.supervisor.directory / "STOP").touch()
        for _ in range(s.MAX_DELETE_ATTEMPTS):
            self.supervisor.tick()
        self.assertFalse(self.supervisor.ended)
        self.assertFalse(self.supervisor.delete_verified)
        self.assertEqual(len(self.deletes()), s.MAX_DELETE_ATTEMPTS)
        self.assertTrue((self.supervisor.directory / "NEEDS_ATTENTION.json").exists())
        self.provider.delete_failures = 0
        self.supervisor.tick()
        self.assertTrue(self.supervisor.ended)
        self.assertTrue(self.supervisor.delete_verified)

    def test_shutdown_reserve_starts_cleanup_before_lifetime_cap(self):
        self.connect(); self.ready()
        self.advance((3 - s.SHUTDOWN_RESERVE - s.PRIOR_SIDECAR_SPEND) / s.INCREMENTAL_RATE * 3600 + 1)
        self.supervisor.tick()
        self.assertEqual(self.supervisor.reason, "initial_lifetime_allowance_exhausted")
        self.assertLess(self.supervisor.high_consumed, 3)

    def test_failed_allocation_spend_is_preserved_in_same_allowance(self):
        self.supervisor.tick()
        initial = s.read_json(self.supervisor.directory / "ledger.json")
        self.assertAlmostEqual(initial["consumed_usd"], .16)
        self.assertAlmostEqual(initial["prior_sidecar_spend_usd"], .16)
        self.assertAlmostEqual(initial["budget_remaining_usd"], 3 - .16 - .12)
        self.advance(90)
        self.provider.balance += 100
        self.supervisor.tick()
        ledger = s.read_json(self.supervisor.directory / "ledger.json")
        self.assertAlmostEqual(ledger["consumed_usd"], .16 + 90 * s.INCREMENTAL_RATE / 3600)
        restart = self.make()
        self.assertAlmostEqual(restart.financial_state()["consumed_usd"], ledger["consumed_usd"])

    def marker(self, **overrides):
        config = {**self.config, "identity_marker_path": str(self.root / "identity.json")}
        marker = {"schema_version": 1, "identity_source": "provider_api_ssh_binding",
                  "expected_pod_id": config["pod_id"], "run_id": config["run_id"],
                  "identity_nonce": config["identity_nonce"], **overrides}
        path = Path(config["identity_marker_path"])
        path.write_text(json.dumps(marker))
        path.chmod(0o600)
        return config

    def test_container_marker_allows_absent_env_and_matching_real_env(self):
        config = self.marker()
        with patch.dict(os.environ, {}, clear=True):
            s.verify_identity_marker(config)
        with patch.dict(os.environ, {"RUNPOD_POD_ID": config["pod_id"]}, clear=True):
            s.verify_identity_marker(config)

    def test_disagreeing_actual_env_marker_pod_nonce_and_run_fail_closed(self):
        config = self.marker()
        with patch.dict(os.environ, {"RUNPOD_POD_ID": s.FORBIDDEN_POD}, clear=True):
            with self.assertRaises(ValueError):
                s.verify_identity_marker(config)
        for overrides in ({"expected_pod_id": s.FORBIDDEN_POD}, {"identity_nonce": "b" * 64}, {"run_id": "another-run"}):
            with self.subTest(overrides=overrides), patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    s.verify_identity_marker(self.marker(**overrides))

    def test_missing_symlink_and_nonprivate_marker_fail_closed(self):
        config = self.marker()
        path = Path(config["identity_marker_path"])
        with patch.dict(os.environ, {}, clear=True):
            path.chmod(0o644)
            with self.assertRaises(ValueError):
                s.verify_identity_marker(config)
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                s.verify_identity_marker(config)
            actual = self.root / "actual.json"
            actual.write_text("{}"); actual.chmod(0o600)
            path.symlink_to(actual)
            with self.assertRaises(OSError):
                s.verify_identity_marker(config)

    def test_actual_remote_verifier_source_runs_without_runpod_env(self):
        config = self.marker()
        env = dict(os.environ); env.pop("RUNPOD_POD_ID", None)
        result = subprocess.run([sys.executable, "-c", s.remote_identity_check(config)], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertNotIn("RUNPOD_POD_ID", env)

    def test_explicit_queue_authorization_retains_aggregate_cap_and_owned_mirror(self):
        cfg = {**self.config, "scope": "parallel_h100_queue", "allowance_usd": 3.65,
               "queue_job_id": self.config["run_id"] + "-queue"}
        s.validate_config(cfg)
        self.assertLess(cfg["protected_prior_projected_total_usd"] + cfg["allowance_usd"], 20)
        with self.assertRaises(ValueError):
            s.validate_config({**cfg, "allowance_usd": 3.66})
        supervisor = s.StandbySupervisor(cfg, self.root / "queue-supervisor", self.provider,
                         clock=lambda: self.wall, monotonic=lambda: self.mono, command=self.command)
        self.connect(); supervisor.tick()
        finance = s.read_json(supervisor.directory / "ledger.json")
        self.assertAlmostEqual(finance["budget_remaining_usd"], 3.65 - .16 - .12)
        calls = [call.args[0] for call in self.command.call_args_list]
        self.assertTrue(any(any(":/workspace/results/agent-steering/standby-test-queue/" in arg for arg in cmd) for cmd in calls))
        s.write_json(Path(cfg["local_mirror"]) / "service/READY.json",
                     {"expected_pod_id": cfg["pod_id"], "run_id": cfg["run_id"], "ready_unix": self.wall,
                      "status": "queue_model_loaded", "experiment_inbox": True, "engineering_gate_passed": False})
        self.assertTrue(supervisor.model_ready())

    def test_restart_retains_consumption_and_rejects_changed_creation_time(self):
        self.supervisor.tick(); self.advance(90); self.supervisor.tick()
        consumed = self.supervisor.high_consumed
        self.provider.balance += 100
        restart = self.make()
        self.assertGreaterEqual(restart.financial_state()["consumed_usd"], consumed)
        self.config["pod_created_at"] = "2026-09-12T04:01:00Z"
        self.config["startup_deadline_unix"] += 60
        with self.assertRaises(ValueError):
            self.make()

    def test_backward_clock_cannot_extend_allowance(self):
        self.supervisor.tick()
        self.wall -= 3600; self.mono += 4000
        self.supervisor.tick()
        self.assertEqual(self.supervisor.reason, "initial_lifetime_allowance_exhausted")


if __name__ == "__main__":
    unittest.main()
