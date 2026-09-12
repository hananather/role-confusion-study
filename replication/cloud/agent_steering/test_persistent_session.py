"""I test retained-pod handover, renewable leases and financial shutdown without cloud calls."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock
from replication.cloud.agent_steering import persistent_session as s


class Provider:
    def __init__(self):
        self.balance, self.autopay = 15.3444190459, False
        self.account_fails, self.deleted, self.delete_failures = False, False, 0
        self.calls = []
        self.others = []
    def account(self, _):
        if self.account_fails: raise RuntimeError("mock connection loss")
        return {"balance_usd": self.balance, "autopay_enabled": self.autopay}
    def request(self, method, path, body=None):
        self.calls.append((method, path))
        if method == "DELETE":
            if self.delete_failures: self.delete_failures -= 1
            else: self.deleted = True
            return 204, None
        return (404, None) if self.deleted else (200, {"id": "ownedpod", "costPerHr": 3.49})
    def pods(self):
        return self.others + ([] if self.deleted else [{"id": "ownedpod"}])


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        connection = self.root / "connection.json"
        s.write_json(connection, {"host": "203.0.113.1", "port": 2222, "key": "/tmp/mock-key"})
        self.config = {"version": 1, "retain_between_batches": True, "session_id": "test-session",
                       "pod_id": "ownedpod", "pod_created_at": "2026-09-12T02:25:50Z",
                       "original_cap_usd": 20, "credit_reserve_usd": 2,
                       "original_balance_usd": 17.5614986737, "pod_start_balance_usd": 15.3444190459,
                       "prior_spend_usd": 2.22, "rate_usd_h": 3.49, "container_disk_gb": 30,
                       "container_disk_usd_gb_month": .1, "existing_volume_usd_month": 280,
                       "storage_month_hours": 730, "connection_file": str(connection),
                       "account_check": {"query": "query { myself { clientBalance isAutoPayEnabled } }"},
                       "remote_lease_path": "/workspace/session-control/ownedpod/lease.json",
                       "sync_sources": [{"remote": "/workspace/results/agent-steering/old-batch",
                                         "local": str(self.root / "old-batch/results")} ]}
        self.wall = s.timestamp(self.config["pod_created_at"]); self.mono = 0
        self.provider = Provider()
        self.command = Mock(return_value=subprocess.CompletedProcess([], 0))
        self.session = self.make_session()
    def make_session(self):
        return s.Session(self.config, self.root / "session", self.provider,
                         clock=lambda: self.wall, monotonic=lambda: self.mono,
                         command=self.command, sleep=lambda _: None)
    def advance(self, seconds):
        self.wall += seconds; self.mono += seconds
    def deletes(self):
        return [path for method, path in self.provider.calls if method == "DELETE"]

    def test_ready_requires_live_account_owned_pod_and_remote_lease(self):
        self.session.tick()
        ready = s.read_json(self.session.directory / "READY.json")
        self.assertEqual(ready["pod_id"], "ownedpod"); self.assertTrue(ready["lease_published"])
        lease = s.read_json(self.session.directory / "lease.json")
        self.assertEqual(lease["lease_expires_unix"], self.wall + 180)
        self.assertEqual(lease["expected_pod_id"], "ownedpod")
        self.assertFalse(self.deletes())

    def test_old_batch_done_exit_stop_and_deadline_never_terminate_session(self):
        out = self.root / "old-batch/results"; out.mkdir(parents=True)
        for name in ("DONE.json", "EXIT.json", "FAILED.json", "STOP"):
            (out / name).write_text("{}")
        self.advance(8401)
        self.session.tick()
        self.assertFalse(self.deletes()); self.assertFalse(self.session.ended)
        self.assertGreater(s.read_json(self.session.directory / "lease.json")["budget_remaining_usd"], 4)

    def test_only_session_stop_deletes_exact_owned_pod_and_preserves_others(self):
        self.provider.others = [{"id": "someone-else"}]
        (self.session.directory / "STOP").touch()
        self.session.tick()
        self.assertEqual(self.deletes(), ["/pods/ownedpod"])
        receipt = s.read_json(self.session.directory / "termination.json")
        self.assertTrue(receipt["verified_404"]); self.assertFalse(receipt["zero_pods_verified"])
        self.assertEqual(receipt["remaining_pods"], self.provider.others)

    def test_live_credit_reserve_floor_terminates_even_before_old_deadline(self):
        self.provider.balance = 2
        self.session.tick()
        self.assertEqual(self.session.reason, "financial_budget_floor")
        self.assertTrue(self.session.delete_verified)

    def test_provider_billing_lag_cannot_extend_financed_lifetime(self):
        self.advance(12500)
        self.session.tick()
        self.assertEqual(self.session.reason, "financial_budget_floor")

    def test_account_credit_does_not_reset_original_spend_or_buy_unapproved_time(self):
        self.session.tick(); first = self.session.high_spend
        self.provider.balance += 10; self.advance(60); self.session.tick()
        self.assertGreaterEqual(self.session.high_spend, first)
        self.assertAlmostEqual(self.session.credits_observed, 10)
        self.advance(12500); self.session.tick()
        self.assertEqual(self.session.reason, "financial_budget_floor")

    def test_autopay_must_be_off_for_handover_and_lease_renewal(self):
        self.provider.autopay = True
        self.session.tick()
        self.assertFalse((self.session.directory / "READY.json").exists())
        self.assertFalse((self.session.directory / "lease.json").exists())
        self.assertFalse(self.deletes())

    def test_failed_lease_transport_cannot_claim_handover_readiness(self):
        self.command.return_value = subprocess.CompletedProcess([], 1)
        self.session.tick()
        self.assertFalse((self.session.directory / "READY.json").exists())
        self.assertFalse(self.deletes())

    def test_provider_loss_stops_renewal_but_keeps_budget_enforcement(self):
        self.session.tick(); before = (self.session.directory / "lease.json").read_bytes()
        self.provider.account_fails = True; self.advance(300); self.session.tick()
        self.assertEqual((self.session.directory / "lease.json").read_bytes(), before)
        self.assertFalse(self.deletes())
        self.advance(12500); self.session.tick()
        self.assertEqual(self.session.reason, "financial_budget_floor")

    def test_restart_retains_ledger_and_stale_account_budget(self):
        self.session.tick(); self.advance(500)
        replacement = self.make_session(); self.provider.account_fails = True
        replacement.tick()
        self.assertGreater(replacement.high_spend, 2.22)
        self.advance(12500); replacement.tick()
        self.assertEqual(replacement.reason, "financial_budget_floor")

    def test_delete_retries_until_get404(self):
        self.provider.delete_failures = 1
        (self.session.directory / "STOP").touch()
        self.session.tick(); self.assertFalse(self.session.ended)
        self.session.tick(); self.assertTrue(self.session.ended)
        self.assertEqual(len(self.deletes()), 2)

    def test_process_signal_is_not_pod_shutdown_authorization(self):
        self.session.signal_seen = True
        self.session.tick()
        self.assertFalse(self.deletes()); self.assertTrue(self.session.last_snapshot["signal_seen"])

    def test_backward_clock_cannot_extend_financial_allowance(self):
        self.session.tick(); self.wall -= 3600; self.mono += 12500
        self.session.tick()
        self.assertEqual(self.session.reason, "financial_budget_floor")

    def test_budget_and_lease_path_cannot_silently_expand(self):
        for key, value in [("original_cap_usd", 21), ("credit_reserve_usd", 1),
                           ("credit_reserve_usd", float("nan")), ("prior_spend_usd", 0),
                           ("existing_volume_usd_month", 1), ("retain_between_batches", False),
                           ("remote_lease_path", "/workspace/session-control/other/lease.json")]:
            with self.subTest(key=key):
                cfg = dict(self.config); cfg[key] = value
                with self.assertRaises(ValueError): s.validate_config(cfg)


if __name__ == "__main__":
    unittest.main()
