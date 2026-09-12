"""I verify forced lease expiry against fake owned children; no model, signals or GPU calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from replication.cloud.agent_steering import service_guardian as g


class Child:
    pid = 7331
    def __init__(self, clock, *, cooperate=False, exit_at=None):
        self.clock, self.cooperate, self.exit_at = clock, cooperate, exit_at
        self.returncode = None
        self.terminates = self.kills = 0
    def poll(self):
        if self.exit_at is not None and self.clock() >= self.exit_at and self.returncode is None:
            self.returncode = 0
        return self.returncode
    def terminate(self):
        self.terminates += 1
        if self.cooperate: self.returncode = -15
    def kill(self):
        self.kills += 1; self.returncode = -9
    def wait(self, timeout=None):
        return self.poll()


class GuardianTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.mono = 0
        self.config = {"execution_approved": True, "persistent_budget_lease_required": True,
                       "expected_pod_id": "ownedpod", "predecessor_pid": 530,
                       "lease_path": "/workspace/session-control/ownedpod/lease.json",
                       "out_dir": "/workspace/session-control/ownedpod/model-service"}
        self.config_path = self.root / "config.json"; self.config_path.write_text(json.dumps(self.config))
        self.child = Child(lambda: self.mono)
        self.spawn = Mock(side_effect=lambda *_args, **_kwargs: self.child)
        self.exists = Mock(return_value=False)
        self.lease = Mock(return_value={"approved": True})
    def advance(self, seconds): self.mono += seconds
    def guardian(self):
        return g.Guardian(self.config, self.config_path, self.root / "guard", lease_reader=self.lease,
                          spawn=self.spawn, exists=self.exists, monotonic=lambda: self.mono, sleep=self.advance)
    def expire_at(self, seconds):
        def read(_):
            if self.mono >= seconds: raise ValueError("expired")
            return {"approved": True}
        self.lease.side_effect = read

    def test_missing_predecessor_is_rejected_without_model_launch(self):
        self.config.pop("predecessor_pid")
        with self.assertRaisesRegex(ValueError, "preceding"): self.guardian()
        self.spawn.assert_not_called()

    def test_expired_lease_prevents_child_startup(self):
        self.lease.side_effect = ValueError("expired")
        with self.assertRaises(RuntimeError): self.guardian().run()
        self.spawn.assert_not_called()

    def test_named_predecessor_is_waited_for_and_never_signaled(self):
        self.exists.side_effect = lambda pid: self.mono < 3
        self.child.exit_at = 5
        guardian = self.guardian(); self.assertEqual(guardian.run(), 0)
        self.assertGreaterEqual(self.mono, 5)
        self.assertTrue(all(call.args == (530,) for call in self.exists.call_args_list))
        self.assertEqual(self.child.terminates, 0); self.assertEqual(self.child.kills, 0)

    def test_predecessor_wait_is_bounded_by_lease(self):
        self.exists.return_value = True; self.expire_at(2)
        with self.assertRaises(RuntimeError): self.guardian().run()
        self.spawn.assert_not_called()

    def test_lease_expiry_kills_only_owned_stuck_worker_after_bounded_grace(self):
        self.expire_at(3)
        guardian = self.guardian(); self.assertEqual(guardian.run(), 2)
        self.assertEqual((self.child.terminates, self.child.kills), (1, 1))
        self.assertLessEqual(self.mono, 8.2)
        receipt = json.loads((self.root / "guard/GUARDIAN-EXIT.json").read_text())
        self.assertEqual(receipt["worker_pid"], 7331); self.assertEqual(receipt["predecessor_pid"], 530)
        self.assertTrue(receipt["worker_exit_verified"]); self.assertEqual(receipt["allocation_action"], "none")
        self.assertTrue(receipt["sigkill_used"])

    def test_cooperative_worker_needs_no_sigkill(self):
        self.child.cooperate = True; self.expire_at(2)
        self.guardian().run()
        self.assertEqual((self.child.terminates, self.child.kills), (1, 0))

    def test_valid_renewal_keeps_the_same_child_across_old_lease_expiry(self):
        self.child.exit_at = 6
        self.lease.side_effect = lambda _: {"approved": True, "lease_expires_unix": self.mono + 180}
        self.assertEqual(self.guardian().run(), 0)
        self.assertEqual(self.spawn.call_count, 1); self.assertEqual(self.child.terminates, 0)

    def test_spawn_uses_owned_session_and_frozen_package_entrypoint(self):
        self.child.exit_at = 1
        self.guardian().run()
        args, kwargs = self.spawn.call_args
        self.assertIn("replication.cloud.agent_steering.persistent_worker", args[0])
        self.assertEqual(args[0][-1], str(self.config_path.resolve()))
        self.assertTrue(kwargs["start_new_session"]); self.assertTrue(callable(kwargs["preexec_fn"]))
        self.assertEqual(kwargs["cwd"], str(g.SOURCE_ROOT))

    def test_explicit_guardian_stop_ends_only_its_child(self):
        guardian = self.guardian()
        def sleep(seconds):
            self.advance(seconds)
            if self.mono >= 1: guardian.stop_requested = True
        guardian.sleep = sleep
        guardian.run()
        self.assertEqual(self.child.terminates, 1)
        self.assertEqual(guardian.stop_reason, "explicit_guardian_stop")

    def test_mismatched_pod_identity_is_rejected_before_spawn(self):
        with patch.dict(g.os.environ, {"RUNPOD_POD_ID": "different"}):
            with self.assertRaisesRegex(ValueError, "different pod"): self.guardian()
        self.spawn.assert_not_called()

    def test_config_change_while_waiting_prevents_model_startup(self):
        guardian = self.guardian()
        self.config_path.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            guardian.run()
        self.spawn.assert_not_called()


if __name__ == "__main__": unittest.main()
