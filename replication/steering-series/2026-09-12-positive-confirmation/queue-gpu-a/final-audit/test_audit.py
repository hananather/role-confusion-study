"""I test the independent auditor with saved closed evidence and in-memory corruptions."""
import copy
import json
from pathlib import Path
import unittest
import audit_queue as a


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((a.PACKET / "local-config.json").read_bytes())
        cls.plan = json.loads(Path(cls.config["plan_file"]).read_bytes())
        cls.local = Path(cls.config["out_dir"])
        cls.rows = json.loads((cls.local / "attribution-index.json").read_bytes())
        cls.rpc, _ = a.load_rpc(cls.local)
        cls.protocol = a.module(a.PREPARED / "source/replication/cloud/agent_steering/frozen_harness/protocol.py", "test_queue_protocol")

    def setUp(self):
        a.ERRORS.clear(); a.FILES.clear()

    def test_all_ten_closed_readouts(self):
        self.assertEqual(len(self.rows), 10)
        for row, sample in zip(self.rows, self.plan["attribution"]):
            a.audit_attribution(row, sample, self.plan, self.protocol, self.rpc)
        self.assertEqual(a.ERRORS, [])

    def test_independent_contract_rejects_hooks_seed_and_token_cap(self):
        sample = self.plan["attribution"][0]
        for field, value in (("hook_calls", 1), ("seed", 0), ("max_new_tokens", 201)):
            with self.subTest(field=field):
                a.ERRORS.clear()
                row = copy.deepcopy(self.rows[0]); row["stats"][field] = value
                rpc = dict(self.rpc); request, response = rpc[a.attribution_key(sample)]
                # I keep both raw and indexed result copies consistent, so the
                # frozen contract, rather than a simple copy check, must fail.
                rpc[a.attribution_key(sample)] = request, {**response, "stats": row["stats"]}
                a.audit_attribution(row, sample, self.plan, self.protocol, rpc)
                self.assertTrue(a.ERRORS)
                self.assertNotIn("Attribution result differs from its raw RPC", a.ERRORS)

    def test_false_censor_flag_and_changed_baseline_join_are_rejected(self):
        sample = self.plan["attribution"][0]
        for field, value in (("censored", True), ("baseline_episode_sha256", "0"*64)):
            with self.subTest(field=field):
                a.ERRORS.clear(); row = copy.deepcopy(self.rows[0]); row[field] = value
                a.audit_attribution(row, sample, self.plan, self.protocol, self.rpc)
                self.assertTrue(a.ERRORS)

    def test_upload_priority_and_unresolved_labels(self):
        self.assertEqual(a.label(True, True, True, "token_limit"), "U")
        self.assertEqual(a.label(False, True, True, "token_limit"), "C")
        self.assertEqual(a.label(False, False, False, "completed"), "X")
        self.assertEqual(a.label(False, True, False, "completed"), "N")

    def test_available_closed_standard_episodes(self):
        common = a.module(a.SERIES / "bridge-001/final-audit/audit_bridge.py", "test_queue_receipts")
        index = json.loads((self.local / "episode-index.json").read_bytes())
        self.assertGreater(len(index), 0)
        # I inspect only the closed, indexed prefix; the live episode is excluded.
        for row, case in zip(index, self.plan["standard_cases"]):
            a.audit_episode(row, case, self.plan, self.protocol, common, self.rpc)
        self.assertEqual(a.ERRORS + common.ERRORS, [])


if __name__ == "__main__":
    unittest.main()
