"""I test queue preservation and lifecycle with no model, sandbox or provider."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("tested_queue_runner", HERE / "queue_runner.py")
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
PREPARED = q.SERIES / "gpu-b-20260912T041801Z/prepared"


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.plan = q.read(PREPARED / "plan.json")
        self.registration = {"execution_approved": True, "arms": self.plan["engine_arms"],
             "out_dir": "/workspace/results/agent-steering/test-only",
             "plan_sha256": q.sha(PREPARED / "plan.json"), "directions_file_sha256": "directions",
             "probe_file_sha256": "probes"}
        q.save(self.root / "registration.json", self.registration)
        q.save(self.root / "connection.json", {"pod_id": "test-only-pod", "host": "unused", "port": 1, "key": "/unused"})
        q.save(self.root / "lease.json", {"approved": True, "expected_pod_id": "test-only-pod", "budget_remaining_usd": 1,
             "lease_expires_unix": q.CLOSEOUT_UNIX, "shutdown_at_unix": q.CLOSEOUT_UNIX})
        self.config = {"execution_approved": True, "expected_pod_id": "test-only-pod",
             "plan": str(PREPARED / "plan.json"), "plan_sha256": q.sha(PREPARED / "plan.json"),
             "registration": str(self.root / "registration.json"), "registration_sha256": q.sha(self.root / "registration.json"),
             "source_root": str(PREPARED / "source"), "local_out_dir": str(self.root / "results"),
             "remote_out_dir": self.registration["out_dir"], "connection_file": str(self.root / "connection.json"),
             "lease_path": str(self.root / "lease.json"), "sandbox_image_id": "test-only-sandbox",
             "stop_launch_unix": q.STOP_LAUNCH_UNIX, "closeout_unix": q.CLOSEOUT_UNIX}
        q.save(self.root / "config.json", self.config)
        self.calls = []

    def tearDown(self): self.temp.cleanup()

    def modules(self):
        owner = self
        class Backend:
            def __init__(self, *args): self.arm = {"arm_id": "none"}
            def wait_ready(self, timeout_s):
                return {"job_file_sha256": owner.config["registration_sha256"],
                        "backend": {"directions_file_sha256": "directions", "probe_file_sha256": "probes"}}
            def select(self, arm, **context): self.arm = {"arm_id": arm}
            def finish(self, *args): raise AssertionError("An item must not stop the unified GPU job")
        def gate(backend, diagnostics, directory, proof, magnitude):
            q.save(directory / "PASSED.json", {"passed": True})
        def episode(backend, case, html, out, **kwargs):
            owner.calls.append((copy.deepcopy(case), dict(kwargs)))
            out.mkdir(parents=True)
            result = {"status": "completed", "censored": False, "verified_dummy_upload": False,
                 "receipts": [], "generated_tokens": 30, "elapsed_s": 1, "exposure_confirmed": case["variant"] != "benign",
                 "steps": [{"tool_calls": [{"model_visible_output": {"stdout": html}}]},
                           {"generation": {"generated_tokens": 30}, "parsed": {"final_output": "A summary. " + "word " * 30}}]}
            q.save(out / "episode.json", result)
            return result
        ep = types.ModuleType("episode"); ep.run_episode = episode
        client = types.ModuleType("local_backend"); client.RemoteBackend = Backend
        runner = types.ModuleType("runner"); runner.technical_gate = gate
        return {"replication.cloud.agent_steering.episode": ep,
                "replication.cloud.agent_steering.local_backend": client,
                "replication.cloud.agent_steering.runner": runner}

    def test_exact_groups_and_resample(self):
        q.validate_plan(self.plan)
        self.assertEqual([len(i["case_ids"]) for i in self.plan["items"]], [15, 10, 5])
        cases = {c["id"]: c for c in self.plan["cases"]}
        self.assertEqual(cases["004-forgery"]["seed"], 11243)
        self.assertEqual(cases["004-standard"]["seed"], 1242)
        self.assertEqual(self.plan["items"][0]["dev_note"], q.REMINDER)

    def test_reject_changed_reminder(self):
        self.plan["items"][0]["dev_note"] += " changed"
        with self.assertRaises(ValueError): q.validate_plan(self.plan)

    def test_benign_sentinel_stays_absent(self):
        case = next(c for c in self.plan["cases"] if c["variant"] == "benign")
        html = Path(case["fixture_path"]).read_text(); adapted = q.runtime_case(case, html)
        self.assertNotIn(adapted["payload"], html)
        self.assertEqual(case["payload"], "")
        with self.assertRaises(ValueError): q.runtime_case(case, html + adapted["payload"])

    def test_complete_item_one_preserves_all_inputs_and_never_finishes_job(self):
        with patch.dict(sys.modules, self.modules()), patch.object(q.time, "time", return_value=q.START_UNIX+60):
            self.assertEqual(q.run_item(self.root / "config.json", 1), 0)
        self.assertEqual(len(self.calls), 15)
        self.assertTrue(all(args["dev_note"] == q.REMINDER and args["max_new_tokens"] == 4096 and args["max_turns"] == 8 for _, args in self.calls))
        self.assertEqual(sum(c["variant"] == "benign" for c, _ in self.calls), 5)
        done = q.read(self.root / "results/item-1/FINISHED.json")
        self.assertEqual((done["status"], done["recorded"]), ("completed", 15))
        summaries = list((self.root / "results/item-1/episodes").glob("*/*/summary.json"))
        self.assertEqual(len(summaries), 15)
        self.assertTrue(all(q.read(p)["summary_quality_judged"] is False for p in summaries))

    def test_cutoff_starts_no_episode(self):
        with patch.dict(sys.modules, self.modules()), patch.object(q.time, "time", return_value=q.STOP_LAUNCH_UNIX):
            self.assertEqual(q.run_item(self.root / "config.json", 1), 0)
        self.assertEqual(self.calls, [])
        self.assertEqual(q.read(self.root / "results/item-1/FINISHED.json")["status"], "time_cutoff")

    def test_item_three_cannot_overtake_item_two(self):
        with patch.dict(sys.modules, self.modules()):
            with self.assertRaises(ValueError): q.run_item(self.root / "config.json", 3)
        self.assertFalse((self.root / "results/item-3").exists())

    def test_protected_gpu_rejected_before_any_model_request(self):
        self.config["expected_pod_id"] = q.PROTECTED; q.save(self.root / "config.json", self.config)
        with self.assertRaises(ValueError): q.run_item(self.root / "config.json", 1)
        self.assertFalse((self.root / "results").exists())

    def test_expired_lease_preserves_unrun_denominator(self):
        q.save(self.root / "lease.json", {"approved": False})
        with patch.dict(sys.modules, self.modules()), patch.object(q.time, "time", return_value=q.START_UNIX+60):
            with self.assertRaises(RuntimeError): q.run_item(self.root / "config.json", 1)
        summary = q.read(self.root / "results/item-1/summary.json")
        self.assertEqual((summary["assigned"], summary["recorded"], len(summary["unrun_case_ids"])), (15, 0, 15))

    def test_summary_rejects_upload_without_receipt(self):
        case = self.plan["cases"][0]
        q.save(self.root / "episode.json", {})
        with self.assertRaises(ValueError):
            q.episode_summary({"verified_dummy_upload": True, "receipts": []}, case, "html", self.root, self.plan["items"][0], "none")


if __name__ == "__main__": unittest.main()
