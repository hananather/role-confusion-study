"""I test follow-up gates and handoff with temporary files and fake processes only."""
import base64
import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import dispatcher as d


class DispatcherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.now = 10000.
        self.stage = self.root / "stage"
        entry = self.stage / "replication/cloud/agent_steering/new_page_runner.py"
        entry.parent.mkdir(parents=True); entry.write_text("# Frozen test entrypoint\n")
        self.prior = self.root / "prior"; self.prior.mkdir()
        self.local = self.root / "new-local"
        self.lease = self.root / "lease.json"
        self.connection = self.root / "connection.json"
        d.write_json(self.connection, {"host": "host.example", "port": 22, "key": str(self.root / "key")})
        self.refresh_lease()
        planfile, proof, diagnostics = [self.root / name for name in ("plan.json", "proof.json", "diagnostics.json")]
        plan = {"scope": "new_page_descriptive_extension", "page_count": 5, "expected_episode_count": 40,
                "selected_bank_indices": list(range(5)), "jobs": [{}]*40, "engine_arms": [{"arm_id": "none"}], "cases": []}
        d.write_json(planfile, plan); d.write_json(proof, {"passed": True}); d.write_json(diagnostics, [])
        self.registration = self.root / "registration.json"
        d.write_json(self.registration, {"schema_version": 1, "execution_approved": True, "job_id": "newjob",
                     "out_dir": "/workspace/results/agent-steering/newjob", "plan_sha256": d.sha(planfile),
                     "arms": plan["engine_arms"], "directions_file_sha256": "d"*64, "probe_file_sha256": "e"*64})
        self.runner_config = self.root / "runner.json"
        local_config = {"execution_approved": True, "out_dir": str(self.local),
                        "expected_pod_id": "pod1", "remote_results": "/workspace/results/agent-steering/newjob",
                        "lease_path": str(self.lease), "connection_file": str(self.connection), "work_deadline_unix": 20000}
        for field, hashfield, path in (("plan_file", "plan_sha256", planfile), ("diagnostics_file", "diagnostics_sha256", diagnostics),
                                       ("model_free_validation", "model_free_validation_sha256", proof), ("registration_file", "registration_sha256", self.registration)):
            local_config[field], local_config[hashfield] = str(path), d.sha(path)
        d.write_json(self.runner_config, local_config)
        self.config = {"version": 1, "execution_approved": True, "job_id": "newjob", "expected_pod_id": "pod1",
                       "current_run_dir": str(self.prior), "current_runner_pid": 456, "predecessor_pid": 530,
                       "lease_path": str(self.lease), "connection_file": str(self.connection), "all_in_rate_usd_h": 3.88,
                       "registration_file": str(self.registration), "local_results": str(self.local),
                       "remote_results": "/workspace/results/agent-steering/newjob",
                       "remote_registration_path": "/workspace/session-control/pod1/model-service/jobs/newjob.json",
                       "local_launch": {"cwd": str(self.stage), "argv": ["/usr/bin/python3", "-u", "-m", d.PACKAGE+"new_page_runner", "--config", str(self.runner_config)], "logpath": str(self.root/"local.log")},
                       "remote_launch": {"cwd": "/workspace/frozen", "argv": ["/workspace/venv-probes/bin/python", "-u", "-m", d.PACKAGE+"service_guardian", "--config", "/workspace/frozen/service.json", "--state-dir", "/workspace/session-control/pod1/model-service/guardian"], "logpath": "/workspace/session-control/pod1/guardian.log"},
                       "local_frozen_files": [{"path": str(p), "sha256": d.sha(p)} for p in (entry, self.runner_config, self.registration, self.connection, planfile, proof, diagnostics)],
                       "remote_frozen_files": [{"path": "/workspace/frozen/service.json", "sha256": "a"*64}, {"path": "/workspace/frozen/replication/cloud/agent_steering/service_guardian.py", "sha256": "b"*64}]}
        self.config_path = self.root / "dispatcher.json"
        self.state = self.root / "dispatch-state"
        self.calls, self.spawns = [], []
        self.complete_bridge()

    def refresh_lease(self):
        d.write_json(self.lease, {"schema_version": 1, "approved": True, "expected_pod_id": "pod1", "lease_id": "session1",
                     "issued_unix": self.now, "lease_expires_unix": self.now+180, "shutdown_at_unix": 20000, "budget_remaining_usd": 12.})

    def complete_bridge(self):
        outcome = {"status": "completed", "recorded_episodes": 35, "planned_episodes": 35}
        for name in ("FINISHED.json", "status.json"): d.write_json(self.prior/name, outcome)
        d.write_json(self.prior/"summary.json", {"recorded_episodes": 35, "total_planned_episodes": 35})
        rows = [{"case_id": str(i), "arm_id": arm, "elapsed_s": (i+1)*10, "censored": i == 3, "verified_dummy_upload": i == 1}
                for i in range(5) for arm in sorted(d.BRIDGE_ARMS)]
        d.write_json(self.prior/"episode-index.json", rows)
        names = ("zero_token_identity", "zero_edits", "mask_edit_counts", "matched_vector_norm", "finite_downstream_readout", "no_timeout")
        d.write_json(self.prior/"engineering-pilot/PASSED.json", {"passed": True, "samples": [{"case_id": str(i), **{k: True for k in names}} for i in range(5)]})
        d.write_json(self.prior/"full-zero-identity.json", {"pairs_available": 5, "planned_pairs": 5, "unexplained_identity_failure": False,
                     "comparisons": [{"case_id": str(i), "unexplained_identity_failure": False, "generations": [{"both_reached": True, "prompts_identical": True, "tokens_identical": True}]} for i in range(5)]})

    def remote(self, connection, payload):
        self.calls.append(payload)
        return {"guardian_pid": 987, "job_id": "newjob", "expected_pod_id": "pod1", "registration_sha256": payload["registration_sha256"]}

    def ready(self, connection, payload):
        backend = {key: d.read_json(self.registration)[key] for key in ("directions_file_sha256", "probe_file_sha256")}
        return {"terminals": {}, "guardian": {"guardian_pid": 987, "worker_pid": 988, "expected_pod_id": "pod1", "config_sha256": "a"*64},
                "service": {"status": "ready", "pid": 988, "backend": backend},
                "job": {"status": "ready", "pid": 988, "backend": backend, "job_file_sha256": payload["registration_sha256"]}}

    def spawn(self, argv, **kwargs):
        self.spawns.append((argv, kwargs)); return SimpleNamespace(pid=876)

    def dispatcher(self, **kwargs):
        d.write_json(self.config_path, self.config)
        return d.Dispatcher(self.config_path, self.state, remote=self.remote, status=kwargs.pop("status", self.ready), spawn=self.spawn,
                            now=lambda: self.now, sleep=kwargs.pop("sleep", lambda _: None), **kwargs)

    def assert_blocked(self, dispatcher):
        self.assertEqual(dispatcher.run(), 2)
        self.assertTrue((self.state/"BLOCKED.json").exists())
        self.assertEqual(self.spawns, [])

    def test_launch_uses_exact_bytes_and_waits_for_ready(self):
        dispatch = self.dispatcher()
        self.assertEqual(dispatch.run(), 0)
        self.assertEqual(base64.b64decode(self.calls[0]["registration_base64"]), self.registration.read_bytes())
        self.assertEqual(self.spawns[0][0], self.config["local_launch"]["argv"])
        self.assertTrue(self.spawns[0][1]["start_new_session"])
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(dispatch.run(), 0)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(self.local.exists())  # My fake child never runs the experiment.

    def test_waits_30_seconds_for_prior_runner(self):
        (self.prior/"FINISHED.json").unlink()
        waits = []
        def sleep(seconds):
            waits.append(seconds); self.complete_bridge()
        self.assertEqual(self.dispatcher(exists=lambda pid: True, sleep=sleep).run(), 0)
        self.assertEqual(waits, [30])

    def test_no_receipt_and_dead_runner_blocks(self):
        (self.prior/"FINISHED.json").unlink()
        self.assert_blocked(self.dispatcher(exists=lambda pid: False)); self.assertEqual(self.calls, [])

    def test_failed_prior_queue_blocks(self):
        d.write_json(self.prior/"FINISHED.json", {"status": "failed", "recorded_episodes": 35, "planned_episodes": 35})
        self.assert_blocked(self.dispatcher()); self.assertEqual(self.calls, [])

    def test_all_timing_includes_censored_and_adverse_rows(self):
        result = d.prerequisite(self.prior)
        self.assertEqual(result["mean_episode_seconds"], 30)
        self.assertEqual(result["required_seconds"], 30*40*1.3+600)
        self.assertFalse(result["outcome_selection"])

    def test_missing_or_duplicate_bridge_rows_rejected(self):
        rows = d.read_json(self.prior/"episode-index.json"); rows[-1] = rows[0]
        d.write_json(self.prior/"episode-index.json", rows)
        self.assert_blocked(self.dispatcher())

    def test_five_actual_identity_pairs_required(self):
        identity = d.read_json(self.prior/"full-zero-identity.json"); identity["comparisons"][0]["generations"] = []
        d.write_json(self.prior/"full-zero-identity.json", identity)
        self.assert_blocked(self.dispatcher())

    def test_identity_divergence_cannot_be_hidden_by_receipt_flag(self):
        identity = d.read_json(self.prior/"full-zero-identity.json")
        identity["comparisons"][0]["generations"][0]["tokens_identical"] = False
        d.write_json(self.prior/"full-zero-identity.json", identity)
        self.assert_blocked(self.dispatcher())

    def test_explained_timeout_prefix_remains_eligible(self):
        identity = d.read_json(self.prior/"full-zero-identity.json")
        identity["comparisons"][0]["generations"][0].update(tokens_identical=False, timing_censored=True, shared_prefix_identical=True, censored_length_difference=True)
        d.write_json(self.prior/"full-zero-identity.json", identity)
        self.assertIsNotNone(d.prerequisite(self.prior))

    def test_engineering_failure_blocks(self):
        pilot = d.read_json(self.prior/"engineering-pilot/PASSED.json"); pilot["samples"][0]["mask_edit_counts"] = False
        d.write_json(self.prior/"engineering-pilot/PASSED.json", pilot)
        self.assert_blocked(self.dispatcher()); self.assertEqual(self.calls, [])

    def test_insufficient_budget_and_stale_wrong_pod_lease(self):
        lease = d.read_json(self.lease)
        for edit in ({"budget_remaining_usd": .1}, {"lease_expires_unix": self.now-1}, {"expected_pod_id": "another"}, {"budget_remaining_usd": float("nan")}):
            with self.subTest(edit=edit), self.assertRaises(ValueError):
                d.budget_gate({**lease, **edit}, "pod1", 2160, 3.88, self.now)

    def test_changed_source_or_config_blocks_before_remote(self):
        dispatch = self.dispatcher()
        Path(self.config["local_frozen_files"][0]["path"]).write_text("changed")
        self.assert_blocked(dispatch); self.assertEqual(self.calls, [])

    def test_config_mutation_while_waiting_blocks(self):
        dispatch = self.dispatcher()
        self.config_path.write_text("{}")
        self.assert_blocked(dispatch); self.assertEqual(self.calls, [])

    def test_stop_does_not_dispatch_or_delete(self):
        dispatch = self.dispatcher(); (self.state/"STOP").touch()
        self.assert_blocked(dispatch); self.assertEqual(self.calls, [])
        self.assertEqual(d.read_json(self.state/"BLOCKED.json")["allocation_action"], "none")

    def test_uncertain_remote_launch_is_never_replayed(self):
        dispatch = self.dispatcher()
        dispatch.remote = lambda *_: (_ for _ in ()).throw(TimeoutError("lost response"))
        self.assert_blocked(dispatch)
        self.assertTrue((self.state/"LAUNCH-INTENT.json").exists())
        dispatch.remote = self.remote
        self.assert_blocked(dispatch); self.assertEqual(self.calls, [])

    def test_waits_for_actual_matching_model_and_job_readiness(self):
        statuses = [None, True]; waits = []
        def status(connection, payload):
            return self.ready(connection, payload) if statuses.pop(0) else {"terminals": {}, "job": None}
        def sleep(seconds):
            waits.append(seconds); self.now += seconds; self.refresh_lease()
        self.assertEqual(self.dispatcher(status=status, sleep=sleep).run(), 0)
        self.assertEqual(waits, [10])
        self.assertTrue((self.state/"REMOTE-READY.json").exists())

    def test_guardian_terminal_blocks_local_runner_only(self):
        self.assert_blocked(self.dispatcher(status=lambda *_: {"terminals": {"guardian": {"returncode": 2}}}))
        self.assertEqual(len(self.calls), 1)

    def test_ready_with_wrong_assets_blocks(self):
        def status(*args):
            receipt = self.ready(*args); receipt["job"]["backend"]["probe_file_sha256"] = "f"*64; return receipt
        self.assert_blocked(self.dispatcher(status=status))

    def test_sync_registration_preserves_existing_target(self):
        path = self.root/"sync-sources.json"
        old = {"remote": "/workspace/results/agent-steering/old", "local": str(self.root/"old-mirror")}
        d.write_json(path, [old])
        self.config.update(sync_sources_file=str(path), local_sync_results=str(self.root/"new-mirror"))
        self.assertEqual(self.dispatcher().run(), 0)
        self.assertEqual(d.read_json(path), [old, {"remote": self.config["remote_results"], "local": str(self.root/"new-mirror")}])

    def test_existing_local_output_blocks(self):
        self.local.mkdir()
        self.assert_blocked(self.dispatcher()); self.assertEqual(self.calls, [])

    def test_fixed_ssh_program_and_json_stdin(self):
        payload = {"opaque": "$(touch /tmp/never-execute); `true`\n"}
        with patch.object(d.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=b'{"ok":true}', stderr=b"")) as run:
            self.assertEqual(d.remote_launch(d.read_json(self.connection), payload), {"ok": True})
            argv = run.call_args.args[0]
            self.assertNotIn(payload["opaque"], " ".join(argv))
            self.assertEqual(json.loads(run.call_args.kwargs["input"]), payload)
            self.assertFalse(run.call_args.kwargs.get("shell", False))
        compile(d.REMOTE_BOOTSTRAP, "remote-bootstrap", "exec")
        compile(d.REMOTE_STATUS, "remote-status", "exec")


if __name__ == "__main__":
    unittest.main()
