"""I test my new bounded agent lifecycle with fake providers and source fixtures only."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from replication.cloud.agent_steering import control as c


class FakeProvider:
    def __init__(self):
        self.inventory = []
        self.balance = 20
        self.autopay = False
        self.quote = 3.49
        self.get_status = 200
        self.calls = []

    def pods(self):
        return self.inventory

    def account(self, _):
        return {"balance_usd": self.balance, "autopay_enabled": self.autopay}

    def quote_and_volume(self, _manifest):
        return {"rate_usd_h": self.quote}

    def request(self, method, path, body=None, **kwargs):
        self.calls.append((method, path))
        return (204, None) if method == "DELETE" else (self.get_status, {"desiredStatus": "EXITED"})


def manifest():
    return {"version": 1, "run_id": "mock-run", "scope": "agent_steering_bridge", "rate_usd_h": 3.49,
            "planned_hours": 2, "grace_minutes": 20, "cap_usd": 20, "series_cap_usd": 10,
            "prior_spend_usd": 2.22, "container_disk_gb": 30, "container_disk_usd_gb_month": 0.1,
            "storage_month_hours": 730, "existing_volume_reserve_usd": 0.90,
            "existing_volume_usd_month": 280, "network_volume_id": c.VOLUME, "data_center": "EU-FR-1",
            "gpu_type": c.GPU, "cloud": "SECURE", "image": "frozen-image",
            "account_check": {"query": "query { myself { clientBalance isAutoPayEnabled } }"},
            "command": [c.PYTHON, "-u", "-m", "replication.cloud.agent_steering.worker",
                        "--config", "/workspace/" + c.REL + "/config.json"],
            "bridge_plan": "/workspace/" + c.REL + "/bridge-plan.json",
            "model_free_validation": "/workspace/" + c.REL + "/tokenizer-proof.json",
            "runtime_review_receipt": "/workspace/" + c.REL + "/runtime-review.json"}


def packet(directory):
    data = manifest(); data["files"] = []
    def add(relative, value):
        source = directory / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(json.dumps(value) if not isinstance(value, str) else value)
        data["files"].append({"source": str(source), "remote": relative, "sha256": c.digest(source)})
        return source
    for name in sorted(c.REQUIRED_SOURCES):
        add(name, "# My immutable mock source; I never import this fixture.\n")
    vector_path = add(c.REL + "/inputs/block11.npz", "mock vector bytes")
    probe_path = add(c.REL + "/inputs/probes.npz", "mock probe bytes")
    arms = [{"arm_id": key, "direction": value[0], "alpha": value[1], "hooks_enabled": value[2],
             "layer": 11, "mask_mode": "tool"} for key, value in c.ARM_CONTRACT.items()]
    cases = [{"id": f"{i:03d}-forgery", "seed": seed} for i, seed in enumerate(c.BRIDGE_SEEDS)]
    plan = {"execution_approved": True, "stage": "bridge35", "case_count": 5, "expected_episode_count": 35,
            "cases": cases, "arms": arms, "settings": c.BRIDGE_SETTINGS,
            "directions_sha256": c.digest(vector_path), "probe_sha256": c.digest(probe_path),
            "jobs": [{"case_id": case["id"], "arm_id": arm, "seed": case["seed"]} for case in cases for arm in c.ARM_IDS],
            "engineering_pilot": {"required": True, "samples": 5, "arms": ["none", "zero", "role_a16"],
                                  "max_new_tokens": 64,
                                  "gate": ["zero_token_identity", "mask_edit_counts", "finite_downstream_readout"]}}
    plan_path = add(c.REL + "/bridge-plan.json", plan)
    add(c.REL + "/config.json", {"execution_approved": True, "bridge_plan": data["bridge_plan"],
                                  "out_dir": "/workspace/results/agent-steering/mock-run", "arms": arms,
                                  "directions_file": "/workspace/" + c.REL + "/inputs/block11.npz",
                                  "directions_file_sha256": c.digest(vector_path),
                                  "probe_file": "/workspace/" + c.REL + "/inputs/probes.npz",
                                  "probe_file_sha256": c.digest(probe_path)})
    add(c.REL + "/tokenizer-proof.json", {"passed": True, "bridge_plan_sha256": c.digest(plan_path),
        "samples": [{"case_id": case["id"], "prompt_sha256": "a" * 64, "token_count": 214,
                     "page_tokens": 100, "payload_tokens": 20, "roundtrip": True,
                     "initial_prompt_identical": True, "mask_disjoint_from_header": True} for case in cases]})
    add(c.REL + "/runtime-review.json", {"passed": True, "source_sha256": {
        row["remote"]: row["sha256"] for row in data["files"] if row["remote"] in c.REQUIRED_SOURCES}})
    path = directory / "manifest.json"; c.write_json(path, data)
    return path


def change_packet(path, remote, transform):
    data = c.read_json(path)
    entry = next(row for row in data["files"] if row["remote"] == remote)
    target = Path(entry["source"]); value = c.read_json(target); transform(value); c.write_json(target, value)
    entry["sha256"] = c.digest(target)
    c.write_json(path, data)


class BudgetTests(unittest.TestCase):
    def test_two_hour_bridge_cost_carries_both_caps_and_prior_cost(self):
        value = c.cost(manifest())
        self.assertEqual(value["gpu_planned_usd"], 6.98)
        self.assertAlmostEqual(value["gpu_with_grace_usd"], 8.1433)
        self.assertAlmostEqual(value["existing_volume_minimum_reserve_usd"], 0.894977)
        self.assertEqual(value["maximum_series_usd"], 9.0529)
        self.assertEqual(value["maximum_total_usd"], 11.2729)
        self.assertEqual(value["series_cap_usd"], 10)

    def test_fixed_envelope_rejects_excess_and_missing_prior_cost(self):
        for key, value in [("rate_usd_h", 3.5), ("planned_hours", 2.1), ("grace_minutes", 21),
                           ("series_cap_usd", 10.1), ("series_cap_usd", 9), ("cap_usd", 20.1),
                           ("cap_usd", 11), ("prior_spend_usd", 0), ("prior_spend_usd", float("nan")),
                           ("existing_volume_reserve_usd", 0.4), ("existing_volume_usd_month", None)]:
            with self.subTest(key=key, value=value):
                data = manifest(); data[key] = value
                with self.assertRaises(ValueError): c.cost(data)

    def test_environment_removes_account_credentials_and_wrong_model_cache(self):
        env = c.batch_environment({"RUNPOD_API_KEY": "dummy", "HF_TOKEN": "dummy", "HF_HUB_CACHE": "/wrong", "PATH": "/usr/bin"})
        self.assertNotIn("RUNPOD_API_KEY", env); self.assertNotIn("HF_TOKEN", env); self.assertNotIn("HF_HUB_CACHE", env)
        self.assertEqual(env["HF_HOME"], "/workspace/hf/home"); self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")

    def test_preflight_rejects_autopay_other_pod_credit_shortfall_and_rate_increase(self):
        for field, value in [("autopay", True), ("inventory", [{"id": "other"}]), ("balance", 9), ("quote", 3.5)]:
            provider = FakeProvider(); setattr(provider, field, value)
            with self.assertRaises(ValueError): c.preflight(provider, manifest())
        self.assertEqual(c.preflight(FakeProvider(), manifest())["existing_pods"], [])

    def test_null_or_unavailable_regional_quote_fails_closed(self):
        provider = object.__new__(c.Provider)
        for stock, price, counts in [(None, None, None), ("Low", 3.49, [2, 4])]:
            provider.request = Mock(return_value=(200, {"data": {"gpuTypes": [{"id": c.GPU,
                "lowestPrice": {"stockStatus": stock, "uninterruptablePrice": price, "availableGpuCounts": counts}}]}}))
            with self.assertRaises(ValueError): provider.quote_and_volume(manifest())

    def test_positive_single_gpu_quote_allows_null_count_list(self):
        provider = object.__new__(c.Provider)
        provider.request = Mock(side_effect=[(200, {"data": {"gpuTypes": [{"id": c.GPU,
            "lowestPrice": {"stockStatus": "Low", "uninterruptablePrice": 3.49, "availableGpuCounts": None}}]}}),
            (200, {"id": c.VOLUME, "size": 2000, "dataCenterId": "EU-FR-1"})])
        self.assertEqual(provider.quote_and_volume(manifest())["rate_usd_h"], 3.49)

    def test_nonboolean_autopay_cannot_pass(self):
        provider = object.__new__(c.Provider)
        spec = {"query": "query {}", "balance_path": ["data", "myself", "clientBalance"],
                "autopay_path": ["data", "myself", "isAutoPayEnabled"]}
        for value in (None, 0, "false"):
            provider.request = Mock(return_value=(200, {"data": {"myself": {"clientBalance": 20, "isAutoPayEnabled": value}}}))
            with self.assertRaises(RuntimeError): provider.account(spec)


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = packet(Path(self.tmp.name))

    def test_complete_packet_passes_without_loading_or_network(self):
        with patch.object(c, "Provider", side_effect=AssertionError("No provider access")):
            self.assertEqual(c.validate_manifest(self.path)["run_id"], "mock-run")
            for mode in ("plan", "dry-run"):
                with patch("sys.argv", ["control.py", mode, "--manifest", str(self.path)]), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(c.main(), 0)

    def test_changed_frozen_source_fails(self):
        source = Path(c.read_json(self.path)["files"][0]["source"]); source.write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"): c.validate_manifest(self.path)

    def test_missing_transitive_dependency_fails(self):
        data = c.read_json(self.path); data["files"] = data["files"][1:]; c.write_json(self.path, data)
        with self.assertRaisesRegex(ValueError, "dependency"): c.validate_manifest(self.path)

    def test_missing_pilot_gate_fails_before_model(self):
        change_packet(self.path, c.REL + "/bridge-plan.json", lambda p: p["engineering_pilot"].update(gate=[]))
        with self.assertRaisesRegex(ValueError, "engineering pilot"): c.validate_manifest(self.path)

    def test_duplicate_job_cannot_create_nominal_35(self):
        change_packet(self.path, c.REL + "/bridge-plan.json", lambda p: p["jobs"].__setitem__(0, p["jobs"][1]))
        with self.assertRaisesRegex(ValueError, "exactly once"): c.validate_manifest(self.path)

    def test_wrong_case_seed_or_arm_fails(self):
        change_packet(self.path, c.REL + "/bridge-plan.json", lambda p: p["cases"][0].update(seed=99))
        with self.assertRaisesRegex(ValueError, "original seeds"): c.validate_manifest(self.path)

    def test_tokenizer_proof_is_bound_to_plan(self):
        change_packet(self.path, c.REL + "/tokenizer-proof.json", lambda p: p.update(bridge_plan_sha256="b"*64))
        with self.assertRaisesRegex(ValueError, "current bridge plan"): c.validate_manifest(self.path)

    def test_tokenizer_proof_requires_five_real_case_ids(self):
        change_packet(self.path, c.REL + "/tokenizer-proof.json", lambda p: p["samples"][0].update(case_id="missing"))
        with self.assertRaisesRegex(ValueError, "five actual prompts"): c.validate_manifest(self.path)

    def test_review_must_bind_all_current_executable_sources(self):
        change_packet(self.path, c.REL + "/runtime-review.json", lambda p: p["source_sha256"].pop(c.REL + "/worker.py"))
        with self.assertRaisesRegex(ValueError, "source hashes"): c.validate_manifest(self.path)

    def test_unapproved_or_differently_scoped_packet_fails(self):
        change_packet(self.path, c.REL + "/config.json", lambda p: p.update(execution_approved=False))
        with self.assertRaisesRegex(ValueError, "execution"): c.validate_manifest(self.path)

    def test_worker_output_cannot_escape_supervised_directory(self):
        change_packet(self.path, c.REL + "/config.json", lambda p: p.update(out_dir="/workspace/other"))
        with self.assertRaisesRegex(ValueError, "output directory"): c.validate_manifest(self.path)

    def test_worker_cannot_relabel_a_different_dose_as_alpha16(self):
        change_packet(self.path, c.REL + "/config.json", lambda p: p["arms"][2].update(alpha=8))
        with self.assertRaisesRegex(ValueError, "dose"): c.validate_manifest(self.path)

    def test_vector_and_probe_must_be_frozen_exact_inputs(self):
        change_packet(self.path, c.REL + "/config.json", lambda p: p.update(directions_file_sha256="b" * 64))
        with self.assertRaisesRegex(ValueError, "directions_file"): c.validate_manifest(self.path)

    def test_jobs_cannot_change_seed_with_same_case_label(self):
        change_packet(self.path, c.REL + "/bridge-plan.json", lambda p: p["jobs"][0].update(seed=99))
        with self.assertRaisesRegex(ValueError, "case seed"): c.validate_manifest(self.path)

    def test_full_episode_settings_cannot_change_with_same_scope(self):
        change_packet(self.path, c.REL + "/bridge-plan.json", lambda p: p["settings"].update(max_new_tokens=64))
        with self.assertRaisesRegex(ValueError, "generation settings"): c.validate_manifest(self.path)

    def test_chat_command_and_new_volume_are_rejected(self):
        data = c.read_json(self.path); data["command"][3] = "replication.cloud.chat_steering.batch"; c.write_json(self.path, data)
        with self.assertRaisesRegex(ValueError, "file-worker"): c.validate_manifest(self.path)
        data = c.read_json(self.path); data["network_volume_id"] = "different"; c.write_json(self.path, data)
        with self.assertRaisesRegex(ValueError, "existing approved"): c.validate_manifest(self.path)

    def test_approval_binds_new_manifest_and_exact_command(self):
        receipt = self.path.parent / "approval.json"
        c.write_json(receipt, {"approved": True, "approved_by": "Hanan", "manifest_sha256": c.digest(self.path),
                              "command": c.launch_command(self.path, receipt), "cap_usd": 20,
                              "approved_at": "2026-09-12T02:00:00Z"})
        c.authorize(self.path, receipt)
        data = c.read_json(self.path); data["run_id"] = "changed-run"; c.write_json(self.path, data)
        with self.assertRaises(ValueError): c.authorize(self.path, receipt)


class ExitReceiptTests(unittest.TestCase):
    def test_worker_closure_receipt_is_preserved_byte_for_byte(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            worker = {"model_closed": True, "exit_status": "ok", "stage": "stopped", "completed": 41}
            c.write_json(directory / "EXIT.json", worker)
            before = (directory / "EXIT.json").read_bytes()
            c.record_supervisor_exit(directory, 0, "a" * 64)
            self.assertEqual((directory / "EXIT.json").read_bytes(), before)
            receipt = c.read_json(directory / "SUPERVISOR-EXIT.json")
            self.assertEqual(receipt["returncode"], 0)
            self.assertEqual(receipt["writer"], "pod_supervisor")
            self.assertNotIn("worker_exit_missing", receipt)

    def test_failed_worker_without_exit_gets_fallback_and_supervisor_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            c.record_supervisor_exit(directory, 2, "b" * 64, error_type="RuntimeError")
            fallback = c.read_json(directory / "EXIT.json")
            self.assertTrue(fallback["worker_exit_missing"])
            self.assertEqual(fallback["returncode"], 2)
            self.assertEqual(fallback["error_type"], "RuntimeError")
            self.assertEqual(c.read_json(directory / "SUPERVISOR-EXIT.json")["manifest_sha256"], "b" * 64)
            self.assertNotIn("model_closed", fallback)

    def test_supervisor_failure_never_replaces_worker_failure_details(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            worker = {"model_closed": False, "exit_status": "error", "stage": "failed", "close_error": "mock failure"}
            c.write_json(directory / "EXIT.json", worker)
            c.record_supervisor_exit(directory, 2, "c" * 64, error_type="OSError")
            self.assertEqual(c.read_json(directory / "EXIT.json"), worker)

    def test_wrapper_startup_failure_emits_a_fallback_exit_for_watchdog(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest_path = directory / "manifest.json"
            c.write_json(manifest_path, {"run_id": "mock-run"})
            real_path = Path
            def mapped_path(*parts):
                if parts == ("/workspace/results/agent-steering",):
                    return directory / "results"
                return real_path(*parts)
            with patch.object(c, "Path", side_effect=mapped_path), \
                    patch.object(c, "pod_run", side_effect=RuntimeError("mock startup failure")), \
                    patch.object(c, "Provider", side_effect=AssertionError("No provider calls")), \
                    patch("sys.argv", ["control.py", "pod-run", "--manifest", str(manifest_path),
                                       "--created-at", "2026-09-12T02:00:00Z"]), \
                    contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(c.main(), 2)
            result = directory / "results/mock-run"
            self.assertEqual(c.read_json(result / "FAILED.json")["stage"], "pod_wrapper")
            self.assertEqual(c.read_json(result / "SUPERVISOR-EXIT.json")["returncode"], 2)
            self.assertTrue(c.read_json(result / "EXIT.json")["worker_exit_missing"])


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.start = c.timestamp("2026-09-12T00:00:00Z")
        self.wall = self.start; self.mono = 0
        c.write_json(self.directory / "pod.json", {"id": "mockpod", "created_at": "2026-09-12T00:00:00Z",
                     "remote_results": "/workspace/results/agent-steering/mock", "rate_usd_h": 3.49})
        self.provider = FakeProvider()
        self.command = Mock(return_value=subprocess.CompletedProcess([], 0))
        self.watch = c.Watchdog(self.directory, self.provider, clock=lambda: self.wall,
                               monotonic=lambda: self.mono, command=self.command, sleep=lambda _: None)

    def advance(self, seconds):
        self.wall += seconds; self.mono += seconds

    def connect(self):
        c.write_json(self.directory / "connection.json", {"host": "203.0.113.1", "port": 2222, "key": "/tmp/mock-key"})

    def test_hard_deadline_applies_without_ssh(self):
        self.advance(8400); self.provider.get_status = 404
        self.watch.tick()
        self.assertTrue(self.watch.terminated)
        self.assertEqual(self.watch.reason, "hard_deadline")
        self.command.assert_not_called()

    def test_delete_requires_404_and_retries_even_when_exited(self):
        self.advance(8400)
        self.watch.tick()
        self.assertFalse(self.watch.terminated)
        self.provider.get_status = 404
        self.watch.tick()
        self.assertTrue(self.watch.terminated)
        self.assertEqual([m for m, _ in self.provider.calls].count("DELETE"), 2)

    def test_inventory_failure_after_404_keeps_supervising(self):
        self.advance(8400); self.provider.get_status = 404
        self.provider.pods = Mock(side_effect=RuntimeError("network error"))
        self.watch.tick()
        self.assertFalse(self.watch.terminated)

    def test_other_pods_are_reported_and_never_deleted(self):
        self.advance(8400); self.provider.get_status = 404
        self.provider.inventory = [{"id": "unrelated"}]
        self.watch.tick()
        receipt = c.read_json(self.directory / "termination.json")
        self.assertFalse(receipt["zero_pods_verified"])
        self.assertNotIn(("DELETE", "/pods/unrelated"), self.provider.calls)

    def test_fresh_liveness_cannot_hide_stale_progress(self):
        self.connect(); self.advance(601)
        c.write_json(self.watch.results / "heartbeat.json", {"unix": self.wall, "progress_unix": self.start})
        self.provider.get_status = 404; self.watch.tick()
        self.assertEqual(self.watch.reason, "stale_heartbeat")

    def test_missing_progress_expires_after_twenty_minutes(self):
        self.advance(1201); self.provider.get_status = 404
        self.watch.tick()
        self.assertEqual(self.watch.reason, "missing_heartbeat")

    def test_stop_syncs_before_delete(self):
        self.connect(); (self.directory / "STOP").touch()
        self.provider.get_status = 404
        self.watch.tick()
        self.assertTrue(self.watch.final_synced)
        self.assertEqual(self.watch.reason, "STOP")
        args, kwargs = self.command.call_args
        self.assertIn("-rltz", args[0]); self.assertNotIn("-az", args[0])
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)

    def test_sync_cannot_extend_deadline(self):
        self.connect(); self.advance(8398); self.provider.get_status = 404
        def blocked(*args, **kwargs):
            self.assertEqual(kwargs["timeout"], 2)
            self.advance(2)
            raise subprocess.TimeoutExpired("rsync", 2)
        self.command.side_effect = blocked
        self.watch.tick()
        self.assertTrue(self.watch.terminated)

    def test_backward_wall_clock_does_not_extend_deadline(self):
        self.mono = 8400; self.wall = self.start - 3600; self.provider.get_status = 404
        self.watch.tick()
        self.assertTrue(self.watch.terminated)

    def test_shortened_window_is_anchored_to_creation(self):
        data = c.read_json(self.directory / "pod.json"); data["planned_hours"] = 2
        c.write_json(self.directory / "pod.json", data)
        self.watch = c.Watchdog(self.directory, self.provider, clock=lambda: self.wall,
                               monotonic=lambda: self.mono, command=self.command)
        self.advance(8400); self.provider.get_status = 404
        self.watch.tick()
        self.assertTrue(self.watch.terminated)


if __name__ == "__main__":
    unittest.main()
