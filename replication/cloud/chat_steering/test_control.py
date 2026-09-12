"""I exercise budget and termination failures with fake providers; no cloud calls."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import control as c


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
    return {"version": 1, "run_id": "mock-run", "rate_usd_h": 3.49,
            "planned_hours": 2, "grace_minutes": 20, "cap_usd": 20,
            "container_disk_gb": 30, "container_disk_usd_gb_month": 0.1,
            "storage_month_hours": 730, "existing_volume_reserve_usd": 0.90,
            "existing_volume_usd_month": 280,
            "network_volume_id": c.VOLUME, "data_center": "EU-FR-1",
            "gpu_type": c.GPU, "cloud": "SECURE", "image": "frozen-image",
            "account_check": {"query": "query { myself { clientBalance isAutoPayEnabled } }"},
            "command": [c.PYTHON, "-u", "-m", "replication.cloud.chat_steering.batch", "run",
                        "--config", "/workspace/replication/cloud/chat_steering/run-config.json"]}


class BudgetTests(unittest.TestCase):
    def test_two_hours_and_grace_include_disk_and_current_volume_quote(self):
        value = c.cost(manifest())
        self.assertEqual(value["gpu_planned_usd"], 6.98)
        self.assertAlmostEqual(value["gpu_with_grace_usd"], 8.1433)
        self.assertEqual(value["existing_volume_usd_month"], 280)
        self.assertAlmostEqual(value["existing_volume_minimum_reserve_usd"], 0.894977)
        self.assertAlmostEqual(value["maximum_total_usd"], 9.0529)

    def test_five_hour_plan_exceeds_twenty_dollar_cap_with_current_volume_quote(self):
        data = manifest(); data.update(planned_hours=5, existing_volume_reserve_usd=2.05)
        with self.assertRaisesRegex(ValueError, "aggregate cap"):
            c.cost(data)

    def test_existing_volume_needs_monthly_quote_and_sufficient_reserve(self):
        data = manifest(); data.pop("existing_volume_usd_month")
        with self.assertRaisesRegex(ValueError, "monthly quote"):
            c.cost(data)
        for monthly, reserve in [(280, 0.4), (280, 0), (float("nan"), 1), (-1, 1)]:
            data = manifest(); data.update(existing_volume_usd_month=monthly, existing_volume_reserve_usd=reserve)
            with self.assertRaises(ValueError):
                c.cost(data)

    def test_excess_rate_or_storage_fails(self):
        for key, value in [("rate_usd_h", 3.5), ("existing_volume_reserve_usd", 25), ("planned_hours", 6)]:
            data = manifest(); data[key] = value
            with self.assertRaises(ValueError):
                c.cost(data)

    def test_ephemeral_window_still_reserves_retained_volume_charges(self):
        data = manifest()
        data.update(planned_hours=2, cap_usd=20, container_disk_gb=100,
                    existing_volume_reserve_usd=0.90, asset_mode="fresh_ephemeral", network_volume_id=None)
        estimate = c.cost(data)
        self.assertAlmostEqual(estimate["maximum_total_usd"], 9.0753)
        self.assertEqual(estimate["cap_usd"], 20)

    def test_retry_cost_includes_prior_attempts_under_the_same_cap(self):
        data = manifest()
        data.update(planned_hours=2, cap_usd=20, prior_spend_usd=0.20,
                    existing_volume_reserve_usd=0.90)
        estimate = c.cost(data)
        self.assertEqual(estimate["prior_spend_usd"], 0.20)
        self.assertAlmostEqual(estimate["maximum_total_usd"], 9.2529)
        data["cap_usd"] = 9
        with self.assertRaisesRegex(ValueError, "aggregate cap"):
            c.cost(data)

    def test_batch_environment_keeps_kernels_on_the_separate_home_cache(self):
        initial = {"HF_HUB_CACHE": "/workspace/hf", "HF_HOME": "/wrong",
                   "RUNPOD_API_KEY": "dummy", "OPENAI_API_KEY": "dummy", "PATH": "/usr/bin"}
        env = c.batch_environment(initial)
        self.assertNotIn("HF_HUB_CACHE", env)
        self.assertEqual(env["HF_HOME"], "/workspace/hf/home")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")
        self.assertNotIn("RUNPOD_API_KEY", env)
        self.assertNotIn("OPENAI_API_KEY", env)
        self.assertEqual(initial["HF_HUB_CACHE"], "/workspace/hf")

    def test_preflight_rejects_autopay_existing_pods_short_balance_and_higher_rate(self):
        for field, value in [("autopay", True), ("inventory", [{"id": "other"}]),
                             ("balance", 8.5), ("quote", 3.5)]:
            provider = FakeProvider(); setattr(provider, field, value)
            with self.assertRaises(ValueError):
                c.preflight(provider, manifest())
        self.assertEqual(c.preflight(FakeProvider(), manifest())["existing_pods"], [])

    def test_null_regional_quote_fails_closed(self):
        provider = object.__new__(c.Provider)
        provider.request = Mock(return_value=(200, {"data": {"gpuTypes": [{"id": c.GPU,
            "lowestPrice": {"stockStatus": None, "uninterruptablePrice": None, "availableGpuCounts": None}}]}}))
        with self.assertRaisesRegex(ValueError, "regional"):
            provider.quote_and_volume(manifest())

    def test_priced_single_gpu_regional_quote_allows_unspecified_count_list(self):
        provider = object.__new__(c.Provider)
        provider.request = Mock(side_effect=[
            (200, {"data": {"gpuTypes": [{"id": c.GPU, "lowestPrice": {
                "stockStatus": "Low", "uninterruptablePrice": 3.49, "availableGpuCounts": None}}]}}),
            (200, {"id": c.VOLUME, "size": 2000, "dataCenterId": "EU-FR-1"})])
        self.assertEqual(provider.quote_and_volume(manifest())["rate_usd_h"], 3.49)
        query = provider.request.call_args_list[0].args[2]["query"]
        self.assertIn("gpuCount: 1", query)
        self.assertIn('dataCenterId: "EU-FR-1"', query)

    def test_missing_or_nonboolean_autopay_fails_closed(self):
        provider = object.__new__(c.Provider)
        spec = {"query": "query {}", "balance_path": ["data", "myself", "clientBalance"],
                "autopay_path": ["data", "myself", "isAutoPayEnabled"]}
        for value in (None, "false", 0):
            provider.request = Mock(return_value=(200, {"data": {"myself": {
                "clientBalance": 20, "isAutoPayEnabled": value}}}))
            with self.assertRaises(RuntimeError):
                provider.account(spec)

    def test_plan_and_dry_run_do_not_instantiate_provider(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); data = manifest(); data["files"] = []
            names = [c.REL + "/" + name for name in ["control.py", "run.sh", "protocol.py", "batch.py", "runtime.py", "run-config.json"]]
            names += ["replication/cloud/jobcommon.py", "replication/cloud/job_steering.py"]
            for i, relative in enumerate(names):
                source = directory / str(i); source.write_text("frozen bytes")
                data["files"].append({"source": str(source), "remote": relative, "sha256": c.digest(source)})
            path = directory / "manifest.json"; c.write_json(path, data)
            for mode in ("plan", "dry-run"):
                with patch.object(c, "Provider", side_effect=AssertionError("Network access forbidden")), \
                        patch("sys.argv", ["control.py", mode, "--manifest", str(path)]), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(c.main(), 0)
            Path(data["files"][0]["source"]).write_text("changed")
            with self.assertRaisesRegex(ValueError, "changed"):
                c.validate_manifest(path)

    def test_approval_binds_exact_command_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"; path.write_text('{"cap_usd":20}'); receipt = Path(tmp) / "approval.json"
            data = {"approved": True, "approved_by": "Hanan", "manifest_sha256": c.digest(path),
                    "command": c.launch_command(path, receipt), "cap_usd": 20, "approved_at": "2026-09-12T00:00:00Z"}
            c.write_json(receipt, data); c.authorize(path, receipt)
            path.write_text('{"changed":true}')
            with self.assertRaises(ValueError):
                c.authorize(path, receipt)


class WatchdogTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name)
        self.start = c.timestamp("2026-09-12T00:00:00Z")
        self.wall = self.start; self.mono = 0
        c.write_json(self.directory / "pod.json", {"id": "mockpod", "created_at": "2026-09-12T00:00:00Z",
                     "remote_results": "/workspace/results/chat-steering/mock", "rate_usd_h": 3.49})
        self.provider = FakeProvider()
        self.command = Mock(return_value=subprocess.CompletedProcess([], 0))
        self.watch = c.Watchdog(self.directory, self.provider, clock=lambda: self.wall,
                               monotonic=lambda: self.mono, command=self.command, sleep=lambda _: None)

    def advance(self, seconds):
        self.wall += seconds; self.mono += seconds

    def connect(self):
        c.write_json(self.directory / "connection.json", {"host": "203.0.113.1", "port": 2222, "key": "/tmp/mock-key"})

    def test_hard_deadline_applies_without_ssh(self):
        self.advance(19200); self.provider.get_status = 404
        self.watch.tick()
        self.assertTrue(self.watch.terminated)
        self.assertEqual(self.watch.reason, "hard_deadline")
        self.command.assert_not_called()

    def test_delete_requires_404_and_retries_even_when_exited(self):
        self.advance(19200)
        self.watch.tick()
        self.assertFalse(self.watch.terminated)
        self.provider.get_status = 404
        self.watch.tick()
        self.assertTrue(self.watch.terminated)
        self.assertEqual([m for m, _ in self.provider.calls].count("DELETE"), 2)

    def test_inventory_failure_after_404_keeps_supervising(self):
        self.advance(19200); self.provider.get_status = 404
        self.provider.pods = Mock(side_effect=RuntimeError("network error"))
        self.watch.tick()
        self.assertFalse(self.watch.terminated)

    def test_other_pods_are_reported_and_never_deleted(self):
        self.advance(19200); self.provider.get_status = 404
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
        self.connect(); self.advance(19198); self.provider.get_status = 404
        def blocked(*args, **kwargs):
            self.assertEqual(kwargs["timeout"], 2)
            self.advance(2)
            raise subprocess.TimeoutExpired("rsync", 2)
        self.command.side_effect = blocked
        self.watch.tick()
        self.assertTrue(self.watch.terminated)

    def test_backward_wall_clock_does_not_extend_deadline(self):
        self.mono = 19200; self.wall = self.start - 3600; self.provider.get_status = 404
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
