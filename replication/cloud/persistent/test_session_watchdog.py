"""Exercise my session-lifetime decisions without SSH or provider calls."""

import argparse
import contextlib
import io
import json
import os
import shlex
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from session_watchdog import SessionWatchdog


class Clock:
    def __init__(self):
        self.wall = datetime(2026, 9, 11, 3, 0, tzinfo=timezone.utc)
        self.elapsed = 0.0

    def advance(self, seconds):
        self.wall += timedelta(seconds=seconds)
        self.elapsed += seconds


class FakeApi:
    def __init__(self):
        self.deleted = []
        self.results = [{"terminated": True}]

    def delete_pod(self, pod_id, tries=1):
        self.deleted.append(pod_id)
        return self.results.pop(0)

    def get_pod(self, pod_id):
        return {"desiredStatus": "RUNNING"}


class SessionLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.clock = Clock()
        self.api = FakeApi()
        self.command = Mock(return_value=subprocess.CompletedProcess([], 0))
        self.args = argparse.Namespace(session_dir=self.temp.name, pod_id="testpod123",
                                       created_at=self.clock.wall.isoformat(), hours=3,
                                       config=None, interval=30, sync_timeout=20)
        self.supervisor = SessionWatchdog(self.args, api=self.api, now=lambda: self.clock.wall,
                                         monotonic=lambda: self.clock.elapsed, command=self.command)
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)

    def connect(self):
        self.supervisor.config.write_text(json.dumps({"ssh_host": "203.0.113.10", "ssh_port": 22022}))

    def marker(self, relative, text="", *, age=0):
        path = self.supervisor.outputs / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        stamp = self.clock.wall.timestamp() - age
        os.utime(path, (stamp, stamp))
        return path

    def status(self):
        return json.loads((Path(self.temp.name) / "session_status.json").read_text())

    def test_deadline_applies_before_ssh_is_ready(self):
        self.supervisor.tick()
        self.assertEqual(self.status()["state"], "waiting_for_ssh")
        self.clock.advance(3 * 3600)
        self.supervisor.tick()
        self.assertEqual(self.api.deleted, ["testpod123"])
        self.assertEqual(self.status()["termination_reason"], "hard_deadline")
        self.command.assert_not_called()

    def test_per_job_exit_done_and_failure_retain_the_pod(self):
        self.connect()
        self.marker("first/DONE")
        self.marker("first/EXIT", "0")
        self.marker("second/EXIT", "9")
        self.marker("second/FAILED", "private contents must not appear")
        self.marker("third/QUEUE_DONE")
        self.supervisor.tick()
        self.assertEqual(self.api.deleted, [])
        self.assertEqual(self.status()["state"], "needs_attention")
        self.assertEqual(self.status()["failure_markers"], ["second/EXIT", "second/FAILED"])
        self.assertNotIn("private contents", (Path(self.temp.name) / "session_watchdog.log").read_text())

    def test_fresh_root_queue_done_deletes_only_the_pod(self):
        self.connect()
        self.marker("QUEUE_DONE")
        self.supervisor.tick()
        self.assertTrue(self.supervisor.terminated)
        self.assertEqual(self.api.deleted, ["testpod123"])
        self.assertEqual(self.status()["termination_reason"], "QUEUE_DONE")

    def test_old_root_queue_done_is_ignored(self):
        self.connect()
        self.marker("QUEUE_DONE", age=60)
        self.supervisor.tick()
        self.assertEqual(self.api.deleted, [])

    def test_stop_session_does_not_require_working_ssh(self):
        (Path(self.temp.name) / "STOP_SESSION").touch()
        self.supervisor.tick()
        self.assertEqual(self.api.deleted, ["testpod123"])
        self.assertEqual(self.status()["termination_reason"], "STOP_SESSION")

    def test_old_remote_stop_marker_cannot_end_a_new_session(self):
        self.connect()
        self.marker("STOP_SESSION", age=60)
        self.supervisor.tick()
        self.assertEqual(self.api.deleted, [])

    def test_connection_updates_do_not_reset_creation_deadline(self):
        self.clock.advance(100)
        self.connect()
        self.supervisor.config.write_text(json.dumps({"ssh_host": "203.0.113.20", "ssh_port": 22,
                                                      "created_at": "2099-01-01T00:00:00Z", "hours": 99}))
        self.supervisor.tick()
        self.assertEqual(self.supervisor.remaining(), 10700)
        self.assertIn("root@203.0.113.20:/workspace/results/", self.command.call_args.args[0])
        self.assertNotIn("--delete", self.command.call_args.args[0])
        cmd = self.command.call_args.args[0]
        ssh = shlex.split(cmd[cmd.index("-e") + 1])
        self.assertNotIn("-n", ssh, "rsync must retain SSH stdin for its protocol")
        self.assertEqual(self.command.call_args.kwargs["stdin"], subprocess.DEVNULL)

    def test_backward_wall_clock_does_not_extend_lifetime(self):
        self.clock.advance(3 * 3600)
        self.clock.wall -= timedelta(hours=2)
        self.supervisor.tick()
        self.assertEqual(self.api.deleted, ["testpod123"])

    def test_failed_sync_does_not_terminate_job_or_leak_output(self):
        self.connect()
        self.command.side_effect = subprocess.TimeoutExpired("secret-command", 20, stderr="secret-value")
        self.supervisor.tick()
        self.assertEqual(self.api.deleted, [])
        self.assertEqual(self.status()["state"], "needs_attention")
        self.assertNotIn("secret", (Path(self.temp.name) / "session_watchdog.log").read_text())

    def test_transfer_budget_cannot_pass_the_deadline(self):
        self.connect()
        self.clock.advance(3 * 3600 - 2)

        def timeout_at_deadline(*args, **kwargs):
            self.assertEqual(kwargs["timeout"], 2)
            self.clock.advance(2)
            raise subprocess.TimeoutExpired("rsync", 2)

        self.command.side_effect = timeout_at_deadline
        self.supervisor.tick()
        self.assertEqual(self.api.deleted, ["testpod123"])

    def test_unverified_deletion_is_retried_not_marked_complete(self):
        self.api.results = [{"terminated": False}, {"terminated": True}]
        self.clock.advance(3 * 3600)
        self.supervisor.tick()
        self.assertFalse(self.supervisor.terminated)
        self.assertEqual(self.status()["state"], "terminating")
        self.supervisor.tick()
        self.assertEqual(self.api.deleted, ["testpod123", "testpod123"])
        self.assertTrue(self.supervisor.terminated)

    def test_already_deleted_pod_is_verified_by_404(self):
        self.api.results = [{"terminated": False}]
        self.api.get_pod = Mock(side_effect=RuntimeError("HTTP 404"))
        self.clock.advance(3 * 3600)
        self.supervisor.tick()
        self.assertTrue(self.supervisor.terminated)

    def test_local_disk_failure_cannot_prevent_deadline_termination(self):
        self.clock.advance(3 * 3600)
        with patch.object(Path, "write_text", side_effect=OSError("disk full")), \
                patch.object(Path, "open", side_effect=OSError("disk full")):
            self.supervisor.tick()
        self.assertEqual(self.api.deleted, ["testpod123"])
        self.assertTrue(self.supervisor.terminated)

    def test_invalid_config_retains_pod_until_deadline(self):
        self.supervisor.config.write_text('{"ssh_host": "host; malicious"}')
        self.supervisor.tick()
        self.assertEqual(self.status()["state"], "needs_attention")
        self.command.assert_not_called()
        self.clock.advance(3 * 3600)
        self.supervisor.tick()
        self.assertTrue(self.supervisor.terminated)


if __name__ == "__main__":
    unittest.main()
