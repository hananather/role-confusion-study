"""I verify the additive backup repair without a remote call."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch
from replication.cloud.parallel_h100 import rsync_guard as guard


class GuardTests(unittest.TestCase):
    def setUp(self):
        run = "mirror-test-" + uuid.uuid4().hex
        self.root = "/workspace/parallel-lanes/" + run
        self.path = Path("/tmp/mats-prime-" + run + "-identity.json")
        self.config = {"expected_pod_id": "newpod", "run_id": run, "isolated_root": self.root,
                       "queue_job_id": run + "-queue", "identity_marker_path": str(self.path),
                       "identity_source": "provider_api_ssh_binding", "identity_nonce": "a" * 64}
        self.marker = {"schema_version": 1, "identity_source": "provider_api_ssh_binding", "expected_pod_id": "newpod",
                       "run_id": run, "identity_nonce": "a" * 64}
        with self.path.open("x") as file: json.dump(self.marker, file)
        self.path.chmod(0o600)
        self.addCleanup(self.path.unlink)
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start(); self.addCleanup(self.env.stop)
        self.args = ["--server", "--sender", "-logDtprze.iLsfxCIvu", ".", self.root + "/"]

    def test_absent_env_owned_marker_allows_read_only_source(self):
        guard.verify(self.config, self.args)
        args = self.args[:-1] + ["/workspace/results/agent-steering/" + self.config["queue_job_id"] + "/"]
        guard.verify(self.config, args)

    def test_receiver_mode_and_foreign_source_are_rejected(self):
        for args in ([x for x in self.args if x != "--sender"], self.args[:-1] + ["/workspace/"]):
            with self.assertRaises(ValueError): guard.verify(self.config, args)

    def test_foreign_env_and_bad_marker_are_rejected(self):
        with patch.dict(os.environ, {"RUNPOD_POD_ID": "nz1bypfsiv62sc"}):
            with self.assertRaises(ValueError): guard.verify(self.config, self.args)
        self.path.write_text(json.dumps({**self.marker, "identity_nonce": "b" * 64}))
        with self.assertRaises(ValueError): guard.verify(self.config, self.args)

    def test_nonprivate_marker_rejected(self):
        self.path.chmod(0o644)
        with self.assertRaises(ValueError): guard.verify(self.config, self.args)


if __name__ == "__main__": unittest.main()
