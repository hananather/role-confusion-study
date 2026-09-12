#!/workspace/venv-probes/bin/python
"""I authorize artifact reads from this exact provider-bound queue container."""
import argparse
import json
import os
from pathlib import Path
import stat


def verify(config, args):
    pod = config["expected_pod_id"]
    run = config["run_id"]
    root = "/workspace/parallel-lanes/" + run
    if pod == "nz1bypfsiv62sc" or config["isolated_root"] != root or config["queue_job_id"] != run + "-queue":
        raise ValueError("Wrong owned lane")
    actual = os.environ.get("RUNPOD_POD_ID")
    if actual is not None and actual != pod:
        raise ValueError("Pod environment mismatch")
    marker = "/tmp/mats-prime-" + run + "-identity.json"
    if config["identity_marker_path"] != marker or config["identity_source"] != "provider_api_ssh_binding":
        raise ValueError("Wrong identity marker")
    descriptor = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor) as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600
                or info.st_nlink != 1 or not 0 < info.st_size <= 16384):
            raise ValueError("Invalid marker ownership, type or permissions")
        data = json.load(stream)
    expected = {"schema_version": 1, "identity_source": "provider_api_ssh_binding", "expected_pod_id": pod,
                "run_id": run, "identity_nonce": config["identity_nonce"]}
    if data != expected:
        raise ValueError("Marker binding mismatch")
    allowed = {root, "/workspace/results/agent-steering/" + config["queue_job_id"]}
    if ("--server" not in args or "--sender" not in args or not args or args[-1].rstrip("/") not in allowed
            or Path(args[-1]).resolve() != Path(args[-1])):
        raise ValueError("I permit only reads from this lane's two artifact roots")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    options, args = parser.parse_known_args()
    config = json.loads(Path(options.config).read_bytes())
    if options.config != config["isolated_root"] + "/worker-config.json":
        raise ValueError("Wrong frozen queue configuration")
    verify(config, args)
    os.execvp("rsync", ["rsync"] + args)


if __name__ == "__main__":
    main()
