"""I mirror one owned queue's artifacts without modifying its live supervision."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from replication.cloud.agent_steering.control import connection_command


def commands(packet):
    packet = Path(packet)
    config = json.loads((packet / "state/worker-config.json").read_bytes())
    connection = json.loads((packet / "state/connection.json").read_bytes())
    root, run, pod = config["isolated_root"], config["run_id"], config["expected_pod_id"]
    if (pod == "nz1bypfsiv62sc" or connection["pod_id"] != pod or root != "/workspace/parallel-lanes/" + run
            or config["queue_job_id"] != run + "-queue"):
        raise ValueError("I require the exact separately owned lane")
    guard = "/workspace/venv-probes/bin/python " + root + "/control/rsync_guard.py --config " + root + "/worker-config.json"
    base = ["rsync", "-rltz", "--partial", "--timeout=15", "--exclude=.env", "--exclude=.env.*",
            "--rsync-path", guard, "-e", shlex.join(connection_command(connection))]
    host = "root@" + connection["host"] + ":"
    mirror = packet / "mirror"
    job_dest = mirror / "results" / config["queue_job_id"]
    return pod, [(base + ["--include=/service/***", "--include=/model-service/***", "--exclude=*",
                         host + root + "/", str(mirror) + "/"], mirror),
                 (base + [host + "/workspace/results/agent-steering/" + config["queue_job_id"] + "/",
                          str(job_dest) + "/"], job_dest)]


def deadline(packet):
    plan = json.loads((Path(packet) / "plan.json").read_bytes())
    ledger = json.loads((Path(packet) / "state/supervisor/ledger.json").read_bytes())
    closeout = float(plan.get("closeout_unix", float(plan["stop_launch_unix"]) + 600))
    return min(closeout + 120, float(ledger["shutdown_at_unix"]))


def once(packet, *, command=subprocess.run):
    packet = Path(packet)
    pod, copies = commands(packet)
    operation = packet / "operations/mirror-repair"
    operation.mkdir(parents=True, exist_ok=True)
    results = []
    for argv, destination in copies:
        destination.mkdir(parents=True, exist_ok=True)
        try:
            result = command(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE, timeout=25, check=False)
            row = {"returncode": result.returncode, "stderr": result.stderr.decode(errors="replace")[-800:]}
        except (OSError, subprocess.TimeoutExpired) as error:
            row = {"returncode": None, "error_type": type(error).__name__}
        results.append({"destination": str(destination), **row})
    receipt = {"pod_id": pod, "checked_unix": time.time(), "copies": results,
               "success": all(row["returncode"] == 0 for row in results), "allocation_action": "none"}
    (operation / "latest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    with (operation / "events.jsonl").open("a") as file:
        file.write(json.dumps(receipt) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    operation = args.packet / "operations/mirror-repair"
    while time.time() < deadline(args.packet) and not (operation / "MIRROR_STOP").exists():
        started = time.monotonic()
        result = once(args.packet)
        print(json.dumps({"success": result["success"], "checked_unix": result["checked_unix"]}), flush=True)
        if args.once:
            return 0 if result["success"] else 2
        time.sleep(max(1, 60 - (time.monotonic() - started)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
