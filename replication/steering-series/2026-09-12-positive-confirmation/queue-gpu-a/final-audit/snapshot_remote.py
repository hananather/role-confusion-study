"""I hash only the closed remote queue; the parent owns artifact transfer."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
SERIES = HERE.parents[1]
PACKET = SERIES / "queue-gpu-a/launch-packet"
PROGRAM = r'''
import datetime,hashlib,json,pathlib
root=pathlib.Path('/workspace/results/agent-steering/agent-queue-a-items5-6-20260912')
def read(path):
    return json.loads(path.read_bytes()) if path.exists() else None
terminal,client=read(root/'EXIT.json'),read(root/'CLIENT-DONE.json')
closed=bool(terminal and client and terminal.get('status') in ('job_stopped','error') and client.get('status') not in (None,'starting','running') and client.get('planned_readouts')==10 and client.get('planned_episodes')==5)
files={}
if closed:
    before=(root/'EXIT.json').read_bytes(),(root/'CLIENT-DONE.json').read_bytes()
    for path in root.rglob('*'):
        if path.is_file() and not path.is_symlink():
            data=path.read_bytes();files[str(path.relative_to(root))]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    if before!=((root/'EXIT.json').read_bytes(),(root/'CLIENT-DONE.json').read_bytes()):
        raise RuntimeError('Closure receipts changed during read-only snapshot')
print(json.dumps({'snapshot_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'expected_pod_id':'nz1bypfsiv62sc','job_id':'agent-queue-a-items5-6-20260912','remote_job_closed':closed,'exit':terminal,'client':client,'files':files}))
'''


def main():
    target = HERE / "remote-mirror.json"
    if target.exists():
        raise RuntimeError("I preserve completed remote snapshots")
    config = json.loads((PACKET / "local-config.json").read_bytes())
    connection = json.loads(Path(config["connection_file"]).read_bytes())
    mirror = Path(config["local_sync_results"])
    command = ["ssh", "-i", connection["key"], "-p", str(connection["port"]), "-o", "BatchMode=yes",
               "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=accept-new",
               "root@" + connection["host"], "python3 -c " + shlex.quote(PROGRAM)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=45)
    if result.returncode:
        raise RuntimeError("Read-only remote snapshot failed: " + result.stderr[:300])
    snapshot = json.loads(result.stdout)
    if not snapshot["remote_job_closed"]:
        print(json.dumps({"status": "waiting_for_remote_job_closure"})); return 3
    missing, mismatches = [], []
    for relative, record in snapshot["files"].items():
        path = mirror / relative
        if not path.exists():
            missing.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"] or path.stat().st_size != record["bytes"]:
            mismatches.append(relative)
    if missing or mismatches:
        print(json.dumps({"status": "waiting_for_existing_mirror_sync", "missing": missing, "mismatches": mismatches})); return 3
    snapshot.update(mirror_verified=True, local_mirror=str(mirror),
                    method="Fixed read-only SSH SHA-256 snapshot; the parent and existing supervisor own transfers.",
                    remote_program=PROGRAM)
    with target.open("x") as stream:
        json.dump(snapshot, stream, indent=2); stream.write("\n")
    print(json.dumps({"status": "closed_remote_mirror_verified", "files": len(snapshot["files"]), "path": str(target)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
