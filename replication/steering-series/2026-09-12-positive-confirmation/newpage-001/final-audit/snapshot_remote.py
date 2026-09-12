"""I verify an already closed remote job against its existing local artifact mirror."""
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
SERIES = HERE.parents[1]
ROOT = SERIES.parents[2]
RUN = ROOT / "replication/cloud/outbox/agent-steering/agent-newpages-20260912T025000Z"
PROGRAM = r'''
import datetime,hashlib,json,pathlib
root=pathlib.Path('/workspace/results/agent-steering/agent-newpages-20260912T025000Z')
def read(path):
    return json.loads(path.read_bytes()) if path.exists() else None
done,terminal,client=read(root/'DONE.json'),read(root/'EXIT.json'),read(root/'CLIENT-DONE.json')
closed=bool(done and terminal and client and done.get('status')=='job_stopped' and terminal.get('status')=='job_stopped' and done.get('error') is None and terminal.get('error') is None and client.get('status')=='completed' and client.get('recorded_episodes')==40 and client.get('planned_episodes')==40)
files={}
if closed:
    for path in root.rglob('*'):
        if path.is_file() and not path.is_symlink():
            data=path.read_bytes();files[str(path.relative_to(root))]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
print(json.dumps({'snapshot_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'expected_pod_id':'nz1bypfsiv62sc','job_id':'agent-newpages-20260912T025000Z','remote_job_closed':closed,'done':done,'exit':terminal,'files':files}))
'''


def main():
    connection=json.loads((ROOT/'replication/cloud/outbox/agent-steering/agent-bridge-20260912T022500Z/connection.json').read_bytes())
    command=['ssh','-i',connection['key'],'-p',str(connection['port']),'-o','BatchMode=yes','-o','ConnectTimeout=10',
             '-o','StrictHostKeyChecking=accept-new','root@'+connection['host'],'python3 -c '+shlex.quote(PROGRAM)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=45)
    if result.returncode:
        raise RuntimeError("Read-only remote snapshot failed: "+result.stderr[:300])
    snapshot=json.loads(result.stdout)
    if not snapshot['remote_job_closed']:
        print(json.dumps({'status':'waiting_for_remote_job_closure'}));return 3
    missing,mismatches=[],[]
    for relative,record in snapshot['files'].items():
        path=RUN/'results'/relative
        if not path.exists():missing.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:mismatches.append(relative)
    if missing or mismatches:
        print(json.dumps({'status':'waiting_for_existing_mirror_sync','missing':missing,'mismatches':mismatches}));return 3
    snapshot.update(mirror_verified=True,local_mirror=str(RUN/'results'),method='Fixed read-only SSH SHA-256 snapshot; existing monitor performed artifact transfer.',remote_program=PROGRAM)
    target=HERE/'remote-mirror.json'
    with target.open('x') as stream:json.dump(snapshot,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':'closed_remote_mirror_verified','files':len(snapshot['files']),'path':str(target)}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
