"""I provision and inspect one bounded H100 session without logging credentials."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from runpod_api import Api, cost_log_append


def credentials():
    cfg = tomllib.loads((Path.home() / '.runpod/config.toml').read_text())
    return cfg.get('apikey') or cfg.get('api_key')


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['volume', 'launch', 'status', 'ready'])
    p.add_argument('--session-dir', type=Path, required=True)
    args = p.parse_args()
    session = args.session_dir.resolve()
    session.mkdir(parents=True, exist_ok=True)
    api = Api(api_key=credentials())
    if args.action == 'volume':
        if (session / 'volume.json').exists():
            print((session / 'volume.json').read_text())
            return
        result = api.call('POST', '/networkvolumes', {
            'name': 'mats-replication-2tb-' + session.name,
            'size': 2000, 'dataCenterId': 'EU-FR-1',
        }, retries=1)['data']
        save(session / 'volume.json', result)
        print(json.dumps(result))
    elif args.action == 'launch':
        if (session / 'pod.json').exists():
            raise SystemExit('I already recorded a pod for this session; inspect it before any retry.')
        if (session / 'launch-intent.json').exists():
            raise SystemExit('I must reconcile the previous creation request against live inventory before retrying.')
        if not (HERE / 'session_watchdog.py').exists():
            raise SystemExit('I require the session supervisor before provisioning.')
        balance = api.balance()
        if not balance or float(balance.get('clientBalance') or 0) < 12:
            raise SystemExit('I require at least $12 available before provisioning this session.')
        volume = json.loads((session / 'volume.json').read_text())
        spec = {'name': 'mats-persistent-' + session.name,
                'cloudType': 'SECURE', 'gpuTypeIds': ['NVIDIA H100 80GB HBM3'], 'gpuCount': 1,
                'dataCenterIds': [volume['dataCenterId']], 'networkVolumeId': volume['id'],
                'imageName': 'runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04',
                'containerDiskInGb': 30, 'volumeInGb': 0, 'volumeMountPath': '/workspace',
                'ports': ['22/tcp'], 'supportPublicIp': True, 'interruptible': False,
                'env': {'PUBLIC_KEY': (Path.home() / '.ssh/id_ed25519.pub').read_text().strip()}}
        now = datetime.now(timezone.utc).isoformat()
        save(session / 'launch-intent.json', {'created_at': now, 'hours': 3, 'gpu_rate_ceiling': 3.49,
             'storage_size_gb': 2000, 'network_volume_id': volume['id'], 'data_center': volume['dataCenterId'],
             'balance_before': balance['clientBalance'], 'previous_pod': 'mdx2jpzha1js58',
             'authorization': 'Hanan requested 2 TB, another H100 during handover, then one H100 for sequential experiments.'})
        # I use one create attempt: an ambiguous response must be reconciled with live inventory.
        result = api.call('POST', '/pods', spec, retries=1)['data']
        brief = {k: result.get(k) for k in ['id', 'name', 'desiredStatus', 'costPerHr', 'createdAt', 'lastStartedAt', 'publicIp', 'portMappings']}
        brief.update(created_at=now, network_volume_id=volume['id'], storage_size_gb=2000, hours=3)
        save(session / 'pod.json', brief)
        if float(result.get('costPerHr') or 999) > 3.49:
            api.delete_pod(result['id'])
            raise SystemExit('The returned GPU rate exceeded my recorded ceiling; I retired the pod and retained storage.')
        save(session / 'connection.json', {})
        env = os.environ.copy()
        env['RUNPOD_API_KEY'] = api.api_key
        log = open(session / 'supervisor-console.log', 'ab', buffering=0)
        command = ['caffeinate', '-i', sys.executable, str(HERE / 'session_watchdog.py'),
                   '--session-dir', str(session), '--pod-id', result['id'], '--created-at', now,
                   '--hours', '3', '--config', str(session / 'connection.json')]
        try:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, start_new_session=True, close_fds=True)
        except OSError:
            api.delete_pod(result['id'])
            raise
        save(session / 'supervisor-process.json', {'pid': proc.pid, 'created_at': now})
        cost_log_append(HERE.parent / 'cost-log.csv', {'date_utc': now, 'run_id': session.name,
            'experiment': 'persistent-sequential-session', 'pod_id': result['id'], 'gpu': 'NVIDIA H100 SXM',
            'cloud': 'SECURE', 'rate_usd_h': result.get('costPerHr'), 'start_utc': now, 'planned_hours': 3,
            'balance_before': balance['clientBalance'], 'status': 'created; lifetime supervisor launched',
            'notes': '2 TB independent network volume retained; individual jobs do not delete GPU; hard lifetime 3h.'})
        print(json.dumps(brief, indent=2))
    else:
        meta = json.loads((session / 'pod.json').read_text())
        pod = api.get_pod(meta['id'])
        brief = {k: pod.get(k) for k in ['id', 'name', 'desiredStatus', 'costPerHr', 'publicIp', 'portMappings']}
        if args.action == 'ready' and pod.get('publicIp') and (pod.get('portMappings') or {}).get('22'):
            save(session / 'connection.json', {'ssh_host': pod['publicIp'],
                'ssh_port': int(pod['portMappings']['22']), 'ssh_key': str(Path.home() / '.ssh/id_ed25519')})
        print(json.dumps(brief, indent=2))


if __name__ == '__main__':
    main()
