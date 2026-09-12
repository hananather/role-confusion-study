"""I close the finished inference job, verify its backup, and audit saved outcomes."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
import time

from .closeout import generate
from .mirror_loop import once
from .standby_supervisor import remote_identity_check
from ..agent_steering.control import connection_command
from ..agent_steering.local_backend import RemoteBackend


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finalize(packet):
    packet = Path(packet).resolve()
    config = json.loads((packet / 'local-config.json').read_bytes())
    sequence = json.loads((packet / 'state/sequence.json').read_bytes())
    if not sequence.get('finished_at') or sequence['status'] == 'running':
        raise ValueError('I must let the fixed sequence finish before closing its job')
    if config['expected_pod_id'] == 'nz1bypfsiv62sc':
        raise ValueError('The original pod is not mine to close')
    for item in sequence['items']:
        if not (Path(config['local_out_dir']) / f"item-{item['item']}" / 'FINISHED.json').exists():
            raise ValueError('A launched item has no final accounting')
    connection = json.loads(Path(config['connection_file']).read_bytes())
    if connection['pod_id'] != config['expected_pod_id']:
        raise ValueError('Wrong owned connection')
    plan = json.loads(Path(config['plan']).read_bytes())
    service_plan = json.loads((packet / 'plan.json').read_bytes())
    supervisor = json.loads((packet / 'state/supervisor-config.json').read_bytes())
    pod = json.loads((packet / 'state/pod.json').read_bytes())
    if {config['expected_pod_id'], supervisor['pod_id'], pod['id'], sequence['pod_id']} != {pod['id']}:
        raise ValueError('My pod receipts must all bind the same GPU B')
    if {plan['run_id'], service_plan['run_id'], supervisor['run_id']} != {plan['run_id']}:
        raise ValueError('My plans must all bind the same owned queue')
    if config['remote_out_dir'] != '/workspace/results/agent-steering/' + plan['run_id'] + '-queue':
        raise ValueError('I can close only the exact owned GPU B job directory')
    if supervisor['remote_root'] != '/workspace/parallel-lanes/' + plan['run_id']:
        raise ValueError('I require the exact owned GPU B service directory')
    if sha(config['plan']) != config['plan_sha256'] or sha(packet / 'local-config.json') != sequence['local_config_sha256']:
        raise ValueError('My registered plan and configuration must remain unchanged')
    def remote_python(script):
        return subprocess.run(connection_command(connection) + ['root@' + connection['host'],
            shlex.join(['/workspace/venv-probes/bin/python', '-B', '-c', script])],
            capture_output=True, check=True, timeout=30)
    remote_python(remote_identity_check(supervisor))
    backend = RemoteBackend(connection, config['remote_out_dir'], packet / 'final-rpc', plan['engine_arms'])
    backend.finish({'sequence_status': sequence['status'], 'unstarted_items': sequence['unstarted_items'],
                    'allocation_action': 'none', 'pod_retained_by_supervisor': True})
    for _ in range(30):
        done = backend.read_remote('DONE.json')
        if done:
            break
        if backend.read_remote('FAILED.json'):
            raise RuntimeError('The inference job ended with an infrastructure error')
        time.sleep(1)
    else:
        raise RuntimeError('The inference job has not acknowledged its close-out')
    copied = once(packet)
    if not copied['success']:
        raise RuntimeError('The final backup did not succeed')
    script = remote_identity_check(supervisor) + (
        'import pathlib,hashlib,json\n'
        'root=pathlib.Path(' + repr(config['remote_out_dir']) + ')\n'
        'files={}\n'
        'for path in sorted(root.rglob("*")):\n'
        ' if path.is_symlink(): raise ValueError("Unexpected artifact symlink")\n'
        ' if path.is_file(): files[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()\n'
        'source_root=pathlib.Path(' + repr(supervisor['remote_root'] + '/source') + ')\n'
        'source_files={}\n'
        'for entry in ' + repr(service_plan['files']) + ':\n'
        ' path=source_root/entry["path"]\n'
        ' if path.is_symlink(): raise ValueError("Unexpected source symlink")\n'
        ' source_files[entry["path"]]=hashlib.sha256(path.read_bytes()).hexdigest()\n'
        'print(json.dumps({"job":files,"source":source_files}))\n')
    reply = json.loads(remote_python(script).stdout)
    remote = reply['job']
    mirror = packet / 'mirror/results' / (plan['run_id'] + '-queue')
    mismatches = [name for name, expected in remote.items()
                  if not (mirror / name).is_file() or sha(mirror / name) != expected]
    source_mismatches = [entry['path'] for entry in service_plan['files']
                         if sha(packet / 'source' / entry['path']) != entry['sha256']]
    remote_source_mismatches = [entry['path'] for entry in service_plan['files']
                               if reply['source'].get(entry['path']) != entry['sha256']]
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = packet / ('closeout-' + stamp)
    report = generate(config['plan'], packet, out)
    receipt = {'verified_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'pod_id': config['expected_pod_id'], 'sequence_status': sequence['status'],
               'job_close_receipt': done, 'remote_job_files': remote,
               'remote_files_verified': len(remote), 'backup_mismatches': mismatches,
               'local_frozen_service_source_mismatches': source_mismatches,
               'remote_frozen_service_source_mismatches': remote_source_mismatches,
               'remote_frozen_service_source_hashes': reply['source'],
               'prediction_copy_sha256': sha(plan['evidence']['predictions-2026-09-12.json']['path']),
               'closeout': str(out), 'totals': report['totals'], 'allocation_action': 'none'}
    (out / 'sync-and-close-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if mismatches or source_mismatches or remote_source_mismatches:
        raise RuntimeError('Close-out retained verification discrepancies')
    print(json.dumps({key: receipt[key] for key in ('closeout', 'remote_files_verified', 'totals', 'allocation_action')}))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--packet', type=Path, required=True)
    finalize(parser.parse_args().packet)
