"""I dispatch the frozen GPU-B items in order and preserve every unrun item."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
import time


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def main(packet):
    packet = Path(packet).resolve()
    config_path = packet / 'local-config.json'
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    runner = Path(config['source_root']) / 'replication/cloud/parallel_h100/queue_runner.py'
    state = {'started_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'pid': os.getpid(), 'pod_id': config['expected_pod_id'], 'items': [],
             'local_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
             'status': 'running', 'stop_launch_unix': config['stop_launch_unix']}
    save(packet / 'state/sequence.json', state)
    for item in (1, 2, 3):
        if time.time() >= config['stop_launch_unix']:
            state['status'] = 'launch_cutoff'
            break
        if config_path.read_bytes() != config_bytes:
            raise ValueError('The bound execution configuration changed')
        command = [sys.executable, '-B', str(runner), 'run', '--config', str(config_path), '--item', str(item)]
        row = {'item': item, 'argv': command, 'started_unix': time.time()}
        state['items'].append(row)
        log_path = packet / 'state' / f'item-{item}.log'
        with log_path.open('xb') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT)
            row['pid'] = process.pid
            save(packet / 'state/sequence.json', state)
            returncode = process.wait()
        finished = Path(config['local_out_dir']) / f'item-{item}' / 'FINISHED.json'
        receipt = json.loads(finished.read_bytes()) if finished.exists() else {'status': 'missing_finish_receipt'}
        row.update(returncode=returncode, finished_unix=time.time(), status=receipt['status'],
                   recorded=receipt.get('recorded', 0))
        save(packet / 'state/sequence.json', state)
        if returncode != 0 or receipt['status'] not in ('completed', 'gate_failed'):
            state['status'] = receipt['status']
            break
    else:
        state['status'] = 'all_items_closed'
    state['unstarted_items'] = [item for item in (1, 2, 3) if item not in {r['item'] for r in state['items']}]
    state['finished_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state['allocation_action'] = 'none'
    save(packet / 'state/sequence.json', state)
    return 0 if state['status'] != 'failed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--packet', type=Path, required=True)
    raise SystemExit(main(parser.parse_args().packet))
