"""I run my approved experiment stages sequentially on one persistent GPU."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def gate(stage):
    if stage.get('gate') != 'probes':
        return
    meta = json.loads((Path(stage['output']) / 'metadata.json').read_text())
    failures = []
    for field in ('custom_forward_verified', 'role_counts_equal'):
        if not meta.get(field): failures.append(field)
    if meta.get('partial'): failures.append('partial run')
    if meta.get('n_probes') != 384 or meta.get('n_probes_planned') != 384:
        failures.append('expected 384 probes')
    if any(dtype in str(meta.get('expert_dtype', '')) for dtype in ('bfloat16', 'float16', 'float32')):
        failures.append('MXFP4 expert precision')
    if not meta.get('expert_dtype'): failures.append('expert precision absent')
    if failures: raise RuntimeError('Probe validation failed: ' + ', '.join(failures))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--queue', type=Path, required=True)
    ap.add_argument('--session-out', type=Path, required=True)
    ap.add_argument('--deadline', required=True)
    args = ap.parse_args()
    os.environ.update(HF_HOME='/workspace/hf/home', TOKENIZERS_PARALLELISM='false',
                      TRITON_CACHE_DIR='/workspace/.triton', PYTHONUNBUFFERED='1',
                      MPLBACKEND='Agg', UV_CACHE_DIR='/workspace/.uv-cache',
                      UV_PYTHON_INSTALL_DIR='/workspace/.uv-python')
    deadline = datetime.fromisoformat(args.deadline.replace('Z', '+00:00')).timestamp()
    args.session_out.mkdir(parents=True, exist_ok=True)
    lock = (args.session_out / 'queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    journal = args.session_out / 'queue-journal.jsonl'
    def event(kind, **fields):
        row = {'time_utc': now(), 'event': kind, **fields}
        with journal.open('a') as f: f.write(json.dumps(row) + '\n')
        save(args.session_out / 'queue-status.json', row)
        print(json.dumps(row), flush=True)
    event('worker_started', deadline=args.deadline)
    while time.time() < deadline - 60:
        manifest = json.loads(args.queue.read_text())
        stages = manifest['stages']
        completed = 0
        for stage in stages:
            output = Path(stage['output'])
            receipt = args.session_out / 'stage-receipts' / (stage['name'] + '.json')
            signature = hashlib.sha256(json.dumps(stage, sort_keys=True).encode()).hexdigest()
            if receipt.exists():
                previous = json.loads(receipt.read_text())
                if previous.get('stage_sha256') != signature:
                    raise RuntimeError('Completed stage specification changed: ' + stage['name'])
                if previous.get('status') == 'complete':
                    completed += 1
                    continue
            output.mkdir(parents=True, exist_ok=True)
            remaining = deadline - time.time() - 60
            minimum = stage.get('minimum_seconds', 60)
            if remaining < minimum:
                event('insufficient_remaining_time', stage=stage['name'], remaining_seconds=remaining)
                return 3
            hours = min(stage.get('max_hours', remaining / 3600), remaining / 3600)
            command = [part.replace('{hours}', f'{hours:.4f}') for part in stage['command']]
            event('stage_started', stage=stage['name'], command=command, output=str(output))
            started = now()
            log_path = args.session_out / 'stage-logs' / (stage['name'] + '.log')
            log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with log_path.open('ab', buffering=0) as log:
                    proc = subprocess.run(command, cwd=manifest['cwd'], stdin=subprocess.DEVNULL,
                                          stdout=log, stderr=subprocess.STDOUT, timeout=hours * 3600)
                if proc.returncode != 0: raise RuntimeError('process exit ' + str(proc.returncode))
                gate(stage)
            except Exception as error:
                save(output / 'NEEDS_ATTENTION', {'time_utc': now(), 'stage': stage['name'],
                                                 'error': str(error), 'error_type': type(error).__name__})
                event('needs_attention', stage=stage['name'], error=str(error))
                # I retain this pod, environment and outputs for repair under the session cap.
                return 2
            save(receipt, {'stage_sha256': signature, 'status': 'complete', 'started_at': started,
                           'completed_at': now(), 'output': str(output)})
            event('stage_complete', stage=stage['name'])
            completed += 1
            # I reread the manifest before selecting the next stage, allowing reviewed additions.
            break
        else:
            if completed == len(stages) and manifest.get('sealed'):
                event('queue_complete', stages=completed)
                Path('/workspace/results/QUEUE_DONE').write_text(now() + '\n')
                return 0
            event('queue_idle', stages=completed)
            time.sleep(min(30, max(1, deadline - time.time() - 60)))
    event('session_deadline_near')
    return 3


if __name__ == '__main__':
    sys.exit(main())
