#!/usr/bin/env python3
"""I inspect my persistent worker every 20 seconds without controlling its lifetime.

I distinguish a quiet log from idle compute using the worker process tree's
cumulative CPU/I/O, GPU utilization, and meaningful stage/output changes. Fresh
heartbeats alone do not establish progress. This monitor never stops a process,
deletes a pod, or changes a remote file. Its local attention flag is advisory.

Usage: python3 progress_monitor.py --session-dir cloud/persistent/sessions/SESSION
       python3 progress_monitor.py --self-test
"""

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


# This program runs through SSH's stdin. I read only session files, /proc process
# counters, and nvidia-smi; I never read environments, credentials, or full argv.
REMOTE_COLLECTOR = r'''
import datetime, hashlib, json, os, pathlib, re, stat as stat_types, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
def safe_text(text):
    text = re.sub(r"(?i)Bearer\s+\S+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)([\w-]*(?:api[_-]?key|token|secret|password)[\w-]*\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", text)
    text = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{12,}|hf_[A-Za-z0-9]{12,}|rpa_[A-Za-z0-9_-]{12,})\b", "[REDACTED]", text)
    return text[:240]
def filtered(data):
    if not isinstance(data, dict):
        return {}
    allowed = {'phase', 'status', 'event', 'stage', 'done', 'total', 'batch', 'batches',
               'tokens', 'n_completed', 'completed', 'planned', 'layer', 'layer_ix',
               'sec_per_probe', 'sec_per_batch', 'n_probes', 'n_probes_planned', 'n_items',
               'time_utc', 'updated_utc', 'output'}
    return {k: (safe_text(v) if isinstance(v, str) else v) for k, v in data.items()
            if k in allowed and isinstance(v, (str, int, float, bool, type(None)))}
def output_telemetry(output, stage_log, metadata):
    # I make a bounded number of direct stat calls, regardless of checkpoint count.
    # Root directory mtimes are excluded: heartbeat atomic writes change them.
    candidates = []
    if output:
        names = ('train.log', 'tokens.parquet', 'tokens.parquet.tmp', 'page_means.parquet',
                 'page_means.parquet.tmp', 'prompts.parquet', 'gptoss-20b.pkl',
                 'gptoss-20b-basesplit.pkl', 'probes.npz', 'probes-basesplit.npz',
                 'gardening-layer12-float16.npy', 'native-probabilities-unrounded.npy',
                 'tomato-role-projections-gptoss-20b.csv', 'curves.csv', 'span_means.csv',
                 'EXTRACTION_DONE', 'ANALYSIS_DONE', 'DONE')
        candidates.extend((name, output / name, False) for name in names)
        candidates.extend(('@directory:' + name, output / name, True) for name in ('parts', 'figures'))
        completed = metadata.get('n_completed')
        if isinstance(completed, int) and completed >= 0:
            for index in sorted({max(0, completed - 1), completed}):
                for prefix, extension in (('tokens', 'parquet'), ('page', 'parquet'), ('item', 'json')):
                    name = f'parts/{prefix}-{index:05d}.{extension}'
                    candidates.append((name, output / name, False))
                    candidates.append((name + '.tmp', output / (name + '.tmp'), False))
    if stage_log:
        candidates.append(('stage-console', stage_log, False))
    observed = []
    for name, path, directory in candidates:
        try:
            info = path.stat()
            if (directory and stat_types.S_ISDIR(info.st_mode)) or (not directory and stat_types.S_ISREG(info.st_mode)):
                observed.append((name, 0 if directory else info.st_size, info.st_mtime_ns))
        except OSError:
            pass
    return observed, len(candidates)
queue_raw = read_json(root / 'queue-status.json')
queue = filtered(queue_raw)
output = pathlib.Path(queue_raw['output']) if isinstance(queue_raw.get('output'), str) else None
if output is not None:
    try:
        output.resolve().relative_to(root.resolve())
    except (ValueError, OSError):
        output = None
stage = str(queue_raw.get('stage') or '')
stage_log = root / 'stage-logs' / (stage + '.log') if re.fullmatch(r'[A-Za-z0-9_-]+', stage) else None
worker_record = read_json(root / 'worker-pid.json')
try:
    worker_pid = int(worker_record.get('pid', 0))
except (ValueError, TypeError):
    worker_pid = 0
processes = {}
ticks = os.sysconf(os.sysconf_names['SC_CLK_TCK'])
for directory in pathlib.Path('/proc').iterdir():
    if not directory.name.isdigit():
        continue
    try:
        stat = (directory / 'stat').read_text()
        after = stat[stat.rfind(')') + 2:].split()
        processes[int(directory.name)] = {'pid': int(directory.name), 'ppid': int(after[1]),
            'state': after[0], 'start_ticks': int(after[19]),
            'cpu_seconds': sum(int(after[i]) for i in (11, 12, 13, 14)) / ticks}
    except (OSError, ValueError, IndexError):
        continue
selected = {worker_pid} if worker_pid in processes else set()
while True:
    children = {pid for pid, proc in processes.items() if proc['ppid'] in selected}
    updated = selected | children
    if updated == selected:
        break
    selected = updated
tree = []
for pid in sorted(selected):
    proc = processes[pid]
    proc['io_bytes'] = 0
    try:
        counters = dict(line.split(':', 1) for line in pathlib.Path('/proc', str(pid), 'io').read_text().splitlines())
        # rchar/wchar also detect buffered reads/writes missed by disk byte counters.
        proc['io_bytes'] = sum(int(counters.get(k, 0)) for k in ('rchar', 'wchar', 'read_bytes', 'write_bytes'))
    except (OSError, ValueError):
        pass
    tree.append(proc)
worker_alive = worker_pid in processes and processes[worker_pid]['state'] not in ('Z', 'X')
gpu = []
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
    for line in result.stdout.splitlines():
        values = [float(part.strip()) for part in line.split(',')]
        if len(values) == 3:
            gpu.append(dict(utilization_percent=values[0], memory_used_mib=values[1], memory_total_mib=values[2]))
except (OSError, ValueError, subprocess.TimeoutExpired):
    pass
progress = filtered(read_json(output / 'progress.json')) if output else {}
heartbeat = filtered(read_json(output / 'heartbeat')) if output else {}
metadata = filtered(read_json(output / 'metadata.json')) if output else {}
markers = []
if output:
    for name in ('FAILED', 'NEEDS_ATTENTION', 'EXIT'):
        path = output / name
        if path.is_file():
            if name == 'EXIT':
                try:
                    if path.read_text()[:64].strip() == '0':
                        continue
                except OSError:
                    pass
            markers.append(name)
files, output_stat_attempts = output_telemetry(output, stage_log, metadata)
fingerprint = hashlib.sha256(json.dumps(sorted(files)).encode()).hexdigest()
errors = []
if stage_log and stage_log.is_file():
    try:
        with stage_log.open('rb') as handle:
            handle.seek(max(0, stage_log.stat().st_size - 16384))
            tail = handle.read().decode(errors='replace')
        errors = [safe_text(line.strip()) for line in tail.splitlines()
                  if re.search(r'(?i)\b(traceback|[A-Za-z]*Error|exception|failed|killed|out of memory|fatal)\b', line)][-5:]
    except OSError:
        pass
snapshot = {'remote_time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'queue': queue, 'progress': progress, 'heartbeat': heartbeat, 'metadata': metadata,
            'worker_pid': worker_pid or None, 'worker_alive': worker_alive, 'processes': tree,
            'gpu': gpu, 'failure_markers': markers, 'recent_error_lines': errors,
            'output_fingerprint': fingerprint, 'output_files_observed': len(files),
            'output_bytes_observed': sum(value[1] for value in files),
            'output_stat_attempts': output_stat_attempts,
            'output_observation': 'bounded named artifacts, recent item checkpoints, and parts/figures directory metadata'}
print(json.dumps(snapshot))
'''


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def semantic_fingerprint(snapshot):
    excluded = {'time_utc', 'updated_utc'}
    data = {key: {k: v for k, v in snapshot.get(key, {}).items() if k not in excluded}
            for key in ('queue', 'progress', 'metadata')}
    data['outputs'] = snapshot.get('output_fingerprint')
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


class ProgressTracker:
    """I classify observed activity; I never equate allocated GPU memory with work."""

    def __init__(self, idle_seconds=120):
        self.idle_seconds = idle_seconds
        self.previous = None
        self.previous_time = None
        self.last_activity = None
        self.missing_stage_since = None

    def assess(self, snapshot, observed):
        previous = self.previous
        elapsed = max(0.001, observed - self.previous_time) if self.previous_time is not None else 20
        if self.last_activity is None:
            self.last_activity = observed
        current_procs = {(p['pid'], p['start_ticks']): p for p in snapshot.get('processes', [])}
        old_procs = {(p['pid'], p['start_ticks']): p for p in previous.get('processes', [])} if previous else {}
        cpu_delta = sum(max(0, p['cpu_seconds'] - old_procs[key]['cpu_seconds'])
                        for key, p in current_procs.items() if key in old_procs)
        io_delta = sum(max(0, p.get('io_bytes', 0) - old_procs[key].get('io_bytes', 0))
                       for key, p in current_procs.items() if key in old_procs)
        cpu_active = cpu_delta >= max(0.05, elapsed * 0.005)
        io_active = io_delta >= 65536
        gpu_known = bool(snapshot.get('gpu'))
        gpu_active = any(g['utilization_percent'] > 0 for g in snapshot.get('gpu', []))
        changed = previous is not None and semantic_fingerprint(snapshot) != semantic_fingerprint(previous)
        new_process = previous is not None and bool(set(current_procs) - set(old_procs))
        activity = cpu_active or io_active or gpu_active or changed or new_process
        if activity:
            self.last_activity = observed
        idle_for = max(0, observed - self.last_activity)
        queue = snapshot.get('queue', {})
        event = queue.get('event')
        worker_alive = snapshot.get('worker_alive', False)
        active_children = [p for p in current_procs.values()
                           if p['pid'] != snapshot.get('worker_pid') and p.get('state') not in ('Z', 'X')]
        stage_expected = event == 'stage_started'
        if stage_expected and worker_alive and not active_children:
            if self.missing_stage_since is None:
                self.missing_stage_since = observed
        else:
            self.missing_stage_since = None
        reasons = []
        if event == 'needs_attention' or snapshot.get('failure_markers'):
            reasons.append('stage_failure')
        if snapshot.get('worker_pid') and not worker_alive and event != 'queue_complete':
            reasons.append('queue_worker_not_alive')
        if self.missing_stage_since is not None and observed - self.missing_stage_since >= 20:
            reasons.append('stage_process_missing')
        if event in ('insufficient_remaining_time', 'session_deadline_near'):
            reasons.append(event)
        if idle_for >= self.idle_seconds and gpu_known and event != 'queue_complete':
            reasons.append('no_progress_or_cpu_gpu_activity')
        state = 'needs_attention' if reasons else ('queue_complete' if event == 'queue_complete' else (
            'active' if activity else 'observing'))
        self.previous, self.previous_time = snapshot, observed
        return {'state': state, 'reasons': reasons, 'stage': queue.get('stage'), 'queue_event': event,
                'phase': snapshot.get('progress', {}).get('phase') or snapshot.get('heartbeat', {}).get('phase'),
                'cpu_seconds_since_previous': round(cpu_delta, 4),
                'cpu_core_equivalents': round(cpu_delta / elapsed, 4), 'process_io_bytes_since_previous': io_delta,
                'gpu_activity': gpu_active, 'gpu_observed': gpu_known, 'meaningful_progress_changed': changed,
                'new_process': new_process, 'inactive_seconds': round(idle_for, 1),
                'idle_threshold_seconds': self.idle_seconds}


def collect(connection, remote_root, timeout):
    host = connection.get('ssh_host')
    if not isinstance(host, str) or not re.fullmatch(r'[A-Za-z0-9.:-]+', host):
        raise ValueError('SSH endpoint is not ready')
    port = int(connection.get('ssh_port', 22))
    if not 1 <= port <= 65535:
        raise ValueError('Invalid SSH port')
    key = str(Path(connection.get('ssh_key', '~/.ssh/id_ed25519')).expanduser())
    # remote_root contains only a validated session identifier, never shell syntax.
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no', '-o',
               'UserKnownHostsFile=/dev/null', '-o', 'ConnectTimeout=5', '-i', key,
               '-p', str(port), 'root@' + host, 'python3 - ' + remote_root]
    result = subprocess.run(command, input=REMOTE_COLLECTOR, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError('SSH snapshot unavailable')
    return json.loads(result.stdout)


def session_finished(directory):
    try:
        status = json.loads((directory / 'session_status.json').read_text())
        if status.get('state') == 'terminated':
            return 'session_terminated'
        deadline = status.get('deadline_utc')
        if deadline and utc_now() >= dt.datetime.fromisoformat(deadline.replace('Z', '+00:00')):
            return 'session_deadline_passed'
    except (OSError, ValueError):
        pass
    return None


def run(args):
    directory = Path(args.session_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    session_id = args.session_id or directory.name
    if not re.fullmatch(r'[A-Za-z0-9_-]+', session_id):
        raise ValueError('Invalid session identifier')
    connection_path = Path(args.config).expanduser() if args.config else directory / 'connection.json'
    lock = (directory / 'progress_monitor.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('A progress monitor already owns this session')
    tracker = ProgressTracker(args.idle_seconds)
    next_due = time.monotonic()
    unavailable_since = None
    first_attention = None
    try:
        while True:
            finished = session_finished(directory)
            if finished:
                atomic_json(directory / 'latest-monitor.json', {'time_utc': utc_now().isoformat(), 'state': finished})
                return 0
            try:
                connection = json.loads(connection_path.read_text())
                snapshot = collect(connection, '/workspace/results/' + session_id, args.ssh_timeout)
                unavailable_since = None
                assessment = tracker.assess(snapshot, time.monotonic())
                record = {'time_utc': utc_now().isoformat(), **assessment, 'snapshot': snapshot}
            except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
                if unavailable_since is None:
                    unavailable_since = time.monotonic()
                # An inaccessible pod is unknown, never evidence of idle compute.
                unavailable = time.monotonic() - unavailable_since
                record = {'time_utc': utc_now().isoformat(), 'state': 'needs_attention' if unavailable >= 60 else 'connection_unavailable',
                          'reasons': ['monitor_connection_unavailable'] if unavailable >= 60 else [],
                          'unavailable_seconds': round(unavailable, 1), 'error_type': type(error).__name__}
            with (directory / 'monitoring.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            atomic_json(directory / 'latest-monitor.json', record)
            if record.get('state') == 'needs_attention':
                if first_attention is None:
                    first_attention = record['time_utc']
                atomic_json(directory / 'NEEDS_ATTENTION.json', {**record, 'first_observed_utc': first_attention,
                                                              'action_policy': 'advisory only; monitor never stops or deletes'})
            elif record.get('state') != 'connection_unavailable':
                first_attention = None
                (directory / 'NEEDS_ATTENTION.json').unlink(missing_ok=True)
            if args.once:
                return 0
            next_due += args.interval
            time.sleep(max(0, next_due - time.monotonic()))
    finally:
        lock.close()


def self_test():
    import ast
    import copy
    import stat
    from types import SimpleNamespace
    base = {'queue': {'event': 'stage_started', 'stage': 'fit', 'time_utc': 'first'},
            'progress': {'phase': 'fit', 'done': 1}, 'metadata': {}, 'heartbeat': {'time_utc': 'first'},
            'worker_pid': 1, 'worker_alive': True,
            'processes': [{'pid': 1, 'start_ticks': 1, 'cpu_seconds': 0, 'io_bytes': 0, 'state': 'S'},
                          {'pid': 2, 'start_ticks': 2, 'cpu_seconds': 0, 'io_bytes': 0, 'state': 'S'}],
            'gpu': [{'utilization_percent': 0, 'memory_used_mib': 12000}],
            'output_fingerprint': 'same', 'failure_markers': []}
    tracker = ProgressTracker()
    tracker.assess(copy.deepcopy(base), 0)
    for t in range(20, 121, 20):
        item = copy.deepcopy(base)
        item['heartbeat']['time_utc'] = str(t)
        item['queue']['time_utc'] = str(t)
        result = tracker.assess(item, t)
    assert result['reasons'] == ['no_progress_or_cpu_gpu_activity'], result
    for kind in ('cpu', 'io', 'gpu', 'progress', 'output'):
        tracker = ProgressTracker()
        tracker.assess(copy.deepcopy(base), 0)
        for t in range(20, 201, 20):
            item = copy.deepcopy(base)
            if kind == 'cpu': item['processes'][1]['cpu_seconds'] = t
            if kind == 'io': item['processes'][1]['io_bytes'] = t * 65536
            if kind == 'gpu': item['gpu'][0]['utilization_percent'] = 100
            if kind == 'progress': item['progress']['done'] = t
            if kind == 'output': item['output_fingerprint'] = str(t)
            result = tracker.assess(item, t)
            assert not result['reasons'], (kind, result)
    dead = copy.deepcopy(base)
    dead['worker_alive'] = False
    assert 'queue_worker_not_alive' in ProgressTracker().assess(dead, 0)['reasons']
    failed = copy.deepcopy(base)
    failed['failure_markers'] = ['FAILED']
    assert 'stage_failure' in ProgressTracker().assess(failed, 0)['reasons']
    done = copy.deepcopy(dead)
    done['queue']['event'] = 'queue_complete'
    assert ProgressTracker().assess(done, 0)['state'] == 'queue_complete'
    unknown = copy.deepcopy(base)
    unknown['gpu'] = []
    tracker = ProgressTracker()
    tracker.assess(unknown, 0)
    assert 'no_progress_or_cpu_gpu_activity' not in tracker.assess(unknown, 200)['reasons']
    stale_error = copy.deepcopy(base)
    stale_error['recent_error_lines'] = ['Old CUDA OutOfMemoryError from the preceding attempt']
    assert 'stage_failure' not in ProgressTracker().assess(stale_error, 0)['reasons']
    # Exercise the actual remote helper using fake paths: no file creation or GPU.
    node = next(node for node in ast.parse(REMOTE_COLLECTOR).body
                if isinstance(node, ast.FunctionDef) and node.name == 'output_telemetry')
    namespace = {'stat_types': stat}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<telemetry-helper>', 'exec'), namespace)
    class FakePath:
        calls = 0
        def __init__(self, value): self.value = value
        def __truediv__(self, value): return FakePath(self.value + '/' + value)
        def stat(self):
            FakePath.calls += 1
            directory = self.value.endswith(('/parts', '/figures'))
            return SimpleNamespace(st_mode=stat.S_IFDIR if directory else stat.S_IFREG,
                                   st_size=1024, st_mtime_ns=123)
        def rglob(self, *_): raise AssertionError('Recursive file scans are forbidden')
    for completed in (0, 1, 1200, 1000000):
        FakePath.calls = 0
        observed, attempts = namespace['output_telemetry'](FakePath('/stage'), FakePath('/stage.log'), {'n_completed': completed})
        assert FakePath.calls == attempts <= 33, (completed, attempts)
        assert not any('heartbeat' in name or 'metadata' in name for name, _, _ in observed)
    # Parse the remote program without executing it or touching /proc/SSH/GPU.
    compile(REMOTE_COLLECTOR, '<remote-collector>', 'exec')
    print('Passed: idle120s, CPU/I/O/GPU work, progress/output changes, process/stage failure, completion, unknown GPU, stale error text, at most33 artifact stats regardless of checkpoint count.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--session-dir')
    parser.add_argument('--session-id')
    parser.add_argument('--config')
    parser.add_argument('--interval', type=float, default=20)
    parser.add_argument('--ssh-timeout', type=float, default=10)
    parser.add_argument('--idle-seconds', type=float, default=120)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.session_dir:
        parser.error('--session-dir is required')
    if not 1 <= args.interval <= 20 or not 1 <= args.ssh_timeout < args.interval or args.idle_seconds < 120:
        parser.error('Use interval1..20s, SSH timeout shorter than interval, and idle threshold at least120s')
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
