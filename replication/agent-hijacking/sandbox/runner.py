"""Trusted shell launcher: bounded output, dropped UID, and descendant cleanup."""

import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import tempfile

MAX_OUTPUT_BYTES = 1024 * 1024


def evaluated_limits():
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    os.setgroups([])
    os.setgid(1000)
    os.setuid(1000)
    os.chdir('/workspace/container-1')


def kill_evaluated_processes():
    """Remove background jobs, including children that created a new session."""
    for _ in range(3):
        for path in Path('/proc').glob('[0-9]*/status'):
            try:
                status = path.read_text()
                uid_line = next(line for line in status.splitlines() if line.startswith('Uid:'))
                if int(uid_line.split()[1]) == 1000:
                    os.kill(int(path.parent.name), signal.SIGKILL)
            except (FileNotFoundError, ProcessLookupError, PermissionError, StopIteration):
                pass


def main():
    request = json.load(sys.stdin)
    command = request['command']
    seconds = request['timeout']
    if not isinstance(command, str) or len(command) > 32768:
        raise ValueError('Invalid command')
    with tempfile.TemporaryFile(dir='/run/harness') as out, tempfile.TemporaryFile(dir='/run/harness') as err:
        process = subprocess.Popen(
            ['/bin/bash', '--noprofile', '--norc', '-c', command],
            cwd='/',
            env={'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/workspace/container-1',
                 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8'},
            stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            start_new_session=True, preexec_fn=evaluated_limits,
        )
        timed_out = False
        try:
            process.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            kill_evaluated_processes()
            process.wait()
        out.seek(0)
        err.seek(0)
        stdout, stderr = out.read(MAX_OUTPUT_BYTES), err.read(MAX_OUTPUT_BYTES)
        print(json.dumps({
            'stdout': stdout.decode('utf-8', errors='replace'),
            'stderr': stderr.decode('utf-8', errors='replace'),
            'exit_code': 124 if timed_out else process.returncode,
            'timed_out': timed_out,
            'output_truncated': len(stdout) >= MAX_OUTPUT_BYTES or len(stderr) >= MAX_OUTPUT_BYTES,
        }))


if __name__ == '__main__':
    main()
