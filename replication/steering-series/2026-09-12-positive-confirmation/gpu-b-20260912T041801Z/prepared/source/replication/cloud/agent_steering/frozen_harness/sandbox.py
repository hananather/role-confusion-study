"""Credential-free Docker shell environment with only a private page and receiver.

Model text reaches bash only inside the container, as standard input to a
trusted launcher. The Docker CLI always receives an argument list on the host.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any
import uuid


DEFAULT_IMAGE = 'mats-agent-hijacking-sandbox:2026-09-11'
SANDBOX_DIR = Path(__file__).resolve().parents[1] / 'sandbox'
PAGE_URL = 'http://127.0.0.1:8765/page.html'
UPLOAD_URL = 'http://127.0.0.1:8765/api/'


def remove_container(run, target: str, *, attempts: int = 3, retry_delay: float = .25) -> dict[str, Any]:
    """Remove one exact container and require an explicit missing-container result.

    A Docker transport error is never evidence of absence. Both the controller
    and detached watchdog use this bounded procedure.
    """
    last_error = 'Cleanup did not run'
    for attempt in range(1, attempts + 1):
        try:
            run(['rm', '--force', target], check=False, timeout=5)
        except (subprocess.SubprocessError, OSError):
            # A timeout can follow successful removal; inspect resolves that ambiguity.
            pass
        try:
            result = run(['container', 'inspect', target], check=False, timeout=5)
            missing = re.search(r'No such (?:container|object):\s*' + re.escape(target) + r'(?:\s|$)',
                                result.stderr or '')
            if result.returncode != 0 and missing:
                return {'cleanup_verified': True, 'cleanup_attempts': attempt, 'cleanup_error': None}
            if result.returncode == 0:
                last_error = 'Docker still reports the container present'
            else:
                last_error = 'Docker inspection could not verify absence: ' + (result.stderr or '').strip()[:1000]
        except (subprocess.SubprocessError, OSError) as error:
            last_error = 'Docker inspection failed: ' + type(error).__name__ + ': ' + str(error)[:1000]
        if attempt < attempts:
            time.sleep(retry_delay)
    return {'cleanup_verified': False, 'cleanup_attempts': attempts, 'cleanup_error': last_error}


def build_image(image: str = DEFAULT_IMAGE) -> str:
    """Build only the dedicated, three-file Docker context and return its ID."""
    docker = shutil.which('docker')
    if not docker:
        raise RuntimeError('Docker CLI is unavailable')
    subprocess.run([docker, 'build', '--tag', image, str(SANDBOX_DIR)], check=True, timeout=600)
    result = subprocess.run([docker, 'image', 'inspect', image, '--format', '{{.Id}}'],
                            check=True, capture_output=True, text=True, timeout=15)
    return result.stdout.strip()


class DockerSandbox:
    """One fresh container per episode; inference stays outside this boundary."""

    def __init__(self, image: str, html: str, secret: str, episode_id: str,
                 lifetime_seconds: int = 900):
        if not isinstance(html, str) or len(html.encode('utf-8')) > 8 * 1024 * 1024:
            raise ValueError('Page must be text of at most 8 MiB')
        if not isinstance(secret, str) or not secret or len(secret.encode('utf-8')) > 16384:
            raise ValueError('Dummy secret must contain 1–16384 UTF-8 bytes')
        if not isinstance(lifetime_seconds, int) or not 1 <= lifetime_seconds <= 86400:
            raise ValueError('Lifetime must be an integer between 1 and 86400 seconds')
        self.image, self.html, self.secret = image, html, secret
        self.episode_id = episode_id
        self.lifetime_seconds = lifetime_seconds
        self.container_id: str | None = None
        self._container_name: str | None = None
        self._watchdog: subprocess.Popen | None = None
        self._closed = False
        self._cleanup_started = False
        self._last_receipts: list[dict[str, Any]] = []
        self._docker = shutil.which('docker')
        if not self._docker:
            raise RuntimeError('Docker CLI is unavailable')
        self.metadata: dict[str, Any] = {
            'backend': 'docker', 'image': image, 'episode_id': episode_id,
            'page_url': PAGE_URL, 'upload_url': UPLOAD_URL,
            'cwd': '/workspace/container-1', 'evaluated_uid': 1000,
            'network': 'none', 'host_mounts': [], 'readonly_rootfs': True, 'init': True,
            'memory_bytes': 512 * 1024 * 1024, 'cpus': 1, 'pids_limit': 96,
            'workspace_tmpfs_bytes': 64 * 1024 * 1024,
            'tmp_tmpfs_bytes': 16 * 1024 * 1024,
            'trusted_tmpfs_bytes': 32 * 1024 * 1024,
            'output_limit_bytes_per_stream': 1024 * 1024, 'max_upload_bytes': 65536,
            'max_receipts': 128, 'lifetime_seconds': lifetime_seconds,
            'max_receipt_log_bytes': 8 * 1024 * 1024,
            'trusted_capabilities': ['CHOWN', 'SETUID', 'SETGID', 'KILL'],
            'page_sha256': hashlib.sha256(html.encode()).hexdigest(),
            'dummy_secret_sha256': hashlib.sha256(secret.encode()).hexdigest(),
            'cleanup_verified': False, 'cleanup_error': None,
        }

    def _run(self, args, *, input=None, timeout=30, check=True):
        return subprocess.run([self._docker, *args], input=input, text=True,
                              capture_output=True, timeout=timeout, check=check)

    def start(self):
        if self.container_id is not None or self._closed:
            raise RuntimeError('Sandbox cannot be started twice')
        name = 'mats-hijack-' + re.sub(r'[^a-zA-Z0-9_.-]', '-', self.episode_id)[:32] + '-' + uuid.uuid4().hex[:8]
        self._container_name = name
        try:
            result = self._run([
                'run', '--detach', '--init', '--name', name, '--network', 'none',
                '--read-only', '--cap-drop', 'ALL', '--cap-add', 'SETUID',
                '--cap-add', 'SETGID', '--cap-add', 'KILL', '--cap-add', 'CHOWN',
                '--security-opt', 'no-new-privileges:true',
                '--memory', '512m', '--memory-swap', '512m', '--cpus', '1',
                '--pids-limit', '96', '--ulimit', 'nofile=256:256',
                '--shm-size', '16m', '--log-driver', 'none',
                '--tmpfs', '/workspace:rw,noexec,nosuid,nodev,size=64m,mode=0755',
                '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size=16m,mode=1777',
                '--tmpfs', '/run/harness:rw,noexec,nosuid,nodev,size=32m,mode=0700',
                '--label', 'mats.agent-hijacking=true', self.image,
            ])
            self.container_id = result.stdout.strip()
            self.metadata['container_id'] = self.container_id
            self.metadata['started_at_unix'] = time.time()
            self._watchdog = subprocess.Popen(
                [sys.executable, str(SANDBOX_DIR / 'watchdog.py'), self._docker,
                 self.container_id, str(self.lifetime_seconds)],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            self.metadata['watchdog_pid'] = self._watchdog.pid
            setup = (
                "import os,sys; from pathlib import Path; "
                "p=Path('/run/harness/config.tmp'); p.write_text(sys.stdin.read()); "
                "p.chmod(0o600); p.rename('/run/harness/config.json')"
            )
            self._run(['exec', '-i', '--user', '0', self.container_id, 'python', '-c', setup],
                      input=json.dumps({'html': self.html, 'secret': self.secret}, ensure_ascii=False), timeout=15)
            ready = (
                "import time,urllib.request\n"
                "for _ in range(100):\n"
                " try:\n"
                "  urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=.2).read();break\n"
                " except OSError:time.sleep(.05)\n"
                "else:raise RuntimeError('Private page server unavailable')"
            )
            self._run(['exec', '--user', '0', self.container_id, 'python', '-c', ready], timeout=10)
            details = json.loads(self._run(['inspect', self.container_id]).stdout)[0]
            self.metadata['image_id'] = details['Image']
            self.metadata['architecture'] = self._run(
                ['image', 'inspect', self.image, '--format', '{{.Architecture}}']).stdout.strip()
            return self
        except BaseException:
            self.close()
            raise

    def shell(self, command: str, timeout: int = 30) -> dict[str, Any]:
        if self.container_id is None or self._closed or self._cleanup_started:
            raise RuntimeError('Sandbox is not running')
        if not isinstance(command, str) or len(command) > 32768:
            raise ValueError('Shell command must be a string of at most 32768 characters')
        if not isinstance(timeout, int) or not 1 <= timeout <= 60:
            raise ValueError('Shell timeout must be an integer from 1 to 60 seconds')
        try:
            result = self._run(
                ['exec', '-i', '--user', '0', self.container_id,
                 'python', '/opt/harness/runner.py'],
                input=json.dumps({'command': command, 'timeout': timeout}), timeout=timeout + 10,
            )
            return json.loads(result.stdout)
        except (subprocess.SubprocessError, json.JSONDecodeError) as error:
            raise RuntimeError('Sandbox shell failed or its lifetime expired') from error

    def receipts(self) -> list[dict[str, Any]]:
        if self.container_id is not None and not self._closed:
            result = self._run(['exec', '--user', '0', self.container_id, 'cat',
                                '/run/harness/receipts.jsonl'], check=False, timeout=10)
            if result.returncode == 0:
                self._last_receipts = [json.loads(line) for line in result.stdout.splitlines()
                                       if line.endswith('}')]
        return [dict(row) for row in self._last_receipts]

    def close(self):
        if self._closed:
            return
        self._cleanup_started = True
        receipt_error = None
        if self.container_id:
            try:
                self.receipts()
            except (subprocess.SubprocessError, OSError, ValueError) as error:
                receipt_error = str(error)[:1000]
                self.metadata['receipt_collection_error'] = receipt_error
        target = self.container_id or self._container_name
        outcome = (remove_container(self._run, target) if target else
                   {'cleanup_verified': True, 'cleanup_attempts': 0, 'cleanup_error': None})
        self.metadata.update(outcome)
        self.metadata['cleanup_checked_at_unix'] = time.time()
        if not outcome['cleanup_verified']:
            # Keep the watchdog active and allow a later explicit cleanup retry.
            raise RuntimeError('Sandbox cleanup unverified for ' + str(target) + ': ' + outcome['cleanup_error'])
        self._closed = True
        self.metadata['closed_at_unix'] = time.time()
        if self._watchdog:
            if self._watchdog.poll() is None:
                self._watchdog.terminate()
                try:
                    self._watchdog.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._watchdog.kill()
                    self._watchdog.wait(timeout=2)
            self.metadata['watchdog_returncode'] = self._watchdog.poll()
        if receipt_error:
            raise RuntimeError('Sandbox removed, but final receipt collection failed: ' + receipt_error)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()
