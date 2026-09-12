"""Independent host backstop; remove one exact container after a fixed lifetime."""

import subprocess
from pathlib import Path
import sys
import time


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from agent_hijacking.sandbox import remove_container

    docker, container_id, seconds = sys.argv[1:]
    time.sleep(float(seconds))

    def run(args, **kwargs):
        return subprocess.run([docker, *args], stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, **kwargs)

    outcome = remove_container(run, container_id, attempts=3, retry_delay=1)
    if not outcome['cleanup_verified']:
        print(outcome['cleanup_error'], file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
