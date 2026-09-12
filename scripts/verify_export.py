"""I verify this review package without consulting its original archive."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root.resolve()
    manifest = json.loads((root / 'provenance/export-manifest.json').read_text())
    failures = []
    for relative, expected in manifest['files'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError('An export path escapes the clone: ' + relative)
        if not path.is_file():
            failures.append({'file': relative, 'problem': 'missing'})
            continue
        content = path.read_bytes()
        if len(content) != expected['bytes'] or hashlib.sha256(content).hexdigest() != expected['sha256']:
            failures.append({'file': relative, 'problem': 'bytes or hash changed'})
        if expected.get('archive_path') and expected.get('source_sha256') != expected['sha256']:
            failures.append({'file': relative, 'problem': 'copied source differs from archive'})
    print(json.dumps({'passed': not failures, 'listed_files': len(manifest['files']),
                      'failures': failures}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
