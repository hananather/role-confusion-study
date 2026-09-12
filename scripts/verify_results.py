"""I verify the exported saved evidence without inference, network access or writes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ORIGINAL_ROOT = Path('/Users/hananather/Desktop/MATS 12.0')
SNAPSHOT = Path('replication/steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    def resolve_saved(value):
        path = Path(value)
        relative = path.relative_to(ORIGINAL_ROOT) if path.is_absolute() else path
        result = (root / relative).resolve()
        if not result.is_relative_to(root):
            raise ValueError('An evidence path escapes the export: ' + str(value))
        return result

    snapshot = root / SNAPSHOT
    manifest = json.loads((snapshot / 'artifact-manifest.json').read_text())
    verification = json.loads((snapshot / 'verification.json').read_text())
    failures = []
    for source, expected in manifest['files'].items():
        path = resolve_saved(source)
        if not path.is_file():
            failures.append(('missing', source))
            continue
        content = path.read_bytes()
        if len(content) != expected['bytes'] or hashlib.sha256(content).hexdigest() != expected['sha256']:
            failures.append(('hash_or_size', source))
    table = snapshot / 'arm-by-page.csv'
    if hashlib.sha256(table.read_bytes()).hexdigest() != verification['table_sha256']:
        failures.append(('table_hash', str(table.relative_to(root))))
    with table.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    counts = Counter(row['outcome'] for row in rows)
    recorded = [row for row in rows if row['recorded'] == 'True']
    identities = {(row['run_id'], row['queue_item'], row['case_id'], row['arm_id']) for row in rows}
    expected_counts = {'rows': 110, 'recorded': 100, 'unrun': 10, 'censored': 3}
    actual_counts = {'rows': len(rows), 'recorded': len(recorded), 'unrun': counts['UNRUN'],
                     'censored': sum(row['censored'] == 'True' for row in rows)}
    if actual_counts != expected_counts or len(identities) != len(rows):
        failures.append(('accounting', actual_counts))
    for row in recorded:
        path = resolve_saved(row['episode_file'])
        if not path.is_file():
            continue
        content = path.read_bytes()
        episode = json.loads(content)
        if hashlib.sha256(content).hexdigest() != row['episode_sha256']:
            failures.append(('episode_hash', row['episode_file']))
        for key in ('case_id', 'seed', 'censored', 'verified_dummy_upload'):
            if str(episode[key]) != row[key]:
                failures.append(('episode_field', row['episode_file'], key))
    print(json.dumps({'passed': not failures, 'counts': actual_counts,
                      'indexed_artifacts': len(manifest['files']), 'failures': failures}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
