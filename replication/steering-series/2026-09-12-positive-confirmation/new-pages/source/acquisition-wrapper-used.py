"""I apply the frozen page recipe with logged prior-title exclusions and a wall cap."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import threading
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
FROZEN = ROOT / 'replication/cloud/agent_steering/frozen_harness/prepare.py'
SOURCE = ROOT / 'prompt-injection-as-role-confusion'
PRIOR = ROOT / 'replication/steering-agent/runs/forgery5-tool_minus_cot-a16/input-manifest.json'
TARGET, SEED, MAX_CANDIDATES, WALL_SECONDS = 100, 20260912, 1000, 900
START = time.monotonic()
STATE = {'stage': 'starting', 'source_candidates': 0, 'excluded_prior_titles': 0, 'accepted': 0}
STOP = threading.Event()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def append_json(path, value):
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')


def heartbeat():
    while not STOP.is_set():
        journal = OUT / 'retrieval.jsonl'
        counts = {}
        if journal.exists():
            for line in journal.read_text().splitlines():
                try:
                    status = json.loads(line)['status']
                    counts[status] = counts.get(status, 0) + 1
                except (json.JSONDecodeError, KeyError):
                    pass
        STATE['accepted'] = counts.get('selected', 0)
        write_json(OUT / 'preparation-heartbeat.json', {
            **STATE, 'utc': datetime.now(timezone.utc).isoformat(),
            'elapsed_s': round(time.monotonic() - START, 3), 'retrieval_status_counts': counts,
            'target_pages': TARGET, 'max_source_candidates': MAX_CANDIDATES,
            'wall_cap_s': WALL_SECONDS, 'model_calls': 0, 'gpu_calls': 0})
        STOP.wait(15)


def alarm_handler(signum, frame):
    raise TimeoutError('Prespecified 900-second public-page preparation deadline')


def main():
    if any((OUT / name).exists() for name in ('raw', 'manifest.json', 'preparation-receipt.json', 'source-candidates.jsonl')):
        raise RuntimeError('I do not overwrite or resume a prior acquisition in this directory')
    (OUT / 'source').mkdir(exist_ok=True)
    shutil.copyfile(FROZEN, OUT / 'source/frozen-prepare.py')
    shutil.copyfile(SOURCE / 'experiments/cot-forgery-agent-evals/prompts/injections.yaml', OUT / 'source/injections.yaml')
    shutil.copyfile(SOURCE / 'experiments/cot-forgery-agent-evals/prompts/classify-injection-output.yaml', OUT / 'source/classify-injection-output.yaml')
    prior = json.loads(PRIOR.read_text())
    prior_titles = sorted({c['title'] for c in prior['cases']})
    excluded = {title.strip().casefold() for title in prior_titles}
    assert len(excluded) == 5
    write_json(OUT / 'preparation-plan.json', {
        'seed': SEED, 'target_pages': TARGET, 'max_source_candidates': MAX_CANDIDATES,
        'wall_cap_s': WALL_SECONDS, 'prior_titles_excluded_before_count': prior_titles,
        'prior_manifest': str(PRIOR), 'prior_manifest_sha256': digest(PRIOR),
        'frozen_prepare': str(FROZEN), 'frozen_prepare_sha256': digest(FROZEN),
        'wrapper_sha256': digest(__file__), 'source_yaml_sha256': digest(OUT / 'source/injections.yaml'),
        'eligibility': {'class_stripped_bytes_max': 100*1024, 'raw_transport_bytes_max': 512*1024, 'body_boundary_required': True},
        'recipe_changes': ['I exclude the five original titles before acceptance/counting and journal their original stream indices.', 'I cap original source candidates at1000 and wall time at900seconds.'],
        'model_calls': 0, 'gpu_calls': 0, 'paid_api_calls': 0,
        'prospective': True, 'outcome_based_selection': False})
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(WALL_SECONDS)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    receipt = {'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'launchable': False}
    try:
        os.environ['USE_TORCH'] = '0'
        os.environ['USE_TF'] = '0'
        os.environ['USE_FLAX'] = '0'
        STATE['stage'] = 'loading_public_dataset_metadata'
        import datasets
        original_load = datasets.load_dataset

        class FilteredDataset:
            def __init__(self, dataset):
                self.dataset = dataset
            def shuffle(self, *, seed, buffer_size):
                shuffled = self.dataset.shuffle(seed=seed, buffer_size=buffer_size)
                def rows():
                    for original_ix, row in enumerate(shuffled):
                        if original_ix >= MAX_CANDIDATES:
                            break
                        is_prior = row['title'].strip().casefold() in excluded
                        STATE['source_candidates'] = original_ix + 1
                        STATE['excluded_prior_titles'] += int(is_prior)
                        STATE['stage'] = 'retrieving_public_html'
                        append_json(OUT / 'source-candidates.jsonl', {
                            'source_candidate': original_ix, 'dataset_id': str(row['id']),
                            'title': row['title'], 'url': row['url'],
                            'status': 'excluded_prior_title' if is_prior else 'passed_to_frozen_recipe'})
                        if not is_prior:
                            yield row
                return rows()

        def filtered_load(*args, **kwargs):
            return FilteredDataset(original_load(*args, **kwargs))
        datasets.load_dataset = filtered_load
        spec = importlib.util.spec_from_file_location('frozen_prepare_for_new_pages', FROZEN)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        pages = module.freeze_pages(OUT, TARGET, SEED, MAX_CANDIDATES)
        datasets.load_dataset = original_load
        assert len(pages) == TARGET
        assert not excluded.intersection(p['title'].strip().casefold() for p in pages)
        if len({p['dataset_id'] for p in pages}) != TARGET or len({p['sha256'] for p in pages}) != TARGET:
            raise RuntimeError('Duplicate source identity or raw-page content found; I retain the receipt rather than silently substituting')
        STATE['stage'] = 'building_frozen_input_manifest'
        manifest = module.build_manifest(OUT, pages, SOURCE, SEED)
        manifest['preparation_wrapper_sha256'] = digest(__file__)
        manifest['prior_title_exclusions'] = prior_titles
        manifest['adaptations'].append('five original titles excluded before counting;900-second wall cap and1000 original-source-candidate cap')
        target = OUT / 'manifest.json'
        write_json(target, manifest)
        (OUT / 'manifest.sha256').write_text(digest(target) + '  manifest.json\n')
        receipt.update(status='complete', launchable=True, pages=TARGET,
                       fixture_cases=len(manifest['cases']), forged_cases=sum(c['variant']=='forgery' for c in manifest['cases']),
                       manifest_sha256=digest(target), source_yaml_sha256=manifest['payload_source_sha256'],
                       frozen_prepare_sha256=digest(FROZEN), wrapper_sha256=digest(__file__),
                       unique_page_ids=TARGET, unique_raw_hashes=TARGET)
        STATE['stage'] = 'complete'
    except Exception as error:
        message = re.sub(r'https?://\S+\?\S+', '[URL query omitted]', str(error))
        receipt.update(status='aborted', launchable=False, error_type=type(error).__name__, error=message[:1200])
        STATE['stage'] = 'aborted'
    finally:
        signal.alarm(0)
        STOP.set()
        thread.join(timeout=2)
        receipt.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_s=round(time.monotonic()-START,3),
                       source_candidates=STATE['source_candidates'], excluded_prior_titles=STATE['excluded_prior_titles'],
                       raw_files_preserved=len(list((OUT/'raw').glob('*.html'))) if (OUT/'raw').exists() else 0,
                       model_calls=0, gpu_calls=0, paid_api_calls=0)
        write_json(OUT/'preparation-receipt.json', receipt)
        write_json(OUT/'preparation-heartbeat.json', {**STATE,'utc':receipt['finished_utc'],'elapsed_s':receipt['elapsed_s']})
        print(json.dumps(receipt,ensure_ascii=False), flush=True)
    if receipt['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
