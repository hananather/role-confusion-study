"""I freeze all accepted pages from the stopped acquisition, without new requests."""
from pathlib import Path
import hashlib, importlib.util, json, re, shutil
from collections import Counter
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan = json.loads((OUT/'preparation-plan.json').read_text())
receipt = json.loads((OUT/'preparation-receipt.json').read_text())
assert receipt['status']=='aborted' and receipt['source_candidates']==1000
assert not (OUT/'manifest.json').exists()
rows = [json.loads(x) for x in (OUT/'retrieval.jsonl').read_text().splitlines()]
pages = [r for r in rows if r['status']=='selected']
assert len(pages)==77 and len({r['sha256'] for r in pages})==77 and len({r['dataset_id'] for r in pages})==77
excluded = {x.strip().casefold() for x in plan['prior_titles_excluded_before_count']}
assert not any(p['title'].strip().casefold() in excluded for p in pages)
assert [p['page_id'] for p in pages] == [f'{i:03d}' for i in range(77)]
for p in pages:
    assert sha(OUT/p['raw_path']) == p['sha256']
frozen = OUT/'source/frozen-prepare.py'
assert sha(frozen)==plan['frozen_prepare_sha256']
source = OUT/'source/reference'
rel = Path('experiments/cot-forgery-agent-evals')
(source/rel/'prompts').mkdir(parents=True,exist_ok=True)
for name in ['injections.yaml','classify-injection-output.yaml']:
    shutil.copyfile(OUT/'source'/name,source/rel/'prompts'/name)
shutil.copyfile(ROOT/'prompt-injection-as-role-confusion'/rel/'01-run-injections-gpt-oss.ipynb',source/rel/'01-run-injections-gpt-oss.ipynb')
assert sha(source/rel/'prompts/injections.yaml') == plan['source_yaml_sha256']
spec=importlib.util.spec_from_file_location('unchanged_frozen_prepare',frozen)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
manifest=module.build_manifest(OUT,pages,source,20260912)
http_errors=Counter()
for r in rows:
    if r['status']=='retrieval_error':
        m=re.search(r'\b[45][0-9][0-9]\b',r.get('detail',''))
        http_errors[m.group(0) if m else r.get('error','unknown')]+=1
manifest.update(
    corpus_status='incomplete_100_page_bank', requested_page_count=100, accepted_page_count=77,
    acquisition_complete=False, outcome_based_selection=False,
    prior_title_exclusions=plan['prior_titles_excluded_before_count'],
    preparation_wrapper_sha256=plan['wrapper_sha256'], frozen_prepare_sha256=plan['frozen_prepare_sha256'],
    acquisition_receipt_sha256=sha(OUT/'preparation-receipt.json'),
    partial_freeze_reason='The collector reached its prespecified1000-source-candidate cap after sustained HTTP429 responses. I froze ALL77 accepted pages in their original journal order; no model outcomes were observed.',
    future_execution_scope='Only a separately frozen throughput-budgeted prefix of this bank; not a completed100-page sample.')
manifest['adaptations'].extend(['five prior titles excluded before acceptance','incomplete100-page target: all77 accepted pages frozen after candidate-cap stop and HTTP429 access failures'])
target=OUT/'manifest.json'
target.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
(OUT/'manifest.sha256').write_text(sha(target)+'  manifest.json\n')
verification={
    'utc':datetime.now(timezone.utc).isoformat(),'status':'frozen_incomplete_bank',
    'requested_pages':100,'accepted_pages':77,'forgery_cases':77,'standard_cases':77,
    'unique_dataset_ids':77,'unique_raw_hashes':77,'prior_title_overlap':0,
    'source_candidates':1000,'prior_titles_encountered_and_excluded':receipt['excluded_prior_titles'],
    'retrieval_status_counts':dict(Counter(r['status'] for r in rows)),
    'http_error_counts':dict(http_errors),'journal_order_preserved':True,
    'manifest_sha256':sha(target),'source_yaml_sha256':plan['source_yaml_sha256'],
    'frozen_prepare_sha256':plan['frozen_prepare_sha256'],
    'original_acquisition_wrapper_sha256':plan['wrapper_sha256'],
    'freeze_script_sha256':sha(__file__),'no_additional_requests':True,
    'model_calls':0,'gpu_calls':0,'paid_api_calls':0,
    'does_not_authorize_execution':True}
(OUT/'partial-bank-receipt.json').write_text(json.dumps(verification,indent=2)+'\n')
files={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in {'frozen-files.json','preparation-heartbeat.json'} and '__pycache__' not in p.parts}
(OUT/'frozen-files.json').write_text(json.dumps({'sha256_files':files},indent=2)+'\n')
print(json.dumps(verification,indent=2))
