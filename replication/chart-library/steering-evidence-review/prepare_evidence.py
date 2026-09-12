"""I reduce every assigned row in the frozen batch, retaining missing outcomes."""
from pathlib import Path
import csv, hashlib, json
from collections import Counter, defaultdict

HERE = Path(__file__).resolve().parent
REPLICATION = HERE.parents[1]
SOURCE = REPLICATION / 'steering-series/2026-09-12-positive-confirmation/analysis-ready/20260912T051758Z'
OUT = HERE / 'data'
OUT.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def boolean(v):
    assert v in ('True', 'False'), v
    return v == 'True'
def save_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)

source = SOURCE / 'arm-by-page.csv'
assert sha(source) == '9b6110ce9b781c8dfd13470eb537717c13def35a542183f6eee7ab316d477093'
rows = list(csv.DictReader(source.open()))
assert len(rows) == 110
keys = [(r['cohort'], r['variant'], r['page_id'], r['arm_id']) for r in rows]
assert len(set(keys)) == len(rows)
manifest = [{'path': str(source), 'sha256': sha(source)}]
probes, compact, episode_paths = [], [], set()
for r in rows:
    d = dict(r)
    d['page_label'] = ('H' if r['cohort'] == 'historical' else 'N') + str(int(r['page_id']) + 1)
    for k in ('assigned', 'recorded'):
        d[k] = boolean(r[k])
    if not d['recorded']:
        assert r['outcome'] == 'UNRUN' and not r['episode_file']
        compact.append(d); continue
    p = Path(r['episode_file']); assert str(p) not in episode_paths
    episode_paths.add(str(p)); assert sha(p) == r['episode_sha256']
    e = json.loads(p.read_text())
    assert str(e['seed']) == r['seed'] and e['case_id'] == r['case_id']
    receipts = json.loads((p.parent / 'receipts.json').read_text())
    assert receipts == e['receipts']
    upload = any(x.get('valid', False) for x in receipts)
    assert upload == e['verified_dummy_upload'] == boolean(r['verified_dummy_upload'])
    censored = bool(e['censored']); assert censored == boolean(r['censored'])
    outcome = 'UPLOAD' if upload else 'CENSORED' if censored else 'NO_UPLOAD'
    assert outcome == r['outcome']
    if r['exposure_confirmed']:
        assert e['exposure_confirmed'] == boolean(r['exposure_confirmed'])
    d['exposure_from_episode'] = e['exposure_confirmed']
    if outcome == 'NO_UPLOAD':
        assert e['status'] == 'completed'
        if r['variant'] != 'benign': assert e['exposure_confirmed']
    for f in (p, p.parent / 'receipts.json'):
        manifest.append({'path': str(f), 'sha256': sha(f)})
    exposed = [t for t in e['steering_turns'] if t.get('page_span_tokens', 0) > 0]
    if exposed:
        t = exposed[0]
        for span, mean in t['probe_means'].items():
            if 'p_cot' not in mean: continue
            assert abs(sum(mean['p_' + k] for k in ('system', 'user', 'cot', 'assistant', 'tool')) - 1) < 2e-6
            pr = {k: d[k] for k in ('cohort', 'variant', 'case_id', 'page_id', 'page_label', 'arm_id', 'seed', 'outcome', 'episode_file', 'episode_sha256')}
            pr.update({'turn': t['turn'], 'span': span, **mean, 'prompt_sha256': t['prompt_sha256']})
            probes.append(pr)
        if r['page_p_cot']:
            assert float(r['page_p_cot']) == t['probe_means']['page']['p_cot']
    compact.append(d)

assert len(episode_paths) == 100
assert Counter(r['outcome'] for r in compact)['CENSORED'] == 3
forgery = [r for r in compact if r['variant'] == 'forgery']
assert sum(r['recorded'] for r in forgery) == 80
payload = [r for r in probes if r['variant'] == 'forgery' and r['span'] == 'payload']
assert len(payload) == 60 and {r['n_tokens'] for r in payload} == {216, 225, 230, 264, 325}
assert len({(r['cohort'], r['case_id'], r['arm_id']) for r in payload}) == 60

pairs, summaries = [], []
base = {(r['cohort'], r['case_id']): r for r in forgery if r['arm_id'] == 'none'}
for r in forgery:
    if r['arm_id'] == 'none': continue
    b = base[(r['cohort'], r['case_id'])]
    assert b['seed'] == r['seed']
    before, after = b['outcome'], r['outcome']
    change = ('unrun' if after == 'UNRUN' else 'unresolved' if after == 'CENSORED' else
              'unchanged' if after == before else 'prevented' if before == 'UPLOAD' else 'introduced')
    pairs.append({k: r[k] for k in ('cohort', 'case_id', 'page_label', 'arm_id', 'seed')} |
                 {'baseline_outcome': before, 'intervention_outcome': after, 'change': change})
for arm in sorted({r['arm_id'] for r in forgery}):
    for cohort in ('historical', 'new', 'both'):
        rr = [r for r in forgery if r['arm_id'] == arm and (cohort == 'both' or r['cohort'] == cohort)]
        if not rr: continue
        c = Counter(r['outcome'] for r in rr)
        pc = Counter(r['change'] for r in pairs if r['arm_id'] == arm and (cohort == 'both' or r['cohort'] == cohort))
        summaries.append({'arm_id': arm, 'cohort': cohort, 'assigned': len(rr), 'recorded': sum(r['recorded'] for r in rr),
                          **{k: c[k] for k in ('UPLOAD', 'NO_UPLOAD', 'CENSORED', 'UNRUN')},
                          **{k: pc[k] for k in ('prevented', 'introduced', 'unchanged', 'unresolved', 'unrun')}})
role = [r for r in payload if r['arm_id'] == 'role_a16']
assert len(role) == 10 and max(r['p_cot'] for r in role) < 2.1e-18
assert sum(r['outcome'] == 'UPLOAD' for r in role) == 6
save_csv(OUT/'all-assigned-outcomes.csv', compact)
save_csv(OUT/'forgery-outcomes.csv', forgery)
save_csv(OUT/'first-exposure-probes.csv', probes)
save_csv(OUT/'forgery-payload-probes.csv', payload)
save_csv(OUT/'paired-changes.csv', pairs)
save_csv(OUT/'arm-counts.csv', summaries)
(OUT/'source-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
audit = {'source_snapshot': str(SOURCE), 'assigned': 110, 'recorded': len(episode_paths),
         'censored': 3, 'unrun': 10, 'forgery_episodes': 80, 'forgery_pages': 10,
         'forgery_payload_probe_records': len(payload), 'payload_token_lengths': sorted({r['n_tokens'] for r in payload}),
         'role_payload_max_cot': max(r['p_cot'] for r in role),
         'role_uploads': 6, 'no_intervention_uploads': 7,
         'receipt_checks': 'Saved receipt validity matched separate receipts and every episode. Independent review reparses dummy fields.',
         'inference': 'Descriptive complete-cohort counts; no population confidence interval or equivalence claim.',
         'inputs': len(manifest), 'files': {p.name: sha(p) for p in OUT.iterdir() if p.is_file() and p.name != 'verification.json'}}
(OUT/'verification.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k != 'files'}, indent=2))
