"""I combine the closed GPU A and B evidence without changing their inputs."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import shutil

SERIES = Path(__file__).resolve().parent.parent
A = SERIES / 'analysis-ready/20260912T050153Z'
B = SERIES.parents[1] / 'cloud/outbox/parallel-h100/gpu-b-20260912T041801Z/closeout-20260912T051046Z'

def read(path):
    return json.loads(path.read_text())

def digest(path):
    data = path.read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}

def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')

def main():
    now = datetime.now(timezone.utc)
    dest = SERIES / 'analysis-ready' / now.strftime('%Y%m%dT%H%M%SZ')
    dest.mkdir()
    (dest / 'inputs').mkdir()
    a_manifest = read(A / 'artifact-manifest.json')
    a_index = read(A / 'completed-episode-index.json')
    a_rows = list(csv.DictReader((A / 'arm-by-page.csv').open()))
    b = read(B / 'closeout.json')
    files = dict(a_manifest['files'])
    for path, expected in {**files, **b['sources']}.items():
        actual = digest(Path(path))
        assert actual == expected, ('source_changed', path, expected, actual)
        files[path] = actual
    index_by_file = {str(Path(r['episode_path']) / 'episode.json'): r for r in a_index}
    rows = []
    for r in a_rows:
        x = index_by_file[r['episode_file']]
        e = read(Path(r['episode_file']))
        assert digest(Path(r['episode_file']))['sha256'] == r['episode_sha256']
        rows.append(dict(
            source_gpu='A', run_id=x['run_id'], queue_item=6 if r['variant']=='standard' else '',
            cohort=r['cohort'], page_id=r['page_id'], case_id=r['case_id'], variant=r['variant'],
            arm_id=r['arm_id'], seed=r['seed'], assigned=True, recorded=True,
            outcome=r['outcome'].upper(), unrun_reason=None, censored=e['censored'],
            exposure_confirmed=e.get('exposure_confirmed'), emitted_upload_attempt=e.get('emitted_upload_attempt'),
            verified_dummy_upload=e['verified_dummy_upload'],
            candidate_summary=x.get('summary_present_heuristic', x.get('candidate_summary_present')),
            summary_quality_judged=x.get('summary_quality_judged', False),
            page_p_tool=r['page_p_tool'], page_p_user=r['page_p_user'], page_p_cot=r['page_p_cot'],
            probe_turn=r['probe_turn'], episode_file=r['episode_file'], episode_sha256=r['episode_sha256'],
            source_table=str(A/'arm-by-page.csv')))
    b_index = []
    for r in b['rows']:
        path = Path(r['episode_path']) / 'episode.json'
        e = read(path) if r['recorded_in_index'] else None
        if e:
            assert digest(path)['sha256'] == r['episode_sha256']
            assert e['censored'] == r['censored'] and e['verified_dummy_upload'] == r['verified_dummy_upload']
            assert e['seed'] == r['seed'] and e['case_id'] == r['case_id']
            for p in path.parent.rglob('*'):
                if p.is_file():
                    files[str(p)] = digest(p)
            b_index.append({**r, 'run_id': b['run_id'], 'source_gpu': 'B',
                            'emitted_upload_attempt': e['emitted_upload_attempt']})
        rows.append(dict(
            source_gpu='B', run_id=b['run_id'], queue_item=r['item'], cohort=r['cohort'],
            page_id=r['page_id'], case_id=r['case_id'], variant=r['variant'],
            arm_id=r['treatment'] if r['item'] != 2 else r['engine_arm_id'], seed=r['seed'],
            assigned=r['assigned'], recorded=r['recorded_in_index'], outcome=r['outcome'],
            unrun_reason=r['unrun_reason'], censored=r['censored'], exposure_confirmed=r['exposure_confirmed'],
            emitted_upload_attempt=e['emitted_upload_attempt'] if e else None,
            verified_dummy_upload=r['verified_dummy_upload'], candidate_summary=r['summary_present'],
            summary_quality_judged=r['summary_quality_judged'], page_p_tool=r['p_tool'],
            page_p_user=r['p_user'], page_p_cot=r['p_cot'], probe_turn=r['first_page_probe_turn'],
            episode_file=str(path) if e else None, episode_sha256=r['episode_sha256'],
            source_table=str(B/'arm-by-page.csv')))
    assert len(rows) == 110 and sum(r['recorded'] for r in rows) == 100
    assert sum(r['outcome']=='UNRUN' for r in rows) == 10
    assert sum(r['censored'] is True for r in rows) == 3
    assert len({(r['run_id'],r['queue_item'],r['case_id'],r['arm_id']) for r in rows}) == 110
    rows.sort(key=lambda r:(r['cohort'],r['page_id'],r['variant'],r['arm_id']))
    with (dest/'arm-by-page.csv').open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    totals=defaultdict(Counter)
    for r in rows:
        c=totals[(r['cohort'],r['variant'],r['arm_id'])]
        c['assigned']+=1; c['recorded']+=int(r['recorded']); c[r['outcome']]+=1
        c['candidate_summaries']+=int(r['candidate_summary'] is True)
        c['summary_presence_unavailable']+=int(r['recorded'] and r['candidate_summary'] is None)
    save(dest/'arm-summary.json', [{'cohort':k[0],'variant':k[1],'arm_id':k[2],**dict(v)} for k,v in sorted(totals.items())])
    source_inputs={}
    for prefix, root, names in [('gpu-a',A,['arm-by-page.csv','artifact-manifest.json','completed-episode-index.json','attribution-index.json']),
                              ('gpu-b',B,['arm-by-page.csv','closeout.json','gate-readouts.csv','sync-and-close-receipt.json','retained-pods-live.json','independent-item2-audit.json'])]:
        for name in names:
            src=root/name
            if not src.exists():
                assert name in ['retained-pods-live.json','independent-item2-audit.json']
                continue
            target=dest/'inputs'/f'{prefix}-{name}'
            shutil.copyfile(src,target); source_inputs[str(src)]=digest(src)
    save(dest/'completed-episode-index.json', a_index+b_index)
    shutil.copyfile(A/'attribution-index.json',dest/'attribution-index.json')
    shutil.copyfile(B/'gate-readouts.csv',dest/'gate-readouts.csv')
    manifest={'created_at':now.isoformat(),'full_agent_episodes_recorded':100,
              'full_agent_episodes_uncensored':97,'censored_episodes':3,'unrun_assigned_slots':10,
              'authorship_readouts':10,'source_inputs':source_inputs,
              'audit_receipts':a_manifest['audit_receipts']+[str(B/'sync-and-close-receipt.json')],
              'files':files}
    save(dest/'artifact-manifest.json',manifest)
    save(dest/'verification.json',{'verified_at':now.isoformat(),'source_hash_errors':0,
          'a_input_rows':80,'b_input_rows':30,'combined_rows':110,'recorded_episodes':100,
          'unrun_slots':10,'censored_episodes':3,'hashed_local_artifacts':len(files),
          'table_sha256':digest(dest/'arm-by-page.csv')['sha256'],
          'normalization_notes':['A candidate_summary is populated from the saved index summary_present_heuristic or candidate_summary_present fields, preserving missing values as unavailable; the previous CSV left it blank.',
          'B absent episode files and endpoint values remain blank for all UNRUN slots.',
          'Hooks-off probe fields remain blank; gate readouts remain separate from behavioral rows.',
          'Input tables are copied without modification.']})
    save(SERIES/'analysis-ready/latest.json',{'snapshot':str(dest),'closed_full_episodes':100,
          'authorship_readouts':10,'unrun_assigned_slots':10,'censored_episodes':3,
          'artifact_file_count':len(files),'arm_by_page':str(dest/'arm-by-page.csv'),
          'prior_snapshot':str(A),'gpu_b_included':True})
    print(json.dumps({'snapshot':str(dest),'rows':110,'recorded_episodes':100,'unrun':10,'files':len(files)}))

if __name__ == '__main__':
    main()
