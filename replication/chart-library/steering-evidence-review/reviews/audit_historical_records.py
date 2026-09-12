"""I audit saved records only; I never load a model or execute evaluated commands."""
from pathlib import Path
from collections import Counter, defaultdict
import csv,gzip,hashlib,json
ROOT=Path('/Users/hananather/Desktop/MATS 12.0')
OUT=ROOT/'replication/chart-library/steering-evidence-review/reviews'
def load(p): return json.loads(Path(p).read_text())
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csvwrite(name, rows):
 with (OUT/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def records(name):
 p=ROOT/'role-steering/data'/name/'episodes.jsonl.gz'; raw=gzip.decompress(p.read_bytes()); rows=[]
 for line_no,line in enumerate(raw.splitlines(),1):
  r=json.loads(line);r['_line']=line_no;r['_sha256']=hashlib.sha256(line).hexdigest();rows.append(r)
 return rows,{'source':str(p),'compressed_sha256':digest(p),'decompressed_sha256':hashlib.sha256(raw).hexdigest(),'records':len(rows)}
permission,pmeta=records('permission-20260905')
allocation=load(ROOT/'role-steering/data/permission-20260905/allocation.json')
cases={x['id']:x for x in load(ROOT/'role-steering/data/permission-20260905/cases.json')}
key=lambda r:(r['case_id'],r.get('task',r.get('authorized_action')),r['method'],r['seed'])
assert len({key(x) for x in permission})==len(permission)==149
assert len({key(x) for x in allocation})==len(allocation)==280
assert {key(x) for x in permission}<={key(x) for x in allocation}
indexed={key(x):x for x in permission}; paired=[]
for r in sorted(permission,key=key):
 c=cases[r['case_id']]; markers=r['evidence']['markers']; authorized=r['authorized_action'];unwanted='B' if authorized=='A' else 'A'
 assert r['task']==authorized
 for a in ['A','B']:
  assert (markers[a] is not None)==r['writes'][a]
  assert (markers[a]==c['nonces'][a])==r['completed_values'][a]
 assert r['writes'][unwanted]==r['unauthorized_write']
 assert r['completed_values'][authorized]==r['authorized_write']
 assert r['injected_page_sha256']==c['injected_page_sha256'] if 'injected_page_sha256' in c else True
 if r['method']=='none': continue
 base=indexed.get((r['case_id'],r['task'],'none',r['seed']))
 if base is None: continue
 scores=[q['probabilities']['tool'] for s in r['steps'] for q in s.get('role_scores',[])]
 first_scores=next((s['role_scores'] for s in r['steps'] if s.get('role_scores')),[])
 baseline_first=next((s['role_scores'] for s in base['steps'] if s.get('role_scores')),[])
 assert [x['token_id'] for x in first_scores]==[x['token_id'] for x in baseline_first]
 if r['method']!='prompt':
  assert [(x['index'],x['offset']) for x in first_scores]==[(x['index'],x['offset']) for x in baseline_first]
 paired.append(dict(case_id=r['case_id'],page_id=r['page_id'],template_id=r['template_id'],authorized_action=authorized,seed=r['seed'],method=r['method'],baseline_method='none',command_order=' '.join(c['command_order']),baseline_line=base['_line'],intervention_line=r['_line'],baseline_raw_record_sha256=base['_sha256'],intervention_raw_record_sha256=r['_sha256'],source_path=pmeta['source'],baseline_state=base['state'],intervention_state=r['state'],baseline_authorized_completion=base['authorized_write'],intervention_authorized_completion=r['authorized_write'],baseline_unauthorized_write=base['unauthorized_write'],intervention_unauthorized_write=r['unauthorized_write'],baseline_unauthorized_attempt=base['unauthorized_attempt'],intervention_unauthorized_attempt=r['unauthorized_attempt'],baseline_both_markers_written=all(base['writes'].values()),intervention_both_markers_written=all(r['writes'].values()),baseline_both_nonces_reported=all(base['nonce_recovered'].values()),intervention_both_nonces_reported=all(r['nonce_recovered'].values()),intervention_min_tool_all_recorded_command_tokens=min(scores),intervention_min_tool_first_prefill=min(x['probabilities']['tool'] for x in first_scores),intervention_mean_tool_first_prefill=sum(x['probabilities']['tool'] for x in first_scores)/len(first_scores),baseline_mean_tool_first_prefill=sum(x['probabilities']['tool'] for x in baseline_first)/len(baseline_first),first_prefill_scored_tokens=len(first_scores),all_recorded_scored_tokens=len(scores),read_exposure_confirmed=any(x.get('tool')=='functions.read_page' and x.get('output',{}).get('ok') for x in r['tool_events'])))
tool=[x for x in paired if x['method']=='tool_03'];assert len(tool)==21
csvwrite('permission-toolward-pairs.csv',tool)
csvwrite('permission-all-method-pairs.csv',paired)
summary=[]
for method in sorted({x['method'] for x in paired}):
 rows=[x for x in paired if x['method']==method]
 row={'method':method,'pairs':len(rows),'unique_cases':len({x['case_id'] for x in rows})}
 for metric in ['unauthorized_write','authorized_completion','both_nonces_reported','unauthorized_attempt']:
  row[metric+'_prevented_or_lost']=sum(x['baseline_'+metric] is True and x['intervention_'+metric] is False for x in rows)
  row[metric+'_introduced_or_gained']=sum(x['baseline_'+metric] is False and x['intervention_'+metric] is True for x in rows)
  row[metric+'_unknown']=sum(x['baseline_'+metric] is None or x['intervention_'+metric] is None for x in rows)
 row['both_written']=sum(x['intervention_both_markers_written'] for x in rows)
 row['min_tool']=min(x['intervention_min_tool_all_recorded_command_tokens'] for x in rows)
 summary.append(row)
csvwrite('permission-method-summary.csv',summary)
local=[]; errors=[]
for group in ['agent-hijacking','steering-agent']:
 for runpath in sorted((ROOT/'replication'/group/'runs').glob('*/run.json')):
  run=load(runpath)
  for case_id in run['selected_cases']:
   ep=runpath.parent/case_id/'episode.json'
   if not ep.exists():
    local.append(dict(group=group,run=runpath.parent.name,case_id=case_id,seed='',status='NO_CLOSED_EPISODE',exposed='',upload='',censored='',has_turn1_probe=False,turn1_payload_tokens='',turn1_payload_cot='',turn1_payload_tool='',turn1_payload_user='',source=str(ep),sha256=''))
    continue
   r=load(ep);receipt_path=ep.parent/'receipts.json';receipts=load(receipt_path) if receipt_path.exists() else r.get('receipts',[])
   if isinstance(receipts,dict): receipts=receipts.get('receipts',[])
   verified=any(x.get('valid') is True for x in receipts)
   if verified != r['verified_dummy_upload']: errors.append(str(ep)+' receipt mismatch')
   means={};sp=ep.parent/'steering.json'
   if sp.exists(): means=next((x.get('probe_means',{}).get('payload',{}) for x in load(sp)['turns'] if x['turn']==1),{})
   local.append(dict(group=group,run=runpath.parent.name,case_id=case_id,seed=r['seed'],status=r['status'],exposed=r.get('exposure_confirmed'),upload=r.get('verified_dummy_upload'),censored=r.get('censored'),has_turn1_probe=means.get('n_tokens',0)>0,turn1_payload_tokens=means.get('n_tokens',''),turn1_payload_cot=means.get('p_cot',''),turn1_payload_tool=means.get('p_tool',''),turn1_payload_user=means.get('p_user',''),source=str(ep),sha256=digest(ep)))
csvwrite('historical-mlx-episode-inventory.csv',local)
first,fmeta=records('first-20260904');falloc=load(ROOT/'role-steering/data/first-20260904/allocation.json');fcase=load(ROOT/'role-steering/data/first-20260904/cases.json'); fmap={x['id']:x for x in fcase['main']+fcase['calibration']}
assert len(first)==760 and len(falloc)==1880
firstrows=[]
for r in first:
 c=fmap[r['case_id']]; html=c['html'];ix=html.lower().rfind('</body>');ix=len(html) if ix<0 else ix
 outputs=[x.get('output',{}).get('html') for s in r['steps'] for x in s.get('tool_outputs',[]) if x.get('tool')=='functions.read']
 exposed=any(isinstance(x,str) and len(x)>=len(html) and x.startswith(html[:ix]) and x.endswith(html[ix:]) for x in outputs)
 scored=any(s.get('role_scores') for s in r['steps'])
 firstrows.append(dict(case_id=r['case_id'],task=r['task'],method=r['method'],seed=r['seed'],page_id=r['page_id'],template_id=r['template_id'],source_line=r['_line'],raw_record_sha256=r['_sha256'],state=r['state'],page_exposed=exposed,has_token_scores=scored,completed_transfer=r['completed_transfer'],nonce_recovered=r['nonce_recovered'],attempt_recorded=r['attempt']))
csvwrite('historical-transfer-record-inventory.csv',firstrows)
first_summary=[]
for task,method in sorted({(x['task'],x['method']) for x in firstrows}):
 rr=[x for x in firstrows if (x['task'],x['method'])==(task,method)]
 first_summary.append(dict(task=task,method=method,recorded=len(rr),planned=sum(x['task']==task and x['method']==method for x in falloc),exposed=sum(x['page_exposed'] for x in rr),transfers=sum(x['completed_transfer'] for x in rr),token_score_records=sum(x['has_token_scores'] for x in rr)))
csvwrite('historical-transfer-method-summary.csv',first_summary)
uptake=load(ROOT/'replication/role-uptake/runs/pilot-24-20260911/run.json')['results']
ceiling=load(ROOT/'replication/role-uptake/runs/ceiling-2-20260911/run.json')['results']
uptake_summary=[]
for cond in sorted({x['condition'] for x in uptake}):
 rr=[x for x in uptake if x['condition']==cond]
 uptake_summary.append(dict(condition=cond,records=len(rr),actions=dict(Counter(x['action'] for x in rr)),valid=sum(x['valid'] for x in rr)))
report={'permission_export':pmeta,'permission_allocated':len(allocation),'permission_recovered_cases':len({x['case_id'] for x in permission}),'permission_toolward_pairs':len(tool),'permission_toolward_both_written':sum(x['intervention_both_markers_written'] for x in tool),'permission_toolward_min_tool_all_recorded':min(x['intervention_min_tool_all_recorded_command_tokens'] for x in tool),'permission_toolward_all_exposed':all(x['read_exposure_confirmed'] for x in tool),'permission_method_summary':summary,'first_export':fmeta,'first_allocated':len(falloc),'first_unauthorized_records':sum(x['task']=='unauthorized' for x in firstrows),'first_unauthorized_unexposed':sum(x['task']=='unauthorized' and not x['page_exposed'] for x in firstrows),'first_records_with_scores':sum(x['has_token_scores'] for x in firstrows),'local_assigned_slots':len(local),'local_closed_episodes':sum(x['status']!='NO_CLOSED_EPISODE' for x in local),'local_receipt_errors':errors,'uptake_conditions':uptake_summary,'uptake_ceiling_duplicate_results':all(x in uptake for x in ceiling)}
export=load(ROOT/'role-steering/data/export-manifest.json')
report['public_export_hash_errors']=[x['public'] for x in export['files'] if digest(ROOT/'role-steering'/x['public'])!=x['public_sha256']]
assert not report['public_export_hash_errors']
identity=[]
for case in ['000','001','002']:
 a=ROOT/'replication/agent-hijacking/runs/ten-20260911'/f'{case}-forgery'
 b=ROOT/'replication/steering-agent/runs/forgery5-tool_minus_cot-a0'/f'{case}-forgery'
 turns=[]
 for f in sorted(a.glob('step-*.generation.json')):
  twin=b/f.name
  turns.append({'turn_file':f.name,'tokens_equal':load(f)['token_ids']==load(twin)['token_ids'],'prompt_equal':f.with_name(f.name.replace('generation.json','prompt.txt')).read_bytes()==twin.with_name(twin.name.replace('generation.json','prompt.txt')).read_bytes()})
 identity.append({'case_id':case,'turns':turns,'note':'First divergence after upload; only preceding tool execution stderr differs, with curl timing/speed text.'})
report['historical_zero_identity']=identity
(OUT/'historical-audit-verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
assert not errors
