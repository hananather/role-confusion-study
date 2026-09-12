"""I stage and launch only this reviewed successor after verified prior closure; no provider calls."""
from __future__ import annotations
import argparse
import base64
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from prepare_inputs import PACKET, save, sha
from run_queue import verify_packet, remaining
from replication.cloud.agent_steering.dispatcher import register_sync

def ssh(connection, program, data):
    command = ['ssh','-i',connection['key'],'-p',str(connection['port']),'-o','BatchMode=yes','-o','ConnectTimeout=10',
        '-o','StrictHostKeyChecking=accept-new','root@'+connection['host'],'python3 -c '+shlex.quote(program)]
    p = subprocess.run(command, input=json.dumps(data).encode(), capture_output=True, timeout=45, check=True)
    return json.loads(p.stdout) if p.stdout.strip() else None

PRECHECK = '''
import json,os,pathlib,sys
p=json.load(sys.stdin)
if os.environ.get('RUNPOD_POD_ID') not in (None,p['pod']): raise ValueError('Wrong pod')
try: os.kill(p['predecessor'],0)
except ProcessLookupError: pass
else: raise ValueError('Old worker is still present; no replacement is started')
for name in ('root','service','results'):
 if pathlib.Path(p[name]).exists(): raise ValueError('New successor path already exists: '+name)
print(json.dumps({'passed':True,'predecessor_absence_verified':True}))
'''

BOOTSTRAP = '''
import base64,hashlib,json,os,pathlib,subprocess,sys,time
p=json.load(sys.stdin);root=pathlib.Path(p['root'])
for rel,digest in p['files'].items():
 f=root/rel
 if f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest()!=digest: raise ValueError('Staged bytes changed: '+rel)
cfgpath=root/'registration/service-config.json'; cfg=json.loads(cfgpath.read_bytes())
if hashlib.sha256(cfgpath.read_bytes()).hexdigest()!=p['service_sha']:raise ValueError('Service config changed')
if time.time()>=p['launch_cutoff']:raise ValueError('Launch cutoff passed')
try:os.kill(cfg['predecessor_pid'],0)
except ProcessLookupError:pass
else:raise ValueError('Old worker unexpectedly present')
lease=json.loads(pathlib.Path(cfg['lease_path']).read_bytes());now=time.time()
if lease.get('approved') is not True or lease.get('expected_pod_id')!=cfg['expected_pod_id'] or now>=min(lease['lease_expires_unix'],lease['shutdown_at_unix'],p['closeout']):raise ValueError('Expired lease')
jobraw=(root/'registration/registration.json').read_bytes()
if hashlib.sha256(jobraw).hexdigest()!=p['job_sha']:raise ValueError('Job config changed')
job=json.loads(jobraw);service=pathlib.Path(cfg['out_dir']);service.mkdir(parents=True,exist_ok=False)
(service/'jobs').mkdir();target=service/'jobs'/(job['job_id']+'.json');tmp=target.with_suffix('.uploading');tmp.write_bytes(jobraw);os.replace(tmp,target)
env=dict(os.environ)
for key in list(env):
 if 'API_KEY' in key or key.endswith('HF_TOKEN') or key in ('HUGGING_FACE_HUB_TOKEN','HUGGINGFACE_TOKEN','HF_HUB_CACHE','PYTHONPATH','PYTHONHOME'):env.pop(key,None)
env.update(HF_HOME='/workspace/hf/home',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false')
argv=['/workspace/venv-probes/bin/python','-u','-m','replication.cloud.agent_steering.service_guardian','--config',str(cfgpath),'--state-dir',str(service/'guardian')]
with (service/'guardian-console.log').open('xb') as log:
 child=subprocess.Popen(argv,cwd=str(root/'source'),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'guardian_pid':child.pid,'service':str(service),'job_id':job['job_id'],'allocation_action':'none'}))
'''

STATUS = '''
import json,pathlib,sys
p=json.load(sys.stdin);s=pathlib.Path(p['service']);r=pathlib.Path(p['results'])
def read(f):return json.loads(f.read_bytes()) if f.exists() else None
print(json.dumps({'guardian':read(s/'guardian/GUARDIAN-READY.json'),'service':read(s/'SERVICE-READY.json'),'job':read(r/'READY.json'),
 'terminal':[read(f) for f in (s/'guardian/GUARDIAN-EXIT.json',s/'SERVICE-EXIT.json',s/'FAILED.json',r/'FAILED.json',r/'EXIT.json')]}))
'''

def launch(registration):
    registration=Path(registration).resolve();spec=json.loads((registration/'launch.json').read_text())
    config=json.loads((registration/'local-config.json').read_text())
    if sha(registration/'local-config.json')!=spec['local_config_sha256']:raise ValueError('Local config changed')
    plan,job=verify_packet(config)
    if time.time()>=plan['launch_cutoff_unix']:raise ValueError('No new item may start after05:08:01UTC')
    remaining(config,plan)
    state=registration/'launch-state';state.mkdir(exist_ok=False)
    save(state/'INTENT.json',{'launch_spec_sha256':sha(registration/'launch.json'),'plan_sha256':config['plan_sha256'],'unix':time.time()})
    connection=json.loads(Path(config['connection_file']).read_text());service=json.loads((registration/'service-config.json').read_text())
    data={'root':spec['remote_root'],'service':service['out_dir'],'results':job['out_dir'],'pod':config['expected_pod_id'],'predecessor':service['predecessor_pid']}
    save(state/'PRECHECK.json',ssh(connection,PRECHECK,data))
    shell=['ssh','-i',connection['key'],'-p',str(connection['port']),'-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','ConnectTimeout=10']
    endpoint='root@'+connection['host']+':'+spec['remote_root']+'/'
    subprocess.run(['rsync','-rlt','--exclude=__pycache__','--exclude=*.pyc','--timeout=30','-e',shlex.join(shell),str(PACKET)+'/',endpoint],check=True,timeout=120)
    subprocess.run(['rsync','-rlt','--exclude=launch-state','--timeout=30','-e',shlex.join(shell),str(registration)+'/',endpoint+'registration/'],check=True,timeout=120)
    files={str(p.relative_to(PACKET)):sha(p) for p in PACKET.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc'}
    payload={**data,'files':files,'service_sha':spec['service_config_sha256'],'job_sha':config['registration_sha256'],'launch_cutoff':plan['launch_cutoff_unix'],'closeout':plan['session_closeout_unix']}
    verify_packet(config)
    if sha(registration/'local-config.json')!=spec['local_config_sha256']:raise ValueError('Local config changed during staging')
    receipt=ssh(connection,BOOTSTRAP,payload);save(state/'REMOTE-LAUNCHED.json',receipt)
    deadline=min(time.time()+300,plan['launch_cutoff_unix'])
    while time.time()<deadline:
        remaining(config,plan);status=ssh(connection,STATUS,data)
        if any(r is not None for r in status['terminal']):save(state/'REMOTE-TERMINAL.json',status);raise RuntimeError('Successor terminated before readiness')
        if status['job']:
            g,s,j=status['guardian'],status['service'],status['job']
            if (not g or not s or g['guardian_pid']!=receipt['guardian_pid'] or g['worker_pid']!=s['pid'] or s['pid']!=j['pid']
                or g['config_sha256']!=spec['service_config_sha256'] or j['job_file_sha256']!=config['registration_sha256']):raise ValueError('Successor readiness identity mismatch')
            for key in ('directions_file_sha256','probe_file_sha256'):
                if j['backend'][key]!=job[key] or s['backend'][key]!=job[key]:raise ValueError('Loaded asset mismatch')
            save(state/'REMOTE-READY.json',status);break
        time.sleep(5)
    else:raise TimeoutError('No ready successor within startup/cutoff window')
    if time.time()>=plan['launch_cutoff_unix']:raise ValueError('Cutoff reached before local runner')
    verify_packet(config)
    if sha(registration/'local-config.json')!=spec['local_config_sha256']:raise ValueError('Local config changed during readiness')
    register_sync(config)
    argv=[str(PACKET.parents[3]/'.venv/bin/python'),'-u',str(PACKET/'code/run_queue.py'),'--config',str(registration/'local-config.json')]
    with (state/'local-console.log').open('xb') as log:
        child=subprocess.Popen(argv,cwd=str(PACKET),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(state/'LAUNCHED.json',{'local_runner_pid':child.pid,'remote':receipt,'unix':time.time(),'allocation_action':'none'})
    print(json.dumps({'local_runner_pid':child.pid,'status':'launched','state':str(state)}))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--registration',required=True);launch(ap.parse_args().registration)
