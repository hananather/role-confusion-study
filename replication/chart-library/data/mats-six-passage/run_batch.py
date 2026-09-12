"""I measure twelve authored six-passage illustrations and retain every candidate."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
import sys,json,hashlib,time,atexit
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=ROOT/'cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z'
MODEL=Path.home()/'.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3'
TOKENIZER=Path.home()/'.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee'
sys.path.insert(0,str(ROOT.parent/'prompt-injection-as-role-confusion'))
from utils.substring_assignments import flag_message_types
CLASSES=['system','user','cot','assistant']
PASSAGES=['system','user','cot','assistant','user','cot','assistant']
CONDITIONS=['proper_tags','basic_no_format','everything_in_user_tags']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def prepare():
 if (HERE/'input-freeze.json').exists():raise RuntimeError('Inputs already frozen; refusing overwrite')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True,add_eos_token=False,add_bos_token=False)
 tok.chat_template=(SOURCE/'appendix-e-L12/custom-chat-template.jinja').read_text()
 system=json.loads((SOURCE/'appendix-e-L12/original-messages.json').read_text())[0]['content']
 cases=json.loads((HERE/'candidates.json').read_text());assert len(cases)==12
 frozen=[]
 for case in cases:
  ident=case['id'];assert ident.startswith('mats-') and len(case['messages'])==6
  assert [x['role'] for x in case['messages']]==PASSAGES[1:]
  dest=HERE/ident;dest.mkdir(exist_ok=False)
  texts=[system]+[x['content'] for x in case['messages']]
  chat=[{'role':'system','content':system}]
  for i in [1,4]:
   chat.extend([{'role':'user','content':texts[i]},{'role':'assistant','content':'<think>'+texts[i+1]+'</think>'+texts[i+2]}])
  joined='\n'.join(texts);prefix=tok.bos_token or ''
  formatted={'proper_tags':prefix+tok.apply_chat_template(chat,tokenize=False,add_generation_prompt=False),'basic_no_format':prefix+joined,'everything_in_user_tags':prefix+tok.apply_chat_template([{'role':'user','content':joined}],tokenize=False,add_generation_prompt=False)}
  tokenrows=[];records=[]
  for pi,condition in enumerate(CONDITIONS):
   text=formatted[condition];enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);last=0
   for ti,((a,b),tid) in enumerate(zip(enc.offset_mapping,enc.input_ids)):
    begin=max(a,last);piece=text[begin:b] if begin<b else '';last=max(last,b)
    tokenrows.append(dict(prompt_ix=pi,prompt_key=condition,token_ix=ti,token=piece,token_id=tid))
   assert len(enc.input_ids)<=2048
   records.append(dict(prompt_ix=pi,prompt_key=condition,text=text,token_ids=enc.input_ids,n_tokens=len(enc.input_ids)))
  df=flag_message_types(pd.DataFrame(tokenrows),texts).drop(columns=['base_message'])
  df['base_message_type']=df.base_message_ix.map(dict(enumerate(PASSAGES)))
  df['segment_token_ix']=df.groupby(['prompt_ix','base_message_ix']).cumcount()+1
  df['sample_ix']=np.arange(len(df));df['seg_ix']=df.base_message_ix
  write(dest/'original-messages.json',texts);write(dest/'rendered-prompts-and-token-ids.json',records)
  df.to_csv(dest/'all-tokens-and-labels.csv',index=False)
  # I use exactly the earlier first-120-token-per-passage display rule.
  eligible=df[df.base_message_type.isin(CLASSES[1:]) & (df.segment_token_ix<=120)].copy()
  keys=['base_message_ix','segment_token_ix']
  matching=eligible.groupby(keys).agg(n=('prompt_key','nunique'),text_count=('token','nunique'),id_count=('token_id','nunique')).query('n==3 and text_count==1 and id_count==1').reset_index()[keys]
  shown=eligible.merge(matching,on=keys).sort_values(['prompt_ix','token_ix'])
  shown['display_token_ix']=shown.groupby('prompt_ix').cumcount()+1
  assert shown.groupby('prompt_key').size().nunique()==1
  assert set(shown.base_message_ix)==set(range(1,7))
  shown.to_csv(dest/'display-index.csv',index=False)
  write(dest/'display-selection.json',{'rule':'First 120 tokens per non-System passage, then identical token text and token ID at matching passage-relative position in all three formats. Chosen before measurements.','full_content_tokens':df[df.base_message_type.isin(CLASSES[1:])].groupby('prompt_key').size().to_dict(),'display_tokens':shown.groupby('prompt_key').size().to_dict(),'display_segment_counts':shown[shown.prompt_key=='proper_tags'].groupby('base_message_ix').size().to_dict()})
  files=['original-messages.json','rendered-prompts-and-token-ids.json','all-tokens-and-labels.csv','display-index.csv','display-selection.json']
  frozen.append({'id':ident,'title':case['title'],'files':{f:sha(dest/f) for f in files},'forward_tokens':[r['n_tokens'] for r in records],'display_tokens_each':len(shown)//3})
 write(HERE/'input-freeze.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'stage':'Distill','north_star':'A readable paper-style example of the role-probe measurement; no population inference from selection.','criteria':'The user requested multiple examples and visual selection. I preserve all 12 outcomes and label the featured plot as selected after inspection. Clarity, role-band separation, readable text and six-passage structure guide the choice.','authorship':'All candidates are authored illustrative conversations, not real conversations with Neel or model-generated outputs. No quotation of Neel is claimed.','n_candidates':len(cases),'candidate_source_sha256':sha(HERE/'candidates.json'),'candidates':frozen})
 print('Frozen',[(x['id'],x['display_tokens_each'],max(x['forward_tokens'])) for x in frozen],flush=True)
def run():
 import importlib.metadata
 freeze=json.loads((HERE/'input-freeze.json').read_text())
 assert sha(HERE/'candidates.json')==freeze['candidate_source_sha256']
 for case in freeze['candidates']:
  for f,digest in case['files'].items():assert sha(HERE/case['id']/f)==digest
  assert not (HERE/case['id']/'token-probabilities.csv').exists()
 lock=ROOT/'MODEL-IN-USE.md';ownership=f'# Local MATS six-passage batch\n\nPID {os.getpid()}; 12 authored candidates, 36 short forwards, one model.\n'
 with lock.open('x') as h:h.write(ownership)
 def release():
  if lock.exists() and lock.read_text()==ownership:lock.unlink()
 atexit.register(release)
 import mlx.core as mx
 import mlx.nn as nn
 from mlx_lm import load
 probe=SOURCE/'probes-full/probes.npz';z=np.load(probe);w=z['suca_L12__coef'].astype(np.float64);b=z['suca_L12__intercept'].astype(np.float64)
 assert w.shape==(4,2880)
 start=time.time();model,_=load(str(MODEL));mx.eval(model.parameters());store={}
 class Recorder(nn.Module):
  def __init__(self,wrapped):super().__init__();self.wrapped=wrapped
  def __call__(self,x):
   y=self.wrapped(x);store['h']=y;return y
 model.model.layers[12].post_attention_layernorm=Recorder(model.model.layers[12].post_attention_layernorm)
 summaries=[];timings=[];dtypes=set()
 for case in freeze['candidates']:
  dest=HERE/case['id'];records=json.loads((dest/'rendered-prompts-and-token-ids.json').read_text());states=[]
  for rec in records:
   t=time.time();out=model(mx.array([rec['token_ids']]));mx.eval(out,store['h']);dtypes.add(str(store['h'].dtype))
   h=np.array(store['h'][0].astype(mx.float32)).astype(np.float16)
   assert h.shape==(rec['n_tokens'],2880) and np.isfinite(h).all();states.append(h)
   dt=time.time()-t;timings.append({'candidate':case['id'],'condition':rec['prompt_key'],'tokens':len(h),'seconds':dt})
   print(case['id'],rec['prompt_key'],len(h),round(dt,2),flush=True)
   del out;store.clear();mx.clear_cache()
  h=np.concatenate(states);np.save(dest/'layer12-float16.npy',h)
  logits=h.astype(np.float64)@w.T+b;logits-=logits.max(axis=1,keepdims=True);p=np.exp(logits);p/=p.sum(axis=1,keepdims=True)
  np.save(dest/'probabilities.npy',p)
  df=pd.read_csv(dest/'all-tokens-and-labels.csv',keep_default_na=False)
  for col in ['base_message_ix','segment_token_ix','seg_ix']:df[col]=pd.to_numeric(df[col],errors='coerce')
  for j,role in enumerate(CLASSES):df['p_'+role]=p[:,j]
  df['prob']=df.p_cot;df.to_csv(dest/'token-probabilities.csv',index=False)
  ix=pd.read_csv(dest/'display-index.csv',keep_default_na=False)
  ix['prob']=p[ix.sample_ix.astype(int),2]
  for j,role in enumerate(CLASSES):ix['p_'+role]=p[ix.sample_ix.astype(int),j]
  ix['layer_ix']=12;ix['target_role']='cot';ix.to_csv(dest/'displayed-rows.csv',index=False)
  means=ix.groupby(['prompt_key','base_message_ix','base_message_type']).prob.agg(['mean','size']).reset_index();means['candidate']=case['id'];summaries.extend(means.to_dict('records'))
  write(dest/'result.json',{'candidate':case['id'],'n_display_tokens_each':len(ix)//3,'states_sha256':sha(dest/'layer12-float16.npy'),'probabilities_sha256':sha(dest/'probabilities.npy'),'display_sha256':sha(dest/'displayed-rows.csv')})
 pd.DataFrame(summaries).to_csv(HERE/'all-passage-means.csv',index=False)
 write(HERE/'provenance.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'n_candidates':12,'n_model_forwards':len(timings),'elapsed_seconds':time.time()-start,'peak_memory_gb':mx.get_peak_memory()/1e9,'model_snapshot':MODEL.name,'model_id':'mlx-community/gpt-oss-20b-MXFP4-Q8','tokenizer_snapshot':TOKENIZER.name,'probe_sha256':sha(probe),'probe_key':'suca_L12','class_order':CLASSES,'hook':'zero-based layer12 post_attention_layernorm output, pre-MLP','runtime':'MLX','activation_dtypes':sorted(dtypes),'storage_dtype':'float16','projection_dtype':'float64','batch_size':1,'no_generation':True,'input_freeze_sha256':sha(HERE/'input-freeze.json'),'script_sha256':sha(__file__),'versions':{x:importlib.metadata.version(x) for x in ['mlx','mlx-lm','numpy','transformers']},'forwards':timings,'interpretation':'Twelve authored candidate illustrations selected for display after measurement. Not an independent sample for population inference. Existing gardening runtime control applies to this same model/probe configuration; it does not establish calibration.'})
 print('Completed; model releases on process exit.',flush=True)
if __name__=='__main__':
 if '--prepare' in sys.argv:prepare()
 else:run()
