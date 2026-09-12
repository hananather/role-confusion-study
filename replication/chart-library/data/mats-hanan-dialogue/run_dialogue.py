"""I generate a Hanan-specific MATS exchange, then measure its six passages."""
import os
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
import sys,json,re,time,hashlib,atexit
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SRC=ROOT/'cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z'
MODEL=Path.home()/'.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3'
TOKENIZER=Path.home()/'.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee'
sys.path.insert(0,str(ROOT.parent/'prompt-injection-as-role-confusion'))
from utils.substring_assignments import flag_message_types
CLASSES=['system','user','cot','assistant'];ROLES=['system','user','cot','assistant','user','cot','assistant']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 if (HERE/'prompt-freeze.json').exists():raise RuntimeError('Preserved run exists; refusing overwrite')
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True,add_eos_token=False,add_bos_token=False)
 tok.chat_template=(SRC/'appendix-e-L12/custom-chat-template.jinja').read_text()
 system=json.loads((SRC/'appendix-e-L12/original-messages.json').read_text())[0]['content']
 system=system.replace('Reasoning: medium','Reasoning: low')+'\nKeep each final answer concise: at most 100 words, without a long introduction.'
 users=["I'm Hanan, and I want to apply to Neel Nanda's MATS stream. I've been reading Prompt Injection as Role Confusion and experimenting with GPT-OSS-20B. Can you suggest a small project about whether steering the model's role representation changes which instructions it follows? I'd like one result worth showing Neel, even if it ruins my favourite hypothesis.",
 "Okay, suppose the Toolness probe shoots up but the model still follows the injected instruction. Is that a useful result, or have I built an expensive mood ring? What's the first control you would run?"]
 write(HERE/'prompt-freeze.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'system':system,'user_prompts':users,'authorship':'The two user prompts are authored in Hanan’s voice for this new illustration; they are not a historical transcript. The local model will generate both analysis passages and final answers. No Neel quotation or actual admission outcome is claimed.','stage':'Distill','north_star':'An original, personal MATS conversation in the same six-passage display structure.','selection':'One new exchange; retain both complete responses and all measured scores. No search over generated outcomes.','generation':{'temperature':0.0,'max_new_tokens_per_turn':1536,'timeout_seconds_per_turn':180,'reasoning_effort':'low','final_answer_instruction':'at most 100 words'}})
 lock=ROOT/'MODEL-IN-USE.md';ownership=f'# Local Hanan MATS dialogue\n\nPID {os.getpid()}; two bounded generations followed by three short measurement forwards.\n'
 with lock.open('x') as f:f.write(ownership)
 def release():
  if lock.exists() and lock.read_text()==ownership:lock.unlink()
 atexit.register(release)
 import mlx.core as mx
 import mlx.nn as nn
 from mlx_lm import load,stream_generate
 from mlx_lm.sample_utils import make_sampler
 import importlib.metadata
 start=time.time();model,mlx_tok=load(str(MODEL));mx.eval(model.parameters())
 chat=[{'role':'system','content':system}];texts=[system];generated=[]
 for turn,user in enumerate(users,1):
  chat.append({'role':'user','content':user});texts.append(user)
  prompt=(tok.bos_token or '')+tok.apply_chat_template(chat,tokenize=False,add_generation_prompt=True)+'<|channel|>analysis<|message|>'
  ids=tok.encode(prompt,add_special_tokens=False)
  assert ids==mlx_tok.encode(prompt,add_special_tokens=False)
  begun=time.time();tokens=[];finish=None
  stream=stream_generate(model,mlx_tok,prompt=ids,max_tokens=1536,sampler=make_sampler(temp=0.0))
  for r in stream:
   tokens.append(int(r.token));finish=r.finish_reason
   if len(tokens)%128==0:print('generation',turn,len(tokens),round(time.time()-begun,1),flush=True)
   if finish:break
   if time.time()-begun>180:finish='timeout';break
  stream.close()
  raw='<|start|>assistant<|channel|>analysis<|message|>'+tok.decode(tokens,skip_special_tokens=False)
  (HERE/f'turn-{turn}-raw.txt').write_text(raw)
  write(HERE/f'turn-{turn}-generation.json',{'turn':turn,'prompt':prompt,'prompt_token_ids':ids,'response_token_ids':tokens,'finish_reason':finish,'seconds':time.time()-begun})
  analysis=re.search(r'<\|channel\|>analysis<\|message\|>(.*?)<\|end\|>',raw,re.S)
  final=re.search(r'<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|fim_suffix\|>|<\|end\|>|$)',raw,re.S)
  if not analysis or not final or finish in ['length','timeout']:raise RuntimeError(f'Turn {turn} lacks complete analysis and final; retained raw output')
  thought=analysis.group(1);answer=final.group(1)
  assert thought.strip() and answer.strip()
  texts.extend([thought,answer]);chat.append({'role':'assistant','content':'<think>'+thought+'</think>'+answer})
  generated.append({'turn':turn,'analysis':thought,'final':answer,'tokens':len(tokens),'seconds':time.time()-begun,'finish_reason':finish})
  print('Completed generation',turn,len(tokens),finish,flush=True)
  mx.clear_cache()
 write(HERE/'original-messages.json',texts);write(HERE/'generated-conversation.json',generated)
 joined='\n'.join(texts);prefix=tok.bos_token or ''
 prompts={'proper_tags':prefix+tok.apply_chat_template(chat,tokenize=False,add_generation_prompt=False),'basic_no_format':prefix+joined,'everything_in_user_tags':prefix+tok.apply_chat_template([{'role':'user','content':joined}],tokenize=False,add_generation_prompt=False)}
 tokenrows=[];records=[]
 for pi,(condition,text) in enumerate(prompts.items()):
  enc=tok(text,add_special_tokens=False,return_offsets_mapping=True);last=0
  for ti,((a,b),tid) in enumerate(zip(enc.offset_mapping,enc.input_ids)):
   begin=max(a,last);piece=text[begin:b] if begin<b else '';last=max(last,b)
   tokenrows.append(dict(prompt_ix=pi,prompt_key=condition,token_ix=ti,token=piece,token_id=tid))
  assert len(enc.input_ids)<=4096
  records.append(dict(prompt_ix=pi,prompt_key=condition,text=text,token_ids=enc.input_ids,n_tokens=len(enc.input_ids)))
 df=flag_message_types(pd.DataFrame(tokenrows),texts).drop(columns=['base_message'])
 df['base_message_type']=df.base_message_ix.map(dict(enumerate(ROLES)))
 df['segment_token_ix']=df.groupby(['prompt_ix','base_message_ix']).cumcount()+1
 df['seg_ix']=df.base_message_ix;df['sample_ix']=np.arange(len(df))
 write(HERE/'rendered-prompts-and-token-ids.json',records);df.to_csv(HERE/'all-tokens-and-labels.csv',index=False)
 eligible=df[df.base_message_type.isin(CLASSES[1:])&(df.segment_token_ix<=120)]
 keys=['base_message_ix','segment_token_ix']
 matches=eligible.groupby(keys).agg(n=('prompt_key','nunique'),text_count=('token','nunique'),id_count=('token_id','nunique')).query('n==3 and text_count==1 and id_count==1').reset_index()[keys]
 shown=eligible.merge(matches,on=keys).sort_values(['prompt_ix','token_ix']);shown['display_token_ix']=shown.groupby('prompt_ix').cumcount()+1
 assert shown.groupby('prompt_key').size().nunique()==1 and set(shown.base_message_ix)==set(range(1,7))
 write(HERE/'measurement-freeze.json',{'frozen_utc':datetime.now(timezone.utc).isoformat(),'original_messages_sha256':sha(HERE/'original-messages.json'),'generation_outputs_retained':True,'display_rule':'First 120 tokens per non-System passage, then same text and token ID at matching passage-relative position in all three formats; no score-based selection.','full_content_tokens':df[df.base_message_type.isin(CLASSES[1:])].groupby('prompt_key').size().to_dict(),'display_tokens':shown.groupby('prompt_key').size().to_dict(),'display_segment_counts':shown[shown.prompt_key=='proper_tags'].groupby('base_message_ix').size().to_dict()})
 store={}
 class Recorder(nn.Module):
  def __init__(self,wrapped):super().__init__();self.wrapped=wrapped
  def __call__(self,x):
   y=self.wrapped(x);store['h']=y;return y
 model.model.layers[12].post_attention_layernorm=Recorder(model.model.layers[12].post_attention_layernorm)
 states=[];timings=[]
 for rec in records:
  begun=time.time();out=model(mx.array([rec['token_ids']]));mx.eval(out,store['h'])
  h=np.array(store['h'][0].astype(mx.float32)).astype(np.float16);assert np.isfinite(h).all();states.append(h)
  timings.append({'condition':rec['prompt_key'],'tokens':len(h),'seconds':time.time()-begun})
  print('measurement',rec['prompt_key'],len(h),round(time.time()-begun,2),flush=True)
  del out;store.clear();mx.clear_cache()
 h=np.concatenate(states);np.save(HERE/'layer12-float16.npy',h)
 probe=SRC/'probes-full/probes.npz';z=np.load(probe);w=z['suca_L12__coef'].astype(np.float64);b=z['suca_L12__intercept'].astype(np.float64)
 logits=h.astype(np.float64)@w.T+b;logits-=logits.max(axis=1,keepdims=True);p=np.exp(logits);p/=p.sum(axis=1,keepdims=True)
 np.save(HERE/'probabilities.npy',p)
 for j,role in enumerate(CLASSES):df['p_'+role]=p[:,j];shown['p_'+role]=p[shown.sample_ix.astype(int),j]
 df['prob']=df.p_cot;df.to_csv(HERE/'token-probabilities.csv',index=False)
 shown['prob']=shown.p_cot;shown['target_role']='cot';shown['layer_ix']=12;shown.to_csv(HERE/'displayed-rows.csv',index=False)
 shown.groupby(['prompt_key','base_message_ix','base_message_type']).prob.agg(['mean','size']).reset_index().to_csv(HERE/'passage-means.csv',index=False)
 write(HERE/'provenance.json',{'completed_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.time()-start,'peak_memory_gb':mx.get_peak_memory()/1e9,'model_id':'mlx-community/gpt-oss-20b-MXFP4-Q8','model_snapshot':MODEL.name,'tokenizer_snapshot':TOKENIZER.name,'probe_key':'suca_L12','probe_sha256':sha(probe),'class_order':CLASSES,'hook':'zero-based layer12 post_attention_layernorm output, pre-MLP','n_generated_assistant_turns':2,'n_measurement_forwards':3,'temperature':0.0,'generation_reasoning_effort':'low','answers_and_analysis':'Generated once by the local model; unchanged outputs replayed for measurement in three formats. User prompts authored for this illustration.','storage_dtype':'float16','projection_dtype':'float64','runtime':'MLX','versions':{x:importlib.metadata.version(x) for x in ['mlx','mlx-lm','numpy','transformers']},'measurement_forwards':timings,'prompt_freeze_sha256':sha(HERE/'prompt-freeze.json'),'measurement_freeze_sha256':sha(HERE/'measurement-freeze.json'),'script_sha256':sha(__file__),'interpretation':'One original Hanan-specific illustrative exchange, no claim about admissions or Neel’s judgment. Analysis text is the local model’s generated channel, not a verified explanation of mechanism.'})
 print('Complete; model releases on exit.',flush=True)
if __name__=='__main__':main()
