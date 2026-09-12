"""I measure a prespecified authored MATS example, plus a matched gardening control."""
import os
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
import json, hashlib, sys, time, argparse, atexit
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
REPO=ROOT.parent/'prompt-injection-as-role-confusion'
SOURCE=ROOT/'cloud/persistent/sessions/20260911T030050Z/outputs/20260911T030050Z'
MODEL=Path.home()/'.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3'
TOKENIZER=Path.home()/'.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/6cee5e81ee83917806bbde320786a8fb61efebee'
sys.path.insert(0,str(REPO))
from utils.substring_assignments import flag_message_types
ROLES=['system','user','cot','assistant']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v): Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
def prepare():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(TOKENIZER),local_files_only=True,add_eos_token=False,add_bos_token=False)
    tok.chat_template=(SOURCE/'appendix-e-L12/custom-chat-template.jinja').read_text()
    messages=[json.loads((SOURCE/'appendix-e-L12/original-messages.json').read_text())[0]['content'],
      "For a MATS project with Neel, how can I turn a broad idea into one clear research question?",
      "The user is asking how to scope a research project. I should connect the choice of experiments to the claim they want to test. A small project can be informative if it has a clear baseline and a result that would change the conclusion. I should recommend one question first, then explain when more experiments are useful.",
      "Start with one question you can answer clearly. Write down your claim, a simple baseline, and what evidence would change your mind. Run the smallest experiment that distinguishes the competing explanations. Add experiments when they resolve a specific uncertainty, and report results that contradict your initial idea too."]
    if not isinstance(messages[0],str):
        raise TypeError('Gardening system message is not text')
    joined='\n'.join(messages); prefix=tok.bos_token or ''
    prompts={'basic_no_format':prefix+joined,'everything_in_user_tags':prefix+tok.apply_chat_template([{'role':'user','content':joined}],tokenize=False,add_generation_prompt=False),'proper_tags':prefix+tok.apply_chat_template([{'role':'system','content':messages[0]},{'role':'user','content':messages[1]},{'role':'assistant','content':'<think>'+messages[2]+'</think>'+messages[3]}],tokenize=False,add_generation_prompt=False)}
    rows=[]; records=[]
    for pi,(condition,text) in enumerate(prompts.items()):
        enc=tok(text,add_special_tokens=False,return_offsets_mapping=True); last=0
        for ti,((a,b),tid) in enumerate(zip(enc.offset_mapping,enc.input_ids)):
            emit=max(a,last); piece=text[emit:b] if emit<b else ''; last=max(last,b)
            rows.append(dict(prompt_ix=pi,prompt_key=condition,token_ix=ti,token=piece,token_id=tid))
        records.append(dict(prompt_ix=pi,prompt_key=condition,text=text,token_ids=enc.input_ids,n_tokens=len(enc.input_ids)))
    labels=flag_message_types(pd.DataFrame(rows),messages)
    labels['base_message_type']=labels.base_message_ix.map(dict(enumerate(ROLES)))
    labels['sample_ix']=np.arange(len(labels))
    labels['segment_token_ix']=labels.groupby(['prompt_ix','base_message_ix']).cumcount()+1
    write(HERE/'original-messages.json',messages); write(HERE/'rendered-prompts-and-token-ids.json',records)
    labels.to_csv(HERE/'all-tokens-and-labels.csv',index=False)
    write(HERE/'input-freeze.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),stage='Distill: original illustration',north_star='I illustrate the measurement machinery on one new authored conversation, without selecting on probe results.',authorship='Coding-agent-authored illustrative conversation; neither a real conversation with Neel nor generated model reasoning. Neel reference is a paraphrase of supplied writing guidance, not a quotation.',prediction='I expect the tagged analysis passage to receive higher CoT scores than the question and final answer. Untagged conditions may retain some differentiation; all outcomes are retained.',selection='One fixed example; all non-system content tokens retained in full CSV; display only exact token matches at same message-relative position across all three formats, no truncation.',input_sha256={p.name:sha(p) for p in [HERE/'original-messages.json',HERE/'rendered-prompts-and-token-ids.json',HERE/'all-tokens-and-labels.csv']},n_tokens=[r['n_tokens'] for r in records]))
    print('Prepared',[(r['prompt_key'],r['n_tokens']) for r in records],flush=True)
def run():
    if (HERE/'token-probabilities.csv').exists(): raise RuntimeError('Results already exist; refusing overwrite')
    freeze=json.loads((HERE/'input-freeze.json').read_text())
    for name,digest in freeze['input_sha256'].items(): assert sha(HERE/name)==digest
    lock=ROOT/'MODEL-IN-USE.md'
    if lock.exists(): raise RuntimeError('MODEL-IN-USE.md exists; I will not load another model copy')
    ownership=f'# Local MATS illustration measurement\n\nPID {os.getpid()}; six bounded forwards, one model.\n'
    with lock.open('x') as handle: handle.write(ownership)
    def release():
        if lock.exists() and lock.read_text()==ownership: lock.unlink()
    atexit.register(release)
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load
    import importlib.metadata
    records=json.loads((HERE/'rendered-prompts-and-token-ids.json').read_text())
    garden=json.loads((SOURCE/'appendix-e-L12/rendered-prompts-and-token-ids.json').read_text())
    assert max(r['n_tokens'] for r in records+garden)<=2048
    probe=SOURCE/'probes-full/probes.npz'; z=np.load(probe); w=z['suca_L12__coef'].astype(np.float64); b=z['suca_L12__intercept'].astype(np.float64)
    assert w.shape==(4,2880) and b.shape==(4,)
    start=time.time(); model,_=load(str(MODEL)); mx.eval(model.parameters()); inner=model.model; store={}
    class Recorder(nn.Module):
        def __init__(self,wrapped): super().__init__(); self.wrapped=wrapped
        def __call__(self,x):
            y=self.wrapped(x); store['h']=y; return y
    inner.layers[12].post_attention_layernorm=Recorder(inner.layers[12].post_attention_layernorm)
    summaries=[]; observed_dtypes=[]
    for name,items in [('mats',records),('gardening-control',garden)]:
        states=[]
        for rec in items:
            t=time.time(); output=model(mx.array([rec['token_ids']])); mx.eval(output,store['h'])
            observed_dtypes.append(str(store['h'].dtype)); h=np.array(store['h'][0].astype(mx.float32)); assert h.shape==(rec['n_tokens'],2880) and np.isfinite(h).all()
            states.append(h.astype(np.float16)); print(name,rec['prompt_key'],h.shape,round(time.time()-t,2),flush=True)
            del output; store.clear(); mx.clear_cache()
        h=np.concatenate(states); np.save(HERE/(name+'-layer12-float16.npy'),h)
        logits=h.astype(np.float64)@w.T+b; logits-=logits.max(1,keepdims=True); p=np.exp(logits); p/=p.sum(1,keepdims=True)
        np.save(HERE/(name+'-probabilities.npy'),p)
        if name=='gardening-control':
            native=np.load(SOURCE/'appendix-e-L12/native-probabilities-unrounded.npy')
            index=np.load(SOURCE/'appendix-e-L12/native-probability-sample-indices.npy')
            original_states=np.load(SOURCE/'appendix-e-L12/gardening-layer12-float16.npy')
            assert original_states.shape==h.shape
            native_table=pd.read_csv(SOURCE/'appendix-e-L12/all-tokens-and-labels.csv')
            assert native_table.token_id.tolist()==[tid for record in items for tid in record['token_ids']]
            diff=p[index]-native
            write(HERE/'gardening-runtime-comparison.json',dict(scope='Same recorded unpadded token IDs and retained content tokens; backend, weight precision and H100 batching differ. Descriptive compatibility check only.',n_content_tokens=len(index),mean_absolute_probability_difference_pp={role:float(np.mean(np.abs(diff[:,j]))*100) for j,role in enumerate(ROLES)},max_absolute_probability_difference_pp=float(np.max(np.abs(diff))*100),mean_difference_pp={role:float(np.mean(diff[:,j])*100) for j,role in enumerate(ROLES)}))
        if name=='mats':
            df=pd.read_csv(HERE/'all-tokens-and-labels.csv',keep_default_na=False)
            for column in ['base_message_ix','segment_token_ix']: df[column]=pd.to_numeric(df[column],errors='coerce')
            for j,role in enumerate(ROLES): df[role]=p[:,j]
            df.to_csv(HERE/'token-probabilities.csv',index=False)
            content=df[df.base_message_type.isin(ROLES[1:])].copy(); content['segment_token_ix']=content.segment_token_ix.astype(int)
            keys=['base_message_type','segment_token_ix']; common=content.groupby(keys).agg(n=('prompt_key','nunique'),tokens=('token','nunique')).query('n==3 and tokens==1').reset_index()[keys]
            shown=content.merge(common,on=keys).sort_values(['prompt_ix','token_ix']); shown['display_token_ix']=shown.groupby('prompt_ix').cumcount()+1; shown['prob']=shown['cot']; shown['seg_ix']=shown['base_message_ix'].astype(int)
            shown.to_csv(HERE/'displayed-rows.csv',index=False)
            summaries=content.groupby(['prompt_key','base_message_type'])[ROLES].mean().reset_index().to_dict('records')
            write(HERE/'display-selection.json',dict(full_content_tokens=content.groupby('prompt_key').size().to_dict(),displayed_tokens=shown.groupby('prompt_key').size().to_dict(),omitted_token_records=content.merge(common,on=keys,how='left',indicator=True).query('_merge=="left_only"')[['prompt_key','token_ix','token','base_message_type']].to_dict('records')))
    write(HERE/'provenance.json',dict(completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-start,model_id='mlx-community/gpt-oss-20b-MXFP4-Q8',model_snapshot=MODEL.name,tokenizer_snapshot=TOKENIZER.name,probe_sha256=sha(probe),probe_key='suca_L12',class_order=ROLES,hook='zero-based layer 12 post_attention_layernorm output (pre-MLP)',runtime='MLX',batch_size=1,no_generation=True,activation_runtime_dtypes=sorted(set(observed_dtypes)),activations='Stored float16; projected float64 using saved float32 coefficients',divergences=['MLX kernels and 8-bit non-expert weights differ from native H100 experiment.','Batch 1 no padding; H100 control used padded batch. Cross-runtime comparison is descriptive, not calibration validation.'],versions={k:importlib.metadata.version(k) for k in ['mlx','mlx-lm','numpy','transformers']},peak_memory_gb=mx.get_peak_memory()/1e9,mean_probabilities=summaries,script_sha256=sha(__file__)))
if __name__=='__main__':
    if '--prepare' in sys.argv: prepare()
    else: run()
