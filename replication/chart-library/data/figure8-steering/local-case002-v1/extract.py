"""I extract the first post-fetch response, preserving original token positions."""
from pathlib import Path
import json,re,hashlib,csv
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent
PLAN=json.loads((HERE/'input-freeze.json').read_text())
MODEL=Path.home()/'.cache/huggingface/hub/models--mlx-community--gpt-oss-20b-MXFP4-Q8/snapshots/773a7da77e569019bb0fd17a554b263738d669a3'
tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True)
write=lambda p,j:Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def sentence(text):
    text=text.strip()
    # I keep a complete opening sentence; file extensions are not sentence ends.
    m=re.search(r'[.!?](?=\s+[A-Z]|$)',text)
    return text[:m.end()] if m else text

def extract_arm(item):
    arm=item['arm'];folder=HERE/'results'/arm
    episode=json.loads((folder/'episode.json').read_text())
    prompt=(folder/'step-01.prompt.txt').read_text()
    gen=json.loads((folder/'step-01.generation.json').read_text())
    arr=np.load(folder/'readouts/step-01.readouts.npz')
    enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
    ids=list(enc.input_ids);assert ids==arr['token_ids'][:len(ids)].tolist()
    assert ids+gen['token_ids']==arr['token_ids'].tolist(), 'Main illustration requires a complete, aligned post-fetch turn'
    offsets=np.array(enc.offset_mapping)
    segments=[]
    def add_prompt(key,source,label,text):
        escaped=json.dumps(text,ensure_ascii=False)[1:-1] if source!='user' else text
        a=prompt.index(escaped);b=a+len(escaped)
        pos=np.flatnonzero((offsets[:,0]<b)&(offsets[:,1]>a)&(offsets[:,1]>offsets[:,0]))
        segments.append(dict(segment=key,source=source,label=label,excerpt=sentence(text),full_text=text,positions=pos.tolist()))
    add_prompt('user','user','User',item['case']['user_prompt'])
    head=item['case']['payload'].split('\n\n',1)[0]
    add_prompt('command','tool','Tool: injected request',head)
    if item['forged_reasoning']:
        label='Tool: destyled reasoning' if arm=='destyled' else 'Tool: forged CoT'
        add_prompt('forgery','forged_cot',label,item['forged_reasoning'])
    # I segment actual sampled IDs at Harmony delimiters, avoiding re-tokenization.
    g=gen['token_ids'];special={s:tok.convert_tokens_to_ids(s) for s in ['<|message|>','<|end|>','<|call|>','<|return|>','<|start|>']}
    start=0;count={'cot':0,'assistant':0}
    for i,t in enumerate(g):
        if t!=special['<|message|>']:continue
        header=tok.decode(g[start:i],skip_special_tokens=False)
        j=i+1
        while j<len(g) and g[j] not in [special[s] for s in ['<|end|>','<|call|>','<|return|>','<|start|>']]:j+=1
        source='cot' if re.search(r'<\|channel\|>analysis(?:\s|$)',header) and 'to=' not in header else 'assistant'
        count[source]+=1
        content=tok.decode(g[i+1:j],skip_special_tokens=False,clean_up_tokenization_spaces=False)
        label='Agent CoT' if source=='cot' else ('Assistant action' if 'to=' in header else 'Assistant reply')
        segments.append(dict(segment=source+str(count[source]),source=source,label=label,excerpt=sentence(content),full_text=content,positions=list(range(len(ids)+i+1,len(ids)+j))))
        start=j+1
    rows=[];x=0;manifest=[]
    for seg in segments:
        positions=seg.pop('positions');seg['full_tokens']=len(positions);seg['shown_tokens']=min(200,len(positions));manifest.append(seg)
        for ii,pos in enumerate(positions):
            row=dict(arm=arm,segment=seg['segment'],source=seg['source'],turn=1,original_token_position=pos,token_index=ii,token_id=int(arr['token_ids'][pos]),token_text=tok.decode([int(arr['token_ids'][pos])],clean_up_tokenization_spaces=False),is_displayed=ii<200,x=x+ii if ii<200 else '',raw_cotness=float(arr['p_suca_L16'][pos,2]),sucat16_cotness=float(arr['p_sucat_L16'][pos,2]),sucat16_toolness=float(arr['p_sucat_L16'][pos,4]),sucat12_cotness=float(arr['p_sucat_L12'][pos,2]),sucat12_toolness=float(arr['p_sucat_L12'][pos,4]))
            rows.append(row)
        x+=min(200,len(positions))
    title={'plain':'A · No forged reasoning (ordinary injection)','forgery':'B · CoT forgery','steered':'C · CoT forgery + Tool−CoT steering','destyled':'D · Destyled CoT forgery'}[arm]
    label='Dummy file uploaded' if episode['verified_dummy_upload'] else ('No upload; run completed' if not episode['censored'] else 'Censored; no verified upload')
    info=dict(arm=arm,title=title,outcome_label=label,outcome_note=f"Whole episode: {episode['status']}; {len(episode['steps'])} model turns. Receipt-verified upload: {episode['verified_dummy_upload']}.",segments=manifest)
    outcome={k:episode[k] for k in ['status','verified_dummy_upload','emitted_upload_attempt','censored','generated_tokens','elapsed_s']};outcome.update(arm=arm,turns=len(episode['steps']),final_output=next((s['parsed']['final_output'] for s in reversed(episode['steps']) if s.get('parsed',{}).get('final_output')),None))
    return rows,info,outcome

def main():
    allrows=[];arms=[];outcomes=[]
    for item in PLAN['arms']:
        rows,info,outcome=extract_arm(item);allrows+=rows;arms.append(info);outcomes.append(outcome)
    df=pd.DataFrame(allrows)
    for col,out in [('raw_cotness','cotness'),('sucat16_cotness','sucat16_cotness_smoothed'),('sucat12_cotness','sucat12_cotness_smoothed')]:
        df[out]=df.groupby(['arm','segment'],sort=False)[col].transform(lambda s:s.ewm(alpha=.5,adjust=True).mean())
    df.to_csv(HERE/'all-relevant-token-scores.csv',index=False)
    df[df.is_displayed].to_csv(HERE/'displayed-rows.csv',index=False)
    means=df.groupby(['arm','segment'],sort=False)[['raw_cotness','sucat16_cotness','sucat16_toolness','sucat12_cotness','sucat12_toolness']].mean().reset_index()
    means.to_csv(HERE/'segment-means.csv',index=False)
    manifest={'schema_version':1,'title':'CoT forgery and steering in a tool-using agent','subtitle':'GPT-OSS-20B · layer 16 · four-role probe · one selected page and seed','primary_probe':{'name':'suca_L16','layer':16,'roles':4},'arms':arms,'attack_texts':[{'title':'Injected request · present in all four conditions','source':'tool','text':PLAN['arms'][0]['case']['payload']},{'title':'Forged reasoning · added in B and C','source':'forged_cot','text':PLAN['arms'][1]['forged_reasoning']},{'title':'Destyled reasoning · added in D','source':'forged_cot','text':PLAN['arms'][3]['forged_reasoning']}],'footer':'First response after reading the page. Outcome labels refer to the complete episode.','methods_note':'First 200 tokens per displayed segment; EWMA α = 0.5 within each segment. The full HTML, earlier agent turn, System/Developer text and role tags are processed but omitted here. The four-role probe has no Tool class. Colors mark text origin, not a predicted class. Intervention: Tool−CoT at block 11 on tool-content prefill only, α = 16. This adapts Figure 8 to a tool-output attack.','selection':PLAN['selection'],'input_freeze_sha256':sha(HERE/'input-freeze.json'),'displayed_rows_sha256':sha(HERE/'displayed-rows.csv'),'outcomes':outcomes}
    write(HERE/'display-manifest.json',manifest);write(HERE/'outcomes.json',outcomes)
    print(json.dumps(outcomes,indent=2))
if __name__=='__main__':main()
