"""I make an offline viewer for saved trajectories without changing or judging them."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'none'; img-src 'none'">
<title>Saved trajectories</title>
<style>
:root{color-scheme:light;--ink:#202733;--muted:#5f6976;--line:#dce1e7;--paper:#fff;--bg:#f5f6f8;--accent:#334f78}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}
header{padding:28px max(24px,4vw) 20px;background:white;border-bottom:1px solid var(--line)}
h1{font-size:28px;margin:0 0 6px;letter-spacing:-.6px}p{margin:6px 0;color:var(--muted)}
.notice{margin-top:14px;padding:10px 13px;border-left:3px solid var(--accent);background:#eff3f8;color:var(--ink)}
.toolbar{position:sticky;top:0;z-index:2;display:flex;gap:12px;flex-wrap:wrap;padding:16px max(24px,4vw);background:rgba(245,246,248,.97);border-bottom:1px solid var(--line)}
label{display:flex;flex-direction:column;gap:4px;font-size:12px;font-weight:600;color:var(--muted)}
select,input,button{font:inherit;color:var(--ink);background:white;border:1px solid #cbd2dc;border-radius:6px;padding:8px 10px;min-height:38px}
select{max-width:420px}input{min-width:220px}button{cursor:pointer;align-self:flex-end}button:disabled{opacity:.4;cursor:default}
main{padding:22px max(24px,4vw)}#count{margin:0 0 14px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr));align-items:start;gap:18px}
.card{min-width:0;background:white;border:1px solid var(--line);border-radius:9px;overflow:hidden}.cardhead{padding:16px 18px;border-bottom:1px solid var(--line)}
h2{font-size:17px;line-height:1.35;margin:0 0 7px}.meta{font-size:12px;color:var(--muted);overflow-wrap:anywhere}.flags{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}.flag{font-size:12px;background:#f0f2f5;border:1px solid #e0e4ea;border-radius:20px;padding:2px 8px}
.section{padding:13px 18px;border-bottom:1px solid #edf0f3}.section:last-child{border-bottom:0}.section h3,summary{font-size:13px;font-weight:650;margin:0 0 7px;color:#3f4d5f}summary{cursor:pointer;margin:0}details[open]>summary{margin-bottom:10px}
pre{font:13px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre-wrap;overflow-wrap:anywhere;margin:0;max-height:640px;overflow:auto}
.empty{font-size:13px;color:var(--muted);font-style:italic}.footer{padding:20px max(24px,4vw);border-top:1px solid var(--line);font-size:12px;color:var(--muted);overflow-wrap:anywhere}
@media(max-width:650px){header,.toolbar,main,.footer{padding-left:16px;padding-right:16px}.toolbar label{width:100%}select,input{max-width:100%;width:100%}}
</style></head><body>
<header><h1>Saved trajectories</h1><p id="overview"></p>
<p>Harmfulness judging and review of whether the emitted reasoning recognizes an injection remain unfinished.</p>
<div id="scope" class="notice"></div></header>
<div class="toolbar">
<label>Prompt<select id="prompt"></select></label>
<label>Run<select id="arm"></select></label>
<label>Find in question<input id="search" type="search" placeholder="Question text or prompt ID"></label>
<button id="previous" type="button">Previous prompt</button><button id="next" type="button">Next prompt</button>
</div><main><p id="count"></p><div id="cards" class="grid"></div></main>
<footer class="footer">This viewer works offline. Source: __SOURCE_LABEL__<br>Snapshot SHA256: __SOURCE_HASH__<br><span id="warnings"></span></footer>
<script id="trajectory-data" type="application/json">__DATA_JSON__</script>
<script>
'use strict';
const data=JSON.parse(document.getElementById('trajectory-data').textContent);
const rows=data.rows;
const $=id=>document.getElementById(id);
const stringify=value=>value===null||value===undefined?'—':typeof value==='string'?value:JSON.stringify(value);
const bool=value=>value===true?'Yes':value===false?'No':'Unknown';
const armKey=r=>JSON.stringify([r.variant,r.arm_id||r.arm||'unspecified',r.stage,r.layer,r.alpha,r.mask,r.max_new_tokens||r.max_tokens]);
const promptKey=r=>JSON.stringify([r.prompt_id,r.kind||null,r.split||null]);
const prettyDirection={none:'Unsteered',refusal_last:'Refusal direction',user_minus_cot:'User − reasoning',tool_minus_cot:'Tool − reasoning',tool_minus_user:'Tool − user',reverse:'Reversed direction',random_0:'Random direction 1',random_1:'Random direction 2',random_2:'Random direction 3'};
function armLabel(r){
 const variant=r.variant==='base'?'Base':r.variant==='forgery_generic'?'Forged':stringify(r.variant);
 const direction=prettyDirection[r.direction]||stringify(r.direction);
 return variant+' · '+direction+(r.direction&&r.direction!=='none'?' · strength '+stringify(r.alpha)+' · layer '+stringify(r.layer)+' · '+stringify(r.mask):'')+' · '+stringify(r.max_new_tokens||r.max_tokens)+' token cap';
}
function option(parent,value,label){const node=document.createElement('option');node.value=value;node.textContent=label;parent.append(node);}
const prompts=new Map(),arms=new Map();for(const r of rows){if(!prompts.has(promptKey(r)))prompts.set(promptKey(r),r);if(!arms.has(armKey(r)))arms.set(armKey(r),r);}
option($('prompt'),'ALL','All prompts');for(const [key,r] of prompts)option($('prompt'),key,'Prompt '+stringify(r.prompt_id)+(r.split?' · '+r.split:''));
option($('arm'),'ALL','All runs for this prompt');for(const [key,r] of arms)option($('arm'),key,armLabel(r));
if(prompts.size)$('prompt').selectedIndex=1;
const finished=rows.filter(r=>r.final_ended===true).length,censored=rows.filter(r=>r.censored===true).length;
$('overview').textContent=rows.length+' saved trajectories · '+prompts.size+' prompts · '+arms.size+' run settings · '+finished+' final channels ended · '+censored+' saved censor flags';
const baselineOnly=rows.length>0&&rows.every(r=>r.variant==='base'&&(!r.direction||r.direction==='none')&&(!r.alpha||r.alpha===0));
$('scope').textContent=!rows.length?'This snapshot has no saved trajectories yet.':baselineOnly?'Only baseline trajectories are present in this snapshot. Forged and steered trajectories are not included in this file.':'Each card shows a saved trajectory. Runs are paired by prompt ID, prompt kind, and split. Blank or missing runs remain absent from this snapshot.';
$('warnings').textContent=data.warnings.length?data.warnings.map(w=>'Line '+w.line+': '+w.issue).join(' · '):'All saved JSONL records were parsed.';
function element(tag,cls,text){const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n;}
function section(card,title,value,open){
 const box=element(open?'section':'details','section');
 box.append(element(open?'h3':'summary','',title));
 if(value!==null&&value!==undefined&&String(value).length)box.append(element('pre','',stringify(value)));else box.append(element('p','empty','No text recorded.'));
 card.append(box);
}
function render(){
 const selectedPrompt=$('prompt').value,selectedArm=$('arm').value,query=$('search').value.toLowerCase();
 const visible=rows.filter(r=>(selectedPrompt==='ALL'||promptKey(r)===selectedPrompt)&&(selectedArm==='ALL'||armKey(r)===selectedArm)&&(!query||(stringify(r.question)+' '+stringify(r.prompt_id)).toLowerCase().includes(query)));
 $('cards').replaceChildren();$('count').textContent='Showing '+visible.length+' of '+rows.length+' saved trajectories.';
 for(const r of visible){
  const card=element('article','card'),head=element('div','cardhead');head.append(element('h2','',armLabel(r)));
  head.append(element('div','meta','Prompt '+stringify(r.prompt_id)+' · source line '+r._source_line_1based+' · '+stringify(r.n_gen)+' generated tokens'));
  const flags=element('div','flags');for(const [label,value] of [['Final started',bool(r.final_started)],['Final ended',bool(r.final_ended)],['Censored',bool(r.censored)],['Stop',stringify(r.stop_reason)]])flags.append(element('span','flag',label+': '+value));
  head.append(flags);card.append(head);
  section(card,'Question',r.question,true);section(card,'Supplied policy',r.policy,false);
  section(card,'Emitted analysis',r.cot,true);section(card,'Final channel',r.final,true);
  section(card,'Full raw output',r.raw_output===undefined?r.output:r.raw_output,false);
  const metadata={source_line_1based:r._source_line_1based,prompt_id:r.prompt_id,variant:r.variant,arm_id:r.arm_id||r.arm,stage:r.stage,kind:r.kind,split:r.split,direction:r.direction,layer:r.layer,alpha:r.alpha,mask:r.mask,max_new_tokens:r.max_new_tokens||r.max_tokens,edited_positions:r.edited_positions,probe_relation_to_edit:r.probe_relation_to_edit,policy_probe_probabilities:r.sucat_L12_policy_probabilities,provisional_success_saved_unreviewed:r.provisional_success,canned_refusal_saved_unreviewed:r.canned_refusal};
  section(card,'Saved run metadata · provisional labels unreviewed',JSON.stringify(metadata,null,2),false);$('cards').append(card);
 }
 $('previous').disabled=$('prompt').selectedIndex<=1;$('next').disabled=$('prompt').selectedIndex<1||$('prompt').selectedIndex>=$('prompt').options.length-1;
}
for(const id of ['prompt','arm'])$(id).addEventListener('change',render);$('search').addEventListener('input',()=>{if($('search').value)$('prompt').value='ALL';render();});
$('previous').addEventListener('click',()=>{$('prompt').selectedIndex-=1;render();});$('next').addEventListener('click',()=>{$('prompt').selectedIndex+=1;render();});render();
</script></body></html>'''


def build_viewer(generations, output, allow_partial_tail=False):
    source,target=Path(generations).resolve(),Path(output).resolve()
    if source==target:
        raise ValueError("The HTML output must differ from the raw input")
    if target.exists():
        raise ValueError("I require a new HTML filename to preserve prior review artifacts")
    raw=source.read_bytes();lines=raw.splitlines();rows=[];warnings=[]
    for number,line in enumerate(lines,start=1):
        if not line.strip():
            warnings.append({"line":number,"issue":"blank line"});continue
        try:record=json.loads(line)
        except (json.JSONDecodeError,UnicodeDecodeError) as exc:
            if allow_partial_tail and number==len(lines) and not raw.endswith(b"\n"):
                warnings.append({"line":number,"issue":"incomplete final line excluded"});continue
            raise ValueError(f"Invalid JSON on source line {number}") from exc
        if not isinstance(record,dict) or "prompt_id" not in record:
            raise ValueError(f"Source line {number} lacks a trajectory object/prompt ID")
        rows.append({**record,"_source_line_1based":number})
    payload=json.dumps({"rows":rows,"warnings":warnings},ensure_ascii=False,allow_nan=False)
    # I escape HTML delimiters in the JSON container; every displayed field also
    # uses textContent. Saved instructions, HTML and script fragments remain text.
    payload=payload.replace("&","\\u0026").replace("<","\\u003c").replace(">","\\u003e").replace("\u2028","\\u2028").replace("\u2029","\\u2029")
    digest=hashlib.sha256(raw).hexdigest()
    page=PAGE.replace("__SOURCE_LABEL__",html.escape(str(source))).replace("__SOURCE_HASH__",digest).replace("__DATA_JSON__",payload)
    target.parent.mkdir(parents=True,exist_ok=True);target.write_text(page)
    return {"html":str(target),"source":str(source),"source_sha256":digest,"rows":len(rows),"warnings":warnings,
            "judging":"unfinished","recognition_review":"unfinished"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations",type=Path,required=True)
    parser.add_argument("--html",type=Path,required=True)
    parser.add_argument("--allow-partial-tail",action="store_true")
    args=parser.parse_args()
    print(json.dumps(build_viewer(args.generations,args.html,args.allow_partial_tail)))


if __name__=="__main__":main()
