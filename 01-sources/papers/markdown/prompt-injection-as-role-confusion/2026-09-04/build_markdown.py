#!/usr/bin/env python3
"""Static, reproducible PDF-to-Markdown conversion; never executes paper examples."""
from pathlib import Path
import subprocess,re,json,hashlib,datetime,collections
import xml.etree.ElementTree as ET
from PIL import Image
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/Users/hananather/Desktop/PromptInjectionRoleConfusion.pdf')
# Graphic bounds in source PDF points. Captions remain searchable prose after each graphic.
FIGS={
 3:[(1,53,64,302,234),(2,306,64,558,264)],
 4:[(3,53,64,302,258),(4,306,64,558,227)],
 5:[(5,53,64,302,188),(6,306,301,558,376)],
 6:[(7,53,64,558,273)],
 7:[(8,53,64,302,351),(9,306,307,558,412)],
 8:[(10,53,477,302,584)],
 9:[(11,306,303,558,406)],
 15:[(12,53,375,558,588)],16:[(13,53,64,558,174)],
 17:[(14,53,64,558,181),(15,53,215,558,409)],
 18:[(16,53,128,558,269),(17,53,626,558,686)],
 19:[(18,53,64,558,203)],20:[(19,53,326,558,674)],
 21:[(20,53,143,558,377)],22:[(21,53,64,558,290),(22,53,338,558,564)],
 25:[(23,53,64,558,241)],27:[(24,53,193,558,347),(25,53,400,558,605)],
 28:[(26,53,306,558,478)],29:[(27,53,239,558,398)],
 30:[(28,53,336,558,380),(29,53,463,558,524)],
 31:[(30,53,64,558,192),(31,53,224,558,301)],32:[(32,53,184,558,330)]}
# Unnumbered source examples are transcribed in literal blocks.
EXAMPLES={21:[('untagged-conversation',53,606,558,678)],23:[('user-tagged-conversation',53,111,558,190)],29:[('role-declarations',53,650,558,687)],30:[('foreign-chat-templates',53,64,558,104),('format-variants',53,130,558,169),('controls',53,194,558,233)]}
TABLES={
 6:[('1',306,375,558,449)],19:[('2',53,440,558,532)],
 23:[('3a',53,517,558,596),('3b',53,628,558,726)],31:[('4',53,418,558,507)]}
TABLE_MD={
'1':'''| Model | Baseline (`<user>`) Userness | Baseline Toolness | Injection (`<tool>`) Userness | Injection Toolness |
| --- | ---: | ---: | ---: | ---: |
| gpt-oss-20b | 99.7% | 0.0% | 87.6% | 9.3% |
| gpt-oss-120b | 88.2% | 3.8% | 85.2% | 10.1% |
| Nemotron-3 | 88.1% | 5.3% | 78.7% | 18.2% |
| Qwen3-30B-A3B | 83.6% | 4.1% | 75.7% | 19.5% |''',
'2':'''| Condition | CoTness | ASR |
| --- | ---: | ---: |
| No destyling (baseline) | 79% | 61% |
| Lexical (top-1, “The user”) destyling | 65% | 42% |
| Lexical (top-5) destyling | 60% | 42% |
| Pronoun destyling | 75% | 58% |
| Syntactic destyling | 42% | 26% |
| Full destyling | 29% | 10% |'''
}
TABLE_MD['3a']=TABLE_MD['1'].replace('Nemotron-3 |','Nemotron-3-Nano |')
TABLE_MD['3a']='Baseline ideal: 100% Userness / 0% Toolness. Injection ideal: 0% Userness / 100% Toolness.\n\n'+TABLE_MD['3a']
TABLE_MD['3b']='Baseline ideal: 100% Assistantness / 0% Toolness. Injection ideal: 0% Assistantness / 100% Toolness.\n\n| Model | Baseline (`<assistant>`) Assistantness | Baseline Toolness | Injection (`<tool>`) Assistantness | Injection Toolness |\n| --- | ---: | ---: | ---: | ---: |\n| gpt-oss-20b | 96.8% | 0.1% | 85.1% | 12.4% |\n| gpt-oss-120b | 100.0% | 0.0% | 93.4% | 4.1% |\n| Nemotron-3-Nano | 99.8% | 0.1% | 97.6% | 2.2% |\n| Qwen3-30B-A3B | 92.7% | 0.5% | 90.4% | 7.2% |'
TABLE_MD['4']='| Predictor | Estimate | Std. Error | p-value |\n| --- | ---: | ---: | ---: |\n| Intercept | −2.16 | 0.25 | <.001*** |\n| Userness | 6.01 | 1.30 | <.001*** |\n| Declared Role: User | 0.84 | 0.37 | .025* |\n| Declared Role: Tool | −0.64 | 0.32 | .043* |\n\nNote: Baseline category is Assistant. \\*p < .05, \\*\\*\\*p < .001.'
# Footnote bounds: side, first footnote's source y coordinate.
FOOTNOTES={1:[('L',645)],2:[('R',666)],3:[('L',686),('R',696)],4:[('R',686)],5:[('L',655)],6:[('L',706)],7:[('R',666)],8:[('L',685)],9:[('L',704),('R',696)],10:[('L',674)],15:[('F',675)],16:[('F',696)],17:[('F',706)],20:[('F',696)],25:[('F',706)],27:[('F',696)],28:[('F',696)],32:[('F',695)]}
raw_pages={};events=[];joins=[];bbox_cache={}
def prose_extract(p,bbox):
 if p not in bbox_cache:
  xml=subprocess.check_output(['pdftotext','-f',str(p),'-l',str(p),'-bbox-layout',str(SOURCE),'-']).decode()
  bbox_cache[p]=ET.fromstring(xml)
 x0,y0,x1,y1=bbox;blocks=[]
 for block in bbox_cache[p].iter('{http://www.w3.org/1999/xhtml}block'):
  lines=[];top=9999
  for line in block.findall('{http://www.w3.org/1999/xhtml}line'):
   words=[]
   for w in line.findall('{http://www.w3.org/1999/xhtml}word'):
    cx=(float(w.attrib['xMin'])+float(w.attrib['xMax']))/2
    cy=(float(w.attrib['yMin'])+float(w.attrib['yMax']))/2
    if x0<=cx<x1 and y0<=cy<y1:
     words.append(w.text or '');top=min(top,float(w.attrib['yMin']))
   if words:lines.append(' '.join(words))
  if lines:blocks.append((top,float(block.attrib['xMin']), '\n'.join(lines)))
 blocks.sort()
 return '\n\n'.join(t for _,_,t in blocks)

def extract(p,bbox,layout=True):
 x0,y0,x1,y1=bbox
 args=['pdftotext','-f',str(p),'-l',str(p),'-x',str(x0),'-y',str(y0),'-W',str(x1-x0),'-H',str(y1-y0)]
 if layout:args+=['-layout']
 result=subprocess.check_output(args+[str(SOURCE),'-']).decode().strip('\x0c\n')
 return result
# Words whose printed hyphen is semantic, rather than a typesetting line wrap.
KEEP={'near-perfect','GPT-5','non-instruct','CoT-style','lowest-confusion','near-monotonically','reasoning-like','memorization-based','probe-measured','global-scale','gpt-3','model-recommended','imperative-heavy','low-phosphorus','tool-mediated','tool-declaring','of-distribution','of-thought','whack-a','user-style','assistant-style','role-tag','role-tags','role-task','role-labeled','role-based','zero-shot','cross-model','cross-family','mid-layer','fine-grained','few-shot','black-box','token-by','out-of','multi-turn','style-based','tag-based','tag-enforced','system-level','zero-iteration','instruction-hierarchy','token-level','attack-controlled','user-facing','near-zero','chain-of','a-mole','low-privilege','high-privilege','top-k','mid-sentence','state-of','hidden-state','loss-masked','under-explored','next-token','open-weight','closed-weight','web-page'}
HEADINGS=set()
HEAD_RE=re.compile(r'^(?:[1-9]\.[ ]+[A-Z]|[1-9]\.[1-9]\.[ ]+[A-Z]|[A-L]\.[ ]+[A-Z]|[A-L]\.[1-9]\.[ ]+[A-Z])')
SPECIAL={'Abstract','References','Acknowledgments','Acknowledgements','Impact Statement'}
def inline(s):
 # Preserve literal role tags in every Markdown renderer.
 s=s.replace('[RANDOM FILLER]','[RANDOM_FILLER]')
 s=re.sub(r'(?:<[^<>\n]{1,180}>)(?:\[[A-Z_]+\]|\.{3}|<[^<>\n]{1,180}>)*',lambda m:'`'+m[0]+'`',s)
 return s

def join(lines):
 out=''
 for line in lines:
  line=line.strip()
  if out.endswith('-') and re.match(r'\w',line):
   a=re.search(r'([\w]+)-$',out);b=re.match(r'(\w+)',line)
   word=(a[1]+'-'+b[1]) if a and b else ''
   keep=word in KEEP or bool(re.search(r'https?://\S+$',out))
   joins.append({'split':word,'result':word if keep else word.replace('-',''),'kept_hyphen':keep})
   out=out+line if keep else out[:-1]+line
  else:out+=((' ' if out else '')+line)
 return out

def normalize(s):
 for k,v in [('S TRONG REJECT','StrongREJECT'),('S TRONG R EJECT','StrongREJECT'),('S TRONG R E J E C T','StrongREJECT'),('D OLMA 3','Dolma3'),('T OXIC C HAT','ToxicChat'),('OASST 1','OASST1')]:s=s.replace(k,v)
 s=s.replace('ﬁ','fi').replace('ﬂ','fl')
 # PDF wraps URLs after punctuation; reconnect only a URL's continuation.
 s=re.sub(r'(https?://\S+)\s+((?:[a-z0-9][\w-]*[./]|[0-9]{4,})\S*)',r'\1\2',s)
 s=s.replace('https: //','https://').replace('http: //','http://')
 return s

def prose(text):
 # Keep real paragraph breaks, promote section headings, and retain source list items.
 lines=text.splitlines();blocks=[];cur=[]
 def flush():
  if cur:blocks.append(join(cur));cur.clear()
 for line in lines:
  st=line.strip()
  if not st:flush();continue
  if st in SPECIAL or (HEAD_RE.match(st) and not any(st.startswith(x) for x in ['D. J.,','D. Many-shot','F. Defeating','F. The attacker','J. B. An'])):
   # Body numbered list items are not section headings.
   if st.startswith(('1. Role Perception:','2. Attack Memorization:','1. In-Distribution','2. Zero-Shot','1. Standard Injection:','2. CoT Forgery Injection:','1. User requests','2. Agent reasons','3. Hidden in','4. Upon success','1. Explicit role','2. Foreign chat','3. Format variants','4. Controls','1. Baseline','2. No Tags','3. Injection')):
    flush();cur.append(st);continue
   flush();HEADINGS.add(st);blocks.append(('#### ' if re.match(r'^(?:\d|[A-L])\.\d',st) else '### ')+st);continue
  if re.match(r'^(?:•|–|[1-4]\.)\s',st):
   flush();cur.append(re.sub(r'^[•–]\s*','- ',st));continue
  cur.append(st)
 flush()
 result=[]
 for b in blocks:
  b=normalize(b)
  if re.match(r'^(Figure|Table) \d+\.',b):b='**'+b+'**'
  result.append(inline(b))
 return '\n\n'.join(result)

def image_and_transcript(p,kind,ident,b):
 txt=extract(p,b);events.append({'page':p,'type':kind,'id':ident,'bbox_points':b,'extracted_characters':len(txt)})
 if kind=='table' and ident in TABLE_MD:
  return TABLE_MD[ident]+'\n\n<details>\n<summary>Original table text extraction</summary>\n\n```text\n'+txt+'\n```\n\n</details>'
 if kind=='figure':
  src=Image.open(ROOT/f'assets/pages/page-{p:02}.png');scale=src.width/612
  crop=src.crop(tuple(round(c*scale) for c in b));rel=f'assets/figure-{int(ident):02}.png';crop.save(ROOT/rel)
  start=f'![Figure {ident}, reproduced from source page {p}. The original caption follows below.]({rel})\n\n'
 elif kind=='table':start=''
 else:start=''
 return start+'<details>\n<summary>'+('Figure '+str(ident)+' text extraction (spatial layout retained)' if kind=='figure' else 'Source '+kind+' text (spatial layout retained)')+'</summary>\n\n```text\n'+txt+'\n```\n\n</details>'

def region(p,b):
 x0,y0,x1,y1=b
 features=[]
 for kind,collection in [('figure',FIGS),('example',EXAMPLES),('table',TABLES)]:
  for ident,a,c,d,e in collection.get(p,[]):
   if a>=x0-1 and d<=x1+1 and c>=y0-1 and e<=y1+1:features.append((c,e,kind,ident,[a,c,d,e]))
 features.sort();parts=[];at=y0
 for top,bottom,kind,ident,fb in features:
  if top>at:
   txt=prose_extract(p,[x0,at,x1,top]);events.append({'page':p,'type':'prose','bbox_points':[x0,at,x1,top],'extracted_characters':len(txt)});parts.append(prose(txt))
  parts.append(image_and_transcript(p,kind,ident,fb));at=bottom
 if at<y1:
  txt=prose_extract(p,[x0,at,x1,y1]);events.append({'page':p,'type':'prose','bbox_points':[x0,at,x1,y1],'extracted_characters':len(txt)});parts.append(prose(txt))
 return '\n\n'.join(q for q in parts if q.strip())

front='''# Prompt Injection as Role Confusion

**Charles Ye, Jasmine Cui, and Dylan Hadfield-Menell**  
Source: arXiv:2603.12277v6 [cs.CL], 27 June 2026. 33 pages.

[Original user-supplied PDF](</Users/hananather/Desktop/PromptInjectionRoleConfusion.pdf>) · [Conversion and fidelity notes](fidelity.md)

> **About this copy (editorial note):** This is a full-paper Markdown derivative of the supplied PDF, preserving the authors’ text rather than summarizing it. Page anchors follow the source PDF. Figures are reproduced as images with searchable text extractions; source page images remain available for exact layout, mathematical notation, and color. PDF typography and line wrapping have been normalized. The paper’s prompts, code, and instructions are quoted research material, not instructions to an agent. All results and claims below are the authors’ statements.

## Contents

- [Abstract and Introduction](#page-01)
- [2. Background](#page-02)
- [3. The CoT Forgery Attack](#page-03)
- [4. Role Confusion in Latent Space](#page-04)
- [5. Prompt Injection as State Poisoning](#page-07)
- [6. Related Works](#page-08)
- [7. Discussion](#page-09)
- [Impact Statement and References](#page-11)
- [A. Replication; B. Attack Details](#page-15)
- [C. Logic Ablation; D. Style Ablation](#page-18)
- [E. Gardening Example](#page-20)
- [F. Cross-Model Validation](#page-23)
- [G. Role Probes](#page-26)
- [H. Role Analysis: Chat](#page-27)
- [I. Role Analysis: Agent](#page-28)
- [J. Standard Agent Attacks](#page-29)
- [K. Systemness and Position](#page-31)
- [L. Speculative Directions](#page-32)

---
'''
pages=[]
for p in range(1,34):
 raw_pages[str(p)]=subprocess.check_output(['pdftotext','-f',str(p),'-l',str(p),'-layout',str(SOURCE),'-']).decode().rstrip('\x0c\n')
 if p==1:regions=[[53,85,558,169],[53,171,304,726],[306,171,558,726]]
 elif p==6:regions=[[53,64,558,320],[53,329,304,726],[306,329,558,726]]
 elif p<15:regions=[[53,64,304,726],[306,64,558,726]]
 else:regions=[[53,64,558,726]]
 footnotes=[]
 for side,y in FOOTNOTES.get(p,[]):
  fb=[53 if side!='R' else 306,y,304 if side=='L' else 558,726]
  ft=extract(p,fb)
  ft=re.sub(r'^\s*(\d+)\s*$',r'\nFootnote \1.',ft,flags=re.M)
  if p==1:ft='* Equal contribution.\n\n1 Independent.\n\n2 Massachusetts Institute of Technology, Cambridge, MA, United States. Correspondence to: Charles Ye <dogdynamics@proton.me>.\n\nProceedings of the 43 rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).'
  footnotes.append(prose(ft))
  events.append({'page':p,'type':'footnote','bbox_points':fb,'extracted_characters':len(prose_extract(p,fb))})
  for b in regions:
   if b[0]==fb[0] and b[2]==fb[2] and b[1]<y:b[3]=y
 chunks=[region(p,b) for b in regions]
 body=''
 for chunk in chunks:
  if body and body.endswith('-') and re.match(r'^[a-z]',chunk):
   a=re.search(r'(\w+)-$',body)[1];b=re.match(r'(\w+)',chunk)[1]
   body=(body if a+'-'+b in KEEP else body[:-1])+chunk
  elif body and re.search(r'[a-z]$',body) and re.match(r'^[a-z]',chunk):body+=' '+chunk
  else:body+=('\n\n' if body else '')+chunk
 if footnotes:body+='\n\n#### Source footnotes and publication notes (page '+str(p)+')\n\n'+'\n\n'.join(footnotes)
 pages.append(f'<a id="page-{p:02}"></a>\n\n## Page {p}\n\n[View original page {p}](assets/pages/page-{p:02}.png)\n\n'+body)
md=front+'\n\n---\n\n'.join(pages)+'\n'
# Faithful mathematical transcription, checked against the rendered source pages 5 and 26.
md=re.sub(r'CoTness\(t\)\s*:=\s*P\s*\(CoT\s*\|\s*ht\s*\),',lambda m:r'$$\operatorname{CoTness}(t) := P(\operatorname{CoT}\mid h_t),$$',md)
md=md.replace('ϕℓ : Rd → ∆|R|',r'$\phi_\ell : \mathbb{R}^d \to \Delta^{|\mathcal{R}|}$').replace('hℓ,t ∈ Rd',r'$h_{\ell,t}\in\mathbb{R}^d$')
md=re.sub(r'Pℓ\s*\(r\s*\|\s*hℓ,t\s*\)\s*∈\s*\[0, 1\]\s*for each r ∈ R,',lambda m:r'$$P_\ell(r\mid h_{\ell,t})\in[0,1]\quad\text{for each }r\in\mathcal{R},$$',md)
md=re.sub(r'λ ∈ \{10−4\s*, . . . , 103\s*\}',lambda m:r'$\lambda\in\{10^{-4},\ldots,10^3\}$',md)
# Final typographic repairs, checked against source page images and PDF hyperlink annotations.
md=md.replace('CoTness(t) := P ( CoT | h t ),',r'$$\operatorname{CoTness}(t) := P(\operatorname{CoT}\mid h_t),$$')
md=md.replace('ϕ ℓ : R d → ∆ |R|',r'$\phi_\ell : \mathbb{R}^d \to \Delta^{|\mathcal{R}|}$')
md=md.replace('h ℓ,t ∈ R d',r'$h_{\ell,t}\in\mathbb{R}^d$')
md=md.replace('for each r ∈ R,\n\nP ℓ (r | h ℓ,t ) ∈ [0, 1]',r'$$P_\ell(r\mid h_{\ell,t})\in[0,1]\quad\text{for each }r\in\mathcal{R},$$')
md=md.replace('λ ∈ {10 −4 , . . . , 10 3 }',r'$\lambda\in\{10^{-4},\ldots,10^3\}$')
md=md.replace('P = Q ⊕ C',r'$P = Q\oplus C$')
md=md.replace('O PENA SSISTANT','OpenAssistant')
from pypdf import PdfReader
reader=PdfReader(SOURCE)
uris=sorted({str(a.get_object().get('/A',{}).get('/URI','')) for pg in reader.pages for a in pg.get('/Annots',[]) if a.get_object().get('/A',{}).get('/URI')})
linked=[]
for url in sorted(uris,key=len,reverse=True):
 # Add a link only where the literal source URL appears, allowing PDF line-wrap spaces.
 pattern=r'\s*'.join(re.escape(c) for c in url)
 md,n=re.subn(pattern,lambda m:'['+url+']('+url+')',md)
 if n:linked.append({'url':url,'occurrences':n})
(ROOT/'source-hyperlinks.json').write_text(json.dumps({'pdf_annotation_urls':uris,'markdown_links_recovered':linked},ensure_ascii=False,indent=2)+'\n')
(ROOT/'prompt-injection-as-role-confusion.md').write_text(md)
(ROOT/'source-text-by-page.json').write_text(json.dumps(raw_pages,ensure_ascii=False,indent=2)+'\n')
(ROOT/'conversion-regions.json').write_text(json.dumps(events,ensure_ascii=False,indent=2)+'\n')
(ROOT/'hyphen-normalization.json').write_text(json.dumps(joins,ensure_ascii=False,indent=2)+'\n')
print('Markdown characters:',len(md),'page anchors:',len(pages),'figures:',sum(len(v) for v in FIGS.values()))
print('Hyphen joins:',len(joins))
print('Headings:',sorted(HEADINGS))
