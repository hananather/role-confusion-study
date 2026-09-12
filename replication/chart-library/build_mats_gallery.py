"""I show all twelve measured candidates and preserve the visual selection context."""
from pathlib import Path
import json,html
import pandas as pd
H=Path(__file__).resolve().parent
D=H/'data/mats-six-passage'
cases=json.loads((D/'candidates.json').read_text())
selection=json.loads((D/'selection.json').read_text()) if (D/'selection.json').exists() else {}
selected=selection.get('candidate')
means=pd.read_csv(D/'all-passage-means.csv')
colors={'user':'#0084d1','cot':'#e17100','assistant':'#009966'}
labels=['User 1','CoT 1','Assistant 1','User 2','CoT 2','Assistant 2']
parts=[]
for case in cases:
 ident=case['id'];prefix=f'figures/mats-six-passage/{ident}/{ident}'
 title=html.escape(case['title']);badge=' · Featured illustration' if ident==selected else ''
 passages=''.join(f'<div class="passage" style="border-color:{colors[m["role"]]}"><b>{label}</b><p>{html.escape(m["content"])}</p></div>' for label,m in zip(labels,case['messages']))
 rows=[]
 for condition,label in [('proper_tags','Correct tags'),('basic_no_format','No tags'),('everything_in_user_tags','All User tags')]:
  q=means[(means.candidate==ident)&(means.prompt_key==condition)].sort_values('base_message_ix')
  rows.append('<tr><th>'+label+'</th>'+''.join(f'<td>{100*x:.1f}%</td>' for x in q['mean'])+'</tr>')
 table='<div class="scroll"><table><thead><tr><th>CoTness mean</th>'+''.join('<th>'+x+'</th>' for x in labels)+'</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
 parts.append(f'<section id="{ident}"><h2>{ident}: {title}{badge}</h2><a href="{prefix}.png"><img loading="lazy" src="{prefix}.png" alt="Three measured CoTness panels for {ident}, with all six authored passages."></a><p class="links"><a href="{prefix}.pdf">PDF</a> · <a href="{prefix}.svg">SVG</a> · <a href="data/mats-six-passage/{ident}/displayed-rows.csv">Plotted measurements</a></p><details><summary>Read all six passages and their measured means</summary>{passages}{table}</details></section>')
css="""@font-face{font-family:Termes;src:url('fonts/texgyretermes-regular.otf')}@font-face{font-family:Termes;src:url('fonts/texgyretermes-bold.otf');font-weight:700}*{box-sizing:border-box}body{font-family:Termes,serif;color:#252525;background:#fff;margin:0;font-size:20px;line-height:1.5}main{max-width:1150px;margin:auto;padding:32px}p{max-width:850px}h1{font-size:38px;line-height:1.15}h2{font-size:27px;margin-top:50px;padding-top:20px;border-top:1px solid #ccc}img{width:100%;height:auto}a{color:#384f66;text-underline-offset:3px}nav{display:flex;flex-wrap:wrap;gap:14px;font-size:17px}.links{font-size:17px}.passage{border-left:3px solid;padding-left:16px;margin:18px 0}.passage p{margin-top:5px}summary{cursor:pointer;font-size:18px}table{border-collapse:collapse;font-size:16px;width:100%;white-space:nowrap}th,td{border-bottom:1px solid #ddd;text-align:left;padding:8px}.scroll{overflow-x:auto}@media(max-width:650px){main{padding:20px 16px}body{font-size:18px}h1{font-size:30px}}
"""
nav=''.join(f'<a href="#{c["id"]}">{c["id"]}</a>' for c in cases)
intro='All twelve conversations were authored before this batch was measured. Each has two User → CoT → Assistant turns, rendered under three formatting conditions. We preserve every measured result and select the featured example after inspecting the plots for visual clarity. This is an illustration gallery, not a population estimate.'
sel=f'<p><strong>Selected: {selected}.</strong> {html.escape(selection.get("reason",""))}</p>' if selected else '<p>Selection pending.</p>'
page=f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MATS example comparison</title><style>{css}</style><main><p><a href="blog-draft.html">Return to the report</a></p><h1>Twelve MATS examples, measured and compared</h1><p>{intro}</p>{sel}<nav>{nav}</nav>{"".join(parts)}<p><a href="data/mats-six-passage/input-freeze.json">Frozen batch</a> · <a href="data/mats-six-passage/provenance.json">Run provenance</a> · <a href="data/mats-six-passage/audit.md">Independent audit</a></p></main></html>'
(H/'mats-example-gallery.html').write_text(page)
print({'gallery':str(H/'mats-example-gallery.html'),'candidates':len(parts),'selected':selected})
