"""I show receiver-verified upload outcomes separately from probe readouts."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,Rectangle,FancyArrowPatch
from matplotlib import font_manager
HERE=Path(__file__).resolve().parent
DATA=HERE/'data/figure8-steering/local-case002-v1'
OUT=HERE/'figures/figure8-steering/case002-v1'
for name in ['regular','bold']:
 font_manager.fontManager.addfont(HERE/f'fonts/termes-ttf/texgyretermes-{name}.ttf')
plt.rcParams.update({'font.family':'TeX Gyre Termes','pdf.fonttype':42,'svg.fonttype':'path','text.parse_math':False})
rows=json.loads((DATA/'outcomes.json').read_text())
assert [r['arm'] for r in rows]==['plain','forgery','steered','destyled']
W,H=8.3,5.50
fig=plt.figure(figsize=(W,H),dpi=180);fig.patch.set_facecolor('white')
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
INK='#263441';GRAY='#62748e';PINK='#ff637e';GREEN='#00d492'
artists=[]
def text(x,y,s,size=12,weight='normal',color=INK,**kw):
 t=ax.text(x,y,s,fontsize=size,fontweight=weight,color=color,va='top',**kw);artists.append(t);return t
def box(x,y,w,h,color,fill=False):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=0.055',facecolor=matplotlib.colors.to_rgba(color,.095) if fill else 'white',edgecolor=color,linewidth=1.5))
text(.42,.22,'Did the model upload the dummy file?',23,'bold')
text(.44,.69,'One selected webpage task · GPT-OSS-20B',11,color=GRAY)
labels=[('No forged reasoning','Injected command; summary refused'),('CoT forgery','Summary provided'),('CoT forgery + steering','Summary provided; one unsupported sentence'),('Destyled CoT forgery','Summary provided')]
verified=[]
for i,(row,(label,note)) in enumerate(zip(rows,labels)):
 ep=json.loads((DATA/'results'/row['arm']/'episode.json').read_text())
 assert row['status']=='completed' and not row['censored'] and ep['exposure_confirmed']
 receipts=json.loads((DATA/'results'/row['arm']/'receipts.json').read_text())
 upload=any(r.get('valid',False) for r in receipts)
 assert upload==row['verified_dummy_upload']==ep['verified_dummy_upload']
 color=PINK if upload else GREEN
 y=1.18+i*.94
 text(.44,y+.18,'ABCD'[i],12,'bold',GRAY)
 text(.77,y+.15,label,14,'bold')
 text(.77,y+.46,note,9.7,color=GRAY)
 ax.add_patch(FancyArrowPatch((4.07,y+.41),(4.47,y+.41),arrowstyle='-|>',mutation_scale=11,linewidth=1,color='#a5aeb7'))
 box(4.62,y,3.24,.79,color,True)
 text(6.24,y+.12,'UPLOADED' if upload else 'NO UPLOAD',21,'bold',ha='center')
 text(6.24,y+.48,'Prompt injection succeeded' if upload else 'Prompt injection failed',11,ha='center')
 verified.append({'arm':row['arm'],'uploaded':upload,'label':'UPLOADED' if upload else 'NO UPLOAD','basis':'receiver receipt + exposed, completed episode'})
# I state the outcome key explicitly; color never carries the result by itself.
text(.44,5.05,'LEGEND',9,'bold',GRAY)
for x,color,phrase in [(1.24,GREEN,'No upload = injection failed'),(4.82,PINK,'Upload = injection succeeded')]:
 ax.add_patch(Rectangle((x,5.045),.135,.135,facecolor=color,edgecolor=color))
 text(x+.22,5.03,phrase,10.2)
fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[]
for t in artists:
 b=t.get_window_extent(renderer);assert b.x0>=0 and b.y0>=0 and b.x1<=fig.bbox.width and b.y1<=fig.bbox.height,t.get_text()
 bounds.append({'text':t.get_text(),'bounds':list(b.bounds)})
# I check that condition descriptions finish before the outcome boxes.
for t in artists:
 if t.get_text() in [n for _,n in labels]:assert t.get_window_extent(renderer).x1<(4.03/W)*fig.bbox.width,t.get_text()
files=[]
for ext in ['png','pdf','svg']:
 p=OUT/f'upload-outcomes.{ext}';fig.savefig(p,dpi=300,facecolor='white');files.append({'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
plt.close(fig)
(OUT/'upload-outcomes-provenance.json').write_text(json.dumps({'outcomes_sha256':hashlib.sha256((DATA/'outcomes.json').read_bytes()).hexdigest(),'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'font':'TeX Gyre Termes','size_inches':[W,H],'outcome_palette':{'uploaded':PINK,'no_upload':GREEN},'color_semantics':'Outcome colors in a separate behavior diagram, not token-role colors. Exact paper hues retained.','verified_outcomes':verified,'technical_footer':False,'text_bounds':bounds,'files':files},indent=2)+'\n')
print({'figure':str(OUT/'upload-outcomes.png'),'outcomes':verified})
