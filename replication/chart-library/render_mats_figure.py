"""I render the frozen MATS example from measured, matched probe scores."""
from pathlib import Path
import hashlib,json,textwrap
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
HERE=Path(__file__).resolve().parent
DATA=HERE/'data/mats-example/displayed-rows.csv'
OUT=HERE/'figures/mats-example';OUT.mkdir(exist_ok=True)
FONT=HERE/'fonts/termes-ttf'
for variant in ['regular','bold','italic','bolditalic']:
 font_manager.fontManager.addfont(FONT/f'texgyretermes-{variant}.ttf')
plt.rcParams.update({'font.family':'TeX Gyre Termes','font.size':9,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','text.parse_math':False,'axes.linewidth':.5,'savefig.facecolor':'white'})
COLORS={'user':'#00a6f4','cot':'#fd9a00','assistant':'#00d492'}
LABELS={'user':'User question','cot':'Authored reasoning','assistant':'Assistant answer'}
CONDITIONS=['proper_tags','basic_no_format','everything_in_user_tags']
d=pd.read_csv(DATA,keep_default_na=False,float_precision='round_trip')
assert set(d.prompt_key)==set(CONDITIONS)
assert d.prob.between(0,1).all()
assert not d.duplicated(['prompt_key','display_token_ix']).any()
if 'target_role' in d: assert set(d.target_role)=={'cot'}
if 'layer_ix' in d: assert set(d.layer_ix)=={12}
base=d[d.prompt_key==CONDITIONS[0]].sort_values('display_token_ix')
n=len(base)
assert base.display_token_ix.tolist()==list(range(1,n+1))
assert n and set(base.base_message_type)==set(COLORS)
for condition in CONDITIONS:
 q=d[d.prompt_key==condition].sort_values('display_token_ix')
 assert len(q)==n
 assert q[['display_token_ix','token','base_message_type']].reset_index(drop=True).equals(base[['display_token_ix','token','base_message_type']].reset_index(drop=True))
fig,axs=plt.subplots(3,1,figsize=(6.75,3.85),sharex=True,sharey=True)
for ax,condition,title,letter in zip(axs,CONDITIONS,['Correct role tags','No role tags','All text in one User message'],'abc'):
 q=d[d.prompt_key==condition].sort_values('display_token_ix')
 for role,color in COLORS.items():
  r=q[q.base_message_type==role]
  ax.scatter(r.display_token_ix,r.prob,s=3,linewidths=0,color=color,zorder=3)
 ax.set_ylim(-.025,1.025);ax.set_xlim(.5,n+.5)
 ax.set_yticks([0,.5,1],labels=['0','50','100'])
 ax.grid(axis='y',color='#d9d9d9',linewidth=.3);ax.set_axisbelow(True)
 for spine in ax.spines.values():spine.set_color('#333333')
 ax.tick_params(length=2,width=.4,labelsize=8)
 ax.set_title(f'{letter.upper()} · {title}',fontsize=9,fontweight='bold',pad=5)
 for role in ['cot','assistant']:
  start=base.loc[base.base_message_type==role,'display_token_ix'].min()
  ax.axvline(start-.5,lw=.45,color='#b7b7b7',ls=(0,(2,3)),zorder=1)
centers=[base.loc[base.base_message_type==role,'display_token_ix'].mean() for role in COLORS]
axs[-1].set_xticks(centers,labels=[LABELS[r] for r in COLORS])
axs[-1].tick_params(axis='x',length=0,pad=7,labelsize=8)
axs[-1].set_xlabel('Matched content tokens, in conversation order',fontsize=8,labelpad=7)
for ax in axs[:-1]:ax.tick_params(axis='x',bottom=False)
question=json.loads((DATA.parent/'original-messages.json').read_text())[1]
question=' '.join(question.split())
if len(question)>140: question=question[:137].rsplit(' ',1)[0]+'…'
fig.text(.115,.985,'\n'.join(textwrap.wrap(question,95)),fontsize=9,fontstyle='italic',va='top')
fig.text(.022,.54,'CoT probe probability (%)',rotation=90,ha='center',va='center',fontsize=9)
handles=[Line2D([],[],marker='o',ls='',ms=4,color=c,label={'user':'User','cot':'CoT','assistant':'Assistant'}[r]) for r,c in COLORS.items()]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.015),ncol=3,frameon=False,handletextpad=.3,columnspacing=2,fontsize=8)
fig.subplots_adjust(left=.115,right=.985,bottom=.23,top=.86,hspace=.55)
fig.canvas.draw()
for ext in ['pdf','svg','png']:
 fig.savefig(OUT/f'mats-role-readout.{ext}',dpi=600)
plt.close(fig)
meta={'source':str(DATA),'source_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'n_display_tokens_per_condition':n,'n_points':len(d),'palette':COLORS,'font':'TeX Gyre Termes','font_embedding':'TrueType outlines derived from source OTF; cubic-to-quadratic tolerance 0.25 font units','size_inches':[6.75,3.85],'artifact_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('mats-role-readout.*')}}
(OUT/'render-provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
