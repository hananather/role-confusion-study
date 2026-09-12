"""I clarify the six report figures while retaining their saved measurements and paper style."""
from pathlib import Path
import hashlib, json, importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures/review-v0.2';OUT.mkdir(parents=True,exist_ok=True)
SESSION=HERE.parent/'cloud/persistent/sessions/20260911T030050Z'
SOURCE=SESSION/'outputs/20260911T030050Z'
FONT=Path('/usr/local/texlive/2019/texmf-dist/fonts/opentype/public/tex-gyre')
for v in ['regular','bold','italic','bolditalic']:font_manager.fontManager.addfont(FONT/f'texgyretermes-{v}.otf')
plt.rcParams.update({'font.family':'TeX Gyre Termes','font.size':9,'text.usetex':False,'text.parse_math':False,'pdf.fonttype':42,'axes.linewidth':.5,'savefig.facecolor':'white'})
COLORS={'system':'#90a1b9','user':'#00a6f4','cot':'#fd9a00','assistant':'#00d492','tool':'#7e6cff'}
TEXT={'user':'#0084d1','cot':'#e17100','assistant':'#009966'}
LABELS={'system':'System','user':'User','cot':'CoT','assistant':'Assistant','tool':'Tool'}
MARKERS={'system':'D','user':'o','cot':'^','assistant':'s','tool':'v'}
GRAY='#62748e';SOURCES=[];PLOTTED={}
spec=importlib.util.spec_from_file_location('saved_paper_style',HERE.parent/'cloud/persistent/appendix_e_figures.py')
paper=importlib.util.module_from_spec(spec);spec.loader.exec_module(paper)
def fingerprint(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def read(p):
 SOURCES.append(fingerprint(p));return pd.read_csv(p,keep_default_na=False,float_precision='round_trip')
def panel(ax):
 ax.grid(color='#d9d9d9',linewidth=.3);ax.set_axisbelow(True)
 for x in ax.spines.values():x.set_color('#333333')
 ax.tick_params(width=.4,length=2,pad=3)
def save(fig,name):
 for text in fig.texts:
  if text.get_position()[1]>.9:text.set_verticalalignment('top')
 fig.canvas.draw()
 for ext in ['png','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=300)
 plt.close(fig)
def passage_labels(rows):
 labels=[];counts={}
 for seg,g in rows.groupby('seg_ix',sort=True):
  role=g.base_message_type.iloc[0];counts[role]=counts.get(role,0)+1
  labels.append((int(g.display_token_ix.iloc[0]),role,counts[role],g.token.tolist()))
 return labels

garden=read(SOURCE/'appendix-e-L12/figure-7-displayed-rows.csv')
fig,axs=plt.subplots(3,1,figsize=(6.75,3.6),sharex=True)
for ax,condition,title in zip(axs,['proper_tags','basic_no_format','everything_in_user_tags'],['A · Correct tags','B · No tags','C · All text in one User message']):
 rows=garden[garden.prompt_key==condition].sort_values('display_token_ix');assert len(rows)==512
 panel(ax)
 for role in ['user','cot','assistant']:
  q=rows[rows.base_message_type==role];ax.scatter(q.display_token_ix,q.prob,s=1.1,linewidths=0,color=COLORS[role],alpha=.9)
 ax.set_ylim(-.02,1.02);ax.set_xlim(1,512);ax.set_yticks([0,1],labels=['0%','100%']);ax.set_title(title,fontsize=8,fontweight='bold',pad=3)
 labels=passage_labels(rows);ax.set_xticks([x[0] for x in labels]);ax.tick_params(axis='x',labelbottom=ax==axs[-1])
 if ax==axs[-1]:
  excerpts=[paper.wrap_tokens(x[3],6 if x[1]=='user' else 12 if x[1]=='cot' else 20,ellipsis='…') for x in labels]
  ax.set_xticklabels(excerpts,ha='left',fontsize=7);ax.tick_params(axis='x',pad=14)
  for tick,x in zip(ax.get_xticklabels(),labels):tick.set_color(TEXT[x[1]])
  for start,role,n,_ in labels:ax.annotate(f'{LABELS[role]} {n}',(start,0),xycoords=('data','axes fraction'),xytext=(0,-2),textcoords='offset points',ha='left',va='top',fontsize=6.7,color='#252525')
fig.text(.11,.975,'Local GPT-OSS-20B role readout on the authors’ gardening conversation',fontweight='bold',fontsize=9)
fig.text(.11,.937,'Layer 12 · four-role probe · 512 displayed tokens from one conversation',fontsize=8)
fig.text(.018,.52,'CoTness',rotation=90,va='center',fontsize=8,fontweight='bold')
fig.text(.11,.018,'Same matched display subset across conditions; the numerical replication gap is reported in the text.',fontsize=7)
fig.subplots_adjust(left=.11,right=.97,top=.84,bottom=.20,hspace=.55)
save(fig,'00-gardening');PLOTTED['00-gardening']={'rows':1536,'tokens_each':512}

rows=read(SESSION/'presentation/probe-offset-illustration/plotted-probabilities.csv')
display=garden[garden.prompt_key=='everything_in_user_tags'].sort_values('display_token_ix');labels=passage_labels(display)
excerpts=[paper.wrap_tokens(x[3],4 if x[1]=='user' else 7 if x[1]=='cot' else 10,max_lines=2,ellipsis='…') for x in labels]
for fraction,name in [(.01,'01-small-offsets'),(.05,'02-large-offsets')]:
 fig,axes=plt.subplots(4,3,figsize=(9,6.8),sharex=True,sharey=True)
 for edit_ix,edited_role in enumerate(['user','tool']):
  for col,sign in enumerate([-1,0,1]):
   part=rows[(rows.offset_fraction==fraction)&(rows.edited_role==edited_role)&(rows.sign==sign)].sort_values('display_token_ix');assert len(part)==512
   for metric_ix,target in enumerate(['user','tool']):
    row=edit_ix*2+metric_ix;ax=axes[row,col];panel(ax)
    for role in ['user','cot','assistant']:
     q=part[part.base_message_type==role];ax.scatter(q.display_token_ix,q['p_'+target],s=1.1,linewidths=0,color=COLORS[role],alpha=.9)
    ax.set_ylim(-.02,1.02);ax.set_xlim(1,512);ax.set_yticks([0,1],labels=['0%','100%']);ax.tick_params(axis='y',labelsize=7)
    ax.set_xticks([x[0] for x in labels]);ax.tick_params(axis='x',labelbottom=row in [1,3])
    if row in [1,3]:
     ax.set_xticklabels(excerpts,ha='left',fontsize=6);ax.tick_params(axis='x',pad=12)
     for tick,x in zip(ax.get_xticklabels(),labels):tick.set_color(TEXT[x[1]])
     for start,role,n,_ in labels:ax.annotate({'user':'U','cot':'C','assistant':'A'}[role]+str(n),(start,0),xycoords=('data','axes fraction'),xytext=(0,-2),textcoords='offset points',ha='left',va='top',fontsize=6,color='#252525')
    if col==0:ax.set_ylabel(target.title()+'ness',fontsize=8,fontweight='bold',labelpad=6)
    if metric_ix==0:ax.set_title('Unmodified' if sign==0 else ('Subtract ' if sign<0 else 'Add ')+edited_role.title()+' direction',fontsize=8,fontweight='bold',pad=3)
 fig.text(.12,.977,f'±{fraction:.0%} offsets along saved User and Tool probe directions',fontsize=10,fontweight='bold')
 fig.text(.12,.945,'GPT-OSS-20B · layer 12 · five-role probe · one conversation · 512 matched tokens · all User tags',fontsize=8)
 fig.text(.12,.092,'Passage key: U = User, C = CoT (reasoning), A = Assistant; 1 and 2 identify the two turns of each role.',fontsize=7.4)
 fig.text(.12,.067,'Reference norm = 45.2471, the median across 921 forwarded tokens; 1% / 5% give lengths 0.4525 / 2.2624.',fontsize=7.4)
 fig.text(.12,.042,'Every token and both directions use the same fixed offset length at each strength. Colors retain the paper’s role meanings.',fontsize=7.4)
 fig.text(.12,.017,'Offline rescoring; no downstream model computation or behavior. “Subtract” applies an opposite offset, not component removal.',fontsize=7.4)
 fig.subplots_adjust(left=.12,right=.95,top=.89,bottom=.19,wspace=.18,hspace=.68)
 save(fig,name);PLOTTED[name]={'probability_points':6144,'tokens_each_panel':512,'offset_fraction':fraction}

means=read(HERE/'data/offset-dose-response.csv');assert len(means)==30
fig,axs=plt.subplots(2,2,figsize=(7.2,5.0),sharex=True,sharey=True)
for col,direction in enumerate(['user','tool']):
 for row,target in enumerate(['user','tool']):
  ax=axs[row,col];panel(ax)
  for role in ['user','cot','assistant']:
   q=means[(means.edited_role==direction)&(means.base_message_type==role)].sort_values('signed_offset_percent')
   ax.plot(q.signed_offset_percent,q['p_'+target],color=COLORS[role],lw=.7,marker=MARKERS[role],ms=3.2,label=LABELS[role])
  ax.axvline(0,color='#999999',lw=.4,ls='--');ax.set_ylim(-.02,1.02);ax.set_xlim(-5.35,5.35)
  ax.set_yticks([0,.5,1],labels=['0%','50%','100%']);ax.set_xticks([-5,-1,0,1,5],labels=['−5','−1','0','+1','+5'])
  if row==0:ax.set_title(LABELS[direction]+' direction',fontweight='bold')
  if col==0:ax.set_ylabel(LABELS[target]+'ness',fontweight='bold')
  if row==1:ax.set_xlabel('Signed offset (% of reference norm)')
q=means[(means.edited_role=='tool')&(means.base_message_type=='user')].set_index('signed_offset_percent')
deltas={role:100*(q.loc[1,'p_'+role]-q.loc[0,'p_'+role]) for role in ['user','tool']}
fig.text(.11,.977,'A small Tool offset raises both scores for original User text',fontweight='bold',fontsize=10.5)
fig.text(.11,.94,f'0 → +1% Tool: Userness +{deltas["user"]:.1f} points; Toolness +{deltas["tool"]:.1f} points (before rounding the means)',fontsize=8)
handles,names=axs[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.54,.085),ncol=3,frameon=False,handlelength=1.6)
fig.text(.11,.064,'GPT-OSS-20B · layer 12 · five-role probe · one conversation; each mean uses the same source-role tokens.',fontsize=7.2)
fig.text(.11,.038,'Reference norm = 45.2471 across 921 forwarded tokens. Lines join five evaluated strengths; no intermediate values measured.',fontsize=7)
fig.text(.11,.012,'Offline rescoring of 512 saved states; no model continuation or behavior. Marker shapes also identify source roles.',fontsize=7.2)
fig.subplots_adjust(left=.11,right=.96,top=.855,bottom=.245,hspace=.24,wspace=.18)
save(fig,'03-offset-dose-response');PLOTTED['03-offset-dose-response']={'probability_means':60,'groups':30,'small_tool_deltas_pp':deltas}

r=read(HERE/'data/rh6-plotted.csv');assert len(r)==24
fig,axs=plt.subplots(2,2,figsize=(7.2,5.0),sharex=True,sharey='row')
for row,metric in enumerate(['p_user','log_user_tool']):
 for col,space in enumerate(['sucat','uat']):
  ax=axs[row,col];panel(ax);ax.axhline(0,color='#999999',lw=.5,ls='--',zorder=1)
  for order,label in [('B','Marker first'),('A','Marker second')]:
   q=r[(r.metric==metric)&(r.space==space)&(r.order==order)].sort_values('layer')
   ax.errorbar(q.layer,q.plot_mean,yerr=np.vstack((q.plot_mean-q.plot_lo,q.plot_hi-q.plot_mean)),label=label,color=GRAY,lw=.85,ls='-' if order=='B' else '--',marker='o',ms=3,mew=.65,mfc=GRAY if order=='B' else 'white',elinewidth=.65,capsize=2,capthick=.65,zorder=3)
  ax.set_xlim(7.4,16.6);ax.set_xticks([8,12,16])
  if row==0:
   ax.set_ylim(-5,25);ax.set_yticks([-5,0,10,20]);ax.set_title('SUCAT · five-role probe' if space=='sucat' else 'UAT · three-role probe',fontweight='bold')
  else:ax.set_ylim(-.08,1.65);ax.set_yticks([0,.5,1,1.5],labels=['0','0.5','1.0','1.5']);ax.set_xlabel('Layer index')
axs[0,0].set_ylabel('Adjusted Userness effect\n(percentage points)',fontweight='bold')
axs[1,0].set_ylabel('Adjusted ln(User/Tool) effect',fontweight='bold')
fig.text(.13,.975,'Adjusted relative effects are positive in all 12 plotted settings',fontweight='bold',fontsize=10.5)
fig.text(.13,.935,'Marker: (permission − ordinary request) − replacement-span effect · 100 template–page units',fontsize=8)
handles,names=axs[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.55,.078),ncol=2,frameon=False,handlelength=2.5)
fig.text(.13,.054,'User/Tool = mean User probability / mean Tool probability within each span; logs precede paired contrasts.',fontsize=7)
fig.text(.13,.03,'Mean adjusted Userness is negative at layer 12 with SUCAT and positive with UAT. No action outcomes measured.',fontsize=7)
fig.text(.13,.006,'Pointwise 95% paired-bootstrap intervals. Lines join the three measured layers; intervening layers are not displayed.',fontsize=7)
fig.subplots_adjust(left=.13,right=.96,top=.855,bottom=.22,hspace=.24,wspace=.18)
save(fig,'04-rh6-readout');PLOTTED['04-rh6-readout']={'estimates':24,'interval_bounds':48,'units_each':100}

r=read(HERE/'data/probe-recall-by-layer.csv');assert len(r)==240
fig,axs=plt.subplots(1,2,figsize=(7.2,3.6),sharey=True)
for ax,split,title in zip(axs,['prompt','base'],['Prompt split: 124 held-out variants','Text split: 24 held-out base texts']):
 panel(ax)
 for role in ['system','user','cot','assistant','tool']:
  q=r[(r.split==split)&(r.role==role)].sort_values('layer_ix');assert len(q)==24
  ax.plot(q.layer_ix,q.recall,color=COLORS[role],lw=.65,marker=MARKERS[role],ms=2.5,label=LABELS[role])
 ax.set_ylim(-.02,1.02);ax.set_xlim(-.3,23.3);ax.set_yticks([0,.5,1],labels=['0%','50%','100%']);ax.set_xticks([0,8,12,16,23]);ax.set_xlabel('Layer index');ax.set_title(title,fontsize=9,fontweight='bold')
axs[0].set_ylabel('Recall within each true role',fontweight='bold')
fig.text(.11,.965,'Held-out role recall varies by layer and split',fontweight='bold',fontsize=11)
fig.text(.11,.915,'GPT-OSS-20B · five-role probes · all 24 layers · 249 base texts in the full corpus',fontsize=8.5)
handles,names=axs[0].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.53,.08),ncol=5,frameon=False,handlelength=1.5,columnspacing=1)
fig.text(.11,.044,'Color and marker shape identify the true role label of held-out neutral-text tokens. Recall is token-weighted.',fontsize=7.2)
fig.text(.11,.014,'Separately fitted probes and different test sets; this evaluates role decoding under the neutral-text construction.',fontsize=7.2)
fig.subplots_adjust(left=.11,right=.97,top=.8,bottom=.31,wspace=.2)
save(fig,'05-probe-recall');PLOTTED['05-probe-recall']={'recall_values':240}

manifest={'purpose':'Review copies with clearer labels, redundant role markers and measured-value callouts; source figures preserved.','sources':SOURCES,'renderer':fingerprint(Path(__file__)),'palette':COLORS,'font':'TeX Gyre Termes','role_markers':MARKERS,'plotted':PLOTTED,'artifacts':[fingerprint(p) for p in sorted(OUT.glob('*')) if p.suffix in ['.png','.pdf']],'new_model_runs':0,'new_gpu_jobs':0}
(HERE/'data/review-figures-provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
print({'figures':len(PLOTTED),'artifacts':len(manifest['artifacts']),'output':str(OUT),'new_model_runs':0})
