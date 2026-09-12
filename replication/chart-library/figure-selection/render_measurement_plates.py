"""I refine four measurement plates from frozen data without changing a plotted value."""
from pathlib import Path
import hashlib,json,textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
LIB=HERE.parent
SESSION=LIB.parent/'cloud/persistent/sessions/20260911T030050Z'
SRC=SESSION/'outputs/20260911T030050Z/appendix-e-L12'
OUT=HERE/'figures'; DATA=HERE/'data'
OUT.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
FONT=LIB/'fonts'
for kind in ['regular','bold','italic']:
 font_manager.fontManager.addfont(FONT/f'texgyretermes-{kind}.otf')
plt.rcParams.update({'font.family':'TeX Gyre Termes','font.size':9,'text.usetex':False,'text.parse_math':False,'pdf.fonttype':42,'svg.fonttype':'none','axes.linewidth':.55,'savefig.facecolor':'white'})
COLORS={'user':'#00a6f4','cot':'#fd9a00','assistant':'#00d492'}
TEXT={'user':'#0084d1','cot':'#e17100','assistant':'#009966'}
ROLES=['user','cot','assistant','user','cot','assistant']
ROLELABELS=['User 1','CoT 1','Assistant 1','User 2','CoT 2','Assistant 2']
SHORT=['U1','C1','A1','U2','C2','A2']
LABEL={'user':'User','cot':'CoT','assistant':'Assistant','tool':'Tool'}
INK='#1b303f';GRAY='#62748e'
MARKERS={'user':'o','cot':'^','assistant':'s'}
GARDEN=SRC/'figure-7-displayed-rows.csv'
OFFSETS=SESSION/'presentation/probe-offset-illustration/plotted-probabilities.csv'
DOSE=LIB/'data/offset-dose-response.csv'
MESSAGES=SRC/'original-messages.json'
garden=pd.read_csv(GARDEN,keep_default_na=False,float_precision='round_trip')
offsets=pd.read_csv(OFFSETS,keep_default_na=False,float_precision='round_trip')
dose=pd.read_csv(DOSE,keep_default_na=False,float_precision='round_trip')
messages=json.loads(MESSAGES.read_text())
excerpts=[messages[1]['content'].split('. ')[0]+'.', messages[2]['content'].split('. ')[0]+'.', messages[3]['content'].split('\n')[0], messages[4]['content'].split('!! ')[1].split('?')[0]+'?', messages[5]['content'].split('. ')[0]+'.', messages[6]['content'].split('\n')[0].strip().replace('**','')]
for i,excerpt in enumerate(excerpts,1):
 assert excerpt.replace('**','') in messages[i]['content'].replace('**','')
base=garden[garden.prompt_key=='everything_in_user_tags'].sort_values('display_token_ix')
passages=[]
for seg,q in base.groupby('seg_ix',sort=True):
 passages.append({'start':int(q.display_token_ix.iloc[0]),'n':len(q),'role':q.base_message_type.iloc[0]})
assert len(passages)==6
ALL=[]
def fingerprint(p):return {'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
def panel(ax):
 ax.grid(axis='x',color='#dddddd',lw=.35);ax.set_axisbelow(True)
 for spine in ax.spines.values():spine.set_color('#454545')
 ax.tick_params(width=.45,length=2,pad=2,labelsize=8)
 ax.set_ylim(-.02,1.02);ax.set_xlim(1,512);ax.set_yticks([0,1],labels=['0%','100%'])
 ax.set_xticks([p['start'] for p in passages],labels=[])
def scatter(ax,q,target):
 for role in COLORS:
  p=q[q.base_message_type==role]
  a=ax.scatter(p.display_token_ix,p[target],s=1.05,linewidths=0,color=COLORS[role],alpha=.92,rasterized=False)
  assert np.array_equal(np.asarray(a.get_offsets()),p[['display_token_ix',target]].to_numpy())
def title(fig,main,subtitle):
 fig.text(.09,.973,main,fontsize=12,fontweight='bold',color=INK,va='top')
 fig.text(.09,.926,subtitle,fontsize=9,color=GRAY,va='top')
def transcript(fig,ylabel,ytext,angle=35,fontsize=7.2):
 labels=[]
 for i,(role,label,excerpt) in enumerate(zip(ROLES,ROLELABELS,excerpts)):
  x=.09+i*.1465
  t=fig.text(x,ylabel,label,fontsize=8.3,fontweight='bold',va='top',color=INK);labels.append(t)
  wrapped=textwrap.fill(excerpt,width=25,break_long_words=False,break_on_hyphens=False)
  assert wrapped.split()==excerpt.split()
  t=fig.text(x,ytext,wrapped,fontsize=fontsize,color=TEXT[role],rotation=angle,ha='left',va='top',linespacing=1.15);labels.append(t)
 return labels
def save(fig,name,meta,checked_text):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();bounds=[]
 for t in checked_text:
  bb=t.get_window_extent(renderer).transformed(fig.dpi_scale_trans.inverted())
  assert bb.x0>=-.01 and bb.y0>=-.01 and bb.x1<=fig.get_figwidth()+.01 and bb.y1<=fig.get_figheight()+.01,(name,t.get_text(),list(bb.bounds))
  bounds.append({'text':t.get_text(),'bounds_inches':list(bb.bounds)})
 for ext in ['png','pdf','svg']:fig.savefig(OUT/f'{name}.{ext}',dpi=450)
 meta.update({'id':name,'renderer':fingerprint(Path(__file__)),'palette':COLORS,'text_palette':TEXT,'font':'TeX Gyre Termes','size_inches':list(fig.get_size_inches()),'plotted_values_unchanged':True,'text_extents':bounds,'artifacts':[fingerprint(OUT/f'{name}.{ext}') for ext in ['png','pdf','svg']],'new_model_runs':0})
 (DATA/f'{name}-provenance.json').write_text(json.dumps(meta,indent=2)+'\n');ALL.append(meta);plt.close(fig)
# I keep all three formats and add complete source openings below the gardening plate.
fig=plt.figure(figsize=(8.3,5.1))
title(fig,'Role patterns persist when message tags change','GPT-OSS-20B · layer 12 · one gardening conversation · 512 matched tokens')
axes=[]
for i,(condition,name) in enumerate(zip(['proper_tags','basic_no_format','everything_in_user_tags'],['A · Correct tags','B · No tags','C · All text in one User message'])):
 ax=fig.add_axes([.09,.727-i*.174,.875,.118]);axes.append(ax);panel(ax)
 q=garden[garden.prompt_key==condition].sort_values('display_token_ix');assert len(q)==512
 scatter(ax,q,'prob');ax.set_title(name,fontsize=9.5,fontweight='bold',pad=5)
fig.text(.025,.61,'CoTness',rotation=90,va='center',fontweight='bold',fontsize=10)
for i,p in enumerate(passages):
 mid=axes[-1].transData.transform((p['start']+(p['n']-1)/2,0))[0]/fig.bbox.width
 fig.add_artist(Line2D([mid,.09+i*.1465+.025],[.37,.326],color='#b8b8b8',lw=.5,transform=fig.transFigure))
t=transcript(fig,.315,.275)
meta={'title':'Role patterns persist when message tags change','caption':'The authors’ gardening example retains a visible reasoning-role pattern under changed tags, while our mean scores remain below the paper’s reported values.','scope':'Qualitative Figure 7 match, with an unresolved numerical gap; one supplied conversation, a four-role probe, and 512 matched displayed tokens per format.','sources':[fingerprint(GARDEN),fingerprint(MESSAGES)],'points':1536,'excerpts':excerpts,'technical_details':'All panels use the same matched display subset. Originally reasoning tokens average 64.3%, 71.8% and 71.7% across the displayed conditions. The complete-context model forward and the subset are described in the original session GARDENING-RESULTS.md. Colors identify original passage roles, rather than the displayed tags.'}
save(fig,'gardening',meta,t)
# I keep the complete 4-by-3 offset comparison and print each source opening once.
for fraction,name in [(.01,'small-offsets'),(.05,'large-offsets')]:
 fig=plt.figure(figsize=(8.3,6.75))
 headline='Small offsets change the role readout unevenly' if fraction==.01 else 'Large offsets change the separation between passage roles'
 title(fig,headline,f'±{fraction:.0%} reference-norm offset · layer 12 · 512 matched tokens · offline readout')
 axs=[]
 for erow,edited in enumerate(['user','tool']):
  for c,sign in enumerate([-1,0,1]):
   q=offsets[(offsets.offset_fraction==fraction)&(offsets.edited_role==edited)&(offsets.sign==sign)].sort_values('display_token_ix');assert len(q)==512
   for mrow,target in enumerate(['user','tool']):
    row=erow*2+mrow
    y=[.753,.605,.408,.26][row]
    ax=fig.add_axes([.09+c*.304,y,.267,.111]);axs.append(ax);panel(ax);scatter(ax,q,'p_'+target)
    if c==0:ax.set_ylabel(target.title()+'ness',fontweight='bold',fontsize=9,labelpad=6)
    else:ax.set_yticklabels([])
    if mrow==0:ax.set_title('Unmodified' if sign==0 else ('Subtract ' if sign<0 else 'Add ')+edited.title()+' direction',fontsize=8.7,fontweight='bold',pad=5)
    if mrow==1:ax.set_xticklabels(SHORT,fontsize=6.8,ha='left');ax.tick_params(axis='x',pad=2)
 t=transcript(fig,.214,.182,fontsize=6.9)
 meta={'title':headline,'caption':('At a 1% offset, adding User or Tool coefficients changes the two readouts differently across the same six passages.' if fraction==.01 else 'At 5%, the User offset raises Userness across User, reasoning and Assistant passages, while the Tool offset produces a different redistribution.'),'scope':'Offline rescoring of saved activations: no continued model computation or behavioral outcome. The same single conversation, token cohort, five-role probe and axes are used at both strengths.','sources':[fingerprint(OFFSETS),fingerprint(GARDEN),fingerprint(MESSAGES)],'panels':12,'points':6144,'offset_fraction':fraction,'excerpts':excerpts,'technical_details':'Each offset is sign × fraction × 45.2471 × normalized saved User or Tool coefficient vector. The common reference is the median activation norm across 921 forwarded tokens in the all-User condition; fixed lengths are 0.4525 and 2.2624. Subtraction applies the opposite offset, not component removal. The saved parameterization matters for individual coefficient directions. All 12 panels and every plotted point are preserved. U/C/A identify User/CoT/Assistant; numbers identify the two turns.'}
 save(fig,name,meta,t)
# I preserve all five measured strengths and all role curves in the dose overview.
fig,axs=plt.subplots(2,2,figsize=(8.3,5.4),sharex=True,sharey=True)
title(fig,'Userness and Toolness need not trade off','Mean score by original passage role · one conversation · layer 12 · offline readout')
for col,direction in enumerate(['user','tool']):
 for row,target in enumerate(['user','tool']):
  ax=axs[row,col]
  ax.grid(color='#dddddd',lw=.35);ax.set_axisbelow(True)
  for sp in ax.spines.values():sp.set_color('#454545')
  for role in COLORS:
   q=dose[(dose.edited_role==direction)&(dose.base_message_type==role)].sort_values('signed_offset_percent');assert len(q)==5
   line,=ax.plot(q.signed_offset_percent,q['p_'+target],color=COLORS[role],lw=1.0,marker=MARKERS[role],ms=4,label=LABEL[role])
   assert np.array_equal(line.get_xydata(),q[['signed_offset_percent','p_'+target]].to_numpy())
  ax.axvline(0,color='#aab0b5',lw=.5,ls='--');ax.set_ylim(-.025,1.025);ax.set_xlim(-5.4,5.4)
  ax.set_yticks([0,.5,1],labels=['0%','50%','100%']);ax.set_xticks([-5,-1,0,1,5],labels=['−5','−1','0','+1','+5']);ax.tick_params(width=.45,length=2,pad=3,labelsize=8.5)
  if row==0:ax.set_title('Offset along '+LABEL[direction]+' coefficients',fontweight='bold',fontsize=10)
  if col==0:ax.set_ylabel(target.title()+'ness',fontweight='bold',fontsize=10)
  if row==1:ax.set_xlabel('Signed offset (% of reference norm)',fontsize=9)
q=dose[(dose.edited_role=='tool')&(dose.base_message_type=='user')].set_index('signed_offset_percent')
deltas={role:float(100*(q.loc[1,'p_'+role]-q.loc[0,'p_'+role])) for role in ['user','tool']}
for row,target in enumerate(['user','tool']):
 ax=axs[row,1];p=q.loc[1,'p_'+target]
 if row==0:
  ax.annotate(f'+1% Tool offset\nUserness +{deltas[target]:.1f} points',xy=(1,p),xytext=(2.0,.87),textcoords='data',fontsize=8,color=INK,ha='left',va='center',arrowprops={'arrowstyle':'-','color':'#9099a2','lw':.65})
 else:
  ax.text(-4.65,.83,f'+1% Tool offset on User text\nToolness +{deltas[target]:.1f} points',fontsize=8,color=TEXT['user'],ha='left',va='center')
handles,names=axs[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='lower center',bbox_to_anchor=(.55,.025),ncol=3,frameon=False,handlelength=1.8,title='Original passage role',title_fontsize=8.5,fontsize=9)
fig.subplots_adjust(left=.09,right=.965,top=.818,bottom=.20,hspace=.28,wspace=.18)
meta={'title':'Userness and Toolness need not trade off','caption':'For originally User text, a small positive Tool offset raises both scores; a larger offset raises Toolness while lowering Userness.','scope':'Deterministic offline rescoring of 512 saved token states in one conversation, using five strengths and a five-role probe. Lines connect measured strengths only.','sources':[fingerprint(DOSE),fingerprint(OFFSETS)],'panels':4,'probability_means':60,'small_tool_deltas_pp':deltas,'technical_details':'Each point averages tokens within one original passage-role group, pooling its two turns. All groups and both readouts are retained. The common reference norm is 45.2471 over 921 forwarded tokens. There is no model continuation, behavioral outcome or population confidence interval.'}
save(fig,'offset-dose-response',meta,[])
(DATA/'measurement-plates.json').write_text(json.dumps({'purpose':'I make the existing measurement figures more legible without changing the underlying values.','figures':ALL},indent=2)+'\n')
print(json.dumps({'figures':[m['id'] for m in ALL],'artifacts':12,'measurement_data_unchanged':True},indent=2))
