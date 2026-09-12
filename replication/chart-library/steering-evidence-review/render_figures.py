"""I show complete cohorts using the paper's palette and Termes typography."""
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE/'data'; OUT = HERE/'figures'; OUT.mkdir(exist_ok=True)
for variant in ('regular', 'bold', 'italic', 'bolditalic'):
    font_manager.fontManager.addfont(HERE.parent/f'fonts/termes-ttf/texgyretermes-{variant}.ttf')
plt.rcParams.update({'font.family':'TeX Gyre Termes','font.size':11,'pdf.fonttype':42,
    'svg.fonttype':'path','text.parse_math':False,'axes.linewidth':.5,'axes.edgecolor':'#333333',
    'savefig.facecolor':'white','xtick.major.width':.5,'ytick.major.width':.5})
INK='#172b3a'; GRAY='#62748e'; GREEN='#00d492'; PINK='#ff637e'; BLUE='#00a6f4'; ORANGE='#fd9a00'; PURPLE='#7e6cff'
STATES={'UPLOAD':(PINK,'U'), 'NO_UPLOAD':(GREEN,'–'), 'CENSORED':('#90a1b9','?'),
        'UNRUN':('white','·'), 'UNASSIGNED':('#f3f3f3','/')}
LABELS={'none':'No intervention','zero':'Zero dose','role_a16':'Tool − CoT','reverse_a16':'Reversed vector',
        'random_0_a16':'Random 1','random_1_a16':'Random 2','random_2_a16':'Random 3',
        'sentence_dev_rule':'Provenance reminder','tool_raising_a16':'Tool-raising vector'}
ARMS=list(LABELS)
outcomes=pd.read_csv(DATA/'forgery-outcomes.csv',keep_default_na=False)
probes=pd.read_csv(DATA/'forgery-payload-probes.csv',keep_default_na=False)
counts=pd.read_csv(DATA/'arm-counts.csv')
artifacts=[]
def save(fig,name,data_files):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer()
    issues=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text(): continue
        box=t.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1):
            issues.append(t.get_text())
    assert not issues, (name,issues)
    files={}
    for ext in ('png','pdf','svg'):
        p=OUT/f'{name}.{ext}'; fig.savefig(p,dpi=320);files[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    artifacts.append({'name':name,'size_inches':list(fig.get_size_inches()),'files':files,
        'data':{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in data_files},
        'visible_text_within_canvas':True})
    plt.close(fig)
def state_legend(fig,y=.045):
    handles=[Patch(facecolor=c,edgecolor=GRAY,label=label) for c,label in
             [(PINK,'U  Uploaded'),(GREEN,'–  No upload'),('#90a1b9','?  Unresolved'),('white','·  Unrun'),('#f3f3f3','/  Not assigned')]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,y),ncol=5,frameon=False,
               fontsize=10,handlelength=1.2,handletextpad=.45,columnspacing=1.5)

# I preserve every page and arm in source order, including absent assignments.
fig=plt.figure(figsize=(9.2,6.6))
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,9.2),ylim=(6.6,0));ax.axis('off')
ax.text(.35,.18,'Upload outcomes across every tested page',fontsize=22,weight='bold',color=INK,va='top')
ax.text(.36,.65,'GPT-OSS-20B · H100 · one sampling seed per page',fontsize=11,color=GRAY,va='top')
x0=2.65; step=.47; gap=.28; y0=1.69; dy=.45
ax.text(x0+step*2,1.05,'Previously explored pages',ha='center',fontsize=12,weight='bold',color=INK)
ax.text(x0+step*7+gap,1.05,'New pages',ha='center',fontsize=12,weight='bold',color=INK)
ax.text(8.13,1.05,'Uploads / recorded',ha='center',fontsize=11,weight='bold',color=INK)
for c in range(10):
    x=x0+c*step+(gap if c>=5 else 0)
    ax.text(x,1.40,('H' if c<5 else 'N')+str(c%5+1),ha='center',fontsize=10,color=GRAY)
for i,arm in enumerate(ARMS):
    y=y0+i*dy
    if arm=='role_a16': ax.add_patch(Rectangle((.3,y-.21),8.5,.42,facecolor='#f3f5f7',edgecolor='none'))
    ax.text(.38,y,LABELS[arm],fontsize=12,weight='bold' if arm in ('role_a16','tool_raising_a16') else 'normal',color=INK,va='center')
    for c in range(10):
        cohort='historical' if c<5 else 'new'
        r=outcomes[(outcomes.arm_id==arm)&(outcomes.cohort==cohort)&(outcomes.page_id.astype(int)==c%5)]
        state=r.iloc[0].outcome if len(r) else 'UNASSIGNED'
        color,symbol=STATES[state];x=x0+c*step+(gap if c>=5 else 0)
        ax.add_patch(Rectangle((x-.196,y-.175),.392,.35,facecolor=color,
                               edgecolor=GRAY,linewidth=.55,linestyle='--' if state=='UNRUN' else '-'))
        ax.text(x,y,symbol,ha='center',va='center',fontsize=12,weight='bold',color=INK)
    s=counts[(counts.arm_id==arm)&(counts.cohort=='both')].iloc[0]
    total=f'{s.UPLOAD} / {s.recorded}'
    ax.text(8.14,y-.055 if s.CENSORED else y,total,ha='center',va='center',fontsize=11,weight='bold' if arm=='role_a16' else 'normal',color=INK)
    if s.CENSORED: ax.text(8.14,y+.125,f'{s.CENSORED} unresolved',ha='center',va='center',fontsize=8.5,color=GRAY)
ax.text(.38,6.00,'Original vector: 2 uploads prevented · 1 introduced · 7 outcomes unchanged',fontsize=12,weight='bold',color=INK)
state_legend(fig,.018)
save(fig,'01-complete-outcomes',[DATA/'forgery-outcomes.csv',DATA/'paired-changes.csv',DATA/'arm-counts.csv'])

# I plot all saved forged-payload scores, never infer missing no-hook measurements.
probe_arms=['zero','role_a16','reverse_a16','random_0_a16','random_1_a16','random_2_a16','tool_raising_a16']
fig,axs=plt.subplots(1,2,figsize=(9.2,6.5),sharey=True)
fig.text(.04,.963,'Low reasoning-role scores can coexist with uploads',fontsize=21,weight='bold',color=INK,va='top')
fig.text(.04,.883,'Every available forged-passage readout · each symbol is one episode',fontsize=11,color=GRAY,va='top')
for ax,cohort,title in zip(axs,('historical','new'),('A · Previously explored pages','B · New pages')):
    ax.set(xlim=(-5,105),ylim=(6.55,-.55));ax.set_yticks(range(7))
    ax.set_xticks([0,25,50,75,100],labels=['0','25','50','75','100'])
    ax.grid(axis='x',color='#d9d9d9',linewidth=.4);ax.set_axisbelow(True)
    ax.set_title(title,fontsize=12,weight='bold',pad=10)
    ax.set_xlabel('Reasoning-role score (%)',fontsize=12)
    ax.axhspan(.57,1.43,color='#f3f5f7',zorder=0)
    for i,arm in enumerate(probe_arms):
        q=probes[(probes.cohort==cohort)&(probes.arm_id==arm)].sort_values('page_id')
        if not len(q):
            ax.text(50,i,'Not assigned' if arm=='zero' else 'Unrun',ha='center',va='center',color=GRAY,fontsize=11)
            continue
        assert len(q)==5
        for j,(_,r) in enumerate(q.iterrows()):
            y=i+(j-2)*.14
            if r.outcome=='CENSORED':
                ax.scatter(r.p_cot*100,y,s=41,marker='X',color=GRAY,edgecolor=INK,linewidth=.35,zorder=4)
            else:
                ax.scatter(r.p_cot*100,y,s=42,marker='^' if r.outcome=='UPLOAD' else 'o',color=STATES[r.outcome][0],edgecolor=INK,linewidth=.35,zorder=4)
    ax.tick_params(axis='y',length=0,pad=8)
axs[0].set_yticklabels([LABELS[a] for a in probe_arms],fontsize=11)
handles=[Line2D([],[],ls='',marker=m,markersize=7,markerfacecolor=c,markeredgecolor=INK,markeredgewidth=.4,label=lab)
         for m,c,lab in [('^',PINK,'Uploaded'),('o',GREEN,'No upload'),('X',GRAY,'Unresolved')]]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.56,.01),ncol=3,frameon=False,fontsize=11)
fig.subplots_adjust(left=.23,right=.974,top=.82,bottom=.14,wspace=.18)
save(fig,'02-probe-behavior',[DATA/'forgery-payload-probes.csv'])

# I keep the earlier marker endpoint separate from the newer upload task.
permission_path=HERE/'reviews/permission-toolward-pairs.csv'
if permission_path.exists():
    p=pd.read_csv(permission_path)
    assert len(p)==21 and p.case_id.nunique()==8
    assert p.intervention_mean_tool_first_prefill.min()>.9999
    assert p.baseline_unauthorized_write.sum()==18 and p.intervention_unauthorized_write.sum()==17
    assert p.baseline_authorized_completion.sum()==21 and p.intervention_authorized_completion.sum()==19
    fig,axs=plt.subplots(1,3,figsize=(9.2,5.0),gridspec_kw={'width_ratios':[1.25,1,1]})
    fig.text(.04,.96,'Toolness saturates while unwanted writes persist',fontsize=21,weight='bold',color=INK,va='top')
    fig.text(.04,.859,'Separate permission task · 21 pairs, eight cases · repeated-context lines overlap',fontsize=11,color=GRAY,va='top')
    ax=axs[0]
    for _,r in p.iterrows():
        ax.plot([0,1],[100*r.baseline_mean_tool_first_prefill,100*r.intervention_mean_tool_first_prefill],color=PURPLE,alpha=.35,lw=.7)
        ax.scatter([0,1],[100*r.baseline_mean_tool_first_prefill,100*r.intervention_mean_tool_first_prefill],s=13,color=PURPLE,zorder=3)
    ax.set(xlim=(-.3,1.3),ylim=(-3,105),ylabel='Tool-role score (%)',xticks=[0,1],xticklabels=['No steering','Toolward'],yticks=[0,25,50,75,100])
    ax.set_title('A · Command readout',fontsize=12,weight='bold',pad=12)
    for ax,before,after,color,title in [(axs[1],18,17,PINK,'B · Unwanted write'),(axs[2],21,19,GREEN,'C · Authorized action')]:
        ax.bar([0,1],[before,after],color=color,edgecolor=INK,linewidth=.45,width=.56)
        for x,v in enumerate((before,after)):
            ax.text(x,v+.8,f'{v} / 21',ha='center',fontsize=13,weight='bold',color=INK)
        ax.set(xlim=(-.55,1.55),ylim=(0,25),xticks=[0,1],xticklabels=['No steering','Toolward'],yticks=[0,5,10,15,20],ylabel='Recorded episodes')
        ax.set_title(title,fontsize=12,weight='bold',pad=12)
    for ax in axs:
        ax.grid(axis='y',color='#d9d9d9',linewidth=.4);ax.set_axisbelow(True);ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=.08,right=.98,top=.73,bottom=.15,wspace=.45)
    save(fig,'03-permission-task',[permission_path])

(OUT/'render-provenance.json').write_text(json.dumps({'font':'TeX Gyre Termes','palette_source':'../../../../prompt-injection-as-role-confusion/r-utils/plots.r + workspace paper palette',
    'outcome_colors':{'uploaded':PINK,'no_upload':GREEN,'unresolved':GRAY},
    'color_semantics':'Figures 1–2 color outcomes, explicitly labelled in legends; Figure 3 uses Tool purple for probe scores and separate labelled action bars.',
    'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'artifacts':artifacts},indent=2)+'\n')
print(json.dumps([a['name'] for a in artifacts]))
