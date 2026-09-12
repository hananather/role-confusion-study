"""I render a small set of figures from my saved experimental records."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE=Path(__file__).resolve().parent
REPL=HERE.parent
SESSION=REPL/'cloud/persistent/sessions/20260911T030050Z'
SOURCE=SESSION/'outputs/20260911T030050Z'
PRESENT=SESSION/'presentation'
COLORS={'system':'#90a1b9','user':'#00a6f4','cot':'#fd9a00','assistant':'#00d492','tool':'#7e6cff'}
LABELS={'system':'System','user':'User','cot':'CoT','assistant':'Assistant','tool':'Tool'}
FONT=Path('/usr/local/texlive/2019/texmf-dist/fonts/opentype/public/tex-gyre')
for variant in ['regular','bold','italic','bolditalic']:
    font_manager.fontManager.addfont(FONT/f'texgyretermes-{variant}.otf')
plt.rcParams.update({'font.family':'TeX Gyre Termes','font.size':9,'text.usetex':False,
 'text.parse_math':False,'pdf.fonttype':42,'axes.linewidth':.5,'savefig.facecolor':'white'})
for sub in ['figures','data']: (HERE/sub).mkdir(exist_ok=True)
SOURCES=[]
def source(p):
    p=Path(p)
    SOURCES.append({'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    return p

def panel(ax):
    ax.grid(color='#d9d9d9',linewidth=.3)
    ax.set_axisbelow(True)
    for s in ax.spines.values():s.set_color('#333333')
    ax.tick_params(width=.4,length=2,pad=3)

def save(fig,name):
    for ext in ['png','pdf']:fig.savefig(HERE/'figures'/f'{name}.{ext}',dpi=300)
    plt.close(fig)

def offsets():
    d=pd.read_csv(source(PRESENT/'probe-offset-illustration/plotted-probabilities.csv'),float_precision='round_trip')
    meta=json.loads(source(PRESENT/'probe-offset-illustration/provenance.json').read_text())
    assert len(d)==6144
    assert set(d.base_message_type)=={'user','cot','assistant'}
    keys=['edited_role','sample_ix']
    z=d[d.sign==0]
    a=z[z.offset_fraction==.01].set_index(keys)
    b=z[z.offset_fraction==.05].set_index(keys)
    assert np.array_equal(a.sort_index()[['p_user','p_tool']].values,b.sort_index()[['p_user','p_tool']].values)
    d=d[(d.sign!=0)|(d.offset_fraction==.01)].copy()
    d['signed_offset_percent']=100*d.sign*d.offset_fraction
    means=d.groupby(['edited_role','signed_offset_percent','base_message_type']).agg(
      p_user=('p_user','mean'),p_tool=('p_tool','mean'),n_tokens=('sample_ix','size')).reset_index()
    assert len(means)==30
    means.to_csv(HERE/'data/offset-dose-response.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(7.2,4.8),sharex=True,sharey=True)
    for col,direction in enumerate(['user','tool']):
      for row,target in enumerate(['user','tool']):
        ax=axs[row,col];panel(ax)
        for role in ['user','cot','assistant']:
          q=means[(means.edited_role==direction)&(means.base_message_type==role)].sort_values('signed_offset_percent')
          ax.plot(q.signed_offset_percent,q['p_'+target],color=COLORS[role],lw=.7,marker='o',ms=2.3,label=LABELS[role])
        ax.axvline(0,color='#999999',lw=.4,ls='--')
        ax.set_ylim(-.02,1.02);ax.set_xlim(-5.35,5.35)
        ax.set_yticks([0,.5,1],labels=['0%','50%','100%'])
        ax.set_xticks([-5,-1,0,1,5],labels=['−5','−1','0','+1','+5'])
        if row==0:ax.set_title(LABELS[direction]+' direction',fontweight='bold')
        if col==0:ax.set_ylabel(LABELS[target]+'ness',fontweight='bold')
        if row==1:ax.set_xlabel('Signed offset (% of reference norm)')
    fig.text(.11,.975,'Offset direction and size change the role-score pattern',fontweight='bold',fontsize=11)
    fig.text(.11,.935,'My saved activations · layer 12 · five-role probe · one gardening conversation',fontsize=8.5)
    handles,labels=axs[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.54,.062),ncol=3,frameon=False,handlelength=1.6)
    fig.text(.11,.034,'Each point averages the same source-role tokens; 512 matched tokens in total. Lines join five evaluated offsets.',fontsize=7.2)
    fig.text(.11,.009,'Offline arithmetic only: I edit saved states and rescore them. No downstream model pass or behavior is measured.',fontsize=7.2)
    fig.subplots_adjust(left=.11,right=.96,top=.855,bottom=.20,hspace=.24,wspace=.18)
    save(fig,'03-offset-dose-response')
    return {'n_conversations':1,'n_display_tokens':512,'reference_tokens':meta['reference_token_count'],
      'reference_norm':meta['reference_median_activation_norm'],'new_model_forward':False,'source_role_counts':d[d.signed_offset_percent==0].groupby('edited_role').base_message_type.value_counts().to_dict().__str__()}

def recalls():
    rows=[]
    audit=json.loads(source(SOURCE/'probes-full/split-audit.json').read_text())['splits']
    for split,suffix in [('prompt',''),('base','-basesplit')]:
      d=pd.read_csv(source(SOURCE/f'probes-full/acc_by_role_gptoss-20b{suffix}.csv'))
      d=d[d.role_space=='system,user,cot,assistant,tool'].copy()
      d['correct']=d['count'].where(d.role==d.pred,0)
      r=d.groupby(['layer_ix','role'])[['count','correct']].sum().reset_index()
      r['recall']=r.correct/r['count'];r['split']=split;rows.append(r)
    data=pd.concat(rows,ignore_index=True)
    assert len(data)==240
    assert data.groupby(['split','role'])['count'].nunique().eq(1).all()
    assert data.recall.between(0,1).all()
    data.to_csv(HERE/'data/probe-recall-by-layer.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(7.2,3.35),sharey=True)
    for ax,split,title in zip(axs,['prompt','base'],['Prompt split: 124 held-out variants','Text split: 24 held-out base texts']):
      panel(ax)
      for role in ['system','user','cot','assistant','tool']:
        r=data[(data.split==split)&(data.role==role)]
        ax.plot(r.layer_ix,r.recall,color=COLORS[role],lw=.65,marker='o',ms=1.7,label=LABELS[role])
      ax.set_ylim(-.02,1.02);ax.set_xlim(-.3,23.3)
      ax.set_yticks([0,.5,1],labels=['0%','50%','100%'])
      ax.set_xticks([0,8,12,16,23]);ax.set_xlabel('Layer index')
      ax.set_title(title,fontsize=9,fontweight='bold')
    axs[0].set_ylabel('Correct-role classification',fontweight='bold')
    fig.text(.11,.965,'Probe quality varies by role, depth and split',fontweight='bold',fontsize=11)
    fig.text(.11,.915,'My H100 run · five-role probe · all 24 layers · 249 base texts in the full corpus',fontsize=8.5)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.055),ncol=5,frameon=False,handlelength=1.5,columnspacing=1)
    fig.text(.11,.018,'Token-weighted held-out recall. These splits use different evaluation sets; their difference is descriptive.',fontsize=7.2)
    fig.subplots_adjust(left=.11,right=.97,top=.80,bottom=.28,wspace=.20)
    save(fig,'05-probe-recall')
    return {k:{field:v for field,v in audit[k].items() if field.startswith('n_')} for k in ['prompt:sucat','base:sucat']}

if __name__=='__main__':
    result={'offline_offsets':offsets(),'probe_recall':recalls()}
    for original,target in [('my-h100-gardening-overview','00-gardening'),('probe-offset-illustration/probe-offsets-1pct','01-small-offsets'),('probe-offset-illustration/probe-offsets-5pct','02-large-offsets')]:
      for ext in ['png','pdf']:
        p=source(PRESENT/f'{original}.{ext}');shutil.copyfile(p,HERE/'figures'/f'{target}.{ext}')
    result.update({'sources':SOURCES,'colors':COLORS,'font':'TeX Gyre Termes','new_gpu_jobs':0,'validation':'Exact baseline identity, fixed matched token cohort, counts and bounded probabilities checked.'})
    (HERE/'data/render-provenance.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'figures':5,'new_model_runs':0,'verified':True}))
