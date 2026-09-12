"""I show the complete six-passage dialogue below the unchanged role measurements."""
from pathlib import Path
import json, re, hashlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from render_mats_six_passage import HERE, setup_style, read_candidate, passage_labels, COLORS, TEXT, CONDITIONS, TITLES

def main():
    setup_style()
    source=HERE/'data/mats-hanan-dialogue-v2'
    out=HERE/'figures/mats-hanan-layouts/full-dialogue-balanced'
    out.mkdir(parents=True,exist_ok=True)
    rows,base,csv=read_candidate('mats-hanan-dialogue-v2',data_root=HERE/'data')
    raw=json.loads((source/'original-messages.json').read_text())[1:]
    # I render Markdown emphasis as ordinary text while retaining every word.
    text=[re.sub(r'\*','',x).replace('\u202f',' ').replace('\u00a0',' ') for x in raw]
    fontsize=8.2; leading=10.3; width=8.3; left=.63; right=.25; gap=.20
    colwidths=[1.60,1.60,width-left-right-2*gap-3.20]
    scratch=plt.figure(figsize=(width,1)); scratch.canvas.draw()
    renderer=scratch.canvas.get_renderer(); prop=FontProperties(family='TeX Gyre Termes',size=fontsize)
    def measure(s): return renderer.get_text_width_height_descent(s,prop,ismath=False)[0]*72/scratch.dpi
    def wrap(s,colwidth):
        lines=[]
        for para in s.splitlines():
            if not para.strip():
                if lines and lines[-1]!='':lines.append('')
                continue
            current=''
            for word in para.split():
                trial=(current+' '+word).strip()
                if current and measure(trial)>colwidth*72:
                    lines.append(current);current=word
                else:current=trial
            if current:lines.append(current)
        while lines and not lines[-1]:lines.pop()
        assert ' '.join(' '.join(lines).split())==' '.join(s.split())
        return lines
    lines=[wrap(s,colwidths[i%3]) for i,s in enumerate(text)];plt.close(scratch)
    units=lambda seq:sum(.5 if not line else 1 for line in seq)
    rowheights=[max(units(x) for x in lines[i:i+3])*leading/72+.34 for i in [0,3]]
    plot_top=.75; plotheight=.58; plotgap=.29
    text_top=plot_top+3*plotheight+2*plotgap+.45
    height=text_top+sum(rowheights)+.30+.48
    fig=plt.figure(figsize=(width,height))
    fig.text(left/width,1-.16/height,'Reasoning-role scores across dialogue formats',fontsize=11,fontweight='bold',va='top')
    fig.text(left/width,1-.39/height,'GPT-OSS-20B · layer 12 · four-role probe · 578 matched tokens from one conversation',fontsize=8.8,va='top')
    labels=passage_labels(base);axes=[]
    for i,(condition,title) in enumerate(zip(CONDITIONS,TITLES)):
        top=plot_top+i*(plotheight+plotgap)
        ax=fig.add_axes([left/width,1-(top+plotheight)/height,(width-left-right)/width,plotheight/height]);axes.append(ax)
        part=rows[rows.prompt_key==condition].sort_values('display_token_ix')
        for role,color in COLORS.items():
            group=part[part.base_message_type==role]
            artist=ax.scatter(group.display_token_ix,group.prob,s=1.2,color=color,linewidths=0,alpha=.9)
            assert np.array_equal(np.asarray(artist.get_offsets()),group[['display_token_ix','prob']].to_numpy())
        ax.set(xlim=(1,len(base)),ylim=(-.02,1.02));ax.set_yticks([0,1],['0%','100%'])
        ax.set_xticks([v[0] for v in labels]);ax.grid(color='#d9d9d9',linewidth=.3);ax.set_axisbelow(True)
        ax.tick_params(length=2,width=.4,labelsize=8)
        if i==2:ax.set_xticklabels([f'{role.title() if role!="cot" else "CoT"} {num}' for _,role,num,*_ in labels],ha='left',fontsize=7.5)
        else:ax.set_xticklabels([])
        ax.set_title(title,fontsize=8.8,fontweight='bold',pad=4)
    fig.text(.10/width,1-(plot_top+(3*plotheight+2*plotgap)/2)/height,'CoTness',rotation=90,va='center',fontsize=9,fontweight='bold')
    fig.text(left/width,1-(text_top-.19)/height,'Complete conversation',fontsize=9,fontweight='bold',va='top')
    texts=[];bounds=[]
    roles=['user','cot','assistant']*2
    for i,seq in enumerate(lines):
        row=i//3;col=i%3;start=text_top+(rowheights[0]+.30 if row else 0)
        colwidth=colwidths[col]
        x=left+sum(colwidths[:col])+col*gap
        role=roles[i];name=('CoT' if role=='cot' else role.title())+f' {row+1}'
        origin='Hanan' if role=='user' else 'model'
        fig.text(x/width,1-start/height,name+' · '+origin,color=TEXT[role],fontsize=8.5,fontweight='bold',va='top')
        offset=.22
        for line in seq:
            if line:
                artist=fig.text(x/width,1-(start+offset)/height,line,color=TEXT[role],fontsize=fontsize,va='top')
                texts.append(artist)
            offset+=(.5 if not line else 1)*leading/72
        bounds.append({'passage':name,'rendered_text':text[i],'line_count':len(seq),'font_size':fontsize,'column_width_inches':colwidth})
    fig.text(left/width,.16/height,'Full text is shown; plots retain a matched token subset. Generated advice is preserved as experimental input.',fontsize=7,va='bottom')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    assert all(t.get_window_extent(renderer).x1<fig.bbox.width-3 and t.get_window_extent(renderer).y0>20 for t in texts)
    for suffix in ['png','pdf','svg']:fig.savefig(out/f'full-dialogue-balanced.{suffix}',dpi=400)
    plt.close(fig)
    meta={'title':'Reasoning-role scores across dialogue formats','layout':'Complete dialogue, two rows of three passage blocks','source_sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),'original_messages_sha256':hashlib.sha256((source/'original-messages.json').read_bytes()).hexdigest(),'points_per_panel':len(base),'all_plot_values_exact':True,'no_text_truncation':True,'text_transform':'Markdown emphasis markers removed; whitespace normalized only. Every word preserved.','size_inches':[width,height],'passages':bounds}
    (out/'render-provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps({'output':str(out),'size_inches':[width,height],'font_size':fontsize,'line_counts':[len(x) for x in lines]}))

if __name__=='__main__':main()
