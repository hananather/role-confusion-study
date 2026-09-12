"""I publish the compact complete-excerpt layout as the current MATS figure."""
import json,shutil,hashlib
from render_mats_six_passage import HERE,setup_style,read_candidate
from render_mats_layout_candidates import compact_diagonal

def main():
    setup_style()
    rows,base,_=read_candidate('mats-hanan-dialogue-v2',data_root=HERE/'data')
    messages=json.loads((HERE/'data/mats-hanan-dialogue-v2/original-messages.json').read_text())
    compact_diagonal(rows,base,messages)
    source=HERE/'figures/mats-hanan-layouts/diagonal-35-compact'
    out=HERE/'figures/mats-hanan-dialogue-v2';out.mkdir(exist_ok=True)
    for suffix in ['png','pdf','svg']:
        shutil.copy2(source/f'diagonal-35-compact.{suffix}',out/f'mats-hanan-dialogue-v2.{suffix}')
    record=json.loads((source/'layout-provenance.json').read_text())
    record['selected_layout']='diagonal-35-compact'
    record['selection_reason']='I use complete opening excerpts at a fixed 7.2pt, rotated 35 degrees. The full six-passage text is retained in the unabridged companion.'
    record['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob('mats-hanan-dialogue-v2.*')}
    (out/'render-provenance.json').write_text(json.dumps(record,indent=2)+'\n')

if __name__=='__main__':main()
