"""I package the draft, figures and directly cited evidence for later review."""
from pathlib import Path
import re,shutil,hashlib,json,zipfile,os
from datetime import datetime,timezone
from urllib.parse import unquote
HERE=Path(__file__).resolve().parent
DRAFT=HERE/'role-signals-blog-draft.md'
OUT=HERE/('.review-bundle-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
OUT.mkdir()
text=DRAFT.read_text()
version=re.search(r'Review draft (v[0-9.]+)',text).group(1)
refs=re.findall(r'^\[([^\]]+)\]: <(/[^>]+)>$',text,re.M)
images=re.findall(r'!\[[^\]]*\]\(<(/[^>]+)>\)',text)
n_figures=len(images)
map_paths={}
for label,value in refs:
    src=Path(unquote(value))
    if not src.is_file():raise FileNotFoundError(src)
    if src.name in ['mats-example-gallery.html','mats-layout-gallery.html','cot-forgery-steering.html']:dest=Path(src.name)
    elif src.is_relative_to(HERE/'canonical'):dest=src.relative_to(HERE)
    elif any(src.is_relative_to(HERE/'data'/folder) for folder in ['mats-six-passage','mats-hanan-dialogue','mats-hanan-dialogue-v2']):dest=src.relative_to(HERE)
    elif src.is_relative_to(HERE/'figures/mats-hanan-layouts'):dest=src.relative_to(HERE)
    elif src.is_relative_to(HERE/'figures'):dest=Path('figures')/src.name
    elif src.parent==HERE/'data':dest=Path('data')/src.name
    else:dest=Path('evidence')/(label+src.suffix)
    map_paths[src.resolve()]=dest
for value in images:
    src=Path(value);map_paths[src.resolve()]=Path('figures')/src.name
# I preserve the full twelve-example comparison as portable companion material.
for source_dir,patterns in [(HERE/'figures/mats-six-passage',['**/*.png','**/*.pdf','**/*.svg','**/*provenance.json']), (HERE/'data/mats-six-passage',['*.json','*.csv','*.md','mats-*/*.csv','mats-*/conversation.md','mats-*/display-selection.json'])]:
    for pattern in patterns:
        for src in source_dir.glob(pattern):
            if src.is_file() and src.resolve() not in map_paths:
                map_paths[src.resolve()]=src.relative_to(HERE)
for name in ['texgyretermes-regular.otf','texgyretermes-bold.otf','texgyretermes-italic.otf']:
    src=HERE/'fonts'/name;map_paths[src.resolve()]=Path('fonts')/name
for folder in ['mats-hanan-dialogue','mats-hanan-dialogue-v2']:
    for pattern in ['*.md','*.json','*.csv','turn-*-raw.txt']:
        for src in (HERE/'data'/folder).glob(pattern):
            if src.resolve() not in map_paths:map_paths[src.resolve()]=src.relative_to(HERE)
for src in (HERE/'figures/mats-hanan-layouts').rglob('*'):
    if src.is_file() and src.resolve() not in map_paths:map_paths[src.resolve()]=src.relative_to(HERE)
# I preserve the approved example and the separate measured steering companion.
for folder in ['canonical/mats-dialogue-v1','figures/figure8-steering/case002-v1','data/figure8-steering']:
    for src in (HERE/folder).rglob('*'):
        if not src.is_file() or '__pycache__' in src.parts:continue
        if src.suffix.lower() not in ['.md','.json','.csv','.txt','.html','.png','.pdf','.svg','.py']:continue
        if src.name.endswith('.partial.json') or src.name=='progress.json':continue
        if src.resolve() not in map_paths:map_paths[src.resolve()]=src.relative_to(HERE)
for name in ['cot-forgery-steering.md','cot-forgery-steering.html','CANONICAL-EXAMPLES.md']:
    src=HERE/name;map_paths[src.resolve()]=Path(name)
manifest=[]
for src,dest in map_paths.items():
    target=OUT/dest;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(src,target)
    if src.name=='mats-example-gallery.html':target.write_text(target.read_text().replace('href="blog-draft.html"','href="role-signals-blog-draft.md"'))
    if src.name=='mats-layout-gallery.html':target.write_text(target.read_text().replace('href="blog-draft.html#1-hanan-asks-for-a-mats-project-idea"','href="role-signals-blog-draft.md"'))
    if src.name=='cot-forgery-steering.html':target.write_text(target.read_text().replace('href="blog-draft.html"','href="role-signals-blog-draft.md"'))
    if src.name=='cot-forgery-steering.md':target.write_text(target.read_text().replace('(blog-draft.html)','(role-signals-blog-draft.md)'))
    manifest.append({'file':str(dest),'original_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size})
for src,dest in map_paths.items():text=text.replace(str(src),dest.as_posix())
(OUT/DRAFT.name).write_text(text)
(OUT/'README.md').write_text(f'''# Review draft {version}

Start with [the Markdown article](role-signals-blog-draft.md). Keep its figures, data and evidence folders beside it so images and citations work.

This local review bundle contains all {n_figures} charts in PNG and PDF, the directly cited result tables, provenance records, and review notes. Figure 1 uses the [new Hanan–MATS exchange](data/mats-hanan-dialogue-v2/conversation.md), with complete generated text and both attempts retained. It also includes the earlier [twelve-example comparison gallery](mats-example-gallery.html), with all candidate figures and full conversation text. The draft remains subject to Hanan's review. It has not been published or sent to anyone by preparing this bundle.

The approved [canonical MATS example](canonical/mats-dialogue-v1/README.md) is frozen for reuse. A separate [CoT-forgery steering companion](cot-forgery-steering.html) adds a four-condition behavioral figure, its full attack text, local trajectories and H100 control records. Its binary activation and probability arrays remain in the workspace; the plotted and full relevant token-score tables are included here.

The main article's local links are portable and checked. Some copied audits retain deeper links to the original workspace; the complete model weights, raw activation dumps, and private conversation logs are outside this review bundle. The included checksums identify the cited source records. This is a reading and evidence-review package, not a complete experiment reproduction environment.
''')
(OUT/'manifest.json').write_text(json.dumps({'draft_sha256':hashlib.sha256(DRAFT.read_bytes()).hexdigest(),'files':manifest,'scope':f'All {n_figures} figures and directly cited evidence; full model and raw context are not included.'},indent=2)+'\n')
for link in re.findall(r'<([^>]+)>',text):
    if link.startswith(('figures/','data/','evidence/')):assert (OUT/link).exists(),link
previous=HERE/'review-bundle'
if previous.exists():
    backup=HERE/'archive'/('bundle-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'));previous.rename(backup)
OUT.rename(previous);OUT=previous
archive=HERE/'role-signals-review-bundle.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.rglob('*')):
        if f.is_file():z.write(f,Path('role-signals-review-'+version)/f.relative_to(OUT))
print({'bundle':str(archive),'files':sum(p.is_file() for p in OUT.rglob('*')),'bytes':archive.stat().st_size})
