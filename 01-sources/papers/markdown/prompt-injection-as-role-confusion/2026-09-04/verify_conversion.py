from pathlib import Path
import json,subprocess,xml.etree.ElementTree as ET,hashlib,re,datetime
R=Path(__file__).resolve().parent;src=Path('/Users/hananather/Desktop/PromptInjectionRoleConfusion.pdf');events=json.loads((R/'conversion-regions.json').read_text());md=(R/'prompt-injection-as-role-confusion.md').read_text();ns='{http://www.w3.org/1999/xhtml}'
rows=[]
for p in range(1,34):
 root=ET.fromstring(subprocess.check_output(['pdftotext','-f',str(p),'-l',str(p),'-bbox-layout',str(src),'-']))
 regions=[e['bbox_points'] for e in events if e['page']==p];eligible=0;miss=[]
 for w in root.iter(ns+'word'):
  x=(float(w.attrib['xMin'])+float(w.attrib['xMax']))/2;y=(float(w.attrib['yMin'])+float(w.attrib['yMax']))/2
  if 53<=x<=558 and (85 if p==1 else 64)<=y<726:
   eligible+=1
   if not any(x0<=x<x1 and y0<=y<y1 for x0,y0,x1,y1 in regions):miss.append({'text':w.text,'x':round(x,2),'y':round(y,2)})
 rows.append({'page':p,'source_word_boxes_in_content_area':eligible,'word_boxes_outside_conversion_regions':miss,'page_anchor_present':f'id="page-{p:02}"' in md,'page_render_present':(R/f'assets/pages/page-{p:02}.png').exists()})
metadata={'source_pdf':str(src),'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'source_pages':33,'markdown_sha256':hashlib.sha256((R/'prompt-injection-as-role-confusion.md').read_bytes()).hexdigest(),'markdown_characters':len(md),'page_anchors':len(re.findall(r'<a id="page-\d+"',md)),'figure_images':len(list((R/'assets').glob('figure-*.png'))),'table_ids':['1','2','3a','3b','4'],'source_footnote_numbers':list(range(1,25)),'verified_date':'2026-09-04','page_coverage':rows,'coverage_boundary':'Checks source word centers within x=53..558 points, y=64..726 (page 1 y=85..726). Running headers, page numbers, and rotated arXiv sidebar are represented by navigation/metadata rather than repeated. This is geometric extraction coverage, not proof of perfect character transcription.'}
(R/'extraction-metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False)+'\n')
print('Uncovered source word boxes:',[(r['page'],r['word_boxes_outside_conversion_regions']) for r in rows if r['word_boxes_outside_conversion_regions']])
print('All source footnotes:',all('Footnote '+str(n)+'.' in md for n in range(1,25)))
print('Page anchors',metadata['page_anchors'],'figures',metadata['figure_images'])
