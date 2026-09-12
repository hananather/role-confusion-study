"""I render the review draft from its canonical Markdown source."""
from pathlib import Path
import os,re,html
from urllib.parse import quote,unquote
import mistune
HERE=Path(__file__).resolve().parent
DRAFT=HERE/'role-signals-blog-draft.md'

class ReaderRenderer(mistune.HTMLRenderer):
    def __init__(self):
        super().__init__(escape=False)
        self.headings=[]
    def url_for_reader(self,url):
        if url.startswith('/Users/'):
            return quote(os.path.relpath(unquote(url),HERE),safe='/#')
        return url
    def image(self,text,url,title=None):
        address=self.url_for_reader(url)
        return f'<a class="chart-zoom" href="{html.escape(address)}" target="_blank">'+super().image(text,address,title)+'</a>'
    def link(self,text,url,title=None):
        return super().link(text,self.url_for_reader(url),title)
    def heading(self,text,level,**attrs):
        plain=re.sub('<[^>]+>','',text)
        slug=re.sub('[^a-z0-9]+','-',plain.lower()).strip('-')
        if level==2:self.headings.append((slug,plain))
        return f'<h{level} id="{slug}">{text}</h{level}>\n'

renderer=ReaderRenderer()
md=mistune.create_markdown(renderer=renderer,plugins=['table','strikethrough'])
draft_text=DRAFT.read_text()
title=re.search(r'^# (.+)$',draft_text,re.M).group(1)
version=re.search(r'Review draft (v[0-9.]+)',draft_text).group(1)
body=md(draft_text)
body=body.replace('<table>','<div class="table-scroll"><table>').replace('</table>','</table></div>')
nav=''.join(f'<li><a href="#{s}">{html.escape(t)}</a></li>' for s,t in renderer.headings)
css='''@font-face{font-family:Termes;src:url('fonts/texgyretermes-regular.otf')}@font-face{font-family:Termes;src:url('fonts/texgyretermes-bold.otf');font-weight:700}@font-face{font-family:Termes;src:url('fonts/texgyretermes-italic.otf');font-style:italic}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:28px}body{margin:0;background:#fff;color:#252525;font-family:Termes,'Times New Roman',serif;font-size:20px;line-height:1.55}main{max-width:1120px;padding:30px 32px 80px;margin:auto}p,ul,ol,blockquote{max-width:850px}h1{max-width:950px;font-size:43px;line-height:1.12;margin:28px 0 18px}h2{font-size:29px;line-height:1.22;max-width:910px;margin:42px 0 16px;padding-top:20px;border-top:1px solid #d9d9d9}p{margin:13px 0}li{margin:9px 0}a{color:#384f66;text-underline-offset:3px;text-decoration-thickness:1px}a:hover{color:#121d28}img{display:block;width:100%;height:auto}.chart-zoom{display:block;margin:25px 0 10px}p:has(.chart-zoom){max-width:none}p:has(.chart-zoom)+p{font-size:17px;line-height:1.45;max-width:1040px}.topline{font-size:16px;color:#62748e;display:flex;gap:20px;flex-wrap:wrap;padding:0 0 17px;border-bottom:1px solid #333}details{font-size:17px;max-width:850px;margin:15px 0}summary{cursor:pointer;color:#526579}details ul{columns:2;padding-left:23px}details li{margin:4px 0;break-inside:avoid}.table-scroll{overflow-x:auto;max-width:100%;margin:20px 0}table{border-collapse:collapse;font-size:18px;line-height:1.4;min-width:620px;width:100%;max-width:920px}th,td{text-align:left;padding:10px 13px;border-bottom:1px solid #ddd;vertical-align:top}th{border-top:1px solid #333;border-bottom:1px solid #333}blockquote{border-left:2px solid #9ca9b8;padding-left:20px;margin:20px 0;color:#394452}code{font-family:ui-monospace,monospace;font-size:.77em;background:#f4f5f6;padding:2px 4px;overflow-wrap:anywhere}pre{overflow-x:auto}.skip{position:absolute;left:-10000px}.skip:focus{left:10px;top:10px;background:#fff;padding:10px}footer{border-top:1px solid #bbb;margin-top:35px;padding-top:18px;font-size:16px;color:#62748e}@media(max-width:650px){body{font-size:18px}main{padding:22px 17px 50px}h1{font-size:34px}h2{font-size:25px;margin-top:32px}details ul{columns:1}table{font-size:16px}.topline{gap:13px}p:has(.chart-zoom)+p{font-size:16px}}@media print{.topline,details,footer{display:none}main{max-width:none;padding:0}body{font-size:11pt}h1{font-size:24pt}h2{font-size:17pt}img{break-inside:avoid}a{color:inherit}h2{break-after:avoid}}'''
page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Review draft — {html.escape(title)}</title><style>{css}</style></head><body><a class="skip" href="#article">Skip to article</a><main><div class="topline"><span>Review draft · {version}</span><a href="role-signals-blog-draft.md" download>Markdown source</a><a href="index.html">Visual gallery</a><a href="role-signals-review-bundle.zip" download>Review bundle</a></div><details><summary>Reading route</summary><ul>{nav}</ul></details><article id="article">{body}</article><footer>This preview is generated from the Markdown draft. Figure 1 contains a new local MATS measurement; the gardening replication and previous result figures remain available below. Click any chart for its full-resolution image.</footer></main></body></html>'''
(HERE/'blog-draft.html').write_text(page)
print({'html':str(HERE/'blog-draft.html'),'headings':len(renderer.headings),'images':body.count('<img ')})
