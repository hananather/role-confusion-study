"""I render my review Markdown as a readable, accessible local page."""
from pathlib import Path
import html, re
import mistune

HERE=Path(__file__).resolve().parent
source=(HERE/'report.md').read_text()
css=re.search(r'<style>(.*?)</style>',(HERE.parent/'blog-draft.html').read_text(),re.S)[1]
css=css.replace("url('fonts/", "url('../fonts/").replace('url("fonts/', 'url("../fonts/')
css+='\narticle{max-width:1050px;margin:auto}main{max-width:1140px}.review-nav{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0 28px;font-size:17px}.chart-zoom{display:block}.chart-zoom img{display:block;width:100%;height:auto}h2{scroll-margin-top:28px} .review-note{font-size:16px;color:#52525b} @media(max-width:650px){main{padding:22px 18px}h1{font-size:34px}body{font-size:19px}.review-nav{font-size:15px;gap:12px}} @media print{.review-nav,.topline,.skip{display:none}a{color:inherit}h2,h3{break-after:avoid}.chart-zoom{break-inside:avoid}}'
class Reader(mistune.HTMLRenderer):
    def heading(self,text,level,**attrs):
        anchor=re.sub('[^a-z0-9]+','-',re.sub('<[^>]+>','',text).lower()).strip('-')
        return f'<h{level} id="{anchor}">{text}</h{level}>\n'
    def image(self,text,url,title=None):
        number=re.search(r'/0([123])-',url)
        anchor=' id="figure-'+number.group(1)+'"' if number else ''
        return '<a'+anchor+' class="chart-zoom" href="'+html.escape(url)+'" target="_blank" rel="noopener" aria-label="Open full-size figure: '+html.escape(text,quote=True)+'">'+super().image(text,url,title)+'</a>'
body=mistune.create_markdown(renderer=Reader(),plugins=['table'])(source)
page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Role scores and prompt injection — complete evidence review</title><style>'+css+'</style></head><body><a class="skip" href="#article">Skip to report</a><main><nav class="topline" aria-label="Report resources"><a href="../figure-selection/index.html">Refined figure gallery</a><a href="../index.html">Figure library</a><a href="report.md" download>Markdown</a><a href="review-bundle.zip" download>Review bundle</a><a href="REVIEWS.md">Verification</a></nav><nav class="review-nav" aria-label="Report sections"><a href="#tldr">TLDR</a><a href="#1-the-original-vector-does-not-improve-the-new-page-outcomes">All outcomes</a><a href="#2-suppressing-the-reasoning-role-score-is-not-enough">Probe versus behavior</a><a href="#3-a-defense-must-preserve-the-requested-action">Task completion</a><a href="#methods-coverage-and-sources">Methods</a></nav><article id="article">'+body+'</article></main></body></html>'
(HERE/'index.html').write_text(page)
print(HERE/'index.html')
