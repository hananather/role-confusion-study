"""I render the companion directly from its review Markdown."""
from pathlib import Path
import re,html,mistune
HERE=Path(__file__).resolve().parent
source=HERE/'cot-forgery-steering.md';text=source.read_text()
css=re.search(r'<style>(.*?)</style>',(HERE/'blog-draft.html').read_text(),re.S)[1]
class Reader(mistune.HTMLRenderer):
 def image(self,text,url,title=None):
  anchor = "figure" if "upload-outcomes" in url else "probe-figure"
  return '<a id="'+anchor+'" class="chart-zoom" href="'+html.escape(url)+'" target="_blank">'+super().image(text,url,title)+'</a>'
body=mistune.create_markdown(renderer=Reader(),plugins=['table'])(text).replace('<table>','<div class="table-scroll"><table>').replace('</table>','</table></div>')
page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CoT forgery and steering — review figure</title><style>'+css+'</style></head><body><a class="skip" href="#article">Skip to article</a><main><nav class="topline"><a href="blog-draft.html">Main report</a><a href="cot-forgery-steering.md" download>Markdown</a><a href="figures/figure8-steering/case002-v1/upload-outcomes.pdf">Outcome PDF</a><a href="canonical/mats-dialogue-v1/figure.png">Canonical MATS example</a></nav><article id="article">'+body+'</article></main></body></html>'
(HERE/'cot-forgery-steering.html').write_text(page)
