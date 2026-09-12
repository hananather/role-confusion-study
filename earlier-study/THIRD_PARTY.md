# Attribution and licenses

I built this experiment on [Prompt Injection as Role Confusion](https://github.com/role-confusion/prompt-injection-as-role-confusion) by Charles Ye, Jasmine Cui and Dylan Hadfield-Menell. My frozen upstream reference is commit `ec333c40fd43fe991e1ebf66765051b6d7e35784`. I preserve the upstream MIT permission text verbatim in `LICENSE.md`; my experiment and analysis code are distributed under the same terms. That software license does not replace licenses or rights in third-party text embedded in the data.

## Wikipedia text

Cases, prompts and tool-output transcripts reproduce text by Wikipedia contributors, under [Creative Commons Attribution-ShareAlike 4.0](https://creativecommons.org/licenses/by-sa/4.0/), subject to the [Wikipedia reuse policy](https://en.wikipedia.org/wiki/Wikipedia:Copyrights). I cleaned and normalized source HTML, inserted synthetic task instructions, and reproduced those adapted pages in experiment transcripts. The adapted Wikipedia content remains under CC BY-SA 4.0. Each revision below links to its page and contributor history; `data/page-attribution.json` records source and cleaned-content hashes.

| Page ID | Article and frozen revision |
| --- | --- |
| page-00 | [Marar (caste)](https://en.wikipedia.org/w/index.php?oldid=1345081503) |
| page-01 | [Horst Bollmann](https://en.wikipedia.org/w/index.php?oldid=1295357219) |
| page-02 | [Karim Mahdi Salih](https://en.wikipedia.org/w/index.php?oldid=1369042979) |
| page-03 | [K. Anantharamu](https://en.wikipedia.org/w/index.php?oldid=1371898591) |
| page-04 | [Dayton District](https://en.wikipedia.org/w/index.php?oldid=1347641914) |
| page-05 | [Major Henderson incident](https://en.wikipedia.org/w/index.php?oldid=1314780364) |
| page-06 | [Puke (EP)](https://en.wikipedia.org/w/index.php?oldid=1263939391) |
| page-07 | [Nostradamus ni Kiite Miro](https://en.wikipedia.org/w/index.php?oldid=1290241746) |
| page-08 | [John Wiedeman](https://en.wikipedia.org/w/index.php?oldid=1366077902) |
| page-09 | [At the National Grid](https://en.wikipedia.org/w/index.php?oldid=1363461625) |

## Model and preparation sources

I used `openai/gpt-oss-20b`; the experiment metadata records the exact revision. This repository contains my saved local-classifier and steering arrays, not the language-model weights. Attack templates derive from the upstream MIT-licensed repository.

Preparation used the C4 and Dolma3 dataset revisions recorded in the scientific configurations. I omit the neutral corpus text and its standalone role-context renderings from this release. Those dataset references and source hashes describe my method; they do not grant a new license over scraped component text.
