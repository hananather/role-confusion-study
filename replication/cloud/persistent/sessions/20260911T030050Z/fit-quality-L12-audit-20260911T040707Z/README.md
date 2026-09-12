# My layer-12 fit-quality audit

Captured 2026-09-11T04:07:07.787786+00:00. I preserve the earlier audit and leave the running recipe unchanged.

All **eight prompt-split layer-12 accuracies agree with the independent H200 reference at the three-decimal precision of my log**. The largest absolute difference from the unrounded reference accuracy is **0.045497 percentage point**. All eight held-out token denominators match.

For the gardening probe, **suca L12**, my log reports accuracy **0.703** and negative log likelihood **0.727**. The reference accuracy is **0.703366595**, computed from **37,711 / 53,615** correct tokens. My held-out denominator is also **53,615**. The accuracy difference is **-0.036660 percentage point**.

This agreement supports continuing the unchanged run. It does not establish optimizer convergence, coefficient equality, calibrated role probabilities, or reproduction of the gardening figures. I have no reference negative log likelihood in this confusion-count source. The historical reference is an independent H200 reproduction, not an original paper-trained artifact.

- [Eight matched metrics](matched-L12-accuracies.csv)
- [Captured fit and warning evidence](captured-L12-fit-and-warning-lines.txt)
- [Reference confusion counts](reference-confusion-rows-L12.csv)
- [Absolute paths, timestamps, hashes and formulas](audit.json)
