# My early fit-quality audit

Captured 2026-09-11T03:48:23.078876+00:00. I leave the active recipe and process unchanged.

I found no fresh mechanical fit defect requiring interruption. All 16 prompt-split accuracies at layers 0 and 2 match the archived independent H200 replication within **0.081937 percentage point**. The local log rounds accuracy to three decimals. This compares my run with an independent reproduction, not an original paper-trained artifact.

I recomputed reference accuracy from its confusion counts and checked denominators against my saved split audit. All 16 held-out token denominators agree: **True**. The estimator function body matches the authors' notebook under an AST comparison: **True**.

I retain the convergence concern. cuML 25.08 code 1 means the attempted step fell below the minimum allowed size; the solver reports line-search failure and retains a valid previous iterate when required. Matching early accuracy does not establish coefficient equality, calibrated role probabilities, convergence, or success on layer 12. My next cheap check is to inspect the existing native pickle's iteration count, saved objective, finite coefficients, class order and parameter fields after it is written, without refitting or another model forward pass.

- [Matched 16 rows](matched-early-layer-accuracies.csv)
- [Captured fit and warning evidence](captured-matched-fit-and-warning-lines.txt)
- [Reference confusion counts](reference-confusion-rows-L0-L2.csv)
- [Full provenance, formulas and qualifications](audit.json)
