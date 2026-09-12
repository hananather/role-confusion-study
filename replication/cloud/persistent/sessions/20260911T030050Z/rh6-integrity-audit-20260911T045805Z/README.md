# My RH6 integrity audit

I checked the completed saved run locally without GPU use or source-file changes. Capture: 2026-09-11T04:58:05.055976+00:00.

Overall integrity checks passed: **True**. I found 1,200 unique inputs, 100 templates, all 12 conditions with 100 inputs each, layers 8/12/16 and the actual recorded probe spaces **sucat and uat**.

I independently retokenized all 1,200 exact saved inputs with the cached tokenizer, verified every saved ID and character offset, reconstructed every span's overlap positions and matched all 1,631,880 scored token rows to their original text. I retained the 750 inherited tool-result end offsets beyond text; no command, injection or user-turn span has that overflow.

I checked present-role probability finiteness and normalization, absent-role NaNs, all 7,200 page-context records, and recomputed all 28,800 per-item/span summaries. I independently recomputed all 1,080 aggregate rows, including paired contrasts, denominator counts, positive fractions and the seeded 2,000-draw bootstrap intervals. I also checked all 12 overlap correlations.

This is an integrity audit. It does not establish a causal mechanism, command execution, or optimizer convergence. The implemented log-ratio uses the log of span-mean probabilities; the lexical-overlap statistic is an unadjusted correlation.

- [Checks](checks.csv)
- [Detailed errors, numerical differences and source hashes](audit.json)
