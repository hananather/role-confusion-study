# My completed H100 session

I completed all nine queued stages on one persistent H100, backed up the final results, and terminated the GPU at 2026-09-11 04:54:20 UTC (00:54 Toronto). RunPod independently confirmed zero pods at 04:54:55 UTC. My 2,000 GB network volume remains in EU-FR-1 and continues to incur storage charges. The 20-second collector exited, and the attached monitoring schedule is paused.

| Work | Completed | What I learned |
| --- | --- | --- |
| Probe training | 384 probes; 24 layers; 8 role spaces; prompt and base splits | All saved partition, coefficient and parameter checks passed. Across 96 matched H200 probes, overall accuracy differs by at most 0.0791 percentage points; the largest role-specific recall difference is 0.615 points. Solver convergence remains unestablished. |
| Gardening / Appendix E | 3 conditions; Figures 7 and 20–22 | Role-style signals persist under misleading tags, but the paper's numerical probabilities did not reproduce. CoT means are 64.03%, 71.04% and 70.97% across correct, absent and all-user tags; the tested display subsets do not explain the discrepancy. |
| Appendix K neutral control | 1,330 items | The position contrast depends on layer. All 200 neutral texts overlap the probe corpus, so this is not a held-out replication of the paper's conversation figure. |
| RH6 | 1,200 prefills; 100 template–page pairs | Marker permission increases the relative User/Tool score beyond the paired null in all 12 tested configurations. Absolute Userness depends on the probe; prohibition produces little relative-score change in most configurations. No behavior was tested. |

The independent RH6 audit passed all 44 checks, including 1,631,880 scored-token mappings, 28,800 per-item/span rows and all 1,080 aggregate rows and bootstrap intervals. Its 750 inherited span overflows affect only the broad tool-result reference, not the command, injection or user-turn spans used for these conclusions. Statistical intervals are pointwise; the reports preserve null-matching, lexical-overlap and position limitations.

I repaired a forward-memory failure, a model-format guard and a summary-rounding check while preserving failed evidence. The gardening summary repair reused saved activations and probabilities without another model pass. The tiny pilot's original grouped split overlapped training and test data; I exclude those pilot accuracies and verified all 16 full-data partitions independently. Completing the queue does not erase these qualifications or establish reproduction of every paper result.

## My evidence

- [Operational session, repairs and shutdown](SESSION.md)
- [Live provider shutdown verification](shutdown-provider-verification.json)
- [Final local artifact checksums](final-local-artifact-manifest.json)
- [Full probe completion audit](full-probes-completion-audit-20260911T043828Z/README.md)
- [Native estimator audit](outputs/20260911T030050Z/probes-full/native-probe-audit.json)
- [Gardening results and figures](GARDENING-RESULTS.md)
- [Appendix K completion audit](appendix-k-completion-audit-20260911T044857Z/README.md)
- [RH6 results and limitations](RH6-RESULTS.md)
- [RH6 integrity audit](rh6-integrity-audit-20260911T045805Z/README.md)

All completed experiment outputs are in `outputs/20260911T030050Z/`. The local checksum manifest covers 116 top-level artifacts totaling 307,041,049 bytes; checkpoint parts are also synced but are outside that manifest. Model files, environment and activation caches remain on the retained network volume. This closes the approved queue; broader experiments not in that queue remain outside this completion claim.
