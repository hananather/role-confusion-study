# Which experiments have actually run

Updated after the September 12, 05:14 UTC audit. I keep the seven proposed follow-ups separate from the closed replication batches: 100 recorded agent episodes, ten separate authorship readouts and ten assigned episodes left unrun. Three recorded agent episodes are censored.

| Proposed experiment | Execution status |
| --- | --- |
|1 Paragraph patching|Not run|
|2 Doubt-point counterfactual|Not run|
|3 Wrapper manipulation|Not run|
|4 Who-wrote-it readout|Partial: ten CUDA freeform readouts completed; the original forty-page design is incomplete|
|5 Full 40-page, two-seed, two-dose comparison|Not run|
|6 Geometry, full corpus and both sites|Partially complete: all 24 probe-site layers; full block-output site pending|
|7 Decision-level vector|Not run|

Separate work: all 35 historical CUDA bridge episodes and all 40 new-page episodes are complete, audited and synced locally. These 75 episodes cover ten forgery pages plus ten benign controls. Across the ten forgery pages, verified uploads are unsteered 7/10, role alpha16 6/10, reverse 5/10, and random controls 4/10, 2/10 and 4/10. Two random-2 nonuploads are censored. Every episode encountered its intended page. Both engineering gates passed. The original forty-page, two-seed, two-dose study remains unrun.

The authorized queue completed item 4's twenty-trajectory paired review and item 5's ten authorship readouts. The answers identify page/tool as source on 0/10 items; these preliminary assistant labels come from an added question under a forced-final response format. Item 6 saved and independently audited all five new standard-injection episodes: zero observed uploads, four resolved nonuploads and one censored outcome.

GPU B closed with twenty uncensored episodes and ten UNRUN assignments. Item 1's sentence arm produced 4/10 forgery uploads and five benign candidate summaries. Item 2's Tool−mean(User,CoT) direction passed all five probe checks at the original alpha16 magnitude, without doubling, then produced 0/5 historical-page uploads and five candidate summaries. The five new Tool-raising pages and item 3's five historical standard controls remain UNRUN. [GPU B closeout](../../cloud/outbox/parallel-h100/gpu-b-20260912T041801Z/closeout-20260912T051046Z/README.md).

The Tool-raising result is a provisional positive against matched historical unsteered 3/5, old role 2/5 and best random 1/5. Its observed margin over the best random is one upload. It does not establish generalization or test the full ten-page prediction. Summary presence remains a heuristic, not judged quality or utility preservation.

## The 313-prompt StrongREJECT batch

All 313 unsteered baseline trajectories are saved locally: 309 complete final answers and 4 censored at 5,000 tokens. Every row is the base variant with zero activation edits. No chat forgery or steering comparison has run. A usable 313-row forgery corpus is still missing, and semantic harmfulness judging remains unfinished.

The provisional string rule calls 145 completed answers successful, but this is not a valid harmful-response count. Some answers include a harmless cat fact followed by a refusal, which breaks exact canned-refusal matching. I preserve the raw flags and require proper content labels before reporting attack success.

- [313 raw trajectories](</Users/hananather/Desktop/MATS 12.0/replication/cloud/outbox/chat-steering/baseline-retry-20260912T004400Z/results/generations.jsonl>)
- [313 baseline report](</Users/hananather/Desktop/MATS 12.0/replication/cloud/outbox/chat-steering/baseline-retry-20260912T004400Z/RESULTS.md>)
- [Current artifact index](analysis-ready/latest.json)
- [Machine-readable status](experiment-status.json)

## The closed authorized queue and remaining work

Hanan authorized [the one-hour queue](RUN-QUEUE-2026-09-12.md) at 04:18:01 UTC, with a 05:08:01 UTC launch cutoff. GPU A's retention instruction was superseded by explicit user shutdown at 05:11:30 UTC; [05:11:39 verification](persistent-session/state/user-shutdown-verification.json) records HTTP 404 and absence from inventory. GPU B was [last observed at 05:11:58](../../cloud/outbox/parallel-h100/gpu-b-20260912T041801Z/closeout-20260912T051046Z/retained-pods-live.json) idle with its model retained under its separate financial supervisor and a 05:25:19 UTC deadline.

My highest next recommendation, conditional on new execution authorization, is to complete the five fixed new Tool-raising pages before expanding. I can first judge saved summary quality and recognition blind to arm, and calibrate the authorship question with clean controls. [Roadmap and claim boundaries](queue-gpu-a/roadmap-status.md).

The original seven-experiment battery remains a proposed program, not a queued or fully funded run. Its previous 15–20 H100-hour estimate depends on the original single-continuation design and is not a remaining-time estimate for this one-hour queue. Prior text-defense/declaration/destyle recommendations and the layer-patching-first proposal remain preserved in the review notes. Any next experiment must follow the actual queue results and the current authorization.
