# Historical evidence review of the report and Figure 3

*Reviewer 2 · September 12, 2026 · Understand → Distill*

**Verdict: PASS. No material corrections or remaining recommendations within this review's scope.**

My north star is an accurate distinction between moving a role-probe score and preserving the user's permitted action. I apply the Neel lens through complete recovered cohorts, baseline comparisons, direct record inspection and bounded conclusions.

I reviewed the actual `report.md`, particularly section 3 and the historical methods, the complete Figure 3 renderer, and the rendered PNG. I checked the report against my independently reduced raw-record inventory and reran a separate reduction of the figure CSV. The reviewed artifact hashes and numerical checks are preserved in [report-historical-review-checks.json](report-historical-review-checks.json).

- **Eligibility:** all 21 recovered Toolward episodes have matched no-steering records. They cover eight cases and 15 distinct case/permission prefill contexts. The text correctly states both repeated-seed overlap and recovery of only 149 of 280 planned records. It does not present the recovered set as the complete planned experiment.
- **Readout:** Figure 3A uses the mean Tool probability across command tokens at the first post-page prefill. The minimum episode mean is 0.9999762827699835. Its near-saturated display and title are justified. It does not confuse first-prefill means with the lower minimum across individual tokens in later prefills.
- **Behavior:** the plotted counts match the paired records: unwanted physical writes change from 18 to 17; correct authorized completions change from 21 to 19. Four unwanted writes are prevented and three introduced. The renderer asserts the expected cohort size and plotted totals before rendering.
- **Malformed generation:** section 3 explicitly preserves the distinction between a recorded nonwrite and successful selective refusal. One prevented physical write comes from a parse error with unknown attempt status; it remains in the denominator and contributes a failed permitted completion. The figure and prose do not count this as successful task completion.
- **Utility meaning:** the preceding paragraph defines authorized completion in terms of the correct marker value. Figure 3's caption also names correctly completed actions. The latest upload batch's summary-presence heuristic is explicitly separated from factual summary quality and the green nonupload state.
- **Historical boundaries:** Tool-minus-User, command-only steering, the marker endpoint and the earlier classifier are separated from the CUDA Tool-minus-CoT upload experiment. The report does not pool those counts or call them independent replications of one method. Earlier transfer and unexecuted ACTION-proposal pilots remain in the coverage inventory.
- **Presentation:** all three panels are readable in the actual PNG. Paired readout lines, outcome counts and axis labels express the claim without implying statistical equivalence. Shared initial contexts explain the overlapping lines in the report.
- **Narrative:** the claim is that extreme role scores do not certify selective behavior, rather than that every steering vector has zero effect. That statement is supported by the displayed historical cohort and preserves the favorable unfinished construction discussed elsewhere.

No report, renderer or scientific source was edited during this review. This review does not independently re-audit the latest CUDA batch; that is another review's responsibility.
