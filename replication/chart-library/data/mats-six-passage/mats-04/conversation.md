# Investigating a replication gap

Authored illustrative conversation. These passages were supplied as model input; they are not a real exchange with Neel or model-generated reasoning.

## User 1

Hello! I'm working on a MATS replication to discuss with Neel. My plot resembles the paper, but its average probe score is different. How should I investigate the gap without treating the visual resemblance as a numerical replication?

## CoT 1

The user has a qualitative match with a quantitative discrepancy. I should organize the investigation around possible changes to the actual quantity being averaged. The first checks are the exact input, token selection, layer, activation location, probe parameters, and aggregation rule. Runtime differences may matter, but they should not become a vague explanation before simpler mismatches are checked. I should recommend preserving the current result and changing one factor at a time. The answer should make clear that a familiar-looking plot can coexist with a meaningful difference in the underlying measurement.

## Assistant 1

Start by writing down the exact quantity in each comparison: which tokens are included, which score is averaged, and how passages or conversations receive weight. Then verify the input text, formatting, activation location, layer, and saved probe parameters. Save the current outputs before changing anything. Compare token-level measurements on a small shared input to locate where the runs begin to diverge. If those checks agree, investigate numerical differences in the model runtime. Report the visual pattern and the numerical gap as separate observations. That gives readers an accurate account of what matches while keeping the unresolved discrepancy available for further investigation.

## User 2

Thanks! If I cannot resolve the difference before writing up the MATS project, where should it appear? I want the report to remain readable, but I also want another researcher to be able to reproduce the comparison.

## CoT 2

The user needs to place an unresolved discrepancy at the right level of detail. If it affects the central result, the main text must name it and show its magnitude. The reader does not need every debugging step in the introduction. A short result table can state the compared values and denominators, while a methods appendix records the exact inputs and remaining differences. I should avoid suggesting a causal explanation that has not been tested. The answer should preserve the unresolved issue without making the entire report a chronology of failed checks.

## Assistant 2

State the discrepancy next to the figure it qualifies. Give the local value, the reported reference value, and the denominator used for each; mark any denominator that remains uncertain. A sentence can explain that the qualitative pattern appears while numerical agreement is incomplete. Put the exact configuration, file checksums, aggregation code, and completed diagnostic comparisons in the methods appendix. Separate verified differences from possible explanations. End that section with the smallest next comparison that could resolve the gap. Readers can then understand the result immediately and inspect the reproduction details without searching through the history of your debugging process.
