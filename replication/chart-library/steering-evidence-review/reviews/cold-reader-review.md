# Cold-reader review

**Resolved verdict: PASS. No material clarity recommendations remain.**

On September 12, 2026, I reread the current report and checked the affected linked audit passages. The opening now identifies my GPT-OSS-20B agent experiments and their relationship to *Prompt Injection as Role Confusion* before the TLDR. The paragraph before Figure 1 defines zero dose and explains its purpose alongside the separate no-intervention arm. These additions resolve both requested changes without restructuring the report.

The methods now map the figure labels Random 1, 2 and 3 to source arms `random_0_a16`, `random_1_a16` and `random_2_a16`. The three linked audit Markdown files use explicit source-arm labels. In the passages I flagged, the censored arm is now source `random_2` and the eight-completed-nonupload arm is source `random_1`, consistent with that mapping. The incidental naming concern is resolved at the reader-facing documentation level.

The initial findings below remain as review history. My follow-up checked the revised prose and labels; it does not add a source-data validation or a new rendered-figure audit.

I first read `report.md` and all three PNG figures without consulting project context. I then checked only the linked numerical and statistical audits to resolve the baseline terminology. This is a review of clarity and reader interpretation, not an independent validation of every source.

The title states the central observation accurately. The report explains the attack, intervention, probe, observed behavior and practical consequence in a coherent sequence. Figures 1 and 2 make the central result legible: the original vector has near-zero reasoning-role scores alongside six uploads, and its five new-page outcomes match the baseline. The later favorable vector, random controls, unresolved outcomes and unrun tests remain visible. Figure 3 supplies a useful second task while explicitly preserving its different intervention and endpoint. The proposed next step follows from the unresolved favorable branch.

## Initial material recommendations — resolved

1. **Establish the local test and its provenance before the TLDR.** The opening currently introduces “the original steering direction” before identifying the model, attack task, intervention or meaning of “original.” A reader arriving through the paper could mistake that term for a reproduction of the paper's original method. Replace the generic italic subtitle with a short setup, for example: “I tested whether changing GPT-OSS-20B’s internal role scores stops webpage instructions from causing an unauthorized dummy-file upload. These are my local agent tests, motivated by *Prompt Injection as Role Confusion*.” Keep the existing detailed setup in Section 1. This supplies the experimental question and evidence boundary without changing the claim.

2. **Define “zero dose” before Figure 1.** Both “No intervention” and “Zero dose” appear as separate baselines, but their difference is never stated in plain language. A short sentence before Figure 1 is enough: “The zero-dose runs record role scores with no vector added; the no-intervention runs do not record those scores.” This also prepares readers for the missing baseline in Figure 2. The linked audit supports that distinction and already preserves the narrower matching boundary; no claim of identical complete trajectories is needed.

## Initial stopping decision

I would approve the report's narrative after those additions. The remaining specialist terms occur in the methods or have enough nearby explanation to follow the main result. The body is appropriately bounded; shortening it further could remove useful distinctions between no upload, a completed legitimate task, and an established defense. I do not recommend new experiments, broader claims, removal of the favorable branch, or restructuring the report.

## Initial incidental linked-document consistency — resolved

The linked audits contain apparent stale random-arm names: `claims-statistics-review.md` calls the arm with four verified uploads and two unresolved episodes “Random 2,” while Figure 1 calls it “Random 3.” `full-batch-audit.md` calls the arm with eight completed nonuploads “Random 1,” while Figures 1 and 2 identify it as “Random 2.” I have not audited the source mapping. The numerical reviewer should reconcile these names so following the report's links does not produce a contradictory account.
