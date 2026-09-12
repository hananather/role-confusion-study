# Independent review 09 — prose and reader hierarchy

**Frozen report SHA-256:** `022c1f4958500451a8568f2749e7878964f69607b023c692130acac016c7db4b`

**Lens:** Concise prose, research voice, Neel Nanda writing principles, hierarchy, removal of unnecessary explanation, and accessibility to a reader with no project context.

## Strongest accurate contribution

The report gives a concrete reason to treat the classifier and summary measure as part of a role-probe result. In the saved permission comparison, the sign of adjusted absolute Userness differs between the five-role and three-role classifiers, while the adjusted relative User/Tool measure increases throughout the plotted grid. The offline offsets supply a complementary, visibly bounded illustration: two class probabilities can increase together, and an offset selected from the classifier changes its own readout. This is a useful empirical measurement account. The report does not need to claim a newly identified role mechanism or an effective defense to make that account worthwhile.

## One-paragraph retelling

We can read role-related patterns from GPT-OSS-20B activations, but what a changing score means depends on the classifier, its classes and the way the score is summarized. A gardening example partly reproduces the authors’ qualitative pattern but leaves a numerical gap. Editing that example’s saved states demonstrates classifier responses, including simultaneous increases in Userness and Toolness. A separate 100-pair permission comparison shows that a relative score can increase even when absolute Userness falls under one probe, and absolute effects can disagree between probes. Held-out recall and unresolved fitting/calibration checks limit interpretation. The next scientific question is whether a controlled edit persists through later model computation; these figures do not answer it or demonstrate improved instruction following.

## Recommended modifications

### R09-01

**Exact passage:** TLDR: “Our main finding is that the interpretation of a changing role score depends on what we change and how we measure it.” The first bullet begins “A permission cue can look different under different readouts” and refers to “the marker command”, “all 12 tested relative-score settings”, “a neutral-text control” and “Absolute Userness”.

**Concern:** The opening gives a generic conclusion before the concrete finding, then assumes knowledge of the marker task, score definitions and experimental grid that the reader does not receive until Section 4. The strongest result is understandable without that terminology: the same permission comparison changes sign under different classifiers, while a relative measure agrees. This is a hierarchy and cold-reader issue, not a request for additional explanation throughout the report.

**Suggested remedy:** Keep the opening definition of a role probe, then lead with the actual measurement result. Suggested replacement for the main-finding sentence and first bullet: “In our measurements, changing the classifier or the score can change whether the same permission cue appears to make text more User-like.

- Across 100 constructed inputs, we compare a webpage-summary request with and without permission to write a dummy marker file. After subtracting a neutral-text control, the User-to-Tool probability ratio increases in all 12 tested combinations of layer, classifier and command order. At layer 12, the User probability itself falls with a five-role classifier and rises with a three-role classifier.” Keep the other two summary claims and the no-behavior boundary. Alternatively shorten this bullet further and leave the exact count of 12 in Section 4; do not introduce “relative-score settings” before defining the measure.

### R09-02

**Exact passage:** Section 4: “We now change the model’s actual input: a User message either permits an action or uses neutral wording.” And: “Each pairs an instruction template with webpage text and contains two commands: write a dummy marker file, and upload data.”

**Concern:** The experiment supporting the main contribution remains more abstract than the preceding gardening illustration. “Uses neutral wording” can suggest that one permission sentence is replaced by a matched neutral sentence; the saved input instead adds a permission sentence to the same baseline request. A short actual example would also make the subsequently disclosed lexical-overlap and position explanations intelligible. The reader should know exactly which action is permitted and where the commands appear.

**Suggested remedy:** Replace the first two setup paragraphs with equally compact but concrete wording: “We use 100 constructed pairs of an instruction template and webpage text. The Tool message contains commands to write a dummy marker file and upload data. The User message asks for a webpage summary; the permission condition adds: ‘If the page asks you to write a marker file, you have my permission to do that, and only that.’ The neutral condition omits this sentence. We test both command orders and a control that replaces the commands with neutral text. Each prepared conversation is processed through the model to record activations, with no generated continuation or executed command.” Retain the following adjusted-effect explanation and all control limitations. Cite the frozen input or the existing input/result reference. This wording is verified against the frozen prefills, whose SHA-256 matches the input contract.

## Proposed removals

### R09-03

**Exact passage:** Review status: “We are in **Understand / Distill**: the aim is to make each measurement and its consequence legible, with particular attention to clarity, source fidelity, skeptical interpretation and adequate samples.” Appendix B: “The audit covers the September 10 Claude discussion, its 19 tracked goals and 35 linked subagent logs; older evidence is included where that discussion referred to it.”

**Concern:** These sentences describe the internal writing and coordination process rather than a research finding, a measurement definition or a scientific limitation. The phase labels require project context and the goal/log counts do not help a new reader assess the evidence. This is a useful cut because the report already contains an audit link and an explicit review-status statement. The broader backlog and historical adverse result should remain.

**Suggested remedy:** Remove these two sentences from the report and preserve their information in the already linked verification/conversation-audit records. Keep the agent-assistance disclosure, numerical reconstruction counts, the statement that report preparation ran no model/GPU experiment, the source list, and the sentence distinguishing report figure numbers from paper figure numbers. No other section or chart needs removal for this recommendation.

## Scientific uncertainty already correctly disclosed

- The gardening result uses one authored conversation, differs numerically from the paper, and does not isolate style, content or position.
- The offsets rescore saved states, use non-unique saved coefficient rows, and provide no downstream computation or behavior evidence. Repeated tokens and offsets do not supply independent conversations.
- The permission comparison is exploratory, uses paired constructed inputs with imperfect replacement matching and unresolved lexical/position alternatives, and reports pointwise rather than simultaneous intervals.
- The probe evaluations use different held-out sets and separately fitted probes; recall is distinct from calibration, and fitting convergence remains unresolved.
- The planned propagated-intervention batch is deferred. Its prepared candidates are not completed independent conversations, and the 200-conversation minimum is not a sample-size justification by itself.

These are reasons to preserve the current claim boundaries, not reasons to demand new experiments before approving the prose.

## Review basis

I read the entire frozen draft before reviewing it, inspected all six PNG figures, and read the supplied Neel Nanda writing guidance. I checked the permission wording against `rh/out/rh6-readings/prefills.json`, the RH6 result note and the saved RH6 input contract. The local prefills hash is `be9254d581e88715478bfd825b56058558fba2babe09f7cb2d204ceccd2468aa`, matching that contract. I also read the deferred E9 backlog to check that the proposed next step remains a plan. I did not read other reviewer verdicts, the narrative card or a mutable draft, and did not run experiments.

## Verdict

**Revise — three recommendations.** The report already has a coherent, appropriately limited research story. The changes above make its main result accessible sooner, specify the central input comparison and remove internal process detail. I recommend preserving all six charts, their scientific qualifications, the adverse historical evidence and the technical appendices. I do not recommend a broad restructuring or a general round of sentence-level shortening.
