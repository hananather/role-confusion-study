# Frozen new-page bank: 77 of the requested100 pages

I froze all77 accepted pages in their original acquisition-journal order. They have77 distinct Wikipedia dataset IDs and77 distinct raw hashes, with no title overlap against the five original pilot pages. No model outcomes were observed during selection.

The1000-source-candidate cap ended acquisition after154.8seconds. Of those candidates, two original titles were excluded before counting;77 were accepted,69 exceeded the original class-stripped100KiB filter, and852 returned retrieval errors (851 HTTP429, one404). I did not substitute another source or fetch further pages after the stopped acquisition. This is an incomplete100-page bank.

[manifest.json](manifest.json) contains77 forgery cases and77 corresponding standard-injection cases built by the unchanged frozen preparation function, with seed20260912 and copied source YAML. Standard-injection cases are attacks, not benign controls. The unmodified pages in `raw/` provide the separate benign inputs if a later plan includes them. Every case has its own deterministic seed; the same case seed must be shared across its treatment arms.

[partial-bank-receipt.json](partial-bank-receipt.json) records counts and hashes. [preparation-receipt.json](preparation-receipt.json) preserves the original failed100-page acquisition status. The source-candidate and retrieval journals retain all exclusions and failures. The original wrapper is preserved in `source/acquisition-wrapper-used.py`; the current wrapper adds a future fail-fast HTTP429 guard and records Retry-After, but I have not rerun it.

A later stage may use a frozen budgeted prefix of these77 pages, selected before treatment outcomes. Its manifest/run ID must remain part of the pairing key: local case IDs such as000-forgery repeat IDs from the earlier five-page cohort. The bank does not itself authorize a model run, establish attack success, or supply100 independent examples. Multiple arms or new seeds on a page do not increase the number of distinct pages.

Manifest SHA256: `19b95f66d7afb0b62620a4b5e10f861811a27eb8a41711c0110201d2dc236c97`.
