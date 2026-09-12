# Frozen evidence

I include 760 episode records from `first-20260904` and 149 from `permission-20260905`, with their complete planned allocations, cases and the permission protocol. The JSONL files are gzip-compressed. I retain prompts, generated text, tool evidence, outcome fields and every recorded scored token.

The first episode file decompresses to the original bytes. In the permission file, I omitted only the top-level sandbox identifier from each record. The first run's adjudications retain the recorded decisions, reasons, timestamps and raw-record hashes; reviewer metadata is omitted. Public analysis does not infer who made those decisions.

`export-manifest.json` records original and public hashes, byte counts, and the exact omitted fields for every exported source file. The paths in its `source` column are relative to my preserved private experiment, rather than paths in this repository. Hashes embedded in historical metadata continue to identify the original files and can therefore differ from the public export hashes. `page-attribution.json` identifies the Wikipedia revision and hashes for each page.

The full neutral-text preparation corpus, role-context text snapshots, operational records, and private writing drafts are excluded. These omissions prevent a complete rerun of preparation from this package alone; they do not change the saved-episode analysis. Third-party content is attributed in `../THIRD_PARTY.md`.
