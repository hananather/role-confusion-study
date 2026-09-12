# Verification

The saved-result checks cover the September 12 agent snapshot.

| Check | Recorded result |
| --- | --- |
| Scientific artifact hashes and sizes | 2,224 files match the frozen manifest |
| Assignment accounting | 110 assignments: 100 recorded attempts, 10 unrun; 3 recorded attempts are censored |
| Episode and table consistency | All 100 recorded case IDs, seeds, censoring flags and verified-upload flags match |
| Export integrity | Listed files match the export manifest |
| CPU tests | 24 tests passed for the CUDA adapter, direction gate and closeout |

The [initial record](../provenance/verification.json) and
[expansion record](../provenance/expansion-verification.json) retain the test
environment and output. The timeout test emitted NumPy arithmetic warnings;
all tests passed. The tests use fake inference and temporary files.

For documentation and figure updates, I check file hashes, the document
conversion and navigation links.

[Commands](reproducing.md#saved-results) · [File manifest](../provenance/export-manifest.json)
