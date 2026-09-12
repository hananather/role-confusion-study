# My verification record

I ran the following checks from this isolated review directory with the existing
Python 3.11.8 installation. I installed no dependencies, loaded no model and
made no API or provider calls.

| Check | Observed result |
| --- | --- |
| Saved-evidence hashes and sizes | All 2,224 indexed artifacts match the frozen scientific manifest |
| Assignment accounting | 110 unique assignments, 100 recorded attempts, 10 unrun assignments and 3 censored recorded attempts |
| Raw episode/table consistency | All 100 recorded case IDs, seeds, censoring flags and verified-upload flags match their CSV rows; episode and table hashes match |
| Export integrity | Every file listed in the export manifest matches its recorded bytes and SHA-256 |
| Focused CPU tests | 24 tests passed across the CUDA adapter, Tool-raising gate and closeout modules |

The focused tests cover span masks, hook removal, zero-dose identity with fake
inference, intervention scaling, unchanged original direction arrays, gate
thresholds and separation of commands, receipts, censoring and unrun cells.
NumPy emitted divide-by-zero, overflow and invalid-value warnings from the test
tensor's matrix multiplication during the timeout test. The tests still passed;
I retain those warnings in the [machine-readable record](provenance/verification.json).

The verifiers read saved records. The unit tests use fake model objects and
temporary files. These checks do not reproduce GPU behavior, independently
retrain the probes, assess the factual quality of summaries or turn preliminary
assistant labels into human judgments. I did not rerun the earlier study's
figure/analysis scripts in this export.

To repeat the checks from this clone:

```bash
python3 scripts/verify_results.py
python3 scripts/verify_export.py
python3 -m unittest replication.cloud.agent_steering.test_hf_backend replication.cloud.parallel_h100.test_tool_raising replication.cloud.parallel_h100.test_closeout -v
```

The verifiers require only the standard library. The unit tests additionally
require NumPy, which was already available during export validation.
