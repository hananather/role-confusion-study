# My persistent H100 session

I keep one H100 for consecutive experiments and retain the model, Python environment, probes and activations on a 2 TB network volume. The session lifetime is separate from each experiment's exit status. My current operational record is [the September 11 session](sessions/20260911T030050Z/SESSION.md), with its live [supervisor status](sessions/20260911T030050Z/session_status.json), [queue specification](sessions/20260911T030050Z/queue.json), and [synced outputs](sessions/20260911T030050Z/outputs/).

My approved sequence is probe pilot, full all-layer/two-split probes, Appendix E's three gardening conditions, Appendix K neutral controls, and RH6 readings. [The queue plan](queue-plan.md) records exact commands, input identity, readiness and exclusions. The earlier one-shot launchers must not manage this session: they delete the pod after an individual job.

I run `queue_worker.py` on the pod and `session_watchdog.py` on my Mac. The worker records each stage, halts on validation failures, and preserves completed-stage receipts. The supervisor syncs `/workspace/results/` every 30 seconds and deletes only the GPU pod at its three-hour deadline or the completed queue's root `QUEUE_DONE` marker. Individual `DONE`, `EXIT` and failure markers never terminate this session. My Mac remains awake through `caffeinate`; the supervisor depends on the Mac retaining network access. No account API key is passed to the pod.

I retain the independent network volume after GPU shutdown. Storage remains billed until I delete that volume explicitly; the current 2 TB allocation is prorated hourly. The account balance must continue to cover storage. My synced outputs are a second copy; the large activation caches remain on the volume.

I verify shutdown by checking the live provider inventory. I never interpret a requested delete or a finished Python process as proof that GPU billing stopped. Failed or ambiguous creation requests require live inventory reconciliation before another attempt.

My infrastructure validation includes 14 mocked lifecycle tests, a live write check on the mounted volume, verified uploaded source hashes, live supervisor synchronization, and a sequential-worker check covering failure halt and completed-stage reuse. Scientific CUDA results are established only by their actual stage outputs; model-free and synthetic checks do not establish replication.
