I indexed 313 saved trajectories across 2 arm groups and 313 prompt groups.

Harmfulness judging and review of whether the emitted reasoning recognizes an injection remain unfinished.

The source is [generations.jsonl](</Users/hananather/Desktop/MATS 12.0/replication/cloud/outbox/chat-steering/baseline-retry-20260912T004400Z/results/generations.jsonl>), SHA256 `57cd0771d992e246ca7cf481fb11021bf5f82343bc4693cfa73796c71f6c8063`.

- [Arm counts](</Users/hananather/Desktop/MATS 12.0/replication/cloud/outbox/chat-steering/baseline-retry-20260912T004400Z/review/index/arm-counts.csv>) reports saved completion, censoring, and emitted-reasoning availability.
- [Prompt and arm index](</Users/hananather/Desktop/MATS 12.0/replication/cloud/outbox/chat-steering/baseline-retry-20260912T004400Z/review/index/prompt-arm-index.csv>) gives one row per saved trajectory and its source line number.
- [Pairing matrix](</Users/hananather/Desktop/MATS 12.0/replication/cloud/outbox/chat-steering/baseline-retry-20260912T004400Z/review/index/prompt-arm-matrix.csv>) places each prompt's available arms side by side. Cell values are source line numbers; blank cells indicate missing rows.
- [Coverage and column definitions](</Users/hananather/Desktop/MATS 12.0/replication/cloud/outbox/chat-steering/baseline-retry-20260912T004400Z/review/index/coverage.json>) records the snapshot identity, warnings, duplicate cells, and each matrix column's arm settings.

I preserve saved censoring flags separately from final-channel completion. The long index copies any saved provisional labels as unreviewed proxies.

0 prompt groups have rows in more than one arm column; 0 prompt-arm cells contain duplicate rows.

The first indexed trajectory is [source line 1](</Users/hananather/Desktop/MATS 12.0/replication/cloud/outbox/chat-steering/baseline-retry-20260912T004400Z/results/generations.jsonl:1>).
