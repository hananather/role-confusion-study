# jamesnelmore/role-confusion-extension: reference probe numbers

Extracted 2026-09-10 from the local clone at
`replication/reference/role-confusion-extension`. Read-only extraction. No model was run.
Every number below is copied from a shipped file. Tables marked "derived" were recomputed
from a shipped CSV with the notebook's own formula, and the rows that the notebook printed
were checked against them.

## 1. Provenance

| Item | Value |
|---|---|
| Repo | https://github.com/jamesnelmore/role-confusion-extension |
| Local commit | `b1387b0e19ad3d4880c1c4dbfb4099ee5874ef93`, James Elmore, 2026-09-05 03:22:30 +0000, "Persist probes, deliverables, and the working marimo demo before pod shutdown." |
| Authors' code | Git submodule `vendor/role-confusion` pinned to `ec333c40fd43fe991e1ebf66765051b6d7e35784`. Same commit as our own clone `prompt-injection-as-role-confusion/` and `role-steering/`. The submodule is not checked out locally (zero files). |
| What was run | The authors' `experiments/role-analysis/02-train-role-probes.ipynb`, cells 0 to 20, plus one appended cell 21 that clones the cuML probes into sklearn. One kernel, driven by `nbclient`. Notebook date 2026-09-04. |
| Only source edit | Cell 1: `ws = '/workspace/deliberative-alignment-jailbreaks'` became `ws = '/workspace/prompt-injection-as-role-confusion'`. The report says all other cells 0 to 20 diffed clean. |
| Hardware | NVIDIA H200 NVL, 139.80 GB as printed by the notebook (report says 143 GB, driver 580.126.16). Peak 30 GB VRAM, about 72 GB RSS. |
| Model | `openai/gpt-oss-20b`, HF snapshot `6cee5e81ee83917806bbde320786a8fb61efebee`, MXFP4 experts, attention `kernels-community/vllm-flash-attn3`. |
| Runtime | Cell 14 forward passes 209.7 s. Cell 19 (96 probes) 803.1 s. Whole notebook 1096.6 s. |

Software versions from the report, section 3.6:

```
python 3.12 (kernel reports 3.12.11)   torch 2.9.1+cu128   transformers 4.57.5
flash-attn 2.8.3+cu128torch2.9   kernels 0.11.5   cuml 25.08.00   cudf 25.08.00
cupy 14.2.0   scikit-learn 1.7.2   numpy 2.2.6   pandas 2.3.3   datasets 5.0.1
zstandard 0.25.0
```

Deviations from the authors' `setup_python.sh`, all environment-level, none touching the probe
definition (report section 3):

1. RAPIDS `25.9.*` does not exist. Installed cuML and cuDF 25.08 instead. The report flags this as the one change that can move coefficients, because the solver did not converge cleanly (see caveats).
2. scikit-learn pinned to 1.7.2 because 1.9.0 breaks cuML import. sklearn is only the `Pipeline` wrapper during training.
3. Added `zstandard` so Dolma3 streams.
4. pandas forced from 3.0.5 to 2.3.3 by cuDF's pin.
5. Report section 8 notes the activation cube already lands on CPU in the authors' `gptoss.py`, so no `.cpu()` patch was needed and cell 14 ran unmodified.

Data fingerprints resolved during the run: `allenai/c4` en validation at
`1588ec454efa1a09f29cd18ddd04fe05fc8653a2` (unpinned in the authors' code) and
`allenai/dolma3_mix-150B-1025` at `3a8349c2f7946cdc56f8ccf22c555672be0b3208` (pinned as `3a8349c`).

### Config used (from `config/probe.yaml`, entry `gptoss-20b`, unmodified)

Verified against the same file in our local authors' clone at the same commit.

| Setting | Value |
|---|---|
| `model_prefix` | `gptoss-20b` |
| `n_sample_size` | 250 (yields 62 C4 + 187 Dolma3 = 249 documents) |
| `seq_len` | 1024 |
| `nested_reasoning` | false |
| `train_prefixes` | `[""]` |
| `train_params.C` | 5.0e-3 |
| `train_params.add_scaling` | false |
| `layers_to_probe` | `range(0, 24, 2)`, so 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 (12 layers) |
| Estimator | `cuml.linear_model.LogisticRegression(penalty='l2', max_iter=5000, linesearch_max_iter=100, fit_intercept=True, C=5e-3)` inside `sklearn.pipeline.Pipeline([('clf', ...)])` |
| Split | `cuml.train_test_split` on unique `prompt_ix`, `test_size=0.1`, `random_state=123` |
| `seed` | 123 |
| Feature | pre-MLP residual, the output of `post_attention_layernorm`, cast to float16 for cupy then float32 for the fit |
| Role spaces | 8: `(user, assistant)`, `(user, assistant, tool)`, `(user, cot, assistant)`, `(user, cot, assistant, tool)`, `(system, user, assistant)`, `(system, user, assistant, tool)`, `(system, user, cot, assistant)`, `(system, user, cot, assistant, tool)` |
| Probes | 96 = 8 role spaces x 12 layers |

### Deliverables shipped

| File | Size | SHA-256 (from `SHA256SUMS.txt`, verified locally where present) |
|---|---|---|
| `deliverables/role_probes.pkl` | 7,498,917 bytes | `e0480e398788ed32717dcb7a7c0d50936993d7fb4d37912cae9759b267108b52` |
| `data/role_probes.pkl` | 7,498,917 bytes | identical hash to the deliverables copy |
| `gptoss-20b.pkl` (cuML original) | 5.8 MB | `f48fbb8659df0bddec22307c2cf3eba6c0fd28ff0be8219765a7a048a46caeff`. **Not in the repo.** Listed in the checksums and report only. |
| `deliverables/acc_by_role_gptoss-20b.csv` | 72,783 bytes, 1,245 rows | `6e2884bfbf264a2fdf53ff63a32dd8a48928bbf1b01d7079f6fb8e112bed5aa3` |
| `deliverables/acc_by_pos_gptoss-20b.csv` | 5,131,059 bytes, 98,304 rows | `70f2b70f17c96a7e57db4736c576fff7d4f3260a282f069b68d8e1d566ac6644` |
| `deliverables/02-train-role-probes-EXECUTED.ipynb` | 22 cells with outputs | |
| `deliverables/run_nb.log` | full console log | |
| `deliverables/REPORT.md` | training report | |

## 2. Numbers printed by the executed notebook

### 2.1 Printed shapes and sanity prints

| Cell | Print |
|---|---|
| 1 | `Device 0: NVIDIA H200 NVL`, `Total: 139.80 GB` |
| 3 | `Allocated: 12.85 GB`, `Reserved: 12.90 GB` after model load |
| 4 | `LM loss: 3.984269618988037` |
| 4 | `Hidden states layers (pre-mlp | post-layer): 24 | 24` |
| 4 | `Hidden state size (pre-mlp | post-layer): torch.Size([1280, 2880]) | torch.Size([1280, 2880])` |
| 4 | `Verified custom forward pass successfully matches original model output!` |
| 11 | `input_df`: `[1245 rows x 5 columns]`, `question_ix` 0 to 248, roles in order system, user, tool, cot, assistant |
| 14 | 39 batches of 32 (`0/39` to `39/39`) |
| 14 | `PPL: 263.0306091308594` (the single decoded-sequence check inside `run_and_export_states`) |
| 14 | `6554` (return value of `gc.collect()`) |
| 18 | `0` (grid search left commented out; `gc.collect()` return value) |
| 19 | `Num probes: 96`, `Probe layers: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22` |
| 21 | `max |p_cuml - p_sklearn| over all 96 probes = 1.908e-06` |

The report adds `max_seqlen 1035` for the dataloader. That value is not printed in the notebook.

### 2.2 Role token counts (cell 16, verbatim)

| | role | count |
|---|---|---|
| 0 | assistant | 141078 |
| 1 | cot | 141078 |
| 2 | system | 141078 |
| 3 | tool | 141078 |
| 4 | user | 141078 |

Total 705,390 content tokens. Exact equality across roles is the notebook's own correctness check.

### 2.3 Validation accuracy by layer (cell 20, verbatim; rounded to 2 dp by the notebook)

Columns are role spaces abbreviated by first letter. Chance is 1 / |role space|.

| layer_ix | s,u,a | s,u,a,t | s,u,c,a | s,u,c,a,t | u,a | u,a,t | u,c,a | u,c,a,t |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.23 | 0.19 | 0.16 | 0.14 | 0.46 | 0.29 | 0.25 | 0.19 |
| 2 | 0.35 | 0.36 | 0.26 | 0.28 | 0.71 | 0.51 | 0.43 | 0.36 |
| 4 | 0.40 | 0.38 | 0.31 | 0.30 | 0.65 | 0.54 | 0.47 | 0.38 |
| 6 | 0.59 | 0.63 | 0.59 | 0.51 | 0.85 | 0.74 | 0.69 | 0.61 |
| 8 | 0.73 | 0.82 | 0.81 | 0.70 | 0.92 | 0.88 | 0.85 | 0.80 |
| 10 | 0.69 | 0.76 | 0.74 | 0.62 | 0.89 | 0.82 | 0.79 | 0.72 |
| 12 | 0.70 | 0.72 | 0.70 | 0.59 | 0.88 | 0.80 | 0.77 | 0.67 |
| 14 | 0.77 | 0.79 | 0.80 | 0.64 | 0.92 | 0.85 | 0.83 | 0.75 |
| 16 | 0.88 | 0.87 | 0.88 | 0.72 | 0.97 | 0.91 | 0.90 | 0.82 |
| 18 | 0.88 | 0.84 | 0.85 | 0.71 | 0.96 | 0.90 | 0.89 | 0.79 |
| 20 | 0.86 | 0.80 | 0.82 | 0.67 | 0.96 | 0.87 | 0.87 | 0.75 |
| 22 | 0.84 | 0.82 | 0.83 | 0.67 | 0.96 | 0.86 | 0.86 | 0.75 |

### 2.4 Five-role clone check (cell 21, verbatim, 3 dp)

| layer | acc | coef shape |
|---|---|---|
| L00 | 0.143 | (5, 2880) |
| L02 | 0.276 | (5, 2880) |
| L04 | 0.298 | (5, 2880) |
| L06 | 0.510 | (5, 2880) |
| L08 | 0.704 | (5, 2880) |
| L10 | 0.621 | (5, 2880) |
| L12 | 0.595 | (5, 2880) |
| L14 | 0.638 | (5, 2880) |
| L16 | 0.725 | (5, 2880) |
| L18 | 0.710 | (5, 2880) |
| L20 | 0.667 | (5, 2880) |
| L22 | 0.674 | (5, 2880) |

### 2.5 Per-probe held-out accuracy and log loss, full precision (read from `role_probes.pkl`)

These are the `acc` and `nll` floats stored in each dict. `n_inputs` is the number of tokens
in the role space before the train/test split. Rounding these to 2 dp reproduces table 2.3.

| role_space | layer_ix | acc | nll | n_inputs |
|---|---|---|---|---|
| user,assistant | 0 | 0.46381936887921654 | 0.7318826795883477 | 282156 |
| user,assistant | 2 | 0.7072128089538318 | 0.5781848536880758 | 282156 |
| user,assistant | 4 | 0.6478703559769936 | 0.6869662772056243 | 282156 |
| user,assistant | 6 | 0.8511192289755946 | 0.3793278377124288 | 282156 |
| user,assistant | 8 | 0.9194388310275143 | 0.2287064284264721 | 282156 |
| user,assistant | 10 | 0.8868335146898803 | 0.33238130518335846 | 282156 |
| user,assistant | 12 | 0.875019431058604 | 0.36954386491915947 | 282156 |
| user,assistant | 14 | 0.9157469298927405 | 0.2784696616242943 | 282156 |
| user,assistant | 16 | 0.966617441318203 | 0.09302288416433074 | 282156 |
| user,assistant | 18 | 0.9633530234727188 | 0.09835324831916449 | 282156 |
| user,assistant | 20 | 0.9567853256645422 | 0.1167911881543726 | 282156 |
| user,assistant | 22 | 0.9572516710710399 | 0.11175477325846869 | 282156 |
| user,assistant,tool | 0 | 0.288357822574846 | 1.1630034446716309 | 423234 |
| user,assistant,tool | 2 | 0.5137707039957813 | 1.0531516075134277 | 423234 |
| user,assistant,tool | 4 | 0.5433974927491071 | 1.0790749788284302 | 423234 |
| user,assistant,tool | 6 | 0.7427550995949088 | 0.6967307329177856 | 423234 |
| user,assistant,tool | 8 | 0.8765071070735156 | 0.3583667576313019 | 423234 |
| user,assistant,tool | 10 | 0.8188595124523599 | 0.5286853313446045 | 423234 |
| user,assistant,tool | 12 | 0.8013854598624128 | 0.5487675070762634 | 423234 |
| user,assistant,tool | 14 | 0.8454900644790143 | 0.4572613835334778 | 423234 |
| user,assistant,tool | 16 | 0.9106402358637551 | 0.35041508078575134 | 423234 |
| user,assistant,tool | 18 | 0.8981998609746159 | 0.3848942816257477 | 423234 |
| user,assistant,tool | 20 | 0.8680936743450226 | 0.44811609387397766 | 423234 |
| user,assistant,tool | 22 | 0.8586495361825547 | 0.49347078800201416 | 423234 |
| user,cot,assistant | 0 | 0.2485677988446511 | 1.170615792274475 | 423234 |
| user,cot,assistant | 2 | 0.43006783479949184 | 1.0753793716430664 | 423234 |
| user,cot,assistant | 4 | 0.46554327764328 | 1.120324730873108 | 423234 |
| user,cot,assistant | 6 | 0.6869531867973825 | 0.7720783948898315 | 423234 |
| user,cot,assistant | 8 | 0.8467125290634963 | 0.41639745235443115 | 423234 |
| user,cot,assistant | 10 | 0.7900477000886886 | 0.5659059882164001 | 423234 |
| user,cot,assistant | 12 | 0.7728133464368753 | 0.6007407903671265 | 423234 |
| user,cot,assistant | 14 | 0.8345597929001175 | 0.49976375699043274 | 423234 |
| user,cot,assistant | 16 | 0.9034013279321172 | 0.3484792411327362 | 423234 |
| user,cot,assistant | 18 | 0.892063568158393 | 0.35218727588653564 | 423234 |
| user,cot,assistant | 20 | 0.8679019151945156 | 0.422136515378952 | 423234 |
| user,cot,assistant | 22 | 0.8576667705362065 | 0.4712257385253906 | 423234 |
| user,cot,assistant,tool | 0 | 0.18627249836799403 | 1.4450433254241943 | 564312 |
| user,cot,assistant,tool | 2 | 0.3555534831670242 | 1.3084064722061157 | 564312 |
| user,cot,assistant,tool | 4 | 0.37888650564207776 | 1.3441938161849976 | 564312 |
| user,cot,assistant,tool | 6 | 0.6070689172806117 | 0.9344499111175537 | 564312 |
| user,cot,assistant,tool | 8 | 0.8033759209176536 | 0.48706233501434326 | 564312 |
| user,cot,assistant,tool | 10 | 0.7171687027883987 | 0.6939997673034668 | 564312 |
| user,cot,assistant,tool | 12 | 0.6749044110789891 | 0.787736177444458 | 564312 |
| user,cot,assistant,tool | 14 | 0.7487829898349343 | 0.6198878288269043 | 564312 |
| user,cot,assistant,tool | 16 | 0.820442040473748 | 0.45952892303466797 | 564312 |
| user,cot,assistant,tool | 18 | 0.7912524480089527 | 0.5109291672706604 | 564312 |
| user,cot,assistant,tool | 20 | 0.754005408934067 | 0.590122401714325 | 564312 |
| user,cot,assistant,tool | 22 | 0.7542665298890235 | 0.6097711324691772 | 564312 |
| system,user,assistant | 0 | 0.2275222320765119 | 1.1760603189468384 | 423234 |
| system,user,assistant | 2 | 0.34696421294853663 | 1.1489938497543335 | 423234 |
| system,user,assistant | 4 | 0.3951916393010379 | 1.2130769491195679 | 423234 |
| system,user,assistant | 6 | 0.5935664805004914 | 0.9041313529014587 | 423234 |
| system,user,assistant | 8 | 0.7323281957860927 | 0.6983363628387451 | 423234 |
| system,user,assistant | 10 | 0.6866176082839953 | 0.8204758167266846 | 423234 |
| system,user,assistant | 12 | 0.7014549725544715 | 0.7905177474021912 | 423234 |
| system,user,assistant | 14 | 0.7702485677988447 | 0.6680317521095276 | 423234 |
| system,user,assistant | 16 | 0.8829789784031257 | 0.471918523311615 | 423234 |
| system,user,assistant | 18 | 0.8791917351806131 | 0.5091526508331299 | 423234 |
| system,user,assistant | 20 | 0.8565162156331647 | 0.5543083548545837 | 423234 |
| system,user,assistant | 22 | 0.8358781370598528 | 0.6622288823127747 | 423234 |
| system,user,assistant,tool | 0 | 0.1853772265224284 | 1.4416463375091553 | 564312 |
| system,user,assistant,tool | 2 | 0.362473188473375 | 1.287725567817688 | 564312 |
| system,user,assistant,tool | 4 | 0.3846498181479064 | 1.3400710821151733 | 564312 |
| system,user,assistant,tool | 6 | 0.6254033386179241 | 0.906160831451416 | 564312 |
| system,user,assistant,tool | 8 | 0.8183717243308776 | 0.45107123255729675 | 564312 |
| system,user,assistant,tool | 10 | 0.7606639932854612 | 0.59420245885849 | 564312 |
| system,user,assistant,tool | 12 | 0.7176909446983121 | 0.6910450458526611 | 564312 |
| system,user,assistant,tool | 14 | 0.7850601510771239 | 0.5514707565307617 | 564312 |
| system,user,assistant,tool | 16 | 0.8722558985358575 | 0.3523404002189636 | 564312 |
| system,user,assistant,tool | 18 | 0.8395038701855824 | 0.4086962938308716 | 564312 |
| system,user,assistant,tool | 20 | 0.800391681432435 | 0.49527543783187866 | 564312 |
| system,user,assistant,tool | 22 | 0.8177375734402685 | 0.46982282400131226 | 564312 |
| system,user,cot,assistant | 0 | 0.15814604121980788 | 1.445725440979004 | 564312 |
| system,user,cot,assistant | 2 | 0.26220274177002706 | 1.3223875761032104 | 564312 |
| system,user,cot,assistant | 4 | 0.3057912897510025 | 1.4001282453536987 | 564312 |
| system,user,cot,assistant | 6 | 0.5931922036743449 | 0.9422605037689209 | 564312 |
| system,user,cot,assistant | 8 | 0.8088408094749604 | 0.4692535996437073 | 564312 |
| system,user,cot,assistant | 10 | 0.737405576797538 | 0.6348122358322144 | 564312 |
| system,user,cot,assistant | 12 | 0.7033665951692624 | 0.7270163893699646 | 564312 |
| system,user,cot,assistant | 14 | 0.7975753054182598 | 0.5419626832008362 | 564312 |
| system,user,cot,assistant | 16 | 0.8837452205539494 | 0.35263940691947937 | 564312 |
| system,user,cot,assistant | 18 | 0.8536417047468059 | 0.41482603549957275 | 564312 |
| system,user,cot,assistant | 20 | 0.8235195374428798 | 0.47073644399642944 | 564312 |
| system,user,cot,assistant | 22 | 0.8262986104634897 | 0.46719783544540405 | 564312 |
| system,user,cot,assistant,tool | 0 | 0.14295628036086053 | 1.680220603942871 | 705390 |
| system,user,cot,assistant,tool | 2 | 0.27575413055354586 | 1.584196925163269 | 705390 |
| system,user,cot,assistant,tool | 4 | 0.29818240878822333 | 1.6494677066802979 | 705390 |
| system,user,cot,assistant,tool | 6 | 0.5100919869475985 | 1.2445120811462402 | 705390 |
| system,user,cot,assistant,tool | 8 | 0.7036337059075406 | 0.835594654083252 | 705390 |
| system,user,cot,assistant,tool | 10 | 0.6211112259512455 | 1.0394575595855713 | 705390 |
| system,user,cot,assistant,tool | 12 | 0.5947258848022207 | 1.1149046421051025 | 705390 |
| system,user,cot,assistant,tool | 14 | 0.6379435084973496 | 1.0388473272323608 | 705390 |
| system,user,cot,assistant,tool | 16 | 0.7248512410117087 | 0.815012514591217 | 705390 |
| system,user,cot,assistant,tool | 18 | 0.7098793686417529 | 0.8505873084068298 | 705390 |
| system,user,cot,assistant,tool | 20 | 0.6670013436295716 | 0.9462666511535645 | 705390 |
| system,user,cot,assistant,tool | 22 | 0.674442984334165 | 0.9261564016342163 | 705390 |

### 2.6 Accuracy by role (cell 20)

The notebook displays this 96-row table truncated by pandas. The rows it actually printed:

| role_space | layer_ix | assistant | cot | system | tool | user |
|---|---|---|---|---|---|---|
| system,user,assistant | 0 | 0.39 | NaN | 0.10 | NaN | 0.25 |
| system,user,assistant | 2 | 0.61 | NaN | 0.20 | NaN | 0.29 |
| system,user,assistant | 4 | 0.61 | NaN | 0.26 | NaN | 0.38 |
| system,user,assistant | 6 | 0.78 | NaN | 0.47 | NaN | 0.59 |
| system,user,assistant | 8 | 0.83 | NaN | 0.68 | NaN | 0.71 |
| ... | ... | ... | ... | ... | ... | ... |
| user,cot,assistant,tool | 14 | 0.89 | 0.64 | NaN | 0.61 | 0.84 |
| user,cot,assistant,tool | 16 | 0.94 | 0.73 | NaN | 0.64 | 0.96 |
| user,cot,assistant,tool | 18 | 0.92 | 0.65 | NaN | 0.63 | 0.96 |
| user,cot,assistant,tool | 20 | 0.91 | 0.58 | NaN | 0.58 | 0.94 |
| user,cot,assistant,tool | 22 | 0.91 | 0.59 | NaN | 0.61 | 0.92 |

Full table, **derived** from `acc_by_role_gptoss-20b.csv` with the notebook's formula
(per role: sum of `count` where `pred == role`, divided by the sum of `count` for that role,
rounded to 2 dp). The ten printed rows above match exactly.

| role_space | layer_ix | assistant | cot | system | tool | user |
|---|---|---|---|---|---|---|
| user,assistant | 0 | 0.44 | NaN | NaN | NaN | 0.50 |
| user,assistant | 2 | 0.70 | NaN | NaN | NaN | 0.72 |
| user,assistant | 4 | 0.64 | NaN | NaN | NaN | 0.66 |
| user,assistant | 6 | 0.82 | NaN | NaN | NaN | 0.89 |
| user,assistant | 8 | 0.88 | NaN | NaN | NaN | 0.98 |
| user,assistant | 10 | 0.84 | NaN | NaN | NaN | 0.95 |
| user,assistant | 12 | 0.83 | NaN | NaN | NaN | 0.93 |
| user,assistant | 14 | 0.87 | NaN | NaN | NaN | 0.97 |
| user,assistant | 16 | 0.95 | NaN | NaN | NaN | 0.98 |
| user,assistant | 18 | 0.94 | NaN | NaN | NaN | 0.99 |
| user,assistant | 20 | 0.93 | NaN | NaN | NaN | 0.99 |
| user,assistant | 22 | 0.93 | NaN | NaN | NaN | 0.99 |
| user,assistant,tool | 0 | 0.23 | NaN | NaN | 0.34 | 0.30 |
| user,assistant,tool | 2 | 0.50 | NaN | NaN | 0.49 | 0.54 |
| user,assistant,tool | 4 | 0.49 | NaN | NaN | 0.49 | 0.61 |
| user,assistant,tool | 6 | 0.74 | NaN | NaN | 0.72 | 0.76 |
| user,assistant,tool | 8 | 0.83 | NaN | NaN | 0.92 | 0.88 |
| user,assistant,tool | 10 | 0.80 | NaN | NaN | 0.82 | 0.83 |
| user,assistant,tool | 12 | 0.81 | NaN | NaN | 0.79 | 0.80 |
| user,assistant,tool | 14 | 0.84 | NaN | NaN | 0.86 | 0.84 |
| user,assistant,tool | 16 | 0.84 | NaN | NaN | 0.94 | 0.94 |
| user,assistant,tool | 18 | 0.83 | NaN | NaN | 0.93 | 0.92 |
| user,assistant,tool | 20 | 0.82 | NaN | NaN | 0.88 | 0.89 |
| user,assistant,tool | 22 | 0.81 | NaN | NaN | 0.88 | 0.88 |
| user,cot,assistant | 0 | 0.18 | 0.22 | NaN | NaN | 0.31 |
| user,cot,assistant | 2 | 0.29 | 0.30 | NaN | NaN | 0.61 |
| user,cot,assistant | 4 | 0.32 | 0.33 | NaN | NaN | 0.64 |
| user,cot,assistant | 6 | 0.62 | 0.59 | NaN | NaN | 0.79 |
| user,cot,assistant | 8 | 0.78 | 0.83 | NaN | NaN | 0.90 |
| user,cot,assistant | 10 | 0.73 | 0.73 | NaN | NaN | 0.87 |
| user,cot,assistant | 12 | 0.74 | 0.70 | NaN | NaN | 0.84 |
| user,cot,assistant | 14 | 0.79 | 0.81 | NaN | NaN | 0.88 |
| user,cot,assistant | 16 | 0.81 | 0.91 | NaN | NaN | 0.96 |
| user,cot,assistant | 18 | 0.80 | 0.89 | NaN | NaN | 0.95 |
| user,cot,assistant | 20 | 0.79 | 0.85 | NaN | NaN | 0.93 |
| user,cot,assistant | 22 | 0.75 | 0.85 | NaN | NaN | 0.93 |
| user,cot,assistant,tool | 0 | 0.09 | 0.11 | NaN | 0.26 | 0.35 |
| user,cot,assistant,tool | 2 | 0.23 | 0.23 | NaN | 0.49 | 0.55 |
| user,cot,assistant,tool | 4 | 0.26 | 0.23 | NaN | 0.47 | 0.63 |
| user,cot,assistant,tool | 6 | 0.67 | 0.46 | NaN | 0.60 | 0.73 |
| user,cot,assistant,tool | 8 | 0.86 | 0.73 | NaN | 0.75 | 0.89 |
| user,cot,assistant,tool | 10 | 0.80 | 0.61 | NaN | 0.65 | 0.82 |
| user,cot,assistant,tool | 12 | 0.78 | 0.55 | NaN | 0.60 | 0.78 |
| user,cot,assistant,tool | 14 | 0.89 | 0.64 | NaN | 0.61 | 0.84 |
| user,cot,assistant,tool | 16 | 0.94 | 0.73 | NaN | 0.64 | 0.96 |
| user,cot,assistant,tool | 18 | 0.92 | 0.65 | NaN | 0.63 | 0.96 |
| user,cot,assistant,tool | 20 | 0.91 | 0.58 | NaN | 0.58 | 0.94 |
| user,cot,assistant,tool | 22 | 0.91 | 0.59 | NaN | 0.61 | 0.92 |
| system,user,assistant | 0 | 0.39 | NaN | 0.10 | NaN | 0.25 |
| system,user,assistant | 2 | 0.61 | NaN | 0.20 | NaN | 0.29 |
| system,user,assistant | 4 | 0.61 | NaN | 0.26 | NaN | 0.38 |
| system,user,assistant | 6 | 0.78 | NaN | 0.47 | NaN | 0.59 |
| system,user,assistant | 8 | 0.83 | NaN | 0.68 | NaN | 0.71 |
| system,user,assistant | 10 | 0.80 | NaN | 0.64 | NaN | 0.64 |
| system,user,assistant | 12 | 0.81 | NaN | 0.65 | NaN | 0.67 |
| system,user,assistant | 14 | 0.84 | NaN | 0.76 | NaN | 0.70 |
| system,user,assistant | 16 | 0.85 | NaN | 0.88 | NaN | 0.93 |
| system,user,assistant | 18 | 0.84 | NaN | 0.88 | NaN | 0.92 |
| system,user,assistant | 20 | 0.84 | NaN | 0.85 | NaN | 0.89 |
| system,user,assistant | 22 | 0.82 | NaN | 0.82 | NaN | 0.87 |
| system,user,assistant,tool | 0 | 0.18 | NaN | 0.16 | 0.22 | 0.17 |
| system,user,assistant,tool | 2 | 0.49 | NaN | 0.20 | 0.44 | 0.27 |
| system,user,assistant,tool | 4 | 0.45 | NaN | 0.24 | 0.39 | 0.43 |
| system,user,assistant,tool | 6 | 0.77 | NaN | 0.43 | 0.54 | 0.74 |
| system,user,assistant,tool | 8 | 0.94 | NaN | 0.63 | 0.79 | 0.88 |
| system,user,assistant,tool | 10 | 0.90 | NaN | 0.58 | 0.71 | 0.84 |
| system,user,assistant,tool | 12 | 0.88 | NaN | 0.55 | 0.62 | 0.81 |
| system,user,assistant,tool | 14 | 0.96 | NaN | 0.67 | 0.64 | 0.87 |
| system,user,assistant,tool | 16 | 0.98 | NaN | 0.80 | 0.77 | 0.94 |
| system,user,assistant,tool | 18 | 0.97 | NaN | 0.74 | 0.71 | 0.93 |
| system,user,assistant,tool | 20 | 0.96 | NaN | 0.67 | 0.63 | 0.94 |
| system,user,assistant,tool | 22 | 0.96 | NaN | 0.69 | 0.68 | 0.94 |
| system,user,cot,assistant | 0 | 0.13 | 0.16 | 0.17 | NaN | 0.19 |
| system,user,cot,assistant | 2 | 0.26 | 0.29 | 0.21 | NaN | 0.28 |
| system,user,cot,assistant | 4 | 0.27 | 0.28 | 0.26 | NaN | 0.43 |
| system,user,cot,assistant | 6 | 0.67 | 0.50 | 0.46 | NaN | 0.74 |
| system,user,cot,assistant | 8 | 0.86 | 0.79 | 0.69 | NaN | 0.88 |
| system,user,cot,assistant | 10 | 0.79 | 0.68 | 0.63 | NaN | 0.84 |
| system,user,cot,assistant | 12 | 0.78 | 0.61 | 0.61 | NaN | 0.81 |
| system,user,cot,assistant | 14 | 0.89 | 0.70 | 0.73 | NaN | 0.87 |
| system,user,cot,assistant | 16 | 0.94 | 0.80 | 0.86 | NaN | 0.94 |
| system,user,cot,assistant | 18 | 0.93 | 0.74 | 0.82 | NaN | 0.93 |
| system,user,cot,assistant | 20 | 0.92 | 0.68 | 0.77 | NaN | 0.94 |
| system,user,cot,assistant | 22 | 0.91 | 0.69 | 0.77 | NaN | 0.95 |
| system,user,cot,assistant,tool | 0 | 0.08 | 0.15 | 0.18 | 0.17 | 0.16 |
| system,user,cot,assistant,tool | 2 | 0.20 | 0.25 | 0.22 | 0.42 | 0.23 |
| system,user,cot,assistant,tool | 4 | 0.25 | 0.23 | 0.23 | 0.36 | 0.40 |
| system,user,cot,assistant,tool | 6 | 0.64 | 0.36 | 0.35 | 0.50 | 0.65 |
| system,user,cot,assistant,tool | 8 | 0.90 | 0.61 | 0.51 | 0.63 | 0.81 |
| system,user,cot,assistant,tool | 10 | 0.85 | 0.50 | 0.44 | 0.50 | 0.77 |
| system,user,cot,assistant,tool | 12 | 0.84 | 0.46 | 0.43 | 0.45 | 0.77 |
| system,user,cot,assistant,tool | 14 | 0.86 | 0.53 | 0.51 | 0.47 | 0.82 |
| system,user,cot,assistant,tool | 16 | 0.90 | 0.60 | 0.68 | 0.55 | 0.93 |
| system,user,cot,assistant,tool | 18 | 0.89 | 0.54 | 0.66 | 0.55 | 0.92 |
| system,user,cot,assistant,tool | 20 | 0.87 | 0.50 | 0.60 | 0.47 | 0.92 |
| system,user,cot,assistant,tool | 22 | 0.89 | 0.52 | 0.61 | 0.48 | 0.91 |

Held-out test token counts per role (the CSV `count` sums; constant across layers within a
role space because the split is by `prompt_ix` with a fixed seed):

| role_space | assistant | cot | system | tool | user |
|---|---|---|---|---|---|
| user,assistant | 14408 | | | | 11324 |
| user,assistant,tool | 12261 | | | 11132 | 18326 |
| user,cot,assistant | 12261 | 11132 | | | 18326 |
| user,cot,assistant,tool | 15269 | 14974 | | 11784 | 11588 |
| system,user,assistant | 12261 | | 18326 | | 11132 |
| system,user,assistant,tool | 15269 | | 11588 | 14974 | 11784 |
| system,user,cot,assistant | 15269 | 14974 | 11588 | | 11784 |
| system,user,cot,assistant,tool | 14793 | 13765 | 9414 | 16909 | 12846 |

Five-role confusion counts, derived from the same CSV (rows are true role, columns are predicted):

Layer 12:

| true \ pred | system | user | cot | assistant | tool |
|---|---|---|---|---|---|
| system | 4078 | 2017 | 1485 | 452 | 1382 |
| user | 1738 | 9857 | 308 | 59 | 884 |
| cot | 1947 | 797 | 6372 | 1116 | 3533 |
| assistant | 332 | 27 | 1678 | 12411 | 345 |
| tool | 3413 | 1646 | 3818 | 471 | 7561 |

Layer 16:

| true \ pred | system | user | cot | assistant | tool |
|---|---|---|---|---|---|
| system | 6371 | 1100 | 1059 | 96 | 788 |
| user | 606 | 11933 | 75 | 6 | 226 |
| cot | 687 | 90 | 8213 | 580 | 4195 |
| assistant | 23 | 0 | 1258 | 13337 | 175 |
| tool | 2732 | 621 | 3859 | 459 | 9238 |

### 2.7 Accuracy by token position

`acc_by_pos_gptoss-20b.csv` has 98,304 rows: columns `token_in_seg_ix` (0 to 1023), `count`,
`acc` (4 dp), `model`, `layer_ix`, `role_space`, for all 8 role spaces x 12 layers. Not
reproduced here. The first rows (`user,assistant`, layer 0) have `count` 49 and `acc` from
0.9388 at position 0 down to about 0.67 to 0.86 by positions 20 to 33.

### 2.8 Outputs that are NOT in this notebook

The executed notebook has 22 cells. The authors' original has 56. Everything after the save
cell was dropped, so none of the following exist anywhere in the repo:

- Conversation projection tables (mean prob and argmax accuracy by `conv_type` x role space x layer x role). Original cells 21 to 38.
- Alt-model conversation validation. Original cells 39 to 46.
- Tomato / gardening example and `tomato-projections` CSV. Original cells 47 to 55.
- Any perplexity numbers beyond the single `PPL: 263.0306091308594` decoded-sequence check in cell 14.
- The grid search over `C` (cell 18 is present but its loop is commented out, as in the original).

A grep for "tomato" or "gardening" hits only `REPORT.md`, which mentions them as dropped.

## 3. Pickle structure and weight extraction

Both `deliverables/role_probes.pkl` and `data/role_probes.pkl` are byte-identical
(SHA-256 `e0480e39...08b52`). They are the **portable sklearn clone**, not the cuML original.
The cuML original `gptoss-20b.pkl` is not shipped.

Inspected with `pickletools.dis` and a custom `Unpickler` that stubs every non-numpy class.
Findings:

- Pickle protocol 4. Top level is a Python `list` of 96 `dict`s, in the order the notebook trained them: role space outer loop (`user,assistant` first, `system,user,cot,assistant,tool` last), layer 0 to 22 inner loop.
- Dict keys: `probe`, `acc`, `nll`, `layer_ix`, `role_space`, `roles_map`, `n_inputs`, `C`, `feature`.
  - `acc`, `nll`: Python floats (see table 2.5).
  - `layer_ix`: int. `role_space`: list of str. `roles_map`: dict role to class index in `role_space` order. `n_inputs`: int. `C`: 0.005. `feature`: `"pre_mlp"`.
  - There is **no** `acc_by_role` or `acc_by_pos` inside the pickle. Those live only in the two CSVs. The cuML original would have carried them as DataFrames.
- `probe` class path: `sklearn.linear_model._logistic.LogisticRegression`. No `Pipeline`, no cuml, no cupy anywhere in the file. Only `numpy._core.multiarray._reconstruct`, `numpy.ndarray`, `numpy.dtype`, and the sklearn class are referenced.
- Estimator `__dict__` (every probe): `penalty='l2'`, `dual=False`, `tol=0.0001`, `C=0.005`, `fit_intercept=True`, `intercept_scaling=1`, `class_weight=None`, `random_state=None`, `solver='lbfgs'`, `max_iter=5000`, `multi_class='deprecated'`, `verbose=0`, `warm_start=False`, `n_jobs=None`, `l1_ratio=None`, `n_features_in_=2880`, `_sklearn_version='1.7.2'`, plus the three arrays.
- Arrays are plain numpy, little-endian, no NaNs:

| role space size | `coef_` | `intercept_` | `classes_` |
|---|---|---|---|
| 2 (`user,assistant`) | `(1, 2880)` float64 | `(1,)` float64 | `[0, 1]` int64 |
| 3 | `(3, 2880)` float64 | `(3,)` float64 | `[0, 1, 2]` int64 |
| 4 | `(4, 2880)` float64 | `(4,)` float64 | `[0, 1, 2, 3]` int64 |
| 5 | `(5, 2880)` float64 | `(5,)` float64 | `[0, 1, 2, 3, 4]` int64 |

- The float64 values are a cast of cuML's float32 coefficients (report section 6.2). The clone reproduces cuML `predict_proba` to `1.908e-06` on 2,000 real activations per probe (cell 21).
- The binary probe has one coefficient row. Its sign convention is sklearn's: positive logit means class 1, which is `assistant` under `roles_map = {'user': 0, 'assistant': 1}`.

Five-role `coef_` row L2 norms and intercepts, order system, user, cot, assistant, tool (derived, 4 dp):

| layer | norms | intercepts |
|---|---|---|
| 0 | 1.2729, 1.3443, 1.1130, 0.9532, 1.3888 | 0.0221, 0.0430, 0.0207, 0.0565, -0.1423 |
| 2 | 2.7255, 2.7945, 2.8289, 2.4383, 3.1335 | -0.2167, -0.1877, 0.1637, 0.0088, 0.2318 |
| 4 | 2.4334, 3.2452, 2.6676, 2.4681, 3.2786 | -0.0229, -0.0031, 0.0662, -0.0926, 0.0450 |
| 6 | 3.5687, 5.6549, 3.4479, 5.6398, 4.7231 | -0.1244, 0.0577, 0.1439, -0.0064, -0.0594 |
| 8 | 3.8489, 6.3562, 4.4887, 7.2011, 4.5337 | 0.0698, -0.0820, 0.1440, -0.1016, -0.0302 |
| 10 | 3.6351, 5.4476, 3.2182, 6.8838, 3.8729 | -0.0036, -0.1652, 0.1654, -0.0870, 0.0899 |
| 12 | 3.3234, 4.8831, 2.6903, 6.2987, 3.2085 | 0.0043, -0.1023, 0.1353, -0.1020, 0.0647 |
| 14 | 3.7862, 5.0609, 3.0187, 6.2653, 3.0837 | -0.0372, 0.0536, 0.1032, -0.1497, 0.0305 |
| 16 | 4.3776, 5.9863, 3.6584, 6.3935, 3.3758 | 0.0481, 0.0349, 0.0369, -0.1247, 0.0048 |
| 18 | 4.7089, 6.7600, 4.2396, 6.4907, 3.5452 | 0.0652, 0.0210, 0.0774, -0.0658, -0.1008 |
| 20 | 4.3145, 6.2998, 3.8565, 6.2983, 2.9278 | 0.0501, 0.0215, 0.0100, -0.0893, 0.0077 |
| 22 | 4.5068, 6.1899, 3.8947, 6.3216, 3.7313 | 0.0232, -0.0721, 0.1057, 0.0154, -0.0636 |

### Extraction recipe without cuml, cupy, or even sklearn

Weights are fully recoverable with numpy alone. Stub the sklearn class so the version of
sklearn on the machine (or its absence) does not matter:

```python
import io, pickle, numpy as np

class _Stub:
    def __init__(self, *a, **k): pass
    def __setstate__(self, state): self.__dict__.update(state)

class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy") or module in ("builtins", "collections", "copyreg", "_codecs"):
            return super().find_class(module, name)
        return type(f"{module}.{name}", (_Stub,), {})   # sklearn LogisticRegression -> stub

with open("data/role_probes.pkl", "rb") as f:
    probes = _Unpickler(f).load()

FIVE = ["system", "user", "cot", "assistant", "tool"]
p = next(x for x in probes if x["layer_ix"] == 16 and x["role_space"] == FIVE)
W = p["probe"].__dict__["coef_"]        # (5, 2880) float64
b = p["probe"].__dict__["intercept_"]   # (5,)
u, t = p["roles_map"]["user"], p["roles_map"]["tool"]
userness = W[u] - W.mean(axis=0)
userness_minus_toolness = W[u] - W[t]
```

If sklearn 1.7.x is installed, a plain `pickle.load` also works and `p["probe"].coef_` is the
same array. Older or newer sklearn versions load with an `InconsistentVersionWarning` because
only `coef_`, `intercept_`, `classes_`, `n_features_in_` are needed for `predict_proba`.

Multi-class probability is softmax over `W @ x + b` (sklearn multinomial). Binary probability
is sigmoid of `W[0] @ x + b[0]` for class 1.

The feature the probe expects is the output of `post_attention_layernorm` in the authors'
custom forward, that is after the attention residual add and before the MoE block
(`utils/pretrained_models/gptoss.py:62` in the authors' clone). The consumer in this repo,
`src/local.py` `load_vectors()` and `project_tool_tokens()`, hooks exactly that module.

## 4. File inventory

| Path | What | Size |
|---|---|---|
| `README.md` | Three-run description: OpenRouter attack, local attack with userness projection, local with userness minus toolness | 1.1 KB |
| `PROBE_TRAINING_HANDOFF.md` | Brief given to the GPU agent. Contains an export snippet with two bugs the report corrects (cupy `.get()`, binary `classes_`). | 8 KB |
| `deliverables/REPORT.md` | Training report quoted above | 18 KB |
| `deliverables/02-train-role-probes-EXECUTED.ipynb` | Executed notebook, 22 cells | 307 KB |
| `deliverables/run_nb.log` | Console log incl. 77 cuML L-BFGS warnings with timestamps | 39 KB |
| `deliverables/role_probes.pkl`, `data/role_probes.pkl` | Portable probes (identical) | 7.5 MB each |
| `deliverables/acc_by_role_gptoss-20b.csv` | Test confusion counts per role space x layer | 73 KB, 1,245 rows |
| `deliverables/acc_by_pos_gptoss-20b.csv` | Test accuracy per token position | 5.1 MB, 98,304 rows |
| `deliverables/SHA256SUMS.txt` | Four hashes, including the unshipped `gptoss-20b.pkl` | |
| `data/pages.jsonl` | 912 rows = 456 Wikipedia articles x {clean, base-injection}. Keys `id`, `idx`, `variant`, `title`, `size_kb`, `html`. Injection text comes from the vendor `injection.yaml`. Gitignored but present locally. | 7.3 MB |
| `data/build_pages.py` | Builds `pages.jsonl` from random Wikipedia pages (5 to 10 KB after cleaning) | 4.6 KB |
| `src/eval.py` | Inspect task: bash tool, curl of a locally served page, dpaste.com sinkholed, judge from vendor `classify-output.yaml` | 8.7 KB |
| `src/local.py` | `steered-hf` Inspect provider. Projects a probe direction out of Harmony tool-call and tool-result tokens at the pre-MLP hook of chosen layers, prefill only. | 11 KB |
| `src/run.py` | Four-cell runner (clean/injected x off/project), `KEY_LAYERS = (12, 16)`, `--layer-sweep` | 9.9 KB |
| `src/pairwise.py`, `src/notebook_stats.py`, `src/report.py`, `src/export_summaries.py`, `src/progress.py` | Judge, stats, reporting helpers | |
| `notebooks/results.py`, `notebooks/demo.py` | marimo notebooks for the n=48 paired results | 14 KB, 10 KB |
| `notebooks/results.html` | Exported marimo page titled "Role-confusion eval" | 92 KB |
| `tests/test_so_far.py` | Smoke tests for cell matching and pairing | 5 KB |
| `scripts/next_proj_cell.sh` | Pod-side resume script for the local injected-off cell (idx 0 to 47) | 1 KB |
| `logs/*.eval` (12 files, 2026-09-03) | Inspect logs from OpenRouter runs, zip archives with per-sample JSON | 22 KB to 1.4 MB |
| `logs/four_cell/{clean_off,injected_off}/`, `logs/local_off/injected_off/`, `logs/proj/L12/{clean_project,injected_project}/`, `logs/proj/L12plus/{clean_project,injected_project}/`, `logs/pairwise_dirty/` | Inspect logs for the projection experiments (2026-09-04). Nine files. No `L16` directories are present despite `run.py` naming them. | |
| `vendor/role-confusion/` | Empty submodule directory | 0 files |
| `pyproject.toml`, `uv.lock`, `.env.example`, `.gitignore`, `.gitmodules` | Env | |

No plots (PNG, SVG, PDF) are shipped. No conversation-projection CSVs. No `probe_states.pt`.
Conversation data exists only in the form of agent transcripts inside the `.eval` archives.

## 5. Caveats

1. **The layer-16 peak is visible in table 2.3, with one tie.** At 2 dp every column's maximum is at layer 16. For `system,user,assistant` layers 16 and 18 both print 0.88. At full precision (table 2.5) layer 16 is 0.8830 and layer 18 is 0.8792, so 16 still wins. The report's sentence "Every role space peaks at layer 16" holds, but for that one column only at the third decimal.
2. **Layer 12 is not the peak.** The authors' notebook uses the mid even layer as `TEST_LAYER_IX`. Here 5-role accuracy is 0.595 at layer 12 versus 0.725 at layer 16, and the binary probe is 0.88 versus 0.97. The paper's conversation-projection and tomato figures at layer 12 cannot be checked against this repo because those cells were not run.
3. **Not the paper's numbers.** The report says to cite these as a reproduction. Reasons: the C4 sample is unpinned, cuML 25.08 replaced a nonexistent 25.9 pin, the FA3 kernel is unpinned, and 77 of 96 fits ended on an L-BFGS line-search warning (54 "line search failed (code 1)", 23 "failed to advance"). The five-role probes hit the warning at all 12 layers.
4. **The shipped pickle is the sklearn clone, not the cuML artifact.** `gptoss-20b.pkl` is hashed in `SHA256SUMS.txt` but absent from the repo. Coefficients in the clone are float64 casts of cuML's float32 values; probabilities agree to 1.9e-6.
5. **Per-role weakness in the 5-role space.** From the derived table, `cot` and `tool` are the confusable pair (0.60 and 0.55 at layer 16, versus 0.93 user and 0.90 assistant). The layer-16 confusion matrix shows 4,195 cot tokens predicted as tool and 3,859 tool tokens predicted as cot.
6. **Test set is by document, not by token.** Ten percent of `prompt_ix` values are held out with seed 123, run separately per role space on that space's own `prompt_ix` values. For the 5-role space that is about 125 of 1,245 sequences. Test token counts per role therefore differ by role space (see the counts table), and the held-out documents are not the same set across role spaces.
7. **Report versus notebook discrepancies.** The report says 143 GB and the notebook prints 139.80 GB total (different accounting of the same H200 NVL). The report's `max_seqlen 1035` is not printed anywhere in the notebook or log. The report's numpy version is stated as 2.2.6 in section 3.6 and as 2.5.2 in section 7.
8. **Vendor submodule is empty locally.** Any path in this repo that reads `vendor/role-confusion/...` (injection YAML, judge prompt) would fail here until the submodule is initialised. Our workspace already holds the same commit at `prompt-injection-as-role-confusion/`.
