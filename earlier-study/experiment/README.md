# Experiment source

I preserve the source snapshots used for the two runs in `first-20260904/source/` and `permission-20260905/source/`. These files show how I constructed prompts, trained the local classifier, applied steering, executed actions in a sandbox, and scored outcomes. The snapshots are unchanged; their hashes are in `../data/export-manifest.json`.

Each run also has a scientific `config.json` export and `model-metadata.json`. I removed the preparation corpus and runtime paths as listed in the export manifest. The original input hashes, dataset revisions, generation parameters, model revision and saved diagnostic values remain available. In particular, `role_positions` in the first run's metadata records the native contexts and token positions used for the local probe.

`model-assets.npz` contains the saved linear-classifier weights and bias, steering directions, reference scale and random controls. Both runs used these same bytes. It contains no language-model weights.

These historical source files are for inspection. They require GPU inference and isolated external sandboxes to execute; the complete neutral-text preparation corpus is not included. The commands in the top-level README reproduce analysis and figures from saved records on CPU.
