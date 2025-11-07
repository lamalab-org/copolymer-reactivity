## LLM Comparison Archive

This folder contains historic evaluation runs for the copolymer extraction
pipeline. The content is grouped by purpose to make it easier to locate scripts,
inputs and results.

- `scripts/`: launchers for the different LLM providers that were benchmarked.
- `data/`: static inputs used during the experiments, including the PDF test
  set and OCR artefacts.
- `artifacts/`: auxiliary files produced during the runs (SQLite caches,
  rendered page images, etc.).
- `analysis/`: follow-up evaluation tooling such as the `model_evaluation`
  notebook/script and its wandb outputs.
- `results/`: raw model outputs, separated by provider (`baseline`,
  `assistant`, `claude`, `gpt4o`, `safe`) plus aggregated exports.
- `reports/`: high-level summaries and logs collected after the experiments.

All content is kept read-only for provenance; contemporary pipelines should rely
on the material in `data_extraction/artifacts` instead.

