# `experiments/` — studies orbiting the main pipeline

Self-contained studies that don't belong in `copol_prediction/`'s production path. Some produce figures in the paper; others are exploratory and kept for provenance.

| Subdir | What it does | In paper? |
|---|---|---|
| [`permutation_importance/`](permutation_importance/) | XGBoost permutation importance + per-class SHAP analysis | yes — `fig:permutation_analysis`, `fig:per_class_shap` |
| [`baseline/`](baseline/) | Train comparisons without conf-filter / negative augmentation | yes — `fig:system_comp` |
| [`reaction_conditions_comparison/`](reaction_conditions_comparison/) | Ablation: model with vs without reaction-condition features (the headline result behind the *condition-aware* claim) | yes — Sankey + the ablation numbers in `sec:results` |
| [`case_studies/solvent/`](case_studies/solvent/) | The Minsk 1973 nine-solvent case study | yes — `fig:solvent_case_study` |
| [`case_studies/lab_experiments/`](case_studies/lab_experiments/) | Prospective lab copolymerisations (GC, NMR, SEC) | yes — `fig:lab_exp`, `fig:GC_*`, `fig:NMR_*`, `fig:SEC_*`, `tab:gc-retention-times`, `tab:lab-experiments-summary` |
| [`feature_comparison/`](feature_comparison/) | Quantum descriptors vs. Morgan fingerprints | exploratory |
| [`filter_comparison/`](filter_comparison/) | Sweep over data-filter combinations | exploratory |
| [`case_studies/negative_data/`](case_studies/negative_data/) | logP-based negative-data baseline | exploratory |
| [`archive/`](archive/) | Superseded scripts kept for provenance | — |

## How to run a study

Every experiment is self-contained — `cd` into its directory, `python` the runner. Each has its own short README with the inputs it expects (almost always the central `copol_prediction/artifacts/data_splits/`) and the outputs it writes (`results/` next to the runner).

```bash
# Permutation importance + SHAP
python experiments/permutation_importance/run_permutation_importance.py

# Reaction-conditions ablation
python experiments/reaction_conditions_comparison/run_comparison.py

# Baseline-vs-released comparison
python experiments/baseline/train_baseline_feature.py
python experiments/baseline/compare_models.py
python experiments/baseline/plot_no_filter_train_val_performance.py

# Solvent case study (Minsk 1973)
python experiments/case_studies/solvent/solvent_case_study.py

# Lab-experiment plots + tables
python experiments/case_studies/lab_experiments/plot_lab_experiments_timeseries.py
python experiments/case_studies/lab_experiments/plot_gc_chromatograms.py
python experiments/case_studies/lab_experiments/plot_nmr_spectra.py
python experiments/case_studies/lab_experiments/plot_sec_curves.py
python experiments/case_studies/lab_experiments/make_latex_gc_table.py
python experiments/case_studies/lab_experiments/make_latex_analysis_table.py
```

## Shared assumptions

- All experiments read the central train/val/test splits from `copol_prediction/artifacts/data_splits/`. Recut them with `python copol_prediction/create_data_split.py` if `paper_dataset/processed_data.csv` changes.
- Most use 5-fold cross-validation with Optuna for hyper-parameter search.
- Plots use the LamaLab matplotlib style from `copol_prediction/analysis/lamalab.mplstyle`.

## `archive/`

Scripts here are superseded by the current pipeline (the canonical split generator is `copol_prediction/create_data_split.py`, the canonical feature-prep is `copol_prediction/preprocess_splits_full_features.py`, etc.). Kept for provenance; don't run them.
