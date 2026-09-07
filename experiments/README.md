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

```
experiments/
├── feature_comparison/         # Compare different molecular features
│   ├── data/                  # Morgan fingerprint train/test data
│   ├── fingerprint/           # Morgan fingerprint feature processing
│   ├── results/               # Metrics, plots
│   └── run_comparison.py      # Trains + compares both variants
├── filter_comparison/         # Compare data filtering strategies
│   └── sweep_filters.py
├── reaction_conditions_comparison/  # Compare with/without reaction conditions
│   ├── results/               # Metrics, plots
│   └── run_comparison.py      # Trains + compares both variants
├── baseline/                   # Database-lookup baseline vs full model
├── case_studies/                # Lab experiments, solvent & negative-data case studies
├── permutation_importance/      # SHAP / permutation feature importance
├── archive/                    # Old/deprecated scripts
└── run_all.sh                  # Run all experiments
```

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

This creates `feature_comparison/data/train_morgan.csv`, `feature_comparison/data/test_morgan.csv` (derived data only).

**Note:** Normal splits (`train.csv`, `test.csv`) are NOT copied anymore.
All scripts should use the central split directly from `copol_prediction/artifacts/data_splits/`

### 3. Run Experiments

**Option A: Run all experiments**
```bash
./run_all.sh
```

**Option B: Run specific experiments**

Feature comparison (quantum-chemical descriptors vs Morgan fingerprints):
```bash
cd feature_comparison && python run_comparison.py
```

Filter comparison:
```bash
cd filter_comparison && python sweep_filters.py
```

Reaction conditions comparison (with vs without reaction condition features):
```bash
cd reaction_conditions_comparison && python run_comparison.py
```

## 📊 Experiments

### Feature Comparison

**Goal**: Compare different molecular feature representations

- **Baseline**: 15 quantum chemical descriptors (Fukui indices, HOMO-LUMO gaps)
- **Morgan Fingerprint**: 2048-bit Morgan fingerprints + other features

Both variants use the same voting model (XGBoost + Tanimoto-similarity lookup);
`run_comparison.py` trains and compares them in one go.

**Results**: See `feature_comparison/results/`

### Filter Comparison

**Goal**: Evaluate impact of different data filtering strategies

- No filtering (baseline)
- Polymer type filtering
- Method filtering
- Combined filters

**Results**: See `filter_comparison/output/`

### Reaction Conditions Comparison

**Goal**: Compare model performance with and without reaction condition features

- **Full Model**: All features including reaction conditions (temperature, polytype embeddings, method embeddings, solvent properties)
- **No Reaction Conditions**: Model trained without reaction condition features (only molecular descriptors and HOMO-LUMO differences)

**Excluded features**: `temperature`, `polytype_emb_1`, `polytype_emb_2`, `method_emb_1`, `method_emb_2`, `solvent_logP`, `solvent_TPSA`, `solvent_HBD`, `solvent_FractionCSP3`

Both variants use the same voting model; `run_comparison.py` trains and
compares them in one go.

**Results**: Plots saved to `reaction_conditions_comparison/results/`

### Other Experiments

- **`baseline/`**: Compares a pure database-lookup baseline (Tanimoto similarity)
  against the full model and a model trained using only the baseline
  prediction as a feature. See `baseline/README.md`.
- **`case_studies/`**: Lab-experiment validation, solvent case study, and
  negative-data case study.
- **`permutation_importance/`**: SHAP-based and permutation-based feature
  importance analysis.

## 📝 Notes

- All experiments use the **same train/test splits** for fair comparison
- Models are trained with XGBoost + 5-fold cross-validation
- Hyperparameters are tuned with Optuna (50 trials)
- Results include confusion matrices, per-class metrics, and macro metrics

## 🗂 Archive

The `archive/` directory contains old scripts kept for reference but not part of the current workflow.
