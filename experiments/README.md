# Experiments

Systematic experiments for copolymer microstructure prediction model development and validation.

## 📁 Structure

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

## 🚀 Quick Start

### 1. Create Central Train/Test Split

First, create the central split used by all experiments:

```bash
cd ../copol_prediction
python create_data_split.py
cd ../experiments
```

This creates splits in `copol_prediction/artifacts/data_splits/`

### 2. Create Experiment-Specific Data (Optional)

```bash
# Create Morgan fingerprint version (only if needed)
python archive/create_train_test_split.py --fingerprints
```

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
