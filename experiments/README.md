# Experiments

Systematic experiments for copolymer microstructure prediction model development and validation.

## 📁 Structure

```
experiments/
├── feature_comparison/         # Compare different molecular features
│   ├── baseline/              # Quantum chemical descriptors
│   ├── fingerprint/           # Morgan fingerprints
│   └── comparison/            # Analysis and plots
├── filter_comparison/         # Compare data filtering strategies
│   └── sweep_filters.py
├── data/                      # Shared train/test splits
├── archive/                   # Old/deprecated scripts
├── create_train_test_split.py # Split generation script
└── run_all.sh                 # Run all experiments
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

### 2. Create Experiment-Specific Data

```bash
# Copy baseline data and create Morgan fingerprint version
python create_train_test_split.py --fingerprints
```

This creates `data/train.csv`, `data/test.csv`, `data/train_morgan.csv`, `data/test_morgan.csv`

### 3. Run Experiments

**Option A: Run all experiments**
```bash
./run_all.sh
```

**Option B: Run specific experiments**

Feature comparison:
```bash
cd feature_comparison/baseline && python train.py
cd ../fingerprint && python train.py
cd ../comparison && python compare.py
```

Filter comparison:
```bash
cd filter_comparison && python sweep_filters.py
```

## 📊 Experiments

### Feature Comparison

**Goal**: Compare different molecular feature representations

- **Baseline**: 15 quantum chemical descriptors (Fukui indices, HOMO-LUMO gaps)
- **Morgan Fingerprint**: 2048-bit Morgan fingerprints + other features

**Results**: See `feature_comparison/README.md` and plots in `feature_comparison/comparison/plots/`

### Filter Comparison

**Goal**: Evaluate impact of different data filtering strategies

- No filtering (baseline)
- Polymer type filtering
- Method filtering
- Combined filters

**Results**: See `filter_comparison/README.md`

## 📝 Notes

- All experiments use the **same train/test splits** for fair comparison
- Models are trained with XGBoost + 5-fold cross-validation
- Hyperparameters are tuned with Optuna (50 trials)
- Results include confusion matrices, per-class metrics, and macro metrics

## 🗂 Archive

The `archive/` directory contains old scripts kept for reference but not part of the current workflow.
