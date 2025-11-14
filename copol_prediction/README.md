# Copolymerization Prediction

ML system for predicting copolymerization reactivity ratios from molecular descriptors.

## Overview

Predicts r-product (r₁ × r₂) class:
- **Class 0**: < 1 (Alternating)
- **Class 1**: 1-25 (Random to block-like)
- **Class 2**: > 25 (Homopolymer)

## Quick Start

### Installation
```bash
pip install pandas numpy scikit-learn xgboost joblib morfeus-ml
# For API: pip install fastapi uvicorn
```

### Setup (First Time)
```bash
# 1. Create central train/test split
cd ../experiments && python create_data_split.py

# 2. Calculate molecular features (cached, ~1-5 min/monomer)
cd ../copol_prediction && python monomer_feature_calculation.py
```

### Training
```bash
# Train final model (~20 min, includes automatic analysis)
python train_final_model.py

# Or test all filter combinations (~3 hours)
cd ../experiments && python sweep_filters.py
```

### Prediction
```python
from copolpredictor.inference import CopolymerPredictor

predictor = CopolymerPredictor("artifacts/model_bundle")
result = predictor.predict_with_confidence(features)
```

Or via REST API:
```bash
python api.py  # http://localhost:8000/docs
```

## Central Data Split

All scripts use a **central train/test split** (created once, reused everywhere):

```bash
cd ../experiments && python create_data_split.py [--remove-specialized]
```

Creates:
- `artifacts/data_splits/train.csv` (~80% of groups)
- `artifacts/data_splits/test.csv` (~20% of groups)
- `artifacts/data_splits/split_info.json`

**Benefits:** Reproducible, fair comparison, no data leakage (group-based split by `reaction_id`)

**Usage in code:**
```python
from copol_prediction import load_data_split
df_train, df_test = load_data_split.load_train_test_split()
```

## Scripts

| Script | Purpose | Time |
|--------|---------|------|
| `train_final_model.py` | Train production model + analysis | ~20 min |
| `analysis/analyze_model.py` | Generate analysis plots | < 1 min |
| `../experiments/sweep_filters.py` | Test 16 filter combinations | ~3 hours |
| `../experiments/create_data_split.py` | Create central split | < 1 min |
| `monomer_feature_calculation.py` | Calculate molecular features | 1-5 min/monomer |
| `api.py` | REST API server | Instant |

### train_final_model.py

Trains model with hyperparameter optimization and **automatically runs analysis**.

```bash
python train_final_model.py [options]

Options:
  --output-dir DIR         Model directory (default: artifacts/model_bundle)
  --hyperparam-iter N      Search iterations (default: 25)
  --augmentation-samples N Augmentation samples (default: 5)
  --random-state N         Random seed (default: 42)
```

**Configuration** (edit lines 371-373 in file):
```python
config = {
    'add_negative_data': True,    # Add synthetic negatives
    'use_augmentation': False,    # Gaussian augmentation
}
```

### analysis/analyze_model.py

Generate analysis plots (automatically runs after training).

```bash
python analysis/analyze_model.py --all [--compare-holdout]

Key options:
  --compare-holdout        Generate plots for all data + holdout
  --holdout-only          Only holdout set
  --filtering             Dynamic confidence filtering
  --min-retention N       Min retention rate (default: 0.7)
```

**Generated plots:**
- Confusion matrices (absolute & normalized)
- Confidence distributions (correct vs incorrect)
- Feature importance
- Calibration curves per class
- Error analysis by class
- Confidence vs r-product
- Confidence filtering analysis

### sweep_filters.py

Tests all 16 filter combinations (4×4 matrix) on same holdout set.

```bash
cd ../experiments && python sweep_filters.py [--n-iter N]
```

**Combinations tested:**
- Rows: `remove_specialized` × `add_negative_data` (4 combos)
- Cols: `use_augmentation` × `apply_polymerization_filter` (4 combos)

Results saved to `artifacts/experiments_holdout/` with heatmap visualizations.

## Python API

```python
from copolpredictor.inference import CopolymerPredictor, batch_predict

# Single prediction
predictor = CopolymerPredictor("artifacts/model_bundle")
result = predictor.predict_with_confidence(features)
# Returns: {'predictions': [1], 'probabilities': [...], 'confidence': [0.85]}

# Batch prediction
batch_predict("input.csv", "output.csv")
```

## REST API

```bash
python api.py  # Runs at http://localhost:8000
```

**Endpoints:**
- `GET /health` - Health check
- `GET /model/info` - Model metadata
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {...}}'
```

## Data Format

**Required columns:**
- `monomer1_smiles`, `monomer2_smiles` - SMILES strings
- `constant_1`, `constant_2` - Reactivity ratios (r₁, r₂)
- `temperature`, `solvent_smiles`, `polymerization_type`, `method`
- `reaction_id` - Unique group identifier

## Features

~15 features used:
- **Molecular:** Fukui indices, HOMO/LUMO, orbital interactions
- **Conditions:** Temperature, solvent properties (logP, TPSA, HBD, FractionCSP3)
- **Embeddings:** Method and polymerization type (PCA-reduced)

## Model Pipeline

1. Load central train/test split (group-based, ~20% test)
2. Optional: Add negative data, augmentation
3. Hyperparameter search (RandomizedSearchCV, 5-fold GroupKFold)
4. Train final model on full training set
5. Evaluate on holdout
6. Save model bundle + metadata
7. Generate analysis plots

## Performance

Typical holdout results:
- **Accuracy:** 75-85%
- **F1 (weighted):** 0.75-0.85

**Confidence interpretation:**
- \> 0.8: High confidence
- 0.6-0.8: Medium confidence
- < 0.6: Low confidence (validate experimentally)

## Project Structure

```
copol_prediction/
├── train_final_model.py       # Main training script
├── api.py                      # REST API
├── load_data_split.py          # Load central split
├── monomer_feature_calculation.py
├── analysis/
│   ├── analyze_model.py        # Analysis plots
│   ├── plot_config.py          # Plot styling
│   ├── error_analysis.py
│   └── permutation_analysis.py
├── artifacts/
│   ├── data_splits/            # Central train/test split
│   ├── model_bundle/           # Trained model
│   └── experiments_holdout/    # Sweep results
└── output/
    ├── analysis/               # Analysis plots
    └── processed_data.csv

src/copolpredictor/             # Core library
├── data_processing.py
├── model_training.py
├── evaluation.py
├── inference.py                # CopolymerPredictor
└── ...
```

## Modules (src/copolpredictor/)

| Module | Purpose |
|--------|---------|
| `data_processing.py` | Data loading & preprocessing |
| `data_augmentation.py` | Gaussian augmentation |
| `model_training.py` | Training, CV, model saving |
| `evaluation.py` | Metrics & evaluation |
| `calibration.py` | Model calibration |
| `holdout_utils.py` | Holdout set management |
| `inference.py` | CopolymerPredictor class |
| `prediction_utils.py` | Feature definitions |

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | `python train_final_model.py` |
| Missing features | `python monomer_feature_calculation.py` |
| No train/test split | `cd ../experiments && python create_data_split.py` |
| API port in use | `lsof -ti:8000 \| xargs kill` |
| Quick test | `python train_final_model.py --hyperparam-iter 5` |

## Common Commands

```bash
# Quick test (fewer iterations)
python train_final_model.py --hyperparam-iter 5

# Manual analysis (if needed)
python analysis/analyze_model.py --all --compare-holdout

# Recreate data split
cd ../experiments && python create_data_split.py

# Kill API
lsof -ti:8000 | xargs kill

# Run specific analysis
python analysis/analyze_model.py --confusion --confidence --features
```

## Notes

- **Plot Styling:** All plots use LamaLab matplotlib style from `plots_and_figures/lamalab.mplstyle`
- **Confidence Filtering:** Dynamic thresholding per class to improve accuracy
- **Reproducibility:** Fixed random seed (42), central split ensures consistency
- **Legacy:** Old `classification.py` kept for reference, use new modular scripts

## Migration from classification.py

Old monolithic script → New modular system:
- `classification.py::main()` → `train_final_model.py`
- `classification.py::sweep_filters_and_plot()` → `experiments/sweep_filters.py`
- Manual model loading → `CopolymerPredictor` class

All functionality preserved in new modules under `src/copolpredictor/`.
