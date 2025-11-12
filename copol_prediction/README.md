# Copolymerization Prediction

ML system for predicting copolymerization reactivity ratios from molecular descriptors.

## Overview

Predicts r-product (r₁ × r₂) class:
- **Class 0**: < 1 (Alternating)
- **Class 1**: 1-25 (Random to weak block)
- **Class 2**: > 25 (Strong block)

## Quick Start

```bash
# 1. Calculate molecular features (one-time)
python monomer_feature_calculation.py

# 2. Train model
python train_final_model.py

# 3. Make predictions
python api.py  # Start REST API
# or
from copolpredictor.inference import CopolymerPredictor
predictor = CopolymerPredictor("artifacts/model_bundle")
result = predictor.predict_with_confidence(features)
```

## Scripts

| Script | Purpose | Time |
|--------|---------|------|
| `monomer_feature_calculation.py` | Calculate molecular features | 1-5 min/monomer |
| `train_final_model.py` | Train production model | ~20 min |
| `sweep_filters.py` | Test filter combinations | ~3 hours |
| `analyze_model.py` | Generate analysis plots | < 1 min |
| `api.py` | REST API for predictions | Instant |

### train_final_model.py

Train production model with hyperparameter optimization.

```bash
python train_final_model.py [options]

Options:
  --data-path PATH         Input CSV (default: ../data_extraction/extracted_reactions.csv)
  --output-dir DIR         Model directory (default: artifacts/model_bundle)
  --hyperparam-iter N      Search iterations (default: 25)
  --augmentation-samples N Augmentation samples (default: 5)
  --random-state N         Random seed (default: 42)
```

### sweep_filters.py

Test all 16 filter combinations (4×4 matrix) on the same holdout set.

```bash
python sweep_filters.py [options]

Options:
  --n-iter N               Iterations per config (default: 10)
  --output-dir DIR         Results directory
  --plots-dir DIR          Plots directory
  --augmentation-samples N Augmentation samples (default: 5)
```

**Filter combinations tested (4×4 = 16):**
- Rows: `remove_specialized` × `add_negative_data` (4 combos)
- Cols: `use_augmentation` × `apply_polymerization_filter` (4 combos)

**Generated visualizations:**
- `filter_matrix_holdout_f1_weighted.png` - 4×4 heatmap of F1 scores
- `filter_matrix_holdout_accuracy.png` - 4×4 heatmap of accuracy
- `filter_matrix_holdout_f1_macro.png` - 4×4 heatmap of macro F1
- Traditional bar plots for comparison

Results saved to `artifacts/experiments_holdout/` and plots to `output/model_comp/`.

**Note:** All combinations are tested on the **same holdout set** for fair comparison.

### analyze_model.py

Generate analysis plots for trained models.

```bash
python analyze_model.py [options]

Options:
  --model-path PATH     Model bundle path (default: artifacts/model_bundle)
  --data-path PATH      Data CSV path (default: output/processed_data.csv)
  --output-dir DIR      Output directory (default: output/analysis)
  --holdout-only        Use only holdout set
  --compare-holdout     Generate plots for both all data and holdout set
  
  # Plot selection (omit to generate all)
  --all                 Generate all plots
  --confusion           Confusion matrix
  --confidence          Confidence distribution
  --features            Feature importance
  --calibration         Calibration curves
  --errors              Error analysis by class
  --confidence-vs-r1r2  Confidence vs r-product
  --filtering           Dynamic confidence filtering analysis
  --min-retention N     Minimum retention rate (default: 0.7)
```

**Generated plots:**
- `confusion_matrix.png` - Confusion matrix (absolute & normalized)
- `confidence_distribution.png` - Confidence score distributions
- `feature_importance.png` - Top feature importances
- `calibration_curves.png` - Calibration curves per class
- `error_analysis_by_class.png` - Error analysis breakdown
- `confidence_vs_r1r2.png` - Confidence vs r-product value
- `confidence_filtering_analysis.png` - Dynamic filtering analysis (4 subplots)
- `confidence_filtering_report.txt` - Detailed filtering report

**Plot Styling:** All plots use consistent colors and styling from `plot_config.py`, which loads the LamaLab matplotlib style (`plots_and_figures/lamalab.mplstyle`). Customize colors by editing `plot_config.py`.

**Examples:**
```bash
# All plots (all data)
python analyze_model.py --all

# Compare all data vs holdout set
python analyze_model.py --all --compare-holdout

# Only holdout set
python analyze_model.py --all --holdout-only

# Specific plots
python analyze_model.py --confusion --confidence --features

# With confidence filtering (keeps min 70% per class, removes bad high-conf predictions)
python analyze_model.py --all --min-retention 0.7

# Compare holdout with filtering
python analyze_model.py --compare-holdout --filtering --min-retention 0.8
```

**Confidence Filtering:**  
Dynamically finds optimal confidence threshold per class:
- Keeps minimum 70% of predictions per class (adjustable)
- Removes incorrect predictions with high confidence if they outnumber correct ones
- Generates detailed report showing accuracy improvement after filtering

### api.py

REST API for predictions.

```bash
python api.py
# API runs at http://localhost:8000
# Docs at http://localhost:8000/docs
```

**Endpoints:**
- `GET /health` - Health check
- `GET /model/info` - Model info
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {...}}'
```

## Python API

```python
from copolpredictor.inference import CopolymerPredictor

# Load model
predictor = CopolymerPredictor("artifacts/model_bundle")

# Single prediction
result = predictor.predict_with_confidence(features)
# Returns: {'predictions': [1], 'probabilities': [...], 'confidence': [0.85]}

# Batch prediction
from copolpredictor.inference import batch_predict
batch_predict("input.csv", "output.csv")
```

## Configuration

Edit `train_final_model.py` (lines 402-407):

```python
config = {
    'remove_specialized': False,   # Remove specialized reactions
    'add_negative_data': True,     # Add synthetic negatives
    'use_augmentation': False,     # Gaussian augmentation
}
```

## Data Format

Required columns:
- `monomer1_smiles`, `monomer2_smiles` - SMILES strings
- `constant_1`, `constant_2` - Reactivity ratios
- `temperature`, `solvent_smiles`, `polymerization_type`, `method`
- `reaction_id` - Unique identifier

## Features

~15 features used:
- Molecular: Fukui indices, HOMO/LUMO, orbital interactions
- Conditions: Temperature, solvent properties, method (embedded)

## Modules

```
src/copolpredictor/
├── data_processing.py    # Data loading & preprocessing
├── data_augmentation.py  # Data augmentation
├── model_training.py     # Training & saving
├── evaluation.py         # Evaluation & metrics
├── calibration.py        # Model calibration
├── holdout_utils.py      # Holdout management
└── inference.py          # Prediction API
```

**Key functions:**
- `train_xgboost_with_cv()` - Train with CV
- `save_model_bundle()` / `load_model_bundle()` - Model I/O
- `evaluate_model()` - Evaluation
- `CopolymerPredictor` - High-level inference

## Model Pipeline

1. Persistent holdout split (~20%, group-based by reaction_id)
2. Hyperparameter search (RandomizedSearchCV, 5-fold GroupKFold)
3. Train on full training set with best params
4. Evaluate on holdout
5. Save bundle: `model.joblib`, `meta.json`

## Performance

Typical holdout results:
- Accuracy: 75-85%
- F1 (weighted): 0.75-0.85

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `python train_final_model.py` |
| Missing features | Run `python monomer_feature_calculation.py` |
| API port in use | `lsof -ti:8000 \| xargs kill` |

## Requirements

```bash
pip install pandas numpy scikit-learn xgboost joblib

# For API
pip install fastapi uvicorn

# For features
pip install morfeus-ml
```

See `requirements_api.txt` for full API dependencies.

## Migration from classification.py

Old `classification.py` is kept for reference. Use instead:
- `train_final_model.py` → replaces `main()`
- `sweep_filters.py` → replaces `sweep_filters_and_plot()`
- `CopolymerPredictor` → replaces manual model loading

All functions preserved in new modules.
