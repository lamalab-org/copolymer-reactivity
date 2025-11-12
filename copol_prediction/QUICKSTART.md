# Quick Start

## Installation

```bash
pip install pandas numpy scikit-learn xgboost joblib
```

## Three Steps

### 1. Calculate Features

```bash
python monomer_feature_calculation.py
```

⏱️ 1-5 min/monomer (cached)

### 2. Train Model

```bash
# Single model
python train_final_model.py

# Or test all filter combinations
python sweep_filters.py
```

⏱️ 20 min (single) or 3 hours (sweep)

### 2b. Analyze Model (Optional)

```bash
# Generate all analysis plots
python analyze_model.py --all

# Compare all data vs holdout set
python analyze_model.py --all --compare-holdout

# Or specific plots
python analyze_model.py --confusion --confidence
```

⏱️ < 1 minute  
**Note:** Plots use LamaLab style (`plots_and_figures/lamalab.mplstyle`) and colors from `plot_config.py`. Use `--compare-holdout` to generate plots for both all data and holdout set.

### 3. Predict

**Python:**
```python
from copolpredictor.inference import CopolymerPredictor

predictor = CopolymerPredictor("artifacts/model_bundle")
result = predictor.predict_with_confidence(features)
```

**REST API:**
```bash
python api.py  # http://localhost:8000/docs
```

## Results

- **Class 0**: < 1 (Alternating)
- **Class 1**: 1-25 (Random/weak block)
- **Class 2**: > 25 (Strong block)

**Confidence:**
- \> 0.8: High
- 0.6-0.8: Medium
- < 0.6: Low (validate experimentally)

## Common Issues

```bash
# Model not found
python train_final_model.py

# Missing features
python monomer_feature_calculation.py

# API port in use
lsof -ti:8000 | xargs kill

# Quick test (fewer iterations)
python train_final_model.py --hyperparam-iter 5

# Filter sweep (tests all 16 combinations, ~3-5 hours)
python sweep_filters.py
# Creates 4×4 heatmaps showing all filter combinations on same holdout set
```

See [README.md](README.md) for details.
