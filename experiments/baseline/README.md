# Baseline Feature Experiment

This experiment compares three approaches:
1. **Baseline (Database Lookup)**: Direct predictions using Tanimoto similarity to find nearest training point
2. **Base Model**: Normal final model with all features
3. **Baseline-Feature Model**: Model trained using only baseline predictions as features (one-hot encoded)

## Usage

### 1. Train Baseline-Feature Model

```bash
cd experiments/baseline
python train_baseline_feature.py --output-dir results
```

This will:
- Compute baseline predictions for training set (using leave-one-out)
- Compute baseline predictions for test set
- Train an XGBoost model using only baseline predictions (one-hot encoded) as features
- Use the same hyperparameters and training procedure as the final model

### 2. Compare All Models

```bash
python compare_models.py \
    --base-model-path ../../copol_prediction/artifacts/model_bundle \
    --baseline-feature-model-path results \
    --output-dir comparison
```

This will:
- Load predictions from all three models
- Calculate metrics (accuracy, precision, recall, F1) for each
- Create comparison plots showing:
  - Per-class accuracy
  - Macro accuracy
  - Per-class precision
  - Per-class recall

## Results

The comparison script generates:
- `comparison/model_comparison.png` - Comparison plots
- `comparison/model_comparison.pdf` - PDF version
- Console output with detailed metrics table

## Notes

- The baseline-feature model uses only 3 features (one-hot encoded baseline predictions)
- Training uses the same hyperparameter search space as the final model
- All models are evaluated on the same test set from `copol_prediction/artifacts/data_splits/`
