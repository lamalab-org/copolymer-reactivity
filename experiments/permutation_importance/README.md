# Permutation Feature Importance Analysis

This experiment performs permutation feature importance analysis using `feature_columns_2` from `prediction_utils`.

## Overview

The experiment:
1. Loads the global train/test split from `experiments/data/`
2. Trains an XGBoost model using `feature_columns_2` features
3. Performs permutation importance analysis on the test set
4. Visualizes results as bar plots

## Usage

```bash
cd experiments/permutation_importance
python train.py [options]
```

### Options

- `--output-dir`: Output directory (default: `results`)
- `--random-state`: Random seed (default: 42)
- `--hyperparam-iter`: Number of hyperparameter search iterations (default: 25)
- `--n-repeats`: Number of permutation repeats (default: 10)
- `--scoring`: Scoring metric for permutation importance (default: `f1_macro`)
- `--top-n`: Number of top features to plot (default: 30)

### Example

```bash
python train.py --output-dir results --n-repeats 20 --top-n 40
```

## Output

The experiment generates:
- `permutation_importance_detailed.csv`: Detailed permutation importance results for all features
- `permutation_importance_barplot.png`: Bar plot visualization of top N features
- `meta.json`: Experiment metadata and summary
- `model.joblib`: Trained model

## Features

This experiment uses `feature_columns_2` which includes:
- Extended molecular descriptors (IP, EA, HOMO, LUMO, Fukui indices, etc.)
- HOMO-LUMO differences
- Temperature and solvent properties
- Polymerization type and method embeddings

