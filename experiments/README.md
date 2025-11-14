# Experiments

Compare different monomer representations for copolymerization prediction.

## Setup (First Time)

### 1. Create central train/test split

```bash
cd ../copol_prediction

# Default: keep all datapoints
python create_data_split.py

# Or: remove specialized datapoints
python create_data_split.py --remove-specialized

cd ../experiments
```

This creates the central split used by **all** scripts:
- `copol_prediction/artifacts/data_splits/train.csv`
- `copol_prediction/artifacts/data_splits/test.csv`

**Note**: Use `--remove-specialized` to exclude specialized reactions from the **test set** (training keeps all reactions).

### 2. Create experiment-specific data

```bash
# Copy baseline data and create Morgan fingerprint version
python create_train_test_split.py --fingerprints
```

This creates:
- `data/train.csv`, `data/test.csv` (copies of central split)
- `data/train_morgan.csv`, `data/test_morgan.csv` (with Morgan fingerprints)

## Structure

```
experiments/
├── baseline/              # Quantum chemical features
├── fingerprint/           # Morgan fingerprints
├── data/                  # Train/test splits (created by setup)
├── create_train_test_split.py
└── compare_results.py
```

## Usage

Run all experiments:
```bash
./run_all.sh
```

Or run individually:
```bash
cd baseline && python train.py
cd fingerprint && python train.py
python compare_results.py
```

## Results

Results in each experiment's `results/` folder:
- `results.json` - Metrics and metadata
- `model.joblib` - Trained model

Comparison: `comparison.csv`

