# `copol_prediction/` — Pipeline 2: reactions → model + API

This is **Pipeline 2** of the repo (see the [top-level README](../README.md) for the conceptual map). It takes the curated dataset produced by `data_extraction/`, computes per-monomer descriptors, trains the architecture-classification model, and serves predictions via a FastAPI service.

## Data flow

```
                processed_data.csv             ← dataset (4,969 measurement rows,
                  ▲                              3,791 unique reactions, 1,206 papers)
                  │
                  │ ↓ monomer_feature_calculation.py          (XTB descriptors)
                  │ ↓ create_data_split.py                    (stratified by monomer pair)
                  ▼
       artifacts/data_splits/{train,val,test}.csv
                  │
                  │ ↓ train_final_model.py                    (XGBoost + calibration)
                  ▼
       artifacts/model_bundle/                                ← the released model
                  │
                  │ ↓ api/app.py                              (FastAPI service)
                  ▼
       POST /predict, /preprocess_all, /optimize_reaction,  ← what powers
            /find_architecture_switch, /paper_metrics, ...     polycarp.cheminfo.org

       artifacts/model_bundle/  +  artifacts/data_splits/
                  │
                  │ ↓ reproduce_paper_metrics.py              ← regression check
                  ▼
       artifacts/paper_metrics.json + REPRODUCED on stdout
```

## Classes

The classifier predicts one of three architecture classes, derived from the reactivity-ratio product `r₁·r₂`:

| index | name | rule | meaning |
|---:|---|---|---|
| 0 | `alternating` | `r₁·r₂ < 1` | monomers prefer the comonomer over self-addition |
| 1 | `random` | `1 ≤ r₁·r₂ ≤ 25` | catch-all (covers azeotropic, mildly-alternating, and mildly-blocky systems) |
| 2 | `gradient` | `r₁·r₂ > 25` | monomers prefer self-addition; composition drift along the chain |

The canonical mapping lives in [`api/class_labels.py`](api/class_labels.py) and is imported everywhere the names appear — single source of truth.

## Voting model

Two architecture-prediction models cooperate:

- **XGBoost** on the XTB-derived feature vector — the classifier proper.
- **Lookup**: the architecture of the top-1 nearest-neighbour literature reaction (Tanimoto on Morgan fingerprints of the monomer pair + solvent).

The **voting** layer keeps only predictions where the two agree (the "coverage" metric in the paper). Disagreement is exposed on every `/preprocess_all` response so the web UI can flag low-confidence cases.

## Layout

```
copol_prediction/
├── processed_data.csv                   ← live dataset (the API reads this for the
│                                          nearest-neighbour lookup pool)
│
├── create_data_split.py                 ← monomer-pair-stratified 70/10/20 split
├── train_final_model.py                 ← XGBoost training + calibration
├── monomer_feature_calculation.py       ← XTB descriptor pipeline
├── mayo_lewis_classification.py         ← r1·r2 → class assignment
├── reproduce_paper_metrics.py           ← canonical regression check
├── preprocess_splits_full_features.py   ← splits with full feature set (perm. importance)
├── REPRODUCE.md                         ← reproduction recipe details
│
├── analysis/                            ← paper figures (plot_model_figure, plot_class_curves)
├── api/                                 ← FastAPI service deployed at polycarp.cheminfo.org
├── filter/                              ← curation + augmented-negatives pool
├── utils/                               ← load_data_split helpers
│
└── artifacts/
    ├── model_bundle/                    ← XGBoost model + calibration + metadata
    ├── data_splits/                     ← train/val/test (6,774 rows / 3,387 reactions)
    ├── data_splits_full_features/       ← splits with descriptors for permutation analysis
    └── paper_metrics.json               ← cached output of reproduce_paper_metrics.py
                                           served unmodified by GET /paper_metrics
```

### Setup (First Time)
```bash
# 1. Create central train/test split
python create_data_split.py

# 2. Calculate molecular features (cached, ~1-5 min/monomer)
python monomer_feature_calculation.py
```

Loads `artifacts/model_bundle/` + `artifacts/data_splits/`, evaluates plain XGBoost and the voting model on both splits, asserts every cell of the paper's table reproduces within ±0.005. Exits non-zero on drift. Also the basis of `tests/test_api_parity.py::test_paper_metrics_endpoint`.

### Re-split the dataset

```bash
python create_data_split.py
```

Reads `processed_data.csv`, applies the paper filter (`r₁·r₂ ≥ 0`, `r₁·r₂` not null, drop rows with NaN features), stratifies by **monomer pair** (`frozenset({canon(m1), canon(m2)})`) so all rows for a given pair land in the same split (prevents leakage), writes `artifacts/data_splits/{train,val,test}.csv` + `split_info.json`.

### Retrain the model

```bash
python train_final_model.py

# Or test all filter combinations (~3 hours)
cd ../experiments/filter_comparison && python sweep_filters.py
```

Reads the splits, runs `RandomizedSearchCV` over XGBoost hyper-parameters (5-fold GroupKFold by `monomer_pair_key`), fits the final model on train+val, calibrates on a held-out subset, writes the new `artifacts/model_bundle/`. Run `python reproduce_paper_metrics.py` afterwards to confirm the new bundle matches (or to capture the new numbers if you intend to update the paper).

### Compute monomer descriptors

```bash
python monomer_feature_calculation.py
```

Runs XTB for each unique monomer SMILES that doesn't yet have a cached descriptor file under [`api/molecule_properties/`](api/molecule_properties/). First-time computation per monomer takes ~1–5 min; subsequent runs are cached.

### Serve the API locally

See [`api/README.md`](api/README.md). One-liner with Docker:

```bash
cd api && docker compose up
```

## Library entry point

```python
from copolpredictor.inference import CopolymerPredictor

predictor = CopolymerPredictor("artifacts/model_bundle")
result = predictor.predict_with_confidence(features)
```

See [`src/copolpredictor/`](../src/copolpredictor/) for module docs.

## Central Data Split

All scripts use a **central train/test split** (created once, reused everywhere):

```bash
python create_data_split.py [--remove-specialized]
```

Creates:
- `artifacts/data_splits/train.csv` (~80% of groups)
- `artifacts/data_splits/test.csv` (~20% of groups)
- `artifacts/data_splits/split_info.json`

**Benefits:** Reproducible, fair comparison, no data leakage (group-based split by `reaction_id`)

**Usage in code:**
```python
from copol_prediction.utils import load_data_split
df_train, df_test = load_data_split.load_train_test_split()
```

## Scripts

| Script | Purpose | Time |
|--------|---------|------|
| `train_final_model.py` | Train production model + analysis | ~20 min |
| `analysis/analyze_model.py` | Generate analysis plots | < 1 min |
| `../experiments/filter_comparison/sweep_filters.py` | Test 16 filter combinations | ~3 hours |
| `create_data_split.py` | Create central split | < 1 min |
| `monomer_feature_calculation.py` | Calculate molecular features | 1-5 min/monomer |
| `api/app.py` | REST API server | Instant |

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

### experiments/filter_comparison/sweep_filters.py

Tests all 16 filter combinations (4×4 matrix) on same holdout set.

```bash
cd ../experiments/filter_comparison
python sweep_filters.py [--n-iter N]
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
cd api && python app.py  # Runs at http://localhost:8000
```

**See [`api/README.md`](api/README.md) for full API documentation.**

**Endpoints:**
- `GET /health` - Health check
- `GET /model/info` - Model metadata
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /docs` - Interactive API documentation (Swagger UI)

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
├── monomer_feature_calculation.py
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── load_data_split.py     # Load central split utility
├── api/                        # REST API
│   ├── app.py                  # FastAPI application
│   ├── README.md               # API documentation
│   ├── baseline_lookup.py      # Nearest-neighbor lookup
│   ├── reaction_optimization.py # Solvent / temperature grid search
│   ├── morfeus_patch.py        # XTB compatibility patch
│   ├── requirements.txt
│   ├── Dockerfile              # Docker deployment
│   ├── compose.yaml            # Docker Compose config
│   ├── data/                   # PCA embeddings
│   └── molecule_properties/    # Precomputed monomer features
├── analysis/                   # Analysis tools
│   ├── analyze_model.py       # Main analysis script
│   ├── plot_config.py          # Plot styling
│   └── lamalab.mplstyle        # Plot style file
├── artifacts/
│   ├── data_splits/            # Central train/test split
│   ├── model_bundle/           # Trained model
│   └── experiments_holdout/    # Sweep results
└── output/
    ├── analysis/               # Generated plots
    └── processed_data.csv      # Processed dataset

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
| No train/test split | `python create_data_split.py` |
| API port in use | `lsof -ti:8000 \| xargs kill` |
| Quick test | `python train_final_model.py --hyperparam-iter 5` |

## Common Commands

```bash
# Quick test (fewer iterations)
python train_final_model.py --hyperparam-iter 5

# Manual analysis (if needed)
python analysis/analyze_model.py --all --compare-holdout

# Recreate data split
python create_data_split.py

# Kill API
lsof -ti:8000 | xargs kill

# Run specific analysis
cd analysis && python analyze_model.py --confusion --confidence --features
```

## Notes

- **Plot Styling:** All plots use LamaLab matplotlib style from `analysis/plot_config.py`
- **Confidence Filtering:** Dynamic thresholding per class to improve accuracy
- **Reproducibility:** Fixed random seed (42), central split ensures consistency
- **Legacy:** Old `classification.py` kept for reference, use new modular scripts

## Migration from classification.py

Old monolithic script → New modular system:
- `classification.py::main()` → `train_final_model.py`
- `classification.py::sweep_filters_and_plot()` → `experiments/sweep_filters.py`
- Manual model loading → `CopolymerPredictor` class

All functionality preserved in new modules under `src/copolpredictor/`.
