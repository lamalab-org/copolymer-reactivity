# Project Structure - copol_prediction/

Clean and organized directory structure for the copolymerization prediction pipeline.

## Main Directory

Core scripts in the main `copol_prediction/` directory:

```
copol_prediction/
├── train_final_model.py            # Main training script
├── monomer_feature_calculation.py  # Calculate molecular features
├── utils/                          # Utility functions
│   ├── __init__.py
│   └── load_data_split.py          # Load train/test split
└── README.md                       # Main documentation
```

## Subdirectories

### api/ - REST API

Production-ready REST API for predictions.

```
api/
├── app.py                    # FastAPI application
├── baseline_lookup.py        # Nearest-neighbor lookup
├── reaction_optimization.py  # Solvent / temperature grid search
├── morfeus_patch.py          # XTB compatibility patch
├── requirements.txt
├── Dockerfile
├── compose.yaml
├── README.md
├── data/                     # PCA embeddings
└── molecule_properties/      # Precomputed monomer features
```

**Start API**: `cd api && docker compose up -d`

### analysis/ - Analysis Tools

All analysis and visualization tools.

```
analysis/
├── analyze_model.py        # Main analysis script
├── plot_config.py          # Plot styling configuration
└── lamalab.mplstyle        # Plot style file
```
(Permutation importance lives in experiments/permutation_importance/.)

**Run analysis**: `cd analysis && python analyze_model.py --all`

### artifacts/ - Training Artifacts

Generated during training and experiments.

```
artifacts/
├── data_splits/            # Central train/test split
│   ├── train.csv
│   ├── test.csv
│   └── split_info.json
├── model_bundle/           # Trained production model
│   ├── model.joblib
│   ├── model.xgb.json
│   ├── meta.json
│   ├── SELECTED_FILTERS.json
│   └── holdout_results/
└── experiments_holdout/    # Filter sweep results
    └── *.json
```

### output/ - Generated Output

Analysis results and processed data.

```
output/
├── analysis/               # Generated analysis plots
│   ├── *.png              # Various analysis visualizations
│   ├── metrics.csv        # Performance metrics
│   └── feature_importance.txt
├── molecule_properties/    # Cached molecular features
│   └── *.json             # One file per monomer
├── processed_data.csv      # Processed dataset
└── *.json                  # Various cached data
```

### filter/ - Data Filtering

Tools for data filtering and augmentation.

```
filter/
├── artificial_datapoints/  # Synthetic data generation
└── llm_specialized_filter/ # LLM-based filtering
```

### utils/ - Utility Functions

Helper functions and utilities.

```
utils/
├── __init__.py
└── load_data_split.py     # Load central train/test split
```

## Directory Navigation

### For Users

**Make predictions**:
```bash
cd api
docker compose up -d
# Open http://localhost:8000/docs
```

**View analysis**:
```bash
cd output/analysis
# Open PNG files
```

### For Developers

**Train new model**:
```bash
python train_final_model.py
```

**Run analysis**:
```bash
cd analysis
python analyze_model.py --all --compare-holdout
```

**Test filters**:
```bash
cd ../experiments
python sweep_filters.py
```

**Calculate features**:
```bash
python monomer_feature_calculation.py
```

## File Organization Principles

1. **Main directory**: Core training and utility scripts only
2. **api/**: Everything related to the REST API
3. **analysis/**: All analysis and visualization tools
4. **artifacts/**: Generated during training (models, splits)
5. **output/**: Generated results (plots, data)
6. **filter/**: Data filtering and augmentation

## Related Documentation

- Main docs: `README.md`
- API docs: `api/README.md`
- API setup: `api/SETUP_COMPLETE.md`

---

**Last updated**: 2025-11-14

