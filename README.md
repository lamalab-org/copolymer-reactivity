# Copolymerization Reactivity Prediction

Machine learning system for extracting and predicting copolymerization reactivity ratios from scientific literature.

## 🎯 Overview

This project combines automated literature data extraction with machine learning to predict copolymerization reactivity patterns. The system extracts reactivity ratio data (r₁, r₂) from scientific papers and trains a classifier on the r-product (r₁ × r₂).

For class definitions and bin edges, see the model artifact (`copol_prediction/artifacts/model_bundle/meta.json`) and the paper (TODO: link).

### Key Features

✅ Automated literature data extraction  
✅ ML-based reactivity prediction  
✅ REST API for predictions    
✅ Comprehensive analysis tools 

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/lamalab-org/copolymer-reactivity
cd copolymer-reactivity

# Install package
pip install -e .

# Or install from GitHub directly
pip install git+https://github.com/lamalab-org/copolymer-reactivity
```

### 2. Use the Prediction API

The fastest way to use the trained model is the pre-built container:

```bash
cd copol_prediction/api
docker compose up
```

This pulls `ghcr.io/lamalab-org/copolymer-reactivity:latest` and starts the API on port 8000. To run from source instead:

```bash
cd copol_prediction/api
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open http://localhost:8000/docs for interactive API documentation.

**See [`copol_prediction/api/README.md`](copol_prediction/api/README.md) for complete API documentation.**

### 3. Use the Python Library

```python
from copolpredictor.inference import CopolymerPredictor

# Load trained model
predictor = CopolymerPredictor("copol_prediction/artifacts/model_bundle")

# Make prediction
features = {...}  # 19 molecular and reaction features — see "Required Features" below
result = predictor.predict_with_confidence(features)

print(f"Predicted class: {result['predictions'][0]}")
print(f"Confidence: {result['confidence'][0]:.2%}")
```

## 📁 Project Structure

```
├── copol_prediction/          # ML prediction pipeline
│   ├── api/                   # REST API (FastAPI)
│   ├── analysis/              # Model analysis tools
│   ├── utils/                 # Utility functions
│   ├── artifacts/             # Trained models & data splits
│   └── README.md              # Prediction pipeline docs
│
├── data_extraction/           # Literature data extraction
│   ├── obtain_data.py         # Main extraction script
│   ├── output/                # Extracted data
│   └── README.md              # Extraction docs
│
├── experiments/               # Experiments & filter sweeps
│   ├── sweep_filters.py       # Test filter combinations
│   ├── baseline/              # Baseline models
│   ├── fingerprint/           # Fingerprint-based models
│   └── README.md              # Experiments docs
│
├── src/                       # Core libraries
│   ├── copolextractor/        # Data extraction library
│   └── copolpredictor/        # ML prediction library
├── tests/                     # Unit tests
├── dump/                      # Legacy code (archived)
└── pyproject.toml             # Package configuration
```

## 🧪 Main Components

### 1. Prediction API 

**Location**: [`copol_prediction/api/`](copol_prediction/api/)

REST API for making predictions:

```bash
cd copol_prediction/api
docker compose up
# Open http://localhost:8000/docs
```

**Features**:
- FastAPI with automatic validation
- Batch predictions
- Docker deployment ready

📖 **Full documentation**: [`copol_prediction/api/README.md`](copol_prediction/api/README.md)

### 2. ML Prediction Pipeline

**Location**: [`copol_prediction/`](copol_prediction/)

Complete machine learning pipeline for training and evaluating models:

```bash
cd copol_prediction

# Train production model (~20 min)
python train_final_model.py

# Run analysis
cd analysis && python analyze_model.py --all

# Test filter combinations (~3 hours)
cd ../../experiments && python sweep_filters.py
```

**Key Scripts**:
- `train_final_model.py` - Train production model with automatic analysis
- `monomer_feature_calculation.py` - Calculate molecular features
- `analysis/analyze_model.py` - Generate analysis plots
- `sweep_filters.py` - Test all filter combinations

📖 **Full documentation**: [`copol_prediction/README.md`](copol_prediction/README.md)

### 3. Data Extraction

**Location**: [`data_extraction/`](data_extraction/)

Automated extraction of copolymerization data from scientific literature:

```bash
cd data_extraction
python obtain_data.py
```

**Features**:
- CrossRef API integration
- LLM-based data extraction 
- Automatic monomer name resolution
- Confidence scoring

📖 **Full documentation**: [`data_extraction/README.md`](data_extraction/README.md)


### 4. Core Libraries

**Location**: [`src/`](src/)

Two main Python packages:

#### copolextractor
Library for extracting copolymerization data from literature.

```python
from copolextractor import crossref_search, utils

# Search for papers
papers = crossref_search.search_papers("copolymerization")

# Resolve monomer names
smiles = utils.name_to_smiles("styrene")
```

#### copolpredictor
Library for ML-based reactivity prediction.

```python
from copolpredictor.inference import CopolymerPredictor
from copolpredictor import data_processing, model_training

# Inference
predictor = CopolymerPredictor("path/to/model")
result = predictor.predict(features)

# Training
model = model_training.train_final_model(X, y, params)
```

**Modules**:
- `data_processing.py` - Data loading & preprocessing
- `data_augmentation.py` - Gaussian augmentation
- `model_training.py` - Model training & hyperparameter optimization
- `evaluation.py` - Metrics & evaluation
- `inference.py` - Production inference
- `calibration.py` - Probability calibration

## 📊 Model Performance

```bash
# headline numbers for the currently deployed model
jq '{holdout_accuracy, holdout_f1_weighted, holdout_f1_macro, cv_score, n_features}' \
  copol_prediction/artifacts/model_bundle/meta.json

# full confusion matrices + per-class breakdown
cat copol_prediction/artifacts/model_bundle/all_metrics.txt

# voting-ensemble metrics
jq . copol_prediction/artifacts/model_bundle/voting_test_metrics.json
```

For methodology and reported numbers see the paper (TODO: add citation/link).

**Model location**: `copol_prediction/artifacts/model_bundle/`

### Required Features

```bash
# from a running API
curl http://localhost:8000/features

# or from the artifact directly
jq '.feature_columns, .n_features' copol_prediction/artifacts/model_bundle/meta.json
```

If you have SMILES + reaction conditions instead of precomputed features, `POST /preprocess_all` builds the model-ready feature dict for you (see the Quick Start example above).

## 🔧 Usage Examples

### Python API

```python
# Prediction from SMILES + reaction conditions via the local API.
# /preprocess_all builds the feature dict the model needs from the
# cached molecular-property JSONs, so you don't hand-build the schema.
import requests

API = "http://localhost:8000"
features = requests.post(f"{API}/preprocess_all", json={
    "monomer1_smiles": "C=COC(C)=O",                # vinyl acetate
    "monomer2_smiles": "C=COC(=O)c1ccccc1",         # vinyl benzoate
    "solvent_smiles": "c1ccccc1",                   # benzene
    "method": "solvent",
    "polytype": "free radical",
    "temperature": 79.6,
}, timeout=120).json()["features"]

pred = requests.post(f"{API}/predict", json={"features": features}).json()
print(f"Class {pred['predicted_class']} ({pred['predicted_class_name']}), "
      f"confidence {pred['confidence']:.2%}")
```

If you already have precomputed features and prefer to bypass the API, use the `CopolymerPredictor` class directly (`from copolpredictor.inference import CopolymerPredictor`) — the required feature names are in the artifact, not here.

### REST API

```bash
# Start API
cd copol_prediction/api && docker compose up -d

# /preprocess_all takes SMILES + reaction conditions and returns the model-ready feature dict
FEATURES=$(curl -sS -X POST http://localhost:8000/preprocess_all \
  -H "Content-Type: application/json" \
  -d '{"monomer1_smiles":"C=COC(C)=O","monomer2_smiles":"C=COC(=O)c1ccccc1",
       "solvent_smiles":"c1ccccc1","method":"solvent",
       "polytype":"free radical","temperature":79.6}' | jq -c '.features')

curl -sS -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"features\": $FEATURES}"
```

### Command Line

```bash
# Train new model
cd copol_prediction
python train_final_model.py

# Extract data from literature
cd data_extraction
python obtain_data.py

# Run experiments
cd experiments
python sweep_filters.py
```

## 📖 Detailed Documentation

Each major component has its own detailed README:

| Component | Documentation |
|-----------|---------------|
| **Prediction API** | [`copol_prediction/api/README.md`](copol_prediction/api/README.md) |
| **ML Pipeline** | [`copol_prediction/README.md`](copol_prediction/README.md) |
| **Data Extraction** | [`data_extraction/README.md`](data_extraction/README.md) |
| **Experiments** | [`experiments/README.md`](experiments/README.md) |

## 🚢 Deployment

### Docker

```bash
cd copol_prediction/api
docker-compose up -d
```

The API will be available at http://localhost:8000

### Manual Deployment

```bash
# Install dependencies
pip install -r copol_prediction/api/requirements.txt

# Start with Gunicorn (4 workers)
cd copol_prediction/api
gunicorn app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## 📊 Data

### Extracted Data

The system has extracted and processed data from ~400 scientific papers:

- **Location**: `data_extraction/output/copol_database/`
- **Format**: JSON files (one per paper)
- **Contents**: Monomer pairs, reactivity ratios, reaction conditions

### Processed Data

- **Location**: `copol_prediction/output/processed_data.csv`
- **Samples**: ~1,100 copolymerization reactions
- **Features**: Molecular descriptors + reaction conditions
- **Labels**: r-product class (0, 1, or 2)

### Data Split

Centralized train/test split:
- **Location**: `copol_prediction/artifacts/data_splits/`
- **Split**: ~80% train / ~20% test
- **Method**: Group-based (by `reaction_id`) to prevent data leakage

## 🔬 Research & Development

### Experiments

The `experiments/` directory contains baseline comparisons:

- **Baseline models**: Simple feature sets
- **Fingerprint models**: Morgan fingerprints
- **Filter sweeps**: Testing 16 combinations of preprocessing filters

Run all experiments:
```bash
cd experiments
./run_all.sh
```

### Analysis

Comprehensive analysis tools in `copol_prediction/analysis/`:

```bash
cd copol_prediction
python analysis/analyze_model.py --all --compare-holdout
```

**Generated plots**:
- Confusion matrices
- Confidence distributions
- Feature importance
- Calibration curves
- Error analysis
- Confidence filtering

## 🛠️ Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e .
pip install pytest black isort flake8

# Install pre-commit hooks (optional)
pip install pre-commit
pre-commit install
```

### Code Structure

- `src/copolextractor/` - Data extraction library
- `src/copolpredictor/` - ML prediction library
- `copol_prediction/` - Training & analysis scripts
- `data_extraction/` - Extraction scripts
- `experiments/` - Baseline experiments
- `tests/` - Unit tests

### Adding New Features

1. Add feature calculation in `src/copolpredictor/data_processing.py`
2. Update feature list in `prediction_utils.py`
3. Retrain model with `copol_prediction/train_final_model.py`
4. Update API documentation

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{wilhelmi2024copolymer,
  author = {Schilling-Wilhelmi, Mara; Jablonka, Kevin M.},
  title = {Copolymerization Reactivity Prediction},
  year = {2025},
  url = {https://github.com/lamalab-org/copolymer-reactivity}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🐛 Issues

If you encounter any problems or have suggestions, please [open an issue](https://github.com/lamalab-org/copolymer-reactivity/issues).

## 📧 Contact

Mara Schilling-Wilhelmi - mara.wilhelmi@uni-jena.de

Project Link: [https://github.com/lamalab-org/copolymer-reactivity](https://github.com/lamalab-org/copolymer-reactivity)


## 📚 Additional Resources

### Documentation
- [API Documentation](copol_prediction/api/README.md) - REST API usage
- [ML Pipeline](copol_prediction/README.md) - Model training & evaluation
- [Data Extraction](data_extraction/README.md) - Literature data extraction
- [Experiments](experiments/README.md) - Baseline comparisons

### Quick Links
- **Interactive API**: http://localhost:8000/docs (when running)
- **Model Performance**: `copol_prediction/output/analysis/`
- **Extracted Data**: `data_extraction/output/copol_database/`
- **Trained Model**: `copol_prediction/artifacts/model_bundle/`

