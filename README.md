# Copolymerization Reactivity Prediction

Machine learning system for extracting and predicting copolymerization reactivity ratios from scientific literature.

## 🎯 Overview

This project combines automated literature data extraction with machine learning to predict copolymerization reactivity patterns. The system extracts reactivity ratio data (r₁, r₂) from scientific papers and trains models to predict the r-product (r₁ × r₂) class, which indicates the copolymerization behavior:

- **Class 0**: r₁·r₂ < 1 → Alternating copolymerization
- **Class 1**: 1 ≤ r₁·r₂ ≤ 25 → Random to weak block formation
- **Class 2**: r₁·r₂ > 25 → Strong block formation

### Key Features

✅ Automated literature data extraction  
✅ ML-based reactivity prediction (78.6% accuracy)  
✅ REST API for predictions    
✅ Comprehensive analysis tools 

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/marawilhelmi/polymer-reactivity
cd polymer-reactivity

# Install package
pip install -e .

# Or install from GitHub directly
pip install git+https://github.com/marawilhelmi/polymer-reactivity
```

### 2. Use the Prediction API

The fastest way to use the trained model:

```bash
cd copol_prediction/api
pip install -r requirements.txt
./start.sh
```

Then open http://localhost:8000/docs for interactive API documentation.

**See [`copol_prediction/api/README.md`](copol_prediction/api/README.md) for complete API documentation.**

### 3. Use the Python Library

```python
from copolpredictor.inference import CopolymerPredictor

# Load trained model
predictor = CopolymerPredictor("copol_prediction/artifacts/model_bundle")

# Make prediction
features = {...}  # 15 molecular and reaction features
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
./start.sh
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

Current production model (trained 2025-11-14):

| Metric | Value |
|--------|-------|
| Holdout Accuracy | 78.6% |
| Holdout F1 (weighted) | 79.0% |
| Holdout F1 (macro) | 67.8% |
| CV Score | 84.6% |
| Features | 15 |
| Classes | 3 |

**Model location**: `copol_prediction/artifacts/model_bundle/`

### Required Features

The model uses 15 molecular and reaction features:

**Molecular descriptors** (8):
- `fukui_radical_max_1/2` - Fukui radical indices
- `delta_HOMO_LUMO_AA/AB/BB/BA` - Orbital interactions

**Reaction conditions** (7):
- `temperature` - Reaction temperature (°C)
- `solvent_logP`, `solvent_TPSA`, `solvent_HBD`, `solvent_FractionCSP3` - Solvent properties
- `polytype_emb_1/2` - Polymerization type embeddings
- `method_emb_1/2` - Method embeddings

## 🔧 Usage Examples

### Python API

```python
from copolpredictor.inference import CopolymerPredictor

# Initialize predictor
predictor = CopolymerPredictor("copol_prediction/artifacts/model_bundle")

# Make prediction
features = {
    "fukui_radical_max_1": 0.15,
    "fukui_radical_max_2": 0.18,
    "delta_HOMO_LUMO_AA": -5.2,
    "delta_HOMO_LUMO_AB": -4.8,
    "delta_HOMO_LUMO_BB": -5.5,
    "delta_HOMO_LUMO_BA": -4.9,
    "temperature": 60.0,
    "polytype_emb_1": 0.23,
    "polytype_emb_2": -0.15,
    "method_emb_1": 0.45,
    "method_emb_2": -0.32,
    "solvent_logP": 2.1,
    "solvent_TPSA": 20.5,
    "solvent_HBD": 0.0,
    "solvent_FractionCSP3": 0.67
}

result = predictor.predict_with_confidence(features)

print(f"Class: {result['predictions'][0]}")
print(f"Confidence: {result['confidence'][0]:.2%}")
```

### REST API

```bash
# Start API
cd copol_prediction/api && ./start.sh

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"features": {...}}'
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

If you encounter any problems or have suggestions, please [open an issue](https://github.com/marawilhelmi/polymer-reactivity/issues).

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

