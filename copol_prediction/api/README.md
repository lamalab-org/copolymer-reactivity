# Copolymerization Prediction API

REST API for predicting copolymerization reactivity using machine learning with XTB quantum chemistry calculations.

## 🚀 Quick Start

### Docker (Recommended)

```bash
# Start
docker compose up -d

# Test
curl http://localhost:8000/health
docker exec copol-prediction-api conda run -n copol python tests/test_xtb.py

# Stop
docker compose down
```

**URLs:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

### Local Development

```bash
pip install -r requirements.txt
python app.py
# Or: ./start.sh
```

**Note:** Requires XTB installation (`conda install -c conda-forge xtb-python`)

## 📊 Model

- **Features**: 15
- **Classes**: 3 (r-product ranges: <1, 1-25, >25)
- **Accuracy**: 78.6% (holdout), 84.6% (CV)
- **Training Date**: 2025-11-14

## 🔧 API Endpoints

### Core Endpoints

```bash
# Health check
GET /health

# Model info
GET /model/info

# Prediction
POST /predict
POST /predict/batch
```

### Preprocessing

```bash
# Convert SMILES to features
POST /preprocess/solvent   # Fast (takes solvent SMILES)
POST /preprocess/monomer   # Slow (XTB calculation on first request, takes monomer SMILES)
POST /preprocess_all       # Combined preprocessing (monomers + solvent + embeddings → features ready for prediction)
```

### Embeddings

```bash
GET /embeddings/method/{name}
GET /embeddings/polytype/{name}
```

## 💻 Usage Example

### Simple Approach (Recommended)

```python
import requests

API_URL = "http://localhost:8000"

# Preprocess everything at once (recommended for web integrations)
preprocessed = requests.post(
    f"{API_URL}/preprocess_all",
    json={
        "monomer1_smiles": "C=CC1=CC=CC=C1",  # styrene
        "monomer2_smiles": "CC(=O)OC(C)=C",   # methyl methacrylate
        "solvent_smiles": "CC1=CC=CC=C1",     # toluene
        "method": "solvent",
        "polytype": "free radical",
        "temperature": 60.0
    }
).json()

if preprocessed["success"]:
    # Make prediction directly with preprocessed features
    result = requests.post(
        f"{API_URL}/predict",
        json={"features": preprocessed["features"]}
    ).json()
    
    print(f"Predicted class: {result['predicted_class']} ({result['r_product_range']})")
    print(f"Confidence: {result['confidence']:.2%}")
else:
    print(f"Preprocessing failed: {preprocessed.get('error')}")
```

### Step-by-Step Approach

```python
import requests

API_URL = "http://localhost:8000"

# Preprocess monomers (uses XTB for quantum chemistry, input as SMILES)
monomer1 = requests.post(
    f"{API_URL}/preprocess/monomer",
    json={"monomer_smiles": "C=CC1=CC=CC=C1"}  # styrene SMILES
).json()

monomer2 = requests.post(
    f"{API_URL}/preprocess/monomer",
    json={"monomer_smiles": "CC(=O)OC(C)=C"}  # methyl methacrylate SMILES
).json()

# Get other features (solvent SMILES)
solvent = requests.post(
    f"{API_URL}/preprocess/solvent",
    json={"solvent_smiles": "CC1=CC=CC=C1"}  # toluene SMILES
).json()

method_emb = requests.get(f"{API_URL}/embeddings/method/solution").json()
polytype_emb = requests.get(f"{API_URL}/embeddings/polytype/free radical").json()

# Calculate HOMO-LUMO deltas
homo_1, lumo_1 = monomer1["features"]["homo"], monomer1["features"]["lumo"]
homo_2, lumo_2 = monomer2["features"]["homo"], monomer2["features"]["lumo"]

# Make prediction
features = {
    "fukui_radical_max_1": monomer1["features"]["fukui_radical_max"],
    "fukui_radical_max_2": monomer2["features"]["fukui_radical_max"],
    "delta_HOMO_LUMO_AA": homo_1 - lumo_1,
    "delta_HOMO_LUMO_AB": homo_1 - lumo_2,
    "delta_HOMO_LUMO_BB": homo_2 - lumo_2,
    "delta_HOMO_LUMO_BA": homo_2 - lumo_1,
    "temperature": 60.0,
    "polytype_emb_1": polytype_emb["pca_1"],
    "polytype_emb_2": polytype_emb["pca_2"],
    "method_emb_1": method_emb["pca_1"],
    "method_emb_2": method_emb["pca_2"],
    "solvent_logP": solvent["features"]["solvent_logP"],
    "solvent_TPSA": solvent["features"]["solvent_TPSA"],
    "solvent_HBD": solvent["features"]["solvent_HBD"],
    "solvent_FractionCSP3": solvent["features"]["solvent_FractionCSP3"]
}

result = requests.post(f"{API_URL}/predict", json={"features": features}).json()
print(f"Predicted class: {result['predicted_class']} ({result['r_product_range']})")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🐋 Docker

### Resources

Recommended for XTB calculations:
- **CPUs**: 2
- **Memory**: 6GB

Adjust in `docker-compose.yml` if needed.

### XTB Cache

- First monomer calculation: **slow** (minutes)
- Subsequent requests: **fast** (cached)
- Cache: `./molecule_properties/` (persisted via volume)

### Management

```bash
# Rebuild
docker compose build --build-arg BUILDPLATFORM=linux/amd64
docker compose up -d

# Logs
docker compose logs -f

# Shell access
docker exec -it copol-prediction-api /bin/bash

# Clean up
docker compose down -v
```

## 📁 Directory Structure

```
api/
├── app.py                    # Main application
├── morfeus_patch.py          # XTB compatibility patch
├── requirements.txt          # Dependencies
├── Dockerfile
├── docker-compose.yml
├── start.sh
├── README.md
├── data/                     # Embeddings
├── config/                   # Configuration
├── tests/                    # Test scripts
├── cache/                    # Runtime cache
└── molecule_properties/      # Monomer cache (1147 files)
```

## 📖 Resources

- Interactive API docs: http://localhost:8000/docs
- Model training: `../train_final_model.py`
- XTB docs: https://xtb-docs.readthedocs.io/
