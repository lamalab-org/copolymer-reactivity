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

### Public HTTPS Access (ngrok)

To expose your API publicly with HTTPS for testing with external websites:

```bash
# 1. Install ngrok: https://ngrok.com/download
# 2. Authenticate: ngrok config add-authtoken YOUR_TOKEN
# 3. Start API: python app.py
# 4. In another terminal: ngrok http 8000
# 5. Use the https:// URL shown by ngrok
```

See `NGROK_SETUP.md` for detailed instructions.

### Local Development

```bash
pip install -r requirements.txt
python app.py
```

**Note:** Requires XTB installation (`conda install -c conda-forge xtb-python`)

### Testing

Run comprehensive tests for all features:

```bash
# Test against local API
python test_all_features.py

# Test against ngrok URL (for public HTTPS access)
python test_all_features.py --url https://your-ngrok-url.ngrok-free.app
```

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

### Dataset Query

```bash
POST /check_doi   # Check if a DOI exists in the dataset
```

### Reaction Optimization (NEW!)

```bash
POST /optimize_reaction   # Explore 3x3 grid of predictions (3 temperatures × 3 solvents)
```

The `/optimize_reaction` endpoint performs reaction optimization by:
- Varying temperature: base_temp - step, base_temp, base_temp + step (default step: 20°C)
- Finding similar solvents: selects solvents with similar logP values from the dataset
- Generating predictions: creates a 3x3 grid of predictions (9 total)
- Returns: predicted class and probabilities for each combination

### Similar Papers & Nearest Neighbors (NEW!)

The `/preprocess_all` endpoint now automatically returns:

1. **10 most similar papers** from the dataset based on:
   - Monomer similarity (Tanimoto)
   - Solvent similarity
   - Temperature proximity
   - Method/Polytype embeddings

2. **10 nearest data points** from the training database using baseline lookup:
   - Based on Tanimoto similarity of monomer and solvent SMILES fingerprints
   - Returns: class, monomer names, solvent name, conditions, source (DOI), similarity score
   - Uses the same approach as `experiments/baseline` database lookup model

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
    # Access nearest neighbors
    if preprocessed.get("nearest_neighbors"):
        print("\nTop 3 nearest neighbors:")
        for neighbor in preprocessed["nearest_neighbors"][:3]:
            print(f"  Rank {neighbor['rank']}: {neighbor['monomer1_name']} + {neighbor['monomer2_name']} "
                  f"(similarity: {neighbor['similarity']:.3f}, class: {neighbor.get('class', neighbor.get('predicted_class'))})")
    
    # Make prediction directly with preprocessed features
    result = requests.post(
        f"{API_URL}/predict",
        json={"features": preprocessed["features"]}
    ).json()
    
    print(f"\nPredicted class: {result['predicted_class']} ({result['r_product_range']})")
    print(f"Confidence: {result['confidence']:.2%}")
else:
    print(f"Preprocessing failed: {preprocessed.get('error')}")
```

### Reaction Optimization Example

```python
import requests

API_URL = "http://localhost:8000"

# Optimize reaction conditions
optimization = requests.post(
    f"{API_URL}/optimize_reaction",
    json={
        "monomer1_smiles": "C=CC1=CC=CC=C1",  # styrene
        "monomer2_smiles": "CC(=O)OC(C)=C",   # methyl methacrylate
        "solvent_smiles": "CC1=CC=CC=C1",     # toluene (base solvent)
        "method": "solvent",
        "polytype": "free radical",
        "temperature": 60.0,  # Base temperature
        "temperature_step": 20.0,  # ±20°C variation
        "n_solvents": 3  # 3 solvents
    }
).json()

if optimization["success"]:
    print(f"Generated {len(optimization['predictions'])} predictions")
    
    # Display results
    for pred in optimization['predictions']:
        print(f"\nTemp: {pred['temperature']:.1f}°C, "
              f"Solvent: {pred['solvent_name']} (logP: {pred['solvent_logp']:.3f})")
        print(f"  Class: {pred['predicted_class']}, "
              f"Confidence: {pred['confidence']:.4f}")
else:
    print(f"Optimization failed: {optimization.get('error')}")
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

### DOI Check Example

Check if a paper (by DOI) exists in the training dataset:

```python
import requests

API_URL = "http://localhost:8000"

# Check if DOI exists in dataset
result = requests.post(
    f"{API_URL}/check_doi",
    json={"doi": "10.1016/0014-3057(84)90010-7"}
).json()

if result["exists"]:
    print(f"✓ DOI found in dataset!")
    print(f"DOI: {result['doi']}")
    print(f"Normalized: {result['normalized_doi']}")
else:
    print(f"✗ DOI not found in dataset")

# Also works with full URL
result = requests.post(
    f"{API_URL}/check_doi",
    json={"doi": "https://doi.org/10.1016/0014-3057(84)90010-7"}
).json()

print(f"Exists: {'YES' if result['exists'] else 'NO'}")
```

## 📊 Confidence Score

The API returns a confidence score (0-1) for each prediction. The confidence is calculated using a **weighted metric**:

```python
confidence = 0.7 × max_probability + 0.3 × margin_to_second_best
```

**Interpretation:**
- **> 0.80**: Very certain (e.g., `[0.90, 0.05, 0.05]` → 94% confidence)
- **0.60-0.80**: Quite certain (e.g., `[0.70, 0.20, 0.10]` → 64% confidence)
- **0.50-0.60**: Somewhat uncertain (e.g., `[0.65, 0.35, 0.00]` → 55% confidence)
- **< 0.50**: Uncertain (e.g., `[0.50, 0.45, 0.05]` → 43% confidence)

For more details, see `CONFIDENCE_EXPLAINED.md`.

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
├── app.py                    # FastAPI application
├── baseline_lookup.py        # Nearest-neighbor lookup
├── reaction_optimization.py  # Solvent / temperature grid search
├── morfeus_patch.py          # XTB compatibility patch
├── requirements.txt
├── Dockerfile
├── compose.yaml
├── README.md
├── data/                     # PCA embeddings (method / polytype)
├── config/                   # qcengine config
├── cache/                    # Runtime cache (gitignored)
└── molecule_properties/      # Precomputed monomer features
```

## 📖 Resources

- Interactive API docs: http://localhost:8000/docs
- Model training: `../train_final_model.py`
- XTB docs: https://xtb-docs.readthedocs.io/
