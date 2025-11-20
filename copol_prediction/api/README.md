# Copolymerization Prediction API

REST API for predicting copolymerization reactivity using machine learning.

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd copol_prediction/api
pip install -r requirements.txt
```

### 2. Start the API

```bash
# Development mode (with auto-reload)
python app.py

# Or with uvicorn directly
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- **API**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

## 📊 Model Information

Current model trained on: **2025-11-14**

- **Features**: 15
- **Classes**: 3 (r-product ranges: <1, 1-25, >25)
- **Holdout Accuracy**: 78.6%
- **Holdout F1 (weighted)**: 79.0%
- **CV Score**: 84.6%

### Required Features

The model requires these 15 features:

1. `fukui_radical_max_1` - Fukui radical maximum for monomer 1
2. `fukui_radical_max_2` - Fukui radical maximum for monomer 2
3. `delta_HOMO_LUMO_AA` - HOMO-LUMO delta for AA interaction
4. `delta_HOMO_LUMO_AB` - HOMO-LUMO delta for AB interaction
5. `delta_HOMO_LUMO_BB` - HOMO-LUMO delta for BB interaction
6. `delta_HOMO_LUMO_BA` - HOMO-LUMO delta for BA interaction
7. `temperature` - Reaction temperature (°C)
8. `polytype_emb_1` - Polymerization type embedding (dimension 1)
9. `polytype_emb_2` - Polymerization type embedding (dimension 2)
10. `method_emb_1` - Method embedding (dimension 1)
11. `method_emb_2` - Method embedding (dimension 2)
12. `solvent_logP` - Solvent logP
13. `solvent_TPSA` - Solvent topological polar surface area
14. `solvent_HBD` - Solvent hydrogen bond donors
15. `solvent_FractionCSP3` - Solvent fraction of sp³ carbons

## 🔧 API Endpoints

### 1. Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-14T10:30:00",
  "model_loaded": true
}
```

### 2. Model Information
```bash
GET /model/info
```

**Response:**
```json
{
  "model_version": "1.0.0",
  "n_features": 15,
  "feature_names": ["fukui_radical_max_1", ...],
  "class_labels": [0, 1, 2],
  "created_at": "2025-11-14T09:22:31.654067Z",
  "model_path": "../artifacts/model_bundle"
}
```

### 3. Get Required Features
```bash
GET /features
```

**Response:**
```json
{
  "required_features": ["fukui_radical_max_1", ...],
  "n_features": 15
}
```

### 4. Single Prediction
```bash
POST /predict
```

**Request Body:**
```json
{
  "features": {
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
}
```

**Response:**
```json
{
  "predicted_class": 1,
  "class_probabilities": {
    "class_0": 0.15,
    "class_1": 0.70,
    "class_2": 0.15
  },
  "confidence": 0.85,
  "r_product_range": "1-25 (Random to weak block)",
  "timestamp": "2025-11-14T10:30:00"
}
```

### 5. Batch Prediction
```bash
POST /predict/batch
```

**Request Body:**
```json
{
  "samples": [
    {
      "fukui_radical_max_1": 0.15,
      "fukui_radical_max_2": 0.18,
      ...
    },
    {
      "fukui_radical_max_1": 0.20,
      "fukui_radical_max_2": 0.22,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "predicted_class": 1,
      "class_probabilities": {...},
      "confidence": 0.85,
      "r_product_range": "1-25 (Random to weak block)",
      "timestamp": "2025-11-14T10:30:00"
    },
    ...
  ],
  "total_samples": 2,
  "timestamp": "2025-11-14T10:30:00"
}
```

### 6. Preprocess Solvent
```bash
POST /preprocess/solvent
```

Converts a solvent name to SMILES and calculates solvent features.

**Request Body:**
```json
{
  "solvent_name": "toluene"
}
```

**Response:**
```json
{
  "solvent_name": "toluene",
  "solvent_smiles": "Cc1ccccc1",
  "features": {
    "solvent_logP": 2.73,
    "solvent_TPSA": 0.0,
    "solvent_HBD": 0.0,
    "solvent_FractionCSP3": 0.14
  },
  "success": true,
  "error": null
}
```

### 7. Preprocess Monomer
```bash
POST /preprocess/monomer
```

Converts a monomer name to SMILES, checks for existing features in cache, or calculates new features if needed.

**Request Body:**
```json
{
  "monomer_name": "styrene"
}
```

**Response:**
```json
{
  "monomer_name": "styrene",
  "monomer_smiles": "C=Cc1ccccc1",
  "features": {
    "fukui_radical_max": 0.203,
    "homo": -0.416,
    "lumo": -0.207
  },
  "success": true,
  "error": null,
  "from_cache": true
}
```

**Note:** If features need to be calculated, this may take several minutes as it performs quantum chemical calculations.

### 8. Get Available Methods
```bash
GET /embeddings/methods
```

Returns a list of all available method strings that have embeddings.

**Response:**
```json
{
  "methods": ["bulk", "emulsion", "solution", "solvent", ...],
  "count": 30
}
```

### 9. Get Available Polytypes
```bash
GET /embeddings/polytypes
```

Returns a list of all available polymerization type strings that have embeddings.

**Response:**
```json
{
  "polytypes": ["free radical", "cationic", "anionic", ...],
  "count": 75
}
```

### 10. Get Method Embeddings
```bash
GET /embeddings/method/{method_name}
```

Returns PCA-reduced embeddings for a specific method string.

**Example:**
```bash
GET /embeddings/method/solution
```

**Response:**
```json
{
  "pca_1": -3.403,
  "pca_2": 2.568
}
```

### 11. Get Polytype Embeddings
```bash
GET /embeddings/polytype/{polytype_name}
```

Returns PCA-reduced embeddings for a specific polymerization type string.

**Example:**
```bash
GET /embeddings/polytype/free radical
```

**Response:**
```json
{
  "pca_1": 7.497,
  "pca_2": -0.463
}
```

## 💻 Usage Examples

### Complete Workflow: From Names to Prediction

```python
import requests

API_URL = "http://localhost:8000"

# Step 1: Preprocess monomers
monomer1_response = requests.post(
    f"{API_URL}/preprocess/monomer",
    json={"monomer_name": "styrene"}
)
monomer1 = monomer1_response.json()

monomer2_response = requests.post(
    f"{API_URL}/preprocess/monomer",
    json={"monomer_name": "methyl methacrylate"}
)
monomer2 = monomer2_response.json()

# Step 2: Preprocess solvent
solvent_response = requests.post(
    f"{API_URL}/preprocess/solvent",
    json={"solvent_name": "toluene"}
)
solvent = solvent_response.json()

# Step 3: Get embeddings
method_response = requests.get(f"{API_URL}/embeddings/method/solution")
method_emb = method_response.json()

polytype_response = requests.get(f"{API_URL}/embeddings/polytype/free radical")
polytype_emb = polytype_response.json()

# Step 4: Calculate HOMO-LUMO deltas
homo_1 = monomer1["features"]["homo"]
lumo_1 = monomer1["features"]["lumo"]
homo_2 = monomer2["features"]["homo"]
lumo_2 = monomer2["features"]["lumo"]

# Step 5: Prepare features for prediction
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

# Step 6: Make prediction
response = requests.post(
    f"{API_URL}/predict",
    json={"features": features}
)

result = response.json()
print(f"Predicted class: {result['predicted_class']}")
print(f"Range: {result['r_product_range']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Simple Prediction with Pre-calculated Features

```python
import requests

# API URL
API_URL = "http://localhost:8000"

# Example features
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

# Make prediction
response = requests.post(
    f"{API_URL}/predict",
    json={"features": features}
)

result = response.json()
print(f"Predicted class: {result['predicted_class']}")
print(f"Range: {result['r_product_range']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
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
  }'
```

## 🐛 Troubleshooting

### Problem: "Model not loaded"

**Solution**: Ensure the model bundle is in the correct directory:
```bash
ls ../artifacts/model_bundle/
# Should contain: model.joblib, meta.json, etc.
```

### Problem: "Missing required features"

**Solution**: All 15 features must be present in the request. Check with:
```bash
curl http://localhost:8000/features
```

### Problem: Port already in use

**Solution**: Use a different port:
```bash
uvicorn app:app --port 8001
```

## 📚 Additional Information

- See `../train_final_model.py` for model training details
