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

## 💻 Usage Examples

### Python with requests

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

## 🚢 Production Deployment

### With Gunicorn (recommended for production)

```bash
# Install gunicorn if not installed
pip install gunicorn

# Start with 4 worker processes
gunicorn app:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

### Environment Variables

```bash
# Customize model path (optional)
export MODEL_PATH="/path/to/model_bundle"

# Start API
python app.py
```

## 🐳 Docker Deployment

See `Dockerfile` and `docker-compose.yml` for container deployment.

```bash
# Build Docker image
docker build -t copol-api .

# Run container
docker run -p 8000:8000 copol-api
```

## 📝 Class Interpretation

The model classifies copolymerization reactions into 3 classes based on the r-product:

- **Class 0**: r₁·r₂ < 1 → **Strong alternating tendency**
  - Monomers strongly prefer to react with the other monomer
  - Leads to alternating copolymer

- **Class 1**: 1 ≤ r₁·r₂ ≤ 25 → **Random to weak block formation**
  - Monomers show similar or slightly preferred reactivity
  - Leads to random or weakly block-forming copolymers

- **Class 2**: r₁·r₂ > 25 → **Strong block formation**
  - Monomers strongly prefer to react with themselves
  - Leads to block copolymers or homopolymer mixtures

## 🔒 Security

For production deployment, consider implementing:

1. **API Keys / Authentication**
2. **Rate Limiting** (e.g., with `slowapi`)
3. **HTTPS** activation
4. **CORS** configuration if needed
5. Extended **Input Validation**

## 📊 Monitoring

The API provides basic monitoring via:

- `/health` - Health check endpoint
- `/model/info` - Model metadata

For production monitoring, we recommend:
- Prometheus metrics
- ELK Stack for logs
- Sentry for error tracking

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

- See `test_api.py` for API tests
- See `example_client.py` for usage examples
- See `/docs` (when API is running) for interactive API documentation
- See `../train_final_model.py` for model training details

## 🤝 Support

For questions or issues:
1. Check the interactive documentation at `/docs`
2. Check the API logs
3. Test with the included `test_api.py` script

