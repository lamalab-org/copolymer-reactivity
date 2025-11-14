#!/usr/bin/env python3
"""
FastAPI REST API for copolymerization prediction.

This module provides a REST API interface for making predictions
with the trained copolymerization model.

Usage:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import os
import sys
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path

# FastAPI dependencies
try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    print("Error: FastAPI not installed. Install with: pip install fastapi uvicorn")
    sys.exit(1)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from copolpredictor.inference import CopolymerPredictor


# ============================================================================
# API Models (Request/Response schemas)
# ============================================================================

class PredictionInput(BaseModel):
    """Input schema for single prediction."""
    features: Dict[str, float] = Field(
        ...,
        description="Dictionary of feature values",
        example={
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
    )


class BatchPredictionInput(BaseModel):
    """Input schema for batch prediction."""
    samples: List[Dict[str, float]] = Field(
        ...,
        description="List of feature dictionaries"
    )


class PredictionOutput(BaseModel):
    """Output schema for prediction."""
    predicted_class: int = Field(..., description="Predicted class (0, 1, or 2)")
    class_probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    r_product_range: str = Field(..., description="Human-readable r-product range")
    timestamp: str = Field(..., description="Prediction timestamp")


class BatchPredictionOutput(BaseModel):
    """Output schema for batch prediction."""
    predictions: List[PredictionOutput]
    total_samples: int
    timestamp: str


class ModelInfo(BaseModel):
    """Model information schema."""
    model_version: str
    n_features: int
    feature_names: List[str]
    class_labels: List[int]
    created_at: str
    model_path: str


class HealthCheck(BaseModel):
    """Health check response schema."""
    status: str
    timestamp: str
    model_loaded: bool


# ============================================================================
# FastAPI Application
# ============================================================================

# Initialize FastAPI app
app = FastAPI(
    title="Copolymerization Prediction API",
    description="REST API for predicting copolymerization reactivity using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global predictor instance
predictor: Optional[CopolymerPredictor] = None

# Model path (can be configured via environment variable)
MODEL_PATH = os.environ.get("MODEL_PATH", "../artifacts/model_bundle")


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor
    try:
        # Resolve relative path from api directory
        model_path = Path(__file__).parent / MODEL_PATH
        print(f"Loading model from {model_path}...")
        predictor = CopolymerPredictor(str(model_path))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("API will start but predictions will fail until model is loaded")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("Shutting down API...")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Copolymerization Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy" if predictor else "model_not_loaded",
        timestamp=datetime.now().isoformat(),
        model_loaded=predictor is not None
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return ModelInfo(
        model_version="1.0.0",
        n_features=len(predictor.features),
        feature_names=predictor.features,
        class_labels=predictor.class_labels,
        created_at=predictor.metadata.get('created_at', 'unknown'),
        model_path=MODEL_PATH
    )


@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """
    Make a single prediction.
    
    Args:
        input_data: Feature values for prediction
        
    Returns:
        Prediction results with class, probabilities, and confidence
    """
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Make prediction
        results = predictor.predict_with_confidence(input_data.features)
        
        # Format output
        pred_class = int(results['predictions'][0])
        proba = results['probabilities'][0]
        confidence = float(results['confidence'][0])
        
        # Map to range label
        range_labels = {
            0: "< 1 (Strong alternating tendency)",
            1: "1-25 (Random to weak block)",
            2: "> 25 (Strong block tendency)"
        }
        
        return PredictionOutput(
            predicted_class=pred_class,
            class_probabilities={
                f"class_{i}": float(proba[i]) 
                for i in range(len(proba))
            },
            confidence=confidence,
            r_product_range=range_labels[pred_class],
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(input_data: BatchPredictionInput):
    """
    Make predictions for multiple samples.
    
    Args:
        input_data: List of feature dictionaries
        
    Returns:
        Batch prediction results
    """
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        predictions = []
        
        for sample in input_data.samples:
            results = predictor.predict_with_confidence(sample)
            
            pred_class = int(results['predictions'][0])
            proba = results['probabilities'][0]
            confidence = float(results['confidence'][0])
            
            range_labels = {
                0: "< 1 (Strong alternating tendency)",
                1: "1-25 (Random to weak block)",
                2: "> 25 (Strong block tendency)"
            }
            
            predictions.append(PredictionOutput(
                predicted_class=pred_class,
                class_probabilities={
                    f"class_{i}": float(proba[i]) 
                    for i in range(len(proba))
                },
                confidence=confidence,
                r_product_range=range_labels[pred_class],
                timestamp=datetime.now().isoformat()
            ))
        
        return BatchPredictionOutput(
            predictions=predictions,
            total_samples=len(predictions),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/features", response_model=Dict[str, List[str]])
async def get_required_features():
    """Get list of required features for prediction."""
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return {
        "required_features": predictor.features,
        "n_features": len(predictor.features)
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle any unhandled exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "error": str(exc)
        }
    )


# ============================================================================
# Main (for direct execution)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Copolymerization Prediction API...")
    print(f"Model path: {MODEL_PATH}")
    print("\nAPI will be available at:")
    print("  - http://localhost:8000")
    print("  - Documentation: http://localhost:8000/docs")
    print("  - ReDoc: http://localhost:8000/redoc")
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

