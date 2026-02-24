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
import json
import hashlib
import math
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path

# FastAPI dependencies
try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, Field
except ImportError:
    print("Error: FastAPI not installed. Install with: pip install fastapi uvicorn")
    sys.exit(1)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from copolpredictor.inference import CopolymerPredictor
from copolpredictor.data_processing import load_molecular_data
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pandas as pd

# Import paper similarity module
try:
    from paper_similarity import find_similar_papers, format_similarity_output
    SIMILARITY_AVAILABLE = True
except ImportError:
    SIMILARITY_AVAILABLE = False
    print("Warning: paper_similarity module not available. Similar papers feature disabled.")

# Import baseline lookup module
try:
    from baseline_lookup import find_top_k_nearest_neighbors
    BASELINE_LOOKUP_AVAILABLE = True
except ImportError:
    BASELINE_LOOKUP_AVAILABLE = False
    print("Warning: baseline_lookup module not available. Nearest neighbors feature disabled.")

# Import reaction optimization module
try:
    from reaction_optimization import create_optimization_grid
    REACTION_OPTIMIZATION_AVAILABLE = True
except ImportError:
    REACTION_OPTIMIZATION_AVAILABLE = False
    print("Warning: reaction_optimization module not available. Reaction optimization feature disabled.")

# Import solubility check module
try:
    from solubility_check import load_solubility_model, get_solubility_issue_flag
    SOLUBILITY_CHECK_AVAILABLE = True
except ImportError:
    SOLUBILITY_CHECK_AVAILABLE = False
    print("Warning: solubility_check module not available. Solubility check feature disabled.")

# Import monomer feature calculation functions
try:
    from morfeus.conformer import ConformerEnsemble
    import qcengine

    # Try to import patched XTB class, fall back to original if not available
    try:
        # Static mapping from classes to English text descriptions
        class_descriptions_map = {
            0: "alternating",
            1: "random to block like",
            2: "homopolymer",
        }
        from morfeus_patch import XTB

        print("✓ Using patched XTB class for better compatibility")
    except ImportError:
        from morfeus import XTB

        print("⚠ Using original XTB class (may have compatibility issues with newer XTB versions)")

    MORFEUS_AVAILABLE = True

    # Configure QCEngine to find xtb binary via environment variable
    # QCEngine looks for QC_{PROGRAM}_EXE environment variable
    xtb_path = "/opt/xtb/bin/xtb"
    if os.path.exists(xtb_path):
        try:
            os.environ["QC_XTB_EXE"] = xtb_path
            print(f"✓ QCEngine configured with XTB at: {xtb_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not configure QCEngine XTB path: {e}")
except ImportError:
    MORFEUS_AVAILABLE = False
    print("Warning: morfeus not available. Monomer feature calculation will be limited.")


# ============================================================================
# Helper Functions
# ============================================================================

def clean_json_values(obj: Any, replace_with_zero: bool = False) -> Any:
    """
    Recursively clean NaN, Inf, and -Inf values from dictionaries/lists
    to make them JSON-serializable. 
    
    Args:
        obj: Object to clean
        replace_with_zero: If True, replace NaN/Inf with 0.0 instead of None
    """
    if isinstance(obj, dict):
        return {k: clean_json_values(v, replace_with_zero) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_values(item, replace_with_zero) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0 if replace_with_zero else None
        return obj
    elif isinstance(obj, (int, str, bool, type(None))):
        return obj
    else:
        # Try to convert to float and check
        try:
            float_val = float(obj)
            if math.isnan(float_val) or math.isinf(float_val):
                return 0.0 if replace_with_zero else None
            return float_val
        except (ValueError, TypeError):
            return obj


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
    monomer1_smiles: Optional[str] = Field(None, description="Optional: SMILES of monomer 1 (for solubility check)")
    monomer2_smiles: Optional[str] = Field(None, description="Optional: SMILES of monomer 2 (for solubility check)")
    solvent_smiles: Optional[str] = Field(None, description="Optional: SMILES of solvent (for solubility check)")


class BatchPredictionInput(BaseModel):
    """Input schema for batch prediction."""
    samples: List[Dict[str, float]] = Field(
        ...,
        description="List of feature dictionaries"
    )


class SimilarPaper(BaseModel):
    """Similar paper information."""
    rank: int = Field(..., description="Ranking (1-10)")
    doi: str = Field(..., description="DOI of the paper")
    paper_name: str = Field(..., description="Paper filename/name")
    similarity_score: float = Field(..., description="Overall similarity score (0-1)")
    match_quality: str = Field(..., description="Match quality label")
    details: Dict[str, float] = Field(..., description="Component similarity scores")
    reaction_info: Dict = Field(..., description="Reaction information from paper")


class NearestNeighbor(BaseModel):
    """Nearest neighbor data point from training database."""
    rank: int = Field(..., description="Ranking (1-10)")
    similarity: float = Field(..., description="Similarity score (0-1), higher is more similar")
    predicted_class: int = Field(..., description="Predicted class (0, 1, or 2)", alias="class")
    predicted_class_name: str = Field(..., description="Predicted class name: 'alternating', 'random to block like', or 'homopolymer'")
    monomer1_name: str = Field(..., description="First monomer name")
    monomer2_name: str = Field(..., description="Second monomer name")
    monomer1_smiles: str = Field(..., description="First monomer SMILES")
    monomer2_smiles: str = Field(..., description="Second monomer SMILES")
    solvent_name: str = Field(..., description="Solvent name")
    solvent_smiles: str = Field(..., description="Solvent SMILES")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    method: Optional[str] = Field(None, description="Polymerization method")
    polytype: Optional[str] = Field(None, description="Polymerization type")
    source: Optional[str] = Field(None, description="DOI or original source")
    reaction_id: Optional[str] = Field(None, description="Reaction ID")
    
    class Config:
        populate_by_name = True


class PredictionOutput(BaseModel):
    """Output schema for prediction."""
    predicted_class: int = Field(..., description="Predicted class (0, 1, or 2)")
    class_probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    r_product_range: str = Field(..., description="Human-readable r-product range")
    class_descriptions: Dict[str, str] = Field(..., description="Description for each class label")
    models_agree: Optional[bool] = Field(None, description="Whether XGBoost and Lookup models agree on the prediction")
    below_threshold: bool = Field(False, description="Whether confidence is below the 0.7 threshold")
    lookup_class: Optional[int] = Field(None, description="Predicted class from the Lookup (nearest-neighbor) model")
    similar_papers: Optional[List[SimilarPaper]] = Field(None, description="Most similar papers from dataset")
    nearest_neighbors: Optional[List[NearestNeighbor]] = Field(None, description="Top 10 nearest data points from training database (baseline lookup)")
    solubility_issue: Optional[int] = Field(None, description="Solubility issue flag: 0 = no issues, 1 = has issues, -1 = unknown")
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


class SolventPreprocessOutput(BaseModel):
    """Output schema for solvent preprocessing."""
    solvent_smiles: Optional[str] = Field(None, description="SMILES representation of the solvent")
    features: Dict[str, Optional[float]] = Field(..., description="Calculated solvent features")
    success: bool = Field(..., description="Whether preprocessing was successful")
    error: Optional[str] = Field(None, description="Error message if preprocessing failed")


class MonomerPreprocessOutput(BaseModel):
    """Output schema for monomer preprocessing."""
    monomer_smiles: Optional[str] = Field(None, description="SMILES representation of the monomer")
    features: Dict[str, Optional[float]] = Field(..., description="Calculated monomer features")
    success: bool = Field(..., description="Whether preprocessing was successful")
    error: Optional[str] = Field(None, description="Error message if preprocessing failed")
    from_cache: bool = Field(..., description="Whether features were loaded from cache")


class PreprocessAllInput(BaseModel):
    """Input schema for combined preprocessing."""
    monomer1_smiles: str = Field(..., description="SMILES string of monomer 1")
    monomer2_smiles: str = Field(..., description="SMILES string of monomer 2")
    solvent_smiles: str = Field(..., description="SMILES string of solvent")
    method: str = Field(default='solvent', description="Polymerisation method")
    polytype: str = Field(default='free radical', description="Polymerisation type")
    temperature: float = Field(default=60.0, description="Temperature in Celsius")


class OptimizeReactionInput(BaseModel):
    """Input schema for reaction optimization."""
    monomer1_smiles: str = Field(..., description="SMILES string of monomer 1")
    monomer2_smiles: str = Field(..., description="SMILES string of monomer 2")
    solvent_smiles: str = Field(..., description="SMILES string of base solvent")
    method: str = Field(default='solvent', description="Polymerisation method")
    polytype: str = Field(default='free radical', description="Polymerisation type")
    temperature: float = Field(default=60.0, description="Base temperature in Celsius")
    temperature_step: float = Field(default=20.0, description="Temperature step size in Celsius (default: 20.0)")
    n_solvents: int = Field(default=3, description="Number of solvents to use (default: 3)")


class PreprocessAllOutput(BaseModel):
    """Output schema for combined preprocessing."""
    features: Dict[str, float] = Field(..., description="All calculated features ready for prediction")
    success: bool = Field(..., description="Whether preprocessing was successful")
    error: Optional[str] = Field(None, description="Error message if preprocessing failed")
    similar_papers: Optional[List[SimilarPaper]] = Field(None, description="Most similar papers from dataset")
    nearest_neighbors: Optional[List[NearestNeighbor]] = Field(None, description="Top 10 nearest data points from training database (baseline lookup)")
    lookup_class: Optional[int] = Field(None, description="Predicted class from the Lookup (nearest-neighbor) model (top-1 neighbor)")
    solubility_issue: Optional[int] = Field(None, description="Solubility issue flag: 0 = no issues, 1 = has issues, -1 = unknown")


class OptimizationPrediction(BaseModel):
    """Single prediction result in optimization grid."""
    temperature: float = Field(..., description="Temperature in Celsius")
    solvent_smiles: str = Field(..., description="Solvent SMILES")
    solvent_name: str = Field(..., description="Solvent name")
    solvent_logp: float = Field(..., description="Solvent logP value")
    predicted_class: int = Field(..., description="Predicted class (0, 1, or 2)")
    predicted_class_name: str = Field(..., description="Predicted class name: 'alternating', 'random to block like', or 'homopolymer'")
    class_probabilities: Dict[str, float] = Field(..., description="Probability for each class")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    solubility_issue: Optional[int] = Field(None, description="Solubility issue flag: 0 = no issues, 1 = has issues, -1 = unknown")


class OptimizeReactionOutput(BaseModel):
    """Output schema for reaction optimization."""
    success: bool = Field(..., description="Whether optimization was successful")
    error: Optional[str] = Field(None, description="Error message if optimization failed")
    predictions: List[OptimizationPrediction] = Field(..., description="3x3 grid of predictions (3 temperatures × 3 solvents)")
    base_temperature: float = Field(..., description="Base temperature used")
    temperature_step: float = Field(..., description="Temperature step size used")
    base_solvent_logp: float = Field(..., description="Base solvent logP value")
    timestamp: str = Field(..., description="Optimization timestamp")


class DOICheckInput(BaseModel):
    """Input schema for DOI check."""
    doi: str = Field(..., description="DOI to check (e.g., '10.1016/0014-3057(84)90010-7' or 'https://doi.org/10.1016/0014-3057(84)90010-7')")


class DOICheckOutput(BaseModel):
    """Output schema for DOI check."""
    doi: str = Field(..., description="The queried DOI")
    exists: bool = Field(..., description="Whether the DOI exists in the dataset")
    normalized_doi: str = Field(..., description="The normalized DOI used for matching")
    timestamp: str = Field(..., description="Check timestamp")


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

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[CopolymerPredictor] = None

# Model path (can be configured via environment variable)
MODEL_PATH = os.environ.get("MODEL_PATH", "../artifacts/model_bundle")

# Dataset path for DOI checking (can be configured via environment variable)
DATASET_PATH = os.environ.get("DATASET_PATH", "../processed_data.csv")

# Global embedding dictionaries
method_embeddings: Dict[str, Dict[str, float]] = {}
polytype_embeddings: Dict[str, Dict[str, float]] = {}

# Global dataset cache
dataset_df: Optional[pd.DataFrame] = None

# Global training data cache for baseline lookup
train_df: Optional[pd.DataFrame] = None

# Global fingerprint cache for baseline lookup
fingerprint_cache: Optional[Dict] = None

# Global solubility model
solubility_model = None


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model, embeddings, and dataset on startup."""
    global predictor, method_embeddings, polytype_embeddings, dataset_df, train_df, fingerprint_cache, solubility_model

    # Load embeddings
    try:
        api_dir = Path(__file__).parent
        method_emb_path = api_dir / "data" / "method_emb_pca_values.json"
        polytype_emb_path = api_dir / "data" / "polytype_emb_pca_values.json"

        if method_emb_path.exists():
            try:
                with open(method_emb_path, 'r') as f:
                    method_embeddings = json.load(f)
                print(f"✓ Loaded {len(method_embeddings)} method embeddings")
            except json.JSONDecodeError as e:
                print(f"✗ Error parsing method embeddings JSON: {e}")
                method_embeddings = {}
            except Exception as e:
                print(f"✗ Error loading method embeddings: {e}")
                method_embeddings = {}
        else:
            print(f"⚠ Warning: Method embeddings file not found at {method_emb_path}")
            method_embeddings = {}

        if polytype_emb_path.exists():
            try:
                with open(polytype_emb_path, 'r') as f:
                    polytype_embeddings = json.load(f)
                print(f"✓ Loaded {len(polytype_embeddings)} polytype embeddings")
            except json.JSONDecodeError as e:
                print(f"✗ Error parsing polytype embeddings JSON: {e}")
                polytype_embeddings = {}
            except Exception as e:
                print(f"✗ Error loading polytype embeddings: {e}")
                polytype_embeddings = {}
        else:
            print(f"⚠ Warning: Polytype embeddings file not found at {polytype_emb_path}")
            polytype_embeddings = {}
    except Exception as e:
        print(f"✗ Error loading embeddings: {e}")
        method_embeddings = {}
        polytype_embeddings = {}

    # Load model
    try:
        # Resolve relative path from api directory
        model_path = Path(__file__).parent / MODEL_PATH
        print(f"Loading model from {model_path}...")
        predictor = CopolymerPredictor(str(model_path))
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("API will start but predictions will fail until model is loaded")

    # Load dataset for DOI checking
    try:
        # Resolve relative path from api directory
        dataset_path = Path(__file__).parent / DATASET_PATH
        if dataset_path.exists():
            print(f"Loading dataset from {dataset_path}...")
            dataset_df = pd.read_csv(dataset_path)
            print(f"✓ Dataset loaded successfully ({len(dataset_df)} rows)")
        else:
            print(f"⚠ Warning: Dataset file not found at {dataset_path}")
            print("DOI checking will not be available")
            dataset_df = None
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("DOI checking will not be available")
        dataset_df = None

    # Load training data for baseline lookup
    if BASELINE_LOOKUP_AVAILABLE:
        try:
            # Try to load from data splits
            api_dir = Path(__file__).parent
            split_dir = api_dir.parent / "artifacts" / "data_splits"
            train_path = split_dir / "train.csv"
            
            if train_path.exists():
                print(f"Loading training data from {train_path}...")
                train_df = pd.read_csv(train_path)
                print(f"✓ Training data loaded successfully ({len(train_df)} rows)")

                # Add negative data to lookup pool
                neg_path = api_dir.parent / "filter" / "artificial_datapoints" / "processed_combined_augmented.csv"
                if neg_path.exists():
                    df_neg = pd.read_csv(neg_path)
                    if 'Class' in df_neg.columns:
                        df_neg = df_neg.rename(columns={'Class': 'r_product_class'})
                    df_neg['r_product_class'] = df_neg['r_product_class'].astype(int)
                    train_df = pd.concat([train_df, df_neg], ignore_index=True)
                    print(f"✓ Added {len(df_neg)} negative data points to lookup pool ({len(train_df)} total)")
                else:
                    print(f"⚠ Warning: Negative data not found at {neg_path}")
                
                # Precompute fingerprints for all unique SMILES in training data
                if BASELINE_LOOKUP_AVAILABLE:
                    try:
                        from baseline_lookup import load_fingerprint_cache, save_fingerprint_cache, compute_fingerprints_for_smiles
                        
                        print("Loading fingerprint cache...")
                        fingerprint_cache = load_fingerprint_cache()
                        
                        # Get all unique SMILES from training data
                        unique_monomer1 = set(train_df['monomer1_smiles'].dropna().unique())
                        unique_monomer2 = set(train_df['monomer2_smiles'].dropna().unique())
                        unique_solvents = set(train_df['solvent_smiles'].dropna().unique())
                        all_unique_smiles = list(unique_monomer1 | unique_monomer2 | unique_solvents)
                        
                        # Compute fingerprints for any missing SMILES
                        if fingerprint_cache is None:
                            print("Fingerprint cache not found. Computing fingerprints for all SMILES...")
                            fingerprint_cache = compute_fingerprints_for_smiles(all_unique_smiles)
                            save_fingerprint_cache(fingerprint_cache)
                            print(f"✓ Computed and cached fingerprints for {len(fingerprint_cache)} SMILES")
                        else:
                            # Check if we need to compute any missing fingerprints
                            missing_smiles = [s for s in all_unique_smiles if s not in fingerprint_cache]
                            if missing_smiles:
                                print(f"Computing fingerprints for {len(missing_smiles)} missing SMILES...")
                                new_fps = compute_fingerprints_for_smiles(missing_smiles, cache_dict=fingerprint_cache)
                                fingerprint_cache.update(new_fps)
                                save_fingerprint_cache(fingerprint_cache)
                                print(f"✓ Updated cache with {len(missing_smiles)} new fingerprints")
                            else:
                                print(f"✓ Fingerprint cache loaded ({len(fingerprint_cache)} entries)")
                    except Exception as e:
                        print(f"⚠ Warning: Failed to load/precompute fingerprint cache: {e}")
                        import traceback
                        traceback.print_exc()
                        fingerprint_cache = None
            else:
                print(f"⚠ Warning: Training data not found at {train_path}")
                print("Baseline lookup will not be available")
                train_df = None
        except Exception as e:
            print(f"✗ Error loading training data: {e}")
            print("Baseline lookup will not be available")
            train_df = None
    else:
        train_df = None

    # Load solubility model
    if SOLUBILITY_CHECK_AVAILABLE:
        try:
            solubility_model = load_solubility_model()
        except Exception as e:
            print(f"✗ Error loading solubility model: {e}")
            print("Solubility check will not be available")
            solubility_model = None
    else:
        solubility_model = None


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
        # Static mapping from classes to English text descriptions
        class_descriptions_map = {
            0: "alternating",
            1: "random to block like",
            2: "homopolymer",
        }

        # Make XGBoost prediction
        results = predictor.predict_with_confidence(input_data.features)

        # Format output
        pred_class = int(results['predictions'][0])
        proba = results['probabilities'][0]
        confidence = float(results['confidence'][0])

        # Voting: Lookup prediction via nearest-neighbor
        nearest_neighbors_list = None
        lookup_class_value = None
        models_agree_flag = None

        smiles_available = (
            input_data.monomer1_smiles
            and input_data.monomer2_smiles
            and input_data.solvent_smiles
        )

        if smiles_available and BASELINE_LOOKUP_AVAILABLE and train_df is not None:
            try:
                global fingerprint_cache
                fp_dict_to_use = fingerprint_cache if fingerprint_cache is not None else None
                neighbors = find_top_k_nearest_neighbors(
                    test_monomer1_smiles=input_data.monomer1_smiles,
                    test_monomer2_smiles=input_data.monomer2_smiles,
                    test_solvent_smiles=input_data.solvent_smiles,
                    df_train=train_df,
                    k=10,
                    fp_dict=fp_dict_to_use
                )
                if neighbors:
                    nearest_neighbors_list = [
                        NearestNeighbor(**neighbor) for neighbor in neighbors
                    ]
                    lookup_class_value = int(neighbors[0]['predicted_class'])
                    models_agree_flag = (pred_class == lookup_class_value)
            except Exception as e:
                print(f"Warning: Lookup prediction failed: {e}")

        below_threshold_flag = confidence < 0.7

        # Solubility check: Only if SMILES are provided
        solubility_issue_flag = None
        if SOLUBILITY_CHECK_AVAILABLE and solubility_model is not None:
            if smiles_available:
                try:
                    solubility_issue_flag = get_solubility_issue_flag(
                        monomer1_smiles=input_data.monomer1_smiles,
                        monomer2_smiles=input_data.monomer2_smiles,
                        solvent_smiles=input_data.solvent_smiles,
                        model=solubility_model
                    )
                except Exception as e:
                    print(f"Warning: Solubility check failed: {e}")
                    solubility_issue_flag = None

        return PredictionOutput(
            predicted_class=pred_class,
            class_probabilities={
                f"class_{i}": float(proba[i])
                for i in range(len(proba))
            },
            confidence=confidence,
            r_product_range=class_descriptions_map[pred_class],
            class_descriptions={
                f"class_{i}": desc
                for i, desc in class_descriptions_map.items()
            },
            models_agree=models_agree_flag,
            below_threshold=below_threshold_flag,
            lookup_class=lookup_class_value,
            nearest_neighbors=nearest_neighbors_list,
            solubility_issue=solubility_issue_flag,
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
        # Static mapping from classes to English text descriptions
        class_descriptions_map = {
            0: "alternating",
            1: "random to block like",
            2: "homopolymer",
        }
        
        predictions = []

        for sample in input_data.samples:
            results = predictor.predict_with_confidence(sample)

            pred_class = int(results['predictions'][0])
            proba = results['probabilities'][0]
            confidence = float(results['confidence'][0])
            
            # Solubility check not available for batch predictions (no SMILES)
            solubility_issue_flag = None

            predictions.append(PredictionOutput(
                predicted_class=pred_class,
                class_probabilities={
                    f"class_{i}": float(proba[i])
                    for i in range(len(proba))
                },
                confidence=confidence,
                r_product_range=class_descriptions_map[pred_class],
                class_descriptions={
                    f"class_{i}": desc
                    for i, desc in class_descriptions_map.items()
                },
                nearest_neighbors=None,
                solubility_issue=solubility_issue_flag,
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


@app.get("/features")
async def get_required_features():
    """Get list of required features for prediction."""
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    try:
        features = predictor.features if predictor.features else []
        return {
            "required_features": features,
            "n_features": len(features) if features else 0
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving features: {str(e)}"
        )


@app.post("/preprocess_all", response_model=PreprocessAllOutput)
async def preprocess_all(input_data: PreprocessAllInput):
    """
    Preprocess all inputs and return features ready for prediction.
    Combines monomer preprocessing, solvent preprocessing, and embeddings.
    """
    try:
        # Preprocess monomers
        base_path = Path(__file__).parent / "molecule_properties"

        m1_data = load_monomer_features(input_data.monomer1_smiles, base_path)
        if not m1_data:
            return PreprocessAllOutput(
                features={},
                success=False,
                error="Monomer 1 features not found in cache"
            )
        m1_features = extract_monomer_features_for_model(m1_data)

        m2_data = load_monomer_features(input_data.monomer2_smiles, base_path)
        if not m2_data:
            return PreprocessAllOutput(
                features={},
                success=False,
                error="Monomer 2 features not found in cache"
            )
        m2_features = extract_monomer_features_for_model(m2_data)

        # Preprocess solvent
        solvent_features = calculate_solvent_features(input_data.solvent_smiles)
        if any(v is None for v in solvent_features.values()):
            return PreprocessAllOutput(
                features={},
                success=False,
                error="Failed to calculate solvent features"
            )

        # Get embeddings
        if input_data.method not in method_embeddings:
            return PreprocessAllOutput(
                features={},
                success=False,
                error=f"Method '{input_data.method}' not found in embeddings"
            )
        method_emb = method_embeddings[input_data.method]

        if input_data.polytype not in polytype_embeddings:
            return PreprocessAllOutput(
                features={},
                success=False,
                error=f"Polytype '{input_data.polytype}' not found in embeddings"
            )
        polytype_emb = polytype_embeddings[input_data.polytype]

        # Combine all features - include all features that the model might need
        # Always include all features, even if None (model will handle NaN filling)
        features = {
            # Monomer 1 features
            "fukui_radical_max_1": m1_features.get("fukui_radical_max"),
            "global_electrophilicity_1": m1_features.get("global_electrophilicity"),
            "global_nucleophilicity_1": m1_features.get("global_nucleophilicity"),
            "dipole_x_1": m1_features.get("dipole_x"),
            "dipole_y_1": m1_features.get("dipole_y"),
            "dipole_z_1": m1_features.get("dipole_z"),
            
            # Monomer 2 features
            "fukui_radical_max_2": m2_features.get("fukui_radical_max"),
            "global_electrophilicity_2": m2_features.get("global_electrophilicity"),
            "global_nucleophilicity_2": m2_features.get("global_nucleophilicity"),
            "dipole_x_2": m2_features.get("dipole_x"),
            "dipole_y_2": m2_features.get("dipole_y"),
            "dipole_z_2": m2_features.get("dipole_z"),
            
            # HOMO-LUMO differences
            "delta_HOMO_LUMO_AA": (m1_features.get("homo") - m1_features.get("lumo")) 
                                   if (m1_features.get("homo") is not None and m1_features.get("lumo") is not None) 
                                   else None,
            "delta_HOMO_LUMO_AB": (m1_features.get("homo") - m2_features.get("lumo")) 
                                   if (m1_features.get("homo") is not None and m2_features.get("lumo") is not None) 
                                   else None,
            "delta_HOMO_LUMO_BB": (m2_features.get("homo") - m2_features.get("lumo")) 
                                   if (m2_features.get("homo") is not None and m2_features.get("lumo") is not None) 
                                   else None,
            "delta_HOMO_LUMO_BA": (m2_features.get("homo") - m1_features.get("lumo")) 
                                   if (m2_features.get("homo") is not None and m1_features.get("lumo") is not None) 
                                   else None,
            
            # Other features
            "temperature": input_data.temperature,
            "polytype_emb_1": polytype_emb["pca_1"],
            "polytype_emb_2": polytype_emb["pca_2"],
            "method_emb_1": method_emb["pca_1"],
            "method_emb_2": method_emb["pca_2"],
            "solvent_logP": solvent_features["solvent_logP"],
            "solvent_TPSA": solvent_features["solvent_TPSA"],
            "solvent_HBD": solvent_features["solvent_HBD"],
            "solvent_FractionCSP3": solvent_features["solvent_FractionCSP3"]
        }
        
        # If model is loaded, ensure all required features are present
        # Fill missing features with None if model requires them
        if predictor and predictor.features:
            for required_feature in predictor.features:
                if required_feature not in features:
                    features[required_feature] = None

        # Find similar papers if dataset is available
        similar_papers_list = None
        if SIMILARITY_AVAILABLE and dataset_df is not None:
            try:
                method_emb_tuple = (method_emb["pca_1"], method_emb["pca_2"])
                polytype_emb_tuple = (polytype_emb["pca_1"], polytype_emb["pca_2"])
                
                similar_papers = find_similar_papers(
                    dataset_df,
                    input_data.monomer1_smiles,
                    input_data.monomer2_smiles,
                    input_data.solvent_smiles,
                    input_data.temperature,
                    method_emb_tuple,
                    polytype_emb_tuple,
                    top_n=10
                )
                
                similar_papers_list = format_similarity_output(similar_papers)
            except Exception as e:
                print(f"Warning: Failed to find similar papers: {e}")
                similar_papers_list = None

        # Find nearest neighbors using baseline lookup
        nearest_neighbors_list = None
        lookup_class_value = None
        if BASELINE_LOOKUP_AVAILABLE and train_df is not None:
            try:
                # Use precomputed fingerprint cache if available
                global fingerprint_cache
                fp_dict_to_use = fingerprint_cache if fingerprint_cache is not None else None
                neighbors = find_top_k_nearest_neighbors(
                    test_monomer1_smiles=input_data.monomer1_smiles,
                    test_monomer2_smiles=input_data.monomer2_smiles,
                    test_solvent_smiles=input_data.solvent_smiles,
                    df_train=train_df,
                    k=10,
                    fp_dict=fp_dict_to_use
                )
                
                # Convert to Pydantic models
                if neighbors:
                    print(f"✓ Found {len(neighbors)} nearest neighbors")
                    nearest_neighbors_list = [
                        NearestNeighbor(**neighbor) for neighbor in neighbors
                    ]
                    lookup_class_value = int(neighbors[0]['predicted_class'])
                else:
                    nearest_neighbors_list = []
                    print(f"⚠ Warning: find_top_k_nearest_neighbors returned empty list")
                    print(f"  train_df size: {len(train_df) if train_df is not None else 'None'}")
                    print(f"  fingerprint_cache size: {len(fingerprint_cache) if fingerprint_cache is not None else 'None'}")
                    print(f"  test SMILES: m1={input_data.monomer1_smiles[:50]}, m2={input_data.monomer2_smiles[:50]}, s={input_data.solvent_smiles[:50]}")
            except Exception as e:
                print(f"✗ Error: Failed to find nearest neighbors: {e}")
                import traceback
                traceback.print_exc()
                nearest_neighbors_list = None

        # Check solubility
        solubility_issue_flag = None
        if SOLUBILITY_CHECK_AVAILABLE and solubility_model is not None:
            try:
                solubility_issue_flag = get_solubility_issue_flag(
                    monomer1_smiles=input_data.monomer1_smiles,
                    monomer2_smiles=input_data.monomer2_smiles,
                    solvent_smiles=input_data.solvent_smiles,
                    model=solubility_model
                )
            except Exception as e:
                print(f"Warning: Solubility check failed: {e}")
                solubility_issue_flag = None

        # Clean all data to remove NaN/Inf values before JSON serialization
        # For features dict, replace NaN/Inf with 0.0 (model expects float, not None)
        # For similar_papers, also replace with 0.0 since schema expects float
        cleaned_features = clean_json_values(features, replace_with_zero=True)
        cleaned_similar_papers = clean_json_values(similar_papers_list, replace_with_zero=True) if similar_papers_list else None
        cleaned_nearest_neighbors = clean_json_values(nearest_neighbors_list) if nearest_neighbors_list else None
        
        return PreprocessAllOutput(
            features=cleaned_features,
            success=True,
            error=None,
            similar_papers=cleaned_similar_papers,
            nearest_neighbors=cleaned_nearest_neighbors,
            lookup_class=lookup_class_value,
            solubility_issue=solubility_issue_flag
        )

    except Exception as e:
        import traceback
        error_detail = f"Preprocessing failed: {str(e)}"
        print(f"Error in preprocess_all: {error_detail}")
        traceback.print_exc()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_detail
        )


def calculate_solvent_features(smiles: str) -> Dict[str, Optional[float]]:
    """
    Calculate solvent features from SMILES string.
    Returns only the features needed by the model:
    - solvent_logP
    - solvent_TPSA
    - solvent_HBD
    - solvent_FractionCSP3
    """

    def is_invalid(smiles):
        if not smiles or not isinstance(smiles, str):
            return True
        smiles_clean = smiles.strip().lower()
        return smiles_clean in {"", "na", "nan", "none"}

    if is_invalid(smiles):
        return {
            "solvent_logP": None,
            "solvent_TPSA": None,
            "solvent_HBD": None,
            "solvent_FractionCSP3": None
        }

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "solvent_logP": None,
                "solvent_TPSA": None,
                "solvent_HBD": None,
                "solvent_FractionCSP3": None
            }

        return {
            "solvent_logP": float(Descriptors.MolLogP(mol)),
            "solvent_TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
            "solvent_HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
            "solvent_FractionCSP3": float(Descriptors.FractionCSP3(mol))
        }
    except Exception as e:
        return {
            "solvent_logP": None,
            "solvent_TPSA": None,
            "solvent_HBD": None,
            "solvent_FractionCSP3": None
        }


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize smiles using RDKit (local copy without cache to avoid SQLite issues)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return Chem.MolToSmiles(mol)


class SolventPreprocessInput(BaseModel):
    """Input schema for solvent preprocessing."""
    solvent_smiles: str = Field(..., description="SMILES string of the solvent")


@app.post("/preprocess/solvent", response_model=SolventPreprocessOutput)
async def preprocess_solvent(input_data: SolventPreprocessInput):
    """
    Preprocess a solvent: take SMILES and calculate features.

    Args:
        solvent_smiles: Solvent SMILES string

    Returns:
        SMILES representation and calculated features
    """
    try:
        solvent_smiles = input_data.solvent_smiles

        # Calculate features
        features = calculate_solvent_features(solvent_smiles)

        # Check if any features are None (indicating calculation failure)
        if any(v is None for v in features.values()):
            return SolventPreprocessOutput(
                solvent_smiles=solvent_smiles,
                features=features,
                success=False,
                error="Failed to calculate some features from SMILES"
            )

        return SolventPreprocessOutput(
            solvent_smiles=solvent_smiles,
            features=features,
            success=True,
            error=None
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Solvent preprocessing failed: {str(e)}"
        )


@app.get("/embeddings/method/{method_name}", response_model=Dict[str, Optional[float]])
async def get_method_embeddings(method_name: str):
    """
    Get PCA-reduced embeddings for a specific method string.

    Args:
        method_name: The method string to get embeddings for

    Returns:
        Dictionary with pca_1 and pca_2 values
    """
    if method_name not in method_embeddings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Method '{method_name}' not found."
        )

    return method_embeddings[method_name]


@app.get("/embeddings/polytype/{polytype_name}", response_model=Dict[str, Optional[float]])
async def get_polytype_embeddings(polytype_name: str):
    """
    Get PCA-reduced embeddings for a specific polymerization type string.

    Args:
        polytype_name: The polytype string to get embeddings for

    Returns:
        Dictionary with pca_1 and pca_2 values
    """
    if polytype_name not in polytype_embeddings:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Polytype '{polytype_name}' not found."
        )

    return polytype_embeddings[polytype_name]


def get_smiles_md5(smiles: str) -> str:
    """Create MD5 hash from SMILES string for consistent filename."""
    return hashlib.md5(smiles.encode('utf-8')).hexdigest()


def load_monomer_features(smiles: str, base_path: Optional[Path] = None) -> Optional[Dict]:
    """
    Load monomer features from JSON file if it exists.
    Uses MD5 hash of SMILES as filename. Falls back to canonical SMILES matching
    for backward compatibility with old files.
    Returns None if no match is found.
    """
    if base_path is None:
        base_path = Path(__file__).parent / "molecule_properties"
    elif isinstance(base_path, str):
        base_path = Path(base_path)

    # 1) Fast path: MD5 hash lookup
    md5_hash = get_smiles_md5(smiles)
    file_path = base_path / f"{md5_hash}.json"

    if file_path.exists():
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            for key in ['charges', 'fukui_electrophilicity', 'fukui_nucleophilicity', 'fukui_radical']:
                if key in data and isinstance(data[key], dict) and data[key]:
                    data[key + '_min'] = min(data[key].values())
                    data[key + '_max'] = max(data[key].values())
                    data[key + '_mean'] = sum(data[key].values()) / len(data[key].values())

            return data
        except Exception as e:
            print(f"Error loading monomer features from {file_path}: {e}")
            return None

    # 2) Fallback: canonical SMILES-based lookup (for backward compatibility)
    try:
        target_canonical = canonicalize_smiles(smiles)
    except Exception as e:
        print(f"Warning: failed to canonicalize SMILES '{smiles}': {e}")
        return None

    try:
        for candidate_path in base_path.glob("*.json"):
            try:
                with open(candidate_path, 'r') as f:
                    candidate_data = json.load(f)
            except Exception:
                continue

            stored_smiles = candidate_data.get("smiles")
            if not stored_smiles:
                continue

            try:
                if canonicalize_smiles(stored_smiles) == target_canonical:
                    # Found a match; post-process fields as above
                    for key in ['charges', 'fukui_electrophilicity', 'fukui_nucleophilicity', 'fukui_radical']:
                        if key in candidate_data and isinstance(candidate_data[key], dict) and candidate_data[key]:
                            candidate_data[key + '_min'] = min(candidate_data[key].values())
                            candidate_data[key + '_max'] = max(candidate_data[key].values())
                            candidate_data[key + '_mean'] = sum(candidate_data[key].values()) / len(
                                candidate_data[key].values())
                    return candidate_data
            except Exception:
                continue
    except Exception as e:
        print(f"Warning: error while searching canonical match for '{smiles}': {e}")

    return None


def extract_monomer_features_for_model(data: Dict) -> Dict[str, Optional[float]]:
    """
    Extract all features needed for the model from the full molecular data.
    Returns: fukui_radical_max, global_electrophilicity, global_nucleophilicity, 
             dipole (x, y, z), homo, lumo
    """
    # Extract dipole components (dipole is a list [x, y, z])
    dipole = data.get("dipole")
    dipole_x = dipole[0] if isinstance(dipole, list) and len(dipole) > 0 else None
    dipole_y = dipole[1] if isinstance(dipole, list) and len(dipole) > 1 else None
    dipole_z = dipole[2] if isinstance(dipole, list) and len(dipole) > 2 else None
    
    features = {
        "fukui_radical_max": data.get("fukui_radical_max"),
        "global_electrophilicity": data.get("global_electrophilicity"),
        "global_nucleophilicity": data.get("global_nucleophilicity"),
        "dipole_x": dipole_x,
        "dipole_y": dipole_y,
        "dipole_z": dipole_z,
        "homo": data.get("homo"),
        "lumo": data.get("lumo")
    }
    return features


def calculate_monomer_features(smiles: str, base_path: Optional[Path] = None) -> Dict:
    """
    Calculate monomer features using morfeus (same as monomer_feature_calculation.py).
    This is a long-running operation and should be used asynchronously.
    """
    if not MORFEUS_AVAILABLE:
        raise RuntimeError("morfeus is not available. Cannot calculate monomer features.")

    if base_path is None:
        base_path = Path(__file__).parent / "molecule_properties"
    elif isinstance(base_path, str):
        base_path = Path(base_path)

    from morfeus.conformer import ConformerEnsemble

    # Optimize conformer
    ce = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94")
    ce.prune_rmsd()
    ce.sort()

    try:
        ce.optimize_qc_engine(
            program="xtb", model={"method": "GFN-FF"}, procedure="geometric"
        )
    except Exception as e:
        print(f"GFN-FF optimization failed for {smiles}: {e}")
        ce = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94")
        ce.prune_rmsd()
        ce.sort()

    try:
        ce.optimize_qc_engine(
            program="xtb", model={"method": "GFN2-xTB"}, procedure="geometric"
        )
    except Exception as e:
        print(f"GFN2-xTB optimization failed for {smiles}: {e}")

    ce.sp_qc_engine(program="xtb", model={"method": "GFN2-xTB"})
    best_conformer = ce.conformers[0]

    elements = best_conformer.elements.tolist()
    coordinates = best_conformer.coordinates.tolist()
    energy = best_conformer.energy

    # Calculate properties
    xtb = XTB(elements, coordinates)

    properties = {
        "smiles": smiles,
        "best_conformer_coordinates": coordinates,
        "best_conformer_elements": elements,
        "best_conformer_energy": energy,
        "ip": xtb.get_ip(),
        "ip_corrected": xtb.get_ip(corrected=True),
        "ea": xtb.get_ea(),
        "homo": xtb.get_homo(),
        "lumo": xtb.get_lumo(),
        "charges": xtb.get_charges(),
        "dipole": xtb.get_dipole().tolist(),
        "global_electrophilicity": xtb.get_global_descriptor("electrophilicity", corrected=True),
        "global_nucleophilicity": xtb.get_global_descriptor("nucleophilicity", corrected=True),
        "fukui_electrophilicity": xtb.get_fukui("electrophilicity"),
        "fukui_nucleophilicity": xtb.get_fukui("nucleophilicity"),
        "fukui_radical": xtb.get_fukui("radical"),
    }

    # Process dict fields
    for key in ['charges', 'fukui_electrophilicity', 'fukui_nucleophilicity', 'fukui_radical']:
        if key in properties and isinstance(properties[key], dict) and properties[key]:
            properties[key + '_min'] = min(properties[key].values())
            properties[key + '_max'] = max(properties[key].values())
            properties[key + '_mean'] = sum(properties[key].values()) / len(properties[key].values())

    # Save to file
    if base_path is None:
        base_path = Path(__file__).parent / "molecule_properties"

    base_path.mkdir(exist_ok=True)
    md5_hash = get_smiles_md5(smiles)
    file_path = base_path / f"{md5_hash}.json"

    with open(file_path, 'w') as f:
        json.dump(properties, f, indent=4)

    return properties


class MonomerPreprocessInput(BaseModel):
    """Input schema for monomer preprocessing."""
    monomer_smiles: str = Field(..., description="SMILES string of the monomer")


@app.post("/preprocess/monomer", response_model=MonomerPreprocessOutput)
async def preprocess_monomer(input_data: MonomerPreprocessInput):
    """
    Preprocess a monomer: take SMILES, check for existing features,
    or calculate new features if needed.

    Args:
        monomer_smiles: Monomer SMILES string

    Returns:
        SMILES representation and calculated features
    """
    try:
        monomer_smiles = input_data.monomer_smiles

        # Try to load existing features (direct lookup or canonical SMILES match)
        base_path = Path(__file__).parent / "molecule_properties"
        existing_data = load_monomer_features(monomer_smiles, base_path)

        if existing_data is not None:
            # Features found in cache
            features = extract_monomer_features_for_model(existing_data)
            # Return only the minimal features for backward compatibility
            minimal_features = {
                "fukui_radical_max": features.get("fukui_radical_max"),
                "homo": features.get("homo"),
                "lumo": features.get("lumo")
            }
            return MonomerPreprocessOutput(
                monomer_smiles=monomer_smiles,
                features=minimal_features,
                success=True,
                error=None,
                from_cache=True
            )

        # Features not found, need to calculate
        if not MORFEUS_AVAILABLE:
            return MonomerPreprocessOutput(
                monomer_smiles=monomer_smiles,
                features={
                    "fukui_radical_max": None,
                    "homo": None,
                    "lumo": None
                },
                success=False,
                error="morfeus is not available. Cannot calculate monomer features.",
                from_cache=False
            )

        # Calculate features (this is a long-running operation)
        # Note: In production, you might want to run this in a background task
        calculated_data = calculate_monomer_features(monomer_smiles, base_path)
        all_features = extract_monomer_features_for_model(calculated_data)
        # Return only the minimal features for backward compatibility
        minimal_features = {
            "fukui_radical_max": all_features.get("fukui_radical_max"),
            "homo": all_features.get("homo"),
            "lumo": all_features.get("lumo")
        }

        return MonomerPreprocessOutput(
            monomer_smiles=monomer_smiles,
            features=minimal_features,
            success=True,
            error=None,
            from_cache=False
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Monomer preprocessing failed: {str(e)}"
        )


@app.post("/check_doi", response_model=DOICheckOutput)
async def check_doi(input_data: DOICheckInput):
    """
    Check if a given DOI exists in the dataset.
    
    Args:
        input_data: DOICheckInput containing the DOI to check
        
    Returns:
        DOICheckOutput with existence status and normalized DOI
    """
    if dataset_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dataset not loaded. DOI checking is not available."
        )
    
    try:
        # Normalize the DOI for comparison
        doi_input = input_data.doi.strip()
        
        # Remove common prefixes if present
        if doi_input.startswith("https://doi.org/"):
            normalized_doi = doi_input.replace("https://doi.org/", "")
        elif doi_input.startswith("http://doi.org/"):
            normalized_doi = doi_input.replace("http://doi.org/", "")
        elif doi_input.startswith("doi.org/"):
            normalized_doi = doi_input.replace("doi.org/", "")
        else:
            normalized_doi = doi_input
            
        # Check if the DOI exists in the dataset
        # The dataset stores DOIs in the 'original_source' column
        # They can be in various formats, so we check for multiple formats
        exists = False
        
        if 'original_source' in dataset_df.columns:
            # Check for exact match with normalized DOI
            exists = dataset_df['original_source'].astype(str).str.contains(
                normalized_doi, 
                case=False, 
                na=False,
                regex=False
            ).any()
        
        return DOICheckOutput(
            doi=doi_input,
            exists=exists,
            normalized_doi=normalized_doi,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"DOI check failed: {str(e)}"
        )


@app.post("/optimize_reaction", response_model=OptimizeReactionOutput)
async def optimize_reaction(input_data: OptimizeReactionInput):
    """
    Perform reaction optimization by exploring different solvent and temperature combinations.
    
    Creates a 3x3 grid of predictions:
    - 3 temperatures: base_temp - step, base_temp, base_temp + step
    - 3 solvents: similar solvents based on logP from the dataset
    
    Args:
        input_data: OptimizeReactionInput with monomers, base solvent, temperature, etc.
        
    Returns:
        OptimizeReactionOutput with 3x3 grid of predictions
    """
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    if not REACTION_OPTIMIZATION_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reaction optimization module not available"
        )
    
    if dataset_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dataset not loaded. Reaction optimization requires dataset for finding similar solvents."
        )
    
    try:
        # Get base solvent logP
        base_solvent_features = calculate_solvent_features(input_data.solvent_smiles)
        base_logp = base_solvent_features.get('solvent_logP')
        
        if base_logp is None:
            raise ValueError(f"Could not determine logP for solvent: {input_data.solvent_smiles}")
        
        # Prepare solubility check function
        solubility_check_func = None
        if SOLUBILITY_CHECK_AVAILABLE and solubility_model is not None:
            def check_solubility(monomer1_smiles, monomer2_smiles, solvent_smiles):
                return get_solubility_issue_flag(
                    monomer1_smiles=monomer1_smiles,
                    monomer2_smiles=monomer2_smiles,
                    solvent_smiles=solvent_smiles,
                    model=solubility_model
                )
            solubility_check_func = check_solubility
        
        # Create optimization grid
        predictions = create_optimization_grid(
            monomer1_smiles=input_data.monomer1_smiles,
            monomer2_smiles=input_data.monomer2_smiles,
            base_solvent_smiles=input_data.solvent_smiles,
            base_temperature=input_data.temperature,
            method=input_data.method,
            polytype=input_data.polytype,
            dataset_df=dataset_df,
            method_embeddings=method_embeddings,
            polytype_embeddings=polytype_embeddings,
            predictor=predictor,
            load_monomer_features_func=load_monomer_features,
            extract_monomer_features_func=extract_monomer_features_for_model,
            calculate_solvent_features_func=calculate_solvent_features,
            temperature_step=input_data.temperature_step,
            n_solvents=input_data.n_solvents,
            solubility_check_func=solubility_check_func
        )
        
        if not predictions:
            return OptimizeReactionOutput(
                success=False,
                error="No valid predictions could be generated",
                predictions=[],
                base_temperature=input_data.temperature,
                temperature_step=input_data.temperature_step,
                base_solvent_logp=float(base_logp),
                timestamp=datetime.now().isoformat()
            )
        
        # Clean predictions to remove NaN/Inf values before JSON serialization
        # Replace NaN/Inf with 0.0 for numeric fields
        cleaned_predictions = clean_json_values(predictions, replace_with_zero=True)
        
        # Convert to Pydantic models
        prediction_models = [
            OptimizationPrediction(**pred) for pred in cleaned_predictions
        ]
        
        return OptimizeReactionOutput(
            success=True,
            error=None,
            predictions=prediction_models,
            base_temperature=input_data.temperature,
            temperature_step=input_data.temperature_step,
            base_solvent_logp=float(base_logp),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Reaction optimization failed: {str(e)}"
        )


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
