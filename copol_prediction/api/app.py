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
from copolpredictor.data_processing import load_molecular_data
from copolextractor.utils import name_to_smiles
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# Import monomer feature calculation functions
try:
    from morfeus.conformer import ConformerEnsemble
    import qcengine
    
    # Try to import patched XTB class, fall back to original if not available
    try:
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


class SolventPreprocessInput(BaseModel):
    """Input schema for solvent preprocessing."""
    solvent_name: str = Field(..., description="Name of the solvent")


class SolventPreprocessOutput(BaseModel):
    """Output schema for solvent preprocessing."""
    solvent_name: str = Field(..., description="Input solvent name")
    solvent_smiles: Optional[str] = Field(None, description="SMILES representation of the solvent")
    features: Dict[str, Optional[float]] = Field(..., description="Calculated solvent features")
    success: bool = Field(..., description="Whether preprocessing was successful")
    error: Optional[str] = Field(None, description="Error message if preprocessing failed")


class MonomerPreprocessInput(BaseModel):
    """Input schema for monomer preprocessing."""
    monomer_name: str = Field(..., description="Name of the monomer")


class MonomerPreprocessOutput(BaseModel):
    """Output schema for monomer preprocessing."""
    monomer_name: str = Field(..., description="Input monomer name")
    monomer_smiles: Optional[str] = Field(None, description="SMILES representation of the monomer")
    features: Dict[str, Optional[float]] = Field(..., description="Calculated monomer features")
    success: bool = Field(..., description="Whether preprocessing was successful")
    error: Optional[str] = Field(None, description="Error message if preprocessing failed")
    from_cache: bool = Field(..., description="Whether features were loaded from cache")


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

# Global embedding dictionaries
method_embeddings: Dict[str, Dict[str, float]] = {}
polytype_embeddings: Dict[str, Dict[str, float]] = {}


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Load model and embeddings on startup."""
    global predictor, method_embeddings, polytype_embeddings
    
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


@app.post("/preprocess/solvent", response_model=SolventPreprocessOutput)
async def preprocess_solvent(input_data: SolventPreprocessInput):
    """
    Preprocess a solvent: convert name to SMILES and calculate features.
    
    Args:
        input_data: Solvent name
        
    Returns:
        SMILES representation and calculated features
    """
    try:
        # Convert name to SMILES
        solvent_smiles = name_to_smiles(input_data.solvent_name, force_retry=True)
        
        if solvent_smiles is None:
            return SolventPreprocessOutput(
                solvent_name=input_data.solvent_name,
                solvent_smiles=None,
                features={
                    "solvent_logP": None,
                    "solvent_TPSA": None,
                    "solvent_HBD": None,
                    "solvent_FractionCSP3": None
                },
                success=False,
                error=f"Could not convert solvent name '{input_data.solvent_name}' to SMILES"
            )
        
        # Calculate features
        features = calculate_solvent_features(solvent_smiles)
        
        # Check if any features are None (indicating calculation failure)
        if any(v is None for v in features.values()):
            return SolventPreprocessOutput(
                solvent_name=input_data.solvent_name,
                solvent_smiles=solvent_smiles,
                features=features,
                success=False,
                error="Failed to calculate some features from SMILES"
            )
        
        return SolventPreprocessOutput(
            solvent_name=input_data.solvent_name,
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


@app.get("/embeddings/methods")
async def get_available_methods():
    """
    Get list of all available method strings for selection.
    
    Returns:
        List of all method names that have embeddings
    """
    global method_embeddings
    try:
        if not method_embeddings:
            # Try to reload embeddings if they're empty
            api_dir = Path(__file__).parent
            method_emb_path = api_dir / "data" / "method_emb_pca_values.json"
            if method_emb_path.exists():
                try:
                    with open(method_emb_path, 'r') as f:
                        method_embeddings = json.load(f)
                    print(f"✓ Reloaded {len(method_embeddings)} method embeddings")
                except Exception as e:
                    print(f"Error reloading method embeddings: {e}")
        
        methods = sorted(list(method_embeddings.keys())) if method_embeddings else []
        return {
            "methods": methods,
            "count": len(methods)
        }
    except Exception as e:
        import traceback
        error_detail = f"Error retrieving methods: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving methods: {str(e)}"
        )


@app.get("/embeddings/polytypes")
async def get_available_polytypes():
    """
    Get list of all available polymerization type strings for selection.
    
    Returns:
        List of all polytype names that have embeddings
    """
    global polytype_embeddings
    try:
        if not polytype_embeddings:
            # Try to reload embeddings if they're empty
            api_dir = Path(__file__).parent
            polytype_emb_path = api_dir / "data" / "polytype_emb_pca_values.json"
            if polytype_emb_path.exists():
                try:
                    with open(polytype_emb_path, 'r') as f:
                        polytype_embeddings = json.load(f)
                    print(f"✓ Reloaded {len(polytype_embeddings)} polytype embeddings")
                except Exception as e:
                    print(f"Error reloading polytype embeddings: {e}")
        
        polytypes = sorted(list(polytype_embeddings.keys())) if polytype_embeddings else []
        return {
            "polytypes": polytypes,
            "count": len(polytypes)
        }
    except Exception as e:
        import traceback
        error_detail = f"Error retrieving polytypes: {str(e)}\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving polytypes: {str(e)}"
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
            detail=f"Method '{method_name}' not found. Use /embeddings/methods to see available options."
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
            detail=f"Polytype '{polytype_name}' not found. Use /embeddings/polytypes to see available options."
        )
    
    return polytype_embeddings[polytype_name]


def get_safe_filename(smiles: str) -> str:
    """Create a safe filename from SMILES by replacing problematic characters."""
    return smiles.replace('/', '_').replace('\\', '_').replace(':', '_')


def load_monomer_features(smiles: str, base_path: Optional[Path] = None) -> Optional[Dict]:
    """
    Load monomer features from JSON file if it exists.
    Returns None if file doesn't exist.
    """
    if base_path is None:
        base_path = Path(__file__).parent / "molecule_properties"
    elif isinstance(base_path, str):
        base_path = Path(base_path)
    
    safe_smiles = get_safe_filename(smiles)
    file_path = base_path / f"{safe_smiles}.json"
    
    if not file_path.exists():
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Process dict fields: take min, max, mean (same as in data_processing.py)
        for key in ['charges', 'fukui_electrophilicity', 'fukui_nucleophilicity', 'fukui_radical']:
            if key in data and isinstance(data[key], dict) and data[key]:
                data[key + '_min'] = min(data[key].values())
                data[key + '_max'] = max(data[key].values())
                data[key + '_mean'] = sum(data[key].values()) / len(data[key].values())
        
        return data
    except Exception as e:
        print(f"Error loading monomer features from {file_path}: {e}")
        return None


def extract_monomer_features_for_model(data: Dict) -> Dict[str, Optional[float]]:
    """
    Extract only the features needed for the model from the full molecular data.
    Returns: fukui_radical_max, homo, lumo
    """
    features = {
        "fukui_radical_max": data.get("fukui_radical_max"),
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
    safe_smiles = get_safe_filename(smiles)
    file_path = base_path / f"{safe_smiles}.json"
    
    with open(file_path, 'w') as f:
        json.dump(properties, f, indent=4)
    
    return properties


@app.post("/preprocess/monomer", response_model=MonomerPreprocessOutput)
async def preprocess_monomer(input_data: MonomerPreprocessInput):
    """
    Preprocess a monomer: convert name to SMILES, check for existing features,
    or calculate new features if needed.
    
    Args:
        input_data: Monomer name
        
    Returns:
        SMILES representation and calculated features
    """
    try:
        # Convert name to SMILES
        monomer_smiles = name_to_smiles(input_data.monomer_name, force_retry=True)
        
        if monomer_smiles is None:
            return MonomerPreprocessOutput(
                monomer_name=input_data.monomer_name,
                monomer_smiles=None,
                features={
                    "fukui_radical_max": None,
                    "homo": None,
                    "lumo": None
                },
                success=False,
                error=f"Could not convert monomer name '{input_data.monomer_name}' to SMILES",
                from_cache=False
            )
        
        # Try to load existing features
        base_path = Path(__file__).parent / "molecule_properties"
        existing_data = load_monomer_features(monomer_smiles, base_path)
        
        if existing_data is not None:
            # Features found in cache
            features = extract_monomer_features_for_model(existing_data)
            return MonomerPreprocessOutput(
                monomer_name=input_data.monomer_name,
                monomer_smiles=monomer_smiles,
                features=features,
                success=True,
                error=None,
                from_cache=True
            )
        
        # Features not found, need to calculate
        if not MORFEUS_AVAILABLE:
            return MonomerPreprocessOutput(
                monomer_name=input_data.monomer_name,
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
        features = extract_monomer_features_for_model(calculated_data)
        
        return MonomerPreprocessOutput(
            monomer_name=input_data.monomer_name,
            monomer_smiles=monomer_smiles,
            features=features,
            success=True,
            error=None,
            from_cache=False
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Monomer preprocessing failed: {str(e)}"
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

