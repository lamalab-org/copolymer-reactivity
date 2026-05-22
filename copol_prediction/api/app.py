#!/usr/bin/env python3
"""
FastAPI REST API for copolymerization prediction.

This module provides a REST API interface for making predictions
with the trained copolymerization model.

Usage:
    uvicorn app:app --reload --host 0.0.0.0 --port 8000
"""

import functools
import hashlib
import json
import math
import os
import platform
import socket
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# FastAPI dependencies
try:
    from fastapi import FastAPI, HTTPException, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, ConfigDict, Field
except ImportError:
    print("Error: FastAPI not installed. Install with: pip install fastapi uvicorn")
    sys.exit(1)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

# Canonical class -> human-readable label mapping, shared with the lookup and
# reaction-optimization modules (single source of truth — see class_labels.py).
from class_labels import CLASS_LABELS
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from copolpredictor.inference import CopolymerPredictor

# Import baseline lookup module
try:
    from baseline_lookup import find_top_k_nearest_neighbors

    BASELINE_LOOKUP_AVAILABLE = True
except ImportError:
    BASELINE_LOOKUP_AVAILABLE = False
    print("Warning: baseline_lookup module not available. Nearest neighbors feature disabled.")

# Import reaction optimization module
try:
    from reaction_optimization import create_optimization_grid, find_architecture_switches

    REACTION_OPTIMIZATION_AVAILABLE = True
except ImportError:
    REACTION_OPTIMIZATION_AVAILABLE = False
    print(
        "Warning: reaction_optimization module not available. Reaction optimization feature disabled."
    )

# Import monomer feature calculation functions
try:
    import qcengine
    from morfeus.conformer import ConformerEnsemble

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

    features: Dict[str, Optional[float]] = Field(
        ...,
        description="Dictionary of feature values (null is allowed; the model treats missing features as NaN/0)",
        json_schema_extra={
            # Real values from the published holdout (artifacts/data_splits/test.csv row 0).
            # Use the exact feature names the trained model expects — see /features.
            "example": {
                "charges_min_1": -0.448977,
                "fukui_electrophilicity_min_1": -0.037725,
                "fukui_electrophilicity_max_1": 0.187582,
                "fukui_nucleophilicity_min_1": -0.017029,
                "fukui_radical_min_1": -0.027106,
                "homo_1": -0.409195,
                "charges_min_2": -0.432152,
                "fukui_electrophilicity_min_2": -0.027193,
                "fukui_electrophilicity_max_2": 0.149151,
                "fukui_nucleophilicity_min_2": -0.006415,
                "fukui_radical_min_2": -0.009479,
                "homo_2": -0.378648,
                "delta_HOMO_LUMO_AB": -0.140381,
                "delta_HOMO_LUMO_BA": -0.097219,
                "temperature": 60.0,
                "polytype_emb_1": 7.496726,
                "polytype_emb_2": -0.463191,
                "solvent_logp": -0.0014,
                "solvent_FractionCSP3": 1.0,
            }
        },
    )


class BatchPredictionInput(BaseModel):
    """Input schema for batch prediction."""

    samples: List[Dict[str, Optional[float]]] = Field(
        ..., description="List of feature dictionaries (null values are allowed)"
    )


class NearestNeighbor(BaseModel):
    """Nearest neighbor data point from training database."""

    rank: int = Field(..., description="Ranking (1-10)")
    similarity: float = Field(..., description="Similarity score (0-1), higher is more similar")
    predicted_class: int = Field(..., description="Predicted class (0, 1, or 2)", alias="class")
    predicted_class_name: str = Field(..., description="Human-readable class label")
    monomer1_name: str = Field(..., description="First monomer name")
    monomer2_name: str = Field(..., description="Second monomer name")
    monomer1_smiles: str = Field(..., description="First monomer SMILES")
    monomer2_smiles: str = Field(..., description="Second monomer SMILES")
    solvent_name: str = Field(..., description="Solvent name")
    solvent_smiles: str = Field(..., description="Solvent SMILES")
    temperature: Optional[float] = Field(None, description="Temperature in Celsius")
    method: Optional[str] = Field(None, description="Polymerization method")
    polytype: Optional[str] = Field(None, description="Polymerization type")
    source: Optional[str] = Field(
        None,
        description=(
            "Raw provenance string as extracted by the LLM — may be a DOI, "
            "a citation, or free text. Prefer `doi` / `doi_url` below."
        ),
    )
    doi: Optional[str] = Field(
        None,
        description=(
            "DOI of the source paper, recovered from the processed-paper "
            "filename. Null for synthetic/augmented neighbours and for the "
            "rare real paper that has no DOI."
        ),
    )
    doi_url: Optional[str] = Field(
        None, description="Resolvable https://doi.org/ link, or null when `doi` is null"
    )
    reaction_id: Optional[str] = Field(None, description="Reaction ID")

    model_config = ConfigDict(populate_by_name=True)


class PredictionOutput(BaseModel):
    """Output schema for prediction."""

    predicted_class: int = Field(..., description="Predicted class (0, 1, or 2)")
    predicted_class_name: str = Field(..., description="Human-readable class label")
    class_probabilities: Dict[str, float] = Field(
        ...,
        description=(
            "Mapping from human-readable class name (see `class_descriptions` "
            "on /model/info, or `predicted_class_name`) to its probability."
        ),
    )
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    below_threshold: bool = Field(
        False, description="Whether confidence is below the 0.7 threshold"
    )
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
    class_descriptions: Dict[str, str] = Field(
        ..., description="Human-readable label for each class"
    )
    created_at: str
    model_path: str


class BuildInfo(BaseModel):
    """Build-time provenance baked into the image. All fields are best-effort
    — they default to "unknown" outside Docker (env vars unset)."""

    git_sha: str = Field(..., description="Commit SHA the image was built from")
    git_branch: str = Field(..., description="Branch or tag the image was built from")
    build_time: str = Field(..., description="When the image was built (ISO 8601 UTC)")


class RuntimeInfo(BaseModel):
    """Runtime / process info — handy for debugging restarts and env drift."""

    python_version: str = Field(..., description="Major.minor.patch")
    started_at: str = Field(..., description="When this process started (ISO 8601 UTC)")
    uptime_seconds: float = Field(..., description="Seconds since process start")
    hostname: str = Field(..., description="Container hostname")


class HealthCheck(BaseModel):
    """Health check response schema."""

    status: str
    timestamp: str
    model_loaded: bool
    build: BuildInfo
    runtime: RuntimeInfo


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
    method: str = Field(default="solvent", description="Polymerisation method")
    polytype: str = Field(default="free radical", description="Polymerisation type")
    temperature: float = Field(default=60.0, description="Temperature in Celsius")


# Literal aliases keep the OpenAPI schema (and Swagger UI) showing exactly
# the allowed values — a frontend <select> can forward its option value
# verbatim.
SolventSet = Literal["top3", "common", "chlorinated", "aromatic"]
TemperatureMode = Literal["40-80", "20-100", "fixed60", "step20"]


class OptimizeReactionInput(BaseModel):
    """Input schema for reaction optimization."""

    monomer1_smiles: str = Field(..., description="SMILES string of monomer 1")
    monomer2_smiles: str = Field(..., description="SMILES string of monomer 2")
    solvent_smiles: str = Field(..., description="SMILES string of base solvent")
    method: str = Field(default="solvent", description="Polymerisation method")
    polytype: str = Field(default="free radical", description="Polymerisation type")
    temperature: float = Field(default=60.0, description="Base temperature in Celsius")
    temperature_step: float = Field(
        default=20.0,
        description="Temperature step in Celsius — only used when temperature_mode='step20'",
    )
    n_solvents: int = Field(
        default=3, description="Number of solvents — only used when solvent_set='top3'"
    )
    solvent_set: SolventSet = Field(
        default="top3",
        description=(
            "Which solvents to sweep. 'top3': dataset solvents nearest the base "
            "solvent in logP (uses n_solvents). 'common'/'chlorinated'/'aromatic': "
            "curated sets."
        ),
    )
    temperature_mode: TemperatureMode = Field(
        default="step20",
        description=(
            "Which temperatures to sweep. '40-80'=[40,60,80], '20-100'=[20,60,100], "
            "'fixed60'=[60], 'step20'=base ± temperature_step."
        ),
    )


class PreprocessAllOutput(BaseModel):
    """Output schema for combined preprocessing."""

    features: Dict[str, Optional[float]] = Field(
        ...,
        description="All calculated features ready for prediction. Individual values may be null when the underlying XTB descriptor could not be computed for a given monomer.",
    )
    success: bool = Field(..., description="Whether preprocessing was successful")
    error: Optional[str] = Field(None, description="Error message if preprocessing failed")
    nearest_neighbors: Optional[List[NearestNeighbor]] = Field(
        None, description="Top 10 nearest data points from training database (baseline lookup)"
    )
    lookup_class: Optional[int] = Field(
        None,
        description="Predicted class from the Lookup (nearest-neighbor) model (top-1 neighbor)",
    )


class OptimizationPrediction(BaseModel):
    """Single prediction result in optimization grid."""

    temperature: float = Field(..., description="Temperature in Celsius")
    solvent_smiles: str = Field(..., description="Solvent SMILES")
    solvent_name: str = Field(..., description="Solvent name")
    solvent_logp: float = Field(..., description="Solvent logP value")
    predicted_class: int = Field(..., description="Predicted class (0, 1, or 2)")
    predicted_class_name: str = Field(..., description="Human-readable class label")
    class_probabilities: Dict[str, float] = Field(
        ...,
        description=(
            "Mapping from human-readable class name (see `class_descriptions` "
            "on /model/info, or `predicted_class_name`) to its probability."
        ),
    )
    confidence: float = Field(..., description="Prediction confidence (0-1)")


class OptimizeReactionOutput(BaseModel):
    """Output schema for reaction optimization."""

    success: bool = Field(..., description="Whether optimization was successful")
    error: Optional[str] = Field(None, description="Error message if optimization failed")
    predictions: List[OptimizationPrediction] = Field(
        ...,
        description="Grid of predictions, one per (temperature × solvent) combination",
    )
    base_temperature: float = Field(..., description="Base temperature used")
    temperature_step: float = Field(..., description="Temperature step size used")
    base_solvent_logp: float = Field(..., description="Base solvent logP value")
    timestamp: str = Field(..., description="Optimization timestamp")


class ArchitectureSwitchInput(BaseModel):
    """Input schema for the counterfactual architecture-switch search."""

    monomer1_smiles: str = Field(..., description="SMILES string of monomer 1")
    monomer2_smiles: str = Field(..., description="SMILES string of monomer 2")
    solvent_smiles: str = Field(..., description="SMILES of the current/base solvent")
    method: str = Field(default="solvent", description="Polymerisation method")
    polytype: str = Field(default="free radical", description="Polymerisation type")
    temperature: float = Field(default=60.0, description="Current/base temperature in Celsius")
    solvent_set: SolventSet = Field(
        default="common", description="Which solvents to search — see /optimize_reaction"
    )
    temperature_mode: TemperatureMode = Field(
        default="40-80", description="Which temperatures to search — see /optimize_reaction"
    )
    temperature_step: float = Field(
        default=20.0, description="Temperature step — only used when temperature_mode='step20'"
    )
    n_solvents: int = Field(
        default=3, description="Number of solvents — only used when solvent_set='top3'"
    )
    top_n: int = Field(default=5, description="Max number of counterfactuals to return")


class ArchitectureSwitchCandidate(BaseModel):
    """A condition set that flips the predicted architecture."""

    temperature: float = Field(..., description="Temperature in Celsius")
    solvent_smiles: str = Field(..., description="Solvent SMILES")
    solvent_name: str = Field(..., description="Solvent name")
    solvent_logp: float = Field(..., description="Solvent logP value")
    predicted_class: int = Field(..., description="Predicted class (0, 1, or 2)")
    predicted_class_name: str = Field(..., description="Human-readable class label")
    class_probabilities: Dict[str, float] = Field(
        ..., description="Probability per class, keyed by human-readable class name"
    )
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    delta_logp: float = Field(..., description="solvent_logp minus the base solvent's logP")
    delta_temperature: float = Field(..., description="temperature minus the base temperature (°C)")
    reference: Optional[NearestNeighbor] = Field(
        None,
        description=(
            "Closest real reaction in the training data for this counterfactual's "
            "solvent — grounds the suggested condition change in a literature data "
            "point (with `doi` / `doi_url` when available). Preferentially the "
            "closest reaction with the *same* monomer pair; null when baseline "
            "lookup is unavailable or no neighbour was found."
        ),
    )
    reference_same_monomers: Optional[bool] = Field(
        None,
        description=(
            "True when `reference` is a reaction with the same monomer pair as "
            "the query. False when the training data has no reaction for this "
            "monomer pair and `reference` is the closest *different*-monomer "
            "reaction instead (treat as a weaker analogy). Null when there is "
            "no reference."
        ),
    )


class ArchitectureSwitchOutput(BaseModel):
    """Output schema for the counterfactual architecture-switch search."""

    success: bool = Field(..., description="Whether the search ran")
    error: Optional[str] = Field(None, description="Error message if the search failed")
    baseline: Optional[OptimizationPrediction] = Field(
        None, description="Prediction for the unchanged (base solvent + base temperature) reaction"
    )
    counterfactuals: List[ArchitectureSwitchCandidate] = Field(
        default_factory=list,
        description=(
            "Condition sets whose predicted architecture differs from the baseline, "
            "ranked by smallest |delta_logp| (then |delta_temperature|). Empty if no "
            "evaluated condition flips the architecture."
        ),
    )
    n_evaluated: int = Field(0, description="Total (solvent × temperature) cells evaluated")
    timestamp: str = Field(..., description="Search timestamp")


class DOICheckInput(BaseModel):
    """Input schema for DOI check."""

    doi: str = Field(
        ...,
        description="DOI to check (e.g., '10.1016/0014-3057(84)90010-7' or 'https://doi.org/10.1016/0014-3057(84)90010-7')",
    )


class DOICheckOutput(BaseModel):
    """Output schema for DOI check."""

    doi: str = Field(..., description="The queried DOI")
    exists: bool = Field(..., description="Whether the DOI exists in the dataset")
    normalized_doi: str = Field(..., description="The normalized DOI used for matching")
    timestamp: str = Field(..., description="Check timestamp")


# ============================================================================
# Application State
# ============================================================================

# Global predictor instance
predictor: Optional[CopolymerPredictor] = None

# Configurable paths.
# Defaults point to the baked-in locations inside the Docker image (/app/…).
# Override via env vars for local development (e.g. MODEL_PATH=../artifacts/model_bundle).
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/artifacts/model_bundle")
DATASET_PATH = os.environ.get("DATASET_PATH", "/app/processed_data.csv")
TRAIN_DATA_PATH = os.environ.get("TRAIN_DATA_PATH", "/app/artifacts/data_splits/train.csv")
NEGATIVE_DATA_PATH = os.environ.get(
    "NEGATIVE_DATA_PATH",
    "/app/artificial_datapoints.csv",
)

# Deterministic seed for live-XTB conformer generation (see
# calculate_monomer_features). Overridable for ablation; in production we
# always want the same SMILES → same cached features.
CONFORMER_RANDOM_SEED = int(os.environ.get("CONFORMER_RANDOM_SEED", "42"))

# Build provenance baked in at `docker build` time via --build-arg (see
# .github/workflows/docker-image.yml). Outside Docker, env vars are unset
# and we report "unknown" — debugging signal, not a contract.
BUILD_INFO = BuildInfo(
    git_sha=os.environ.get("GIT_SHA", "unknown"),
    git_branch=os.environ.get("GIT_BRANCH", "unknown"),
    build_time=os.environ.get("BUILD_TIME", "unknown"),
)

# Process-uptime tracking. _MONOTONIC is for measuring seconds (immune to
# wall-clock jumps); _AT is the human-readable wall-clock start time.
PROCESS_STARTED_AT = datetime.utcnow().isoformat() + "Z"
PROCESS_STARTED_MONOTONIC = time.time()

# Global embedding dictionaries
method_embeddings: Dict[str, Dict[str, float]] = {}
polytype_embeddings: Dict[str, Dict[str, float]] = {}

# Global dataset cache
dataset_df: Optional[pd.DataFrame] = None

# Global training data cache for baseline lookup
train_df: Optional[pd.DataFrame] = None

# Global fingerprint cache for baseline lookup
fingerprint_cache: Optional[Dict] = None


# ============================================================================
# Lifespan (startup / shutdown)
# ============================================================================


@asynccontextmanager
async def lifespan(app):
    """Load model, embeddings, and dataset on startup; cleanup on shutdown."""
    global predictor, method_embeddings, polytype_embeddings, dataset_df, train_df, fingerprint_cache

    # Load embeddings
    try:
        api_dir = Path(__file__).parent
        method_emb_path = api_dir / "data" / "method_emb_pca_values.json"
        polytype_emb_path = api_dir / "data" / "polytype_emb_pca_values.json"

        if method_emb_path.exists():
            try:
                with open(method_emb_path, "r") as f:
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
                with open(polytype_emb_path, "r") as f:
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
            api_dir = Path(__file__).parent
            train_path = Path(TRAIN_DATA_PATH)
            if not train_path.is_absolute():
                train_path = api_dir / train_path

            if train_path.exists():
                print(f"Loading training data from {train_path}...")
                train_df = pd.read_csv(train_path)
                print(f"✓ Training data loaded successfully ({len(train_df)} rows)")

                neg_path = Path(NEGATIVE_DATA_PATH)
                if not neg_path.is_absolute():
                    neg_path = api_dir / neg_path
                if neg_path.exists():
                    df_neg = pd.read_csv(neg_path)
                    if "Class" in df_neg.columns:
                        df_neg = df_neg.rename(columns={"Class": "r_product_class"})
                    df_neg["r_product_class"] = df_neg["r_product_class"].astype(int)
                    train_df = pd.concat([train_df, df_neg], ignore_index=True)
                    print(
                        f"✓ Added {len(df_neg)} negative data points to lookup pool ({len(train_df)} total)"
                    )
                else:
                    print(f"⚠ Warning: Negative data not found at {neg_path}")

                # Precompute fingerprints for all unique SMILES in training data
                if BASELINE_LOOKUP_AVAILABLE:
                    try:
                        from baseline_lookup import (
                            compute_fingerprints_for_smiles,
                            load_fingerprint_cache,
                            save_fingerprint_cache,
                        )

                        print("Loading fingerprint cache...")
                        fingerprint_cache = load_fingerprint_cache()

                        # Get all unique SMILES from training data
                        unique_monomer1 = set(train_df["monomer1_smiles"].dropna().unique())
                        unique_monomer2 = set(train_df["monomer2_smiles"].dropna().unique())
                        unique_solvents = set(train_df["solvent_smiles"].dropna().unique())
                        all_unique_smiles = list(
                            unique_monomer1 | unique_monomer2 | unique_solvents
                        )

                        # Compute fingerprints for any missing SMILES
                        if fingerprint_cache is None:
                            print(
                                "Fingerprint cache not found. Computing fingerprints for all SMILES..."
                            )
                            fingerprint_cache = compute_fingerprints_for_smiles(all_unique_smiles)
                            save_fingerprint_cache(fingerprint_cache)
                            print(
                                f"✓ Computed and cached fingerprints for {len(fingerprint_cache)} SMILES"
                            )
                        else:
                            # Check if we need to compute any missing fingerprints
                            missing_smiles = [
                                s for s in all_unique_smiles if s not in fingerprint_cache
                            ]
                            if missing_smiles:
                                print(
                                    f"Computing fingerprints for {len(missing_smiles)} missing SMILES..."
                                )
                                new_fps = compute_fingerprints_for_smiles(
                                    missing_smiles, cache_dict=fingerprint_cache
                                )
                                fingerprint_cache.update(new_fps)
                                save_fingerprint_cache(fingerprint_cache)
                                print(
                                    f"✓ Updated cache with {len(missing_smiles)} new fingerprints"
                                )
                            else:
                                print(
                                    f"✓ Fingerprint cache loaded ({len(fingerprint_cache)} entries)"
                                )
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

    yield

    # Shutdown
    print("Shutting down API...")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Copolymerization Prediction API",
    description="REST API for predicting copolymerization reactivity using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        "health": "/health",
    }


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy" if predictor else "model_not_loaded",
        timestamp=datetime.now().isoformat(),
        model_loaded=predictor is not None,
        build=BUILD_INFO,
        runtime=RuntimeInfo(
            python_version=platform.python_version(),
            started_at=PROCESS_STARTED_AT,
            uptime_seconds=round(time.time() - PROCESS_STARTED_MONOTONIC, 1),
            hostname=socket.gethostname(),
        ),
    )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    return ModelInfo(
        model_version="1.0.0",
        n_features=len(predictor.features),
        feature_names=predictor.features,
        class_labels=predictor.class_labels,
        class_descriptions={str(k): v for k, v in CLASS_LABELS.items()},
        created_at=predictor.metadata.get("created_at", "unknown"),
        model_path=MODEL_PATH,
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
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    try:
        results = predictor.predict_with_confidence(input_data.features)

        pred_class = int(results["predictions"][0])
        proba = results["probabilities"][0]
        confidence = float(results["confidence"][0])

        return PredictionOutput(
            predicted_class=pred_class,
            predicted_class_name=CLASS_LABELS[pred_class],
            class_probabilities={CLASS_LABELS[i]: float(proba[i]) for i in range(len(proba))},
            confidence=confidence,
            below_threshold=confidence < 0.7,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Prediction failed: {str(e)}"
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
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    try:
        predictions = []

        for sample in input_data.samples:
            results = predictor.predict_with_confidence(sample)

            pred_class = int(results["predictions"][0])
            proba = results["probabilities"][0]
            confidence = float(results["confidence"][0])

            predictions.append(
                PredictionOutput(
                    predicted_class=pred_class,
                    predicted_class_name=CLASS_LABELS[pred_class],
                    class_probabilities={
                        CLASS_LABELS[i]: float(proba[i]) for i in range(len(proba))
                    },
                    confidence=confidence,
                    below_threshold=confidence < 0.7,
                    timestamp=datetime.now().isoformat(),
                )
            )

        return BatchPredictionOutput(
            predictions=predictions,
            total_samples=len(predictions),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Batch prediction failed: {str(e)}"
        )


@app.get("/features")
async def get_required_features():
    """Get list of required features for prediction."""
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    try:
        features = predictor.features if predictor.features else []
        return {"required_features": features, "n_features": len(features) if features else 0}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving features: {str(e)}",
        )


@functools.lru_cache(maxsize=1)
def _load_paper_metrics() -> Optional[Dict]:
    """Load the precomputed train/test performance artifact (memoised).

    `paper_metrics.json` is generated by reproduce_paper_metrics.py and ships
    next to the model bundle; it carries aggregate metrics, confusion matrices
    and per-row individual predictions. Returns None if the artifact is absent.
    """
    path = (Path(__file__).parent / MODEL_PATH).resolve().parent / "paper_metrics.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@app.get("/paper_metrics")
async def get_paper_metrics():
    """Train/test performance of the released model — the paper's results.

    Returns aggregate metrics (per-class accuracy/precision/F1, confusion
    matrices) for the plain XGBoost and voting models on both splits, plus the
    per-row individual predictions. Served from a precomputed artifact
    (`reproduce_paper_metrics.py --json`), cached in memory — no live compute.
    """
    metrics = _load_paper_metrics()
    if metrics is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="paper_metrics.json artifact not found",
        )
    return metrics


@app.post("/preprocess_all", response_model=PreprocessAllOutput)
async def preprocess_all(input_data: PreprocessAllInput):
    """
    Preprocess all inputs and return features ready for prediction.
    Combines monomer preprocessing, solvent preprocessing, and embeddings.
    """
    try:
        base_path = Path(__file__).parent / "molecule_properties"

        # Compute nearest neighbors early so the lookup remains available even
        # when live XTB feature calculation fails for one of the monomers.
        nearest_neighbors_list = None
        lookup_class_value = None
        if BASELINE_LOOKUP_AVAILABLE and train_df is not None:
            try:
                global fingerprint_cache
                fp_dict_to_use = fingerprint_cache if fingerprint_cache is not None else None
                neighbors = find_top_k_nearest_neighbors(
                    test_monomer1_smiles=input_data.monomer1_smiles,
                    test_monomer2_smiles=input_data.monomer2_smiles,
                    test_solvent_smiles=input_data.solvent_smiles,
                    df_train=train_df,
                    k=10,
                    fp_dict=fp_dict_to_use,
                )

                if neighbors:
                    print(f"✓ Found {len(neighbors)} nearest neighbors")
                    nearest_neighbors_list = [NearestNeighbor(**neighbor) for neighbor in neighbors]
                    lookup_class_value = int(neighbors[0]["predicted_class"])
                else:
                    nearest_neighbors_list = []
                    print("⚠ Warning: find_top_k_nearest_neighbors returned empty list")
            except Exception as e:
                print(f"✗ Error: Failed to find nearest neighbors: {e}")
                import traceback

                traceback.print_exc()
                nearest_neighbors_list = None

        m1_data, err = get_or_compute_monomer_data(
            input_data.monomer1_smiles, base_path, "Monomer 1"
        )
        if err:
            cleaned_nearest_neighbors = (
                clean_json_values(nearest_neighbors_list)
                if nearest_neighbors_list is not None
                else None
            )
            return PreprocessAllOutput(
                features={},
                success=False,
                error=err,
                nearest_neighbors=cleaned_nearest_neighbors,
                lookup_class=lookup_class_value,
            )
        m1_features = extract_monomer_features_for_model(m1_data)

        m2_data, err = get_or_compute_monomer_data(
            input_data.monomer2_smiles, base_path, "Monomer 2"
        )
        if err:
            cleaned_nearest_neighbors = (
                clean_json_values(nearest_neighbors_list)
                if nearest_neighbors_list is not None
                else None
            )
            return PreprocessAllOutput(
                features={},
                success=False,
                error=err,
                nearest_neighbors=cleaned_nearest_neighbors,
                lookup_class=lookup_class_value,
            )
        m2_features = extract_monomer_features_for_model(m2_data)

        # Preprocess solvent
        solvent_features = calculate_solvent_features(input_data.solvent_smiles)
        if any(v is None for v in solvent_features.values()):
            return PreprocessAllOutput(
                features={}, success=False, error="Failed to calculate solvent features"
            )

        # Get embeddings
        if input_data.method not in method_embeddings:
            return PreprocessAllOutput(
                features={},
                success=False,
                error=f"Method '{input_data.method}' not found in embeddings",
            )
        method_emb = method_embeddings[input_data.method]

        if input_data.polytype not in polytype_embeddings:
            return PreprocessAllOutput(
                features={},
                success=False,
                error=f"Polytype '{input_data.polytype}' not found in embeddings",
            )
        polytype_emb = polytype_embeddings[input_data.polytype]

        features = assemble_model_features(
            m1_features=m1_features,
            m2_features=m2_features,
            solvent_features=solvent_features,
            polytype_emb=polytype_emb,
            method_emb=method_emb,
            temperature=input_data.temperature,
        )

        # If model is loaded, ensure all required features are present
        # Fill missing features with None if model requires them
        if predictor and predictor.features:
            for required_feature in predictor.features:
                if required_feature not in features:
                    features[required_feature] = None

        cleaned_features = clean_json_values(features, replace_with_zero=True)
        cleaned_nearest_neighbors = (
            clean_json_values(nearest_neighbors_list)
            if nearest_neighbors_list is not None
            else None
        )

        return PreprocessAllOutput(
            features=cleaned_features,
            success=True,
            error=None,
            nearest_neighbors=cleaned_nearest_neighbors,
            lookup_class=lookup_class_value,
        )

    except Exception as e:
        import traceback

        error_detail = f"Preprocessing failed: {str(e)}"
        print(f"Error in preprocess_all: {error_detail}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_detail)


@functools.lru_cache(maxsize=256)
def calculate_solvent_features(smiles: str) -> Dict[str, Optional[float]]:
    """
    Calculate solvent features from SMILES string.
    Returns only the features needed by the model:
    - solvent_logP
    - solvent_TPSA
    - solvent_HBD
    - solvent_FractionCSP3

    Memoised: solvent descriptors are a pure function of the SMILES and the
    same handful of solvents are hit repeatedly across /preprocess_all,
    /optimize_reaction and /find_architecture_switch. Callers must treat the
    returned dict as read-only — it is shared across cache hits.
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
            "solvent_FractionCSP3": None,
        }

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "solvent_logP": None,
                "solvent_TPSA": None,
                "solvent_HBD": None,
                "solvent_FractionCSP3": None,
            }

        return {
            "solvent_logP": float(Descriptors.MolLogP(mol)),
            "solvent_TPSA": float(rdMolDescriptors.CalcTPSA(mol)),
            "solvent_HBD": float(rdMolDescriptors.CalcNumHBD(mol)),
            "solvent_FractionCSP3": float(Descriptors.FractionCSP3(mol)),
        }
    except Exception as e:
        return {
            "solvent_logP": None,
            "solvent_TPSA": None,
            "solvent_HBD": None,
            "solvent_FractionCSP3": None,
        }


def canonicalize_smiles(smiles: str) -> str:
    """Canonicalize smiles using RDKit (local copy without cache to avoid SQLite issues)"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")
    return Chem.MolToSmiles(mol)


def monomer_pair_key(smiles1: str, smiles2: str) -> Optional[frozenset]:
    """Order-insensitive canonical key for a monomer pair, or None if either
    SMILES can't be parsed. Two reactions share a key iff they use the same
    two monomers (regardless of which is labelled 1 vs 2)."""
    try:
        return frozenset({canonicalize_smiles(smiles1), canonicalize_smiles(smiles2)})
    except Exception:
        return None


# Per-row monomer_pair_key for the lookup pool, memoised by DataFrame identity
# (the pool is built once at startup). Canonicalising ~thousands of SMILES is
# not free, so the first /find_architecture_switch call pays it once.
_train_monomer_pairs_memo: Dict[int, List[Optional[frozenset]]] = {}


def train_monomer_pair_keys(df: pd.DataFrame) -> List[Optional[frozenset]]:
    """monomer_pair_key for every row of `df`, in row order (memoised)."""
    cached = _train_monomer_pairs_memo.get(id(df))
    if cached is not None:
        return cached
    keys = [
        monomer_pair_key(m1, m2) for m1, m2 in zip(df["monomer1_smiles"], df["monomer2_smiles"])
    ]
    _train_monomer_pairs_memo[id(df)] = keys
    return keys


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
                error="Failed to calculate some features from SMILES",
            )

        return SolventPreprocessOutput(
            solvent_smiles=solvent_smiles, features=features, success=True, error=None
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Solvent preprocessing failed: {str(e)}",
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
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Method '{method_name}' not found."
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
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Polytype '{polytype_name}' not found."
        )

    return polytype_embeddings[polytype_name]


def get_smiles_md5(smiles: str) -> str:
    """Create MD5 hash from SMILES string for consistent filename."""
    return hashlib.md5(smiles.encode("utf-8")).hexdigest()


@functools.lru_cache(maxsize=512)
def _load_monomer_json(file_path_str: str, _mtime: float) -> Optional[Dict]:
    """Parse a monomer-properties JSON and attach min/max/mean aggregates.

    Memoised on (path, mtime): these JSONs carry conformer-coordinate
    arrays so the parse is not free, and the same monomers are read on
    every request that touches them. `_mtime` is in the key purely so a
    regenerated cache file invalidates the stale entry. The returned dict
    is shared across cache hits — callers must treat it as read-only.
    """
    try:
        with open(file_path_str, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading monomer features from {file_path_str}: {e}")
        return None

    for key in ("charges", "fukui_electrophilicity", "fukui_nucleophilicity", "fukui_radical"):
        values = data.get(key)
        if isinstance(values, dict) and values:
            nums = list(values.values())
            data[key + "_min"] = min(nums)
            data[key + "_max"] = max(nums)
            data[key + "_mean"] = sum(nums) / len(nums)
    return data


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

    # 1) Fast path: MD5 hash lookup (parse + aggregates are memoised).
    md5_hash = get_smiles_md5(smiles)
    file_path = base_path / f"{md5_hash}.json"

    if file_path.exists():
        return _load_monomer_json(str(file_path), file_path.stat().st_mtime)

    # 2) Fallback: canonical SMILES-based lookup (for backward compatibility)
    try:
        target_canonical = canonicalize_smiles(smiles)
    except Exception as e:
        print(f"Warning: failed to canonicalize SMILES '{smiles}': {e}")
        return None

    try:
        for candidate_path in base_path.glob("*.json"):
            try:
                with open(candidate_path, "r") as f:
                    candidate_data = json.load(f)
            except Exception:
                continue

            stored_smiles = candidate_data.get("smiles")
            if not stored_smiles:
                continue

            try:
                if canonicalize_smiles(stored_smiles) == target_canonical:
                    # Found a match; post-process fields as above
                    for key in [
                        "charges",
                        "fukui_electrophilicity",
                        "fukui_nucleophilicity",
                        "fukui_radical",
                    ]:
                        if (
                            key in candidate_data
                            and isinstance(candidate_data[key], dict)
                            and candidate_data[key]
                        ):
                            candidate_data[key + "_min"] = min(candidate_data[key].values())
                            candidate_data[key + "_max"] = max(candidate_data[key].values())
                            candidate_data[key + "_mean"] = sum(candidate_data[key].values()) / len(
                                candidate_data[key].values()
                            )
                    return candidate_data
            except Exception:
                continue
    except Exception as e:
        print(f"Warning: error while searching canonical match for '{smiles}': {e}")

    return None


def extract_monomer_features_for_model(data: Dict) -> Dict[str, Optional[float]]:
    """
    Extract all per-monomer features the model (or downstream callers) may
    need from the cached XTB JSON. Keys here are bare descriptor names; the
    suffix `_1`/`_2` is added later when assembling the model feature vector.
    """
    dipole = data.get("dipole")
    dipole_x = dipole[0] if isinstance(dipole, list) and len(dipole) > 0 else None
    dipole_y = dipole[1] if isinstance(dipole, list) and len(dipole) > 1 else None
    dipole_z = dipole[2] if isinstance(dipole, list) and len(dipole) > 2 else None

    return {
        # Aggregates (computed by load_monomer_features from the raw dicts).
        "charges_min": data.get("charges_min"),
        "charges_max": data.get("charges_max"),
        "charges_mean": data.get("charges_mean"),
        "fukui_electrophilicity_min": data.get("fukui_electrophilicity_min"),
        "fukui_electrophilicity_max": data.get("fukui_electrophilicity_max"),
        "fukui_electrophilicity_mean": data.get("fukui_electrophilicity_mean"),
        "fukui_nucleophilicity_min": data.get("fukui_nucleophilicity_min"),
        "fukui_nucleophilicity_max": data.get("fukui_nucleophilicity_max"),
        "fukui_nucleophilicity_mean": data.get("fukui_nucleophilicity_mean"),
        "fukui_radical_min": data.get("fukui_radical_min"),
        "fukui_radical_max": data.get("fukui_radical_max"),
        "fukui_radical_mean": data.get("fukui_radical_mean"),
        # Scalars.
        "best_conformer_energy": data.get("best_conformer_energy"),
        "ip": data.get("ip"),
        "ip_corrected": data.get("ip_corrected"),
        "ea": data.get("ea"),
        "homo": data.get("homo"),
        "lumo": data.get("lumo"),
        "global_electrophilicity": data.get("global_electrophilicity"),
        "global_nucleophilicity": data.get("global_nucleophilicity"),
        "dipole_x": dipole_x,
        "dipole_y": dipole_y,
        "dipole_z": dipole_z,
    }


def assemble_model_features(
    m1_features: Dict[str, Optional[float]],
    m2_features: Dict[str, Optional[float]],
    solvent_features: Dict[str, Optional[float]],
    polytype_emb: Dict[str, float],
    method_emb: Dict[str, float],
    temperature: float,
) -> Dict[str, Optional[float]]:
    """
    Assemble a feature dict keyed by the model's expected column names.

    The trained model expects e.g. `charges_min_1` / `solvent_logp` (lowercase).
    Earlier versions of this code produced a different (legacy) feature set,
    which the model silently ignored — all model features fell back to zero,
    producing meaningless predictions. This helper is the single source of
    truth so the same vector is built from /preprocess_all and /optimize_reaction.
    """

    def _delta_hl(homo: Optional[float], lumo: Optional[float]) -> Optional[float]:
        if homo is None or lumo is None:
            return None
        return homo - lumo

    features: Dict[str, Optional[float]] = {}

    # Per-monomer descriptors: suffix with _1 / _2.
    for suffix, mf in (("_1", m1_features), ("_2", m2_features)):
        for key, value in mf.items():
            features[f"{key}{suffix}"] = value

    # HOMO-LUMO gaps used by the model and downstream callers.
    features["delta_HOMO_LUMO_AA"] = _delta_hl(m1_features.get("homo"), m1_features.get("lumo"))
    features["delta_HOMO_LUMO_AB"] = _delta_hl(m1_features.get("homo"), m2_features.get("lumo"))
    features["delta_HOMO_LUMO_BB"] = _delta_hl(m2_features.get("homo"), m2_features.get("lumo"))
    features["delta_HOMO_LUMO_BA"] = _delta_hl(m2_features.get("homo"), m1_features.get("lumo"))

    features["temperature"] = temperature
    features["polytype_emb_1"] = polytype_emb["pca_1"]
    features["polytype_emb_2"] = polytype_emb["pca_2"]
    features["method_emb_1"] = method_emb["pca_1"]
    features["method_emb_2"] = method_emb["pca_2"]

    # Solvent features. The trained model uses the lowercase column name
    # `solvent_logp`; we also keep `solvent_logP` for backward compatibility
    # with any consumer that still reads the camelCase version.
    features["solvent_logp"] = solvent_features["solvent_logP"]
    features["solvent_logP"] = solvent_features["solvent_logP"]
    features["solvent_TPSA"] = solvent_features["solvent_TPSA"]
    features["solvent_HBD"] = solvent_features["solvent_HBD"]
    features["solvent_FractionCSP3"] = solvent_features["solvent_FractionCSP3"]

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

    # Seed the RDKit conformer search (morfeus forwards `random_seed` to
    # rdDistGeom.EmbedMultipleConfs via `params.randomSeed`) so back-to-back
    # cache-miss recomputes for the same SMILES land on the same conformer.
    # This is reproducibility for *future* cache entries — it can't make a
    # freshly-computed entry agree bit-exactly with the legacy JSONs already
    # on disk, which were generated by a prior pipeline run with an
    # unknown (possibly unseeded) seed.
    ce = ConformerEnsemble.from_rdkit(smiles, optimize="MMFF94", random_seed=CONFORMER_RANDOM_SEED)
    ce.prune_rmsd()
    ce.sort()

    try:
        ce.optimize_qc_engine(program="xtb", model={"method": "GFN-FF"}, procedure="geometric")
    except Exception as e:
        print(f"GFN-FF optimization failed for {smiles}: {e}")
        ce = ConformerEnsemble.from_rdkit(
            smiles, optimize="MMFF94", random_seed=CONFORMER_RANDOM_SEED
        )
        ce.prune_rmsd()
        ce.sort()

    try:
        ce.optimize_qc_engine(program="xtb", model={"method": "GFN2-xTB"}, procedure="geometric")
    except Exception as e:
        print(f"GFN2-xTB optimization failed for {smiles}: {e}")

    ce.sp_qc_engine(program="xtb", model={"method": "GFN2-xTB"})
    best_conformer = ce.conformers[0]

    elements = best_conformer.elements.tolist()
    coordinates = best_conformer.coordinates.tolist()
    energy = best_conformer.energy

    # Calculate properties. Each XTB call is guarded individually because some
    # descriptors (e.g. dipole, fukui) occasionally come back as None for
    # specific molecules, and we don't want a single missing field to nuke the
    # whole cache entry.
    xtb = XTB(elements, coordinates)

    def _safe(name, fn):
        try:
            return fn()
        except Exception as e:
            print(f"⚠ {name} failed for {smiles}: {e}")
            return None

    dipole = _safe("dipole", xtb.get_dipole)
    properties = {
        "smiles": smiles,
        "best_conformer_coordinates": coordinates,
        "best_conformer_elements": elements,
        "best_conformer_energy": energy,
        "ip": _safe("ip", xtb.get_ip),
        "ip_corrected": _safe("ip_corrected", lambda: xtb.get_ip(corrected=True)),
        "ea": _safe("ea", xtb.get_ea),
        "homo": _safe("homo", xtb.get_homo),
        "lumo": _safe("lumo", xtb.get_lumo),
        "charges": _safe("charges", xtb.get_charges),
        "dipole": dipole.tolist() if dipole is not None else None,
        "global_electrophilicity": _safe(
            "global_electrophilicity",
            lambda: xtb.get_global_descriptor("electrophilicity", corrected=True),
        ),
        "global_nucleophilicity": _safe(
            "global_nucleophilicity",
            lambda: xtb.get_global_descriptor("nucleophilicity", corrected=True),
        ),
        "fukui_electrophilicity": _safe(
            "fukui_electrophilicity", lambda: xtb.get_fukui("electrophilicity")
        ),
        "fukui_nucleophilicity": _safe(
            "fukui_nucleophilicity", lambda: xtb.get_fukui("nucleophilicity")
        ),
        "fukui_radical": _safe("fukui_radical", lambda: xtb.get_fukui("radical")),
    }

    # Process dict fields
    for key in ["charges", "fukui_electrophilicity", "fukui_nucleophilicity", "fukui_radical"]:
        if key in properties and isinstance(properties[key], dict) and properties[key]:
            properties[key + "_min"] = min(properties[key].values())
            properties[key + "_max"] = max(properties[key].values())
            properties[key + "_mean"] = sum(properties[key].values()) / len(
                properties[key].values()
            )

    # Save to file
    if base_path is None:
        base_path = Path(__file__).parent / "molecule_properties"

    base_path.mkdir(exist_ok=True)
    md5_hash = get_smiles_md5(smiles)
    file_path = base_path / f"{md5_hash}.json"

    with open(file_path, "w") as f:
        json.dump(properties, f, indent=4)

    return properties


def get_or_compute_monomer_data(
    smiles: str,
    base_path: Path,
    label: str = "Monomer",
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Return cached monomer features for `smiles`, falling back to a live XTB
    calculation on cache miss when `morfeus` / `xtb-python` are available.

    Returns (data, error). On success error is None; on failure data is None.
    """
    data = load_monomer_features(smiles, base_path)
    if data is not None:
        return data, None
    if not MORFEUS_AVAILABLE:
        return None, f"{label} not in cache and live XTB unavailable"
    try:
        print(f"⚙ Cache miss for {label}; computing features via XTB ({smiles})")
        return calculate_monomer_features(smiles, base_path), None
    except Exception as e:
        return None, f"{label} feature computation failed: {e}"


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
                "lumo": features.get("lumo"),
            }
            return MonomerPreprocessOutput(
                monomer_smiles=monomer_smiles,
                features=minimal_features,
                success=True,
                error=None,
                from_cache=True,
            )

        # Features not found, need to calculate
        if not MORFEUS_AVAILABLE:
            return MonomerPreprocessOutput(
                monomer_smiles=monomer_smiles,
                features={"fukui_radical_max": None, "homo": None, "lumo": None},
                success=False,
                error="morfeus is not available. Cannot calculate monomer features.",
                from_cache=False,
            )

        # Calculate features (this is a long-running operation)
        # Note: In production, you might want to run this in a background task
        calculated_data = calculate_monomer_features(monomer_smiles, base_path)
        all_features = extract_monomer_features_for_model(calculated_data)
        # Return only the minimal features for backward compatibility
        minimal_features = {
            "fukui_radical_max": all_features.get("fukui_radical_max"),
            "homo": all_features.get("homo"),
            "lumo": all_features.get("lumo"),
        }

        return MonomerPreprocessOutput(
            monomer_smiles=monomer_smiles,
            features=minimal_features,
            success=True,
            error=None,
            from_cache=False,
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Monomer preprocessing failed: {str(e)}",
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
            detail="Dataset not loaded. DOI checking is not available.",
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

        if "original_source" in dataset_df.columns:
            # Check for exact match with normalized DOI
            exists = (
                dataset_df["original_source"]
                .astype(str)
                .str.contains(normalized_doi, case=False, na=False, regex=False)
                .any()
            )

        return DOICheckOutput(
            doi=doi_input,
            exists=exists,
            normalized_doi=normalized_doi,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"DOI check failed: {str(e)}"
        )


@app.post("/optimize_reaction", response_model=OptimizeReactionOutput)
async def optimize_reaction(input_data: OptimizeReactionInput):
    """
    Perform reaction optimization by exploring solvent × temperature combinations.

    `solvent_set` selects which solvents to sweep (logP-nearest from the
    dataset, or a curated set); `temperature_mode` selects which temperatures.
    Returns one prediction per (temperature × solvent) cell.
    """
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    if not REACTION_OPTIMIZATION_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reaction optimization module not available",
        )

    if dataset_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dataset not loaded. Reaction optimization requires dataset for finding similar solvents.",
        )

    try:
        # Get base solvent logP
        base_solvent_features = calculate_solvent_features(input_data.solvent_smiles)
        base_logp = base_solvent_features.get("solvent_logP")

        if base_logp is None:
            raise ValueError(f"Could not determine logP for solvent: {input_data.solvent_smiles}")

        # Warm the monomer-feature cache (live XTB on miss) so the inner
        # optimization loop can rely on cached lookups across the grid.
        base_path = Path(__file__).parent / "molecule_properties"
        for label, smi in (
            ("Monomer 1", input_data.monomer1_smiles),
            ("Monomer 2", input_data.monomer2_smiles),
        ):
            _, err = get_or_compute_monomer_data(smi, base_path, label)
            if err:
                raise HTTPException(status_code=400, detail=err)

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
            solvent_set=input_data.solvent_set,
            temperature_mode=input_data.temperature_mode,
        )

        if not predictions:
            return OptimizeReactionOutput(
                success=False,
                error="No valid predictions could be generated",
                predictions=[],
                base_temperature=input_data.temperature,
                temperature_step=input_data.temperature_step,
                base_solvent_logp=float(base_logp),
                timestamp=datetime.now().isoformat(),
            )

        # Clean predictions to remove NaN/Inf values before JSON serialization
        # Replace NaN/Inf with 0.0 for numeric fields
        cleaned_predictions = clean_json_values(predictions, replace_with_zero=True)

        # Convert to Pydantic models
        prediction_models = [OptimizationPrediction(**pred) for pred in cleaned_predictions]

        return OptimizeReactionOutput(
            success=True,
            error=None,
            predictions=prediction_models,
            base_temperature=input_data.temperature,
            temperature_step=input_data.temperature_step,
            base_solvent_logp=float(base_logp),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Reaction optimization failed: {str(e)}",
        )


@app.post("/find_architecture_switch", response_model=ArchitectureSwitchOutput)
async def find_architecture_switch(input_data: ArchitectureSwitchInput):
    """
    Counterfactual search: find the closest reaction conditions that flip the
    predicted copolymer architecture.

    Predicts the baseline reaction (the given solvent + temperature), then
    sweeps a solvent × temperature grid and returns the condition sets whose
    predicted architecture differs from the baseline — ranked by smallest
    change in solvent logP (then temperature). `counterfactuals` is empty when
    nothing in the searched space changes the architecture.
    """
    if not predictor:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model not loaded"
        )

    if not REACTION_OPTIMIZATION_AVAILABLE:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Reaction optimization module not available",
        )

    if dataset_df is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Dataset not loaded. Architecture-switch search requires the dataset.",
        )

    try:
        # Warm the monomer-feature cache (live XTB on miss) before the sweep.
        base_path = Path(__file__).parent / "molecule_properties"
        for label, smi in (
            ("Monomer 1", input_data.monomer1_smiles),
            ("Monomer 2", input_data.monomer2_smiles),
        ):
            _, err = get_or_compute_monomer_data(smi, base_path, label)
            if err:
                raise HTTPException(status_code=400, detail=err)

        result = find_architecture_switches(
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
            solvent_set=input_data.solvent_set,
            temperature_mode=input_data.temperature_mode,
            temperature_step=input_data.temperature_step,
            n_solvents=input_data.n_solvents,
            top_n=input_data.top_n,
        )

        # Ground each counterfactual in literature: find the closest real
        # reaction for this counterfactual's solvent, so the UI can link to
        # its DOI. The reference should use the *same* monomer pair as the
        # query (a different-monomer reaction is only a weak analogy), so the
        # lookup pool is pre-filtered to same-monomer rows when any exist.
        # NN lookup depends only on (monomers, solvent) — not temperature —
        # so results are memoised per solvent across the counterfactual list.
        if BASELINE_LOOKUP_AVAILABLE and train_df is not None and result["counterfactuals"]:
            query_pair = monomer_pair_key(input_data.monomer1_smiles, input_data.monomer2_smiles)
            same_monomer_df = None
            if query_pair is not None:
                pair_keys = train_monomer_pair_keys(train_df)
                positions = [i for i, k in enumerate(pair_keys) if k == query_pair]
                if positions:
                    same_monomer_df = train_df.iloc[positions]

            # When the pair is in the training data, search only those rows;
            # otherwise fall back to the full pool and flag the mismatch.
            lookup_pool = same_monomer_df if same_monomer_df is not None else train_df
            same_monomers = same_monomer_df is not None and query_pair is not None

            nn_by_solvent: Dict[str, Optional[Dict]] = {}
            for cf in result["counterfactuals"]:
                solvent = cf["solvent_smiles"]
                if solvent not in nn_by_solvent:
                    try:
                        neighbors = find_top_k_nearest_neighbors(
                            test_monomer1_smiles=input_data.monomer1_smiles,
                            test_monomer2_smiles=input_data.monomer2_smiles,
                            test_solvent_smiles=solvent,
                            df_train=lookup_pool,
                            k=1,
                            fp_dict=fingerprint_cache,
                        )
                        nn_by_solvent[solvent] = neighbors[0] if neighbors else None
                    except Exception as e:
                        print(f"⚠ NN reference lookup failed for solvent {solvent}: {e}")
                        nn_by_solvent[solvent] = None
                ref = nn_by_solvent[solvent]
                cf["reference"] = ref
                cf["reference_same_monomers"] = (
                    same_monomers if (ref is not None and query_pair is not None) else None
                )

        cleaned = clean_json_values(result, replace_with_zero=True)
        return ArchitectureSwitchOutput(
            success=True,
            error=None,
            baseline=OptimizationPrediction(**cleaned["baseline"]),
            counterfactuals=[ArchitectureSwitchCandidate(**c) for c in cleaned["counterfactuals"]],
            n_evaluated=cleaned["n_evaluated"],
            timestamp=datetime.now().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Architecture-switch search failed: {str(e)}",
        )


# ============================================================================
# Error Handlers
# ============================================================================


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle any unhandled exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "error": str(exc)},
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

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
