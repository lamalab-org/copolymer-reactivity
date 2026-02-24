"""
Solubility check module using the simple logP model from experiments/case_studies/negative_data.

This module checks if both monomers are soluble in the solvent using a binary classification model
that uses only logP features (monomer1_logP, monomer2_logP, solvent_logP).

Class 0 = Soluble (Alternating/Random) - No solubility issues
Class 1 = Insoluble (Homopolymer) - Solubility issues
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from copolpredictor.inference import CopolymerPredictor
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    print("Warning: CopolymerPredictor not available. Solubility check will be disabled.")


# Global solubility model instance
solubility_model: Optional[CopolymerPredictor] = None


def load_solubility_model(model_path: Optional[str] = None) -> Optional[CopolymerPredictor]:
    """
    Load the solubility (logP) model.
    
    Args:
        model_path: Path to model bundle. If None, uses default path.
        
    Returns:
        CopolymerPredictor instance or None if loading fails
    """
    global solubility_model
    
    if not PREDICTOR_AVAILABLE:
        return None
    
    if solubility_model is not None:
        return solubility_model
    
    if model_path is None:
        # Default path: experiments/case_studies/negative_data/results/model_bundle_simple_logp
        api_dir = Path(__file__).parent
        project_root = api_dir.parent.parent
        default_path = project_root / "experiments" / "case_studies" / "negative_data" / "results" / "model_bundle_simple_logp"
        model_path = str(default_path)
    
    try:
        if not os.path.exists(model_path):
            print(f"Warning: Solubility model not found at {model_path}")
            return None
        
        print(f"Loading solubility model from {model_path}...")
        solubility_model = CopolymerPredictor(model_path)
        print("✓ Solubility model loaded successfully")
        return solubility_model
    except Exception as e:
        print(f"Warning: Failed to load solubility model: {e}")
        return None


def calculate_logp(smiles: str) -> Optional[float]:
    """
    Calculate logP for a SMILES string.
    
    Args:
        smiles: SMILES string
        
    Returns:
        logP value or None if calculation fails
    """
    try:
        if pd.isna(smiles) or not smiles or not isinstance(smiles, str):
            return None
        
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        
        return float(Descriptors.MolLogP(mol))
    except Exception:
        return None


def check_solubility(
    monomer1_smiles: str,
    monomer2_smiles: str,
    solvent_smiles: str,
    model: Optional[CopolymerPredictor] = None
) -> Tuple[bool, Optional[float]]:
    """
    Check if both monomers are soluble in the solvent.
    
    Uses the simple logP model to predict solubility:
    - Class 0 = Soluble (no issues)
    - Class 1 = Insoluble (solubility issues)
    
    Args:
        monomer1_smiles: SMILES of first monomer
        monomer2_smiles: SMILES of second monomer
        solvent_smiles: SMILES of solvent
        model: Optional CopolymerPredictor instance. If None, uses global model.
        
    Returns:
        Tuple of (has_solubility_issue, confidence):
            - has_solubility_issue: True if Class 1 (insoluble), False if Class 0 (soluble)
            - confidence: Prediction confidence (0-1) or None if check failed
    """
    # Load model if not provided
    if model is None:
        model = load_solubility_model()
    
    if model is None:
        # If model not available, return None (unknown)
        return (None, None)
    
    # Calculate logP values
    monomer1_logp = calculate_logp(monomer1_smiles)
    monomer2_logp = calculate_logp(monomer2_smiles)
    solvent_logp = calculate_logp(solvent_smiles)
    
    # Check if all logP values are available
    if any(x is None for x in [monomer1_logp, monomer2_logp, solvent_logp]):
        return (None, None)
    
    # Prepare features
    features = {
        'monomer1_logP': float(monomer1_logp),
        'monomer2_logP': float(monomer2_logp),
        'solvent_logP': float(solvent_logp)
    }
    
    try:
        # Make prediction
        results = model.predict_with_confidence(features)
        
        pred_class = int(results['predictions'][0])
        confidence = float(results['confidence'][0])
        
        # Class 0 = Soluble (no issues), Class 1 = Insoluble (has issues)
        has_issue = (pred_class == 1)
        
        return (has_issue, confidence)
    except Exception as e:
        print(f"Warning: Solubility check failed: {e}")
        return (None, None)


def get_solubility_issue_flag(
    monomer1_smiles: str,
    monomer2_smiles: str,
    solvent_smiles: str,
    model: Optional[CopolymerPredictor] = None
) -> int:
    """
    Get solubility issue flag (0 = no issues, 1 = has issues).
    
    This is a convenience wrapper around check_solubility that returns
    an integer flag suitable for API responses.
    
    Args:
        monomer1_smiles: SMILES of first monomer
        monomer2_smiles: SMILES of second monomer
        solvent_smiles: SMILES of solvent
        model: Optional CopolymerPredictor instance
        
    Returns:
        Integer flag: 0 = no solubility issues, 1 = solubility issues, -1 = unknown/check failed
    """
    has_issue, confidence = check_solubility(monomer1_smiles, monomer2_smiles, solvent_smiles, model)
    
    if has_issue is None:
        return -1  # Unknown/check failed
    elif has_issue:
        return 1  # Has solubility issues
    else:
        return 0  # No solubility issues
