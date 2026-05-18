"""
Reaction optimization module for exploring different solvent and temperature combinations.

This module provides functionality to:
1. Find similar solvents based on logP
2. Generate a 3x3 grid of predictions (3 temperatures × 3 solvents)
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


def calculate_solvent_logp(smiles: str) -> Optional[float]:
    """Calculate logP for a solvent SMILES string."""
    try:
        if pd.isna(smiles) or not smiles:
            return None

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None

        return float(Descriptors.MolLogP(mol))
    except Exception:
        return None


def find_similar_solvents(
    target_logp: float, dataset_df: pd.DataFrame, n_solvents: int = 3, tolerance: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Find solvents with similar logP values from the dataset.

    Args:
        target_logp: Target logP value
        dataset_df: DataFrame containing solvent data
        n_solvents: Number of similar solvents to return (default: 3)
        tolerance: Maximum logP difference to consider (default: 1.0)

    Returns:
        List of dictionaries with solvent information:
            - smiles: Solvent SMILES
            - name: Solvent name
            - logp: logP value
            - logp_diff: Difference from target logP
    """
    if dataset_df is None or len(dataset_df) == 0:
        return []

    # Get unique solvents with their logP values
    solvents = []
    seen_smiles = set()

    for _, row in dataset_df.iterrows():
        solvent_smiles = row.get("solvent_smiles")
        if pd.isna(solvent_smiles) or not solvent_smiles:
            continue

        # Skip if we've already seen this SMILES
        if solvent_smiles in seen_smiles:
            continue
        seen_smiles.add(solvent_smiles)

        # Get logP (prefer from dataset, calculate if needed)
        logp = row.get("solvent_logP")
        if pd.isna(logp):
            logp = calculate_solvent_logp(solvent_smiles)

        if logp is None or pd.isna(logp):
            continue

        solvent_name = row.get("solvent", "")
        if pd.isna(solvent_name) or not solvent_name:
            solvent_name = solvent_smiles

        solvents.append(
            {
                "smiles": solvent_smiles,
                "name": str(solvent_name),
                "logp": float(logp),
                "logp_diff": abs(float(logp) - target_logp),
            }
        )

    # Filter by tolerance, exclude solvents with exactly the same logP (logp_diff == 0.0)
    # We want solvents that are similar but not identical
    similar_solvents = [
        s
        for s in solvents
        if s["logp_diff"] <= tolerance and s["logp_diff"] > 0.0  # Exclude exact matches
    ]
    similar_solvents.sort(key=lambda x: x["logp_diff"])

    # If we don't have enough within tolerance, expand search (still exclude exact matches)
    if len(similar_solvents) < n_solvents:
        # Sort all solvents by logP difference, excluding exact matches
        all_solvents = [s for s in solvents if s["logp_diff"] > 0.0]
        all_solvents.sort(key=lambda x: x["logp_diff"])
        similar_solvents = all_solvents[:n_solvents]

    return similar_solvents[:n_solvents]


def generate_temperature_grid(base_temperature: float, step: float = 20.0) -> List[float]:
    """
    Generate a grid of 3 temperatures around the base temperature.

    Args:
        base_temperature: Base temperature in Celsius
        step: Temperature step size (default: 20.0°C)

    Returns:
        List of 3 temperatures: [base - step, base, base + step]
    """
    return [
        max(0.0, base_temperature - step),  # Ensure non-negative
        base_temperature,
        base_temperature + step,
    ]


def create_optimization_grid(
    monomer1_smiles: str,
    monomer2_smiles: str,
    base_solvent_smiles: str,
    base_temperature: float,
    method: str,
    polytype: str,
    dataset_df: pd.DataFrame,
    method_embeddings: Dict[str, Dict[str, float]],
    polytype_embeddings: Dict[str, Dict[str, float]],
    predictor,
    load_monomer_features_func,
    extract_monomer_features_func,
    calculate_solvent_features_func,
    temperature_step: float = 20.0,
    n_solvents: int = 3,
) -> List[Dict]:
    """
    Create a 3x3 grid of predictions by varying temperature and solvent.

    Args:
        monomer1_smiles: First monomer SMILES
        monomer2_smiles: Second monomer SMILES
        base_solvent_smiles: Base solvent SMILES
        base_temperature: Base temperature in Celsius
        method: Polymerization method
        polytype: Polymerization type
        dataset_df: Dataset DataFrame for finding similar solvents
        method_embeddings: Method embeddings dictionary
        polytype_embeddings: Polytype embeddings dictionary
        predictor: CopolymerPredictor instance
        load_monomer_features_func: Function to load monomer features
        extract_monomer_features_func: Function to extract monomer features
        calculate_solvent_features_func: Function to calculate solvent features
        temperature_step: Temperature step size (default: 20.0°C)
        n_solvents: Number of solvents to use (default: 3)

    Returns:
        List of prediction results, each containing:
            - temperature: Temperature used
            - solvent_smiles: Solvent SMILES used
            - solvent_name: Solvent name
            - solvent_logp: Solvent logP
            - predicted_class: Predicted class
            - class_probabilities: Class probabilities
            - confidence: Prediction confidence
    """
    # Get base solvent logP
    base_solvent_features = calculate_solvent_features_func(base_solvent_smiles)
    base_logp = base_solvent_features.get("solvent_logP")

    if base_logp is None:
        # Fallback: calculate logP
        base_logp = calculate_solvent_logp(base_solvent_smiles)

    if base_logp is None:
        raise ValueError(f"Could not determine logP for solvent: {base_solvent_smiles}")

    # Find similar solvents (exclude base solvent, search for n_solvents - 1 additional ones)
    # We want: base solvent + (n_solvents - 1) similar solvents = n_solvents total

    # First, find similar solvents but exclude the base solvent
    similar_solvents = find_similar_solvents(
        target_logp=base_logp,
        dataset_df=dataset_df,
        n_solvents=n_solvents + 5,  # Get extra to filter out base
        tolerance=1.0,
    )

    # Filter out the base solvent from similar solvents
    similar_solvents = [s for s in similar_solvents if s["smiles"] != base_solvent_smiles]

    # Take only (n_solvents - 1) most similar ones
    similar_solvents = similar_solvents[: n_solvents - 1]

    # Get base solvent name from dataset
    base_solvent_name = base_solvent_smiles  # Fallback to SMILES
    for _, row in dataset_df.iterrows():
        if row.get("solvent_smiles") == base_solvent_smiles:
            potential_name = row.get("solvent", "")
            if pd.notna(potential_name) and potential_name:
                base_solvent_name = str(potential_name)
                break

    # Create base solvent info
    base_solvent_info = {
        "smiles": base_solvent_smiles,
        "name": base_solvent_name,
        "logp": base_logp,
        "logp_diff": 0.0,
    }

    # Combine: base solvent + similar solvents, then sort by logP difference
    all_solvents = [base_solvent_info] + similar_solvents
    all_solvents.sort(key=lambda x: x["logp_diff"])

    # Final list: base + (n_solvents - 1) similar = n_solvents total
    similar_solvents = all_solvents[:n_solvents]

    # Generate temperature grid
    temperatures = generate_temperature_grid(base_temperature, temperature_step)

    # Load monomer features once (they don't change)
    from pathlib import Path

    base_path = Path(__file__).parent / "molecule_properties"
    m1_data = load_monomer_features_func(monomer1_smiles, base_path)
    m2_data = load_monomer_features_func(monomer2_smiles, base_path)

    if not m1_data or not m2_data:
        raise ValueError("Could not load monomer features")

    m1_features = extract_monomer_features_func(m1_data)
    m2_features = extract_monomer_features_func(m2_data)

    # Get embeddings
    if method not in method_embeddings:
        raise ValueError(f"Method '{method}' not found in embeddings")
    method_emb = method_embeddings[method]

    if polytype not in polytype_embeddings:
        raise ValueError(f"Polytype '{polytype}' not found in embeddings")
    polytype_emb = polytype_embeddings[polytype]

    # Generate predictions for all combinations
    results = []

    for temp in temperatures:
        for solvent_info in similar_solvents:
            solvent_smiles = solvent_info["smiles"]
            solvent_name = solvent_info["name"]
            solvent_logp = solvent_info["logp"]

            # Calculate solvent features
            solvent_features = calculate_solvent_features_func(solvent_smiles)

            if any(
                v is None
                for v in [
                    solvent_features.get("solvent_logP"),
                    solvent_features.get("solvent_TPSA"),
                    solvent_features.get("solvent_HBD"),
                    solvent_features.get("solvent_FractionCSP3"),
                ]
            ):
                # Skip if solvent features can't be calculated
                continue

            # Build feature vector using the same model-aware helper as
            # /preprocess_all so the optimizer feeds the model the columns it
            # was actually trained on (e.g. `solvent_logp`, `charges_min_1`).
            from app import assemble_model_features

            features = assemble_model_features(
                m1_features=m1_features,
                m2_features=m2_features,
                solvent_features=solvent_features,
                polytype_emb=polytype_emb,
                method_emb=method_emb,
                temperature=temp,
            )

            # Make prediction
            try:
                pred_results = predictor.predict_with_confidence(features)

                pred_class = int(pred_results["predictions"][0])
                proba = pred_results["probabilities"][0]
                confidence = float(pred_results["confidence"][0])

                from app import CLASS_LABELS

                predicted_class_name = CLASS_LABELS.get(pred_class, "unknown")

                results.append(
                    {
                        "temperature": float(temp),
                        "solvent_smiles": solvent_smiles,
                        "solvent_name": solvent_name,
                        "solvent_logp": solvent_logp,
                        "predicted_class": pred_class,
                        "predicted_class_name": predicted_class_name,
                        "class_probabilities": {
                            f"class_{i}": float(proba[i]) for i in range(len(proba))
                        },
                        "confidence": confidence,
                    }
                )
            except Exception as e:
                # Skip this combination if prediction fails
                print(f"Warning: Prediction failed for temp={temp}, solvent={solvent_smiles}: {e}")
                continue

    return results
