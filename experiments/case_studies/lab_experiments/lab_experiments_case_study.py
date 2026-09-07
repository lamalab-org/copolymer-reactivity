#!/usr/bin/env python3
"""
Lab Experiments Case Study

Use the final copolymerization model to predict the class for two
experimentally tested copolymerisations:

  1) Acrylonitrile + N-vinyl-5-pyrrolidone in chloroform
  2) Styrene + 1-octene in chloroform
  3) Butyl acrylate + Vinyl acetate in toluene

Assumptions (from lab setup):
  - Polymerisation type: free radical
  - Method: solvent polymerisation
  - Temperature: 70 °C

The script:
  - builds feature vectors via the same logic as the API (`preprocess_all`)
  - uses the final model bundle in `copol_prediction/artifacts/model_bundle`
  - prints predicted class and confidence for each reaction.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------------

# PROJECT_ROOT: go up 4 levels from this file:
# experiments/case_studies/lab_experiments/lab_experiments_case_study.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, str(Path(PROJECT_ROOT) / "src"))
sys.path.insert(0, str(Path(PROJECT_ROOT) / "copol_prediction"))
sys.path.insert(0, str(Path(PROJECT_ROOT) / "copol_prediction" / "api"))

# Reuse preprocessing helpers from API
from app import (  # type: ignore
    calculate_solvent_features,
    extract_monomer_features_for_model,
    load_monomer_features,
)

from copolpredictor import prediction_utils
from copolpredictor.inference import CopolymerPredictor

try:
    # Nearest-neighbor lookup (used for "voting": only predict when both agree)
    from baseline_lookup import find_top_k_nearest_neighbors  # type: ignore

    BASELINE_LOOKUP_AVAILABLE = True
except Exception:
    BASELINE_LOOKUP_AVAILABLE = False


MODEL_PATH = os.path.join(PROJECT_ROOT, "copol_prediction", "artifacts", "model_bundle")
API_DATA_DIR = os.path.join(PROJECT_ROOT, "copol_prediction", "api", "data")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "copol_prediction", "artifacts", "data_splits")


def load_embeddings() -> tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    """
    Load method and polytype embeddings from the same JSON files
    the API uses.
    """
    import json

    method_path = Path(API_DATA_DIR) / "method_emb_pca_values.json"
    polytype_path = Path(API_DATA_DIR) / "polytype_emb_pca_values.json"

    with open(method_path, "r") as f:
        method_emb = json.load(f)
    with open(polytype_path, "r") as f:
        polytype_emb = json.load(f)

    return method_emb, polytype_emb


def build_features_for_reaction(
    monomer1_smiles: str,
    monomer2_smiles: str,
    solvent_smiles: str,
    method: str,
    polytype: str,
    temperature: float,
    method_embeddings: Dict[str, Dict[str, float]],
    polytype_embeddings: Dict[str, Dict[str, float]],
    predictor: CopolymerPredictor,
) -> Dict[str, float]:
    """
    Build a single feature dict using the same logic as `preprocess_all`
    in the API (without nearest neighbors, papers, etc.).
    """
    base_path = Path(PROJECT_ROOT) / "copol_prediction" / "api" / "molecule_properties"

    # --- Monomer features (cached JSON, created by monomer_feature_calculation.py) ---
    m1_data = load_monomer_features(monomer1_smiles, base_path)
    if not m1_data:
        raise RuntimeError(f"No cached monomer features found for monomer1: {monomer1_smiles}")
    m1_feat = extract_monomer_features_for_model(m1_data)

    m2_data = load_monomer_features(monomer2_smiles, base_path)
    if not m2_data:
        raise RuntimeError(f"No cached monomer features found for monomer2: {monomer2_smiles}")
    m2_feat = extract_monomer_features_for_model(m2_data)

    # --- Solvent features (RDKit descriptors, same as API) ---
    solv_feat = calculate_solvent_features(solvent_smiles)
    if any(v is None for v in solv_feat.values()):
        raise RuntimeError(f"Failed to calculate solvent features for {solvent_smiles}")

    # --- Embeddings ---
    if method not in method_embeddings:
        raise RuntimeError(f"Method '{method}' not found in method embeddings.")
    if polytype not in polytype_embeddings:
        raise RuntimeError(f"Polytype '{polytype}' not found in polytype embeddings.")

    method_emb = method_embeddings[method]
    polytype_emb = polytype_embeddings[polytype]

    # --- Combine all features (mirrors preprocess_all) ---
    features: Dict[str, Any] = {
        # Monomer 1
        "fukui_radical_max_1": m1_feat.get("fukui_radical_max"),
        "global_electrophilicity_1": m1_feat.get("global_electrophilicity"),
        "global_nucleophilicity_1": m1_feat.get("global_nucleophilicity"),
        "dipole_x_1": m1_feat.get("dipole_x"),
        "dipole_y_1": m1_feat.get("dipole_y"),
        "dipole_z_1": m1_feat.get("dipole_z"),
        # Monomer 2
        "fukui_radical_max_2": m2_feat.get("fukui_radical_max"),
        "global_electrophilicity_2": m2_feat.get("global_electrophilicity"),
        "global_nucleophilicity_2": m2_feat.get("global_nucleophilicity"),
        "dipole_x_2": m2_feat.get("dipole_x"),
        "dipole_y_2": m2_feat.get("dipole_y"),
        "dipole_z_2": m2_feat.get("dipole_z"),
        # HOMO-LUMO combinations
        "delta_HOMO_LUMO_AA": (
            (m1_feat.get("homo") - m1_feat.get("lumo"))
            if (m1_feat.get("homo") is not None and m1_feat.get("lumo") is not None)
            else None
        ),
        "delta_HOMO_LUMO_AB": (
            (m1_feat.get("homo") - m2_feat.get("lumo"))
            if (m1_feat.get("homo") is not None and m2_feat.get("lumo") is not None)
            else None
        ),
        "delta_HOMO_LUMO_BB": (
            (m2_feat.get("homo") - m2_feat.get("lumo"))
            if (m2_feat.get("homo") is not None and m2_feat.get("lumo") is not None)
            else None
        ),
        "delta_HOMO_LUMO_BA": (
            (m2_feat.get("homo") - m1_feat.get("lumo"))
            if (m2_feat.get("homo") is not None and m1_feat.get("lumo") is not None)
            else None
        ),
        # Other features
        "temperature": float(temperature),
        "polytype_emb_1": polytype_emb["pca_1"],
        "polytype_emb_2": polytype_emb["pca_2"],
        "method_emb_1": method_emb["pca_1"],
        "method_emb_2": method_emb["pca_2"],
        "solvent_logP": solv_feat["solvent_logP"],
        "solvent_TPSA": solv_feat["solvent_TPSA"],
        "solvent_HBD": solv_feat["solvent_HBD"],
        "solvent_FractionCSP3": solv_feat["solvent_FractionCSP3"],
    }

    # Ensure all model-required features exist (fill missing with None)
    for req in predictor.features:
        if req not in features:
            features[req] = None

    # Convert None to np.nan for XGBoost
    return {k: (np.nan if v is None else float(v)) for k, v in features.items()}


def main():
    print("=" * 60)
    print("LAB EXPERIMENTS CASE STUDY – FINAL MODEL PREDICTIONS")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from {MODEL_PATH} ...")
    predictor = CopolymerPredictor(MODEL_PATH)
    print(f"  ✓ Model loaded ({len(predictor.features)} features)")

    # Load training set for lookup voting
    train_df = None
    if BASELINE_LOOKUP_AVAILABLE:
        train_path = os.path.join(SPLIT_DIR, "train.csv")
        if os.path.exists(train_path):
            try:
                train_df = pd.read_csv(train_path)
                print(f"  ✓ Lookup pool loaded ({len(train_df)} rows) from {train_path}")
            except Exception as e:
                print(f"  ⚠ Failed to load lookup pool from {train_path}: {e}")
                train_df = None
        else:
            print(f"  ⚠ Lookup pool not found at {train_path} (voting disabled)")

    # Load embeddings
    print("\nLoading embeddings (method & polytype) ...")
    method_emb, polytype_emb = load_embeddings()
    print(f"  ✓ Methods:  {list(method_emb.keys())}")
    print(f"  ✓ Polytypes:{list(polytype_emb.keys())}")

    # All lab reactions share these conditions
    method = "solvent"
    polytype = "free radical"
    temperature = 70.0

    # Define lab systems (SMILES)
    systems = [
        {
            "name": "Acrylonitrile + N-vinyl-5-pyrrolidone in chloroform",
            "monomer1_smiles": "C=CC#N",  # acrylonitrile
            "monomer2_smiles": "C=CN1CCCC1=O",  # N-vinyl-2-pyrrolidone
            "solvent_smiles": "ClC(Cl)Cl",  # chloroform
        },
        {
            "name": "Styrene + 1-octene in chloroform",
            "monomer1_smiles": "C=CC1=CC=CC=C1",  # styrene
            "monomer2_smiles": "CCCCCCCC=C",  # 1-octene
            "solvent_smiles": "ClC(Cl)Cl",  # chloroform
        },
        {
            "name": "Butyl acrylate + Vinyl acetate in toluene",
            "monomer1_smiles": "C=CC(=O)OCCCC",  # butyl acrylate
            "monomer2_smiles": "C=COC(C)=O",  # vinyl acetate
            "solvent_smiles": "CC1=CC=CC=C1",  # toluene
        },
        {
            "name": "Acrylonitrile + Ethyl methacrylate in chloroform",
            "monomer1_smiles": "C=CC#N",  # acrylonitrile
            "monomer2_smiles": "C=C(C)C(=O)OCC",  # ethyl methacrylate
            "solvent_smiles": "ClC(Cl)Cl",  # chloroform
        },
        {
            "name": "Vinyl acetate + Ethyl methacrylate in chloroform",
            "monomer1_smiles": "C=COC(C)=O",  # vinyl acetate
            "monomer2_smiles": "C=C(C)C(=O)OCC",  # ethyl methacrylate
            "solvent_smiles": "ClC(Cl)Cl",  # chloroform
        },
        {
            "name": "Methacrylate + N-vinyl-2-pyrrolidone in chloroform",
            "monomer1_smiles": "C=C(C)C(=O)O",  # methacrylate (methacrylic acid)
            "monomer2_smiles": "C=CN1CCCC1=O",  # N-vinyl-2-pyrrolidone
            "solvent_smiles": "ClC(Cl)Cl",  # chloroform
        },
    ]

    class_map = {
        0: "Alternating",
        1: "Random",
        2: "Gradient",
    }

    rows = []

    for sys_idx, sys_desc in enumerate(systems, start=1):
        print("\n" + "-" * 60)
        print(f"System {sys_idx}: {sys_desc['name']}")
        try:
            features = build_features_for_reaction(
                monomer1_smiles=sys_desc["monomer1_smiles"],
                monomer2_smiles=sys_desc["monomer2_smiles"],
                solvent_smiles=sys_desc["solvent_smiles"],
                method=method,
                polytype=polytype,
                temperature=temperature,
                method_embeddings=method_emb,
                polytype_embeddings=polytype_emb,
                predictor=predictor,
            )

            # Predict
            result = predictor.predict_with_confidence(pd.DataFrame([features]))
            pred_class = int(result["predictions"][0])
            confidence = float(result["confidence"][0])

            print(f"  Predicted class   : {pred_class} ({class_map.get(pred_class, 'unknown')})")
            print(f"  Confidence        : {confidence:.3f}")

            # Voting: Lookup + XGBoost (only predict when both agree)
            lookup_class = None
            models_agree = None
            voted_class = None
            voted_confidence = None
            top_neighbor_similarity = None

            if train_df is not None:
                try:
                    neighbors = find_top_k_nearest_neighbors(
                        test_monomer1_smiles=sys_desc["monomer1_smiles"],
                        test_monomer2_smiles=sys_desc["monomer2_smiles"],
                        test_solvent_smiles=sys_desc["solvent_smiles"],
                        df_train=train_df,
                        k=10,
                    )
                    if neighbors:
                        lookup_class = int(neighbors[0].get("predicted_class"))
                        top_neighbor_similarity = float(neighbors[0].get("similarity"))

                        # Show nearest-neighbor datapoint (human-readable fields)
                        nn = neighbors[0]
                        nn_m1 = nn.get("monomer1_name") or nn.get("monomer1_smiles")
                        nn_m2 = nn.get("monomer2_name") or nn.get("monomer2_smiles")
                        nn_solv = nn.get("solvent_name") or nn.get("solvent_smiles")
                        nn_temp = nn.get("temperature")
                        nn_method = nn.get("method")
                        nn_polytype = nn.get("polytype")

                        print("  Nearest neighbor   :")
                        print(f"    - Monomers       : {nn_m1}  +  {nn_m2}")
                        print(f"    - Solvent        : {nn_solv}")
                        if nn_temp is not None:
                            print(f"    - Temperature    : {float(nn_temp):.1f} °C")
                        nn_r1 = nn.get("r1")
                        nn_r2 = nn.get("r2")
                        nn_r1r2 = nn.get("r1r2")
                        if (nn_r1 is not None) or (nn_r2 is not None) or (nn_r1r2 is not None):
                            r1_s = f"{float(nn_r1):.4g}" if nn_r1 is not None else "NA"
                            r2_s = f"{float(nn_r2):.4g}" if nn_r2 is not None else "NA"
                            rprod_s = f"{float(nn_r1r2):.4g}" if nn_r1r2 is not None else "NA"
                            print(f"    - r-values       : r1={r1_s}, r2={r2_s}, r1r2={rprod_s}")
                        if nn_method:
                            print(f"    - Method         : {nn_method}")
                        if nn_polytype:
                            print(f"    - Polytype       : {nn_polytype}")

                        models_agree = bool(pred_class == lookup_class)
                        if models_agree:
                            voted_class = pred_class
                            voted_confidence = confidence
                except Exception as e:
                    print(f"  ⚠ Lookup voting failed: {e}")

            if train_df is not None:
                if lookup_class is None:
                    print("  Voting (Lookup+XGB): lookup unavailable → no voted prediction")
                else:
                    print(f"  Lookup class       : {lookup_class}")
                    if top_neighbor_similarity is not None:
                        print(f"  Top similarity     : {top_neighbor_similarity:.3f}")
                    print(f"  Models agree       : {models_agree}")
                    if voted_class is None:
                        print("  VOTED prediction   : (abstain due to disagreement)")
                    else:
                        print(
                            f"  VOTED class        : {voted_class} ({class_map.get(voted_class, 'unknown')})"
                        )

            rows.append(
                {
                    "system": sys_desc["name"],
                    "monomer1_smiles": sys_desc["monomer1_smiles"],
                    "monomer2_smiles": sys_desc["monomer2_smiles"],
                    "solvent_smiles": sys_desc["solvent_smiles"],
                    "temperature_C": temperature,
                    "method": method,
                    "polytype": polytype,
                    # Base model (XGBoost) output
                    "xgb_predicted_class": pred_class,
                    "xgb_predicted_class_name": class_map.get(pred_class, "unknown"),
                    "xgb_confidence": confidence,
                    # Lookup + XGBoost voting output
                    "lookup_class": lookup_class,
                    "lookup_top_similarity": top_neighbor_similarity,
                    "models_agree": models_agree,
                    "voted_class": voted_class,
                    "voted_class_name": (
                        class_map.get(voted_class, "unknown") if voted_class is not None else None
                    ),
                    "voted_confidence": voted_confidence,
                }
            )
        except Exception as e:
            print(f"  ✗ Error for this system: {e}")

    # Save small CSV summary next to this script
    if rows:
        out_path = os.path.join(os.path.dirname(__file__), "lab_experiments_predictions.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print("\n" + "=" * 60)
        print(f"Predictions saved to: {out_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
