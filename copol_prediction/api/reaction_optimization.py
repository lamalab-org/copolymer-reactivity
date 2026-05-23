"""
Reaction optimization module for exploring different solvent and temperature combinations.

This module provides functionality to:
1. Find similar solvents based on logP
2. Generate a 3x3 grid of predictions (3 temperatures × 3 solvents)
"""

import functools
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from class_labels import CLASS_LABELS
from rdkit import Chem
from rdkit.Chem import Descriptors


@functools.lru_cache(maxsize=256)
def _logp_of(smiles: str) -> Optional[float]:
    """RDKit MolLogP for a SMILES string — memoised (pure function)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return float(Descriptors.MolLogP(mol))
    except Exception:
        return None


def calculate_solvent_logp(smiles: str) -> Optional[float]:
    """Calculate logP for a solvent SMILES string (memoised via _logp_of)."""
    if pd.isna(smiles) or not smiles:
        return None
    return _logp_of(str(smiles))


# Memo for the dataset's distinct (smiles, name, logp) solvents. The dataset
# is loaded once at startup and never mutated, so memoising by id() is safe
# for the process lifetime — keyed by id() so a (hypothetical) reload yields
# a fresh table.
_UNIQUE_SOLVENTS_MEMO: Dict[int, List[Dict[str, Any]]] = {}


def _unique_solvents(dataset_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Distinct solvents from the dataset as {smiles, name, logp} dicts.

    Computed once per DataFrame (memoised). Replaces a per-call
    .iterrows() scan over the full ~5000-row dataset — the unique-solvent
    table is static at runtime, so there is nothing to recompute.
    """
    cached = _UNIQUE_SOLVENTS_MEMO.get(id(dataset_df))
    if cached is not None:
        return cached

    keep = [c for c in ("solvent_smiles", "solvent", "solvent_logP") if c in dataset_df.columns]
    sub = dataset_df[keep].drop_duplicates("solvent_smiles")

    out: List[Dict[str, Any]] = []
    for record in sub.itertuples(index=False):
        row = record._asdict()
        smiles = row.get("solvent_smiles")
        if pd.isna(smiles) or not smiles:
            continue
        logp = row.get("solvent_logP")
        if logp is None or pd.isna(logp):
            logp = calculate_solvent_logp(smiles)
        if logp is None or pd.isna(logp):
            continue
        name = row.get("solvent")
        if name is None or pd.isna(name) or not name:
            name = smiles
        out.append({"smiles": smiles, "name": str(name), "logp": float(logp)})

    _UNIQUE_SOLVENTS_MEMO[id(dataset_df)] = out
    return out


def find_similar_solvents(
    target_logp: float, dataset_df: pd.DataFrame, n_solvents: int = 3, tolerance: float = 1.0
) -> List[Dict[str, Any]]:
    """
    Find solvents with similar logP values from the dataset.

    Returns a list of {smiles, name, logp, logp_diff} dicts, where
    logp_diff = |logp - target_logp|. Solvents at exactly the target logP
    are excluded (we want similar-but-not-identical solvents).
    """
    if dataset_df is None or len(dataset_df) == 0:
        return []

    # The distinct-solvent table is static; only logp_diff depends on the
    # per-call target_logp, so recompute just that.
    solvents = [
        {**s, "logp_diff": abs(s["logp"] - target_logp)} for s in _unique_solvents(dataset_df)
    ]

    similar_solvents = [s for s in solvents if 0.0 < s["logp_diff"] <= tolerance]
    similar_solvents.sort(key=lambda x: x["logp_diff"])

    # If too few within tolerance, expand to the nearest n regardless.
    if len(similar_solvents) < n_solvents:
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


# ============================================================================
# Predefined solvent sets and temperature modes
#
# The string keys here map 1:1 to the frontend <select> option values, so a
# UI dropdown choice can be forwarded to the API verbatim with no translation.
# ============================================================================

# Curated solvent sets. logP is computed on demand (calculate_solvent_logp),
# so only SMILES + display name are stored here.
SOLVENT_SETS: Dict[str, List[Dict[str, str]]] = {
    "common": [
        {"smiles": "O", "name": "water"},
        {"smiles": "CO", "name": "methanol"},
        {"smiles": "CCO", "name": "ethanol"},
        {"smiles": "CC(C)=O", "name": "acetone"},
        {"smiles": "C1CCOC1", "name": "tetrahydrofuran"},
        {"smiles": "Cc1ccccc1", "name": "toluene"},
        {"smiles": "CN(C)C=O", "name": "N,N-dimethylformamide"},
        {"smiles": "CS(C)=O", "name": "dimethyl sulfoxide"},
    ],
    "chlorinated": [
        {"smiles": "ClC(Cl)Cl", "name": "chloroform"},
        {"smiles": "ClCCl", "name": "dichloromethane"},
        {"smiles": "ClCCCl", "name": "1,2-dichloroethane"},
        {"smiles": "ClC(Cl)(Cl)Cl", "name": "carbon tetrachloride"},
        {"smiles": "Clc1ccccc1", "name": "chlorobenzene"},
    ],
    "aromatic": [
        {"smiles": "c1ccccc1", "name": "benzene"},
        {"smiles": "Cc1ccccc1", "name": "toluene"},
        {"smiles": "Cc1ccccc1C", "name": "o-xylene"},
        {"smiles": "Clc1ccccc1", "name": "chlorobenzene"},
        {"smiles": "COc1ccccc1", "name": "anisole"},
    ],
}

# "top3" is not a static list — it means "the dataset solvents closest in
# logP to the base solvent" (the original behaviour of this module).
SOLVENT_SET_CHOICES = ("top3",) + tuple(SOLVENT_SETS)

# Temperature schemes. Each maps to an explicit list of temperatures (°C).
TEMPERATURE_MODE_CHOICES = ("40-80", "20-100", "fixed60", "step20")


def resolve_temperatures(
    temperature_mode: str, base_temperature: float, temperature_step: float = 20.0
) -> List[float]:
    """Translate a temperature_mode key into an explicit list of temperatures.

    `step20` keeps the original base ± step behaviour (so it still honours
    the caller's `temperature` and `temperature_step`); the others are fixed.
    """
    if temperature_mode == "40-80":
        return [40.0, 60.0, 80.0]
    if temperature_mode == "20-100":
        return [20.0, 60.0, 100.0]
    if temperature_mode == "fixed60":
        return [60.0]
    if temperature_mode == "step20":
        return generate_temperature_grid(base_temperature, temperature_step)
    raise ValueError(
        f"Unknown temperature_mode '{temperature_mode}'. "
        f"Expected one of {TEMPERATURE_MODE_CHOICES}."
    )


def resolve_solvents(
    solvent_set: str,
    base_solvent_smiles: str,
    base_logp: Optional[float],
    dataset_df: pd.DataFrame,
    n_solvents: int = 3,
) -> List[Dict[str, Any]]:
    """Translate a solvent_set key into a list of solvent dicts.

    Each dict has: smiles, name, logp, logp_diff (|logp - base_logp|).
    `top3` reproduces the logP-nearest-from-dataset behaviour; the named
    sets return their curated members (logP computed via RDKit).
    """
    if solvent_set == "top3":
        if base_logp is None:
            raise ValueError("base solvent logP required for solvent_set='top3'")
        similar = find_similar_solvents(
            target_logp=base_logp, dataset_df=dataset_df, n_solvents=n_solvents + 5, tolerance=1.0
        )
        similar = [s for s in similar if s["smiles"] != base_solvent_smiles]
        base_name = base_solvent_smiles
        for _, row in dataset_df.iterrows():
            if row.get("solvent_smiles") == base_solvent_smiles:
                potential = row.get("solvent", "")
                if pd.notna(potential) and potential:
                    base_name = str(potential)
                    break
        base_info = {
            "smiles": base_solvent_smiles,
            "name": base_name,
            "logp": base_logp,
            "logp_diff": 0.0,
        }
        combined = [base_info] + similar
        combined.sort(key=lambda s: s["logp_diff"])
        return combined[:n_solvents]

    if solvent_set in SOLVENT_SETS:
        out: List[Dict[str, Any]] = []
        for entry in SOLVENT_SETS[solvent_set]:
            logp = calculate_solvent_logp(entry["smiles"])
            if logp is None:
                continue
            out.append(
                {
                    "smiles": entry["smiles"],
                    "name": entry["name"],
                    "logp": logp,
                    "logp_diff": abs(logp - base_logp) if base_logp is not None else 0.0,
                }
            )
        return out

    raise ValueError(f"Unknown solvent_set '{solvent_set}'. Expected one of {SOLVENT_SET_CHOICES}.")


def _run_condition_grid(
    monomer1_smiles: str,
    monomer2_smiles: str,
    solvents: List[Dict[str, Any]],
    temperatures: List[float],
    method: str,
    polytype: str,
    method_embeddings: Dict[str, Dict[str, float]],
    polytype_embeddings: Dict[str, Dict[str, float]],
    predictor,
    load_monomer_features_func,
    extract_monomer_features_func,
    calculate_solvent_features_func,
) -> List[Dict]:
    """Predict every (solvent x temperature) combination.

    Monomer features and embeddings are resolved once; the prediction loop
    is shared by create_optimization_grid and find_architecture_switches.
    """
    from pathlib import Path

    base_path = Path(__file__).parent / "molecule_properties"
    m1_data = load_monomer_features_func(monomer1_smiles, base_path)
    m2_data = load_monomer_features_func(monomer2_smiles, base_path)
    if not m1_data or not m2_data:
        raise ValueError("Could not load monomer features")
    m1_features = extract_monomer_features_func(m1_data)
    m2_features = extract_monomer_features_func(m2_data)

    if method not in method_embeddings:
        raise ValueError(f"Method '{method}' not found in embeddings")
    method_emb = method_embeddings[method]
    if polytype not in polytype_embeddings:
        raise ValueError(f"Polytype '{polytype}' not found in embeddings")
    polytype_emb = polytype_embeddings[polytype]

    # `assemble_model_features` is imported here (not at module top) to avoid a
    # circular import: app.py imports this module at startup. `CLASS_LABELS`
    # lives in a leaf module, so it is safe to import at the top level.
    from app import assemble_model_features

    # Solvent features do not depend on temperature — compute them once per
    # solvent up front rather than once per (solvent × temperature) cell.
    # Solvents whose features can't be computed are dropped here.
    solvent_feature_map: Dict[str, Dict] = {}
    for solvent_info in solvents:
        smi = solvent_info["smiles"]
        if smi in solvent_feature_map:
            continue
        sf = calculate_solvent_features_func(smi)
        if any(
            sf.get(k) is None
            for k in ("solvent_logP", "solvent_TPSA", "solvent_HBD", "solvent_FractionCSP3")
        ):
            continue
        solvent_feature_map[smi] = sf

    results: List[Dict] = []
    for temp in temperatures:
        for solvent_info in solvents:
            solvent_smiles = solvent_info["smiles"]
            solvent_features = solvent_feature_map.get(solvent_smiles)
            if solvent_features is None:
                continue

            features = assemble_model_features(
                m1_features=m1_features,
                m2_features=m2_features,
                solvent_features=solvent_features,
                polytype_emb=polytype_emb,
                method_emb=method_emb,
                temperature=temp,
            )
            try:
                pred = predictor.predict_with_confidence(features)
                pred_class = int(pred["predictions"][0])
                proba = pred["probabilities"][0]
                results.append(
                    {
                        "temperature": float(temp),
                        "solvent_smiles": solvent_smiles,
                        "solvent_name": solvent_info["name"],
                        "solvent_logp": solvent_info["logp"],
                        "predicted_class": pred_class,
                        "predicted_class_name": CLASS_LABELS.get(pred_class, "unknown"),
                        "class_probabilities": {
                            CLASS_LABELS[i]: float(proba[i]) for i in range(len(proba))
                        },
                        "confidence": float(pred["confidence"][0]),
                    }
                )
            except Exception as e:
                print(f"Warning: Prediction failed for temp={temp}, solvent={solvent_smiles}: {e}")
                continue
    return results


def _base_solvent_logp(base_solvent_smiles: str, calculate_solvent_features_func) -> float:
    """Resolve the base solvent's logP, raising if it cannot be computed."""
    base_logp = calculate_solvent_features_func(base_solvent_smiles).get("solvent_logP")
    if base_logp is None:
        base_logp = calculate_solvent_logp(base_solvent_smiles)
    if base_logp is None:
        raise ValueError(f"Could not determine logP for solvent: {base_solvent_smiles}")
    return float(base_logp)


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
    solvent_set: str = "top3",
    temperature_mode: str = "step20",
) -> List[Dict]:
    """
    Predict a grid of (solvent x temperature) combinations.

    `solvent_set` and `temperature_mode` select which solvents / temperatures
    to sweep; their defaults reproduce the original "base + nearest-logP,
    base +/- step" 3x3 behaviour. See SOLVENT_SET_CHOICES /
    TEMPERATURE_MODE_CHOICES for the accepted values.

    Returns a list of prediction dicts: temperature, solvent_smiles,
    solvent_name, solvent_logp, predicted_class, predicted_class_name,
    class_probabilities, confidence.
    """
    base_logp = _base_solvent_logp(base_solvent_smiles, calculate_solvent_features_func)
    solvents = resolve_solvents(solvent_set, base_solvent_smiles, base_logp, dataset_df, n_solvents)
    temperatures = resolve_temperatures(temperature_mode, base_temperature, temperature_step)
    return _run_condition_grid(
        monomer1_smiles,
        monomer2_smiles,
        solvents,
        temperatures,
        method,
        polytype,
        method_embeddings,
        polytype_embeddings,
        predictor,
        load_monomer_features_func,
        extract_monomer_features_func,
        calculate_solvent_features_func,
    )


def find_architecture_switches(
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
    solvent_set: str = "common",
    temperature_mode: str = "40-80",
    temperature_step: float = 20.0,
    n_solvents: int = 3,
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Counterfactual search: find condition sets that flip the predicted
    architecture, ranked by how little they change.

    The baseline reaction (base solvent + base temperature) is predicted,
    then a solvent x temperature grid is swept. Cells whose predicted class
    differs from the baseline are ranked by smallest |delta logP| from the
    base solvent, with |delta temperature| as the tie-breaker.

    Returns {baseline, counterfactuals, n_evaluated}: `baseline` is the
    starting-point prediction; `counterfactuals` is the top-N ranked list,
    each carrying delta_logp / delta_temperature.
    """
    base_logp = _base_solvent_logp(base_solvent_smiles, calculate_solvent_features_func)

    # Search solvents — the chosen set plus the base solvent itself, so the
    # baseline cell is always present in the grid.
    solvents = resolve_solvents(solvent_set, base_solvent_smiles, base_logp, dataset_df, n_solvents)
    if not any(s["smiles"] == base_solvent_smiles for s in solvents):
        solvents = [
            {
                "smiles": base_solvent_smiles,
                "name": base_solvent_smiles,
                "logp": base_logp,
                "logp_diff": 0.0,
            }
        ] + solvents

    # Search temperatures — the chosen mode plus the base temperature.
    temperatures = resolve_temperatures(temperature_mode, base_temperature, temperature_step)
    if not any(abs(t - base_temperature) < 1e-9 for t in temperatures):
        temperatures = [base_temperature] + temperatures

    grid = _run_condition_grid(
        monomer1_smiles,
        monomer2_smiles,
        solvents,
        temperatures,
        method,
        polytype,
        method_embeddings,
        polytype_embeddings,
        predictor,
        load_monomer_features_func,
        extract_monomer_features_func,
        calculate_solvent_features_func,
    )

    # The baseline cell: base solvent at base temperature.
    baseline = next(
        (
            c
            for c in grid
            if c["solvent_smiles"] == base_solvent_smiles
            and abs(c["temperature"] - base_temperature) < 1e-9
        ),
        None,
    )
    if baseline is None:
        raise ValueError("Baseline reaction could not be predicted")

    # Counterfactuals: cells whose architecture differs from the baseline.
    counterfactuals = []
    for cell in grid:
        if cell["predicted_class"] == baseline["predicted_class"]:
            continue
        enriched = dict(cell)
        enriched["delta_logp"] = cell["solvent_logp"] - base_logp
        enriched["delta_temperature"] = cell["temperature"] - base_temperature
        counterfactuals.append(enriched)

    # Rank: smallest |delta logP| first, |delta temperature| as tie-breaker.
    counterfactuals.sort(key=lambda c: (abs(c["delta_logp"]), abs(c["delta_temperature"])))

    return {
        "baseline": baseline,
        "counterfactuals": counterfactuals[:top_n],
        "n_evaluated": len(grid),
    }
