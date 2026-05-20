"""
Baseline lookup module for finding nearest neighbors in the training database.

This module implements the database lookup approach from experiments/baseline
to find the most similar data points based on Tanimoto similarity of monomers and solvents.
"""

import hashlib
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs, rdFingerprintGenerator

# A DOI is `10.<registrant>/<suffix>`. Training-data JSON files are named
# after the DOI with the single '/' replaced by '_' (dots preserved), e.g.
# 10.1002/pol.1959.1203512832  ->  10.1002_pol.1959.1203512832.json
_DOI_RE = re.compile(r"10\.\d{4,}/\S+")


def doi_from_source_filename(filename: Optional[str]) -> Optional[str]:
    """Recover the DOI from a training row's `source_filename`.

    `source_filename` is the machine-assigned name of the extracted paper
    (set in copolextractor/data_into_csv.py), derived from the DOI the
    download pipeline fetched — so it is a far more reliable provenance
    signal than the LLM-populated `original_source` column.

    Returns the bare DOI (e.g. "10.1002/pol.1959.1203512832"), or None when
    the input is missing or not DOI-shaped (a handful of old papers have no
    DOI and are filed under a citation-style name instead).
    """
    if not isinstance(filename, str) or not filename:
        return None
    stem = filename[:-5] if filename.endswith(".json") else filename
    # Restore only the registrant/suffix separator: the first '_'. DOI
    # suffixes may legitimately contain '_', so a global replace is wrong.
    candidate = stem.replace("_", "/", 1)
    return candidate if _DOI_RE.fullmatch(candidate) else None


def doi_url(doi: Optional[str]) -> Optional[str]:
    """Resolvable https://doi.org/ link for a bare DOI, or None."""
    return f"https://doi.org/{doi}" if doi else None


def get_fingerprint_cache_path(cache_dir: Optional[Path] = None) -> Path:
    """Get the path to the fingerprint cache file."""
    if cache_dir is None:
        cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir / "fingerprints_cache.pkl"


def load_fingerprint_cache(cache_path: Optional[Path] = None) -> Optional[Dict]:
    """
    Load fingerprint cache from disk.

    Args:
        cache_path: Path to cache file (default: cache/fingerprints_cache.pkl)

    Returns:
        Dictionary mapping SMILES to fingerprints, or None if cache doesn't exist
    """
    if cache_path is None:
        cache_path = get_fingerprint_cache_path()

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "rb") as f:
            cache_data = pickle.load(f)
            return cache_data.get("fingerprints", {})
    except Exception as e:
        print(f"Warning: Failed to load fingerprint cache: {e}")
        return None


def save_fingerprint_cache(fp_dict: Dict, cache_path: Optional[Path] = None):
    """
    Save fingerprint cache to disk.

    Args:
        fp_dict: Dictionary mapping SMILES to fingerprints
        cache_path: Path to cache file (default: cache/fingerprints_cache.pkl)
    """
    if cache_path is None:
        cache_path = get_fingerprint_cache_path()

    try:
        cache_data = {"fingerprints": fp_dict, "version": "1.0"}  # For future cache invalidation
        with open(cache_path, "wb") as f:
            pickle.dump(cache_data, f)
        print(f"✓ Saved fingerprint cache with {len(fp_dict)} entries to {cache_path}")
    except Exception as e:
        print(f"Warning: Failed to save fingerprint cache: {e}")


def compute_fingerprints_for_smiles(
    smiles_list, radius: int = 2, n_bits: int = 2048, cache_dict: Optional[Dict] = None
):
    """
    Compute fingerprints for a list of SMILES strings.
    Uses cache_dict to avoid recomputing existing fingerprints.

    Args:
        smiles_list: List of SMILES strings
        radius: Morgan fingerprint radius
        n_bits: Number of bits in fingerprint
        cache_dict: Optional dictionary of precomputed fingerprints

    Returns:
        Dictionary mapping SMILES to fingerprints (None for invalid SMILES)
    """
    if cache_dict is None:
        cache_dict = {}

    fp_dict = cache_dict.copy()  # Start with cached fingerprints
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    for smiles in smiles_list:
        if smiles in fp_dict:
            continue  # Already computed or cached

        try:
            if pd.isna(smiles) or not smiles:
                fp_dict[smiles] = None
                continue

            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                fp_dict[smiles] = None
            else:
                fp_dict[smiles] = mfpgen.GetFingerprint(mol)
        except Exception:
            fp_dict[smiles] = None

    return fp_dict


def find_top_k_nearest_neighbors(
    test_monomer1_smiles: str,
    test_monomer2_smiles: str,
    test_solvent_smiles: str,
    df_train: pd.DataFrame,
    k: int = 10,
    feature_cols: Optional[List[str]] = None,
    fp_dict: Optional[Dict] = None,
) -> List[Dict]:
    """
    Find the top k nearest neighbors in the training database using baseline lookup approach.

    This function uses Tanimoto similarity on SMILES fingerprints to find similar reactions.

    Args:
        test_monomer1_smiles: SMILES string of first monomer
        test_monomer2_smiles: SMILES string of second monomer
        test_solvent_smiles: SMILES string of solvent
        df_train: Training DataFrame (must contain monomer1_smiles, monomer2_smiles, solvent_smiles)
        k: Number of nearest neighbors to return (default: 10)
        feature_cols: Optional list of feature columns to use for tie-breaking
        fp_dict: Optional precomputed fingerprint dictionary {smiles: fp}

    Returns:
        List of dictionaries, each containing:
            - rank: Ranking (1-based)
            - similarity: Combined similarity score (0-1)
            - predicted_class: Predicted class (r_product_class)
            - monomer1_name: First monomer name (falls back to SMILES if name not available)
            - monomer2_name: Second monomer name (falls back to SMILES if name not available)
            - solvent_name: Solvent name (falls back to SMILES if name not available)
            - temperature: Temperature in Celsius
            - r1: Reactivity ratio r1 (constant_1), if available
            - r2: Reactivity ratio r2 (constant_2), if available
            - r1r2: Product r1*r2, if available
            - method: Polymerization method
            - polytype: Polymerization type
            - source: DOI or original source
            - reaction_id: Reaction ID
    """
    # Check required columns
    required_cols = ["monomer1_smiles", "monomer2_smiles", "solvent_smiles"]
    for col in required_cols:
        if col not in df_train.columns:
            raise ValueError(f"Required column '{col}' not found in training DataFrame")

    # Compute fingerprints if not provided
    if fp_dict is None:
        # Try to load from cache
        cached_fps = load_fingerprint_cache()

        unique_monomer1 = set(df_train["monomer1_smiles"].dropna().unique())
        unique_monomer2 = set(df_train["monomer2_smiles"].dropna().unique())
        unique_solvents = set(df_train["solvent_smiles"].dropna().unique())
        all_unique_smiles = list(unique_monomer1 | unique_monomer2 | unique_solvents)
        all_unique_smiles.extend([test_monomer1_smiles, test_monomer2_smiles, test_solvent_smiles])

        # Compute fingerprints, using cache if available
        fp_dict = compute_fingerprints_for_smiles(all_unique_smiles, cache_dict=cached_fps)

        # Save updated cache (only if we computed new fingerprints)
        if cached_fps is None or len(fp_dict) > len(cached_fps):
            save_fingerprint_cache(fp_dict)

    # Get fingerprints for test point
    test_mon1_fp = fp_dict.get(test_monomer1_smiles)
    test_mon2_fp = fp_dict.get(test_monomer2_smiles)
    test_solv_fp = fp_dict.get(test_solvent_smiles)

    # Check if test fingerprints are available
    if test_mon1_fp is None or test_mon2_fp is None or test_solv_fp is None:
        missing = []
        if test_mon1_fp is None:
            missing.append(f"monomer1 ({test_monomer1_smiles[:50]})")
        if test_mon2_fp is None:
            missing.append(f"monomer2 ({test_monomer2_smiles[:50]})")
        if test_solv_fp is None:
            missing.append(f"solvent ({test_solvent_smiles[:50]})")
        print(f"⚠ Warning: Missing fingerprints for: {', '.join(missing)}")
        print(f"  Available fingerprints in cache: {len(fp_dict)}")
        print(
            f"  Test SMILES in cache: m1={test_monomer1_smiles in fp_dict}, m2={test_monomer2_smiles in fp_dict}, s={test_solvent_smiles in fp_dict}"
        )

        # Try to compute missing fingerprints
        missing_smiles = []
        if test_mon1_fp is None:
            missing_smiles.append(test_monomer1_smiles)
        if test_mon2_fp is None:
            missing_smiles.append(test_monomer2_smiles)
        if test_solv_fp is None:
            missing_smiles.append(test_solvent_smiles)

        if missing_smiles:
            print(f"  Computing fingerprints for {len(missing_smiles)} missing SMILES...")
            new_fps = compute_fingerprints_for_smiles(missing_smiles, cache_dict=fp_dict)
            fp_dict.update(new_fps)
            # Update test fingerprints
            test_mon1_fp = fp_dict.get(test_monomer1_smiles)
            test_mon2_fp = fp_dict.get(test_monomer2_smiles)
            test_solv_fp = fp_dict.get(test_solvent_smiles)

            # Check again
            if test_mon1_fp is None or test_mon2_fp is None or test_solv_fp is None:
                print(f"  ✗ Still missing fingerprints after computation")
                return []
            else:
                print(f"  ✓ Successfully computed missing fingerprints")

    # Precompute fingerprint lists for all training points
    train_mon1_fps = [fp_dict.get(sm) for sm in df_train["monomer1_smiles"]]
    train_mon2_fps = [fp_dict.get(sm) for sm in df_train["monomer2_smiles"]]
    train_solv_fps = [fp_dict.get(sm) for sm in df_train["solvent_smiles"]]

    # Filter out None fingerprints for training data
    valid_indices = []
    for i, (m1, m2, s) in enumerate(zip(train_mon1_fps, train_mon2_fps, train_solv_fps)):
        if m1 is not None and m2 is not None and s is not None:
            valid_indices.append(i)

    if not valid_indices:
        print("Warning: No valid fingerprints found in training data")
        return []

    # Calculate similarities only for valid indices
    # Monomer direct: (test_mon1 vs train_mon1, test_mon2 vs train_mon2)
    valid_mon1_fps = [train_mon1_fps[i] for i in valid_indices]
    valid_mon2_fps = [train_mon2_fps[i] for i in valid_indices]
    valid_solv_fps = [train_solv_fps[i] for i in valid_indices]

    mon1_direct = np.array(DataStructs.BulkTanimotoSimilarity(test_mon1_fp, valid_mon1_fps))
    mon2_direct = np.array(DataStructs.BulkTanimotoSimilarity(test_mon2_fp, valid_mon2_fps))
    mon_sim_direct = (mon1_direct + mon2_direct) / 2.0

    # Monomer flipped: (test_mon1 vs train_mon2, test_mon2 vs train_mon1)
    mon1_flipped = np.array(DataStructs.BulkTanimotoSimilarity(test_mon1_fp, valid_mon2_fps))
    mon2_flipped = np.array(DataStructs.BulkTanimotoSimilarity(test_mon2_fp, valid_mon1_fps))
    mon_sim_flipped = (mon1_flipped + mon2_flipped) / 2.0

    # Take best monomer similarity per training point
    mon_similarity = np.maximum(mon_sim_direct, mon_sim_flipped)

    # Solvent similarity
    solv_similarity = np.array(DataStructs.BulkTanimotoSimilarity(test_solv_fp, valid_solv_fps))

    # Combined similarity (average of monomer and solvent)
    combined_similarity = np.array((mon_similarity + solv_similarity) / 2.0)

    # Handle NaN values
    combined_similarity = np.nan_to_num(combined_similarity, nan=0.0)

    # Get top k indices (from valid_indices)
    top_k_valid_indices = np.argsort(combined_similarity)[::-1][:k]
    top_k_indices = [valid_indices[i] for i in top_k_valid_indices]

    # Class name mapping
    class_names = {0: "alternating", 1: "random to block like", 2: "homopolymer"}

    # Build result list
    results = []
    for rank, valid_idx in enumerate(top_k_valid_indices, start=1):
        idx = valid_indices[valid_idx]
        train_row = df_train.iloc[idx]

        # Get predicted class
        predicted_class = int(train_row.get("r_product_class", -1))
        predicted_class_name = class_names.get(predicted_class, "unknown")

        # Get names, fallback to SMILES if names not available
        monomer1_name = train_row.get("monomer1_name", "")
        if pd.isna(monomer1_name) or not monomer1_name:
            monomer1_name = train_row.get("monomer1_smiles", "")

        monomer2_name = train_row.get("monomer2_name", "")
        if pd.isna(monomer2_name) or not monomer2_name:
            monomer2_name = train_row.get("monomer2_smiles", "")

        solvent_name = train_row.get("solvent", "")
        if pd.isna(solvent_name) or not solvent_name:
            solvent_name = train_row.get("solvent_smiles", "")

        # Get SMILES
        monomer1_smiles = str(train_row.get("monomer1_smiles", ""))
        monomer2_smiles = str(train_row.get("monomer2_smiles", ""))
        solvent_smiles = str(train_row.get("solvent_smiles", ""))

        # Reactivity ratios (if present in the split CSV)
        r1_val = train_row.get("constant_1", np.nan)
        r2_val = train_row.get("constant_2", np.nan)
        r1r2_val = train_row.get("r1r2", np.nan)
        r1_out = float(r1_val) if pd.notna(r1_val) else None
        r2_out = float(r2_val) if pd.notna(r2_val) else None
        r1r2_out = float(r1r2_val) if pd.notna(r1r2_val) else None

        _nn_doi = doi_from_source_filename(train_row.get("source_filename"))

        result = {
            "rank": rank,
            "similarity": float(combined_similarity[valid_idx]),
            "predicted_class": predicted_class,
            "predicted_class_name": predicted_class_name,
            "monomer1_name": str(monomer1_name),
            "monomer2_name": str(monomer2_name),
            "monomer1_smiles": monomer1_smiles,
            "monomer2_smiles": monomer2_smiles,
            "solvent_name": str(solvent_name),
            "solvent_smiles": solvent_smiles,
            "temperature": (
                float(train_row.get("temperature", np.nan))
                if pd.notna(train_row.get("temperature"))
                else None
            ),
            "r1": r1_out,
            "r2": r2_out,
            "r1r2": r1r2_out,
            "method": (
                str(train_row.get("method", "")) if pd.notna(train_row.get("method")) else None
            ),
            "polytype": (
                str(train_row.get("polymerization_type", ""))
                if pd.notna(train_row.get("polymerization_type"))
                else None
            ),
            "source": (
                str(train_row.get("original_source", train_row.get("doi", "")))
                if pd.notna(train_row.get("original_source", train_row.get("doi", "")))
                else None
            ),
            # DOI recovered from the processed-paper filename — see
            # doi_from_source_filename for why this beats `source`.
            # None for synthetic/augmented rows (no source_filename) and for
            # the rare real paper that has no DOI.
            "doi": _nn_doi,
            "doi_url": doi_url(_nn_doi),
            "reaction_id": (
                str(train_row.get("reaction_id", ""))
                if pd.notna(train_row.get("reaction_id"))
                else None
            ),
        }

        results.append(result)

    return results


def find_top_k_nearest_neighbors_from_features(
    features: Dict[str, float],
    df_train: pd.DataFrame,
    k: int = 10,
    feature_cols: Optional[List[str]] = None,
    fp_dict: Optional[Dict] = None,
) -> List[Dict]:
    """
    Find the top k nearest neighbors from feature dictionary.

    This is a convenience wrapper that extracts SMILES from features if available,
    or uses feature-based similarity if SMILES are not available.

    Args:
        features: Dictionary of feature values (should contain monomer and solvent info)
        df_train: Training DataFrame
        k: Number of nearest neighbors to return
        feature_cols: Optional list of feature columns to use for similarity
        fp_dict: Optional precomputed fingerprint dictionary

    Returns:
        List of dictionaries with nearest neighbor information
    """
    # Try to extract SMILES from features if available
    # Note: The API might not have SMILES directly in features, so we might need
    # to use a different approach. For now, we'll require SMILES to be passed separately.
    # This function can be extended later if needed.

    # For now, return empty list - this function should be called with explicit SMILES
    return []
