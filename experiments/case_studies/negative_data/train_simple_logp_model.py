#!/usr/bin/env python3
"""
Train a simple binary classification model using ONLY logP features.

Features:
- monomer1_logP
- monomer2_logP
- solvent_logP

Classification: Binary (0/1 vs 2)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "copol_prediction"))

from copolpredictor import data_processing, evaluation, model_training

# Import load_data_split and name_to_smiles
sys.path.insert(0, str(PROJECT_ROOT / "copol_prediction"))
from utils import load_data_split

from copolextractor.utils import name_to_smiles

# Cache for logP values
LOGP_CACHE = {}
CACHE_FILE = Path(__file__).parent / "logp_cache.json"


def load_logp_cache():
    """Load logP cache from file."""
    global LOGP_CACHE
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r") as f:
                LOGP_CACHE = json.load(f)
            print(f"✅ Loaded {len(LOGP_CACHE)} logP values from cache")
        except Exception as e:
            print(f"⚠️ Warning: Could not load cache: {e}")
            LOGP_CACHE = {}
    else:
        LOGP_CACHE = {}


def save_logp_cache():
    """Save logP cache to file."""
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump(LOGP_CACHE, f, indent=2)
        print(f"✅ Saved {len(LOGP_CACHE)} logP values to cache")
    except Exception as e:
        print(f"⚠️ Warning: Could not save cache: {e}")


def calculate_logp(smiles):
    """
    Calculate logP for a SMILES string.
    Uses cache to avoid recalculating.

    Args:
        smiles: SMILES string

    Returns:
        float: logP value or np.nan if calculation fails
    """
    if pd.isna(smiles) or not isinstance(smiles, str) or smiles.strip() == "":
        return np.nan

    # Check cache
    if smiles in LOGP_CACHE:
        return LOGP_CACHE[smiles]

    # Calculate logP
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logp_value = np.nan
        else:
            logp_value = float(Descriptors.MolLogP(mol))
    except Exception as e:
        logp_value = np.nan

    # Store in cache
    LOGP_CACHE[smiles] = logp_value

    return logp_value


def add_logp_features(df):
    """Add logP features for monomers and solvent."""
    print("\n➕ Adding logP features...")

    # Calculate logP for monomer1
    print("  Calculating logP for monomer1...")
    df["monomer1_logP"] = df["monomer1_smiles"].apply(calculate_logp)

    # Calculate logP for monomer2
    print("  Calculating logP for monomer2...")
    df["monomer2_logP"] = df["monomer2_smiles"].apply(calculate_logp)

    # Calculate logP for solvent
    print("  Calculating logP for solvent...")
    df["solvent_logP"] = df["solvent_smiles"].apply(calculate_logp)

    # Report statistics
    n_valid_m1 = df["monomer1_logP"].notna().sum()
    n_valid_m2 = df["monomer2_logP"].notna().sum()
    n_valid_solv = df["solvent_logP"].notna().sum()
    print(f"  ✅ Valid logP values:")
    print(f"     monomer1: {n_valid_m1}/{len(df)}")
    print(f"     monomer2: {n_valid_m2}/{len(df)}")
    print(f"     solvent: {n_valid_solv}/{len(df)}")

    return df


def convert_to_binary_classification(df):
    """
    Convert 3-class classification to binary: 0/1 → 0, 2 → 1

    Args:
        df: DataFrame with 'r_product_class' column

    Returns:
        DataFrame with 'r_product_class_binary' column
    """
    df = df.copy()

    # Convert: 0/1 → 0, 2 → 1
    df["r_product_class_binary"] = (df["r_product_class"] == 2).astype(int)

    # Report distribution
    print("\nBinary class distribution:")
    binary_counts = df["r_product_class_binary"].value_counts().sort_index()
    for cls, count in binary_counts.items():
        label = "Class 0/1 (Alternating/Random)" if cls == 0 else "Class 2 (Homopolymer)"
        print(f"  {label}: {count} samples ({100*count/len(df):.1f}%)")

    return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train simple binary model with ONLY logP features"
    )
    parser.add_argument(
        "--negative-data", type=str, default=None, help="Path to processed negative data CSV"
    )
    parser.add_argument(
        "--negative-test-data", type=str, default=None, help="Path to new negative data test set"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to save model bundle"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--hyperparam-iter", type=int, default=25, help="Number of hyperparameter search iterations"
    )

    return parser.parse_args()


def load_training_data():
    """Load original training data."""
    print(f"\n📥 Loading original training data...")

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent
    split_dir = project_root / "copol_prediction" / "artifacts" / "data_splits"

    try:
        df_train, df_test = load_data_split.load_train_test_split(split_dir=str(split_dir))
        print(f"✅ Loaded {len(df_train)} training samples")
        print(f"   Unique reactions: {df_train['reaction_id'].nunique()}")
        return df_train, df_test
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error loading training data: {e}")
        sys.exit(1)


def prepare_negative_test_data(raw_input_path):
    """
    Prepare negative test data from raw CSV file.
    Converts raw format to standard format with SMILES and logP features.

    Args:
        raw_input_path: Path to raw CSV file (format: Class;monomer_1;monomer_2;Solvent;Temperature;Polymerization_type)

    Returns:
        DataFrame with standard format including SMILES and logP features
    """
    print(f"\n📥 Preparing negative test data from: {raw_input_path}")

    if not Path(raw_input_path).exists():
        print(f"❌ Error: Input file not found: {raw_input_path}")
        return None

    # Read CSV with robust parsing (handle trailing semicolons)
    import io

    with open(raw_input_path, "r") as f:
        all_lines = []
        for line in f:
            cleaned_line = line.rstrip().rstrip(";")  # Remove trailing semicolons
            if cleaned_line:  # Skip empty lines
                all_lines.append(cleaned_line)

    # Parse CSV from cleaned string
    cleaned_csv = "\n".join(all_lines)

    # Check if first line looks like header
    first_line = all_lines[0] if all_lines else ""
    has_header = "Class" in first_line or "monomer" in first_line.lower()

    if has_header:
        df_raw = pd.read_csv(io.StringIO(cleaned_csv), sep=";", engine="python")
    else:
        # No header, use expected column names
        expected_cols = [
            "Class",
            "monomer_1",
            "monomer_2",
            "Solvent",
            "Temperature",
            "Polymerization_type",
        ]
        df_raw = pd.read_csv(
            io.StringIO(cleaned_csv), sep=";", engine="python", header=None, names=expected_cols
        )

    # Ensure we have the expected columns
    expected_cols = [
        "Class",
        "monomer_1",
        "monomer_2",
        "Solvent",
        "Temperature",
        "Polymerization_type",
    ]

    # Map columns if needed (in case column names don't match exactly)
    if len(df_raw.columns) == len(expected_cols):
        # Check if columns match expected names
        if not all(col in df_raw.columns for col in expected_cols):
            df_raw.columns = expected_cols[: len(df_raw.columns)]

    print(f"✅ Loaded {len(df_raw)} raw data points")
    print(f"   Columns: {list(df_raw.columns)}")

    # Convert to standard format
    df_standard = []
    for idx, row in df_raw.iterrows():
        m1_name = str(row.get("monomer_1", "")).strip() if pd.notna(row.get("monomer_1")) else ""
        m2_name = str(row.get("monomer_2", "")).strip() if pd.notna(row.get("monomer_2")) else ""
        solvent_name = str(row.get("Solvent", "")).strip() if pd.notna(row.get("Solvent")) else ""

        # Handle temperature
        temp_val = row.get("Temperature", np.nan)
        if pd.notna(temp_val):
            try:
                temp_val = float(str(temp_val).strip())
            except (ValueError, TypeError):
                temp_val = np.nan

        # Handle class
        cls_val = row.get("Class", np.nan)
        try:
            cls_int = int(float(str(cls_val).strip())) if pd.notna(cls_val) else np.nan
        except (ValueError, TypeError):
            print(f"⚠️ Skipping row {idx}: invalid class value '{cls_val}'")
            continue

        # Skip if essential fields are missing
        if not m1_name or not m2_name:
            print(f"⚠️ Skipping row {idx}: missing monomer names")
            continue

        # Convert names to SMILES
        m1_smiles = name_to_smiles(m1_name)
        m2_smiles = name_to_smiles(m2_name)
        solvent_smiles = name_to_smiles(solvent_name) if solvent_name else None

        if not m1_smiles or not m2_smiles:
            print(f"⚠️ Skipping row {idx}: could not convert to SMILES")
            continue

        new_row = {
            "r_product_class": cls_int,
            "monomer1_name": m1_name,
            "monomer2_name": m2_name,
            "monomer1_smiles": m1_smiles,
            "monomer2_smiles": m2_smiles,
            "solvent": solvent_name,
            "solvent_smiles": solvent_smiles,
            "temperature": temp_val,
            "polymerization_type": (
                str(row.get("Polymerization_type", "")).strip()
                if pd.notna(row.get("Polymerization_type"))
                else ""
            ),
            "reaction_id": idx,
        }
        df_standard.append(new_row)

    df_standard = pd.DataFrame(df_standard)
    print(f"✅ Converted {len(df_standard)} rows to standard format")

    if len(df_standard) < len(df_raw):
        skipped = len(df_raw) - len(df_standard)
        print(f"⚠️ Warning: {skipped} rows were skipped during conversion")
        print(f"   (likely due to missing SMILES conversion or invalid data)")

    return df_standard


def load_negative_data(neg_data_path):
    """Load and prepare negative data."""
    print(f"\n📥 Loading negative data from: {neg_data_path}")

    if not Path(neg_data_path).exists():
        print(f"❌ Error: Negative data file not found: {neg_data_path}")
        sys.exit(1)

    df_neg = pd.read_csv(neg_data_path)

    # Ensure r_product_class exists
    if "r_product_class" not in df_neg.columns and "Class" in df_neg.columns:
        df_neg = df_neg.rename(columns={"Class": "r_product_class"})

    # Ensure reaction_id exists
    if "reaction_id" not in df_neg.columns:
        df_neg["reaction_id"] = df_neg.index

    print(f"✅ Loaded {len(df_neg)} negative data samples")

    return df_neg


def remove_duplicates_by_logp_features(
    df, logp_features=["monomer1_logP", "monomer2_logP", "solvent_logP"]
):
    """
    Remove duplicate rows based on logP features (since temperature is not a feature).

    Args:
        df: DataFrame with logP features
        logp_features: List of feature columns to check for duplicates

    Returns:
        DataFrame with duplicates removed (keeps first occurrence)
    """
    print("\n" + "=" * 60)
    print("REMOVING DUPLICATES (BY LOGP FEATURES)")
    print("=" * 60)

    initial_count = len(df)

    # Check which logP features exist
    available_features = [f for f in logp_features if f in df.columns]
    missing_features = [f for f in logp_features if f not in df.columns]

    if missing_features:
        print(f"⚠️ Warning: Missing logP features: {missing_features}")
        print(f"   Will only check duplicates based on: {available_features}")

    if not available_features:
        print("⚠️ Warning: No logP features found, skipping duplicate removal")
        return df

    # Remove duplicates based on logP features
    df_dedup = df.drop_duplicates(subset=available_features, keep="first")

    removed_count = initial_count - len(df_dedup)
    print(f"\nRemoved {removed_count} duplicate rows ({100*removed_count/initial_count:.1f}%)")
    print(f"  Before: {initial_count} samples")
    print(f"  After:  {len(df_dedup)} samples")

    return df_dedup


def reorder_monomers_by_logp(df):
    """
    Ensure monomer1 has the lower (or equal) logP compared to monomer2.
    Swaps all columns that come in monomer1_*/monomer2_* pairs for affected rows.
    """
    if "monomer1_logP" not in df.columns or "monomer2_logP" not in df.columns:
        return df

    df = df.copy()

    # Rows, bei denen monomer1_logP > monomer2_logP: hier tauschen wir
    mask = df["monomer1_logP"] > df["monomer2_logP"]
    if not mask.any():
        return df

    print(f"\nReordering monomers by logP for {mask.sum()} rows (monomer1_logP > monomer2_logP).")

    # Alle Paare monomer1_*/monomer2_* finden
    mono1_cols = [c for c in df.columns if c.startswith("monomer1_")]
    for c1 in mono1_cols:
        suffix = c1[len("monomer1_") :]
        c2 = "monomer2_" + suffix
        if c2 not in df.columns:
            continue

        tmp = df.loc[mask, c1].copy()
        df.loc[mask, c1] = df.loc[mask, c2]
        df.loc[mask, c2] = tmp

    return df


def sample_normal_data_to_match_negative(df_normal, df_negative, random_state=42):
    """
    Randomly sample normal training data to match the size of negative data.

    Args:
        df_normal: DataFrame with normal training data
        df_negative: DataFrame with negative data (target size)
        random_state: Random seed for reproducibility

    Returns:
        Sampled DataFrame with same size as df_negative
    """
    print("\n" + "=" * 60)
    print("SAMPLING NORMAL DATA TO MATCH NEGATIVE DATA SIZE")
    print("=" * 60)

    target_size = len(df_negative)
    current_size = len(df_normal)

    print(f"\nNormal data: {current_size} samples")
    print(f"Negative data: {target_size} samples")
    print(f"Target size: {target_size} samples")

    if current_size <= target_size:
        print(f"⚠️ Warning: Normal data ({current_size}) <= target size ({target_size})")
        print(f"   Using all normal data without sampling")
        return df_normal

    # Sample to match negative data size
    df_sampled = df_normal.sample(n=target_size, random_state=random_state).reset_index(drop=True)

    print(f"\n✅ Sampled {target_size} samples from {current_size} normal data")
    print(f"   Reduction: {100*(1-target_size/current_size):.1f}%")

    return df_sampled


def split_negative_data(
    df_negative,
    test_size=0.2,
    random_state=42,
    test_only_solvents=None,
    train_only_solvents=None,
    solvent_id_column=None,
):
    """
    Split negative data into train and test sets (80/20).
    Uses monomer+solvent combination-based splitting to ensure fair split:
    - No identical monomer1+monomer2+solvent combinations in both train and test
    - Splits based on unique combinations, not reaction_id

    Optional solvent-level control:
    - test_only_solvents: iterable of solvent identifiers that should go *only* into the test set
    - train_only_solvents: iterable of solvent identifiers that should go *only* into the train set
    - solvent_id_column: name of the column to use for solvent IDs
      (e.g. 'solvent', 'Solvent' oder 'solvent_smiles'). If None, the
      solvent-level control is disabled and the behavior is identical to before.

    Args:
        df_negative: DataFrame with negative data (must have logP features already)
        test_size: Fraction of data to use for test (default 0.2)
        random_state: Random seed for reproducibility
        test_only_solvents: Optional iterable of solvent IDs for test-only
        train_only_solvents: Optional iterable of solvent IDs for train-only
        solvent_id_column: Optional column name used to match solvent IDs

    Returns:
        Tuple of (df_neg_train, df_neg_test)
    """
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 60)
    print("SPLITTING NEGATIVE DATA (80/20) - BY MONOMER+SOLVENT COMBINATIONS")
    print("=" * 60)

    df_negative = df_negative.copy()

    # ------------------------------------------------------------------
    # 1) Optional: solvent-level forcing (some solvents only in train/test)
    # ------------------------------------------------------------------
    forced_train_mask = pd.Series(False, index=df_negative.index)
    forced_test_mask = pd.Series(False, index=df_negative.index)

    has_forced_solvents = (
        (test_only_solvents is not None or train_only_solvents is not None)
        and solvent_id_column is not None
        and solvent_id_column in df_negative.columns
    )

    if has_forced_solvents:
        # Normalize to sets of strings for matching
        test_only_set = set(str(s) for s in (test_only_solvents or []))
        train_only_set = set(str(s) for s in (train_only_solvents or []))

        solvent_series = df_negative[solvent_id_column].astype(str)

        forced_test_mask = solvent_series.isin(test_only_set)
        forced_train_mask = solvent_series.isin(train_only_set) & ~forced_test_mask

        # Detect conflicting specifications
        conflict_mask = forced_test_mask & forced_train_mask
        if conflict_mask.any():
            conflict_vals = solvent_series[conflict_mask].unique().tolist()
            print(
                f"  ⚠️ Warning: Some solvents are in BOTH test_only_solvents and "
                f"train_only_solvents. They will be treated as test-only. "
                f"Solvents: {conflict_vals}"
            )
            # Give priority to test-only (already ensured by & ~forced_test_mask above)

        print(
            f"\nSolvent-level forcing active (column='{solvent_id_column}'):\n"
            f"  Forced to TRAIN only: {forced_train_mask.sum()} rows\n"
            f"  Forced to TEST only:  {forced_test_mask.sum()} rows"
        )
    elif test_only_solvents is not None or train_only_solvents is not None:
        print(
            "  ⚠️ Solvent-level forcing requested but 'solvent_id_column' is missing "
            "or not found in dataframe. Ignoring test_only_solvents/train_only_solvents."
        )

    # Rows that will still be split according to combinations
    remaining_mask = ~(forced_train_mask | forced_test_mask)
    df_remaining = df_negative[remaining_mask].copy()

    # ------------------------------------------------------------------
    # 2) Original combination-based split on the remaining pool
    # ------------------------------------------------------------------
    # Check if logP features exist (needed for fair splitting)
    logp_features = ["monomer1_logP", "monomer2_logP", "solvent_logP"]
    available_logp = [f for f in logp_features if f in df_remaining.columns]

    if len(df_remaining) == 0:
        print("\nAll rows are forced by solvent; skipping combination-based split.")
        df_split_train = df_remaining.copy()
        df_split_test = df_remaining.copy()
        combo_col = None
    else:
        if len(available_logp) == len(logp_features):
            # Use logP features to create unique combinations
            # Create a combination key based on logP values (rounded to avoid float precision issues)
            df_remaining["_combo_key"] = (
                df_remaining["monomer1_logP"].round(4).astype(str)
                + "_"
                + df_remaining["monomer2_logP"].round(4).astype(str)
                + "_"
                + df_remaining["solvent_logP"].round(4).astype(str)
            )
            combo_col = "_combo_key"
            print("  Using logP-based combinations for splitting")
        else:
            # Fallback: Use monomer names + solvent
            print("  ⚠️ logP features not available, using monomer names + solvent")
            # Normalize names (handle case differences)
            m1 = (
                df_remaining.get("monomer1_name", df_remaining.get("monomer_1", ""))
                .astype(str)
                .str.lower()
                .str.strip()
            )
            m2 = (
                df_remaining.get("monomer2_name", df_remaining.get("monomer_2", ""))
                .astype(str)
                .str.lower()
                .str.strip()
            )
            solv = (
                df_remaining.get("solvent", df_remaining.get("Solvent", ""))
                .astype(str)
                .str.lower()
                .str.strip()
            )
            df_remaining["_combo_key"] = m1 + "_" + m2 + "_" + solv
            combo_col = "_combo_key"

        # Get unique combinations
        unique_combos = df_remaining[combo_col].unique()

        print(f"\nFound {len(unique_combos)} unique monomer+solvent combinations (remaining pool)")
        print(f"Total samples in remaining pool: {len(df_remaining)}")

        # Split unique combinations (not individual rows)
        combo_train, combo_test = train_test_split(
            unique_combos, test_size=test_size, random_state=random_state
        )

        # Split data based on combinations
        df_split_train = df_remaining[df_remaining[combo_col].isin(combo_train)].copy()
        df_split_test = df_remaining[df_remaining[combo_col].isin(combo_test)].copy()

        print(f"\nNegative data split (remaining pool):")
        print(f"  Train: {len(df_split_train)} samples ({len(combo_train)} unique combinations)")
        print(f"  Test:  {len(df_split_test)} samples ({len(combo_test)} unique combinations)")
        if len(df_remaining) > 0:
            print(
                f"  Split ratio (remaining pool): {len(df_split_train)}/{len(df_remaining)} "
                f"= {len(df_split_train)/len(df_remaining):.1%} train"
            )

        # Verify no overlap BEFORE removing temporary column
        train_combos_set = set(combo_train)
        test_combos_set = set(combo_test)
        overlap = train_combos_set & test_combos_set

        if overlap:
            print(f"  ⚠️ Warning: {len(overlap)} combinations appear in both train and test!")
            print(f"     Overlapping combinations: {list(overlap)[:5]}...")
        else:
            print(f"  ✅ Verified: No overlap between train and test combinations")

    # ------------------------------------------------------------------
    # 3) Combine forced rows with split rows and clean up
    # ------------------------------------------------------------------
    df_forced_train = df_negative[forced_train_mask].copy()
    df_forced_test = df_negative[forced_test_mask].copy()

    df_neg_train = pd.concat([df_forced_train, df_split_train], ignore_index=True)
    df_neg_test = pd.concat([df_forced_test, df_split_test], ignore_index=True)

    if combo_col is not None:
        df_neg_train = df_neg_train.drop(columns=[combo_col], errors="ignore")
        df_neg_test = df_neg_test.drop(columns=[combo_col], errors="ignore")

    print("\nFinal negative data split (including forced solvents):")
    print(f"  Train: {len(df_neg_train)} samples")
    print(f"  Test:  {len(df_neg_test)} samples")

    return df_neg_train, df_neg_test


def combine_training_data(df_original, df_negative):
    """Combine original training data with negative data."""
    print("\n" + "=" * 60)
    print("COMBINING TRAINING DATA")
    print("=" * 60)

    print(f"\nOriginal training data: {len(df_original)} samples")
    print(f"Negative training data: {len(df_negative)} samples")

    # Combine datasets
    df_combined = pd.concat([df_original, df_negative], ignore_index=True)

    print(f"\n✅ Combined dataset: {len(df_combined)} samples")
    print(f"   Unique reactions: {df_combined['reaction_id'].nunique()}")

    return df_combined


def train_model(df_train, config):
    """
    Train binary classification model with ONLY logP features.

    Args:
        df_train: Training dataframe (original + negative data)
        config: Configuration dictionary

    Returns:
        Dictionary with trained model and training info
    """
    print("\n" + "=" * 60)
    print("MODEL TRAINING (BINARY: 0/1 vs 2, ONLY LOGP FEATURES)")
    print("=" * 60)

    # Features: ONLY logP
    features = ["monomer1_logP", "monomer2_logP", "solvent_logP"]

    # Check if all features exist
    missing_features = [f for f in features if f not in df_train.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    # Prepare training data
    X_train = df_train[features].copy()
    y_train = df_train["r_product_class_binary"].astype(int).values
    groups = df_train["reaction_id"].astype(str).values

    # Fill NaN values
    nan_counts = X_train.isna().sum()
    if nan_counts.sum() > 0:
        print(f"\n⚠️ Warning: Found NaN values in features:")
        for feat, count in nan_counts[nan_counts > 0].items():
            print(f"   {feat}: {count} NaN values ({100*count/len(X_train):.1f}%)")
        X_train = X_train.fillna(X_train.median())  # Fill with median
        print("   Filled NaN values with median")

    print(f"\nTraining set:")
    print(f"  Samples: {len(X_train)}")
    print(f"  Features: {features}")
    print(f"  Unique reactions: {len(np.unique(groups))}")

    # Calculate class weights
    class_weights = model_training.calculate_class_weights(y_train)

    print("\nClass weights:")
    for cls, weight in sorted(class_weights.items()):
        label = "Class 0/1 (Alternating/Random)" if cls == 0 else "Class 2 (Homopolymer)"
        count = (y_train == cls).sum()
        print(f"  {label}: {weight:.4f} (n={count})")

    # Define hyperparameter search space
    param_grid = {
        "n_estimators": [300, 400, 500],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.05, 0.06, 0.07],
        "subsample": [0.85, 0.9, 0.95],
        "colsample_bytree": [0.85, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 0.3],
        "reg_lambda": [1.0, 1.5, 2.0],
        "min_child_weight": [2, 3, 5],
        "gamma": [0.3, 0.5, 0.7],
    }

    # Train with CV and hyperparameter search
    print("\nStarting hyperparameter search...")
    train_results = model_training.train_xgboost_with_cv(
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        param_grid=param_grid,
        n_iter=config["hyperparam_iter"],
        cv=5,
        random_state=config["random_state"],
        class_weights=class_weights,
        n_jobs=-1,
    )

    print("\nBest hyperparameters:")
    for param, value in train_results["best_params"].items():
        print(f"  {param}: {value}")
    print(f"\nBest CV score (F1 weighted): {train_results['best_score']:.4f}")

    # Train final model on full training set
    print("\nTraining final model on full dataset...")
    final_model = model_training.train_final_model(
        X_train=X_train,
        y_train=y_train,
        params=train_results["best_params"],
        class_weights=class_weights,
        random_state=config["random_state"],
    )

    return {
        "model": final_model,
        "best_params": train_results["best_params"],
        "cv_score": train_results["best_score"],
        "class_weights": class_weights,
        "features": features,
    }


def print_detailed_predictions(model, df_test, features, dataset_name="Test Set"):
    """
    Print detailed predictions for each data point: input features and output predictions.

    Args:
        model: Trained model
        df_test: Test dataframe
        features: List of feature names
        dataset_name: Name of the dataset for printing
    """
    print(f"\n" + "=" * 80)
    print(f"DETAILED PREDICTIONS FOR {dataset_name.upper()}")
    print("=" * 80)

    # Get model features
    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            model_features = booster.feature_names
            if not model_features:
                model_features = features
        except:
            model_features = features
    else:
        model_features = features

    # Check for missing features
    missing_features = set(model_features) - set(df_test.columns)
    if missing_features:
        for feat in missing_features:
            df_test[feat] = np.nan

    X_test = df_test[model_features].copy()

    # Fill NaN values
    nan_counts = X_test.isna().sum()
    if nan_counts.sum() > 0:
        X_test = X_test.fillna(0)

    # Get true labels
    if "r_product_class_binary" in df_test.columns:
        y_test = df_test["r_product_class_binary"].astype(int).values
    else:
        y_test = (df_test["r_product_class"] == 2).astype(int).values

    # Make predictions
    y_pred = model.predict(X_test)

    # Get prediction probabilities
    try:
        y_proba = model.predict_proba(X_test)
    except:
        y_proba = None

    # Print for each data point
    print(f"\nTotal samples: {len(df_test)}")
    print(
        f"\n{'ID':<6} {'True':<6} {'Pred':<6} {'Prob[0]':<10} {'Prob[1]':<10} {'monomer1_logP':<15} {'monomer2_logP':<15} {'solvent_logP':<15} {'monomer1':<25} {'monomer2':<25} {'solvent':<15}"
    )
    print("-" * 180)

    for idx in range(len(df_test)):
        row = df_test.iloc[idx]

        # Get monomer and solvent names
        m1_name = str(row.get("monomer1_name", "N/A"))[:24]
        m2_name = str(row.get("monomer2_name", "N/A"))[:24]
        solvent_name = str(row.get("solvent", "N/A"))[:14]

        # Get logP values
        m1_logp = X_test.iloc[idx].get("monomer1_logP", np.nan)
        m2_logp = X_test.iloc[idx].get("monomer2_logP", np.nan)
        solv_logp = X_test.iloc[idx].get("solvent_logP", np.nan)

        # Get probabilities
        if y_proba is not None:
            prob_0 = f"{y_proba[idx, 0]:.4f}" if len(y_proba[idx]) > 0 else "N/A"
            prob_1 = f"{y_proba[idx, 1]:.4f}" if len(y_proba[idx]) > 1 else "N/A"
        else:
            prob_0 = "N/A"
            prob_1 = "N/A"

        # Format logP values
        m1_logp_str = f"{m1_logp:.4f}" if pd.notna(m1_logp) else "NaN"
        m2_logp_str = f"{m2_logp:.4f}" if pd.notna(m2_logp) else "NaN"
        solv_logp_str = f"{solv_logp:.4f}" if pd.notna(solv_logp) else "NaN"

        # True label and prediction
        true_label = y_test[idx]
        pred_label = y_pred[idx]

        # Mark correct/incorrect
        correct = "✓" if true_label == pred_label else "✗"

        print(
            f"{idx:<6} {true_label:<6} {pred_label:<6} {prob_0:<10} {prob_1:<10} {m1_logp_str:<15} {m2_logp_str:<15} {solv_logp_str:<15} {m1_name:<25} {m2_name:<25} {solvent_name:<15} {correct}"
        )

    print("-" * 180)
    print(f"\nSummary:")
    print(f"  Correct predictions: {(y_test == y_pred).sum()}/{len(y_test)}")
    print(f"  Accuracy: {(y_test == y_pred).mean():.4f}")


def evaluate_on_test(model, df_test, features, is_binary=True, dataset_name="Test Set"):
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        df_test: Test dataframe
        features: List of feature names
        is_binary: Whether this is binary classification
        dataset_name: Name of the dataset for printing

    Returns:
        Evaluation results dictionary
    """
    print("\n" + "=" * 60)
    print(f"{dataset_name.upper()} EVALUATION")
    print("=" * 60)

    # Get the features the model actually expects
    if hasattr(model, "feature_names_in_"):
        model_features = list(model.feature_names_in_)
    elif hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            model_features = booster.feature_names
            if not model_features:
                model_features = features
        except:
            model_features = features
    else:
        model_features = features

    print(f"Model expects {len(model_features)} features: {model_features}")
    print(f"Test set has {len(df_test.columns)} columns")

    # Check for missing features
    missing_features = set(model_features) - set(df_test.columns)
    if missing_features:
        print(f"\n⚠️ Warning: Missing {len(missing_features)} features in test set:")
        for feat in sorted(missing_features):
            print(f"   - {feat}")

        # Add missing features with NaN
        for feat in missing_features:
            df_test[feat] = np.nan

    # Use model_features to ensure correct order
    X_test = df_test[model_features].copy()

    # Fill NaN values
    nan_counts = X_test.isna().sum()
    if nan_counts.sum() > 0:
        print(f"\n⚠️ Warning: Found NaN values in features:")
        for feat, count in nan_counts[nan_counts > 0].items():
            print(f"   {feat}: {count} NaN values ({100*count/len(X_test):.1f}%)")

        # Fill with median from training (or 0 if not available)
        X_test = X_test.fillna(0)
        print("   Filled NaN values with 0")

    # Get true labels (binary or 3-class)
    if is_binary and "r_product_class_binary" in df_test.columns:
        y_test = df_test["r_product_class_binary"].astype(int).values
    else:
        # Convert 3-class to binary
        y_test = (df_test["r_product_class"] == 2).astype(int).values

    # Evaluate (binary classification)
    results = evaluation.evaluate_model(model, X_test, y_test, labels=[0, 1])
    evaluation.print_evaluation_results(results, title=f"{dataset_name} Performance")

    # Calculate macro accuracy (average of per-class accuracy)
    from sklearn.metrics import accuracy_score, precision_score

    accuracy_per_class = []
    y_pred = results["predictions"]
    for cls in [0, 1]:
        cls_mask = y_test == cls
        if cls_mask.sum() > 0:
            cls_accuracy = accuracy_score(y_test[cls_mask], y_pred[cls_mask])
            accuracy_per_class.append(cls_accuracy)
        else:
            accuracy_per_class.append(0.0)
    accuracy_macro = np.mean(accuracy_per_class)
    results["accuracy_macro"] = accuracy_macro

    # Calculate macro precision
    precision_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    results["precision_macro"] = precision_macro

    # Print confusion matrix separately for better visibility
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    print(f"\n📊 Confusion Matrix ({dataset_name}):")
    print("=" * 60)
    print("                Predicted")
    print("              Class 0  Class 1")
    print(f"Actual Class 0  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"Actual Class 1  {cm[1,0]:6d}  {cm[1,1]:6d}")
    print("=" * 60)
    print(f"  Accuracy (macro): {accuracy_macro:.4f}")

    # Calculate per-class metrics
    tn, fp, fn, tp = cm.ravel()
    precision_class_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    recall_class_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision_class_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_class_1 = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"\nPer-Class Metrics:")
    print(f"  Class 0 (Alternating/Random):")
    print(f"    Precision: {precision_class_0:.4f}")
    print(f"    Recall: {recall_class_0:.4f}")
    print(f"  Class 1 (Homopolymer):")
    print(f"    Precision: {precision_class_1:.4f}")
    print(f"    Recall: {recall_class_1:.4f}")

    return results


def save_model(
    model_info, train_normal_results, train_neg_results, test_results, test_neg_results, config
):
    """Save model bundle and results."""
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    metadata = {
        "best_params": model_info["best_params"],
        "cv_score": float(model_info["cv_score"]),
        "train_normal_accuracy": float(train_normal_results["accuracy"]),
        "train_normal_f1_weighted": float(train_normal_results["f1_weighted"]),
        "train_normal_f1_macro": float(train_normal_results["f1_macro"]),
        "train_neg_accuracy": float(train_neg_results["accuracy"]),
        "train_neg_f1_weighted": float(train_neg_results["f1_weighted"]),
        "train_neg_f1_macro": float(train_neg_results["f1_macro"]),
        "test_accuracy": float(test_results["accuracy"]),
        "test_f1_weighted": float(test_results["f1_weighted"]),
        "test_f1_macro": float(test_results["f1_macro"]),
        "test_neg_accuracy": float(test_neg_results["accuracy"]),
        "test_neg_f1_weighted": float(test_neg_results["f1_weighted"]),
        "test_neg_f1_macro": float(test_neg_results["f1_macro"]),
        "class_weights": {int(k): float(v) for k, v in model_info["class_weights"].items()},
        "training_config": {
            "binary_classification": True,
            "class_mapping": "0/1 → 0, 2 → 1",
            "features": "ONLY logP (monomer1_logP, monomer2_logP, solvent_logP)",
            "trained_with_negative_data": True,
            "negative_data_source": config["negative_data_path"],
            "random_state": config["random_state"],
            "hyperparam_iter": config["hyperparam_iter"],
        },
    }

    # Save model bundle
    bundle_path = model_training.save_model_bundle(
        model=model_info["model"],
        feature_list=model_info["features"],
        class_labels=[0, 1],  # Binary classification
        out_dir=config["output_dir"],
        metadata=metadata,
    )

    print(f"\n✓ Model bundle saved to: {bundle_path}")


def plot_case_study_performance(
    results_train_normal, results_train_neg, results_test_neg, output_dir
):
    """Create case study performance plot comparing train vs test performance."""
    os.makedirs(output_dir, exist_ok=True)

    datasets = ["Train\n(Normal)", "Train\n(Negative)", "Test\n(Negative)"]

    accuracies = [
        results_train_normal.get("accuracy_macro", results_train_normal.get("accuracy", 0.0)),
        results_train_neg.get("accuracy_macro", results_train_neg.get("accuracy", 0.0)),
        results_test_neg.get("accuracy_macro", results_test_neg.get("accuracy", 0.0)),
    ]

    precisions = [
        results_train_normal.get("precision_macro", results_train_normal.get("f1_macro", 0.0)),
        results_train_neg.get("precision_macro", results_train_neg.get("f1_macro", 0.0)),
        results_test_neg.get("precision_macro", results_test_neg.get("f1_macro", 0.0)),
    ]

    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    x = np.arange(len(datasets))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, accuracies, width, label="Macro Accuracy", alpha=0.8, color="#1f77b4"
    )
    bars2 = ax.bar(
        x + width / 2, precisions, width, label="Macro Precision", alpha=0.8, color="#ff7f0e"
    )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "LogP Model Performance\n(Trained on: Normal + Negative Train Data)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0, 1])

    # Add value labels on bars
    for bars, values in [(bars1, accuracies), (bars2, precisions)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, "case_study_logp_performance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {plot_path}")

    plot_path_pdf = os.path.join(output_dir, "case_study_logp_performance.pdf")
    plt.savefig(plot_path_pdf, bbox_inches="tight")
    print(f"✅ Saved: {plot_path_pdf}")

    plt.close()


def plot_logp_grid_and_performance(
    df_original,
    df_neg_train,
    df_neg_test,
    train_normal_results,
    train_neg_results,
    test_results,
    test_neg_results,
    output_dir,
):
    """
    Create a two-panel plot:
    - Left: logP-space scatter of monomer1_logP vs monomer2_logP, colored by source (normal/negative)
      and marked by split (train/test).
    - Right: bar plot with macro accuracy for train/test (normal/negative).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prepare combined dataframe with source/split flags
    df_orig_train = df_original.copy()
    df_orig_train["source"] = "normal"
    df_orig_train["split"] = "train"

    df_neg_train_plot = df_neg_train.copy()
    df_neg_train_plot["source"] = "negative"
    df_neg_train_plot["split"] = "train"

    df_neg_test_plot = df_neg_test.copy()
    df_neg_test_plot["source"] = "negative"
    df_neg_test_plot["split"] = "test"

    df_plot = pd.concat(
        [df_orig_train, df_neg_train_plot, df_neg_test_plot],
        ignore_index=True,
    )

    fig, (ax_grid, ax_bar) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel 1: logP-grid
    color_map = {"normal": "#1f77b4", "negative": "#d62728"}
    marker_map = {"train": "o", "test": "s"}
    alpha_map = {"train": 0.5, "test": 0.9}

    for source in ["normal", "negative"]:
        for split in ["train", "test"]:
            mask = (df_plot["source"] == source) & (df_plot["split"] == split)
            if not mask.any():
                continue
            ax_grid.scatter(
                df_plot.loc[mask, "monomer1_logP"],
                df_plot.loc[mask, "monomer2_logP"],
                c=color_map[source],
                marker=marker_map[split],
                alpha=alpha_map[split],
                label=f"{source.capitalize()} ({split})",
                edgecolors="none",
                s=20,
            )

    ax_grid.set_xlabel("monomer1_logP")
    ax_grid.set_ylabel("monomer2_logP")
    ax_grid.set_title("Verteilung der Daten im logP-Raum")
    ax_grid.legend(fontsize=8, loc="best")
    ax_grid.grid(alpha=0.3)

    # Panel 2: performance bars
    labels = [
        "Train\nNormal",
        "Train\nNegativ",
        "Test\nNormal",
        "Test\nNegativ",
    ]
    accuracies = [
        train_normal_results.get("accuracy_macro", train_normal_results.get("accuracy", 0.0)),
        train_neg_results.get("accuracy_macro", train_neg_results.get("accuracy", 0.0)),
        test_results.get("accuracy_macro", test_results.get("accuracy", 0.0)),
        test_neg_results.get("accuracy_macro", test_neg_results.get("accuracy", 0.0)),
    ]

    x = np.arange(len(labels))
    width = 0.6

    bars = ax_bar.bar(x, accuracies, width, color="#3A3B73", alpha=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_ylabel("Macro Accuracy")
    ax_bar.set_title("Modell-Performance (Train/Test)")
    ax_bar.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, accuracies):
        height = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    plt.tight_layout()
    out_path = os.path.join(output_dir, "logp_grid_and_performance.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {out_path}")
    plt.close()


def plot_logp_ternary(
    df_original,
    df_test_normal,
    df_neg_train,
    df_neg_test,
    output_dir,
):
    """
    Ternary-style plot für (monomer1_logP, monomer2_logP, solvent_logP).
    Jeder Punkt wird auf ein Dreieck projiziert (relative Anteile der drei logP-Werte).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Daten vorbereiten: nur Zeilen mit allen drei logP-Werten
    cols = ["monomer1_logP", "monomer2_logP", "solvent_logP"]

    def _prepare(df, source):
        d = df.copy()
        d = d.dropna(subset=cols)
        d["source"] = source
        return d

    # Alle normalen Daten (Train + Test) in eine Gruppe
    df_norm_all = pd.concat(
        [
            _prepare(df_original, "normal"),
            _prepare(df_test_normal, "normal"),
        ],
        ignore_index=True,
    )

    # Alle negativen Daten (Train + Test) in eine Gruppe
    df_neg_all = pd.concat(
        [
            _prepare(df_neg_train, "negative"),
            _prepare(df_neg_test, "negative"),
        ],
        ignore_index=True,
    )

    df_plot = pd.concat(
        [df_norm_all, df_neg_all],
        ignore_index=True,
    )

    if df_plot.empty:
        print("⚠️ No data with valid logP values for ternary plot; skipping.")
        return

    # logP-Werte pro Achse auf [0, 1] normalisieren (min/max pro Achse),
    # danach auf Anteile (baryzentrisch) normieren.
    vals_raw = df_plot[cols]
    mins = vals_raw.min()
    maxs = vals_raw.max()
    ranges = maxs - mins
    # Schutz gegen Division durch 0 (falls alle Werte auf einer Achse identisch sind)
    ranges[ranges == 0] = 1.0

    vals_scaled = (vals_raw - mins) / ranges
    vals_scaled = vals_scaled.clip(lower=0.0, upper=1.0)

    vals = vals_scaled.values
    sums = vals.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0  # Schutz gegen Division durch 0
    fracs = vals / sums  # f1, f2, f3; Summe = 1

    f1 = fracs[:, 0]  # monomer1
    f2 = fracs[:, 1]  # monomer2
    f3 = fracs[:, 2]  # solvent

    # Baryzentrische Koordinaten -> 2D
    # Ecken: A(0,0)=monomer1, B(1,0)=monomer2, C(0.5, sqrt(3)/2)=solvent
    x = f2 + 0.5 * f3
    y = (np.sqrt(3) / 2.0) * f3

    fig, ax = plt.subplots(figsize=(6, 6))

    # Dreiecksrahmen zeichnen
    tri_x = [0.0, 1.0, 0.5, 0.0]
    tri_y = [0.0, 0.0, np.sqrt(3) / 2.0, 0.0]
    ax.plot(tri_x, tri_y, "k-", linewidth=1)

    color_map = {"normal": "#1f77b4", "negative": "#d62728"}
    # Niedrigere Opazität, damit sich dichte Bereiche nicht komplett überdecken
    alpha_map = {"normal": 0.25, "negative": 0.4}

    for source in ["normal", "negative"]:
        mask = df_plot["source"] == source
        if not mask.any():
            continue
        ax.scatter(
            x[mask],
            y[mask],
            c=color_map[source],
            marker="o",
            alpha=alpha_map[source],
            label=source.capitalize(),
            edgecolors="none",
            s=25,
        )

    # Achsenticks ausblenden, komplettes Dreieck anzeigen
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, (np.sqrt(3) / 2.0) + 0.05)

    ax.text(0.0, -0.05, "monomer1_logP", ha="center", va="top")
    ax.text(1.0, -0.05, "monomer2_logP", ha="center", va="top")
    ax.text(0.5, (np.sqrt(3) / 2.0) + 0.03, "solvent_logP", ha="center", va="bottom")

    ax.set_title("Ternary-Plot der logP-Werte")
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "logp_ternary.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved: {out_path}")
    plt.close()


def main():
    """Main training pipeline."""
    args = parse_args()

    # Set defaults - everything in experiments/case_studies/negative_data
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent.parent

    if args.negative_data is None:
        # Try local path first, then fallback to original location
        local_path = script_dir / "processed_combined_augmented.csv"
        if local_path.exists():
            args.negative_data = local_path
        else:
            args.negative_data = (
                project_root
                / "copol_prediction"
                / "filter"
                / "artificial_datapoints"
                / "processed_combined_augmented.csv"
            )

    if args.output_dir is None:
        args.output_dir = script_dir / "results" / "model_bundle_simple_logp"

    # Load logP cache
    load_logp_cache()

    # Configuration
    config = {
        "output_dir": str(args.output_dir),
        "random_state": args.random_state,
        "hyperparam_iter": args.hyperparam_iter,
        "negative_data_path": str(args.negative_data),
    }

    print("=" * 60)
    print("TRAIN SIMPLE BINARY MODEL (ONLY LOGP FEATURES)")
    print("=" * 60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\nFeatures: monomer1_logP, monomer2_logP, solvent_logP")
    print("Classification: Binary (0/1 vs 2)")

    # Load original training data
    df_original, df_test_normal = load_training_data()

    # Load negative data
    df_neg_all = load_negative_data(args.negative_data)

    # Add logP features FIRST (needed for deduplication and fair splitting)
    print("\n" + "=" * 60)
    print("ADDING LOGP FEATURES")
    print("=" * 60)
    df_original = add_logp_features(df_original)
    df_neg_all = add_logp_features(df_neg_all)
    df_test_normal = add_logp_features(df_test_normal)

    # Reorder monomers so that monomer1_logP <= monomer2_logP
    df_original = reorder_monomers_by_logp(df_original)
    df_neg_all = reorder_monomers_by_logp(df_neg_all)
    df_test_normal = reorder_monomers_by_logp(df_test_normal)

    # Save cache
    save_logp_cache()

    # Remove duplicates FIRST (before splitting) based on logP features
    # This ensures train and test don't have the same reactions
    df_original = remove_duplicates_by_logp_features(df_original)
    df_neg_all = remove_duplicates_by_logp_features(df_neg_all)
    df_test_normal = remove_duplicates_by_logp_features(df_test_normal)

    # NOW split negative data into train/test (80/20) AFTER deduplication
    # Split is based on monomer+solvent combinations (logP features) for fair split
    df_neg_train, df_neg_test = split_negative_data(
        df_neg_all, test_size=0.2, random_state=config["random_state"]
    )

    # Sample normal data to match negative data size
    df_original = sample_normal_data_to_match_negative(
        df_original, df_neg_train, random_state=config["random_state"]
    )

    # Convert to binary classification
    print("\n" + "=" * 60)
    print("CONVERTING TO BINARY CLASSIFICATION")
    print("=" * 60)
    df_original = convert_to_binary_classification(df_original)
    df_neg_train = convert_to_binary_classification(df_neg_train)
    df_neg_test = convert_to_binary_classification(df_neg_test)
    df_test_normal = convert_to_binary_classification(df_test_normal)

    # Combine datasets (only negative TRAIN data goes into training)
    df_train_combined = combine_training_data(df_original, df_neg_train)

    # Train model
    model_info = train_model(df_train_combined, config)

    # Evaluate on training data - split into normal and negative
    print("\n" + "=" * 60)
    print("EVALUATING ON TRAINING DATA (SPLIT)")
    print("=" * 60)

    # Evaluate on normal training data
    train_normal_results = evaluate_on_test(
        model_info["model"],
        df_original,
        model_info["features"],
        is_binary=True,
        dataset_name="Training Data (Normal)",
    )

    # Evaluate on negative training data (the part used for training)
    train_neg_results = evaluate_on_test(
        model_info["model"],
        df_neg_train,
        model_info["features"],
        is_binary=True,
        dataset_name="Training Data (Negative)",
    )

    # Print detailed predictions for negative training data
    print_detailed_predictions(
        model_info["model"],
        df_neg_train,
        model_info["features"],
        dataset_name="Training Data (Negative)",
    )

    # Evaluate on negative test data (the held-out 20%)
    test_neg_split_results = evaluate_on_test(
        model_info["model"],
        df_neg_test,
        model_info["features"],
        is_binary=True,
        dataset_name="Negative Test Data (Split)",
    )

    # Print detailed predictions for negative test split
    print_detailed_predictions(
        model_info["model"],
        df_neg_test,
        model_info["features"],
        dataset_name="Negative Test Data (Split)",
    )

    # For this experiment, we treat the held-out split from
    # processed_combined_augmented.csv as the negative test set.
    test_neg_results = test_neg_split_results

    # Evaluate on normal test set
    test_results = evaluate_on_test(
        model_info["model"],
        df_test_normal,
        model_info["features"],
        is_binary=True,
        dataset_name="Normal Test Set",
    )

    # Save model
    save_model(
        model_info, train_normal_results, train_neg_results, test_results, test_neg_results, config
    )

    # Create case study performance plot
    print("\n" + "=" * 60)
    print("CREATING CASE STUDY PERFORMANCE PLOT")
    print("=" * 60)
    plot_case_study_performance(
        train_normal_results, train_neg_results, test_neg_results, config["output_dir"]
    )

    # Create combined logP-grid + performance plot
    print("\n" + "=" * 60)
    print("CREATING LOGP GRID + PERFORMANCE PLOT")
    print("=" * 60)
    plot_logp_grid_and_performance(
        df_original,
        df_neg_train,
        df_neg_test,
        train_normal_results,
        train_neg_results,
        test_results,
        test_neg_results,
        config["output_dir"],
    )

    # Create ternary logP plot
    print("\n" + "=" * 60)
    print("CREATING LOGP TERNARY PLOT")
    print("=" * 60)
    plot_logp_ternary(
        df_original,
        df_test_normal,
        df_neg_train,
        df_neg_test,
        config["output_dir"],
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nModel saved to: {config['output_dir']}")
    print(f"\nResults summary:")
    print(f"\nTraining Data (Normal):")
    print(
        f"  Accuracy (macro): {train_normal_results.get('accuracy_macro', train_normal_results['accuracy']):.4f}"
    )
    print(f"  F1 (macro): {train_normal_results['f1_macro']:.4f}")

    print(f"\nTraining Data (Negative):")
    print(
        f"  Accuracy (macro): {train_neg_results.get('accuracy_macro', train_neg_results['accuracy']):.4f}"
    )
    print(f"  F1 (macro): {train_neg_results['f1_macro']:.4f}")

    print(f"\nNormal Test Set:")
    print(f"  Accuracy (macro): {test_results.get('accuracy_macro', test_results['accuracy']):.4f}")
    print(f"  F1 (macro): {test_results['f1_macro']:.4f}")

    print(f"\nNegative Test Data (held-out split):")
    print(
        f"  Accuracy (macro): {test_neg_results.get('accuracy_macro', test_neg_results['accuracy']):.4f}"
    )
    print(f"  F1 (macro): {test_neg_results['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
