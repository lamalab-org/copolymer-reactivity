#!/usr/bin/env python3
"""
Create experiment-specific data (uses central train/test split).

This script:
1. Loads the central train/test split from copol_prediction/artifacts/data_splits/
2. Creates experiment-specific versions (e.g., with Morgan fingerprints)

NOTE: This script only creates derived data (Morgan fingerprints).
The normal splits (train.csv, test.csv) should NOT be duplicated.
All scripts should use the central split directly from copol_prediction/artifacts/data_splits/

The central split should be created FIRST by running:
    cd ../copol_prediction && python create_data_split.py
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fingerprints", action="store_true", help="Create Morgan fingerprint version"
    )
    return parser.parse_args()


def load_central_split():
    """Load the central train/test split created by copol_prediction/create_data_split.py"""
    print("=" * 70)
    print("LOADING CENTRAL TRAIN/TEST SPLIT")
    print("=" * 70)

    train_path = "../copol_prediction/artifacts/data_splits/train.csv"
    test_path = "../copol_prediction/artifacts/data_splits/test.csv"

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("\nError: Central train/test split not found!")
        print("Expected files:")
        print(f"  - {train_path}")
        print(f"  - {test_path}")
        print("\nYou must create the central split first:")
        print("  cd ../copol_prediction")
        print("  python create_data_split.py")
        print("  cd ../experiments")
        sys.exit(1)

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"\n✓ Loaded central split:")
    print(f"  Train: {len(df_train)} samples ({df_train['reaction_id'].nunique()} groups)")
    print(f"  Test:  {len(df_test)} samples ({df_test['reaction_id'].nunique()} groups)")

    return df_train, df_test


def copy_baseline_split(df_train, df_test, output_dir="data"):
    """
    DEPRECATED: Do not copy baseline splits anymore.

    All scripts should use the central split directly from:
    copol_prediction/artifacts/data_splits/

    This function is kept for backward compatibility but does nothing.
    """
    print("\n" + "=" * 70)
    print("NOTE: Baseline splits are no longer copied")
    print("=" * 70)
    print("\nAll scripts should use the central split directly:")
    print("  copol_prediction/artifacts/data_splits/train.csv")
    print("  copol_prediction/artifacts/data_splits/test.csv")
    print("\nUse load_data_split.load_train_test_split() in your scripts.")
    return df_train, df_test


def create_morgan_split(df_train, df_test, output_dir="data", n_bits=2048, radius=2):
    """Create Morgan fingerprint version of the train/test split."""
    print("\n" + "=" * 70)
    print("CREATING MORGAN FINGERPRINT DATA")
    print("=" * 70)
    print(f"Parameters: {n_bits} bits, radius {radius}")

    # Import from feature_comparison/fingerprint
    fingerprint_dir = os.path.join(
        os.path.dirname(__file__), "..", "feature_comparison", "fingerprint"
    )
    sys.path.insert(0, fingerprint_dir)
    import data_processing_morgan

    print("\nGenerating Morgan fingerprints for train set...")
    df_train_morgan = data_processing_morgan.add_morgan_fingerprint_features(
        df_train, n_bits=n_bits, radius=radius
    )

    print("\nGenerating Morgan fingerprints for test set...")
    df_test_morgan = data_processing_morgan.add_morgan_fingerprint_features(
        df_test, n_bits=n_bits, radius=radius
    )

    # Remove any NaN rows that might have been introduced
    print("\nChecking for NaN in Morgan fingerprint data...")
    train_before = len(df_train_morgan)
    test_before = len(df_test_morgan)

    # Get all feature columns (morgan + others)
    morgan_features = [f"morgan_bit_{i}_1" for i in range(n_bits)] + [
        f"morgan_bit_{i}_2" for i in range(n_bits)
    ]
    other_features = [
        "temperature",
        "polytype_emb_1",
        "polytype_emb_2",
        "method_emb_1",
        "method_emb_2",
        "solvent_logP",
        "solvent_TPSA",
        "solvent_HBD",
        "solvent_FractionCSP3",
    ]
    all_features = morgan_features + other_features
    available_features = [c for c in all_features if c in df_train_morgan.columns]

    # Remove NaN
    mask_train = ~df_train_morgan[available_features].isna().any(axis=1)
    mask_test = ~df_test_morgan[available_features].isna().any(axis=1)
    df_train_morgan = df_train_morgan[mask_train].reset_index(drop=True)
    df_test_morgan = df_test_morgan[mask_test].reset_index(drop=True)

    train_removed = train_before - len(df_train_morgan)
    test_removed = test_before - len(df_test_morgan)
    if train_removed > 0:
        print(f"  Removed {train_removed} NaN rows from train set")
    if test_removed > 0:
        print(f"  Removed {test_removed} NaN rows from test set")
    print(f"  Final: train={len(df_train_morgan)}, test={len(df_test_morgan)}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save
    train_path = os.path.join(output_dir, "train_morgan.csv")
    test_path = os.path.join(output_dir, "test_morgan.csv")

    df_train_morgan.to_csv(train_path, index=False)
    df_test_morgan.to_csv(test_path, index=False)

    print(f"\n✓ Saved to: {train_path}")
    print(f"✓ Saved to: {test_path}")

    print("\n" + "=" * 70)
    print("MORGAN DATA CREATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    args = parse_args()

    print("\n" + "=" * 70)
    print("EXPERIMENT DATA PREPARATION")
    print("=" * 70)
    print("\nThis script creates experiment-specific data from the central split")
    print()

    # Load central split
    df_train, df_test = load_central_split()

    # NOTE: We no longer copy baseline splits to experiments/data/
    # All scripts should use the central split directly

    # Optionally create Morgan fingerprint version
    if args.fingerprints:
        # Save Morgan data in feature_comparison directory
        morgan_output_dir = os.path.join("feature_comparison", "data")
        create_morgan_split(df_train, df_test, output_dir=morgan_output_dir)
    else:
        print("\n⚠️  No action specified. Use --fingerprints to create Morgan fingerprint data.")
        print("   Normal splits should be loaded directly from the central location.")

    print("\n" + "=" * 70)
    print("EXPERIMENT DATA READY")
    print("=" * 70)
    print("\nYou can now run experiments with consistent train/test splits!")
