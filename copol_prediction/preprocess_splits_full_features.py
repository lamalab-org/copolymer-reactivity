#!/usr/bin/env python3
"""
Preprocess existing train/val/test splits with full feature_columns_all.

Loads the current splits from artifacts/data_splits, enriches each with
molecular features from the monomer JSON files (so all feature_columns_all
columns are present), and saves to artifacts/data_splits_full_features.

Use this when you need full features (e.g. for permutation importance)
but the original preprocessing was done with a reduced set.

Usage:
  cd copol_prediction
  python preprocess_splits_full_features.py [--molecule-properties PATH] [--output-dir DIR]
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from copolpredictor import data_processing, prediction_utils
from utils.load_data_split import load_train_val_test_split


def parse_args():
    p = argparse.ArgumentParser(description="Enrich train/val/test splits with full feature_columns_all from monomer files")
    p.add_argument(
        "--split-dir",
        default="artifacts/data_splits",
        help="Directory with train.csv, val.csv, test.csv",
    )
    p.add_argument(
        "--output-dir",
        default="artifacts/data_splits_full_features",
        help="Output directory for enriched splits",
    )
    p.add_argument(
        "--molecule-properties",
        default="output/molecule_properties",
        help="Directory with monomer JSON files (SMILES.json)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    split_dir = os.path.join(script_dir, args.split_dir)
    out_dir = os.path.join(script_dir, args.output_dir)
    mol_path = os.path.join(script_dir, args.molecule_properties)

    if not os.path.isdir(mol_path):
        print(f"Warning: Molecule properties path not found: {mol_path}")
        print("  Enrichment will fill NaN for molecular columns.")

    print("Loading splits from", split_dir)
    df_train, df_val, df_test = load_train_val_test_split(split_dir=split_dir)
    wanted = prediction_utils.feature_columns_all
    missing_before = [c for c in wanted if c not in df_train.columns]
    print(f"  feature_columns_all: {len(wanted)} columns")
    print(f"  Missing in train before enrichment: {len(missing_before)}")

    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        print(f"\nEnriching {name} ({len(df)} rows)...")
        df = data_processing.enrich_df_with_molecular_features(
            df.copy(), base_path=mol_path, feature_columns=wanted
        )
        missing_after = [c for c in wanted if c not in df.columns]
        added = len(wanted) - len(missing_after)
        print(f"  After enrichment: {added} of {len(wanted)} feature_columns_all present")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(out_path, index=False)
        print(f"  Saved to {out_path}")

    print("\nDone. Use these splits e.g. with:")
    print(f"  --split-dir {args.output_dir}")
    print("(in scripts that accept a split directory).")


if __name__ == "__main__":
    main()
