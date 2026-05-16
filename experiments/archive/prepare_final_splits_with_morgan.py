#!/usr/bin/env python3
"""
Prepare final splits with Morgan fingerprints for comparison experiment.
Uses the same splits as the final model from copol_prediction/artifacts/data_splits.
"""

import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """Generate Morgan fingerprint for SMILES."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except Exception as e:
        print(f"Error generating fingerprint for {smiles}: {e}")
        return None


def add_morgan_features(df, n_bits=2048, radius=2):
    """Add Morgan fingerprint features to dataframe."""
    print(f"Adding Morgan fingerprints (radius={radius}, n_bits={n_bits})...")

    new_rows = []
    skipped = 0

    for index, row in df.iterrows():
        try:
            monomer1_smiles = row["monomer1_smiles"]
            monomer2_smiles = row["monomer2_smiles"]

            if pd.isna(monomer1_smiles) or pd.isna(monomer2_smiles):
                skipped += 1
                continue

            fp1 = get_morgan_fingerprint(monomer1_smiles, radius=radius, n_bits=n_bits)
            fp2 = get_morgan_fingerprint(monomer2_smiles, radius=radius, n_bits=n_bits)

            if fp1 is None or fp2 is None:
                skipped += 1
                continue

            # Add fingerprint features
            fp1_dict = {f"morgan_bit_{i}_1": int(fp1[i]) for i in range(n_bits)}
            fp2_dict = {f"morgan_bit_{i}_2": int(fp2[i]) for i in range(n_bits)}

            new_row = {**row.to_dict(), **fp1_dict, **fp2_dict}
            new_rows.append(new_row)

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            skipped += 1

    result_df = pd.DataFrame(new_rows)
    print(f"Added Morgan features to {len(result_df)} reactions (skipped {skipped})")

    return result_df


def main():
    print("=" * 60)
    print("PREPARING FINAL SPLITS WITH MORGAN FINGERPRINTS")
    print("=" * 60)

    # Paths
    final_splits_dir = os.path.join(
        os.path.dirname(__file__), "../copol_prediction/artifacts/data_splits"
    )
    output_dir = os.path.join(os.path.dirname(__file__), "data")

    train_path = os.path.join(final_splits_dir, "train.csv")
    test_path = os.path.join(final_splits_dir, "test.csv")

    # Load final splits
    print(f"\nLoading final splits from: {final_splits_dir}")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print(f"Train samples: {len(df_train)}")
    print(f"Test samples: {len(df_test)}")

    # Copy splits to experiments/data (for baseline)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCopying splits to: {output_dir}")
    df_train.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    df_test.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print("✓ Saved train.csv and test.csv (for baseline)")

    # Add Morgan fingerprints
    print("\n" + "=" * 60)
    print("ADDING MORGAN FINGERPRINTS")
    print("=" * 60)

    df_train_morgan = add_morgan_features(df_train, n_bits=2048, radius=2)
    df_test_morgan = add_morgan_features(df_test, n_bits=2048, radius=2)

    # Save Morgan versions
    train_morgan_path = os.path.join(output_dir, "train_morgan.csv")
    test_morgan_path = os.path.join(output_dir, "test_morgan.csv")

    df_train_morgan.to_csv(train_morgan_path, index=False)
    df_test_morgan.to_csv(test_morgan_path, index=False)

    print(f"\n✓ Saved train_morgan.csv: {len(df_train_morgan)} samples")
    print(f"✓ Saved test_morgan.csv: {len(df_test_morgan)} samples")

    # Verify column counts
    morgan_cols = [c for c in df_train_morgan.columns if c.startswith("morgan_bit_")]
    print(f"\nMorgan fingerprint features: {len(morgan_cols)}")

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nFiles ready in: {output_dir}")
    print("  - train.csv & test.csv (for baseline with quantum features)")
    print("  - train_morgan.csv & test_morgan.csv (for fingerprint experiment)")


if __name__ == "__main__":
    main()
