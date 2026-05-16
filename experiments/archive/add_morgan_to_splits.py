#!/usr/bin/env python3
"""
Add Morgan fingerprints to the final splits.
"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import RDKit carefully
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    print("ERROR: RDKit not installed!")
    sys.exit(1)


def get_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """Generate Morgan fingerprint for SMILES."""
    if pd.isna(smiles) or not isinstance(smiles, str) or not smiles.strip():
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.int8)
    except:
        return None


def process_split(input_path, output_path, n_bits=2048, radius=2):
    """Process one split file and add Morgan features."""
    print(f"\nProcessing: {os.path.basename(input_path)}")

    # Load data in chunks to avoid memory issues
    df = pd.read_csv(input_path)
    print(f"  Loaded {len(df)} samples")

    # Process in batches
    batch_size = 100
    results = []

    for i in tqdm(range(0, len(df), batch_size), desc="  Adding Morgan FPs"):
        batch = df.iloc[i : i + batch_size].copy()

        for idx, row in batch.iterrows():
            smiles1 = row["monomer1_smiles"]
            smiles2 = row["monomer2_smiles"]

            fp1 = get_morgan_fingerprint(smiles1, radius=radius, n_bits=n_bits)
            fp2 = get_morgan_fingerprint(smiles2, radius=radius, n_bits=n_bits)

            if fp1 is not None and fp2 is not None:
                # Add fingerprint columns
                for bit_idx in range(n_bits):
                    row[f"morgan_bit_{bit_idx}_1"] = int(fp1[bit_idx])
                    row[f"morgan_bit_{bit_idx}_2"] = int(fp2[bit_idx])
                results.append(row)

    result_df = pd.DataFrame(results)
    print(f"  Successfully processed {len(result_df)} samples")

    # Save
    result_df.to_csv(output_path, index=False)
    print(f"  Saved to: {os.path.basename(output_path)}")

    return len(result_df)


def main():
    print("=" * 60)
    print("ADDING MORGAN FINGERPRINTS TO FINAL SPLITS")
    print("=" * 60)

    data_dir = os.path.join(os.path.dirname(__file__), "data")

    # Process train
    train_in = os.path.join(data_dir, "train.csv")
    train_out = os.path.join(data_dir, "train_morgan.csv")
    n_train = process_split(train_in, train_out)

    # Process test
    test_in = os.path.join(data_dir, "test.csv")
    test_out = os.path.join(data_dir, "test_morgan.csv")
    n_test = process_split(test_in, test_out)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"Train samples with Morgan FPs: {n_train}")
    print(f"Test samples with Morgan FPs: {n_test}")


if __name__ == "__main__":
    main()
