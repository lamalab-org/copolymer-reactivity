"""
Data processing with Morgan fingerprints for monomer representation.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from copolextractor import utils


def _normalize_doi_for_key(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    s = s.strip()
    lowered = s.lower()
    for p in ("https://doi.org/", "http://doi.org/", "doi:", "doi "):
        if lowered.startswith(p):
            s = s[len(p):]
            break
    return s.strip().lower()


def _build_cache_key_from_row(original_source: str) -> str:
    doi = _normalize_doi_for_key(original_source)
    return f"doi::{doi}" if doi else ""


def _load_specialized_cache(cache_path: str) -> dict:
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except:
        return {}


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


def add_morgan_fingerprint_features(df, n_bits=2048, radius=2):
    """Add Morgan fingerprint features for both monomers."""
    print(f"\nGenerating Morgan fingerprints (radius={radius}, n_bits={n_bits})...")
    
    new_rows = []
    skipped = 0
    
    for index, row in df.iterrows():
        try:
            monomer1_smiles = row['monomer1_smiles']
            monomer2_smiles = row['monomer2_smiles']

            if pd.isna(monomer1_smiles) or pd.isna(monomer2_smiles):
                skipped += 1
                continue

            fp1 = get_morgan_fingerprint(monomer1_smiles, radius=radius, n_bits=n_bits)
            fp2 = get_morgan_fingerprint(monomer2_smiles, radius=radius, n_bits=n_bits)

            if fp1 is None or fp2 is None:
                skipped += 1
                continue

            fp1_dict = {f'morgan_bit_{i}_1': int(fp1[i]) for i in range(n_bits)}
            fp2_dict = {f'morgan_bit_{i}_2': int(fp2[i]) for i in range(n_bits)}

            new_row = {**row.to_dict(), **fp1_dict, **fp2_dict}
            new_rows.append(new_row)

        except Exception as e:
            print(f"Error processing row {index}: {e}")
            skipped += 1

    result_df = pd.DataFrame(new_rows)
    print(f"Generated fingerprints for {len(result_df)} reactions (skipped {skipped})")
    
    return result_df


def create_flipped_dataset(df, n_bits=2048):
    """Create flipped dataset with swapped monomers."""
    flipped_rows = []

    for index, row in df.iterrows():
        flipped_row = row.copy()
        
        if 'constant_1' in row:
            flipped_row['constant_1'] = row['constant_2']
            flipped_row['constant_2'] = row['constant_1']
            if 'constant_conf_1' in row:
                flipped_row['constant_conf_1'] = row.get('constant_conf_2')
                flipped_row['constant_conf_2'] = row.get('constant_conf_1')

        flipped_row['monomer1_smiles'] = row['monomer2_smiles']
        flipped_row['monomer2_smiles'] = row['monomer1_smiles']
        flipped_row['monomer1_name'] = row['monomer2_name']
        flipped_row['monomer2_name'] = row['monomer1_name']

        for i in range(n_bits):
            bit1_col = f'morgan_bit_{i}_1'
            bit2_col = f'morgan_bit_{i}_2'
            if bit1_col in row and bit2_col in row:
                flipped_row[bit1_col] = row[bit2_col]
                flipped_row[bit2_col] = row[bit1_col]

        flipped_rows.append(flipped_row)

    return pd.DataFrame(flipped_rows)


def process_embeddings(df, column_name, prefix):
    """Process embeddings with PCA."""
    if column_name not in df.columns:
        return df

    unique_values = [v for v in df[column_name].unique() if not pd.isna(v)]
    if len(unique_values) == 0:
        return df

    embeddings = []
    for value in unique_values:
        embedding = utils.get_or_create_embedding(value)
        if embedding is not None:
            embeddings.append({"name": value, "embedding": embedding})

    if len(embeddings) >= 2:
        embedding_matrix = [item["embedding"] for item in embeddings]
        pca = PCA(n_components=min(2, len(embedding_matrix)))
        reduced_embeddings = pca.fit_transform(embedding_matrix)
        embedding_map = {item["name"]: reduced for item, reduced in zip(embeddings, reduced_embeddings)}

        df[f"{prefix}_1"] = df[column_name].apply(
            lambda x: embedding_map.get(x, [None, None])[0] if not pd.isna(x) else None
        )
        df[f"{prefix}_2"] = df[column_name].apply(
            lambda x: embedding_map.get(x, [None, None])[1] if not pd.isna(x) else None
        )

    valid_values = [item["name"] for item in embeddings]
    df = df[df[column_name].isin(valid_values) | df[column_name].isna()]

    return df


def add_solvent_features(df):
    """Add solvent features."""
    def is_invalid(smiles):
        if pd.isna(smiles):
            return True
        if not isinstance(smiles, str):
            return True
        smiles = smiles.strip().lower()
        return smiles in {"", "na", "nan", "none"}

    def calc_features(smiles):
        if is_invalid(smiles):
            return [None] * 10
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [None] * 10
        return [
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.MolMR(mol),
            rdMolDescriptors.CalcLabuteASA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol),
            Descriptors.HeavyAtomCount(mol)
        ]

    feature_cols = [
        'solvent_logP', 'solvent_TPSA', 'solvent_HBA', 'solvent_HBD',
        'solvent_FractionCSP3', 'solvent_MolMR', 'solvent_LabuteASA',
        'solvent_NumRotatableBonds', 'solvent_RingCount', 'solvent_HeavyAtomCount'
    ]

    feature_values = df['solvent_smiles'].apply(calc_features)
    feature_df = pd.DataFrame(feature_values.tolist(), columns=feature_cols)
    df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)

    return df


def convert_numeric_columns(df, columns):
    """Convert columns to numeric."""
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def load_and_preprocess_data_morgan(input_path, specialized_cache_path=None, n_bits=2048, radius=2):
    """Load and preprocess data with Morgan fingerprints."""
    print(f"Loading data from {input_path}")

    df = pd.read_csv(input_path, decimal='.')
    print(f"Initial datapoints: {len(df)}")

    if specialized_cache_path:
        cache = _load_specialized_cache(specialized_cache_path)
        def _lookup_specialized(row):
            key = _build_cache_key_from_row(row.get("original_source", ""))
            if key and key in cache:
                cls = str(cache[key].get("classification", "")).strip().lower()
                if cls in ("normal", "specialized", "unclear"):
                    return cls
            return "unclear"

        if "specialized_filter" not in df.columns:
            df["specialized_filter"] = df.apply(_lookup_specialized, axis=1)

    df.dropna(subset=['constant_1', 'constant_2', 'monomer1_smiles', 'monomer2_smiles'], inplace=True)
    df = convert_numeric_columns(df, ['constant_1', 'constant_2', 'temperature'])
    df['r1r2'] = df['constant_1'] * df['constant_2']

    # Confidence intervals
    mask_conf = df[['constant_conf_1', 'constant_conf_2']].notnull().all(axis=1)
    df.loc[mask_conf, 'constant_1_plus'] = df.loc[mask_conf, 'constant_1'] + df.loc[mask_conf, 'constant_conf_1']
    df.loc[mask_conf, 'constant_1_minus'] = df.loc[mask_conf, 'constant_1'] - df.loc[mask_conf, 'constant_conf_1']
    df.loc[mask_conf, 'constant_2_plus'] = df.loc[mask_conf, 'constant_2'] + df.loc[mask_conf, 'constant_conf_2']
    df.loc[mask_conf, 'constant_2_minus'] = df.loc[mask_conf, 'constant_2'] - df.loc[mask_conf, 'constant_conf_2']

    c1_variants = {'orig': 'constant_1', 'plus': 'constant_1_plus', 'minus': 'constant_1_minus'}
    c2_variants = {'orig': 'constant_2', 'plus': 'constant_2_plus', 'minus': 'constant_2_minus'}

    for c1_key, c1_col in c1_variants.items():
        for c2_key, c2_col in c2_variants.items():
            if c1_key == 'orig' and c2_key == 'orig':
                continue
            product_col = f'product_c1{c1_key}_c2{c2_key}'
            df.loc[mask_conf, product_col] = df.loc[mask_conf, c1_col] * df.loc[mask_conf, c2_col]

    df = add_solvent_features(df)
    
    print("\n" + "="*60)
    print("USING MORGAN FINGERPRINTS FOR MONOMER REPRESENTATION")
    print("="*60)
    df = add_morgan_fingerprint_features(df, n_bits=n_bits, radius=radius)

    if len(df) == 0:
        return None

    df['reaction_id'] = df.index
    df_flipped = create_flipped_dataset(df, n_bits=n_bits)
    combined_df = pd.concat([df, df_flipped])
    print(f"Total datapoints after flipping: {len(combined_df)}")

    combined_df = process_embeddings(combined_df, "polymerization_type", "polytype_emb")
    combined_df = process_embeddings(combined_df, "method", "method_emb")

    return combined_df

