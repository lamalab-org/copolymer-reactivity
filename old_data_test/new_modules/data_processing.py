"""
Data processing module for the copolymerization prediction model
Contains functions for loading, preprocessing, and transforming data
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from copolextractor import utils


def load_molecular_data(smiles, base_path='../../copol_prediction/output/molecule_properties'):
    """Load molecular properties from JSON file"""
    try:
        file_path = os.path.join(base_path, f'{smiles}.json')
        with open(file_path, 'r') as handle:
            d = json.load(handle)

        # Process dict fields: take min, max, mean
        for key in ['charges', 'fukui_electrophilicity', 'fukui_nucleophilicity', 'fukui_radical']:
            if key in d and isinstance(d[key], dict) and d[key]:
                d[key + '_min'] = min(d[key].values())
                d[key + '_max'] = max(d[key].values())
                d[key + '_mean'] = sum(d[key].values()) / len(d[key].values())

        # Extract dipole components if present
        if 'dipole' in d and isinstance(d['dipole'], list) and len(d['dipole']) >= 3:
            d['dipole_x'] = d['dipole'][0]
            d['dipole_y'] = d['dipole'][1]
            d['dipole_z'] = d['dipole'][2]

        return d
    except FileNotFoundError:
        print(f"File not found for SMILES: {smiles}")
        return None
    except Exception as e:
        print(f"Error processing molecular data for {smiles}: {e}")
        return None


def molecular_features(smiles):
    """Extract numerical features from molecular data"""
    d = load_molecular_data(smiles)
    if d is None:
        return None
    # Select only float values
    d = {k: v for k, v in d.items() if isinstance(v, float)}
    return d


def add_molecular_features(df):
    """Add molecular features to DataFrame for both monomers"""
    new_rows = []
    for index, row in df.iterrows():
        try:
            monomer1_smiles = row['monomer1_smiles']
            monomer2_smiles = row['monomer2_smiles']

            # Skip entries without SMILES
            if pd.isna(monomer1_smiles) or pd.isna(
                    monomer2_smiles) or monomer1_smiles is None or monomer2_smiles is None:
                print(f"  Skipping row {index}: Missing SMILES")
                continue

            # Get molecular features from files - do not use dummy features
            monomer1_data = molecular_features(monomer1_smiles)
            monomer2_data = molecular_features(monomer2_smiles)

            # Skip row if any molecular data is missing
            if monomer1_data is None:
                print(f"  Skipping row {index}: Missing molecular data for monomer1: {monomer1_smiles}")
                continue

            if monomer2_data is None:
                print(f"  Skipping row {index}: Missing molecular data for monomer2: {monomer2_smiles}")
                continue

            # Add _1 and _2 to keys
            monomer1_data = {f'{k}_1': v for k, v in monomer1_data.items()}
            monomer2_data = {f'{k}_2': v for k, v in monomer2_data.items()}

            # Create new row with all data
            new_row = {
                **row,
                **monomer1_data,
                **monomer2_data
            }
            new_rows.append(new_row)
            print(f"  Successfully processed row {index}")

        except Exception as e:
            print(f"  Error processing row {index}: {e}")

    result_df = pd.DataFrame(new_rows)
    print(f"Final dataframe shape after adding molecular features: {result_df.shape}")
    return result_df


def create_flipped_dataset(df):
    """Create another dataset with flipped monomers, preserving reaction_id for proper train/test splits"""
    flipped_rows = []

    for index, row in df.iterrows():
        flipped_row = row.copy()
        # Swap monomer fields
        flipped_row['monomer1_smiles'] = row['monomer2_smiles']
        flipped_row['monomer2_smiles'] = row['monomer1_smiles']
        flipped_row['monomer1_name'] = row['monomer2_name']
        flipped_row['monomer2_name'] = row['monomer1_name']
        flipped_row['constant_1'] = row['constant_2']
        flipped_row['constant_2'] = row['constant_1']

        # Swap other monomer-specific fields if they exist
        for key_pair in [
            ('constant_conf_1', 'constant_conf_2'),
            ('e_value_1', 'e_value_2'),
            ('q_value_1', 'q_value_2')
        ]:
            key1, key2 = key_pair
            if key1 in row and key2 in row:
                flipped_row[key1] = row[key2]
                flipped_row[key2] = row[key1]

        # Swap all monomer features that end with _1 and _2
        for key in list(row.keys()):
            if key.endswith('_1') and key.replace('_1', '_2') in row:
                flipped_row[key] = row[key.replace('_1', '_2')]
                flipped_row[key.replace('_1', '_2')] = row[key]

        flipped_rows.append(flipped_row)

    return pd.DataFrame(flipped_rows)


def process_embeddings(df, column_name, prefix):
    """
    Processes a specified column into embeddings and applies PCA
    """
    if column_name not in df.columns:
        print(f"Column {column_name} not found in DataFrame")
        return df

    print(f"Processing embeddings for {column_name}...")

    # Get unique non-NaN values
    unique_values = [v for v in df[column_name].unique() if not pd.isna(v)]

    if len(unique_values) == 0:
        print(f"No valid values found in {column_name}")
        return df

    embeddings = []
    embedding_map = {}

    # Generate embeddings for unique values
    for value in unique_values:
        embedding = utils.get_or_create_embedding(value)
        if embedding is not None:
            embeddings.append({"name": value, "embedding": embedding})

    # If we have enough embeddings, apply PCA
    if len(embeddings) >= 2:
        # Convert embeddings into matrix for PCA
        embedding_matrix = [item["embedding"] for item in embeddings]

        # Use PCA to reduce dimensions to 2
        pca = PCA(n_components=min(2, len(embedding_matrix)))
        reduced_embeddings = pca.fit_transform(embedding_matrix)

        # Map reduced embeddings back to the column values
        embedding_map = {item["name"]: reduced for item, reduced in zip(embeddings, reduced_embeddings)}

        # Add PCA components to the DataFrame
        df[f"{prefix}_1"] = df[column_name].apply(
            lambda x: embedding_map.get(x, [None, None])[0] if not pd.isna(x) else None
        )
        df[f"{prefix}_2"] = df[column_name].apply(
            lambda x: embedding_map.get(x, [None, None])[1] if not pd.isna(x) else None
        )

        print(f"PCA reduced embeddings for {column_name} added as {prefix}_1 and {prefix}_2.")

        # Save embeddings to file
        utils.save_embeddings({item["name"]: item["embedding"] for item in embeddings},
                              f"output/{prefix}_embeddings.json")
    else:
        print(f"Not enough valid embeddings for {column_name} to perform PCA")

    # Filter out rows where this category doesn't have an embedding
    valid_values = [item["name"] for item in embeddings]
    before_count = len(df)
    df = df[df[column_name].isin(valid_values) | df[column_name].isna()]
    after_count = len(df)

    if before_count > after_count:
        print(f"Removed {before_count - after_count} rows with values in {column_name} that couldn't be embedded")

    return df


def load_and_preprocess_data(input_path="../data_extraction/new/extracted_reactions.csv"):
    """
    Main function to load and preprocess data

    Loads the data, adds molecular properties, creates reaction IDs,
    creates flipped datasets, and processes embeddings

    Returns:
        DataFrame: The preprocessed DataFrame, ready for model training
    """
    print(f"Loading data from {input_path}")

    try:
        # Load data as CSV
        df = pd.read_csv(input_path)
        print(f"Initial datapoints: {len(df)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    # Display columns for debugging
    print("\nAvailable columns:")
    for col in sorted(df.columns):
        print(f"- {col}")

    # Remove rows with missing values
    df.dropna(subset=['constant_1', 'constant_2', 'monomer1_smiles', 'monomer2_smiles'], inplace=True)
    print(f"DataFrame shape after dropping rows with missing values: {df.shape}")

    # Add molecular features
    print("Adding molecular features...")
    df = add_molecular_features(df)
    print(f"DataFrame shape after adding molecular features: {df.shape}")

    # Check if any data remains
    if len(df) == 0:
        print("No data left after adding molecular features. Check if feature files exist.")
        return None

    # Create unique reaction ID BEFORE flipping the dataset
    print("Creating unique reaction IDs...")
    df['reaction_id'] = df.index
    print(f"Created {df['reaction_id'].nunique()} unique reaction IDs")

    # Create flipped dataset
    print("Creating flipped dataset...")
    df_flipped = create_flipped_dataset(df)

    # Combine original and flipped datasets
    combined_df = pd.concat([df, df_flipped])
    print(f"Total datapoints after augmentation: {len(combined_df)}")

    # Process embeddings for categorical features
    print("\nProcessing embeddings for categorical features...")
    combined_df = process_embeddings(combined_df, "polymerization_type", "polytype_emb")
    combined_df = process_embeddings(combined_df, "method", "method_emb")

    # Save processed data
    combined_df.to_csv("processed_data.csv", index=False)
    print("Data saved to processed_data.csv")

    return combined_df