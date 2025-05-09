import pandas as pd
import json
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Descriptors import MolLogP
import os
from sklearn.decomposition import PCA
from openai import OpenAI


# Load existing embeddings from file if available
def load_embeddings(file_path="embeddings.json"):
    """Load embeddings from a JSON file if it exists."""
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            embeddings = json.load(file)
            print(f"Loaded {len(embeddings)} embeddings from {file_path}.")
            return {item["name"]: item["embedding"] for item in embeddings}
    return {}


# Save embeddings to a file
def save_embeddings(embeddings_dict, file_path="output_2/method_embeddings.json"):
    """Save embeddings to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    embeddings_list = [{"name": name, "embedding": embedding} for name, embedding in embeddings_dict.items()]
    with open(file_path, "w") as file:
        json.dump(embeddings_list, file, indent=4)
    print(f"Saved {len(embeddings_list)} embeddings to {file_path}.")


# Initialize global embedding cache
embedding_cache = load_embeddings()

# Initialize OpenAI client
# Note: This requires an API key set via environment variable or directly
try:
    client = OpenAI()
except Exception as e:
    print(f"Warning: Could not initialize OpenAI client: {e}")
    print("Will use dummy embeddings instead.")
    client = None


# Function: Check cache and get embedding
def get_or_create_embedding(client, text, model="text-embedding-3-small"):
    """
    Retrieve embeddings for a given text.
    """
    global embedding_cache  # Ensure global access to the embedding cache

    # Handle None or NaN text inputs safely
    if text is None or pd.isna(text):
        print("Warning: Found None or NaN in text, skipping embedding.")
        return None  # Return None for missing values to ensure they're filtered out

    # Clean the text by removing newlines for consistency
    text_cleaned = text.replace("\n", " ")

    # Check if the embedding already exists in the cache
    if text_cleaned in embedding_cache:
        return embedding_cache[text_cleaned]  # Return the cached embedding

    # If OpenAI client is not available, use a deterministic hash-based embedding
    if client is None:
        print(f"Using hash-based embedding for: {text_cleaned}")
        # Generate a stable but random-like embedding based on text hash
        import hashlib
        hash_val = int(hashlib.md5(text_cleaned.encode()).hexdigest(), 16)
        np.random.seed(hash_val % 2 ** 32)
        hash_embedding = np.random.normal(0, 1, 1536).tolist()
        embedding_cache[text_cleaned] = hash_embedding
        return hash_embedding

    try:
        # If not in the cache, generate the embedding using the OpenAI API
        response = client.embeddings.create(input=[text_cleaned], model=model)
        embedding = response.data[0].embedding  # Extract the embedding from the response

        # Store the new embedding in the cache for future use
        embedding_cache[text_cleaned] = embedding
        return embedding
    except Exception as e:
        print(f"Error getting embedding for '{text_cleaned}': {e}")
        return None  # Return None on error to ensure row will be filtered


# Function: Convert str to embeddings and apply PCA
def process_embeddings(df, client, column_name, prefix):
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

    # Generate embeddings or fetch from cache for unique values
    for value in unique_values:
        embedding = get_or_create_embedding(client, value)  # Fetch embedding
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
        save_embeddings({item["name"]: item["embedding"] for item in embeddings},
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


# Function to load molecular data from JSON files
def load_molecular_data(smiles):
    """Load molecular properties from JSON file"""
    try:
        with open(f'../copol_prediction/output/molecule_properties/{smiles}.json', 'r') as handle:
            d = json.load(handle)

        # Process dict fields: take min, max, mean
        for key in ['charges', 'fukui_electrophilicity', 'fukui_nucleophilicity', 'fukui_radical']:
            d[key + '_min'] = min(d[key].values())
            d[key + '_max'] = max(d[key].values())
            d[key + '_mean'] = sum(d[key].values()) / len(d[key].values())

        # Extract dipole components
        d['dipole_x'] = d['dipole'][0]
        d['dipole_y'] = d['dipole'][1]
        d['dipole_z'] = d['dipole'][2]

        return d
    except FileNotFoundError:
        print(f"File not found for SMILES: {smiles}")
        return None


def molecular_features(smiles):
    """Extract numerical features from molecular data"""
    d = load_molecular_data(smiles)
    if d is None:
        return None
    # Select only float values
    d = {k: v for k, v in d.items() if isinstance(v, float)}
    return d


def is_within_deviation(actual_product, expected_product, deviation=0.10):
    """Check if product is within acceptable deviation"""
    if expected_product == 0:
        return actual_product == 0
    return abs(actual_product - expected_product) / abs(expected_product) <= deviation


# This function is no longer used but kept for reference
def create_dummy_features():
    """Create dummy molecular features when real data is not available"""
    # Note: This function is no longer used as we now skip rows with missing data
    dummy = {}
    for feature in ['ip', 'ip_corrected', 'ea', 'homo', 'lumo',
                    'global_electrophilicity', 'global_nucleophilicity',
                    'best_conformer_energy', 'charges_min', 'charges_max',
                    'charges_mean', 'fukui_electrophilicity_min',
                    'fukui_electrophilicity_max', 'fukui_electrophilicity_mean',
                    'fukui_nucleophilicity_min', 'fukui_nucleophilicity_max',
                    'fukui_nucleophilicity_mean', 'fukui_radical_min',
                    'fukui_radical_max', 'fukui_radical_mean', 'dipole_x',
                    'dipole_y', 'dipole_z']:
        dummy[feature] = 0.0
    return dummy


def create_flipped_dataset(df):
    """Create another dataset with flipped monomers"""
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
        flipped_row['constant_conf_1'] = row['constant_conf_2']
        flipped_row['constant_conf_2'] = row['constant_conf_1']
        flipped_row['e_value_1'] = row['e_value_2']
        flipped_row['e_value_2'] = row['e_value_1']
        flipped_row['q_value_1'] = row['q_value_2']
        flipped_row['q_value_2'] = row['q_value_1']

        # Swap all monomer features
        for key in list(row.keys()):
            if key.endswith('_1') and key.replace('_1', '_2') in row:
                flipped_row[key] = row[key.replace('_1', '_2')]
                flipped_row[key.replace('_1', '_2')] = row[key]

        flipped_rows.append(flipped_row)

    return pd.DataFrame(flipped_rows)


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
            monomer1_data = None
            monomer2_data = None

            try:
                monomer1_data = molecular_features(monomer1_smiles)
            except Exception as e:
                print(f"  Error loading molecular features for monomer1: {e}")

            try:
                monomer2_data = molecular_features(monomer2_smiles)
            except Exception as e:
                print(f"  Error loading molecular features for monomer2: {e}")

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


def main():
    # Load the data
    input_path = ("../data_extraction/new/extracted_reactions.csv")
    print(f"Loading data from {input_path}")

    try:
        # Load data as CSV instead of JSON
        df = pd.read_csv(input_path)
        print(f"Initial datapoints: {len(df)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Display column names for debugging
    print("\nAvailable columns:")
    for col in sorted(df.columns):
        print(f"- {col}")

    # Clean up data - remove rows with missing essential values
    df.dropna(subset=['constant_1', 'constant_2', 'monomer1_smiles', 'monomer2_smiles'], inplace=True)
    print(f"DataFrame shape after dropping rows with missing values: {df.shape}")

    # Add molecular features
    print("Adding molecular features...")
    df = add_molecular_features(df)
    print(f"DataFrame shape after adding molecular features: {df.shape}")

    # Check if we have any data left
    if len(df) == 0:
        print("No data left after adding molecular features. Check if feature files exist.")
        return

    # Create a unique reaction ID for each row BEFORE flipping
    print("Creating unique reaction IDs...")
    df['reaction_id'] = df.apply(
        lambda
            row: f"{row['monomer1_smiles']}_{row['monomer2_smiles']}_{row.get('temperature', '')}_{row.get('method', '')}_{row.get('polymerization_type', '')}_{row.get('solvent', '')}",
        axis=1
    )
    print(f"Created {df['reaction_id'].nunique()} unique reaction IDs")

    # Create flipped dataset to augment data
    print("Creating flipped dataset...")
    df_flipped = create_flipped_dataset(df)

    # Combine original and flipped datasets
    combined_df = pd.concat([df, df_flipped])
    print(f"Total datapoints after augmentation: {len(combined_df)}")

    # Process embeddings for categorical features
    print("\nProcessing embeddings for categorical features...")
    combined_df = process_embeddings(combined_df, client, "polymerization_type", "polytype_emb")
    combined_df = process_embeddings(combined_df, client, "method", "method_emb")

    # Save processed data
    combined_df.to_csv("processed_data.csv", index=False)
    print("Data saved to processed_data.csv")

    print(combined_df['polymerization_type'].unique())

    # Always train the model, no matter how many data points
    if len(combined_df) > 0:
        print("\n=== Starting model training ===")
        train_model(combined_df)
        evaluate_learning_curve(combined_df)
    else:
        print("No data to train model. Exiting.")


def create_grouped_kfold_splits(df, n_splits=5, random_state=42, id_column='reaction_id'):
    """
    Create K-Fold splits that keep reactions with the same ID together
    in either train or test set.

    Args:
        df: DataFrame with reaction data
        n_splits: Number of folds
        random_state: Random seed for reproducibility
        id_column: Column to use as unique identifier for reactions

    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Check if the ID column exists, if not create one based on index
    if id_column not in df.columns:
        print(f"Warning: {id_column} not found, creating a reaction ID based on index")
        df['temp_reaction_id'] = df.index
        id_column = 'temp_reaction_id'

    # Get unique reaction IDs
    unique_reaction_ids = df[id_column].unique()

    # Shuffle the unique IDs
    np.random.seed(random_state)
    np.random.shuffle(unique_reaction_ids)

    # Create folds based on unique reaction IDs
    folds = np.array_split(unique_reaction_ids, n_splits)

    # Create train/test splits
    splits = []
    for i in range(n_splits):
        test_ids = folds[i]
        test_mask = df[id_column].isin(test_ids)
        test_indices = np.where(test_mask)[0]
        train_indices = np.where(~test_mask)[0]
        splits.append((train_indices, test_indices))

    return splits


def train_model(df):
    """Train and evaluate XGBoost model with proper handling of flipped monomer pairs"""
    print("\nTraining model...")

    # Import plotly at the beginning to ensure it's available
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        plotly_available = True
    except ImportError:
        print("Warning: Plotly not installed. Interactive plots will not be created.")
        print("Install with: pip install plotly")
        plotly_available = False

    # Filter for radical polymerization types
    radical_types = [
        'free radical', 'Free radical', 'Free Radical',
        'atom transfer radical polymerization',
        'atom-transfer radical polymerization',
        'nickel-mediated radical', 'bulk',
        'Radical', 'radical',
        'controlled radical',
        'controlled/living radical',
        'conventional radical polymerization',
        'reversible deactivation radical polymerization',
        'reversible addition-fragmentation chain transfer polymerization',
        'reversible addition-fragmentation chain transfer',
        'Homogeneous Radical',
        'Radiation-induced', 'radiation-induced',
        'Radiation-Initiated',
        'photo-induced polymerization',
        'photopolymerization',
        'thermal polymerization',
        'thermal',
        'group transfer polymerization',
        'Emulsion',
        'Homogeneous Radical',
        'semicontinuous emulsion',
        'emulsion'
    ]

    # Count datapoints before filtering
    original_count = len(df)

    # Filter data to keep only radical polymerization types
    df = df[df['polymerization_type'].isin(radical_types)]

    # Count datapoints after filtering
    filtered_count = len(df)
    removed_count = original_count - filtered_count

    print(f"Original datapoints: {original_count}")
    print(f"Filtered datapoints (radical polymerization): {filtered_count}")
    print(f"Removed datapoints (non-radical polymerization): {removed_count}")

    # Create target variable
    df['r1r2'] = df['constant_1'] * df['constant_2']

    # Drop rows where r1r2 is less than 0
    original_count = len(df)
    df = df[df['r1r2'] >= 0]
    dropped_negative_count = original_count - len(df)
    print(f"Dropped {dropped_negative_count} rows where r1r2 < 0")

    # Define the features we want to use
    # Add embedding dimensions to numerical features if they exist
    embedding_features = []
    if 'polytype_emb_1' in df.columns:
        embedding_features.extend(['polytype_emb_1', 'polytype_emb_2'])
    if 'method_emb_1' in df.columns:
        embedding_features.extend(['method_emb_1', 'method_emb_2'])

    numerical_features = ['temperature', 'ip_corrected_1', 'ea_1', 'homo_1', 'lumo_1',
                          'global_electrophilicity_1', 'global_nucleophilicity_1',
                          'best_conformer_energy_1', 'charges_min_1', 'charges_max_1',
                          'charges_mean_1', 'fukui_electrophilicity_min_1',
                          'fukui_electrophilicity_max_1', 'fukui_electrophilicity_mean_1',
                          'fukui_nucleophilicity_min_1', 'fukui_nucleophilicity_max_1',
                          'fukui_nucleophilicity_mean_1', 'fukui_radical_min_1',
                          'fukui_radical_max_1', 'fukui_radical_mean_1', 'dipole_x_1',
                          'dipole_y_1', 'dipole_z_1', 'ip_corrected_2', 'ea_2', 'homo_2',
                          'lumo_2', 'global_electrophilicity_2', 'global_nucleophilicity_2',
                          'charges_min_2', 'charges_max_2', 'charges_mean_2',
                          'fukui_electrophilicity_min_2', 'fukui_electrophilicity_max_2',
                          'fukui_electrophilicity_mean_2', 'fukui_nucleophilicity_min_2',
                          'fukui_nucleophilicity_max_2', 'fukui_nucleophilicity_mean_2',
                          'fukui_radical_min_2', 'fukui_radical_max_2', 'fukui_radical_mean_2',
                          'dipole_x_2', 'dipole_y_2', 'dipole_z_2', 'solvent_logp'] + embedding_features

    # We will only use one-hot encoding for categorical features not represented as embeddings
    categorical_features = []
    if not embedding_features:
        # Only use categorical features if we don't have embeddings
        categorical_features = ['polymerization_type', 'method']

    # Filter to keep only columns that exist in the DataFrame
    existing_numerical = [col for col in numerical_features if col in df.columns]
    existing_categorical = [col for col in categorical_features if col in df.columns]

    # Print counts of missing values in model features before filtering
    model_features = existing_numerical + existing_categorical
    missing_in_features = df[model_features].isna().sum()
    for col, count in sorted(zip(missing_in_features.index, missing_in_features), key=lambda x: x[1], reverse=True):
        if count > 0:
            print(f"- {col}: {count} missing values")

    # Count rows with missing values in model features
    rows_with_missing = df[model_features].isna().any(axis=1).sum()
    print(
        f"\nRows with missing values in model features: {rows_with_missing} ({rows_with_missing / len(df) * 100:.2f}%)")

    # Only drop rows where model features are missing
    na_count_before = len(df)
    df = df.dropna(subset=model_features)
    na_count_after = na_count_before - len(df)
    print(f"Dropped {na_count_after} rows with NA values in model features")
    print(f"Remaining rows for modeling: {len(df)}")

    # If no data left, exit
    if len(df) == 0:
        print("No data left after filtering. Exiting model training.")
        return

    # List to store the actual feature names in the dataset
    actual_features = []

    # Process numerical features
    print("\nUsing these numerical features:")
    for feature in existing_numerical:
        actual_features.append(feature)
        print(f"- {feature}")

    # Process categorical features and one-hot encode them
    if existing_categorical:
        print("\nOne-hot encoding these categorical features:")
        for feature in existing_categorical:
            # Create dummies for this categorical feature
            dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            # Add dummy columns to features list
            actual_features.extend(dummies.columns.tolist())
            print(f"- {feature} (creates {len(dummies.columns)} dummy features)")

    # Check if we have enough features to train the model
    if len(actual_features) < 5:  # Arbitrary threshold
        print(f"Error: Only {len(actual_features)} features found. Not enough for a good model.")
        return

    print(f"\nFinal feature count for modeling: {len(actual_features)}")

    # Define transformer for these features
    transformer = ColumnTransformer([
        ('numerical', StandardScaler(), actual_features)
    ], remainder='drop')

    # Transform features
    X = transformer.fit_transform(df)

    # Transform target with PowerTransformer
    pt = PowerTransformer(method='yeo-johnson')
    y = pt.fit_transform(df['r1r2'].values.reshape(-1, 1)).ravel()

    # Parameter grid for RandomizedSearchCV - simplified for faster results
    param_grid = {
        'n_estimators': [100, 500, 1000, 5000],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': np.logspace(-3, 0, 10),
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': np.linspace(0, 1, 10),
        'reg_lambda': np.linspace(0, 1, 10)
    }

    # Create a reaction ID for grouping related data points
    if 'reaction_id' not in df.columns:
        print("Creating reaction IDs for proper data splitting...")
        # This creates a unique ID for each monomer pair, regardless of order
        df['reaction_id'] = df.apply(
            lambda row: "_".join(sorted([str(row['monomer1_smiles']), str(row['monomer2_smiles'])])),
            axis=1
        )

    # Create grouped K-Fold splits that keep monomer pairs together
    n_splits = 5
    kf_splits = create_grouped_kfold_splits(df, n_splits=n_splits, random_state=42, id_column='reaction_id')
    print(f"\nUsing {n_splits}-fold cross-validation with grouped splits (keeps flipped monomer pairs together)")

    fold_scores = []
    all_y_true = []
    all_y_pred = []
    all_y_train_true = []
    all_y_train_pred = []

    # Store all indices for later use in visualization
    all_test_indices = []
    all_train_indices = []

    for fold, (train_idx, test_idx) in enumerate(kf_splits, 1):
        print(f"Fold {fold}")

        # Store indices for later visualization
        all_train_indices.extend(train_idx)
        all_test_indices.extend(test_idx)

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        # Train model
        model = XGBRegressor(random_state=42)

        # Perform RandomizedSearchCV with fewer iterations for speed
        random_search = RandomizedSearchCV(
            model, param_distributions=param_grid,
            n_iter=10, cv=3, verbose=0, random_state=42, n_jobs=-1
        )
        random_search.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        # Get best model
        best_model = random_search.best_estimator_

        # Print the best hyperparameters
        print("\nBest Hyperparameters:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")

        # Optionally, you can also print the best score achieved during search
        print(f"Best CV score: {random_search.best_score_:.4f}")

        # Make predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        # Store values
        all_y_train_true.extend(y_train)
        all_y_train_pred.extend(y_pred_train)
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred_test)

        # Calculate metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        print(f"Train R2: {r2_train:.4f}")
        print(f"Test R2: {r2_test:.4f}")

        fold_scores.append((r2_train, r2_test))

    # Calculate average scores
    avg_train_r2 = np.mean([score[0] for score in fold_scores])
    avg_test_r2 = np.mean([score[1] for score in fold_scores])

    print(f"\nAverage Train R2: {avg_train_r2:.4f}")
    print(f"\nAverage Test R2: {avg_test_r2:.4f}")

    # Transform values back to original scale
    all_y_true_inv = pt.inverse_transform(np.array(all_y_true).reshape(-1, 1)).ravel()
    all_y_pred_inv = pt.inverse_transform(np.array(all_y_pred).reshape(-1, 1)).ravel()
    all_y_train_true_inv = pt.inverse_transform(np.array(all_y_train_true).reshape(-1, 1)).ravel()
    all_y_train_pred_inv = pt.inverse_transform(np.array(all_y_train_pred).reshape(-1, 1)).ravel()

    # Create arrays for the test and train data points
    test_data = np.column_stack((all_y_true_inv, all_y_pred_inv))
    train_data = np.column_stack((all_y_train_true_inv, all_y_train_pred_inv))

    # Prepare indices for visualization
    # First convert all_test_indices and all_train_indices to numpy arrays if they aren't already
    all_test_indices = np.array(all_test_indices)
    all_train_indices = np.array(all_train_indices)

    # Randomly select points if there are more than requested
    if len(test_data) > 100:
        # Set random seed for reproducibility
        np.random.seed(42)
        # Select 100 random indices without replacement
        selected_test_indices = np.random.choice(len(test_data), size=100, replace=False)
        selected_test_data = test_data[selected_test_indices]
        # Map back to original dataframe indices
        selected_orig_test_indices = all_test_indices[selected_test_indices]
    else:
        selected_test_data = test_data
        selected_orig_test_indices = all_test_indices

    if len(train_data) > 200:
        # Set random seed for reproducibility
        np.random.seed(43)  # Use different seed than test set
        # Select 200 random indices without replacement
        selected_train_indices = np.random.choice(len(train_data), size=200, replace=False)
        selected_train_data = train_data[selected_train_indices]
        # Map back to original dataframe indices
        selected_orig_train_indices = all_train_indices[selected_train_indices]
    else:
        selected_train_data = train_data
        selected_orig_train_indices = all_train_indices

    # Create static matplotlib plot (same as before)
    plt.figure(figsize=(10, 8))

    # Plot the training and test points with different colors and markers
    plt.scatter(selected_train_data[:, 0], selected_train_data[:, 1], alpha=0.5, color='blue',
                label='Training points (200 samples)', marker='o')
    plt.scatter(selected_test_data[:, 0], selected_test_data[:, 1], alpha=0.8, color='red',
                label='Test points (100 samples)', marker='x')

    # Add the diagonal line (perfect prediction)
    plt.plot([0, 5], [0, 5], color='green', linestyle='--', label='Perfect prediction')

    # Set axis limits from 0 to 5
    plt.xlim(0, 5)
    plt.ylim(0, 5)

    # Add gridlines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add labels and title
    plt.xlabel('True r_product', fontsize=12)
    plt.ylabel('Predicted r_product', fontsize=12)
    plt.title(f'XGBoost Model Performance (Test R² = {avg_test_r2:.4f})', fontsize=14)

    # Add legend
    plt.legend(loc='upper left')

    # Add annotation with R² value
    plt.annotate(f'R² = {avg_test_r2:.4f}', xy=(0.05, 0.92), xycoords='axes fraction',
                 fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Save figure with higher resolution and tighter layout
    plt.tight_layout()
    plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Create interactive plot if plotly is available
    if plotly_available:
        try:
            # Concatenate true and predicted values
            y_true_combined = np.concatenate([all_y_train_true_inv, all_y_true_inv])
            y_pred_combined = np.concatenate([all_y_train_pred_inv, all_y_pred_inv])

            # Create interactive visualization
            create_interactive_plot(df, y_true_combined, y_pred_combined,
                                    selected_orig_train_indices, selected_orig_test_indices,
                                    avg_test_r2)
        except Exception as e:
            print(f"Error creating interactive plot: {e}")
            print("Falling back to static plot only.")

    # Save model feature importances if we have a model
    try:
        # Train a final model on all data using the best hyperparameters from the last fold
        final_model = XGBRegressor(**random_search.best_params_, random_state=42)
        final_model.fit(X, y)

        # Get feature importances
        importances = final_model.feature_importances_

        # Create DataFrame for feature importances
        importance_df = pd.DataFrame({
            'Feature': actual_features,
            'Importance': importances
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)

        # Save to CSV
        importance_df.to_csv('feature_importances.csv', index=False)

        # Plot feature importances for top 20 features
        n_features = min(20, len(actual_features))
        plt.figure(figsize=(12, 8))
        plt.bar(range(n_features), importance_df['Importance'].head(n_features))
        plt.xticks(range(n_features), importance_df['Feature'].head(n_features), rotation=90)
        plt.title('Top Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importances.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Also create interactive feature importance plot
        if plotly_available:
            try:
                fig = go.Figure()

                # Get top 20 features
                top_features = importance_df.head(n_features)

                # Add bar chart
                fig.add_trace(go.Bar(
                    x=top_features['Feature'],
                    y=top_features['Importance'],
                    marker_color='darkblue',
                    hovertemplate='<b>%{x}</b><br>Importance: %{y:.4f}<extra></extra>'
                ))

                # Update layout
                fig.update_layout(
                    title='Top Feature Importances',
                    xaxis_title='Feature',
                    yaxis_title='Importance',
                    xaxis_tickangle=-90,
                    plot_bgcolor='white',
                    width=1000,
                    height=600
                )

                # Add grid lines
                fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

                # Save the interactive plot
                fig.write_html('feature_importances_interactive.html')
                print("Interactive feature importance plot saved as 'feature_importances_interactive.html'")
            except Exception as e:
                print(f"Error creating interactive feature importance plot: {e}")
    except Exception as e:
        print(f"Could not create feature importance plot: {e}")

    print("Model training completed.")


def evaluate_learning_curve(df):
    """
    Evaluate model performance with different dataset sizes and plot the learning curve.
    This helps understand the impact of dataset size on model performance.
    Uses reaction-based splitting to prevent data leakage.

    Args:
        df: DataFrame with features and target
    """
    print("\n=== Learning Curve Analysis ===")

    # Create the target variable
    df['r1r2'] = df['constant_1'] * df['constant_2']

    # Drop rows where r1r2 is less than 0
    df = df[df['r1r2'] >= 0]
    print(f"Total valid datapoints: {len(df)}")

    # Create a reaction ID for grouping related data points
    if 'reaction_id' not in df.columns:
        print("Creating reaction IDs for proper data splitting...")
        # This creates a unique ID for each monomer pair, regardless of order
        df['reaction_id'] = df.apply(
            lambda row: "_".join(sorted([str(row['monomer1_smiles']), str(row['monomer2_smiles'])])),
            axis=1
        )

    # Count unique reactions
    unique_reactions = df['reaction_id'].nunique()
    print(f"Number of unique reactions: {unique_reactions}")

    # Define the sample sizes to test
    # Ensure the sizes don't exceed the actual data size
    total_size = len(df)
    sample_sizes = [
        min(100, total_size),
        min(250, total_size),
        min(500, total_size),
        min(1000, total_size),
        min(2000, total_size),
        total_size
    ]
    # Remove duplicates and ensure the list is sorted
    sample_sizes = sorted(list(set(sample_sizes)))

    print(f"Testing model with sample sizes: {sample_sizes}")

    # Define the features - similar to train_model function
    embedding_features = []
    if 'polytype_emb_1' in df.columns:
        embedding_features.extend(['polytype_emb_1', 'polytype_emb_2'])
    if 'method_emb_1' in df.columns:
        embedding_features.extend(['method_emb_1', 'method_emb_2'])

    numerical_features = ['temperature', 'ip_corrected_1', 'ea_1', 'homo_1', 'lumo_1',
                          'global_electrophilicity_1', 'global_nucleophilicity_1',
                          'best_conformer_energy_1', 'charges_min_1', 'charges_max_1',
                          'charges_mean_1', 'fukui_electrophilicity_min_1',
                          'fukui_electrophilicity_max_1', 'fukui_electrophilicity_mean_1',
                          'fukui_nucleophilicity_min_1', 'fukui_nucleophilicity_max_1',
                          'fukui_nucleophilicity_mean_1', 'fukui_radical_min_1',
                          'fukui_radical_max_1', 'fukui_radical_mean_1', 'dipole_x_1',
                          'dipole_y_1', 'dipole_z_1', 'ip_corrected_2', 'ea_2', 'homo_2',
                          'lumo_2', 'global_electrophilicity_2', 'global_nucleophilicity_2',
                          'charges_min_2', 'charges_max_2', 'charges_mean_2',
                          'fukui_electrophilicity_min_2', 'fukui_electrophilicity_max_2',
                          'fukui_electrophilicity_mean_2', 'fukui_nucleophilicity_min_2',
                          'fukui_nucleophilicity_max_2', 'fukui_nucleophilicity_mean_2',
                          'fukui_radical_min_2', 'fukui_radical_max_2', 'fukui_radical_mean_2',
                          'dipole_x_2', 'dipole_y_2', 'dipole_z_2', 'solvent_logp'] + embedding_features

    categorical_features = []
    if not embedding_features:
        categorical_features = ['polymerization_type', 'method']

    # Filter to keep only columns that exist in the DataFrame
    existing_numerical = [col for col in numerical_features if col in df.columns]
    existing_categorical = [col for col in categorical_features if col in df.columns]

    # Handle missing values
    model_features = existing_numerical + existing_categorical
    df = df.dropna(subset=model_features)
    print(f"Total datapoints after removing rows with missing values: {len(df)}")

    # Results storage
    results = {
        'sample_size': [],
        'train_r2': [],
        'test_r2': [],
        'train_std': [],
        'test_std': []
    }

    # For each sample size
    for size in sample_sizes:
        print(f"\nEvaluating model with {size} samples")

        # Sample by reaction ID to ensure proper grouping
        if size < len(df):
            # Sample unique reaction IDs
            unique_ids = df['reaction_id'].unique()
            np.random.seed(42)  # For reproducibility

            # Estimate how many reaction IDs we need to get approximately 'size' rows
            avg_rows_per_id = len(df) / len(unique_ids)
            n_ids_needed = min(int(size / avg_rows_per_id * 1.1), len(unique_ids))  # Add 10% buffer

            # Sample the reaction IDs
            sampled_ids = np.random.choice(unique_ids, size=n_ids_needed, replace=False)

            # Get rows for these reaction IDs
            sampled_df = df[df['reaction_id'].isin(sampled_ids)]

            # If we have too many rows, randomly subsample
            if len(sampled_df) > size:
                sampled_df = sampled_df.sample(n=size, random_state=42)
        else:
            sampled_df = df.copy()

        print(f"Actual sample size: {len(sampled_df)}")

        # Process the data (similar to train_model)
        # List to store feature names
        actual_features = []

        # Process numerical features
        for feature in existing_numerical:
            actual_features.append(feature)

        # Process categorical features with one-hot encoding
        for feature in existing_categorical:
            dummies = pd.get_dummies(sampled_df[feature], prefix=feature, drop_first=False)
            sampled_df = pd.concat([sampled_df, dummies], axis=1)
            actual_features.extend(dummies.columns.tolist())

        # Define transformer for features
        transformer = ColumnTransformer([
            ('numerical', StandardScaler(), actual_features)
        ], remainder='drop')

        # Transform features
        X = transformer.fit_transform(sampled_df)

        # Transform target with PowerTransformer
        pt = PowerTransformer(method='yeo-johnson')
        y = pt.fit_transform(sampled_df['r1r2'].values.reshape(-1, 1)).ravel()

        # Create reaction-based cross-validation splits
        n_splits = 5
        fold_train_scores = []
        fold_test_scores = []

        # Create grouped k-fold splits based on reaction_id
        grouped_kf = create_grouped_kfold_splits(sampled_df, n_splits=n_splits,
                                                 random_state=42, id_column='reaction_id')

        for fold, (train_idx, test_idx) in enumerate(grouped_kf, 1):
            # Ensure we have enough data in both train and test sets
            if len(train_idx) < 10 or len(test_idx) < 10:
                print(f"Warning: Fold {fold} has insufficient data, skipping")
                continue

            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            print(f"Fold {fold}: Train size = {len(X_train)}, Test size = {len(X_test)}")

            # Train model - using fixed parameters for speed
            model = XGBRegressor(
                n_estimators=200,  # Reduced from 5000 for speed
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )

            # Fit model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            # Calculate metrics
            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)

            fold_train_scores.append(r2_train)
            fold_test_scores.append(r2_test)

            print(f"  Fold {fold} Train R²: {r2_train:.4f}")
            print(f"  Fold {fold} Test R²: {r2_test:.4f}")

        # Skip this sample size if we couldn't compute any valid scores
        if len(fold_train_scores) == 0 or len(fold_test_scores) == 0:
            print(f"Skipping sample size {size} - insufficient data for valid cross-validation")
            continue

        # Average and std of scores
        avg_train_r2 = np.mean(fold_train_scores)
        avg_test_r2 = np.mean(fold_test_scores)
        std_train_r2 = np.std(fold_train_scores)
        std_test_r2 = np.std(fold_test_scores)

        print(f"Sample size: {size}")
        print(f"Average Train R²: {avg_train_r2:.4f} ± {std_train_r2:.4f}")
        print(f"Average Test R²: {avg_test_r2:.4f} ± {std_test_r2:.4f}")

        # Store results
        results['sample_size'].append(size)
        results['train_r2'].append(avg_train_r2)
        results['test_r2'].append(avg_test_r2)
        results['train_std'].append(std_train_r2)
        results['test_std'].append(std_test_r2)

    # Plot learning curve
    plt.figure(figsize=(12, 8))

    # Plot the scores
    plt.plot(results['sample_size'], results['train_r2'], 'o-', color='blue',
             label='Training score')
    plt.plot(results['sample_size'], results['test_r2'], 'o-', color='red',
             label='Cross-validation score')

    # Add error bands
    plt.fill_between(results['sample_size'],
                     np.array(results['train_r2']) - np.array(results['train_std']),
                     np.array(results['train_r2']) + np.array(results['train_std']),
                     alpha=0.1, color='blue')
    plt.fill_between(results['sample_size'],
                     np.array(results['test_r2']) - np.array(results['test_std']),
                     np.array(results['test_r2']) + np.array(results['test_std']),
                     alpha=0.1, color='red')

    # Add a table with the exact values
    table_data = []
    for i, size in enumerate(results['sample_size']):
        table_data.append([
            size,
            f"{results['train_r2'][i]:.4f} ± {results['train_std'][i]:.4f}",
            f"{results['test_r2'][i]:.4f} ± {results['test_std'][i]:.4f}"
        ])

    plt.table(
        cellText=table_data,
        colLabels=['Sample Size', 'Train R²', 'Test R²'],
        cellLoc='center',
        loc='lower center',
        bbox=[0.2, -0.4, 0.6, 0.2]
    )

    # Add grid lines for better readability
    plt.grid(True, linestyle='--', alpha=0.7)

    # Set axis limits
    plt.ylim(0, 1.0)
    plt.xlim(min(results['sample_size']) - 50, max(results['sample_size']) + 50)

    # Add labels and title
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Learning Curve: Model Performance vs. Training Set Size', fontsize=14)

    # Add legend
    plt.legend(loc='lower right')

    # Save with extra space for the table
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)
    plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('learning_curve_results.csv', index=False)

    print("\nLearning curve analysis completed and saved to learning_curve.png")
    return results_df


def create_interactive_plot(df, y_true_inv, y_pred_inv, train_indices, test_indices, avg_test_r2):
    """
    Create an interactive scatter plot of true vs predicted values with hover information.

    Args:
        df: Original DataFrame with all features
        y_true_inv: True values (inverse transformed)
        y_pred_inv: Predicted values (inverse transformed)
        train_indices: Indices of training data points to plot
        test_indices: Indices of test data points to plot
        avg_test_r2: Average R² score for test data

    Returns:
        None (saves the plot as an HTML file)
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": False}]])

    # Get original feature values for hover text
    hover_data = df.iloc[np.concatenate([train_indices, test_indices])]

    # Format hover text for training data
    train_hover_texts = []
    for idx in train_indices:
        row = df.iloc[idx]
        hover_text = f"<b>Monomer 1:</b> {row.get('monomer1_name', 'N/A')}<br>" + \
                     f"<b>Monomer 2:</b> {row.get('monomer2_name', 'N/A')}<br>" + \
                     f"<b>Temperature:</b> {row.get('temperature', 'N/A')} K<br>" + \
                     f"<b>Solvent:</b> {row.get('solvent', 'N/A')}<br>" + \
                     f"<b>Method:</b> {row.get('method', 'N/A')}<br>" + \
                     f"<b>Polymerization type:</b> {row.get('polymerization_type', 'N/A')}<br>" + \
                     f"<b>True Value:</b> {y_true_inv[idx]:.3f}<br>" + \
                     f"<b>Prediction:</b> {y_pred_inv[idx]:.3f}"
        train_hover_texts.append(hover_text)

    # Format hover text for test data
    test_hover_texts = []
    for idx in test_indices:
        row = df.iloc[idx]
        hover_text = f"<b>Monomer 1:</b> {row.get('monomer1_name', 'N/A')}<br>" + \
                     f"<b>Monomer 2:</b> {row.get('monomer2_name', 'N/A')}<br>" + \
                     f"<b>Temperature:</b> {row.get('temperature', 'N/A')} K<br>" + \
                     f"<b>Solvent:</b> {row.get('solvent', 'N/A')}<br>" + \
                     f"<b>Method:</b> {row.get('method', 'N/A')}<br>" + \
                     f"<b>Polymerization type:</b> {row.get('polymerization_type', 'N/A')}<br>" + \
                     f"<b>True Value:</b> {y_true_inv[idx]:.3f}<br>" + \
                     f"<b>Prediction:</b> {y_pred_inv[idx]:.3f}"
        test_hover_texts.append(hover_text)

    # Add scatter plots for training and test data
    fig.add_trace(
        go.Scatter(
            x=y_true_inv[train_indices],
            y=y_pred_inv[train_indices],
            mode='markers',
            marker=dict(
                color='blue',
                size=10,
                opacity=0.5
            ),
            text=train_hover_texts,
            hoverinfo='text',
            name='Training Points'
        )
    )

    fig.add_trace(
        go.Scatter(
            x=y_true_inv[test_indices],
            y=y_pred_inv[test_indices],
            mode='markers',
            marker=dict(
                color='red',
                size=10,
                symbol='x',
                opacity=0.8
            ),
            text=test_hover_texts,
            hoverinfo='text',
            name='Test Points'
        )
    )

    # Add the perfect prediction diagonal line
    fig.add_trace(
        go.Scatter(
            x=[0, 5],
            y=[0, 5],
            mode='lines',
            line=dict(color='green', width=2, dash='dash'),
            name='Perfect Prediction'
        )
    )

    # Update layout
    fig.update_layout(
        title=f'XGBoost Model Performance (Test R² = {avg_test_r2:.4f})',
        xaxis_title='True r_product',
        yaxis_title='Predicted r_product',
        xaxis=dict(range=[0, 5]),
        yaxis=dict(range=[0, 5]),
        legend=dict(
            x=0.01,
            y=0.99,
            bgcolor='rgba(255, 255, 255, 0.5)'
        ),
        hovermode='closest',
        plot_bgcolor='white',
        width=900,
        height=700
    )

    # Add grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

    # Add annotation with R² value
    fig.add_annotation(
        x=0.15,
        y=0.95,
        xref='paper',
        yref='paper',
        text=f'R² = {avg_test_r2:.4f}',
        showarrow=False,
        font=dict(size=14),
        bgcolor='white',
        bordercolor='black',
        borderwidth=1,
        borderpad=4
    )

    # Save the interactive plot as an HTML file
    fig.write_html('interactive_model_performance.html')
    print("Interactive plot saved as 'interactive_model_performance.html'")

    # Also save as static image for reports
    fig.write_image('model_performance_interactive.png', scale=2)

    return fig


if __name__ == "__main__":
    main()