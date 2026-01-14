import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import joblib
import json
import os

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "experiments", "data", "train.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "experiments", "feature_comparison", "baseline", "results_final", "model.joblib")
META_PATH = os.path.join(PROJECT_ROOT, "experiments", "feature_comparison", "baseline", "results_final", "meta.json")

# Case study parameters
TEMPERATURE = 90.0
POLYMERIZATION_TYPE = "free radical"
METHOD = "solvent"
MONOMER1_NAME = "Acrylamide"
MONOMER1_SMILES = "C=CC(N)=O"
MONOMER2_NAME = "styrene"
MONOMER2_SMILES = "C=Cc1ccccc1"

# Solvents from the original case study
solvents = [
    "Benzene",
    "o-Dichlorobenzene",
    "Benzonitrile",
    "Dioxane",
    "Bis(2-methoxyethyl) ether",
    "Ethanol",
    "2-(2-Methoxyethoxy)ethanol",
    "DMSO",
    "Methanol",
]

# Minsk et al. data from the original script
minsk_classes = [1, 1, 1, 1, 1, 0, 0, 0, 0]
model_classes = [1, 1, 1, 1, 0, 0, 0, 1, 0]
model_confidence = [0.609, 0.686, 0.395, 0.430, 0.583, 0.493, 0.760, 0.367, 0.569]

# Solvent SMILES mapping (common solvents)
SOLVENT_SMILES = {
    "Benzene": "c1ccccc1",
    "o-Dichlorobenzene": "c1ccc(c(c1)Cl)Cl",
    "Benzonitrile": "C(#N)c1ccccc1",
    "Dioxane": "C1COCCO1",
    "Bis(2-methoxyethyl) ether": "COCCOCCOC",
    "Ethanol": "CCO",
    "2-(2-Methoxyethoxy)ethanol": "COCCOCCO",
    "DMSO": "CS(=O)C",
    "Methanol": "CO",
}

CLASS_COLORS = {
    0: "#3A3B73",  # < 1
    1: "#e27f07",  # 1–25
    2: "#6a040f",  # > 25
}


def load_model_and_features():
    """Load the trained model and feature names."""
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, 'r') as f:
        meta = json.load(f)
    features = meta['feature_columns']
    return model, features


def prepare_case_study_data(df_train, features):
    """
    Prepare case study data for the 9 solvents with Acrylamide + Styrene.
    Uses the actual monomer features from the training dataset.
    """
    # Find rows in training data with these monomers
    # Try different case variations
    monomer1_variants = ['acrylamide', 'Acrylamide', 'ACRYLAMIDE']
    monomer2_variants = ['styrene', 'Styrene', 'STYRENE']
    
    mask = (
        (df_train['monomer1_name'].isin(monomer1_variants) & df_train['monomer2_name'].isin(monomer2_variants)) |
        (df_train['monomer2_name'].isin(monomer1_variants) & df_train['monomer1_name'].isin(monomer2_variants))
    )
    
    monomer_rows = df_train[mask]
    
    if len(monomer_rows) == 0:
        # Try with SMILES instead
        mask = (
            (df_train['monomer1_smiles'] == MONOMER1_SMILES) & (df_train['monomer2_smiles'] == MONOMER2_SMILES) |
            (df_train['monomer2_smiles'] == MONOMER1_SMILES) & (df_train['monomer1_smiles'] == MONOMER2_SMILES)
        )
        monomer_rows = df_train[mask]
    
    if len(monomer_rows) == 0:
        raise ValueError(f"No training data found for {MONOMER1_NAME} + {MONOMER2_NAME}")
    
    # Take the first row as template for monomer features
    template = monomer_rows.iloc[0]
    
    # Check if monomers are in the right order, if not swap the features
    swap_needed = False
    if 'monomer1_name' in template.index:
        if template['monomer1_name'].lower() != MONOMER1_NAME.lower():
            swap_needed = True
    
    # Create case study dataframe
    case_study_rows = []
    
    for solvent_name in solvents:
        # Find solvent features in training data
        solvent_matches = df_train[df_train['solvent'].str.lower() == solvent_name.lower()]
        
        if len(solvent_matches) == 0:
            # Try with SMILES
            if solvent_name in SOLVENT_SMILES:
                solvent_smiles = SOLVENT_SMILES[solvent_name]
                solvent_matches = df_train[df_train['solvent_smiles'] == solvent_smiles]
        
        if len(solvent_matches) > 0:
            solvent_row = solvent_matches.iloc[0]
        else:
            # Use default solvent features (all zeros except temperature)
            print(f"Warning: No solvent data found for {solvent_name}, using default values")
            solvent_row = None
        
        # Build feature dictionary
        row_data = {}
        
        # Copy monomer features from template
        for feat in features:
            if feat.endswith('_1') or feat.endswith('_2'):
                if swap_needed and feat.endswith('_1'):
                    # Swap monomer 1 and 2 features
                    swapped_feat = feat[:-1] + '2'
                    row_data[feat] = template.get(swapped_feat, 0)
                elif swap_needed and feat.endswith('_2'):
                    swapped_feat = feat[:-1] + '1'
                    row_data[feat] = template.get(swapped_feat, 0)
                else:
                    row_data[feat] = template.get(feat, 0)
            elif feat == 'temperature':
                row_data[feat] = TEMPERATURE
            elif feat.startswith('solvent_'):
                if solvent_row is not None and feat in solvent_row.index:
                    row_data[feat] = solvent_row[feat]
                else:
                    row_data[feat] = 0  # Default value
            elif feat in ['polytype_emb_1', 'polytype_emb_2']:
                row_data[feat] = template.get(feat, 0)
            elif feat in ['method_emb_1', 'method_emb_2']:
                row_data[feat] = template.get(feat, 0)
            else:
                row_data[feat] = template.get(feat, 0)
        
        case_study_rows.append(row_data)
    
    df_case = pd.DataFrame(case_study_rows)
    return df_case


def main():
    print("Loading training data...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    print(f"Training data shape: {df_train.shape}")
    
    print("\nLoading model and features...")
    model, features = load_model_and_features()
    print(f"Features: {features}")
    
    print("\nPreparing case study data...")
    df_case = prepare_case_study_data(df_train, features)
    print(f"Case study data shape: {df_case.shape}")
    
    # Prepare training data with same features
    print("\nPreparing training data features...")
    df_train_clean = df_train.dropna(subset=features)
    X_train = df_train_clean[features].values
    print(f"Training data (clean) shape: {X_train.shape}")
    
    X_case = df_case[features].values
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_case_scaled = scaler.transform(X_case)
    
    # Apply PCA
    print("\nApplying PCA...")
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_case_pca = pca.transform(X_case_scaled)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.3f}")
    
    # Find 100 nearest neighbors for case study points
    print("\nFinding nearest neighbors...")
    nn = NearestNeighbors(n_neighbors=100, metric='euclidean')
    nn.fit(X_train_pca)
    
    # Get all neighbors for all case study points
    all_neighbor_indices = set()
    for i in range(len(X_case_pca)):
        distances, indices = nn.kneighbors([X_case_pca[i]])
        all_neighbor_indices.update(indices[0])
    
    neighbor_indices = list(all_neighbor_indices)
    print(f"Found {len(neighbor_indices)} unique nearest neighbors")
    
    # Save nearest neighbors to CSV
    print("\nSaving nearest neighbors data...")
    neighbor_data = df_train_clean.iloc[neighbor_indices].copy()
    neighbor_data['pca_component_1'] = X_train_pca[neighbor_indices, 0]
    neighbor_data['pca_component_2'] = X_train_pca[neighbor_indices, 1]
    
    # Add columns to show which case study point(s) this neighbor is near
    for i, solvent in enumerate(solvents):
        distances, indices = nn.kneighbors([X_case_pca[i]])
        neighbor_data[f'near_{solvent.replace(" ", "_")}'] = False
        neighbor_data.loc[neighbor_data.index[neighbor_data.index.isin(df_train_clean.iloc[indices[0]].index)], f'near_{solvent.replace(" ", "_")}'] = True
        
        # Also add distance to each case study point
        all_distances = np.linalg.norm(X_train_pca[neighbor_indices] - X_case_pca[i], axis=1)
        neighbor_data[f'distance_to_{solvent.replace(" ", "_")}'] = all_distances
    
    # Select important columns for the CSV
    important_cols = ['monomer1_name', 'monomer2_name', 'solvent', 'temperature', 
                      'polymerization_type', 'method', 'r1r2', 'r_product_class',
                      'pca_component_1', 'pca_component_2']
    
    # Add the near_* and distance_* columns
    near_cols = [col for col in neighbor_data.columns if col.startswith('near_') or col.startswith('distance_')]
    
    # Filter to only existing columns
    export_cols = [col for col in important_cols if col in neighbor_data.columns] + near_cols
    
    neighbor_csv_path = os.path.join(os.path.dirname(__file__), "nearest_neighbors.csv")
    neighbor_data[export_cols].to_csv(neighbor_csv_path, index=False)
    print(f"Nearest neighbors saved to: {neighbor_csv_path}")
    print(f"Columns: {export_cols}")
    
    # Save case study features to CSV
    print("\nSaving case study features...")
    case_study_features = df_case.copy()
    case_study_features['solvent_name'] = solvents
    case_study_features['minsk_class'] = minsk_classes
    case_study_features['model_class'] = model_classes
    case_study_features['model_confidence'] = model_confidence
    case_study_features['agreement'] = [minsk_classes[i] == model_classes[i] for i in range(len(solvents))]
    case_study_features['pca_component_1'] = X_case_pca[:, 0]
    case_study_features['pca_component_2'] = X_case_pca[:, 1]
    
    # Reorder columns: identifiers first, then all features, then PCA components
    id_cols = ['solvent_name', 'minsk_class', 'model_class', 'model_confidence', 'agreement']
    pca_cols = ['pca_component_1', 'pca_component_2']
    feature_cols_ordered = id_cols + features + pca_cols
    
    case_csv_path = os.path.join(os.path.dirname(__file__), "case_study_features.csv")
    case_study_features[feature_cols_ordered].to_csv(case_csv_path, index=False)
    print(f"Case study features saved to: {case_csv_path}")
    
    print(f"\n{'='*60}")
    print("FEATURES USED IN PCA (from trained model):")
    print(f"{'='*60}")
    for i, feat in enumerate(features, 1):
        print(f"  {i:2d}. {feat}")
    print(f"{'='*60}")
    print(f"Total: {len(features)} features")
    print(f"These are the EXACT features used in model training.")
    print(f"{'='*60}\n")
    
    # Create plot
    print("\nCreating PCA plot...")
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot training neighbors in gray
    ax.scatter(
        X_train_pca[neighbor_indices, 0],
        X_train_pca[neighbor_indices, 1],
        c='lightgray',
        alpha=0.3,
        s=20,
        label='100 nearest training samples',
        zorder=1
    )
    
    # Determine colors for case study points
    colors = []
    for i in range(len(solvents)):
        if minsk_classes[i] == model_classes[i]:
            colors.append('green')
        else:
            colors.append('red')
    
    # Plot case study points
    for i in range(len(solvents)):
        ax.scatter(
            X_case_pca[i, 0],
            X_case_pca[i, 1],
            c=colors[i],
            s=200,
            alpha=0.8,
            edgecolors='black',
            linewidth=2,
            zorder=3
        )
        
        # Add solvent labels
        ax.annotate(
            solvents[i],
            (X_case_pca[i, 0], X_case_pca[i, 1]),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3),
            zorder=4
        )
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.8, label='Model matches Minsk et al.'),
        Patch(facecolor='red', alpha=0.8, label='Model disagrees with Minsk et al.'),
        Patch(facecolor='lightgray', alpha=0.3, label='100 nearest training samples'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True)
    
    ax.set_xlabel('PC1', fontsize=13)
    ax.set_ylabel('PC2', fontsize=13)
    ax.set_title(
        f'PCA Visualization: {MONOMER1_NAME} + {MONOMER2_NAME}\n'
        f'({TEMPERATURE}°C, {POLYMERIZATION_TYPE}, {METHOD})',
        fontsize=14,
        fontweight='bold'
    )
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(os.path.dirname(__file__), "solvent_case_study_pca.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    
    output_path_pdf = os.path.join(os.path.dirname(__file__), "solvent_case_study_pca.pdf")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    print(f"Plot saved to: {output_path_pdf}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total case study points: {len(solvents)}")
    print(f"Matching predictions: {sum([1 for i in range(len(solvents)) if minsk_classes[i] == model_classes[i]])}")
    print(f"Disagreeing predictions: {sum([1 for i in range(len(solvents)) if minsk_classes[i] != model_classes[i]])}")
    print("\nDisagreements:")
    for i in range(len(solvents)):
        if minsk_classes[i] != model_classes[i]:
            print(f"  - {solvents[i]}: Minsk={minsk_classes[i]}, Model={model_classes[i]} (conf={model_confidence[i]:.3f})")
    print("="*60)


if __name__ == "__main__":
    main()

