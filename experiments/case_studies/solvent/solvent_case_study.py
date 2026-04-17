"""
Solvent Case Study Plot

This script creates a plot showing:
- X-axis: Discrete positions for each case study solvent (ordered by minimum similarity)
- Y-axis: Monomer Tanimoto similarity (continuous)
- Background: Nearest neighbor points from dataset with class colors at low opacity
- Case study points: colored by predicted class, shape by correctness
- Three molecule boxes showing examples at different similarity levels
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Draw
from sklearn.neighbors import NearestNeighbors
import os
import random
import sys
from pathlib import Path

# Add paths to import copolpredictor and copol_prediction as package
# PROJECT_ROOT: go up 4 levels from experiments/case_studies/solvent/solvent_case_study.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# So that `import copolpredictor` and `import copol_prediction` work
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(PROJECT_ROOT) / "src"))

from copolpredictor.inference import CopolymerPredictor

# Paths (use final model bundle = newest model)
# Training data: use the same split as the main model bundle
TRAIN_DATA_PATH = os.path.join(
    PROJECT_ROOT, "copol_prediction", "artifacts", "data_splits", "train.csv"
)
CASE_STUDY_PATH = os.path.join(os.path.dirname(__file__), "case_study_features.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "copol_prediction", "artifacts", "model_bundle")

# Case study parameters
MONOMER1_SMILES = "C=CC(N)=O"  # Acrylamide
MONOMER2_SMILES = "C=Cc1ccccc1"  # Styrene

# Solvents from the case study
CASE_SOLVENTS = [
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

# Marker shapes for classes
CLASS_MARKERS = {
    0: 'o',  # Circle for class 0
    1: 's',  # Square for class 1
    2: '^',  # Triangle for class 2
}

# Database points color
DATASET_COLOR = "#808080"  # Gray for database points

# Number of nearest neighbors per case study point
N_NEIGHBORS_PER_POINT = 50

# Number of molecular structures to display
N_MOLECULE_BOXES = 15

# Monomer similarity bins
MONOMER_BINS = [0.0, 0.3, 0.5, 0.7, 0.85, 1.0]
MONOMER_BIN_LABELS = ['0.0-0.3', '0.3-0.5', '0.5-0.7', '0.7-0.85', '0.85-1.0']


def calculate_tanimoto_similarity(smiles1: str, smiles2: str, radius: int = 2, n_bits: int = 2048) -> float:
    """Calculate Tanimoto similarity between two molecules using Morgan fingerprints."""
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if mol1 is None or mol2 is None:
            return 0.0
        
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=n_bits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=n_bits)
        
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception as e:
        print(f"Warning: Could not calculate Tanimoto similarity: {e}")
        return 0.0


def calculate_monomer_similarity(mon1_smiles: str, mon2_smiles: str, 
                                  query_mon1: str, query_mon2: str) -> float:
    """Calculate monomer similarity considering both orientations."""
    # Direct orientation
    sim_direct = (
        calculate_tanimoto_similarity(mon1_smiles, query_mon1) +
        calculate_tanimoto_similarity(mon2_smiles, query_mon2)
    ) / 2.0
    
    # Flipped orientation
    sim_flipped = (
        calculate_tanimoto_similarity(mon1_smiles, query_mon2) +
        calculate_tanimoto_similarity(mon2_smiles, query_mon1)
    ) / 2.0
    
    return max(sim_direct, sim_flipped)


def draw_molecule_structure(smiles: str, size=(100, 100)):
    """Draw a molecule structure from SMILES and return as numpy array."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Generate 2D coordinates
        AllChem.Compute2DCoords(mol)
        
        # Draw molecule
        img = Draw.MolToImage(mol, size=size, kekulize=True)
        return np.array(img)
    except Exception as e:
        print(f"Warning: Could not draw molecule structure: {e}")
        return None


def calculate_solvent_similarity_to_set(target_solvent_smiles: str, solvent_set_smiles: list) -> float:
    """Calculate maximum similarity to a set of solvents."""
    similarities = []
    for ref_solvent in solvent_set_smiles:
        sim = calculate_tanimoto_similarity(target_solvent_smiles, ref_solvent)
        similarities.append(sim)
    return max(similarities) if similarities else 0.0


def assign_to_bin(value: float, bins: list) -> int:
    """Assign a value to a bin index."""
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return i
    return len(bins) - 2  # Last bin


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Solvent case study plot")
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help=f"Path to model bundle (default: {MODEL_PATH})"
    )
    args = parser.parse_args()
    
    # Store original solvent list (before reordering)
    original_case_solvents = CASE_SOLVENTS.copy()
    
    print("Loading training data...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    print(f"Training data shape: {df_train.shape}")
    
    print("\nLoading case study features...")
    df_case = pd.read_csv(CASE_STUDY_PATH)
    print(f"Case study data shape: {df_case.shape}")
    
    # Extract case study info
    case_minsk_classes = df_case['minsk_class'].tolist()
    # Optional: whether Final Model agrees with Lookup (voting definition)
    # If missing, treat all as agreement=True.
    case_agreement = (
        df_case["agreement"].tolist()
        if "agreement" in df_case.columns
        else [True] * len(df_case)
    )
    
    # Load model and make predictions
    print(f"\nLoading model from {args.model_path}...")
    try:
        predictor = CopolymerPredictor(args.model_path)
        print(f"  ✓ Model loaded ({len(predictor.features)} features)")
        
        # Prepare features for prediction
        # Get required features from model
        required_features = predictor.features
        
        # Check which features are available in df_case
        missing_features = set(required_features) - set(df_case.columns)
        if missing_features:
            print(f"  ⚠️  Warning: Missing features: {missing_features}")
            print("   Attempting to continue with available features...")
            available_features = [f for f in required_features if f in df_case.columns]
            if len(available_features) < len(required_features) * 0.8:
                raise ValueError(f"Too many missing features: {len(missing_features)} missing")
            required_features = available_features
        
        # Prepare feature matrix
        X_case = df_case[required_features].copy()
        
        # Fill NaN values if any
        if X_case.isna().sum().sum() > 0:
            print("  ⚠️  Warning: Found NaN values, filling with 0")
            X_case = X_case.fillna(0)
        
        # Make predictions
        print("  🔮 Making predictions...")
        y_pred = predictor.predict(X_case)
        case_model_classes = y_pred.tolist()
        
        print(f"  ✓ Predictions made: {case_model_classes}")
        
    except Exception as e:
        print(f"  ✗ Error loading model or making predictions: {e}")
        print("  Falling back to predictions from CSV...")
        import traceback
        traceback.print_exc()
        case_model_classes = df_case['model_class'].tolist()
    
    print(f"\nCase study solvents: {original_case_solvents}")
    
    # Filter training data
    print(f"\nPreparing training data...")
    df_train_clean = df_train[
        df_train['monomer1_smiles'].notna() & 
        df_train['monomer2_smiles'].notna() & 
        df_train['solvent_smiles'].notna() &
        df_train['r_product_class'].notna()
    ].copy()
    
    print(f"Clean training data shape: {df_train_clean.shape}")
    
    # Get all case study solvent SMILES
    case_solvent_smiles_list = [SOLVENT_SMILES[s] for s in original_case_solvents if s in SOLVENT_SMILES]
    
    # Calculate similarities for ALL training points
    print("\nCalculating similarities for all training points...")
    all_monomer_sims = []
    all_solvent_sims = []
    all_classes = []
    
    for idx, row in df_train_clean.iterrows():
        # Monomer similarity to case study monomers
        mon_sim = calculate_monomer_similarity(
            row['monomer1_smiles'],
            row['monomer2_smiles'],
            MONOMER1_SMILES,
            MONOMER2_SMILES
        )
        all_monomer_sims.append(mon_sim)
        
        # Solvent similarity: max similarity to any case study solvent
        sol_sim = calculate_solvent_similarity_to_set(
            row['solvent_smiles'],
            case_solvent_smiles_list
        )
        all_solvent_sims.append(sol_sim)
        all_classes.append(int(row['r_product_class']))
    
    all_monomer_sims = np.array(all_monomer_sims)
    all_solvent_sims = np.array(all_solvent_sims)
    
    # Assign each training point to the most similar case study solvent
    print("\nAssigning training points to most similar case study solvents...")
    all_solvent_assignments = []
    
    for idx, row in df_train_clean.iterrows():
        solvent_similarities = []
        for case_solvent in original_case_solvents:
            if case_solvent in SOLVENT_SMILES:
                sim = calculate_tanimoto_similarity(
                    row['solvent_smiles'],
                    SOLVENT_SMILES[case_solvent]
                )
                solvent_similarities.append(sim)
            else:
                solvent_similarities.append(0.0)
        
        # Find best matching case study solvent
        best_match_idx = np.argmax(solvent_similarities)
        all_solvent_assignments.append(best_match_idx)
    
    # Assign monomer similarities to bins
    all_monomer_bins = [assign_to_bin(sim, MONOMER_BINS) for sim in all_monomer_sims]
    
    # Calculate statistics for each solvent to determine ordering
    print("\nCalculating solvent statistics for ordering...")
    solvent_stats = []
    
    for solvent_idx, case_solvent in enumerate(original_case_solvents):
        # Find points assigned to this solvent
        matching_indices = [i for i, assign in enumerate(all_solvent_assignments) if assign == solvent_idx]
        
        if len(matching_indices) > 0:
            matching_sims = [all_monomer_sims[i] for i in matching_indices]
            matching_classes = [all_classes[i] for i in matching_indices]
            
            # Find the point that is farthest away (minimum similarity)
            min_sim = min(matching_sims) if matching_sims else 1.0
            farthest_distance = 1.0 - min_sim  # Distance from perfect similarity (1.0)
            
            solvent_stats.append({
                'idx': solvent_idx,
                'name': case_solvent,
                'farthest_distance': farthest_distance,
                'min_sim': min_sim,
                'matching_indices': matching_indices
            })
        else:
            solvent_stats.append({
                'idx': solvent_idx,
                'name': case_solvent,
                'farthest_distance': 1.0,
                'min_sim': 0.0,
                'matching_indices': []
            })
    
    # Sort solvents: left = lowest min similarity (farthest), right = highest min similarity (closest)
    # Sort by minimum similarity in ascending order (lowest first = left)
    # Special handling: "Bis(2-methoxyethyl) ether" should be rightmost, "2-(2-Methoxyethoxy)ethanol" should be left of it
    def sort_key(s):
        min_sim = s['min_sim']
        name = s['name']
        # Special case: ensure "Bis(2-methoxyethyl) ether" comes last (rightmost)
        if "Bis(2-methoxyethyl)" in name:
            # Give it a very high value to ensure it comes last (rightmost)
            return 10.0  # Very high to push to right
        elif "2-(2-Methoxyethoxy)" in name:
            # Give it a high value but lower than Bis, so it comes just before Bis (second from right)
            return 9.0
        return min_sim
    
    solvent_stats.sort(key=sort_key)
    
    # Create reordered solvent list and mapping
    reordered_solvents = [s['name'] for s in solvent_stats]
    old_to_new_idx = {s['idx']: new_idx for new_idx, s in enumerate(solvent_stats)}
    
    print(f"\nReordered solvents (left=lowest min sim, right=highest min sim):")
    for i, s in enumerate(solvent_stats):
        print(f"  {i}: {s['name']} (min_sim={s['min_sim']:.3f})")
    
    # Reorder case study data to match new solvent order (by matching names)
    reordered_minsk_classes = [0] * len(original_case_solvents)
    reordered_model_classes = [0] * len(original_case_solvents)
    reordered_agreement = [True] * len(original_case_solvents)
    
    for old_idx, old_name in enumerate(original_case_solvents):
        new_idx = reordered_solvents.index(old_name)
        reordered_minsk_classes[new_idx] = case_minsk_classes[old_idx]
        reordered_model_classes[new_idx] = case_model_classes[old_idx]
        reordered_agreement[new_idx] = bool(case_agreement[old_idx])
    
    # Use reordered solvents for the rest of the code
    case_solvents_ordered = reordered_solvents
    
    # Remap all_solvent_assignments to new indices
    all_solvent_assignments_remapped = [old_to_new_idx[assign] for assign in all_solvent_assignments]
    
    # Find nearest neighbors for each case study point
    print(f"\nFinding {N_NEIGHBORS_PER_POINT} nearest neighbors per case study point...")
    
    all_neighbor_indices = set()
    
    for new_solvent_idx, case_solvent in enumerate(case_solvents_ordered):
        # Find points assigned to this solvent (using remapped indices)
        matching_indices = [i for i, assign in enumerate(all_solvent_assignments_remapped) if assign == new_solvent_idx]
        
        if len(matching_indices) > 0:
            # Sort by monomer similarity (descending) and take top N
            matching_sims = [(i, all_monomer_sims[i]) for i in matching_indices]
            matching_sims.sort(key=lambda x: x[1], reverse=True)
            top_neighbors = [idx for idx, _ in matching_sims[:N_NEIGHBORS_PER_POINT]]
            all_neighbor_indices.update(top_neighbors)
    
    neighbor_indices = list(all_neighbor_indices)
    print(f"Found {len(neighbor_indices)} unique nearest neighbors")
    
    # Create grid plot
    print("\nCreating grid plot...")
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # All solvents are shown; we no longer filter by agreement with Lookup
    visible_indices = list(range(len(case_solvents_ordered)))
    # Map old solvent indices to compact x-positions 0..(n_visible-1)
    old_to_new_pos = {old_idx: new_idx for new_idx, old_idx in enumerate(visible_indices)}
    
    # Store actual coordinates for showcase points (with remapped x)
    showcase_coordinates = {}
    
    # Set random seed for reproducible jitter
    np.random.seed(42)
    
    # Plot background points with continuous monomer similarity,
    # but only for solvents that are actually predicted (agreement)
    for neighbor_idx in neighbor_indices:
        solvent_pos_old = all_solvent_assignments_remapped[neighbor_idx]
        if solvent_pos_old not in old_to_new_pos:
            continue  # skip neighbors belonging to non-predicted solvents
        solvent_pos = old_to_new_pos[solvent_pos_old]
        monomer_sim = all_monomer_sims[neighbor_idx]  # Use actual similarity value
        cls = all_classes[neighbor_idx]
        
        # No jitter - points lie exactly on solvent lines
        actual_x = solvent_pos
        actual_y = monomer_sim  # Continuous value
        
        # Store coordinates for showcase points
        showcase_coordinates[neighbor_idx] = (actual_x, actual_y)
        
        # Color: gray for database points, marker by class
        ax.scatter(
            actual_x,
            actual_y,
            c=DATASET_COLOR,
            marker=CLASS_MARKERS[cls],
            alpha=0.5,
            s=60,
            edgecolors='none',
            zorder=1
        )
    
    # Plot single case study row: one point per solvent (final model only)
    y_pos = 1.1
    for old_idx in visible_indices:
        new_x = old_to_new_pos[old_idx]
        pred_class = reordered_model_classes[old_idx]  # same as baseline when agree
        is_correct = (reordered_minsk_classes[old_idx] == pred_class)
        agrees_with_lookup = bool(reordered_agreement[old_idx])
        color = CLASS_COLORS[0] if is_correct else CLASS_COLORS[2]
        marker = CLASS_MARKERS[pred_class]
        ax.scatter(
            new_x,
            y_pos,
            c=color,
            s=200,
            alpha=0.85 if agrees_with_lookup else 0.28,
            edgecolors='black',
            linewidth=2.5 if agrees_with_lookup else 1.5,
            marker=marker,
            zorder=3
        )
    
    # Configure axes
    ax.set_xticks(list(range(len(visible_indices))))
    
    # Shorten long solvent names for x-axis (only for visible solvents)
    x_labels_all = []
    for solvent in case_solvents_ordered:
        if len(solvent) > 20:
            if "Bis(2-methoxyethyl)" in solvent:
                x_labels_all.append("Bis(2-methoxy-\nethyl) ether")
            elif "2-(2-Methoxyethoxy)" in solvent:
                x_labels_all.append("2-(2-Methoxy-\nethoxy)ethanol")
            else:
                x_labels_all.append(solvent[:15] + "...")
        else:
            x_labels_all.append(solvent.replace(" ", "\n", 1) if len(solvent) > 12 else solvent)
    
    # Only label solvents that are actually predicted (agreement), in compact order
    x_labels_visible = [x_labels_all[old_idx] for old_idx in visible_indices]
    ax.set_xticklabels(x_labels_visible, rotation=45, ha='right', fontsize=10)
    
    # Remove tick marks on x-axis (we have the vertical lines)
    ax.tick_params(axis='x', length=0)
    
    # Y-axis: continuous monomer similarity (0.0 to 1.0) + case study row at 1.1 (agreed only)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0', 'Case Study'], fontsize=11)
    
    ax.set_xlabel('Solvent', fontsize=14, fontweight='bold')
    ax.set_ylabel('Monomer Tanimoto Similarity\n(to Acrylamide + Styrene)', fontsize=14, fontweight='bold')
    
    # Add gridlines - main plot area
    # X-limits: from a bit left of first column to just before molecule boxes
    ax.set_xlim(-1.2, len(visible_indices) + 1.0)
    ax.set_ylim(0.0, 1.2)  # Extended to accommodate case study row at 1.1
    
    # Disable automatic grid
    ax.grid(False)
    ax.set_axisbelow(True)
    
    # Draw vertical grid lines manually (from y=0 to y=1.1, only for predicted solvents)
    grid_top = 1.1
    for new_x in range(len(visible_indices)):
        x_pos = new_x  # On each solvent with prediction
        ax.plot([x_pos, x_pos], [0.0, grid_top], 
               color='gray', linestyle='-', linewidth=1.5, alpha=0.3, zorder=1)
    
    # Add Y-axis line at left edge (from y=0 to grid_top)
    ax.plot([-1.2, -1.2], [0.0, grid_top], 
           color='black', linewidth=1.5, alpha=0.7, zorder=2)
    
    # Move Y-axis spine to left edge (but make it invisible, we draw manually)
    ax.spines['left'].set_visible(False)
    
    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # Legend - simplified: one row of points (agreed predictions only)
    legend_elements = []
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=CLASS_COLORS[0], 
               markersize=12, markeredgecolor='black', markeredgewidth=1.5,
               label='Correct', linestyle='None', alpha=0.8)
    )
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=CLASS_COLORS[2], 
               markersize=12, markeredgecolor='black', markeredgewidth=1.5,
               label='Incorrect', linestyle='None', alpha=0.8)
    )

    # Agreement cue (opacity)
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='gray',
               markersize=10, markeredgecolor='black', markeredgewidth=1.2,
               label='No model/lookup agreement (faded)', linestyle='None', alpha=0.28)
    )
    
    # Add separator
    legend_elements.append(Line2D([0], [0], color='none', label=''))
    
    # Shapes: Class markers with new labels (new class semantics)
    class_labels = {
        0: 'Alternating',
        1: 'Random / block-like',
        2: 'Gradient'
    }
    for cls in sorted(CLASS_COLORS.keys()):
        legend_elements.append(
            Line2D([0], [0], marker=CLASS_MARKERS[cls], color='w', 
                   markerfacecolor='gray', 
                   markersize=10, markeredgecolor='black', markeredgewidth=1.0,
                   label=class_labels[cls], linestyle='None', alpha=0.7)
        )
    
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11, 
              frameon=True, framealpha=0.95, edgecolor='gray',
              bbox_to_anchor=(0.02, 0.0))  # Position in left lower corner, at similarity 0.0
    
    # Remove box around plot
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Find 3 key points for molecule boxes
    print("\nFinding key points for molecule boxes...")
    
    # 1. Point at similarity ~1.0 (highest similarity, case study molecules)
    case_study_point = None
    case_study_sim = 1.0  # Case study has perfect similarity to itself
    
    # 2. Point approximately in the middle (similarity ~0.5)
    middle_candidates = []
    for neighbor_idx in neighbor_indices:
        sim = all_monomer_sims[neighbor_idx]
        if 0.4 <= sim <= 0.6:  # Middle range
            middle_candidates.append((neighbor_idx, sim, abs(sim - 0.5)))
    
    middle_point = None
    if middle_candidates:
        middle_candidates.sort(key=lambda x: x[2])  # Sort by distance to 0.5
        middle_point = middle_candidates[0][0]
    
    # 3. Most deviating point (lowest similarity or different class)
    deviating_candidates = []
    for neighbor_idx in neighbor_indices:
        sim = all_monomer_sims[neighbor_idx]
        cls = all_classes[neighbor_idx]
        # Find which solvent this point belongs to
        solvent_pos = all_solvent_assignments_remapped[neighbor_idx]
        case_class = reordered_model_classes[solvent_pos]
        # Deviation: low similarity or wrong class
        deviation = (1.0 - sim) + (1.0 if cls != case_class else 0.0)
        deviating_candidates.append((neighbor_idx, sim, deviation))
    
    most_deviating_point = None
    if deviating_candidates:
        deviating_candidates.sort(key=lambda x: -x[2])  # Sort by deviation (descending)
        most_deviating_point = deviating_candidates[0][0]
    
    # Add molecule boxes to the right of the plot (closer to last solvent)
    last_col_x = len(visible_indices) - 1 if len(visible_indices) > 0 else 0
    box_x_position = last_col_x + 0.5
    box_width = 0.6
    box_spacing = 0.4
    
    # Box 1: Case study molecules at similarity 1.0
    box1_y = 1.0
    mon1_img = draw_molecule_structure(MONOMER1_SMILES, size=(120, 120))
    mon2_img = draw_molecule_structure(MONOMER2_SMILES, size=(120, 120))
    if mon1_img is not None and mon2_img is not None:
        combined_img = np.hstack([mon1_img, mon2_img])
        imagebox = OffsetImage(combined_img, zoom=0.6)
        # Calculate box center for text positioning (slightly left of geometric center)
        # Box width in data coordinates: image_width * zoom / dpi (dpi=100 default)
        box_width_data = (combined_img.shape[1] * 0.6) / 100.0
        box_center_x = box_x_position + box_width_data / 2.0 - 0.05
        ab = AnnotationBbox(imagebox, (box_x_position, box1_y), 
                           xycoords='data', frameon=True, 
                           box_alignment=(0, 0.5), pad=0.3,
                           bboxprops=dict(boxstyle='round,pad=0.3', 
                                         facecolor='white', edgecolor='black', 
                                         linewidth=2.0, alpha=0.95))
        ax.add_artist(ab)
        # Add label just below the box
        ax.text(box_center_x, box1_y - 0.14, 'Similarity: 1.0', 
                ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Box 2: Middle point
    if middle_point is not None:
        middle_sim = all_monomer_sims[middle_point]
        box2_y = middle_sim  # Use actual similarity position of the selected point
        row = df_train_clean.iloc[middle_point]
        mon1_smiles = row['monomer1_smiles']
        mon2_smiles = row['monomer2_smiles']
        mon1_img = draw_molecule_structure(mon1_smiles, size=(120, 120))
        mon2_img = draw_molecule_structure(mon2_smiles, size=(120, 120))
        if mon1_img is not None and mon2_img is not None:
            combined_img = np.hstack([mon1_img, mon2_img])
            imagebox = OffsetImage(combined_img, zoom=0.6)
            # Calculate box center for text positioning (slightly left of geometric center)
            box_width_data = (combined_img.shape[1] * 0.6) / 100.0
            box_center_x = box_x_position + box_width_data / 2.0 - 0.05
            ab = AnnotationBbox(imagebox, (box_x_position, box2_y), 
                               xycoords='data', frameon=True, 
                               box_alignment=(0, 0.5), pad=0.3,
                               bboxprops=dict(boxstyle='round,pad=0.3', 
                                             facecolor='white', edgecolor='black', 
                                             linewidth=2.0, alpha=0.95))
            ax.add_artist(ab)
            # Add label just below the box
            ax.text(box_center_x, box2_y - 0.14, f'Similarity: {middle_sim:.2f}', 
                    ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Box 3: Most deviating point (at its actual y-position)
    if most_deviating_point is not None:
        deviating_sim = all_monomer_sims[most_deviating_point]
        box3_y = deviating_sim  # Use actual similarity position
        row = df_train_clean.iloc[most_deviating_point]
        mon1_smiles = row['monomer1_smiles']
        mon2_smiles = row['monomer2_smiles']
        mon1_img = draw_molecule_structure(mon1_smiles, size=(120, 120))
        mon2_img = draw_molecule_structure(mon2_smiles, size=(120, 120))
        if mon1_img is not None and mon2_img is not None:
            combined_img = np.hstack([mon1_img, mon2_img])
            imagebox = OffsetImage(combined_img, zoom=0.6)
            # Calculate box center for text positioning (slightly left of geometric center)
            box_width_data = (combined_img.shape[1] * 0.6) / 100.0
            box_center_x = box_x_position + box_width_data / 2.0 - 0.05
            ab = AnnotationBbox(imagebox, (box_x_position, box3_y), 
                               xycoords='data', frameon=True, 
                               box_alignment=(0, 0.5), pad=0.3,
                               bboxprops=dict(boxstyle='round,pad=0.3', 
                                             facecolor='white', edgecolor='black', 
                                             linewidth=2.0, alpha=0.95))
            ax.add_artist(ab)
            # Add label just below the box (slightly closer)
            label_y = box3_y - 0.14
            ax.text(box_center_x, label_y, f'Similarity: {deviating_sim:.2f}', 
                    ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Add dashed horizontal lines to show y-positions of molecule boxes
    # Lines should go up to the molecule boxes (on the right side)
    line_end_x = len(case_solvents_ordered) + 0.05  # End just before the boxes start
    
    # Line 1: at box1_y (1.0)
    ax.plot([-1.2, line_end_x], [box1_y, box1_y], color='gray', linestyle='--', linewidth=1.0, alpha=0.5, zorder=0)
    
    # Line 2: at box2_y (middle point)
    if middle_point is not None:
        ax.plot([-1.2, line_end_x], [box2_y, box2_y], color='gray', linestyle='--', linewidth=1.0, alpha=0.5, zorder=0)
    
    # Line 3: at box3_y (most deviating point)
    if most_deviating_point is not None:
        ax.plot([-1.2, line_end_x], [box3_y, box3_y], color='gray', linestyle='--', linewidth=1.0, alpha=0.5, zorder=0)
    
    plt.tight_layout(pad=0.5)
    
    # Save plot
    output_path = os.path.join(os.path.dirname(__file__), "solvent_case_study.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    
    output_path_pdf = os.path.join(os.path.dirname(__file__), "solvent_case_study.pdf")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    print(f"Plot saved to: {output_path_pdf}")
    
    plt.close(fig)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Case study solvents: {len(case_solvents_ordered)}")
    
    # Performance (final model only, no Lookup)
    n_total = len(case_solvents_ordered)
    model_correct = sum(
        1 for i in range(n_total)
        if reordered_minsk_classes[i] == reordered_model_classes[i]
    )

    print(f"\nPerformance of final model on all {n_total} case-study solvents:")
    print(f"  Correct: {model_correct}/{n_total} "
          f"({model_correct / n_total * 100:.1f}%)")
    
    print(f"\nNearest neighbor samples: {len(neighbor_indices)}")
    
    neighbor_classes = [all_classes[i] for i in neighbor_indices]
    print(f"Class distribution in nearest neighbors:")
    for cls in sorted(CLASS_COLORS.keys()):
        count = neighbor_classes.count(cls)
        print(f"  Class {cls}: {count} samples ({100*count/len(neighbor_indices):.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
