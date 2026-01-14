"""
Tanimoto Similarity Grid Plot for Case Study

This script creates a grid plot showing:
- X-axis: Discrete positions for each case study solvent
- Y-axis: Monomer similarity bins
- Background: Nearest neighbor points from dataset with class colors at low opacity
- Case study points: colored by predicted class, shape by correctness
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

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "experiments", "data", "train.csv")
CASE_STUDY_PATH = os.path.join(os.path.dirname(__file__), "case_study_features.csv")

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
    print("Loading training data...")
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    print(f"Training data shape: {df_train.shape}")
    
    print("\nLoading case study features...")
    df_case = pd.read_csv(CASE_STUDY_PATH)
    print(f"Case study data shape: {df_case.shape}")
    
    # Extract case study info
    case_minsk_classes = df_case['minsk_class'].tolist()
    case_model_classes = df_case['model_class'].tolist()
    
    print(f"\nCase study solvents: {CASE_SOLVENTS}")
    
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
    case_solvent_smiles_list = [SOLVENT_SMILES[s] for s in CASE_SOLVENTS if s in SOLVENT_SMILES]
    
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
        for case_solvent in CASE_SOLVENTS:
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
    
    # Find nearest neighbors for each case study point
    print(f"\nFinding {N_NEIGHBORS_PER_POINT} nearest neighbors per case study point...")
    
    all_neighbor_indices = set()
    
    for solvent_idx, case_solvent in enumerate(CASE_SOLVENTS):
        # Find points assigned to this solvent
        matching_indices = [i for i, assign in enumerate(all_solvent_assignments) if assign == solvent_idx]
        
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
    
    # Store actual coordinates for showcase points (with jitter)
    showcase_coordinates = {}
    
    # Set random seed for reproducible jitter
    np.random.seed(42)
    
    # Plot background points with continuous monomer similarity
    for neighbor_idx in neighbor_indices:
        solvent_pos = all_solvent_assignments[neighbor_idx]
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
    
    # Plot case study points
    for i, solvent_name in enumerate(CASE_SOLVENTS):
        is_correct = (case_minsk_classes[i] == case_model_classes[i])
        
        # Color: Class 0 color for correct, Class 2 color for incorrect
        color = CLASS_COLORS[0] if is_correct else CLASS_COLORS[2]
        
        # Marker: based on predicted class
        predicted_class = case_model_classes[i]
        marker = CLASS_MARKERS[predicted_class]
        
        # Case study points are above 1.0 (in extra row above plot)
        y_pos = 1.1
        
        ax.scatter(
            i,
            y_pos,
            c=color,
            s=200,
            alpha=0.8,
            edgecolors='black',
            linewidth=2.5,
            marker=marker,
            zorder=3
        )
    
    # Configure axes
    ax.set_xticks(range(len(CASE_SOLVENTS)))
    
    # Shorten long solvent names for x-axis
    x_labels = []
    for solvent in CASE_SOLVENTS:
        if len(solvent) > 20:
            if "Bis(2-methoxyethyl)" in solvent:
                x_labels.append("Bis(2-methoxy-\nethyl) ether")
            elif "2-(2-Methoxyethoxy)" in solvent:
                x_labels.append("2-(2-Methoxy-\nethoxy)ethanol")
            else:
                x_labels.append(solvent[:15] + "...")
        else:
            x_labels.append(solvent.replace(" ", "\n", 1) if len(solvent) > 12 else solvent)
    
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    
    # Remove tick marks on x-axis (we have the vertical lines)
    ax.tick_params(axis='x', length=0)
    
    # Y-axis: continuous monomer similarity (0.0 to 1.0) + case study row at 1.1
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.1])
    ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0', 'Case Study'], fontsize=11)
    
    ax.set_xlabel('Solvent', fontsize=14, fontweight='bold')
    ax.set_ylabel('Monomer Tanimoto Similarity\n(to Acrylamide + Styrene)', fontsize=14, fontweight='bold')
    
    # Add gridlines - main plot area
    ax.set_xlim(-1.2, len(CASE_SOLVENTS) + 0.5)  # Slightly extended for structures
    ax.set_ylim(-0.05, 1.25)  # Extended to accommodate case study points above 1.0
    
    # Disable automatic grid
    ax.grid(False)
    ax.set_axisbelow(True)
    
    # Draw vertical grid lines manually (from y=0 to y=1.1, on solvents)
    for i in range(len(CASE_SOLVENTS)):
        x_pos = i  # On each solvent
        ax.plot([x_pos, x_pos], [0.0, 1.1], 
               color='gray', linestyle='-', linewidth=1.5, alpha=0.3, zorder=1)
    
    # Add Y-axis line at left edge (from y=0 to y=1.1)
    ax.plot([-1.2, -1.2], [0.0, 1.1], 
           color='black', linewidth=1.5, alpha=0.7, zorder=2)
    
    # Move Y-axis spine to left edge (but make it invisible, we draw manually)
    ax.spines['left'].set_visible(False)
    
    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    # Legend - simplified: explain colors and shapes once
    legend_elements = []
    
    # Colors: Correct (Class 0 color) and Incorrect (Class 2 color)
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=CLASS_COLORS[0], 
               markersize=12, markeredgecolor='black', markeredgewidth=1.5,
               label='Case Study: Correct', linestyle='None', alpha=0.8)
    )
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', 
               markerfacecolor=CLASS_COLORS[2], 
               markersize=12, markeredgecolor='black', markeredgewidth=1.5,
               label='Case Study: Incorrect', linestyle='None', alpha=0.8)
    )
    
    # Add separator
    legend_elements.append(Line2D([0], [0], color='none', label=''))
    
    # Shapes: Class markers
    for cls in sorted(CLASS_COLORS.keys()):
        legend_elements.append(
            Line2D([0], [0], marker=CLASS_MARKERS[cls], color='w', 
                   markerfacecolor='gray', 
                   markersize=10, markeredgecolor='black', markeredgewidth=1.0,
                   label=f'Class {cls}', linestyle='None', alpha=0.7)
        )
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=11, 
              frameon=True, framealpha=0.95, edgecolor='gray')
    
    # Remove box around plot
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    # Add molecular structures from database points
    print("\nAdding molecular structures from database points...")
    
    # Select one point per solvent, plus additional ones for solvents with two clusters
    showcase_points = []
    used_indices = set()
    
    # Solvents with two clusters (need 2 boxes each)
    solvents_with_two_clusters = [
        4,  # Bis(2-methoxyethyl) ether
        6,  # 2-(2-Methoxyethoxy)ethanol
    ]
    
    # First: one point per solvent (excluding case study monomers)
    for col_idx in range(len(CASE_SOLVENTS)):
        candidates = []
        for neighbor_idx in neighbor_indices:
            if neighbor_idx in used_indices:
                continue
            if all_solvent_assignments[neighbor_idx] == col_idx:
                # Check if this monomer pair is too similar to case study monomers
                row = df_train_clean.iloc[neighbor_idx]
                mon1_smiles = row['monomer1_smiles']
                mon2_smiles = row['monomer2_smiles']
                
                # Calculate similarity to case study monomers
                case_sim = calculate_monomer_similarity(
                    mon1_smiles,
                    mon2_smiles,
                    MONOMER1_SMILES,
                    MONOMER2_SMILES
                )
                
                # Exclude if too similar (threshold: 0.95 to avoid exact matches)
                if case_sim < 0.95:
                    mon_sim = all_monomer_sims[neighbor_idx]
                    candidates.append((neighbor_idx, mon_sim))
        
        if len(candidates) > 0:
            # For solvents with two clusters, select from different similarity ranges
            if col_idx in solvents_with_two_clusters:
                # Sort by similarity
                candidates.sort(key=lambda x: x[1])
                # Take one from lower half and one from upper half
                lower_candidate = candidates[len(candidates) // 4][0]
                upper_candidate = candidates[3 * len(candidates) // 4][0]
                
                showcase_points.append({
                    'index': lower_candidate,
                    'col': col_idx,
                    'x': col_idx,
                    'y': all_monomer_sims[lower_candidate]
                })
                showcase_points.append({
                    'index': upper_candidate,
                    'col': col_idx,
                    'x': col_idx,
                    'y': all_monomer_sims[upper_candidate]
                })
                used_indices.add(lower_candidate)
                used_indices.add(upper_candidate)
            else:
                # For other solvents, take one from middle
                candidates.sort(key=lambda x: x[1])
                selected_idx = candidates[len(candidates) // 2][0]
                
                showcase_points.append({
                    'index': selected_idx,
                    'col': col_idx,
                    'x': col_idx,
                    'y': all_monomer_sims[selected_idx]
                })
                used_indices.add(selected_idx)
    
    print(f"Selected {len(showcase_points)} points for structure display")
    
    # Position structures DIAGONALLY (right-up) from their points
    # This uses the whitespace in the upper-right area
    for i, point_info in enumerate(showcase_points):
        idx = point_info['index']
        row = df_train_clean.iloc[idx]
        
        # Get actual coordinates (with jitter) for this point
        if idx in showcase_coordinates:
            actual_point_x, actual_point_y = showcase_coordinates[idx]
        else:
            # Fallback to grid position if not found
            actual_point_x = point_info['x']
            actual_point_y = point_info['y']
        
        # Structure diagonal to the right and up (where most whitespace is)
        # REDUCED offset_x to avoid shifting into next column
        offset_x = 0.2  # Reduced from 1.2 - closer to point, won't shift to next column
        # Adjust offset_y based on position - more space at top
        if actual_point_y > 0.7:
            offset_y = 0.15  # Less upward for high similarity points
        else:
            offset_y = 0.25  # More upward for lower similarity points
        
        struct_x = actual_point_x + offset_x
        struct_y = actual_point_y + offset_y
        
        # Get monomer SMILES
        mon1_smiles = row['monomer1_smiles']
        mon2_smiles = row['monomer2_smiles']
        
        # Draw both monomers (slightly smaller size)
        mon1_img = draw_molecule_structure(mon1_smiles, size=(85, 85))
        mon2_img = draw_molecule_structure(mon2_smiles, size=(85, 85))
        
        # Create a combined box with both monomers
        if mon1_img is not None and mon2_img is not None:
            # Combine both images horizontally
            combined_img = np.hstack([mon1_img, mon2_img])
            
            imagebox = OffsetImage(combined_img, zoom=0.45)
            ab = AnnotationBbox(imagebox, (struct_x, struct_y), 
                               xycoords='data', frameon=True, 
                               box_alignment=(0, 0.5), pad=0.2,
                               bboxprops=dict(boxstyle='round,pad=0.2', 
                                             facecolor='white', edgecolor='black', 
                                             linewidth=1.0, alpha=0.95))
            ax.add_artist(ab)
            
            # Draw line from actual data point to structure box
            line_start_x = actual_point_x
            line_start_y = actual_point_y
            line_end_x = struct_x  # Left edge of the box
            line_end_y = struct_y  # Position of the box
            
            ax.plot([line_start_x, line_end_x], [line_start_y, line_end_y], 
                   color='gray', linestyle='--', linewidth=0.8, alpha=0.5, zorder=2)
    
    plt.tight_layout(pad=0.5)
    
    # Save plot
    output_path = os.path.join(os.path.dirname(__file__), "solvent_case_study_tanimoto_grid.png")
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nPlot saved to: {output_path}")
    
    output_path_pdf = os.path.join(os.path.dirname(__file__), "solvent_case_study_tanimoto_grid.pdf")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    print(f"Plot saved to: {output_path_pdf}")
    
    plt.close(fig)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Case study solvents: {len(CASE_SOLVENTS)}")
    correct = sum([case_minsk_classes[i] == case_model_classes[i] for i in range(len(CASE_SOLVENTS))])
    print(f"Correct predictions: {correct}")
    print(f"Incorrect predictions: {len(CASE_SOLVENTS) - correct}")
    print(f"\nNearest neighbor samples: {len(neighbor_indices)}")
    
    neighbor_classes = [all_classes[i] for i in neighbor_indices]
    print(f"Class distribution in nearest neighbors:")
    for cls in sorted(CLASS_COLORS.keys()):
        count = neighbor_classes.count(cls)
        print(f"  Class {cls}: {count} samples ({100*count/len(neighbor_indices):.1f}%)")
    print("="*60)


if __name__ == "__main__":
    main()
