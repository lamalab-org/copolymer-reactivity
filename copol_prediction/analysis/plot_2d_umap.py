#!/usr/bin/env python3
"""
2D UMAP visualization of copolymerization data with molecule structures.

Creates a UMAP projection colored by r1r2 classes, with molecule structures
displayed for representative datapoints.

Usage:
    python plot_2d_umap.py --data-path path/to/processed_data.csv --output-dir output/data_analysis
"""

import os
import sys
import argparse

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid segfaults

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
# Import dimensionality reduction methods
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# UMAP disabled due to segfault on macOS
UMAP_AVAILABLE = False

# Define configuration locally to avoid import issues
TWO_COL_WIDTH_INCH = 7

CLASS_COLORS = {
    0: '#3A3B73',  # Class 0 - Alternating
    1: '#e27f07',  # Class 1 - Block-like
    2: '#6a040f',  # Class 2 - Homopolymer
}

CLASS_LABELS_SHORT = {
    0: "Class 0: Alternating",
    1: "Class 1: Block-like",
    2: "Class 2: Homopolymer",
}

def get_class_label(class_num, style='short'):
    """Get label for a class."""
    return CLASS_LABELS_SHORT.get(class_num, f"Class {class_num}")

def setup_plot_style():
    """Setup matplotlib style."""
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10

# Define feature columns locally to avoid import issues
feature_columns = [
    'fukui_radical_max_1',
    'fukui_radical_max_2',
    'delta_HOMO_LUMO_AA', 'delta_HOMO_LUMO_AB', 'delta_HOMO_LUMO_BB', 'delta_HOMO_LUMO_BA',
    'temperature',
    'polytype_emb_1', 'polytype_emb_2', 'method_emb_1', 'method_emb_2', 'solvent_logP',
    'solvent_TPSA',
    'solvent_HBD',
    'solvent_FractionCSP3'
]

# RDKit disabled due to segfault on macOS
# We'll use monomer names instead of structures
RDKIT_AVAILABLE = False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create 2D UMAP visualization")
    parser.add_argument("--data-path", default="../output/processed_data.csv", 
                       help="Path to processed data CSV")
    parser.add_argument("--output-dir", default="../output/data_analysis", 
                       help="Output directory")
    parser.add_argument("--n-molecules", type=int, default=10, 
                       help="Number of molecule structures to display")
    parser.add_argument("--random-state", type=int, default=42, 
                       help="Random state for reproducibility")
    return parser.parse_args()


def prepare_features(df, feature_cols):
    """Prepare and scale features for UMAP."""
    # Filter to available features
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Columns to keep in output dataframe
    keep_cols = available_features + ['r1r2']
    if 'monomer1_name' in df.columns:
        keep_cols.append('monomer1_name')
    if 'monomer2_name' in df.columns:
        keep_cols.append('monomer2_name')
    
    # Get feature matrix and drop NaN rows (only based on feature columns)
    df_clean = df[keep_cols].dropna(subset=available_features + ['r1r2'])
    X = df_clean[available_features].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, df_clean


def assign_classes(df):
    """Assign r1r2 classes based on product values."""
    classes = []
    for r1r2 in df['r1r2']:
        if r1r2 < 1:
            classes.append(0)  # Alternating
        elif r1r2 <= 25:
            classes.append(1)  # Block-like
        else:
            classes.append(2)  # Homopolymer
    return np.array(classes)


def select_diverse_points(embedding, n_points=10, random_state=42):
    """
    Select diverse points from the embedding using K-means clustering.
    Returns indices of points closest to cluster centers.
    """
    # Use K-means to find diverse clusters
    kmeans = KMeans(n_clusters=n_points, random_state=random_state, n_init=10)
    kmeans.fit(embedding)
    
    # Find points closest to each cluster center
    indices = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(embedding - center, axis=1)
        closest_idx = np.argmin(distances)
        indices.append(closest_idx)
    
    return indices


def get_monomer_info(df, idx):
    """Get monomer information for a datapoint."""
    info_parts = []
    
    # Try to get monomer names
    if 'monomer1_name' in df.columns:
        m1 = df.iloc[idx]['monomer1_name']
        if not pd.isna(m1) and str(m1).strip() != '':
            info_parts.append(str(m1)[:20])
    
    if 'monomer2_name' in df.columns:
        m2 = df.iloc[idx]['monomer2_name']
        if not pd.isna(m2) and str(m2).strip() != '':
            info_parts.append(str(m2)[:20])
    
    # Fallback to r1r2 value if no monomer names found
    if len(info_parts) == 0 and 'r1r2' in df.columns:
        r1r2 = df.iloc[idx]['r1r2']
        info_parts.append(f"r1×r2={r1r2:.2f}")
    
    # Also add r1r2 value if we have monomer names
    if len(info_parts) > 0 and 'r1r2' in df.columns:
        r1r2 = df.iloc[idx]['r1r2']
        info_parts.append(f"({r1r2:.2f})")
    
    return "\n".join(info_parts) if info_parts else f"Sample {idx}"


def create_umap_plot(df, X_scaled, classes, output_dir, n_molecules=10, random_state=42):
    """Create 2D visualization with molecule structures using UMAP, t-SNE, or PCA."""
    print("\nCreating 2D projection...")
    
    # Try different dimensionality reduction methods
    if UMAP_AVAILABLE:
        print("  Using UMAP...")
        try:
            reducer = umap.UMAP(n_components=2, random_state=random_state, n_neighbors=15, min_dist=0.1)
            embedding = reducer.fit_transform(X_scaled)
            method_name = "UMAP"
        except Exception as e:
            print(f"  ✗ UMAP failed: {e}")
            print("  Falling back to t-SNE...")
            reducer = TSNE(n_components=2, random_state=random_state, perplexity=30, n_iter=1000)
            embedding = reducer.fit_transform(X_scaled)
            method_name = "t-SNE"
    else:
        print("  Using t-SNE...")
        reducer = TSNE(n_components=2, random_state=random_state, perplexity=30, n_iter=1000)
        embedding = reducer.fit_transform(X_scaled)
        method_name = "t-SNE"
    
    print(f"  ✓ {method_name} embedding shape: {embedding.shape}")
    
    # Select diverse points for annotation
    if n_molecules > 0:
        print(f"\nSelecting {n_molecules} diverse datapoints...")
        diverse_indices = select_diverse_points(embedding, n_points=n_molecules, random_state=random_state)
        print(f"  Selected indices: {diverse_indices}")
    else:
        diverse_indices = []
    
    # Create figure
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, TWO_COL_WIDTH_INCH))
    
    # Plot all points colored by class
    for cls in [0, 1, 2]:
        mask = classes == cls
        ax.scatter(
            embedding[mask, 0], 
            embedding[mask, 1],
            c=CLASS_COLORS[cls],
            label=get_class_label(cls, style='short'),
            alpha=0.6,
            s=20,
            edgecolors='none'
        )
    
    # Add annotations for diverse points
    if len(diverse_indices) > 0:
        print("\nAnnotating representative datapoints...")
        
        for idx in diverse_indices:
            # Get position in embedding
            x, y = embedding[idx]
            
            # Get monomer info
            info = get_monomer_info(df, idx)
            
            # Add larger marker for this point
            ax.scatter(
                [x], [y], 
                s=150, 
                c='white', 
                edgecolors='black', 
                linewidths=1.5,
                zorder=10,
                alpha=0.8
            )
            
            # Add text annotation with background box
            ax.annotate(
                info,
                xy=(x, y),
                xytext=(10, 10),
                textcoords='offset points',
                fontsize=6,
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    edgecolor='gray',
                    linewidth=0.5,
                    alpha=0.9
                ),
                arrowprops=dict(
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0.3',
                    color='gray',
                    linewidth=0.5
                ),
                zorder=11
            )
            print(f"  ✓ Added annotation for point {idx}: {info}")
    
    # Styling
    ax.set_xlabel(f'{method_name} 1', fontsize=9)
    ax.set_ylabel(f'{method_name} 2', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='best')
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    
    # Save as PNG
    filename_base = f'2d_{method_name.lower().replace("-", "")}_projection'
    path_png = os.path.join(output_dir, f'{filename_base}.png')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved PNG to {path_png}")
    
    # Save as PDF
    path_pdf = os.path.join(output_dir, f'{filename_base}.pdf')
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"  ✓ Saved PDF to {path_pdf}")
    
    plt.close()
    
    # Print class distribution
    print("\n  Class distribution:")
    for cls in [0, 1, 2]:
        count = np.sum(classes == cls)
        percentage = count / len(classes) * 100
        print(f"    {get_class_label(cls, style='short')}: {count} ({percentage:.1f}%)")


def main():
    """Main pipeline."""
    args = parse_args()
    
    # Setup
    setup_plot_style()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("2D VISUALIZATION OF COPOLYMER DATA")
    print("="*60)
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Datapoints to annotate: {args.n_molecules}")
    
    # Load data
    print("\nLoading data...")
    try:
        df = pd.read_csv(args.data_path)
        print(f"  ✓ Loaded {len(df)} samples")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        sys.exit(1)
    
    # Prepare features
    print("\nPreparing features...")
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"  ✓ Using {len(available_features)} features")
    
    X_scaled, df_clean = prepare_features(df, feature_columns)
    print(f"  ✓ Feature matrix shape: {X_scaled.shape}")
    
    # Assign classes
    classes = assign_classes(df_clean)
    
    # Create UMAP plot
    create_umap_plot(
        df_clean, 
        X_scaled, 
        classes, 
        args.output_dir, 
        n_molecules=args.n_molecules,
        random_state=args.random_state
    )
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

