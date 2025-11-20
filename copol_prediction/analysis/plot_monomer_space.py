#!/usr/bin/env python3
"""
2D visualization of monomer space with average r1×r2 values.

Creates a 2D projection of unique monomers, colored by their average r1×r2 value
across all copolymerizations they participate in.

Usage:
    python plot_monomer_space.py --data-path path/to/processed_data.csv --output-dir output/data_analysis
"""

import os
import sys
import argparse

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from PIL import Image
import io

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, Descriptors
    RDKIT_AVAILABLE = True
except:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Run with required_permissions=['all'] to enable molecule structures.")

# Define configuration locally
TWO_COL_WIDTH_INCH = 7

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create 2D monomer space visualization")
    parser.add_argument("--data-path", default="../output/processed_data.csv", 
                       help="Path to processed data CSV")
    parser.add_argument("--output-dir", default="../output/data_analysis", 
                       help="Output directory")
    parser.add_argument("--n-molecules", type=int, default=10, 
                       help="Number of molecule structures to display")
    parser.add_argument("--random-state", type=int, default=42, 
                       help="Random state for reproducibility")
    return parser.parse_args()


def extract_monomers(df):
    """Extract all unique monomers from the dataset with their SMILES and features."""
    print("\nExtracting unique monomers...")
    
    monomers = {}
    
    # Process monomer 1
    for idx, row in df.iterrows():
        if pd.notna(row.get('monomer1_smiles')) and pd.notna(row.get('r1r2')):
            smiles = row['monomer1_smiles']
            name = row.get('monomer1_name', smiles)
            r1r2 = row['r1r2']
            
            if smiles not in monomers:
                monomers[smiles] = {
                    'name': name,
                    'smiles': smiles,
                    'r1r2_values': [],
                    'fukui_radical_max': [],
                }
            
            monomers[smiles]['r1r2_values'].append(r1r2)
            if pd.notna(row.get('fukui_radical_max_1')):
                monomers[smiles]['fukui_radical_max'].append(row['fukui_radical_max_1'])
    
    # Process monomer 2
    for idx, row in df.iterrows():
        if pd.notna(row.get('monomer2_smiles')) and pd.notna(row.get('r1r2')):
            smiles = row['monomer2_smiles']
            name = row.get('monomer2_name', smiles)
            r1r2 = row['r1r2']
            
            if smiles not in monomers:
                monomers[smiles] = {
                    'name': name,
                    'smiles': smiles,
                    'r1r2_values': [],
                    'fukui_radical_max': [],
                }
            
            monomers[smiles]['r1r2_values'].append(r1r2)
            if pd.notna(row.get('fukui_radical_max_2')):
                monomers[smiles]['fukui_radical_max'].append(row['fukui_radical_max_2'])
    
    print(f"  ✓ Found {len(monomers)} unique monomers")
    
    # Calculate averages
    for smiles in monomers:
        monomers[smiles]['avg_r1r2'] = np.mean(monomers[smiles]['r1r2_values'])
        monomers[smiles]['std_r1r2'] = np.std(monomers[smiles]['r1r2_values'])
        monomers[smiles]['n_occurrences'] = len(monomers[smiles]['r1r2_values'])
        if len(monomers[smiles]['fukui_radical_max']) > 0:
            monomers[smiles]['avg_fukui'] = np.mean(monomers[smiles]['fukui_radical_max'])
        else:
            monomers[smiles]['avg_fukui'] = None
    
    return monomers


def calculate_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    """Calculate Morgan fingerprint for a molecule."""
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return None


def calculate_descriptors(smiles):
    """Calculate RDKit molecular descriptors."""
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        desc = {
            'MolWt': Descriptors.MolWt(mol),
            'LogP': Descriptors.MolLogP(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'TPSA': Descriptors.TPSA(mol),
            'NumRotatableBonds': Descriptors.NumRotatableBonds(mol),
            'NumAromaticRings': Descriptors.NumAromaticRings(mol),
        }
        return np.array(list(desc.values()))
    except:
        return None


def prepare_monomer_features(monomers):
    """Prepare feature matrix for monomers."""
    print("\nCalculating molecular features...")
    
    valid_smiles = []
    features = []
    
    if RDKIT_AVAILABLE:
        print("  Using Morgan fingerprints...")
        for smiles, data in monomers.items():
            fp = calculate_morgan_fingerprint(smiles)
            if fp is not None:
                valid_smiles.append(smiles)
                features.append(fp)
    else:
        print("  ⚠ RDKit not available - using placeholder features")
        # Use simple character-based features as fallback
        for smiles, data in monomers.items():
            # Simple features: length, character counts, etc.
            feat = [
                len(smiles),
                smiles.count('C'),
                smiles.count('O'),
                smiles.count('N'),
                smiles.count('='),
                smiles.count('#'),
            ]
            valid_smiles.append(smiles)
            features.append(feat)
    
    features = np.array(features)
    print(f"  ✓ Feature matrix shape: {features.shape}")
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    return features_scaled, valid_smiles


def select_diverse_monomers(embedding, n_points=10, random_state=42):
    """Select diverse monomers from the embedding using K-means clustering."""
    kmeans = KMeans(n_clusters=min(n_points, len(embedding)), random_state=random_state, n_init=10)
    kmeans.fit(embedding)
    
    indices = []
    for center in kmeans.cluster_centers_:
        distances = np.linalg.norm(embedding - center, axis=1)
        closest_idx = np.argmin(distances)
        if closest_idx not in indices:
            indices.append(closest_idx)
    
    return indices[:n_points]


def draw_molecule(smiles, size=(150, 150)):
    """Draw molecule structure from SMILES string in black."""
    if not RDKIT_AVAILABLE:
        return None
    
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        
        # Draw molecule with black atoms/bonds
        drawer = Draw.MolDraw2DCairo(size[0], size[1])
        drawer.drawOptions().useBWAtomPalette()  # Black and white
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img_bytes = drawer.GetDrawingText()
        
        # Convert to numpy array
        from PIL import Image
        img = Image.open(io.BytesIO(img_bytes))
        return np.array(img)
    except Exception as e:
        print(f"  Warning: Could not draw molecule: {e}")
        return None


def create_monomer_plot(monomers, embedding, valid_smiles, output_dir, n_molecules=10, random_state=42):
    """Create 2D visualization of monomer space."""
    print("\nCreating monomer space plot...")
    
    # Get occurrence counts for coloring
    colors = np.array([monomers[smiles]['n_occurrences'] for smiles in valid_smiles])
    
    # Select diverse monomers for annotation
    if n_molecules > 0:
        print(f"  Selecting {n_molecules} diverse monomers...")
        diverse_indices = select_diverse_monomers(embedding, n_points=n_molecules, random_state=random_state)
        print(f"  Selected {len(diverse_indices)} monomers")
    else:
        diverse_indices = []
    
    # Create figure
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, TWO_COL_WIDTH_INCH))
    
    # Plot all monomers with continuous color scale (by occurrence count, capped at 250)
    scatter = ax.scatter(
        embedding[:, 0],
        embedding[:, 1],
        c=colors,
        cmap='RdBu',
        alpha=0.7,
        s=50,
        edgecolors='white',
        linewidths=0.5,
        vmin=0,
        vmax=100
    )
    
    # Add colorbar (smaller)
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Occurrences', fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    
    # Add molecule structures for diverse monomers
    if RDKIT_AVAILABLE and len(diverse_indices) > 0:
        print("  Drawing molecule structures...")
        
        for idx in diverse_indices:
            x, y = embedding[idx]
            smiles = valid_smiles[idx]
            
            img = draw_molecule(smiles, size=(150, 150))
            
            if img is not None:
                # Create image box
                imagebox = OffsetImage(img, zoom=0.25)
                ab = AnnotationBbox(
                    imagebox, (x, y),
                    frameon=True,
                    pad=0.1,
                    bboxprops=dict(
                        boxstyle="round,pad=0.1",
                        facecolor='white',
                        edgecolor='gray',
                        linewidth=0.5,
                        alpha=0.8
                    )
                )
                ax.add_artist(ab)
                
                n_occ = monomers[smiles]['n_occurrences']
                name = monomers[smiles]['name'][:15]
                print(f"    ✓ {name}: {n_occ} occurrences")
    elif len(diverse_indices) > 0:
        print("  Adding text annotations (RDKit not available)...")
        for idx in diverse_indices:
            x, y = embedding[idx]
            smiles = valid_smiles[idx]
            name = monomers[smiles]['name'][:15]
            avg_r1r2 = monomers[smiles]['avg_r1r2']
            
            ax.annotate(
                f"{name}\n({avg_r1r2:.2f})",
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
                )
            )
    
    # Styling
    ax.set_xlabel('t-SNE 1', fontsize=9)
    ax.set_ylabel('t-SNE 2', fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_aspect('equal', 'box')
    
    plt.tight_layout()
    
    # Save
    path_png = os.path.join(output_dir, 'monomer_space_2d.png')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved PNG to {path_png}")
    
    path_pdf = os.path.join(output_dir, 'monomer_space_2d.pdf')
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"  ✓ Saved PDF to {path_pdf}")
    
    plt.close()
    
    # Print statistics
    print(f"\n  Monomer statistics:")
    print(f"    Total unique monomers: {len(valid_smiles)}")
    print(f"    Occurrence range: {int(colors.min())} - {int(colors.max())}")
    print(f"    Median occurrences: {int(np.median(colors))}")
    print(f"    Mean occurrences: {colors.mean():.1f}")


def main():
    """Main pipeline."""
    args = parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("2D MONOMER SPACE VISUALIZATION")
    print("="*60)
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Molecules to display: {args.n_molecules}")
    
    if not RDKIT_AVAILABLE:
        print("\n⚠ RDKit not available - molecule structures will not be displayed")
        print("  Run with --required-permissions=['all'] to enable RDKit")
    
    # Load data
    print("\nLoading data...")
    try:
        df = pd.read_csv(args.data_path)
        print(f"  ✓ Loaded {len(df)} samples")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        sys.exit(1)
    
    # Extract unique monomers
    monomers = extract_monomers(df)
    
    # Prepare features
    features, valid_smiles = prepare_monomer_features(monomers)
    
    # Create t-SNE embedding
    print("\nCreating t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=args.random_state, perplexity=min(30, len(features)-1), max_iter=1000)
    embedding = tsne.fit_transform(features)
    print(f"  ✓ Embedding shape: {embedding.shape}")
    
    # Create plot
    create_monomer_plot(
        monomers,
        embedding,
        valid_smiles,
        args.output_dir,
        n_molecules=args.n_molecules,
        random_state=args.random_state
    )
    
    # Save monomer data to CSV
    csv_path = os.path.join(args.output_dir, 'monomer_statistics.csv')
    monomer_df = pd.DataFrame([
        {
            'smiles': smiles,
            'name': data['name'],
            'avg_r1r2': data['avg_r1r2'],
            'std_r1r2': data['std_r1r2'],
            'n_occurrences': data['n_occurrences'],
        }
        for smiles, data in monomers.items()
        if smiles in valid_smiles
    ])
    monomer_df = monomer_df.sort_values('avg_r1r2', ascending=False)
    monomer_df.to_csv(csv_path, index=False)
    print(f"\n  ✓ Saved monomer statistics to {csv_path}")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

