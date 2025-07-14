import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from collections import Counter, namedtuple
import os
import matplotlib.colors as mcolors
import matplotlib.cm as cm

plt.style.use("lamalab.mplstyle")

# Define MonomerInfo namedtuple for organizing data
MonomerInfo = namedtuple('MonomerInfo', ['smiles', 'count'])


def load_data(csv_path):
    """Load data from CSV file"""
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Print shape and column names for verification
    print(f"Loaded dataframe with shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    return df


def calculate_fingerprints(df):
    """Calculate molecular fingerprints using RDKit"""
    print("Calculating molecular fingerprints...")

    # Extract unique monomers
    monomer1_smiles = df['monomer1_smiles'].dropna().unique()
    monomer2_smiles = df['monomer2_smiles'].dropna().unique()

    # Combine all unique monomers
    all_smiles = np.concatenate([monomer1_smiles, monomer2_smiles])
    unique_smiles = np.unique(all_smiles)

    print(f"Found {len(unique_smiles)} unique monomers")

    # Count occurrences of each monomer
    smiles_counter = Counter(df['monomer1_smiles'].tolist() + df['monomer2_smiles'].tolist())

    # Create list of valid monomers with their info
    valid_monomers = []
    fingerprints = []

    for smiles in unique_smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Calculate Morgan fingerprint (ECFP4)
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                fp_array = np.zeros((1,))
                DataStructs.ConvertToNumpyArray(fp, fp_array)

                count = smiles_counter.get(smiles, 0)

                valid_monomers.append(MonomerInfo(smiles=smiles, count=count))
                fingerprints.append(fp_array)
            else:
                print(f"Warning: Could not parse SMILES: {smiles}")
        except Exception as e:
            print(f"Error processing SMILES {smiles}: {str(e)}")

    print(f"Successfully processed {len(valid_monomers)} valid monomers")
    return np.array(fingerprints), valid_monomers


def create_2d_map(fingerprints, valid_monomers, df):
    """Create 2D map using t-SNE with color-coded points based on r_product (product of constant_1 and constant_2)"""
    print("Creating 2D map using t-SNE...")

    # t-SNE dimension reduction
    perplexity = min(30, len(fingerprints) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    coords = tsne.fit_transform(fingerprints)

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Create a mapping of monomer pairs to their r_product values
    monomer_pair_to_r_product = {}

    print("Calculating r_product values (constant_1 * constant_2) for all monomer pairs...")
    for _, row in df.iterrows():
        # Calculate r_product as constant_1 * constant_2
        const1 = row.get('constant_1', None)
        const2 = row.get('constant_2', None)

        if pd.notna(const1) and pd.notna(const2):
            r_product = const1 * const2

            if pd.notna(row.get('monomer1_smiles', None)) and pd.notna(row.get('monomer2_smiles', None)):
                pair_key = (row['monomer1_smiles'], row['monomer2_smiles'])
                monomer_pair_to_r_product[pair_key] = r_product

                monomer_names = []
                if pd.notna(row.get('monomer1_name')):
                    monomer_names.append(row['monomer1_name'])
                else:
                    monomer_names.append("Unknown_monomer1")

                if pd.notna(row.get('monomer2_name')):
                    monomer_names.append(row['monomer2_name'])
                else:
                    monomer_names.append("Unknown_monomer2")

                print(
                    f"Pair: {monomer_names[0]} + {monomer_names[1]}, constant_1: {const1}, constant_2: {const2}, r_product: {r_product}")

    # Map each monomer to its average r_product value across all pairs
    monomer_to_avg_r_product = {}

    for (m1, m2), r_val in monomer_pair_to_r_product.items():
        if m1 not in monomer_to_avg_r_product:
            monomer_to_avg_r_product[m1] = []
        if m2 not in monomer_to_avg_r_product:
            monomer_to_avg_r_product[m2] = []

        monomer_to_avg_r_product[m1].append(r_val)
        monomer_to_avg_r_product[m2].append(r_val)

    # Calculate average r_product for each monomer
    print("\nCalculating average r_product for each monomer:")
    for monomer, r_values in monomer_to_avg_r_product.items():
        monomer_to_avg_r_product[monomer] = np.mean(r_values)
        print(
            f"Monomer: {monomer}, avg r_product: {monomer_to_avg_r_product[monomer]:.4f} (from {len(r_values)} reactions)")

    # Get r_product values for each valid monomer
    r_product_values = [monomer_to_avg_r_product.get(m.smiles, np.nan) for m in valid_monomers]

    # Create colormap - using coolwarm for r_product visualization
    try:
        # For matplotlib >= 3.7
        cmap = plt.colormaps['coolwarm']
    except:
        # For older matplotlib versions
        cmap = cm.get_cmap('coolwarm')

    # Set fixed colorbar range from 0 to 2 as requested
    norm = mcolors.Normalize(vmin=0, vmax=2)  # Fixed range 0-2 for visualization purposes

    # Replace NaN values with a default color (gray)
    colors = []
    for val in r_product_values:
        if np.isnan(val):
            colors.append('lightgray')
        else:
            colors.append(cmap(norm(val)))

    # Scale sizes between 100 and 1000 based on count
    counts = np.array([m.count for m in valid_monomers])
    min_size = 100
    max_size = 1000

    if max(counts) > min(counts):
        sizes = min_size + (counts - min(counts)) * (max_size - min_size) / (max(counts) - min(counts))
    else:
        sizes = np.full_like(counts, min_size, dtype=float)

    # Find min/max of coordinates to set axis limits with padding
    x_min, x_max = min(coords[:, 0]), max(coords[:, 0])
    y_min, y_max = min(coords[:, 1]), max(coords[:, 1])

    # Add padding (10% on each side)
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1

    # Set axis limits with padding
    plt.xlim(x_min - x_padding, x_max + x_padding)
    plt.ylim(y_min - y_padding, y_max + y_padding)

    # Main scatter plot
    scatter = plt.scatter(coords[:, 0], coords[:, 1],
                          alpha=0.6,
                          c=colors,
                          s=sizes)

    # Add colorbar for r_product values
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca(),
                        label='Mean r-product value over all reactions of one monomer')

    # Add legend for sizes only
    size_legend_elements = []

    if max(counts) > min(counts):
        size_legend_elements = [
            plt.Line2D([0], [0], marker='o', color='gray',
                       markersize=np.sqrt(s / np.pi),
                       label=f'{c} occurrences')
            for s, c in zip([min_size, (min_size + max_size) / 2, max_size],
                            [min(counts), (min(counts) + max(counts)) // 2, max(counts)])
        ]

        # Place size legend
        plt.legend(handles=size_legend_elements, loc='upper left')

    # Show representative molecules
    n_structures = min(10, len(valid_monomers))
    structure_size = 200
    zoom_factor = 0.3


    indices = np.linspace(0, len(valid_monomers) - 1, n_structures, dtype=int)

    for idx in indices:
        try:
            mol = Chem.MolFromSmiles(valid_monomers[idx].smiles)
            if mol is not None:
                img = Draw.MolToImage(mol, size=(structure_size, structure_size))
                imagebox = OffsetImage(img, zoom=zoom_factor)
                ab = AnnotationBbox(imagebox, (coords[idx, 0], coords[idx, 1]),
                                    box_alignment=(0., 0.),
                                    pad=0.3,
                                    frameon=True,
                                    bboxprops=dict(facecolor='white', edgecolor='gray', linewidth=1.0))
                plt.gca().add_artist(ab)
        except Exception as e:
            print(f"Error rendering molecule {valid_monomers[idx].smiles}: {str(e)}")

    plt.title("Unique Monomers in 5K Copolymerizations in Copol Database", fontsize=14, pad=20)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)

    # Optimize layout
    plt.tight_layout()

    # Save map
    plt.savefig("monomer_map.png", dpi=300, bbox_inches='tight')
    print("Map saved as monomer_map.png")
    plt.close()

    # Print statistics
    print("\nVisualization statistics:")
    total_monomers = len(valid_monomers)
    total_occurrences = sum(m.count for m in valid_monomers)
    print(f"- Total unique monomers: {total_monomers}")
    print(f"- Total monomer occurrences: {total_occurrences}")

    # Print r_product statistics
    #if valid_r_products:
        #print(f"- r_product (constant_1*constant_2) min: {min(valid_r_products):.4f}")
        #print(f"- r_product (constant_1*constant_2) max: {max(valid_r_products):.4f}")
        #print(f"- r_product (constant_1*constant_2) mean: {np.mean(valid_r_products):.4f}")
        #print(
            #f"- Monomers with r_product value: {len(valid_r_products)} ({len(valid_r_products) / total_monomers * 100:.1f}%)")


def main():
    """Main function to orchestrate the visualization process"""
    # Path to your CSV file - you can replace this with your actual path
    csv_path = "extracted_reactions.csv"  # Replace with your actual path

    # Load data
    df = load_data(csv_path)

    # Calculate fingerprints and get valid monomers
    fingerprints, valid_monomers = calculate_fingerprints(df)

    # Create and save the 2D map
    create_2d_map(fingerprints, valid_monomers, df)

    print("Done!")


if __name__ == "__main__":
    main()