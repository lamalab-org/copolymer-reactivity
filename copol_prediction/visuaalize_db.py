from copolextractor.mongodb_storage import CoPolymerDB
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import IPythonConsole
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from collections import Counter


def show_all_keys(d, parent_key='', keys=None):
    """Recursively collect all keys from a document, including nested ones"""
    if keys is None:
        keys = set()

    for key, value in d.items():
        current_key = f"{parent_key}.{key}" if parent_key else key
        keys.add(current_key)

        # If value is a dictionary, recursively get its keys
        if isinstance(value, dict):
            show_all_keys(value, current_key, keys)
        # If value is a list and contains dictionaries, get keys from first dict
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            show_all_keys(value[0], current_key, keys)

    return keys


def show_database_structure():
    """Display all available keys in the MongoDB collection"""
    db = CoPolymerDB()

    # Get all documents to analyze complete structure
    all_keys = set()

    print("\nCollecting all keys from the database...")
    # Look at first 100 documents to get a good sample of keys
    for doc in db.collection.find().limit(100):
        doc_keys = show_all_keys(doc)
        all_keys.update(doc_keys)

    # Sort and display all found keys
    print("\nAll available fields in the database:")
    for key in sorted(all_keys):
        print(f"- {key}")

    print(f"\nTotal number of unique fields: {len(all_keys)}")


class MonomerInfo:
    def __init__(self, smiles, source, count=1):
        self.smiles = smiles
        self.source = source
        self.count = count


def get_all_monomers():
    """Retrieve all monomers and their sources from CoPolymerDB"""
    db = CoPolymerDB()

    # Fetch all entries with SMILES and original_source
    all_entries = list(db.collection.find({}, {
        'monomer1_s': 1,
        'monomer2_s': 1,
        'original_source': 1,
        '_id': 0
    }))

    # Count occurrences of each SMILES
    smiles_count = Counter()
    smiles_source = {}

    for entry in all_entries:
        source = entry.get('original_source', 'unknown')
        if 'monomer1_s' in entry and entry['monomer1_s']:
            smiles_count[entry['monomer1_s']] += 1
            smiles_source[entry['monomer1_s']] = source
        if 'monomer2_s' in entry and entry['monomer2_s']:
            smiles_count[entry['monomer2_s']] += 1
            smiles_source[entry['monomer2_s']] = source

    # Create MonomerInfo objects with counts
    monomers_info = [
        MonomerInfo(smiles, smiles_source[smiles], count)
        for smiles, count in smiles_count.items()
    ]

    # Print statistics about monomer frequencies
    print("\nMonomer frequency statistics:")
    counts = [info.count for info in monomers_info]
    print(f"Min occurrences: {min(counts)}")
    print(f"Max occurrences: {max(counts)}")
    print(f"Mean occurrences: {np.mean(counts):.1f}")
    print(f"Median occurrences: {np.median(counts):.1f}")

    return monomers_info


def calculate_morgan_fingerprints(monomer_info_list):
    """Calculate Morgan fingerprints for SMILES structures"""
    fingerprints = []
    valid_monomers = []
    none_smiles = 0
    parse_errors = 0
    fingerprint_errors = 0

    print("\nAnalyzing SMILES validity:")

    for monomer_info in monomer_info_list:
        # Check for None or empty SMILES
        if monomer_info.smiles is None or not monomer_info.smiles.strip():
            none_smiles += 1
            print(f"None/Empty SMILES found (source: {monomer_info.source})")
            continue

        try:
            mol = Chem.MolFromSmiles(monomer_info.smiles)
            if mol is None:
                parse_errors += 1
                print(f"Failed to parse SMILES: {monomer_info.smiles} (source: {monomer_info.source})")
                continue

            # Morgan fingerprint with radius 2 and 1024 bits
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fingerprints.append(list(fp.ToBitString()))
            valid_monomers.append(monomer_info)
        except Exception as e:
            fingerprint_errors += 1
            print(f"Error processing SMILES {monomer_info.smiles}: {str(e)}")
            continue

    total_invalid = none_smiles + parse_errors + fingerprint_errors
    print(f"\nValidity Analysis:")
    print(f"Total monomers processed: {len(monomer_info_list)}")
    print(f"None/Empty SMILES: {none_smiles}")
    print(f"Failed SMILES parsing: {parse_errors}")
    print(f"Failed fingerprint calculation: {fingerprint_errors}")
    print(f"Total invalid: {total_invalid}")
    print(f"Total valid: {len(valid_monomers)}")

    return np.array(fingerprints, dtype=np.float32), valid_monomers


def create_2d_map(fingerprints, valid_monomers):
    """Create 2D map using t-SNE with color-coded and size-coded points"""
    # t-SNE dimension reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(fingerprints) - 1))
    coords = tsne.fit_transform(fingerprints)

    # Create visualization
    plt.figure(figsize=(15, 10))

    # Prepare colors and sizes based on source and frequency
    colors = ['red' if m.source == 'copol database' else 'blue' for m in valid_monomers]

    # Scale sizes between 100 and 1000 based on count
    counts = np.array([m.count for m in valid_monomers])
    min_size = 100
    max_size = 1000
    sizes = min_size + (counts - min(counts)) * (max_size - min_size) / (max(counts) - min(counts))

    # Main scatter plot
    scatter = plt.scatter(coords[:, 0], coords[:, 1],
                          alpha=0.6,
                          c=colors,
                          s=sizes)

    # Add legend for colors
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='red', markersize=10,
                   label='copol database'),
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='blue', markersize=10,
                   label='other sources')
    ]

    # Add legend for sizes
    size_legend_elements = [
        plt.Line2D([0], [0], marker='o', color='gray',
                   markersize=np.sqrt(s / np.pi),
                   label=f'{c} occurrences')
        for s, c in zip([min_size, (min_size + max_size) / 2, max_size],
                        [min(counts), (min(counts) + max(counts)) // 2, max(counts)])
    ]

    # Combine both legends
    all_legend_elements = legend_elements + size_legend_elements
    plt.legend(handles=all_legend_elements, loc='upper right')

    # Show representative molecules
    n_structures = min(30, len(valid_monomers))
    structure_size = 120
    zoom_factor = 0.25
    indices = np.linspace(0, len(valid_monomers) - 1, n_structures, dtype=int)

    for idx in indices:
        mol = Chem.MolFromSmiles(valid_monomers[idx].smiles)
        img = Draw.MolToImage(mol, size=(structure_size, structure_size))
        imagebox = OffsetImage(img, zoom=zoom_factor)
        ab = AnnotationBbox(imagebox, (coords[idx, 0], coords[idx, 1]),
                            box_alignment=(0., 0.),
                            pad=0.3,
                            frameon=True,
                            bboxprops=dict(facecolor='white', edgecolor='gray'))
        plt.gca().add_artist(ab)

    plt.title("2D Mapping of Monomers by Source and Frequency", fontsize=14, pad=20)
    plt.xlabel("t-SNE Component 1", fontsize=12)
    plt.ylabel("t-SNE Component 2", fontsize=12)

    # Optimize layout
    plt.tight_layout()

    # Save map
    plt.savefig("monomer_map.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics
    print("\nVisualization statistics:")
    source_counts = Counter(info.source for info in valid_monomers)
    for source, count in source_counts.most_common():
        total_occurrences = sum(m.count for m in valid_monomers if m.source == source)
        print(f"- '{source}': {count} unique monomers, {total_occurrences} total occurrences")


def analyze_reaction_methods():
    """Analyze the reaction_conditions.method field in the database"""
    db = CoPolymerDB()

    # Fetch all entries with reaction_conditions.method
    method_entries = list(db.collection.find(
        {},
        {'reaction_conditions.polymerization_type': 1, '_id': 0}
    ))

    # Extract methods and count them
    methods = []
    empty_count = 0
    for entry in method_entries:
        # Check if reaction_conditions exists and has method
        try:
            method = entry.get('reaction_conditions', {}).get('polymerization_type')
            if method:
                methods.append(method)
            else:
                empty_count += 1
        except AttributeError:
            empty_count += 1

    # Count occurrences
    method_counts = Counter(methods)

    # Print results
    print(f"\nAnalysis of reaction_conditions.method:")
    print(f"Total documents analyzed: {len(method_entries)}")
    print(f"Documents with method specified: {len(methods)}")
    print(f"Documents without method: {empty_count}")

    print("\nMethod distribution:")
    for method, count in method_counts.most_common():
        percentage = (count / len(method_entries)) * 100
        print(f"- '{method}': {count} occurrences ({percentage:.1f}%)")


def main():
    # Retrieve all monomers
    print("Loading monomers from database...")
    all_monomers = get_all_monomers()

    analyze_reaction_methods()

    show_database_structure()

    # Calculate fingerprints
    print("\nCalculating Morgan fingerprints...")
    fingerprints, valid_monomers = calculate_morgan_fingerprints(all_monomers)

    # Create 2D map
    print("\nCreating 2D map...")
    create_2d_map(fingerprints, valid_monomers)
    print("2D map has been saved as 'monomer_map.png'")


if __name__ == "__main__":
    main()