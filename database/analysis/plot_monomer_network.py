"""
Create a network plot showing monomer co-occurrences in the database.

Nodes represent monomers, edges represent pairs of monomers that appear together
in the database.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from collections import Counter, defaultdict
import requests
import time
import json
import re
import numpy as np

# Try to import RDKit for molecular fingerprints
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit not available. Install with: conda install -c conda-forge rdkit")
    print("Will use fallback positioning (not chemically meaningful)")

# Try to import sklearn for dimensionality reduction
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")
    print("Will use fallback positioning")

# Add copol_prediction to path to import plot_config
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'copol_prediction' / 'analysis'))
from plot_config import (
    SEQUENTIAL_COLORS, 
    TWO_COL_WIDTH_INCH
)

# Load data
data_path = Path(__file__).parent.parent.parent.parent / 'copol_prediction' / 'processed_data.csv'
print(f"Loading data from: {data_path}")

df = pd.read_csv(data_path)

# Cache file for publication years
cache_file = Path(__file__).parent / 'publication_years_cache.json'
publication_years_cache = {}
if cache_file.exists():
    with open(cache_file, 'r') as f:
        publication_years_cache = json.load(f)
    print(f"  Loaded {len(publication_years_cache)} cached publication years")

def _first_nonempty(*values):
    """Return first value that is not NA/empty-string, else None."""
    for v in values:
        if v is None:
            continue
        if isinstance(v, float) and pd.isna(v):
            continue
        s = str(v).strip()
        if s and s.lower() != "nan":
            return s
    return None


_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"<>]+)", re.IGNORECASE)


def extract_doi_from_source(source_value):
    """Extract DOI from a source URL or string.

    Supports:
    - 'https://doi.org/10.xxxx/....'
    - 'http://doi.org/10.xxxx/....'
    - raw '10.xxxx/....'
    """
    if source_value is None or (isinstance(source_value, float) and pd.isna(source_value)):
        return None

    source_str = str(source_value).strip()
    if not source_str or source_str.lower() == "nan":
        return None

    # Normalize common DOI URL prefixes
    source_str = source_str.replace("https://doi.org/", "").replace("http://doi.org/", "")

    # Direct DOI
    if source_str.startswith("10."):
        return source_str.split()[0].strip().rstrip(".,;)")

    # DOI somewhere inside text
    m = _DOI_RE.search(source_str)
    if m:
        return m.group(1).rstrip(".,;)")

    return None

def get_publication_year_from_crossref(doi):
    """Get publication year from Crossref API. Results are cached."""
    # Check cache first (including None values for 404s)
    if doi in publication_years_cache:
        cached_value = publication_years_cache[doi]
        # Return None if cached as None (404), otherwise return the year
        return cached_value if cached_value is not None else None
    
    # Skip if DOI is None or empty
    if not doi:
        return None
    
    try:
        url = f"https://api.crossref.org/works/{doi}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json().get("message", {})
            issued = data.get("issued", {})
            if issued:
                date_parts = issued.get("date-parts", [])
                if date_parts and len(date_parts[0]) > 0:
                    year = date_parts[0][0]
                    publication_years_cache[doi] = year
                    # Save cache after each successful fetch
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, 'w') as f:
                        json.dump(publication_years_cache, f, indent=2)
                    time.sleep(0.1)  # Be nice to Crossref API
                    return year
        elif response.status_code == 404:
            # DOI not found - cache as None to avoid repeated requests
            publication_years_cache[doi] = None
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(publication_years_cache, f, indent=2)
            # Only print warning for first few 404s to avoid spam
            none_count = sum(1 for v in publication_years_cache.values() if v is None)
            if none_count <= 10:
                print(f"  Warning: DOI not found (404): {doi}")
            elif none_count == 11:
                print(f"  ... (suppressing further 404 warnings - many DOIs may not be in Crossref)")
        else:
            # Other errors - don't cache, might be temporary
            if response.status_code != 429:  # Rate limiting
                print(f"  Warning: Crossref returned status {response.status_code} for DOI {doi}")
    except Exception as e:
        print(f"  Warning: Error fetching year for DOI {doi}: {e}")
    
    return None

import re
import pandas as pd

def _norm_name(x: str) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()

def _norm_smiles(x: str) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _has_any(s: str, patterns) -> bool:
    return any(p in s for p in patterns)


def _has_double_bond(smi: str) -> bool:
    """
    Robust check for a C=C-like double bond in typical monomer SMILES.
    Catches e.g. 'C=C', 'FC(=C(F)F)F', 'C=CC', etc.
    """
    if not smi:
        return False
    # Common explicit "C=C"
    if "C=C" in smi:
        return True
    # Catch patterns like "(=C" (tetrafluoroethylene etc.)
    if "(=C" in smi or "=C(" in smi:
        return True
    # Generic fallback: any '=' between two carbons (very permissive but fine at the end)
    return bool(re.search(r"C=.*=?.*C", smi)) or ("=" in smi)


def classify_monomer(monomer_name, monomer_smiles):
    """
    9-class monomer classification scheme for the copolymerization set.
    """
    name = _norm_name(monomer_name)
    smi = _norm_smiles(monomer_smiles)

    # 1) (Meth)acrylonitriles
    if (
        "acrylonitrile" in name
        or "methacrylonitrile" in name
        or _has_any(smi, ["C=CC#N", "C=C(C)C#N"])
    ):
        return "(Meth)acrylonitriles"

    # 2) Anhydrides / unsaturated diacids & diesters (maleic/itaconic + maleate/fumarate/aconitate)
    if any(
        k in name for k in ["maleic", "maleate", "fumar", "fumarate", "itaconic", "itaconate", "aconitate"]
    ):
        return "Anhydrides/Diacids"

    # 3) (Meth)acrylates (acrylates + methacrylates together)
    if (
        ("methacry" in name)
        or ("acryl" in name and "amide" not in name and "nitrile" not in name)
        or "acrylic acid" in name
        or _has_any(
            smi,
            [
                "C=CC(=O)O",
                "C=CC(=O)OC",
                "C=CC(=O)[O-]",
                "C=C(C)C(=O)O",
                "C=C(C)C(=O)OC",
                "C=C(C)C(=O)[O-]",
            ],
        )
    ):
        return "(Meth)acrylates"

    # 4) (Meth)acrylamides & imides
    if (
        "acrylamide" in name
        or "methacrylamide" in name
        or "maleimide" in name
        or _has_any(smi, ["C=CC(=O)N", "C=C(C)C(=O)N"])
    ):
        return "(Meth)acrylamides/imides"

    # 5) Styrenics
    if (
        "styrene" in name
        or "methylstyrene" in name
        or "chlorostyrene" in name
        or "methoxystyrene" in name
        or "styrene sulfonate" in name
        or _has_any(smi, ["C=Cc1ccccc1", "C=CC1=CC=CC=C1"])
    ):
        return "Styrenics"

    # 6) Conjugated Dienes
    if (
        "butadiene" in name
        or "isoprene" in name
        or "chloroprene" in name
        or "diene" in name
        or _has_any(smi, ["C=CC=C", "C=C-C=C"])
    ):
        return "Conjugated Dienes"

    # 7) Vinyl derivatives (everything "vinyl X", including ether/ester/halides/N-vinyl)
    # Important: comes AFTER acrylates/styrenics to avoid misclassifying vinyl-substituted styrenes.
    if "vinyl" in name or re.search(r"\b\d*-?vinyl", name):
        return "Vinyl Derivatives"

    # 8) Olefins / alpha-olefins (including fluorinated ethylenes)
    if (
        name in ["ethylene", "propylene", "propene", "isobutylene"]
        or re.search(r"\b\d+-?(hexene|octene)\b", name)
        or "tetrafluoroethylene" in name
        or "chlorotrifluoroethylene" in name
        or ("ethylene" in name and any(k in name for k in ["fluoro", "chloro", "trifluoro", "tetrafluoro"]))
        or (_has_double_bond(smi) and "c1ccccc1" not in smi.lower())  # crude brake against pure aromatics
    ):
        return "Olefins"

    # 9) Everything else
    return "Other"


def test_classify_monomer_smoke():
    """
    Lightweight smoke test for the classify_monomer function.
    Checks a curated set of name/SMILES examples against expected classes.
    """
    cases = [
        # (Meth)acrylates (both acrylates and methacrylates are now combined)
        ("methyl methacrylate", "C=C(C)C(=O)OC", "(Meth)acrylates"),
        ("Methacrylic Acid", "C=C(C)C(=O)O", "(Meth)acrylates"),
        ("methyl acrylate", "C=CC(=O)OC", "(Meth)acrylates"),
        ("acrylic acid", "C=CC(=O)O", "(Meth)acrylates"),
        # Name-driven: cyanoacrylate should still end up in (Meth)acrylates
        ("Methyl alpha-Cyanoacrylate", "C=CC(=O)OC", "(Meth)acrylates"),

        # (Meth)acrylamides/imides
        ("acrylamide", "C=CC(=O)N", "(Meth)acrylamides/imides"),
        ("N,N-Dimethylacrylamide", "C=CC(=O)N(C)C", "(Meth)acrylamides/imides"),
        # Name-driven: maleimide-type should land in (Meth)acrylamides/imides
        ("N-Phenylmaleimide", "", "(Meth)acrylamides/imides"),

        # (Meth)acrylonitriles
        ("acrylonitrile", "C=CC#N", "(Meth)acrylonitriles"),
        ("methacrylonitrile", "C=C(C)C#N", "(Meth)acrylonitriles"),

        # Styrenics
        ("Styrene", "C=Cc1ccccc1", "Styrenics"),
        ("p-Chlorostyrene", "C=Cc1ccc(Cl)cc1", "Styrenics"),
        ("Sodium Styrene Sulfonate", "", "Styrenics"),

        # Conjugated Dienes
        ("butadiene", "C=CC=C", "Conjugated Dienes"),
        ("Isoprene", "C=C(C)C=C", "Conjugated Dienes"),
        ("chloroprene", "C=CC(Cl)=C", "Conjugated Dienes"),

        # Anhydrides/Diacids (all maleic/itaconic/maleate/fumarate types are now combined)
        ("Maleic anhydride", "O=C1OC(=O)C=CH1", "Anhydrides/Diacids"),
        ("Itaconic Anhydride", "", "Anhydrides/Diacids"),
        ("Trimethyl aconitate", "", "Anhydrides/Diacids"),

        # Vinyl Derivatives (all vinyl types are now combined)
        ("vinyl chloride", "C=CCl", "Vinyl Derivatives"),
        ("vinylidene chloride", "C=C(Cl)Cl", "Vinyl Derivatives"),
        ("Vinyl acetate", "C=COC(=O)C", "Vinyl Derivatives"),
        ("Vinyl benzoate", "C=COC(=O)c1ccccc1", "Vinyl Derivatives"),
        ("acetic acid, monochloro-, vinyl ester", "", "Vinyl Derivatives"),
        ("Vinyl ethyl ether", "C=COCC", "Vinyl Derivatives"),
        ("2-Chloroethyl Vinyl Ether", "C=COCCCl", "Vinyl Derivatives"),
        ("Divinyl Ether", "C=COC=C", "Vinyl Derivatives"),
        ("1-vinylpyrrolidone", "", "Vinyl Derivatives"),
        ("2-Vinylpyridine", "", "Vinyl Derivatives"),
        ("1-vinylimidazole", "", "Vinyl Derivatives"),
        # via "vinylcarbaz" pattern
        ("N-Vinyl carbazol", "", "Vinyl Derivatives"),

        # Olefins
        ("ethylene", "C=C", "Olefins"),
        ("tetrafluoroethylene", "FC(=C(F)F)F", "Olefins"),
        ("chlorotrifluoroethylene", "FC(=C(F)Cl)F", "Olefins"),
        ("propylene", "C=CC", "Olefins"),
        ("1-hexene", "CCCCCC=C", "Olefins"),
        ("Octene-1", "CCCCCCCC=C", "Olefins"),
        ("isobutylene", "C=C(C)C", "Olefins"),

        # Allyl should NOT be mis-binned into vinyl halides/esters/ethers
        # With current scheme we treat them as olefinic (no dedicated allyl class)
        ("Allyl chloride", "C=CCCl", "Olefins"),
        ("Allyl acetate", "CC(=O)OCC=C", "Olefins"),
    ]

    failed = []
    for name, smi, expected in cases:
        got = classify_monomer(name, smi)
        if got != expected:
            failed.append((name, smi, expected, got))

    if failed:
        print("❌ classify_monomer smoke test FAILED:")
        for name, smi, exp, got in failed:
            print(f"- {name!r} | {smi!r} | expected={exp!r}, got={got!r}")
        raise AssertionError(f"{len(failed)} cases failed")
    else:
        print("✅ classify_monomer smoke test PASSED")

# To run manually (not executed by default):
if __name__ == "__main__":
    # Run smoke test for the monomer classification first
    test_classify_monomer_smoke()


# Extract monomer pairs and collect publication years
print(f"\nLoaded {len(df)} entries")
print("Extracting monomer pairs and publication years...")

# Create graph and collect years for each monomer
G = nx.Graph()
monomer_years = defaultdict(list)  # Store all years for each monomer

# First pass: extract DOIs and fetch years
print("  Fetching publication years from Crossref...")
unique_dois = set()
for idx, row in df.iterrows():
    source_val = _first_nonempty(row.get("source"), row.get("original_source"))
    doi = extract_doi_from_source(source_val)
    if doi:
        unique_dois.add(doi)

print(f"  Found {len(unique_dois)} unique DOIs")
print(f"  Fetching years (this may take a while, using cache when available)...")
print(f"    Note: Some DOIs may return 404 (not found in Crossref) - this is normal")
for i, doi in enumerate(unique_dois, 1):
    if i % 50 == 0:
        print(f"    Progress: {i}/{len(unique_dois)}")
    get_publication_year_from_crossref(doi)

# Print summary of fetched years
successful_years = sum(1 for v in publication_years_cache.values() if v is not None and v is not False)
failed_years = sum(1 for v in publication_years_cache.values() if v is None)
print(f"\n  Summary: {successful_years} DOIs with years found, {failed_years} DOIs not found (404)")

# Second pass: build graph and associate years with monomers
# Also track which edges appear in which time periods and classify monomers
print("  Building graph and associating years with monomers...")
print("  Classifying monomers into classes...")
edge_counts = Counter()
edge_years = defaultdict(list)  # Track years for each edge
monomer_classes = {}  # Store class for each monomer SMILES
# Count how many data points (rows) each monomer appears in
monomer_datapoint_counts = Counter()

for idx, row in df.iterrows():
    monomer1 = row.get('monomer1_smiles')
    monomer2 = row.get('monomer2_smiles')
    monomer1_name = row.get('monomer1_name', monomer1)
    monomer2_name = row.get('monomer2_name', monomer2)
    
    # Skip if either monomer is missing
    if pd.isna(monomer1) or pd.isna(monomer2):
        continue
    
    # Classify monomers
    if monomer1 not in monomer_classes:
        monomer_classes[monomer1] = classify_monomer(monomer1_name, monomer1)
    if monomer2 not in monomer_classes:
        monomer_classes[monomer2] = classify_monomer(monomer2_name, monomer2)
    
    # Get publication year for this entry
    source_val = _first_nonempty(row.get("source"), row.get("original_source"))
    doi = extract_doi_from_source(source_val)
    year = None
    if doi and doi in publication_years_cache:
        year = publication_years_cache[doi]
    
    # Use SMILES as node identifiers (more unique than names)
    # But store names and classes for display
    if monomer1 not in G:
        G.add_node(monomer1, name=monomer1_name, class_name=monomer_classes[monomer1])
    if monomer2 not in G:
        G.add_node(monomer2, name=monomer2_name, class_name=monomer_classes[monomer2])
    
    # Store year for each monomer
    if year:
        monomer_years[monomer1].append(year)
        monomer_years[monomer2].append(year)
    
    # Add edge (or increment weight if edge already exists)
    edge_key = (monomer1, monomer2)
    if G.has_edge(monomer1, monomer2):
        G[monomer1][monomer2]['weight'] += 1
    else:
        G.add_edge(monomer1, monomer2, weight=1)
    
    # Track year for this edge occurrence
    if year:
        edge_years[edge_key].append(year)
    
    edge_counts[edge_key] += 1
    # Count how many data rows each monomer appears in
    monomer_datapoint_counts[monomer1] += 1
    monomer_datapoint_counts[monomer2] += 1

# Calculate average publication year for each monomer
monomer_avg_years = {}
for monomer, years in monomer_years.items():
    if years:
        monomer_avg_years[monomer] = sum(years) / len(years)

print(f"  ✓ Found {G.number_of_nodes()} unique monomers")
print(f"  ✓ Found {G.number_of_edges()} unique monomer pairs")

# Filter: Only keep nodes that appear at least N times
min_occurrences = 5
print(f"\nFiltering: Only showing monomers that appear at least {min_occurrences} times...")

# Calculate node degrees (number of connections)
node_degrees = dict(G.degree())

# Create subgraph with only frequently occurring monomers
frequent_nodes = {node for node, degree in node_degrees.items() if degree >= min_occurrences}
G_filtered = G.subgraph(frequent_nodes).copy()

print(f"  ✓ Filtered to {G_filtered.number_of_nodes()} nodes (from {G.number_of_nodes()})")
print(f"  ✓ Filtered to {G_filtered.number_of_edges()} edges (from {G.number_of_edges()})")

# Calculate chemical fingerprints and reduce to 2D for positioning
print("\nCalculating chemical fingerprints and 2D positions...")

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
    except Exception as e:
        print(f"    Warning: Could not calculate fingerprint for {smiles[:30]}: {e}")
        return None

# Calculate fingerprints for all filtered nodes
node_list = list(G_filtered.nodes())
fingerprints = []
valid_nodes = []

print(f"  Calculating fingerprints for {len(node_list)} nodes...")
for i, smiles in enumerate(node_list):
    if i % 50 == 0 and i > 0:
        print(f"    Progress: {i}/{len(node_list)}")
    fp = calculate_morgan_fingerprint(smiles)
    if fp is not None:
        fingerprints.append(fp)
        valid_nodes.append(smiles)

if len(fingerprints) == 0:
    print("  ⚠ No valid fingerprints calculated, using fallback layout")
    fingerprints = None
else:
    fingerprints = np.array(fingerprints)
    print(f"  ✓ Calculated {len(fingerprints)} fingerprints")

# Reduce to 2D using t-SNE or PCA
pos_2d = {}
if fingerprints is not None and SKLEARN_AVAILABLE:
    print("  Reducing to 2D using t-SNE (this may take a while)...")
    try:
        # Use t-SNE for better separation of chemically similar molecules
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(fingerprints)-1))
        coords_2d = tsne.fit_transform(fingerprints)
        
        # Create position dictionary
        for i, smiles in enumerate(valid_nodes):
            pos_2d[smiles] = (coords_2d[i, 0], coords_2d[i, 1])
        
        print(f"  ✓ Calculated 2D positions for {len(pos_2d)} nodes using t-SNE")
    except Exception as e:
        print(f"  ⚠ t-SNE failed: {e}, trying PCA...")
        try:
            pca = PCA(n_components=2, random_state=42)
            coords_2d = pca.fit_transform(fingerprints)
            
            for i, smiles in enumerate(valid_nodes):
                pos_2d[smiles] = (coords_2d[i, 0], coords_2d[i, 1])
            
            print(f"  ✓ Calculated 2D positions for {len(pos_2d)} nodes using PCA")
        except Exception as e2:
            print(f"  ⚠ PCA also failed: {e2}, using fallback layout")
            pos_2d = None
else:
    pos_2d = None

# Fallback: use network layout if fingerprint-based positioning failed
if pos_2d is None or len(pos_2d) < len(node_list):
    print("  Using network-based layout as fallback...")
    # Fill in missing positions with network layout
    if pos_2d is None:
        pos_2d = {}
    
    # Get nodes that need positions
    missing_nodes = set(node_list) - set(pos_2d.keys())
    if missing_nodes:
        G_missing = G_filtered.subgraph(missing_nodes)
        if G_missing.number_of_nodes() > 0:
            try:
                pos_missing = nx.kamada_kawai_layout(G_missing, weight='weight', seed=42)
            except:
                pos_missing = nx.spring_layout(G_missing, k=2, iterations=100, seed=42)
            pos_2d.update(pos_missing)
    
    # If we still don't have all positions, use network layout for all
    if len(pos_2d) < len(node_list):
        try:
            pos_network = nx.kamada_kawai_layout(G_filtered, weight='weight', seed=42)
        except:
            pos_network = nx.spring_layout(G_filtered, k=2, iterations=100, seed=42)
        pos_2d = pos_network
        print("  ✓ Using network-based layout for all nodes")

# Create network plots for different time period configurations
print("\nCreating network plots for different time periods...")

# Define time period configuration (final: before and after 1970)
time_period_configs = [
    {
        'name': 'two_periods_1970',
        'periods': [
            ("<1970", None, 1969),
            ("≥1970", 1970, None),
        ]
    },
]

# Define class order and colors for the 9-class scheme
preferred_class_order = [
    '(Meth)acrylonitriles',
    'Anhydrides/Diacids',
    '(Meth)acrylates',
    '(Meth)acrylamides/imides',
    'Styrenics',
    'Conjugated Dienes',
    'Vinyl Derivatives',
    'Olefins',
    'Other',
]

# Collect all classes that appear in any period for legend
all_classes_in_data = set()
for node in G_filtered.nodes():
    class_name = G_filtered.nodes[node].get('class_name', 'Other')
    all_classes_in_data.add(class_name)

# Process each time period configuration
for config in time_period_configs:
    config_name = config['name']
    time_periods = config['periods']
    
    print(f"\n{'='*60}")
    print(f"Processing configuration: {config_name}")
    print(f"{'='*60}")
    
    # Create subgraphs for each time period
    period_graphs = []
    for period_name, start_year, end_year in time_periods:
        print(f"\n  Processing period: {period_name}")
        
        # Find edges that appear in this time period
        period_edges = set()
        period_nodes = set()
        
        for edge_key, years in edge_years.items():
            # Check if any occurrence of this edge is in the time period
            for year in years:
                if start_year is None and end_year is not None:
                    if year <= end_year:
                        period_edges.add(edge_key)
                        period_nodes.add(edge_key[0])
                        period_nodes.add(edge_key[1])
                        break
                elif start_year is not None and end_year is None:
                    if year >= start_year:
                        period_edges.add(edge_key)
                        period_nodes.add(edge_key[0])
                        period_nodes.add(edge_key[1])
                        break
                elif start_year is not None and end_year is not None:
                    if start_year <= year <= end_year:
                        period_edges.add(edge_key)
                        period_nodes.add(edge_key[0])
                        period_nodes.add(edge_key[1])
                        break
        
        # Filter to only include nodes that meet the minimum occurrence threshold
        period_nodes = period_nodes & frequent_nodes
        
        if len(period_nodes) == 0:
            print(f"    ⚠ No nodes found for period {period_name}")
            period_graphs.append((period_name, None))
            continue
        
        # Create subgraph for this period
        G_period = G_filtered.subgraph(period_nodes).copy()
        
        # Only include edges that are in this period
        edges_to_remove = []
        for u, v in G_period.edges():
            edge_key = (u, v) if (u, v) in period_edges else (v, u)
            if edge_key not in period_edges:
                edges_to_remove.append((u, v))
        
        for u, v in edges_to_remove:
            G_period.remove_edge(u, v)
        
        print(f"    ✓ Found {G_period.number_of_nodes()} nodes and {G_period.number_of_edges()} edges")
        period_graphs.append((period_name, G_period))

    # Create figure with subplots (number depends on number of periods)
    n_periods = len(time_periods)
    fig, axes = plt.subplots(1, n_periods, figsize=(TWO_COL_WIDTH_INCH * (2.5 if n_periods == 3 else 2.0), TWO_COL_WIDTH_INCH * 1.0))
    
    # Handle case where there's only one subplot (axes is not an array)
    if n_periods == 1:
        axes = [axes]

    # Use the fingerprint-based positions (calculated for all filtered nodes)
    pos = pos_2d

    # Draw each period's graph
    for ax_idx, (period_name, G_period) in enumerate(period_graphs):
        ax = axes[ax_idx]
        
        if G_period is None or G_period.number_of_nodes() == 0:
            period_title_map = {
                "<1970": "Before 1970",
                "≥1970": "After 1970",
            }
            display_title = period_title_map.get(period_name, period_name)
            ax.text(0.5, 0.5, f'No data\nfor {display_title}', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            continue
        
        print(f"  Drawing {period_name}...")
        
        # Get positions for nodes in this period
        pos_period = {node: pos[node] for node in G_period.nodes() if node in pos}
        
        # Draw edges - thickness scales with number of co-occurrences (weight)
        # The weight represents how many times two monomers appear together
        edges = G_period.edges()
        weights = [G_period[u][v].get('weight', 1) for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        # Scale edge widths: thicker edges for more co-occurrences
        # Normalize to range 0.3 to 6.0 based on weight (number of reactions)
        # Larger range makes differences more visible
        if max_weight > 1:
            edge_widths = [0.3 + (w / max_weight) * 5.7 for w in weights]
        else:
            edge_widths = [0.3] * len(weights)
        
        nx.draw_networkx_edges(G_period, pos_period, alpha=0.3, 
                               width=edge_widths, 
                               edge_color='gray', ax=ax)
        
        # Draw nodes - size based on degree, color based on class
        node_degrees_period = dict(G_period.degree())
        node_sizes = [node_degrees_period[node] * 100 for node in G_period.nodes()]
        
        # Get classes for nodes and assign colors
        # Get all unique classes in this period
        period_classes = set()
        for node in G_period.nodes():
            class_name = G_period.nodes[node].get('class_name', 'Other')
            period_classes.add(class_name)
        
        # Create color mapping (use global preferred_class_order)
        class_to_color = {}
        for i, class_name in enumerate(preferred_class_order):
            if class_name in all_classes_in_data:
                class_to_color[class_name] = SEQUENTIAL_COLORS[i % len(SEQUENTIAL_COLORS)]
        
        # Add colors for any remaining classes
        remaining_classes = sorted(all_classes_in_data - set(preferred_class_order))
        for i, class_name in enumerate(remaining_classes):
            class_to_color[class_name] = SEQUENTIAL_COLORS[(len(preferred_class_order) + i) % len(SEQUENTIAL_COLORS)]
        
        # Get node colors
        node_colors = [class_to_color.get(G_period.nodes[node].get('class_name', 'Other'), '#CCCCCC') 
                       for node in G_period.nodes()]
        
        nx.draw_networkx_nodes(G_period, pos_period, node_size=node_sizes, 
                               node_color=node_colors, 
                               alpha=0.8, ax=ax)
        
        # No labels - nodes are colored by class instead
        
        # Map period names to display titles
        period_title_map = {
            "<1970": "Before 1970",
            "≥1970": "After 1970",
        }
        display_title = period_title_map.get(period_name, period_name)
        
        ax.set_title(display_title, fontsize=18, pad=10)
        ax.axis('off')
        
        # Add legend for classes (only in first subplot to avoid repetition)
        if ax_idx == 0:
            # Create legend entries for all classes that appear in any period
            legend_elements = []
            for class_name in preferred_class_order:
                if class_name in all_classes_in_data:
                    color = class_to_color[class_name]
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                      markerfacecolor=color, markersize=8, 
                                                      label=class_name))
            # Add any remaining classes
            for class_name in remaining_classes:
                color = class_to_color[class_name]
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                  markerfacecolor=color, markersize=8, 
                                                  label=class_name))
            
            if legend_elements:
                ax.legend(
                    handles=legend_elements,
                    loc='upper left',
                    fontsize=14,          # increased legend font size
                    frameon=False,
                    bbox_to_anchor=(1.02, 1),
                )

    # After drawing, print monomers shown in this configuration, sorted by datapoint count
    print("\n  Monomers shown in this configuration (sorted by datapoint count):")
    for period_name, G_period in period_graphs:
        if G_period is None or G_period.number_of_nodes() == 0:
            continue
        print(f"\n    Period: {period_name}")
        # Get nodes in this period and sort by datapoint count
        nodes_in_period = list(G_period.nodes())
        nodes_sorted = sorted(
            nodes_in_period,
            key=lambda n: monomer_datapoint_counts.get(n, 0),
            reverse=True,
        )
        for node in nodes_sorted:
            name = G_period.nodes[node].get("name", node)
            class_name = G_period.nodes[node].get("class_name", "Unknown")
            count = monomer_datapoint_counts.get(node, 0)
            print(f"      {name}  |  class={class_name}  |  datapoints={count}")

    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    # Save the plot with configuration name
    output_path = output_dir / f'monomer_network_{config_name}.pdf'
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"\n✓ Plot saved to: {output_path}")
    
    # Also save as PNG
    output_path_png = output_dir / f'monomer_network_{config_name}.png'
    fig.savefig(output_path_png, bbox_inches='tight', dpi=300)
    print(f"✓ Plot saved to: {output_path_png}")
    
    plt.close(fig)  # Close figure to free memory

# Print some statistics
print(f"\nNetwork Statistics (Original):")
print(f"  Number of nodes: {G.number_of_nodes()}")
print(f"  Number of edges: {G.number_of_edges()}")
print(f"  Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

print(f"\nNetwork Statistics (Filtered, ≥{min_occurrences} connections):")
print(f"  Number of nodes: {G_filtered.number_of_nodes()}")
print(f"  Number of edges: {G_filtered.number_of_edges()}")
if G_filtered.number_of_nodes() > 0:
    print(f"  Average degree: {sum(dict(G_filtered.degree()).values()) / G_filtered.number_of_nodes():.2f}")

# Find most connected monomers (from filtered graph)
degrees_filtered = dict(G_filtered.degree())
top_monomers = sorted(degrees_filtered.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"\nTop 10 most connected monomers (in filtered network):")
for i, (smiles, degree) in enumerate(top_monomers, 1):
    name = G_filtered.nodes[smiles].get('name', smiles[:30])
    print(f"  {i}. {name} (degree: {degree})")

# Try to show plot, but don't fail if display is not available
try:
    plt.show()
except Exception as e:
    print(f"\nNote: Could not display plot interactively: {e}")
    print("Plots have been saved successfully.")
