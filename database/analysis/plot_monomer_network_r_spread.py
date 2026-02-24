"""
Monomer network plot variant: edge width = spread of r-product values.

Same data and layout as plot_monomer_network.py, but line thickness indicates
the spread of r-product (r1*r2) for that monomer pair: if a pair has values
e.g. from 1 to 100 (high spread), the edge is thick; if values are similar
(low spread), the edge is thin.
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
# Go three levels up from database/analysis to project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'copol_prediction' / 'analysis'))
from plot_config import (
    SEQUENTIAL_COLORS,
    TWO_COL_WIDTH_INCH
)

# Load data
data_path = Path(__file__).parent.parent.parent / 'copol_prediction' / 'processed_data.csv'
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
    """Extract DOI from a source URL or string."""
    if source_value is None or (isinstance(source_value, float) and pd.isna(source_value)):
        return None
    source_str = str(source_value).strip()
    if not source_str or source_str.lower() == "nan":
        return None
    source_str = source_str.replace("https://doi.org/", "").replace("http://doi.org/", "")
    if source_str.startswith("10."):
        return source_str.split()[0].strip().rstrip(".,;)")
    m = _DOI_RE.search(source_str)
    if m:
        return m.group(1).rstrip(".,;)")
    return None


def get_publication_year_from_crossref(doi):
    """Get publication year from Crossref API. Results are cached."""
    if doi in publication_years_cache:
        cached_value = publication_years_cache[doi]
        return cached_value if cached_value is not None else None
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
                    cache_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(cache_file, 'w') as f:
                        json.dump(publication_years_cache, f, indent=2)
                    time.sleep(0.1)
                    return year
        elif response.status_code == 404:
            publication_years_cache[doi] = None
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(publication_years_cache, f, indent=2)
            none_count = sum(1 for v in publication_years_cache.values() if v is None)
            if none_count <= 10:
                print(f"  Warning: DOI not found (404): {doi}")
            elif none_count == 11:
                print(f"  ... (suppressing further 404 warnings)")
        else:
            if response.status_code != 429:
                print(f"  Warning: Crossref returned status {response.status_code} for DOI {doi}")
    except Exception as e:
        print(f"  Warning: Error fetching year for DOI {doi}: {e}")
    return None


def _norm_name(x):
    return "" if pd.isna(x) else str(x).strip().lower()


def _norm_smiles(x):
    return "" if pd.isna(x) else str(x).strip()


def _has_any(s, patterns):
    return any(p in s for p in patterns)


def _has_double_bond(smi):
    if not smi:
        return False
    if "C=C" in smi or "(=C" in smi or "=C(" in smi:
        return True
    return bool(re.search(r"C=.*=?.*C", smi)) or ("=" in smi)


def classify_monomer(monomer_name, monomer_smiles):
    """9-class monomer classification scheme."""
    name = _norm_name(monomer_name)
    smi = _norm_smiles(monomer_smiles)
    if "acrylonitrile" in name or "methacrylonitrile" in name or _has_any(smi, ["C=CC#N", "C=C(C)C#N"]):
        return "(Meth)acrylonitriles"
    if any(k in name for k in ["maleic", "maleate", "fumar", "fumarate", "itaconic", "itaconate", "aconitate"]):
        return "Anhydrides/Diacids"
    if (
        ("methacry" in name)
        or ("acryl" in name and "amide" not in name and "nitrile" not in name)
        or "acrylic acid" in name
        or _has_any(smi, ["C=CC(=O)O", "C=CC(=O)OC", "C=CC(=O)[O-]", "C=C(C)C(=O)O", "C=C(C)C(=O)OC", "C=C(C)C(=O)[O-]"])
    ):
        return "(Meth)acrylates"
    if "acrylamide" in name or "methacrylamide" in name or "maleimide" in name or _has_any(smi, ["C=CC(=O)N", "C=C(C)C(=O)N"]):
        return "(Meth)acrylamides/imides"
    if "styrene" in name or "methylstyrene" in name or "chlorostyrene" in name or "methoxystyrene" in name or "styrene sulfonate" in name or _has_any(smi, ["C=Cc1ccccc1", "C=CC1=CC=CC=C1"]):
        return "Styrenics"
    if "butadiene" in name or "isoprene" in name or "chloroprene" in name or "diene" in name or _has_any(smi, ["C=CC=C", "C=C-C=C"]):
        return "Conjugated Dienes"
    if "vinyl" in name or re.search(r"\b\d*-?vinyl", name):
        return "Vinyl Derivatives"
    if (
        name in ["ethylene", "propylene", "propene", "isobutylene"]
        or re.search(r"\b\d+-?(hexene|octene)\b", name)
        or "tetrafluoroethylene" in name
        or "chlorotrifluoroethylene" in name
        or ("ethylene" in name and any(k in name for k in ["fluoro", "chloro", "trifluoro", "tetrafluoro"]))
        or (_has_double_bond(smi) and "c1ccccc1" not in smi.lower())
    ):
        return "Olefins"
    return "Other"


# --- Build graph and collect r-product values per edge ---
print(f"\nLoaded {len(df)} entries")
print("Extracting monomer pairs, publication years, and r-product values...")

G = nx.Graph()
monomer_years = defaultdict(list)
edge_counts = Counter()
edge_years = defaultdict(list)
edge_r_product_values = defaultdict(list)  # r-product (r1*r2) per edge
monomer_classes = {}
monomer_datapoint_counts = Counter()

# Ensure we have r-product: use r1r2 if present, else constant_1 * constant_2
if 'r1r2' not in df.columns and 'constant_1' in df.columns and 'constant_2' in df.columns:
    df['r1r2'] = pd.to_numeric(df['constant_1'], errors='coerce') * pd.to_numeric(df['constant_2'], errors='coerce')

unique_dois = set()
for idx, row in df.iterrows():
    source_val = _first_nonempty(row.get("source"), row.get("original_source"))
    doi = extract_doi_from_source(source_val)
    if doi:
        unique_dois.add(doi)

print(f"  Found {len(unique_dois)} unique DOIs")
print(f"  Fetching years (using cache when available)...")
for i, doi in enumerate(unique_dois, 1):
    if i % 50 == 0:
        print(f"    Progress: {i}/{len(unique_dois)}")
    get_publication_year_from_crossref(doi)

successful_years = sum(1 for v in publication_years_cache.values() if v is not None and v is not False)
failed_years = sum(1 for v in publication_years_cache.values() if v is None)
print(f"\n  Summary: {successful_years} DOIs with years found, {failed_years} DOIs not found (404)")

print("  Building graph and collecting r-product per edge...")
for idx, row in df.iterrows():
    monomer1 = row.get('monomer1_smiles')
    monomer2 = row.get('monomer2_smiles')
    monomer1_name = row.get('monomer1_name', monomer1)
    monomer2_name = row.get('monomer2_name', monomer2)

    if pd.isna(monomer1) or pd.isna(monomer2):
        continue

    if monomer1 not in monomer_classes:
        monomer_classes[monomer1] = classify_monomer(monomer1_name, monomer1)
    if monomer2 not in monomer_classes:
        monomer_classes[monomer2] = classify_monomer(monomer2_name, monomer2)

    source_val = _first_nonempty(row.get("source"), row.get("original_source"))
    doi = extract_doi_from_source(source_val)
    year = publication_years_cache.get(doi) if doi else None

    if monomer1 not in G:
        G.add_node(monomer1, name=monomer1_name, class_name=monomer_classes[monomer1])
    if monomer2 not in G:
        G.add_node(monomer2, name=monomer2_name, class_name=monomer_classes[monomer2])

    if year:
        monomer_years[monomer1].append(year)
        monomer_years[monomer2].append(year)

    edge_key = (monomer1, monomer2)
    if G.has_edge(monomer1, monomer2):
        G[monomer1][monomer2]['weight'] += 1
    else:
        G.add_edge(monomer1, monomer2, weight=1)

    if year:
        edge_years[edge_key].append(year)

    # Collect r-product for this edge
    r12 = row.get('r1r2')
    if pd.notna(r12):
        try:
            val = float(r12)
            if val > 0:
                edge_r_product_values[edge_key].append(val)
        except (TypeError, ValueError):
            pass

    edge_counts[edge_key] += 1
    monomer_datapoint_counts[monomer1] += 1
    monomer_datapoint_counts[monomer2] += 1

# Compute r-product spread per edge: ratio max/min (high = large spread, e.g. 1 vs 100 -> 100),
# then transform with log10(ratio + 1) for plotting so extreme ratios don't dominate.
edge_r_product_spread = {}
for edge_key, vals in edge_r_product_values.items():
    if len(vals) < 2:
        edge_r_product_spread[edge_key] = 0.0
        continue
    mn, mx = min(vals), max(vals)
    if mn <= 0:
        edge_r_product_spread[edge_key] = 0.0
        continue
    ratio = mx / mn
    edge_r_product_spread[edge_key] = np.log10(ratio + 1.0)

max_spread = max(edge_r_product_spread.values()) if edge_r_product_spread else 1.0
print(f"  ✓ Found {G.number_of_nodes()} unique monomers, {G.number_of_edges()} unique monomer pairs")
print(f"  ✓ r-product spread: {len([s for s in edge_r_product_spread.values() if s > 0])} edges with spread > 0, max log10(ratio+1) = {max_spread:.2f}")

# Filter: only nodes with at least N connections
min_occurrences = 5
print(f"\nFiltering: Only showing monomers that appear at least {min_occurrences} times...")
node_degrees = dict(G.degree())
frequent_nodes = {node for node, degree in node_degrees.items() if degree >= min_occurrences}
G_filtered = G.subgraph(frequent_nodes).copy()
print(f"  ✓ Filtered to {G_filtered.number_of_nodes()} nodes, {G_filtered.number_of_edges()} edges")

# Fingerprints and 2D layout (same as original script)
def calculate_morgan_fingerprint(smiles, radius=2, n_bits=2048):
    if not RDKIT_AVAILABLE:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except Exception:
        return None

node_list = list(G_filtered.nodes())
fingerprints = []
valid_nodes = []
for i, smiles in enumerate(node_list):
    if i % 50 == 0 and i > 0:
        print(f"    Fingerprint progress: {i}/{len(node_list)}")
    fp = calculate_morgan_fingerprint(smiles)
    if fp is not None:
        fingerprints.append(fp)
        valid_nodes.append(smiles)

pos_2d = {}
if fingerprints and SKLEARN_AVAILABLE:
    fingerprints = np.array(fingerprints)
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(fingerprints) - 1))
        coords_2d = tsne.fit_transform(fingerprints)
        for i, smiles in enumerate(valid_nodes):
            pos_2d[smiles] = (coords_2d[i, 0], coords_2d[i, 1])
        print(f"  ✓ 2D positions via t-SNE")
    except Exception:
        try:
            pca = PCA(n_components=2, random_state=42)
            coords_2d = pca.fit_transform(fingerprints)
            for i, smiles in enumerate(valid_nodes):
                pos_2d[smiles] = (coords_2d[i, 0], coords_2d[i, 1])
            print(f"  ✓ 2D positions via PCA")
        except Exception:
            pos_2d = None
else:
    pos_2d = None

if pos_2d is None or len(pos_2d) < len(node_list):
    if pos_2d is None:
        pos_2d = {}
    missing_nodes = set(node_list) - set(pos_2d.keys())
    if missing_nodes:
        G_missing = G_filtered.subgraph(missing_nodes)
        if G_missing.number_of_nodes() > 0:
            try:
                pos_missing = nx.kamada_kawai_layout(G_missing, weight='weight', seed=42)
            except Exception:
                pos_missing = nx.spring_layout(G_missing, k=2, iterations=100, seed=42)
            pos_2d.update(pos_missing)
    if len(pos_2d) < len(node_list):
        try:
            pos_2d = nx.kamada_kawai_layout(G_filtered, weight='weight', seed=42)
        except Exception:
            pos_2d = nx.spring_layout(G_filtered, k=2, iterations=100, seed=42)
        print("  ✓ Using network layout for all nodes")

# Time period config (same as original)
time_period_configs = [
    {'name': 'two_periods_1970', 'periods': [("<1970", None, 1969), ("≥1970", 1970, None)]},
]
preferred_class_order = [
    '(Meth)acrylonitriles', 'Anhydrides/Diacids', '(Meth)acrylates', '(Meth)acrylamides/imides',
    'Styrenics', 'Conjugated Dienes', 'Vinyl Derivatives', 'Olefins', 'Other',
]
all_classes_in_data = set(G_filtered.nodes[node].get('class_name', 'Other') for node in G_filtered.nodes())

for config in time_period_configs:
    config_name = config['name']
    time_periods = config['periods']
    period_graphs = []
    for period_name, start_year, end_year in time_periods:
        period_edges = set()
        period_nodes = set()
        for edge_key, years in edge_years.items():
            for year in years:
                if start_year is None and end_year is not None and year <= end_year:
                    period_edges.add(edge_key)
                    period_nodes.add(edge_key[0])
                    period_nodes.add(edge_key[1])
                    break
                elif start_year is not None and end_year is None and year >= start_year:
                    period_edges.add(edge_key)
                    period_nodes.add(edge_key[0])
                    period_nodes.add(edge_key[1])
                    break
                elif start_year is not None and end_year is not None and start_year <= year <= end_year:
                    period_edges.add(edge_key)
                    period_nodes.add(edge_key[0])
                    period_nodes.add(edge_key[1])
                    break
        period_nodes &= frequent_nodes
        if len(period_nodes) == 0:
            period_graphs.append((period_name, None))
            continue
        G_period = G_filtered.subgraph(period_nodes).copy()
        for u, v in list(G_period.edges()):
            edge_key = (u, v) if (u, v) in period_edges else (v, u)
            if edge_key not in period_edges:
                G_period.remove_edge(u, v)
        period_graphs.append((period_name, G_period))

    n_periods = len(time_periods)
    fig, axes = plt.subplots(1, n_periods, figsize=(TWO_COL_WIDTH_INCH * (2.5 if n_periods == 3 else 2.0), TWO_COL_WIDTH_INCH * 1.0))
    if n_periods == 1:
        axes = [axes]
    pos = pos_2d

    for ax_idx, (period_name, G_period) in enumerate(period_graphs):
        ax = axes[ax_idx]
        if G_period is None or G_period.number_of_nodes() == 0:
            display_title = {"<1970": "Before 1970", "≥1970": "After 1970"}.get(period_name, period_name)
            ax.text(0.5, 0.5, f'No data\nfor {display_title}', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.axis('off')
            continue

        pos_period = {node: pos[node] for node in G_period.nodes() if node in pos}

        # Edge widths from r-product spread (thick = high spread, thin = low)
        # First, collect spreads for edges in this period and calculate max_spread for this period
        period_spreads = []
        for u, v in G_period.edges():
            edge_key = (u, v) if (u, v) in edge_r_product_spread else (v, u)
            spread = edge_r_product_spread.get(edge_key, 0.0)
            if spread > 0:  # Only include edges with actual spread
                period_spreads.append(spread)
        
        max_spread_period = max(period_spreads) if period_spreads else 1.0
        print(f"    Period {period_name}: {len(period_spreads)} edges with spread > 0, max_spread = {max_spread_period:.2f}")
        
        # Use power scaling so differences are more pronounced: thin stays very thin, thick gets much thicker
        edges = list(G_period.edges())
        edge_spreads_period = []
        edge_widths = []
        for u, v in edges:
            edge_key = (u, v) if (u, v) in edge_r_product_spread else (v, u)
            spread = edge_r_product_spread.get(edge_key, 0.0)
            edge_spreads_period.append(spread)
            if max_spread_period > 0 and spread > 0:
                t = (spread / max_spread_period)
                t = t ** 1.5  # power so low spread stays thin, high spread stands out
                w = 0.2 + t * 9.0  # range 0.2 (thin) to 9.2 (thick)
            else:
                w = 0.2  # Very thin for edges with no spread
            edge_widths.append(w)

        # Identify the three thickest edges (highest r-spread) to highlight
        edge_spread_pairs = list(zip(edges, edge_spreads_period, edge_widths))
        edge_spread_pairs.sort(key=lambda x: x[1], reverse=True)
        top3_edges = {tuple(sorted((u, v))) for (u, v), _, _ in edge_spread_pairs[:3]}
        
        # Separate edges into regular (gray) and highlighted (#393b74)
        regular_edges = []
        regular_widths = []
        highlighted_edges = []
        highlighted_widths = []
        
        for (u, v), spread, width in zip(edges, edge_spreads_period, edge_widths):
            edge_tuple = tuple(sorted((u, v)))
            if edge_tuple in top3_edges:
                highlighted_edges.append((u, v))
                highlighted_widths.append(width)
            else:
                regular_edges.append((u, v))
                regular_widths.append(width)
        
        # Draw regular edges in gray
        if regular_edges:
            nx.draw_networkx_edges(G_period, pos_period, edgelist=regular_edges, 
                                   alpha=0.3, width=regular_widths, 
                                   edge_color='gray', ax=ax)
        
        # Draw highlighted edges in #393b74 (thicker and more opaque)
        if highlighted_edges:
            nx.draw_networkx_edges(G_period, pos_period, edgelist=highlighted_edges, 
                                   alpha=0.7, width=highlighted_widths, 
                                   edge_color='#393b74', ax=ax)

        node_degrees_period = dict(G_period.degree())
        node_sizes = [node_degrees_period[node] * 100 for node in G_period.nodes()]
        class_to_color = {}
        for i, class_name in enumerate(preferred_class_order):
            if class_name in all_classes_in_data:
                class_to_color[class_name] = SEQUENTIAL_COLORS[i % len(SEQUENTIAL_COLORS)]
        for class_name in sorted(all_classes_in_data - set(preferred_class_order)):
            class_to_color[class_name] = SEQUENTIAL_COLORS[(len(preferred_class_order) + len(class_to_color)) % len(SEQUENTIAL_COLORS)]
        node_colors = [class_to_color.get(G_period.nodes[node].get('class_name', 'Other'), '#CCCCCC') for node in G_period.nodes()]
        nx.draw_networkx_nodes(G_period, pos_period, node_size=node_sizes, node_color=node_colors, alpha=0.8, ax=ax)

        display_title = {"<1970": "Before 1970", "≥1970": "After 1970"}.get(period_name, period_name)
        ax.set_title(display_title, fontsize=18, pad=10)
        ax.axis('off')

        if ax_idx == 0:
            legend_elements = []
            # Node class legend
            for class_name in preferred_class_order:
                if class_name in all_classes_in_data:
                    legend_elements.append(
                        plt.Line2D(
                            [0],
                            [0],
                            marker='o',
                            color='w',
                            markerfacecolor=class_to_color[class_name],
                            markersize=8,
                            label=class_name,
                        )
                    )
            for class_name in sorted(all_classes_in_data - set(preferred_class_order)):
                legend_elements.append(
                    plt.Line2D(
                        [0],
                        [0],
                        marker='o',
                        color='w',
                        markerfacecolor=class_to_color[class_name],
                        markersize=8,
                        label=class_name,
                    )
                )

            # Separator between node legend and edge legend
            legend_elements.append(
                plt.Line2D([0], [0], linestyle='none', label='')
            )

            # Short textual explanation of edge spread
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    linestyle='none',
                    color='w',
                    label='Edge width ∝ r-product spread',
                )
            )

            # Edge thickness legend for r-product spread
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    linestyle='-',
                    color='gray',
                    linewidth=0.5,
                    label='Thin edge: similar r-products (low spread)',
                )
            )
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    linestyle='-',
                    color='gray',
                    linewidth=3.0,
                    label='Thicker edge: varying r-products (higher spread)',
                )
            )
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    linestyle='-',
                    color='#393b74',
                    linewidth=4.0,
                    label='Highlighted: 3 pairs with largest spread',
                )
            )

            if legend_elements:
                ax.legend(
                    handles=legend_elements,
                    loc='upper left',
                    fontsize=11,
                    frameon=False,
                    bbox_to_anchor=(1.02, 1),
                )

    plt.tight_layout()
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    base_name = f'monomer_network_r_spread_{config_name}'
    fig.savefig(output_dir / f'{base_name}.pdf', bbox_inches='tight', dpi=300)
    fig.savefig(output_dir / f'{base_name}.png', bbox_inches='tight', dpi=300)
    print(f"\n✓ Saved {output_dir / base_name}.pdf and .png")
    plt.close(fig)

print("\nDone. Edge width = r-product spread (thick = large range e.g. 1–100, thin = similar values).")
