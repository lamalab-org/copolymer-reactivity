"""
r-product over time for the 3 most frequent monomer pairs.

Scatter plot: x = decade (every 10 years), y = r-product (r1*r2). One subplot per pair, all three side by side.
Uses the same data and DOI/year cache as plot_monomer_network.py (processed_data.csv).
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re
import json
import requests
import time
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'copol_prediction' / 'analysis'))
from plot_config import SEQUENTIAL_COLORS, TWO_COL_WIDTH_INCH

data_path = Path(__file__).parent.parent.parent.parent / 'copol_prediction' / 'processed_data.csv'
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)

cache_file = Path(__file__).parent / 'publication_years_cache.json'
publication_years_cache = {}
if cache_file.exists():
    with open(cache_file, 'r') as f:
        publication_years_cache = json.load(f)
    print(f"  Loaded {len(publication_years_cache)} cached publication years")

def _first_nonempty(*values):
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
    if source_value is None or (isinstance(source_value, float) and pd.isna(source_value)):
        return None
    s = str(source_value).strip().replace("https://doi.org/", "").replace("http://doi.org/", "")
    if not s or s.lower() == "nan":
        return None
    if s.startswith("10."):
        return s.split()[0].strip().rstrip(".,;)")
    m = _DOI_RE.search(s)
    return m.group(1).rstrip(".,;)") if m else None

def get_publication_year_from_crossref(doi):
    if doi in publication_years_cache:
        v = publication_years_cache[doi]
        return v if v is not None else None
    if not doi:
        return None
    try:
        r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=10)
        if r.status_code == 200:
            date_parts = r.json().get("message", {}).get("issued", {}).get("date-parts", [])
            if date_parts and len(date_parts[0]) > 0:
                year = date_parts[0][0]
                publication_years_cache[doi] = year
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump(publication_years_cache, f, indent=2)
                time.sleep(0.1)
                return year
        elif r.status_code == 404:
            publication_years_cache[doi] = None
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'w') as f:
                json.dump(publication_years_cache, f, indent=2)
    except Exception as e:
        print(f"  Warning: {e}")
    return None

def pair_key(m1, m2):
    if pd.isna(m1) or pd.isna(m2):
        return None
    a, b = str(m1).strip(), str(m2).strip()
    return (min(a, b), max(a, b)) if a and b else None

# Fetch years for all DOIs
unique_dois = set()
for _, row in df.iterrows():
    doi = extract_doi_from_source(_first_nonempty(row.get("source"), row.get("original_source")))
    if doi:
        unique_dois.add(doi)
print(f"  Fetching years for {len(unique_dois)} DOIs...")
for i, doi in enumerate(unique_dois, 1):
    if i % 50 == 0:
        print(f"    {i}/{len(unique_dois)}")
    get_publication_year_from_crossref(doi)

# Count pair frequencies
pair_count = Counter()
pair_name = {}  # (m1,m2) -> (name1, name2) from first row
for _, row in df.iterrows():
    pk = pair_key(row.get('monomer1_smiles'), row.get('monomer2_smiles'))
    if pk is None:
        continue
    pair_count[pk] += 1
    if pk not in pair_name:
        n1 = row.get('monomer1_name', pk[0][:20])
        n2 = row.get('monomer2_name', pk[1][:20])
        pair_name[pk] = (str(n1)[:25], str(n2)[:25])

top3 = [pk for pk, _ in pair_count.most_common(3)]
if len(top3) < 3:
    print(f"  Only {len(top3)} pairs found; using all.")
    top3 = top3 or list(pair_count.keys())[:3]

# r-product: use r1r2 column or constant_1 * constant_2
if 'r1r2' not in df.columns and 'constant_1' in df.columns and 'constant_2' in df.columns:
    df['r1r2'] = pd.to_numeric(df['constant_1'], errors='coerce') * pd.to_numeric(df['constant_2'], errors='coerce')

def to_float(x):
    if pd.isna(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

def year_to_decade(year):
    """Bin year to decade (1965 -> 1960, 1973 -> 1970)."""
    if year is None:
        return None
    return (year // 10) * 10

# Collect (decade, r_product) per pair
data_by_pair = defaultdict(list)
for _, row in df.iterrows():
    pk = pair_key(row.get('monomer1_smiles'), row.get('monomer2_smiles'))
    if pk not in top3:
        continue
    doi = extract_doi_from_source(_first_nonempty(row.get("source"), row.get("original_source")))
    year = get_publication_year_from_crossref(doi) if doi else None
    if year is None:
        continue
    r_product = to_float(row.get('r1r2'))
    if r_product is not None and r_product >= 0:
        decade = year_to_decade(year)
        data_by_pair[pk].append({'decade': decade, 'r_product': r_product})

# Plot: up to 3 subplots side by side, scatter decade vs r-product
n_plots = min(3, len(top3)) or 1
fig, axes = plt.subplots(1, n_plots, figsize=(TWO_COL_WIDTH_INCH * 2.2, TWO_COL_WIDTH_INCH * 0.9))
if n_plots == 1:
    axes = [axes]
for i, pk in enumerate(top3):
    ax = axes[i]
    rows = data_by_pair.get(pk, [])
    if not rows:
        ax.set_title(pair_name.get(pk, (pk[0][:15], pk[1][:15])))
        ax.axis('off')
        continue
    decades = [r['decade'] for r in rows]
    r_products = [r['r_product'] for r in rows]
    ax.scatter(decades, r_products, c=SEQUENTIAL_COLORS[0], alpha=0.6, s=32, edgecolors='none', label='r-product')
    # Average r-product per decade (different color, larger point)
    count_by_decade = Counter(decades)
    sum_by_decade = defaultdict(float)
    for r in rows:
        sum_by_decade[r['decade']] += r['r_product']
    decades_uniq = sorted(sum_by_decade.keys())
    mean_r = [sum_by_decade[d] / count_by_decade[d] for d in decades_uniq]
    ax.scatter(decades_uniq, mean_r, c='#C41E3A', s=80, marker='o', edgecolors='white', linewidths=1.5, zorder=5, label='Mean')
    # Secondary y-axis: number of data points per decade
    count_by_decade = Counter(decades)
    decades_uniq = sorted(count_by_decade.keys())
    counts = [count_by_decade[d] for d in decades_uniq]
    ax2 = ax.twinx()
    ax2.bar(decades_uniq, counts, width=4, alpha=0.25, color=SEQUENTIAL_COLORS[2], label='N')
    ax2.set_ylabel("Number of data points", fontsize=10)
    ax2.tick_params(axis='y', labelsize=9)
    n1, n2 = pair_name.get(pk, (pk[0][:18], pk[1][:18]))
    ax.set_title(f"{n1} / {n2}", fontsize=11)
    ax.set_xlabel("Decade", fontsize=10)
    ax.set_ylabel("r-product (r₁·r₂)", fontsize=10)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

plt.suptitle("r-product by decade (3 most frequent monomer pairs)", fontsize=14, y=1.02)
plt.tight_layout()
out_dir = Path(__file__).parent / 'figures'
out_dir.mkdir(exist_ok=True)
base = out_dir / 'top3_pairs_r_over_time'
fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight', dpi=300)
fig.savefig(base.with_suffix('.png'), bbox_inches='tight', dpi=300)
print(f"\n✓ Saved {base}.pdf and .png")
plt.close(fig)
