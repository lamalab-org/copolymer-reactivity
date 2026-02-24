"""
Share of monomer categories over time (same style as polymerization types/methods).

Uses the same data and year bins as plot_polymerization_trends.py, and the same
monomer classification (9 classes) as plot_monomer_network.py. Stacked area: share
of each monomer category per 5-year period.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import re
import json
import requests
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'copol_prediction' / 'analysis'))
from plot_config import SEQUENTIAL_COLORS, TWO_COL_WIDTH_INCH

# --- Same data and cache as other evaluation scripts ---
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

# --- Monomer classification (same 9 classes as plot_monomer_network.py) ---
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

# --- Year bins (5-year, same as polymerization_trends) ---
def create_year_bin(year, bin_size=5):
    if year is None:
        return None
    if bin_size <= 1:
        return str(year)
    bin_start = (year // bin_size) * bin_size
    bin_end = bin_start + bin_size - 1
    return f"{bin_start}-{bin_end}"

def bin_to_numeric(bin_label):
    parts = bin_label.split('-')
    if len(parts) == 2:
        return (int(parts[0]) + int(parts[1])) / 2
    return int(parts[0])

# --- Fetch years for all DOIs ---
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

# --- Count monomer occurrences per (year_bin, class): each row contributes 1 to class of monomer1 and 1 to class of monomer2 ---
bin_size = 5
class_counts = defaultdict(lambda: defaultdict(float))

print("  Classifying monomers and counting per decade...")
for _, row in df.iterrows():
    doi = extract_doi_from_source(_first_nonempty(row.get("source"), row.get("original_source")))
    year = get_publication_year_from_crossref(doi) if doi else None
    if year is None:
        continue
    bin_label = create_year_bin(year, bin_size)
    if bin_label is None:
        continue
    m1_smi = row.get('monomer1_smiles')
    m2_smi = row.get('monomer2_smiles')
    m1_name = row.get('monomer1_name', m1_smi)
    m2_name = row.get('monomer2_name', m2_smi)
    if pd.isna(m1_smi) or pd.isna(m2_smi):
        continue
    c1 = classify_monomer(m1_name, m1_smi)
    c2 = classify_monomer(m2_name, m2_smi)
    class_counts[c1][bin_label] += 1
    class_counts[c2][bin_label] += 1

unique_bins = sorted(set(b for c in class_counts for b in class_counts[c].keys()))
if not unique_bins:
    print("  No binned data. Exiting.")
    sys.exit(0)

numeric_bins = [bin_to_numeric(b) for b in unique_bins]
total_per_bin = {}
for b in unique_bins:
    total_per_bin[b] = sum(class_counts[c].get(b, 0) for c in class_counts)

# Order and colors (same as network plot)
preferred_class_order = [
    '(Meth)acrylonitriles', 'Anhydrides/Diacids', '(Meth)acrylates', '(Meth)acrylamides/imides',
    'Styrenics', 'Conjugated Dienes', 'Vinyl Derivatives', 'Olefins', 'Other',
]
classes_in_data = set(class_counts.keys())
class_sorted = sorted(
    [c for c in preferred_class_order if c in classes_in_data],
    key=lambda c: sum(class_counts[c].values()),
    reverse=True,
)
class_sorted += sorted(classes_in_data - set(preferred_class_order), key=lambda c: sum(class_counts[c].values()), reverse=True)

# --- Plot: one stacked area (Monomer categories) ---
fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH_INCH * 1.2, TWO_COL_WIDTH_INCH * 0.9))
ax.set_title('Monomer categories (share per period)', fontsize=14, pad=10)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Share', fontsize=12)
ax.set_ylim(0, 1)
ax.grid(False)

proportions = []
labels = []
colors = []
for i, cat in enumerate(class_sorted):
    total_per_bin_list = [total_per_bin.get(b, 0) for b in unique_bins]
    counts = [class_counts[cat].get(b, 0) for b in unique_bins]
    prop = [c / t if t > 0 else 0 for c, t in zip(counts, total_per_bin_list)]
    proportions.append(prop)
    labels.append(cat)
    colors.append(SEQUENTIAL_COLORS[i % len(SEQUENTIAL_COLORS)])

ax.stackplot(numeric_bins, *proportions, labels=labels, colors=colors, alpha=0.85)
ax.legend(loc='lower right', fontsize=9, frameon=True, ncol=1)
ax.set_xlim(min(numeric_bins) - 2, max(numeric_bins) + 2)

plt.tight_layout()
out_dir = Path(__file__).parent / 'figures'
out_dir.mkdir(exist_ok=True)
base = out_dir / 'monomer_categories_trends'
fig.savefig(base.with_suffix('.pdf'), bbox_inches='tight', dpi=300)
fig.savefig(base.with_suffix('.png'), bbox_inches='tight', dpi=300)
print(f"\n✓ Saved {base}.pdf and .png")
plt.close(fig)
