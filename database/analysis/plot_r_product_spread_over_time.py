"""
Plot how the r-product spread changes over time.

Definition (consistent with `plot_monomer_network_r_spread.py`):
- For each monomer pair, and for each year with >= 2 data points,
  we compute the r-product spread as:

      spread = log10( max(r1*r2) / min(r1*r2) + 1 )

- For a given time bin (10-year bins), we collect all such spread values
  from all monomer pairs and summarize them with:
    - minimum spread
    - median spread
    - maximum spread

Three lines are plotted over time: min, median, max r-product spread.
Only years/edge-pairs with at least 2 r-product values contribute (no
single-point deltas).
"""

from pathlib import Path
import sys
from collections import defaultdict
import json
import time
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add copol_prediction to path to import plot_config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "copol_prediction" / "analysis"))
from plot_config import SEQUENTIAL_COLORS, TWO_COL_WIDTH_INCH


# ---------------------------------------------------------------------------
# Helper functions for DOI/year handling (copied/trimmed from other scripts)
# ---------------------------------------------------------------------------

cache_file = Path(__file__).parent / "publication_years_cache.json"
publication_years_cache: dict[str, int | None] = {}
if cache_file.exists():
    with open(cache_file, "r") as f:
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


def get_publication_year_from_crossref(doi: str | None) -> int | None:
    """Get publication year from Crossref API. Results are cached.

    Note: In this project we typically rely on the existing cache and
    rarely need to hit the API again.
    """
    if not doi:
        return None

    # Check cache first (including None values for 404s)
    if doi in publication_years_cache:
        cached_value = publication_years_cache[doi]
        return cached_value if cached_value is not None else None

    # Try fetching (best effort). If network is disabled this will just fail silently.
    import requests

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
                    with open(cache_file, "w") as f:
                        json.dump(publication_years_cache, f, indent=2)
                    time.sleep(0.1)
                    return year
        elif response.status_code == 404:
            publication_years_cache[doi] = None
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(publication_years_cache, f, indent=2)
    except Exception:
        # For this analysis script it's fine to fail gracefully.
        pass

    return None


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

data_path = Path(__file__).parent.parent.parent / "copol_prediction" / "processed_data.csv"
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)

# Ensure we have r-product: use r1r2 if present, else constant_1 * constant_2
if "r1r2" not in df.columns and "constant_1" in df.columns and "constant_2" in df.columns:
    df["r1r2"] = pd.to_numeric(df["constant_1"], errors="coerce") * pd.to_numeric(
        df["constant_2"], errors="coerce"
    )

print(f"\nLoaded {len(df)} entries")
print("Extracting r-product values and publication years...")

# First pass: collect unique DOIs and ensure we have years in the cache
unique_dois: set[str] = set()
for _, row in df.iterrows():
    source_val = _first_nonempty(row.get("source"), row.get("original_source"))
    doi = extract_doi_from_source(source_val)
    if doi:
        unique_dois.add(doi)

print(f"  Found {len(unique_dois)} unique DOIs")
print("  Fetching years (using cache when available)...")
for i, doi in enumerate(unique_dois, 1):
    if i % 50 == 0:
        print(f"    Progress: {i}/{len(unique_dois)}")
    get_publication_year_from_crossref(doi)

successful_years = sum(
    1 for v in publication_years_cache.values() if v is not None and v is not False
)
failed_years = sum(1 for v in publication_years_cache.values() if v is None)
print(
    f"\n  Summary: {successful_years} DOIs with years found, "
    f"{failed_years} DOIs not found (404 or missing)."
)


# ---------------------------------------------------------------------------
# Collect r-product values per edge and year
# ---------------------------------------------------------------------------

# r_product_by_edge_year[(edge_key)][year] -> list of r1r2 values
r_product_by_edge_year: dict[tuple[str, str], dict[int, list[float]]] = defaultdict(
    lambda: defaultdict(list)
)

for _, row in df.iterrows():
    monomer1 = row.get("monomer1_smiles")
    monomer2 = row.get("monomer2_smiles")

    if pd.isna(monomer1) or pd.isna(monomer2):
        continue

    # Normalize edge key (order-independent)
    edge_key = tuple(sorted((str(monomer1), str(monomer2))))

    # Year
    source_val = _first_nonempty(row.get("source"), row.get("original_source"))
    doi = extract_doi_from_source(source_val)
    year = publication_years_cache.get(doi) if doi else None
    if year is None:
        continue

    # r-product
    r12 = row.get("r1r2")
    if pd.notna(r12):
        try:
            val = float(r12)
            if val > 0:
                r_product_by_edge_year[edge_key][int(year)].append(val)
        except (TypeError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Compute r-product spread per edge per year (same definition as network plot)
# ---------------------------------------------------------------------------

# For each edge and year, we only consider years with >= 2 r-product values.
# Spread is log10(max/min + 1) for that pair in that year.
spreads_by_year: dict[int, list[float]] = defaultdict(list)

for edge_key, year_dict in r_product_by_edge_year.items():
    for year, values in year_dict.items():
        if len(values) < 2:
            # Only one value for this pair/year: no spread information
            continue
        mn, mx = min(values), max(values)
        if mn <= 0:
            continue
        ratio = mx / mn
        spread = np.log10(ratio + 1.0)
        if spread > 0:
            spreads_by_year[year].append(spread)

if not spreads_by_year:
    print("⚠ No r-product spread values with Δ>0 and >=2 points per pair/year found.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Group by 10-year bins and compute min/median/max per bin
# ---------------------------------------------------------------------------

all_years = sorted(spreads_by_year.keys())
min_year, max_year = min(all_years), max(all_years)
print(f"\nYear range with spread data: {min_year}–{max_year}")

bin_size = 10
spreads_by_bin: dict[int, list[float]] = defaultdict(list)

for year, spreads in spreads_by_year.items():
    bin_start = (year // bin_size) * bin_size  # e.g. 1973 -> 1970
    spreads_by_bin[bin_start].extend(spreads)

bins_sorted = sorted(spreads_by_bin.keys())

bin_maxs = [float(np.max(spreads_by_bin[b])) for b in bins_sorted]
bin_counts = [len(spreads_by_bin[b]) for b in bins_sorted]

print(f"  ✓ {len(bins_sorted)} time bins with data")
print(
    f"  Max spread range across bins: "
    f"{min(bin_maxs):.2f} – {max(bin_maxs):.2f} (log10(max/min + 1))"
)


# ---------------------------------------------------------------------------
# Plot: min / median / max r-product spread over time
# ---------------------------------------------------------------------------

print("\nCreating r-product spread over time plot...")

fig, ax = plt.subplots(
    figsize=(TWO_COL_WIDTH_INCH, TWO_COL_WIDTH_INCH * 0.7)
)

bins_as_years = bins_sorted

# Single line: max spread per time bin
line_max, = ax.plot(
    bins_as_years,
    bin_maxs,
    linestyle="-",
    marker="o",
    color=SEQUENTIAL_COLORS[0],
    linewidth=2.0,
    markersize=6,
    label="Max spread per bin",
)

# Highlight the top 3 bins with the highest max spread
if len(bin_maxs) >= 1:
    # indices of bins sorted by descending max spread
    top_indices = np.argsort(bin_maxs)[::-1][: min(3, len(bin_maxs))]
    top_x = [bins_as_years[i] for i in top_indices]
    top_y = [bin_maxs[i] for i in top_indices]
    ax.scatter(
        top_x,
        top_y,
        s=80,
        color=SEQUENTIAL_COLORS[1],
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
        label="Top 3 bins by max spread",
    )

ax.set_xlabel("Year (10-year bins)", fontsize=14)
ax.set_ylabel("r-product spread  log₁₀(max/min + 1)", fontsize=14)
ax.set_title("r-product Spread Over Time", fontsize=16, pad=10)
ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.3)
ax.legend(loc="best", fontsize=12)

# Optional: annotate how many pairs per bin on a secondary y-axis
ax2 = ax.twinx()
ax2.bar(
    bins_as_years,
    bin_counts,
    width=bin_size * 0.7,
    alpha=0.15,
    color="gray",
)
ax2.set_ylabel("Number of pair-year spreads", fontsize=11, color="gray")
ax2.tick_params(axis="y", labelcolor="gray")

plt.tight_layout()

output_dir = Path(__file__).parent / "figures"
output_dir.mkdir(exist_ok=True)

pdf_path = output_dir / "r_product_spread_over_time.pdf"
png_path = output_dir / "r_product_spread_over_time.png"

fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
fig.savefig(png_path, bbox_inches="tight", dpi=300)

print(f"\n✓ Saved plot to:\n  {pdf_path}\n  {png_path}")

plt.close(fig)

