"""
Create line plots showing how polymerization types and methods change over time.

Uses the same data as plot_monomer_network.py and groups data into 5-year bins
to show temporal trends.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from collections import Counter, defaultdict
import json
import re
import numpy as np

# Add copol_prediction to path to import plot_config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'copol_prediction' / 'analysis'))
from plot_config import (
    SEQUENTIAL_COLORS, 
    TWO_COL_WIDTH_INCH,
    setup_plot_style,
)

# Load data
data_path = Path(__file__).parent.parent.parent / 'copol_prediction' / 'processed_data.csv'
print(f"Loading data from: {data_path}")

setup_plot_style()

df = pd.read_csv(data_path)

def _get_publication_year_series(df: pd.DataFrame) -> pd.Series:
    """Use local `publication_year` column (no network)."""
    if "publication_year" not in df.columns:
        return pd.Series([], dtype="float64")
    return pd.to_numeric(df["publication_year"], errors="coerce")

# Grouping config: load from JSON files
_script_dir = Path(__file__).parent
_types_json = _script_dir / 'polymerization_types.json'
_methods_json = _script_dir / 'polymerization_methods.json'


def _group_key_to_display_name(key):
    """Convert group key to display name (e.g. 'radical_polymerization' -> 'Radical polymerization')."""
    return key.replace('_', ' ').title()


def _load_grouping_mapping(json_path):
    """Load a grouping JSON and return a dict: normalized_value -> display_name."""
    if not json_path.exists():
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mapping = {}
    for group_key, variants in data.items():
        display_name = _group_key_to_display_name(group_key)
        for v in variants:
            normalized = str(v).strip().lower()
            if normalized:
                mapping[normalized] = display_name
    return mapping


poly_type_to_group = _load_grouping_mapping(_types_json)
method_to_group = _load_grouping_mapping(_methods_json)

if poly_type_to_group:
    print(f"  Loaded {len(poly_type_to_group)} polymerization type mappings from {_types_json.name}")
else:
    print(f"  Warning: No polymerization_types.json found or empty; raw values will be used.")

if method_to_group:
    print(f"  Loaded {len(method_to_group)} method mappings from {_methods_json.name}")
else:
    print(f"  Warning: No polymerization_methods.json found or empty; raw values will be used.")

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
    raise RuntimeError("Crossref lookup disabled: use local `publication_year` column instead.")


def normalize_value(value):
    """Normalize a string value (lowercase, strip whitespace, handle NaN)."""
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if not s or s == "nan":
        return None
    return s


def group_polymerization_type(poly_type):
    """Map polymerization type to group using polymerization_types.json."""
    if not poly_type:
        return None
    normalized = poly_type.lower().strip()
    if normalized in poly_type_to_group:
        return poly_type_to_group[normalized]
    # Fallback: if value contains a known variant, use that group (longest match)
    best_match = None
    best_len = 0
    for variant, display_name in poly_type_to_group.items():
        if variant in normalized or normalized in variant:
            if len(variant) > best_len:
                best_len = len(variant)
                best_match = display_name
    if best_match:
        return best_match
    return poly_type.strip().title()


def group_method(method):
    """Map method to group using polymerization_methods.json."""
    if not method:
        return None
    normalized = method.lower().strip()
    if normalized in method_to_group:
        return method_to_group[normalized]
    # Fallback: if value contains a known variant, use that group (longest match)
    best_match = None
    best_len = 0
    for variant, display_name in method_to_group.items():
        if variant in normalized or normalized in variant:
            if len(variant) > best_len:
                best_len = len(variant)
                best_match = display_name
    if best_match:
        return best_match
    return method.strip().title()


def create_year_bin(year, bin_size=1):
    """Create a bin label for a year (e.g., bin_size=1 -> '1965'; bin_size=5 -> '1965-1969')."""
    if year is None:
        return None
    if bin_size <= 1:
        return str(year)
    bin_start = (year // bin_size) * bin_size
    bin_end = bin_start + bin_size - 1
    return f"{bin_start}-{bin_end}"


# Extract publication years
print(f"\nLoaded {len(df)} entries")
print("Extracting publication years...")

print("  Using local publication_year column (no Crossref calls).")
pub_year_series = _get_publication_year_series(df)
if pub_year_series.empty:
    raise RuntimeError("Missing `publication_year` column in processed_data.csv (required for polymerization trends).")

# Second pass: collect data with years, polymerization types, and methods
print("\n  Collecting polymerization types and methods with years...")

# First, collect all unique values BEFORE grouping
all_poly_types = Counter()
all_methods = Counter()

for idx, row in df.iterrows():
    poly_type_raw = normalize_value(row.get("polymerization_type"))
    method_raw = normalize_value(row.get("method"))
    
    if poly_type_raw:
        all_poly_types[poly_type_raw] += 1
    if method_raw:
        all_methods[method_raw] += 1

# (Removed) Dumping all raw types/methods to a txt file.
# Keep console output focused on the final grouped trends + summary stats.

# Now collect data with years and apply grouping
data_with_years = []

# Track original values and their groupings for reporting
poly_type_mapping = defaultdict(set)
method_mapping = defaultdict(set)
# Track unassigned values (those that didn't match any group)
unassigned_poly_types = Counter()
unassigned_methods = Counter()

# Get all known group display names (values in the mapping dict are already display names)
known_poly_type_groups = set(poly_type_to_group.values())
known_method_groups = set(method_to_group.values())

for idx, row in df.iterrows():
    try:
        yv = pub_year_series.iloc[int(idx)]
        year = int(yv) if np.isfinite(yv) else None
    except Exception:
        year = None
    if year is None:
        continue
    
    poly_type_raw = normalize_value(row.get("polymerization_type"))
    method_raw = normalize_value(row.get("method"))
    
    # Group similar types and methods
    poly_type = group_polymerization_type(poly_type_raw) if poly_type_raw else None
    method = group_method(method_raw) if method_raw else None
    
    # Track mappings
    if poly_type_raw and poly_type:
        poly_type_mapping[poly_type].add(poly_type_raw)
        # Check if this was assigned to a known group
        if poly_type not in known_poly_type_groups:
            # This value was not assigned to any group (kept as-is)
            unassigned_poly_types[poly_type_raw] += 1
    
    if method_raw and method:
        method_mapping[method].add(method_raw)
        # Check if this was assigned to a known group
        if method not in known_method_groups:
            # This value was not assigned to any group (kept as-is)
            unassigned_methods[method_raw] += 1
    
    if poly_type or method:
        data_with_years.append({
            'year': year,
            'polymerization_type': poly_type,
            'method': method
        })

print(f"  ✓ Collected {len(data_with_years)} entries with years and polymerization data")

# Print grouping information
if poly_type_mapping:
    print(f"\n  Polymerization type groupings ({len(poly_type_mapping)} groups):")
    for grouped_type, original_types in sorted(poly_type_mapping.items()):
        if len(original_types) > 1:
            print(f"    '{grouped_type}': {sorted(original_types)}")
        else:
            print(f"    '{grouped_type}': {sorted(original_types)[0]}")

if method_mapping:
    print(f"\n  Method groupings ({len(method_mapping)} groups):")
    for grouped_method, original_methods in sorted(method_mapping.items()):
        if len(original_methods) > 1:
            print(f"    '{grouped_method}': {sorted(original_methods)}")
        else:
            print(f"    '{grouped_method}': {sorted(original_methods)[0]}")

# Print top 20 unassigned values
if unassigned_poly_types:
    print(f"\n  Top 20 unassigned polymerization types (not in any group):")
    for i, (poly_type, count) in enumerate(unassigned_poly_types.most_common(20), 1):
        print(f"    {i}. '{poly_type}': {count} occurrences")

if unassigned_methods:
    print(f"\n  Top 20 unassigned methods (not in any group):")
    for i, (method, count) in enumerate(unassigned_methods.most_common(20), 1):
        print(f"    {i}. '{method}': {count} occurrences")

# Create year bins (5-year bins)
bin_size = 5
year_bins = []
for entry in data_with_years:
    year = entry['year']
    year_bin = create_year_bin(year, bin_size)
    if year_bin:
        entry['year_bin'] = year_bin
        year_bins.append(year_bin)

unique_bins = sorted(set(year_bins))
print(f"\n  Year bins: {unique_bins[0]} to {unique_bins[-1]} ({len(unique_bins)} bins)")

# Exclude "Rest Unspecified" group from plots
UNSPECIFIED_GROUP_NAME = _group_key_to_display_name("rest_unspecified")

# Count occurrences per bin for polymerization types
print("\n  Counting polymerization types per bin...")
poly_type_counts = defaultdict(lambda: defaultdict(int))
for entry in data_with_years:
    if 'year_bin' in entry and entry['polymerization_type']:
        poly_type = entry['polymerization_type']
        if poly_type == UNSPECIFIED_GROUP_NAME:
            continue
        bin_label = entry['year_bin']
        poly_type_counts[poly_type][bin_label] += 1

# Count occurrences per bin for methods
print("  Counting methods per bin...")
method_counts = defaultdict(lambda: defaultdict(int))
for entry in data_with_years:
    if 'year_bin' in entry and entry['method']:
        method = entry['method']
        if method == UNSPECIFIED_GROUP_NAME:
            continue
        bin_label = entry['year_bin']
        method_counts[method][bin_label] += 1

# Prepare data for plotting
def prepare_plot_data(counts_dict, unique_bins):
    """Convert counts dictionary to DataFrame for plotting."""
    plot_data = {}
    for key in counts_dict:
        values = [counts_dict[key].get(bin_label, 0) for bin_label in unique_bins]
        plot_data[key] = values
    return pd.DataFrame(plot_data, index=unique_bins)

# Filter to only show types/methods that appear in at least N bins or have at least M total occurrences
min_bins = 3
min_total = 10

def filter_plot_data(counts_dict, unique_bins, min_bins=min_bins, min_total=min_total):
    """Filter to only include types/methods that meet minimum criteria."""
    filtered = {}
    for key, bin_counts in counts_dict.items():
        non_zero_bins = sum(1 for count in bin_counts.values() if count > 0)
        total_count = sum(bin_counts.values())
        if non_zero_bins >= min_bins or total_count >= min_total:
            filtered[key] = bin_counts
    return filtered

poly_type_counts_filtered = filter_plot_data(poly_type_counts, unique_bins)
method_counts_filtered = filter_plot_data(method_counts, unique_bins)

print(f"\n  Filtered to {len(poly_type_counts_filtered)} polymerization types (from {len(poly_type_counts)})")
print(f"  Filtered to {len(method_counts_filtered)} methods (from {len(method_counts)})")

# Smooth temporal trends using a rolling mean (same approach as monomer temporal evolution plot)
rolling_window_years = 10
window_bins = int(max(1, round(rolling_window_years / bin_size)))

poly_type_df = prepare_plot_data(poly_type_counts_filtered, unique_bins)
method_df = prepare_plot_data(method_counts_filtered, unique_bins)

if window_bins > 1 and not poly_type_df.empty:
    poly_type_df = poly_type_df.rolling(window=window_bins, min_periods=max(1, window_bins // 2), center=True).mean()
if window_bins > 1 and not method_df.empty:
    method_df = method_df.rolling(window=window_bins, min_periods=max(1, window_bins // 2), center=True).mean()

# Create plots
print("\nCreating plots...")

# Convert bin labels to numeric for plotting (use midpoint of bin)
def bin_to_numeric(bin_label):
    """Convert bin label like '1965-1969' to numeric value (1967)."""
    parts = bin_label.split('-')
    if len(parts) == 2:
        return (int(parts[0]) + int(parts[1])) / 2
    return int(parts[0])

numeric_bins = [bin_to_numeric(bin_label) for bin_label in unique_bins]

# Compute total per bin (for normalization)
total_per_bin_type = poly_type_df.sum(axis=1).to_dict() if not poly_type_df.empty else {}
total_per_bin_method = method_df.sum(axis=1).to_dict() if not method_df.empty else {}

# Create figure with two subplots side by side (stacked area = proportional share per year)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH_INCH * 2, TWO_COL_WIDTH_INCH * 0.9), layout="constrained")

# Sort by total count for consistent order (largest at bottom in stack)
poly_type_sorted = sorted(poly_type_counts_filtered.items(),
                         key=lambda x: sum(x[1].values()), reverse=True)
method_sorted = sorted(method_counts_filtered.items(),
                       key=lambda x: sum(x[1].values()), reverse=True)

# Plot 1: Polymerization Types (stacked area, 100% per bin)
ax1.set_title("A  Polymerization types temporal evolution", loc="left", fontsize=14, pad=6)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Share', fontsize=12)
ax1.set_ylim(0, 1)

proportions_type = []
labels_type = []
colors_type = []
for i, (poly_type, bin_counts) in enumerate(poly_type_sorted):
    total_per_bin = [total_per_bin_type.get(bin_label, 0) for bin_label in unique_bins]
    if not poly_type_df.empty and poly_type in poly_type_df.columns:
        counts = poly_type_df[poly_type].reindex(unique_bins).fillna(0).to_numpy().tolist()
    else:
        counts = [bin_counts.get(bin_label, 0) for bin_label in unique_bins]
    prop = [c / t if t > 0 else 0 for c, t in zip(counts, total_per_bin)]
    proportions_type.append(prop)
    labels_type.append(poly_type if poly_type else "Unknown")
    colors_type.append(SEQUENTIAL_COLORS[i % len(SEQUENTIAL_COLORS)])

ax1.stackplot(numeric_bins, *proportions_type, labels=labels_type, colors=colors_type, alpha=0.85)
ax1.legend(loc='lower left', fontsize=9, frameon=True, ncol=1)
ax1.grid(False)
ax1.set_xlim(min(numeric_bins) - 2, max(numeric_bins) + 2)

# Plot 2: Methods (stacked area, 100% per bin)
ax2.set_title("B  Polymerization methods temporal evolution", loc="left", fontsize=14, pad=6)
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Share', fontsize=12)
ax2.set_ylim(0, 1)

proportions_method = []
labels_method = []
colors_method = []
for i, (method, bin_counts) in enumerate(method_sorted):
    total_per_bin = [total_per_bin_method.get(bin_label, 0) for bin_label in unique_bins]
    if not method_df.empty and method in method_df.columns:
        counts = method_df[method].reindex(unique_bins).fillna(0).to_numpy().tolist()
    else:
        counts = [bin_counts.get(bin_label, 0) for bin_label in unique_bins]
    prop = [c / t if t > 0 else 0 for c, t in zip(counts, total_per_bin)]
    proportions_method.append(prop)
    labels_method.append(method if method else "Unknown")
    colors_method.append(SEQUENTIAL_COLORS[i % len(SEQUENTIAL_COLORS)])

ax2.stackplot(numeric_bins, *proportions_method, labels=labels_method, colors=colors_method, alpha=0.85)
ax2.legend(loc='lower left', fontsize=9, frameon=True, ncol=1)
ax2.grid(False)
ax2.set_xlim(min(numeric_bins) - 2, max(numeric_bins) + 2)

# Create output directory if it doesn't exist
output_dir = Path(__file__).parent / 'figures'
output_dir.mkdir(exist_ok=True)

# Save the plot
output_path = output_dir / 'polymerization_trends.pdf'
fig.savefig(output_path, bbox_inches='tight', dpi=300)
print(f"\n✓ Plot saved to: {output_path}")

# Also save as PNG
output_path_png = output_dir / 'polymerization_trends.png'
fig.savefig(output_path_png, bbox_inches='tight', dpi=300)
print(f"✓ Plot saved to: {output_path_png}")

# Print summary statistics
print(f"\nSummary Statistics:")
print(f"  Total entries with years: {len(data_with_years)}")
print(f"  Year range: {min([e['year'] for e in data_with_years])} - {max([e['year'] for e in data_with_years])}")
print(f"\n  Top polymerization types (by total count):")
for poly_type, bin_counts in sorted(poly_type_counts_filtered.items(), 
                                    key=lambda x: sum(x[1].values()), reverse=True)[:10]:
    total = sum(bin_counts.values())
    print(f"    {poly_type}: {total} occurrences")
print(f"\n  Top methods (by total count):")
for method, bin_counts in sorted(method_counts_filtered.items(), 
                                 key=lambda x: sum(x[1].values()), reverse=True)[:10]:
    total = sum(bin_counts.values())
    print(f"    {method}: {total} occurrences")

# Try to show plot, but don't fail if display is not available
try:
    plt.show()
except Exception as e:
    print(f"\nNote: Could not display plot interactively: {e}")
    print("Plots have been saved successfully.")
