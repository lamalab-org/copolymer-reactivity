import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from copolpredictor.prediction_utils import feature_columns

# Load your dataset
csv_path = "processed_data.csv"  # Change this to your actual CSV path
df = pd.read_csv(csv_path)

# Filter to existing columns and drop rows with missing values
feature_columns = [col for col in feature_columns if col in df.columns]
df_features = df[feature_columns].dropna()

# Prepare subplots: 4 columns, enough rows for all features
num_features = len(feature_columns)
cols = 4
rows = math.ceil(num_features / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

# Plot each feature
for i, col in enumerate(feature_columns):
    sns.histplot(df_features[col], kde=True, bins=30, ax=axes[i])
    axes[i].set_title(col)
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Frequency')

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
output_path = "output/data_analysis/feature_histograms_grid.png"
plt.savefig(output_path, dpi=300)
print(f"Saved plot to {output_path}")

# Compute Pearson correlation matrix
corr_matrix = df_features.corr()

# Set up the figure
plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,        # show correlation values
    fmt=".2f",         # format to 2 decimal places
    cmap="coolwarm",   # color scale
    square=True,
    cbar_kws={"shrink": 0.75}
)

plt.title("Feature Correlation Matrix", fontsize=16)
plt.tight_layout()

# Save to file
corr_output_path = "output/data_analysis/feature_correlation_heatmap.png"
plt.savefig(corr_output_path, dpi=300)
print(f"Saved correlation heatmap to {corr_output_path}")

# Check if 'r1r2' exists in the DataFrame
if 'r1r2' not in df.columns:
    raise ValueError("'r1r2' column (target) is missing from the CSV.")

# Combine features and target into one DataFrame, drop NaNs
plot_df = df[feature_columns + ['r1r2']].dropna()

# Prepare subplots
num_features = len(feature_columns)
cols = 4
rows = math.ceil(num_features / cols)

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

# Plot each feature vs. r1r2
for i, feature in enumerate(feature_columns):
    ax = axes[i]
    sns.scatterplot(data=plot_df, x=feature, y='r1r2', ax=ax, alpha=0.7)
    ax.set_title(f'{feature} vs. r1r2')
    ax.set_xlabel(feature)
    ax.set_ylabel('r1r2')

# Remove unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
scatter_output_path = "output/data_analysis/features_vs_r1r2_scatter.png"
plt.savefig(scatter_output_path, dpi=300)
print(f"Saved scatter plots to {scatter_output_path}")

# ==== Analysis of same monomer pairs =====
output_csv_path = "output/data_analysis/grouped_by_unique_monomer_pairs_with_features.csv"

feature_columns = [
        # Other features
        'temperature',
        'polytype_emb_1', 'polytype_emb_2', 'method_emb_1', 'method_emb_2', 'solvent_logP'
    ]
extra_columns = ['solvent', 'r1r2', 'reaction_id', 'PDF_name']
required_columns = ['monomer1_smiles', 'monomer2_smiles', 'monomer1_name', 'monomer2_name', 'reaction_id']
all_cols = required_columns + feature_columns + extra_columns

# 1. Load + clean
df = pd.read_csv(csv_path)
df = df.dropna(subset=['monomer1_smiles', 'monomer2_smiles'])
df = df[[col for col in all_cols if col in df.columns]]

# 2. Remove exact reaction duplicates
df = df.drop_duplicates(subset='reaction_id')

# 3. Create unordered monomer pair key (ignores order)
df['monomer_pair_key'] = df.apply(
    lambda row: tuple(sorted([row['monomer1_smiles'], row['monomer2_smiles']])),
    axis=1
)

# 4. Group by monomer pair key
grouped = df.groupby('monomer_pair_key')

# Create numeric group_id from unique monomer_pair_key
pair_to_id = {pair: idx for idx, pair in enumerate(df['monomer_pair_key'].unique())}
df['group_id'] = df['monomer_pair_key'].map(pair_to_id)

# 5. Prepare final output
df['monomer1_smiles_norm'] = df['monomer_pair_key'].apply(lambda x: x[0])
df['monomer2_smiles_norm'] = df['monomer_pair_key'].apply(lambda x: x[1])

# Save result
df.to_csv(output_csv_path, index=False)

# 6. Summary
num_pairs = df.groupby(['monomer1_smiles_norm', 'monomer2_smiles_norm']).ngroups
avg_repeats = round(len(df) / num_pairs, 2) if num_pairs > 0 else 0

print(f"\n✅ Saved deduplicated + grouped data to: {output_csv_path}")
print(f"🔁 Unique monomer pairs (unordered): {num_pairs}")
print(f"📊 Avg entries per pair: {avg_repeats}")

# Count how many rows each monomer pair group has
group_counts = df.groupby(['group_id', 'monomer1_smiles_norm', 'monomer2_smiles_norm']).size().reset_index(name='count')

# Sort by count descending and take top 10
top_10 = group_counts.sort_values(by='count', ascending=False).head(10)

# Print result
print("\n🔥 Top 10 monomer pairs by number of datapoints:")
print(top_10.to_string(index=False))

# === Setup ===
output_path = "output/data_analysis/top_monomer_pairs.png"

# Count number of entries per group (by monomer names)
group_counts = df.groupby(
    ['group_id', 'monomer1_name', 'monomer2_name']
).size().reset_index(name='count')

# Top 10 most frequent monomer name pairs
top_10 = group_counts.sort_values(by='count', ascending=False).head(10)

print("\n🔥 Top 10 monomer pairs by number of datapoints:")
print(top_10.to_string(index=False))

# Prepare subplot grid (e.g. 5x2)
n = len(top_10)
cols = 2
rows = math.ceil(n / cols)
fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4.5 * rows))
axes = axes.flatten()

# Plot histograms into subplots
for i, (_, row) in enumerate(top_10.iterrows()):
    gid = row['group_id']
    name1 = row['monomer1_name']
    name2 = row['monomer2_name']
    subset = df[df['group_id'] == gid]

    ax = axes[i]
    sns.histplot(subset['r1r2'], bins=30, kde=False, ax=ax)
    ax.set_title(f"{name1} + {name2} (n={len(subset)})", fontsize=18)
    ax.set_xlabel("r1r2", fontsize=18)
    ax.set_ylabel("Count", fontsize=18)

# Hide any unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig(output_path, dpi=300)
print(f"\n📈 All top 10 histograms saved to: {output_path}")


# ==== Suggestions on SMILES, displayed as Names; check both orders ====
from itertools import combinations
import os

# -------- Parameters --------
min_count_single = 10      # SMILES must appear at least this many times (as single) to enter candidate pool
top_n = 50                 # how many suggestions to keep for each table
near_miss_max_count = 1    # consider pairs with total unordered count <= this value as missing/near-miss
dedup_by_reaction = True   # if True, de-duplicate by reaction_id when counting unordered presence

# -------- Outputs --------
os.makedirs("output/data_analysis", exist_ok=True)
missing_csv = "output/data_analysis/missing_pairs_smiles_display_names.csv"
near_miss_csv = "output/data_analysis/near_miss_pairs_smiles_display_names.csv"

# -------- Safety checks --------
required_cols = {"monomer1_smiles","monomer2_smiles","monomer1_name","monomer2_name"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"CSV must contain columns: {required_cols}")

# -------- Normalize SMILES for keys (strip) --------
for c in ["monomer1_smiles","monomer2_smiles","monomer1_name","monomer2_name"]:
    df[c] = df[c].astype(str).str.strip()

def norm_smiles(x: str) -> str:
    return (x or "").strip()

# -------- Build ordered & unordered keys on SMILES --------
df = df.copy()
df["smiles1"] = df["monomer1_smiles"].map(norm_smiles)
df["smiles2"] = df["monomer2_smiles"].map(norm_smiles)
df["pair_unordered"] = df.apply(lambda r: tuple(sorted([r["smiles1"], r["smiles2"]])), axis=1)
df["pair_ordered_12"] = list(zip(df["smiles1"], df["smiles2"]))
df["pair_ordered_21"] = list(zip(df["smiles2"], df["smiles1"]))  # convenience; not used in counting directly

# -------- Optional de-duplication by reaction_id for unordered presence --------
if dedup_by_reaction and "reaction_id" in df.columns:
    df_unordered_base = df.drop_duplicates(subset="reaction_id")
else:
    df_unordered_base = df

# Unordered counts (existence regardless of order)
unordered_counts = df_unordered_base["pair_unordered"].value_counts()

# Ordered counts (direction matters)
ordered_counts_12 = df["pair_ordered_12"].value_counts()
ordered_counts_21 = df.apply(lambda r: (r["smiles2"], r["smiles1"]), axis=1).value_counts()

# -------- Single-monomer frequencies on SMILES (use unordered_base to avoid bias) --------
singles = (
    pd.concat([
        df_unordered_base[["smiles1"]].rename(columns={"smiles1":"smiles"}),
        df_unordered_base[["smiles2"]].rename(columns={"smiles2":"smiles"})
    ], ignore_index=True)
    .value_counts("smiles")
    .rename("count")
    .reset_index()
)

frequent = singles[singles["count"] >= min_count_single].sort_values("count", ascending=False).reset_index(drop=True)

# -------- SMILES -> display name mapping (most frequent name in dataset for that SMILES) --------
name_candidates = pd.concat([
    df[["smiles1","monomer1_name"]].rename(columns={"smiles1":"smiles","monomer1_name":"name"}),
    df[["smiles2","monomer2_name"]].rename(columns={"smiles2":"smiles","monomer2_name":"name"}),
], ignore_index=True).dropna()

# pick most common name per SMILES
name_map = (
    name_candidates.groupby(["smiles","name"]).size()
    .reset_index(name="cnt")
    .sort_values(["smiles","cnt"], ascending=[True, False])
    .drop_duplicates(subset=["smiles"])
    .set_index("smiles")["name"]
    .to_dict()
)

def disp(smiles: str) -> str:
    return name_map.get(smiles, smiles)  # fallback to SMILES if no name

# -------- Build candidate pairs from frequent SMILES --------
from itertools import combinations
cand_smiles_pairs = [tuple(sorted(p)) for p in combinations(frequent["smiles"], 2)]

# -------- Score and filter: unordered present <= near_miss_max_count --------
results = []
single_count_map = dict(zip(singles["smiles"], singles["count"]))

for s1, s2 in cand_smiles_pairs:
    # unordered & ordered counts
    unordered = int(unordered_counts.get((s1, s2), 0))
    ordered_12 = int(ordered_counts_12.get((s1, s2), 0))
    ordered_21 = int(ordered_counts_21.get((s1, s2), 0))

    # only keep missing / near-miss per threshold (unordered)
    if unordered <= near_miss_max_count:
        c1 = int(single_count_map.get(s1, 0))
        c2 = int(single_count_map.get(s2, 0))
        results.append({
            # SMILES (for computation / exactness)
            "smiles_1": s1,
            "smiles_2": s2,
            # Display names (for readability)
            "name_1": disp(s1),
            "name_2": disp(s2),
            # frequencies of singles
            "mono1_count": c1,
            "mono2_count": c2,
            # counts for pair
            "observed_unordered": unordered,
            "observed_order_12": ordered_12,
            "observed_order_21": ordered_21,
            # simple priority heuristic
            "expected_score": c1 * c2,
            # convenience: flags
            "strictly_missing_both_orders": int(unordered == 0 and ordered_12 == 0 and ordered_21 == 0)
        })

suggestions = pd.DataFrame(results).sort_values(
    by=["observed_unordered", "expected_score"], ascending=[True, False]
).reset_index(drop=True)

# Split into strictly-missing and near-miss (unordered >0 up to threshold)
missing_only = suggestions[suggestions["observed_unordered"] == 0].head(top_n)
near_miss = suggestions[(suggestions["observed_unordered"] > 0) &
                        (suggestions["observed_unordered"] <= near_miss_max_count)].head(top_n)

# -------- Save with NAMES first (SMILES kept as reference columns) --------
cols_order = [
    "name_1","name_2","smiles_1","smiles_2",
    "mono1_count","mono2_count",
    "observed_unordered","observed_order_12","observed_order_21",
    "expected_score","strictly_missing_both_orders"
]
missing_only[cols_order].to_csv(missing_csv, index=False)
near_miss[cols_order].to_csv(near_miss_csv, index=False)

print(f"✅ Saved strictly-missing suggestions (by SMILES; names shown) to {missing_csv}")
print(f"✅ Saved near-miss suggestions (by SMILES; names shown) to {near_miss_csv}")

print("\n🔥 Top strictly-missing pairs (unordered count = 0):")
print(missing_only[cols_order].to_string(index=False))

print("\n⚠️ Top near-miss pairs (unordered count ≤ threshold):")
print(near_miss[cols_order].to_string(index=False))
