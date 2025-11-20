#!/usr/bin/env python3
"""
Data analysis script for copolymerization prediction dataset.

Generates various analysis plots and statistics for the dataset.

Usage:
    python data_analysis.py --data-path path/to/processed_data.csv --output-dir output/data_analysis
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from pathlib import Path
from itertools import combinations

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from copolpredictor.prediction_utils import feature_columns
from plot_config import (
    setup_plot_style,
    ONE_COL_WIDTH_INCH,
    TWO_COL_WIDTH_INCH,
    CLASS_COLORS,
    SEQUENTIAL_COLORS,
)


def format_feature_name(name):
    """Format feature name for display."""
    # Special replacements with numbered suffixes (specific ones first)
    name = name.replace('polytype_emb_1', 'polymerization type emb. 1')
    name = name.replace('polytype_emb_2', 'polymerization type emb. 2')
    name = name.replace('method_emb_1', 'polymerization method emb. 1')
    name = name.replace('method_emb_2', 'polymerization method emb. 2')
    # General cases without numbers
    name = name.replace('polytype_emb', 'polymerization type emb.')
    name = name.replace('method_emb', 'polymerization method emb.')
    
    # Delta HOMO-LUMO formatting
    if 'delta_HOMO_LUMO' in name or 'delta_homo_lumo' in name:
        # Replace delta with symbol
        name = name.replace('delta_HOMO_LUMO', 'Δ HOMO-LUMO')
        name = name.replace('delta_homo_lumo', 'Δ HOMO-LUMO')
        # Replace AA, AB, BA, BB with 1-1, 1-2, 2-1, 2-2
        name = name.replace('_AA', ' 1-1')
        name = name.replace('_AB', ' 1-2')
        name = name.replace('_BA', ' 2-1')
        name = name.replace('_BB', ' 2-2')
    
    # Replace remaining underscores with spaces
    name = name.replace('_', ' ')
    
    return name


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze copolymerization dataset")
    parser.add_argument("--data-path", default="../output/processed_data.csv", help="Path to processed data CSV")
    parser.add_argument("--output-dir", default="../output/data_analysis", help="Output directory for plots and analysis")
    return parser.parse_args()


def setup_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")


def plot_feature_histograms(df, feature_cols, output_dir):
    """Plot histograms for all features."""
    print("\nGenerating feature histograms...")
    
    df_features = df[feature_cols].dropna()
    num_features = len(feature_cols)
    
    # Calculate grid size - 3 columns per row
    cols = 3
    rows = math.ceil(num_features / cols)
    
    # Use TWO_COL width, height depends on number of rows
    fig_width = TWO_COL_WIDTH_INCH
    fig_height = rows * (TWO_COL_WIDTH_INCH / cols) * 0.8  # Maintain aspect ratio
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    # Handle different subplot configurations
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        ax = axes[i]
        sns.histplot(df_features[col], kde=True, bins=30, ax=ax, color=SEQUENTIAL_COLORS[0])
        ax.set_title(format_feature_name(col), fontsize=6)
        ax.set_xlabel('', fontsize=5)
        ax.set_ylabel('Frequency', fontsize=5)
        ax.tick_params(labelsize=5)
        ax.grid(False)
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    # Save as PNG
    path_png = os.path.join(output_dir, 'feature_histograms_grid.png')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved PNG to {path_png}")
    
    # Save as PDF
    path_pdf = os.path.join(output_dir, 'feature_histograms_grid.pdf')
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"  ✓ Saved PDF to {path_pdf}")
    
    plt.close()


def plot_class_distribution(df, output_dir):
    """Plot class distribution as bar plot."""
    print("\nGenerating class distribution plot...")
    
    if 'r1r2' not in df.columns:
        print("  ⚠ Warning: 'r1r2' column missing, skipping class distribution plot")
        return
    
    r1r2_clean = df['r1r2'].dropna()
    
    # Calculate class counts
    class_0 = (r1r2_clean < 1).sum()
    class_1 = ((r1r2_clean >= 1) & (r1r2_clean <= 25)).sum()
    class_2 = (r1r2_clean > 25).sum()
    
    # Class labels in two lines
    class_labels = [
        "Class 0:\nAlternating",
        "Class 1:\nBlock-like",
        "Class 2:\nHomopolymer"
    ]
    
    counts = [class_0, class_1, class_2]
    colors = [CLASS_COLORS[0], CLASS_COLORS[1], CLASS_COLORS[2]]
    
    # Use ONE_COL width for single plot
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_INCH, ONE_COL_WIDTH_INCH / 1.618))
    
    # Create bar plot with narrower bars
    bars = ax.bar(class_labels, counts, color=colors, alpha=0.8, edgecolor='none', width=0.5)
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}',
                ha='center', va='bottom', fontsize=6)
    
    ax.set_ylabel('Count', fontsize=8)
    ax.set_xlabel('', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.set_xticklabels(class_labels, rotation=0, ha='center', fontsize=6)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save as PNG
    path_png = os.path.join(output_dir, 'class_distribution.png')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved PNG to {path_png}")
    
    # Save as PDF
    path_pdf = os.path.join(output_dir, 'class_distribution.pdf')
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"  ✓ Saved PDF to {path_pdf}")
    
    plt.close()


def plot_r1r2_histogram(df, output_dir):
    """Plot histogram for r1r2 values."""
    print("\nGenerating r1r2 histogram...")
    
    if 'r1r2' not in df.columns:
        print("  ⚠ Warning: 'r1r2' column missing, skipping r1r2 histogram")
        return
    
    df_clean = df['r1r2'].dropna()
    
    # Filter data to 0-5 range for better visualization
    df_filtered = df_clean[(df_clean >= 0) & (df_clean <= 5)]
    
    # Use ONE_COL width for single plot
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_INCH, ONE_COL_WIDTH_INCH / 1.618))
    
    # Create bins explicitly in the 0-5 range
    bins = np.linspace(0, 5, 21)  # 20 bins from 0 to 5
    
    sns.histplot(df_filtered, kde=True, bins=bins, ax=ax, color=SEQUENTIAL_COLORS[0])
    ax.set_xlabel('r-product', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_xlim(0, 5)
    ax.tick_params(labelsize=7)
    ax.grid(False)
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save as PNG
    path_png = os.path.join(output_dir, 'r1r2_histogram.png')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved PNG to {path_png}")
    
    # Save as PDF
    path_pdf = os.path.join(output_dir, 'r1r2_histogram.pdf')
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"  ✓ Saved PDF to {path_pdf}")
    
    plt.close()
    
    # Print statistics for filtered data (0-5 range)
    print(f"  r1r2 statistics (0-5 range):")
    print(f"    Total samples: {len(df_clean)}")
    print(f"    Samples in range: {len(df_filtered)} ({len(df_filtered)/len(df_clean)*100:.1f}%)")
    print(f"    Mean: {df_filtered.mean():.2f}")
    print(f"    Median: {df_filtered.median():.2f}")
    print(f"    Std: {df_filtered.std():.2f}")
    print(f"    Min: {df_filtered.min():.2f}")
    print(f"    Max: {df_filtered.max():.2f}")


def plot_correlation_heatmap(df, feature_cols, output_dir):
    """Plot correlation heatmap for features."""
    print("\nGenerating correlation heatmap...")
    
    df_features = df[feature_cols].dropna()
    corr_matrix = df_features.corr()
    
    # Format column/row names
    formatted_names = [format_feature_name(col) for col in feature_cols]
    corr_matrix.columns = formatted_names
    corr_matrix.index = formatted_names
    
    # Use TWO_COL width, square aspect ratio
    fig_size = min(TWO_COL_WIDTH_INCH, 7)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    sns.heatmap(
        corr_matrix,
        annot=False,  # Too many features for annotations
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.75},
        ax=ax
    )
    
    ax.tick_params(labelsize=5)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    
    # Save as PNG
    path_png = os.path.join(output_dir, 'feature_correlation_heatmap.png')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved PNG to {path_png}")
    
    # Save as PDF
    path_pdf = os.path.join(output_dir, 'feature_correlation_heatmap.pdf')
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"  ✓ Saved PDF to {path_pdf}")
    
    plt.close()


def plot_features_vs_target(df, feature_cols, output_dir):
    """Plot features vs r1r2 target variable."""
    print("\nGenerating feature vs target scatter plots...")
    
    if 'r1r2' not in df.columns:
        print("  ⚠ Warning: 'r1r2' column missing, skipping scatter plots")
        return
    
    plot_df = df[feature_cols + ['r1r2']].dropna()
    num_features = len(feature_cols)
    
    # Calculate grid size - 3 columns per row
    cols = 3
    rows = math.ceil(num_features / cols)
    
    # Use TWO_COL width, height depends on number of rows
    fig_width = TWO_COL_WIDTH_INCH
    fig_height = rows * (TWO_COL_WIDTH_INCH / cols) * 0.8  # Maintain aspect ratio
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    # Handle different subplot configurations
    if rows == 1 and cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, feature in enumerate(feature_cols):
        ax = axes[i]
        formatted_name = format_feature_name(feature)
        sns.scatterplot(data=plot_df, x=feature, y='r1r2', ax=ax, 
                       alpha=0.5, s=5, color=CLASS_COLORS[0], edgecolor='none')
        ax.set_title(f'{formatted_name} vs r1r2', fontsize=6)
        ax.set_xlabel(formatted_name, fontsize=5)
        ax.set_ylabel('r1r2', fontsize=5)
        ax.tick_params(labelsize=5)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    # Save as PNG
    path_png = os.path.join(output_dir, 'features_vs_r1r2_scatter.png')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved PNG to {path_png}")
    
    # Save as PDF
    path_pdf = os.path.join(output_dir, 'features_vs_r1r2_scatter.pdf')
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"  ✓ Saved PDF to {path_pdf}")
    
    plt.close()


def analyze_monomer_pairs(df, output_dir):
    """Analyze and plot top monomer pairs."""
    print("\nAnalyzing monomer pairs...")
    
    required_columns = ['monomer1_smiles', 'monomer2_smiles', 'monomer1_name', 'monomer2_name', 'reaction_id', 'r1r2']
    if not all(col in df.columns for col in required_columns):
        print("  ⚠ Warning: Required columns missing, skipping monomer pair analysis")
        return
    
    # Clean and prepare data
    df_clean = df.dropna(subset=['monomer1_smiles', 'monomer2_smiles']).copy()
    df_clean = df_clean.drop_duplicates(subset='reaction_id')
    
    # Create unordered monomer pair key
    df_clean['monomer_pair_key'] = df_clean.apply(
        lambda row: tuple(sorted([row['monomer1_smiles'], row['monomer2_smiles']])),
        axis=1
    )
    
    # Group by monomer pair key
    pair_to_id = {pair: idx for idx, pair in enumerate(df_clean['monomer_pair_key'].unique())}
    df_clean['group_id'] = df_clean['monomer_pair_key'].map(pair_to_id)
    
    # Save grouped data
    output_csv_path = os.path.join(output_dir, 'grouped_by_unique_monomer_pairs.csv')
    df_clean.to_csv(output_csv_path, index=False)
    print(f"  ✓ Saved grouped data to {output_csv_path}")
    
    # Count how many rows each monomer pair group has
    group_counts = df_clean.groupby(
        ['group_id', 'monomer1_name', 'monomer2_name']
    ).size().reset_index(name='count')
    
    # Top 10 most frequent monomer pairs
    top_10 = group_counts.sort_values(by='count', ascending=False).head(10)
    
    print("\n  Top 10 monomer pairs by number of datapoints:")
    print(top_10.to_string(index=False))
    
    # Plot histograms for top 10 pairs
    plot_top_monomer_pairs(df_clean, top_10, output_dir)


def plot_top_monomer_pairs(df, top_10, output_dir):
    """Plot r1r2 histograms for top 10 monomer pairs."""
    print("\nGenerating top monomer pairs histograms...")
    
    n = len(top_10)
    cols = 2
    rows = math.ceil(n / cols)
    
    # Use TWO_COL width, dynamic height
    fig_width = TWO_COL_WIDTH_INCH
    fig_height = max(3, rows * 1.5)
    
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(top_10.iterrows()):
        gid = row['group_id']
        name1 = row['monomer1_name']
        name2 = row['monomer2_name']
        subset = df[df['group_id'] == gid]
        
        ax = axes[i]
        sns.histplot(subset['r1r2'], bins=30, kde=False, ax=ax, color=CLASS_COLORS[0])
        ax.set_title(f"{name1} + {name2} (n={len(subset)})", fontsize=8)
        ax.set_xlabel("r1r2", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    
    # Save as PNG
    path_png = os.path.join(output_dir, 'top_monomer_pairs.png')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved PNG to {path_png}")
    
    # Save as PDF
    path_pdf = os.path.join(output_dir, 'top_monomer_pairs.pdf')
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"  ✓ Saved PDF to {path_pdf}")
    
    plt.close()


def print_dataset_statistics(df, output_dir):
    """Print comprehensive dataset statistics."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    # Unique reaction IDs
    if 'reaction_id' in df.columns:
        unique_reactions = df['reaction_id'].nunique()
        total_rows = len(df)
        print(f"\nUnique reaction IDs: {unique_reactions:,}")
        print(f"Total rows: {total_rows:,}")
        if unique_reactions < total_rows:
            print(f"  (Average {total_rows/unique_reactions:.1f} rows per reaction ID)")
    else:
        print("\n⚠ 'reaction_id' column not found")
    
    # Unique monomers
    unique_monomers = set()
    if 'monomer1_smiles' in df.columns:
        unique_monomers.update(df['monomer1_smiles'].dropna().unique())
    if 'monomer2_smiles' in df.columns:
        unique_monomers.update(df['monomer2_smiles'].dropna().unique())
    print(f"\nUnique monomers: {len(unique_monomers):,}")
    
    # Unique solvents
    if 'solvent_smiles' in df.columns:
        unique_solvents = df['solvent_smiles'].nunique()
        print(f"Unique solvents: {unique_solvents:,}")
    elif 'solvent' in df.columns:
        unique_solvents = df['solvent'].nunique()
        print(f"Unique solvents: {unique_solvents:,}")
    else:
        print("⚠ Solvent column not found")
    
    # Unique temperatures
    if 'temperature' in df.columns:
        temp_clean = df['temperature'].dropna()
        unique_temps = temp_clean.nunique()
        mean_temp = temp_clean.mean()
        median_temp = temp_clean.median()
        std_temp = temp_clean.std()
        min_temp = temp_clean.min()
        max_temp = temp_clean.max()
        
        print(f"\nTemperature statistics:")
        print(f"  Unique temperatures: {unique_temps:,}")
        print(f"  Mean: {mean_temp:.2f} °C")
        print(f"  Median: {median_temp:.2f} °C")
        print(f"  Std: {std_temp:.2f} °C")
        print(f"  Range: {min_temp:.2f} - {max_temp:.2f} °C")
    else:
        print("\n⚠ 'temperature' column not found")
    
    # r1r2 (r-product) statistics
    if 'r1r2' in df.columns:
        r1r2_clean = df['r1r2'].dropna()
        mean_r1r2 = r1r2_clean.mean()
        median_r1r2 = r1r2_clean.median()
        std_r1r2 = r1r2_clean.std()
        min_r1r2 = r1r2_clean.min()
        max_r1r2 = r1r2_clean.max()
        q25_r1r2 = r1r2_clean.quantile(0.25)
        q75_r1r2 = r1r2_clean.quantile(0.75)
        
        print(f"\nr₁×r₂ (r-product) statistics:")
        print(f"  Total samples: {len(r1r2_clean):,}")
        print(f"  Mean: {mean_r1r2:.4f}")
        print(f"  Median: {median_r1r2:.4f}")
        print(f"  Std: {std_r1r2:.4f}")
        print(f"  Min: {min_r1r2:.4f}")
        print(f"  Max: {max_r1r2:.4f}")
        print(f"  25th percentile: {q25_r1r2:.4f}")
        print(f"  75th percentile: {q75_r1r2:.4f}")
        
        # Class distribution
        class_0 = (r1r2_clean < 1).sum()
        class_1 = ((r1r2_clean >= 1) & (r1r2_clean <= 25)).sum()
        class_2 = (r1r2_clean > 25).sum()
        total_class = len(r1r2_clean)
        
        print(f"\n  Class distribution:")
        print(f"    Class 0 (r₁×r₂ < 1): {class_0:,} ({class_0/total_class*100:.1f}%)")
        print(f"    Class 1 (1 ≤ r₁×r₂ ≤ 25): {class_1:,} ({class_1/total_class*100:.1f}%)")
        print(f"    Class 2 (r₁×r₂ > 25): {class_2:,} ({class_2/total_class*100:.1f}%)")
    else:
        print("\n⚠ 'r1r2' column not found")
    
    # Additional statistics for monomers
    if 'monomer1_name' in df.columns and 'monomer2_name' in df.columns:
        # Count occurrences of each monomer
        m1_counts = df['monomer1_name'].value_counts()
        m2_counts = df['monomer2_name'].value_counts()
        all_monomer_counts = pd.concat([m1_counts, m2_counts]).groupby(level=0).sum().sort_values(ascending=False)
        
        print(f"\nMost frequent monomers (top 10):")
        for i, (name, count) in enumerate(all_monomer_counts.head(10).items(), 1):
            print(f"  {i:2d}. {name[:40]:40s} : {count:,} occurrences")
    
    # Save statistics to file
    stats_file = os.path.join(output_dir, 'dataset_statistics.txt')
    with open(stats_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DATASET STATISTICS\n")
        f.write("="*60 + "\n\n")
        
        if 'reaction_id' in df.columns:
            f.write(f"Unique reaction IDs: {unique_reactions:,}\n")
            f.write(f"Total rows: {total_rows:,}\n")
            if unique_reactions < total_rows:
                f.write(f"  (Average {total_rows/unique_reactions:.1f} rows per reaction ID)\n")
        
        f.write(f"\nUnique monomers: {len(unique_monomers):,}\n")
        
        if 'solvent_smiles' in df.columns or 'solvent' in df.columns:
            f.write(f"Unique solvents: {unique_solvents:,}\n")
        
        if 'temperature' in df.columns:
            f.write(f"\nTemperature statistics:\n")
            f.write(f"  Unique temperatures: {unique_temps:,}\n")
            f.write(f"  Mean: {mean_temp:.2f} °C\n")
            f.write(f"  Median: {median_temp:.2f} °C\n")
            f.write(f"  Std: {std_temp:.2f} °C\n")
            f.write(f"  Range: {min_temp:.2f} - {max_temp:.2f} °C\n")
        
        if 'r1r2' in df.columns:
            f.write(f"\nr₁×r₂ (r-product) statistics:\n")
            f.write(f"  Total samples: {len(r1r2_clean):,}\n")
            f.write(f"  Mean: {mean_r1r2:.4f}\n")
            f.write(f"  Median: {median_r1r2:.4f}\n")
            f.write(f"  Std: {std_r1r2:.4f}\n")
            f.write(f"  Min: {min_r1r2:.4f}\n")
            f.write(f"  Max: {max_r1r2:.4f}\n")
            f.write(f"  25th percentile: {q25_r1r2:.4f}\n")
            f.write(f"  75th percentile: {q75_r1r2:.4f}\n")
            f.write(f"\n  Class distribution:\n")
            f.write(f"    Class 0 (r₁×r₂ < 1): {class_0:,} ({class_0/total_class*100:.1f}%)\n")
            f.write(f"    Class 1 (1 ≤ r₁×r₂ ≤ 25): {class_1:,} ({class_1/total_class*100:.1f}%)\n")
            f.write(f"    Class 2 (r₁×r₂ > 25): {class_2:,} ({class_2/total_class*100:.1f}%)\n")
        
        if 'monomer1_name' in df.columns and 'monomer2_name' in df.columns:
            f.write(f"\nMost frequent monomers (top 10):\n")
            for i, (name, count) in enumerate(all_monomer_counts.head(10).items(), 1):
                f.write(f"  {i:2d}. {name[:40]:40s} : {count:,} occurrences\n")
    
    print(f"\n  ✓ Statistics saved to {stats_file}")


def find_missing_pairs(df, output_dir, min_count_single=10, top_n=50, near_miss_max_count=1):
    """Find missing and near-miss monomer pairs."""
    print("\nFinding missing and near-miss monomer pairs...")
    
    required_cols = {"monomer1_smiles", "monomer2_smiles", "monomer1_name", "monomer2_name"}
    if not required_cols.issubset(df.columns):
        print("  ⚠ Warning: Required columns missing, skipping missing pairs analysis")
        return
    
    # Normalize SMILES
    for c in ["monomer1_smiles", "monomer2_smiles", "monomer1_name", "monomer2_name"]:
        df[c] = df[c].astype(str).str.strip()
    
    df = df.copy()
    df["smiles1"] = df["monomer1_smiles"].str.strip()
    df["smiles2"] = df["monomer2_smiles"].str.strip()
    df["pair_unordered"] = df.apply(lambda r: tuple(sorted([r["smiles1"], r["smiles2"]])), axis=1)
    df["pair_ordered_12"] = list(zip(df["smiles1"], df["smiles2"]))
    
    # De-duplicate by reaction_id for unordered presence
    if "reaction_id" in df.columns:
        df_unordered_base = df.drop_duplicates(subset="reaction_id")
    else:
        df_unordered_base = df
    
    # Unordered counts
    unordered_counts = df_unordered_base["pair_unordered"].value_counts()
    
    # Ordered counts
    ordered_counts_12 = df["pair_ordered_12"].value_counts()
    ordered_counts_21 = df.apply(lambda r: (r["smiles2"], r["smiles1"]), axis=1).value_counts()
    
    # Single-monomer frequencies
    singles = (
        pd.concat([
            df_unordered_base[["smiles1"]].rename(columns={"smiles1": "smiles"}),
            df_unordered_base[["smiles2"]].rename(columns={"smiles2": "smiles"})
        ], ignore_index=True)
        .value_counts("smiles")
        .rename("count")
        .reset_index()
    )
    
    frequent = singles[singles["count"] >= min_count_single].sort_values("count", ascending=False).reset_index(drop=True)
    
    # SMILES -> name mapping
    name_candidates = pd.concat([
        df[["smiles1", "monomer1_name"]].rename(columns={"smiles1": "smiles", "monomer1_name": "name"}),
        df[["smiles2", "monomer2_name"]].rename(columns={"smiles2": "smiles", "monomer2_name": "name"}),
    ], ignore_index=True).dropna()
    
    name_map = (
        name_candidates.groupby(["smiles", "name"]).size()
        .reset_index(name="cnt")
        .sort_values(["smiles", "cnt"], ascending=[True, False])
        .drop_duplicates(subset=["smiles"])
        .set_index("smiles")["name"]
        .to_dict()
    )
    
    def disp(smiles: str) -> str:
        return name_map.get(smiles, smiles)
    
    # Build candidate pairs
    cand_smiles_pairs = [tuple(sorted(p)) for p in combinations(frequent["smiles"], 2)]
    
    # Score and filter
    results = []
    single_count_map = dict(zip(singles["smiles"], singles["count"]))
    
    for s1, s2 in cand_smiles_pairs:
        unordered = int(unordered_counts.get((s1, s2), 0))
        ordered_12 = int(ordered_counts_12.get((s1, s2), 0))
        ordered_21 = int(ordered_counts_21.get((s1, s2), 0))
        
        if unordered <= near_miss_max_count:
            c1 = int(single_count_map.get(s1, 0))
            c2 = int(single_count_map.get(s2, 0))
            results.append({
                "smiles_1": s1,
                "smiles_2": s2,
                "name_1": disp(s1),
                "name_2": disp(s2),
                "mono1_count": c1,
                "mono2_count": c2,
                "observed_unordered": unordered,
                "observed_order_12": ordered_12,
                "observed_order_21": ordered_21,
                "expected_score": c1 * c2,
            })
    
    suggestions = pd.DataFrame(results).sort_values(
        by=["observed_unordered", "expected_score"], ascending=[True, False]
    ).reset_index(drop=True)
    
    # Split into missing and near-miss
    missing_only = suggestions[suggestions["observed_unordered"] == 0].head(top_n)
    near_miss = suggestions[(suggestions["observed_unordered"] > 0) &
                           (suggestions["observed_unordered"] <= near_miss_max_count)].head(top_n)
    
    # Save results
    cols_order = [
        "name_1", "name_2", "smiles_1", "smiles_2",
        "mono1_count", "mono2_count",
        "observed_unordered", "observed_order_12", "observed_order_21",
        "expected_score"
    ]
    
    missing_csv = os.path.join(output_dir, 'missing_pairs.csv')
    near_miss_csv = os.path.join(output_dir, 'near_miss_pairs.csv')
    
    missing_only[cols_order].to_csv(missing_csv, index=False)
    near_miss[cols_order].to_csv(near_miss_csv, index=False)
    
    print(f"  ✓ Saved missing pairs to {missing_csv}")
    print(f"  ✓ Saved near-miss pairs to {near_miss_csv}")
    
    print(f"\n  Top strictly-missing pairs (unordered count = 0): {len(missing_only)}")
    if len(missing_only) > 0:
        print(missing_only[["name_1", "name_2", "mono1_count", "mono2_count"]].head(10).to_string(index=False))


def main():
    """Main analysis pipeline."""
    args = parse_args()
    
    # Setup
    setup_plot_style()
    setup_output_dir(args.output_dir)
    
    print("="*60)
    print("DATA ANALYSIS")
    print("="*60)
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    
    # Load data
    print("\nLoading data...")
    try:
        df = pd.read_csv(args.data_path)
        print(f"  ✓ Loaded {len(df)} samples")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        sys.exit(1)
    
    # Get feature columns that exist in the dataframe
    all_available_features = [col for col in feature_columns if col in df.columns]
    
    # Filter to only desired features: temperature, solvent_logP, and delta HOMO-LUMO
    desired_features = []
    for col in all_available_features:
        if col in ['temperature', 'solvent_logP', 'solvent_logp']:
            desired_features.append(col)
        elif 'delta_HOMO_LUMO' in col or 'delta_homo_lumo' in col:
            desired_features.append(col)
    
    available_features = desired_features
    print(f"  ✓ Found {len(available_features)} features (filtered to temperature, solvent_logP, and delta HOMO-LUMO)")
    
    # Print dataset statistics first
    print_dataset_statistics(df, args.output_dir)
    
    # Generate plots and analyses
    print("\n" + "="*60)
    print("GENERATING PLOTS AND ANALYSES")
    print("="*60)
    
    plot_class_distribution(df, args.output_dir)
    plot_r1r2_histogram(df, args.output_dir)
    plot_feature_histograms(df, available_features, args.output_dir)
    plot_correlation_heatmap(df, available_features, args.output_dir)
    plot_features_vs_target(df, available_features, args.output_dir)
    analyze_monomer_pairs(df, args.output_dir)
    find_missing_pairs(df, args.output_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
