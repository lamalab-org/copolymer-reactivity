#!/usr/bin/env python3
"""
Create database analysis plots:
1. Class distribution (bar chart)
2. Feature histograms grid (6 key features)
3. r1r2 histogram

Uses processed data from copol_prediction/output/processed_data.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
import sys

# Add copol_prediction to path to import plot_config
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'copol_prediction' / 'analysis'))
from plot_config import (
    setup_plot_style,
    CLASS_COLORS,
    get_class_label,
    TWO_COL_WIDTH_INCH,
    ONE_COL_WIDTH_INCH,
    SEQUENTIAL_COLORS,
    CONFIDENCE_PLOT_CONFIG,
    ERROR_ANALYSIS_CONFIG,
)


def load_data():
    """Load processed data from multiple possible locations."""
    project_root = Path(__file__).parent.parent.parent
    
    # Try different possible paths
    possible_paths = [
        project_root / 'copol_prediction' / 'output' / 'processed_data.csv',
        project_root / 'data_extraction' / 'artifacts' / 'datasets' / 'extracted_reactions.csv',
        project_root / 'copol_prediction' / 'artifacts' / 'data_splits' / 'train.csv',  # Use train split as fallback
    ]
    
    data_path = None
    for path in possible_paths:
        if path.exists():
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(f"Data file not found. Tried: {[str(p) for p in possible_paths]}")
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} samples")
    
    # If using train split, combine with val and test for complete dataset
    if 'train.csv' in str(data_path):
        split_dir = data_path.parent
        try:
            df_val = pd.read_csv(split_dir / 'val.csv')
            df_test = pd.read_csv(split_dir / 'test.csv')
            df = pd.concat([df, df_val, df_test], ignore_index=True)
            print(f"  Combined with val and test: {len(df)} total samples")
        except FileNotFoundError:
            pass  # Just use train data
    
    return df


def plot_class_distribution(df, output_dir):
    """Plot class distribution as bar chart."""
    
    # Create r_product_class if not exists
    if 'r_product_class' not in df.columns:
        if 'r1r2' in df.columns:
            bins = [-np.inf, 1, 25, np.inf]
            labels = [0, 1, 2]
            df['r_product_class'] = pd.cut(df['r1r2'], bins=bins, labels=labels, right=False).astype(int)
            
            # Override extremes
            if {'constant_1', 'constant_2'}.issubset(df.columns):
                extreme_mask = (
                    ((df['constant_1'] <= 0.1) & (df['constant_2'] > 25)) |
                    ((df['constant_2'] <= 0.1) & (df['constant_1'] > 25))
                )
                df.loc[extreme_mask, 'r_product_class'] = 2
        else:
            print("  Warning: Cannot create r_product_class - missing r1r2 column")
            return
    
    # Count classes
    class_counts = df['r_product_class'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_INCH, ONE_COL_WIDTH_INCH * 0.7))
    
    classes = [get_class_label(i, style='short') for i in range(3)]
    colors = [CLASS_COLORS.get(i) for i in range(3)]
    
    bars = ax.bar(range(3), [class_counts.get(i, 0) for i in range(3)], 
                  color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    
    # Add count labels on bars (smaller font)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + max(class_counts) * 0.01,
                f'{int(height)}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.set_ylabel('Count', fontsize=9)
    ax.set_xlabel('Class', fontsize=9)
    ax.set_xticks(range(3))
    ax.set_xticklabels(classes, fontsize=8)
    # No title
    
    # Thicker axis lines
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add padding between y-axis and first bar
    ax.set_xlim(-0.5, 2.5)
    
    ax.tick_params(labelsize=8, width=0.5)
    ax.grid(False)
    
    plt.tight_layout()
    
    for ext in ['png', 'pdf']:
        path = output_dir / f'class_distribution.{ext}'
        plt.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"  ✓ Saved {path}")
    plt.close()


def plot_feature_histograms(df, output_dir):
    """Plot grid of histograms for key features."""
    
    # Select features to plot (check for both logP and logp)
    feature_cols = [
        'delta_HOMO_LUMO_AA',
        'delta_HOMO_LUMO_AB',
        'delta_HOMO_LUMO_BB',
        'delta_HOMO_LUMO_BA',
        'temperature',
    ]
    
    # Check for solvent_logP or solvent_logp
    if 'solvent_logP' in df.columns:
        feature_cols.append('solvent_logP')
    elif 'solvent_logp' in df.columns:
        feature_cols.append('solvent_logp')
    
    # Check which features exist
    available_features = [f for f in feature_cols if f in df.columns]
    if len(available_features) == 0:
        print("  Warning: No feature columns found for histogram plot")
        return
    
    # Create 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(TWO_COL_WIDTH_INCH, TWO_COL_WIDTH_INCH * 0.7))
    axes_flat = axes.flatten()
    
    for idx, feat in enumerate(feature_cols):
        if idx >= len(axes_flat):
            break
        
        ax = axes_flat[idx]
        
        if feat not in df.columns:
            ax.text(0.5, 0.5, f'{feat}\n(not available)', ha='center', va='center',
                   transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Get data, remove NaN
        data = df[feat].dropna()
        
        if len(data) == 0:
            ax.text(0.5, 0.5, f'{feat}\n(no data)', ha='center', va='center',
                   transform=ax.transAxes, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Create histogram with proper styling from plot_config
        n_bins = CONFIDENCE_PLOT_CONFIG.get('bins', 30)
        hist_color = SEQUENTIAL_COLORS[0]  # Use first sequential color
        ax.hist(data, bins=n_bins, alpha=CONFIDENCE_PLOT_CONFIG.get('alpha', 0.6),
                color=hist_color, edgecolor=CONFIDENCE_PLOT_CONFIG.get('edgecolor', 'black'),
                linewidth=CONFIDENCE_PLOT_CONFIG.get('linewidth', 0.5))
        
        # Add KDE curve (same color as histogram bars)
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 200)
            kde_values = kde(x_range)
            # Normalize KDE to match histogram scale
            hist_counts, _ = np.histogram(data, bins=n_bins)
            kde_scale = hist_counts.max() / kde_values.max() if kde_values.max() > 0 else 1
            ax.plot(x_range, kde_values * kde_scale, color=hist_color,
                   linewidth=2, label='KDE')
        except ImportError:
            pass  # scipy not available
        
        # Format feature name for title
        title = feat.replace('_', ' ')
        if 'delta' in title.lower():
            title = title.replace('delta', 'Δ')
        if 'HOMO' in title:
            title = title.replace('HOMO LUMO', 'HOMO-LUMO')
        
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.tick_params(labelsize=9)
        ax.grid(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide unused subplots
    for idx in range(len(feature_cols), len(axes_flat)):
        axes_flat[idx].axis('off')
    
    plt.tight_layout()
    
    for ext in ['png', 'pdf']:
        path = output_dir / f'feature_histograms_grid.{ext}'
        plt.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"  ✓ Saved {path}")
    plt.close()


def plot_r1r2_histogram(df, output_dir):
    """Plot histogram of r1r2 (r-product) values."""
    
    if 'r1r2' not in df.columns:
        print("  Warning: r1r2 column not found")
        return
    
    data = df['r1r2'].dropna()
    
    if len(data) == 0:
        print("  Warning: No r1r2 data available")
        return
    
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_INCH * 1.3, ONE_COL_WIDTH_INCH * 0.75))
    
    # Create histogram with proper styling - 20 bins for range 0-5
    n_bins = 20
    hist_color = SEQUENTIAL_COLORS[0]  # Use first sequential color
    # Define bins explicitly for range 0-5
    bins = np.linspace(0, 5, n_bins + 1)
    hist_counts, bin_edges = np.histogram(data, bins=bins)
    ax.hist(data, bins=bins, alpha=CONFIDENCE_PLOT_CONFIG.get('alpha', 0.6),
            color=hist_color, edgecolor=CONFIDENCE_PLOT_CONFIG.get('edgecolor', 'black'),
            linewidth=CONFIDENCE_PLOT_CONFIG.get('linewidth', 0.5))
    
    # Add KDE curve (same color as histogram bars)
    try:
        from scipy.stats import gaussian_kde
        # Filter data to range 0-5 for KDE calculation to get better curve
        data_filtered = data[(data >= 0) & (data <= 5)]
        if len(data_filtered) > 1:
            kde = gaussian_kde(data_filtered)
            x_range = np.linspace(0, 5, 200)  # Limit KDE range to 0-5
            kde_density = kde(x_range)
            # Convert KDE density to counts: density * bin_width * total_samples
            bin_width = (5 - 0) / n_bins
            kde_counts = kde_density * bin_width * len(data_filtered)
            ax.plot(x_range, kde_counts, color=hist_color,
                   linewidth=1.5, label='KDE')
    except ImportError:
        pass  # scipy not available
    
    ax.set_xlabel('r-product', fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    # No title
    # Add padding between Y-axis and first bar
    ax.set_xlim(-0.2, 5)  # Limit x-axis to 0-5 with padding on left
    # Add padding to Y-axis top
    y_max = hist_counts.max() if len(hist_counts) > 0 else 1
    ax.set_ylim(bottom=0, top=y_max * 1.1)  # 10% padding at top
    ax.tick_params(labelsize=9)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    for ext in ['png', 'pdf']:
        path = output_dir / f'r1r2_histogram.{ext}'
        plt.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"  ✓ Saved {path}")
    plt.close()


def main():
    """Main function to generate all plots."""
    # Setup plot style first
    setup_plot_style()
    
    print("=" * 60)
    print("DATABASE ANALYSIS PLOTS")
    print("=" * 60)
    
    # Load data
    df = load_data()
    
    # Create output directory
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Generate plots
    print("\n[1/3] Generating class distribution plot...")
    plot_class_distribution(df, output_dir)
    
    print("\n[2/3] Generating feature histograms grid...")
    plot_feature_histograms(df, output_dir)
    
    print("\n[3/3] Generating r1r2 histogram...")
    plot_r1r2_histogram(df, output_dir)
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
