#!/usr/bin/env python3
"""
Plot generation script for voting-model filter sweep results.

Reads results from sweep_filters.py and generates all plots.
Can be run independently to regenerate plots without re-training.

Usage:
    python plot_sweep_results.py [--results-path PATH] [--plots-dir DIR]
"""

import os
import sys
import argparse
import ast
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'copol_prediction'))

try:
    from copol_prediction.analysis.plot_config import (
        setup_plot_style,
        HEATMAP_CMAP,
        TWO_COL_WIDTH_INCH,
        ONE_COL_WIDTH_INCH,
        CLASS_COLORS,
        get_class_label,
        CONFUSION_MATRIX_CONFIG,
    )
except ImportError:
    def setup_plot_style():
        pass
    HEATMAP_CMAP = 'Blues'
    TWO_COL_WIDTH_INCH = 7
    ONE_COL_WIDTH_INCH = 3.5
    CLASS_COLORS = {0: '#3A3B73', 1: '#e27f07', 2: '#6a040f'}
    CONFUSION_MATRIX_CONFIG = {'cmap': 'Blues', 'values_format': 'd'}
    def get_class_label(cid, style='default'):
        labels = {0: "Alternating", 1: "Block-like", 2: "Homopolymer"}
        return labels.get(cid, f"Class {cid}")

# Style for matplotlib
try:
    _STYLE_PATH = os.path.join(_PROJECT_ROOT, 'copol_prediction', 'analysis',
                                'lamalab.mplstyle')
    if os.path.exists(_STYLE_PATH):
        plt.style.use(_STYLE_PATH)
except Exception:
    pass

# Note: Negative data is now only a boolean (XGBoost only, NOT for Lookup)
# The old NEG_TARGETS and NEG_LABELS are kept for backward compatibility
# but should not be used in new code
NEG_TARGETS = ['none', 'xgb_only', 'lookup_only', 'both']
NEG_LABELS = {
    'none': 'Neg: neither',
    'xgb_only': 'Neg: XGB only',
    'lookup_only': 'Neg: Lookup only',
    'both': 'Neg: both',
}


# ---------------------------------------------------------------------------
# Rounding helper: always round 0.5 up
# ---------------------------------------------------------------------------
def round_up_half(val, decimals=2):
    """Round to decimals places, always rounding 0.5 up (not banker's rounding).
    
    Example: 0.725 -> 0.73 (third decimal >= 5, so round second decimal up)
    """
    if pd.isna(val) or np.isnan(val):
        return np.nan
    # Use Decimal with ROUND_HALF_UP to ensure 0.5 always rounds up
    # Convert to Decimal via string to avoid floating point precision issues
    val_decimal = Decimal(str(val))
    quantize_str = '0.' + '0' * decimals
    quantize_decimal = Decimal(quantize_str)
    rounded = val_decimal.quantize(quantize_decimal, rounding=ROUND_HALF_UP)
    return float(rounded)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plots from voting-model filter sweep results"
    )
    parser.add_argument("--results-path", type=str,
                        default="artifacts/experiments_voting/sweep_results.csv")
    parser.add_argument("--plots-dir", type=str,
                        default="output/voting_sweep")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Heatmap: per neg_data_target (2×2 grid of heatmaps)
# ---------------------------------------------------------------------------
def plot_heatmap_grid(results_df, plots_dir, metric='macro_accuracy',
                      metric_label='Macro Accuracy'):
    """Create a 2×2 grid of heatmaps — one per neg_data setting.

    Within each heatmap:
      rows = (spec × poly)  → 4
      cols = aug on/off      → 2
    """
    setup_plot_style()

    fig, axes = plt.subplots(2, 2, figsize=(TWO_COL_WIDTH_INCH, TWO_COL_WIDTH_INCH * 0.85))
    axes_flat = axes.flatten()

    row_labels = []
    for spec in [False, True]:
        for poly in [False, True]:
            s = "Spec+" if spec else "Spec-"
            p = "Poly+" if poly else "Poly-"
            row_labels.append(f"{s}\n{p}")

    col_labels = ["Aug-", "Aug+"]

    # Round vmin/vmax to 2 decimal places (0.5 always rounds up)
    vmin = round_up_half(results_df[metric].min(), decimals=2)
    vmax = round_up_half(results_df[metric].max(), decimals=2)

    # Check if using new format (add_negative_data) or old format (neg_data_target)
    if 'add_negative_data' in results_df.columns:
        # New format: 2x2 grid for (aug, neg) combinations
        for ax_idx, neg_data in enumerate([False, True]):
            for aug_idx, aug in enumerate([False, True]):
                ax = axes_flat[ax_idx * 2 + aug_idx]
                subset = results_df[
                    (results_df['add_negative_data'] == neg_data) &
                    (results_df['use_augmentation'] == aug)
                ]
                
                matrix = np.full((4, 2), np.nan)
                annot_matrix = np.empty((4, 2), dtype=object)
                annot_matrix.fill('')
                for _, row in subset.iterrows():
                    r_idx = int(row['remove_specialized']) * 2 + int(row['apply_polymerization_filter'])
                    c_idx = int(row['use_augmentation'])
                    val = row[metric]
                    rounded_val = round_up_half(val, decimals=2)
                    matrix[r_idx, c_idx] = rounded_val
                    if not np.isnan(rounded_val):
                        annot_matrix[r_idx, c_idx] = f'{rounded_val:.2f}'
                
                neg_label = "Neg+" if neg_data else "Neg-"
                aug_label = "Aug+" if aug else "Aug-"
                title = f"{neg_label} / {aug_label}"
                
                sns.heatmap(
                    matrix, annot=annot_matrix, fmt='', cmap=HEATMAP_CMAP,
                    xticklabels=col_labels, yticklabels=row_labels if aug_idx == 0 else [],
                    ax=ax, vmin=vmin, vmax=vmax,
                    linewidths=0.5, linecolor='gray',
                    annot_kws={'fontsize': 10},
                    cbar=ax_idx == 1 and aug_idx == 1,
                    cbar_kws={'label': metric_label} if (ax_idx == 1 and aug_idx == 1) else {},
                )
                ax.set_title(title, fontsize=11, fontweight='bold')
                ax.tick_params(labelsize=9)
                ax.grid(False)
    else:
        # Old format: use neg_data_target
        for ax_idx, neg_target in enumerate(NEG_TARGETS):
            ax = axes_flat[ax_idx]
            subset = results_df[results_df['neg_data_target'] == neg_target]

        matrix = np.full((4, 2), np.nan)
        annot_matrix = np.empty((4, 2), dtype=object)
        annot_matrix.fill('')
        for _, row in subset.iterrows():
            r_idx = int(row['remove_specialized']) * 2 + int(row['apply_polymerization_filter'])
            c_idx = int(row['use_augmentation'])
            val = row[metric]
            # Round to exactly 2 decimal places (0.5 always rounds up)
            rounded_val = round_up_half(val, decimals=2)
            if ax_idx == 0 and r_idx == 0 and c_idx == 0:  # Print first value as example
                print(f"      Example rounding: {val} -> {rounded_val}")
            matrix[r_idx, c_idx] = rounded_val
            if not np.isnan(rounded_val):
                # Format as string with exactly 2 decimal places
                annot_matrix[r_idx, c_idx] = f'{rounded_val:.2f}'

        sns.heatmap(
            matrix, annot=annot_matrix, fmt='', cmap=HEATMAP_CMAP,
            xticklabels=col_labels, yticklabels=row_labels,
            ax=ax, vmin=vmin, vmax=vmax,
            linewidths=0.5, linecolor='gray',
            annot_kws={'fontsize': 10},
            cbar=ax_idx in [1, 3],
            cbar_kws={'label': metric_label} if ax_idx in [1, 3] else {},
        )
        ax.set_title(NEG_LABELS[neg_target], fontsize=11, fontweight='bold')
        ax.tick_params(labelsize=9)
        ax.grid(False)

    plt.tight_layout()

    for ext in ['png', 'pdf']:
        path = os.path.join(plots_dir, f'heatmap_grid_{metric}.{ext}')
        plt.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"  ✓ Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Combined wide heatmap: rows=(spec×poly), cols=(neg_target×aug)
# ---------------------------------------------------------------------------
def plot_combined_heatmap(results_df, plots_dir, metric='macro_accuracy',
                          metric_label='Macro Accuracy'):
    """Create a single 4×4 heatmap showing all 16 combinations."""
    setup_plot_style()

    row_labels = []
    for spec in [False, True]:
        for poly in [False, True]:
            s = "Spec+" if spec else "Spec-"
            p = "Poly+" if poly else "Poly-"
            row_labels.append(f"{s} / {p}")

    # Check if using new format (add_negative_data) or old format (neg_data_target)
    if 'add_negative_data' in results_df.columns:
        # New format: 4 columns for (aug, neg) combinations
        col_labels = []
        for aug in [False, True]:
            for neg in [False, True]:
                a = "Aug+" if aug else "Aug-"
                n = "Neg+" if neg else "Neg-"
                col_labels.append(f"{a} / {n}")
        
        matrix = np.full((4, 4), np.nan)
        annot_matrix = np.empty((4, 4), dtype=object)
        annot_matrix.fill('')
        
        for _, row in results_df.iterrows():
            r_idx = int(row['remove_specialized']) * 2 + int(row['apply_polymerization_filter'])
            c_idx = int(row['use_augmentation']) * 2 + int(row['add_negative_data'])
            val = row[metric]
            rounded_val = round_up_half(val, decimals=2)
            matrix[r_idx, c_idx] = rounded_val
            if not np.isnan(rounded_val):
                annot_matrix[r_idx, c_idx] = f'{rounded_val:.2f}'
        
        # Make plot square (4x4 grid)
        fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH * 0.9,
                                     TWO_COL_WIDTH_INCH * 0.9))
        
        # Round vmin/vmax to 2 decimal places (0.5 always rounds up)
        vmin = round_up_half(results_df[metric].min(), decimals=2)
        vmax = round_up_half(results_df[metric].max(), decimals=2)
        
        sns.heatmap(
            matrix, annot=annot_matrix, fmt='', cmap=HEATMAP_CMAP,
            xticklabels=col_labels, yticklabels=row_labels,
            ax=ax, vmin=vmin, vmax=vmax, linewidths=0.5, linecolor='gray',
            annot_kws={'fontsize': 14},  # Larger text in cells
            cbar_kws={'label': metric_label, 'shrink': 0.6},  # Smaller colorbar
            square=True,  # Make each cell square
        )
        ax.tick_params(labelsize=12)  # Larger tick labels
        ax.grid(False)
        
        # Adjust colorbar font size
        cbar = ax.collections[0].colorbar
        if cbar is not None:
            cbar.ax.tick_params(labelsize=11)
            cbar.set_label(metric_label, fontsize=12)
        
        # Separator lines between aug groups
        ax.axvline(x=2, color='black', linewidth=2)
    else:
        # Old format: 4×8 heatmap
        col_labels = []
        for neg_target in NEG_TARGETS:
            for aug in [False, True]:
                a = "Aug+" if aug else "Aug-"
                col_labels.append(f"{NEG_LABELS[neg_target]}\n{a}")

        # Round vmin/vmax to 2 decimal places (0.5 always rounds up)
        vmin = round_up_half(results_df[metric].min(), decimals=2)
        vmax = round_up_half(results_df[metric].max(), decimals=2)
        
        matrix = np.full((4, 8), np.nan)
        annot_matrix = np.empty((4, 8), dtype=object)
        annot_matrix.fill('')

        for _, row in results_df.iterrows():
            r_idx = int(row['remove_specialized']) * 2 + int(row['apply_polymerization_filter'])
            neg_idx = NEG_TARGETS.index(row['neg_data_target'])
            c_idx = neg_idx * 2 + int(row['use_augmentation'])
            val = row[metric]
            rounded_val = round_up_half(val, decimals=2)
            matrix[r_idx, c_idx] = rounded_val
            if not np.isnan(rounded_val):
                annot_matrix[r_idx, c_idx] = f'{rounded_val:.2f}'

        fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH * 1.5,
                                         TWO_COL_WIDTH_INCH * 0.45))

        sns.heatmap(
            matrix, annot=annot_matrix, fmt='', cmap=HEATMAP_CMAP,
            xticklabels=col_labels, yticklabels=row_labels,
            ax=ax, vmin=vmin, vmax=vmax, linewidths=0.5, linecolor='gray',
            annot_kws={'fontsize': 9},
            cbar_kws={'label': metric_label},
        )
        ax.tick_params(labelsize=8)
        ax.grid(False)

        # Separator lines between neg_target groups
        for k in range(1, 4):
            ax.axvline(x=k * 2, color='black', linewidth=2)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(plots_dir, f'heatmap_combined_{metric}.{ext}')
        plt.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"  ✓ Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Bar plot: grouped by neg_data_target
# ---------------------------------------------------------------------------
def plot_neg_target_comparison(results_df, plots_dir):
    """Bar plot comparing negative data variants (averaged over other filters)."""
    setup_plot_style()

    # Check if using new format (add_negative_data) or old format (neg_data_target)
    if 'add_negative_data' in results_df.columns:
        # New format: compare neg_data=True vs False
        grouped = results_df.groupby('add_negative_data').agg({
            'macro_accuracy': ['mean', 'std'],
            'macro_precision': ['mean', 'std'],
            'coverage': ['mean', 'std'],
        })
        
        metrics = [
            ('macro_accuracy', 'Macro Accuracy'),
            ('macro_precision', 'Macro Precision'),
            ('coverage', 'Coverage'),
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(TWO_COL_WIDTH_INCH * 1.2,
                                                 TWO_COL_WIDTH_INCH * 0.35))
        
        colors = ['#3A3B73', '#e27f07']
        labels = ['Neg-', 'Neg+']
        
        for ax, (metric, label) in zip(axes, metrics):
            means = [grouped.loc[False, (metric, 'mean')] if False in grouped.index else 0,
                     grouped.loc[True, (metric, 'mean')] if True in grouped.index else 0]
            stds = [grouped.loc[False, (metric, 'std')] if False in grouped.index else 0,
                    grouped.loc[True, (metric, 'std')] if True in grouped.index else 0]
            
            means_rounded = [round_up_half(m, decimals=2) for m in means]
            stds_rounded = [round_up_half(s, decimals=2) for s in stds]
            
            bars = ax.bar(range(2), means_rounded, yerr=stds_rounded, color=colors, alpha=0.85,
                          capsize=3)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7)
            
            ax.set_ylabel(label, fontsize=9)
            ax.set_xticks(range(2))
            ax.set_xticklabels(labels, fontsize=7, rotation=0, ha='center')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)
    else:
        # Old format: use neg_data_target
        grouped = results_df.groupby('neg_data_target').agg({
            'macro_accuracy': ['mean', 'std'],
            'macro_precision': ['mean', 'std'],
            'coverage': ['mean', 'std'],
        })

        metrics = [
            ('macro_accuracy', 'Macro Accuracy'),
            ('macro_precision', 'Macro Precision'),
            ('coverage', 'Coverage'),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(TWO_COL_WIDTH_INCH * 1.2,
                                                 TWO_COL_WIDTH_INCH * 0.35))

        colors = ['#3A3B73', '#e27f07', '#1e8db9', '#6a040f']

        for ax, (metric, label) in zip(axes, metrics):
            means = [grouped.loc[nt, (metric, 'mean')] if nt in grouped.index else 0
                     for nt in NEG_TARGETS]
            stds = [grouped.loc[nt, (metric, 'std')] if nt in grouped.index else 0
                    for nt in NEG_TARGETS]

            # Round means to 2 decimal places (0.5 always rounds up)
            means_rounded = [round_up_half(m, decimals=2) for m in means]
            stds_rounded = [round_up_half(s, decimals=2) for s in stds]
            
            bars = ax.bar(range(4), means_rounded, yerr=stds_rounded, color=colors, alpha=0.85,
                          capsize=3)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                        f'{h:.2f}', ha='center', va='bottom', fontsize=7)

            ax.set_ylabel(label, fontsize=9)
            ax.set_xticks(range(4))
            ax.set_xticklabels([NEG_LABELS[nt] for nt in NEG_TARGETS],
                               fontsize=7, rotation=20, ha='right')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(False)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(plots_dir, f'neg_target_comparison.{ext}')
        plt.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"  ✓ Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Sorted bar chart
# ---------------------------------------------------------------------------
def plot_sorted_bar(results_df, plots_dir, metric='macro_accuracy',
                    metric_label='Macro Accuracy'):
    """Horizontal bar chart of all runs sorted by metric."""
    setup_plot_style()

    df_sorted = results_df.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH,
                                     max(4, len(df_sorted) * 0.28)))
    ax.barh(range(len(df_sorted)), df_sorted[metric], color='#3A3B73',
            alpha=0.85)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted['run_name'], fontsize=7)
    ax.set_xlabel(metric_label, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        path = os.path.join(plots_dir, f'sorted_bar_{metric}.{ext}')
        plt.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"  ✓ Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Coverage heatmap
# ---------------------------------------------------------------------------
def plot_coverage_heatmap(results_df, plots_dir):
    """2×2 grid of heatmaps showing coverage per neg_data_target."""
    plot_heatmap_grid(results_df, plots_dir, metric='coverage',
                      metric_label='Coverage (models agree)')


# ---------------------------------------------------------------------------
# 4x4 Grid of Confusion Matrices
# ---------------------------------------------------------------------------
def plot_confusion_matrix_grid(results_df, plots_dir):
    """Create a SINGLE 4×4 grid showing confusion matrices for all 16 filter combinations.
    
    Grid structure:
    - Rows: (spec, poly) combinations (4 rows)
    - Columns: (aug, neg) combinations (4 columns)
    Each cell contains a 3×3 confusion matrix for the voting model.
    """
    setup_plot_style()
    
    # Create 4x4 grid with larger figure size for better visibility
    fig, axes = plt.subplots(4, 4, figsize=(TWO_COL_WIDTH_INCH * 2.5, TWO_COL_WIDTH_INCH * 2.5))
    
    # Row labels: (spec, poly) combinations
    row_labels = []
    for spec in [False, True]:
        for poly in [False, True]:
            s = "Spec+" if spec else "Spec-"
            p = "Poly+" if poly else "Poly-"
            row_labels.append(f"{s} / {p}")
    
    # Column labels: (aug, neg) combinations
    col_labels = []
    for aug in [False, True]:
        for neg in [False, True]:
            a = "Aug+" if aug else "Aug-"
            n = "Neg+" if neg else "Neg-"
            col_labels.append(f"{a} / {n}")
    
    # Initialize a dictionary to store matrices by position
    matrix_dict = {}
    for _, result_row in results_df.iterrows():
        spec = result_row['remove_specialized']
        poly = result_row['apply_polymerization_filter']
        aug = result_row['use_augmentation']
        neg = result_row['add_negative_data']
        
        # Calculate grid position
        row_idx = int(spec) * 2 + int(poly)
        col_idx = int(aug) * 2 + int(neg)
        
        # Convert confusion matrix to numpy array (handle string format from CSV)
        cm_raw = result_row['confusion_matrix']
        if isinstance(cm_raw, str):
            import ast
            cm_raw = ast.literal_eval(cm_raw)
        cm = np.array(cm_raw, dtype=int)
        if cm.size > 0 and cm.sum() > 0:
            matrix_dict[(row_idx, col_idx)] = cm
    
    # Plot each combination
    for row_idx in range(4):
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            
            if (row_idx, col_idx) not in matrix_dict:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=8)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                cm = matrix_dict[(row_idx, col_idx)]
                
                # Normalize confusion matrix for better visualization
                cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_norm = np.nan_to_num(cm_norm)
                
                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm_norm,
                    display_labels=[get_class_label(i) for i in range(3)]
                )
                disp.plot(
                    cmap=CONFUSION_MATRIX_CONFIG.get('cmap', 'Blues'),
                    ax=ax,
                    values_format='.2f',
                    im_kw={'vmin': 0, 'vmax': 1},
                    text_kw={'fontsize': 7}
                )
                
                # Remove colorbar from individual subplots
                if disp.im_ is not None and disp.im_.colorbar is not None:
                    disp.im_.colorbar.remove()
                
                # Add counts as text overlay (smaller font, below normalized values)
                for i in range(3):
                    for j in range(3):
                        count = cm[i, j]
                        if count > 0:
                            ax.text(j, i + 0.35, f'({count})', ha='center', va='center',
                                   fontsize=5, color='gray', alpha=0.7)
            
            # Set labels only on outer edges
            if row_idx == 3:
                ax.set_xlabel(col_labels[col_idx], fontsize=9, fontweight='bold')
            else:
                ax.set_xlabel('')
                ax.set_xticklabels([])
            
            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=9, fontweight='bold')
            else:
                ax.set_ylabel('')
                ax.set_yticklabels([])
            
            ax.tick_params(labelsize=7)
            ax.grid(False)
    
    # Add overall title
    fig.suptitle('Confusion Matrices: Voting Model (All 16 Filter Combinations)', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Add colorbar for the whole figure (shared across all subplots)
    if len(matrix_dict) > 0:
        # Use the first subplot's image for colorbar
        first_ax = None
        for row_idx in range(4):
            for col_idx in range(4):
                if (row_idx, col_idx) in matrix_dict and len(axes[row_idx, col_idx].images) > 0:
                    first_ax = axes[row_idx, col_idx]
                    break
            if first_ax is not None:
                break
        
        if first_ax is not None and len(first_ax.images) > 0:
            im = first_ax.images[0]
            cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
            cbar.set_label('Normalized Count', fontsize=10)
            cbar.ax.tick_params(labelsize=8)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    for ext in ['png', 'pdf']:
        path = os.path.join(plots_dir, f'confusion_matrix_grid_4x4.{ext}')
        plt.savefig(path, dpi=300 if ext == 'png' else None, bbox_inches='tight')
        print(f"  ✓ Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def plot_sweep_results(results_df, plots_dir):
    """Generate all plots from sweep results."""
    setup_plot_style()
    os.makedirs(plots_dir, exist_ok=True)

    # Round all numeric metrics to 2 decimal places BEFORE plotting (0.5 always rounds up)
    results_df = results_df.copy()
    print(f"\n  Rounding values to 2 decimal places:")
    for metric_col in ['macro_accuracy', 'macro_precision', 'coverage']:
        if metric_col in results_df.columns:
            sample_before = results_df[metric_col].dropna().head(5).tolist()
            results_df[metric_col] = results_df[metric_col].apply(
                lambda x: round_up_half(x, decimals=2)
            )
            sample_after = results_df[metric_col].dropna().head(5).tolist()
            print(f"    {metric_col}:")
            for b, a in zip(sample_before, sample_after):
                print(f"      {b} -> {a}")

    print("\n  Generating heatmap grids …")
    plot_heatmap_grid(results_df, plots_dir, 'macro_accuracy', 'Macro Accuracy')
    plot_heatmap_grid(results_df, plots_dir, 'macro_precision', 'Macro Precision')

    print("\n  Generating combined heatmaps …")
    plot_combined_heatmap(results_df, plots_dir, 'macro_accuracy', 'Macro Accuracy')
    plot_combined_heatmap(results_df, plots_dir, 'macro_precision', 'Macro Precision')

    print("\n  Generating coverage heatmaps …")
    plot_coverage_heatmap(results_df, plots_dir)

    print("\n  Generating neg-target comparison …")
    plot_neg_target_comparison(results_df, plots_dir)

    print("\n  Generating sorted bar charts …")
    plot_sorted_bar(results_df, plots_dir, 'macro_accuracy', 'Macro Accuracy')
    plot_sorted_bar(results_df, plots_dir, 'macro_precision', 'Macro Precision')
    
    print("\n  Generating 4x4 confusion matrix grid …")
    plot_confusion_matrix_grid(results_df, plots_dir)


# ---------------------------------------------------------------------------
# Standalone main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=" * 60)
    print("FILTER SWEEP — PLOT GENERATION")
    print("=" * 60)
    print(f"  Results: {args.results_path}")
    print(f"  Plots:   {args.plots_dir}")

    if not os.path.exists(args.results_path):
        print(f"\nError: Results file not found at {args.results_path}")
        print("Run sweep_filters.py first.")
        sys.exit(1)

    results_df = pd.read_csv(args.results_path)

    # Parse stringified dicts/lists if needed
    for col in ['confusion_matrix']:
        if col in results_df.columns and isinstance(results_df[col].iloc[0], str):
            results_df[col] = results_df[col].apply(ast.literal_eval)

    print(f"  Loaded {len(results_df)} configurations")

    plot_sweep_results(results_df, args.plots_dir)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
