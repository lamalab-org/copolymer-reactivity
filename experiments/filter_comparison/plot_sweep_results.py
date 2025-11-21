#!/usr/bin/env python3
"""
Plot generation script for filter sweep results.

This script loads results from sweep_filters.py and generates all plots.
This allows regenerating plots without rerunning the entire experiment.

Usage:
    python plot_sweep_results.py [--results-path PATH] [--plots-dir DIR]
"""

import os
import sys
import argparse
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../copol_prediction'))

# Import plot configuration from copol_prediction/analysis/plot_config.py
try:
    # Try direct import
    from copol_prediction.analysis.plot_config import (
        setup_plot_style, 
        HEATMAP_CMAP,
        TWO_COL_WIDTH_INCH,
        ONE_COL_WIDTH_INCH,
        CLASS_COLORS,
        get_class_label
    )
except ImportError:
    # Fallback: try relative path import
    try:
        plot_config_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__), 
            '../../copol_prediction/analysis/plot_config.py'
        ))
        if os.path.exists(plot_config_path):
            import importlib.util
            spec = importlib.util.spec_from_file_location("plot_config", plot_config_path)
            plot_config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(plot_config)
            setup_plot_style = plot_config.setup_plot_style
            HEATMAP_CMAP = plot_config.HEATMAP_CMAP
            TWO_COL_WIDTH_INCH = plot_config.TWO_COL_WIDTH_INCH
        else:
            raise ImportError(f"plot_config.py not found at {plot_config_path}")
    except Exception as e:
        # Final fallback
        print(f"Warning: Could not load plot_config.py: {e}")
        print("Using default plot settings")
        def setup_plot_style():
            pass
        HEATMAP_CMAP = 'Blues'
        TWO_COL_WIDTH_INCH = 7
        ONE_COL_WIDTH_INCH = 3
        CLASS_COLORS = {0: '#3A3B73', 1: '#e27f07', 2: '#6a040f'}
        def get_class_label(class_id, style='default'):
            labels = {0: "Class 0:\nAlternating", 1: "Class 1:\nBlock-like", 2: "Class 2:\nHomopolymer"}
            return labels.get(class_id, f"Class {class_id}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate plots from filter sweep results"
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="artifacts/experiments_holdout/sweep_results.csv",
        help="Path to sweep_results.csv"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="output/model_comp",
        help="Directory to save plots"
    )
    
    return parser.parse_args()


def plot_confusion_matrix(cm, labels, save_path, run_name=None):
    """
    Plot and save a confusion matrix.
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        save_path: Path to save the plot
        run_name: Optional run name (not used in title, but for filename)
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot confusion matrix
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    
    # No grid
    ax.grid(False)
    
    # Set labels
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, fontsize=16)
    ax.set_yticklabels(labels, fontsize=16)
    ax.set_xlabel('Predicted Class', fontsize=16, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=16, fontweight='bold')
    
    # Annotate cells with larger font
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]),
                   ha="center", va="center",
                   color="white" if cm[i, j] > cm.max() / 2 else "black",
                   fontsize=18, fontweight='bold')
    
    # No title
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix to {save_path}")


def plot_4x4_matrix(results_df, plots_dir, metric='holdout_f1_macro', metric_label='F1 Score (Macro)'):
    """
    Create 4x4 matrix heatmap showing all filter combinations.
    
    Args:
        results_df: DataFrame with results including 'filters' column
        plots_dir: Directory to save plots
        metric: Metric column name to plot
        metric_label: Label for the metric
    """
    # Use plot style from copol_prediction/analysis/plot_config.py
    setup_plot_style()
    
    # Extract filter values from results
    filter_data = []
    for _, row in results_df.iterrows():
        filters = row['filters']
        if isinstance(filters, str):
            filters = ast.literal_eval(filters)
        filter_data.append({
            'remove_specialized': int(filters.get('remove_specialized', False)),
            'add_negative_data': int(filters.get('add_negative_data', False)),
            'use_augmentation': int(filters.get('use_augmentation', False)),
            'apply_polymerization_filter': int(filters.get('apply_polymerization_filter', False)),
            'metric': row[metric]
        })
    
    filter_df = pd.DataFrame(filter_data)
    
    # Create matrix: rows = specialized + negative (4 combos), cols = aug + poly (4 combos)
    # Row axis: (remove_specialized, add_negative_data)
    # Col axis: (use_augmentation, apply_polymerization_filter)
    
    matrix = np.full((4, 4), np.nan)
    labels_row = []
    labels_col = []
    
    # Generate row labels: (remove_spec, add_neg)
    for spec in [0, 1]:
        for neg in [0, 1]:
            spec_str = "Spec+" if spec else "Spec-"
            neg_str = "Neg+" if neg else "Neg-"
            labels_row.append(f"{spec_str}\n{neg_str}")
    
    # Generate col labels: (augment, poly_filter)
    for aug in [0, 1]:
        for poly in [0, 1]:
            aug_str = "Aug+" if aug else "Aug-"
            poly_str = "Poly+" if poly else "Poly-"
            labels_col.append(f"{aug_str}\n{poly_str}")
    
    # Fill matrix
    for _, row in filter_df.iterrows():
        row_idx = int(row['remove_specialized'] * 2 + row['add_negative_data'])
        col_idx = int(row['use_augmentation'] * 2 + row['apply_polymerization_filter'])
        matrix[row_idx, col_idx] = row['metric']
    
    # Create heatmap with TWO_COL_WIDTH_INCH (same style as analyze_model.py)
    # Use golden ratio for height
    golden = 1.618
    height = TWO_COL_WIDTH_INCH / golden
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, height))
    
    # Use mask for missing values
    mask = np.isnan(matrix)
    
    # Create heatmap
    heatmap = sns.heatmap(
        matrix,
        annot=True,
        fmt='.4f',
        cmap=HEATMAP_CMAP,
        mask=mask,
        cbar_kws={'label': metric_label},
        xticklabels=labels_col,
        yticklabels=labels_row,
        ax=ax,
        vmin=matrix[~mask].min() if not mask.all() else 0,
        vmax=matrix[~mask].max() if not mask.all() else 1,
        linewidths=0.5,
        linecolor='gray',
        annot_kws={'fontsize': 14}  # Larger but not bold
    )
    
    # Adjust colorbar font size
    # Find colorbar in the figure (seaborn creates it as a separate axes)
    for cbar_ax in fig.axes:
        if cbar_ax != ax:  # Colorbar is a different axes than the main plot
            # This is likely the colorbar
            cbar_ax.set_ylabel(metric_label, fontsize=12, fontweight='bold')
            cbar_ax.tick_params(labelsize=11)
            break
    
    # No title
    ax.set_xlabel('Augmentation & Polymerization Filter', fontsize=12, fontweight='bold')
    ax.set_ylabel('Specialized Removal & Negative Data', fontsize=12, fontweight='bold')
    
    # Tick labels (larger but consistent with analyze_model style)
    ax.tick_params(labelsize=11, which='both')
    
    # No grid (same as analyze_model)
    ax.grid(False)
    
    # Remove top and right spines (same style as analyze_model)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save as PNG
    filename_png = f'filter_matrix_{metric}.png'
    path_png = os.path.join(plots_dir, filename_png)
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved PNG to {path_png}")
    
    # Save as PDF
    filename_pdf = f'filter_matrix_{metric}.pdf'
    path_pdf = os.path.join(plots_dir, filename_pdf)
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"  ✓ Saved PDF to {path_pdf}")
    
    plt.close()


def plot_sweep_results(results_df, plots_dir):
    """
    Create visualizations of sweep results.
    
    Args:
        results_df: DataFrame with results
        plots_dir: Directory to save plots
    """
    # Setup plot style from copol_prediction/analysis/plot_config.py
    setup_plot_style()
    
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create 4x4 matrix plots (macro als primär)
    print("\n  Creating 4x4 matrix visualizations...")
    plot_4x4_matrix(results_df, plots_dir, metric='holdout_f1_macro', metric_label='F1 Score (Macro)')
    plot_4x4_matrix(results_df, plots_dir, metric='holdout_accuracy', metric_label='Accuracy')
    plot_4x4_matrix(results_df, plots_dir, metric='holdout_precision_macro', metric_label='Precision (Macro)')
    plot_4x4_matrix(results_df, plots_dir, metric='holdout_recall_macro', metric_label='Recall (Macro)')
    
    # Sort by F1 MACRO
    results_sorted = results_df.sort_values('holdout_f1_macro', ascending=True)
    
    # Plot 1: F1 scores (MACRO)
    plt.figure(figsize=(12, 8))
    plt.barh(results_sorted['run_name'], results_sorted['holdout_f1_macro'], color='#661124')
    plt.xlabel('Macro F1 Score (Holdout)', fontsize=12)
    plt.title('Model Performance Across Filter Combinations (Macro F1)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'F1_score_macro.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved F1 (macro) plot to {plots_dir}/F1_score_macro.png")
    
    # Plot 2: Accuracy
    plt.figure(figsize=(12, 8))
    plt.barh(results_sorted['run_name'], results_sorted['holdout_accuracy'], color='#2d5c8f')
    plt.xlabel('Accuracy (Holdout)', fontsize=12)
    plt.title('Holdout Accuracy Across Filter Combinations', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'Accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved Accuracy plot to {plots_dir}/Accuracy.png")
    
    # Plot 3: Generate confusion matrices for all configurations
    print(f"\n  Generating confusion matrices for all configurations...")
    for _, row in results_df.iterrows():
        cm = np.array(row['confusion_matrix'])
        if isinstance(cm, str):
            cm = ast.literal_eval(cm)
        cm = np.array(cm)
        
        run_name = row['run_name']
        cm_filename = f"confusion_matrix_{run_name}.png"
        cm_path = os.path.join(plots_dir, cm_filename)
        
        plot_confusion_matrix(
            cm=cm,
            labels=[0, 1, 2],
            save_path=cm_path,
            run_name=run_name
        )
    
    # Plot 4: Comparison of metrics (mit MACRO)
    metrics = ['holdout_accuracy', 'holdout_f1_macro', 'holdout_precision_macro', 'holdout_recall_macro']
    metric_labels = ['Accuracy', 'F1 (macro)', 'Precision (macro)', 'Recall (macro)']
    
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(results_sorted))
    width = 0.2
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax.barh(x + i * width, results_sorted[metric], width, label=label)
    
    ax.set_yticks(x + width * 1.5)
    ax.set_yticklabels(results_sorted['run_name'])
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title('Comparison of Macro Metrics Across Configurations', fontsize=14)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'Metrics_comparison_macro.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved metrics comparison to {plots_dir}/Metrics_comparison_macro.png")


def plot_class_distribution_after_neg_data_augmentation(results_df, plots_dir):
    """
    Plot class distribution after negative data augmentation for configurations that use it.
    Similar to plot_class_distribution in data_analysis.py.
    """
    print("\n  Generating class distribution plots after neg data augmentation...")
    
    # Find configurations that use negative data
    neg_data_configs = results_df[results_df['filters'].apply(
        lambda x: x.get('add_negative_data', False) if isinstance(x, dict) else False
    )]
    
    if len(neg_data_configs) == 0:
        print("  No configurations with negative data found, skipping class distribution plot")
        return
    
    # We need to load the actual data to get class counts
    # For now, we'll use the n_train from results, but ideally we'd load the actual data
    # Let's create a plot showing the training set sizes per configuration
    
    # For each configuration with neg data, try to load the data and plot
    for idx, row in neg_data_configs.iterrows():
        run_name = row['run_name']
        filters = row['filters'] if isinstance(row['filters'], dict) else ast.literal_eval(row['filters'])
        
        if not filters.get('add_negative_data', False):
            continue
        
        # Try to load the data using the same logic as in sweep_filters.py
        try:
            # Import necessary modules
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../copol_prediction'))
            from utils import load_data_split
            from copolpredictor import prediction_utils
            
            # Load global split
            script_dir = os.path.dirname(__file__)
            copol_pred_dir = os.path.abspath(os.path.join(script_dir, '../../copol_prediction'))
            original_cwd = os.getcwd()
            os.chdir(copol_pred_dir)
            
            try:
                df_train, df_test = load_data_split.load_train_test_split()
            finally:
                os.chdir(original_cwd)
            
            # Apply filters (simplified - just for getting class distribution)
            # Add negative data
            neg_paths = [
                os.path.join(copol_pred_dir, 'filter/artificial_datapoints/processed_combined_augmented.csv'),
                os.path.join(script_dir, '../../copol_prediction/filter/artificial_datapoints/processed_combined_augmented.csv'),
            ]
            
            for neg_path in neg_paths:
                if os.path.exists(neg_path):
                    df_neg = pd.read_csv(neg_path)
                    if 'Class' in df_neg.columns:
                        df_neg = df_neg.rename(columns={'Class': 'r_product_class'})
                        df_neg['r_product_class'] = df_neg['r_product_class'].astype(int)
                        df_train = pd.concat([df_train, df_neg], ignore_index=True)
                        break
            
            # Apply augmentation if used
            if filters.get('use_augmentation', False):
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
                from copolpredictor import data_augmentation
                df_train = data_augmentation.augment_with_gaussian_samples(
                    df_train,
                    num_samples=5,  # Default
                    std_factor=0.3,
                    random_state=42
                )
            
            # Calculate class distribution
            if 'r_product_class' in df_train.columns:
                class_counts = df_train['r_product_class'].value_counts().sort_index()
                
                # Ensure all classes are present
                for cls in [0, 1, 2]:
                    if cls not in class_counts.index:
                        class_counts[cls] = 0
                
                class_counts = class_counts.sort_index()
                
                # Create plot (same style as data_analysis.py)
                class_labels = [
                    get_class_label(0, style='default'),
                    get_class_label(1, style='default'),
                    get_class_label(2, style='default')
                ]
                
                counts = [class_counts.get(0, 0), class_counts.get(1, 0), class_counts.get(2, 0)]
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
                
                # Save plot as PNG
                filename_png = f'class_distribution_after_neg_aug_{run_name}.png'
                path_png = os.path.join(plots_dir, filename_png)
                plt.savefig(path_png, dpi=300, bbox_inches='tight')
                print(f"  ✓ Saved PNG to {path_png}")
                
                # Save plot as PDF
                filename_pdf = f'class_distribution_after_neg_aug_{run_name}.pdf'
                path_pdf = os.path.join(plots_dir, filename_pdf)
                plt.savefig(path_pdf, bbox_inches='tight')
                print(f"  ✓ Saved PDF to {path_pdf}")
                
                plt.close()
                break  # Only create one plot for now (first config with neg data)
        
        except Exception as e:
            print(f"  Warning: Could not create class distribution plot for {run_name}: {e}")
            continue


def main():
    """Main plotting pipeline."""
    args = parse_args()
    
    print("="*60)
    print("FILTER SWEEP - PLOT GENERATION")
    print("="*60)
    print(f"\nLoading results from: {args.results_path}")
    
    # Load results
    if not os.path.exists(args.results_path):
        print(f"\nError: Results file not found at {args.results_path}")
        print("Please run sweep_filters.py first to generate results.")
        sys.exit(1)
    
    results_df = pd.read_csv(args.results_path)
    
    # Convert filters column from string to dict if needed
    if isinstance(results_df['filters'].iloc[0], str):
        results_df['filters'] = results_df['filters'].apply(ast.literal_eval)
    
    # Convert confusion_matrix column if needed
    if 'confusion_matrix' in results_df.columns:
        if isinstance(results_df['confusion_matrix'].iloc[0], str):
            results_df['confusion_matrix'] = results_df['confusion_matrix'].apply(ast.literal_eval)
    
    print(f"Loaded {len(results_df)} configurations")
    
    # Create plots
    print("\n" + "="*60)
    print("CREATING PLOTS")
    print("="*60)
    
    plot_sweep_results(results_df, args.plots_dir)
    
    # Plot class distribution after neg data augmentation
    plot_class_distribution_after_neg_data_augmentation(results_df, args.plots_dir)
    
    print("\n" + "="*60)
    print("PLOT GENERATION COMPLETE!")
    print("="*60)
    print(f"\nPlots saved to: {args.plots_dir}/")


if __name__ == "__main__":
    main()

