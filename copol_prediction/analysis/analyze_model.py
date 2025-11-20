#!/usr/bin/env python3
"""
Model analysis script for copolymerization prediction.

Generates various analysis plots for trained models.

Usage:
    python analyze_model.py [--all] [--combined] [--confusion] [--confidence] [--features] [--calibration]
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from copolpredictor.inference import CopolymerPredictor
from plot_config import (
    setup_plot_style, 
    CLASS_COLORS, 
    CLASS_LABELS, 
    COMPARISON_COLORS,
    SEQUENTIAL_COLORS,
    HIGHLIGHT_COLORS,
    CONFUSION_MATRIX_CONFIG,
    CONFIDENCE_PLOT_CONFIG,
    FEATURE_IMPORTANCE_CONFIG,
    CALIBRATION_CONFIG,
    ERROR_ANALYSIS_CONFIG,
    get_class_color,
    get_class_label,
    ONE_COL_WIDTH_INCH,
    TWO_COL_WIDTH_INCH,
    ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
    TWO_COL_GOLDEN_RATIO_HEIGHT_INCH,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze trained model")
    parser.add_argument("--model-path", default="../artifacts/model_bundle", help="Path to model bundle")
    parser.add_argument("--data-path", default="../output/processed_data.csv", help="Path to processed data")
    parser.add_argument("--output-dir", default="../output/analysis", help="Output directory for plots")
    parser.add_argument("--holdout-only", action="store_true", help="Use only holdout set")
    parser.add_argument("--compare-holdout", action="store_true", help="Generate plots for both all data and holdout set")
    
    # Plot selection
    parser.add_argument("--all", action="store_true", help="Generate all plots")
    parser.add_argument("--combined", action="store_true", help="Combined confusion matrix and confidence plot")
    parser.add_argument("--confusion", action="store_true", help="Confusion matrix")
    parser.add_argument("--confidence", action="store_true", help="Confidence distribution")
    parser.add_argument("--features", action="store_true", help="Feature importance")
    parser.add_argument("--calibration", action="store_true", help="Calibration curve")
    parser.add_argument("--errors", action="store_true", help="Error analysis by class")
    parser.add_argument("--confidence-vs-r1r2", action="store_true", help="Confidence vs r1r2 plot")
    parser.add_argument("--filtering", action="store_true", help="Confidence filtering analysis")
    parser.add_argument("--min-retention", type=float, default=0.7, help="Minimum retention rate for filtering (default: 0.7)")
    
    return parser.parse_args()


def setup_style():
    """Setup matplotlib style."""
    setup_plot_style()  # Load lamalab.mplstyle and set color scheme


def plot_confusion_matrix_and_confidence(y_true, y_pred, confidence_scores, correct_mask, output_dir, suffix=''):
    """Plot confusion matrix and confidence distribution in combined figure."""
    print(f"Generating combined confusion matrix and confidence plot{' (' + suffix + ')' if suffix else ''}...")
    
    # Create figure with 2 subplots (TWO_COL width, maintaining original 14:5 aspect ratio)
    # Original was (14, 5), so height = width * (5/14)
    height = TWO_COL_WIDTH_INCH * (5/14)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH_INCH, height))
    
    # Left subplot: Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[get_class_label(i) for i in range(3)]
    )
    im = disp.plot(cmap=CONFUSION_MATRIX_CONFIG['cmap'], ax=ax1, 
              values_format=CONFUSION_MATRIX_CONFIG['values_format'],
              im_kw={'vmin': 0, 'vmax': 2500},
              text_kw={'fontsize': 7})
    ax1.set_title('a', fontsize=10, loc='left', fontweight='bold')
    ax1.set_xlabel(ax1.get_xlabel(), fontsize=8)
    ax1.set_ylabel(ax1.get_ylabel(), fontsize=8)
    ax1.tick_params(labelsize=6)
    ax1.grid(False)
    # Adjust colorbar font size
    if im.im_ is not None:
        cbar = im.im_.colorbar
        if cbar is not None:
            cbar.ax.tick_params(labelsize=6)
    
    # Right subplot: Confidence Distribution (Correct vs Incorrect)
    correct_conf = confidence_scores[correct_mask]
    incorrect_conf = confidence_scores[~correct_mask]
    
    ax2.hist(correct_conf, bins=CONFIDENCE_PLOT_CONFIG['bins'], 
             alpha=CONFIDENCE_PLOT_CONFIG['alpha'], label='Correct', 
             color=COMPARISON_COLORS['correct'], edgecolor=CONFIDENCE_PLOT_CONFIG['edgecolor'])
    ax2.hist(incorrect_conf, bins=CONFIDENCE_PLOT_CONFIG['bins'], 
             alpha=CONFIDENCE_PLOT_CONFIG['alpha'], label='Incorrect', 
             color=COMPARISON_COLORS['incorrect'], edgecolor=CONFIDENCE_PLOT_CONFIG['edgecolor'])
    ax2.set_xlabel('Confidence Score', fontsize=8)
    ax2.set_ylabel('Count', fontsize=8)
    ax2.set_title('b', fontsize=10, loc='left', fontweight='bold')
    ax2.legend(fontsize=7)
    ax2.grid(False)
    ax2.tick_params(labelsize=6)
    # Remove top and right spines (box)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save as PNG
    filename_png = f'confusion_and_confidence{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.png'
    path_png = os.path.join(output_dir, filename_png)
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved PNG to {path_png}")
    
    # Save as PDF
    filename_pdf = f'confusion_and_confidence{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.pdf'
    path_pdf = os.path.join(output_dir, filename_pdf)
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"  ✓ Saved PDF to {path_pdf}")
    
    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_dir, suffix=''):
    """Plot confusion matrix."""
    print(f"Generating confusion matrix{' (' + suffix + ')' if suffix else ''}...")
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_INCH, 3))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[get_class_label(i) for i in range(3)]
    )
    disp.plot(cmap=CONFUSION_MATRIX_CONFIG['cmap'], ax=ax, 
              values_format=CONFUSION_MATRIX_CONFIG['values_format'],
              im_kw={'vmin': 0, 'vmax': 2500})
    
    title = 'Confusion Matrix' + (' - ' + suffix if suffix else '')
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()
    
    filename = f'confusion_matrix{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.png'
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to {path}")
    
    # Also save normalized version
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_INCH, 3))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm,
        display_labels=[get_class_label(i) for i in range(3)]
    )
    disp.plot(cmap=CONFUSION_MATRIX_CONFIG['cmap'], ax=ax, values_format='.2f')
    
    title_norm = 'Normalized Confusion Matrix' + (' - ' + suffix if suffix else '')
    plt.title(title_norm, fontsize=14, pad=20)
    plt.tight_layout()
    
    filename_norm = f'confusion_matrix_normalized{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.png'
    path_norm = os.path.join(output_dir, filename_norm)
    plt.savefig(path_norm, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to {path_norm}")


def plot_confidence_distribution(confidence_scores, correct_mask, output_dir, suffix=''):
    """Plot confidence score distribution."""
    print(f"Generating confidence distribution plot{' (' + suffix + ')' if suffix else ''}...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH_INCH, 3.5))
    
    # Overall distribution
    ax1.hist(confidence_scores, bins=CONFIDENCE_PLOT_CONFIG['bins'], 
             edgecolor=CONFIDENCE_PLOT_CONFIG['edgecolor'], 
             alpha=CONFIDENCE_PLOT_CONFIG['alpha'])
    ax1.axvline(confidence_scores.mean(), color=HIGHLIGHT_COLORS['mean'], linestyle='--', 
                label=f'Mean: {confidence_scores.mean():.3f}')
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Overall Confidence Distribution', fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Correct vs incorrect
    correct_conf = confidence_scores[correct_mask]
    incorrect_conf = confidence_scores[~correct_mask]
    
    ax2.hist(correct_conf, bins=CONFIDENCE_PLOT_CONFIG['bins'], 
             alpha=CONFIDENCE_PLOT_CONFIG['alpha'], label='Correct', 
             color=COMPARISON_COLORS['correct'], edgecolor=CONFIDENCE_PLOT_CONFIG['edgecolor'])
    ax2.hist(incorrect_conf, bins=CONFIDENCE_PLOT_CONFIG['bins'], 
             alpha=CONFIDENCE_PLOT_CONFIG['alpha'], label='Incorrect', 
             color=COMPARISON_COLORS['incorrect'], edgecolor=CONFIDENCE_PLOT_CONFIG['edgecolor'])
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Confidence: Correct vs Incorrect', fontsize=13)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    
    filename = f'confidence_distribution{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.png'
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to {path}")
    
    # Print statistics
    print(f"  Mean confidence (correct): {correct_conf.mean():.3f}")
    print(f"  Mean confidence (incorrect): {incorrect_conf.mean():.3f}")


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


def plot_feature_importance(predictor, output_dir, top_n=20):
    """Plot feature importance from model."""
    print("Generating feature importance plot...")
    
    importance_df = predictor.get_feature_importance()
    top_features = importance_df.head(top_n)
    
    # Format feature names
    formatted_names = [format_feature_name(name) for name in top_features['feature']]
    
    # Use TWO_COL width, dynamic height based on number of features
    height = max(4, top_n * 0.2)
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, height))
    
    ax.barh(range(len(top_features)), top_features['importance'], 
            color=FEATURE_IMPORTANCE_CONFIG['color'])
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(formatted_names, fontsize=7)
    ax.set_xlabel('Importance', fontsize=9)
    ax.tick_params(axis='x', labelsize=7)
    ax.invert_yaxis()
    ax.grid(False)
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    
    # Save as PNG
    path_png = os.path.join(output_dir, 'feature_importance.png')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved PNG to {path_png}")
    
    # Save as PDF
    path_pdf = os.path.join(output_dir, 'feature_importance.pdf')
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"  ✓ Saved PDF to {path_pdf}")
    
    plt.close()
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'feature_importance.csv')
    importance_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved CSV to {csv_path}")


def plot_calibration_curve_multiclass(y_true, y_proba, output_dir, suffix=''):
    """Plot calibration curves for each class."""
    print(f"Generating calibration curves{' (' + suffix + ')' if suffix else ''}...")
    
    # Original was (15, 5), ratio 3:1. With width 7, height = 7/3 ≈ 2.33, let's use 3 for better visibility
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL_WIDTH_INCH, 3))
    
    class_names = [get_class_label(i, style='long') for i in range(3)]
    
    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        # Binary indicator for this class
        y_binary = (y_true == i).astype(int)
        y_prob_class = y_proba[:, i]
        
        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(
            y_binary, y_prob_class, n_bins=10, strategy='uniform'
        )
        
        # Plot
        ax.plot(prob_pred, prob_true, marker=CALIBRATION_CONFIG['marker'], 
                linewidth=1.5, 
                markersize=4,
                color=get_class_color(i), label='Model')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        
        ax.set_xlabel('Mean Predicted Probability', fontsize=8)
        ax.set_ylabel('Fraction of Positives', fontsize=8)
        ax.set_title(class_name, fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
        ax.grid(False)
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save as PNG
    filename_png = f'calibration_curves{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.png'
    path_png = os.path.join(output_dir, filename_png)
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved PNG to {path_png}")
    
    # Save as PDF
    filename_pdf = f'calibration_curves{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.pdf'
    path_pdf = os.path.join(output_dir, filename_pdf)
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"  ✓ Saved PDF to {path_pdf}")
    
    plt.close()


def plot_error_analysis_by_class(y_true, y_pred, confidence_scores, output_dir, suffix=''):
    """Analyze errors by true class."""
    print(f"Generating error analysis{' (' + suffix + ')' if suffix else ''}...")
    
    # Original was (15, 5), ratio 3:1. With width 7, use height 3 for better visibility
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL_WIDTH_INCH, 3))
    
    class_names = [get_class_label(i, style='long') for i in range(3)]
    
    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        mask = y_true == i
        correct = (y_true[mask] == y_pred[mask])
        conf = confidence_scores[mask]
        
        # Plot confidence for correct vs incorrect
        correct_conf = conf[correct]
        incorrect_conf = conf[~correct]
        
        ax.hist(correct_conf, bins=ERROR_ANALYSIS_CONFIG['bins'], 
                alpha=ERROR_ANALYSIS_CONFIG['alpha'], 
                label=f'Correct ({len(correct_conf)})', 
                color=COMPARISON_COLORS['correct'], 
                edgecolor=ERROR_ANALYSIS_CONFIG['edgecolor'])
        ax.hist(incorrect_conf, bins=ERROR_ANALYSIS_CONFIG['bins'], 
                alpha=ERROR_ANALYSIS_CONFIG['alpha'], 
                label=f'Incorrect ({len(incorrect_conf)})', 
                color=COMPARISON_COLORS['incorrect'], 
                edgecolor=ERROR_ANALYSIS_CONFIG['edgecolor'])
        
        ax.set_xlabel('Confidence Score', fontsize=8)
        ax.set_ylabel('Count', fontsize=8)
        ax.set_title(class_name, fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
        ax.grid(False)
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save as PNG
    filename_png = f'error_analysis_by_class{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.png'
    path_png = os.path.join(output_dir, filename_png)
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved PNG to {path_png}")
    
    # Save as PDF
    filename_pdf = f'error_analysis_by_class{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.pdf'
    path_pdf = os.path.join(output_dir, filename_pdf)
    plt.savefig(path_pdf, bbox_inches='tight')
    print(f"  ✓ Saved PDF to {path_pdf}")
    
    plt.close()


def plot_confidence_vs_r1r2(df, predictions, confidence_scores, output_dir, suffix=''):
    """Plot confidence vs r1r2 value."""
    print(f"Generating confidence vs r1r2 plot{' (' + suffix + ')' if suffix else ''}...")
    
    # Create plot data
    plot_df = pd.DataFrame({
        'r1r2': df['r1r2'],
        'confidence': confidence_scores,
        'predicted_class': predictions
    })
    
    # Filter extreme values for better visualization
    plot_df = plot_df[(plot_df['r1r2'] > 0.01) & (plot_df['r1r2'] < 100)]
    
    plt.figure(figsize=(TWO_COL_WIDTH_INCH, 3))
    
    # Scatter plot
    for cls in [0, 1, 2]:
        mask = plot_df['predicted_class'] == cls
        plt.scatter(
            plot_df.loc[mask, 'r1r2'],
            plot_df.loc[mask, 'confidence'],
            alpha=0.5, s=20, c=get_class_color(cls),
            label=get_class_label(cls, style='short'), edgecolors='none'
        )
    
    # Class boundaries (r1*r2 product)
    plt.axvline(1, color='gray', linestyle='--', linewidth=1.5, label='Class boundaries')
    plt.axvline(25, color='gray', linestyle='--', linewidth=1.5)
    
    # Moving average
    plot_df_sorted = plot_df.sort_values('r1r2')
    window_size = max(50, len(plot_df) // 50)
    rolling_mean = plot_df_sorted['confidence'].rolling(window=window_size, center=True).mean()
    plt.plot(plot_df_sorted['r1r2'], rolling_mean, 'r-', linewidth=2, 
             label=f'Rolling mean (n={window_size})')
    
    plt.xlabel('r1×r2', fontsize=12)
    plt.ylabel('Confidence Score', fontsize=12)
    title = 'Prediction Confidence vs r-Product' + (' - ' + suffix if suffix else '')
    plt.title(title, fontsize=14)
    plt.xlim(0, 50)
    plt.ylim(0, 1.05)
    plt.legend(loc='best')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    filename = f'confidence_vs_r1r2{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.png'
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to {path}")


def print_classification_report(y_true, y_pred):
    """Print classification metrics."""
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    
    report = classification_report(
        y_true, y_pred,
        target_names=[get_class_label(i, style='long') for i in range(3)],
        digits=3
    )
    print(report)


def find_optimal_threshold_per_class(y_true, y_pred, confidence_scores, min_retention=0.7):
    """
    Find optimal confidence threshold for each class.
    
    Keeps at least min_retention (70%) of predictions per class,
    but removes incorrect predictions with high confidence if they
    outnumber correct ones at that confidence level.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidence_scores: Confidence scores
        min_retention: Minimum fraction of predictions to keep per class
        
    Returns:
        Dictionary with threshold per class and filtered indices
    """
    thresholds = {}
    filtered_indices = []
    
    for cls in [0, 1, 2]:
        # Get predictions for this class
        class_mask = y_pred == cls
        class_indices = np.where(class_mask)[0]
        
        if len(class_indices) == 0:
            continue
        
        # Sort by confidence (descending)
        class_conf = confidence_scores[class_indices]
        class_true = y_true[class_indices]
        class_pred = y_pred[class_indices]
        
        sorted_idx = np.argsort(class_conf)[::-1]
        sorted_conf = class_conf[sorted_idx]
        sorted_true = class_true[sorted_idx]
        sorted_pred = class_pred[sorted_idx]
        
        # Calculate number to keep (at least min_retention)
        min_keep = int(len(class_indices) * min_retention)
        
        # Find optimal threshold
        best_threshold = 0.0
        best_idx = len(class_indices)
        
        for i in range(min_keep, len(class_indices)):
            # Count correct and incorrect up to this point
            correct = (sorted_true[:i] == sorted_pred[:i]).sum()
            incorrect = i - correct
            
            # If incorrect > correct at this confidence level, cut here
            if incorrect > correct and i >= min_keep:
                best_threshold = sorted_conf[i-1]
                best_idx = i
                break
        else:
            # Keep all if no good cutoff found
            best_threshold = sorted_conf[-1] if len(sorted_conf) > 0 else 0.0
            best_idx = len(class_indices)
        
        thresholds[cls] = best_threshold
        
        # Add indices to keep
        keep_mask = class_conf >= best_threshold
        filtered_indices.extend(class_indices[keep_mask])
    
    return thresholds, np.array(filtered_indices)


def analyze_confidence_filtering(y_true, y_pred, confidence_scores, output_dir, min_retention=0.7):
    """
    Perform dynamic confidence filtering analysis.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidence_scores: Confidence scores
        output_dir: Output directory
        min_retention: Minimum retention rate per class
    """
    print("Generating confidence filtering analysis...")
    
    # Find optimal thresholds
    thresholds, filtered_indices = find_optimal_threshold_per_class(
        y_true, y_pred, confidence_scores, min_retention
    )
    
    # Filter data
    y_true_filtered = y_true[filtered_indices]
    y_pred_filtered = y_pred[filtered_indices]
    conf_filtered = confidence_scores[filtered_indices]
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    # Original metrics
    orig_acc = accuracy_score(y_true, y_pred)
    orig_f1 = f1_score(y_true, y_pred, average='weighted')
    
    # Filtered metrics
    filt_acc = accuracy_score(y_true_filtered, y_pred_filtered)
    filt_f1 = f1_score(y_true_filtered, y_pred_filtered, average='weighted')
    
    # Per-class statistics
    class_stats = []
    for cls in [0, 1, 2]:
        orig_mask = y_pred == cls
        filt_mask = y_pred_filtered == cls
        
        orig_count = orig_mask.sum()
        filt_count = filt_mask.sum()
        retention = filt_count / orig_count if orig_count > 0 else 0
        
        orig_acc_cls = accuracy_score(y_true[orig_mask], y_pred[orig_mask]) if orig_count > 0 else 0
        filt_acc_cls = accuracy_score(y_true_filtered[filt_mask], y_pred_filtered[filt_mask]) if filt_count > 0 else 0
        
        class_stats.append({
            'class': cls,
            'threshold': thresholds.get(cls, 0.0),
            'original_count': orig_count,
            'filtered_count': filt_count,
            'retention_rate': retention,
            'original_accuracy': orig_acc_cls,
            'filtered_accuracy': filt_acc_cls,
            'accuracy_gain': filt_acc_cls - orig_acc_cls
        })
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(TWO_COL_WIDTH_INCH, 6))
    
    # 1. Threshold and retention per class
    ax1 = axes[0, 0]
    class_names = [get_class_label(i) for i in range(3)]
    thresholds_list = [s['threshold'] for s in class_stats]
    retention_list = [s['retention_rate'] for s in class_stats]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax1_twin = ax1.twinx()
    color1 = SEQUENTIAL_COLORS[0]
    color2 = SEQUENTIAL_COLORS[1]
    bars1 = ax1.bar(x - width/2, thresholds_list, width, label='Threshold', color=color1, alpha=0.7)
    bars2 = ax1_twin.bar(x + width/2, retention_list, width, label='Retention', color=color2, alpha=0.7)
    
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Confidence Threshold', color=color1)
    ax1_twin.set_ylabel('Retention Rate', color=color2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names)
    ax1.set_title('Threshold and Retention per Class')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1_twin.tick_params(axis='y', labelcolor=color2)
    ax1.grid(alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, thresholds_list)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                f'{val:.3f}', ha='center', va='center', fontsize=9)
    for i, (bar, val) in enumerate(zip(bars2, retention_list)):
        ax1_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                     f'{val:.1%}', ha='center', va='center', fontsize=9)
    
    # 2. Accuracy comparison
    ax2 = axes[0, 1]
    orig_acc_list = [s['original_accuracy'] for s in class_stats]
    filt_acc_list = [s['filtered_accuracy'] for s in class_stats]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax2.bar(x - width/2, orig_acc_list, width, label='Original', 
            color=COMPARISON_COLORS['original'], alpha=0.7)
    ax2.bar(x + width/2, filt_acc_list, width, label='Filtered', 
            color=COMPARISON_COLORS['filtered'], alpha=0.7)
    
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy: Original vs Filtered')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Add value labels
    for i, v in enumerate(orig_acc_list):
        ax2.text(i - width/2, v + 0.02, f'{v:.2%}', ha='center', fontsize=9)
    for i, v in enumerate(filt_acc_list):
        ax2.text(i + width/2, v + 0.02, f'{v:.2%}', ha='center', fontsize=9)
    
    # 3. Overall metrics comparison
    ax3 = axes[1, 0]
    metrics = ['Accuracy', 'F1 (weighted)']
    orig_metrics = [orig_acc, orig_f1]
    filt_metrics = [filt_acc, filt_f1]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax3.bar(x - width/2, orig_metrics, width, label='Original', 
            color=COMPARISON_COLORS['original'], alpha=0.7)
    ax3.bar(x + width/2, filt_metrics, width, label='Filtered', 
            color=COMPARISON_COLORS['filtered'], alpha=0.7)
    
    ax3.set_ylabel('Score')
    ax3.set_title('Overall Metrics Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Add value labels
    for i, v in enumerate(orig_metrics):
        ax3.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    for i, v in enumerate(filt_metrics):
        ax3.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
    
    # 4. Sample count comparison
    ax4 = axes[1, 1]
    orig_counts = [s['original_count'] for s in class_stats]
    filt_counts = [s['filtered_count'] for s in class_stats]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    ax4.bar(x - width/2, orig_counts, width, label='Original', 
            color=COMPARISON_COLORS['original'], alpha=0.7)
    ax4.bar(x + width/2, filt_counts, width, label='Filtered', 
            color=COMPARISON_COLORS['filtered'], alpha=0.7)
    
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Sample Count')
    ax4.set_title('Sample Count per Class')
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_names)
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(orig_counts):
        ax4.text(i - width/2, v + max(orig_counts)*0.02, f'{v}', ha='center', fontsize=9)
    for i, v in enumerate(filt_counts):
        ax4.text(i + width/2, v + max(orig_counts)*0.02, f'{v}', ha='center', fontsize=9)
    
    plt.tight_layout()
    
    path = os.path.join(output_dir, 'confidence_filtering_analysis.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved to {path}")
    
    # Save detailed report
    report_path = os.path.join(output_dir, 'confidence_filtering_report.txt')
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CONFIDENCE FILTERING ANALYSIS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Minimum retention rate: {min_retention:.1%}\n")
        f.write(f"Total samples: {len(y_true)} → {len(y_true_filtered)} ({len(y_true_filtered)/len(y_true):.1%})\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Metric':<20} {'Original':<15} {'Filtered':<15} {'Gain':<15}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Accuracy':<20} {orig_acc:<15.4f} {filt_acc:<15.4f} {filt_acc-orig_acc:+<15.4f}\n")
        f.write(f"{'F1 (weighted)':<20} {orig_f1:<15.4f} {filt_f1:<15.4f} {filt_f1-orig_f1:+<15.4f}\n\n")
        
        f.write("PER-CLASS STATISTICS\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<10} {'Threshold':<12} {'Orig Count':<12} {'Filt Count':<12} {'Retention':<12} {'Orig Acc':<10} {'Filt Acc':<10} {'Gain':<10}\n")
        f.write("-"*80 + "\n")
        
        for stats in class_stats:
            f.write(
                f"{stats['class']:<10} "
                f"{stats['threshold']:<12.4f} "
                f"{stats['original_count']:<12d} "
                f"{stats['filtered_count']:<12d} "
                f"{stats['retention_rate']:<12.1%} "
                f"{stats['original_accuracy']:<10.3f} "
                f"{stats['filtered_accuracy']:<10.3f} "
                f"{stats['accuracy_gain']:+<10.3f}\n"
            )
        
        f.write("\n" + "="*80 + "\n")
        f.write("FILTERED CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        report = classification_report(
            y_true_filtered, y_pred_filtered,
            target_names=[get_class_label(i, style='long') for i in range(3)],
            digits=4
        )
        f.write(report)
    
    print(f"  ✓ Saved report to {report_path}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("CONFIDENCE FILTERING SUMMARY")
    print("="*60)
    print(f"Retention: {len(y_true_filtered)}/{len(y_true)} ({len(y_true_filtered)/len(y_true):.1%})")
    print(f"Accuracy:  {orig_acc:.4f} → {filt_acc:.4f} (Δ {filt_acc-orig_acc:+.4f})")
    print(f"F1:        {orig_f1:.4f} → {filt_f1:.4f} (Δ {filt_f1-orig_f1:+.4f})")
    print("\nPer-class thresholds:")
    for stats in class_stats:
        print(f"  Class {stats['class']}: {stats['threshold']:.4f} "
              f"(retention: {stats['retention_rate']:.1%}, "
              f"acc gain: {stats['accuracy_gain']:+.3f})")
    print("="*60)


def generate_plots_for_dataset(df, predictor, args, suffix=''):
    """Generate plots for a given dataset."""
    # Prepare features
    try:
        X = df[predictor.features]
        
        # Get true labels if available
        if 'r_product_class' in df.columns:
            y_true = df['r_product_class'].values
        else:
            # Create labels from r1r2
            bins = [-np.inf, 1, 25, np.inf]
            labels = [0, 1, 2]
            y_true = pd.cut(df['r1r2'], bins=bins, labels=labels, right=False).astype(int).values
    except Exception as e:
        print(f"  ✗ Error preparing features: {e}")
        return
    
    # Make predictions
    try:
        results = predictor.predict_with_confidence(X)
        y_pred = results['predictions']
        y_proba = results['probabilities']
        confidence = results['confidence']
        correct_mask = y_pred == y_true
        
        print(f"  Accuracy: {correct_mask.mean():.3f}")
    except Exception as e:
        print(f"  ✗ Error making predictions: {e}")
        return
    
    # Print classification report
    print_classification_report(y_true, y_pred)
    
    # Determine which plots to generate
    generate_all = args.all or not any([
        args.combined, args.confusion, args.confidence, args.features, 
        args.calibration, args.errors, args.confidence_vs_r1r2, args.filtering
    ])
    
    # Generate plots
    print(f"\nGenerating plots{' (' + suffix + ')' if suffix else ''}...")
    
    if generate_all or args.combined:
        plot_confusion_matrix_and_confidence(y_true, y_pred, confidence, correct_mask, args.output_dir, suffix)
    
    if generate_all or args.confusion:
        plot_confusion_matrix(y_true, y_pred, args.output_dir, suffix)
    
    if generate_all or args.confidence:
        plot_confidence_distribution(confidence, correct_mask, args.output_dir, suffix)
    
    if (generate_all or args.features) and not suffix:  # Only once for features
        plot_feature_importance(predictor, args.output_dir)
    
    if generate_all or args.calibration:
        plot_calibration_curve_multiclass(y_true, y_proba, args.output_dir, suffix)
    
    if generate_all or args.errors:
        plot_error_analysis_by_class(y_true, y_pred, confidence, args.output_dir, suffix)
    
    if (generate_all or args.confidence_vs_r1r2) and 'r1r2' in df.columns:
        plot_confidence_vs_r1r2(df, y_pred, confidence, args.output_dir, suffix)
    
    if (generate_all or args.filtering) and not suffix:  # Only once for filtering
        analyze_confidence_filtering(y_true, y_pred, confidence, args.output_dir, args.min_retention)


def main():
    """Main analysis pipeline."""
    args = parse_args()
    
    # Setup
    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("MODEL ANALYSIS")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    
    # Load model
    print("\nLoading model...")
    try:
        predictor = CopolymerPredictor(args.model_path)
        print(f"  ✓ Model loaded ({len(predictor.features)} features)")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        sys.exit(1)
    
    # Load data
    print("\nLoading data...")
    try:
        df_all = pd.read_csv(args.data_path)
        print(f"  ✓ Data loaded ({len(df_all)} samples)")
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        sys.exit(1)
    
    # Generate plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)
    
    if args.compare_holdout:
        # Generate plots for both all data and holdout set
        print("\n### All Data ###")
        print(f"  Samples: {len(df_all)}")
        generate_plots_for_dataset(df_all, predictor, args, suffix='All Data')
        
        print("\n### Holdout Set ###")
        from copolpredictor.holdout_utils import get_or_create_holdout_groups, make_base_dataset_for_holdout
        try:
            base_df = make_base_dataset_for_holdout(df_all)
            holdout_groups = get_or_create_holdout_groups(base_df)
            df_holdout = df_all[df_all['reaction_id'].astype(str).isin(holdout_groups)].reset_index(drop=True)
            print(f"  Samples: {len(df_holdout)}")
            generate_plots_for_dataset(df_holdout, predictor, args, suffix='Holdout')
        except Exception as e:
            print(f"  ✗ Could not filter to holdout: {e}")
    
    elif args.holdout_only:
        # Only holdout set
        print("\n### Holdout Set ###")
        from copolpredictor.holdout_utils import get_or_create_holdout_groups, make_base_dataset_for_holdout
        try:
            base_df = make_base_dataset_for_holdout(df_all)
            holdout_groups = get_or_create_holdout_groups(base_df)
            df_holdout = df_all[df_all['reaction_id'].astype(str).isin(holdout_groups)].reset_index(drop=True)
            print(f"  Samples: {len(df_holdout)}")
            generate_plots_for_dataset(df_holdout, predictor, args, suffix='')
        except Exception as e:
            print(f"  ✗ Could not filter to holdout: {e}")
            print(f"  Using all data instead")
            generate_plots_for_dataset(df_all, predictor, args, suffix='')
    
    else:
        # Only all data
        print(f"\n  Samples: {len(df_all)}")
        generate_plots_for_dataset(df_all, predictor, args, suffix='')
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nAll plots saved to: {args.output_dir}/")
    print("\nGenerated plots:")
    for file in sorted(os.listdir(args.output_dir)):
        if file.endswith('.png'):
            print(f"  - {file}")


if __name__ == "__main__":
    main()

