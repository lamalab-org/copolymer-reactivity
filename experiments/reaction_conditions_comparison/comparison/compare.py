#!/usr/bin/env python3
"""
Compare Full Model (with reaction conditions) vs Model without reaction conditions.
Creates bar plots comparing macro metrics and per-class performance.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix

# Add copol_prediction to path to import plot_config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../copol_prediction'))

from copol_prediction.analysis import plot_config
from utils import load_data_split


def load_results(results_dir):
    """Load results from a model directory."""
    results_path = os.path.join(results_dir, 'results.json')
    
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def calculate_macro_accuracy(cm):
    """Calculate macro accuracy (balanced accuracy) from confusion matrix."""
    if cm.size == 0:
        return float("nan")
    row_sums = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_recall = np.where(row_sums > 0, np.diag(cm) / row_sums, np.nan)
    if np.all(np.isnan(per_class_recall)):
        return float("nan")
    return float(np.nanmean(per_class_recall))


def calculate_macro_precision(cm):
    """Calculate macro precision from confusion matrix."""
    if cm.size == 0:
        return float("nan")
    col_sums = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_class_precision = np.where(col_sums > 0, np.diag(cm) / col_sums, np.nan)
    if np.all(np.isnan(per_class_precision)):
        return float("nan")
    return float(np.nanmean(per_class_precision))


def compute_confusion_matrix_from_model(model_dir, df_test, features):
    """Compute confusion matrix by loading model and evaluating on test data."""
    # Load model
    model_path = os.path.join(model_dir, 'model.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    try:
        model = joblib.load(model_path)
        
        # Prepare test data
        X_test = df_test[features]
        y_test = df_test['r_product_class'].astype(int).values
        
        # Predict and compute CM
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
        
        return cm
    except Exception as e:
        raise RuntimeError(f"Could not compute CM from model: {e}")


def create_comparison_plot(full_results, no_cond_results, 
                          full_cm, no_cond_cm, output_path):
    """Create comparison bar plots with macro metrics matching analyze_model.py style."""
    
    # Apply plot style from copol_prediction
    plot_config.setup_plot_style()
    
    # Create figure with 3 subplots using TWO_COL width
    height = plot_config.TWO_COL_WIDTH_INCH * (5/14) * 1.2
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(plot_config.TWO_COL_WIDTH_INCH, height))
    
    # Colors from plot_config
    color_full = plot_config.COMPARISON_COLORS['original']
    color_no_cond = plot_config.COMPARISON_COLORS['filtered']
    
    width = 0.35
    
    # Calculate macro metrics
    full_macro_f1 = full_results['holdout_f1_macro']
    full_macro_prec = calculate_macro_precision(full_cm)
    no_cond_macro_f1 = no_cond_results['holdout_f1_macro']
    no_cond_macro_prec = calculate_macro_precision(no_cond_cm)
    
    # ===== PLOT 1: Macro Precision and Macro F1 Score =====
    metrics = ['Macro\nPrecision', 'Macro\nF1 Score']
    full_vals = [full_macro_prec, full_macro_f1]
    no_cond_vals = [no_cond_macro_prec, no_cond_macro_f1]
    
    x = np.arange(len(metrics))
    
    bars1 = ax1.bar(x - width/2, full_vals, width, label='With Reaction\nConditions', 
                    color=color_full, alpha=0.95)
    bars2 = ax1.bar(x + width/2, no_cond_vals, width, label='Without Reaction\nConditions',
                    color=color_no_cond, alpha=0.95)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=6)
    
    ax1.set_ylabel('Score', fontsize=8)
    ax1.set_title('a', fontsize=10, loc='left', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, fontsize=7)
    ax1.legend(loc='upper right', fontsize=6)
    ax1.set_ylim(0, 1.05)
    ax1.tick_params(labelsize=6)
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # ===== PLOT 2: Precision per Class =====
    def get_per_class_precision(cm):
        col_sums = cm.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(col_sums > 0, np.diag(cm) / col_sums, 0)
    
    full_precisions = get_per_class_precision(full_cm)
    no_cond_precisions = get_per_class_precision(no_cond_cm)
    
    classes = ['Class 0\n(Alternating)', 'Class 1\n(Random)', 'Class 2\n(Block)']
    x2 = np.arange(len(classes))
    
    bars3 = ax2.bar(x2 - width/2, full_precisions, width, label='With Reaction\nConditions',
                    color=color_full, alpha=0.95)
    bars4 = ax2.bar(x2 + width/2, no_cond_precisions, width, label='Without Reaction\nConditions',
                    color=color_no_cond, alpha=0.95)
    
    # Add value labels on bars
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=6)
    
    ax2.set_ylabel('Precision', fontsize=8)
    ax2.set_title('b', fontsize=10, loc='left', fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes, fontsize=6)
    ax2.legend(loc='upper right', fontsize=6)
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(labelsize=6)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # ===== PLOT 3: Number of Features =====
    full_nfeats = full_results['num_features']
    no_cond_nfeats = no_cond_results['num_features']
    
    models = ['With Reaction\nConditions', 'Without Reaction\nConditions']
    feature_counts = [full_nfeats, no_cond_nfeats]
    x3 = np.arange(len(models))
    
    colors = [color_full, color_no_cond]
    bars5 = ax3.bar(x3, feature_counts, width=width, color=colors, alpha=0.95)
    
    # Add value labels on bars
    for i, bar in enumerate(bars5):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + max(feature_counts)*0.02,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=6)
    
    ax3.set_ylabel('Number of Features', fontsize=8)
    ax3.set_title('c', fontsize=10, loc='left', fontweight='bold')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(models, fontsize=7)
    ax3.tick_params(labelsize=6)
    ax3.grid(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save as PNG
    output_path_png = output_path.replace('.pdf', '') + '.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved PNG to: {output_path_png}")
    
    # Save as PDF
    output_path_pdf = output_path.replace('.png', '.pdf')
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✓ Saved PDF to: {output_path_pdf}")
    
    plt.close()
    
    return fig


def print_comparison_table(full_results, no_cond_results, 
                          full_cm, no_cond_cm):
    """Print a detailed comparison table with macro metrics."""
    print("\n" + "="*80)
    print("DETAILED PERFORMANCE COMPARISON (MACRO METRICS)")
    print("="*80)
    
    # Calculate macro metrics
    full_macro_acc = calculate_macro_accuracy(full_cm)
    full_macro_prec = calculate_macro_precision(full_cm)
    no_cond_macro_acc = calculate_macro_accuracy(no_cond_cm)
    no_cond_macro_prec = calculate_macro_precision(no_cond_cm)
    
    print(f"\n{'Metric':<25} {'With Conditions':<25} {'Without Conditions':<25} {'Δ':<10}")
    print("-"*80)
    
    # Macro Accuracy
    delta_acc = no_cond_macro_acc - full_macro_acc
    print(f"{'Macro Accuracy':<25} {full_macro_acc:<25.4f} {no_cond_macro_acc:<25.4f} {delta_acc:+.4f}")
    
    # Macro Precision
    delta_prec = no_cond_macro_prec - full_macro_prec
    print(f"{'Macro Precision':<25} {full_macro_prec:<25.4f} {no_cond_macro_prec:<25.4f} {delta_prec:+.4f}")
    
    # F1 Macro
    full_f1m = full_results.get('holdout_f1_macro', 0)
    no_cond_f1m = no_cond_results.get('holdout_f1_macro', 0)
    delta_f1m = no_cond_f1m - full_f1m
    print(f"{'F1 (macro)':<25} {full_f1m:<25.4f} {no_cond_f1m:<25.4f} {delta_f1m:+.4f}")
    
    # Accuracy
    full_acc = full_results.get('holdout_accuracy', 0)
    no_cond_acc = no_cond_results.get('holdout_accuracy', 0)
    delta_acc_abs = no_cond_acc - full_acc
    print(f"{'Accuracy':<25} {full_acc:<25.4f} {no_cond_acc:<25.4f} {delta_acc_abs:+.4f}")
    
    print("\n" + "-"*80)
    print("CLASS-WISE PRECISION")
    print("-"*80)
    
    def get_per_class_precision(cm):
        col_sums = cm.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(col_sums > 0, np.diag(cm) / col_sums, 0)
    
    full_precisions = get_per_class_precision(full_cm)
    no_cond_precisions = get_per_class_precision(no_cond_cm)
    
    class_names = ['Class 0 (Alternating)', 'Class 1 (Random)', 'Class 2 (Block)']
    
    for i, (name, full_prec, no_cond_prec) in enumerate(zip(class_names, full_precisions, no_cond_precisions)):
        delta = no_cond_prec - full_prec
        print(f"{name:<25} {full_prec:<25.4f} {no_cond_prec:<25.4f} {delta:+.4f}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if full_macro_acc > no_cond_macro_acc:
        print(f"✓ Model WITH reaction conditions performs better overall (+{(full_macro_acc-no_cond_macro_acc)*100:.2f}% macro accuracy)")
        print(f"  - Reaction conditions provide useful information for prediction")
    else:
        print(f"✓ Model WITHOUT reaction conditions performs better overall (+{(no_cond_macro_acc-full_macro_acc)*100:.2f}% macro accuracy)")
        print(f"  - Reaction conditions may not be necessary for good performance")
    
    print(f"\n  Number of features:")
    print(f"  - With conditions: {full_results.get('num_features', 0)}")
    print(f"  - Without conditions: {no_cond_results.get('num_features', 0)}")
    print(f"  - Removed: {full_results.get('num_features', 0) - no_cond_results.get('num_features', 0)} reaction condition features")
    
    print(f"\n  CV Scores (5-fold):")
    print(f"  - With conditions: {full_results.get('cv_score', 0):.4f}")
    print(f"  - Without conditions: {no_cond_results.get('cv_score', 0):.4f}")
    
    print("\n" + "="*80)


def main():
    print("="*80)
    print("COMPARING FULL MODEL VS MODEL WITHOUT REACTION CONDITIONS")
    print("="*80)
    
    # Paths
    full_dir = os.path.join(os.path.dirname(__file__), '../full_model/results')
    no_cond_dir = os.path.join(os.path.dirname(__file__), '../no_reaction_conditions/results')
    output_plot = os.path.join(os.path.dirname(__file__), 'plots/comparison_reaction_conditions.png')
    
    # Create plots directory
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    
    # Load results
    print(f"\nLoading results...")
    print(f"  - Full model: {full_dir}")
    print(f"  - No conditions: {no_cond_dir}")
    
    full_results = load_results(full_dir)
    no_cond_results = load_results(no_cond_dir)
    
    # Load test data
    print(f"\nLoading test data...")
    # Change to copol_prediction directory to use relative paths
    script_dir = os.path.dirname(__file__)
    copol_pred_dir = os.path.join(script_dir, '../../../copol_prediction')
    copol_pred_dir = os.path.abspath(copol_pred_dir)
    
    original_cwd = os.getcwd()
    os.chdir(copol_pred_dir)
    
    try:
        df_train, df_test = load_data_split.load_train_test_split()
    finally:
        os.chdir(original_cwd)
    
    # Compute confusion matrices
    print(f"\nComputing confusion matrices...")
    full_cm = compute_confusion_matrix_from_model(full_dir, df_test, full_results['features'])
    no_cond_cm = compute_confusion_matrix_from_model(no_cond_dir, df_test, no_cond_results['features'])
    
    print(f"  - Full model CM shape: {full_cm.shape}")
    print(f"  - No conditions CM shape: {no_cond_cm.shape}")
    
    # Print comparison table
    print_comparison_table(full_results, no_cond_results, full_cm, no_cond_cm)
    
    # Create plot
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)
    
    create_comparison_plot(full_results, no_cond_results, 
                          full_cm, no_cond_cm, output_plot)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nComparison plot saved to: {output_plot}")


if __name__ == "__main__":
    main()

