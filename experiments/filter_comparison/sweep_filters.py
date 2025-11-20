#!/usr/bin/env python3
"""
Filter sweep script for copolymerization prediction.

This script tests different combinations of data filters and preprocessing
options to find the best configuration for the model.

IMPORTANT: This script uses the global train/test split from experiments/data/
(created by create_train_test_split.py) to ensure consistency with other
experiments (baseline, fingerprint, etc.). All filter configurations are
evaluated on the same holdout set for fair comparison.

Usage:
    python sweep_filters.py [--data-path PATH] [--output-dir DIR]
"""

import os
import sys
import json
import argparse
import itertools
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from copolpredictor import (
    data_processing,
    data_augmentation,
    model_training,
    evaluation,
    holdout_utils,
    prediction_utils
)

# Import plot configuration
try:
    from copol_prediction.analysis.plot_config import setup_plot_style, HEATMAP_CMAP
except ImportError:
    def setup_plot_style():
        pass
    HEATMAP_CMAP = 'Blues'


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Sweep over filter combinations for copolymerization prediction"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="../data_extraction/extracted_reactions.csv",
        help="Path to input data CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/experiments_holdout",
        help="Directory to save results"
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default="output/model_comp",
        help="Directory to save plots"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=50,
        help="Hyperparameter search iterations per run (increased to match classification.py)"
    )
    parser.add_argument(
        "--augmentation-samples",
        type=int,
        default=5,
        help="Number of augmented samples per datapoint"
    )
    
    return parser.parse_args()


def generate_filter_combinations(search_space: Dict[str, List[bool]]):
    """
    Generate all combinations from a boolean search space.
    
    Args:
        search_space: Dictionary mapping filter names to lists of boolean values
        
    Yields:
        Tuple of (combination dict, keys list)
    """
    keys = list(search_space.keys())
    for values in itertools.product(*(search_space[k] for k in keys)):
        yield dict(zip(keys, values))


# Note: This script now uses the global train/test split from experiments/data/
# for consistency with other experiments (baseline, fingerprint, etc.)


def plot_confusion_matrix(cm, labels, title, save_path):
    """
    Plot and save a confusion matrix.
    
    Args:
        cm: Confusion matrix array
        labels: Class labels
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title, fontsize=14)
    plt.colorbar()
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.ylabel('True Class', fontsize=12)
    
    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved confusion matrix to {save_path}")


def prepare_filtered_data(df, filters, config, *, random_state=42, debug=True):
    """
    Load global train/test split and apply filters.
    Uses the pre-split data from experiments/data/ for consistency across all experiments.
    Returns (train_df, test_df, features).
    """

    def log(msg):
        if debug:
            print(msg)

    # ---------------- 1) Load global train/test split ----------------
    log("\n[Split] Loading global train/test split...")
    
    train_path = os.path.join(os.path.dirname(__file__), '../data/train.csv')
    test_path = os.path.join(os.path.dirname(__file__), '../data/test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Global train/test split not found!\n"
            f"Expected:\n  - {train_path}\n  - {test_path}\n"
            f"Run: cd ../copol_prediction && python create_data_split.py"
        )
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    log(f"[Split] Loaded train: {len(df_train)} samples ({df_train['reaction_id'].nunique()} groups)")
    log(f"[Split] Loaded test: {len(df_test)} samples ({df_test['reaction_id'].nunique()} groups)")

    # ---------------- 2) Apply filters to both train and test ----------------
    def apply_filters_to_set(df_set, set_name):
        """Apply filters to a dataset (train or test)."""
        w = df_set.copy()
        t0 = len(w)
        
        # Basic validation (should already be done, but double-check)
        w = w[w['r1r2'].notna() & (w['r1r2'] >= 0)]
        if len(w) < t0:
            log(f"[{set_name}] r1r2 valid: {t0} → {len(w)}")

        # specialized filter
        if filters.get('remove_specialized', False):
            if 'llm_specialized_filter' not in w.columns:
                for spec_path in [
                    "../../copol_prediction/filter/llm_specialized_filter/classified_output.csv",
                    "../copol_prediction/filter/llm_specialized_filter/classified_output.csv",
                ]:
                    if os.path.exists(spec_path):
                        df_spec = pd.read_csv(spec_path)
                        if {'specialized_filter','reaction_id'}.issubset(df_spec.columns):
                            df_spec = df_spec[['reaction_id','specialized_filter']].rename(
                                columns={'specialized_filter':'llm_specialized_filter'}
                            )
                            w = w.merge(df_spec, on='reaction_id', how='left')
                            log(f"[{set_name}] merged specialized filter from {spec_path}")
                            break
            if 'llm_specialized_filter' in w.columns:
                t1 = len(w)
                w = w[w['llm_specialized_filter'] != 'specialized']
                log(f"[{set_name}] remove specialized: {t1} → {len(w)}")

        # polymerization filter
        if filters.get('apply_polymerization_filter', False):
            if 'polymerization_type' in w.columns:
                t2 = len(w)
                w = w[w['polymerization_type'].notna() & (w['polymerization_type'] != "")]
                log(f"[{set_name}] polymerization filter: {t2} → {len(w)}")

        # Ensure target class exists (should already exist from split creation)
        if 'r_product_class' not in w.columns:
            bins = [-np.inf, 1, 25, np.inf]
            labels = [0, 1, 2]
            w['r_product_class'] = pd.cut(w['r1r2'], bins=bins, labels=labels, right=False).astype(int)
            
            # extreme override
            if {'constant_1','constant_2'}.issubset(w.columns):
                mask_ext = (((w['constant_1'] <= 0.1) & (w['constant_2'] > 25)) |
                            ((w['constant_2'] <= 0.1) & (w['constant_1'] > 25)))
                w.loc[mask_ext, 'r_product_class'] = 2

        return w
    
    df_train = apply_filters_to_set(df_train, "Train")
    df_test = apply_filters_to_set(df_test, "Test")

    # ---------------- 3) Get features and remove NaN ----------------
    available_features = [c for c in prediction_utils.feature_columns if c in df_train.columns]
    if not available_features:
        raise RuntimeError("No features found from prediction_utils.feature_columns.")
    
    log(f"\n[Features] Using {len(available_features)} features")

    # Remove NaN in features/target for both sets
    for df_set, set_name in [(df_train, "Train"), (df_test, "Test")]:
        X_set = df_set[available_features]
        y_set = df_set['r_product_class'].astype(int)
        mask = ~(X_set.isna().any(axis=1) | y_set.isna())
        before = len(df_set)
        if set_name == "Train":
            df_train = df_set[mask].reset_index(drop=True)
            after = len(df_train)
        else:
            df_test = df_set[mask].reset_index(drop=True)
            after = len(df_test)
        if before > after:
            log(f"[{set_name}] NaN-drop: {before} → {after}")

    # ---------------- 4) Add negative data to TRAIN only ----------------
    if filters.get('add_negative_data', False):
        neg_paths = [
            "../../copol_prediction/filter/artificial_datapoints/processed_combined_augmented.csv",
            "../copol_prediction/filter/artificial_datapoints/processed_combined_augmented.csv",
        ]
        loaded = False
        for p in neg_paths:
            if os.path.exists(p):
                dn = pd.read_csv(p)
                if 'Class' in dn.columns:
                    dn = dn.rename(columns={'Class':'r_product_class'})
                    dn['r_product_class'] = dn['r_product_class'].astype(int)
                    if 'reaction_id' not in dn.columns:
                        dn['reaction_id'] = [f"neg_{i}" for i in range(len(dn))]
                    df_train = pd.concat([df_train, dn], ignore_index=True)
                    log(f"[Train] add negative: +{len(dn)} from {p}")
                    loaded = True
                    break
        if not loaded:
            log("[Train] ⚠ negative data not found")

    # Final check
    if len(df_test) == 0:
        raise RuntimeError("Test set is empty after filtering")

    log(f"\n[Final] Train={len(df_train)} samples ({df_train['reaction_id'].nunique()} groups)")
    log(f"[Final] Test={len(df_test)} samples ({df_test['reaction_id'].nunique()} groups)")
    
    return df_train, df_test, available_features



def run_single_configuration(df, filters, config):
    """
    Train and evaluate model with specific filter configuration.
    Uses global train/test split for consistency.
    
    Args:
        df: Input dataframe (not used, kept for compatibility)
        filters: Dictionary of filter settings
        config: Configuration dictionary
        
    Returns:
        Dictionary with results
    """
    print("\n" + "="*60)
    run_name = f"spec{int(filters['remove_specialized'])}_poly{int(filters.get('apply_polymerization_filter', False))}_neg{int(filters['add_negative_data'])}_aug{int(filters['use_augmentation'])}"
    print(f"Running: {run_name}")
    print(f"  Filters requested:")
    print(f"    remove_specialized: {filters['remove_specialized']}")
    print(f"    apply_polymerization_filter: {filters.get('apply_polymerization_filter', False)}")
    print(f"    add_negative_data: {filters['add_negative_data']}")
    print(f"    use_augmentation: {filters['use_augmentation']}")
    print(f"  Hyperparameter search iterations: {config['n_iter']}")
    print("="*60)
    
    # Prepare data (loads global split and applies filters)
    df_train, df_test, features = prepare_filtered_data(df, filters, config)
    
    print(f"\n  Data sizes after filtering:")
    print(f"    Training set: {len(df_train)} samples")
    print(f"    Test set: {len(df_test)} samples")
    print(f"    Number of features: {len(features)}")
    
    # Apply augmentation if configured (TRAIN ONLY)
    if filters['use_augmentation']:
        original_train_len = len(df_train)
        df_train_aug = data_augmentation.augment_with_gaussian_samples(
            df_train,
            num_samples=config['augmentation_samples'],
            std_factor=0.3,
            random_state=config['random_state']
        )
        added = len(df_train_aug) - original_train_len
        print(f"  [+] Augmentation: added {added} samples (total: {len(df_train_aug)})")
    else:
        df_train_aug = df_train
        print(f"  [-] No augmentation applied (training with {len(df_train)} samples)")
    
    # Prepare training data
    X_train = df_train_aug[features]
    y_train = df_train_aug['r_product_class'].astype(int).values
    groups = df_train_aug['reaction_id'].astype(str).values
    
    # Calculate class weights
    class_weights = model_training.calculate_class_weights(y_train)
    
    # Print class distribution
    class_counts = pd.Series(y_train).value_counts().sort_index()
    print(f"\n  Class distribution in training set:")
    for cls, count in class_counts.items():
        pct = 100 * count / len(y_train)
        print(f"    Class {cls}: {count:4d} samples ({pct:5.1f}%) | weight: {class_weights.get(cls, 1.0):.3f}")
    
    # Hyperparameter grid (matching classification.py)
    param_grid = {
        'n_estimators': [100, 300, 600, 800],
        'max_depth': [3, 5, 8],
        'learning_rate': [0.04, 0.05, 0.06, 0.07],
        'subsample': [0.8, 0.9, 0.95],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5, 0.6],
        'reg_lambda': [1, 1.5, 2, 3],
        'min_child_weight': [2, 3, 5, 7],
        'gamma': [0.5, 0.6],
    }
    
    # Train with CV
    print("  Training with cross-validation...")
    train_result = model_training.train_xgboost_with_cv(
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        param_grid=param_grid,
        n_iter=config['n_iter'],
        cv=5,
        random_state=config['random_state'],
        class_weights=class_weights,
        n_jobs=-1
    )
    
    print(f"  Best CV score: {train_result['best_score']:.4f}")
    
    # Train final model
    final_model = model_training.train_final_model(
        X_train=X_train,
        y_train=y_train,
        params=train_result['best_params'],
        class_weights=class_weights,
        random_state=config['random_state']
    )
    
    # Evaluate on test set
    if len(df_test) > 0:
        X_test = df_test[features]
        y_test = df_test['r_product_class'].astype(int).values
        
        # Print test class distribution
        test_class_counts = pd.Series(y_test).value_counts().sort_index()
        print(f"\n  Class distribution in test set:")
        for cls, count in test_class_counts.items():
            pct = 100 * count / len(y_test)
            print(f"    Class {cls}: {count:4d} samples ({pct:5.1f}%)")
        
        test_results = evaluation.evaluate_model(
            model=final_model,
            X_test=X_test,
            y_test=y_test,
            labels=[0, 1, 2]
        )
        
        # Print macro metrics
        print("\n" + "-"*60)
        print("TEST SET EVALUATION RESULTS")
        print("-"*60)
        print(f"  Accuracy:          {test_results['accuracy']:.4f}")
        print(f"  F1 (macro):        {test_results['f1_macro']:.4f}")
        print(f"  Precision (macro): {test_results['precision_macro']:.4f}")
        print(f"  Recall (macro):    {test_results['recall_macro']:.4f}")
        print(f"  F1 (weighted):     {test_results['f1_weighted']:.4f} (for comparison)")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = test_results['confusion_matrix']
        print("     Predicted")
        print("        0    1    2")
        for i, row in enumerate(cm):
            if i == 0:
                print(f"True 0 [{row[0]:4d} {row[1]:4d} {row[2]:4d}]")
            else:
                print(f"     {i} [{row[0]:4d} {row[1]:4d} {row[2]:4d}]")
        
        # Print per-class metrics from classification report
        print("\nPer-Class Metrics:")
        report = test_results['classification_report']
        print(report)
        
        print("-"*60 + "\n")
        
        # Save confusion matrix as individual plot
        cm_filename = f"confusion_matrix_{run_name}.png"
        plot_confusion_matrix(
            cm=test_results['confusion_matrix'],
            labels=[0, 1, 2],
            title=f"Confusion Matrix: {run_name}",
            save_path=os.path.join(config['plots_dir'], cm_filename)
        )
        
        # Save test results
        test_filename = f"test_{run_name}.json"
        evaluation.save_holdout_metrics_json(
            y_true=y_test,
            y_pred=test_results['predictions'],
            labels=[0, 1, 2],
            out_dir=config['output_dir'],
            filename=test_filename
        )
        
        return {
            'run_name': run_name,
            'filters': filters,
            'best_params': train_result['best_params'],
            'cv_score': train_result['best_score'],
            'holdout_accuracy': test_results['accuracy'],
            'holdout_f1_macro': test_results['f1_macro'],
            'holdout_f1_weighted': test_results['f1_weighted'],
            'holdout_precision_macro': test_results['precision_macro'],
            'holdout_recall_macro': test_results['recall_macro'],
            'confusion_matrix': test_results['confusion_matrix'].tolist(),
            'cm_plot': cm_filename,
            'n_train': len(df_train),
            'n_holdout': len(df_test)
        }
    else:
        print("  Warning: Empty test set!")
        return None


def plot_4x4_matrix(results_df, plots_dir, metric='holdout_f1_macro', metric_label='F1 Score (Macro)'):
    """
    Create 4x4 matrix heatmap showing all filter combinations.
    
    Args:
        results_df: DataFrame with results including 'filters' column
        plots_dir: Directory to save plots
        metric: Metric column name to plot
        metric_label: Label for the metric
    """
    setup_plot_style()
    
    # Extract filter values from results
    filter_data = []
    for _, row in results_df.iterrows():
        filters = row['filters']
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
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use mask for missing values
    mask = np.isnan(matrix)
    
    sns.heatmap(
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
        linecolor='gray'
    )
    
    ax.set_title(f'Filter Sweep Results: {metric_label}\n(All combinations on same holdout set)', 
                 fontsize=14, pad=20)
    ax.set_xlabel('Augmentation & Polymerization Filter', fontsize=12)
    ax.set_ylabel('Specialized Removal & Negative Data', fontsize=12)
    
    plt.tight_layout()
    
    filename = f'filter_matrix_{metric}.png'
    path = os.path.join(plots_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved 4x4 matrix to {path}")


def plot_sweep_results(results_df, plots_dir):
    """
    Create visualizations of sweep results.
    
    Args:
        results_df: DataFrame with results
        plots_dir: Directory to save plots
    """
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
    
    # Plot 3: Note about individual confusion matrices
    print(f"  Individual confusion matrices saved for each configuration in {plots_dir}/")
    
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


def main():
    """Main sweep pipeline."""
    args = parse_args()
    
    config = {
        'data_path': args.data_path,
        'output_dir': args.output_dir,
        'plots_dir': args.plots_dir,
        'random_state': args.random_state,
        'n_iter': args.n_iter,
        'augmentation_samples': args.augmentation_samples,
    }
    
    print("="*60)
    print("FILTER SWEEP - COPOLYMERIZATION PREDICTION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data path: {config['data_path']}")
    print(f"  Output dir: {config['output_dir']}")
    print(f"  Plots dir: {config['plots_dir']}")
    print(f"  Random state: {config['random_state']}")
    print(f"  Hyperparam iterations: {config['n_iter']}")
    
    # Define search space (all 4x4 = 16 combinations)
    search_space = {
        "remove_specialized": [False, True],
        "apply_polymerization_filter": [False, True],  # Now enabled for 4x4 matrix
        "add_negative_data": [False, True],
        "use_augmentation": [False, True],
    }
    
    print(f"\nSearch space:")
    for key, values in search_space.items():
        print(f"  {key}: {values}")
    
    total_combinations = np.prod([len(v) for v in search_space.values()])
    print(f"\nTotal combinations: {total_combinations}")
    
    # Check for global train/test split
    print("\n" + "="*60)
    print("CHECKING GLOBAL TRAIN/TEST SPLIT")
    print("="*60)
    
    train_path = os.path.join(os.path.dirname(__file__), '../data/train.csv')
    test_path = os.path.join(os.path.dirname(__file__), '../data/test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("\nError: Global train/test split not found!")
        print(f"Expected files:")
        print(f"  - {train_path}")
        print(f"  - {test_path}")
        print("\nPlease run the following commands to create the split:")
        print("  cd ../copol_prediction")
        print("  python create_data_split.py")
        print("  cd ../experiments")
        print("  python archive/create_train_test_split.py")
        sys.exit(1)
    
    # Verify split files
    df_train_check = pd.read_csv(train_path)
    df_test_check = pd.read_csv(test_path)
    print(f"\n✓ Found global split:")
    print(f"  Train: {len(df_train_check)} samples ({df_train_check['reaction_id'].nunique()} groups)")
    print(f"  Test: {len(df_test_check)} samples ({df_test_check['reaction_id'].nunique()} groups)")
    
    # Create output directories
    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['plots_dir'], exist_ok=True)
    
    # Run sweep
    print("\n" + "="*60)
    print("RUNNING SWEEP")
    print("="*60)
    
    results = []
    for i, filters in enumerate(generate_filter_combinations(search_space), 1):
        print(f"\n[{i}/{total_combinations}] Configuration: {filters}")
        
        try:
            # Note: df=None is passed but not used (data loaded from global split)
            result = run_single_configuration(df=None, filters=filters, config=config)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    if not results:
        print("No successful runs!")
        sys.exit(1)
    
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = os.path.join(config['output_dir'], 'sweep_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")
    
    # Save to JSON
    json_path = os.path.join(config['output_dir'], 'sweep_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {json_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    results_sorted = results_df.sort_values('holdout_f1_macro', ascending=False)
    
    print("\nTop 5 configurations by F1 (macro) score:")
    print(results_sorted[['run_name', 'holdout_accuracy', 'holdout_f1_macro', 'holdout_precision_macro', 'holdout_recall_macro']].head())
    
    best_config = results_sorted.iloc[0]
    print(f"\n🏆 Best configuration: {best_config['run_name']}")
    print(f"   Accuracy: {best_config['holdout_accuracy']:.4f}")
    print(f"   F1 (macro): {best_config['holdout_f1_macro']:.4f}")
    print(f"   Precision (macro): {best_config['holdout_precision_macro']:.4f}")
    print(f"   Recall (macro): {best_config['holdout_recall_macro']:.4f}")
    print(f"   Filters: {best_config['filters']}")
    
    # Create plots
    print("\n" + "="*60)
    print("CREATING PLOTS")
    print("="*60)
    
    plot_sweep_results(results_df, config['plots_dir'])
    
    print("\n" + "="*60)
    print("SWEEP COMPLETE!")
    print("="*60)
    print(f"\nResults: {csv_path}")
    print(f"Plots: {config['plots_dir']}/")
    print(f"\nTo train final model with best config, use:")
    print(f"  python train_final_model.py")


if __name__ == "__main__":
    main()

