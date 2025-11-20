#!/usr/bin/env python3
"""
Permutation Feature Importance Analysis Experiment.

This script:
1. Loads the global train/test split
2. Trains a model using feature_columns_2
3. Performs permutation importance analysis
4. Visualizes results as bar plots
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from copolpredictor import (
    model_training,
    evaluation,
    prediction_utils
)

# Import permutation analysis from copol_prediction
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../copol_prediction/analysis'))
from permutation_analysis import (
    calculate_permutation_importance
)


def parse_args():
    parser = argparse.ArgumentParser(description="Permutation Feature Importance Analysis")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hyperparam-iter", type=int, default=25)
    parser.add_argument("--n-repeats", type=int, default=10, help="Number of permutation repeats")
    parser.add_argument("--scoring", type=str, default="f1_macro", help="Scoring metric for permutation importance")
    parser.add_argument("--top-n", type=int, default=30, help="Number of top features to plot")
    return parser.parse_args()


def load_presplit_data():
    """Load pre-split train/test data."""
    print("\n" + "="*60)
    print("LOADING PRE-SPLIT DATA")
    print("="*60)
    
    train_path = os.path.join(os.path.dirname(__file__), '../data/train.csv')
    test_path = os.path.join(os.path.dirname(__file__), '../data/test.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Pre-split data not found!")
        print(f"Expected files:")
        print(f"  - {train_path}")
        print(f"  - {test_path}")
        print(f"\nRun: python archive/create_train_test_split.py")
        sys.exit(1)
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"Loaded train: {len(df_train)} samples ({df_train['reaction_id'].nunique()} groups)")
    print(f"Loaded test: {len(df_test)} samples ({df_test['reaction_id'].nunique()} groups)")
    
    # Get available features from feature_columns_2
    available_features = [c for c in prediction_utils.feature_columns_2 if c in df_train.columns]
    print(f"\nUsing {len(available_features)} features from feature_columns_2")
    print(f"Total features in feature_columns_2: {len(prediction_utils.feature_columns_2)}")
    
    missing_features = set(prediction_utils.feature_columns_2) - set(available_features)
    if missing_features:
        print(f"\n⚠️  Missing {len(missing_features)} features:")
        for feat in sorted(missing_features):
            print(f"    - {feat}")
    
    # Remove NaN in features/target
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
            print(f"[{set_name}] Removed {before - after} rows with NaN")
    
    print(f"\nFinal sizes:")
    print(f"  Train: {len(df_train)} samples")
    print(f"  Test: {len(df_test)} samples")
    
    return df_train, df_test, available_features


def train_model(df_train, features, config):
    """Train XGBoost model with cross-validation."""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    X_train = df_train[features]
    y_train = df_train['r_product_class'].astype(int).values
    groups = df_train['reaction_id'].astype(str).values
    
    class_weights = model_training.calculate_class_weights(y_train)
    
    # Print class distribution
    class_counts = pd.Series(y_train).value_counts().sort_index()
    print(f"\nClass distribution in training set:")
    for cls, count in class_counts.items():
        pct = 100 * count / len(y_train)
        print(f"  Class {cls}: {count:4d} samples ({pct:5.1f}%) | weight: {class_weights.get(cls, 1.0):.3f}")
    
    param_grid = {
        'n_estimators': [500, 600, 700],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.04, 0.05, 0.06],
        'subsample': [0.85, 0.9, 0.95],
        'colsample_bytree': [0.85, 0.9, 1.0],
        'reg_alpha': [0.0, 0.1, 0.3],
        'reg_lambda': [1.0, 1.5, 2.0],
        'min_child_weight': [2, 3, 5],
        'gamma': [0.3, 0.5, 0.7],
    }
    
    print(f"\nTraining with {config['hyperparam_iter']} hyperparameter search iterations...")
    train_results = model_training.train_xgboost_with_cv(
        X_train=X_train, y_train=y_train, groups=groups,
        param_grid=param_grid, n_iter=config['hyperparam_iter'],
        cv=5, random_state=config['random_state'],
        class_weights=class_weights, n_jobs=-1
    )
    
    print(f"\nBest CV score: {train_results['best_score']:.4f}")
    print(f"Best parameters: {train_results['best_params']}")
    
    final_model = model_training.train_final_model(
        X_train=X_train, y_train=y_train,
        params=train_results['best_params'],
        class_weights=class_weights,
        random_state=config['random_state']
    )
    
    return {
        'model': final_model,
        'best_params': train_results['best_params'],
        'cv_score': train_results['best_score'],
        'class_weights': class_weights,
        'features': features
    }


def format_feature_name(name):
    """Format feature name for display (from copol_prediction/analysis/analyze_model.py)."""
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


def evaluate_model(model_info, df_test):
    """Evaluate model on test set."""
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    X_test = df_test[model_info['features']]
    y_test = df_test['r_product_class'].astype(int).values
    
    # Print test class distribution
    test_class_counts = pd.Series(y_test).value_counts().sort_index()
    print(f"\nClass distribution in test set:")
    for cls, count in test_class_counts.items():
        pct = 100 * count / len(y_test)
        print(f"  Class {cls}: {count:4d} samples ({pct:5.1f}%)")
    
    results = evaluation.evaluate_model(
        model_info['model'], X_test, y_test, labels=[0, 1, 2]
    )
    
    evaluation.print_evaluation_results(results, title="Test Set Performance")
    
    return results


def plot_feature_importance_barplot(results_df, output_dir, top_n=30):
    """
    Create a bar plot of permutation feature importance.
    Styled like plots from analyze_model.py (no box, no grid, no title).
    
    Args:
        results_df: DataFrame with permutation importance results
        output_dir: Directory to save plots
        top_n: Number of top features to plot
    """
    # Get top N features
    top_features = results_df.head(top_n).copy()
    
    # Format feature names for display
    formatted_names = [format_feature_name(name) for name in top_features['feature']]
    
    # Use TWO_COL width, dynamic height based on number of features (like analyze_model.py)
    TWO_COL_WIDTH_INCH = 7
    height = max(4, top_n * 0.2)
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, height))
    
    # Create horizontal bar plot with error bars
    y_pos = np.arange(len(top_features))
    colors = plt.cm.RdBu(np.linspace(0, 1, len(top_features)))
    
    ax.barh(y_pos, top_features['importance_mean'],
            xerr=top_features['importance_std'],
            capsize=4, alpha=0.8, color=colors)
    
    # Customize plot with formatted feature names
    ax.set_yticks(y_pos)
    ax.set_yticklabels(formatted_names, fontsize=7)
    ax.set_xlabel('Permutation Importance (Decrease in Score)', fontsize=9)
    ax.tick_params(axis='x', labelsize=7)
    ax.invert_yaxis()  # Highest importance at top
    
    # Remove grid and box (like analyze_model.py)
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'permutation_importance_barplot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')

    plot_path = os.path.join(output_dir, 'permutation_importance_barplot.pdf')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Saved bar plot to: {plot_path}")
    
    return plot_path


def run_permutation_analysis_experiment(model_info, df_test, config):
    """Run permutation importance analysis."""
    print("\n" + "="*60)
    print("PERMUTATION IMPORTANCE ANALYSIS")
    print("="*60)
    
    X_test = df_test[model_info['features']]
    y_test = df_test['r_product_class'].astype(int).values
    
    # Map scoring string to sklearn scoring
    # Note: sklearn's permutation_importance expects callable or string
    scoring_map = {
        'f1_macro': 'f1_macro',
        'f1': 'f1_macro',
        'accuracy': 'accuracy',
        'roc_auc': 'roc_auc_ovr'  # For multi-class
    }
    sklearn_scoring = scoring_map.get(config['scoring'], 'f1_macro')
    
    print(f"\nCalculating permutation importance with {config['scoring']} metric...")
    print(f"Number of repeats: {config['n_repeats']}")
    print(f"Using sklearn scoring: {sklearn_scoring}")
    
    # Calculate permutation importance
    results_df, perm_importance = calculate_permutation_importance(
        model=model_info['model'],
        X_test=X_test.values,  # Convert DataFrame to numpy array
        y_test=y_test,
        feature_names=model_info['features'],
        scoring=sklearn_scoring,
        n_repeats=config['n_repeats'],
        random_state=config['random_state']
    )
    
    # Save detailed results
    results_path = os.path.join(config['output_dir'], 'permutation_importance_detailed.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Saved detailed results to: {results_path}")
    
    # Create bar plot
    plot_path = plot_feature_importance_barplot(
        results_df, config['output_dir'], top_n=config['top_n']
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("PERMUTATION IMPORTANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total features analyzed: {len(model_info['features'])}")
    print(f"\nTop 10 most important features:")
    for i, (idx, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:<45} {row['importance_mean']:.6f} ± {row['importance_std']:.6f}")
    
    print(f"\nBottom 5 least important features:")
    for i, (idx, row) in enumerate(results_df.tail(5).iterrows(), 1):
        print(f"  {i:2d}. {row['feature']:<45} {row['importance_mean']:.6f} ± {row['importance_std']:.6f}")
    
    return {
        'permutation_results': results_df,
        'raw_importance': perm_importance,
        'plot_path': plot_path
    }


def save_metadata(model_info, test_results, perm_results, config):
    """Save experiment metadata."""
    metadata = {
        'experiment': 'permutation_importance',
        'timestamp': datetime.now().isoformat(),
        'features_used': 'feature_columns_2',
        'num_features': len(model_info['features']),
        'cv_score': float(model_info['cv_score']),
        'test_accuracy': float(test_results['accuracy']),
        'test_f1_macro': float(test_results['f1_macro']),
        'test_f1_weighted': float(test_results['f1_weighted']),
        'permutation_scoring': config['scoring'],
        'permutation_n_repeats': config['n_repeats'],
        'best_params': model_info['best_params'],
        'top_10_features': perm_results['permutation_results'].head(10)['feature'].tolist()
    }
    
    metadata_path = os.path.join(config['output_dir'], 'meta.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved metadata to: {metadata_path}")


def main():
    args = parse_args()
    
    config = {
        'output_dir': args.output_dir,
        'random_state': args.random_state,
        'hyperparam_iter': args.hyperparam_iter,
        'n_repeats': args.n_repeats,
        'scoring': args.scoring,
        'top_n': args.top_n,
    }
    
    print("="*60)
    print("PERMUTATION FEATURE IMPORTANCE EXPERIMENT")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Output directory: {config['output_dir']}")
    print(f"  Random state: {config['random_state']}")
    print(f"  Hyperparameter iterations: {config['hyperparam_iter']}")
    print(f"  Permutation repeats: {config['n_repeats']}")
    print(f"  Scoring metric: {config['scoring']}")
    print(f"  Top N features to plot: {config['top_n']}")
    print(f"  Features: feature_columns_2")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Load data
    df_train, df_test, features = load_presplit_data()
    
    # Train model
    model_info = train_model(df_train, features, config)
    
    # Evaluate model
    test_results = evaluate_model(model_info, df_test)
    
    # Run permutation analysis
    perm_results = run_permutation_analysis_experiment(model_info, df_test, config)
    
    # Save metadata
    save_metadata(model_info, test_results, perm_results, config)
    
    # Save model
    model_training.save_model_bundle(
        model=model_info['model'],
        feature_list=model_info['features'],
        class_labels=[0, 1, 2],
        out_dir=config['output_dir'],
        metadata={
            'experiment': 'permutation_importance',
            'cv_score': model_info['cv_score'],
            'test_accuracy': test_results['accuracy'],
            'test_f1_macro': test_results['f1_macro']
        }
    )
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {config['output_dir']}/")
    print(f"  - permutation_importance_detailed.csv")
    print(f"  - permutation_importance_barplot.png")
    print(f"  - meta.json")
    print(f"  - model.joblib")


if __name__ == "__main__":
    main()

