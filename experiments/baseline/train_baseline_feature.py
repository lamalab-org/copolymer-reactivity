#!/usr/bin/env python3
"""
Train model using baseline (database lookup) predictions as input feature.

This script:
1. Computes baseline predictions for train/test sets using Tanimoto similarity
2. Uses these baseline predictions as the only feature for model training
3. Trains with same hyperparameters as final model
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from copolpredictor import (
    data_processing,
    data_augmentation,
    model_training,
    evaluation,
    holdout_utils,
    prediction_utils
)
from copol_prediction.utils import load_data_split
from copol_prediction.analysis.analyze_model import (
    compute_naive_baseline_predictions_with_similarity,
    compute_fingerprints_for_smiles,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train model with baseline predictions as feature"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save model bundle"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--augmentation-samples",
        type=int,
        default=5,
        help="Number of augmented samples per datapoint"
    )
    parser.add_argument(
        "--hyperparam-iter",
        type=int,
        default=25,
        help="Number of hyperparameter search iterations"
    )
    parser.add_argument(
        "--final-model-path",
        type=str,
        default="../../copol_prediction/artifacts/model_bundle",
        help="Path to final model to inherit training configuration from"
    )
    parser.add_argument(
        "--remove-specialized",
        action="store_true",
        help="Remove specialized datapoints from training (overrides final model config)"
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable augmentation (overrides final model config)"
    )
    
    return parser.parse_args()


def load_final_model_config(final_model_path):
    """
    Load training configuration from final model metadata.
    
    Args:
        final_model_path: Path to final model bundle
        
    Returns:
        Dictionary with training configuration
    """
    try:
        from copolpredictor.inference import CopolymerPredictor
        predictor = CopolymerPredictor(final_model_path)
        training_config = predictor.metadata.get('training_config', {})
        
        config = {
            'remove_specialized': training_config.get('specialized_removed_from_training', False),
            'use_augmentation': training_config.get('augmentation_used', False),
            'augmentation_samples': training_config.get('augmentation_samples', 5),
            'add_negative_data': training_config.get('negative_data_used', False),
        }
        
        print(f"  ✓ Loaded configuration from final model:")
        print(f"    - Specialized removed: {config['remove_specialized']}")
        print(f"    - Augmentation used: {config['use_augmentation']}")
        print(f"    - Augmentation samples: {config['augmentation_samples']}")
        print(f"    - Negative data used: {config['add_negative_data']}")
        
        return config
    except Exception as e:
        print(f"  ⚠️  Warning: Could not load final model config: {e}")
        print(f"     Using defaults: specialized filter OFF, augmentation OFF")
        return {
            'remove_specialized': False,
            'use_augmentation': False,
            'augmentation_samples': 5,
            'add_negative_data': False,
        }


def filter_training_data(df_train, remove_specialized=False):
    """
    Filter training data by removing specialized datapoints if needed.
    
    Args:
        df_train: Training dataframe
        remove_specialized: Whether to remove specialized datapoints
        
    Returns:
        Filtered training dataframe
    """
    if not remove_specialized:
        return df_train
    
    df_filtered = df_train.copy()
    
    # Load specialized filter classifications
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    spec_path = os.path.join(
        project_root,
        "copol_prediction/filter/llm_specialized_filter/classified_output.csv"
    )
    
    if os.path.exists(spec_path):
        df_spec = pd.read_csv(spec_path)
        if {'specialized_filter', 'reaction_id'}.issubset(df_spec.columns):
            df_spec = df_spec[['reaction_id', 'specialized_filter']].rename(
                columns={'specialized_filter': 'llm_specialized_filter'}
            )
            df_filtered = df_filtered.merge(df_spec, on='reaction_id', how='left')
            
            if 'llm_specialized_filter' in df_filtered.columns:
                before_count = len(df_filtered)
                df_filtered = df_filtered[df_filtered['llm_specialized_filter'] != 'specialized']
                removed_count = before_count - len(df_filtered)
                if removed_count > 0:
                    print(f"  Removed {removed_count} specialized datapoints from training set")
    
    return df_filtered.reset_index(drop=True)


def prepare_data_with_baseline_feature(config):
    """
    Load pre-split data and compute baseline predictions as feature.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (df_train, df_test, baseline_feature_name)
    """
    print("\n" + "="*60)
    print("DATA PREPARATION WITH BASELINE FEATURE")
    print("="*60)
    
    # Load central train/test split
    print("Loading central train/test split...")
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        split_dir = os.path.join(project_root, "copol_prediction", "artifacts", "data_splits")
        
        df_train, df_test = load_data_split.load_train_test_split(split_dir=split_dir)
        load_data_split.print_split_info(split_dir=split_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease create the central split first:")
        print("  cd copol_prediction && python create_data_split.py")
        sys.exit(1)
    
    # Apply specialized filter to training data if configured
    if config.get('remove_specialized', False):
        print("\nApplying specialized filter to training data...")
        df_train = filter_training_data(df_train, remove_specialized=True)
        print(f"  Training set after filtering: {len(df_train)} samples")
    
    # Check required columns for baseline computation
    required_cols = ['monomer1_smiles', 'monomer2_smiles', 'solvent_smiles']
    for col in required_cols:
        if col not in df_train.columns or col not in df_test.columns:
            raise ValueError(f"Required column '{col}' not found in dataframes")
    
    # Base model features (same list as final model uses)
    base_features = [c for c in prediction_utils.feature_columns if c in df_train.columns]
    print(f"\nUsing {len(base_features)} base model features (same feature set as final model)")

    # Feature columns for tie-breaking in baseline lookup
    feature_cols = base_features
    
    # Build fingerprint cache ONCE (unique SMILES across train+test)
    print("\nBuilding fingerprint cache (unique SMILES across train+test)...")
    unique_smiles = set()
    for col in ['monomer1_smiles', 'monomer2_smiles', 'solvent_smiles']:
        unique_smiles |= set(df_train[col].dropna().unique())
        unique_smiles |= set(df_test[col].dropna().unique())
    fp_dict = compute_fingerprints_for_smiles(list(unique_smiles))
    print(f"  Fingerprint cache size: {len(fp_dict)} (valid: {len([v for v in fp_dict.values() if v is not None])})")

    print(f"\nComputing baseline predictions for training set...")
    y_train = df_train['r_product_class'].astype(int).values
    
    # For training: use leave-one-out approach (for each training point, 
    # find nearest neighbor from other training points)
    # Note: Use filtered training data for lookup pool
    print("  Computing baseline for training set (leave-one-out)...")
    baseline_train = []
    baseline_train_sim = []
    for idx, row in df_train.iterrows():
        # Create temporary train set without current point
        df_train_temp = df_train.drop(index=idx).reset_index(drop=True)
        y_train_temp = df_train_temp['r_product_class'].astype(int).values
        
        # Find baseline for this single point
        df_test_single = df_train.loc[[idx]].reset_index(drop=True)
        baseline_pred, baseline_sim = compute_naive_baseline_predictions_with_similarity(
            df_test_single, df_train_temp, y_train_temp, feature_cols, fp_dict=fp_dict
        )
        baseline_train.append(baseline_pred[0])
        baseline_train_sim.append(baseline_sim[0])
    
    baseline_train = np.array(baseline_train)
    baseline_train_sim = np.array(baseline_train_sim)
    
    print(f"\nComputing baseline predictions for test set...")
    # Use filtered training data for lookup pool
    baseline_test, baseline_test_sim = compute_naive_baseline_predictions_with_similarity(
        df_test, df_train, y_train, feature_cols, fp_dict=fp_dict
    )
    
    # Add baseline predictions as feature (one-hot encoded)
    print("\nAdding baseline predictions as features...")
    for df, baseline_pred, baseline_sim, name in [
        (df_train, baseline_train, baseline_train_sim, 'train'),
        (df_test, baseline_test, baseline_test_sim, 'test'),
    ]:
        # One-hot encode baseline predictions
        df['baseline_class_0'] = (baseline_pred == 0).astype(int)
        df['baseline_class_1'] = (baseline_pred == 1).astype(int)
        df['baseline_class_2'] = (baseline_pred == 2).astype(int)
        # Distance feature: 1 - similarity (clip to [0,1])
        df['baseline_distance'] = np.clip(1.0 - baseline_sim, 0.0, 1.0)
        print(f"  {name}: Added 3 baseline features")
    
    print(f"\nFinal dataset:")
    print(f"  Train: {len(df_train)} samples ({df_train['reaction_id'].nunique()} groups)")
    print(f"  Test:  {len(df_test)} samples ({df_test['reaction_id'].nunique()} groups)")
    
    # Final feature set: base model features + baseline features
    baseline_features = ['baseline_class_0', 'baseline_class_1', 'baseline_class_2', 'baseline_distance']
    final_features = base_features + baseline_features
    print(f"\nFinal feature set size: {len(final_features)} (base + 3 baseline one-hots)")
    
    return df_train, df_test, final_features


def train_model(df_train, features, config):
    """
    Train the model with hyperparameter optimization.
    Same as train_final_model.py but using base features + baseline one-hot features.
    """
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    # Augment training data if configured
    if config['use_augmentation']:
        df_train_aug = data_augmentation.augment_with_gaussian_samples(
            df_train,
            num_samples=config['augmentation_samples'],
            std_factor=0.3,
            random_state=config['random_state']
        )
        print(f"Augmented training set: {len(df_train_aug)} samples")
    else:
        df_train_aug = df_train
    
    # Prepare training data
    X_train = df_train_aug[features]
    y_train = df_train_aug['r_product_class'].astype(int).values
    groups = df_train_aug['reaction_id'].astype(str).values
    
    # Calculate class weights
    class_weights = model_training.calculate_class_weights(y_train)
    print("\nClass weights:")
    for cls, weight in class_weights.items():
        print(f"  Class {cls}: {weight:.4f}")
    
    # Define hyperparameter search space (same as final model)
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
    
    # Train with CV and hyperparameter search
    print("\nStarting hyperparameter search...")
    train_results = model_training.train_xgboost_with_cv(
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        param_grid=param_grid,
        n_iter=config['hyperparam_iter'],
        cv=5,
        random_state=config['random_state'],
        class_weights=class_weights,
        n_jobs=-1
    )
    
    print("\nBest hyperparameters:")
    for param, value in train_results['best_params'].items():
        print(f"  {param}: {value}")
    print(f"\nBest CV score (F1 weighted): {train_results['best_score']:.4f}")
    
    # Train final model on full training set
    print("\nTraining final model on full training set...")
    final_model = model_training.train_final_model(
        X_train=X_train,
        y_train=y_train,
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


def evaluate_on_holdout(model, df_holdout, features):
    """Evaluate model on holdout set."""
    print("\n" + "="*60)
    print("HOLDOUT EVALUATION")
    print("="*60)
    
    X_holdout = df_holdout[features]
    y_holdout = df_holdout['r_product_class'].astype(int).values
    
    results = evaluation.evaluate_model(model, X_holdout, y_holdout, labels=[0, 1, 2])
    evaluation.print_evaluation_results(results, title="Holdout Set Performance")
    
    return results


def save_model(model_info, holdout_results, config):
    """Save model bundle and results."""
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    # Get split info
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    split_dir = os.path.join(project_root, "copol_prediction", "artifacts", "data_splits")
    split_info = load_data_split.get_split_info(split_dir=split_dir)
    specialized_removed = split_info.get('remove_specialized_from_test', False) if split_info else False
    
    # Prepare metadata
    metadata = {
        'best_params': model_info['best_params'],
        'cv_score': float(model_info['cv_score']),
        'holdout_accuracy': float(holdout_results['accuracy']),
        'holdout_f1_weighted': float(holdout_results['f1_weighted']),
        'holdout_f1_macro': float(holdout_results['f1_macro']),
        'class_weights': {int(k): float(v) for k, v in model_info['class_weights'].items()},
        'training_config': {
            'augmentation_used': config['use_augmentation'],
            'augmentation_samples': config['augmentation_samples'],
            'negative_data_used': config.get('add_negative_data', False),
            'specialized_removed_from_training': config.get('remove_specialized', False),
            'specialized_removed_from_test': specialized_removed,  # Always False (test/val never filtered)
            'used_central_split': True,
            'random_state': config['random_state'],
            'feature_type': 'baseline_predictions_only'
        }
    }
    
    # Save model bundle
    bundle_path = model_training.save_model_bundle(
        model=model_info['model'],
        feature_list=model_info['features'],
        class_labels=[0, 1, 2],
        out_dir=config['output_dir'],
        metadata=metadata
    )
    
    print(f"\n✓ Model bundle saved to: {bundle_path}")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Load configuration from final model (if available)
    print("="*60)
    print("BASELINE FEATURE MODEL TRAINING")
    print("="*60)
    print("\nLoading training configuration from final model...")
    final_config = load_final_model_config(args.final_model_path)
    
    # Override with command-line arguments if provided
    if args.remove_specialized:
        final_config['remove_specialized'] = True
        print(f"  → Override: specialized filter ENABLED (from command line)")
    if args.no_augmentation:
        final_config['use_augmentation'] = False
        print(f"  → Override: augmentation DISABLED (from command line)")
    
    # Configuration
    config = {
        'output_dir': args.output_dir,
        'random_state': args.random_state,
        'augmentation_samples': args.augmentation_samples if not args.no_augmentation else final_config['augmentation_samples'],
        'hyperparam_iter': args.hyperparam_iter,
        'use_augmentation': final_config['use_augmentation'],
        'add_negative_data': final_config['add_negative_data'],
        'remove_specialized': final_config['remove_specialized'],
    }
    
    print("\nFinal configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nℹ️  Note: Using baseline predictions (database lookup) as feature")
    print("   Using central train/test split from copol_prediction/artifacts/data_splits/")
    
    # Prepare data with baseline feature
    df_train, df_test, features = prepare_data_with_baseline_feature(config)
    
    # Train model
    model_info = train_model(df_train, features, config)
    
    # Evaluate on holdout
    holdout_results = evaluate_on_holdout(model_info['model'], df_test, features)
    
    # Save model
    save_model(model_info, holdout_results, config)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {config['output_dir']}")


if __name__ == "__main__":
    main()
