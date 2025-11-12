#!/usr/bin/env python3
"""
Final model training script for copolymerization prediction.

This script trains the production model with optimized settings and saves
it as a bundle for deployment.

Usage:
    python train_final_model.py [--data-path PATH] [--output-dir DIR]
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train final copolymerization prediction model"
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
        default="artifacts/model_bundle",
        help="Directory to save model bundle"
    )
    parser.add_argument(
        "--holdout-path",
        type=str,
        default="artifacts/test_ids.csv",
        help="Path to persistent holdout group IDs"
    )
    parser.add_argument(
        "--process-data",
        action="store_true",
        help="Process data from scratch (otherwise load processed_data.csv)"
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
    
    return parser.parse_args()


def prepare_data(df, config):
    """
    Prepare data for training.
    
    Args:
        df: Input dataframe
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_df, holdout_df, base_df)
    """
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    # Create base dataset for holdout splitting
    base_df = holdout_utils.make_base_dataset_for_holdout(df)
    
    # Apply filters
    df_filtered = df.copy()
    df_filtered = df_filtered[df_filtered['r1r2'].notna()]
    df_filtered = df_filtered[df_filtered['r1r2'] >= 0]
    
    # Remove specialized if configured
    if config['remove_specialized'] and 'llm_specialized_filter' in df_filtered.columns:
        original_len = len(df_filtered)
        df_filtered = df_filtered[df_filtered['llm_specialized_filter'] != 'specialized']
        print(f"Removed {original_len - len(df_filtered)} specialized samples")
    
    # Create target classes
    bins = [-np.inf, 1, 25, np.inf]
    labels = [0, 1, 2]
    df_filtered['r_product_class'] = pd.cut(
        df_filtered['r1r2'], bins=bins, labels=labels, right=False
    ).astype(int)
    
    # Override extremes
    if {'constant_1', 'constant_2'}.issubset(df_filtered.columns):
        extreme_mask = (
            ((df_filtered['constant_1'] <= 0.1) & (df_filtered['constant_2'] > 25)) |
            ((df_filtered['constant_2'] <= 0.1) & (df_filtered['constant_1'] > 25))
        )
        df_filtered.loc[extreme_mask, 'r_product_class'] = 2
        print(f"Marked {extreme_mask.sum()} extreme cases as class 2")
    
    # Get available features
    available_features = [c for c in prediction_utils.feature_columns if c in df_filtered.columns]
    print(f"Using {len(available_features)} features")
    
    # Remove NaN rows
    X_all = df_filtered[available_features]
    y_all = df_filtered['r_product_class'].astype(int)
    mask = ~(pd.isna(X_all).any(axis=1) | pd.isna(y_all))
    df_clean = df_filtered[mask].reset_index(drop=True)
    print(f"Clean dataset: {len(df_clean)} samples")

    # Add negative data if configured
    if config['add_negative_data']:
        neg_path = "filter/artificial_datapoints/processed_combined_augmented.csv"
        if os.path.exists(neg_path):
            df_neg = pd.read_csv(neg_path)
            if 'Class' in df_neg.columns:
                df_neg = df_neg.rename(columns={'Class': 'r_product_class'})
                df_neg['r_product_class'] = df_neg['r_product_class'].astype(int)
                df_filtered = pd.concat([df_clean, df_neg], ignore_index=True)
                print(f"Added {len(df_neg)} negative data points. Dataset has now {len(df_filtered)}")
        else:
            print(f"Warning: Negative data file not found at {neg_path}")
    
    # Split into train and holdout
    holdout_groups = holdout_utils.get_or_create_holdout_groups(
        base_df, 
        group_col='reaction_id',
        test_groups_path=config['holdout_path']
    )
    
    df_train, df_holdout = holdout_utils.split_train_holdout(
        df_filtered, holdout_groups, group_col='reaction_id'
    )
    
    print(f"Training set: {len(df_train)} samples ({df_train['reaction_id'].nunique()} groups)")
    print(f"Holdout set: {len(df_holdout)} samples ({df_holdout['reaction_id'].nunique()} groups)")
    
    return df_train, df_holdout, available_features


def train_model(df_train, features, config):
    """
    Train the model with hyperparameter optimization.
    
    Args:
        df_train: Training dataframe
        features: List of feature names
        config: Configuration dictionary
        
    Returns:
        Dictionary with trained model and training info
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
    
    # Define hyperparameter search space
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
    """
    Evaluate model on holdout set.
    
    Args:
        model: Trained model
        df_holdout: Holdout dataframe
        features: List of feature names
        
    Returns:
        Evaluation results dictionary
    """
    print("\n" + "="*60)
    print("HOLDOUT EVALUATION")
    print("="*60)
    
    X_holdout = df_holdout[features]
    y_holdout = df_holdout['r_product_class'].astype(int).values
    
    results = evaluation.evaluate_model(model, X_holdout, y_holdout, labels=[0, 1, 2])
    evaluation.print_evaluation_results(results, title="Holdout Set Performance")
    
    return results


def save_model(model_info, holdout_results, config):
    """
    Save model bundle and results.
    
    Args:
        model_info: Dictionary with model and training info
        holdout_results: Holdout evaluation results
        config: Configuration dictionary
    """
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
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
            'negative_data_used': config['add_negative_data'],
            'specialized_removed': config['remove_specialized'],
            'random_state': config['random_state']
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
    
    # Save holdout metrics
    evaluation.save_holdout_metrics_json(
        y_true=holdout_results['predictions'],  # This should be y_true
        y_pred=holdout_results['predictions'],
        labels=[0, 1, 2],
        out_dir=os.path.join(config['output_dir'], "holdout_results"),
        filename="final_model_holdout.json"
    )
    
    print(f"\n✓ Model bundle saved to: {bundle_path}")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Configuration
    config = {
        'data_path': args.data_path,
        'output_dir': args.output_dir,
        'holdout_path': args.holdout_path,
        'process_data': args.process_data,
        'random_state': args.random_state,
        'augmentation_samples': args.augmentation_samples,
        'hyperparam_iter': args.hyperparam_iter,
        # Training settings (can be adjusted)
        'remove_specialized': False,
        'add_negative_data': True,
        'use_augmentation': False,
    }
    
    print("="*60)
    print("COPOLYMERIZATION PREDICTION - FINAL MODEL TRAINING")
    print("="*60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Load data
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    if config['process_data']:
        print("Processing data from scratch...")
        df = data_processing.load_and_preprocess_data(config['data_path'])
        if df is None or len(df) == 0:
            print("Error: No data available after preprocessing")
            sys.exit(1)
    else:
        processed_path = "output/processed_data.csv"
        if not os.path.exists(processed_path):
            print(f"Processed data not found at {processed_path}")
            print("Run with --process-data flag to process from scratch")
            sys.exit(1)
        df = pd.read_csv(processed_path)
        print(f"Loaded processed data: {len(df)} samples")
    
    # Prepare data
    df_train, df_holdout, features = prepare_data(df, config)
    
    # Train model
    model_info = train_model(df_train, features, config)
    
    # Evaluate on holdout
    holdout_results = evaluate_on_holdout(model_info['model'], df_holdout, features)
    
    # Save model
    save_model(model_info, holdout_results, config)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {config['output_dir']}")
    print("\nTo use the model:")
    print("  from copolpredictor.inference import CopolymerPredictor")
    print(f"  predictor = CopolymerPredictor('{config['output_dir']}')")
    print("  predictions = predictor.predict(X)")


if __name__ == "__main__":
    main()

