#!/usr/bin/env python3
"""
Train full model with all features (including reaction conditions).
This is the baseline model for comparison.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
# Add copol_prediction to path for utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../copol_prediction'))

from copolpredictor import (
    model_training,
    evaluation,
    prediction_utils
)
from utils import load_data_split


def parse_args():
    parser = argparse.ArgumentParser(description="Train full model with all features")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hyperparam-iter", type=int, default=25)
    return parser.parse_args()


def load_presplit_data(config):
    """Load pre-split train/test data using global split."""
    print("\n" + "="*60)
    print("LOADING PRE-SPLIT DATA")
    print("="*60)
    
    # Load central train/test split
    print("Loading central train/test split...")
    try:
        # Change to copol_prediction directory to use relative paths
        script_dir = os.path.dirname(__file__)
        copol_pred_dir = os.path.join(script_dir, '../../../copol_prediction')
        copol_pred_dir = os.path.abspath(copol_pred_dir)
        
        original_cwd = os.getcwd()
        os.chdir(copol_pred_dir)
        
        try:
            df_train, df_test = load_data_split.load_train_test_split()
            load_data_split.print_split_info()
        finally:
            os.chdir(original_cwd)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease create the central split first:")
        print("  cd copol_prediction && python create_data_split.py")
        sys.exit(1)
    
    # Get available features (all features)
    available_features = [c for c in prediction_utils.feature_columns if c in df_train.columns]
    print(f"\nUsing {len(available_features)} features (full model with reaction conditions)")
    print(f"  Train: {len(df_train)} samples ({df_train['reaction_id'].nunique()} groups)")
    print(f"  Test:  {len(df_test)} samples ({df_test['reaction_id'].nunique()} groups)")
    
    # Add negative data if configured
    if config['add_negative_data']:
        neg_path = "../../../copol_prediction/filter/artificial_datapoints/processed_combined_augmented.csv"
        if os.path.exists(neg_path):
            df_neg = pd.read_csv(neg_path)
            if 'Class' in df_neg.columns:
                df_neg = df_neg.rename(columns={'Class': 'r_product_class'})
                df_neg['r_product_class'] = df_neg['r_product_class'].astype(int)
            df_train = pd.concat([df_train, df_neg], ignore_index=True)
            print(f"Added {len(df_neg)} negative data points to training set")
        else:
            print(f"Warning: Negative data file not found at {neg_path}")
    
    print(f"\nFinal dataset:")
    print(f"  Train: {len(df_train)} samples ({df_train['reaction_id'].nunique()} groups)")
    print(f"  Test:  {len(df_test)} samples ({df_test['reaction_id'].nunique()} groups)")
    
    return df_train, df_test, available_features


def train_model(df_train, features, config):
    """Train the model with hyperparameter optimization."""
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    # Prepare training data
    X_train = df_train[features]
    y_train = df_train['r_product_class'].astype(int).values
    groups = df_train['reaction_id'].astype(str).values
    
    # Calculate class weights
    class_weights = model_training.calculate_class_weights(y_train)
    print("\nClass weights:")
    for cls, weight in class_weights.items():
        print(f"  Class {cls}: {weight:.4f}")
    
    # Define hyperparameter search space (same as train_final_model.py)
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
        X_train=X_train, y_train=y_train, groups=groups,
        param_grid=param_grid, n_iter=config['hyperparam_iter'],
        cv=5, random_state=config['random_state'],
        class_weights=class_weights, n_jobs=-1
    )
    
    print("\nBest hyperparameters:")
    for param, value in train_results['best_params'].items():
        print(f"  {param}: {value}")
    print(f"\nBest CV score (F1 weighted): {train_results['best_score']:.4f}")
    
    # Train final model on full training set
    print("\nTraining final model on full training set...")
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


def evaluate_and_save(model_info, df_holdout, config):
    """Evaluate model and save results."""
    print("\n" + "="*60)
    print("EVALUATION & SAVING")
    print("="*60)
    
    X_holdout = df_holdout[model_info['features']]
    y_holdout = df_holdout['r_product_class'].astype(int).values
    
    results = evaluation.evaluate_model(model_info['model'], X_holdout, y_holdout, labels=[0, 1, 2])
    evaluation.print_evaluation_results(results, title="Holdout Performance")
    
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    metadata = {
        'experiment': 'full_model_with_reaction_conditions',
        'timestamp': datetime.now().isoformat(),
        'cv_score': float(model_info['cv_score']),
        'holdout_accuracy': float(results['accuracy']),
        'holdout_f1_weighted': float(results['f1_weighted']),
        'holdout_f1_macro': float(results['f1_macro']),
        'best_params': model_info['best_params'],
        'num_features': len(model_info['features']),
        'features': model_info['features']
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    model_training.save_model_bundle(
        model=model_info['model'],
        feature_list=model_info['features'],
        class_labels=[0, 1, 2],
        out_dir=output_dir,
        metadata=metadata
    )
    
    print(f"\nSaved to: {output_dir}")


def main():
    args = parse_args()
    
    config = {
        'output_dir': args.output_dir,
        'random_state': args.random_state,
        'hyperparam_iter': args.hyperparam_iter,
        'add_negative_data': True,
        'use_augmentation': False,
    }
    
    print("="*60)
    print("FULL MODEL EXPERIMENT (WITH REACTION CONDITIONS)")
    print("="*60)
    
    df_train, df_test, features = load_presplit_data(config)
    model_info = train_model(df_train, features, config)
    evaluate_and_save(model_info, df_test, config)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

