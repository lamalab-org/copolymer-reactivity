#!/usr/bin/env python3
"""
Train model with Morgan fingerprints for monomer representation.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from copolpredictor import (
    model_training,
    evaluation,
    holdout_utils
)

import data_processing_morgan


def parse_args():
    parser = argparse.ArgumentParser(description="Train with Morgan fingerprints")
    parser.add_argument("--data-path", type=str, default="../../data_extraction/artifacts/datasets/extracted_reactions.csv")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--n-bits", type=int, default=2048)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--hyperparam-iter", type=int, default=25)
    return parser.parse_args()


def get_morgan_feature_columns(n_bits):
    """Get Morgan fingerprint + other feature columns."""
    morgan_features = []
    for i in range(n_bits):
        morgan_features.append(f'morgan_bit_{i}_1')
        morgan_features.append(f'morgan_bit_{i}_2')
    
    other_features = [
        'temperature',
        'polytype_emb_1', 'polytype_emb_2',
        'method_emb_1', 'method_emb_2',
        'solvent_logP', 'solvent_TPSA',
        'solvent_HBD', 'solvent_FractionCSP3'
    ]
    
    return morgan_features + other_features


def load_presplit_data(config):
    """Load pre-split train/test data with Morgan fingerprints."""
    print("\n" + "="*60)
    print("LOADING PRE-SPLIT DATA")
    print("="*60)
    
    train_path = os.path.join(os.path.dirname(__file__), '../../data/train_morgan.csv')
    test_path = os.path.join(os.path.dirname(__file__), '../../data/test_morgan.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Pre-split Morgan fingerprint data not found!")
        print(f"Expected files:")
        print(f"  - {train_path}")
        print(f"  - {test_path}")
        print(f"\nRun: python create_train_test_split.py --fingerprints")
        sys.exit(1)
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"Loaded train: {len(df_train)} samples ({df_train['reaction_id'].nunique()} groups)")
    print(f"Loaded test: {len(df_test)} samples ({df_test['reaction_id'].nunique()} groups)")
    
    # Get Morgan feature columns
    feature_columns = get_morgan_feature_columns(config['n_bits'])
    available_features = [c for c in feature_columns if c in df_train.columns]
    
    print(f"Using {len(available_features)} features")
    print(f"  Morgan bits: {config['n_bits'] * 2}")
    print(f"  Other: {len(available_features) - config['n_bits'] * 2}")
    
    # Note: NaN removal already done in create_train_test_split.py
    
    return df_train, df_test, available_features


def train_model(df_train, features, config):
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)
    
    X_train = df_train[features]
    y_train = df_train['r_product_class'].astype(int).values
    groups = df_train['reaction_id'].astype(str).values
    
    class_weights = model_training.calculate_class_weights(y_train)
    
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
    
    train_results = model_training.train_xgboost_with_cv(
        X_train=X_train, y_train=y_train, groups=groups,
        param_grid=param_grid, n_iter=config['hyperparam_iter'],
        cv=5, random_state=config['random_state'],
        class_weights=class_weights, n_jobs=-1
    )
    
    print(f"\nBest CV score: {train_results['best_score']:.4f}")
    
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
        'experiment': 'fingerprint',
        'timestamp': datetime.now().isoformat(),
        'morgan_n_bits': config['n_bits'],
        'morgan_radius': config['radius'],
        'cv_score': float(model_info['cv_score']),
        'holdout_accuracy': float(results['accuracy']),
        'holdout_f1_weighted': float(results['f1_weighted']),
        'holdout_f1_macro': float(results['f1_macro']),
        'best_params': model_info['best_params'],
        'num_features': len(model_info['features'])
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
        'data_path': args.data_path,
        'output_dir': args.output_dir,
        'n_bits': args.n_bits,
        'radius': args.radius,
        'random_state': args.random_state,
        'hyperparam_iter': args.hyperparam_iter,
        'remove_specialized': False,
    }
    
    print("="*60)
    print("FINGERPRINT EXPERIMENT")
    print("="*60)
    print(f"Morgan fingerprints: {config['n_bits']} bits, radius {config['radius']}")
    
    df_train, df_test, features = load_presplit_data(config)
    model_info = train_model(df_train, features, config)
    evaluate_and_save(model_info, df_test, config)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()

