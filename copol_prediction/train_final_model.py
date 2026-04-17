#!/usr/bin/env python3
"""
Final model training script for copolymerization prediction.

This script trains the production model with optimized settings and saves
it as a bundle for deployment.

Usage:
    python train_final_model.py [--output-dir DIR] [--analysis-output-dir DIR]

Paths are resolved under copol_prediction/ when relative (works from repo root).
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add parent directory to path
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..'))

from copolpredictor import (
    data_processing,
    data_augmentation,
    model_training,
    evaluation,
    holdout_utils,
    prediction_utils
)
from utils import load_data_split


def _resolve_under_script(path: str) -> str:
    """Relative paths are resolved under copol_prediction/ (this script's directory)."""
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)
    return os.path.normpath(os.path.join(_SCRIPT_DIR, path))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train final copolymerization prediction model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "artifacts", "model_bundle"),
        help="Directory to save model bundle (default: copol_prediction/artifacts/model_bundle)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use-augmentation",
        action="store_true",
        help="Enable Gaussian training augmentation (default: off)",
    )
    parser.add_argument(
        "--augmentation-samples",
        type=int,
        default=5,
        help="Number of augmented samples per datapoint (only if --use-augmentation)",
    )
    parser.add_argument(
        "--hyperparam-iter",
        type=int,
        default=25,
        help="Number of hyperparameter search iterations"
    )
    parser.add_argument(
        "--no-negative-data",
        action="store_true",
        help="Train model without negative data (for comparison studies)"
    )
    parser.add_argument(
        "--remove-specialized",
        action="store_true",
        help="Remove specialized datapoints from training set (only affects training, not validation/test)"
    )

    parser.add_argument(
        "--analysis-output-dir",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "output", "analysis"),
        help="Where to write analysis plots (default: copol_prediction/output/analysis)",
    )
    parser.add_argument(
        "--calibration-method",
        type=str,
        choices=["none", "platt", "isotonic"],
        default="isotonic",
        help="Post-hoc probability calibration fit on validation voting subset (default: isotonic).",
    )
    parser.add_argument(
        "--cv-prune-100-path",
        type=str,
        default=os.path.normpath(
            os.path.join(
                _SCRIPT_DIR,
                "..",
                "experiments",
                "cv_pruning",
                "results_val_full",
                "ids_error_ge_100pct_enriched.csv",
            )
        ),
        help=(
            "Optional path to ids_error_ge_100pct_enriched.csv from CV-pruning. "
            "If present, those reaction_ids are removed from TRAIN before fitting."
        ),
    )
    
    return parser.parse_args()


def prepare_data(config):
    """
    Load pre-split data and prepare for training.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_df, holdout_df, available_features)
    """
    print("\n" + "="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    # Load central train/validation/test split
    print("Loading central train/validation/test split...")
    try:
        split_dir = os.path.join(_SCRIPT_DIR, "artifacts", "data_splits")
        df_train, df_val, df_test = load_data_split.load_train_val_test_split(split_dir=split_dir)
        load_data_split.print_split_info(split_dir=split_dir)
        print(f"\nNote: Using Train for training, Validation for hyperparameter tuning (optional), Test for final evaluation")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease create the central split first:")
        print("  cd copol_prediction && python create_data_split.py")
        sys.exit(1)
    
    # Get available features
    available_features = [c for c in prediction_utils.feature_columns if c in df_train.columns]
    print(f"\nUsing {len(available_features)} features")
    print(f"  Train: {len(df_train)} samples ({df_train['reaction_id'].nunique()} groups)")
    print(f"  Test:  {len(df_test)} samples ({df_test['reaction_id'].nunique()} groups)")
    
    # Note: NaN removal already done in create_data_split.py

    # Apply specialized filter to training data only (if configured)
    if config.get('remove_specialized', False):
        if 'specialized_filter' in df_train.columns:
            before_count = len(df_train)
            df_train = df_train[df_train['specialized_filter'] != 'specialized'].reset_index(drop=True)
            removed_count = before_count - len(df_train)
            print(f"Removed {removed_count} specialized datapoints from training set")
            print(f"  Training set: {len(df_train)} samples ({df_train['reaction_id'].nunique()} groups)")
        else:
            print("Warning: 'specialized_filter' column not found in training data")

    # Apply CV-pruning: remove 100% error-rate reaction_ids from TRAIN only
    prune_path = config.get("cv_prune_100_path")
    if prune_path and os.path.exists(prune_path):
        try:
            df_prune = pd.read_csv(prune_path)
            if "reaction_id" in df_prune.columns:
                prune_ids = set(df_prune["reaction_id"].astype(str).tolist())
                before_rows = len(df_train)
                before_groups = df_train["reaction_id"].astype(str).nunique()
                df_train = df_train[~df_train["reaction_id"].astype(str).isin(prune_ids)].reset_index(drop=True)
                after_rows = len(df_train)
                after_groups = df_train["reaction_id"].astype(str).nunique()
                print(
                    f"Applied CV-pruning (100% error-rate): removed "
                    f"{before_rows - after_rows} rows / {before_groups - after_groups} groups "
                    f"from training set"
                )
            else:
                print(f"Warning: 'reaction_id' column not found in prune list: {prune_path}")
        except Exception as e:
            print(f"Warning: Failed to apply CV-pruning from {prune_path}: {e}")
    elif prune_path:
        print(f"Warning: CV-pruning file not found at {prune_path}. Proceeding without pruning.")
    
    # Add negative data if configured
    if config['add_negative_data']:
        neg_path = os.path.join(_SCRIPT_DIR, "filter", "artificial_datapoints", "processed_combined_augmented.csv")
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
    print(f"  Train:      {len(df_train)} samples ({df_train['reaction_id'].nunique()} groups)")
    print(f"  Validation: {len(df_val)} samples ({df_val['reaction_id'].nunique()} groups)")
    print(f"  Test:       {len(df_test)} samples ({df_test['reaction_id'].nunique()} groups)")
    
    # Note: Validation set is available but not used for training
    # It can be used for hyperparameter tuning if needed
    # For now, we train on Train and evaluate on Test
    
    return df_train, df_val, df_test, available_features


def fit_and_save_validation_calibration(
    *,
    model,
    df_train_lookup_pool: pd.DataFrame,
    df_val: pd.DataFrame,
    features: list[str],
    out_dir: str,
    method: str,
) -> str | None:
    """
    Fit per-class OVR calibrators on the VALIDATION set (voting subset),
    save to <out_dir>/calibration.joblib, and return the path.
    """
    if method == "none":
        print("\nCalibration disabled (--calibration-method none).")
        return None

    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression
    import joblib
    from analysis.analyze_model import (
        compute_fingerprints_for_smiles,
        compute_naive_baseline_predictions_with_similarity,
    )

    print("\n" + "=" * 60)
    print("FITTING PROBABILITY CALIBRATION (validation voting subset)")
    print("=" * 60)

    df_train_lookup_pool = df_train_lookup_pool.dropna(subset=features + ["r_product_class"]).reset_index(drop=True)
    df_val = df_val.dropna(subset=features + ["r_product_class"]).reset_index(drop=True)

    X_val = df_val[features]
    y_val = df_val["r_product_class"].astype(int).values

    # Base model probabilities on validation
    y_proba_raw = model.predict_proba(X_val)

    # Voting subset (Lookup + XGB agree)
    smiles_cols = ["monomer1_smiles", "monomer2_smiles", "solvent_smiles"]
    all_smiles = set()
    for data in [df_train_lookup_pool, df_val]:
        for c in smiles_cols:
            if c in data.columns:
                all_smiles.update(data[c].dropna().unique())
    fp_dict = compute_fingerprints_for_smiles(list(all_smiles))
    y_train_lookup = df_train_lookup_pool["r_product_class"].astype(int).values
    lookup_pred, _sim = compute_naive_baseline_predictions_with_similarity(
        df_val, df_train_lookup_pool, y_train_lookup, features, fp_dict=fp_dict
    )
    lookup_pred = np.asarray(lookup_pred).astype(int)
    xgb_pred = model.predict(X_val)
    agree = (xgb_pred == lookup_pred)

    y_val_v = y_val[agree]
    y_proba_v = y_proba_raw[agree]
    print(f"  Validation voting subset: {int(agree.sum())}/{len(agree)} ({agree.mean()*100:.1f}%)")

    # Fit per-class calibrators on voting subset
    calibrators = []
    for cls in range(3):
        y_binary = (y_val_v == cls).astype(int)
        p_cls = y_proba_v[:, cls]
        if method == "platt":
            cal = LogisticRegression(max_iter=1000)
            cal.fit(p_cls.reshape(-1, 1), y_binary)
        elif method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(p_cls, y_binary)
        else:
            raise ValueError(f"Unknown calibration method: {method}")
        calibrators.append(cal)

    payload = {
        "method": method,
        "fitted_on": "validation_voting_subset",
        "n_val_total": int(len(y_val)),
        "n_val_voting": int(len(y_val_v)),
        "calibrators": calibrators,
    }

    os.makedirs(out_dir, exist_ok=True)
    cal_path = os.path.join(out_dir, "calibration.joblib")
    joblib.dump(payload, cal_path)
    print(f"  ✓ Saved calibration payload to {cal_path}")
    return cal_path


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
    results["y_true"] = y_holdout
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
    
    # Get split info to know if specialized were removed
    split_info = load_data_split.get_split_info(
        split_dir=os.path.join(_SCRIPT_DIR, "artifacts", "data_splits")
    )
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
            'negative_data_used': config['add_negative_data'],
            'specialized_removed_from_training': config.get('remove_specialized', False),
            'specialized_removed_from_test': specialized_removed,  # This is always False (test/val never filtered)
            'cv_prune_100_path': config.get("cv_prune_100_path"),
            'used_central_split': True,
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
        y_true=holdout_results.get("y_true"),
        y_pred=holdout_results.get("predictions"),
        labels=[0, 1, 2],
        out_dir=os.path.join(config['output_dir'], "holdout_results"),
        filename="final_model_holdout.json"
    )
    
    print(f"\n✓ Model bundle saved to: {bundle_path}")


def save_all_metrics_to_file(model, df_train, df_test, features, output_dir, config):
    """
    Evaluate model on both train and test sets and save all metrics to a text file.
    
    Args:
        model: Trained model
        df_train: Training dataframe
        df_test: Test dataframe
        features: List of feature names
        output_dir: Output directory
        config: Configuration dictionary
    """
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )
    
    print("\n" + "="*60)
    print("CALCULATING ALL METRICS")
    print("="*60)
    
    # Evaluate on training set
    print("Evaluating on training set...")
    X_train = df_train[features]
    y_train = df_train['r_product_class'].astype(int).values
    y_train_pred = model.predict(X_train)
    
    train_cm = confusion_matrix(y_train, y_train_pred, labels=[0, 1, 2])
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision_weighted = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_recall_weighted = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
    train_precision_macro = precision_score(y_train, y_train_pred, average='macro', zero_division=0)
    train_recall_macro = recall_score(y_train, y_train_pred, average='macro', zero_division=0)
    train_f1_macro = f1_score(y_train, y_train_pred, average='macro', zero_division=0)
    
    # Per-class metrics for train
    train_precision_per_class = precision_score(y_train, y_train_pred, average=None, zero_division=0, labels=[0, 1, 2])
    train_recall_per_class = recall_score(y_train, y_train_pred, average=None, zero_division=0, labels=[0, 1, 2])
    train_f1_per_class = f1_score(y_train, y_train_pred, average=None, zero_division=0, labels=[0, 1, 2])
    
    # Evaluate on test set
    print("Evaluating on test set...")
    X_test = df_test[features]
    y_test = df_test['r_product_class'].astype(int).values
    y_test_pred = model.predict(X_test)
    
    test_cm = confusion_matrix(y_test, y_test_pred, labels=[0, 1, 2])
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision_weighted = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_recall_weighted = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
    test_precision_macro = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_recall_macro = recall_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    
    # Per-class metrics for test
    test_precision_per_class = precision_score(y_test, y_test_pred, average=None, zero_division=0, labels=[0, 1, 2])
    test_recall_per_class = recall_score(y_test, y_test_pred, average=None, zero_division=0, labels=[0, 1, 2])
    test_f1_per_class = f1_score(y_test, y_test_pred, average=None, zero_division=0, labels=[0, 1, 2])
    
    # Classification reports
    train_classification_report = classification_report(y_train, y_train_pred, labels=[0, 1, 2])
    test_classification_report = classification_report(y_test, y_test_pred, labels=[0, 1, 2])
    
    # Write to file
    metrics_file = os.path.join(output_dir, "all_metrics.txt")
    with open(metrics_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("COMPLETE MODEL EVALUATION METRICS\n")
        f.write("="*80 + "\n\n")
        
        # Configuration
        f.write("CONFIGURATION\n")
        f.write("-"*80 + "\n")
        f.write(f"Random State: {config['random_state']}\n")
        f.write(f"Augmentation Used: {config['use_augmentation']}\n")
        f.write(f"Augmentation Samples: {config['augmentation_samples']}\n")
        f.write(f"Negative Data Used: {config['add_negative_data']}\n")
        f.write(f"Number of Features: {len(features)}\n")
        f.write(f"Train Samples: {len(df_train)}\n")
        f.write(f"Test Samples: {len(df_test)}\n")
        f.write("\n")
        
        # Training Set Metrics
        f.write("="*80 + "\n")
        f.write("TRAINING SET METRICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("-"*80 + "\n")
        f.write("                Predicted\n")
        f.write("              Class 0  Class 1  Class 2\n")
        f.write(f"Actual Class 0  {train_cm[0,0]:6d}  {train_cm[0,1]:6d}  {train_cm[0,2]:6d}\n")
        f.write(f"Actual Class 1  {train_cm[1,0]:6d}  {train_cm[1,1]:6d}  {train_cm[1,2]:6d}\n")
        f.write(f"Actual Class 2  {train_cm[2,0]:6d}  {train_cm[2,1]:6d}  {train_cm[2,2]:6d}\n")
        f.write("\n")
        
        f.write("Overall Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"Accuracy:              {train_accuracy:.6f}\n")
        f.write(f"Precision (weighted):   {train_precision_weighted:.6f}\n")
        f.write(f"Recall (weighted):      {train_recall_weighted:.6f}\n")
        f.write(f"F1 Score (weighted):    {train_f1_weighted:.6f}\n")
        f.write(f"Precision (macro):      {train_precision_macro:.6f}\n")
        f.write(f"Recall (macro):         {train_recall_macro:.6f}\n")
        f.write(f"F1 Score (macro):       {train_f1_macro:.6f}\n")
        f.write("\n")
        
        f.write("Per-Class Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<10} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}\n")
        f.write("-"*80 + "\n")
        for i, class_label in enumerate([0, 1, 2]):
            f.write(f"Class {class_label:<6} {train_precision_per_class[i]:<15.6f} {train_recall_per_class[i]:<15.6f} {train_f1_per_class[i]:<15.6f}\n")
        f.write("\n")
        
        f.write("Classification Report:\n")
        f.write("-"*80 + "\n")
        f.write(train_classification_report)
        f.write("\n\n")
        
        # Test Set Metrics
        f.write("="*80 + "\n")
        f.write("TEST SET METRICS\n")
        f.write("="*80 + "\n\n")
        
        f.write("Confusion Matrix:\n")
        f.write("-"*80 + "\n")
        f.write("                Predicted\n")
        f.write("              Class 0  Class 1  Class 2\n")
        f.write(f"Actual Class 0  {test_cm[0,0]:6d}  {test_cm[0,1]:6d}  {test_cm[0,2]:6d}\n")
        f.write(f"Actual Class 1  {test_cm[1,0]:6d}  {test_cm[1,1]:6d}  {test_cm[1,2]:6d}\n")
        f.write(f"Actual Class 2  {test_cm[2,0]:6d}  {test_cm[2,1]:6d}  {test_cm[2,2]:6d}\n")
        f.write("\n")
        
        f.write("Overall Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"Accuracy:              {test_accuracy:.6f}\n")
        f.write(f"Precision (weighted):   {test_precision_weighted:.6f}\n")
        f.write(f"Recall (weighted):      {test_recall_weighted:.6f}\n")
        f.write(f"F1 Score (weighted):    {test_f1_weighted:.6f}\n")
        f.write(f"Precision (macro):      {test_precision_macro:.6f}\n")
        f.write(f"Recall (macro):         {test_recall_macro:.6f}\n")
        f.write(f"F1 Score (macro):       {test_f1_macro:.6f}\n")
        f.write("\n")
        
        f.write("Per-Class Metrics:\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Class':<10} {'Precision':<15} {'Recall':<15} {'F1 Score':<15}\n")
        f.write("-"*80 + "\n")
        for i, class_label in enumerate([0, 1, 2]):
            f.write(f"Class {class_label:<6} {test_precision_per_class[i]:<15.6f} {test_recall_per_class[i]:<15.6f} {test_f1_per_class[i]:<15.6f}\n")
        f.write("\n")
        
        f.write("Classification Report:\n")
        f.write("-"*80 + "\n")
        f.write(test_classification_report)
        f.write("\n")
    
    print(f"\n✓ All metrics saved to: {metrics_file}")
    return metrics_file


def evaluate_voting_on_test_set(model, df_train_lookup_pool, df_test, features, output_dir):
    """
    Evaluate XGB-only vs Voting (XGB + Lookup) on the TEST set.

    Voting definition matches the analysis scripts: keep only samples where
    XGBoost and Lookup agree; metrics are computed on that retained subset.
    """
    import json
    from sklearn.metrics import (
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    print("\n" + "=" * 60)
    print("TEST SET EVALUATION (Voting: XGB + Lookup)")
    print("=" * 60)

    # Local imports (RDKit-heavy)
    from analysis.analyze_model import (
        compute_fingerprints_for_smiles,
        compute_naive_baseline_predictions_with_similarity,
    )

    # Drop NaNs for fair comparison
    df_train_lookup_pool = df_train_lookup_pool.dropna(subset=features + ["r_product_class"]).reset_index(drop=True)
    df_test = df_test.dropna(subset=features + ["r_product_class"]).reset_index(drop=True)

    X_test = df_test[features]
    y_test = df_test["r_product_class"].astype(int).values

    # XGBoost predictions (used only to define the voting subset)
    xgb_pred = model.predict(X_test)

    # Lookup predictions (with cached fingerprints)
    smiles_cols = ["monomer1_smiles", "monomer2_smiles", "solvent_smiles"]
    all_smiles = set()
    for data in [df_train_lookup_pool, df_test]:
        for c in smiles_cols:
            if c in data.columns:
                all_smiles.update(data[c].dropna().unique())
    fp_dict = compute_fingerprints_for_smiles(list(all_smiles))

    y_train_lookup = df_train_lookup_pool["r_product_class"].astype(int).values
    lookup_pred, _sim = compute_naive_baseline_predictions_with_similarity(
        df_test, df_train_lookup_pool, y_train_lookup, features, fp_dict=fp_dict
    )
    lookup_pred = np.asarray(lookup_pred).astype(int)

    agree = (xgb_pred == lookup_pred)
    y_test_v = y_test[agree]
    xgb_pred_v = xgb_pred[agree]

    def _metrics(yt, yp):
        if len(yt) == 0:
            return {
                "n": 0,
                "balanced_accuracy": None,
                "precision_macro": None,
                "recall_macro": None,
                "f1_macro": None,
                "confusion_matrix": None,
            }
        return {
            "n": int(len(yt)),
            "balanced_accuracy": float(balanced_accuracy_score(yt, yp)),
            "precision_macro": float(precision_score(yt, yp, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(yt, yp, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "confusion_matrix": confusion_matrix(yt, yp, labels=[0, 1, 2]).tolist(),
        }

    payload = {
        "test_total": int(len(y_test)),
        "voting": {
            "coverage": float(agree.mean()) if len(agree) else 0.0,
            "agree_n": int(agree.sum()),
            **_metrics(y_test_v, xgb_pred_v),
        }
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "voting_test_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Test total: {payload['test_total']}")
    print(f"Voting coverage: {payload['voting']['coverage']:.3f} ({payload['voting']['agree_n']} retained)")
    print(f"✓ Voting test metrics saved to: {out_path}")
    return out_path


def run_analysis(model_path, data_path, output_dir):
    """
    Run model analysis after training.

    Evaluates the voting model (XGBoost + Lookup) on the test set.
    Plots are generated twice: unfiltered and voting-filtered (threshold 0.7).

    Args:
        model_path: Path to trained model bundle
        data_path: Path to processed data (unused, kept for signature compat)
        output_dir: Output directory for analysis plots
    """
    print("\n" + "=" * 60)
    print("RUNNING MODEL ANALYSIS  (test set, voting model)")
    print("=" * 60)

    try:
        from analysis import analyze_model
        from utils.load_data_split import load_train_val_test_split
        from analysis.plot_class_curves import plot_class_curves

        class AnalysisArgs:
            def __init__(self):
                self.model_path = model_path
                self.output_dir = output_dir
                self.all = True
                self.combined = False
                self.confusion = False
                self.confidence = False
                self.features = False
                self.calibration = True  # explicit: multiclass reliability / calibration_curves.* on voting test subset
                self.errors = False
                self.confidence_vs_r1r2 = False
                self.filtering = False
                self.min_retention = 0.7
                self.confidence_threshold = 0.7
                self.latex_table = True
                self.n_folds = 5

        analyze_model.setup_style()
        os.makedirs(output_dir, exist_ok=True)

        predictor = analyze_model.CopolymerPredictor(model_path)
        print(f"  ✓ Model loaded ({len(predictor.features)} features)")

        # Load test set from the global train/validation/test split
        split_dir = os.path.join(_SCRIPT_DIR, "artifacts", "data_splits")
        df_train, df_val, df_test = load_train_val_test_split(split_dir=split_dir)
        print(f"  ✓ Test set loaded ({len(df_test)} samples)")

        args = AnalysisArgs()

        if args.latex_table:
            analyze_model.create_latex_performance_table(
                predictor, output_dir,
                n_folds=args.n_folds,
                confidence_threshold=args.confidence_threshold
            )
            analyze_model.create_latex_per_class_table(
                predictor, output_dir,
                confidence_threshold=args.confidence_threshold
            )

        analyze_model.generate_plots_for_dataset(df_test, predictor, args, suffix='')

        # Also generate Mayo–Lewis "class curves" figure (uses constants from split)
        try:
            df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
            out_class_curves = plot_class_curves(df_all=df_all, output_dir=output_dir)
            print(f"  ✓ Class curves written to: {out_class_curves}")
        except Exception as e:
            print(f"  ⚠ Failed to generate class curves: {e}")

        cal_base = os.path.join(output_dir, "calibration_curves")
        cal_ok = os.path.isfile(cal_base + ".png") or os.path.isfile(cal_base + ".pdf")
        if cal_ok:
            print(f"  ✓ Calibration curves written under: {output_dir}/ (calibration_curves.png/pdf)")
        else:
            print(
                f"  ⚠ Calibration curves not found at {cal_base}.png/pdf — "
                "check console for errors during plot_calibration_curve_multiclass"
            )

        print(f"\n  ✓ Analysis complete! Plots saved to: {output_dir}/")

    except Exception as e:
        print(f"\n  ✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        print("  You can run analysis manually with:")
        print(f"    python analysis/analyze_model.py --model-path {model_path}")


def main():
    """Main training pipeline."""
    args = parse_args()
    args.output_dir = _resolve_under_script(args.output_dir)
    args.analysis_output_dir = _resolve_under_script(args.analysis_output_dir)
    if args.cv_prune_100_path:
        args.cv_prune_100_path = _resolve_under_script(args.cv_prune_100_path)

    # Configuration
    config = {
        'output_dir': args.output_dir,
        'analysis_output_dir': args.analysis_output_dir,
        'random_state': args.random_state,
        'augmentation_samples': args.augmentation_samples,
        'hyperparam_iter': args.hyperparam_iter,
        # Training settings
        # Negative data is now disabled by default for final model training
        'add_negative_data': False,
        'use_augmentation': bool(args.use_augmentation),
        'remove_specialized': args.remove_specialized,  # Remove specialized datapoints if flag is set
        # CV-pruning (data error analysis filter @ 100%)
        'cv_prune_100_path': args.cv_prune_100_path,
        'calibration_method': args.calibration_method,
    }
    
    print("="*60)
    print("COPOLYMERIZATION PREDICTION - FINAL MODEL TRAINING")
    print("="*60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nℹ️  Note: Using central train/validation/test split from artifacts/data_splits/")
    print("   To recreate split: python create_data_split.py")
    
    # Prepare data (loads central split)
    df_train, df_val, df_holdout, features = prepare_data(config)
    
    # Train model
    model_info = train_model(df_train, features, config)
    
    # Evaluate on holdout
    holdout_results = evaluate_on_holdout(model_info['model'], df_holdout, features)
    
    # Save model
    save_model(model_info, holdout_results, config)
    
    # Save all metrics to text file
    save_all_metrics_to_file(
        model=model_info['model'],
        df_train=df_train,
        df_test=df_holdout,
        features=features,
        output_dir=config['output_dir'],
        config=config
    )

    # Evaluate voting on TEST set (XGBoost + Lookup)
    # Lookup pool is TRAIN (+ optional negative data, if available)
    df_lookup_pool = df_train.copy()
    neg_path = os.path.join(_SCRIPT_DIR, "filter", "artificial_datapoints", "processed_combined_augmented.csv")
    if os.path.exists(neg_path):
        try:
            df_neg = pd.read_csv(neg_path)
            if "Class" in df_neg.columns and "r_product_class" not in df_neg.columns:
                df_neg = df_neg.rename(columns={"Class": "r_product_class"})
            if "r_product_class" in df_neg.columns:
                df_neg["r_product_class"] = df_neg["r_product_class"].astype(int)
                df_lookup_pool = pd.concat([df_lookup_pool, df_neg], ignore_index=True)
                print(f"\n✓ Added {len(df_neg)} negative datapoints to lookup pool ({len(df_lookup_pool)} total)")
        except Exception as e:
            print(f"\nWarning: Failed to add negative data to lookup pool: {e}")

    evaluate_voting_on_test_set(
        model=model_info["model"],
        df_train_lookup_pool=df_lookup_pool,
        df_test=df_holdout,
        features=features,
        output_dir=config["output_dir"],
    )

    # Fit calibration on VALIDATION voting subset and save into model bundle directory
    cal_path = fit_and_save_validation_calibration(
        model=model_info["model"],
        df_train_lookup_pool=df_lookup_pool,
        df_val=df_val,
        features=features,
        out_dir=config["output_dir"],
        method=config["calibration_method"],
    )
    if cal_path:
        # Update meta.json to reflect that calibration exists
        try:
            meta_path = os.path.join(config["output_dir"], "meta.json")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta.setdefault("training_config", {})
            meta["training_config"]["calibration_method"] = config["calibration_method"]
            meta["training_config"]["calibration_fitted_on"] = "validation_voting_subset"
            meta["training_config"]["calibration_file"] = "calibration.joblib"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
            print(f"\n✓ Updated bundle metadata with calibration info: {meta_path}")
        except Exception as e:
            print(f"\nWarning: Failed to update meta.json with calibration info: {e}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {config['output_dir']}")
    print("\nTo use the model:")
    print("  from copolpredictor.inference import CopolymerPredictor")
    print(f"  predictor = CopolymerPredictor('{config['output_dir']}')")
    print("  predictions = predictor.predict(X)")
    
    # Run automatic analysis (always under copol_prediction/output/analysis by default)
    run_analysis(
        model_path=config['output_dir'],
        data_path=os.path.join(_SCRIPT_DIR, "output", "processed_data.csv"),
        output_dir=config['analysis_output_dir'],
    )


if __name__ == "__main__":
    main()

