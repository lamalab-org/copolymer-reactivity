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
from utils import load_data_split


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train final copolymerization prediction model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/model_bundle",
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
    
    # Load central train/test split
    print("Loading central train/test split...")
    try:
        df_train, df_test = load_data_split.load_train_test_split()
        load_data_split.print_split_info()
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease create the central split first:")
        print("  cd ../experiments && python create_data_split.py")
        sys.exit(1)
    
    # Get available features
    available_features = [c for c in prediction_utils.feature_columns if c in df_train.columns]
    print(f"\nUsing {len(available_features)} features")
    print(f"  Train: {len(df_train)} samples ({df_train['reaction_id'].nunique()} groups)")
    print(f"  Test:  {len(df_test)} samples ({df_test['reaction_id'].nunique()} groups)")
    
    # Note: NaN removal already done in create_data_split.py

    # Add negative data if configured
    if config['add_negative_data']:
        neg_path = "filter/artificial_datapoints/processed_combined_augmented.csv"
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
    
    # Get split info to know if specialized were removed
    split_info = load_data_split.get_split_info()
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
            'specialized_removed_from_test': specialized_removed,
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
        y_true=holdout_results['predictions'],  # This should be y_true
        y_pred=holdout_results['predictions'],
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


def run_analysis(model_path, data_path, output_dir):
    """
    Run model analysis after training.
    
    Args:
        model_path: Path to trained model bundle
        data_path: Path to processed data
        output_dir: Output directory for analysis plots
    """
    print("\n" + "="*60)
    print("RUNNING MODEL ANALYSIS")
    print("="*60)
    
    try:
        # Import analyze_model module
        from analysis import analyze_model
        
        # Create mock args for analyze_model
        class AnalysisArgs:
            def __init__(self):
                self.model_path = model_path
                self.data_path = data_path
                self.output_dir = output_dir
                self.holdout_only = False
                self.compare_holdout = True
                self.all = True
                self.confusion = False
                self.confidence = False
                self.features = False
                self.calibration = False
                self.errors = False
                self.confidence_vs_r1r2 = False
                self.filtering = False
                self.min_retention = 0.7
        
        # Run analysis
        analyze_model.setup_style()
        os.makedirs(output_dir, exist_ok=True)
        
        predictor = analyze_model.CopolymerPredictor(model_path)
        print(f"  ✓ Model loaded ({len(predictor.features)} features)")
        
        df_all = pd.read_csv(data_path)
        print(f"  ✓ Data loaded ({len(df_all)} samples)")
        
        args = AnalysisArgs()
        
        # Generate plots for both all data and holdout
        print("\n### All Data ###")
        print(f"  Samples: {len(df_all)}")
        analyze_model.generate_plots_for_dataset(df_all, predictor, args, suffix='All Data')
        
        print("\n### Holdout Set ###")
        from copolpredictor.holdout_utils import get_or_create_holdout_groups, make_base_dataset_for_holdout
        try:
            base_df = make_base_dataset_for_holdout(df_all)
            holdout_groups = get_or_create_holdout_groups(base_df)
            df_holdout = df_all[df_all['reaction_id'].astype(str).isin(holdout_groups)].reset_index(drop=True)
            print(f"  Samples: {len(df_holdout)}")
            analyze_model.generate_plots_for_dataset(df_holdout, predictor, args, suffix='Holdout')
        except Exception as e:
            print(f"  ✗ Could not filter to holdout: {e}")
        
        print("\n  ✓ Analysis complete! Plots saved to: {output_dir}/")
        
    except Exception as e:
        print(f"\n  ✗ Analysis failed: {e}")
        print("  You can run analysis manually with:")
        print(f"    python analysis/analyze_model.py --model-path {model_path}")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    # Configuration
    config = {
        'output_dir': args.output_dir,
        'random_state': args.random_state,
        'augmentation_samples': args.augmentation_samples,
        'hyperparam_iter': args.hyperparam_iter,
        # Training settings (can be adjusted)
        'add_negative_data': True,
        'use_augmentation': False,
    }
    
    print("="*60)
    print("COPOLYMERIZATION PREDICTION - FINAL MODEL TRAINING")
    print("="*60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nℹ️  Note: Using central train/test split from artifacts/data_splits/")
    print("   To recreate split: cd ../experiments && python create_data_split.py")
    
    # Prepare data (loads central split)
    df_train, df_holdout, features = prepare_data(config)
    
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
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nModel saved to: {config['output_dir']}")
    print("\nTo use the model:")
    print("  from copolpredictor.inference import CopolymerPredictor")
    print(f"  predictor = CopolymerPredictor('{config['output_dir']}')")
    print("  predictions = predictor.predict(X)")
    
    # Run automatic analysis
    run_analysis(
        model_path=config['output_dir'],
        data_path="output/processed_data.csv",
        output_dir="output/analysis"
    )


if __name__ == "__main__":
    main()

