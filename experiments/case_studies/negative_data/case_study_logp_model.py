#!/usr/bin/env python3
"""
Case study: Train logP model on all training data (normal + negative)
and evaluate on train (normal), train (negative), and a test split
of the same negative data.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "copol_prediction"))

from copolpredictor import (
    model_training,
    evaluation,
    data_processing
)

# Import from train_simple_logp_model
sys.path.insert(0, str(Path(__file__).parent))
from train_simple_logp_model import (
    load_training_data,
    load_negative_data,
    prepare_negative_test_data,
    add_logp_features,
    convert_to_binary_classification,
    combine_training_data,
    split_negative_data,
    remove_duplicates_by_logp_features,
    sample_normal_data_to_match_negative,
    train_model,
    print_detailed_predictions,
    load_logp_cache,
    save_logp_cache,
    reorder_monomers_by_logp,
)


def evaluate_model_on_dataset(model, df_test, features, dataset_name):
    """Evaluate model on a dataset and return macro metrics."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix
    )
    
    # Get model features
    if hasattr(model, 'feature_names_in_'):
        model_features = list(model.feature_names_in_)
    elif hasattr(model, 'get_booster'):
        try:
            booster = model.get_booster()
            model_features = booster.feature_names
            if not model_features:
                model_features = features
        except:
            model_features = features
    else:
        model_features = features
    
    # Check for missing features
    missing_features = set(model_features) - set(df_test.columns)
    if missing_features:
        for feat in missing_features:
            df_test[feat] = np.nan
    
    X_test = df_test[model_features].copy()
    
    # Check for NaN values BEFORE filling
    nan_counts = X_test.isna().sum()
    if nan_counts.sum() > 0:
        print(f"\n⚠️ Warning: Found NaN values in features:")
        for feat, count in nan_counts[nan_counts > 0].items():
            print(f"   {feat}: {count} NaN values ({100*count/len(X_test):.1f}%)")
            # Show which rows have NaN
            nan_rows = X_test[X_test[feat].isna()].index
            print(f"      Rows with NaN: {list(nan_rows[:5])}...")
        X_test = X_test.fillna(0)
        print("   Filled NaN values with 0")
    
    # Check feature statistics
    print(f"\n📊 Feature statistics for {dataset_name}:")
    for feat in model_features:
        print(f"   {feat}: min={X_test[feat].min():.3f}, max={X_test[feat].max():.3f}, mean={X_test[feat].mean():.3f}, std={X_test[feat].std():.3f}")
    
    # Get true labels (binary)
    # Use r_product_class_binary if available (after convert_to_binary_classification)
    # Otherwise convert from r_product_class
    if 'r_product_class_binary' in df_test.columns:
        y_test = df_test['r_product_class_binary'].astype(int).values
    else:
        y_test = (df_test['r_product_class'] == 2).astype(int).values
    
    print(f"\n📊 Dataset info for {dataset_name}:")
    print(f"  Total samples: {len(y_test)}")
    print(f"  Class 0 (Alternating/Random): {(y_test == 0).sum()}")
    print(f"  Class 1 (Homopolymer): {(y_test == 1).sum()}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Debug: Print feature values and predictions for first few samples
    print(f"\n🔍 Debug: First 5 samples predictions:")
    print(f"  Features shape: {X_test.shape}")
    print(f"  Feature names: {list(X_test.columns)}")
    for i in range(min(5, len(X_test))):
        print(f"\n  Sample {i+1}:")
        print(f"    Features: {dict(X_test.iloc[i])}")
        print(f"    True label: {y_test[i]}")
        print(f"    Predicted: {y_pred[i]}")
    
    # Calculate macro accuracy (average of per-class accuracy)
    accuracy_per_class = []
    for cls in [0, 1]:
        cls_mask = (y_test == cls)
        if cls_mask.sum() > 0:
            cls_accuracy = accuracy_score(y_test[cls_mask], y_pred[cls_mask])
            accuracy_per_class.append(cls_accuracy)
        else:
            accuracy_per_class.append(0.0)
    accuracy_macro = np.mean(accuracy_per_class)
    
    # Calculate macro precision
    precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Calculate macro recall
    recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Calculate macro F1
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    
    print(f"\n📊 Results for {dataset_name}:")
    print(f"  Accuracy (macro): {accuracy_macro:.4f}")
    print(f"  Precision (macro): {precision_macro:.4f}")
    print(f"  Recall (macro): {recall_macro:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Class 0  Class 1")
    print(f"Actual Class 0  {cm[0,0]:6d}  {cm[0,1]:6d}")
    print(f"Actual Class 1  {cm[1,0]:6d}  {cm[1,1]:6d}")
    print(f"\n  Prediction distribution:")
    pred_counts = pd.Series(y_pred).value_counts().sort_index()
    for cls, count in pred_counts.items():
        print(f"    Predicted Class {cls}: {count} samples")
    
    return {
        'accuracy_macro': accuracy_macro,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'y_true': y_test,
        'y_pred': y_pred
    }


def plot_case_study_performance(results_train_normal, results_train_neg, results_test_neg_split, output_dir):
    """Create case study performance plot comparing train vs test performance."""
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = [
        'Train\n(Normal)',
        'Train\n(Negative)',
        'Test\n(Neg Split)',
    ]
    
    accuracies = [
        results_train_normal['accuracy_macro'],
        results_train_neg['accuracy_macro'],
        results_test_neg_split['accuracy_macro'],
    ]
    
    precisions = [
        results_train_normal['precision_macro'],
        results_train_neg['precision_macro'],
        results_test_neg_split['precision_macro'],
    ]
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Macro Accuracy', alpha=0.8, color='#1f77b4')
    bars2 = ax.bar(x + width/2, precisions, width, label='Macro Precision', alpha=0.8, color='#ff7f0e')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('LogP Model Performance\n(Trained on: Normal + Negative Train Data)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Add value labels on bars
    for bars, values in [(bars1, accuracies), (bars2, precisions)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'case_study_logp_performance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {plot_path}")
    
    plot_path_pdf = os.path.join(output_dir, 'case_study_logp_performance.pdf')
    plt.savefig(plot_path_pdf, bbox_inches='tight')
    print(f"✅ Saved: {plot_path_pdf}")
    
    plt.close()


def main():
    """Main case study pipeline."""
    script_dir = Path(__file__).parent
    
    # Paths
    negative_data_path = script_dir / "processed_combined_augmented.csv"
    output_dir = script_dir / "results" / "case_study_logp"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load logP cache
    load_logp_cache()
    
    # Configuration
    config = {
        'output_dir': str(output_dir),
        'random_state': 42,
        'hyperparam_iter': 25,
        'negative_data_path': str(negative_data_path),
    }
    
    print("="*60)
    print("CASE STUDY: LOGP MODEL")
    print("="*60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("\nFeatures: monomer1_logP, monomer2_logP, solvent_logP")
    print("Classification: Binary (0/1 vs 2)")
    
    # Load original training data
    df_original, df_test_normal = load_training_data()
    
    # Load negative data
    df_neg_all = load_negative_data(negative_data_path)
    
    # Add logP features FIRST (needed for deduplication and fair splitting)
    print("\n" + "="*60)
    print("ADDING LOGP FEATURES")
    print("="*60)
    df_original = add_logp_features(df_original)
    df_neg_all = add_logp_features(df_neg_all)
    
    # Reorder monomers so that monomer1_logP <= monomer2_logP
    df_original = reorder_monomers_by_logp(df_original)
    df_neg_all = reorder_monomers_by_logp(df_neg_all)
    
    # Save cache
    save_logp_cache()
    
    # Remove duplicates FIRST (before splitting) based on logP features
    # This ensures train and test don't have the same reactions
    df_original = remove_duplicates_by_logp_features(df_original)
    df_neg_all = remove_duplicates_by_logp_features(df_neg_all)
    
    # NOW split negative data into train/test (80/20) AFTER deduplication
    # Split is based on monomer+solvent combinations (logP features) for fair split
    df_neg_train, df_neg_test = split_negative_data(df_neg_all, test_size=0.2, random_state=config['random_state'])
    
    # Sample normal data to match negative data size
    df_original = sample_normal_data_to_match_negative(df_original, df_neg_train, random_state=config['random_state'])
    
    # Convert to binary classification
    print("\n" + "="*60)
    print("CONVERTING TO BINARY CLASSIFICATION")
    print("="*60)
    df_original = convert_to_binary_classification(df_original)
    df_neg_train = convert_to_binary_classification(df_neg_train)
    df_neg_test = convert_to_binary_classification(df_neg_test)
    
    # Combine datasets (only negative TRAIN data goes into training)
    df_train_combined = combine_training_data(df_original, df_neg_train)
    
    # Train model
    model_info = train_model(df_train_combined, config)
    
    # Evaluate on three datasets
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Evaluate on normal training data
    results_train_normal = evaluate_model_on_dataset(
        model_info['model'],
        df_original,
        model_info['features'],
        "Training Data (Normal)"
    )
    
    # Evaluate on negative training data (the part used for training)
    results_train_neg = evaluate_model_on_dataset(
        model_info['model'],
        df_neg_train,
        model_info['features'],
        "Training Data (Negative)"
    )
    
    # Print detailed predictions for negative training data
    print_detailed_predictions(
        model_info['model'],
        df_neg_train,
        model_info['features'],
        dataset_name="Training Data (Negative)"
    )
    
    # Evaluate on negative test data (the held-out 20% split from processed_combined_augmented.csv)
    results_test_neg_split = evaluate_model_on_dataset(
        model_info['model'],
        df_neg_test,
        model_info['features'],
        "Negative Test Data (Split)"
    )
    
    # Print detailed predictions for negative test split
    print_detailed_predictions(
        model_info['model'],
        df_neg_test,
        model_info['features'],
        dataset_name="Negative Test Data (Split)"
    )
    
    # Create plot
    print("\n" + "="*60)
    print("CREATING CASE STUDY PERFORMANCE PLOT")
    print("="*60)
    plot_case_study_performance(
        results_train_normal,
        results_train_neg,
        results_test_neg_split,
        output_dir
    )
    
    print("\n" + "="*60)
    print("CASE STUDY COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
