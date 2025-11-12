"""
Evaluation utilities for copolymerization prediction models.

This module provides functions for model evaluation, metrics calculation,
and results visualization.
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def save_holdout_metrics_json(
    y_true,
    y_pred,
    labels=None,
    out_dir="artifacts/experiments_holdout",
    filename=None
):
    """
    Save hold-out evaluation results to JSON.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels (e.g., [0,1,2])
        out_dir: Output directory
        filename: Output filename (auto-generated if None)
        
    Returns:
        Path to saved file
    """
    os.makedirs(out_dir, exist_ok=True)

    # If labels are not provided, infer them from the data
    if labels is None:
        labels = sorted(set(map(int, set(y_true))))

    report = classification_report(y_true, y_pred, labels=list(labels), output_dict=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))

    payload = {
        "timestamp": datetime.datetime.now().isoformat(),
        "labels": list(labels),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }

    if filename is None:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"holdout_{ts}.json"

    fpath = os.path.join(out_dir, filename)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    return fpath


def save_results_with_confidence(
    fold_scores_df,
    features,
    class_counts,
    overall_accuracy,
    overall_f1,
    y_true,
    y_pred,
    y_pred_proba,
    confidence_scores,
    df_clean,
    output_dir="output"
):
    """
    Save classification results with confidence scores.
    
    Args:
        fold_scores_df: DataFrame with fold-wise scores
        features: List of feature names
        class_counts: Class distribution
        overall_accuracy: Overall accuracy
        overall_f1: Overall F1 score
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Prediction probabilities
        confidence_scores: Confidence scores
        df_clean: Original dataframe with features
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save fold scores
    fold_scores_df.to_csv(f'{output_dir}/xgboost_fold_scores.csv', index=False)

    n_classes = y_pred_proba.shape[1]

    # Build predictions dictionary
    predictions_dict = {
        'true_label': y_true,
        'predicted_label': y_pred,
        'confidence_score': confidence_scores,
        'correct_prediction': np.array(y_pred) == np.array(y_true)
    }

    # Add per-class probabilities
    for cls_idx in range(n_classes):
        predictions_dict[f'proba_class_{cls_idx}'] = y_pred_proba[:, cls_idx]

    predictions_df = pd.DataFrame(predictions_dict)

    # Add features from original dataframe
    if len(predictions_df) == len(df_clean):
        exclude_columns = ['true_label', 'predicted_label', 'confidence_score', 'correct_prediction']
        for col in df_clean.columns:
            if col not in exclude_columns and col not in predictions_df.columns:
                predictions_df[col] = df_clean[col].values
    else:
        print(f"Warning: Predictions length ({len(predictions_df)}) doesn't match data length ({len(df_clean)})")

    # Save predictions
    predictions_df.to_csv(f'{output_dir}/predictions_with_confidence.csv', index=False)

    # Calculate confidence statistics
    correct_predictions = predictions_df['correct_prediction']
    high_conf = predictions_df['confidence_score'] > 0.8
    med_conf = (predictions_df['confidence_score'] > 0.6) & (predictions_df['confidence_score'] <= 0.8)
    low_conf = predictions_df['confidence_score'] <= 0.6

    confidence_stats = {
        'high_confidence_accuracy': correct_predictions[high_conf].mean() if high_conf.sum() > 0 else 0,
        'medium_confidence_accuracy': correct_predictions[med_conf].mean() if med_conf.sum() > 0 else 0,
        'low_confidence_accuracy': correct_predictions[low_conf].mean() if low_conf.sum() > 0 else 0,
        'high_confidence_count': high_conf.sum(),
        'medium_confidence_count': med_conf.sum(),
        'low_confidence_count': low_conf.sum(),
        'mean_confidence': predictions_df['confidence_score'].mean(),
        'std_confidence': predictions_df['confidence_score'].std()
    }

    # Save summary statistics
    summary_stats = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
        summary_stats[f'{metric}_mean'] = fold_scores_df[metric].mean()
        summary_stats[f'{metric}_std'] = fold_scores_df[metric].std()

    summary_stats.update(confidence_stats)
    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv(f'{output_dir}/summary.csv', index=False)

    print(f"\nResults saved to:")
    print(f"- {output_dir}/xgboost_fold_scores.csv")
    print(f"- {output_dir}/predictions_with_confidence.csv")
    print(f"- {output_dir}/summary.csv")


def plot_feature_importance(model, feature_names, top_n=10, output_path="output/feature_importance.png"):
    """
    Plot feature importance for XGBoost model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        top_n: Number of top features to plot
        output_path: Path to save plot
    """
    # Check if model is a CalibratedClassifierCV wrapper
    if hasattr(model, "estimator"):
        model = model.estimator

    # Extract feature importances
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        print("Model has no feature_importances_. Skipping plot.")
        return

    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importances')
    plt.barh(range(top_n), importance[indices], color='#661124')
    plt.yticks(range(top_n), [feature_names[i] for i in indices], fontsize=14)
    plt.xlabel('Feature Importance', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Feature importance plot saved to {output_path}")


def evaluate_model(model, X_test, y_test, labels=None):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        labels: List of class labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        # Weighted metrics (für Vergleich)
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        # Macro metrics (Haupt-Metriken)
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'classification_report': classification_report(y_test, y_pred, labels=labels),
        'confusion_matrix': confusion_matrix(y_test, y_pred, labels=labels),
        'predictions': y_pred,
        'probabilities': y_proba
    }
    
    return results


def print_evaluation_results(results, title="Evaluation Results"):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary with evaluation metrics
        title: Title for the output
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1 (weighted): {results['f1_weighted']:.4f}")
    print(f"F1 (macro):    {results['f1_macro']:.4f}")
    print("\nClassification Report:")
    print(results['classification_report'])
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])

