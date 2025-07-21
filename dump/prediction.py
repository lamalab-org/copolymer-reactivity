"""
Extension to main.py to add bucket-based classification model
"""

import os
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier


# Import the BucketClassifier
from copolpredictor.bucket_model import DistributionBucketRegressor
from copolpredictor import models, data_processing, result_visualization as visualization
import copolextractor.utils as utils
from copolpredictor.models import DualDistanceModelTrainer
from copolpredictor.models import ProductVsIndividualModelTrainer

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler


def run_bucket_classifier(df, random_state=42, optimize=True):
    """
    Run simplified bucket distribution model on the data
    Args:
        df: DataFrame with features and target
        random_state: Seed for reproducibility
        optimize: Whether to run hyperparameter optimization
    """
    print("\n=== Running Distribution-Based Bucket Model ===")
    os.makedirs("output/classification", exist_ok=True)
    os.makedirs("output/classification/prob", exist_ok=True)

    # Init model
    model = DistributionBucketRegressor(random_state=random_state)

    # Prepare data
    X, y, df_prepared = model.prepare_data(df)

    # Simple hyperparameter optimization if requested
    best_params = None
    if optimize:
        print("\n=== Running Hyperparameter Optimization ===")

        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 300, 500, 700],
            'max_depth': [3, 4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }

        # Create base classifier
        base_model = XGBClassifier(
            objective='multi:softprob',
            eval_metric='rmse',
            random_state=random_state
        )

        # Create KFold splits
        kf_splits = utils.create_grouped_kfold_splits(df_prepared, n_splits=3, random_state=random_state)

        # Run RandomizedSearchCV
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=10,  # Try 10 parameter combinations
            scoring='neg_mean_squared_error',
            cv=[(train, test) for train, test in kf_splits],
            verbose=1,
            random_state=random_state
        )

        # Fit the search (only on training data)
        search.fit(X, y)

        # Get best parameters
        best_params = search.best_params_
        print("\n=== Best Hyperparameters ===")
        for param, value in best_params.items():
            print(f"{param}: {value}")

    # Train the final model with best parameters
    if best_params:
        print("\n=== Training Model with Optimized Hyperparameters ===")
    else:
        print("\n=== Training Model with Default Hyperparameters ===")

    # Get train and test data with predictions
    test_truths, test_preds, test_uncerts, train_truths, train_preds, train_uncerts = model.fit(df,
                                                                                                best_params=best_params)

    # Evaluate performance
    metrics = model.evaluate(test_truths, test_preds, test_uncerts, train_truths, train_preds, train_uncerts)

    # Save metrics
    metrics_df = pd.DataFrame([
        {"Set": "Test", "Metric": "RMSE", "Value": metrics["test"]["rmse"]},
        {"Set": "Test", "Metric": "R2", "Value": metrics["test"]["r2"]},
        {"Set": "Test", "Metric": "Uncertainty", "Value": metrics["test"]["uncertainty"]},
        {"Set": "Train", "Metric": "RMSE", "Value": metrics["train"]["rmse"]},
        {"Set": "Train", "Metric": "R2", "Value": metrics["train"]["r2"]},
        {"Set": "Train", "Metric": "Uncertainty", "Value": metrics["train"]["uncertainty"]},
    ])
    metrics_df.to_csv("output/classification/metrics.csv", index=False)

    print("\n=== Distribution-Based Model Results ===")
    print(metrics_df)

    # Save detailed predictions
    results_df = pd.DataFrame({
        "set": ["test"] * len(test_truths) + ["train"] * len(train_truths),
        "true_r1r2": np.concatenate([test_truths, train_truths]),
        "pred_r1r2": np.concatenate([test_preds, train_preds]),
        "uncertainty": np.concatenate([test_uncerts, train_uncerts])
    })
    results_df.to_csv("output/classification/predictions_detailed.csv", index=False)

    print(f"Saved {len(results_df)} predictions ({len(test_truths)} test, {len(train_truths)} train)")

    return model, metrics


def save_detailed_predictions(predictions, df_prepared, original_df):
    """
    Save detailed prediction results for error analysis
    Ensures method and polymerization_type are included in the output

    Args:
        predictions: Dictionary with prediction results
        df_prepared: DataFrame with prepared data (including buckets)
        original_df: Original DataFrame with all features
    """
    # Get test indices and predicted bucket indices
    test_indices = predictions['test_indices']
    pred_buckets = predictions['test_pred_buckets']
    true_buckets = predictions['test_true_buckets']

    # Create a results DataFrame
    results_df = pd.DataFrame()

    true_buckets_array = np.array(true_buckets)
    pred_buckets_array = np.array(pred_buckets)

    comparison = (true_buckets_array == pred_buckets_array)
    results_df['bucket_correct'] = np.where(comparison, 1, 0)

    # Add r1r2 data from the prepared DataFrame
    if 'r1r2' in df_prepared.columns:
        true_r1r2 = df_prepared.iloc[test_indices]['r1r2'].values
        results_df['true_r1r2'] = true_r1r2

    # Add predicted r1r2 using bucket centers
    bucket_centers = predictions['bucket_centers']
    pred_r1r2 = np.array([bucket_centers[idx] for idx in pred_buckets])
    results_df['pred_r1r2'] = pred_r1r2

    # Calculate errors
    results_df['abs_error'] = np.abs(results_df['true_r1r2'] - results_df['pred_r1r2'])
    results_df['rel_error'] = results_df['abs_error'] / np.maximum(results_df['true_r1r2'], 0.0001) * 100

    # Add important columns from the original DataFrame
    important_columns = [
        'monomer1_name', 'monomer2_name', 'solvent_name',
        'monomer1_smiles', 'monomer2_smiles', 'solvent_smiles',
        'reaction_id', 'reference', 'temperature', 'polymerization_type',
        'method'  # Explicitly include method
    ]

    # Only add columns that exist
    existing_columns = [col for col in important_columns if col in original_df.columns]

    # For each important column, add it to the results DataFrame
    if 'reaction_id' in df_prepared.columns:
        reaction_ids = df_prepared.iloc[test_indices]['reaction_id'].values

        # Add each column if it exists
        for col in existing_columns:
            if col in original_df.columns:
                # Create a mapping of reaction_id to column value
                id_to_value = dict(zip(original_df['reaction_id'], original_df[col]))
                # Map reaction_ids to the corresponding values
                results_df[col] = [id_to_value.get(rid) for rid in reaction_ids]

    # Verify that method and polymerization_type are included if available in the original data
    for col in ['method', 'polymerization_type']:
        if col not in results_df.columns and col in original_df.columns:
            print(f"Warning: Column '{col}' not transferred properly. Adding it manually.")

            # If we have reaction_id, we can still add the missing column
            if 'reaction_id' in results_df.columns:
                id_to_value = dict(zip(original_df['reaction_id'], original_df[col]))
                results_df[col] = [id_to_value.get(rid) for rid in results_df['reaction_id']]

    # Add distribution-based predictions if available
    if 'r1r2_pred_dist' in predictions:
        results_df['pred_r1r2_dist'] = predictions['r1r2_pred_dist']
        # Calculate errors for distribution-based predictions
        results_df['abs_error_dist'] = np.abs(results_df['true_r1r2'] - results_df['pred_r1r2_dist'])
        results_df['rel_error_dist'] = results_df['abs_error_dist'] / np.maximum(results_df['true_r1r2'], 0.0001) * 100

    # Save the results
    results_df.to_csv("output/classification/detailed_predictions.csv", index=False)
    print("Saved detailed predictions to output/classification/detailed_predictions.csv")
    print(f"Columns included in detailed predictions: {results_df.columns.tolist()}")


def run_binary_classification(df, random_state=42):
    """
    Performs binary classification of r_product using grouped K-fold cross-validation.
    Class 0: r_product < 0.01 OR r_product > 100
    Class 1: 0.01 ≤ r_product ≤ 100
    """

    print("=== XGBoost Binary Classification Model for R-Product ===")

    # Create r_product (r1 * r2)
    df['r1r2'] = df['constant_1'] * df['constant_2']

    # Create binary target class
    # Class 0: r_product < 0.01 OR r_product > 100
    # Class 1: 0.01 <= r_product <= 100
    df['r_product_class'] = ((df['r1r2'] >= 0.01) & (df['r1r2'] <= 100)).astype(int)

    # Show class distribution
    class_counts = df['r_product_class'].value_counts()
    print(f"\nClass distribution:")
    print(f"Class 0 (r_product < 0.01 OR > 100): {class_counts[0]} samples ({class_counts[0] / len(df) * 100:.1f}%)")
    print(f"Class 1 (0.01 ≤ r_product ≤ 100): {class_counts[1]} samples ({class_counts[1] / len(df) * 100:.1f}%)")

    # Feature selection
    feature_columns = [
        # Molecular descriptors for Monomer 1
        'best_conformer_energy_1', 'ip_1', 'ip_corrected_1', 'ea_1', 'homo_1', 'lumo_1',
        'global_electrophilicity_1', 'global_nucleophilicity_1', 'charges_min_1', 'charges_max_1',
        'charges_mean_1', 'fukui_electrophilicity_min_1', 'fukui_electrophilicity_max_1',
        'fukui_electrophilicity_mean_1', 'fukui_nucleophilicity_min_1', 'fukui_nucleophilicity_max_1',
        'fukui_nucleophilicity_mean_1', 'fukui_radical_min_1', 'fukui_radical_max_1',
        'fukui_radical_mean_1', 'dipole_x_1', 'dipole_y_1', 'dipole_z_1',

        # Molecular descriptors for Monomer 2
        'best_conformer_energy_2', 'ip_2', 'ip_corrected_2', 'ea_2', 'homo_2', 'lumo_2',
        'global_electrophilicity_2', 'global_nucleophilicity_2', 'charges_min_2', 'charges_max_2',
        'charges_mean_2', 'fukui_electrophilicity_min_2', 'fukui_electrophilicity_max_2',
        'fukui_electrophilicity_mean_2', 'fukui_nucleophilicity_min_2', 'fukui_nucleophilicity_max_2',
        'fukui_nucleophilicity_mean_2', 'fukui_radical_min_2', 'fukui_radical_max_2',
        'fukui_radical_mean_2', 'dipole_x_2', 'dipole_y_2', 'dipole_z_2',

        # HOMO-LUMO differences
        'delta_HOMO_LUMO_AA', 'delta_HOMO_LUMO_AB', 'delta_HOMO_LUMO_BB', 'delta_HOMO_LUMO_BA',

        # Other features
        'temperature', 'solvent_logp',
        'polytype_emb_1', 'polytype_emb_2', 'method_emb_1', 'method_emb_2'
    ]

    # Check available features
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"\nAvailable features: {len(available_features)}")

    # Prepare data
    X = df[available_features].values
    y = df['r_product_class'].values

    # Remove rows with NaN values
    mask = ~(pd.isna(X).any(axis=1) | pd.isna(y))
    X = X[mask]
    y = y[mask]
    df_clean = df[mask].reset_index(drop=True)

    print(f"Final dataset size: {len(X)} samples, {len(available_features)} features")

    # Create grouped K-Fold splits
    n_splits = 5
    kf_splits = utils.create_grouped_kfold_splits(df_clean, n_splits=n_splits, random_state=random_state)
    print(f"\nUsing {n_splits}-fold cross-validation with grouped splits (keeps flipped monomer pairs together)")

    # Get model and parameter grid
    model = xgb.XGBClassifier(random_state=random_state, eval_metric='logloss', use_label_encoder=False)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }

    # Storage for scores and predictions
    fold_scores = []
    all_y_true = []
    all_y_pred = []
    all_y_pred_proba = []
    all_prediction_confidence = []
    all_models = []

    # Cross-validation loop
    for fold, (train_idx, test_idx) in enumerate(kf_splits, 1):
        print(f"\nFold {fold}")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        # Hyperparameter optimization
        random_search = RandomizedSearchCV(
            model, param_distributions=param_grid,
            n_iter=20, cv=3, scoring='f1', verbose=0, random_state=random_state, n_jobs=-1
        )

        # Fit model
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        # Predictions with confidence estimation
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        # Calculate prediction confidence using multiple methods
        confidence_scores = calculate_prediction_confidence(best_model, X_test, y_pred_proba)

        # Calculate fold metrics
        fold_accuracy = accuracy_score(y_test, y_pred)
        fold_precision = precision_score(y_test, y_pred)
        fold_recall = recall_score(y_test, y_pred)
        fold_f1 = f1_score(y_test, y_pred)
        fold_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"  Accuracy: {fold_accuracy:.4f}")
        print(f"  Precision: {fold_precision:.4f}")
        print(f"  Recall: {fold_recall:.4f}")
        print(f"  F1-Score: {fold_f1:.4f}")
        print(f"  AUC: {fold_auc:.4f}")

        # Store results
        fold_scores.append({
            'fold': fold,
            'accuracy': fold_accuracy,
            'precision': fold_precision,
            'recall': fold_recall,
            'f1_score': fold_f1,
            'auc': fold_auc,
            'best_params': random_search.best_params_
        })

        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_pred_proba.extend(y_pred_proba)
        all_prediction_confidence.extend(confidence_scores)
        all_models.append(best_model)

    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    overall_precision = precision_score(all_y_true, all_y_pred)
    overall_recall = recall_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred)
    overall_auc = roc_auc_score(all_y_true, all_y_pred_proba)

    print(f"\n=== Overall Results (5-Fold CV) ===")
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1-Score: {overall_f1:.4f}")
    print(f"Overall AUC: {overall_auc:.4f}")

    # Calculate mean and std across folds
    fold_scores_df = pd.DataFrame(fold_scores)
    print(f"\n=== Cross-Validation Statistics ===")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
        mean_val = fold_scores_df[metric].mean()
        std_val = fold_scores_df[metric].std()
        print(f"{metric.upper()}: {mean_val:.4f} (+/- {std_val * 2:.4f})")

    # Detailed evaluation
    print(f"\n=== Detailed Evaluation ===")
    print("\nClassification Report:")
    print(classification_report(all_y_true, all_y_pred,
                                target_names=['Class 0 (< 0.01 or > 100)', 'Class 1 (0.01-100)']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(cm)

    # Create visualizations with confidence
    create_classification_plots(all_y_true, all_y_pred, all_y_pred_proba, fold_scores_df, all_prediction_confidence)

    # Feature importance from best model
    plot_feature_importance(all_models[0], available_features)

    # Save results with confidence scores
    save_results_with_confidence(fold_scores_df, available_features, class_counts, overall_accuracy, overall_f1,
                                 overall_auc,
                                 all_y_true, all_y_pred, all_y_pred_proba, all_prediction_confidence, df_clean)

    return {
        'fold_scores': fold_scores_df,
        'overall_metrics': {
            'accuracy': overall_accuracy,
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'auc': overall_auc
        },
        'models': all_models,
        'confidence_scores': all_prediction_confidence
    }


def calculate_prediction_confidence(model, X_test, y_pred_proba):
    """
    Calculate prediction confidence using entropy-based uncertainty.
    Lower entropy = higher confidence
    """

    # Get probabilities for both classes
    y_pred_proba_both = model.predict_proba(X_test)

    # Avoid log(0) by clipping probabilities
    epsilon = 1e-15
    y_pred_proba_both = np.clip(y_pred_proba_both, epsilon, 1 - epsilon)

    # Calculate entropy: H = -sum(p * log(p))
    entropy = -np.sum(y_pred_proba_both * np.log(y_pred_proba_both), axis=1)

    # For binary classification, maximum entropy is log(2)
    max_entropy = np.log(2)

    # Convert entropy to confidence: confidence = 1 - (entropy / max_entropy)
    # This gives values from 0 (minimum confidence) to 1 (maximum confidence)
    confidence = 1 - (entropy / max_entropy)

    return confidence


def create_classification_plots(y_true, y_pred, y_pred_proba, fold_scores_df, confidence_scores):
    """Creates visualizations for binary classification with confidence"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix - Overall')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()

    # 3. Confidence Distribution
    y_true_array = np.array(y_true)
    confidence_array = np.array(confidence_scores)
    correct_predictions = (np.array(y_pred) == y_true_array)

    axes[0, 2].hist(confidence_array[correct_predictions], bins=20, alpha=0.7,
                    label='Correct Predictions', density=True, color='green')
    axes[0, 2].hist(confidence_array[~correct_predictions], bins=20, alpha=0.7,
                    label='Incorrect Predictions', density=True, color='red')
    axes[0, 2].set_xlabel('Prediction Confidence')
    axes[0, 2].set_ylabel('Density')
    axes[0, 2].set_title('Confidence Distribution')
    axes[0, 2].legend()

    # 4. Cross-Validation Scores
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    means = [fold_scores_df[metric].mean() for metric in metrics]
    stds = [fold_scores_df[metric].std() for metric in metrics]

    x = np.arange(len(metrics))
    axes[1, 0].bar(x, means, yerr=stds, capsize=5)
    axes[1, 0].set_xlabel('Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Cross-Validation Performance')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([m.upper() for m in metrics])

    # 5. Confidence vs Accuracy
    # Bin predictions by confidence and calculate accuracy in each bin
    confidence_bins = np.linspace(0, 1, 11)
    bin_accuracies = []
    bin_counts = []

    for i in range(len(confidence_bins) - 1):
        mask = (confidence_array >= confidence_bins[i]) & (confidence_array < confidence_bins[i + 1])
        if np.sum(mask) > 0:
            bin_accuracy = np.mean(correct_predictions[mask])
            bin_accuracies.append(bin_accuracy)
            bin_counts.append(np.sum(mask))
        else:
            bin_accuracies.append(0)
            bin_counts.append(0)

    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    axes[1, 1].plot(bin_centers, bin_accuracies, 'o-', label='Actual Accuracy')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Confidence Calibration')
    axes[1, 1].legend()

    # 6. Prediction Probability Distribution with Confidence
    y_pred_proba_array = np.array(y_pred_proba)
    scatter = axes[1, 2].scatter(y_pred_proba_array, confidence_array,
                                 c=correct_predictions, cmap='RdYlGn', alpha=0.6)
    axes[1, 2].set_xlabel('Predicted Probability')
    axes[1, 2].set_ylabel('Confidence Score')
    axes[1, 2].set_title('Probability vs Confidence')
    axes[1, 2].axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=axes[1, 2], label='Correct Prediction')

    plt.tight_layout()
    plt.savefig('output/xgboost_binary_classification_plots.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_results_with_confidence(fold_scores_df, features, class_counts, overall_accuracy, overall_f1, overall_auc,
                                 y_true, y_pred, y_pred_proba, confidence_scores, df_clean):
    """Saves XGBoost binary classification results with confidence scores and all features"""

    # Save fold scores
    fold_scores_df.to_csv('output/xgboost_binary_fold_scores.csv', index=False)

    # Create predictions dataframe with all features
    predictions_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'predicted_probability': y_pred_proba,
        'confidence_score': confidence_scores,
        'correct_prediction': np.array(y_pred) == np.array(y_true)
    })

    # Add all available features from the cleaned dataframe
    # First, ensure we have the same number of rows
    if len(predictions_df) == len(df_clean):
        # Add all columns from df_clean except those that might conflict
        exclude_columns = ['true_label', 'predicted_label', 'predicted_probability', 'confidence_score',
                           'correct_prediction']

        for col in df_clean.columns:
            if col not in exclude_columns and col not in predictions_df.columns:
                predictions_df[col] = df_clean[col].values
    else:
        print(f"Warning: Predictions length ({len(predictions_df)}) doesn't match clean data length ({len(df_clean)})")
        print("Features will not be added to predictions file")

    # Save predictions with all features
    predictions_df.to_csv('output/xgboost_predictions_with_confidence.csv', index=False)

    print(f"Saved predictions with {len(predictions_df.columns)} columns including all features")

    # Calculate confidence statistics
    correct_predictions = predictions_df['correct_prediction']
    high_confidence_mask = predictions_df['confidence_score'] > 0.8
    medium_confidence_mask = (predictions_df['confidence_score'] > 0.6) & (predictions_df['confidence_score'] <= 0.8)
    low_confidence_mask = predictions_df['confidence_score'] <= 0.6

    confidence_stats = {
        'high_confidence_accuracy': correct_predictions[
            high_confidence_mask].mean() if high_confidence_mask.sum() > 0 else 0,
        'medium_confidence_accuracy': correct_predictions[
            medium_confidence_mask].mean() if medium_confidence_mask.sum() > 0 else 0,
        'low_confidence_accuracy': correct_predictions[
            low_confidence_mask].mean() if low_confidence_mask.sum() > 0 else 0,
        'high_confidence_count': high_confidence_mask.sum(),
        'medium_confidence_count': medium_confidence_mask.sum(),
        'low_confidence_count': low_confidence_mask.sum(),
        'mean_confidence': predictions_df['confidence_score'].mean(),
        'std_confidence': predictions_df['confidence_score'].std()
    }

    # Save summary statistics
    summary_stats = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
        summary_stats[f'{metric}_mean'] = fold_scores_df[metric].mean()
        summary_stats[f'{metric}_std'] = fold_scores_df[metric].std()

    # Add confidence statistics
    summary_stats.update(confidence_stats)

    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv('output/xgboost_binary_summary.csv', index=False)

    # Save detailed report
    with open('output/xgboost_binary_classification_report.txt', 'w') as f:
        f.write("XGBoost Binary Classification Report with Confidence\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Classification Task: R-Product Binary Classification\n")
        f.write(f"Class 0: r_product < 0.01 OR r_product > 100\n")
        f.write(f"Class 1: 0.01 ≤ r_product ≤ 100\n\n")
        f.write(f"Class Distribution:\n")
        f.write(f"Class 0: {class_counts[0]} samples ({class_counts[0] / sum(class_counts) * 100:.1f}%)\n")
        f.write(f"Class 1: {class_counts[1]} samples ({class_counts[1] / sum(class_counts) * 100:.1f}%)\n\n")
        f.write(f"Number of Features: {len(features)}\n\n")

        f.write("5-Fold Cross-Validation Results:\n")
        f.write("-" * 30 + "\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
            mean_val = fold_scores_df[metric].mean()
            std_val = fold_scores_df[metric].std()
            f.write(f"{metric.upper()}: {mean_val:.4f} (+/- {std_val * 2:.4f})\n")
        f.write(f"\nOverall Performance:\n")
        f.write(f"Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"F1-Score: {overall_f1:.4f}\n")
        f.write(f"AUC: {overall_auc:.4f}\n\n")

        f.write("Confidence Analysis:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Mean Confidence: {confidence_stats['mean_confidence']:.4f}\n")
        f.write(f"Std Confidence: {confidence_stats['std_confidence']:.4f}\n\n")
        f.write(f"High Confidence (>0.8): {confidence_stats['high_confidence_count']} predictions, "
                f"Accuracy: {confidence_stats['high_confidence_accuracy']:.4f}\n")
        f.write(f"Medium Confidence (0.6-0.8): {confidence_stats['medium_confidence_count']} predictions, "
                f"Accuracy: {confidence_stats['medium_confidence_accuracy']:.4f}\n")
        f.write(f"Low Confidence (≤0.6): {confidence_stats['low_confidence_count']} predictions, "
                f"Accuracy: {confidence_stats['low_confidence_accuracy']:.4f}\n")

    print(f"\nResults saved to:")
    print(f"- output/xgboost_binary_fold_scores.csv")
    print(f"- output/xgboost_predictions_with_confidence.csv (with all features)")
    print(f"- output/xgboost_binary_summary.csv")
    print(f"- output/xgboost_binary_classification_report.txt")
    print(f"- output/xgboost_binary_classification_plots.png")
    print(f"- output/xgboost_feature_importance.png")
    """Creates visualizations for binary classification"""

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix - Overall')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    axes[0, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--')
    axes[0, 1].set_xlabel('False Positive Rate')
    axes[0, 1].set_ylabel('True Positive Rate')
    axes[0, 1].set_title('ROC Curve')
    axes[0, 1].legend()

    # 3. Cross-Validation Scores
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    means = [fold_scores_df[metric].mean() for metric in metrics]
    stds = [fold_scores_df[metric].std() for metric in metrics]

    x = np.arange(len(metrics))
    axes[1, 0].bar(x, means, yerr=stds, capsize=5)
    axes[1, 0].set_xlabel('Metrics')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Cross-Validation Performance')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([m.upper() for m in metrics])

    # 4. Prediction Probability Distribution
    y_true_array = np.array(y_true)
    y_pred_proba_array = np.array(y_pred_proba)
    axes[1, 1].hist(y_pred_proba_array[y_true_array == 0], bins=20, alpha=0.7, label='Class 0 (< 0.01 or > 100)',
                    density=True)
    axes[1, 1].hist(y_pred_proba_array[y_true_array == 1], bins=20, alpha=0.7, label='Class 1 (0.01-100)', density=True)
    axes[1, 1].set_xlabel('Predicted Probability')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Prediction Probability Distribution')
    axes[1, 1].legend()
    axes[1, 1].axvline(x=0.5, color='red', linestyle='--', label='Decision Threshold')

    plt.tight_layout()
    plt.savefig('output/xgboost_binary_classification_plots.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20):
    """Plots feature importance for XGBoost model"""

    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(12, 8))
    plt.title(f'Top {top_n} Feature Importances - XGBoost')
    plt.barh(range(top_n), importance[indices])
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('output/xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_results(fold_scores_df, features, class_counts, overall_accuracy, overall_f1, overall_auc):
    """Saves XGBoost binary classification results to files"""

    # Save fold scores
    fold_scores_df.to_csv('output/xgboost_binary_fold_scores.csv', index=False)

    # Save summary statistics
    summary_stats = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
        summary_stats[f'{metric}_mean'] = fold_scores_df[metric].mean()
        summary_stats[f'{metric}_std'] = fold_scores_df[metric].std()

    summary_df = pd.DataFrame([summary_stats])
    summary_df.to_csv('output/xgboost_binary_summary.csv', index=False)

    # Save detailed report
    with open('output/xgboost_binary_classification_report.txt', 'w') as f:
        f.write("XGBoost Binary Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Classification Task: R-Product Binary Classification\n")
        f.write(f"Class 0: r_product < 0.01 OR r_product > 100\n")
        f.write(f"Class 1: 0.01 ≤ r_product ≤ 100\n\n")
        f.write(f"Class Distribution:\n")
        f.write(f"Class 0: {class_counts[0]} samples ({class_counts[0] / sum(class_counts) * 100:.1f}%)\n")
        f.write(f"Class 1: {class_counts[1]} samples ({class_counts[1] / sum(class_counts) * 100:.1f}%)\n\n")
        f.write(f"Number of Features: {len(features)}\n\n")

        f.write("5-Fold Cross-Validation Results:\n")
        f.write("-" * 30 + "\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc']:
            mean_val = fold_scores_df[metric].mean()
            std_val = fold_scores_df[metric].std()
            f.write(f"{metric.upper()}: {mean_val:.4f} (+/- {std_val * 2:.4f})\n")
        f.write(f"\nOverall Performance:\n")
        f.write(f"Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"F1-Score: {overall_f1:.4f}\n")
        f.write(f"AUC: {overall_auc:.4f}\n")

    print(f"\nResults saved to:")
    print(f"- output/xgboost_binary_fold_scores.csv")
    print(f"- output/xgboost_binary_summary.csv")
    print(f"- output/xgboost_binary_classification_report.txt")
    print(f"- output/xgboost_binary_classification_plots.png")
    print(f"- output/xgboost_feature_importance.png")


# Integration into main function
def main(process_data=True):
    """Main function to run binary classification model"""
    import os

    # Set random seed for reproducibility
    random_state = 42

    # Create output directory
    os.makedirs("output", exist_ok=True)

    print("=== Copolymerization Binary Classification ===")

    if process_data:
        # Load processed data
        df = pd.read_csv("output/processed_data.csv")
    else:
        df = pd.read_csv("output/processed_data.csv")

    # Run Binary Classification model
    print("\nRunning XGBoost Binary Classification model...")
    results = run_binary_classification(df, random_state)

    print("\n=== Modeling Complete ===")
    print("Results saved to output/")

    return results


def run_xgboost(df, random_state):
    """Run XGBoost model and save both train and test predictions for error analysis"""
    # Initialize model trainer
    trainer = models.ModelTrainer(model_type="xgboost", random_state=random_state)

    # Store the original DataFrame before any processing
    original_df = df.copy()

    # Prepare data for modeling
    X, y, features, df_prepared = trainer.prepare_data(df)

    if X is None or y is None:
        print("Error preparing data for XGBoost model.")
        return

    # Train and evaluate model
    fold_scores, predictions = trainer.train_and_evaluate(X, y, df_prepared)

    # Train final model
    final_model = trainer.train_final_model(X, y)

    # Get feature importances
    importance_df = trainer.get_feature_importances(final_model)

    if importance_df is not None:
        importance_df.to_csv("output/xgboost_feature_importances.csv", index=False)

    # Create visualizations
    visualization.plot_model_performance(
        predictions,
        title="XGBoost Model Performance",
        save_path="output/xgboost_performance.png"
    )

    if importance_df is not None:
        visualization.plot_feature_importances(
            importance_df,
            title="XGBoost Feature Importances",
            save_path="output/xgboost_importances.png"
        )

    # MODIFIED: Save both train and test results for error analysis
    # Check if the predictions dictionary contains the necessary keys
    required_keys = ['test_pred', 'test_true', 'train_pred', 'train_true', 'test_indices', 'train_indices']
    missing_keys = [key for key in required_keys if key not in predictions]

    if missing_keys:
        print(f"Missing required keys for complete analysis: {missing_keys}")
        print(f"Available keys in predictions: {list(predictions.keys())}")
        return

    # Create a combined DataFrame for both train and test data
    results_df = pd.DataFrame()

    # Process test data
    test_indices = predictions['test_indices']
    test_predictions = predictions['test_pred']
    test_true_values = predictions['test_true']

    # Process train data
    train_indices = predictions['train_indices']
    train_predictions = predictions['train_pred']
    train_true_values = predictions['train_true']

    # Function to add data to the results DataFrame
    def add_data_to_results(indices, predictions, true_values, data_type, df_prepared, original_df, features):
        temp_df = pd.DataFrame()

        # Add a column to identify train/test data
        temp_df['data_type'] = [data_type] * len(indices)

        # Add input features
        if isinstance(indices, list) and len(indices) > 0:
            indices = np.array(indices)

        if all(feature in df_prepared.columns for feature in features):
            for feature in features:
                temp_df[feature] = df_prepared.iloc[indices][feature].reset_index(drop=True)

        # Ensure method and polymerization_type are copied from the prepared or original DataFrame
        special_columns = ['method', 'polymerization_type']
        for col in special_columns:
            # Check if column exists in df_prepared first
            if col in df_prepared.columns:
                temp_df[col] = df_prepared.iloc[indices][col].reset_index(drop=True)
            # If not in df_prepared but in original_df, we'll need to map by reaction_id later

        # Get identification columns from original DataFrame if possible
        if 'reaction_id' in df_prepared.columns:
            reaction_ids = df_prepared.iloc[indices]['reaction_id'].values

            # Important identification columns - include method and polymerization_type
            important_columns = [
                'monomer1_name', 'monomer2_name', 'solvent_name',
                'monomer1_smiles', 'monomer2_smiles', 'solvent_smiles',
                'reaction_id', 'reference', 'source_filename',
                'method', 'polymerization_type'  # Explicitly include these
            ]

            # Check which columns exist in the original DataFrame
            existing_columns = [col for col in important_columns if col in original_df.columns]

            if existing_columns and 'reaction_id' in existing_columns:
                # For each sample, find the corresponding row in the original DataFrame
                for i, reaction_id in enumerate(reaction_ids):
                    original_row = original_df[original_df['reaction_id'] == reaction_id]

                    if not original_row.empty:
                        for col in existing_columns:
                            if i >= len(temp_df):
                                # Expand temp_df if needed
                                new_row = pd.DataFrame([{col: original_row[col].values[0]}])
                                temp_df = pd.concat([temp_df, new_row], ignore_index=True)
                            else:
                                # Add or update the column value
                                temp_df.at[i, col] = original_row[col].values[0]

        # Add true values
        temp_df['true_r1r2'] = true_values

        # Try to add r12 and r21 if they're in the prepared dataframe
        if 'r12' in df_prepared.columns and 'r21' in df_prepared.columns:
            temp_df['true_r12'] = df_prepared.iloc[indices]['r12'].values
            temp_df['true_r21'] = df_prepared.iloc[indices]['r21'].values
        elif 'constant_1' in df_prepared.columns and 'constant_2' in df_prepared.columns:
            temp_df['true_r12'] = df_prepared.iloc[indices]['constant_1'].values
            temp_df['true_r21'] = df_prepared.iloc[indices]['constant_2'].values

        # Add predictions
        temp_df['pred_r1r2'] = predictions

        # Calculate errors
        temp_df['error_r1r2'] = temp_df['true_r1r2'] - temp_df['pred_r1r2']
        temp_df['abs_error_r1r2'] = np.abs(temp_df['error_r1r2'])
        temp_df['rel_error_r1r2'] = np.abs(temp_df['error_r1r2'] / np.maximum(temp_df['true_r1r2'], 0.0001)) * 100

        # Check if any of our important columns are still missing
        for col in ['method', 'polymerization_type']:
            if col not in temp_df.columns and col in original_df.columns:
                print(f"Warning: Column '{col}' not transferred properly. Adding it manually.")

                # If we have reaction_id, we can still add the missing column
                if 'reaction_id' in temp_df.columns:
                    id_to_value = dict(zip(original_df['reaction_id'], original_df[col]))
                    temp_df[col] = [id_to_value.get(rid) for rid in temp_df['reaction_id']]

        return temp_df

    # Add test data
    test_df = add_data_to_results(test_indices, test_predictions, test_true_values, 'test', df_prepared, original_df,
                                  features)
    print(f"Processed {len(test_df)} test samples")

    # Add train data
    train_df = add_data_to_results(train_indices, train_predictions, train_true_values, 'train', df_prepared,
                                   original_df, features)
    print(f"Processed {len(train_df)} train samples")

    # Combine both DataFrames
    results_df = pd.concat([test_df, train_df], ignore_index=True)

    # Save the results
    results_df.to_csv("output/xgboost_predictions_for_error_analysis.csv", index=False)
    print(
        f"Saved {len(results_df)} samples ({len(test_df)} test, {len(train_df)} train) to output/xgboost_predictions_for_error_analysis.csv")

    # Calculate R² separately for train and test
    # Check for NaN values before calculating metrics
    print("Checking for NaN values...")

    # Filter out NaN values from test data
    test_mask = ~(np.isnan(test_true_values) | np.isnan(test_predictions))
    test_true_filtered = test_true_values[test_mask]
    test_pred_filtered = test_predictions[test_mask]
    print(f"Removed {len(test_true_values) - len(test_true_filtered)} NaN values from test data")

    # Filter out NaN values from train data
    train_mask = ~(np.isnan(train_true_values) | np.isnan(train_predictions))
    train_true_filtered = train_true_values[train_mask]
    train_pred_filtered = train_predictions[train_mask]
    print(f"Removed {len(train_true_values) - len(train_true_filtered)} NaN values from train data")

    # Calculate metrics on filtered data
    if len(test_true_filtered) > 0:
        test_r2 = r2_score(test_true_filtered, test_pred_filtered)
        test_rmse = np.sqrt(mean_squared_error(test_true_filtered, test_pred_filtered))
        print(f"XGBoost Test R² score: {test_r2:.4f}")
        print(f"XGBoost Test RMSE: {test_rmse:.4f}")
    else:
        print("Warning: No valid test data for metric calculation")
        test_r2 = None
        test_rmse = None

    if len(train_true_filtered) > 0:
        train_r2 = r2_score(train_true_filtered, train_pred_filtered)
        train_rmse = np.sqrt(mean_squared_error(train_true_filtered, train_pred_filtered))
        print(f"XGBoost Train R² score: {train_r2:.4f}")
        print(f"XGBoost Train RMSE: {train_rmse:.4f}")
    else:
        print("Warning: No valid train data for metric calculation")
        train_r2 = None
        train_rmse = None

    print(f"XGBoost Avg CV R² score: {predictions['avg_test_r2']:.4f}")

    # Save a summary of metrics
    metrics_summary = pd.DataFrame({
        'Metric': ['R² Score', 'RMSE'],
        'Train': [train_r2, train_rmse],
        'Test': [test_r2, test_rmse],
        'CV_Average': [predictions['avg_test_r2'], None]
    })
    metrics_summary.to_csv("output/xgboost_metrics_summary.csv", index=False)

    # Print the columns included in the output
    print(f"Columns included in the analysis: {results_df.columns.tolist()}")


def plot_predictions(true_values, predicted_values, title, save_path, x_label="True Values",
                     y_label="Predicted Values"):
    """Plot true vs predicted values with a 0-5 range"""
    plt.figure(figsize=(8, 8))

    # Create the scatter plot
    plt.scatter(true_values, predicted_values, alpha=0.6)

    # Add perfect prediction line
    min_val = 0
    max_val = 5
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    # Set limits and labels
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add R² value as text
    r2 = r2_score(true_values, predicted_values)
    plt.text(0.05, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))

    # Save and close
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_training_history(train_losses, val_losses, save_path):
    """Plot training and validation loss history"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_neural_network(df, random_state):
    """Run improved Neural Network model"""
    try:
        # Set PyTorch random seed for reproducibility
        torch.manual_seed(random_state)

        # Train the neural network model with improved settings
        print("Training neural network model...")
        model, metrics, scalers, loss_history = train_simple_nn(
            df,
            epochs=500,  # Increased epochs
            lr=3e-4,  # Adjusted learning rate
            batch_size=64,  # Larger batch size
            random_state=random_state,
            transform_targets=True  # Use PowerTransformer with yeo-johnson
        )

        # Unpack scalers
        feature_scaler, target_transformer = scalers

        # Plot training history
        train_losses, val_losses = loss_history
        plot_training_history(train_losses, val_losses, "output/nn_training_history.png")

        # Save metrics
        metrics_df = pd.DataFrame([{
            'Metric': 'R² Score',
            'R1': metrics['r12_r2'],
            'R2': metrics['r21_r2'],
            'Product': metrics['product_r2'],
            'MSE_R1': metrics['r12_mse'],
            'MSE_R2': metrics['r21_mse'],
            'MSE_Product': metrics['product_mse']
        }])

        metrics_df.to_csv("output/nn_metrics.csv", index=False)

        # Prepare a fresh test loader for plotting
        train_loader, test_loader, input_size, new_feature_scaler, new_target_transformer = prepare_data(
            df, batch_size=128, random_state=random_state, transform_targets=True
        )

        # Get predictions for plotting - use the target_transformer to get original scale values
        true_values, predicted_values = get_predictions_from_model(model, test_loader, new_target_transformer)

        # Plot predictions for each target
        plot_predictions(
            true_values[:, 0],
            predicted_values[:, 0],
            title="Neural Network: R1 Predictions",
            save_path="output/nn_r1_predictions.png",
            x_label="True R1",
            y_label="Predicted R1"
        )

        plot_predictions(
            true_values[:, 1],
            predicted_values[:, 1],
            title="Neural Network: R2 Predictions",
            save_path="output/nn_r2_predictions.png",
            x_label="True R2",
            y_label="Predicted R2"
        )

        plot_predictions(
            true_values[:, 2],
            predicted_values[:, 2],
            title="Neural Network: R Product Predictions",
            save_path="output/nn_rproduct_predictions.png",
            x_label="True R Product",
            y_label="Predicted R Product"
        )

        # Additional plot: Computed R1*R2 vs True R Product
        computed_product = predicted_values[:, 0] * predicted_values[:, 1]
        plot_predictions(
            true_values[:, 2],
            computed_product,
            title="Neural Network: Computed R1*R2 vs True Product",
            save_path="output/nn_computed_product.png",
            x_label="True R Product",
            y_label="Computed R1*R2"
        )

        # Distribution plots
        plt.figure(figsize=(10, 6))
        plt.hist(true_values[:, 0], bins=30, alpha=0.5, label='True R1')
        plt.hist(predicted_values[:, 0], bins=30, alpha=0.5, label='Predicted R1')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of True vs Predicted R1 Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/nn_r1_distribution.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(true_values[:, 1], bins=30, alpha=0.5, label='True R2')
        plt.hist(predicted_values[:, 1], bins=30, alpha=0.5, label='Predicted R2')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of True vs Predicted R2 Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/nn_r2_distribution.png")
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(true_values[:, 2], bins=30, alpha=0.5, label='True Product')
        plt.hist(predicted_values[:, 2], bins=30, alpha=0.5, label='Predicted Product')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of True vs Predicted Product Values')
        plt.legend()
        plt.tight_layout()
        plt.savefig("output/nn_product_distribution.png")
        plt.close()

        # Save model
        try:
            model_info = {
                'model_state': model.state_dict(),
                'feature_scaler': feature_scaler,
                'target_transformer': target_transformer,
                'input_size': model.shared_layers[0].in_features
            }
            torch.save(model_info, "output/nn_model.pt")
            print("Model saved successfully to output/nn_model.pt")
        except Exception as e:
            print(f"Error saving model: {e}")

        avg_r2 = (metrics['r12_r2'] + metrics['r21_r2'] + metrics['product_r2']) / 3
        print(f"Neural Network average R² score: {avg_r2:.4f}")
        print("Prediction plots saved to output/ directory")

    except Exception as e:
        print(f"Error running neural network model: {e}")
        import traceback
        traceback.print_exc()


def run_dual_distance_xgboost(df, random_state):
    """Run Dual Distance XGBoost model with uncertainty estimation

    This function uses the DualDistanceModelTrainer which inherits from ModelTrainer.
    The prepare_data method is inherited from the parent ModelTrainer class.
    """
    # First, ensure you have the proper imports at the top of your file:
    # from copolpredictor.models import ModelTrainer  # Base class
    # Add the DualDistanceModelTrainer class definition to your models.py file,
    # then import it:
    # from copolpredictor.models import DualDistanceModelTrainer

    # For now, if you haven't added it to models.py yet, you can define it here
    # after the ModelTrainer import

    # Initialize dual distance model trainer
    # This inherits all methods from ModelTrainer, including prepare_data
    trainer = DualDistanceModelTrainer(model_type="xgboost", random_state=random_state)

    # Store the original DataFrame before any processing
    original_df = df.copy()

    # Use the inherited prepare_data method from ModelTrainer
    # This method handles all the feature engineering, scaling, and transformation
    X, y, features, df_prepared = trainer.prepare_data(df)

    if X is None or y is None:
        print("Error preparing data for Dual Distance XGBoost model.")
        return

    print(f"Data prepared successfully with {len(features)} features")
    print(f"Transformer: {trainer.transformer}")
    print(f"Target transformer: {trainer.target_transformer}")

    # Train and evaluate dual distance models
    # This uses the custom train_dual_distance_models method from DualDistanceModelTrainer
    results = trainer.train_dual_distance_models(X, y, df_prepared)

    # Train final models on all data
    final_model_0, final_model_1 = trainer.train_final_dual_models(X, y)

    # Create visualizations
    # 1. Performance plot with uncertainty
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Test set predictions
    ax1.scatter(results['test_true'], results['test_pred'],
                c=results['test_uncertainty'], cmap='viridis', alpha=0.6)
    ax1.plot([0, 5], [0, 5], 'r--', lw=2)
    ax1.set_xlabel('True R1R2')
    ax1.set_ylabel('Predicted R1R2')
    ax1.set_title(f'Dual Distance XGBoost Test Performance\nR² = {results["avg_test_r2"]:.4f}')
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)
    cbar1 = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar1.set_label('Uncertainty')

    # Uncertainty vs Error plot
    test_errors = np.abs(results['test_true'] - results['test_pred'])
    ax2.scatter(results['test_uncertainty'], test_errors, alpha=0.6)
    ax2.set_xlabel('Uncertainty')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title(f'Uncertainty vs Error\nCorrelation = {results["uncertainty_error_correlation"]:.4f}')

    # Add trend line
    z = np.polyfit(results['test_uncertainty'], test_errors, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(results['test_uncertainty'].min(), results['test_uncertainty'].max(), 100)
    ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8)

    plt.tight_layout()
    plt.savefig("output/dual_distance_xgboost_performance.png")
    plt.close()

    # 2. Uncertainty distribution
    plt.figure(figsize=(10, 6))
    plt.hist(results['test_uncertainty'], bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(results['avg_uncertainty'], color='red', linestyle='--',
                label=f'Mean = {results["avg_uncertainty"]:.4f}')
    plt.xlabel('Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Uncertainties')
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/dual_distance_uncertainty_distribution.png")
    plt.close()

    # Save detailed results
    results_df = pd.DataFrame()

    # Process test data
    test_indices = results['test_indices']
    train_indices = results['train_indices']

    # Create results for test set
    test_df = pd.DataFrame({
        'data_type': ['test'] * len(results['test_true']),
        'true_r1r2': results['test_true'],
        'pred_r1r2': results['test_pred'],
        'uncertainty': results['test_uncertainty'],
        'abs_error': np.abs(results['test_true'] - results['test_pred']),
        'rel_error': np.abs(results['test_true'] - results['test_pred']) / np.maximum(results['test_true'],
                                                                                      0.0001) * 100
    })

    # Create results for train set
    train_df = pd.DataFrame({
        'data_type': ['train'] * len(results['train_true']),
        'true_r1r2': results['train_true'],
        'pred_r1r2': results['train_pred'],
        'uncertainty': results['train_uncertainty'],
        'abs_error': np.abs(results['train_true'] - results['train_pred']),
        'rel_error': np.abs(results['train_true'] - results['train_pred']) / np.maximum(results['train_true'],
                                                                                        0.0001) * 100
    })

    # Add metadata from original dataframe
    if 'reaction_id' in df_prepared.columns:
        # For test set
        test_reaction_ids = df_prepared.iloc[test_indices]['reaction_id'].values
        train_reaction_ids = df_prepared.iloc[train_indices]['reaction_id'].values

        # Important columns to include
        important_columns = [
            'monomer1_name', 'monomer2_name', 'solvent_name',
            'monomer1_smiles', 'monomer2_smiles', 'solvent_smiles',
            'reaction_id', 'reference', 'temperature',
            'polymerization_type', 'method'
        ]

        # Add columns that exist
        existing_columns = [col for col in important_columns if col in original_df.columns]

        # Map data for test set
        for i, (idx, rid) in enumerate(zip(test_indices, test_reaction_ids)):
            if i < len(test_df):
                original_row = original_df[original_df['reaction_id'] == rid]
                if not original_row.empty:
                    for col in existing_columns:
                        test_df.at[i, col] = original_row[col].values[0]

        # Map data for train set
        for i, (idx, rid) in enumerate(zip(train_indices, train_reaction_ids)):
            if i < len(train_df):
                original_row = original_df[original_df['reaction_id'] == rid]
                if not original_row.empty:
                    for col in existing_columns:
                        train_df.at[i, col] = original_row[col].values[0]

    # Combine train and test results
    results_df = pd.concat([test_df, train_df], ignore_index=True)

    # Save results
    results_df.to_csv("output/dual_distance_xgboost_predictions.csv", index=False)
    print(f"Saved {len(results_df)} predictions to output/dual_distance_xgboost_predictions.csv")

    # Save metrics summary
    metrics_summary = pd.DataFrame({
        'Metric': ['R² Score', 'RMSE', 'Mean Uncertainty', 'Uncertainty-Error Correlation'],
        'Train': [
            results['avg_train_r2'],
            results['avg_train_rmse'],
            np.mean(results['train_uncertainty']),
            '-'
        ],
        'Test': [
            results['avg_test_r2'],
            results['avg_test_rmse'],
            results['avg_uncertainty'],
            results['uncertainty_error_correlation']
        ]
    })
    metrics_summary.to_csv("output/dual_distance_xgboost_metrics.csv", index=False)

    # Analyze high vs low uncertainty predictions
    threshold = np.percentile(results['test_uncertainty'], 75)
    high_unc_mask = results['test_uncertainty'] > threshold
    low_unc_mask = results['test_uncertainty'] <= threshold

    high_unc_errors = test_errors[high_unc_mask]
    low_unc_errors = test_errors[low_unc_mask]

    print("\n=== Uncertainty Analysis ===")
    print(f"High uncertainty threshold (75th percentile): {threshold:.4f}")
    print(f"High uncertainty predictions: {np.sum(high_unc_mask)} samples")
    print(f"  Mean absolute error: {np.mean(high_unc_errors):.4f}")
    print(f"  Std of error: {np.std(high_unc_errors):.4f}")
    print(f"Low uncertainty predictions: {np.sum(low_unc_mask)} samples")
    print(f"  Mean absolute error: {np.mean(low_unc_errors):.4f}")
    print(f"  Std of error: {np.std(low_unc_errors):.4f}")

    return trainer, results


def run_product_vs_individual_xgboost(df, random_state):
    """Run Product vs Individual R1/R2 XGBoost model with uncertainty estimation"""
    # Import the extended class
    # from copolpredictor.models import ProductVsIndividualModelTrainer

    # Initialize model trainer
    trainer = ProductVsIndividualModelTrainer(model_type="xgboost", random_state=random_state)

    # Store the original DataFrame
    original_df = df.copy()

    # Prepare data using inherited method from ModelTrainer
    X, y, features, df_prepared = trainer.prepare_data(df)

    if X is None or y is None:
        print("Error preparing data for Product vs Individual model.")
        return

    # Train and evaluate models
    results = trainer.train_product_vs_individual_models(X, y, df_prepared)

    # Train final models on all data
    final_product_model, final_individual_model = trainer.train_final_models(X, y, df_prepared)

    # Create comprehensive visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Direct Product Model Performance
    ax = axes[0, 0]
    scatter1 = ax.scatter(results['test_true'], results['test_pred_direct'],
                          c=results['test_uncertainty'], cmap='viridis', alpha=0.6, s=30)
    ax.plot([0, 5], [0, 5], 'r--', lw=2)
    ax.set_xlabel('True R1R2')
    ax.set_ylabel('Predicted R1R2 (Direct)')
    ax.set_title(f'Direct Product Model\nR² = {results["avg_test_r2_direct"]:.4f}')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    plt.colorbar(scatter1, ax=ax, label='Uncertainty')

    # 2. Indirect Product Model Performance (R1 × R2)
    ax = axes[0, 1]
    scatter2 = ax.scatter(results['test_true'], results['test_pred_indirect'],
                          c=results['test_uncertainty'], cmap='viridis', alpha=0.6, s=30)
    ax.plot([0, 5], [0, 5], 'r--', lw=2)
    ax.set_xlabel('True R1R2')
    ax.set_ylabel('Predicted R1×R2 (Indirect)')
    ax.set_title(f'Indirect Product Model\nR² = {results["avg_test_r2_indirect"]:.4f}')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    plt.colorbar(scatter2, ax=ax, label='Uncertainty')

    # 3. Final Combined Prediction
    ax = axes[0, 2]
    scatter3 = ax.scatter(results['test_true'], results['test_pred'],
                          c=results['test_uncertainty'], cmap='viridis', alpha=0.6, s=30)
    ax.plot([0, 5], [0, 5], 'r--', lw=2)
    ax.set_xlabel('True R1R2')
    ax.set_ylabel('Final Predicted R1R2')
    ax.set_title(f'Combined Final Model\nR² = {results["avg_test_r2"]:.4f}')
    ax.set_xlim(0, 5)
    ax.set_ylim(0, 5)
    plt.colorbar(scatter3, ax=ax, label='Uncertainty')

    # 4. Model Agreement Plot
    ax = axes[1, 0]
    ax.scatter(results['test_pred_direct'], results['test_pred_indirect'],
               c=results['test_uncertainty'], cmap='plasma', alpha=0.6, s=30)
    ax.plot([0, 5], [0, 5], 'k--', lw=2, label='Perfect Agreement')
    ax.set_xlabel('Direct Product Prediction')
    ax.set_ylabel('Indirect (R1×R2) Prediction')
    ax.set_title('Model Agreement Analysis')
    ax.legend()
    plt.colorbar(ax.collections[0], ax=ax, label='Uncertainty')

    # 5. Uncertainty vs Error
    ax = axes[1, 1]
    test_errors = np.abs(results['test_true'] - results['test_pred'])
    ax.scatter(results['test_uncertainty'], test_errors, alpha=0.6, s=30)
    ax.set_xlabel('Uncertainty')
    ax.set_ylabel('Absolute Error')
    ax.set_title(f'Uncertainty Calibration\nCorr = {results["uncertainty_error_correlation"]:.4f}')

    # Add trend line
    z = np.polyfit(results['test_uncertainty'], test_errors, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(results['test_uncertainty'].min(), results['test_uncertainty'].max(), 100)
    ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, lw=2)

    # 6. Individual R1 and R2 predictions
    ax = axes[1, 2]
    # Get true r1 and r2 values from the prepared dataframe
    test_indices = results['test_indices']
    true_r1 = df_prepared.iloc[test_indices]['constant_1'].values
    true_r2 = df_prepared.iloc[test_indices]['constant_2'].values

    # Create a 2D histogram showing R1 vs R2 prediction quality
    from matplotlib.colors import LogNorm
    h = ax.hist2d(results['test_pred_r1'], results['test_pred_r2'],
                  bins=20, cmap='YlOrRd', norm=LogNorm())
    ax.set_xlabel('Predicted R1')
    ax.set_ylabel('Predicted R2')
    ax.set_title('R1 vs R2 Predictions Distribution')
    plt.colorbar(h[3], ax=ax, label='Count')

    plt.tight_layout()
    plt.savefig("output/product_vs_individual_xgboost_analysis.png", dpi=150)
    plt.close()

    # Additional visualization: R1 and R2 individual performance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # R1 predictions
    ax1.scatter(true_r1, results['test_pred_r1'], alpha=0.6, s=30)
    ax1.plot([0, max(true_r1)], [0, max(true_r1)], 'r--', lw=2)
    ax1.set_xlabel('True R1')
    ax1.set_ylabel('Predicted R1')
    ax1.set_title(f'R1 Predictions\nR² = {r2_score(true_r1, results["test_pred_r1"]):.4f}')
    ax1.grid(True, alpha=0.3)

    # R2 predictions
    ax2.scatter(true_r2, results['test_pred_r2'], alpha=0.6, s=30)
    ax2.plot([0, max(true_r2)], [0, max(true_r2)], 'r--', lw=2)
    ax2.set_xlabel('True R2')
    ax2.set_ylabel('Predicted R2')
    ax2.set_title(f'R2 Predictions\nR² = {r2_score(true_r2, results["test_pred_r2"]):.4f}')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/r1_r2_individual_predictions.png", dpi=150)
    plt.close()

    # Uncertainty distribution analysis
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(results['test_uncertainty'], bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(results['avg_uncertainty'], color='red', linestyle='--',
                label=f'Mean = {results["avg_uncertainty"]:.4f}')
    plt.xlabel('Uncertainty')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Uncertainties')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    # Box plot comparing uncertainties by error quartiles
    error_quartiles = pd.qcut(test_errors, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    uncertainty_by_quartile = [results['test_uncertainty'][error_quartiles == q] for q in ['Q1', 'Q2', 'Q3', 'Q4']]
    plt.boxplot(uncertainty_by_quartile, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    plt.xlabel('Error Quartiles')
    plt.ylabel('Uncertainty')
    plt.title('Uncertainty by Error Magnitude')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("output/product_vs_individual_uncertainty_analysis.png")
    plt.close()

    # Save detailed results
    results_df = pd.DataFrame()

    # Test set results
    test_df = pd.DataFrame({
        'data_type': ['test'] * len(results['test_true']),
        'true_r1r2': results['test_true'],
        'pred_r1r2_final': results['test_pred'],
        'pred_r1r2_direct': results['test_pred_direct'],
        'pred_r1r2_indirect': results['test_pred_indirect'],
        'pred_r1': results['test_pred_r1'],
        'pred_r2': results['test_pred_r2'],
        'uncertainty': results['test_uncertainty'],
        'abs_error': np.abs(results['test_true'] - results['test_pred']),
        'abs_error_direct': np.abs(results['test_true'] - results['test_pred_direct']),
        'abs_error_indirect': np.abs(results['test_true'] - results['test_pred_indirect']),
        'model_disagreement': np.abs(results['test_pred_direct'] - results['test_pred_indirect'])
    })

    # Add relative errors
    test_df['rel_error'] = test_df['abs_error'] / np.maximum(test_df['true_r1r2'], 0.0001) * 100
    test_df['rel_error_direct'] = test_df['abs_error_direct'] / np.maximum(test_df['true_r1r2'], 0.0001) * 100
    test_df['rel_error_indirect'] = test_df['abs_error_indirect'] / np.maximum(test_df['true_r1r2'], 0.0001) * 100

    # Train set results
    train_df = pd.DataFrame({
        'data_type': ['train'] * len(results['train_true']),
        'true_r1r2': results['train_true'],
        'pred_r1r2_final': results['train_pred'],
        'uncertainty': results['train_uncertainty'],
        'abs_error': np.abs(results['train_true'] - results['train_pred']),
        'rel_error': np.abs(results['train_true'] - results['train_pred']) / np.maximum(results['train_true'],
                                                                                        0.0001) * 100
    })

    # Add metadata from original dataframe
    if 'reaction_id' in df_prepared.columns:
        # For test set
        test_reaction_ids = df_prepared.iloc[test_indices]['reaction_id'].values
        train_reaction_ids = df_prepared.iloc[results['train_indices']]['reaction_id'].values

        # Add true r1 and r2 values
        test_df['true_r1'] = df_prepared.iloc[test_indices]['constant_1'].values
        test_df['true_r2'] = df_prepared.iloc[test_indices]['constant_2'].values

        # Important columns to include
        important_columns = [
            'monomer1_name', 'monomer2_name', 'solvent_name',
            'monomer1_smiles', 'monomer2_smiles', 'solvent_smiles',
            'reaction_id', 'reference', 'temperature',
            'polymerization_type', 'method'
        ]

        # Add columns that exist
        existing_columns = [col for col in important_columns if col in original_df.columns]

        # Map data for test set
        for i, rid in enumerate(test_reaction_ids):
            if i < len(test_df):
                original_row = original_df[original_df['reaction_id'] == rid]
                if not original_row.empty:
                    for col in existing_columns:
                        test_df.at[i, col] = original_row[col].values[0]

        # Map data for train set (simplified - fewer columns)
        for i, rid in enumerate(train_reaction_ids):
            if i < len(train_df):
                original_row = original_df[original_df['reaction_id'] == rid]
                if not original_row.empty:
                    train_df.at[i, 'reaction_id'] = rid
                    if 'method' in existing_columns:
                        train_df.at[i, 'method'] = original_row['method'].values[0]
                    if 'polymerization_type' in existing_columns:
                        train_df.at[i, 'polymerization_type'] = original_row['polymerization_type'].values[0]

    # Combine train and test results
    results_df = pd.concat([test_df, train_df], ignore_index=True)

    # Save results
    results_df.to_csv("output/product_vs_individual_xgboost_predictions.csv", index=False)
    print(f"\nSaved {len(results_df)} predictions to output/product_vs_individual_xgboost_predictions.csv")

    # Save metrics summary
    metrics_summary = pd.DataFrame({
        'Model': ['Direct Product', 'Indirect (R1×R2)', 'Final Combined'],
        'Test_R2': [
            results['avg_test_r2_direct'],
            results['avg_test_r2_indirect'],
            results['avg_test_r2']
        ],
        'Description': [
            'Predicts r1r2 directly',
            'Predicts r1 and r2 separately, then multiplies',
            'Weighted combination of both approaches'
        ]
    })

    # Add individual R1 and R2 metrics
    individual_metrics = pd.DataFrame({
        'Model': ['Individual R1', 'Individual R2'],
        'Test_R2': [
            np.mean([s['test_r2_r1'] for s in results['fold_scores']]),
            np.mean([s['test_r2_r2'] for s in results['fold_scores']])
        ],
        'Description': [
            'R1 prediction accuracy',
            'R2 prediction accuracy'
        ]
    })

    metrics_summary = pd.concat([metrics_summary, individual_metrics], ignore_index=True)

    # Add uncertainty metrics
    uncertainty_metrics = pd.DataFrame({
        'Metric': ['Mean Uncertainty', 'Uncertainty-Error Correlation', 'Model Disagreement (Mean)'],
        'Value': [
            results['avg_uncertainty'],
            results['uncertainty_error_correlation'],
            np.mean(np.abs(results['test_pred_direct'] - results['test_pred_indirect']))
        ],
        'Description': [
            'Average prediction uncertainty',
            'How well uncertainty predicts actual errors',
            'Average disagreement between direct and indirect models'
        ]
    })

    metrics_summary.to_csv("output/product_vs_individual_metrics_summary.csv", index=False)
    uncertainty_metrics.to_csv("output/product_vs_individual_uncertainty_metrics.csv", index=False)

    # Analyze which approach works better for different ranges
    print("\n=== Performance Analysis by Value Range ===")
    ranges = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0)]

    for low, high in ranges:
        mask = (results['test_true'] >= low) & (results['test_true'] < high)
        if np.sum(mask) > 0:
            r2_direct = r2_score(results['test_true'][mask], results['test_pred_direct'][mask])
            r2_indirect = r2_score(results['test_true'][mask], results['test_pred_indirect'][mask])
            r2_final = r2_score(results['test_true'][mask], results['test_pred'][mask])

            print(f"\nRange [{low:.1f}, {high:.1f}): {np.sum(mask)} samples")
            print(f"  Direct R²: {r2_direct:.4f}")
            print(f"  Indirect R²: {r2_indirect:.4f}")
            print(f"  Final R²: {r2_final:.4f}")
            print(f"  Better approach: {'Direct' if r2_direct > r2_indirect else 'Indirect'}")

    # Analyze high vs low uncertainty predictions
    threshold = np.percentile(results['test_uncertainty'], 75)
    high_unc_mask = results['test_uncertainty'] > threshold
    low_unc_mask = results['test_uncertainty'] <= threshold

    high_unc_errors = test_errors[high_unc_mask]
    low_unc_errors = test_errors[low_unc_mask]

    print("\n=== Uncertainty-Based Performance Analysis ===")
    print(f"Uncertainty threshold (75th percentile): {threshold:.4f}")
    print(f"\nHigh uncertainty predictions: {np.sum(high_unc_mask)} samples")
    print(f"  Mean absolute error: {np.mean(high_unc_errors):.4f}")
    print(f"  Error std dev: {np.std(high_unc_errors):.4f}")
    print(f"  R² score: {r2_score(results['test_true'][high_unc_mask], results['test_pred'][high_unc_mask]):.4f}")

    print(f"\nLow uncertainty predictions: {np.sum(low_unc_mask)} samples")
    print(f"  Mean absolute error: {np.mean(low_unc_errors):.4f}")
    print(f"  Error std dev: {np.std(low_unc_errors):.4f}")
    print(f"  R² score: {r2_score(results['test_true'][low_unc_mask], results['test_pred'][low_unc_mask]):.4f}")

    return trainer, results


def main(process_data=False):
    """Main function to run both XGBoost regression and bucket classification models"""
    # Input data path
    data_path = "../data_extraction/extracted_reactions.csv"

    # Set random seed for reproducibility
    random_state = 42

    # Create output directory
    #os.makedirs("output", exist_ok=False)

    print("=== Copolymerization Prediction Model ===")

    if process_data:
        # Step 1: Load and preprocess data
        print("\nStep 1: Loading and preprocessing data...")
        df = data_processing.load_and_preprocess_data(data_path)

        if df is None or len(df) == 0:
            print("Error: No data available for modeling.")
            return

        # Save processed data
        df.to_csv("output/processed_data.csv", index=False)
    else:
        df = pd.read_csv("output/processed_data.csv")

    # Step 2: Run XGBoost regression model (as before)
    print("\nStep 2: Running XGBoost regression model...")
    #run_xgboost(df, random_state)

    # Step 2b: Run Dual Distance XGBoost model
    print("\nStep 2b: Running Dual Distance XGBoost model...")
    #run_dual_distance_xgboost(df, random_state)

    # Step 2c: Run Product vs Individual R1/R2 XGBoost model
    print("\nStep 2c: Running Product vs Individual R1/R2 XGBoost model...")
    #run_product_vs_individual_xgboost(df, random_state)

    # Step 3: Run Bucket Classification model
    print("\nStep 3: Running Bucket Classification model...")
    #run_bucket_classifier(df, random_state)

    # Step 4: Run Binary Classification model (NEW)
    print("\nStep 4: Running Binary Classification model...")
    run_binary_classification(df, random_state)

    print("\n=== Modeling Complete ===")
    print("Results saved to output/")


if __name__ == "__main__":
    main()