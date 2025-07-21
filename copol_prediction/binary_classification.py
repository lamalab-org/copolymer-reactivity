import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import matplotlib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from copolpredictor import data_processing
import copolextractor.utils as utils
from error_analysis import perform_error_analysis
from copolpredictor.prediction_utils import feature_columns

matplotlib.use('Agg')  # Set non-interactive backend


def is_extreme_r(val):
    return val <= 0


def run_binary_classification(df, random_state=42):
    """
    Performs binary classification of r_product using grouped K-fold cross-validation.
    Class 0: r_product < 0.01 OR r_product > 100
    Class 1: 0.01 ≤ r_product ≤ 100
    """

    print("=== XGBoost Binary Classification Model for R-Product ===")

    # Create binary target class
    # Class 0: r_product < 0.01 OR r_product > 100
    # Class 1: 0.01 <= r_product <= 100
    df['r_product_class'] = ((df['r1r2'] >= 0.01) & (df['r1r2'] <= 100)).astype(int)

    # Filter by polymerization method
    original_count = len(df)
    df = df[df['polymerization_type'].isin(utils.RADICAL_TYPES)]
    filtered_count = len(df)
    print(f"Filtered datapoints (radical polymerization): {filtered_count}")
    print(f"Removed datapoints (non-radical polymerization): {original_count - filtered_count}")

    # Remove rows where r1r2 is less than 0
    original_count = len(df)
    df = df[df['r1r2'] >= 0]
    print(f"Dropped {original_count - len(df)} rows where r1r2 < 0")

    # Show class distribution
    class_counts = df['r_product_class'].value_counts()
    print(f"\nClass distribution:")
    print(f"Class 0 (r_product < 0.01 OR > 100): {class_counts[0]} samples ({class_counts[0] / len(df) * 100:.1f}%)")
    print(f"Class 1 (0.01 ≤ r_product ≤ 100): {class_counts[1]} samples ({class_counts[1] / len(df) * 100:.1f}%)")

    # Check available features
    available_features = [col for col in feature_columns if col in df.columns]

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
    all_y_pred_r_combined = []

    # Cross-validation loop
    for fold, (train_idx, test_idx) in enumerate(kf_splits, 1):
        print(f"\nFold {fold}")

        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        print(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")

        class_weights = {0: 5, 1: 1}
        sample_weights = np.array([class_weights[label] for label in y_train])

        # Hyperparameter optimization
        random_search = RandomizedSearchCV(
            model, param_distributions=param_grid,
            n_iter=10, cv=3, scoring='f1', verbose=0, random_state=random_state, n_jobs=-1
        )

        # Fit the XGBoost classifier with randomized hyperparameter search
        random_search.fit(X_train, y_train, sample_weight=sample_weights)

        best_model = random_search.best_estimator_
        best_model.fit(X_train, y_train, sample_weight=sample_weights)

        # Predictions with confidence estimation
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        # Calculate prediction confidence using multiple methods
        confidence_scores = calculate_prediction_confidence(best_model, X_test, y_pred_proba)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

        print("Single r-value models:")

        # Training of single r-value classifier
        df['r1_extreme_class'] = df['constant_1'].apply(lambda x: int(not is_extreme_r(x)))
        df['r2_extreme_class'] = df['constant_2'].apply(lambda x: int(not is_extreme_r(x)))

        # Train separate classifiers for r1 and r2
        results_r_single = run_single_r_classifier_r1_r2(df_clean, available_features, test_idx, random_state)

        # Combine predictions: if either r1 OR r2 is predicted as extreme → overall = extreme
        y_pred_r_single_combined = results_r_single['combined_pred']

        print("\n=== Combined r1/r2 Extreme Classifier ===")

        # Compare the predictions from both classifiers
        agreement_mask = (y_pred == y_pred_r_single_combined)
        agreement_ratio = agreement_mask.mean()
        print(f"  Model agreement with r_single_extreme: {agreement_ratio:.2%}")

        # Store for later
        all_y_pred_r_combined.extend(y_pred_r_single_combined)


    # Calculate overall metrics
    overall_accuracy = accuracy_score(all_y_true, all_y_pred)
    overall_precision = precision_score(all_y_true, all_y_pred)
    overall_recall = recall_score(all_y_true, all_y_pred)
    overall_f1 = f1_score(all_y_true, all_y_pred)
    overall_auc = roc_auc_score(all_y_true, all_y_pred_proba)

    # Feature importance from best model
    plot_feature_importance(all_models[0], available_features)

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Comparison of product vs. single value predictions
    cm_compare = confusion_matrix(all_y_pred, all_y_pred_r_combined)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_compare, annot=True, fmt='d', cmap='Blues', xticklabels=['Extreme (r1/r2)', 'No Extreme (r1/r2)'],
                yticklabels=['Extreme (Product)', 'No Extreme (Product)'])
    plt.xlabel('r_single_extreme_prediction')
    plt.ylabel('r_product_prediction')
    plt.title('Confusion Matrix: Product vs Single-r Predictions')
    plt.tight_layout()
    plt.savefig('output/confusion_matrix_product_vs_single.png', dpi=300)
    plt.close()

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

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(cm)

    all_y_pred_corrected = apply_rule_based_override(
        y_pred_product=all_y_pred,
        y_pred_single_r=all_y_pred_r_combined
    )

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    print("\n=== Evaluation After Rule-Based Correction ===")
    print(f"Accuracy:  {accuracy_score(all_y_true, all_y_pred_corrected):.4f}")
    print(f"Precision: {precision_score(all_y_true, all_y_pred_corrected):.4f}")
    print(f"Recall:    {recall_score(all_y_true, all_y_pred_corrected):.4f}")
    print(f"F1-Score:  {f1_score(all_y_true, all_y_pred_corrected):.4f}")

    # Delta to original predictions
    delta_f1 = f1_score(all_y_true, all_y_pred_corrected) - f1_score(all_y_true, all_y_pred)
    print(f"\nF1 improvement due to override: {delta_f1:+.4f}")

    print("\nConfusion Matrix after Correction:")
    cm = confusion_matrix(all_y_true, all_y_pred_corrected)
    print(cm)

    # Save results with confidence scores
    save_results_with_confidence(fold_scores_df, available_features, class_counts, overall_accuracy, overall_f1,
                                 overall_auc,
                                 all_y_true, all_y_pred_corrected, all_y_pred_proba, all_prediction_confidence, df_clean)

    perform_error_analysis(all_y_true, all_y_pred_corrected, all_y_pred_proba, all_prediction_confidence, df_clean,
                                         kf_splits, detailed_error_analyis=True)

    from sklearn.calibration import CalibratedClassifierCV

    print("\n=== Global Calibration on Full Dataset ===")

    # Step 1: Fit final XGBoost model on full data
    # Use average best_params across folds or pick best from first fold
    best_params = fold_scores[0]['best_params']  # or implement a best-of-all strategy

    from sklearn.model_selection import train_test_split
    X_temp, X_val, y_temp, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    final_model = xgb.XGBClassifier(
        **best_params,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )

    # Sample weights (optional)
    sample_weights = np.array([5 if label == 0 else 1 for label in y_train])

    # Train on X_train_final
    final_model.fit(X_train, y_train, sample_weight=sample_weights)

    # Calibrate on holdout set
    calibrated_model = CalibratedClassifierCV(final_model, method='sigmoid', cv=5)
    calibrated_model.fit(X_calib, y_calib)
    from sklearn.calibration import calibration_curve

    y_val_pred = calibrated_model.predict(X_val)
    y_val_proba = calibrated_model.predict_proba(X_val)[:, 1]

    cm = confusion_matrix(y_val, y_val_pred)
    print(f"\n=== CONFUSION MATRIX after Calibration ===")
    print(cm)

    # 2. Calibration curve
    prob_true, prob_pred = calibration_curve(y_val, y_val_pred, n_bins=10)

    # === Plot 1: Calibration Curve ===
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, 'o-', color='blue', label='Calibrated (Global)')
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Global Confidence Calibration (cv=5)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/calibration_curve_global.png", dpi=300)
    plt.close()

    # === Recalculate entropy-based confidence (optional, for consistency)
    calibrated_confidence = calculate_prediction_confidence(calibrated_model, X, y_val_proba)

    y_pred_calibrated_corrected = apply_rule_based_override(
        y_pred_product=y_val_pred,
        y_pred_single_r=all_y_pred_r_combined
    )

    cm = confusion_matrix(y_val, y_pred_calibrated_corrected)
    print(f"\n=== CONFUSION MATRIX after Calibration and correction ===")
    print(cm)

    # === Run error analysis on calibrated model
    print("\n=== Error Analysis: Calibrated Global Model ===")

    perform_error_analysis(
        all_y_true=y_val,
        all_y_pred=y_val_pred,
        all_y_pred_proba=y_val_proba,
        all_prediction_confidence=calibrated_confidence,
        df_clean=df_clean,
        kf_splits=kf_splits,
        detailed_error_analyis=False
    )

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


def apply_rule_based_override(y_pred_product, y_pred_single_r):
    """
    Applies a rule-based override:
    If the product classifier predicts class 1 (not extreme),
    but either r1 or r2 predicts class 0 (extreme), override to class 0.

    Parameters:
    -----------
    y_pred_product : array-like of shape (n_samples,)
        Predictions from the r_product classifier (0 or 1).

    y_pred_single_r : array-like of shape (n_samples,)
        Combined predictions from r1/r2 classifier (0 or 1).
        (Assumes class 0 means 'extreme'.)

    Returns:
    --------
    y_pred_corrected : np.ndarray
        Corrected predictions after applying the override rule.

    num_overridden : int
        Number of predictions that were changed (1 → 0).
    """
    y_pred_product = np.array(y_pred_product)
    y_pred_single_r = np.array(y_pred_single_r)

    y_pred_corrected = y_pred_product.copy()
    #override_mask = (y_pred_single_r == 0) & (y_pred_product == 1)
    #y_pred_corrected[override_mask] = 0
    #num_overridden = override_mask.sum()

    #print(f"\nRule-based overrides applied: {num_overridden} predictions corrected to class 0")

    return y_pred_corrected


def run_single_r_classifier_r1_r2(df, feature_columns, test_idx, random_state=42):
    """
    Trains two separate XGBoost classifiers to predict whether r1 and r2 are in the extreme range,
    using the same test indices as the r_product classifier for consistency.

    Returns predictions for both classifiers and combined output.
    Class 0: value is extreme (<0.01 or >50)
    Class 1: value is normal
    """
    from sklearn.metrics import classification_report, accuracy_score
    import xgboost as xgb

    df = df.copy()

    # Create individual binary labels for r1 and r2
    df['r1_extreme_class'] = df['constant_1'].apply(lambda x: int(not is_extreme_r(x)))  # 1 = normal, 0 = extreme
    df['r2_extreme_class'] = df['constant_2'].apply(lambda x: int(not is_extreme_r(x)))

    # Extract features and clean NaNs
    X = df[feature_columns].values
    mask = ~(pd.isna(X).any(axis=1))
    df_clean = df[mask].reset_index(drop=True)
    X = X[mask]

    # Prepare target labels
    y_r1 = df_clean['r1_extreme_class'].values
    y_r2 = df_clean['r2_extreme_class'].values

    # Split indices
    train_idx = [i for i in range(len(df_clean)) if i not in test_idx]
    X_train, X_test = X[train_idx], X[test_idx]
    y_r1_train, y_r1_test = y_r1[train_idx], y_r1[test_idx]
    y_r2_train, y_r2_test = y_r2[train_idx], y_r2[test_idx]

    # Model and hyperparameter grid
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

    # === Train classifier for r1 ===
    random_search_r1 = RandomizedSearchCV(
        model, param_distributions=param_grid,
        n_iter=10, cv=3, scoring='f1', verbose=0, random_state=random_state, n_jobs=-1
    )
    random_search_r1.fit(X_train, y_r1_train)
    best_model_r1 = random_search_r1.best_estimator_

    y_pred_r1 = best_model_r1.predict(X_test)
    y_proba_r1 = best_model_r1.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_r1_test, y_pred_r1):.4f}")

    # === Train classifier for r2 ===
    random_search_r2 = RandomizedSearchCV(
        model, param_distributions=param_grid,
        n_iter=10, cv=3, scoring='f1', verbose=0, random_state=random_state, n_jobs=-1
    )
    random_search_r2.fit(X_train, y_r2_train)
    best_model_r2 = random_search_r2.best_estimator_

    y_pred_r2 = best_model_r2.predict(X_test)
    y_proba_r2 = best_model_r2.predict_proba(X_test)[:, 1]

    print(f"Accuracy: {accuracy_score(y_r2_test, y_pred_r2):.4f}")

    # === Combine predictions ===
    # Combined logic: only if BOTH are normal → combined is normal (1), else extreme (0)
    y_pred_combined = ((y_pred_r1 == 1) & (y_pred_r2 == 1)).astype(int)

    return {
        'r1_pred': y_pred_r1,
        'r2_pred': y_pred_r2,
        'r1_proba': y_proba_r1,
        'r2_proba': y_proba_r2,
        'combined_pred': y_pred_combined,
        'y_r1_true': y_r1_test,
        'y_r2_true': y_r2_test
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


    print(f"\nResults saved to:")
    print(f"- output/xgboost_binary_fold_scores.csv")
    print(f"- output/xgboost_predictions_with_confidence.csv (with all features)")
    print(f"- output/xgboost_binary_summary.csv")
    print(f"- output/xgboost_feature_importance.png")


def plot_feature_importance(model, feature_names, top_n=20):
    """Plots feature importance for XGBoost model"""

    # Check if model is a CalibratedClassifierCV wrapper
    if hasattr(model, "estimator"):
        model = model.estimator  # Compatible access for scikit-learn ≥0.24

    # Now extract feature importances
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        print("Model has no feature_importances_. Skipping plot.")
        return

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


# Integration into main function
def main(process_data=False):
    """Main function to run binary classification model"""
    import os
    # Input data path
    data_path = "../data_extraction/extracted_reactions.csv"

    # Set random seed for reproducibility
    random_state = 42

    # Create output directory
    os.makedirs("output", exist_ok=True)

    print("=== Copolymerization Binary Classification ===")

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

    # Run Binary Classification model
    print("\nRunning XGBoost Binary Classification model...")
    results = run_binary_classification(df, random_state)

    print("\n=== Modeling Complete ===")
    print("Results saved to output/")

    return results


if __name__ == "__main__":
    main()