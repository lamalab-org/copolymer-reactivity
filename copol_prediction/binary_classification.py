import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib

matplotlib.use('Agg')  # Set non-interactive backend

import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
from copolpredictor import models, data_processing, result_visualization as visualization
import copolextractor.utils as utils
from error_analysis import perform_error_analysis


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
    all_y_pred_r_combined = []

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
            n_iter=10, cv=3, scoring='f1', verbose=0, random_state=random_state, n_jobs=-1
        )

        # Fit model
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

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

        df['r1_extreme_class'] = df['constant_1'].apply(lambda x: int(not is_extreme_r(x)))
        df['r2_extreme_class'] = df['constant_2'].apply(lambda x: int(not is_extreme_r(x)))

        # Train separate classifiers for r1 and r2
        results_r_single = run_single_r_classifier_r1_r2(df_clean, available_features, test_idx, random_state)

        # Combine predictions: if either r1 OR r2 is predicted as extreme → overall = extreme
        y_pred_r_single_combined = results_r_single['combined_pred']

        print("\n=== Combined r1/r2 Extreme Classifier ===")
        from sklearn.metrics import classification_report

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

    y_true_r_single = [
        int(is_extreme_r(row['constant_1']) or is_extreme_r(row['constant_2']))
        for _, row in df_clean.iterrows()
    ]

    # Convert to array
    y_true_r_single = np.array(y_true_r_single)
    y_pred_r_single = np.array(all_y_pred_r_combined)

    # Compute scores
    print(f"Accuracy:  {accuracy_score(y_true_r_single, y_pred_r_single):.4f}")
    print(f"Precision: {precision_score(y_true_r_single, y_pred_r_single):.4f}")
    print(f"Recall:    {recall_score(y_true_r_single, y_pred_r_single):.4f}")
    print(f"F1-Score:  {f1_score(y_true_r_single, y_pred_r_single):.4f}")

    from sklearn.metrics import classification_report

    # Recalculate true labels (if not already stored)
    y_true_combined = [
        int(not (is_extreme_r(row['constant_1']) or is_extreme_r(row['constant_2'])))
        for _, row in df_clean.iterrows()
    ]

    print(classification_report(y_true_combined, all_y_pred_r_combined,
                                target_names=["Class 0 (extreme r1 or r2)", "Class 1 (both normal)"]))

    # Overall agreement comparison
    overall_agreement = (np.array(all_y_pred) == np.array(all_y_pred_r_combined)).mean()
    print(f"\n=== Agreement Between Classifiers (Overall) ===")
    print(f"Prediction agreement (r_product vs r_single_extreme): {overall_agreement:.2%}")

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm_compare = confusion_matrix(all_y_pred, all_y_pred_r_combined)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_compare, annot=True, fmt='d', cmap='Blues', xticklabels=['No Extreme (r1/r2)', 'Extreme (r1/r2)'],
                yticklabels=['No Extreme (Product)', 'Extreme (Product)'])
    plt.xlabel('r_single_extreme_prediction')
    plt.ylabel('r_product_prediction')
    plt.title('Confusion Matrix: Product vs Single-r Predictions')
    plt.tight_layout()
    plt.savefig('output/confusion_matrix_product_vs_single.png', dpi=300)
    plt.close()

    # Detailed evaluation
    print(f"\n=== Detailed Evaluation ===")
    print("\nClassification Report:")
    from sklearn.metrics import classification_report

    print(classification_report(all_y_true, all_y_pred,
                                target_names=['Class 0 (< 0.01 or > 100)', 'Class 1 (0.01-100)']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_y_true, all_y_pred)
    print(cm)


    # Copy original product predictions
    all_y_pred_corrected = np.array(all_y_pred).copy()

    # Override rule: if single_r predicts extreme (0) but product prediction is 1 → force to 0
    override_mask = (np.array(all_y_pred_r_combined) == 0) & (np.array(all_y_pred) == 1)
    all_y_pred_corrected[override_mask] = 0

    # Print how many predictions were changed
    num_overridden = override_mask.sum()
    print(f"\nRule-based overrides applied: {num_overridden} predictions corrected to class 0")

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

    print("\n=== Evaluation After Rule-Based Correction ===")
    print(f"Accuracy:  {accuracy_score(all_y_true, all_y_pred_corrected):.4f}")
    print(f"Precision: {precision_score(all_y_true, all_y_pred_corrected):.4f}")
    print(f"Recall:    {recall_score(all_y_true, all_y_pred_corrected):.4f}")
    print(f"F1-Score:  {f1_score(all_y_true, all_y_pred_corrected):.4f}")

    print("\nClassification Report (Corrected):")
    print(classification_report(all_y_true, all_y_pred_corrected,
                                target_names=['Class 0 (<0.01 or >100)', 'Class 1 (0.01–100)']))

    # Delta to original predictions
    delta_f1 = f1_score(all_y_true, all_y_pred_corrected) - f1_score(all_y_true, all_y_pred)
    print(f"\nF1 improvement due to override: {delta_f1:+.4f}")

    print("\nConfusion Matrix after Correction:")
    cm = confusion_matrix(all_y_true, all_y_pred_corrected)
    print(cm)

    # Create visualizations with confidence
    create_classification_plots(all_y_true, all_y_pred_corrected, all_y_pred_proba, fold_scores_df, all_prediction_confidence)

    # Feature importance from best model
    plot_feature_importance(all_models[0], available_features)

    # Save results with confidence scores
    save_results_with_confidence(fold_scores_df, available_features, class_counts, overall_accuracy, overall_f1,
                                 overall_auc,
                                 all_y_true, all_y_pred_corrected, all_y_pred_proba, all_prediction_confidence, df_clean)

    perform_error_analysis(all_y_true, all_y_pred_corrected, all_y_pred_proba, all_prediction_confidence, df_clean,
                                         kf_splits)

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

    print("\n--- r1 Extreme Classifier Report ---")
    print(classification_report(y_r1_test, y_pred_r1, target_names=["Extreme", "Normal"]))
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

    print("\n--- r2 Extreme Classifier Report ---")
    print(classification_report(y_r2_test, y_pred_r2, target_names=["Extreme", "Normal"]))
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


def create_classification_plots(y_true, y_pred, y_pred_proba, fold_scores_df, confidence_scores):
    """Creates visualizations for binary classification with confidence"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    plt.style.use("lamalab.mplstyle")

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], vmin=0, vmax=1000)
    axes[0, 0].set_title('Confusion Matrix - Overall')
    axes[0, 0].set_ylabel('True Label')
    axes[0, 0].set_xlabel('Predicted Label')

    # 3. Confidence Distribution
    y_true_array = np.array(y_true)
    confidence_array = np.array(confidence_scores)
    correct_predictions = (np.array(y_pred) == y_true_array)

    axes[0, 1].hist(confidence_array[correct_predictions], bins=20, alpha=0.7,
                    label='Correct Predictions', density=True, color='#661124')
    axes[0, 1].hist(confidence_array[~correct_predictions], bins=20, alpha=0.7,
                    label='Incorrect Predictions', density=True, color='#194A81')
    axes[0, 1].axvline(x=0.3, color='black', linestyle='--', alpha=0.7, label='Threshold 0.3')
    axes[0, 1].set_xlabel('Prediction Confidence')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Confidence Distribution')
    axes[0, 1].legend()

    # 4. Cross-Validation Scores
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
    means = [fold_scores_df[metric].mean() for metric in metrics]
    stds = [fold_scores_df[metric].std() for metric in metrics]

    x = np.arange(len(metrics))
    axes[1, 0].bar(x, means, yerr=stds, capsize=5, color='#194A81')
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

    plt.tight_layout()
    plt.savefig('xgboost_binary_classification_plots.png', dpi=300, bbox_inches='tight')
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


def perform_integrated_error_analysis(all_y_true, all_y_pred, all_y_pred_proba, all_prediction_confidence, df_clean,
                                      kf_splits):
    """
    Perform error analysis directly with the correct data alignment
    """

    print(f"\n" + "=" * 60)
    print(f"INTEGRATED ERROR ANALYSIS")
    print(f"=" * 60)

    # Create all test indices from K-fold splits to map back to df_clean
    all_test_indices = []
    for fold, (train_idx, test_idx) in enumerate(kf_splits, 1):
        all_test_indices.extend(test_idx)

    # Create the complete predictions dataframe with correct alignment
    predictions_df = pd.DataFrame({
        'true_label': all_y_true,
        'predicted_label': all_y_pred,
        'predicted_probability': all_y_pred_proba,
        'confidence_score': all_prediction_confidence,
        'correct_prediction': np.array(all_y_pred) == np.array(all_y_true)
    })

    # Add features from df_clean using the correct indices
    for col in df_clean.columns:
        if col not in predictions_df.columns:
            # Map the correct rows from df_clean to predictions
            predictions_df[col] = df_clean.iloc[all_test_indices][col].values

    print(f"Total predictions: {len(predictions_df)}")

    # Basic error statistics
    total_predictions = len(predictions_df)
    correct_predictions = predictions_df['correct_prediction'].sum()
    incorrect_predictions = (~predictions_df['correct_prediction']).sum()

    print(f"\nBasic Statistics:")
    print(f"Correct predictions: {correct_predictions} ({correct_predictions / total_predictions * 100:.1f}%)")
    print(f"Incorrect predictions: {incorrect_predictions} ({incorrect_predictions / total_predictions * 100:.1f}%)")

    # Analyze error types
    false_positives = (predictions_df['predicted_label'] == 1) & (predictions_df['true_label'] == 0)
    false_negatives = (predictions_df['predicted_label'] == 0) & (predictions_df['true_label'] == 1)

    print(f"\nError Types:")
    print(f"False Positives (predicted normal, actually extreme): {false_positives.sum()}")
    print(f"False Negatives (predicted extreme, actually normal): {false_negatives.sum()}")

    # Extract all errors
    errors_df = predictions_df[~predictions_df['correct_prediction']].copy()

    if len(errors_df) == 0:
        print("\n🎉 No errors found! Perfect predictions!")
        return

    print(f"\n=== Saving All Errors for Manual Analysis ===")
    print(f"Total errors to analyze: {len(errors_df)}")

    # Sort by confidence (most confident errors first - these are most interesting)
    errors_df = errors_df.sort_values('confidence_score', ascending=False)

    # Save all errors with all features
    errors_df.to_csv('output/all_errors_for_manual_analysis.csv', index=False)
    print(f"✓ Saved all {len(errors_df)} errors to: all_errors_for_manual_analysis.csv")

    # Create a summary of high confidence errors
    high_conf_errors = errors_df[errors_df['confidence_score'] > 0.5]
    if len(high_conf_errors) > 0:
        high_conf_errors.to_csv('output/high_confidence_errors.csv', index=False)
        print(f"✓ High confidence errors (>0.5): {len(high_conf_errors)} - saved to: high_confidence_errors.csv")

    # Create error type specific files
    fp_errors = errors_df[errors_df['predicted_label'] == 1]  # False positives
    fn_errors = errors_df[errors_df['predicted_label'] == 0]  # False negatives

    if len(fp_errors) > 0:
        fp_errors.to_csv('output/false_positive_errors.csv', index=False)
        print(f"✓ False positive errors: {len(fp_errors)} - saved to: false_positive_errors.csv")

    if len(fn_errors) > 0:
        fn_errors.to_csv('output/false_negative_errors.csv', index=False)
        print(f"✓ False negative errors: {len(fn_errors)} - saved to: false_negative_errors.csv")

    # Print some examples of the most confident errors
    print(f"\n=== Most Confident Errors (Top 5) ===")
    print("These are the most problematic - model was very confident but wrong:")

    for i, (_, row) in enumerate(errors_df.head(5).iterrows()):
        error_type = "False Positive" if row['predicted_label'] == 1 else "False Negative"
        print(f"\n{i + 1}. {error_type}")
        print(f"   Confidence: {row['confidence_score']:.4f}")
        print(f"   Predicted: {row['predicted_label']}, True: {row['true_label']}")
        print(f"   Probability: {row['predicted_probability']:.4f}")

        # Only print values that exist and are not NaN
        if 'r1r2' in row and pd.notna(row['r1r2']):
            print(f"   R-product: {row['r1r2']:.6f}")
        if 'monomer1_name' in row and pd.notna(row['monomer1_name']) and 'monomer2_name' in row and pd.notna(
                row['monomer2_name']):
            print(f"   Monomers: {row['monomer1_name']} + {row['monomer2_name']}")
        if 'solvent' in row and pd.notna(row['solvent']):
            print(f"   Solvent: {row['solvent']}")
        if 'temperature' in row and pd.notna(row['temperature']):
            print(f"   Temperature: {row['temperature']}")

    # Simple confidence analysis
    print(f"\n=== Confidence Analysis of Errors ===")
    print(f"Error confidence statistics:")
    print(f"  Mean: {errors_df['confidence_score'].mean():.4f}")
    print(f"  Median: {errors_df['confidence_score'].median():.4f}")
    print(f"  Min: {errors_df['confidence_score'].min():.4f}")
    print(f"  Max: {errors_df['confidence_score'].max():.4f}")

    # Confidence bins
    high_conf = (errors_df['confidence_score'] > 0.7).sum()
    med_conf = ((errors_df['confidence_score'] > 0.3) & (errors_df['confidence_score'] <= 0.7)).sum()
    low_conf = (errors_df['confidence_score'] <= 0.3).sum()

    print(f"  High confidence (>0.7): {high_conf} errors")
    print(f"  Medium confidence (0.3-0.7): {med_conf} errors")
    print(f"  Low confidence (≤0.3): {low_conf} errors")

    # R-product analysis if available
    if 'r1r2' in errors_df.columns:
        print(f"\n=== R-Product Analysis of Errors ===")
        r_values = errors_df['r1r2'].dropna()
        if len(r_values) > 0:
            print(f"R-product statistics for errors:")
            print(f"  Mean: {r_values.mean():.6f}")
            print(f"  Median: {r_values.median():.6f}")
            print(f"  Range: {r_values.min():.6f} - {r_values.max():.6f}")

            # Boundary analysis
            near_lower = (r_values < 0.02).sum()
            near_upper = (r_values > 50).sum()
            extreme_low = (r_values < 0.001).sum()
            extreme_high = (r_values > 1000).sum()

            print(f"  Near boundaries:")
            print(f"    Very low (<0.001): {extreme_low}")
            print(f"    Low boundary (<0.02): {near_lower}")
            print(f"    High boundary (>50): {near_upper}")
            print(f"    Very high (>1000): {extreme_high}")

    # Create a simple summary plot
    create_simple_error_plot(errors_df)

    print(f"\n" + "=" * 60)
    print(f"ERROR ANALYSIS FILES CREATED:")
    print(f"• all_errors_for_manual_analysis.csv - ALL {len(errors_df)} errors with all features")
    if len(high_conf_errors) > 0:
        print(f"• high_confidence_errors.csv - {len(high_conf_errors)} high confidence errors")
    if len(fp_errors) > 0:
        print(f"• false_positive_errors.csv - {len(fp_errors)} false positive errors")
    if len(fn_errors) > 0:
        print(f"• false_negative_errors.csv - {len(fn_errors)} false negative errors")
    print(f"• simple_error_analysis_plot.png - Basic visualization")
    print(f"\nOpen these CSV files in Excel/Python for detailed manual analysis!")
    print(f"=" * 60)


def create_simple_error_plot(errors_df):
    """Create a simple visualization of errors"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Error types
    error_types = ['False Positive', 'False Negative']
    error_counts = [
        (errors_df['predicted_label'] == 1).sum(),  # False positives
        (errors_df['predicted_label'] == 0).sum()  # False negatives
    ]

    axes[0, 0].bar(error_types, error_counts, color=['orange', 'red'], alpha=0.7)
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Error Types')
    for i, count in enumerate(error_counts):
        axes[0, 0].text(i, count + 5, str(count), ha='center', va='bottom')

    # 2. Confidence distribution
    axes[0, 1].hist(errors_df['confidence_score'], bins=20, alpha=0.7, color='red', edgecolor='black')
    axes[0, 1].set_xlabel('Confidence Score')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Confidence Distribution of Errors')

    # 3. R-product distribution (if available)
    if 'r1r2' in errors_df.columns:
        r_values = errors_df['r1r2'].dropna()
        if len(r_values) > 0:
            # Focus on reasonable range
            r_values_clipped = r_values[r_values <= 10]
            axes[1, 0].hist(r_values_clipped, bins=30, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 0].set_xlabel('R-product (clipped at 10)')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('R-product Distribution of Errors')
            axes[1, 0].axvline(x=0.01, color='blue', linestyle='--', alpha=0.7, label='Lower boundary')
            axes[1, 0].legend()

    # 4. Confidence vs R-product (if available)
    if 'r1r2' in errors_df.columns:
        r_values = errors_df['r1r2'].dropna()
        conf_values = errors_df.loc[errors_df['r1r2'].notna(), 'confidence_score']

        if len(r_values) > 0:
            # Plot with reasonable R-product range
            mask = r_values <= 10
            r_plot = r_values[mask]
            conf_plot = conf_values[mask]

            scatter = axes[1, 1].scatter(r_plot, conf_plot, alpha=0.6, c='red', s=20)
            axes[1, 1].set_xlabel('R-product (clipped at 10)')
            axes[1, 1].set_ylabel('Confidence Score')
            axes[1, 1].set_title('Confidence vs R-product')
            axes[1, 1].axvline(x=0.01, color='blue', linestyle='--', alpha=0.5, label='Boundary')
            axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('output/simple_error_analysis_plot.png', dpi=300, bbox_inches='tight')
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

    # Perform comprehensive error analysis directly with the correct data

    return results


if __name__ == "__main__":
    main()