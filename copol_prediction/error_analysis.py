import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib

matplotlib.use('Agg')


def perform_error_analysis(all_y_true, all_y_pred, all_y_pred_proba, all_prediction_confidence, df_clean, kf_splits, detailed_error_analyis):
    """
    Error analysis with data from the Binary Classification
    """

    # Create all test indices from K-fold splits
    all_test_indices = []
    for fold, (train_idx, test_idx) in enumerate(kf_splits, 1):
        all_test_indices.extend(test_idx)

    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'true_label': all_y_true,
        'predicted_label': all_y_pred,
        'predicted_probability': all_y_pred_proba,
        'confidence_score': all_prediction_confidence,
        'correct_prediction': np.array(all_y_pred) == np.array(all_y_true)
    })

    # Add features from df_clean using correct indices
    for col in df_clean.columns:
        if col not in predictions_df.columns:
            predictions_df[col] = df_clean.iloc[all_test_indices][col].values

    print(f"Created predictions dataframe with {len(predictions_df)} predictions")

    # Extract errors
    errors_df = predictions_df[~predictions_df['correct_prediction']].copy()
    errors_df = errors_df.sort_values('confidence_score', ascending=False)

    # Run all analysis functions

    if detailed_error_analyis:
        analyze_confidence_errors(predictions_df)
        analyze_r_product_errors(predictions_df)
        analyze_chemical_errors(predictions_df)
        analyze_confidence_thresholds(predictions_df)
        analyze_threshold_03_detailed(predictions_df)
    else:
        analyze_confidence_thresholds(predictions_df)
        analyze_threshold_03_detailed(predictions_df)

    create_comprehensive_error_plots(predictions_df, errors_df)



def analyze_confidence_errors(df):
    """Analyze relationship between confidence and errors"""

    print(f"\n=== Confidence Analysis ===")

    # Confidence statistics by correctness
    correct_confidence = df[df['correct_prediction']]['confidence_score']
    incorrect_confidence = df[~df['correct_prediction']]['confidence_score']

    print(
        f"Correct predictions - Mean confidence: {correct_confidence.mean():.4f}, Std: {correct_confidence.std():.4f}")
    print(
        f"Incorrect predictions - Mean confidence: {incorrect_confidence.mean():.4f}, Std: {incorrect_confidence.std():.4f}")

    # Low confidence errors
    low_confidence_threshold = 0.5
    low_confidence_errors = df[(~df['correct_prediction']) & (df['confidence_score'] < low_confidence_threshold)]
    high_confidence_errors = df[(~df['correct_prediction']) & (df['confidence_score'] > 0.7)]

    print(f"Low confidence errors (<{low_confidence_threshold}): {len(low_confidence_errors)}")
    print(f"High confidence errors (>0.7): {len(high_confidence_errors)} - These are particularly concerning!")


def analyze_r_product_errors(df):
    """Analyze r_product values where errors occur"""

    print(f"\n=== R-Product Error Analysis ===")

    # Check if r1r2 column exists
    if 'r1r2' not in df.columns:
        print("r1r2 column not found in dataframe. Skipping r_product analysis.")
        return

    # Separate by error types
    false_positives = df[(df['predicted_label'] == 1) & (df['true_label'] == 0)]
    false_negatives = df[(df['predicted_label'] == 0) & (df['true_label'] == 1)]

    print(f"R-product statistics:")
    print(f"False Positives (predicted normal, actually extreme): {len(false_positives)} cases")

    if len(false_positives) > 0:
        fp_r_values = false_positives['r1r2'].dropna()
        print(f"  Valid r_product values: {len(fp_r_values)}")

        if len(fp_r_values) > 0:
            try:
                # Convert to numpy for scalar operations
                fp_array = fp_r_values.values

                mean_val = np.mean(fp_array)
                median_val = np.median(fp_array)
                min_val = np.min(fp_array)
                max_val = np.max(fp_array)

                print(f"  Mean r_product: {mean_val:.6f}")
                print(f"  Median r_product: {median_val:.6f}")
                print(f"  Range: {min_val:.6f} - {max_val:.6f}")

                # Check how close to boundaries
                near_lower = np.sum(fp_array < 0.02)
                near_upper = np.sum(fp_array > 50)
                print(f"  Near lower boundary (<0.02): {near_lower}")
                print(f"  Near upper boundary (>50): {near_upper}")

            except Exception as e:
                print(f"  Error calculating statistics: {e}")
        else:
            print(f"  No valid r_product values found (all NaN)")
    else:
        print(f"  No false positives found")

    print(f"False Negatives (predicted extreme, actually normal): {len(false_negatives)} cases")

    if len(false_negatives) > 0:
        fn_r_values = false_negatives['r1r2'].dropna()
        print(f"  Valid r_product values: {len(fn_r_values)}")

        if len(fn_r_values) > 0:
            try:
                # Convert to numpy for scalar operations
                fn_array = fn_r_values.values

                mean_val = np.mean(fn_array)
                median_val = np.median(fn_array)
                min_val = np.min(fn_array)
                max_val = np.max(fn_array)

                print(f"  Mean r_product: {mean_val:.6f}")
                print(f"  Median r_product: {median_val:.6f}")
                print(f"  Range: {min_val:.6f} - {max_val:.6f}")

                # Check boundary regions
                boundary_lower = np.sum((fn_array >= 0.01) & (fn_array <= 0.05))
                boundary_upper = np.sum((fn_array >= 50) & (fn_array <= 100))
                print(f"  Near lower boundary (0.01-0.05): {boundary_lower}")
                print(f"  Near upper boundary (50-100): {boundary_upper}")

            except Exception as e:
                print(f"  Error calculating statistics: {e}")
        else:
            print(f"  No valid r_product values found (all NaN)")
    else:
        print(f"  No false negatives found")


def analyze_chemical_errors(df):
    """Analyze chemical aspects of errors (monomers, solvents)"""

    print(f"\n=== Chemical Error Analysis ===")

    # Monomer analysis
    if 'monomer1_name' in df.columns and 'monomer2_name' in df.columns:
        incorrect_df = df[~df['correct_prediction']]

        print(f"Most problematic monomer 1:")
        monomer1_errors = incorrect_df['monomer1_name'].value_counts().head(5)
        for monomer, count in monomer1_errors.items():
            total_monomer1 = (df['monomer1_name'] == monomer).sum()
            error_rate = count / total_monomer1 if total_monomer1 > 0 else 0
            print(f"  {monomer}: {count} errors out of {total_monomer1} ({error_rate:.2%})")

        print(f"Most problematic monomer 2:")
        monomer2_errors = incorrect_df['monomer2_name'].value_counts().head(5)
        for monomer, count in monomer2_errors.items():
            total_monomer2 = (df['monomer2_name'] == monomer).sum()
            error_rate = count / total_monomer2 if total_monomer2 > 0 else 0
            print(f"  {monomer}: {count} errors out of {total_monomer2} ({error_rate:.2%})")

    # Solvent analysis
    if 'solvent' in df.columns:
        print(f"Most problematic solvents:")
        solvent_errors = df[~df['correct_prediction']]['solvent'].value_counts().head(5)
        for solvent, count in solvent_errors.items():
            total_solvent = (df['solvent'] == solvent).sum()
            error_rate = count / total_solvent if total_solvent > 0 else 0
            print(f"  {solvent}: {count} errors out of {total_solvent} ({error_rate:.2%})")


def analyze_confidence_thresholds(df):
    """Analyze the trade-off of filtering predictions by confidence threshold"""

    print(f"\n=== Confidence Threshold Analysis ===")

    # Test different confidence thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    results = []

    for threshold in thresholds:
        # Keep only predictions above threshold
        kept_mask = df['confidence_score'] >= threshold
        removed_mask = df['confidence_score'] < threshold

        # Statistics for kept predictions
        kept_predictions = df[kept_mask]

        if len(kept_predictions) > 0:
            kept_accuracy = kept_predictions['correct_prediction'].mean()
        else:
            kept_accuracy = 0

        # Count what we're losing
        total_removed = removed_mask.sum()
        correct_removed = (removed_mask & df['correct_prediction']).sum()
        incorrect_removed = (removed_mask & ~df['correct_prediction']).sum()

        # Count what we're keeping
        total_kept = kept_mask.sum()

        results.append({
            'threshold': threshold,
            'total_kept': total_kept,
            'total_removed': total_removed,
            'correct_removed': correct_removed,
            'incorrect_removed': incorrect_removed,
            'kept_accuracy': kept_accuracy,
            'percent_data_kept': total_kept / len(df) * 100,
            'percent_errors_removed': incorrect_removed / (~df['correct_prediction']).sum() * 100 if (~df[
                'correct_prediction']).sum() > 0 else 0,
            'percent_correct_removed': correct_removed / df['correct_prediction'].sum() * 100 if df[
                                                                                                     'correct_prediction'].sum() > 0 else 0
        })

    # Convert to DataFrame for easier analysis
    results_df = pd.DataFrame(results)

    print(f"Confidence Threshold Trade-off Analysis:")
    print(f"{'Threshold':<10} {'Data Kept':<12} {'Accuracy':<10} {'Errors Removed':<15} {'Correct Lost':<13}")
    print("-" * 70)

    for _, row in results_df.iterrows():
        print(f"{row['threshold']:<10.1f} "
              f"{row['percent_data_kept']:<12.1f}% "
              f"{row['kept_accuracy']:<10.3f} "
              f"{row['percent_errors_removed']:<15.1f}% "
              f"{row['percent_correct_removed']:<13.1f}%")

    # Save results
    results_df.to_csv('output/confidence_threshold_analysis.csv', index=False)


def analyze_threshold_03_detailed(df):
    """Detailed analysis for threshold 0.3: new metrics and detailed error analysis"""

    print(f"\n" + "=" * 60)
    print(f"DETAILED ANALYSIS FOR CONFIDENCE THRESHOLD 0.3")
    print(f"=" * 60)

    # Filter predictions with confidence >= 0.3
    threshold = 0.5
    filtered_df = df[df['confidence_score'] >= threshold].copy()

    print(f"Original dataset: {len(df)} predictions")
    print(f"After filtering (conf ≥ {threshold}): {len(filtered_df)} predictions")
    print(f"Removed: {len(df) - len(filtered_df)} predictions ({(len(df) - len(filtered_df)) / len(df) * 100:.1f}%)")

    if len(filtered_df) == 0:
        print("No predictions left after filtering!")
        return

    # Calculate all metrics for filtered data
    y_true_filtered = filtered_df['true_label']
    y_pred_filtered = filtered_df['predicted_label']
    y_pred_proba_filtered = filtered_df['predicted_probability']

    # Calculate metrics
    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    precision = precision_score(y_true_filtered, y_pred_filtered, zero_division=0)
    recall = recall_score(y_true_filtered, y_pred_filtered, zero_division=0)
    f1 = f1_score(y_true_filtered, y_pred_filtered, zero_division=0)
    auc = roc_auc_score(y_true_filtered, y_pred_proba_filtered) if len(np.unique(y_true_filtered)) > 1 else 0

    # Compare with original metrics
    original_accuracy = accuracy_score(df['true_label'], df['predicted_label'])
    original_precision = precision_score(df['true_label'], df['predicted_label'], zero_division=0)
    original_recall = recall_score(df['true_label'], df['predicted_label'], zero_division=0)
    original_f1 = f1_score(df['true_label'], df['predicted_label'], zero_division=0)
    original_auc = roc_auc_score(df['true_label'], df['predicted_probability']) if len(
        np.unique(df['true_label'])) > 1 else 0

    print(f"\n=== COMPARISON WITH ORIGINAL METRICS ===")
    print(f"{'Metric':<12} {'Original':<10} {'Filtered':<10} {'Change':<10}")
    print("-" * 45)
    print(f"{'Accuracy':<12} {original_accuracy:<10.4f} {accuracy:<10.4f} {accuracy - original_accuracy:<+10.4f}")
    print(f"{'Precision':<12} {original_precision:<10.4f} {precision:<10.4f} {precision - original_precision:<+10.4f}")
    print(f"{'Recall':<12} {original_recall:<10.4f} {recall:<10.4f} {recall - original_recall:<+10.4f}")
    print(f"{'F1-Score':<12} {original_f1:<10.4f} {f1:<10.4f} {f1 - original_f1:<+10.4f}")
    print(f"{'AUC':<12} {original_auc:<10.4f} {auc:<10.4f} {auc - original_auc:<+10.4f}")

    from sklearn.metrics import confusion_matrix

    # Print Confusion Matrix
    cm = confusion_matrix(y_true_filtered, y_pred_filtered)
    print(f"\n=== CONFUSION MATRIX (Filtered, conf ≥ {threshold}) ===")
    print(cm)

    # Save remaining errors after filtering
    errors_df_filtered = filtered_df[~filtered_df['correct_prediction']].copy()
    if len(errors_df_filtered) > 0:
        errors_df_filtered = errors_df_filtered.sort_values('confidence_score', ascending=False)
        errors_df_filtered.to_csv('output/detailed_errors_threshold_03.csv', index=False)
        print(f"\nRemaining errors after filtering: {len(errors_df_filtered)}")
        print(f"Saved to: detailed_errors_threshold_03.csv")


def create_comprehensive_error_plots(predictions_df, errors_df):
    """Create comprehensive error analysis visualizations"""

    fig, axes = plt.subplots(2, 2, figsize=(20, 15))

    # 1. Confidence distribution by correctness
    correct_confidence = predictions_df[predictions_df['correct_prediction']]['confidence_score']
    incorrect_confidence = predictions_df[~predictions_df['correct_prediction']]['confidence_score']

    axes[0, 0].hist(correct_confidence, bins=20, alpha=0.7, label='Correct', density=True, color='#661124')
    axes[0, 0].hist(incorrect_confidence, bins=20, alpha=0.7, label='Incorrect', density=True, color='#194A81')
    axes[0, 0].set_xlabel('Confidence Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Confidence Distribution by Correctness')
    axes[0, 0].legend()

    # 4. Confidence vs Accuracy calibration
    confidence_array = predictions_df['confidence_score'].values
    correct_predictions = predictions_df['correct_prediction'].values

    confidence_bins = np.linspace(0, 1, 11)
    bin_accuracies = []

    for i in range(len(confidence_bins) - 1):
        mask = (confidence_array >= confidence_bins[i]) & (confidence_array < confidence_bins[i + 1])
        if np.sum(mask) > 0:
            bin_accuracy = np.mean(correct_predictions[mask])
            bin_accuracies.append(bin_accuracy)
        else:
            bin_accuracies.append(0)

    bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
    axes[0, 1].plot(bin_centers, bin_accuracies, 'o-', label='Actual Accuracy', color='#194A81')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    axes[0, 1].set_xlabel('Confidence')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Confidence Calibration')
    axes[0, 1].legend()

    # 6. Temperature distribution (if available)
    if 'temperature' in predictions_df.columns:
        temp_correct = predictions_df[predictions_df['correct_prediction']]['temperature'].dropna()
        temp_incorrect = predictions_df[~predictions_df['correct_prediction']]['temperature'].dropna()

        if len(temp_correct) > 0 and len(temp_incorrect) > 0:
            axes[1, 0].hist(temp_correct, bins=20, alpha=0.7, label='Correct', density=True, color='#661124')
            axes[1, 0].hist(temp_incorrect, bins=20, alpha=0.7, label='Incorrect', density=True, color='#194A81')
            axes[1, 0].set_xlabel('Temperature')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Temperature Distribution')
            axes[1, 0].legend()

    # 7. R1 vs R2 scatter (if available)
    if 'constant_1' in predictions_df.columns and 'constant_2' in predictions_df.columns:
        correct_mask = predictions_df['correct_prediction']

        # Plot subset to avoid overcrowding
        n_plot = min(1000, len(predictions_df))
        plot_indices = np.random.choice(len(predictions_df), n_plot, replace=False)

        plot_df = predictions_df.iloc[plot_indices]
        plot_correct = plot_df['correct_prediction']

        axes[1, 1].scatter(plot_df[plot_correct]['constant_1'], plot_df[plot_correct]['constant_2'],
                           alpha=0.8, color='#661124', label='Correct', s=20)
        axes[1, 1].scatter(plot_df[~plot_correct]['constant_1'], plot_df[~plot_correct]['constant_2'],
                           alpha=1, color='#4093C3', label='Incorrect', s=40)
        axes[1, 1].set_xlabel('R1 (constant_1)')
        axes[1, 1].set_ylabel('R2 (constant_2)')
        axes[1, 1].set_title('R1 vs R2 by Correctness')
        axes[1, 1].legend()
        axes[1, 1].set_xlim(0, 5)
        axes[1, 1].set_ylim(0, 5)

    plt.tight_layout()
    plt.savefig('output/error_analysis.png', dpi=300, bbox_inches='tight')

    print(f"Comprehensive error analysis plots saved to: comprehensive_error_analysis_plots.png")
