from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


matplotlib.use('Agg')


def perform_error_analysis(
    all_y_true,
    all_y_pred,
    all_y_pred_proba,
    all_prediction_confidence,
    df_clean,
    kf_splits=None,
    detailed_error_analyis=False
):
    """
    Perform error analysis for both CV-based and holdout-based evaluation.

    Parameters:
        all_y_true: list or array of true labels
        all_y_pred: list or array of predicted labels
        all_y_pred_proba: list or array of predicted probabilities
        all_prediction_confidence: list or array of confidence scores
        df_clean: cleaned dataframe with full data
        kf_splits: list of (train_idx, test_idx) tuples or None (for holdout)
        detailed_error_analyis: whether to run extended diagnostics
    """

    # Determine indices for mapping predictions to df_clean
    if kf_splits is not None:
        # Cross-validation mode: collect all test indices
        all_test_indices = []
        for fold, (train_idx, test_idx) in enumerate(kf_splits, 1):
            all_test_indices.extend(test_idx)
        df_predictions = df_clean.iloc[all_test_indices].reset_index(drop=True)
    else:
        # Holdout mode: assume df_clean is already aligned with predictions
        df_predictions = df_clean.reset_index(drop=True)

    # Construct predictions DataFrame
    if isinstance(all_y_pred_proba, (list, np.ndarray)) and np.ndim(all_y_pred_proba) == 2:
        top_class_probs = np.max(all_y_pred_proba, axis=1)
    else:
        top_class_probs = all_y_pred_proba

    predicted_class_probas = np.array([
        proba[class_idx] for proba, class_idx in zip(all_y_pred_proba, all_y_pred)
    ])

    predictions_df = pd.DataFrame({
        'true_label': all_y_true,
        'predicted_label': all_y_pred,
        'predicted_probability': predicted_class_probas,
        'confidence_score': np.max(all_y_pred_proba, axis=1),
        'correct_prediction': np.array(all_y_pred) == np.array(all_y_true)
    })

    # Merge in metadata from df_clean
    for col in df_predictions.columns:
        if col not in predictions_df.columns:
            predictions_df[col] = df_predictions[col].values

    print(f"Created predictions dataframe with {len(predictions_df)} predictions")

    # Extract and sort errors
    errors_df = predictions_df[~predictions_df['correct_prediction']].copy()
    errors_df = errors_df.sort_values('predicted_probability', ascending=False)

    all_y_pred_proba = np.array(all_y_pred_proba)

    # Add class-wise probabilities to the DataFrame (optional, for interactive plots)
    n_classes = all_y_pred_proba.shape[1]
    for i in range(n_classes):
        predictions_df[f'prob_class_{i}'] = all_y_pred_proba[:, i]

    # Run analyses
    if detailed_error_analyis:
        analyze_confidence_errors(predictions_df)
        analyze_r_product_errors(predictions_df)
        #analyze_chemical_errors(predictions_df)
        analyze_confidence_thresholds_multiclass(predictions_df)
        plot_predicted_prob_vs_r1r2_by_class(predictions_df)
        interactive_confidence_vs_r1r2(predictions_df)


    else:
        analyze_confidence_thresholds_multiclass(predictions_df)

    create_comprehensive_error_plots(predictions_df, errors_df)


def analyze_confidence_errors(df):
    """Analyze relationship between confidence and errors"""

    print(f"\n=== Confidence Analysis ===")

    # Confidence statistics by correctness
    correct_confidence = df[df['correct_prediction']]['predicted_probability']
    incorrect_confidence = df[~df['correct_prediction']]['predicted_probability']

    print(
        f"Correct predictions - Mean confidence: {correct_confidence.mean():.4f}, Std: {correct_confidence.std():.4f}")
    print(
        f"Incorrect predictions - Mean confidence: {incorrect_confidence.mean():.4f}, Std: {incorrect_confidence.std():.4f}")

    # Low confidence errors
    low_confidence_threshold = 0.5
    low_confidence_errors = df[(~df['correct_prediction']) & (df['predicted_probability'] < low_confidence_threshold)]
    high_confidence_errors = df[(~df['correct_prediction']) & (df['predicted_probability'] > 0.7)]

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
    """Analyze chemical aspects of errors (monomers, solvents), and plot PCA of monomer fingerprints colored by average error rate."""

    print(f"\n=== Chemical Error Analysis ===")

    # Monomer 1: Top 10 monomers by error rate (not absolute count)
    if 'monomer1_name' in df.columns:
        incorrect_df = df[~df['correct_prediction']]

        print(f"\nTop 10 Monomer 1 by Error Rate:")

        # Count total and incorrect predictions per monomer
        total_counts = df['monomer1_name'].value_counts()
        error_counts = incorrect_df['monomer1_name'].value_counts()

        # Combine into a DataFrame
        monomer_error_df = pd.DataFrame({
            'total': total_counts,
            'errors': error_counts
        }).fillna(0)
        monomer_error_df['error_rate'] = monomer_error_df['errors'] / monomer_error_df['total']
        monomer_error_df = monomer_error_df.sort_values('error_rate', ascending=False).head(100)

        for monomer, row in monomer_error_df.iterrows():
            print(f"  {monomer}: {int(row['errors'])} errors out of {int(row['total'])} ({row['error_rate']:.2%})")

    # Solvent error analysis
    if 'solvent' in df.columns:
        print(f"Most problematic solvents:")
        solvent_errors = df[~df['correct_prediction']]['solvent'].value_counts().head(5)
        for solvent, count in solvent_errors.items():
            total_solvent = (df['solvent'] == solvent).sum()
            error_rate = count / total_solvent if total_solvent > 0 else 0
            print(f"  {solvent}: {count} errors out of {total_solvent} ({error_rate:.2%})")

    # PCA on monomer fingerprints
    if 'monomer1_name' in df.columns and 'monomer1_smiles' in df.columns:
        df = df.copy()
        df['error'] = (~df['correct_prediction']).astype(int)

        # Aggregate error rate and count per monomer
        monomer_stats = df.groupby('monomer1_name').agg({
            'error': 'mean',
            'monomer1_smiles': 'first',
            'correct_prediction': 'count'
        }).rename(columns={'error': 'avg_error', 'correct_prediction': 'count'})

        # Generate fingerprints
        fps = []
        colors = []
        sizes = []
        labels = []

        for monomer, row in monomer_stats.iterrows():
            smi = row['monomer1_smiles']
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
                fps.append(np.array(fp))
                colors.append(row['avg_error'])
                sizes.append(row['count'] * 5)  # scale point size
                labels.append(monomer)

        if not fps:
            print("No valid SMILES for monomers – skipping PCA plot.")
            return

        X_fp = np.array(fps)
        X_pca = PCA(n_components=2).fit_transform(X_fp)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=colors,
            cmap='Reds',
            alpha=0.8,
            edgecolor='k'
        )
        plt.colorbar(scatter, label="Avg. Error Rate")
        plt.xlabel("PC1", fontsize=14)
        plt.ylabel("PC2", fontsize=14)
        plt.tight_layout()
        plt.savefig("output/monomer1_pca_error.png", dpi=300)
        plt.close()
        print("Saved: output/monomer1_pca_error.png")
    else:
        print("Missing 'monomer1_name' or 'monomer1_smiles' for PCA plot.")


def analyze_confidence_thresholds_multiclass(df, n_classes=3):
    """Analyze the trade-off of filtering predictions by confidence threshold for multiclass classification."""

    print(f"\n=== Confidence Threshold Analysis (Multiclass) ===")

    thresholds = [round(val, 2) for val in np.arange(0.05, 1.0, 0.05)]
    results = []

    initial_class_counts = df['true_label'].value_counts().to_dict()

    for threshold in thresholds:
        kept_mask = df['confidence_score'] >= threshold
        removed_mask = df['confidence_score'] < threshold
        kept_predictions = df[kept_mask]
        kept_class_counts = kept_predictions['true_label'].value_counts().to_dict()

        kept_accuracy = kept_predictions['correct_prediction'].mean() if len(kept_predictions) > 0 else 0
        total_removed = removed_mask.sum()
        correct_removed = (removed_mask & df['correct_prediction']).sum()
        incorrect_removed = (removed_mask & ~df['correct_prediction']).sum()
        total_kept = kept_mask.sum()

        result = {
            'threshold': threshold,
            'total_kept': total_kept,
            'kept_accuracy': kept_accuracy,
            'total_removed': total_removed,
            'correct_removed': correct_removed,
            'incorrect_removed': incorrect_removed,
            'percent_data_kept': total_kept / len(df) * 100,
            'percent_errors_removed': incorrect_removed / (~df['correct_prediction']).sum() * 100 if (~df['correct_prediction']).sum() > 0 else 0,
            'percent_correct_removed': correct_removed / df['correct_prediction'].sum() * 100 if df['correct_prediction'].sum() > 0 else 0
        }

        for cls in range(n_classes):
            kept_cls = kept_class_counts.get(cls, 0)
            total_cls = initial_class_counts.get(cls, 0)
            result[f'kept_class_{cls}'] = kept_cls
            result[f'percent_kept_class_{cls}'] = (kept_cls / total_cls * 100) if total_cls > 0 else 0

        results.append(result)

    results_df = pd.DataFrame(results)

    # Print header
    class_headers = " ".join([f"{'K_C'+str(c):<8}" for c in range(n_classes)])
    percent_headers = " ".join([f"%C{c}" for c in range(n_classes)])
    header = (
        f"{'Threshold':<10} {'Data Kept':<10} {'Accuracy':<10} {'Err Rem%':<10} {'Corr Lost%':<11}"
        f"{class_headers} {percent_headers}"
    )
    print(header)
    print("-" * len(header))

    for _, row in results_df.iterrows():
        line = (
            f"{row['threshold']:<10.2f} {row['percent_data_kept']:<10.1f} {row['kept_accuracy']:<10.3f} "
            f"{row['percent_errors_removed']:<10.1f} {row['percent_correct_removed']:<11.1f}"
        )
        line += " " + " ".join([f"{int(row[f'kept_class_{c}']):<8}" for c in range(n_classes)])
        line += " " + " ".join([f"{row[f'percent_kept_class_{c}']:.1f}%" for c in range(n_classes)])
        print(line)

    # Threshold selection
    conditions = [results_df[f'percent_kept_class_{c}'] >= 70 for c in range(n_classes)]
    valid_thresholds_df = results_df[np.logical_and.reduce(conditions)]

    if not valid_thresholds_df.empty:
        recommended_threshold = valid_thresholds_df['threshold'].max()
        print(f"\nUsing threshold = {recommended_threshold} (retains ≥70% of samples from each class)")
    else:
        recommended_threshold = thresholds[0]
        print(f"\nWARNING: No threshold could retain ≥70% of samples from all classes.")
        print(f"Defaulting to lowest threshold = {recommended_threshold}")

    filtered_df = df[df['confidence_score'] >= recommended_threshold].copy()
    print(f"\nOriginal dataset: {len(df)} predictions")
    print(f"After filtering (conf ≥ {recommended_threshold}): {len(filtered_df)} predictions")
    print(f"Removed: {len(df) - len(filtered_df)} predictions "
          f"({(len(df) - len(filtered_df)) / len(df) * 100:.1f}%)")

    # Confusion Matrix
    cm = confusion_matrix(filtered_df['true_label'], filtered_df['predicted_label'])
    print(f"\n=== CONFUSION MATRIX (Filtered, conf ≥ {recommended_threshold}) ===")
    print(cm)

    # Accuracy per class: diagonal / row sum
    acc_per_class = cm.diagonal() / cm.sum(axis=1)

    # Weighted macro accuracy (equal class weights)
    weighted_acc = (acc_per_class * np.array([1 / 3, 1 / 3, 1 / 3])).sum()

    # Print class-wise accuracy
    for i, acc in enumerate(acc_per_class):
        print(f"Class {i} Accuracy: {acc:.4f}")

    print(f"\nWeighted Accuracy (macro average): {weighted_acc:.4f}")

    # Save remaining errors after filtering
    errors_df_filtered = filtered_df[~filtered_df['correct_prediction']].copy()
    if len(errors_df_filtered) > 0:
        errors_df_filtered = errors_df_filtered.sort_values('predicted_probability', ascending=False)
        errors_df_filtered.to_csv(f'output/detailed_errors_threshold_{recommended_threshold:.1f}.csv', index=False)
        print(f"\nRemaining errors after filtering: {len(errors_df_filtered)}")
        print(f"Saved to: detailed_errors_threshold_{recommended_threshold:.1f}.csv")

    return results_df, recommended_threshold, cm


def create_comprehensive_error_plots(predictions_df, errors_df):
    """Multiclass error visualization incl. calibration curves and confusion matrix"""

    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import confusion_matrix
    import numpy as np

    n_classes = len(np.unique(predictions_df['true_label']))

    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    cmap = plt.get_cmap('tab10')

    # === 1. Confidence Distribution by Correctness ===
    correct_conf = predictions_df[predictions_df['correct_prediction']]['confidence_score']
    incorrect_conf = predictions_df[~predictions_df['correct_prediction']]['confidence_score']

    axes[0, 0].hist(correct_conf, bins=20, alpha=0.7, label='Correct', density=True, color='#009688')
    axes[0, 0].hist(incorrect_conf, bins=20, alpha=0.7, label='Incorrect', density=True, color='#e91e63')
    axes[0, 0].set_title('Confidence Distribution')
    axes[0, 0].set_xlabel('Max Confidence (across classes)')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].legend()

    # === 2. Confidence vs True r1r2 + trendline ===
    if 'confidence_score' in predictions_df.columns and 'r1r2' in predictions_df.columns:
        df_trend = predictions_df.copy()
        df_trend = df_trend[df_trend['r1r2'].notna()]
        df_trend['r1r2_bin'] = pd.cut(df_trend['r1r2'], bins=np.linspace(0, 2, 11), include_lowest=True)

        binned = df_trend.groupby('r1r2_bin').agg(
            mean_r1r2=('r1r2', 'mean'),
            mean_conf=('confidence_score', 'mean')
        ).reset_index()

        axes[0, 1].scatter(df_trend['r1r2'], df_trend['confidence_score'], alpha=0.3, label='Samples', s=15, color='#2266ac')
        axes[0, 1].plot(binned['mean_r1r2'], binned['mean_conf'], color='#661124', marker='o', label='Binned Trend')
        axes[0, 1].set_xlim(0, 2)
        axes[0, 1].set_ylim(0, 1.05)
        axes[0, 1].set_xlabel('True r1r2', fontsize=14)
        axes[0, 1].set_ylabel('Confidence', fontsize=14)
        axes[0, 1].tick_params(labelsize=12)
        axes[0, 1].legend(fontsize=12)
        axes[0, 1].set_title("Confidence vs. True r1r2", fontsize=16)

    # === 3. Confusion Matrix ===
    y_true = predictions_df["true_label"]
    y_pred = predictions_df["predicted_label"]
    cm = confusion_matrix(y_true, y_pred, labels=range(n_classes))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], vmax=cm.max() * 0.5, annot_kws={"size": 14})
    axes[1, 0].set_title("Confusion Matrix")
    axes[1, 0].set_xlabel("Predicted Label")
    axes[1, 0].set_ylabel("True Label")

    # === 4. R1 vs R2 Scatterplot (if present) ===
    if 'constant_1' in predictions_df.columns and 'constant_2' in predictions_df.columns:
        n_plot = min(1000, len(predictions_df))
        plot_df = predictions_df.sample(n=n_plot, random_state=42)
        correct = plot_df['correct_prediction']

        axes[1, 1].scatter(plot_df[correct]['constant_1'], plot_df[correct]['constant_2'],
                           alpha=0.8, color='#2266ac', label='Correct', s=30)
        axes[1, 1].scatter(plot_df[~correct]['constant_1'], plot_df[~correct]['constant_2'],
                           alpha=0.9, color='#661124', label='Incorrect', s=40)
        axes[1, 1].set_xlabel('r1', fontsize=16)
        axes[1, 1].set_ylabel('r2',fontsize=16)
        axes[1, 1].tick_params(axis='x', labelsize=14)
        axes[1, 1].tick_params(axis='y', labelsize=14)

        axes[1, 1].set_xlim(0, 5)
        axes[1, 1].set_ylim(0, 5)
        axes[1, 1].legend(fontsize=14)
    else:
        axes[1, 1].text(0.5, 0.5, "No R1/R2 data available", ha='center', va='center')
        axes[1, 1].set_axis_off()

    plt.tight_layout()
    plt.savefig('output/error_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved: output/error_analysis.png")


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_predicted_prob_vs_r1r2_by_class(predictions_df, class_labels=[0, 1, 2]):
    """
    Plots the predicted probability for each class vs. true r1r2 value (actual reaction product),
    with a separate plot for each predicted class.

    Parameters:
        predictions_df: DataFrame that must contain columns:
                        - 'predicted_probability': prob of predicted class (already extracted)
                        - 'predicted_label'
                        - 'r1r2' (true product)
        class_labels: list of class indices to plot
        log_r1r2: whether to use log10 scale for r1r2 axis
    """
    for cls in class_labels:
        df_cls = predictions_df[predictions_df['predicted_label'] == cls].copy()

        if df_cls.empty:
            print(f"No predictions for class {cls}")
            continue

        x = df_cls['r1r2']
        y = df_cls['predicted_probability']


        plt.figure(figsize=(8, 5))
        sns.scatterplot(x=x, y=y, alpha=0.5, edgecolor=None, s=40)
        sns.regplot(x=x, y=y, scatter=False, lowess=True, color='red')

        plt.title(f"Predicted Probability vs r1r2 for Class {cls}")
        plt.xlabel("r1r2")
        plt.xlim(0, 30)
        plt.ylabel("Predicted Probability for Class")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"output/predicted_prob_vs_r1r2_class_{cls}.png")
        plt.show()


import plotly.express as px
import pandas as pd

def interactive_confidence_vs_r1r2(predictions_df, proba_columns_prefix="prob_class_"):
    """
    Create an interactive scatter plot of confidence vs r1r2, colored by predicted class.
    Tooltip shows class probabilities and true/pred labels.
    """

    # Check required columns
    required = ['r1r2', 'confidence_score', 'predicted_label', 'true_label']
    for col in required:
        if col not in predictions_df.columns:
            raise ValueError(f"Missing column: {col}")

    # Add class-wise probabilities (optional: rename them if needed)
    proba_cols = [col for col in predictions_df.columns if col.startswith(proba_columns_prefix)]
    if not proba_cols:
        raise ValueError("No class probability columns found.")

    # Build hover text
    predictions_df['hover_text'] = (
        "True Label: " + predictions_df['true_label'].astype(str) +
        "<br>Predicted: " + predictions_df['predicted_label'].astype(str) +
        "<br>r1r2: " + predictions_df['r1r2'].round(4).astype(str)
    )

    for col in proba_cols:
        predictions_df['hover_text'] += f"<br>{col}: " + predictions_df[col].round(3).astype(str)

    # Create plot
    fig = px.scatter(
        predictions_df,
        x='r1r2',
        y='confidence_score',
        color='predicted_label',
        hover_name='hover_text',
        labels={
            'r1r2': 'True r1r2',
            'confidence_score': 'Confidence',
            'predicted_label': 'Predicted Class'
        },
        color_discrete_sequence=px.colors.qualitative.Set1,
        opacity=0.7
    )

    fig.update_layout(
        title="Interactive: Confidence vs True r1r2 (Colored by Predicted Class)",
        xaxis=dict(title="True r1r2", range=[0, 30]),
        yaxis=dict(title="Prediction Confidence", range=[0, 1.05]),
        font=dict(size=14)
    )

    fig.show()

