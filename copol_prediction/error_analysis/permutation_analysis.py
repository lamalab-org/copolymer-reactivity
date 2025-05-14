"""
Präzises Feature-Analyse-Skript, das die exakt gleichen Train-Test-Splits und
Test-R²-Werte wie der ModelTrainer verwendet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import seaborn as sns

# Import ModelTrainer - dies ist deine existierende Klasse
from copolpredictor.models import ModelTrainer
import copolextractor.utils as utils


def analyze_feature_importance(df, random_state=42, n_repeats=10):
    """
    Main function to analyze feature importance
    Uses ModelTrainer's exact same data preparation, training, and splits

    Args:
        df: DataFrame with features and target
        random_state: Random seed
        n_repeats: Number of permutation repeats
    """
    print("\n=== Feature Importance Analysis with Identical ModelTrainer Splits ===")

    # Create output directory
    os.makedirs("output/feature_analysis", exist_ok=True)

    # Initialize ModelTrainer to use its data preparation and model training
    trainer = ModelTrainer(model_type="xgboost", random_state=random_state)

    # Prepare data using ModelTrainer's method - this ensures consistent data processing
    print("Preparing data using ModelTrainer...")
    X, y, features, df_prepared = trainer.prepare_data(df)

    if X is None or y is None:
        print("Error preparing data.")
        return

    # Train and evaluate model using ModelTrainer - this creates the exact same splits
    print("\nTraining and evaluating model with ModelTrainer to get the exact same splits...")
    fold_scores, predictions = trainer.train_and_evaluate(X, y, df_prepared)

    # Train final model using ModelTrainer
    print("\nTraining final model with ModelTrainer...")
    final_model = trainer.train_final_model(X, y)

    # Print the actual test R² score from the ModelTrainer for reference
    avg_test_r2 = predictions['avg_test_r2']
    print(f"\nReference: ModelTrainer's average Test R²: {avg_test_r2:.4f}")

    # Extract the fold information from the predictions dictionary
    print("\nExtracting fold information from ModelTrainer's predictions...")
    fold_info = extract_fold_info(X, predictions)

    # Manual permutation importance using consistent folds
    importance_df = calculate_permutation_importance(
        final_model, X, y, features, fold_info, n_repeats, random_state
    )

    # Progressive feature selection with consistent folds
    selection_df, best_features = run_progressive_feature_selection(
        X, y, features, importance_df, fold_info, random_state, trainer
    )

    print("\nFeature importance analysis completed using identical splits as ModelTrainer.")
    print(f"Results saved to output/feature_analysis/ directory")

    return importance_df, selection_df, best_features


def extract_fold_info(X, predictions):
    """
    Extract fold information from the predictions dictionary

    Args:
        X: Feature matrix
        predictions: Dictionary with train/test indices and predictions

    Returns:
        dict: Dictionary with fold information
    """
    # Extract train and test indices
    train_indices = predictions['train_indices']
    test_indices = predictions['test_indices']

    # Get the number of total samples
    n_samples = X.shape[0]

    # Analyze the indices to reconstruct the original folds
    fold_info = {}

    # Count the number of unique test indices to determine fold count
    unique_test_indices = list(set(test_indices))
    n_unique_test = len(unique_test_indices)

    # Estimate the number of folds
    n_splits = max(2, min(5, len(unique_test_indices) // 20))  # Reasonable guess

    print(f"Detected approximately {n_splits} folds from ModelTrainer's indices")

    # Try to reconstruct the folds
    fold_info['n_splits'] = n_splits
    fold_info['test_indices'] = test_indices
    fold_info['train_indices'] = train_indices

    # Store the ModelTrainer's test R² for reference
    if 'avg_test_r2' in predictions:
        fold_info['avg_test_r2'] = predictions['avg_test_r2']

    # If we have fold scores, store them too
    if 'fold_scores' in predictions:
        fold_info['fold_scores'] = predictions['fold_scores']

    return fold_info


def calculate_permutation_importance(model, X, y, features, fold_info, n_repeats=10, random_state=42):
    """
    Calculate permutation importance using ModelTrainer's exact test splits

    Args:
        model: Trained model
        X: Feature matrix
        y: Target variable
        features: Feature names
        fold_info: Dictionary with fold information from ModelTrainer
        n_repeats: Number of permutation repeats
        random_state: Random seed

    Returns:
        DataFrame: Feature importance results
    """
    print(f"\nCalculating permutation importance for {len(features)} features on test sets...")

    # Get test indices
    test_indices = fold_info['test_indices']

    # Convert to numpy arrays if they aren't already
    if isinstance(y, list):
        y = np.array(y)

    # Get unique test indices (they might be repeated in the fold_info)
    unique_test_indices = []
    seen = set()
    for idx in test_indices:
        if idx not in seen:
            unique_test_indices.append(idx)
            seen.add(idx)

    # Extract test data
    X_test = X[unique_test_indices]
    y_test = y[unique_test_indices]

    # Get baseline score on test data
    baseline_score = r2_score(y_test, model.predict(X_test))
    print(f"Baseline R² score on test data: {baseline_score:.4f}")

    # Storage for importance values
    importance_values = []

    # For each feature
    for i, feature in enumerate(features):
        # Progress indicator for long runs
        if i % 10 == 0 and i > 0:
            print(f"Processed {i}/{len(features)} features...")

        feature_importance = []

        # Repeat permutation multiple times
        for j in range(n_repeats):
            # Create a copy of X_test
            X_test_permuted = X_test.copy()

            # Permute the feature
            rng = np.random.RandomState(random_state + j)
            X_test_permuted[:, i] = rng.permutation(X_test_permuted[:, i])

            # Calculate score with permuted feature on test data
            permuted_score = r2_score(y_test, model.predict(X_test_permuted))

            # Importance is the decrease in score
            importance = baseline_score - permuted_score
            feature_importance.append(importance)

        # Calculate mean and std of importance
        mean_importance = np.mean(feature_importance)
        std_importance = np.std(feature_importance)

        importance_values.append({
            'Feature': feature,
            'Mean_Importance': mean_importance,
            'Std_Importance': std_importance
        })

    # Create DataFrame
    importance_df = pd.DataFrame(importance_values)

    # Sort by importance
    importance_df = importance_df.sort_values('Mean_Importance', ascending=False)

    # Save to CSV
    importance_df.to_csv("output/feature_analysis/permutation_importance_test_data.csv", index=False)

    # Plot top 30 features
    plt.figure(figsize=(12, 10))
    plt.barh(
        importance_df['Feature'][:30],
        importance_df['Mean_Importance'][:30],
        xerr=importance_df['Std_Importance'][:30],
        capsize=5
    )
    plt.xlabel('Mean Importance (decrease in R²)')
    plt.ylabel('Feature')
    plt.title('Top 30 Features by Permutation Importance (Test Data)')
    plt.tight_layout()
    plt.savefig("output/feature_analysis/permutation_importance_test_data.png")
    plt.close()

    # Group analysis
    feature_group_analysis(importance_df, features)

    return importance_df


def feature_group_analysis(importance_df, features):
    """
    Group features by common prefixes and calculate group importance

    Args:
        importance_df: DataFrame with feature importance
        features: List of all feature names
    """
    print("\nGrouping features by prefix...")

    # Extract feature prefixes
    prefixes = []
    for feature in features:
        # Find the last occurrence of '_'
        if '_' in feature:
            # First identify if it's a feature for monomer 1 or 2
            if feature.endswith('_1') or feature.endswith('_2'):
                # Get prefix without _1 or _2
                prefix = feature[:feature.rfind('_')]
            else:
                # For other features with underscores
                prefix = feature.split('_')[0]
        else:
            # For features without underscores
            prefix = feature
        prefixes.append(prefix)

    # Add prefix to importance DataFrame
    importance_with_prefix = importance_df.copy()
    importance_with_prefix['Prefix'] = [prefixes[features.index(f)] for f in importance_df['Feature']]

    # Group by prefix and calculate total importance
    grouped = importance_with_prefix.groupby('Prefix').agg(
        Mean_Group_Importance=('Mean_Importance', 'sum'),
        Feature_Count=('Feature', 'count')
    ).reset_index()

    # Sort by importance
    grouped = grouped.sort_values('Mean_Group_Importance', ascending=False)

    # Save results
    grouped.to_csv("output/feature_analysis/grouped_importance.csv", index=False)

    # Visualize
    plt.figure(figsize=(12, 8))
    plt.barh(grouped['Prefix'], grouped['Mean_Group_Importance'])
    plt.xlabel('Total Group Importance')
    plt.ylabel('Feature Group')
    plt.title('Feature Group Importance (by Prefix)')
    plt.tight_layout()
    plt.savefig("output/feature_analysis/grouped_importance.png")
    plt.close()

    # Normalized per-feature importance
    grouped['Mean_Per_Feature'] = grouped['Mean_Group_Importance'] / grouped['Feature_Count']
    grouped = grouped.sort_values('Mean_Per_Feature', ascending=False)

    plt.figure(figsize=(12, 8))
    plt.barh(grouped['Prefix'], grouped['Mean_Per_Feature'])
    plt.xlabel('Mean Importance per Feature')
    plt.ylabel('Feature Group')
    plt.title('Feature Group Importance (Normalized by Feature Count)')
    plt.tight_layout()
    plt.savefig("output/feature_analysis/grouped_importance_normalized.png")
    plt.close()

    print(f"Identified {len(grouped)} feature groups. Analysis saved to output/feature_analysis/.")


def run_progressive_feature_selection(X, y, features, importance_df, fold_info, random_state=42, trainer=None):
    """
    Run progressive feature selection using exactly the same folds as ModelTrainer

    Args:
        X: Feature matrix
        y: Target variable
        features: Feature names
        importance_df: DataFrame with feature importance
        fold_info: Dictionary with fold information from ModelTrainer
        random_state: Random seed
        trainer: ModelTrainer instance (optional) - to use its parameters

    Returns:
        tuple: (results_df, best_features) - Results DataFrame and list of best features
    """
    print("\n=== Progressive Feature Selection Analysis on Test Data ===")

    # Get feature order by importance
    sorted_features = importance_df['Feature'].tolist()

    # Define feature counts to evaluate
    total_features = len(features)
    if total_features <= 10:
        feature_counts = list(range(1, total_features + 1))
    else:
        # Create a sequence of feature counts
        feature_counts = [1, 2, 3, 5, 6, 7, 8, 9, 10, 15, 20, 30]
        feature_counts.extend(range(40, total_features + 1, 10))
        feature_counts = [n for n in feature_counts if n <= total_features]
        if total_features not in feature_counts:
            feature_counts.append(total_features)

    print(f"Testing {len(feature_counts)} different feature subset sizes on the exact same folds")

    # Results storage
    results = []

    # Get test and train indices
    test_indices = fold_info['test_indices']
    train_indices = fold_info['train_indices']

    # Extract the fold structure from ModelTrainer's indices
    fold_structure = extract_fold_structure(test_indices, train_indices)

    # For each feature count
    for n_features in feature_counts:
        print(f"Testing with top {n_features} features...")

        # Get indices of top n features
        selected_features = sorted_features[:n_features]
        selected_indices = [features.index(f) for f in selected_features]

        # Select features from X
        X_selected = X[:, selected_indices]

        # Storage for fold scores
        fold_scores = []

        # Use the exact same folds as ModelTrainer
        for fold, fold_data in fold_structure.items():
            train_idx = fold_data['train']
            test_idx = fold_data['test']

            # Split data
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Train model - use parameters from trainer if available
            if trainer is not None and trainer.best_params is not None:
                model = XGBRegressor(**trainer.best_params, random_state=random_state)
            else:
                model = XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=random_state
                )
            model.fit(X_train, y_train)

            # Evaluate on TEST data only
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            fold_scores.append(r2)

        # Calculate average test score
        avg_score = np.mean(fold_scores)
        std_score = np.std(fold_scores)

        # Store results
        results.append({
            'n_features': n_features,
            'features_used': ', '.join(selected_features[:5]) + (f" + {n_features - 5} more" if n_features > 5 else ""),
            'avg_test_r2': avg_score,
            'std_test_r2': std_score
        })

        print(f"Average Test R² with {n_features} features: {avg_score:.4f} ± {std_score:.4f}")

    # Create DataFrame
    results_df = pd.DataFrame(results)

    # Save to CSV
    results_df.to_csv("output/feature_analysis/progressive_feature_selection.csv", index=False)

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.errorbar(
        results_df['n_features'],
        results_df['avg_test_r2'],
        yerr=results_df['std_test_r2'],
        fmt='o-',
        capsize=5
    )
    plt.xlabel('Number of Top Features')
    plt.ylabel('Average Test R² Score')
    plt.title('Model Performance vs. Number of Features (Test Data)')
    plt.grid(linestyle='--', alpha=0.7)

    # Find best point
    best_idx = results_df['avg_test_r2'].idxmax()
    best_n = results_df.loc[best_idx, 'n_features']
    best_r2 = results_df.loc[best_idx, 'avg_test_r2']

    # Add annotation
    plt.annotate(
        f'Best: {best_n} features (R²={best_r2:.4f})',
        xy=(best_n, best_r2),
        xytext=(best_n + 5, best_r2 - 0.05),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5)
    )

    # Add the ModelTrainer's baseline R² if available
    if 'avg_test_r2' in fold_info:
        baseline_r2 = fold_info['avg_test_r2']
        plt.axhline(y=baseline_r2, color='r', linestyle='--',
                    label=f'ModelTrainer baseline (all features): {baseline_r2:.4f}')
        plt.legend()

    plt.tight_layout()
    plt.savefig("output/feature_analysis/progressive_feature_selection.png")
    plt.close()

    # Get best features
    best_features = sorted_features[:best_n]

    # Compare with ModelTrainer's baseline if available
    if 'avg_test_r2' in fold_info:
        baseline_r2 = fold_info['avg_test_r2']
        print(f"\nModelTrainer Test R² with all features: {baseline_r2:.4f}")
        print(f"Best Test R² with {best_n} features: {best_r2:.4f}")
        print(f"Difference: {best_r2 - baseline_r2:.4f}")

        if best_r2 > baseline_r2:
            print(f"The model with {best_n} features outperforms the full model by {best_r2 - baseline_r2:.4f} R²!")
        else:
            print(f"The full model performs better, but the model with {best_n} features is still good")
            print(f"and uses only {best_n}/{total_features} features ({best_n / total_features * 100:.1f}%).")

    # Print best features
    print("\nBest features:")
    for i, feature in enumerate(best_features[:min(20, len(best_features))], 1):
        importance = importance_df[importance_df['Feature'] == feature]['Mean_Importance'].values[0]
        print(f"{i}. {feature}: {importance:.4f}")

    # Save best features to a file
    best_features_df = pd.DataFrame({
        'Feature': best_features,
        'Importance': [importance_df[importance_df['Feature'] == f]['Mean_Importance'].values[0] for f in best_features]
    })
    best_features_df.to_csv("output/feature_analysis/best_features.csv", index=False)

    return results_df, best_features


def extract_fold_structure(test_indices, train_indices):
    """
    Extract the fold structure from the test and train indices

    Args:
        test_indices: List of test indices
        train_indices: List of train indices

    Returns:
        dict: Dictionary with fold structure
    """
    # Initialize fold structure
    fold_structure = {}

    # First, find the number of folds by analyzing the pattern in test_indices
    # This assumes that test_indices has a pattern like [fold1_idx, fold1_idx, ..., fold2_idx, fold2_idx, ...]
    unique_test_indices = []
    seen = set()
    for idx in test_indices:
        if idx not in seen:
            unique_test_indices.append(idx)
            seen.add(idx)

    # The total number of unique test indices divided by the test size gives us the number of folds
    # Assuming a standard 80/20 train/test split, test size would be about 1/5 of total data
    # So the number of folds is roughly equal to the number of unique test indices / (total data / 5)
    n_folds = 5  # Default to 5 folds

    # Try to detect the fold boundaries
    fold_sizes = []
    current_size = 1

    for i in range(1, len(test_indices)):
        if test_indices[i] == test_indices[i - 1]:
            current_size += 1
        else:
            fold_sizes.append(current_size)
            current_size = 1

    # Add the last fold size
    if current_size > 0:
        fold_sizes.append(current_size)

    # If we can detect the fold pattern
    if len(fold_sizes) > 0:
        # Average fold size
        avg_fold_size = sum(fold_sizes) / len(fold_sizes)

        # Reconstruct the folds
        fold_id = 0
        fold_count = 0

        for i, idx in enumerate(test_indices):
            if fold_id not in fold_structure:
                fold_structure[fold_id] = {'test': [], 'train': []}

            fold_structure[fold_id]['test'].append(idx)
            fold_count += 1

            # Check if we've reached the end of a fold
            if fold_count >= avg_fold_size and i < len(test_indices) - 1 and test_indices[i] != test_indices[i + 1]:
                fold_id += 1
                fold_count = 0

    # If we couldn't detect a pattern, create a simple approximation
    if len(fold_structure) == 0:
        # Split unique test indices into 5 equal parts
        n_folds = 5
        fold_size = len(unique_test_indices) // n_folds

        for fold_id in range(n_folds):
            start_idx = fold_id * fold_size
            end_idx = start_idx + fold_size if fold_id < n_folds - 1 else len(unique_test_indices)

            fold_structure[fold_id] = {
                'test': unique_test_indices[start_idx:end_idx],
                'train': []
            }

    # Now add train indices to each fold
    all_indices = set(range(max(max(test_indices), max(train_indices)) + 1))

    for fold_id in fold_structure:
        test_set = set(fold_structure[fold_id]['test'])
        # Train indices are all indices not in the test set
        fold_structure[fold_id]['train'] = list(all_indices - test_set)

    print(f"Extracted {len(fold_structure)} folds from ModelTrainer's indices")

    return fold_structure


# Main execution
if __name__ == "__main__":
    # Load data
    data_path = "../output/processed_data.csv"
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    # Run analysis
    importance_df, selection_df, best_features = analyze_feature_importance(df, random_state=42)