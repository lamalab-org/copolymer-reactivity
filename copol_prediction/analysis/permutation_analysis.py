import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')


def calculate_permutation_importance(model, X_test, y_test, feature_names,
                                     scoring='f1', n_repeats=10, random_state=42):
    """
    Calculate permutation feature importance for a trained model.

    Parameters:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        scoring: Scoring metric ('f1', 'accuracy', 'roc_auc')
        n_repeats: Number of permutation repeats
        random_state: Random seed

    Returns:
        DataFrame with permutation importance results
    """
    print(f"Calculating permutation importance with {scoring} metric...")

    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X_test, y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    # Create results dataframe
    results_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std,
        'importance_max': perm_importance.importances.max(axis=1),
        'importance_min': perm_importance.importances.min(axis=1)
    })

    # Sort by mean importance
    results_df = results_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)

    return results_df, perm_importance


def plot_permutation_importance(results_df, top_n=30, save_path='output/permutation_importance.png'):
    """
    Plot permutation importance with error bars.

    Parameters:
        results_df: DataFrame from calculate_permutation_importance
        top_n: Number of top features to plot
        save_path: Path to save the plot
    """
    # Get top N features
    top_features = results_df.head(top_n).copy()

    # Create the plot
    plt.figure(figsize=(12, max(8, top_n * 0.3)))

    # Create horizontal bar plot with error bars
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_features['importance_mean'],
             xerr=top_features['importance_std'],
             capsize=3, alpha=0.7)

    plt.yticks(y_pos, top_features['feature'])
    plt.xlabel('Permutation Importance (Decrease in Score)')
    plt.title(f'Top {top_n} Features - Permutation Importance')
    plt.gca().invert_yaxis()  # Highest importance at top

    # Add grid for better readability
    plt.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Permutation importance plot saved to: {save_path}")


def analyze_feature_redundancy(results_df, X_test, feature_names, threshold=0.001):
    """
    Identify potentially redundant features based on low permutation importance.

    Parameters:
        results_df: DataFrame from calculate_permutation_importance
        X_test: Test features for correlation analysis
        feature_names: List of feature names
        threshold: Importance threshold below which features are considered redundant

    Returns:
        Dictionary with analysis results
    """
    print(f"\nAnalyzing feature redundancy (threshold: {threshold})...")

    # Identify low-importance features
    low_importance = results_df[results_df['importance_mean'] <= threshold].copy()
    high_importance = results_df[results_df['importance_mean'] > threshold].copy()

    print(f"Features with importance <= {threshold}: {len(low_importance)}")
    print(f"Features with importance > {threshold}: {len(high_importance)}")

    # Calculate correlations for low-importance features with high-importance ones
    if len(low_importance) > 0 and len(high_importance) > 0:
        X_test_df = pd.DataFrame(X_test, columns=feature_names)

        correlation_analysis = []
        for low_feat in low_importance['feature']:
            max_corr = 0
            max_corr_feature = None

            for high_feat in high_importance['feature']:
                if low_feat in X_test_df.columns and high_feat in X_test_df.columns:
                    corr = abs(X_test_df[low_feat].corr(X_test_df[high_feat]))
                    if corr > max_corr:
                        max_corr = corr
                        max_corr_feature = high_feat

            correlation_analysis.append({
                'low_importance_feature': low_feat,
                'max_correlation': max_corr,
                'correlated_with': max_corr_feature,
                'importance': low_importance[low_importance['feature'] == low_feat]['importance_mean'].iloc[0]
            })

        correlation_df = pd.DataFrame(correlation_analysis)
        correlation_df = correlation_df.sort_values('max_correlation', ascending=False)
    else:
        correlation_df = pd.DataFrame()

    return {
        'low_importance_features': low_importance,
        'high_importance_features': high_importance,
        'correlation_analysis': correlation_df,
        'n_redundant': len(low_importance),
        'n_important': len(high_importance)
    }


def suggest_feature_removal(redundancy_analysis, correlation_threshold=0.8):
    """
    Suggest features for removal based on redundancy analysis.

    Parameters:
        redundancy_analysis: Output from analyze_feature_redundancy
        correlation_threshold: Correlation threshold for redundancy

    Returns:
        List of features suggested for removal
    """
    suggestions = []

    correlation_df = redundancy_analysis['correlation_analysis']

    if len(correlation_df) > 0:
        # Suggest removal of low-importance features that are highly correlated with important ones
        highly_correlated = correlation_df[correlation_df['max_correlation'] >= correlation_threshold]

        for _, row in highly_correlated.iterrows():
            suggestions.append({
                'feature_to_remove': row['low_importance_feature'],
                'reason': f'Low importance ({row["importance"]:.4f}) and highly correlated ({row["max_correlation"]:.3f}) with {row["correlated_with"]}',
                'importance': row['importance'],
                'correlation': row['max_correlation']
            })

    # Also suggest features with extremely low importance (close to zero or negative)
    extremely_low = redundancy_analysis['low_importance_features'][
        redundancy_analysis['low_importance_features']['importance_mean'] <= 0.0001
        ]

    for _, row in extremely_low.iterrows():
        if not any(s['feature_to_remove'] == row['feature'] for s in suggestions):
            suggestions.append({
                'feature_to_remove': row['feature'],
                'reason': f'Extremely low importance ({row["importance_mean"]:.6f})',
                'importance': row['importance_mean'],
                'correlation': None
            })

    return suggestions


def run_permutation_analysis(models, X_test, y_test, feature_names,
                             output_dir='output', scoring='f1'):
    """
    Run complete permutation feature importance analysis.

    Parameters:
        models: List of trained models from cross-validation
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        output_dir: Directory to save outputs
        scoring: Scoring metric for permutation importance

    Returns:
        Dictionary with all analysis results
    """
    print("=" * 50)
    print("PERMUTATION FEATURE IMPORTANCE ANALYSIS")
    print("=" * 50)

    # Use the first model for analysis (could also ensemble multiple models)
    model = models

    # Calculate permutation importance
    results_df, perm_importance = calculate_permutation_importance(
        model, X_test, y_test, feature_names, scoring=scoring
    )

    # Save detailed results
    results_df.to_csv(f'{output_dir}/permutation_importance_detailed.csv', index=False)

    # Plot importance
    plot_permutation_importance(results_df, top_n=30,
                                save_path=f'{output_dir}/permutation_importance.png')

    # Analyze feature redundancy
    redundancy_analysis = analyze_feature_redundancy(
        results_df, X_test, feature_names, threshold=0.001
    )

    # Get removal suggestions
    removal_suggestions = suggest_feature_removal(redundancy_analysis)

    # Print summary
    print(f"\n=== PERMUTATION IMPORTANCE SUMMARY ===")
    print(f"Total features analyzed: {len(feature_names)}")
    print(f"Features with importance > 0.001: {redundancy_analysis['n_important']}")
    print(f"Features with importance <= 0.001: {redundancy_analysis['n_redundant']}")

    print(f"\nTop 10 most important features:")
    for i, row in results_df.head(10).iterrows():
        print(f"  {i + 1:2d}. {row['feature']:<40} {row['importance_mean']:.6f} ± {row['importance_std']:.6f}")

    if len(removal_suggestions) > 0:
        print(f"\n=== FEATURE REMOVAL SUGGESTIONS ===")
        print(f"Suggested {len(removal_suggestions)} features for removal:")

        suggestions_df = pd.DataFrame(removal_suggestions)
        suggestions_df.to_csv(f'{output_dir}/feature_removal_suggestions.csv', index=False)

        for i, suggestion in enumerate(removal_suggestions[:10], 1):  # Show top 10
            print(f"  {i:2d}. {suggestion['feature_to_remove']:<40}")
            print(f"      Reason: {suggestion['reason']}")
    else:
        print("\nNo clear feature removal suggestions based on current thresholds.")

    # Save redundancy analysis
    if len(redundancy_analysis['correlation_analysis']) > 0:
        redundancy_analysis['correlation_analysis'].to_csv(
            f'{output_dir}/feature_correlation_analysis.csv', index=False
        )

    print(f"\nAnalysis complete. Results saved to {output_dir}/")

    return {
        'permutation_results': results_df,
        'redundancy_analysis': redundancy_analysis,
        'removal_suggestions': removal_suggestions,
        'raw_importance': perm_importance
    }


def create_feature_importance_comparison(tree_importance, perm_importance, feature_names,
                                         save_path='output/importance_comparison.png'):
    """
    Compare tree-based feature importance with permutation importance.

    Parameters:
        tree_importance: Array of tree-based importances
        perm_importance: Array of permutation importances
        feature_names: List of feature names
        save_path: Path to save comparison plot
    """
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'feature': feature_names,
        'tree_importance': tree_importance,
        'permutation_importance': perm_importance
    })

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(comparison_df['tree_importance'],
                comparison_df['permutation_importance'],
                alpha=0.6)

    plt.xlabel('Tree-based Feature Importance')
    plt.ylabel('Permutation Feature Importance')
    plt.title('Tree-based vs Permutation Feature Importance')

    # Add diagonal line for reference
    max_val = max(comparison_df['tree_importance'].max(),
                  comparison_df['permutation_importance'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Feature importance comparison plot saved to: {save_path}")

    return comparison_df