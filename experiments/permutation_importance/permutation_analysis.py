"""
SHAP and permutation importance utilities for feature importance analysis.

Provides feature grouping by correlation and SHAP-based feature importance
(voting model, validation set).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Feature grouping by correlation (for group-based permutation importance)
# ---------------------------------------------------------------------------

def _union_find_parent(parent, x):
    if parent[x] != x:
        parent[x] = _union_find_parent(parent, parent[x])
    return parent[x]


def build_feature_groups(X_df, feature_names, correlation_threshold=0.9):
    """
    Group features that have high absolute correlation with each other.
    Features in the same group will be permuted together in group-based importance.

    Parameters:
        X_df: DataFrame with columns including feature_names (e.g. training data)
        feature_names: List of feature names to consider
        correlation_threshold: Minimum absolute correlation to put two features in same group (e.g. 0.85 or 0.9)

    Returns:
        List of lists: each inner list is a group of feature names (singleton or multiple)
    """
    available = [f for f in feature_names if f in X_df.columns]
    if len(available) < 2:
        return [[f] for f in available]

    corr = X_df[available].corr()
    # Union-find: merge features with |corr| >= threshold
    parent = {f: f for f in available}
    for i, fi in enumerate(available):
        for j, fj in enumerate(available):
            if i >= j:
                continue
            if abs(corr.iloc[i, j]) >= correlation_threshold:
                pi = _union_find_parent(parent, fi)
                pj = _union_find_parent(parent, fj)
                if pi != pj:
                    parent[pi] = pj

    # Collect groups by root
    from collections import defaultdict
    root_to_features = defaultdict(list)
    for f in available:
        root = _union_find_parent(parent, f)
        root_to_features[root].append(f)

    return [sorted(g) for g in root_to_features.values()]


def _scorer_from_string(scoring, y_true, y_pred):
    """Return score for (y_true, y_pred) given scoring name."""
    if scoring in ('f1_macro', 'f1'):
        return f1_score(y_true, y_pred, average='macro', zero_division=0)
    if scoring == 'balanced_accuracy':
        return balanced_accuracy_score(y_true, y_pred)
    if scoring == 'accuracy':
        return accuracy_score(y_true, y_pred)
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def calculate_permutation_importance_by_groups(
    model, X_df, y_true, feature_groups,
    scoring='f1_macro', n_repeats=10, random_state=42
):
    """
    Permutation importance by feature groups (correlated features permuted together).

    Parameters:
        model: Predictor with .predict(X) where X is DataFrame with same columns as X_df
        X_df: DataFrame of features (columns must include all features in feature_groups)
        y_true: True labels
        feature_groups: List of lists of feature names (each group permuted together)
        scoring: 'f1_macro', 'balanced_accuracy', or 'accuracy'
        n_repeats: Number of permutation repeats per group
        random_state: Random seed

    Returns:
        results_df: DataFrame with columns group_label, features (tuple/str), importance_mean, importance_std
        (sorted by importance_mean descending)
    """
    rng = np.random.default_rng(random_state)
    X_df = X_df.copy()
    n = len(X_df)

    def score_fn(X):
        pred = model.predict(X)
        return _scorer_from_string(scoring, y_true, pred)

    baseline = score_fn(X_df)
    print(f"  Baseline score ({scoring}): {baseline:.4f}")

    group_importances = []  # list of (repeat, group_idx, drop)
    for gidx, group in enumerate(feature_groups):
        drops = []
        for _ in range(n_repeats):
            X_perm = X_df.copy()
            idx = rng.permutation(n)
            for f in group:
                if f in X_perm.columns:
                    X_perm[f] = X_df[f].iloc[idx].values
            s = score_fn(X_perm)
            drops.append(baseline - s)
        group_importances.append((group, drops))

    rows = []
    for group, drops in group_importances:
        group_label = group[0] if len(group) == 1 else f"{group[0]} (+{len(group)-1})"
        rows.append({
            'group_label': group_label,
            'features': tuple(group),
            'n_features': len(group),
            'importance_mean': np.mean(drops),
            'importance_std': np.std(drops),
        })
    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return results_df


# ---------------------------------------------------------------------------
# SHAP-based feature importance
# ---------------------------------------------------------------------------

def calculate_shap_importance_by_groups(
    model, X_df, feature_groups, max_samples=500
):
    """
    Calculate SHAP importance by feature groups (correlated features grouped together).
    For groups: mean absolute SHAP value across features in the group.

    Parameters:
        model: XGBoost model (must have .get_booster() for TreeExplainer)
        X_df: DataFrame of features (columns must include all features in feature_groups)
        feature_groups: List of lists of feature names (each group gets combined SHAP)
        max_samples: Maximum samples for SHAP computation (for speed)

    Returns:
        results_df: DataFrame with columns group_label, features (tuple), importance_mean, importance_std
        (sorted by importance_mean descending)
        shap_values_per_group: Dict mapping group_label to array of SHAP values per sample
        feature_values_per_group: Dict mapping group_label to array of feature values per sample (mean for groups)
        X_sample: The DataFrame used for SHAP computation (for reference)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP not installed. Install with: pip install shap")

    # Limit samples for speed
    if len(X_df) > max_samples:
        X_sample = X_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"  Computing SHAP on {max_samples} samples (of {len(X_df)} total)")
    else:
        X_sample = X_df

    # Get XGBoost booster
    if hasattr(model, 'get_booster'):
        booster = model.get_booster()
    elif hasattr(model, 'model') and hasattr(model.model, 'get_booster'):
        booster = model.model.get_booster()
    else:
        raise ValueError("Model must be XGBoost with .get_booster() method")

    # Compute SHAP values
    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)

    # Handle multi-class: shap_values can be a list OR a 3D array (samples, features, classes)
    # Use mean absolute SHAP across classes
    if isinstance(shap_values, list):
        # List of arrays (one per class): each has shape (n_samples, n_features)
        shap_abs_list = [np.abs(sv) for sv in shap_values]
        shap_abs = np.stack(shap_abs_list, axis=0).mean(axis=0)  # Shape: (n_samples, n_features)
        print(f"  Multi-class (list): {len(shap_values)} classes, SHAP shape after mean: {shap_abs.shape}")
    elif len(shap_values.shape) == 3:
        # 3D array: (n_samples, n_features, n_classes)
        shap_abs = np.abs(shap_values).mean(axis=2)  # Mean over classes -> Shape: (n_samples, n_features)
        print(f"  Multi-class (3D array): SHAP shape {shap_values.shape} -> {shap_abs.shape} after mean over classes")
    else:
        # Single class: (n_samples, n_features)
        shap_abs = np.abs(shap_values)
        print(f"  Single class: SHAP shape: {shap_abs.shape}")
    
    print(f"  X_sample shape: {X_sample.shape}")

    # Map feature names to indices
    feature_to_idx = {f: i for i, f in enumerate(X_sample.columns)}
    
    rows = []
    shap_values_per_group = {}
    feature_values_per_group = {}
    for group in feature_groups:
        # Get indices for features in this group
        group_indices = [feature_to_idx[f] for f in group if f in feature_to_idx]
        if not group_indices:
            continue
        
        # Mean absolute SHAP for this group (mean over features in group -> one value per sample)
        group_shap = shap_abs[:, group_indices].mean(axis=1)  # Shape: (n_samples,)
        # Ensure it's a 1D numpy array
        group_shap = np.asarray(group_shap).flatten()
        
        # Get feature values: for groups, use mean; for singletons, use the single feature value
        group_feature_cols = [X_sample.columns[i] for i in group_indices]
        if len(group_feature_cols) == 1:
            group_feature_vals = X_sample[group_feature_cols[0]].values
        else:
            # Mean across features in group
            group_feature_vals = X_sample[group_feature_cols].mean(axis=1).values
        group_feature_vals = np.asarray(group_feature_vals).flatten()
        
        # Ensure lengths match
        if len(group_shap) != len(group_feature_vals):
            print(f"    Warning: Length mismatch for {group[0]}: group_shap={len(group_shap)}, group_feature_vals={len(group_feature_vals)}, shap_abs.shape={shap_abs.shape}, X_sample.shape={X_sample.shape}")
            # Skip this group if lengths don't match
            continue
        
        importance_mean = group_shap.mean()
        importance_std = group_shap.std()
        
        group_label = group[0] if len(group) == 1 else f"{group[0]} (+{len(group)-1})"
        rows.append({
            'group_label': group_label,
            'features': tuple(group),
            'n_features': len(group),
            'importance_mean': importance_mean,
            'importance_std': importance_std,
        })
        shap_values_per_group[group_label] = group_shap
        feature_values_per_group[group_label] = group_feature_vals
    
    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return results_df, shap_values_per_group, feature_values_per_group, X_sample


def calculate_permutation_importance(model, X_test, y_test, feature_names,
                                     scoring='f1', n_repeats=10, random_state=42):
    """
    Calculate permutation feature importance for a trained model (per-feature, no grouping).

    Parameters:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        feature_names: List of feature names
        scoring: Scoring metric ('f1', 'accuracy', 'roc_auc')
        n_repeats: Number of permutation repeats
        random_state: Random seed

    Returns:
        DataFrame with permutation importance results, and raw perm_importance object
    """
    print(f"Calculating permutation importance with {scoring} metric...")

    perm_importance = permutation_importance(
        model, X_test, y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    results_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std,
        'importance_max': perm_importance.importances.max(axis=1),
        'importance_min': perm_importance.importances.min(axis=1)
    })
    results_df = results_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
    return results_df, perm_importance


def plot_permutation_importance(results_df, top_n=30, save_path='output/permutation_importance.png'):
    """Plot permutation importance with error bars (per-feature results)."""
    top_features = results_df.head(top_n).copy()
    plt.figure(figsize=(12, max(8, top_n * 0.3)))
    y_pos = np.arange(len(top_features))
    plt.barh(y_pos, top_features['importance_mean'],
             xerr=top_features['importance_std'],
             capsize=3, alpha=0.7)
    plt.yticks(y_pos, top_features['feature'])
    plt.xlabel('Permutation Importance (Decrease in Score)')
    plt.title(f'Top {top_n} Features - Permutation Importance')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Permutation importance plot saved to: {save_path}")


def analyze_feature_redundancy(results_df, X_test, feature_names, threshold=0.001):
    """Identify potentially redundant features based on low permutation importance."""
    low_importance = results_df[results_df['importance_mean'] <= threshold].copy()
    high_importance = results_df[results_df['importance_mean'] > threshold].copy()
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
    """Suggest features for removal based on redundancy analysis."""
    suggestions = []
    correlation_df = redundancy_analysis['correlation_analysis']
    if len(correlation_df) > 0:
        highly_correlated = correlation_df[correlation_df['max_correlation'] >= correlation_threshold]
        for _, row in highly_correlated.iterrows():
            suggestions.append({
                'feature_to_remove': row['low_importance_feature'],
                'reason': f'Low importance ({row["importance"]:.4f}) and highly correlated ({row["max_correlation"]:.3f}) with {row["correlated_with"]}',
                'importance': row['importance'],
                'correlation': row['max_correlation']
            })
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
    Run complete per-feature permutation importance analysis (legacy entry point).
    """
    model = models
    results_df, perm_importance = calculate_permutation_importance(
        model, X_test, y_test, feature_names, scoring=scoring
    )
    results_df.to_csv(f'{output_dir}/permutation_importance_detailed.csv', index=False)
    plot_permutation_importance(results_df, top_n=30, save_path=f'{output_dir}/permutation_importance.png')
    redundancy_analysis = analyze_feature_redundancy(results_df, X_test, feature_names, threshold=0.001)
    removal_suggestions = suggest_feature_removal(redundancy_analysis)
    if len(redundancy_analysis['correlation_analysis']) > 0:
        redundancy_analysis['correlation_analysis'].to_csv(
            f'{output_dir}/feature_correlation_analysis.csv', index=False
        )
    return {
        'permutation_results': results_df,
        'redundancy_analysis': redundancy_analysis,
        'removal_suggestions': removal_suggestions,
        'raw_importance': perm_importance
    }


def create_feature_importance_comparison(tree_importance, perm_importance, feature_names,
                                         save_path='output/importance_comparison.png'):
    """Compare tree-based feature importance with permutation importance."""
    comparison_df = pd.DataFrame({
        'feature': feature_names,
        'tree_importance': tree_importance,
        'permutation_importance': perm_importance
    })
    plt.figure(figsize=(10, 8))
    plt.scatter(comparison_df['tree_importance'], comparison_df['permutation_importance'], alpha=0.6)
    plt.xlabel('Tree-based Feature Importance')
    plt.ylabel('Permutation Feature Importance')
    plt.title('Tree-based vs Permutation Feature Importance')
    max_val = max(comparison_df['tree_importance'].max(), comparison_df['permutation_importance'].max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Feature importance comparison plot saved to: {save_path}")
    return comparison_df
