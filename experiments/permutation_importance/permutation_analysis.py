"""
SHAP and permutation importance utilities for feature importance analysis.

Provides feature grouping by correlation and SHAP-based feature importance
(voting model, validation set).
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not installed. Install with: pip install shap")

warnings.filterwarnings("ignore")


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


def build_semantic_feature_groups(
    feature_names: list[str],
    *,
    include_monomer_1_2_pairs: bool = True,
    include_delta_homo_lumo_pairs: bool = True,
) -> list[list[str]]:
    """
    Build "semantic" feature groups that should be permuted together.

    Motivation (requested):
      - Permute Δ HOMO–LUMO pairs together:
          AA + BB  (1-1 & 2-2)
          AB + BA  (1-2 & 2-1)
      - Optionally permute monomer1/monomer2 counterparts together:
          *_1 + *_2 (e.g. dipole_x_1 with dipole_x_2)

    Notes:
      - This is NOT correlation-based; it's naming-rule based.
      - Only returns groups for features that exist in `feature_names`.
      - Any feature not captured by a multi-feature group is returned as a singleton.
    """
    names = [str(f) for f in feature_names]
    present = set(names)
    used: set[str] = set()
    groups: list[list[str]] = []

    # Dipole moments: group all x/y/z for monomer 1&2 together
    dipole_feats = [
        "dipole_x_1",
        "dipole_y_1",
        "dipole_z_1",
        "dipole_x_2",
        "dipole_y_2",
        "dipole_z_2",
    ]
    dip_present = [f for f in dipole_feats if f in present]
    if len(dip_present) >= 2:
        groups.append(sorted(dip_present))
        used.update(dip_present)

    # Δ HOMO–LUMO explicit pairing (AA+BB and AB+BA)
    if include_delta_homo_lumo_pairs:
        pair_groups = [
            ("delta_HOMO_LUMO_AA", "delta_HOMO_LUMO_BB"),
            ("delta_HOMO_LUMO_AB", "delta_HOMO_LUMO_BA"),
        ]
        for a, b in pair_groups:
            feats = [f for f in (a, b) if f in present]
            if len(feats) >= 2:
                groups.append(sorted(feats))
                used.update(feats)

    # Generic monomer 1/2 pairing by suffix
    if include_monomer_1_2_pairs:
        # map base -> (f1, f2)
        base_to_pair: dict[str, list[str]] = {}
        for f in names:
            if f.endswith("_1") and f[:-2] in (x[:-2] for x in names if x.endswith("_2")):
                base_to_pair.setdefault(f[:-2], []).append(f)
            elif f.endswith("_2") and f[:-2] in (x[:-2] for x in names if x.endswith("_1")):
                base_to_pair.setdefault(f[:-2], []).append(f)
        for base, feats in base_to_pair.items():
            feats = sorted({ff for ff in feats if ff in present})
            if len(feats) == 2 and not any(ff in used for ff in feats):
                groups.append(feats)
                used.update(feats)

    # Remaining features as singletons (preserve original order as much as possible)
    for f in names:
        if f in used:
            continue
        groups.append([f])

    # De-dup safety (shouldn't be needed but keep it robust)
    seen = set()
    unique_groups = []
    for g in groups:
        key = tuple(g)
        if key in seen:
            continue
        seen.add(key)
        unique_groups.append(g)
    return unique_groups


def build_hybrid_feature_groups(
    X_df: pd.DataFrame,
    feature_names: list[str],
    *,
    correlation_threshold: float = 0.9,
    include_monomer_1_2_pairs: bool = True,
    include_delta_homo_lumo_pairs: bool = True,
) -> list[list[str]]:
    """
    Hybrid grouping: enforce semantic groups first, then correlation-group the remainder.

    This yields stable, interpretable groupings while still collapsing highly correlated
    leftovers.
    """
    semantic = build_semantic_feature_groups(
        feature_names,
        include_monomer_1_2_pairs=include_monomer_1_2_pairs,
        include_delta_homo_lumo_pairs=include_delta_homo_lumo_pairs,
    )
    used = {f for g in semantic for f in g}
    remaining = [f for f in feature_names if f not in used]
    corr_groups = build_feature_groups(X_df, remaining, correlation_threshold=correlation_threshold)
    return [g for g in semantic if g] + [g for g in corr_groups if g]


def _scorer_from_string(scoring, y_true, y_pred):
    """Return score for (y_true, y_pred) given scoring name."""
    if scoring in ("f1_macro", "f1"):
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    if scoring == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if scoring == "accuracy":
        return accuracy_score(y_true, y_pred)
    return f1_score(y_true, y_pred, average="macro", zero_division=0)


def calculate_permutation_importance_by_groups(
    model, X_df, y_true, feature_groups, scoring="f1_macro", n_repeats=10, random_state=42
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
        rows.append(
            {
                "group_label": group_label,
                "features": tuple(group),
                "n_features": len(group),
                "importance_mean": np.mean(drops),
                "importance_std": np.std(drops),
            }
        )
    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return results_df


# ---------------------------------------------------------------------------
# SHAP-based feature importance
# ---------------------------------------------------------------------------


def calculate_shap_importance_by_groups(
    model,
    X_df,
    feature_groups,
    max_samples=500,
    reduction="mean",
    group_labels=None,
):
    """
    Calculate SHAP importance by feature groups (correlated features grouped together).
    For groups: mean absolute SHAP value across features in the group.

    Parameters:
        model: XGBoost model (must have .get_booster() for TreeExplainer)
        X_df: DataFrame of features (columns must include all features in feature_groups)
        feature_groups: List of lists of feature names (each group gets combined SHAP)
        max_samples: Maximum samples for SHAP computation (for speed)
        reduction: 'mean' (default) or 'sum' — how to aggregate |SHAP| across classes
        group_labels: Optional list of str, same length as feature_groups, providing
                      explicit group labels. If None, labels are auto-generated.

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
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    elif hasattr(model, "model") and hasattr(model.model, "get_booster"):
        booster = model.model.get_booster()
    else:
        raise ValueError("Model must be XGBoost with .get_booster() method")

    # Compute SHAP values
    print("  Computing SHAP values...")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)

    # Handle multi-class: shap_values can be a list OR a 3D array (samples, features, classes)
    # Aggregate |SHAP| across classes using reduction ('mean' or 'sum')
    reduce_fn = np.sum if reduction == "sum" else np.mean
    if isinstance(shap_values, list):
        shap_abs_list = [np.abs(sv) for sv in shap_values]
        shap_abs = reduce_fn(np.stack(shap_abs_list, axis=0), axis=0)  # (n_samples, n_features)
        print(
            f"  Multi-class (list): {len(shap_values)} classes, SHAP shape after {reduction}: {shap_abs.shape}"
        )
    elif len(shap_values.shape) == 3:
        # 3D array: (n_samples, n_features, n_classes)
        shap_abs = reduce_fn(np.abs(shap_values), axis=2)  # (n_samples, n_features)
        print(
            f"  Multi-class (3D array): SHAP shape {shap_values.shape} -> {shap_abs.shape} after {reduction} over classes"
        )
    else:
        shap_abs = np.abs(shap_values)
        print(f"  Single class: SHAP shape: {shap_abs.shape}")

    print(f"  X_sample shape: {X_sample.shape}")

    # Map feature names to indices
    feature_to_idx = {f: i for i, f in enumerate(X_sample.columns)}

    rows = []
    shap_values_per_group = {}
    feature_values_per_group = {}
    for g_idx, group in enumerate(feature_groups):
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
            print(
                f"    Warning: Length mismatch for {group[0]}: group_shap={len(group_shap)}, group_feature_vals={len(group_feature_vals)}, shap_abs.shape={shap_abs.shape}, X_sample.shape={X_sample.shape}"
            )
            # Skip this group if lengths don't match
            continue

        importance_mean = float(group_shap.mean())
        importance_std = float(group_shap.std())
        q25, q50, q75 = [float(x) for x in np.percentile(group_shap, [25, 50, 75])]

        if group_labels is not None and g_idx < len(group_labels):
            group_label = str(group_labels[g_idx])
        else:
            group_label = group[0] if len(group) == 1 else f"{group[0]} (+{len(group)-1})"
        rows.append(
            {
                "group_label": group_label,
                "features": tuple(group),
                "n_features": len(group),
                "importance_mean": importance_mean,
                "importance_std": importance_std,
                "q25": q25,
                "q50": q50,
                "q75": q75,
            }
        )
        shap_values_per_group[group_label] = group_shap
        feature_values_per_group[group_label] = group_feature_vals

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return results_df, shap_values_per_group, feature_values_per_group, X_sample


def calculate_shap_pairwise_importance_by_groups(
    model,
    X_df: pd.DataFrame,
    y_true: np.ndarray,
    feature_groups,
    *,
    class_a: int,
    class_b: int,
    max_samples: int = 500,
):
    """
    Pairwise SHAP importance for multi-class models.

    Motivation:
      For 3-class problems, averaging |SHAP| across classes can blur signal.
      For a given decision "class_a vs class_b", the relevant quantity is the
      difference of the class scores/logits. For tree models, SHAP is additive
      per class score, so we approximate the pairwise explanation by:

        SHAP_pair = SHAP(class_a) - SHAP(class_b)

      and then summarize importance as mean absolute SHAP_pair.

    This function:
      - filters X_df/y_true to only samples with y_true in {class_a, class_b}
      - computes SHAP values via TreeExplainer
      - builds pairwise SHAP contributions and returns group-based importances,
        analogous to calculate_shap_importance_by_groups.

    Parameters:
        model: XGBoost model (must have .get_booster())
        X_df: DataFrame of features
        y_true: true multiclass labels (used only for filtering)
        feature_groups: list[list[str]] feature groups
        class_a, class_b: the class pair to compare
        max_samples: max samples for SHAP computation (speed)
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP not installed. Install with: pip install shap")

    y_true = np.asarray(y_true).astype(int)
    mask = (y_true == int(class_a)) | (y_true == int(class_b))
    X_pair = X_df.loc[mask].reset_index(drop=True)
    if len(X_pair) == 0:
        raise ValueError(f"No samples found for classes {class_a} vs {class_b}")

    # Limit samples for speed
    if len(X_pair) > max_samples:
        X_sample = X_pair.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(
            f"  Computing pairwise SHAP ({class_a} vs {class_b}) on {max_samples} samples (of {len(X_pair)} total)"
        )
    else:
        X_sample = X_pair

    # Get XGBoost booster
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    elif hasattr(model, "model") and hasattr(model.model, "get_booster"):
        booster = model.model.get_booster()
    else:
        raise ValueError("Model must be XGBoost with .get_booster() method")

    # Compute SHAP values
    print(f"  Computing SHAP values for pairwise comparison: {class_a} vs {class_b} ...")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)

    # Extract per-class SHAP arrays (n_samples, n_features)
    if isinstance(shap_values, list):
        if max(class_a, class_b) >= len(shap_values):
            raise ValueError(
                f"SHAP returned {len(shap_values)} classes; cannot index {class_a},{class_b}"
            )
        shap_a = np.asarray(shap_values[int(class_a)])
        shap_b = np.asarray(shap_values[int(class_b)])
        print(
            f"  Multi-class (list): using SHAP[{class_a}] - SHAP[{class_b}] with shape {shap_a.shape}"
        )
    elif len(shap_values.shape) == 3:
        # (n_samples, n_features, n_classes)
        if max(class_a, class_b) >= shap_values.shape[2]:
            raise ValueError(
                f"SHAP returned {shap_values.shape[2]} classes; cannot index {class_a},{class_b}"
            )
        shap_a = np.asarray(shap_values[:, :, int(class_a)])
        shap_b = np.asarray(shap_values[:, :, int(class_b)])
        print(
            f"  Multi-class (3D): using SHAP[:,:,{class_a}] - SHAP[:,:,{class_b}] with shape {shap_a.shape}"
        )
    else:
        raise ValueError("Pairwise SHAP requires multi-class SHAP output (list or 3D array).")

    shap_pair = shap_a - shap_b
    shap_abs = np.abs(shap_pair)  # (n_samples, n_features)

    # Map feature names to indices
    feature_to_idx = {f: i for i, f in enumerate(X_sample.columns)}

    rows = []
    shap_values_per_group = {}
    feature_values_per_group = {}

    for group in feature_groups:
        group_indices = [feature_to_idx[f] for f in group if f in feature_to_idx]
        if not group_indices:
            continue

        group_shap = shap_abs[:, group_indices].mean(axis=1)
        group_shap = np.asarray(group_shap).flatten()

        group_feature_cols = [X_sample.columns[i] for i in group_indices]
        if len(group_feature_cols) == 1:
            group_feature_vals = X_sample[group_feature_cols[0]].values
        else:
            group_feature_vals = X_sample[group_feature_cols].mean(axis=1).values
        group_feature_vals = np.asarray(group_feature_vals).flatten()

        if len(group_shap) != len(group_feature_vals):
            print(f"    Warning: Length mismatch for {group[0]} (pairwise). Skipping.")
            continue

        importance_mean = float(group_shap.mean())
        importance_std = float(group_shap.std())
        q25, q50, q75 = [float(x) for x in np.percentile(group_shap, [25, 50, 75])]

        group_label = group[0] if len(group) == 1 else f"{group[0]} (+{len(group)-1})"
        rows.append(
            {
                "group_label": group_label,
                "features": tuple(group),
                "n_features": len(group),
                "importance_mean": importance_mean,
                "importance_std": importance_std,
                "q25": q25,
                "q50": q50,
                "q75": q75,
            }
        )
        shap_values_per_group[group_label] = group_shap
        feature_values_per_group[group_label] = group_feature_vals

    results_df = (
        pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    )
    return results_df, shap_values_per_group, feature_values_per_group, X_sample


def _strong_group_label(feature_name: str) -> str:
    """
    Manual/semantic grouping for "strongly grouped" average SHAP plots.

    - Combine monomer1/monomer2 feature pairs: *_1 and *_2 -> "<base> (1&2)"
      e.g. dipole_x_1 + dipole_x_2
    - Combine Δ HOMO–LUMO pairs:
        AA + BB -> "Δ HOMO-LUMO (1-1 & 2-2)"
        AB + BA -> "Δ HOMO-LUMO (1-2 & 2-1)"
    """
    f = str(feature_name)

    if f in ("delta_HOMO_LUMO_AA", "delta_HOMO_LUMO_BB"):
        return "Δ HOMO-LUMO (1-1 & 2-2)"
    if f in ("delta_HOMO_LUMO_AB", "delta_HOMO_LUMO_BA"):
        return "Δ HOMO-LUMO (1-2 & 2-1)"

    # generic monomer-pair grouping
    if f.endswith("_1") or f.endswith("_2"):
        base = f[:-2]
        return f"{base} (1&2)"

    return f


def calculate_shap_average_strong_groups(
    model,
    X_df: pd.DataFrame,
    *,
    max_samples: int = 500,
):
    """
    Average absolute SHAP across all classes, with manual/semantic strong grouping.

    Output has NO error bars by design (just importance_mean), and groups
    are merged by naming rules in _strong_group_label.
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP not installed. Install with: pip install shap")

    # Limit samples for speed
    if len(X_df) > max_samples:
        X_sample = X_df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"  Computing strongly-grouped SHAP on {max_samples} samples (of {len(X_df)} total)")
    else:
        X_sample = X_df.reset_index(drop=True)

    # Get XGBoost booster
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    elif hasattr(model, "model") and hasattr(model.model, "get_booster"):
        booster = model.model.get_booster()
    else:
        raise ValueError("Model must be XGBoost with .get_booster() method")

    # Compute SHAP values
    import shap

    print("  Computing SHAP values (strong groups)...")
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)

    # Convert to per-feature mean(|SHAP|) across classes:
    if isinstance(shap_values, list):
        shap_abs = np.stack([np.abs(sv) for sv in shap_values], axis=0).mean(axis=0)  # (n, f)
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            shap_abs = np.abs(sv).mean(axis=2)  # (n, f)
        else:
            shap_abs = np.abs(sv)  # (n, f)

    feature_names = list(X_sample.columns)
    feature_to_idx = {f: i for i, f in enumerate(feature_names)}

    # Build mapping strong_group -> feature indices
    group_to_features = {}
    for f in feature_names:
        g = _strong_group_label(f)
        group_to_features.setdefault(g, []).append(f)

    rows = []
    for g, feats in group_to_features.items():
        idxs = [feature_to_idx[f] for f in feats if f in feature_to_idx]
        if not idxs:
            continue
        group_shap = shap_abs[:, idxs].mean(axis=1)  # (n,)
        rows.append(
            {
                "group_label": g,
                "features": tuple(sorted(feats)),
                "n_features": int(len(feats)),
                "importance_mean": float(np.mean(group_shap)),
            }
        )

    df = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df, X_sample


def calculate_permutation_importance(
    model, X_test, y_test, feature_names, scoring="f1", n_repeats=10, random_state=42
):
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
        model,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
    )

    results_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm_importance.importances_mean,
            "importance_std": perm_importance.importances_std,
            "importance_max": perm_importance.importances.max(axis=1),
            "importance_min": perm_importance.importances.min(axis=1),
        }
    )
    results_df = results_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return results_df, perm_importance


def plot_permutation_importance(
    results_df, top_n=30, save_path="output/permutation_importance.png"
):
    """Plot permutation importance with error bars (per-feature results)."""
    top_features = results_df.head(top_n).copy()
    plt.figure(figsize=(12, max(8, top_n * 0.3)))
    y_pos = np.arange(len(top_features))
    plt.barh(
        y_pos,
        top_features["importance_mean"],
        xerr=top_features["importance_std"],
        capsize=3,
        alpha=0.7,
    )
    plt.yticks(y_pos, top_features["feature"])
    plt.xlabel("Permutation Importance (Decrease in Score)")
    plt.title(f"Top {top_n} Features - Permutation Importance")
    plt.gca().invert_yaxis()
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Permutation importance plot saved to: {save_path}")


def build_named_groups_for_paper(feature_names: list[str]) -> dict[str, list[str]]:
    """
    Paper groups (labels) used for the permutation-importance barplot.
    Permutation is still computed per-feature; these groups are for plotting only.
    """
    available = set(str(f) for f in (feature_names or []))

    def _take(candidates: list[str]) -> list[str]:
        return [c for c in candidates if c in available]

    groups: dict[str, list[str]] = {}
    groups["Polymerization type embedding"] = _take(["polytype_emb_1", "polytype_emb_2"])
    groups["Charges min"] = _take(["charges_min_1", "charges_min_2"])
    groups["Dipole moment (x, y, z; monomer 1&2)"] = _take(
        ["dipole_x_1", "dipole_x_2", "dipole_y_1", "dipole_y_2", "dipole_z_1", "dipole_z_2"]
    )
    groups["HOMO"] = _take(["homo_1", "homo_2"])
    groups["Fukui index (electrophilicity) max"] = _take(
        ["fukui_electrophilicity_max_1", "fukui_electrophilicity_max_2"]
    )
    groups["Charges mean"] = _take(["charges_mean_1", "charges_mean_2"])
    groups["Solvent LogP"] = _take(["solvent_logp", "solvent_logP"])
    groups["HOMO-LUMO (1-1)"] = _take(["delta_HOMO_LUMO_AA", "delta_HOMO_LUMO_BB"])
    groups["HOMO-LUMO (1-2)"] = _take(["delta_HOMO_LUMO_AB", "delta_HOMO_LUMO_BA"])
    groups["Temperature"] = _take(["temperature"])
    groups["Solvent Fraction of sp3 C"] = _take(["solvent_FractionCSP3"])
    groups["Ionization potential"] = _take(["ip_1", "ip_2"])
    groups["Polymerization method embedding"] = _take(["method_emb_1", "method_emb_2"])
    groups["Fukui index (radical) max"] = _take(["fukui_radical_max_1", "fukui_radical_max_2"])
    groups["Charges max"] = _take(["charges_max_1", "charges_max_2"])
    groups["Fukui index (nucleophilicity) max"] = _take(
        ["fukui_nucleophilicity_max_1", "fukui_nucleophilicity_max_2"]
    )
    groups["Ionization potential (corrected)"] = _take(["ip_corrected_1", "ip_corrected_2"])
    groups["Fukui index (nucleophilicity) mean"] = _take(
        ["fukui_nucleophilicity_mean_1", "fukui_nucleophilicity_mean_2"]
    )
    groups["Global nucleophilicity"] = _take(
        ["global_nucleophilicity_1", "global_nucleophilicity_2"]
    )
    groups["Global electrophilicity"] = _take(
        ["global_electrophilicity_1", "global_electrophilicity_2"]
    )
    groups["LUMO"] = _take(["lumo_1", "lumo_2"])
    groups["Fukui index (radical) mean"] = _take(["fukui_radical_mean_1", "fukui_radical_mean_2"])
    groups["Fukui index (nucleophilicity) min"] = _take(
        ["fukui_nucleophilicity_min_1", "fukui_nucleophilicity_min_2"]
    )
    groups["Solvent number of hydrogen bond donors"] = _take(["solvent_HBD"])
    groups["Fukui index (electrophilicity) mean"] = _take(
        ["fukui_electrophilicity_mean_1", "fukui_electrophilicity_mean_2"]
    )
    groups["Solvent Topological Polar Surface Area"] = _take(["solvent_TPSA"])
    groups["Electron affinity"] = _take(["ea_1", "ea_2"])
    groups["Fukui index (electrophilicity) min"] = _take(
        ["fukui_electrophilicity_min_1", "fukui_electrophilicity_min_2"]
    )
    groups["Best conformer energy"] = _take(["best_conformer_energy_1", "best_conformer_energy_2"])
    groups["Fukui index (radical) min"] = _take(["fukui_radical_min_1", "fukui_radical_min_2"])

    # drop empty
    return {k: v for k, v in groups.items() if v}


def aggregate_per_feature_importance_to_groups(
    per_feature_df: pd.DataFrame,
    named_groups: dict[str, list[str]],
    *,
    feature_col: str = "feature",
    mean_col: str = "importance_mean",
    std_col: str = "importance_std",
) -> pd.DataFrame:
    """
    Aggregate per-feature permutation importance into plot-groups (for plotting only).
    """
    if per_feature_df is None or len(per_feature_df) == 0:
        return pd.DataFrame(
            columns=["group_label", "features", "n_features", "importance_mean", "importance_std"]
        )

    df = per_feature_df.copy()
    if std_col not in df.columns:
        df[std_col] = 0.0

    rows = []
    for label, feats in named_groups.items():
        feats = [str(f) for f in feats]
        sub = df[df[feature_col].astype(str).isin(feats)]
        if len(sub) == 0:
            continue
        means = sub[mean_col].astype(float).values
        stds = sub[std_col].astype(float).values
        rows.append(
            {
                "group_label": str(label),
                "features": tuple(sub[feature_col].astype(str).tolist()),
                "n_features": int(len(sub)),
                "importance_mean": float(np.mean(means)),
                "importance_std": float(np.sqrt(np.mean(stds**2))) if len(stds) else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def calculate_permutation_importance_by_named_groups(
    model,
    X_df: pd.DataFrame,
    y_true,
    named_groups: dict[str, list[str]],
    *,
    scoring: str,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Permutation importance where each *group* is permuted jointly (not per-feature),
    with explicit group labels (used in plot/CSV).
    """
    rng = np.random.default_rng(int(random_state))
    X_df = X_df.copy()
    n = len(X_df)

    y_true = np.asarray(y_true)

    def score_fn(X: pd.DataFrame) -> float:
        pred = model.predict(X)
        return float(_scorer_from_string(scoring, y_true, np.asarray(pred)))

    baseline = score_fn(X_df)
    print(f"  Baseline score ({scoring}): {baseline:.4f}")

    rows = []
    for label, feats in named_groups.items():
        feats = [f for f in feats if f in X_df.columns]
        if not feats:
            continue
        drops = []
        for _ in range(int(n_repeats)):
            X_perm = X_df.copy()
            idx = rng.permutation(n)
            for f in feats:
                X_perm[f] = X_df[f].iloc[idx].values
            drops.append(baseline - score_fn(X_perm))
        rows.append(
            {
                "group_label": str(label),
                "features": tuple(feats),
                "n_features": int(len(feats)),
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops)),
            }
        )

    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def build_pair12_atomic_groups(feature_names: list[str]) -> dict[str, list[str]]:
    """
    Build "atomic" permutation groups:
      - if a feature ends with _1 and the corresponding _2 exists -> permute them jointly
      - if a feature ends with _2 and _1 exists -> handled by the _1 entry (skip)
      - otherwise -> singleton group

    Returns:
      dict mapping group_label -> list of feature names.
      group_label is the base name for _1/_2 pairs (e.g., 'charges_max'),
      otherwise the feature name itself (e.g., 'temperature', 'solvent_logp').
    """
    feats = [str(f) for f in (feature_names or [])]
    feat_set = set(feats)
    groups: dict[str, list[str]] = {}

    # Some paired features should NOT be permuted jointly (we want individual permutation,
    # and later averaging in the paper group aggregation).
    no_joint_pair_bases = {
        "polytype_emb",
        "method_emb",
    }

    for f in feats:
        if f.endswith("_1"):
            base = f[:-2]
            mate = f"{base}_2"
            if base in no_joint_pair_bases:
                groups[f] = [f]
            elif mate in feat_set:
                groups[base] = [f, mate]
            else:
                groups[f] = [f]
        elif f.endswith("_2"):
            base = f[:-2]
            mate = f"{base}_1"
            if base in no_joint_pair_bases:
                groups[f] = [f]
            elif mate in feat_set:
                # handled when we encounter _1
                continue
            groups[f] = [f]
        else:
            groups[f] = [f]

    # stable ordering
    return {k: groups[k] for k in sorted(groups.keys())}


def build_pair12_with_correlation_groups(
    X_df: pd.DataFrame,
    feature_names: list[str],
    *,
    correlation_threshold: float = 0.9,
) -> dict[str, list[str]]:
    """
    Extend pair12 atomic groups with correlation-based merging.

    Steps:
      1. Build pair12 atomic groups (_1/_2 pairs jointly, polytype_emb/method_emb individual).
      2. Compute a representative value per group (mean across features in the group).
      3. Merge groups whose representatives have |corr| >= correlation_threshold.

    Returns a dict mapping group_label -> list[str] of feature names (same format as
    build_pair12_atomic_groups), with merged groups using the label of the first group
    (sorted alphabetically) in the merged set.
    """
    from collections import defaultdict

    atomic = build_pair12_atomic_groups(feature_names)
    # atomic: dict[label -> list[feat]]

    # Build representative series per group (mean of columns in group)
    available_labels = [
        lbl for lbl, feats in atomic.items() if all(f in X_df.columns for f in feats)
    ]
    if len(available_labels) < 2:
        return atomic

    rep_df = pd.DataFrame(
        {
            lbl: X_df[[f for f in atomic[lbl] if f in X_df.columns]].mean(axis=1)
            for lbl in available_labels
        }
    )

    corr = rep_df.corr().abs()

    # Union-find over group labels
    parent = {lbl: lbl for lbl in available_labels}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(available_labels):
        for j, b in enumerate(available_labels):
            if j <= i:
                continue
            if corr.loc[a, b] >= correlation_threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    # merge: keep the alphabetically first label as root
                    root = min(ra, rb)
                    child = max(ra, rb)
                    parent[child] = root

    # Collect merged groups
    root_to_labels: dict[str, list[str]] = defaultdict(list)
    for lbl in available_labels:
        root_to_labels[find(lbl)].append(lbl)

    merged: dict[str, list[str]] = {}
    for root, members in root_to_labels.items():
        all_feats = []
        seen = set()
        for lbl in sorted(members):
            for f in atomic[lbl]:
                if f not in seen:
                    all_feats.append(f)
                    seen.add(f)
        sorted_members = sorted(members)
        # For merged groups (>1 atomic group), join all member labels so the
        # barplot label lists every constituent group (e.g. "homo / ip / ip_corrected")
        group_label = " / ".join(sorted_members) if len(sorted_members) > 1 else sorted_members[0]
        merged[group_label] = all_feats

    # Include labels not available in X_df as-is (singletons / no data)
    for lbl, feats in atomic.items():
        if lbl not in available_labels:
            merged[lbl] = feats

    return {k: merged[k] for k in sorted(merged.keys())}


def aggregate_group_importance_to_named_groups(
    atomic_group_df: pd.DataFrame,
    named_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Aggregate permutation importances computed on "atomic groups" (pairs/singletons)
    into higher-level named groups for plotting/reporting.

    Aggregation:
      - mean: average of atomic group means included in the named group
      - std:  RMS of atomic group stds included in the named group

    atomic_group_df must have columns:
      group_label, features (pipe-joined or tuple), importance_mean, importance_std
    """
    if atomic_group_df is None or len(atomic_group_df) == 0:
        return pd.DataFrame(
            columns=["group_label", "features", "n_features", "importance_mean", "importance_std"]
        )

    df = atomic_group_df.copy()

    # Ensure features column is parsed into list[str]
    def _parse_feats(x):
        if isinstance(x, (tuple, list)):
            return [str(v) for v in x]
        s = str(x)
        if "|" in s:
            return [p for p in s.split("|") if p]
        return [s] if s else []

    df["_features_list"] = df["features"].apply(_parse_feats)
    df["importance_mean"] = df["importance_mean"].astype(float)
    df["importance_std"] = df["importance_std"].astype(float)

    rows = []
    for label, feats in named_groups.items():
        feats_set = set(str(f) for f in feats)
        mask = df["_features_list"].apply(lambda lst: any(f in feats_set for f in lst))
        sub = df.loc[mask]
        if len(sub) == 0:
            continue
        means = sub["importance_mean"].values
        stds = sub["importance_std"].values
        rows.append(
            {
                "group_label": str(label),
                "features": tuple(sorted(feats_set)),
                "n_features": int(len(feats_set)),
                "importance_mean": float(np.mean(means)),
                "importance_std": float(np.sqrt(np.mean(stds**2))) if len(stds) else 0.0,
            }
        )

    out = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return out


def plot_group_permutation_importance_barplot(
    results_df: pd.DataFrame,
    *,
    top_n: int = 50,
    save_path: str,
    xlabel: str,
):
    """
    Grouped barplot (no title) with gradient colors, using current matplotlib style.
    """
    top = results_df.head(int(top_n)).copy()

    try:
        from copol_prediction.analysis.plot_config import TWO_COL_WIDTH_INCH

        width = float(TWO_COL_WIDTH_INCH)
    except Exception:
        width = 7.0
    height = max(3.2, len(top) * 0.22)

    fig, ax = plt.subplots(figsize=(width, height))
    y_pos = np.arange(len(top))
    ax.barh(
        y_pos,
        top["importance_mean"].astype(float).values,
        xerr=top["importance_std"].astype(float).values,
        capsize=4,
        alpha=0.85,
        color=plt.cm.RdBu(np.linspace(0, 1, len(top))),
    )
    ax.set_yticks(y_pos)

    def _cap_first(s: str) -> str:
        s = str(s)
        return (s[:1].upper() + s[1:]) if s else s

    labels = [_cap_first(x) for x in top["group_label"].astype(str).values]
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(str(xlabel), fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis()
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Group permutation importance plot saved to: {save_path}")


def analyze_feature_redundancy(results_df, X_test, feature_names, threshold=0.001):
    """Identify potentially redundant features based on low permutation importance."""
    low_importance = results_df[results_df["importance_mean"] <= threshold].copy()
    high_importance = results_df[results_df["importance_mean"] > threshold].copy()
    if len(low_importance) > 0 and len(high_importance) > 0:
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        correlation_analysis = []
        for low_feat in low_importance["feature"]:
            max_corr = 0
            max_corr_feature = None
            for high_feat in high_importance["feature"]:
                if low_feat in X_test_df.columns and high_feat in X_test_df.columns:
                    corr = abs(X_test_df[low_feat].corr(X_test_df[high_feat]))
                    if corr > max_corr:
                        max_corr = corr
                        max_corr_feature = high_feat
            correlation_analysis.append(
                {
                    "low_importance_feature": low_feat,
                    "max_correlation": max_corr,
                    "correlated_with": max_corr_feature,
                    "importance": low_importance[low_importance["feature"] == low_feat][
                        "importance_mean"
                    ].iloc[0],
                }
            )
        correlation_df = pd.DataFrame(correlation_analysis)
        correlation_df = correlation_df.sort_values("max_correlation", ascending=False)
    else:
        correlation_df = pd.DataFrame()
    return {
        "low_importance_features": low_importance,
        "high_importance_features": high_importance,
        "correlation_analysis": correlation_df,
        "n_redundant": len(low_importance),
        "n_important": len(high_importance),
    }


def suggest_feature_removal(redundancy_analysis, correlation_threshold=0.8):
    """Suggest features for removal based on redundancy analysis."""
    suggestions = []
    correlation_df = redundancy_analysis["correlation_analysis"]
    if len(correlation_df) > 0:
        highly_correlated = correlation_df[
            correlation_df["max_correlation"] >= correlation_threshold
        ]
        for _, row in highly_correlated.iterrows():
            suggestions.append(
                {
                    "feature_to_remove": row["low_importance_feature"],
                    "reason": f'Low importance ({row["importance"]:.4f}) and highly correlated ({row["max_correlation"]:.3f}) with {row["correlated_with"]}',
                    "importance": row["importance"],
                    "correlation": row["max_correlation"],
                }
            )
    extremely_low = redundancy_analysis["low_importance_features"][
        redundancy_analysis["low_importance_features"]["importance_mean"] <= 0.0001
    ]
    for _, row in extremely_low.iterrows():
        if not any(s["feature_to_remove"] == row["feature"] for s in suggestions):
            suggestions.append(
                {
                    "feature_to_remove": row["feature"],
                    "reason": f'Extremely low importance ({row["importance_mean"]:.6f})',
                    "importance": row["importance_mean"],
                    "correlation": None,
                }
            )
    return suggestions


def run_permutation_analysis(
    models, X_test, y_test, feature_names, output_dir="output", scoring="f1"
):
    """
    Run complete per-feature permutation importance analysis (legacy entry point).
    """
    model = models
    results_df, perm_importance = calculate_permutation_importance(
        model, X_test, y_test, feature_names, scoring=scoring
    )
    results_df.to_csv(f"{output_dir}/permutation_importance_detailed.csv", index=False)
    plot_permutation_importance(
        results_df, top_n=30, save_path=f"{output_dir}/permutation_importance.png"
    )
    redundancy_analysis = analyze_feature_redundancy(
        results_df, X_test, feature_names, threshold=0.001
    )
    removal_suggestions = suggest_feature_removal(redundancy_analysis)
    if len(redundancy_analysis["correlation_analysis"]) > 0:
        redundancy_analysis["correlation_analysis"].to_csv(
            f"{output_dir}/feature_correlation_analysis.csv", index=False
        )
    return {
        "permutation_results": results_df,
        "redundancy_analysis": redundancy_analysis,
        "removal_suggestions": removal_suggestions,
        "raw_importance": perm_importance,
    }


def create_feature_importance_comparison(
    tree_importance, perm_importance, feature_names, save_path="output/importance_comparison.png"
):
    """Compare tree-based feature importance with permutation importance."""
    comparison_df = pd.DataFrame(
        {
            "feature": feature_names,
            "tree_importance": tree_importance,
            "permutation_importance": perm_importance,
        }
    )
    plt.figure(figsize=(10, 8))
    plt.scatter(
        comparison_df["tree_importance"], comparison_df["permutation_importance"], alpha=0.6
    )
    plt.xlabel("Tree-based Feature Importance")
    plt.ylabel("Permutation Feature Importance")
    plt.title("Tree-based vs Permutation Feature Importance")
    max_val = max(
        comparison_df["tree_importance"].max(), comparison_df["permutation_importance"].max()
    )
    plt.plot([0, max_val], [0, max_val], "r--", alpha=0.5)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Feature importance comparison plot saved to: {save_path}")
    return comparison_df
