#!/usr/bin/env python3
"""
Compare models for final model selection.

Models compared:
1. Lookup Model (database lookup / nearest neighbor by Tanimoto similarity)
2. XGBoost Model (final model with molecular descriptors)
3. XGBoost + Lookup Features (XGBoost with additional lookup class + distance features)
4. Voting: Lookup + XGBoost (only predict when both agree)
5. Voting: Lookup + XGBoost+Features (only predict when both agree)

For each model:
- Macro accuracy (balanced accuracy)
- Confusion matrix
- Confidence bar plot: correct vs incorrect predictions per confidence bin

For voting models:
- Number of unpredicted data points (disagreement)
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from copolpredictor.inference import CopolymerPredictor
from copol_prediction.utils import load_data_split
from copol_prediction.analysis.analyze_model import (
    compute_naive_baseline_predictions_with_similarity,
    get_class_label
)
from copolpredictor import model_training, prediction_utils
from copolpredictor.data_augmentation import augment_with_gaussian_samples
from copol_prediction.analysis.plot_config import (
    setup_plot_style,
    COMPARISON_COLORS,
    HIGHLIGHT_COLORS,
    CONFIDENCE_PLOT_CONFIG,
    TWO_COL_WIDTH_INCH,
    ONE_COL_WIDTH_INCH
)
from sklearn.metrics import (
    precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


MODEL_COLORS = {
    'Lookup': '#3A3B73',
    'XGBoost': '#e27f07',
    'XGBoost + Lookup Features': '#6a040f',
    'Voting (Lookup + XGBoost)': '#1e8db9',
    'Voting (Lookup + XGBoost+Features)': '#0a0e38',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare models for final model selection"
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="../../copol_prediction/artifacts/model_bundle",
        help="Path to base XGBoost model (final model)"
    )
    parser.add_argument(
        "--baseline-feature-model-path",
        type=str,
        default="results",
        help="Path to XGBoost + lookup features model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison",
        help="Output directory for plots"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading & validation
# ---------------------------------------------------------------------------

def load_and_validate_data():
    """
    Load the central train/validation split and validate:
    - No negative data in validation set
    - No overlapping reaction_ids between train and validation
    """
    print("=" * 60)
    print("DATA LOADING & VALIDATION")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    split_dir = os.path.join(project_root, "copol_prediction", "artifacts", "data_splits")

    df_train, df_val, df_test = load_data_split.load_train_val_test_split(split_dir=split_dir)
    load_data_split.print_split_info(split_dir=split_dir)

    # --- Validation 1: No overlapping reaction_ids between any sets ---
    train_ids = set(df_train['reaction_id'].astype(str).unique())
    val_ids = set(df_val['reaction_id'].astype(str).unique())
    test_ids = set(df_test['reaction_id'].astype(str).unique())
    
    # Check all pairwise overlaps
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    
    has_overlap = False
    if len(train_val_overlap) > 0:
        print(f"\n  ERROR: {len(train_val_overlap)} reaction_ids appear in BOTH train and validation!")
        print(f"  Overlapping IDs (first 10): {list(train_val_overlap)[:10]}")
        has_overlap = True
    
    if len(train_test_overlap) > 0:
        print(f"\n  ERROR: {len(train_test_overlap)} reaction_ids appear in BOTH train and test!")
        print(f"  Overlapping IDs (first 10): {list(train_test_overlap)[:10]}")
        has_overlap = True
    
    if len(val_test_overlap) > 0:
        print(f"\n  ERROR: {len(val_test_overlap)} reaction_ids appear in BOTH validation and test!")
        print(f"  Overlapping IDs (first 10): {list(val_test_overlap)[:10]}")
        has_overlap = True
    
    if has_overlap:
        raise ValueError("Splits have overlapping reaction_ids! This should not happen.")
    else:
        print(f"\n  ✓ OK: No overlapping reaction_ids")
        print(f"    Train:      {len(train_ids)} unique reaction_ids")
        print(f"    Validation: {len(val_ids)} unique reaction_ids")
        print(f"    Test:       {len(test_ids)} unique reaction_ids")
        print(f"    Total:      {len(train_ids) + len(val_ids) + len(test_ids)} unique reaction_ids")

    # --- Validation 2: No negative data in validation ---
    # Negative data would have r_product_class values that indicate artificial data,
    # or come from the negative data file. Check if there's a marker column.
    # The split itself is from the original dataset (r1r2 >= 0), so there should be
    # no artificial negatives. Verify all validation classes are valid (0, 1, 2).
    val_classes = df_val['r_product_class'].unique()
    print(f"  Validation set classes: {sorted(val_classes)}")

    # Check that no negative / artificial data leaked into validation
    # Negative data typically has specific markers or comes from a separate file.
    # The split_info confirms filter "r1r2 >= 0" was applied.
    if 'is_negative' in df_val.columns:
        n_neg = df_val['is_negative'].sum()
        if n_neg > 0:
            print(f"\n  WARNING: {n_neg} negative data points found in validation set! Removing...")
            df_val = df_val[~df_val['is_negative']].reset_index(drop=True)
        else:
            print(f"  OK: No negative data in validation set (checked is_negative column)")
    else:
        # No marker column -- negative data was never part of the base split
        print(f"  OK: Validation set comes from base split (filter: r1r2 >= 0, no negative data)")

    print(f"\n  Final sizes - Train: {len(df_train)} | Validation: {len(df_val)}")
    print("=" * 60)
    return df_train, df_val


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def filter_training_data_for_lookup(df_train, remove_specialized=False):
    """
    Filter training data for lookup predictions.
    
    Args:
        df_train: Training dataframe
        remove_specialized: Whether to remove specialized datapoints
        
    Returns:
        Filtered training dataframe
    """
    df_filtered = df_train.copy()
    
    if remove_specialized:
        # Load specialized filter classifications
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
        spec_path = os.path.join(
            project_root,
            "copol_prediction/filter/llm_specialized_filter/classified_output.csv"
        )
        
        if os.path.exists(spec_path):
            df_spec = pd.read_csv(spec_path)
            if {'specialized_filter', 'reaction_id'}.issubset(df_spec.columns):
                df_spec = df_spec[['reaction_id', 'specialized_filter']].rename(
                    columns={'specialized_filter': 'llm_specialized_filter'}
                )
                df_filtered = df_filtered.merge(df_spec, on='reaction_id', how='left')
                
                if 'llm_specialized_filter' in df_filtered.columns:
                    before_count = len(df_filtered)
                    df_filtered = df_filtered[df_filtered['llm_specialized_filter'] != 'specialized']
                    removed_count = before_count - len(df_filtered)
                    if removed_count > 0:
                        print(f"  Removed {removed_count} specialized datapoints from lookup training set")
    
    return df_filtered.reset_index(drop=True)


def get_lookup_predictions(df_test, df_train, remove_specialized=False):
    """
    Get lookup (nearest-neighbor) predictions and similarities.
    
    Args:
        df_test: Test/validation dataframe
        df_train: Training dataframe
        remove_specialized: Whether to filter specialized datapoints from training
    """
    print("\nComputing Lookup predictions (Tanimoto nearest neighbor)...")
    
    # Filter training data if needed
    if remove_specialized:
        df_train_filtered = filter_training_data_for_lookup(df_train, remove_specialized=True)
    else:
        df_train_filtered = df_train
    
    y_train = df_train_filtered['r_product_class'].astype(int).values

    from copolpredictor import prediction_utils
    feature_cols = [c for c in prediction_utils.feature_columns if c in df_train_filtered.columns]

    pred, sim = compute_naive_baseline_predictions_with_similarity(
        df_test, df_train_filtered, y_train, feature_cols
    )
    print(f"  Training set size: {len(df_train_filtered)} (filtered: {remove_specialized})")
    print(f"  Predictions: {len(pred)} | Similarity range: [{sim.min():.3f}, {sim.max():.3f}]")
    return pred, sim


def train_xgboost_no_filters(df_train, features, random_state=42, n_iter=15):
    """
    Train XGBoost model without any filters (no augmentation, no specialized filter, no negative data).
    Used for confidence threshold sweep plot comparison.
    """
    print(f"  DEBUG train_xgboost_no_filters: Input df_train shape: {df_train.shape}")
    
    # Ensure no augmented rows are present
    if 'r1r2_variant_source' in df_train.columns:
        before = len(df_train)
        print(f"  DEBUG: Filtering augmented rows (before: {before})")
        df_train = df_train[df_train['r1r2_variant_source'] == 'original'].copy()
        print(f"  DEBUG: After filtering (after: {len(df_train)}, removed: {before - len(df_train)})")
    else:
        print(f"  DEBUG: No 'r1r2_variant_source' column - using all {len(df_train)} rows")
    
    X_train = df_train[features]
    y_train = df_train['r_product_class'].astype(int).values
    groups = df_train['reaction_id'].astype(str).values
    
    print(f"  DEBUG: Training with {len(X_train)} samples, {len(features)} features")
    print(f"  DEBUG: Class distribution: {pd.Series(y_train).value_counts().sort_index().to_dict()}")
    
    class_weights = model_training.calculate_class_weights(y_train)
    
    param_grid = {
        'n_estimators': [500, 600, 700],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.04, 0.05, 0.06],
        'subsample': [0.85, 0.9, 0.95],
        'colsample_bytree': [0.85, 0.9, 1.0],
        'reg_alpha': [0.0, 0.1, 0.3],
        'reg_lambda': [1.0, 1.5, 2.0],
        'min_child_weight': [2, 3, 5],
        'gamma': [0.3, 0.5, 0.7],
    }
    
    result = model_training.train_xgboost_with_cv(
        X_train=X_train, y_train=y_train, groups=groups,
        param_grid=param_grid, n_iter=n_iter,
        cv=5, random_state=random_state,
        class_weights=class_weights, n_jobs=-1,
    )
    
    model = model_training.train_final_model(
        X_train=X_train, y_train=y_train,
        params=result['best_params'],
        class_weights=class_weights,
        random_state=random_state,
    )
    
    return model, features


def get_xgboost_predictions(model_path, df_test, use_no_filters_model=False, df_train=None):
    """
    Get predictions from the base XGBoost model.
    
    Args:
        model_path: Path to model bundle
        df_test: Test dataframe
        use_no_filters_model: If True, train a new model without filters instead of loading
        df_train: Training dataframe (required if use_no_filters_model=True)
    """
    if use_no_filters_model and df_train is not None:
        print(f"\nTraining XGBoost model WITHOUT filters (no augmentation, no specialized filter, no negative data)...")
        print(f"  DEBUG get_xgboost_predictions: Input df_train shape: {df_train.shape}")
        
        # Ensure df_train is not augmented - filter out augmented rows if present
        df_train_clean = df_train.copy()
        if 'r1r2_variant_source' in df_train_clean.columns:
            # Remove augmented rows (keep only originals)
            before = len(df_train_clean)
            print(f"  DEBUG: Found 'r1r2_variant_source' column, value counts: {df_train_clean['r1r2_variant_source'].value_counts().to_dict()}")
            df_train_clean = df_train_clean[df_train_clean['r1r2_variant_source'] == 'original'].copy()
            if len(df_train_clean) < before:
                print(f"  Removed {before - len(df_train_clean)} augmented rows (keeping only originals)")
            else:
                print(f"  DEBUG: No augmented rows found (all rows are 'original')")
        else:
            print(f"  DEBUG: 'r1r2_variant_source' column NOT found - assuming all data is original")
        
        print(f"  DEBUG: Using {len(df_train_clean)} samples for training")
        features = [c for c in prediction_utils.feature_columns if c in df_train_clean.columns]
        print(f"  DEBUG: Training model NOW (this will take a while)...")
        model, features = train_xgboost_no_filters(df_train_clean, features, random_state=42, n_iter=15)
        print(f"  DEBUG: Model training completed!")
        
        # Simple predictor wrapper
        X_test = df_test[features]
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        
        max_proba = np.max(proba, axis=1)
        sorted_proba = np.sort(proba, axis=1)
        margin = sorted_proba[:, -1] - sorted_proba[:, -2] if proba.shape[1] > 1 else max_proba
        confidence = 0.7 * max_proba + 0.3 * margin
        
        return pred, confidence, proba
    else:
        print(f"\nLoading XGBoost model from {model_path}...")
        predictor = CopolymerPredictor(model_path)
        X_test = df_test[predictor.features]
        result = predictor.predict_with_confidence(X_test)
        y_proba = predictor.model.predict_proba(X_test)
        return result['predictions'], result['confidence'], y_proba


def train_xgboost_plus_lookup_no_filters(df_train, base_features, random_state=42, n_iter=15):
    """
    Train XGBoost + Lookup Features model without any filters.
    First computes lookup features, then trains XGBoost with base + lookup features.
    """
    print(f"  DEBUG train_xgboost_plus_lookup_no_filters: Input df_train shape: {df_train.shape}")
    
    # Ensure df_train is not augmented - filter out augmented rows if present
    df_train_clean = df_train.copy()
    if 'r1r2_variant_source' in df_train_clean.columns:
        # Remove augmented rows (keep only originals)
        before = len(df_train_clean)
        print(f"  DEBUG: Filtering augmented rows (before: {before})")
        print(f"  DEBUG: Value counts: {df_train_clean['r1r2_variant_source'].value_counts().to_dict()}")
        df_train_clean = df_train_clean[df_train_clean['r1r2_variant_source'] == 'original'].copy()
        if len(df_train_clean) < before:
            print(f"  Removed {before - len(df_train_clean)} augmented rows (keeping only originals)")
        else:
            print(f"  DEBUG: No augmented rows found (all rows are 'original')")
    else:
        print(f"  DEBUG: No 'r1r2_variant_source' column - using all {len(df_train_clean)} rows")
    
    # Compute lookup features for training set
    y_train = df_train_clean['r_product_class'].astype(int).values
    feature_cols = [c for c in base_features if c in df_train_clean.columns]
    
    # Use training set itself for lookup (no filtering)
    baseline_pred, baseline_sim = compute_naive_baseline_predictions_with_similarity(
        df_train_clean, df_train_clean, y_train, feature_cols
    )
    
    df_train_ext = df_train_clean.copy()
    df_train_ext['baseline_class_0'] = (baseline_pred == 0).astype(int)
    df_train_ext['baseline_class_1'] = (baseline_pred == 1).astype(int)
    df_train_ext['baseline_class_2'] = (baseline_pred == 2).astype(int)
    df_train_ext['baseline_distance'] = np.clip(1.0 - baseline_sim, 0.0, 1.0)
    
    # All features: base + lookup features
    all_features = base_features + ['baseline_class_0', 'baseline_class_1', 'baseline_class_2', 'baseline_distance']
    all_features = [f for f in all_features if f in df_train_ext.columns]
    
    model, _ = train_xgboost_no_filters(df_train_ext, all_features, random_state=random_state, n_iter=n_iter)
    return model, all_features


def get_xgboost_plus_lookup_predictions(model_path, df_test, df_train, remove_specialized=False, use_no_filters_model=False):
    """
    Get predictions from XGBoost model trained with additional lookup features
    (baseline_class_0/1/2 + baseline_distance).
    
    Args:
        model_path: Path to model bundle
        df_test: Test/validation dataframe
        df_train: Training dataframe
        remove_specialized: Whether to filter specialized datapoints from training
        use_no_filters_model: If True, train a new model without filters instead of loading
    """
    if use_no_filters_model:
        print(f"\nTraining XGBoost + Lookup Features model WITHOUT filters...")
        base_features = [c for c in prediction_utils.feature_columns if c in df_train.columns]
        model, all_features = train_xgboost_plus_lookup_no_filters(df_train, base_features, random_state=42, n_iter=15)
        
        # Compute lookup features for test set
        y_train = df_train['r_product_class'].astype(int).values
        feature_cols = [c for c in base_features if c in df_train.columns]
        baseline_pred, baseline_sim = compute_naive_baseline_predictions_with_similarity(
            df_test, df_train, y_train, feature_cols
        )
        
        df_test_ext = df_test.copy()
        df_test_ext['baseline_class_0'] = (baseline_pred == 0).astype(int)
        df_test_ext['baseline_class_1'] = (baseline_pred == 1).astype(int)
        df_test_ext['baseline_class_2'] = (baseline_pred == 2).astype(int)
        df_test_ext['baseline_distance'] = np.clip(1.0 - baseline_sim, 0.0, 1.0)
        
        X_test = df_test_ext[all_features]
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        
        max_proba = np.max(proba, axis=1)
        sorted_proba = np.sort(proba, axis=1)
        margin = sorted_proba[:, -1] - sorted_proba[:, -2] if proba.shape[1] > 1 else max_proba
        confidence = 0.7 * max_proba + 0.3 * margin
        
        return pred, confidence, proba
    else:
        print(f"\nLoading XGBoost + Lookup Features model from {model_path}...")
        predictor = CopolymerPredictor(model_path)

        # Filter training data if needed
        if remove_specialized:
            df_train_filtered = filter_training_data_for_lookup(df_train, remove_specialized=True)
        else:
            df_train_filtered = df_train

        # Compute lookup predictions for the test set to create the extra features
        y_train = df_train_filtered['r_product_class'].astype(int).values
        feature_cols = [c for c in prediction_utils.feature_columns if c in df_train_filtered.columns]
        baseline_pred, baseline_sim = compute_naive_baseline_predictions_with_similarity(
            df_test, df_train_filtered, y_train, feature_cols
        )

        df_test_ext = df_test.copy()
        df_test_ext['baseline_class_0'] = (baseline_pred == 0).astype(int)
        df_test_ext['baseline_class_1'] = (baseline_pred == 1).astype(int)
        df_test_ext['baseline_class_2'] = (baseline_pred == 2).astype(int)
        df_test_ext['baseline_distance'] = np.clip(1.0 - baseline_sim, 0.0, 1.0)

        X_test = df_test_ext[predictor.features]
        result = predictor.predict_with_confidence(X_test)
        y_proba = predictor.model.predict_proba(X_test)
        return result['predictions'], result['confidence'], y_proba


# ---------------------------------------------------------------------------
# Voting
# ---------------------------------------------------------------------------

def compute_voting(pred_a, pred_b, name_a, name_b, y_true):
    """
    Voting: only predict where both models agree.

    Returns:
        voting_pred  : np.ndarray (float, NaN where disagree)
        agree_mask   : np.ndarray[bool]
        stats        : dict with agreement / disagreement counts
    """
    title = f"Voting ({name_a} + {name_b})"
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)

    agree_mask = pred_a == pred_b
    voting_pred = pred_a.copy().astype(float)
    voting_pred[~agree_mask] = np.nan

    n_total = len(pred_a)
    n_agree = int(agree_mask.sum())
    n_disagree = n_total - n_agree

    print(f"  Total:        {n_total}")
    print(f"  Agreement:    {n_agree} ({n_agree / n_total * 100:.1f}%)")
    print(f"  Disagreement: {n_disagree} ({n_disagree / n_total * 100:.1f}%)  <-- NOT predicted")

    # Per true-class breakdown
    print(f"\n  {'True class':<14} {'Agree':>7} {'Disagree':>9} {'Agree%':>8}")
    print(f"  {'-' * 40}")
    for cls in sorted(np.unique(y_true)):
        mask_cls = y_true == cls
        n_cls = mask_cls.sum()
        n_cls_agree = (mask_cls & agree_mask).sum()
        n_cls_disagree = n_cls - n_cls_agree
        pct = n_cls_agree / n_cls * 100 if n_cls > 0 else 0
        print(f"  {get_class_label(cls, 'short'):<14} {n_cls_agree:>7} {n_cls_disagree:>9} {pct:>7.1f}%")
    print("=" * 60)

    stats = {
        'n_total': n_total,
        'n_agreement': n_agree,
        'n_disagreement': n_disagree,
    }
    return voting_pred, agree_mask, stats


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def calculate_metrics(y_true, y_pred, name):
    """Calculate macro accuracy and macro precision."""
    metrics = {
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
    }
    return metrics


def print_comparison_table(all_metrics, all_predictions, n_total):
    """Print comparison table of macro accuracy and macro precision.
    
    Note: Voting models are evaluated on the *agreed* subset only (abstention elsewhere),
    so we also print coverage (= n / n_total) to make comparisons explicit.
    """
    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    header = f"{'Model':<38} {'n':>5} {'Cover%':>7} {'Macro Acc':>10} {'Macro Prec':>11}"
    print(header)
    print("-" * 70)

    for name, m in all_metrics.items():
        pred = all_predictions.get(name)
        if pred is not None and hasattr(pred, 'dtype') and pred.dtype == float:
            n = int((~np.isnan(pred)).sum())
        else:
            n = n_total
        cover = (n / n_total * 100.0) if n_total else 0.0
        print(
            f"{name:<38} {n:>5} {cover:>6.1f}% "
            f"{m['balanced_accuracy']:>10.4f} "
            f"{m['precision_macro']:>11.4f}"
        )
    print("=" * 70)


def print_confidence_filter_table(all_predictions, all_confidences, y_true, threshold=0.7):
    """
    Print table showing effect of a confidence threshold filter.

    For each model: macro acc, macro prec after filtering,
    % of total removed, and % removed per true class.
    """
    print("\n" + "=" * 110)
    print(f"CONFIDENCE FILTER ANALYSIS  (threshold >= {threshold})")
    print("=" * 110)

    classes = sorted(np.unique(y_true))
    cls_headers = "".join(f" {'Cls ' + str(c) + ' rem%':>10}" for c in classes)
    header = (
        f"{'Model':<38} {'n_before':>8} {'n_after':>8} {'removed%':>9} "
        f"{'Macro Acc':>10} {'Macro Prec':>11}" + cls_headers
    )
    print(header)
    print("-" * 110)

    desired_order = [
        'Lookup',
        'XGBoost',
        'XGBoost + Lookup Features',
        'Voting (Lookup + XGBoost)',
        'Voting (Lookup + XGBoost+Features)',
    ]
    names = [n for n in desired_order if n in all_predictions and n in all_confidences]
    for n in all_confidences:
        if n not in names:
            names.append(n)

    for name in names:
        conf = all_confidences[name]
        pred = all_predictions[name]

        # For voting models: start from agreed-only predictions
        if name.startswith('Voting'):
            valid_base = ~np.isnan(pred)
        else:
            valid_base = np.ones(len(pred), dtype=bool)

        conf_valid = conf[valid_base]
        pred_valid = pred[valid_base].astype(float)
        yt_valid = y_true[valid_base]

        # For voting, conf may contain NaN for disagreed points
        not_nan = ~np.isnan(conf_valid)
        conf_valid = conf_valid[not_nan]
        pred_valid = pred_valid[not_nan]
        yt_valid = yt_valid[not_nan]

        n_before = len(pred_valid)

        # Apply threshold
        keep = conf_valid >= threshold
        n_after = int(keep.sum())
        n_removed = n_before - n_after
        pct_removed = n_removed / n_before * 100 if n_before > 0 else 0.0

        if n_after > 0:
            yt_kept = yt_valid[keep]
            yp_kept = pred_valid[keep].astype(int)
            mac = balanced_accuracy_score(yt_kept, yp_kept)
            mpr = precision_score(yt_kept, yp_kept, average='macro', zero_division=0)
        else:
            mac = float('nan')
            mpr = float('nan')

        # Per-class removal %
        cls_strs = []
        for c in classes:
            cls_mask = yt_valid == c
            n_cls = int(cls_mask.sum())
            if n_cls > 0:
                n_cls_removed = int((cls_mask & ~keep).sum())
                cls_strs.append(f"{n_cls_removed / n_cls * 100:>10.1f}")
            else:
                cls_strs.append(f"{'--':>10}")

        print(
            f"{name:<38} {n_before:>8} {n_after:>8} {pct_removed:>8.1f}% "
            f"{mac:>10.4f} {mpr:>11.4f}" + "".join(cls_strs)
        )

    print("=" * 110)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _safe_name(name):
    return name.lower().replace(' ', '_').replace('+', 'plus').replace('(', '').replace(')', '').replace('/', '_')


def _model_color(name, idx=0):
    return MODEL_COLORS.get(name, list(MODEL_COLORS.values())[idx % len(MODEL_COLORS)])


def plot_confidence_distributions(all_confidences, all_predictions, y_true, output_dir):
    """
    One figure with all models side-by-side.
    Each subplot: overlapping histograms of confidence for correct (blue)
    vs incorrect (red) predictions, plus a mean-confidence vertical line.
    """
    print("\nPlotting confidence distributions (all models side-by-side)...")
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    desired_order = [
        'Lookup',
        'XGBoost',
        'XGBoost + Lookup Features',
        'Voting (Lookup + XGBoost)',
        'Voting (Lookup + XGBoost+Features)',
    ]
    names = [n for n in desired_order if n in all_confidences]
    for n in all_confidences:
        if n not in names:
            names.append(n)

    n_models = len(names)
    if n_models == 0:
        print("  No confidence data available, skipping.")
        return

    fig, axes = plt.subplots(1, n_models,
                             figsize=(TWO_COL_WIDTH_INCH * n_models / 3, 3.5),
                             sharey=True)
    if n_models == 1:
        axes = [axes]

    n_bins = CONFIDENCE_PLOT_CONFIG.get('bins', 30)
    alpha = CONFIDENCE_PLOT_CONFIG.get('alpha', 0.6)
    ec = CONFIDENCE_PLOT_CONFIG.get('edgecolor', 'black')
    lw = CONFIDENCE_PLOT_CONFIG.get('linewidth', 0.5)

    for ax, name in zip(axes, names):
        conf = all_confidences[name]
        pred = all_predictions[name]

        # For voting models, restrict to agreed predictions
        if name.startswith('Voting'):
            valid = ~np.isnan(pred)
            conf = conf[valid]
            pred = pred[valid].astype(int)
            yt = y_true[valid]
        else:
            pred = pred.astype(int)
            yt = y_true

        correct_mask = pred == yt
        correct_conf = conf[correct_mask]
        incorrect_conf = conf[~correct_mask]

        ax.hist(correct_conf, bins=n_bins, alpha=alpha, label='Correct',
                color=COMPARISON_COLORS['correct'], edgecolor=ec, linewidth=lw)
        ax.hist(incorrect_conf, bins=n_bins, alpha=alpha, label='Incorrect',
                color=COMPARISON_COLORS['incorrect'], edgecolor=ec, linewidth=lw)

        ax.axvline(conf.mean(), color=HIGHLIGHT_COLORS['mean'], linestyle='--',
                   linewidth=1.2, label=f'Mean: {conf.mean():.3f}')

        ax.set_xlabel('Confidence', fontsize=9)
        ax.set_title(name, fontsize=9, fontweight='bold')
        ax.legend(fontsize=6)
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

        # Print stats to console
        mean_c = correct_conf.mean() if len(correct_conf) > 0 else float('nan')
        mean_ic = incorrect_conf.mean() if len(incorrect_conf) > 0 else float('nan')
        print(f"  {name}: mean conf correct={mean_c:.3f}, incorrect={mean_ic:.3f}")

    axes[0].set_ylabel('Count', fontsize=9)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(output_dir, f'confidence_distributions.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved confidence_distributions.png/.pdf")


def plot_confidence_threshold_sweep(all_predictions, all_confidences, y_true,
                                    thresholds, output_dir, reaction_ids=None):
    """
    Two side-by-side plots:
      Left plot  : Macro Accuracy per threshold (lines, one per model)
      Right plot : Number of remaining data points per threshold (lines, one per model)
    Common legend below both plots.
    """
    print("\nPlotting confidence threshold sweep...")

    # Load lamalab style directly
    style_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'copol_prediction', 'analysis', 'lamalab.mplstyle'
    )
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        setup_plot_style()

    os.makedirs(output_dir, exist_ok=True)

    desired_order = [
        'Lookup',
        'XGBoost',
        'XGBoost + Lookup Features',
        'Voting (Lookup + XGBoost)',
        'Voting (Lookup + XGBoost+Features)',
    ]
    names = [n for n in desired_order if n in all_confidences]
    for n in all_confidences:
        if n not in names:
            names.append(n)

    print(f"  Models found in all_confidences: {list(all_confidences.keys())}")
    print(f"  Models to plot: {names}")

    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH_INCH, 3.5))

    all_accs = []
    all_counts = []

    print("\n  Computing accuracy and counts for each model and threshold...")
    for name in names:
        print(f"\n  Processing {name}...")
        conf = all_confidences[name]
        pred = all_predictions[name]
        color = _model_color(name)

        print(f"    Initial: conf shape={conf.shape}, pred shape={pred.shape}, y_true shape={y_true.shape}")
        print(f"    conf: min={np.nanmin(conf):.3f}, max={np.nanmax(conf):.3f}, "
              f"NaN count={np.isnan(conf).sum()}/{len(conf)}")
        print(f"    pred: NaN count={np.isnan(pred).sum()}/{len(pred)}")

        if name.startswith('Voting'):
            valid_base = ~np.isnan(pred)
            print(f"    Voting model: {valid_base.sum()}/{len(pred)} valid predictions")
        else:
            valid_base = np.ones(len(pred), dtype=bool)
            print(f"    Non-voting model: all {len(pred)} predictions valid")

        conf_v = conf[valid_base]
        pred_v = pred[valid_base].astype(float)
        yt_v = y_true[valid_base]
        
        # Get reaction_ids for valid_base if available
        if reaction_ids is not None:
            reaction_ids_v = reaction_ids[valid_base]
        else:
            reaction_ids_v = None

        print(f"    After filtering valid_base: conf_v shape={conf_v.shape}, "
              f"conf_v NaN count={np.isnan(conf_v).sum()}/{len(conf_v)}")
        print(f"    After filtering valid_base: pred_v shape={pred_v.shape}, "
              f"pred_v NaN count={np.isnan(pred_v).sum()}/{len(pred_v)}")
        print(f"    After filtering valid_base: yt_v shape={yt_v.shape}, "
              f"yt_v unique values={np.unique(yt_v)}")

        not_nan = ~np.isnan(conf_v)
        conf_v = conf_v[not_nan]
        pred_v = pred_v[not_nan]
        yt_v = yt_v[not_nan]
        if reaction_ids_v is not None:
            reaction_ids_v = reaction_ids_v[not_nan]
        
        print(f"    After removing NaN conf: conf_v shape={conf_v.shape}, "
              f"pred_v shape={pred_v.shape}, yt_v shape={yt_v.shape}")
        
        # Verify alignment: check if predictions match true labels length
        if len(conf_v) != len(pred_v) or len(pred_v) != len(yt_v):
            print(f"    ERROR: Length mismatch! conf_v={len(conf_v)}, pred_v={len(pred_v)}, yt_v={len(yt_v)}")
        
        # Check overall accuracy before threshold filtering for debugging
        if len(yt_v) > 0:
            overall_acc = balanced_accuracy_score(yt_v, pred_v.astype(int))
            print(f"    Overall accuracy (before threshold filtering): {overall_acc:.4f}")

        accs = []
        counts = []
        print(f"    Threshold sweep results (Macro Accuracy = balanced_accuracy_score):")
        for t in thresholds:
            keep = conf_v >= t
            n_kept = int(keep.sum())
            counts.append(n_kept)
            if n_kept > 0:
                yt_kept = yt_v[keep]
                yp_kept = pred_v[keep].astype(int)
                
                # Check class distribution in kept samples
                # If reaction_ids are available, deduplicate to count each reaction_id only once
                if reaction_ids_v is not None:
                    reaction_ids_kept = reaction_ids_v[keep]
                    # Create a dataframe to deduplicate by reaction_id
                    df_kept = pd.DataFrame({
                        'reaction_id': reaction_ids_kept,
                        'class': yt_kept
                    })
                    # Keep only first occurrence of each reaction_id
                    df_kept_unique = df_kept.drop_duplicates(subset='reaction_id', keep='first')
                    yt_kept_unique = df_kept_unique['class'].values
                    unique_classes, class_counts = np.unique(yt_kept_unique, return_counts=True)
                    class_dist = dict(zip(unique_classes, class_counts))
                    n_unique_reactions = len(df_kept_unique)
                    class_dist_note = f" (unique reactions: {n_unique_reactions})"
                else:
                    unique_classes, class_counts = np.unique(yt_kept, return_counts=True)
                    class_dist = dict(zip(unique_classes, class_counts))
                    class_dist_note = ""
                
                # Verify alignment
                if len(yt_kept) != len(yp_kept):
                    print(f"      ERROR: Length mismatch! yt_kept={len(yt_kept)}, yp_kept={len(yp_kept)}")
                
                # Calculate accuracy
                acc = balanced_accuracy_score(yt_kept, yp_kept)
                
                # Also calculate simple accuracy for comparison
                simple_acc = (yt_kept == yp_kept).mean()
                
                accs.append(acc)
                
                # Warn if sample size is very small or classes are missing
                warning = ""
                if n_kept < 10:
                    warning = " ⚠️ VERY SMALL SAMPLE"
                elif len(class_dist) < 3:
                    missing = set([0, 1, 2]) - set(class_dist.keys())
                    warning = f" ⚠️ MISSING CLASSES: {missing}"
                
                # Check if accuracy seems suspiciously high
                if acc > 0.95 and n_kept < 50:
                    warning += " ⚠️ SUSPICIOUSLY HIGH ACCURACY"
                
                print(f"      t={t:.1f}: n_kept={n_kept:4d}, macro_accuracy={acc:.4f}, "
                      f"simple_acc={simple_acc:.4f}, class_dist={class_dist}{class_dist_note}{warning}")
            else:
                accs.append(float('nan'))
                print(f"      t={t:.1f}: n_kept={n_kept:4d}, macro_accuracy=NaN (no samples)")

        all_accs.extend([a for a in accs if not np.isnan(a)])
        all_counts.extend(counts)

        # Left plot: Accuracy - plot all values, matplotlib will handle NaN
        # Replace NaN with None so matplotlib skips those points but keeps the line
        accs_plot = [a if not np.isnan(a) else None for a in accs]
        ax1.plot(thresholds, accs_plot, marker='o', color=color, label=name, linewidth=1.5, 
                markersize=4, markeredgewidth=0.5)
        
        # Debug: print accuracy range for each model
        valid_accs = [a for a in accs if not np.isnan(a)]
        if valid_accs:
            print(f"    Summary: macro_accuracy range [{min(valid_accs):.3f}, {max(valid_accs):.3f}], "
                  f"{len(valid_accs)}/{len(accs)} valid thresholds")
        else:
            print(f"    Summary: WARNING - no valid macro_accuracy values!")
        
        # Right plot: Counts
        ax2.plot(thresholds, counts, marker='s', color=color, label=name, linewidth=1.5,
                markersize=4, markeredgewidth=0.5)

    # Left plot: Accuracy (original range 0.4-0.7)
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Macro Accuracy')
    ax1.set_xlim(thresholds[0] - 0.03, thresholds[-1] + 0.03)  # Original range
    
    # Dynamic ylim based on actual data
    if all_accs:
        acc_min = min(all_accs)
        acc_max = max(all_accs)
        acc_range = acc_max - acc_min
        acc_pad = acc_range * 0.1 if acc_range > 0 else 0.05
        ax1.set_ylim(max(0.5, acc_min - acc_pad), min(1.0, acc_max + acc_pad))
    else:
        ax1.set_ylim(0.58, 0.82)
    
    ax1.set_xticks(thresholds)  # Original thresholds
    ax1.xaxis.set_minor_locator(plt.NullLocator())
    ax1.yaxis.set_minor_locator(plt.NullLocator())
    ax1.grid(False)  # Remove grid

    # Right plot: Counts (original range 0.4-0.7)
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Number of Predictions')
    ax2.set_xlim(thresholds[0] - 0.03, thresholds[-1] + 0.03)  # Original range
    count_min = min(all_counts) if all_counts else 0
    count_max = max(all_counts) if all_counts else 1
    count_pad = (count_max - count_min) * 0.08
    ax2.set_ylim(count_min - count_pad, count_max + count_pad)
    ax2.set_xticks(thresholds)  # Original thresholds
    ax2.xaxis.set_minor_locator(plt.NullLocator())
    ax2.yaxis.set_minor_locator(plt.NullLocator())
    ax2.grid(False)  # Remove grid

    # Add "a" and "b" labels above the plots (not inside)
    fig.text(0.0, 1.0, 'a', transform=fig.transFigure, fontsize=12, fontweight='bold',
             verticalalignment='top', horizontalalignment='left')
    fig.text(0.5, 1.0, 'b', transform=fig.transFigure, fontsize=12, fontweight='bold',
             verticalalignment='top', horizontalalignment='left')

    # Print final summary of all plotted values
    print("\n  Final summary of all plotted values:")
    print("  " + "="*70)
    for name in names:
        print(f"\n  {name}:")
        # Recompute to show final values
        conf = all_confidences[name]
        pred = all_predictions[name]
        if name.startswith('Voting'):
            valid_base = ~np.isnan(pred)
        else:
            valid_base = np.ones(len(pred), dtype=bool)
        conf_v = conf[valid_base]
        pred_v = pred[valid_base].astype(float)
        yt_v = y_true[valid_base]
        not_nan = ~np.isnan(conf_v)
        conf_v = conf_v[not_nan]
        pred_v = pred_v[not_nan]
        yt_v = yt_v[not_nan]
        
        print(f"    Threshold | Samples | Macro Accuracy")
        print(f"    ----------|---------|----------------")
        for t in thresholds:
            keep = conf_v >= t
            n_kept = int(keep.sum())
            if n_kept > 0:
                acc = balanced_accuracy_score(yt_v[keep], pred_v[keep].astype(int))
                print(f"    {t:8.1f}   | {n_kept:7d} | {acc:.4f}")
            else:
                print(f"    {t:8.1f}   | {n_kept:7d} | NaN")
    print("  " + "="*70)

    # Common legend below both plots (without box)
    fig.legend(names, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=3, frameon=False)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25, top=0.95)
    
    # Save original plot
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(output_dir, f'confidence_threshold_sweep.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved confidence_threshold_sweep.png/.pdf")


def plot_confidence_threshold_sweep_full_range(all_predictions, all_confidences, y_true,
                                               thresholds, output_dir, reaction_ids=None):
    """
    Two side-by-side plots with full X-axis range (0.0 to 1.0):
      Left plot  : Macro Accuracy per threshold (lines, one per model)
      Right plot : Number of remaining data points per threshold (lines, one per model)
    Common legend below both plots.
    """
    print("\nPlotting confidence threshold sweep (full range 0.0-1.0)...")

    # Load lamalab style directly
    style_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'copol_prediction', 'analysis', 'lamalab.mplstyle'
    )
    if os.path.exists(style_path):
        plt.style.use(style_path)
    else:
        setup_plot_style()

    os.makedirs(output_dir, exist_ok=True)

    desired_order = [
        'Lookup',
        'XGBoost',
        'XGBoost + Lookup Features',
        'Voting (Lookup + XGBoost)',
        'Voting (Lookup + XGBoost+Features)',
    ]
    names = [n for n in desired_order if n in all_confidences]
    for n in all_confidences:
        if n not in names:
            names.append(n)

    # Create figure with 2 subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH_INCH, 3.5))
    
    # Make axes thinner
    for ax in [ax1, ax2]:
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    all_accs = []
    all_counts = []

    print("\n  Computing accuracy and counts for each model and threshold...")
    for name in names:
        conf = all_confidences[name]
        pred = all_predictions[name]
        color = _model_color(name)

        if name.startswith('Voting'):
            valid_base = ~np.isnan(pred)
        else:
            valid_base = np.ones(len(pred), dtype=bool)

        conf_v = conf[valid_base]
        pred_v = pred[valid_base].astype(float)
        yt_v = y_true[valid_base]
        
        # Get reaction_ids for valid_base if available
        if reaction_ids is not None:
            reaction_ids_v = reaction_ids[valid_base]
        else:
            reaction_ids_v = None
            
        not_nan = ~np.isnan(conf_v)
        conf_v = conf_v[not_nan]
        pred_v = pred_v[not_nan]
        yt_v = yt_v[not_nan]
        if reaction_ids_v is not None:
            reaction_ids_v = reaction_ids_v[not_nan]

        accs = []
        counts = []
        for t in thresholds:
            keep = conf_v >= t
            n_kept = int(keep.sum())
            counts.append(n_kept)
            if n_kept > 0:
                acc = balanced_accuracy_score(yt_v[keep], pred_v[keep].astype(int))
                accs.append(acc)
            else:
                accs.append(float('nan'))

        all_accs.extend([a for a in accs if not np.isnan(a)])
        all_counts.extend(counts)

        # Left plot: Accuracy
        accs_plot = [a if not np.isnan(a) else None for a in accs]
        ax1.plot(thresholds, accs_plot, marker='o', color=color, label=name, linewidth=1.5, 
                markersize=4, markeredgewidth=0.5)
        
        # Right plot: Counts
        ax2.plot(thresholds, counts, marker='s', color=color, label=name, linewidth=1.5,
                markersize=4, markeredgewidth=0.5)

    # Left plot: Accuracy with X-axis 0-1
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Macro Accuracy')
    ax1.set_xlim(0.0, 1.0)
    if all_accs:
        acc_min = min(all_accs)
        acc_max = max(all_accs)
        acc_range = acc_max - acc_min
        acc_pad = acc_range * 0.1 if acc_range > 0 else 0.05
        ax1.set_ylim(max(0.5, acc_min - acc_pad), min(1.0, acc_max + acc_pad))
    else:
        ax1.set_ylim(0.58, 0.82)
    ax1.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.xaxis.set_minor_locator(plt.NullLocator())
    ax1.yaxis.set_minor_locator(plt.NullLocator())
    ax1.grid(False)  # Remove grid
    
    # Right plot: Counts with X-axis 0-1
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Number of Predictions')
    ax2.set_xlim(0.0, 1.0)
    count_min = min(all_counts) if all_counts else 0
    count_max = max(all_counts) if all_counts else 1
    count_pad = (count_max - count_min) * 0.08
    ax2.set_ylim(count_min - count_pad, count_max + count_pad)
    ax2.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax2.xaxis.set_minor_locator(plt.NullLocator())
    ax2.yaxis.set_minor_locator(plt.NullLocator())
    ax2.grid(False)  # Remove grid

    # Add "a" and "b" labels above the plots (not inside)
    fig.text(0.0, 1.0, 'a', transform=fig.transFigure, fontsize=12, fontweight='bold',
             verticalalignment='top', horizontalalignment='left')
    fig.text(0.5, 1.0, 'b', transform=fig.transFigure, fontsize=12, fontweight='bold',
             verticalalignment='top', horizontalalignment='left')
    
    # Common legend below both plots (without box)
    fig.legend(names, fontsize=7, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=3, frameon=False)
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25, top=0.95)
    
    # Save new plot with full X-axis range
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(output_dir, f'confidence_threshold_sweep_full_range.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved confidence_threshold_sweep_full_range.png/.pdf")


def plot_confusion_matrices(all_predictions, y_true, output_dir):
    """Side-by-side confusion matrices for all models."""
    print("\nPlotting confusion matrices...")
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    desired_order = [
        'Lookup',
        'XGBoost',
        'XGBoost + Lookup Features',
        'Voting (Lookup + XGBoost)',
        'Voting (Lookup + XGBoost+Features)',
    ]
    names = [n for n in desired_order if n in all_predictions]
    for n in all_predictions:
        if n not in names:
            names.append(n)

    class_names = [get_class_label(i, style='short') for i in range(3)]

    n_models = len(names)
    fig_width = max(TWO_COL_WIDTH_INCH * 1.3, 3.0 * n_models)
    fig, axes = plt.subplots(1, n_models, figsize=(fig_width, 3.5))
    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        # Make axes thinner
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        pred = all_predictions[name]

        if name.startswith('Voting'):
            valid = ~np.isnan(pred)
            y_p = pred[valid].astype(int)
            y_t = y_true[valid]
            if len(y_p) == 0:
                ax.text(0.5, 0.5, 'No agreed\npredictions', ha='center', va='center',
                        transform=ax.transAxes)
                ax.set_title(name, fontsize=9, fontweight='bold')
                continue
        else:
            y_p = pred.astype(int)
            y_t = y_true

        cm = confusion_matrix(y_t, y_p, labels=[0, 1, 2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
        disp.plot(cmap='Blues', ax=ax, values_format='d', text_kw={'fontsize': 8})
        ax.set_title(name, fontsize=9, fontweight='bold')
        ax.set_xlabel(ax.get_xlabel(), fontsize=8)
        ax.set_ylabel(ax.get_ylabel(), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(False)
        if disp.im_ is not None and disp.im_.colorbar is not None:
            disp.im_.colorbar.remove()
        fig.colorbar(disp.im_, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

    plt.tight_layout()
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(output_dir, f'confusion_matrices.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved confusion_matrices.png/.pdf")


def plot_prediction_coverage(voting_stats_list, output_dir):
    """
    Bar chart showing how many data points are predicted vs. lost
    for each voting model.
    """
    if not voting_stats_list:
        return

    print("\nPlotting prediction coverage for voting models...")
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)

    names = [s['name'] for s in voting_stats_list]
    made = [s['n_agreement'] for s in voting_stats_list]
    lost = [s['n_disagreement'] for s in voting_stats_list]
    totals = [s['n_total'] for s in voting_stats_list]

    x = np.arange(len(names))
    width = 0.5

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_INCH * 1.4, 3.5))

    bars_made = ax.bar(x, made, width, label='Predicted (agreement)',
                       color=COMPARISON_COLORS.get('correct', '#2266ac'), alpha=0.85,
                       edgecolor='black', linewidth=0.4)
    bars_lost = ax.bar(x, lost, width, bottom=made, label='Not predicted (disagreement)',
                       color=COMPARISON_COLORS.get('incorrect', '#920506'), alpha=0.85,
                       edgecolor='black', linewidth=0.4)

    for i in range(len(names)):
        t = totals[i]
        m, l = made[i], lost[i]
        ax.text(x[i], m / 2, f'{m}\n({m / t * 100:.0f}%)', ha='center', va='center',
                fontsize=7, color='white', fontweight='bold')
        if l > 0:
            ax.text(x[i], m + l / 2, f'{l}\n({l / t * 100:.0f}%)', ha='center', va='center',
                    fontsize=7, color='white', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha='right', fontsize=7)
    ax.set_ylabel('Number of Test Samples')
    ax.set_title('Voting Models – Prediction Coverage')
    ax.legend(fontsize=7, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.grid(False)
    plt.tight_layout()

    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(output_dir, f'prediction_coverage.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved prediction_coverage.png/.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print("=" * 60)
    print("MODEL COMPARISON FOR FINAL MODEL SELECTION")
    print("=" * 60)

    # Resolve output directory relative to this script (stable regardless of cwd)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(script_dir, args.output_dir)

    # --- Load model metadata to get training filters ---
    print("\nReading model training configuration...")
    try:
        # Load base XGBoost model metadata
        predictor_base = CopolymerPredictor(args.base_model_path)
        base_metadata = predictor_base.metadata.get('training_config', {})
        remove_specialized_base = base_metadata.get('specialized_removed_from_training', False)
        use_augmentation_base = base_metadata.get('augmentation_used', False)
        negative_data_base = base_metadata.get('negative_data_used', False)
        
        print(f"  Base XGBoost model config:")
        print(f"    - Specialized removed: {remove_specialized_base}")
        print(f"    - Augmentation used: {use_augmentation_base}")
        print(f"    - Negative data used: {negative_data_base}")
        
        # Load XGBoost + Lookup Features model metadata (if available)
        remove_specialized_lf = remove_specialized_base
        use_augmentation_lf = use_augmentation_base
        try:
            predictor_lf = CopolymerPredictor(args.baseline_feature_model_path)
            lf_metadata = predictor_lf.metadata.get('training_config', {})
            remove_specialized_lf = lf_metadata.get('specialized_removed_from_training', False)
            use_augmentation_lf = lf_metadata.get('augmentation_used', False)
            negative_data_lf = lf_metadata.get('negative_data_used', False)
            
            print(f"\n  XGBoost + Lookup Features model config:")
            print(f"    - Specialized removed: {remove_specialized_lf}")
            print(f"    - Augmentation used: {use_augmentation_lf}")
            print(f"    - Negative data used: {negative_data_lf}")
            
            # Warn if filters differ
            if (remove_specialized_base != remove_specialized_lf or 
                use_augmentation_base != use_augmentation_lf or
                negative_data_base != negative_data_lf):
                print(f"\n  ⚠️  WARNING: Model training filters differ!")
                print(f"     Base model and Lookup Features model were trained with different filters.")
                print(f"     Using filters from Base model for Lookup predictions.")
        except Exception as e:
            print(f"  Note: Could not load XGBoost + Lookup Features model metadata: {e}")
            print(f"        Using Base model filters for Lookup predictions.")
        
        # Use base model filters for Lookup (to match XGBoost predictions)
        remove_specialized = remove_specialized_base
        use_augmentation = use_augmentation_base
        
    except Exception as e:
        print(f"  Warning: Could not load model metadata: {e}")
        print(f"  Using default: specialized filter OFF, augmentation OFF")
        remove_specialized = False
        use_augmentation = False

    # --- Load & validate data ---
    df_train, df_val = load_and_validate_data()
    y_true = df_val['r_product_class'].astype(int).values
    reaction_ids_val = df_val['reaction_id'].astype(str).values
    n_total = len(y_true)

    all_predictions = {}
    all_metrics = {}
    all_confidences = {}
    voting_stats_list = []

    # =================================================================
    # 1. LOOKUP MODEL (nearest-neighbor baseline)
    # =================================================================
    print("\n" + "=" * 60)
    print("1. LOOKUP MODEL")
    print("=" * 60)
    lookup_pred, lookup_sim = get_lookup_predictions(df_val, df_train, remove_specialized=remove_specialized)
    all_predictions['Lookup'] = lookup_pred
    all_metrics['Lookup'] = calculate_metrics(y_true, lookup_pred, 'Lookup')
    lookup_confidence = lookup_sim.copy()
    all_confidences['Lookup'] = lookup_confidence

    # =================================================================
    # 2. XGBOOST MODEL (final model, molecular descriptors only)
    # =================================================================
    print("\n" + "=" * 60)
    print("2. XGBOOST MODEL")
    print("=" * 60)
    try:
        xgb_pred, xgb_conf, xgb_proba = get_xgboost_predictions(args.base_model_path, df_val)
        all_predictions['XGBoost'] = xgb_pred
        all_metrics['XGBoost'] = calculate_metrics(y_true, xgb_pred, 'XGBoost')
        all_confidences['XGBoost'] = xgb_conf
    except Exception as e:
        print(f"  ERROR loading XGBoost model: {e}")
        xgb_pred = None

    # =================================================================
    # 3. XGBOOST + LOOKUP FEATURES (base features + lookup class + distance)
    # =================================================================
    print("\n" + "=" * 60)
    print("3. XGBOOST + LOOKUP FEATURES")
    print("=" * 60)
    try:
        xgb_lf_pred, xgb_lf_conf, xgb_lf_proba = get_xgboost_plus_lookup_predictions(
            args.baseline_feature_model_path, df_val, df_train, remove_specialized=remove_specialized
        )
        all_predictions['XGBoost + Lookup Features'] = xgb_lf_pred
        all_metrics['XGBoost + Lookup Features'] = calculate_metrics(
            y_true, xgb_lf_pred, 'XGBoost + Lookup Features'
        )
        all_confidences['XGBoost + Lookup Features'] = xgb_lf_conf
    except Exception as e:
        print(f"  ERROR loading XGBoost + Lookup Features model: {e}")
        xgb_lf_pred = None

    # =================================================================
    # 4. VOTING: LOOKUP + XGBOOST
    # =================================================================
    if lookup_pred is not None and xgb_pred is not None:
        v_pred, v_mask, v_stats = compute_voting(
            lookup_pred, xgb_pred, 'Lookup', 'XGBoost', y_true
        )
        v_stats['name'] = 'Voting (Lookup + XGBoost)'
        voting_stats_list.append(v_stats)

        all_predictions['Voting (Lookup + XGBoost)'] = v_pred
        y_t_v = y_true[v_mask]
        y_p_v = v_pred[v_mask].astype(int)
        if len(y_t_v) > 0:
            all_metrics['Voting (Lookup + XGBoost)'] = calculate_metrics(
                y_t_v, y_p_v, 'Voting (Lookup + XGBoost)'
            )
            # Fairness check: compare both base models on the SAME agreed subset
            try:
                xgb_acc_agree = balanced_accuracy_score(y_t_v, xgb_pred[v_mask].astype(int))
                lu_acc_agree = balanced_accuracy_score(y_t_v, lookup_pred[v_mask].astype(int))
                print("\n  Agreement-subset check (same samples as voting evaluation):")
                print(f"    XGBoost balanced acc on agree subset: {xgb_acc_agree:.4f}")
                print(f"    Lookup  balanced acc on agree subset: {lu_acc_agree:.4f}")
            except Exception as e:
                print(f"  Warning: could not compute agreement-subset check: {e}")
            # Use XGBoost confidence for agreed predictions
            voting_conf = xgb_conf.copy()
            voting_conf[~v_mask] = np.nan
            all_confidences['Voting (Lookup + XGBoost)'] = voting_conf

    # =================================================================
    # 5. VOTING: LOOKUP + XGBOOST+FEATURES
    # =================================================================
    if lookup_pred is not None and xgb_lf_pred is not None:
        v_pred2, v_mask2, v_stats2 = compute_voting(
            lookup_pred, xgb_lf_pred, 'Lookup', 'XGBoost+Features', y_true
        )
        v_stats2['name'] = 'Voting (Lookup + XGBoost+Features)'
        voting_stats_list.append(v_stats2)

        all_predictions['Voting (Lookup + XGBoost+Features)'] = v_pred2
        y_t_v2 = y_true[v_mask2]
        y_p_v2 = v_pred2[v_mask2].astype(int)
        if len(y_t_v2) > 0:
            all_metrics['Voting (Lookup + XGBoost+Features)'] = calculate_metrics(
                y_t_v2, y_p_v2, 'Voting (Lookup + XGBoost+Features)'
            )
            # Fairness check: compare both base models on the SAME agreed subset
            try:
                xgb_lf_acc_agree = balanced_accuracy_score(y_t_v2, xgb_lf_pred[v_mask2].astype(int))
                lu_acc_agree = balanced_accuracy_score(y_t_v2, lookup_pred[v_mask2].astype(int))
                print("\n  Agreement-subset check (same samples as voting evaluation):")
                print(f"    XGBoost+Features balanced acc on agree subset: {xgb_lf_acc_agree:.4f}")
                print(f"    Lookup          balanced acc on agree subset: {lu_acc_agree:.4f}")
            except Exception as e:
                print(f"  Warning: could not compute agreement-subset check: {e}")
            # Use XGBoost+Features confidence for agreed predictions
            voting_conf2 = xgb_lf_conf.copy()
            voting_conf2[~v_mask2] = np.nan
            all_confidences['Voting (Lookup + XGBoost+Features)'] = voting_conf2

    # =================================================================
    # COMPARISON TABLE
    # =================================================================
    print_comparison_table(all_metrics, all_predictions, n_total)

    # =================================================================
    # CONFIDENCE FILTER ANALYSIS (tables for 0.1 .. 0.7)
    # =================================================================
    thresholds = [round(t * 0.1, 1) for t in range(4, 8)]
    for t in thresholds:
        print_confidence_filter_table(all_predictions, all_confidences, y_true, threshold=t)

    # =================================================================
    # PLOTS
    # =================================================================
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving plots to: {os.path.abspath(output_dir)}")

    # Confusion matrices
    plot_confusion_matrices(all_predictions, y_true, output_dir)

    # Confidence distributions (all models side-by-side, correct vs incorrect)
    plot_confidence_distributions(all_confidences, all_predictions, y_true, output_dir)

    # Prediction coverage for voting models
    plot_prediction_coverage(voting_stats_list, output_dir)

    # Confidence threshold sweep: macro acc + n_points vs threshold
    # Train models WITHOUT filters for this plot
    print("\n" + "=" * 60)
    print("TRAINING MODELS WITHOUT FILTERS FOR CONFIDENCE THRESHOLD SWEEP")
    print("=" * 60)
    print("Training models with basic dataset (no augmentation, no specialized filter, no negative data)...")
    
    all_predictions_no_filters = {}
    all_confidences_no_filters = {}
    
    # Ensure df_train is clean (no augmented rows) for no-filters training
    print(f"\n  DEBUG: Initial df_train shape: {df_train.shape}")
    print(f"  DEBUG: Columns in df_train: {list(df_train.columns)[:10]}...")
    
    df_train_clean = df_train.copy()
    if 'r1r2_variant_source' in df_train_clean.columns:
        before = len(df_train_clean)
        print(f"  DEBUG: Found 'r1r2_variant_source' column")
        print(f"  DEBUG: Value counts: {df_train_clean['r1r2_variant_source'].value_counts().to_dict()}")
        df_train_clean = df_train_clean[df_train_clean['r1r2_variant_source'] == 'original'].copy()
        if len(df_train_clean) < before:
            print(f"  Removed {before - len(df_train_clean)} augmented rows from training data")
            print(f"  Using {len(df_train_clean)} original samples (no augmentation)")
        else:
            print(f"  DEBUG: No augmented rows found (all rows are 'original')")
    else:
        print(f"  DEBUG: 'r1r2_variant_source' column NOT found - assuming all data is original")
        print(f"  Using {len(df_train_clean)} samples (no augmentation column present)")
    
    print(f"  DEBUG: Final df_train_clean shape: {df_train_clean.shape}")
    
    # Lookup (same as before, but without specialized filter)
    print("\n1. Lookup Model (no filters)...")
    lookup_pred_no_filters, lookup_sim_no_filters = get_lookup_predictions(df_val, df_train_clean, remove_specialized=False)
    all_predictions_no_filters['Lookup'] = lookup_pred_no_filters
    all_confidences_no_filters['Lookup'] = lookup_sim_no_filters
    
    # XGBoost (train without filters - don't load optimized model)
    print("\n2. XGBoost Model (training WITHOUT filters - not using optimized model bundle)...")
    try:
        xgb_pred_no_filters, xgb_conf_no_filters, xgb_proba_no_filters = get_xgboost_predictions(
            args.base_model_path, df_val, use_no_filters_model=True, df_train=df_train_clean
        )
        all_predictions_no_filters['XGBoost'] = xgb_pred_no_filters
        all_confidences_no_filters['XGBoost'] = xgb_conf_no_filters
        print(f"  ✓ Trained XGBoost model without filters")
    except Exception as e:
        print(f"  ERROR training XGBoost model without filters: {e}")
        import traceback
        traceback.print_exc()
        xgb_pred_no_filters = None
    
    # XGBoost + Lookup Features (train without filters)
    print("\n3. XGBoost + Lookup Features (training without filters)...")
    try:
        xgb_lf_pred_no_filters, xgb_lf_conf_no_filters, xgb_lf_proba_no_filters = get_xgboost_plus_lookup_predictions(
            args.baseline_feature_model_path, df_val, df_train_clean, remove_specialized=False, use_no_filters_model=True
        )
        all_predictions_no_filters['XGBoost + Lookup Features'] = xgb_lf_pred_no_filters
        all_confidences_no_filters['XGBoost + Lookup Features'] = xgb_lf_conf_no_filters
    except Exception as e:
        print(f"  ERROR training XGBoost + Lookup Features model without filters: {e}")
        xgb_lf_pred_no_filters = None
    
    # Voting models (without filters)
    if lookup_pred_no_filters is not None and xgb_pred_no_filters is not None:
        v_pred_no_filters, v_mask_no_filters, v_stats_no_filters = compute_voting(
            lookup_pred_no_filters, xgb_pred_no_filters, 'Lookup', 'XGBoost', y_true
        )
        all_predictions_no_filters['Voting (Lookup + XGBoost)'] = v_pred_no_filters
        voting_conf_no_filters = xgb_conf_no_filters.copy()
        voting_conf_no_filters[~v_mask_no_filters] = np.nan
        all_confidences_no_filters['Voting (Lookup + XGBoost)'] = voting_conf_no_filters
    
    if lookup_pred_no_filters is not None and xgb_lf_pred_no_filters is not None:
        v_pred2_no_filters, v_mask2_no_filters, v_stats2_no_filters = compute_voting(
            lookup_pred_no_filters, xgb_lf_pred_no_filters, 'Lookup', 'XGBoost+Features', y_true
        )
        all_predictions_no_filters['Voting (Lookup + XGBoost+Features)'] = v_pred2_no_filters
        voting_conf2_no_filters = xgb_lf_conf_no_filters.copy()
        voting_conf2_no_filters[~v_mask2_no_filters] = np.nan
        all_confidences_no_filters['Voting (Lookup + XGBoost+Features)'] = voting_conf2_no_filters
    
    # Original plot: 0.4 to 0.7 (old name) - using models WITHOUT filters
    thresholds_original = [round(t * 0.1, 1) for t in range(4, 8)]  # 0.4, 0.5, 0.6, 0.7
    plot_confidence_threshold_sweep(
        all_predictions_no_filters, all_confidences_no_filters, y_true, thresholds_original, output_dir,
        reaction_ids=reaction_ids_val
    )
    
    # Full range plot: 0.0 to 1.0 (new name) - using models WITHOUT filters
    thresholds_full = [round(t * 0.1, 1) for t in range(0, 11)]  # 0.0, 0.1, 0.2, ..., 1.0
    plot_confidence_threshold_sweep_full_range(
        all_predictions_no_filters, all_confidences_no_filters, y_true, thresholds_full, output_dir,
        reaction_ids=reaction_ids_val
    )

    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE!")
    print("=" * 60)
    print(f"Plots saved to: {output_dir}/")


if __name__ == "__main__":
    main()
