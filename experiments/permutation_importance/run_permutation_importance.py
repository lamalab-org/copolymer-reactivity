#!/usr/bin/env python3
"""
SHAP feature importance (voting model, validation set, group-based).

- Loads the final model bundle (same as compare_models / train_final_model).
- Uses the voting model: only validation samples where XGBoost and Lookup agree.
- Applies same training filters for Lookup as the final model (e.g. specialized removed).
- SHAP importance by feature groups: highly correlated features are grouped together
  (mean absolute SHAP per group).

Usage:
  python run_permutation_importance.py [--model-path PATH] [--correlation-threshold 0.9]
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, '../..'))
sys.path.insert(0, os.path.join(_script_dir, '../../copol_prediction'))
sys.path.insert(0, os.path.join(_script_dir, '..'))

from copolpredictor.inference import CopolymerPredictor
from copolpredictor import prediction_utils, model_training
from utils.load_data_split import load_train_val_test_split
from baseline.compare_models import get_lookup_predictions

# Local analysis (same experiment folder)
sys.path.insert(0, _script_dir)
from permutation_analysis import (
    build_feature_groups,
    calculate_shap_importance_by_groups,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="SHAP feature importance (voting model, validation set, group-based)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(_script_dir, "../../copol_prediction/artifacts/model_bundle"),
        help="Path to final model bundle",
    )
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--top-n", type=int, default=50, help="Number of top feature groups to plot")
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.85,
        help="Min absolute correlation to put two features in same group. Use 1.0 for no grouping (each feature alone).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum samples for SHAP computation (default: 500, use more for accuracy but slower)",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default=None,
        help="Directory with train.csv, val.csv, test.csv (default: copol_prediction/artifacts/data_splits). Use artifacts/data_splits_full_features for full feature_columns_all.",
    )
    parser.add_argument(
        "--train-full-features-model",
        action="store_true",
        help="Train an XGBoost with feature_columns_all on the given split (use with --split-dir artifacts/data_splits_full_features). Saves to output-dir/full_features_model_bundle. Does not change the final model.",
    )
    parser.add_argument(
        "--hyperparam-iter",
        type=int,
        default=15,
        help="Hyperparameter search iterations when using --train-full-features-model (default: 15)",
    )
    return parser.parse_args()


def get_split_dir(split_dir_arg=None):
    if split_dir_arg:
        if os.path.isdir(split_dir_arg):
            return split_dir_arg
        copol_dir = os.path.join(_script_dir, "../../copol_prediction")
        return os.path.join(copol_dir, split_dir_arg)
    copol_dir = os.path.join(_script_dir, "../../copol_prediction")
    return os.path.join(copol_dir, "artifacts", "data_splits")


def load_data(split_dir_arg=None):
    """Load train/val/test; return df_train, df_val, df_test."""
    split_dir = get_split_dir(split_dir_arg)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Split directory not found: {split_dir}. Create the central split first."
        )
    df_train, df_val, df_test = load_train_val_test_split(split_dir=split_dir)
    return df_train, df_val, df_test


def train_full_features_model(df_train, output_dir, random_state=42, hyperparam_iter=15):
    """
    Train XGBoost with feature_columns_all on the given training data.
    Saves bundle to output_dir/full_features_model_bundle. Returns path to that bundle.
    """
    features = [c for c in prediction_utils.feature_columns_all if c in df_train.columns]
    if len(features) < len(prediction_utils.feature_columns_all):
        print(f"  Using {len(features)} of {len(prediction_utils.feature_columns_all)} feature_columns_all present in data")
    X_train = df_train[features]
    y_train = df_train["r_product_class"].astype(int).values
    groups = df_train["reaction_id"].astype(str).values

    class_weights = model_training.calculate_class_weights(y_train)
    param_grid = {
        "n_estimators": [300, 500, 600],
        "max_depth": [4, 5, 6],
        "learning_rate": [0.04, 0.05, 0.06],
        "subsample": [0.85, 0.9, 0.95],
        "colsample_bytree": [0.85, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 0.3],
        "reg_lambda": [1.0, 1.5, 2.0],
        "min_child_weight": [2, 3, 5],
        "gamma": [0.3, 0.5, 0.7],
    }
    print("  Running hyperparameter search...")
    train_results = model_training.train_xgboost_with_cv(
        X_train=X_train,
        y_train=y_train,
        groups=groups,
        param_grid=param_grid,
        n_iter=hyperparam_iter,
        cv=5,
        random_state=random_state,
        class_weights=class_weights,
        n_jobs=-1,
    )
    print("  Training final model on full training set...")
    final_model = model_training.train_final_model(
        X_train=X_train,
        y_train=y_train,
        params=train_results["best_params"],
        class_weights=class_weights,
        random_state=random_state,
    )
    bundle_dir = os.path.join(output_dir, "full_features_model_bundle")
    model_training.save_model_bundle(
        model=final_model,
        feature_list=features,
        class_labels=[0, 1, 2],
        out_dir=bundle_dir,
        metadata={
            "experiment": "permutation_importance_full_features",
            "feature_set": "feature_columns_all",
            "training_config": {
                "specialized_removed_from_training": False,
                "augmentation_used": False,
                "negative_data_used": False,
            },
        },
    )
    print(f"  Saved full-features model to {bundle_dir}")
    return bundle_dir


def get_voting_subset(df_val, df_train, predictor, remove_specialized):
    """
    Restrict validation to samples where XGBoost and Lookup agree (voting model).
    Returns (X_val_voting, y_val_voting, df_val_voting, n_agree, n_total).
    """
    X_val = df_val[predictor.features]
    y_val = df_val["r_product_class"].astype(int).values

    xgb_pred = predictor.predict(X_val)
    lookup_pred, _ = get_lookup_predictions(df_val, df_train, remove_specialized=remove_specialized)
    agree = (xgb_pred == lookup_pred)

    n_total = len(agree)
    n_agree = int(agree.sum())
    df_val_voting = df_val.loc[agree].reset_index(drop=True)
    X_val_voting = df_val_voting[predictor.features]
    y_val_voting = df_val_voting["r_product_class"].astype(int).values

    return X_val_voting, y_val_voting, df_val_voting, n_agree, n_total


def format_feature_name(name):
    """Format feature name for display."""
    name = str(name)
    name = name.replace("polytype_emb_1", "polymerization type emb. 1")
    name = name.replace("polytype_emb_2", "polymerization type emb. 2")
    name = name.replace("method_emb_1", "polymerization method emb. 1")
    name = name.replace("method_emb_2", "polymerization method emb. 2")
    name = name.replace("polytype_emb", "polymerization type emb.")
    name = name.replace("method_emb", "polymerization method emb.")
    if "delta_HOMO_LUMO" in name or "delta_homo_lumo" in name:
        name = name.replace("delta_HOMO_LUMO", "Δ HOMO-LUMO").replace("delta_homo_lumo", "Δ HOMO-LUMO")
        name = name.replace("_AA", " 1-1").replace("_AB", " 1-2").replace("_BA", " 2-1").replace("_BB", " 2-2")
    name = name.replace("_", " ")
    return name


def plot_group_importance_barplot(results_df, output_dir, top_n=50):
    """Bar plot for group-based SHAP importance (group_label on y-axis)."""
    n_groups = len(results_df)
    n_plot = min(top_n, n_groups)
    top = results_df.head(top_n).copy()
    if n_plot < n_groups:
        print(f"  Plotting top {n_plot} of {n_groups} groups (--top-n={top_n})")
    else:
        print(f"  Plotting all {n_plot} groups")
    top["display_label"] = top["group_label"].apply(format_feature_name)

    TWO_COL_WIDTH_INCH = 7
    height = max(4, len(top) * 0.2)
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, height))

    y_pos = np.arange(len(top))
    ax.barh(
        y_pos,
        top["importance_mean"],
        xerr=top["importance_std"],
        capsize=4,
        alpha=0.8,
        color=plt.cm.RdBu(np.linspace(0, 1, len(top))),
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["display_label"], fontsize=7)
    ax.set_xlabel("SHAP importance (mean |SHAP value|)", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis()
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"shap_importance_barplot.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved bar plot to {os.path.join(output_dir, 'shap_importance_barplot.png')}")
    return os.path.join(output_dir, "shap_importance_barplot.png")


def plot_group_importance_beeswarm(results_df, shap_values_per_group, feature_values_per_group, output_dir, top_n=50):
    """Beeswarm plot showing individual SHAP values per sample, colored by feature values."""
    n_groups = len(results_df)
    n_plot = min(top_n, n_groups)
    top = results_df.head(top_n).copy()
    if n_plot < n_groups:
        print(f"  Plotting top {n_plot} of {n_groups} groups (--top-n={top_n})")
    else:
        print(f"  Plotting all {n_plot} groups")
    top["display_label"] = top["group_label"].apply(format_feature_name)

    TWO_COL_WIDTH_INCH = 7
    height = max(4, len(top) * 0.2)
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, height))

    y_pos = np.arange(len(top))
    
    # Set random seed for reproducible jitter
    np.random.seed(42)
    
    # Normalize feature values for color mapping (per group)
    # We'll use a diverging colormap: red for high values, blue for low values
    from matplotlib.colors import Normalize
    
    # Plot individual points (beeswarm-style)
    for i, (_, row) in enumerate(top.iterrows()):
        group_label = row["group_label"]
        if group_label not in shap_values_per_group:
            print(f"  Warning: {group_label} not found in shap_values_per_group")
            continue
        if group_label not in feature_values_per_group:
            print(f"  Warning: {group_label} not found in feature_values_per_group")
            continue
            
        shap_vals = shap_values_per_group[group_label]
        feature_vals = feature_values_per_group[group_label]
        
        # Ensure arrays are 1D
        shap_vals = np.asarray(shap_vals).flatten()
        feature_vals = np.asarray(feature_vals).flatten()
        
        if len(shap_vals) == 0 or len(feature_vals) == 0:
            print(f"  Warning: {group_label} has empty values")
            continue
        if len(shap_vals) != len(feature_vals):
            print(f"  Warning: Length mismatch for {group_label}: shap_vals={len(shap_vals)}, feature_vals={len(feature_vals)}")
            continue
            
        # Normalize feature values for this group (0-1 scale)
        f_min, f_max = feature_vals.min(), feature_vals.max()
        if f_max > f_min:
            feature_vals_norm = (feature_vals - f_min) / (f_max - f_min)
        else:
            feature_vals_norm = np.ones_like(feature_vals) * 0.5
        
        # Add small jitter to y-axis for better visibility
        y_jitter = np.random.normal(y_pos[i], 0.05, size=len(shap_vals))
        
        # Color by feature value: use RdYlBu_r (red=high, blue=low) or similar
        # Map normalized feature values to colors
        colors = plt.cm.RdYlBu_r(feature_vals_norm)
        
        ax.scatter(
            shap_vals,
            y_jitter,
            alpha=0.5,
            s=12,
            c=colors,
            edgecolors='none',
        )
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["display_label"], fontsize=7)
    ax.set_xlabel("SHAP value (|SHAP| per sample)", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis()
    ax.grid(False, axis='y')
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Feature value\n(high → low)", fontsize=7, rotation=0, labelpad=15)
    cbar.ax.tick_params(labelsize=6)
    
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"shap_importance_beeswarm.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved beeswarm plot to {os.path.join(output_dir, 'shap_importance_beeswarm.png')}")
    return os.path.join(output_dir, "shap_importance_beeswarm.png")


def run_shap_analysis(predictor, df_train, X_val_voting, y_val_voting, config):
    """Build feature groups from training data; run group-based SHAP on validation voting set.
    Prefers feature_columns_all from prediction_utils; falls back to model features if data were
    preprocessed with a reduced set and don't contain feature_columns_all."""
    model_features = set(predictor.features)
    in_data = set(df_train.columns)
    # Prefer feature_columns_all; only features that exist in data and are used by the model
    wanted = prediction_utils.feature_columns_all
    feature_names = [c for c in wanted if c in in_data and c in model_features]
    used_fallback = False
    partial_feature_columns_all = False  # True if some feature_columns_all are missing in data
    if not feature_names:
        # Data likely preprocessed with reduced features – use what's there
        feature_names = [c for c in predictor.features if c in in_data]
        used_fallback = True
    else:
        if len(feature_names) < len(wanted):
            partial_feature_columns_all = True
    print(f"  Analyzing {len(feature_names)} features")
    # Build groups from training data (avoid leakage)
    X_train = df_train[feature_names]
    feature_groups = build_feature_groups(
        X_train,
        feature_names,
        correlation_threshold=config["correlation_threshold"],
    )
    n_groups = len(feature_groups)
    n_singles = sum(1 for g in feature_groups if len(g) == 1)
    print(f"  Feature groups: {n_groups} (singletons: {n_singles}, multi-feature: {n_groups - n_singles})")
    # Show group membership for features that might be grouped (e.g. temperature)
    for g in feature_groups:
        if len(g) > 1 and "temperature" in g:
            print(f"  Note: 'temperature' is in a group with {len(g)-1} other(s): {g}")

    print(f"  Calculating group-based SHAP importance (max_samples={config['max_samples']})...")
    # Get the underlying XGBoost model
    xgb_model = predictor.model
    results_df, shap_values_per_group, feature_values_per_group, X_sample = calculate_shap_importance_by_groups(
        model=xgb_model,
        X_df=X_val_voting,
        feature_groups=feature_groups,
        max_samples=config["max_samples"],
    )

    # Save CSV (group_label, features as string, importance_mean, importance_std)
    out_csv = os.path.join(config["output_dir"], "shap_importance_detailed.csv")
    save_df = results_df.copy()
    save_df["features"] = save_df["features"].apply(lambda t: "|".join(t))
    save_df.to_csv(out_csv, index=False)
    print(f"  ✓ Saved {out_csv}")

    plot_path_bar = plot_group_importance_barplot(
        results_df, config["output_dir"], top_n=config["top_n"]
    )
    
    plot_path_beeswarm = plot_group_importance_beeswarm(
        results_df, shap_values_per_group, feature_values_per_group, config["output_dir"], top_n=config["top_n"]
    )

    print(f"\n  Top 10 groups by importance:")
    for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"    {i:2d}. {row['group_label']:<45} {row['importance_mean']:.6f} ± {row['importance_std']:.6f}")

    if used_fallback:
        print("\n  Note: No feature_columns_all in data (preprocessed with reduced set?). Used model features present in data.")
    elif partial_feature_columns_all:
        n_in_data = len(feature_names)
        n_total = len(wanted)
        print(f"\n  Note: Only {n_in_data} of {n_total} feature_columns_all are present in data (and in model). Analysis and plot use this subset. Missing features may be due to reduced preprocessing.")

    return {
        "shap_results": results_df,
        "plot_path_bar": plot_path_bar,
        "plot_path_beeswarm": plot_path_beeswarm,
    }


def main():
    args = parse_args()
    config = {
        "output_dir": args.output_dir,
        "random_state": args.random_state,
        "top_n": args.top_n,
        "correlation_threshold": args.correlation_threshold,
        "max_samples": args.max_samples,
    }
    split_dir_arg = args.split_dir
    if args.train_full_features_model and not split_dir_arg:
        split_dir_arg = "artifacts/data_splits_full_features"
        print("  (--train-full-features-model: using split-dir artifacts/data_splits_full_features)")

    print("=" * 60)
    print("SHAP FEATURE IMPORTANCE (VOTING MODEL, VALIDATION)")
    print("=" * 60)
    print(f"  Output: {config['output_dir']}")
    if split_dir_arg:
        print(f"  Split dir: {split_dir_arg}")
    print(f"  Correlation threshold (grouping): {config['correlation_threshold']}")
    print(f"  Max samples for SHAP: {config['max_samples']}")

    os.makedirs(config["output_dir"], exist_ok=True)

    # Load data first (needed for training full-features model)
    print("\nLoading train/validation/test split...")
    df_train, df_val, df_test = load_data(split_dir_arg=split_dir_arg)

    # Model: either train with feature_columns_all or load existing
    if args.train_full_features_model:
        print("\nTraining XGBoost with feature_columns_all (for this experiment only)...")
        model_path = train_full_features_model(
            df_train,
            config["output_dir"],
            random_state=config["random_state"],
            hyperparam_iter=args.hyperparam_iter,
        )
    else:
        model_path = args.model_path
    print(f"\nUsing model: {model_path}")

    # Load predictor
    print("Loading model bundle...")
    predictor = CopolymerPredictor(model_path)
    meta = predictor.metadata.get("training_config", {})
    remove_specialized = meta.get("specialized_removed_from_training", False)
    print(f"  Specialized removed from training (for Lookup): {remove_specialized}")
    print(f"  Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")

    # Voting subset on validation
    print("\nApplying voting filter (XGBoost + Lookup agree) on validation...")
    X_val_voting, y_val_voting, df_val_voting, n_agree, n_total = get_voting_subset(
        df_val, df_train, predictor, remove_specialized
    )
    print(f"  Validation: {n_agree}/{n_total} samples ({100 * n_agree / n_total:.1f}%) after voting")

    # Group-based SHAP importance
    print("\n" + "=" * 60)
    print("SHAP IMPORTANCE BY FEATURE GROUPS")
    print("=" * 60)
    shap_results = run_shap_analysis(
        predictor, df_train, X_val_voting, y_val_voting, config
    )

    # Metadata
    metadata = {
        "experiment": "permutation_importance",
        "timestamp": datetime.now().isoformat(),
        "model_path": os.path.abspath(model_path),
        "dataset": "validation (voting subset)",
        "n_validation_total": int(n_total),
        "n_validation_voting": int(n_agree),
        "remove_specialized_lookup": remove_specialized,
        "num_feature_groups": len(shap_results["shap_results"]),
        "correlation_threshold": config["correlation_threshold"],
        "max_samples": config["max_samples"],
        "top_groups": shap_results["shap_results"].head(10)["group_label"].tolist(),
        "trained_full_features_model": getattr(args, "train_full_features_model", False),
    }
    meta_path = os.path.join(config["output_dir"], "meta.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Saved metadata to {meta_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"  Results: {config['output_dir']}/")


if __name__ == "__main__":
    main()
