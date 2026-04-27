#!/usr/bin/env python3
"""
Feature importance experiments (Validation set).

This script supports:
  - SHAP importance (grouped) on the validation *voting subset* (XGB + Lookup agree)
  - Permutation importance (grouped) on the validation set (default: full validation)

Key points:
  - Loads the final model bundle (same as `copol_prediction/train_final_model.py` output).
  - Does **not** retrain the classifier and does **not** apply Gaussian augmentation here.
  - Grouping options include "semantic" groups that permute related feature pairs together:
      - Δ HOMO–LUMO: AA+BB (1-1 & 2-2), AB+BA (1-2 & 2-1)
      - Monomer counterparts: *_1 + *_2 (e.g. dipole_x_1 with dipole_x_2)

Usage:
  python run_permutation_importance.py --permutation
  python run_permutation_importance.py --shap
  python run_permutation_importance.py --permutation --grouping hybrid
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
    build_semantic_feature_groups,
    build_hybrid_feature_groups,
    calculate_shap_importance_by_groups,
    calculate_shap_pairwise_importance_by_groups,
    calculate_shap_average_strong_groups,
    calculate_permutation_importance_by_groups,
)


def maybe_apply_cv_prune_100(df_train: pd.DataFrame, predictor: CopolymerPredictor) -> pd.DataFrame:
    """
    If the loaded model bundle was trained with CV-pruning (100% error-rate IDs),
    apply the same reaction_id removal to the TRAIN split used as lookup pool and
    for SHAP grouping.
    """
    meta_cfg = (predictor.metadata or {}).get("training_config", {}) or {}
    prune_path = meta_cfg.get("cv_prune_100_path")
    if not prune_path:
        return df_train
    if not os.path.exists(prune_path):
        print(f"  Warning: cv_prune_100_path not found: {prune_path}. Proceeding without pruning in this experiment.")
        return df_train
    try:
        df_prune = pd.read_csv(prune_path)
        if "reaction_id" not in df_prune.columns:
            print(f"  Warning: 'reaction_id' missing in prune list: {prune_path}. Proceeding without pruning.")
            return df_train
        prune_ids = set(df_prune["reaction_id"].astype(str).tolist())
        before_rows = len(df_train)
        before_groups = df_train["reaction_id"].astype(str).nunique()
        df_train_pruned = df_train[~df_train["reaction_id"].astype(str).isin(prune_ids)].copy().reset_index(drop=True)
        after_rows = len(df_train_pruned)
        after_groups = df_train_pruned["reaction_id"].astype(str).nunique()
        print(
            f"  Applied CV-pruning (100% error-rate): removed "
            f"{before_rows - after_rows} rows / {before_groups - after_groups} groups from TRAIN lookup pool"
        )
        return df_train_pruned
    except Exception as e:
        print(f"  Warning: Failed to apply CV-pruning from {prune_path}: {e}. Proceeding without pruning.")
        return df_train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature importance experiments (SHAP + permutation), validation split"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(_script_dir, "../../copol_prediction/artifacts/model_bundle"),
        help="Path to final model bundle",
    )
    # Default to a stable path inside this experiment folder, regardless of CWD
    parser.add_argument("--output-dir", type=str, default=os.path.join(_script_dir, "results"))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--top-n",
        type=int,
        default=0,
        help="Number of top feature groups to plot (0 = plot all). Default: 0.",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=float,
        default=0.85,
        help="Min absolute correlation to put two features in same group. Use 1.0 for no grouping (each feature alone).",
    )
    parser.add_argument(
        "--grouping",
        type=str,
        choices=["correlation", "semantic", "hybrid"],
        default="hybrid",
        help="How to group features for permutation/SHAP summaries (default: hybrid).",
    )
    parser.add_argument(
        "--permutation",
        action="store_true",
        help="Run grouped permutation importance on validation set.",
    )
    parser.add_argument(
        "--permutation-n-repeats",
        type=int,
        default=25,
        help="Permutation repeats per group (default: 25).",
    )
    parser.add_argument(
        "--permutation-scoring",
        type=str,
        choices=["f1_macro", "balanced_accuracy", "accuracy"],
        default="f1_macro",
        help="Scoring metric for permutation importance (default: f1_macro).",
    )
    parser.add_argument(
        "--permutation-use-voting-subset",
        action="store_true",
        help="If set, compute permutation importance on validation voting subset instead of full validation.",
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Run SHAP analysis (grouped) on validation voting subset.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=500,
        help="Maximum samples for SHAP computation (default: 500, use more for accuracy but slower)",
    )
    parser.add_argument(
        "--pairwise-shap",
        action="store_true",
        help="Compute pairwise SHAP (0vs1, 0vs2, 1vs2) on TRAIN split and save plots/CSVs.",
    )
    parser.add_argument(
        "--pairwise-top-n",
        type=int,
        default=20,
        help="Top N groups to plot for each pairwise SHAP plot (default: 20).",
    )
    parser.add_argument(
        "--per-class-shap",
        action="store_true",
        help="Compute per-class SHAP (one plot per class) on TRAIN split and save plots/CSVs.",
    )
    parser.add_argument(
        "--avg-shap-strong-groups",
        action="store_true",
        help="Compute average |SHAP| across classes with strong semantic grouping (no error bars) on validation voting subset.",
    )
    parser.add_argument(
        "--avg-shap-top-n",
        type=int,
        default=10,
        help="Top N groups to plot for the strongly-grouped average SHAP plot (default: 10).",
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


def plot_group_importance_barplot_to_file(results_df, output_dir, *, filename_base: str, top_n: int = 10):
    """
    Same as plot_group_importance_barplot, but lets us control the output filenames.
    """
    n_groups = len(results_df)
    n_plot = min(top_n, n_groups)
    top = results_df.head(top_n).copy()
    top["display_label"] = top["group_label"].apply(format_feature_name)

    TWO_COL_WIDTH_INCH = 7
    height = max(3.2, len(top) * 0.22)
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, height))

    y_pos = np.arange(len(top))
    # Prefer quantiles/IQR if available; else fall back to mean±std
    if {"q25", "q50", "q75"}.issubset(set(top.columns)):
        center = top["q50"].astype(float).values
        q25 = top["q25"].astype(float).values
        q75 = top["q75"].astype(float).values
        xerr = np.vstack([center - q25, q75 - center])
        xlabel = "SHAP importance (median |SHAP|, IQR)"
    else:
        center = top["importance_mean"].astype(float).values
        xerr = top["importance_std"].astype(float).values
        xlabel = "SHAP importance (mean |SHAP| ± std)"
    ax.barh(
        y_pos,
        center,
        xerr=xerr,
        capsize=4,
        alpha=0.85,
        color=plt.cm.RdBu(np.linspace(0, 1, len(top))),
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["display_label"], fontsize=7)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis()
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"{filename_base}.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    return os.path.join(output_dir, f"{filename_base}.png")


def plot_group_importance_beeswarm_to_file(
    results_df,
    shap_values_per_group,
    feature_values_per_group,
    output_dir,
    *,
    filename_base: str,
    top_n: int = 10,
):
    """
    Beeswarm plot with controlled output filename.
    Uses the same visual style as plot_group_importance_beeswarm.
    """
    from matplotlib.colors import Normalize

    n_groups = len(results_df)
    n_plot = min(top_n, n_groups)
    top = results_df.head(top_n).copy()
    top["display_label"] = top["group_label"].apply(format_feature_name)

    TWO_COL_WIDTH_INCH = 7
    height = max(3.2, len(top) * 0.22)
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, height))

    y_pos = np.arange(len(top))
    np.random.seed(42)  # reproducible jitter

    for i, (_, row) in enumerate(top.iterrows()):
        group_label = row["group_label"]
        if group_label not in shap_values_per_group:
            continue
        if group_label not in feature_values_per_group:
            continue

        shap_vals = np.asarray(shap_values_per_group[group_label]).flatten()
        feature_vals = np.asarray(feature_values_per_group[group_label]).flatten()
        if len(shap_vals) == 0 or len(feature_vals) == 0 or len(shap_vals) != len(feature_vals):
            continue

        f_min, f_max = float(np.min(feature_vals)), float(np.max(feature_vals))
        if f_max > f_min:
            feature_vals_norm = (feature_vals - f_min) / (f_max - f_min)
        else:
            feature_vals_norm = np.ones_like(feature_vals) * 0.5

        y_jitter = np.random.normal(y_pos[i], 0.05, size=len(shap_vals))
        colors = plt.cm.RdYlBu_r(feature_vals_norm)

        ax.scatter(
            shap_vals,
            y_jitter,
            alpha=0.5,
            s=12,
            c=colors,
            edgecolors="none",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["display_label"], fontsize=7)
    ax.set_xlabel("SHAP value (|SHAP| per sample)", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis()
    ax.grid(False, axis="y")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Feature value\n(high → low)", fontsize=7, rotation=0, labelpad=15)
    cbar.ax.tick_params(labelsize=6)

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"{filename_base}.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return os.path.join(output_dir, f"{filename_base}.png")


def plot_group_importance_barplot_no_errorbars(
    results_df,
    output_dir,
    *,
    filename_base: str,
    top_n: int = 20,
):
    top = results_df.head(int(top_n)).copy()
    top = top.iloc[::-1]  # barh: top item at top
    TWO_COL_WIDTH_INCH = 7
    height = max(3.2, len(top) * 0.22)
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, height))
    ax.barh(
        top["group_label"].astype(str),
        top["importance_mean"].astype(float),
        color="#661124",
        alpha=0.9,
    )
    ax.set_xlabel("mean |SHAP| (avg over 3 classes)", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"{filename_base}.{ext}")
        plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return os.path.join(output_dir, f"{filename_base}.png")


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
    print("  Training final model on full training set (no augmentation; raw train rows only)...")
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
    s = str(name)

    # Group labels are often like "<feature> (+N)". We want clean display without the (+N).
    if " (+" in s and s.endswith(")"):
        s = s.split(" (+", 1)[0]

    # Explicit group labels (semantic)
    # If the group starts with dipole_* we'll show one combined dipole label.
    if s.startswith("dipole_"):
        return "Dipole moment (x, y, z; monomer 1&2)"

    # Δ HOMO–LUMO naming (keep AA/AB/BA/BB mapping)
    if "delta_HOMO_LUMO" in s or "delta_homo_lumo" in s:
        s = s.replace("delta_HOMO_LUMO", "Δ HOMO-LUMO").replace("delta_homo_lumo", "Δ HOMO-LUMO")
        s = s.replace("_AA", " (1-1)").replace("_AB", " (1-2)").replace("_BA", " (2-1)").replace("_BB", " (2-2)")
        return s

    # Remove monomer index suffix for display
    if s.endswith("_1") or s.endswith("_2"):
        s = s[:-2]

    # Solvent feature display names
    if s in ("solvent_logp", "solvent_logP"):
        return "Solvent LogP"
    if s == "solvent_TPSA":
        return "Solvent Topological Polar Surface Area"
    if s == "solvent_HBD":
        return "Solvent number of hydrogen bond donors"
    if s == "solvent_FractionCSP3":
        return "Solvent Fraction of sp3 C"

    # Embeddings
    if s == "polytype_emb":
        return "Polymerization type embedding"
    if s == "method_emb":
        return "Polymerization method embedding"

    # Core electronic descriptors
    if s == "ip":
        return "Ionization potential"
    if s == "ip_corrected":
        return "Ionization potential (corrected)"
    if s == "ea":
        return "Electron affinity"
    if s == "homo":
        return "HOMO"
    if s == "lumo":
        return "LUMO"

    # Fukui indices (rename "radical" family to "Fukui index ...")
    if s.startswith("fukui_radical_"):
        tail = s.replace("fukui_radical_", "")
        return f"Fukui index (radical) {tail}".strip()
    if s.startswith("fukui_electrophilicity_"):
        tail = s.replace("fukui_electrophilicity_", "")
        return f"Fukui index (electrophilicity) {tail}".strip()
    if s.startswith("fukui_nucleophilicity_"):
        tail = s.replace("fukui_nucleophilicity_", "")
        return f"Fukui index (nucleophilicity) {tail}".strip()

    # Default: prettify underscores
    return s.replace("_", " ")


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


def _select_feature_names_for_analysis(predictor, df_ref: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Choose feature list to analyze.
    Prefer `feature_columns_all` intersection with model+data; fall back to model features.
    """
    model_features = set(predictor.features)
    in_data = set(df_ref.columns)
    wanted = list(prediction_utils.feature_columns_all)

    # Desired: test *all* features in feature_columns_all (if present in data),
    # but only if the loaded model bundle actually supports them.
    feature_names = [c for c in wanted if c in in_data]
    missing_in_model = [c for c in feature_names if c not in model_features]

    if not feature_names:
        # Fallback: reduced dataset / legacy
        feature_names = [c for c in predictor.features if c in in_data]
        missing_in_model = []
    return feature_names, missing_in_model


def _build_groups_for_mode(
    *,
    grouping: str,
    df_train: pd.DataFrame,
    feature_names: list[str],
    correlation_threshold: float,
) -> list[list[str]]:
    if grouping == "semantic":
        return build_semantic_feature_groups(feature_names)
    if grouping == "correlation":
        return build_feature_groups(df_train[feature_names], feature_names, correlation_threshold=correlation_threshold)
    # hybrid (default)
    return build_hybrid_feature_groups(
        df_train[feature_names],
        feature_names,
        correlation_threshold=correlation_threshold,
    )


def run_permutation_importance_analysis(
    *,
    predictor,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    X_val_voting: pd.DataFrame | None,
    y_val_voting: np.ndarray | None,
    config: dict,
    use_voting_subset: bool,
):
    """
    Grouped permutation importance on validation set (optionally voting subset).
    """
    feature_names, missing_in_model = _select_feature_names_for_analysis(predictor, df_train)
    if missing_in_model:
        raise ValueError(
            "Loaded model bundle does not support all feature_columns_all. "
            f"Missing {len(missing_in_model)} features (e.g. {missing_in_model[:5]}). "
            "Use a bundle trained on feature_columns_all (e.g. run this script with "
            "--train-full-features-model and --split-dir artifacts/data_splits_full_features, "
            "or pass --model-path <.../full_features_model_bundle>)."
        )
    print(f"  Permutation importance: analyzing {len(feature_names)} features")

    groups = _build_groups_for_mode(
        grouping=config["grouping"],
        df_train=df_train,
        feature_names=feature_names,
        correlation_threshold=config["correlation_threshold"],
    )
    n_groups = len(groups)
    n_multi = sum(1 for g in groups if len(g) > 1)
    print(f"  Grouping='{config['grouping']}' -> {n_groups} groups ({n_multi} multi-feature)")

    if use_voting_subset:
        if X_val_voting is None or y_val_voting is None:
            raise ValueError("Voting subset requested, but voting data not provided.")
        X_eval = X_val_voting[feature_names]
        y_eval = y_val_voting
        dataset_label = "validation (voting subset)"
    else:
        df_val_clean = df_val.dropna(subset=feature_names + ["r_product_class"]).reset_index(drop=True)
        X_eval = df_val_clean[feature_names]
        y_eval = df_val_clean["r_product_class"].astype(int).values
        dataset_label = "validation (full)"

    print(f"  Dataset: {dataset_label}  n={len(X_eval)}")

    results_df = calculate_permutation_importance_by_groups(
        model=predictor,
        X_df=X_eval,
        y_true=y_eval,
        feature_groups=groups,
        scoring=config["permutation_scoring"],
        n_repeats=int(config["permutation_n_repeats"]),
        random_state=int(config["random_state"]),
    )

    out_csv = os.path.join(config["output_dir"], "permutation_importance_detailed.csv")
    save_df = results_df.copy()
    save_df["features"] = save_df["features"].apply(lambda t: "|".join(t))
    save_df.to_csv(out_csv, index=False)
    print(f"  ✓ Saved {out_csv}")

    # Plot (top_n == 0 -> all)
    top_n = int(config["top_n"])
    if top_n <= 0:
        top = results_df.copy()
    else:
        top = results_df.head(top_n).copy()
    top["display_label"] = top["group_label"].apply(format_feature_name)
    TWO_COL_WIDTH_INCH = 7
    height = max(3.2, len(top) * 0.22)
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, height))
    y_pos = np.arange(len(top))
    ax.barh(
        y_pos,
        top["importance_mean"].astype(float),
        xerr=top["importance_std"].astype(float),
        capsize=4,
        alpha=0.85,
        color=plt.cm.RdBu(np.linspace(0, 1, len(top))),
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["display_label"], fontsize=7)
    ax.set_xlabel(f"Permutation importance (Δ {config['permutation_scoring']}, mean ± std)", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis()
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        plt.savefig(os.path.join(config["output_dir"], f"permutation_importance_barplot.{ext}"), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved plot {os.path.join(config['output_dir'], 'permutation_importance_barplot.png')}")

    return {
        "permutation_results": results_df,
        "dataset": dataset_label,
        "n_eval": int(len(X_eval)),
        "num_groups": int(len(results_df)),
    }


def main():
    args = parse_args()
    config = {
        "output_dir": args.output_dir,
        "random_state": args.random_state,
        "top_n": args.top_n,
        "correlation_threshold": args.correlation_threshold,
        "max_samples": args.max_samples,
        "grouping": args.grouping,
        "permutation_n_repeats": args.permutation_n_repeats,
        "permutation_scoring": args.permutation_scoring,
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
    print(f"  Grouping mode: {config['grouping']}")
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
    aug_used = meta.get("augmentation_used")
    print("  SHAP: no re-training and no Gaussian augmentation in this script; explaining bundle weights as-is.")
    if aug_used is True:
        print(
            "  ⚠ Bundle reports augmentation_used=True. For consistency with final "
            "training (augmentation off by default), retrain the bundle without --use-augmentation."
        )
    elif aug_used is False:
        print("  ✓ Bundle reports augmentation_used=False.")
    else:
        print("  ℹ Bundle has no augmentation_used in metadata (legacy bundle).")
    print(f"  Specialized removed from training (for Lookup): {remove_specialized}")
    print(f"  Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")

    # Match final training setup: apply CV-pruning (100% list) to TRAIN lookup pool if present
    df_train = maybe_apply_cv_prune_100(df_train, predictor)

    # Build voting subset (needed for SHAP, optional for permutation)
    print("\nApplying voting filter (XGBoost + Lookup agree) on validation...")
    X_val_voting, y_val_voting, df_val_voting, n_agree, n_total = get_voting_subset(
        df_val, df_train, predictor, remove_specialized
    )
    print(f"  Validation voting subset: {n_agree}/{n_total} samples ({100 * n_agree / n_total:.1f}%)")

    ran_any = False
    shap_results = None
    perm_results = None

    # Default behavior: if neither flag set, run permutation (user request focus).
    if not args.permutation and not args.shap and not args.pairwise_shap and not args.per_class_shap and not args.avg_shap_strong_groups:
        args.permutation = True

    if args.permutation:
        ran_any = True
        print("\n" + "=" * 60)
        print("PERMUTATION IMPORTANCE (GROUPED)")
        print("=" * 60)
        try:
            perm_results = run_permutation_importance_analysis(
                predictor=predictor,
                df_train=df_train,
                df_val=df_val,
                X_val_voting=X_val_voting,
                y_val_voting=y_val_voting,
                config=config,
                use_voting_subset=bool(args.permutation_use_voting_subset),
            )
        except ValueError as e:
            # Common footgun: user passes the production bundle (25 features) but requests
            # feature_columns_all. Auto-switch to a full-features bundle if available.
            msg = str(e)
            wants_all = "feature_columns_all" in msg and "Missing" in msg
            full_bundle_candidate = os.path.join(config["output_dir"], "full_features_model_bundle")
            if wants_all and os.path.isdir(full_bundle_candidate):
                print(
                    "\n  ⚠ Loaded bundle does not cover feature_columns_all. "
                    f"Switching to full-features bundle at: {full_bundle_candidate}"
                )
                predictor = CopolymerPredictor(full_bundle_candidate)
                perm_results = run_permutation_importance_analysis(
                    predictor=predictor,
                    df_train=df_train,
                    df_val=df_val,
                    X_val_voting=X_val_voting,
                    y_val_voting=y_val_voting,
                    config=config,
                    use_voting_subset=bool(args.permutation_use_voting_subset),
                )
            elif wants_all and (split_dir_arg is not None) and ("data_splits_full_features" in str(split_dir_arg)):
                print(
                    "\n  ⚠ Loaded bundle does not cover feature_columns_all and no full-features "
                    "bundle found in output-dir. Training a full-features model bundle for this experiment..."
                )
                model_path = train_full_features_model(
                    df_train,
                    config["output_dir"],
                    random_state=config["random_state"],
                    hyperparam_iter=args.hyperparam_iter,
                )
                predictor = CopolymerPredictor(model_path)
                perm_results = run_permutation_importance_analysis(
                    predictor=predictor,
                    df_train=df_train,
                    df_val=df_val,
                    X_val_voting=X_val_voting,
                    y_val_voting=y_val_voting,
                    config=config,
                    use_voting_subset=bool(args.permutation_use_voting_subset),
                )
            else:
                raise

    if args.shap:
        ran_any = True
        # Group-based SHAP importance
        print("\n" + "=" * 60)
        print("SHAP IMPORTANCE BY FEATURE GROUPS (validation voting subset)")
        print("=" * 60)
        shap_results = run_shap_analysis(
            predictor, df_train, X_val_voting, y_val_voting, config
        )

    # ------------------------------------------------------------------
    # Strongly-grouped average SHAP (validation voting subset)
    # ------------------------------------------------------------------
    if args.avg_shap_strong_groups:
        ran_any = True
        print("\n" + "=" * 60)
        print("AVERAGE SHAP (STRONG GROUPS, validation voting subset)")
        print("=" * 60)

        xgb_model = predictor.model
        df_strong, _Xsample = calculate_shap_average_strong_groups(
            model=xgb_model,
            X_df=X_val_voting,
            max_samples=config["max_samples"],
        )

        out_csv = os.path.join(config["output_dir"], "shap_average_strong_groups.csv")
        save_df = df_strong.copy()
        save_df["features"] = save_df["features"].apply(lambda t: "|".join(t))
        save_df.to_csv(out_csv, index=False)
        print(f"  ✓ Saved {out_csv}")

        plot_group_importance_barplot_no_errorbars(
            df_strong,
            config["output_dir"],
            filename_base=f"shap_average_strong_groups_top{int(args.avg_shap_top_n)}",
            top_n=int(args.avg_shap_top_n),
        )
        print(
            f"  ✓ Saved plot {os.path.join(config['output_dir'], f'shap_average_strong_groups_top{int(args.avg_shap_top_n)}.png')}"
        )

    # ------------------------------------------------------------------
    # Pairwise SHAP on TRAIN (binary per pair)
    # ------------------------------------------------------------------
    if args.pairwise_shap:
        ran_any = True
        print("\n" + "=" * 60)
        print("PAIRWISE SHAP (TRAIN)")
        print("=" * 60)

        # Target dataset for SHAP: TRAIN (model features)
        X_train_model = df_train[predictor.features]
        y_train = df_train["r_product_class"].astype(int).values

        # Use same feature_names/grouping logic as run_shap_analysis (but based on TRAIN)
        model_features = set(predictor.features)
        in_data = set(df_train.columns)
        wanted = prediction_utils.feature_columns_all
        feature_names = [c for c in wanted if c in in_data and c in model_features]
        if not feature_names:
            feature_names = [c for c in predictor.features if c in in_data]
        X_train_for_groups = df_train[feature_names]
        feature_groups = build_feature_groups(
            X_train_for_groups,
            feature_names,
            correlation_threshold=config["correlation_threshold"],
        )

        pairs = [(0, 1), (0, 2), (1, 2)]
        xgb_model = predictor.model

        for a, b in pairs:
            print(f"\n  Pair: {a} vs {b}")
            results_df, _sv, _fv, _Xsample = calculate_shap_pairwise_importance_by_groups(
                model=xgb_model,
                X_df=X_train_model,
                y_true=y_train,
                feature_groups=feature_groups,
                class_a=a,
                class_b=b,
                max_samples=config["max_samples"],
            )

            # Save CSV
            out_csv = os.path.join(config["output_dir"], f"shap_pairwise_{a}_vs_{b}.csv")
            save_df = results_df.copy()
            save_df["features"] = save_df["features"].apply(lambda t: "|".join(t))
            save_df.to_csv(out_csv, index=False)
            print(f"  ✓ Saved {out_csv}")

            # Plot top N
            plot_group_importance_barplot_to_file(
                results_df,
                config["output_dir"],
                filename_base=f"shap_pairwise_{a}_vs_{b}_top{int(args.pairwise_top_n)}",
                top_n=int(args.pairwise_top_n),
            )
            print(
                f"  ✓ Saved plot {os.path.join(config['output_dir'], f'shap_pairwise_{a}_vs_{b}_top{int(args.pairwise_top_n)}.png')}"
            )

    # ------------------------------------------------------------------
    # Per-class SHAP on TRAIN (one plot per class)
    # ------------------------------------------------------------------
    if args.per_class_shap:
        ran_any = True
        print("\n" + "=" * 60)
        print("PER-CLASS SHAP (TRAIN)")
        print("=" * 60)

        from permutation_analysis import SHAP_AVAILABLE
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not installed. Install with: pip install shap")

        import shap

        # Dataset: TRAIN (model features)
        X_train_model = df_train[predictor.features]
        y_train = df_train["r_product_class"].astype(int).values

        # Feature grouping (same logic as above)
        model_features = set(predictor.features)
        in_data = set(df_train.columns)
        wanted = prediction_utils.feature_columns_all
        feature_names = [c for c in wanted if c in in_data and c in model_features]
        if not feature_names:
            feature_names = [c for c in predictor.features if c in in_data]
        X_train_for_groups = df_train[feature_names]
        feature_groups = build_feature_groups(
            X_train_for_groups,
            feature_names,
            correlation_threshold=config["correlation_threshold"],
        )

        # Sample for SHAP (speed)
        if len(X_train_model) > config["max_samples"]:
            X_sample = X_train_model.sample(n=config["max_samples"], random_state=42).reset_index(drop=True)
            print(f"  Computing SHAP on {config['max_samples']} samples (of {len(X_train_model)} total)")
        else:
            X_sample = X_train_model.reset_index(drop=True)

        xgb_model = predictor.model
        booster = xgb_model.get_booster() if hasattr(xgb_model, "get_booster") else xgb_model
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_sample)

        # Extract per-class SHAP arrays
        if isinstance(shap_values, list):
            per_class = [np.asarray(sv) for sv in shap_values]  # each (n, f)
        elif len(np.asarray(shap_values).shape) == 3:
            sv3 = np.asarray(shap_values)  # (n, f, c)
            per_class = [sv3[:, :, i] for i in range(sv3.shape[2])]
        else:
            raise ValueError("Expected multi-class SHAP output for per-class SHAP plots.")

        feature_to_idx = {f: i for i, f in enumerate(X_sample.columns)}

        for cls in [0, 1, 2]:
            sv_signed = np.asarray(per_class[int(cls)])  # (n, f)
            sv_abs = np.abs(sv_signed)  # (n, f) for importance stats
            rows = []
            shap_values_per_group = {}
            feature_values_per_group = {}
            for group in feature_groups:
                idxs = [feature_to_idx[f] for f in group if f in feature_to_idx]
                if not idxs:
                    continue
                # Beeswarm should show signed SHAP; barplots should summarize |SHAP|
                group_shap_signed = sv_signed[:, idxs].mean(axis=1).flatten()
                group_shap_abs = sv_abs[:, idxs].mean(axis=1).flatten()
                group_label = group[0] if len(group) == 1 else f"{group[0]} (+{len(group)-1})"

                # Feature values used for beeswarm coloring (mean for groups)
                cols = [X_sample.columns[i] for i in idxs]
                if len(cols) == 1:
                    group_vals = X_sample[cols[0]].values
                else:
                    group_vals = X_sample[cols].mean(axis=1).values
                group_vals = np.asarray(group_vals).flatten()

                rows.append(
                    {
                        "group_label": group_label,
                        "features": tuple(group),
                        "n_features": len(group),
                        "importance_mean": float(np.mean(group_shap_abs)),
                        "importance_std": float(np.std(group_shap_abs)),
                        "q25": float(np.percentile(group_shap_abs, 25)),
                        "q50": float(np.percentile(group_shap_abs, 50)),
                        "q75": float(np.percentile(group_shap_abs, 75)),
                    }
                )
                shap_values_per_group[group_label] = group_shap_signed
                feature_values_per_group[group_label] = group_vals
            results_df = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)

            out_csv = os.path.join(config["output_dir"], f"shap_per_class_{cls}.csv")
            save_df = results_df.copy()
            save_df["features"] = save_df["features"].apply(lambda t: "|".join(t))
            save_df.to_csv(out_csv, index=False)
            print(f"  ✓ Saved {out_csv}")

            plot_group_importance_barplot_to_file(
                results_df,
                config["output_dir"],
                filename_base=f"shap_per_class_{cls}_top{int(args.pairwise_top_n)}",
                top_n=int(args.pairwise_top_n),
            )
            print(
                f"  ✓ Saved plot {os.path.join(config['output_dir'], f'shap_per_class_{cls}_top{int(args.pairwise_top_n)}.png')}"
            )

            plot_group_importance_beeswarm_to_file(
                results_df,
                shap_values_per_group,
                feature_values_per_group,
                config["output_dir"],
                filename_base=f"shap_per_class_{cls}_beeswarm_top{int(args.pairwise_top_n)}",
                top_n=int(args.pairwise_top_n),
            )
            print(
                f"  ✓ Saved beeswarm {os.path.join(config['output_dir'], f'shap_per_class_{cls}_beeswarm_top{int(args.pairwise_top_n)}.png')}"
            )

    # Metadata
    metadata = {
        "experiment": "permutation_importance",
        "timestamp": datetime.now().isoformat(),
        "model_path": os.path.abspath(model_path),
        "note": "Permutation/SHAP use frozen model from bundle; this experiment does not apply training-time augmentation.",
        "bundle_augmentation_used": aug_used,
        "ran": {
            "permutation": bool(args.permutation),
            "shap": bool(args.shap),
            "avg_shap_strong_groups": bool(args.avg_shap_strong_groups),
            "pairwise_shap": bool(args.pairwise_shap),
            "per_class_shap": bool(args.per_class_shap),
        },
        "n_validation_total": int(n_total),
        "n_validation_voting": int(n_agree),
        "remove_specialized_lookup": remove_specialized,
        "grouping": config["grouping"],
        "correlation_threshold": config["correlation_threshold"],
        "max_samples": config["max_samples"],
        "permutation": {
            "scoring": config["permutation_scoring"],
            "n_repeats": int(config["permutation_n_repeats"]),
            "use_voting_subset": bool(args.permutation_use_voting_subset),
            "dataset": (perm_results or {}).get("dataset") if perm_results else None,
            "n_eval": (perm_results or {}).get("n_eval") if perm_results else None,
            "num_groups": (perm_results or {}).get("num_groups") if perm_results else None,
        },
        "shap": {
            "dataset": "validation (voting subset)" if args.shap else None,
            "num_feature_groups": len(shap_results["shap_results"]) if shap_results else None,
            "top_groups": shap_results["shap_results"].head(10)["group_label"].tolist() if shap_results else None,
        },
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
