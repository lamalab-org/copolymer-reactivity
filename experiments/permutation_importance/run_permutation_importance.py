#!/usr/bin/env python3
"""
SHAP feature importance (voting model, validation set, group-based).

- Loads the final model bundle (same as compare_models / train_final_model).
- Does **not** re-train the classifier and does **not** apply Gaussian augmentation here;
  SHAP explains the XGBoost weights exactly as stored in the bundle. Use a bundle trained
  with `train_final_model.py` without `--use-augmentation` for alignment with the no-augmentation pipeline.
- Uses the voting model: only validation samples where XGBoost and Lookup agree.
- Applies same training filters for Lookup as the final model (e.g. specialized removed, CV-prune list on lookup pool when present in bundle metadata).
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
    calculate_shap_pairwise_importance_by_groups,
    calculate_shap_average_strong_groups,
    calculate_permutation_importance_by_named_groups,
    build_pair12_atomic_groups,
    build_pair12_with_correlation_groups,
    plot_group_permutation_importance_barplot,
)

try:
    from copol_prediction.analysis.plot_config import setup_plot_style
except Exception:
    def setup_plot_style():
        pass


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
        description="SHAP feature importance (voting model, validation set, group-based)"
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

    # ------------------------------------------------------------------
    # Permutation importance (per-feature), grouped in plot only
    # ------------------------------------------------------------------
    parser.add_argument(
        "--permutation-per-feature",
        action="store_true",
        help="Compute permutation importance per feature (no joint permutation).",
    )
    parser.add_argument(
        "--permutation-by-groups",
        action="store_true",
        help="Compute permutation importance by permuting the paper groups jointly.",
    )
    parser.add_argument(
        "--permutation-pair12",
        action="store_true",
        help="Permute *_1 and *_2 jointly (atomic groups), then aggregate to paper plot groups.",
    )
    parser.add_argument(
        "--no-voting",
        action="store_true",
        help="Disable voting filter (no Lookup). Use full validation set and XGBoost predictions only.",
    )
    parser.add_argument(
        "--permutation-n-repeats",
        type=int,
        default=10,
        help="Number of permutation repeats per feature (default: 10).",
    )
    parser.add_argument(
        "--permutation-scoring",
        type=str,
        default="balanced_accuracy",
        choices=["f1_macro", "balanced_accuracy", "accuracy"],
        help="Scoring metric for permutation importance (default: balanced_accuracy).",
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
    # Same param_grid as train_final_model.py to ensure comparable tuning
    param_grid = {
        "n_estimators": [500, 600, 700],
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
            "best_params": train_results["best_params"],
            "cv_score": train_results.get("cv_score"),
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
    """Format feature name for display.

    Strategy: explicit long replacements first (so substrings like 'ea' in 'mean'
    or 'ip' in 'dipole' are already gone), then underscores → spaces, then short
    abbreviations via whole-word regex to avoid false matches.
    """
    import re
    name = str(name)

    # 1. Embedding features (specific suffixes before base)
    name = name.replace("polytype_emb_1", "polymerization type emb. 1")
    name = name.replace("polytype_emb_2", "polymerization type emb. 2")
    name = name.replace("method_emb_1", "polymerization method emb. 1")
    name = name.replace("method_emb_2", "polymerization method emb. 2")
    name = name.replace("polytype_emb", "polymerization type emb.")
    name = name.replace("method_emb", "polymerization method emb.")

    # 2. HOMO-LUMO differences
    name = name.replace("delta_HOMO_LUMO", "Δ HOMO-LUMO")
    name = name.replace("delta_homo_lumo", "Δ HOMO-LUMO")
    name = name.replace("_AA", " 1-1").replace("_AB", " 1-2").replace("_BA", " 2-1").replace("_BB", " 2-2")

    # 3. Fukui indices (before generic 'ea' / 'ip' substitution)
    name = name.replace("fukui_electrophilicity_min", "Fukui electrophilicity min")
    name = name.replace("fukui_electrophilicity_max", "Fukui electrophilicity max")
    name = name.replace("fukui_electrophilicity_mean", "Fukui electrophilicity mean")
    name = name.replace("fukui_nucleophilicity_min", "Fukui nucleophilicity min")
    name = name.replace("fukui_nucleophilicity_max", "Fukui nucleophilicity max")
    name = name.replace("fukui_nucleophilicity_mean", "Fukui nucleophilicity mean")
    name = name.replace("fukui_radical_min", "Fukui radical min")
    name = name.replace("fukui_radical_max", "Fukui radical max")
    name = name.replace("fukui_radical_mean", "Fukui radical mean")

    # 4. Dipole components (before 'ip' substitution)
    name = name.replace("dipole_x", "dipole moment x")
    name = name.replace("dipole_y", "dipole moment y")
    name = name.replace("dipole_z", "dipole moment z")

    # 5. Other explicit multi-word features
    name = name.replace("ip_corrected", "ionization potential (corrected)")
    name = name.replace("best_conformer_energy", "best conformer energy")
    name = name.replace("global_electrophilicity", "global electrophilicity")
    name = name.replace("global_nucleophilicity", "global nucleophilicity")
    name = name.replace("charges_min", "charges min")
    name = name.replace("charges_max", "charges max")
    name = name.replace("charges_mean", "charges mean")

    # 6. Solvent descriptors
    name = name.replace("solvent_logp", "solvent LogP")
    name = name.replace("solvent_logP", "solvent LogP")
    name = name.replace("solvent_TPSA", "solvent TPSA")
    name = name.replace("solvent_HBD", "solvent H-bond donors")
    name = name.replace("solvent_FractionCSP3", "solvent fraction Csp3")

    # 7. Remaining underscores → spaces (so 'ip_1' becomes 'ip 1')
    name = name.replace("_", " ")

    # 8. Short abbreviations as whole words only (avoids 'ea' in 'mean', 'ip' in 'dipole')
    name = re.sub(r"\bip\b", "ionization potential", name)
    name = re.sub(r"\bea\b", "electron affinity", name)
    name = re.sub(r"\bhomo\b", "HOMO", name)
    name = re.sub(r"\blumo\b", "LUMO", name)

    return name


def _shap_bar_colors(group_labels):
    """
    Return a list of bar colors, one per group_label.
    Monomer features (quantum chemistry) → series_1 (blue)
    Reaction condition features (temperature, solvent, embedding) → series_2 (red)
    """
    color_monomer = "#143D60"
    color_condition = "#661124"

    condition_keywords = {
        "temperature", "polytype_emb", "method_emb",
        "polymerization", "solvent_logp", "solvent_logP",
        "solvent_frac", "solvent_tpsa", "solvent_hbd", "solvent",
    }

    colors = []
    for lbl in group_labels:
        lbl_lower = str(lbl).lower()
        if any(kw in lbl_lower for kw in condition_keywords):
            colors.append(color_condition)
        else:
            colors.append(color_monomer)
    return colors


def plot_group_importance_barplot(results_df, output_dir, top_n=50):
    """Bar plot for group-based SHAP importance, color-coded by feature category."""
    n_groups = len(results_df)
    n_plot = min(top_n, n_groups)
    top = results_df.head(top_n).copy()
    if n_plot < n_groups:
        print(f"  Plotting top {n_plot} of {n_groups} groups (--top-n={top_n})")
    else:
        print(f"  Plotting all {n_plot} groups")
    top["display_label"] = top["group_label"].apply(format_feature_name)

    try:
        from copol_prediction.analysis.plot_config import TWO_COL_WIDTH_INCH
        width = float(TWO_COL_WIDTH_INCH)
    except Exception:
        width = 7.0
    height = max(4, len(top) * 0.2)
    fig, ax = plt.subplots(figsize=(width, height))

    y_pos = np.arange(len(top))
    bar_colors = _shap_bar_colors(top["group_label"].tolist())
    ax.barh(
        y_pos,
        top["importance_mean"],
        xerr=top["importance_std"],
        capsize=4,
        alpha=0.85,
        color=bar_colors,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["display_label"], fontsize=7)
    ax.set_xlabel("SHAP importance (sum |SHAP| across classes, mean ± std)", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis()
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    c_mon, c_cond = "#143D60", "#661124"
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(facecolor=c_mon, alpha=0.85, label="Monomer descriptor"),
            Patch(facecolor=c_cond, alpha=0.85, label="Reaction condition"),
        ],
        fontsize=7,
        loc="lower right",
        frameon=False,
    )

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


def run_shap_analysis(predictor, df_val, config):
    """
    Group-based SHAP importance on the full validation set using the final model's features.

    - Uses only predictor.features (the final model's feature set).
    - Groups features using build_pair12_atomic_groups (same grouping as permutation analysis).
    - Sums |SHAP| across all three classes.
    - Applies format_feature_name to group labels for the barplot.
    """
    feature_names = [c for c in predictor.features if c in df_val.columns]
    print(f"  Analyzing {len(feature_names)} features (final model feature set)")

    # Same grouping as permutation analysis: _1/_2 pairs joint, emb. features individual
    atomic_groups = build_pair12_atomic_groups(feature_names)
    feature_groups = list(atomic_groups.values())
    raw_labels = list(atomic_groups.keys())
    group_labels = [format_feature_name(lbl) for lbl in raw_labels]

    n_pairs = sum(1 for v in feature_groups if len(v) == 2)
    n_singles = sum(1 for v in feature_groups if len(v) == 1)
    print(f"  Groups: {len(feature_groups)} ({n_pairs} pairs, {n_singles} singletons)")

    X_val = df_val[feature_names]
    print(f"  Calculating group-based SHAP importance (max_samples={config['max_samples']}, reduction=sum)...")
    results_df, shap_values_per_group, feature_values_per_group, X_sample = calculate_shap_importance_by_groups(
        model=predictor.model,
        X_df=X_val,
        feature_groups=feature_groups,
        max_samples=config["max_samples"],
        reduction="sum",
        group_labels=group_labels,
    )

    out_csv = os.path.join(config["output_dir"], "shap_importance_detailed.csv")
    save_df = results_df.copy()
    save_df["features"] = save_df["features"].apply(lambda t: "|".join(t))
    save_df.to_csv(out_csv, index=False)
    print(f"  ✓ Saved {out_csv}")

    plot_path_bar = plot_group_importance_barplot(
        results_df, config["output_dir"], top_n=config["top_n"]
    )
    plot_path_beeswarm = plot_group_importance_beeswarm(
        results_df, shap_values_per_group, feature_values_per_group,
        config["output_dir"], top_n=config["top_n"]
    )

    print(f"\n  Top 10 groups by importance:")
    for i, (_, row) in enumerate(results_df.head(10).iterrows(), 1):
        print(f"    {i:2d}. {row['group_label']:<45} {row['importance_mean']:.6f} ± {row['importance_std']:.6f}")

    return {
        "shap_results": results_df,
        "plot_path_bar": plot_path_bar,
        "plot_path_beeswarm": plot_path_beeswarm,
    }


def run_shap_per_class_analysis(predictor, df_val, config):
    """
    Per-class signed SHAP beeswarm — all 3 classes side by side in one figure.

    - X-axis per subplot: signed SHAP for that class
      (positive = pushes towards class, negative = away)
    - Y-axis: feature groups (shared, labeled only on leftmost subplot)
    - Color: normalized feature value (red = high, blue = low)
    - Single colorbar on the right
    """
    from matplotlib.colors import Normalize
    import shap as shap_lib

    feature_names = [c for c in predictor.features if c in df_val.columns]
    X_val = df_val[feature_names]
    print(f"  Features: {len(feature_names)}")

    atomic_groups = build_pair12_atomic_groups(feature_names)
    feature_groups = list(atomic_groups.values())
    raw_labels = list(atomic_groups.keys())
    group_labels = [format_feature_name(lbl) for lbl in raw_labels]

    max_samples = config["max_samples"]
    if len(X_val) > max_samples:
        X_sample = X_val.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"  Computing SHAP on {max_samples} of {len(X_val)} samples")
    else:
        X_sample = X_val.reset_index(drop=True)

    booster = predictor.model.get_booster()
    explainer = shap_lib.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        per_class_shap = [np.asarray(sv) for sv in shap_values]
    elif np.asarray(shap_values).ndim == 3:
        sv3 = np.asarray(shap_values)
        per_class_shap = [sv3[:, :, c] for c in range(sv3.shape[2])]
    else:
        raise ValueError("Expected multi-class SHAP (list or 3D array).")

    feature_to_idx = {f: i for i, f in enumerate(X_sample.columns)}
    class_names = {0: "alternating", 1: "gradient", 2: "random"}
    top_n = config["top_n"]

    # Build per-group data for each class; use class-0 order for consistent y-axis
    all_class_data = []
    for sv_signed in per_class_shap:
        shap_list, feat_list, labels = [], [], []
        for group, lbl in zip(feature_groups, group_labels):
            idxs = [feature_to_idx[f] for f in group if f in feature_to_idx]
            if not idxs:
                continue
            cols = [X_sample.columns[i] for i in idxs]
            g_shap = sv_signed[:, idxs].mean(axis=1)
            g_feat = X_sample[cols].mean(axis=1).values if len(cols) > 1 else X_sample[cols[0]].values
            shap_list.append(np.asarray(g_shap).flatten())
            feat_list.append(np.asarray(g_feat).flatten())
            labels.append(lbl)
        all_class_data.append((shap_list, feat_list, labels))

    # Sort order by mean |SHAP| of class 0, apply to all classes
    shap0, _, labels0 = all_class_data[0]
    order = np.argsort([-np.mean(np.abs(s)) for s in shap0])[:top_n]

    # Apply order to all classes
    sorted_data = []
    for shap_list, feat_list, labels in all_class_data:
        sorted_data.append((
            [shap_list[i] for i in order],
            [feat_list[i] for i in order],
            [labels[i] for i in order],
        ))
    y_labels = sorted_data[0][2]  # shared y-axis labels
    n_groups = len(y_labels)
    y_pos = np.arange(n_groups)

    try:
        from copol_prediction.analysis.plot_config import TWO_COL_WIDTH_INCH
        total_width = float(TWO_COL_WIDTH_INCH)
    except Exception:
        total_width = 7.0

    plot_height = max(3.2, n_groups * 0.28)
    cbar_height = 0.22   # height of the colorbar strip in inches
    gap = 1.0            # gap between plots and colorbar in inches
    fig_height = plot_height + gap + cbar_height

    fig = plt.figure(figsize=(total_width, fig_height))
    gs = fig.add_gridspec(
        2, 3,
        height_ratios=[plot_height, cbar_height],
        hspace=gap / fig_height,  # proportional gap
        wspace=0.08,
    )
    axes = [fig.add_subplot(gs[0, c]) for c in range(3)]

    np.random.seed(42)
    for col_idx, (ax, (shap_list, feat_list, _)) in enumerate(zip(axes, sorted_data)):
        for i, (shap_vals, feat_vals) in enumerate(zip(shap_list, feat_list)):
            f_min, f_max = float(np.min(feat_vals)), float(np.max(feat_vals))
            feat_norm = (
                (feat_vals - f_min) / (f_max - f_min)
                if f_max > f_min else np.full_like(feat_vals, 0.5)
            )
            y_jitter = np.random.normal(y_pos[i], 0.05, size=len(shap_vals))
            ax.scatter(shap_vals, y_jitter, alpha=0.5, s=8,
                       c=plt.cm.RdYlBu_r(feat_norm), edgecolors="none")

        ax.axvline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
        ax.set_xlabel(
            f"SHAP ({class_names.get(col_idx, str(col_idx))})",
            fontsize=8,
        )
        ax.tick_params(axis="x", labelsize=7)
        ax.set_yticks(y_pos)
        ax.set_ylim(n_groups - 0.5, -0.5)   # inverted, same range on all axes
        ax.grid(False, axis="y")
        ax.grid(True, axis="x", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col_idx == 0:
            ax.set_yticklabels(y_labels, fontsize=7)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)

    # Horizontal colorbar in the reserved bottom strip
    cbar_ax = fig.add_subplot(gs[1, :])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlBu_r, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"], fontsize=7)
    cbar.set_label("Feature value", fontsize=7, labelpad=3)
    cbar.ax.tick_params(labelsize=7)

    for ext in ("pdf", "png"):
        out = os.path.join(config["output_dir"], f"shap_per_class_beeswarm.{ext}")
        plt.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved shap_per_class_beeswarm.pdf/png")


def main():
    args = parse_args()
    setup_plot_style()
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

    # Validation set selection: full validation (no voting) vs voting subset
    if args.no_voting:
        print("\nUsing full validation set (no voting / no Lookup)...")
        df_val_selected = df_val.reset_index(drop=True)
        X_val_selected = df_val_selected[predictor.features]
        y_val_selected = df_val_selected["r_product_class"].astype(int).values
        n_total = int(len(df_val_selected))
        n_agree = int(len(df_val_selected))
        print(f"  Validation: {n_agree}/{n_total} samples (100.0%)")
        # Keep legacy variable names for downstream SHAP section
        X_val_voting, y_val_voting, df_val_voting = X_val_selected, y_val_selected, df_val_selected
    else:
        # Voting subset on validation
        print("\nApplying voting filter (XGBoost + Lookup agree) on validation...")
        X_val_selected, y_val_selected, df_val_selected, n_agree, n_total = get_voting_subset(
            df_val, df_train, predictor, remove_specialized
        )
        print(f"  Validation: {n_agree}/{n_total} samples ({100 * n_agree / n_total:.1f}%) after voting")
        # Keep legacy variable names for downstream SHAP section
        X_val_voting, y_val_voting, df_val_voting = X_val_selected, y_val_selected, df_val_selected

    # ------------------------------------------------------------------
    # Permutation importance: _1/_2 pairs permuted jointly (XGBoost, full validation)
    # polytype_emb_* and method_emb_* are permuted individually (not paired)
    # ------------------------------------------------------------------
    if args.permutation_per_feature or args.permutation_by_groups or args.permutation_pair12:
        print("\n" + "=" * 60)
        print("PERMUTATION IMPORTANCE (PAIR _1/_2 JOINT; FULL VALIDATION; XGBOOST)")
        print("=" * 60)

        # Always use full validation set with XGBoost only (no voting filter).
        # Target: feature_columns_all — intersected with model features and data columns.
        # Use --train-full-features-model (or point --model-path to full_features_model_bundle)
        # to cover all of feature_columns_all.
        model_feature_set = set(predictor.features)
        features = [
            c for c in prediction_utils.feature_columns_all
            if c in df_val.columns and c in model_feature_set
        ]
        missing_from_model = [
            c for c in prediction_utils.feature_columns_all if c not in model_feature_set
        ]
        if missing_from_model:
            print(
                f"  ⚠  {len(missing_from_model)} feature_columns_all not in current model "
                f"(use --train-full-features-model or point --model-path to "
                f"results/full_features_model_bundle to include them):\n"
                f"     {missing_from_model}"
            )
        Xp = df_val[features].copy()
        yp = df_val["r_product_class"].astype(int).values
        print(f"  Samples: {len(yp)}, Features: {len(features)} / {len(prediction_utils.feature_columns_all)} feature_columns_all")

        # Build permutation groups:
        #   - _1/_2 feature pairs are permuted jointly
        #   - polytype_emb_1/2 and method_emb_1/2 remain individual singletons
        #   - if --correlation-threshold < 1.0, additionally merge groups whose
        #     representative values are highly correlated (e.g. correlated RDKit features)
        corr_thresh = float(config["correlation_threshold"])
        if corr_thresh < 1.0:
            atomic_groups = build_pair12_with_correlation_groups(
                Xp, features, correlation_threshold=corr_thresh
            )
            print(f"  Correlation threshold: {corr_thresh} (correlated groups merged)")
        else:
            atomic_groups = build_pair12_atomic_groups(features)
        n_pairs = sum(1 for v in atomic_groups.values() if len(v) == 2)
        n_multi = sum(1 for v in atomic_groups.values() if len(v) > 2)
        n_singles = sum(1 for v in atomic_groups.values() if len(v) == 1)
        print(f"  Groups: {len(atomic_groups)} ({n_pairs} pairs, {n_multi} multi-feature, {n_singles} singletons)")

        atomic_df = calculate_permutation_importance_by_named_groups(
            predictor.model,
            Xp,
            yp,
            atomic_groups,
            scoring="balanced_accuracy",
            n_repeats=int(args.permutation_n_repeats),
            random_state=int(args.random_state),
        )

        out_atomic_csv = os.path.join(args.output_dir, "permutation_importance_pair12_atomic_groups.csv")
        save_atomic = atomic_df.copy()
        save_atomic["features"] = save_atomic["features"].apply(lambda t: "|".join(t))
        save_atomic.to_csv(out_atomic_csv, index=False)
        print(f"  ✓ Saved {out_atomic_csv}")

        # Barplot directly from atomic groups with readable labels
        plot_df = atomic_df.copy()
        plot_df["group_label"] = plot_df["group_label"].apply(format_feature_name)
        xlabel = "Permutation importance (Macro Accuracy, mean ± std)"
        for ext in ("pdf", "png"):
            out_path = os.path.join(args.output_dir, f"permutation_importance_barplot.{ext}")
            plot_group_permutation_importance_barplot(
                plot_df, top_n=50, save_path=out_path, xlabel=xlabel,
            )

    # Group-based SHAP importance
    print("\n" + "=" * 60)
    print("SHAP IMPORTANCE BY FEATURE GROUPS")
    print("=" * 60)
    shap_results = run_shap_analysis(predictor, df_val, config)

    # ------------------------------------------------------------------
    # Strongly-grouped average SHAP (validation voting subset)
    # ------------------------------------------------------------------
    if args.avg_shap_strong_groups:
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
    # Per-class SHAP beeswarm (validation set, final model features)
    # ------------------------------------------------------------------
    if args.per_class_shap:
        print("\n" + "=" * 60)
        print("PER-CLASS SHAP BEESWARM (VALIDATION SET)")
        print("=" * 60)
        run_shap_per_class_analysis(predictor, df_val, config)


    # Metadata
    metadata = {
        "experiment": "permutation_importance",
        "timestamp": datetime.now().isoformat(),
        "model_path": os.path.abspath(model_path),
        "note": "SHAP uses frozen XGBoost from bundle; this experiment does not apply training-time augmentation.",
        "bundle_augmentation_used": aug_used,
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
