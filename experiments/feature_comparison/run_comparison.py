#!/usr/bin/env python3
"""
Feature comparison: Quantum-chemical descriptors vs Morgan fingerprints.

Both variants use the **voting model** (XGBoost + Lookup must agree).
The only difference is which features the XGBoost component is trained on:
  1. Quantum-chemical descriptors  (~15 features, same as the final model)
  2. Morgan fingerprints            (~4105 features)

The Lookup model is identical in both cases (Tanimoto similarity on
molecular fingerprints – independent of the XGBoost feature set).

Negative data is included in the Lookup pool (same as the default final model).

Usage:
    python run_comparison.py [--n-iter 25] [--output-dir results]
"""

import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_score, recall_score

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))

sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "copol_prediction"))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "fingerprint"))

from utils import load_data_split

from copol_prediction.analysis.analyze_model import (
    compute_fingerprints_for_smiles,
    compute_naive_baseline_predictions_with_similarity,
)
from copolpredictor import model_training, prediction_utils
from copolpredictor.data_augmentation import augment_with_gaussian_samples
from copolpredictor.inference import CopolymerPredictor

try:
    from copol_prediction.analysis.plot_config import (
        COMPARISON_COLORS,
        TWO_COL_WIDTH_INCH,
        setup_plot_style,
    )
except ImportError:

    def setup_plot_style():
        pass

    COMPARISON_COLORS = {"original": "#3A3B73", "filtered": "#e27f07"}
    TWO_COL_WIDTH_INCH = 7

try:
    _STYLE_PATH = os.path.join(_PROJECT_ROOT, "copol_prediction", "analysis", "lamalab.mplstyle")
    if os.path.exists(_STYLE_PATH):
        plt.style.use(_STYLE_PATH)
except Exception:
    pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Feature comparison: quantum descriptors vs Morgan fingerprints"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results", help="Directory to save results and plots"
    )
    parser.add_argument(
        "--n-iter", type=int, default=25, help="Hyperparameter search iterations per model"
    )
    parser.add_argument("--n-bits", type=int, default=2048, help="Morgan fingerprint bits")
    parser.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--plot-only", action="store_true", help="Skip training, re-plot from saved results JSON"
    )
    parser.add_argument(
        "--final-model-path",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "copol_prediction", "artifacts", "model_bundle"),
        help="Path to final model bundle (default: copol_prediction/artifacts/model_bundle)",
    )
    parser.add_argument(
        "--use-test",
        action="store_true",
        help="Use test set instead of validation set for evaluation (default: validation set)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def filter_valid_smiles(df):
    """Remove rows where any SMILES column cannot be parsed by RDKit."""
    from rdkit import Chem

    smiles_cols = ["monomer1_smiles", "monomer2_smiles", "solvent_smiles"]
    n_before = len(df)
    mask = pd.Series(True, index=df.index)
    for col in smiles_cols:
        if col in df.columns:
            valid = df[col].apply(
                lambda s: Chem.MolFromSmiles(str(s)) is not None if pd.notna(s) else False
            )
            mask &= valid
    df_out = df[mask].reset_index(drop=True)
    n_removed = n_before - len(df_out)
    if n_removed > 0:
        print(f"    Filtered {n_removed} rows with invalid SMILES " f"({len(df_out)} remaining)")
    return df_out


def load_data():
    """Load central train/val/test split + negative data for Lookup pool.

    Returns (df_train, df_val, df_test, df_neg).
    """
    copol_dir = os.path.join(_PROJECT_ROOT, "copol_prediction")
    split_dir = os.path.join(copol_dir, "artifacts", "data_splits")

    df_train, df_val, df_test = load_data_split.load_train_val_test_split(split_dir=split_dir)

    neg_path = os.path.join(
        copol_dir, "filter", "artificial_datapoints", "processed_combined_augmented.csv"
    )
    df_neg = None
    if os.path.exists(neg_path):
        df_neg = pd.read_csv(neg_path)
        if "Class" in df_neg.columns:
            df_neg = df_neg.rename(columns={"Class": "r_product_class"})
        df_neg["r_product_class"] = df_neg["r_product_class"].astype(int)
        if "reaction_id" not in df_neg.columns:
            df_neg["reaction_id"] = [f"neg_{i}" for i in range(len(df_neg))]
        print(f"  Negative data: {len(df_neg)} samples")
    else:
        print(f"  WARNING: Negative data not found at {neg_path}")

    # Filter out rows with unparseable SMILES to prevent RDKit segfaults
    print("  Validating SMILES …")
    df_train = filter_valid_smiles(df_train)
    df_val = filter_valid_smiles(df_val)
    df_test = filter_valid_smiles(df_test)
    if df_neg is not None:
        df_neg = filter_valid_smiles(df_neg)

    return df_train, df_val, df_test, df_neg


def get_morgan_feature_columns(n_bits):
    """Feature columns for the Morgan fingerprint model."""
    morgan = [f"morgan_bit_{i}_{m}" for i in range(n_bits) for m in [1, 2]]
    other = [
        "temperature",
        "polytype_emb_1",
        "polytype_emb_2",
        "method_emb_1",
        "method_emb_2",
        "solvent_logP",
        "solvent_TPSA",
        "solvent_HBD",
        "solvent_FractionCSP3",
    ]
    return morgan + other


def load_morgan_data():
    """Load pre-computed Morgan fingerprint train/test data.

    These are created by experiments/archive/create_train_test_split.py --fingerprints
    and stored in the feature_comparison/data/ directory.
    """
    data_dir = os.path.join(_SCRIPT_DIR, "data")
    train_path = os.path.join(data_dir, "train_morgan.csv")
    test_path = os.path.join(data_dir, "test_morgan.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Morgan fingerprint data not found!\n"
            f"  Expected: {train_path}\n"
            f"           {test_path}\n"
            f"  Run: cd experiments && python archive/create_train_test_split.py --fingerprints"
        )

    df_train_morgan = pd.read_csv(train_path)
    df_test_morgan = pd.read_csv(test_path)
    return df_train_morgan, df_test_morgan


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------
PARAM_GRID = {
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


def train_xgboost(df_train, features, config, training_config=None):
    """Train XGBoost and return (model, cv_score).

    Args:
        df_train: Training dataframe
        features: Feature columns to use
        config: Experiment config (n_iter, random_state, etc.)
        training_config: Training config from final model (augmentation, specialized filter, etc.)
    """
    df_train_work = df_train.copy()

    # Apply specialized filter if configured
    if training_config and training_config.get("remove_specialized", False):
        if "specialized_filter" in df_train_work.columns:
            before_count = len(df_train_work)
            df_train_work = df_train_work[
                df_train_work["specialized_filter"] != "specialized"
            ].reset_index(drop=True)
            removed_count = before_count - len(df_train_work)
            if removed_count > 0:
                print(f"    Removed {removed_count} specialized datapoints from training set")
        else:
            print("    Warning: 'specialized_filter' column not found in training data")

    # Apply augmentation if configured
    if training_config and training_config.get("use_augmentation", False):
        augmentation_samples = training_config.get("augmentation_samples", 5)
        print(f"    Applying augmentation ({augmentation_samples} samples per row)...")
        df_train_work = augment_with_gaussian_samples(
            df_train_work,
            num_samples=augmentation_samples,
            std_factor=0.3,
            random_state=config["random_state"],
        )
        print(f"    Training set after augmentation: {len(df_train_work)} samples")

    X = df_train_work[features]
    y = df_train_work["r_product_class"].astype(int).values
    groups = df_train_work["reaction_id"].astype(str).values

    class_weights = model_training.calculate_class_weights(y)

    result = model_training.train_xgboost_with_cv(
        X_train=X,
        y_train=y,
        groups=groups,
        param_grid=PARAM_GRID,
        n_iter=config["n_iter"],
        cv=5,
        random_state=config["random_state"],
        class_weights=class_weights,
        n_jobs=-1,
    )
    model = model_training.train_final_model(
        X_train=X,
        y_train=y,
        params=result["best_params"],
        class_weights=class_weights,
        random_state=config["random_state"],
    )
    return model, result["best_score"]


# ---------------------------------------------------------------------------
# Voting evaluation
# ---------------------------------------------------------------------------
def evaluate_voting(xgb_pred, lookup_pred, y_true, label):
    """Compute voting metrics and return a dict."""
    agree = xgb_pred == lookup_pred
    n_agree = int(agree.sum())
    n_total = len(y_true)
    coverage = n_agree / n_total

    y_true_v = y_true[agree]
    y_pred_v = xgb_pred[agree]

    macro_acc = balanced_accuracy_score(y_true_v, y_pred_v)
    macro_prec = precision_score(y_true_v, y_pred_v, average="macro", zero_division=0)
    per_cls_acc = recall_score(y_true_v, y_pred_v, labels=[0, 1, 2], average=None, zero_division=0)
    per_cls_prec = precision_score(
        y_true_v, y_pred_v, labels=[0, 1, 2], average=None, zero_division=0
    )
    cm = sk_confusion_matrix(y_true_v, y_pred_v, labels=[0, 1, 2])

    print(
        f"  [{label}] Macro Acc: {macro_acc:.4f}  |  Macro Prec: {macro_prec:.4f}  "
        f"|  Coverage: {coverage:.1%}  ({n_agree}/{n_total})"
    )

    return {
        "label": label,
        "macro_accuracy": macro_acc,
        "macro_precision": macro_prec,
        "coverage": coverage,
        "n_voting": n_agree,
        "n_total": n_total,
        "per_class_acc": per_cls_acc.tolist(),
        "per_class_prec": per_cls_prec.tolist(),
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def create_comparison_plot(res_quantum, res_morgan, n_feats_quantum, n_feats_morgan, output_dir):
    """Create a 3-panel comparison figure with shared legend below."""
    setup_plot_style()

    c1 = COMPARISON_COLORS.get("original", "#3A3B73")
    c2 = COMPARISON_COLORS.get("filtered", "#e27f07")
    alpha = 0.75

    height = TWO_COL_WIDTH_INCH * 0.42
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(TWO_COL_WIDTH_INCH, height))

    w = 0.25
    pad = 0.4  # x-axis padding beyond data range

    # --- (a) Macro metrics ---
    metrics_labels = ["Macro\nAccuracy", "Macro\nPrecision"]
    q_vals = [res_quantum["macro_accuracy"], res_quantum["macro_precision"]]
    m_vals = [res_morgan["macro_accuracy"], res_morgan["macro_precision"]]
    x = np.arange(len(metrics_labels))

    b1 = ax1.bar(x - w / 2, q_vals, w, label="Quantum Descriptors", color=c1, alpha=alpha)
    b2 = ax1.bar(x + w / 2, m_vals, w, label="Morgan Fingerprints", color=c2, alpha=alpha)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.01,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=6,
            )
    ax1.set_ylabel("Score", fontsize=8)
    ax1.set_title("a", fontsize=10, loc="left", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_labels, fontsize=7)
    ax1.set_xlim(x[0] - pad, x[-1] + pad)
    ax1.set_ylim(0, 1.08)
    ax1.tick_params(labelsize=6)
    ax1.grid(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # --- (b) Per-class accuracy ---
    classes = ["Class 0\n(Alternating)", "Class 1\n(Block-like)", "Class 2\n(Homopolymer)"]
    x2 = np.arange(3)
    ax2.bar(x2 - w / 2, res_quantum["per_class_acc"], w, color=c1, alpha=alpha)
    ax2.bar(x2 + w / 2, res_morgan["per_class_acc"], w, color=c2, alpha=alpha)
    for vals, offset in [
        (res_quantum["per_class_acc"], -w / 2),
        (res_morgan["per_class_acc"], w / 2),
    ]:
        for i, v in enumerate(vals):
            ax2.text(i + offset, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
    ax2.set_ylabel("Accuracy", fontsize=8)
    ax2.set_title("b", fontsize=10, loc="left", fontweight="bold")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes, fontsize=6)
    ax2.set_xlim(x2[0] - pad, x2[-1] + pad)
    ax2.set_ylim(0, 1.08)
    ax2.tick_params(labelsize=6)
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # --- (c) Number of features (log scale) ---
    x3_pos = np.arange(2)
    models = ["Quantum\nDescriptors", "Morgan\nFingerprints"]
    counts = [n_feats_quantum, n_feats_morgan]
    ax3.bar(x3_pos, counts, width=w, color=[c1, c2], alpha=alpha)
    for i, cnt in enumerate(counts):
        ax3.text(i, cnt * 1.15, str(int(cnt)), ha="center", va="bottom", fontsize=7)
    ax3.set_ylabel("Number of Features", fontsize=8)
    ax3.set_title("c", fontsize=10, loc="left", fontweight="bold")
    ax3.set_xticks(x3_pos)
    ax3.set_xticklabels(models, fontsize=7)
    ax3.set_xlim(x3_pos[0] - pad, x3_pos[-1] + pad)
    ax3.set_yscale("log")
    ax3.set_ylim(bottom=1)
    ax3.tick_params(labelsize=6)
    ax3.grid(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # --- Shared legend below all plots ---
    handles = [b1[0], b2[0]]
    labels = ["Quantum Descriptors", "Morgan Fingerprints"]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"feature_comparison.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  ✓ Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    config = {
        "n_iter": args.n_iter,
        "n_bits": args.n_bits,
        "radius": args.radius,
        "random_state": args.random_state,
    }

    output_dir = os.path.join(_SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot-only mode: reload saved results and regenerate plots
    # ------------------------------------------------------------------
    if args.plot_only:
        json_path = os.path.join(output_dir, "comparison_results.json")
        if not os.path.exists(json_path):
            print(f"Error: {json_path} not found. Run without --plot-only first.")
            sys.exit(1)
        print("=" * 60)
        print("PLOT-ONLY MODE — reloading saved results")
        print("=" * 60)
        with open(json_path) as f:
            saved = json.load(f)
        res_quantum = saved["quantum_descriptors"]
        res_morgan = saved["morgan_fingerprints"]
        n_feats_q = res_quantum["n_features"]
        n_feats_m = res_morgan["n_features"]
        print(f"  Loaded results from {json_path}")
        create_comparison_plot(res_quantum, res_morgan, n_feats_q, n_feats_m, output_dir)
        print("\nDone.")
        return

    print("=" * 60)
    print("FEATURE COMPARISON — VOTING MODEL")
    print("  Quantum Descriptors vs Morgan Fingerprints")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data …")
    df_train, df_val, df_test, df_neg = load_data()

    # Choose evaluation set (default: validation set, like SHAP analysis)
    if args.use_test:
        df_eval = df_test
        eval_name = "test"
    else:
        df_eval = df_val
        eval_name = "validation"
    print(f"  Using {eval_name} set for evaluation: {len(df_eval)} samples")

    # Load final model to get correct features and configuration
    print("\n  Loading final model configuration...")
    training_config = {}
    try:
        predictor = CopolymerPredictor(args.final_model_path)
        quantum_features = predictor.features
        training_config = predictor.metadata.get("training_config", {})
        print(f"  ✓ Loaded {len(quantum_features)} features from final model")
        print(
            f"    - Specialized removed: {training_config.get('specialized_removed_from_training', False)}"
        )
        print(f"    - Augmentation used: {training_config.get('augmentation_used', False)}")
        print(f"    - Augmentation samples: {training_config.get('augmentation_samples', 5)}")
        print(f"    - Negative data used: {training_config.get('negative_data_used', False)}")
    except Exception as e:
        print(f"  ⚠️  Warning: Could not load final model: {e}")
        print(f"     Falling back to prediction_utils.feature_columns")
        quantum_features = [c for c in prediction_utils.feature_columns if c in df_train.columns]
    print(f"  Quantum features: {len(quantum_features)}")

    # Lookup pool: train + negative data (only if final model uses negative data)
    use_negative_in_lookup = (
        training_config.get("negative_data_used", False) if training_config else False
    )
    df_lookup_pool = df_train.copy()
    if use_negative_in_lookup and df_neg is not None:
        df_lookup_pool = pd.concat([df_lookup_pool, df_neg], ignore_index=True)
        print(f"\n  Lookup pool: {len(df_lookup_pool)} samples (train + neg)")
    else:
        print(f"\n  Lookup pool: {len(df_lookup_pool)} samples (train only, no negative data)")

    y_true = df_eval["r_product_class"].astype(int).values

    # Morgan fingerprint data
    df_train_morgan, df_test_morgan = load_morgan_data()
    # Match evaluation set: filter Morgan data to same reaction_ids as df_eval
    if args.use_test:
        # Use test set - match by reaction_id
        df_eval_morgan = df_test_morgan[
            df_test_morgan["reaction_id"].isin(df_eval["reaction_id"])
        ].reset_index(drop=True)
        print(f"  Morgan test set: {len(df_eval_morgan)} samples (matched to evaluation set)")
    else:
        # For validation: match validation reaction_ids in Morgan test set
        # Note: Morgan only has train/test, so we use test_morgan but filter to validation reaction_ids
        val_reaction_ids = set(df_eval["reaction_id"].astype(str))
        df_eval_morgan = df_test_morgan[
            df_test_morgan["reaction_id"].astype(str).isin(val_reaction_ids)
        ].reset_index(drop=True)
        print(f"  ⚠️  Note: Using test_morgan.csv filtered to validation reaction_ids")
        print(
            f"     Morgan validation subset: {len(df_eval_morgan)} samples (of {len(df_test_morgan)} total)"
        )
        if len(df_eval_morgan) != len(df_eval):
            print(
                f"     ⚠️  Warning: Sample count mismatch! Validation: {len(df_eval)}, Morgan: {len(df_eval_morgan)}"
            )
            print(f"        Some validation samples may not be in Morgan test set")
    morgan_features = [
        c for c in get_morgan_feature_columns(config["n_bits"]) if c in df_train_morgan.columns
    ]
    print(f"  Morgan features:  {len(morgan_features)}")

    # ------------------------------------------------------------------
    # 2. Pre-compute fingerprints & Lookup predictions (shared)
    # ------------------------------------------------------------------
    print("\n[2/5] Computing Lookup predictions (shared by both models) …")
    smiles_cols = ["monomer1_smiles", "monomer2_smiles", "solvent_smiles"]
    all_smiles = set()
    for data in [df_lookup_pool, df_eval]:
        for col in smiles_cols:
            if col in data.columns:
                all_smiles.update(data[col].dropna().unique())
    fp_dict = compute_fingerprints_for_smiles(list(all_smiles))
    n_valid = sum(1 for v in fp_dict.values() if v is not None)
    print(f"  Fingerprint cache: {n_valid}/{len(all_smiles)} SMILES")

    y_lookup_pool = df_lookup_pool["r_product_class"].astype(int).values
    lookup_pred, _ = compute_naive_baseline_predictions_with_similarity(
        df_eval,
        df_lookup_pool,
        y_lookup_pool,
        quantum_features,
        fp_dict=fp_dict,
    )
    print(f"  Lookup predictions: {len(lookup_pred)}")

    # ------------------------------------------------------------------
    # 3. Train XGBoost — Quantum descriptors
    # ------------------------------------------------------------------
    print("\n[3/5] Training XGBoost with quantum descriptors …")
    print(f"  Using training config from final model:")
    print(
        f"    - Specialized removed: {training_config.get('specialized_removed_from_training', False)}"
    )
    print(f"    - Augmentation used: {training_config.get('augmentation_used', False)}")
    print(f"    - Negative data used: {training_config.get('negative_data_used', False)}")
    xgb_quantum, cv_quantum = train_xgboost(df_train, quantum_features, config, training_config)
    xgb_pred_quantum = xgb_quantum.predict(df_eval[quantum_features])
    print(f"  CV score: {cv_quantum:.4f}")

    # ------------------------------------------------------------------
    # 4. Train XGBoost — Morgan fingerprints
    # ------------------------------------------------------------------
    print("\n[4/5] Training XGBoost with Morgan fingerprints …")
    print(f"  Using same training config as quantum descriptors")
    # Note: Morgan data doesn't have specialized_filter column, so filter won't apply
    # But augmentation will still work
    xgb_morgan, cv_morgan = train_xgboost(df_train_morgan, morgan_features, config, training_config)

    # Ensure df_eval_morgan has same samples as df_eval (match by reaction_id and order)
    eval_reaction_ids = df_eval["reaction_id"].astype(str).values
    df_eval_morgan_matched = df_eval_morgan[
        df_eval_morgan["reaction_id"].astype(str).isin(eval_reaction_ids)
    ].copy()
    # Sort to match order of df_eval
    df_eval_morgan_matched["_sort_key"] = (
        df_eval_morgan_matched["reaction_id"]
        .astype(str)
        .apply(lambda x: list(eval_reaction_ids).index(x) if x in eval_reaction_ids else 999999)
    )
    df_eval_morgan_matched = df_eval_morgan_matched.sort_values("_sort_key").reset_index(drop=True)
    df_eval_morgan_matched = df_eval_morgan_matched.drop(columns=["_sort_key"])

    if len(df_eval_morgan_matched) != len(df_eval):
        print(
            f"  ⚠️  Warning: Only {len(df_eval_morgan_matched)}/{len(df_eval)} samples matched in Morgan data"
        )
        print(f"     Some samples will be excluded from comparison")
        # Filter df_eval and other arrays to matched samples
        matched_mask = (
            df_eval["reaction_id"]
            .astype(str)
            .isin(df_eval_morgan_matched["reaction_id"].astype(str))
        )
        df_eval = df_eval[matched_mask].reset_index(drop=True)
        y_true = df_eval["r_product_class"].astype(int).values
        lookup_pred = lookup_pred[matched_mask.values]
        xgb_pred_quantum = xgb_pred_quantum[matched_mask.values]
        print(f"     Using {len(df_eval)} matched samples for comparison")

    xgb_pred_morgan = xgb_morgan.predict(df_eval_morgan_matched[morgan_features])
    print(f"  CV score: {cv_morgan:.4f}")

    # ------------------------------------------------------------------
    # 5. Voting evaluation & comparison
    # ------------------------------------------------------------------
    print(f"\n[5/5] Evaluating voting models …")
    res_quantum = evaluate_voting(xgb_pred_quantum, lookup_pred, y_true, "Quantum Descriptors")
    res_morgan = evaluate_voting(xgb_pred_morgan, lookup_pred, y_true, "Morgan Fingerprints")

    # ---- Print comparison table ----
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'Quantum':>12} {'Morgan':>12} {'Δ':>10}")
    print("-" * 62)

    for key, label in [
        ("macro_accuracy", "Macro Accuracy"),
        ("macro_precision", "Macro Precision"),
        ("coverage", "Coverage"),
    ]:
        q = res_quantum[key]
        m = res_morgan[key]
        d = m - q
        if key == "coverage":
            print(f"{label:<25} {q:>12.1%} {m:>12.1%} {d:>+10.1%}")
        else:
            print(f"{label:<25} {q:>12.4f} {m:>12.4f} {d:>+10.4f}")

    print(
        f"{'CV score (XGBoost)':<25} {cv_quantum:>12.4f} {cv_morgan:>12.4f} "
        f"{cv_morgan - cv_quantum:>+10.4f}"
    )
    print(
        f"{'Num features (XGBoost)':<25} {len(quantum_features):>12d} "
        f"{len(morgan_features):>12d}"
    )

    print("\n" + "-" * 62)
    print("PER-CLASS ACCURACY (RECALL)")
    print("-" * 62)
    cls_names = ["Class 0 (Alternating)", "Class 1 (Block-like)", "Class 2 (Homopolymer)"]
    for i, name in enumerate(cls_names):
        q = res_quantum["per_class_acc"][i]
        m = res_morgan["per_class_acc"][i]
        print(f"  {name:<25} {q:>10.4f} {m:>10.4f} {m - q:>+10.4f}")
    print("=" * 70)

    # ---- Save results JSON ----
    results_json = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "evaluation_set": eval_name,
        "final_model_path": args.final_model_path,
        "quantum_descriptors": {
            **res_quantum,
            "cv_score": cv_quantum,
            "n_features": len(quantum_features),
            "features": quantum_features,
        },
        "morgan_fingerprints": {
            **res_morgan,
            "cv_score": cv_morgan,
            "n_features": len(morgan_features),
        },
    }
    json_path = os.path.join(output_dir, "comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\n  Results: {json_path}")

    # ---- Plots ----
    print("\nCreating comparison plot …")
    create_comparison_plot(
        res_quantum, res_morgan, len(quantum_features), len(morgan_features), output_dir
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
