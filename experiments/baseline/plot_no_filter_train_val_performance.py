#!/usr/bin/env python3
"""
Create a no-confidence-filter performance comparison bar plot (validation only).

Models:
1) Nearest Neighbor
2) XGBoost (current final model bundle)
3) XGBoost with NN Features (trained with current final-model architecture)
4) Voting (NN and XGBoost)

Plot:
- Validation-only grouped bar plot with:
  class-wise recall, macro accuracy, and macro F1.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from copol_prediction.analysis.analyze_model import (  # noqa: E402
    compute_fingerprints_for_smiles,
    compute_naive_baseline_predictions_with_similarity,
    setup_style as analysis_setup_style,
)
from copol_prediction.analysis.plot_config import (  # noqa: E402
    TWO_COL_WIDTH_INCH,
    get_class_label,
)
from copol_prediction.utils import load_data_split  # noqa: E402
from copolpredictor import model_training  # noqa: E402
from copolpredictor.inference import CopolymerPredictor  # noqa: E402

try:  # Suppress verbose RDKit deprecation logging in lookup loops.
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


MODEL_ORDER = [
    "Nearest Neighbor",
    "XGBoost",
    "XGBoost with NN Features",
    "Voting (NN and XGBoost)",
]

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot validation performance without confidence filtering "
            "for nearest-neighbor, XGBoost, voting, and XGBoost with NN features."
        )
    )
    parser.add_argument(
        "--base-model-path",
        type=str,
        default="../../copol_prediction/artifacts/model_bundle",
        help="Path to current final XGBoost model bundle.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison",
        help="Output directory for validation plot + metrics table.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for retraining XGBoost + lookup-features model.",
    )
    return parser.parse_args()


def _resolve_paths(args):
    script_dir = Path(__file__).resolve().parent

    base_model_path = Path(args.base_model_path)
    if not base_model_path.is_absolute():
        base_model_path = (script_dir / base_model_path).resolve()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (script_dir / output_dir).resolve()

    return base_model_path, output_dir


def load_train_val():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    split_dir = project_root / "copol_prediction" / "artifacts" / "data_splits"

    df_train, df_val, _df_test = load_data_split.load_train_val_test_split(
        split_dir=str(split_dir)
    )
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)


def compute_lookup_predictions(df_target, df_lookup_pool, base_features, fp_dict):
    y_lookup = df_lookup_pool["r_product_class"].astype(int).values
    pred, similarity = compute_naive_baseline_predictions_with_similarity(
        df_target,
        df_lookup_pool,
        y_lookup,
        feature_cols=base_features,
        fp_dict=fp_dict,
    )
    return pred.astype(int), similarity.astype(float)


def build_lookup_feature_frame(df_target, lookup_pred, lookup_sim):
    df_ext = df_target.copy()
    df_ext["baseline_class_0"] = (lookup_pred == 0).astype(int)
    df_ext["baseline_class_1"] = (lookup_pred == 1).astype(int)
    df_ext["baseline_class_2"] = (lookup_pred == 2).astype(int)
    df_ext["baseline_distance"] = np.clip(1.0 - lookup_sim, 0.0, 1.0)
    return df_ext


def compute_validation_metrics(y_true, y_pred, voting_mask=None):
    if voting_mask is None:
        y_eval = y_true
        p_eval = y_pred
    else:
        y_eval = y_true[voting_mask]
        p_eval = y_pred[voting_mask]

    if len(y_eval) == 0:
        return {
            "class_0_acc": float("nan"),
            "class_1_acc": float("nan"),
            "class_2_acc": float("nan"),
            "macro_acc": float("nan"),
            "macro_f1": float("nan"),
            "coverage": 0.0,
            "n_predicted": 0,
            "n_total": int(len(y_true)),
        }

    class_recalls = recall_score(
        y_eval,
        p_eval,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )
    macro_acc = float(balanced_accuracy_score(y_eval, p_eval))
    macro_f1 = float(f1_score(y_eval, p_eval, average="macro", zero_division=0))
    coverage = float(len(y_eval) / len(y_true))

    return {
        "class_0_acc": float(class_recalls[0]),
        "class_1_acc": float(class_recalls[1]),
        "class_2_acc": float(class_recalls[2]),
        "macro_acc": macro_acc,
        "macro_f1": macro_f1,
        "coverage": coverage,
        "n_predicted": int(len(y_eval)),
        "n_total": int(len(y_true)),
    }


def plot_validation_metric_bars(df_metrics, output_dir):
    # Reuse the exact analysis style configuration from copol_prediction/analysis.
    analysis_setup_style()
    font_size = 12

    metric_order = [
        "class_0_acc",
        "class_1_acc",
        "class_2_acc",
        "macro_acc",
        "macro_f1",
    ]
    class_1_label = get_class_label(1, style="short").replace("Random / block-like", "Random")
    metric_labels = {
        "class_0_acc": f"{get_class_label(0, style='short')} Recall",
        "class_1_acc": f"{class_1_label} Recall",
        "class_2_acc": f"{get_class_label(2, style='short')} Recall",
        "macro_acc": "Macro Accuracy",
        "macro_f1": "Macro F1 Score",
    }
    # Match permutation-importance style: RdBu gradient, sampled sparsely.
    rd_bu_sparse = plt.cm.RdBu(np.linspace(0, 1, 13))[::3]  # 5 colors: red -> blue
    metric_colors = {metric: rd_bu_sparse[i] for i, metric in enumerate(metric_order)}

    x = np.arange(len(MODEL_ORDER))
    width = 0.15
    offsets = (np.arange(len(metric_order)) - (len(metric_order) - 1) / 2) * width

    fig, ax = plt.subplots(1, 1, figsize=(TWO_COL_WIDTH_INCH, 3.6))

    all_vals = []
    for i, metric in enumerate(metric_order):
        vals = []
        for model_name in MODEL_ORDER:
            row = df_metrics[df_metrics["model"] == model_name].iloc[0]
            vals.append(float(row[metric]))
        all_vals.extend([v for v in vals if np.isfinite(v)])
        ax.bar(
            x + offsets[i],
            vals,
            width=width,
            color=metric_colors[metric],
            edgecolor="black",
            linewidth=0.35,
            alpha=0.9,
            label=metric_labels[metric],
        )

    if all_vals:
        y_min = min(all_vals)
        y_max = max(all_vals)
        y_range = y_max - y_min
        pad = 0.08 * y_range if y_range > 0 else 0.02
        lower = max(0.0, y_min - pad)
        upper = min(1.0, y_max + pad)
        if upper - lower < 0.06:
            center = 0.5 * (upper + lower)
            lower = max(0.0, center - 0.03)
            upper = min(1.0, center + 0.03)
        ax.set_ylim(lower, upper)
    else:
        ax.set_ylim(0.5, 1.0)

    ax.set_ylabel("Validation Score", fontsize=font_size)
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=20, ha="right", fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.legend(
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
        fontsize=font_size,
    )
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.34, top=0.9)

    for ext in ("png", "pdf"):
        fig.savefig(
            output_dir / f"validation_performance_no_conf_filter.{ext}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def fit_xgb_lookup_feature_model(
    df_train_ext, final_model_best_params, all_features, random_state
):
    X_train = df_train_ext[all_features]
    y_train = df_train_ext["r_product_class"].astype(int).values
    class_weights = model_training.calculate_class_weights(y_train)

    model = model_training.train_final_model(
        X_train=X_train,
        y_train=y_train,
        params=final_model_best_params,
        class_weights=class_weights,
        random_state=random_state,
    )
    return model


def main():
    args = parse_args()
    base_model_path, output_dir = _resolve_paths(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("NO-FILTER VALIDATION PERFORMANCE PLOT")
    print("=" * 70)

    print("\nLoading split data...")
    df_train, df_val = load_train_val()
    print(f"  Train rows: {len(df_train)}")
    print(f"  Validation rows: {len(df_val)}")

    print(f"\nLoading final model bundle: {base_model_path}")
    predictor = CopolymerPredictor(str(base_model_path))
    base_features = [c for c in predictor.features if c in df_train.columns]
    missing = [c for c in predictor.features if c not in df_train.columns]
    if missing:
        raise ValueError(
            f"Missing {len(missing)} feature(s) from split data, e.g. {missing[:5]}"
        )

    # Keep only rows with complete feature + label data for fair split-wise comparisons.
    required_cols = list(base_features) + ["r_product_class"]
    df_train = df_train.dropna(subset=required_cols).reset_index(drop=True)
    df_val = df_val.dropna(subset=required_cols).reset_index(drop=True)
    print(f"\nAfter NaN filtering:")
    print(f"  Train rows: {len(df_train)}")
    print(f"  Validation rows: {len(df_val)}")

    print("\nPreparing fingerprint cache for lookup models...")
    smiles_cols = ["monomer1_smiles", "monomer2_smiles", "solvent_smiles"]
    all_smiles = set()
    for frame in (df_train, df_val):
        for col in smiles_cols:
            if col in frame.columns:
                all_smiles.update(frame[col].dropna().unique())
    fp_dict = compute_fingerprints_for_smiles(list(all_smiles))

    # ------------------------------------------------------------------
    # Predictions for validation (and training only where required for feature model fit)
    # ------------------------------------------------------------------
    y_val = df_val["r_product_class"].astype(int).values

    print("\nComputing lookup predictions (no confidence filter)...")
    lookup_pred_train, lookup_sim_train = compute_lookup_predictions(
        df_train, df_train, base_features, fp_dict
    )
    lookup_pred_val, lookup_sim_val = compute_lookup_predictions(
        df_val, df_train, base_features, fp_dict
    )
    print(f"  Train: lookup done ({len(lookup_pred_train)} predictions)")
    print(f"  Validation: lookup done ({len(lookup_pred_val)} predictions)")

    print("\nComputing XGBoost predictions from current final model...")
    xgb_pred_val = predictor.model.predict(df_val[base_features]).astype(int)
    print(f"  Validation: XGBoost done ({len(xgb_pred_val)} predictions)")

    print("\nTraining XGBoost with NN Features using final-model architecture...")
    best_params = predictor.metadata.get("best_params", None)
    if best_params is None:
        raise ValueError(
            "Final model metadata does not contain 'best_params'. "
            "Cannot align XGBoost+Lookup architecture automatically."
        )

    df_train_ext = build_lookup_feature_frame(
        df_train, lookup_pred_train, lookup_sim_train
    )
    df_val_ext = build_lookup_feature_frame(
        df_val, lookup_pred_val, lookup_sim_val
    )
    lookup_feature_cols = [
        "baseline_class_0",
        "baseline_class_1",
        "baseline_class_2",
        "baseline_distance",
    ]
    all_features = base_features + lookup_feature_cols

    xgb_lf_model = fit_xgb_lookup_feature_model(
        df_train_ext=df_train_ext,
        final_model_best_params=best_params,
        all_features=all_features,
        random_state=args.random_state,
    )
    xgb_lf_pred_val = xgb_lf_model.predict(df_val_ext[all_features]).astype(int)

    print("\nComputing voting predictions (NN and XGBoost)...")
    voting_mask_val = lookup_pred_val == xgb_pred_val
    voting_pred_val = xgb_pred_val
    print(
        f"  Validation: agreement {int(voting_mask_val.sum())}/{len(voting_mask_val)} "
        f"({voting_mask_val.mean() * 100:.1f}%)"
    )

    # ------------------------------------------------------------------
    # Metrics table
    # ------------------------------------------------------------------
    print("\nCalculating metrics...")
    rows = []
    m_lookup = compute_validation_metrics(y_val, lookup_pred_val)
    rows.append({"split": "Validation", "model": "Nearest Neighbor", **m_lookup})

    m_xgb = compute_validation_metrics(y_val, xgb_pred_val)
    rows.append({"split": "Validation", "model": "XGBoost", **m_xgb})

    m_voting = compute_validation_metrics(y_val, voting_pred_val, voting_mask=voting_mask_val)
    rows.append(
        {"split": "Validation", "model": "Voting (NN and XGBoost)", **m_voting}
    )

    m_xgb_lf = compute_validation_metrics(y_val, xgb_lf_pred_val)
    rows.append({"split": "Validation", "model": "XGBoost with NN Features", **m_xgb_lf})

    df_metrics = pd.DataFrame(rows)
    df_metrics["model"] = pd.Categorical(
        df_metrics["model"], categories=MODEL_ORDER, ordered=True
    )
    df_metrics = df_metrics.sort_values(["model"]).reset_index(drop=True)

    csv_path = output_dir / "validation_performance_no_conf_filter.csv"
    df_metrics.to_csv(csv_path, index=False)
    print(f"  Saved metrics table: {csv_path}")

    json_path = output_dir / "validation_performance_no_conf_filter.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"  Saved metrics json:  {json_path}")

    print("\nPlotting bar chart...")
    plot_validation_metric_bars(df_metrics, output_dir)
    print(
        f"  Saved plot: {output_dir / 'validation_performance_no_conf_filter.png'} "
        f"and .pdf"
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
