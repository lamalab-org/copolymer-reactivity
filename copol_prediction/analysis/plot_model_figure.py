#!/usr/bin/env python3
"""
Create paper-style overview figures.

We intentionally split the original "model summary" into 2 figures:

Figure 1: Class curves (1x4)
  (a) Alternating
  (b) Random
  (c) Gradient
  (d) Schematic: how classes are split (diagonal crossing + integral I_rand)

Figure 2: Model performance (1x3)
  (a) Confusion matrix on the TEST set (UNFILTERED; XGB-only predictions)
  (b) Performance comparison: with vs without reaction conditions (macro + per-class)
  (c) SHAP feature importance (top N)

Usage (repo root):
  python -m copol_prediction.analysis.plot_model_figure

Outputs (default):
  copol_prediction/analysis/figures/model_class_curves_figure.png/.pdf
  copol_prediction/analysis/figures/model_performance_figure.png/.pdf
"""

from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Ensure copol_prediction/ is on sys.path when run as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # copol_prediction/
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from copolpredictor.inference import CopolymerPredictor
from copol_prediction.utils import load_data_split
from copol_prediction.analysis.analyze_model import get_class_label
from copol_prediction.analysis import plot_class_curves as pcc
from copol_prediction.analysis.plot_config import CLASS_CURVES_FIGSIZE_INCH, setup_plot_style


def parse_args():
    p = argparse.ArgumentParser(description="Create model overview figures (curves + performance).")
    p.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "artifacts", "model_bundle"),
        help="Path to final model bundle (default: copol_prediction/artifacts/model_bundle).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "figures"),
        help="Directory to save the figure (default: copol_prediction/analysis/figures).",
    )
    p.add_argument(
        "--split-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "artifacts", "data_splits"),
        help="Directory with train/val/test CSVs (default: copol_prediction/artifacts/data_splits).",
    )
    p.add_argument(
        "--reaction-conditions-json",
        type=str,
        default=os.path.join(
            WORKSPACE_ROOT,
            "experiments",
            "reaction_conditions_comparison",
            "results",
            "comparison_results.json",
        ),
        help="Path to reaction conditions comparison_results.json.",
    )
    p.add_argument(
        "--shap-csv",
        type=str,
        default=os.path.join(
            WORKSPACE_ROOT,
            "experiments",
            "permutation_importance",
            "results",
            "shap_average_strong_groups.csv",
        ),
        help="Path to strongly-grouped average SHAP CSV (from experiments/permutation_importance).",
    )
    p.add_argument(
        "--shap-top-n",
        type=int,
        default=10,
        help="Top N SHAP groups to show (default: 10).",
    )
    p.add_argument(
        "--max-curves-per-class",
        type=int,
        default=1500,
        help="Max curves per class for class-curves panels (default: 1500).",
    )
    p.add_argument(
        "--band-quantiles",
        type=float,
        nargs=2,
        default=(10.0, 90.0),
        help="Quantiles for the class-curves band, e.g. 10 90.",
    )
    return p.parse_args()


def _plot_confusion_matrix_unfiltered(ax, predictor: CopolymerPredictor, df_test: pd.DataFrame):
    X = df_test[predictor.features]
    y_true = df_test["r_product_class"].astype(int).values
    y_pred = predictor.predict(X).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[get_class_label(i, style="short") for i in [0, 1, 2]],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("A  Confusion matrix (test, unfiltered)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")


def _plot_reaction_conditions_comparison(ax, json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    w = saved["with_conditions"]
    wo = saved["without_conditions"]

    labels = [
        "Within-pair\nbal. acc",
        "Macro acc",
        "Coverage",
    ]
    with_pair = float(w.get("pair_balanced_accuracy", w["macro_accuracy"]))
    without_pair = float(wo.get("pair_balanced_accuracy", wo["macro_accuracy"]))
    with_vals = [with_pair, float(w["macro_accuracy"]), float(w["coverage"])]
    without_vals = [without_pair, float(wo["macro_accuracy"]), float(wo["coverage"])]

    x = np.arange(len(labels))
    width = 0.35
    b1 = ax.bar(x - width / 2, with_vals, width, label="With cond.", color="#661124", alpha=0.85)
    b2 = ax.bar(x + width / 2, without_vals, width, label="Without cond.", color="#143D60", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("B  With vs without reaction conditions")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower left")

    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom")


def _plot_shap_topn(ax, csv_path: str, top_n: int):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "SHAP file not found\n(re-run permutation importance)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return
    df = df.sort_values("importance_mean", ascending=False).head(int(top_n)).copy()
    df = df.iloc[::-1]  # for barh top at top

    ax.barh(
        df["group_label"].astype(str),
        df["importance_mean"].astype(float).values,
        color="#661124",
        alpha=0.9,
    )
    ax.set_title("C  SHAP feature importance (avg |SHAP|, grouped)")
    ax.set_xlabel("mean |SHAP| (avg over 3 classes)")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_class_curves_row(axes_row, df_all: pd.DataFrame, *, max_curves_per_class: int, band_quantiles):
    f1 = np.linspace(0, 1, 501)
    class_curves = pcc._sample_per_class(
        df_all,
        f1=f1,
        max_curves_per_class=max_curves_per_class,
    )

    panels = [
        ("alternating", "A  Alternating"),
        ("random (to blocky)", "B  Random"),
        ("gradient", "C  Gradient"),
    ]

    for ax, (label, title) in zip(axes_row[:3], panels):
        pcc._plot_dual_band(
            ax,
            f1,
            class_curves.get(label, np.empty((0, len(f1)))),
            label=title,
            band_quantiles=band_quantiles,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$f_1$ (feed conc. monomer 1)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes_row[0].set_ylabel(r"$F_1$ (monomer 1 proportion in polymer)")

    # 4th panel: schematic explanation of class split
    if len(axes_row) >= 4 and hasattr(pcc, "_plot_class_split_explanation"):
        pcc._plot_class_split_explanation(axes_row[3])
        axes_row[3].set_title("D  Class determination")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_plot_style()

    # Data
    df_train, df_val, df_test = load_data_split.load_train_val_test_split(split_dir=args.split_dir)
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Model
    predictor = CopolymerPredictor(args.model_path)

    # Figure 1: class curves (1x4)
    fig1, axes1 = plt.subplots(1, 4, figsize=CLASS_CURVES_FIGSIZE_INCH, sharex=True, sharey=True)
    _plot_class_curves_row(
        axes1,
        df_all,
        max_curves_per_class=args.max_curves_per_class,
        band_quantiles=tuple(args.band_quantiles),
    )
    plt.tight_layout()
    out1_png = os.path.join(args.output_dir, "model_class_curves_figure.png")
    out1_pdf = os.path.join(args.output_dir, "model_class_curves_figure.pdf")
    fig1.savefig(out1_png, dpi=300, bbox_inches="tight")
    fig1.savefig(out1_pdf, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: performance plots (1x3)
    fig2, axes2 = plt.subplots(1, 3, figsize=(13.5, 3.8))
    _plot_confusion_matrix_unfiltered(axes2[0], predictor, df_test)
    _plot_reaction_conditions_comparison(axes2[1], args.reaction_conditions_json)
    _plot_shap_topn(axes2[2], args.shap_csv, args.shap_top_n)
    plt.tight_layout()
    out2_png = os.path.join(args.output_dir, "model_performance_figure.png")
    out2_pdf = os.path.join(args.output_dir, "model_performance_figure.pdf")
    fig2.savefig(out2_png, dpi=300, bbox_inches="tight")
    fig2.savefig(out2_pdf, bbox_inches="tight")
    plt.close(fig2)

    print(f"✓ Saved: {out1_png}")
    print(f"✓ Saved: {out2_png}")


if False and __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create paper-style overview figures.

We intentionally split the original "model summary" into 2 figures:

Figure 1: Class curves (1x4)
  (a) Random/blocky
  (b) Gradient
  (c) Alternating
  (d) Schematic: how classes are split (diagonal crossing + integral I_rand)

Figure 2: Model performance (1x3)
  (a) Confusion matrix on the TEST set (UNFILTERED; XGB-only predictions)
  (b) Performance comparison: with vs without reaction conditions (macro + per-class)
  (c) SHAP feature importance (top N)

Usage (repo root):
  python copol_prediction/analysis/plot_model_figure.py

Outputs:
  copol_prediction/output/analysis/model_class_curves_figure.png/.pdf
  copol_prediction/output/analysis/model_performance_figure.png/.pdf
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Ensure copol_prediction/ is on sys.path when run as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # copol_prediction/
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from copolpredictor.inference import CopolymerPredictor
from copol_prediction.utils import load_data_split
from copol_prediction.analysis.analyze_model import get_class_label
from copol_prediction.analysis import plot_class_curves as pcc
from copol_prediction.analysis.plot_config import CLASS_CURVES_FIGSIZE_INCH, setup_plot_style


def parse_args():
    p = argparse.ArgumentParser(description="Create model overview figures (curves + performance).")
    p.add_argument(
        "--model-path",
        type=str,
        default=os.path.join(PROJECT_ROOT, "artifacts", "model_bundle"),
        help="Path to final model bundle (default: copol_prediction/artifacts/model_bundle).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(SCRIPT_DIR, "figures"),
        help="Directory to save the figure (default: copol_prediction/analysis/figures).",
    )
    p.add_argument(
        "--split-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "artifacts", "data_splits"),
        help="Directory with train/val/test CSVs (default: copol_prediction/artifacts/data_splits).",
    )
    p.add_argument(
        "--reaction-conditions-json",
        type=str,
        default=os.path.join(WORKSPACE_ROOT, "experiments", "reaction_conditions_comparison", "results", "comparison_results.json"),
        help="Path to reaction conditions comparison_results.json.",
    )
    p.add_argument(
        "--shap-csv",
        type=str,
        default=os.path.join(WORKSPACE_ROOT, "experiments", "permutation_importance", "results", "shap_average_strong_groups.csv"),
        help="Path to strongly-grouped average SHAP CSV (from experiments/permutation_importance).",
    )
    p.add_argument(
        "--shap-top-n",
        type=int,
        default=10,
        help="Top N SHAP groups to show (default: 10).",
    )
    p.add_argument(
        "--max-curves-per-class",
        type=int,
        default=1500,
        help="Max curves per class for class-curves panels (default: 1500).",
    )
    p.add_argument(
        "--band-quantiles",
        type=float,
        nargs=2,
        default=(10.0, 90.0),
        help="Quantiles for the class-curves band, e.g. 10 90.",
    )
    return p.parse_args()


def _plot_confusion_matrix_unfiltered(ax, predictor: CopolymerPredictor, df_test: pd.DataFrame):
    X = df_test[predictor.features]
    y_true = df_test["r_product_class"].astype(int).values
    y_pred = predictor.predict(X).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[get_class_label(i, style="short") for i in [0, 1, 2]],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("A  Confusion matrix (test, unfiltered)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")


def _plot_reaction_conditions_comparison(ax, json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        saved = json.load(f)
    w = saved["with_conditions"]
    wo = saved["without_conditions"]

    labels = [
        "Within-pair\nbal. acc",
        "Macro acc",
        "Coverage",
    ]
    with_pair = float(w.get("pair_balanced_accuracy", w["macro_accuracy"]))
    without_pair = float(wo.get("pair_balanced_accuracy", wo["macro_accuracy"]))
    with_vals = [with_pair, float(w["macro_accuracy"]), float(w["coverage"])]
    without_vals = [without_pair, float(wo["macro_accuracy"]), float(wo["coverage"])]

    x = np.arange(len(labels))
    width = 0.35
    b1 = ax.bar(x - width / 2, with_vals, width, label="With cond.", color="#661124", alpha=0.85)
    b2 = ax.bar(x + width / 2, without_vals, width, label="Without cond.", color="#143D60", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("B  With vs without reaction conditions")
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower left")

    for bars in (b1, b2):
        for bar in bars:
            v = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", va="bottom")


def _plot_shap_topn(ax, csv_path: str, top_n: int):
    df = pd.read_csv(csv_path)
    df = df.sort_values("importance_mean", ascending=False).head(int(top_n)).copy()
    df = df.iloc[::-1]  # for barh top at top

    ax.barh(
        df["group_label"].astype(str),
        df["importance_mean"].astype(float).values,
        color="#661124",
        alpha=0.9,
    )
    ax.set_title("C  SHAP feature importance (avg |SHAP|, grouped)")
    ax.set_xlabel("mean |SHAP| (avg over 3 classes)")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_class_curves_row(axes_row, df_all: pd.DataFrame, *, max_curves_per_class: int, band_quantiles):
    f1 = np.linspace(0, 1, 501)
    class_curves = pcc._sample_per_class(
        df_all,
        f1=f1,
        max_curves_per_class=max_curves_per_class,
    )

    panels = [
        ("alternating", "A  Alternating"),
        ("random (to blocky)", "B  Random"),
        ("gradient", "C  Gradient"),
    ]

    for ax, (label, title) in zip(axes_row[:3], panels):
        pcc._plot_dual_band(
            ax,
            f1,
            class_curves.get(label, np.empty((0, len(f1)))),
            label=title,
            band_quantiles=band_quantiles,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$f_1$ (feed conc. monomer 1)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes_row[0].set_ylabel(r"$F_1$ (monomer 1 proportion in polymer)")

    # 4th panel: schematic explanation of class split
    if len(axes_row) >= 4 and hasattr(pcc, "_plot_class_split_explanation"):
        pcc._plot_class_split_explanation(axes_row[3])
        axes_row[3].set_title("D  Class determination")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_plot_style()

    # Data
    df_train, df_val, df_test = load_data_split.load_train_val_test_split(split_dir=args.split_dir)
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    # Model
    predictor = CopolymerPredictor(args.model_path)

    # Figure 1: class curves (1x4)
    fig1, axes1 = plt.subplots(1, 4, figsize=CLASS_CURVES_FIGSIZE_INCH, sharex=True, sharey=True)
    _plot_class_curves_row(
        axes1,
        df_all,
        max_curves_per_class=args.max_curves_per_class,
        band_quantiles=tuple(args.band_quantiles),
    )
    plt.tight_layout()
    out1_png = os.path.join(args.output_dir, "model_class_curves_figure.png")
    out1_pdf = os.path.join(args.output_dir, "model_class_curves_figure.pdf")
    fig1.savefig(out1_png, dpi=300, bbox_inches="tight")
    fig1.savefig(out1_pdf, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: performance plots (1x3)
    fig2, axes2 = plt.subplots(1, 3, figsize=(13.5, 3.8))
    _plot_confusion_matrix_unfiltered(axes2[0], predictor, df_test)
    _plot_reaction_conditions_comparison(axes2[1], args.reaction_conditions_json)
    _plot_shap_topn(axes2[2], args.shap_csv, args.shap_top_n)
    plt.tight_layout()
    out2_png = os.path.join(args.output_dir, "model_performance_figure.png")
    out2_pdf = os.path.join(args.output_dir, "model_performance_figure.pdf")
    fig2.savefig(out2_png, dpi=300, bbox_inches="tight")
    fig2.savefig(out2_pdf, bbox_inches="tight")
    plt.close(fig2)

    print(f"✓ Saved: {out1_png}")
    print(f"✓ Saved: {out2_png}")


if False and __name__ == "__main__":
    main()

