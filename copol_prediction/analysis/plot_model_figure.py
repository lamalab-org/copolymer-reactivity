#!/usr/bin/env python3
"""
Create paper-style overview figures.

Figure 1: Class curves (1x4)
  (a) Alternating
  (b) Random
  (c) Gradient
  (d) Schematic: how classes are split (diagonal crossing + integral I_rand)

Figure 2: Model performance (1x3)
  (a) Confusion matrix on the TEST set (UNFILTERED; XGB-only predictions)
  (b) Prediction transition Sankey: without vs with reaction conditions
  (c) SHAP feature importance (top N)

Usage (repo root):
  python -m copol_prediction.analysis.plot_model_figure

Outputs (default):
  copol_prediction/analysis/figures/model_class_curves_figure.png/.pdf
  copol_prediction/analysis/figures/model_performance_figure.png/.pdf
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Ensure copol_prediction/ is on sys.path when run as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # copol_prediction/
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from copol_prediction.analysis import plot_class_curves as pcc
from copol_prediction.analysis.analyze_model import get_class_label
from copol_prediction.analysis.plot_config import (
    CLASS_CURVES_FIGSIZE_INCH,
    CLASS_LABELS_SHORT,
    TWO_COL_WIDTH_INCH,
    setup_plot_style,
)
from copol_prediction.utils import load_data_split
from copolpredictor.inference import CopolymerPredictor

_SANKEY_CLASS_ORDER = [0, 1, 2, -1]
_SANKEY_CLASS_LABELS = [
    CLASS_LABELS_SHORT.get(0, "Alternating"),
    CLASS_LABELS_SHORT.get(1, "Random"),
    CLASS_LABELS_SHORT.get(2, "Gradient"),
    "No prediction",
]
_SANKEY_CLASS_COLORS = ["#1e8db9", "#9ed5f2", "#ffbc57", "#8A8A8A"]


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
        help="Directory to save figures (default: copol_prediction/analysis/figures).",
    )
    p.add_argument(
        "--split-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "artifacts", "data_splits"),
        help="Directory with train/val/test CSVs.",
    )
    p.add_argument(
        "--sankey-matrix-csv",
        type=str,
        default=os.path.join(
            WORKSPACE_ROOT,
            "experiments",
            "reaction_conditions_comparison",
            "results",
            "prediction_transition_sankey_special_subset_voting_matrix.csv",
        ),
        help="Path to the Sankey transition matrix CSV (from run_comparison.py).",
    )
    p.add_argument(
        "--shap-csv",
        type=str,
        default=os.path.join(
            WORKSPACE_ROOT,
            "experiments",
            "permutation_importance",
            "results",
            "shap_importance_detailed.csv",
        ),
        help="Path to SHAP importance CSV.",
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


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


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
    ax.set_aspect("auto")
    ax.set_title("A", fontsize=10, loc="left")
    ax.set_xlabel("Predicted", fontsize=8)
    ax.set_ylabel("True", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")


def _draw_sankey_on_ax(
    ax, flow: np.ndarray, class_order: list, class_labels: list, class_colors: list
) -> None:
    """Draw a Sankey-style transition diagram on *ax* (no figure creation/saving)."""
    n_classes = len(class_order)
    total = int(flow.sum())
    if total == 0:
        return

    left_totals = flow.sum(axis=1)
    right_totals = flow.sum(axis=0)

    x_left = 0.15
    x_right = 0.85
    node_w = 0.08
    y_top = 0.82
    y_bottom = 0.05
    gap = 0.03
    usable_h = y_top - y_bottom - 2 * gap

    left_heights = (left_totals / total) * usable_h
    right_heights = (right_totals / total) * usable_h

    left_y0 = np.zeros(n_classes)
    right_y0 = np.zeros(n_classes)
    cursor = y_top
    for i in range(n_classes):
        left_y0[i] = cursor - left_heights[i]
        cursor = left_y0[i] - gap
    cursor = y_top
    for j in range(n_classes):
        right_y0[j] = cursor - right_heights[j]
        cursor = right_y0[j] - gap

    for i in range(n_classes):
        ax.add_patch(
            Rectangle(
                (x_left - node_w / 2, left_y0[i]),
                node_w,
                left_heights[i],
                facecolor=class_colors[i],
                edgecolor="white",
                linewidth=1.0,
                alpha=0.9,
                zorder=3,
            )
        )
        ax.text(
            x_left - node_w / 2 - 0.02,
            left_y0[i] + left_heights[i] / 2,
            f"{class_labels[i]}\n(n={int(left_totals[i])})",
            ha="right",
            va="center",
            fontsize=8,
        )

    for j in range(n_classes):
        ax.add_patch(
            Rectangle(
                (x_right - node_w / 2, right_y0[j]),
                node_w,
                right_heights[j],
                facecolor=class_colors[j],
                edgecolor="white",
                linewidth=1.0,
                alpha=0.9,
                zorder=3,
            )
        )
        ax.text(
            x_right + node_w / 2 + 0.02,
            right_y0[j] + right_heights[j] / 2,
            f"{class_labels[j]}\n(n={int(right_totals[j])})",
            ha="left",
            va="center",
            fontsize=8,
        )

    left_offsets = left_y0.copy()
    right_offsets = right_y0.copy()
    cx1 = x_left + 0.22
    cx2 = x_right - 0.22
    x0 = x_left + node_w / 2
    x1 = x_right - node_w / 2

    for i in range(n_classes):
        for j in range(n_classes):
            n_ij = int(flow[i, j])
            if n_ij == 0:
                continue
            h = (n_ij / total) * usable_h
            y0b = left_offsets[i]
            y0t = y0b + h
            y1b = right_offsets[j]
            y1t = y1b + h
            left_offsets[i] = y0t
            right_offsets[j] = y1t

            verts = [
                (x0, y0b),
                (cx1, y0b),
                (cx2, y1b),
                (x1, y1b),
                (x1, y1t),
                (cx2, y1t),
                (cx1, y0t),
                (x0, y0t),
                (x0, y0b),
            ]
            codes = [
                Path.MOVETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.LINETO,
                Path.CURVE4,
                Path.CURVE4,
                Path.CURVE4,
                Path.CLOSEPOLY,
            ]
            ax.add_patch(
                PathPatch(
                    Path(verts, codes),
                    facecolor=class_colors[i],
                    edgecolor="none",
                    alpha=0.42,
                    zorder=2,
                )
            )
            if n_ij >= 8:
                xm = 0.5 * (x0 + x1)
                ym = 0.5 * ((y0b + y0t) / 2 + (y1b + y1t) / 2)
                ax.text(xm, ym, str(n_ij), ha="center", va="center", fontsize=8, color="#1f1f1f")

    ax.text(x_left, y_top + 0.07, "w/o cond.", ha="center", va="bottom", fontsize=8)
    ax.text(x_right, y_top + 0.07, "w/ cond.", ha="center", va="bottom", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def _plot_sankey_from_csv(ax, csv_path: str) -> None:
    """Load the saved transition matrix CSV and draw the Sankey on *ax*."""
    try:
        df = pd.read_csv(csv_path, index_col=0)
    except FileNotFoundError:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Sankey matrix not found.\nRun run_comparison.py first.",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_title("B", fontsize=10, loc="left")
        return

    # Columns are like "with_0", "with_1", "with_2", "with_-1"
    # Index rows are like "without_0", "without_1", "without_2", "without_-1"
    def _parse_key(s: str) -> int:
        return int(s.split("_", 1)[1])

    row_order = [_parse_key(r) for r in df.index]
    col_order = [_parse_key(c) for c in df.columns]

    if row_order != col_order:
        # Align columns to row order
        df = df[[f"with_{k}" for k in row_order]]

    flow = df.values.astype(int)
    class_order = row_order

    # Map class_order positions to labels/colors
    label_map = dict(zip(_SANKEY_CLASS_ORDER, _SANKEY_CLASS_LABELS))
    color_map = dict(zip(_SANKEY_CLASS_ORDER, _SANKEY_CLASS_COLORS))
    labels = [label_map.get(c, str(c)) for c in class_order]
    colors = [color_map.get(c, "#999999") for c in class_order]

    _draw_sankey_on_ax(ax, flow, class_order, labels, colors)
    ax.set_title("B", fontsize=10, loc="left")


def _plot_shap_topn(ax, csv_path: str, top_n: int):
    def _shap_bar_color(label: str) -> str:
        condition_keywords = {"temperature", "polymerization", "polytype", "method_emb", "solvent"}
        if any(kw in str(label).lower() for kw in condition_keywords):
            return "#ffbc57"
        return "#9ed5f2"

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
            fontsize=8,
        )
        return

    df = df.sort_values("importance_mean", ascending=False).head(int(top_n)).copy()
    labels = df["group_label"].astype(str).tolist()
    means = df["importance_mean"].astype(float).values
    stds = (
        df["importance_std"].astype(float).values
        if "importance_std" in df.columns
        else np.zeros(len(means))
    )

    y_pos = np.arange(len(labels))
    bar_colors = [_shap_bar_color(lbl) for lbl in labels]

    ax.barh(y_pos, means, xerr=stds, capsize=3, alpha=0.85, color=bar_colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("C", fontsize=10, loc="left")
    ax.set_xlabel("|SHAP| importance (mean ± std)", fontsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(facecolor="#9ed5f2", alpha=0.85, label="Monomer descriptor"),
            Patch(facecolor="#ffbc57", alpha=0.85, label="Reaction condition"),
        ],
        fontsize=8,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.18),
        ncol=1,
        frameon=False,
    )


def _plot_class_curves_row(
    axes_row, df_all: pd.DataFrame, *, max_curves_per_class: int, band_quantiles
):
    f1 = np.linspace(0, 1, 501)
    class_curves = pcc._sample_per_class(
        df_all,
        f1=f1,
        max_curves_per_class=max_curves_per_class,
    )

    panels = [
        ("alternating", "A  Alternating"),
        ("random", "B  Random"),
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
        ax.set_xlabel(r"$f_1$ (feed conc. monomer 1)", fontsize=12)
        ax.tick_params(labelsize=12)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes_row[0].set_ylabel(r"$F_1$ (monomer 1 proportion in polymer)", fontsize=12)

    if len(axes_row) >= 4 and hasattr(pcc, "_plot_class_split_explanation"):
        pcc._plot_class_split_explanation(axes_row[3])
        axes_row[3].set_title("D", fontsize=12, loc="left")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    setup_plot_style()
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 8,
        }
    )

    df_train, df_val, df_test = load_data_split.load_train_val_test_split(split_dir=args.split_dir)
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
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

    # Figure 2: performance (1x3) — confusion matrix | Sankey | SHAP
    fig2, axes2 = plt.subplots(
        1,
        3,
        figsize=(TWO_COL_WIDTH_INCH, TWO_COL_WIDTH_INCH / 3.0),
        gridspec_kw={"width_ratios": [1.3, 1, 1]},
        constrained_layout=True,
    )
    _plot_confusion_matrix_unfiltered(axes2[0], predictor, df_test)
    _plot_sankey_from_csv(axes2[1], args.sankey_matrix_csv)
    _plot_shap_topn(axes2[2], args.shap_csv, args.shap_top_n)
    out2_png = os.path.join(args.output_dir, "model_performance_figure.png")
    out2_pdf = os.path.join(args.output_dir, "model_performance_figure.pdf")
    fig2.savefig(out2_png, dpi=300, bbox_inches="tight")
    fig2.savefig(out2_pdf, bbox_inches="tight")
    plt.close(fig2)

    print(f"✓ Saved: {out1_png}")
    print(f"✓ Saved: {out2_png}")


if __name__ == "__main__":
    main()
