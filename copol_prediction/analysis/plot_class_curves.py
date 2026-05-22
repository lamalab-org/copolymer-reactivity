#!/usr/bin/env python3
"""
Create the "class curves" figure:
Mayo–Lewis F1(f1) families grouped by architecture class.

Figure: 1x3 subplots (random/blocky, gradient, alternating).
Each panel shows:
  - individual curves (faded)
  - diagonal reference (F1 = f1)
  - quantile band (default 10–90%) and mean curve highlighted

Data source:
  - uses the central train/val/test split (copol_prediction/artifacts/data_splits)
  - extracts (r1, r2) from columns constant_1 / constant_2

Usage (from repo root):
  python copol_prediction/analysis/plot_class_curves.py --output-dir copol_prediction/output/analysis
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure copol_prediction/ is on sys.path when run as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # copol_prediction/
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from copol_prediction.analysis.plot_config import TWO_COL_WIDTH_INCH, setup_plot_style
from copol_prediction.mayo_lewis_classification import (
    classify_reactivity_curve,
    compute_curve_descriptors,
    mayo_lewis,
)
from copol_prediction.utils import load_data_split


def _trapz(y: np.ndarray, x: np.ndarray) -> float:
    """
    Trapezoidal integration compatible with NumPy variants.
    """
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    return float(np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1]) * 0.5))


def parse_args():
    p = argparse.ArgumentParser(description="Plot Mayo–Lewis class curves (1x3 panels).")
    p.add_argument(
        "--split-dir",
        type=str,
        default=None,
        help="Directory with train/val/test CSVs (default: copol_prediction/artifacts/data_splits).",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "figures"),
        help="Directory to save plots (default: copol_prediction/analysis/figures).",
    )
    p.add_argument(
        "--n-f1",
        type=int,
        default=501,
        help="Number of f1 points for curves (default: 501).",
    )
    p.add_argument(
        "--max-curves-per-class",
        type=int,
        default=1500,
        help="Max number of curves to draw per class (default: 1500).",
    )
    p.add_argument(
        "--band-quantiles",
        type=float,
        nargs=2,
        default=(10.0, 90.0),
        help="Quantiles for band, e.g. 10 90 (default: 10 90).",
    )
    return p.parse_args()


def _resolve_split_dir(split_dir_arg: str | None) -> str:
    if split_dir_arg and os.path.isdir(split_dir_arg):
        return split_dir_arg
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # copol_prediction/
    default_dir = os.path.join(project_root, "artifacts", "data_splits")
    if split_dir_arg:
        return os.path.join(project_root, split_dir_arg)
    return default_dir


def _unique_reaction_constants(df: pd.DataFrame) -> pd.DataFrame:
    req = ["reaction_id", "constant_1", "constant_2"]
    missing = [c for c in req if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for class curves: {missing}")
    w = df[req].dropna().copy()
    w["reaction_id"] = w["reaction_id"].astype(str)
    # one row per reaction_id (constants should be stable within a reaction)
    w = w.drop_duplicates(subset=["reaction_id"])
    return w


def _sample_per_class(
    df_all: pd.DataFrame,
    *,
    f1: np.ndarray,
    max_curves_per_class: int,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Return dict class_label -> curves matrix (n_curves, len(f1)).
    """
    rng = np.random.default_rng(seed)
    w = _unique_reaction_constants(df_all)

    labels: List[str] = []
    curves: List[np.ndarray] = []
    for r1, r2 in w[["constant_1", "constant_2"]].itertuples(index=False, name=None):
        r1 = float(r1)
        r2 = float(r2)
        # Skip degenerate pairs that produce non-finite Mayo–Lewis curves
        # (e.g. denominator hits 0 for extreme or invalid inputs)
        F1 = mayo_lewis(f1, r1, r2)
        if not np.all(np.isfinite(F1)):
            continue

        res = classify_reactivity_curve(r1, r2, n_points=5000)
        label = res["class_name"]
        labels.append(label)
        curves.append(F1)

    df_idx = pd.DataFrame({"label": labels})
    curve_mat = np.vstack(curves) if curves else np.empty((0, len(f1)))

    out: Dict[str, np.ndarray] = {}
    for label in ["random (to blocky)", "gradient", "alternating"]:
        idx = np.where(df_idx["label"].values == label)[0]
        if len(idx) == 0:
            out[label] = np.empty((0, len(f1)))
            continue
        if len(idx) > max_curves_per_class:
            idx = rng.choice(idx, size=max_curves_per_class, replace=False)
        out[label] = curve_mat[idx]
    return out


def _signed_area(f1: np.ndarray, curve: np.ndarray) -> float:
    return float(_trapz(curve - f1, f1))


def _split_upper_lower(f1: np.ndarray, curves: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split curves by whether the signed area vs diagonal is >= 0 (upper) or < 0 (lower).
    """
    if curves.size == 0:
        return np.empty((0, len(f1))), np.empty((0, len(f1)))
    upper = []
    lower = []
    for c in curves:
        if _signed_area(f1, c) >= 0:
            upper.append(c)
        else:
            lower.append(c)
    upper = np.vstack(upper) if upper else np.empty((0, len(f1)))
    lower = np.vstack(lower) if lower else np.empty((0, len(f1)))
    return upper, lower


def _mirror_curve(y: np.ndarray) -> np.ndarray:
    # mirror around diagonal F1=f1: y_mirror(x) = 1 - y(1-x)
    return 1.0 - y[::-1]


def _downsample(x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x_old, y_old)


def _plot_dual_band(
    ax,
    f1: np.ndarray,
    curves: np.ndarray,
    *,
    label: str,
    band_quantiles=(10.0, 90.0),
    n_plot_points: int = 60,
):
    ax.plot(f1, f1, "--", lw=1.2, alpha=0.8, color="gray")
    if curves.size == 0:
        ax.set_title(label, loc="left", fontsize=10)
        return

    upper, lower = _split_upper_lower(f1, curves)
    f1_plot = np.linspace(0, 1, n_plot_points)

    # individual curves (faded)
    for c in upper:
        ax.plot(f1, c, alpha=0.03, lw=1.0, color="#ffbc57")
    for c in lower:
        ax.plot(f1, c, alpha=0.03, lw=1.0, color="#9ed5f2")

    def _band_and_mean(mat: np.ndarray):
        q_low, q_high = np.percentile(mat, band_quantiles, axis=0)
        mean = np.mean(mat, axis=0)
        return q_low, q_high, mean

    # upper band (direct)
    if len(upper) > 0:
        q_low_u, q_high_u, mean_u = _band_and_mean(upper)
        ax.fill_between(
            f1_plot,
            _downsample(f1, q_low_u, f1_plot),
            _downsample(f1, q_high_u, f1_plot),
            alpha=0.22,
            color="#ffbc57",
        )
        ax.plot(f1_plot, _downsample(f1, mean_u, f1_plot), lw=2.2, color="#ffbc57")

    # lower band: mirror trick (match notebook logic)
    if len(lower) > 0:
        lower_m = np.array([_mirror_curve(c) for c in lower])
        q_low_lm, q_high_lm, mean_lm = _band_and_mean(lower_m)
        q_low_l = _mirror_curve(q_high_lm)
        q_high_l = _mirror_curve(q_low_lm)
        mean_l = _mirror_curve(mean_lm)
        ax.fill_between(
            f1_plot,
            _downsample(f1, q_low_l, f1_plot),
            _downsample(f1, q_high_l, f1_plot),
            alpha=0.22,
            color="#9ed5f2",
        )
        ax.plot(f1_plot, _downsample(f1, mean_l, f1_plot), lw=2.2, color="#9ed5f2")

    ax.set_title(label, loc="left", fontsize=10)


def _plot_class_split_explanation(ax):
    """
    Schematic panel explaining how curves are split into classes:
    - diagonal intersection(s)
    - integrated deviation from the diagonal (∫ |F1 - f1| df1)
    """
    f1 = np.linspace(1e-4, 1.0 - 1e-4, 1500)

    # Three representative examples (fixed for reproducibility / paper stability)
    examples = [
        {
            "r1": 0.5,
            "r2": 0.5,
            "color": "#ffbc57",
            "label": "example: random-like",
            "text_x": 0.66,
            "text_y": 0.50,
            "extra": "\nIntersection = 0.5",
        },
        {
            "r1": 6.0,
            "r2": 0.2,
            "color": "#9ed5f2",
            "label": "example: gradient-like",
            "text_x": 0.02,
            "text_y": 0.96,
            "extra": "\nno intersection",
        },
        {
            "r1": 0.1,
            "r2": 0.1,
            "color": "#3e3888",
            "label": "example: alternating",
            "text_x": 0.36,
            "text_y": 0.18,
            "extra": "\nIntersection = 0.5",
        },
    ]

    ax.plot(f1, f1, "--", lw=1.2, alpha=0.8, color="gray", label="random line (F1=f1)")

    for ex in examples:
        desc = compute_curve_descriptors(ex["r1"], ex["r2"], n_points=len(f1))
        F1 = desc["F1"]

        ax.plot(f1, F1, lw=2.2, color=ex["color"], alpha=0.98, label=ex["label"], zorder=3)

        # Shade |F1 - f1| area (I_rand)
        above = F1 >= f1
        ax.fill_between(f1, F1, f1, where=above, interpolate=True, color=ex["color"], alpha=0.12)
        ax.fill_between(f1, F1, f1, where=~above, interpolate=True, color=ex["color"], alpha=0.12)

        # Mark main interior diagonal intersection (if present)
        if desc["has_crossing"] and desc["crossing_main"] is not None:
            x = float(desc["crossing_main"])
            ax.scatter([x], [x], s=28, color=ex["color"], zorder=5)

        ax.text(
            ex["text_x"],
            ex["text_y"],
            f"$r_1$={ex['r1']:.2g}, $r_2$={ex['r2']:.2g}\n"
            + f"Integral = {desc['I_rand']:.3f}"
            + ex["extra"],
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color=ex["color"],
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.7),
        )

    ax.set_title("D", loc="left", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$f_1$", fontsize=8)
    ax.set_ylabel(r"$F_1$", fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)


def plot_class_curves(
    *,
    df_all: pd.DataFrame,
    output_dir: str,
    n_f1: int = 501,
    max_curves_per_class: int = 400,
    band_quantiles=(10.0, 90.0),
) -> str:
    setup_plot_style()
    os.makedirs(output_dir, exist_ok=True)
    f1 = np.linspace(0, 1, int(n_f1))
    class_curves = _sample_per_class(df_all, f1=f1, max_curves_per_class=max_curves_per_class)

    classes = ["alternating", "random (to blocky)", "gradient"]
    panel_titles = {
        "alternating": "A  Alternating",
        "random (to blocky)": "B  Random",
        "gradient": "C  Gradient",
    }

    fig = plt.figure(figsize=(TWO_COL_WIDTH_INCH, TWO_COL_WIDTH_INCH * 0.75), layout="constrained")
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 2.5])

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a, sharey=ax_a)
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_a, sharey=ax_a)
    ax_d = fig.add_subplot(gs[:, 1])

    axes_abc = [ax_a, ax_b, ax_c]
    for ax, label in zip(axes_abc, classes):
        _plot_dual_band(
            ax,
            f1,
            class_curves.get(label, np.empty((0, len(f1)))),
            label=panel_titles.get(label, label),
            band_quantiles=band_quantiles,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.tick_params(labelsize=8)
        ax.set_ylabel(r"$F_1$", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Only bottom panel gets x-axis label; hide tick labels on upper two
    for ax in [ax_a, ax_b]:
        plt.setp(ax.get_xticklabels(), visible=False)
    ax_c.set_xlabel(r"$f_1$", fontsize=8)

    _plot_class_split_explanation(ax_d)

    out_png = os.path.join(output_dir, "class_curves.png")
    out_pdf = os.path.join(output_dir, "class_curves.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    return out_png


def main():
    args = parse_args()
    split_dir = _resolve_split_dir(args.split_dir)
    df_train, df_val, df_test = load_data_split.load_train_val_test_split(split_dir=split_dir)
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    out = plot_class_curves(
        df_all=df_all,
        output_dir=os.path.normpath(args.output_dir),
        n_f1=args.n_f1,
        max_curves_per_class=args.max_curves_per_class,
        band_quantiles=tuple(args.band_quantiles),
    )
    print(f"✓ Saved class curves to: {out}")


if __name__ == "__main__":
    main()
