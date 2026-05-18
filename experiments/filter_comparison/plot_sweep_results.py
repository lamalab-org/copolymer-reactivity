#!/usr/bin/env python3
"""
Plot generation script for voting-model filter sweep results.

Reads results from sweep_filters.py and generates all plots.
Can be run independently to regenerate plots without re-training.

Usage:
    python plot_sweep_results.py [--results-path PATH] [--plots-dir DIR]
"""

import argparse
import ast
import os
import sys
from decimal import ROUND_HALF_UP, Decimal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "copol_prediction"))

try:
    from copol_prediction.analysis.plot_config import (
        CLASS_COLORS,
        CONFUSION_MATRIX_CONFIG,
        HEATMAP_CMAP,
        ONE_COL_WIDTH_INCH,
        TWO_COL_WIDTH_INCH,
        get_class_label,
        setup_plot_style,
    )
except ImportError:

    def setup_plot_style():
        pass

    HEATMAP_CMAP = "Blues"
    TWO_COL_WIDTH_INCH = 7
    ONE_COL_WIDTH_INCH = 3.5
    CLASS_COLORS = {0: "#3A3B73", 1: "#e27f07", 2: "#6a040f"}
    CONFUSION_MATRIX_CONFIG = {"cmap": "Blues", "values_format": "d"}

    def get_class_label(cid, style="default"):
        labels = {0: "Alternating", 1: "Block-like", 2: "Homopolymer"}
        return labels.get(cid, f"Class {cid}")


# Style for matplotlib
try:
    _STYLE_PATH = os.path.join(_PROJECT_ROOT, "copol_prediction", "analysis", "lamalab.mplstyle")
    if os.path.exists(_STYLE_PATH):
        plt.style.use(_STYLE_PATH)
except Exception:
    pass

#
# Note: The previous "negative data" sweep dimension was removed.
# Plotting assumes results contain only (spec, poly, aug) combinations.


# ---------------------------------------------------------------------------
# Rounding helper: always round 0.5 up
# ---------------------------------------------------------------------------
def round_up_half(val, decimals=2):
    """Round to decimals places, always rounding 0.5 up (not banker's rounding).

    Example: 0.725 -> 0.73 (third decimal >= 5, so round second decimal up)
    """
    if pd.isna(val) or np.isnan(val):
        return np.nan
    # Use Decimal with ROUND_HALF_UP to ensure 0.5 always rounds up
    # Convert to Decimal via string to avoid floating point precision issues
    val_decimal = Decimal(str(val))
    quantize_str = "0." + "0" * decimals
    quantize_decimal = Decimal(quantize_str)
    rounded = val_decimal.quantize(quantize_decimal, rounding=ROUND_HALF_UP)
    return float(rounded)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate plots from voting-model filter sweep results"
    )
    parser.add_argument(
        "--results-path",
        type=str,
        default="experiments/filter_comparison/output/voting_sweep/sweep_results.csv",
    )
    parser.add_argument(
        "--plots-dir", type=str, default="experiments/filter_comparison/output/voting_sweep"
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Heatmap (single): rows=(spec×poly), cols=(aug)
# ---------------------------------------------------------------------------
def plot_heatmap_grid(
    results_df, plots_dir, metric="macro_accuracy", metric_label="Macro Accuracy"
):
    """Create a single 4×2 heatmap.

    - Rows: (spec × poly) → 4
    - Cols: augmentation  → 2
    """
    setup_plot_style()

    fig, ax = plt.subplots(1, 1, figsize=(ONE_COL_WIDTH_INCH * 1.15, ONE_COL_WIDTH_INCH * 1.10))

    row_labels = []
    for spec in [False, True]:
        for poly in [False, True]:
            row_labels.append(
                "Specialized reaction: "
                + ("on" if spec else "off")
                + "\nPolymerization type: "
                + ("on" if poly else "off")
            )

    col_labels = ["off", "on"]

    # Round vmin/vmax to 2 decimal places (0.5 always rounds up)
    vmin = round_up_half(results_df[metric].min(), decimals=2)
    vmax = round_up_half(results_df[metric].max(), decimals=2)

    matrix = np.full((4, 2), np.nan)
    annot_matrix = np.empty((4, 2), dtype=object)
    annot_matrix.fill("")

    for _, row in results_df.iterrows():
        r_idx = int(row["remove_specialized"]) * 2 + int(row["apply_polymerization_filter"])
        c_idx = int(row["use_augmentation"])
        val = row[metric]
        rounded_val = round_up_half(val, decimals=2)
        matrix[r_idx, c_idx] = rounded_val
        if not np.isnan(rounded_val):
            annot_matrix[r_idx, c_idx] = f"{rounded_val:.2f}"

    sns.heatmap(
        matrix,
        annot=annot_matrix,
        fmt="",
        cmap=HEATMAP_CMAP,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"fontsize": 11},
        cbar=True,
        cbar_kws={"label": metric_label, "shrink": 0.8},
    )
    ax.set_title(metric_label, fontsize=11, fontweight="bold")
    ax.set_xlabel("Augmentation", fontsize=10, fontweight="bold")
    ax.set_ylabel("Filters", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.grid(False)

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        path = os.path.join(plots_dir, f"heatmap_grid_{metric}.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  ✓ Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Combined heatmap: rows=(spec×poly), cols=(aug)
# ---------------------------------------------------------------------------
def plot_combined_heatmap(
    results_df, plots_dir, metric="macro_accuracy", metric_label="Macro Accuracy"
):
    """Create a single 4×2 heatmap showing all combinations."""
    setup_plot_style()

    row_labels = []
    for spec in [False, True]:
        for poly in [False, True]:
            row_labels.append(
                "Specialized reaction: "
                + ("on" if spec else "off")
                + "\nPolymerization type: "
                + ("on" if poly else "off")
            )

    col_labels = ["off", "on"]

    # Round vmin/vmax to 2 decimal places (0.5 always rounds up)
    vmin = round_up_half(results_df[metric].min(), decimals=2)
    vmax = round_up_half(results_df[metric].max(), decimals=2)

    matrix = np.full((4, 2), np.nan)
    annot_matrix = np.empty((4, 2), dtype=object)
    annot_matrix.fill("")

    for _, row in results_df.iterrows():
        r_idx = int(row["remove_specialized"]) * 2 + int(row["apply_polymerization_filter"])
        c_idx = int(row["use_augmentation"])
        val = row[metric]
        rounded_val = round_up_half(val, decimals=2)
        matrix[r_idx, c_idx] = rounded_val
        if not np.isnan(rounded_val):
            annot_matrix[r_idx, c_idx] = f"{rounded_val:.2f}"

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_INCH * 1.30, ONE_COL_WIDTH_INCH * 1.10))

    sns.heatmap(
        matrix,
        annot=annot_matrix,
        fmt="",
        cmap=HEATMAP_CMAP,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"fontsize": 11},
        cbar_kws={"label": metric_label, "shrink": 0.8},
    )
    ax.set_xlabel("Augmentation", fontsize=10, fontweight="bold")
    ax.set_ylabel("Filters", fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)
    ax.grid(False)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(plots_dir, f"heatmap_combined_{metric}.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  ✓ Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Sorted bar chart
# ---------------------------------------------------------------------------
def plot_sorted_bar(results_df, plots_dir, metric="macro_accuracy", metric_label="Macro Accuracy"):
    """Horizontal bar chart of all runs sorted by metric."""
    setup_plot_style()

    df_sorted = results_df.sort_values(metric, ascending=True)

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, max(4, len(df_sorted) * 0.28)))
    ax.barh(range(len(df_sorted)), df_sorted[metric], color="#3A3B73", alpha=0.85)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["run_name"], fontsize=7)
    ax.set_xlabel(metric_label, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(plots_dir, f"sorted_bar_{metric}.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  ✓ Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Coverage heatmap
# ---------------------------------------------------------------------------
def plot_coverage_heatmap(results_df, plots_dir):
    """2×2 grid of heatmaps showing coverage per neg_data_target."""
    plot_heatmap_grid(
        results_df, plots_dir, metric="coverage", metric_label="Coverage (models agree)"
    )


def plot_accuracy_and_coverage_side_by_side(results_df, plots_dir):
    """Create a single figure: Macro Accuracy and Coverage heatmaps side-by-side."""
    setup_plot_style()

    row_labels = []
    for spec in [False, True]:
        for poly in [False, True]:
            row_labels.append(
                "Specialized reaction: "
                + ("on" if spec else "off")
                + "\nPolymerization type: "
                + ("on" if poly else "off")
            )
    col_labels = ["off", "on"]

    def _build(metric: str):
        vmin = round_up_half(results_df[metric].min(), decimals=2)
        vmax = round_up_half(results_df[metric].max(), decimals=2)
        matrix = np.full((4, 2), np.nan)
        annot_matrix = np.empty((4, 2), dtype=object)
        annot_matrix.fill("")
        for _, row in results_df.iterrows():
            r_idx = int(row["remove_specialized"]) * 2 + int(row["apply_polymerization_filter"])
            c_idx = int(row["use_augmentation"])
            val = row[metric]
            rounded_val = round_up_half(val, decimals=2)
            matrix[r_idx, c_idx] = rounded_val
            if not np.isnan(rounded_val):
                annot_matrix[r_idx, c_idx] = f"{rounded_val:.2f}"
        return matrix, annot_matrix, vmin, vmax

    acc_m, acc_a, acc_vmin, acc_vmax = _build("macro_accuracy")
    cov_m, cov_a, cov_vmin, cov_vmax = _build("coverage")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(TWO_COL_WIDTH_INCH * 1.10, ONE_COL_WIDTH_INCH * 1.10),
    )

    sns.heatmap(
        acc_m,
        annot=acc_a,
        fmt="",
        cmap=HEATMAP_CMAP,
        xticklabels=col_labels,
        yticklabels=row_labels,
        ax=axes[0],
        vmin=acc_vmin,
        vmax=acc_vmax,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"fontsize": 11},
        cbar=True,
        cbar_kws={"label": "Macro Accuracy", "shrink": 0.8},
    )
    axes[0].set_yticklabels(row_labels, rotation=0)
    axes[0].set_title("Macro Accuracy", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Augmentation", fontsize=10, fontweight="bold")
    axes[0].set_ylabel("Filters", fontsize=10, fontweight="bold")
    axes[0].tick_params(labelsize=8)
    axes[0].tick_params(axis="y", pad=10)
    axes[0].grid(False)

    sns.heatmap(
        cov_m,
        annot=cov_a,
        fmt="",
        cmap=HEATMAP_CMAP,
        xticklabels=col_labels,
        yticklabels=False,
        ax=axes[1],
        vmin=cov_vmin,
        vmax=cov_vmax,
        linewidths=0.5,
        linecolor="gray",
        annot_kws={"fontsize": 11},
        cbar=True,
        cbar_kws={"label": "Coverage", "shrink": 0.8},
    )
    axes[1].set_title("Coverage", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Augmentation", fontsize=10, fontweight="bold")
    axes[1].set_ylabel("")
    axes[1].tick_params(labelleft=False)
    axes[1].tick_params(labelsize=8)
    axes[1].grid(False)

    # Ensure both heatmaps align vertically
    axes[1].set_ylim(axes[0].get_ylim())

    # Make room for multi-line y tick labels on the left.
    fig.subplots_adjust(left=0.42, wspace=0.35)
    for ext in ["png", "pdf"]:
        path = os.path.join(plots_dir, f"heatmap_macro_accuracy_and_coverage.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  ✓ Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# 4x4 Grid of Confusion Matrices
# ---------------------------------------------------------------------------
def plot_confusion_matrix_grid(results_df, plots_dir):
    """Create a SINGLE 4×2 grid showing confusion matrices for all filter combinations.

    Grid structure:
    - Rows: (spec, poly) combinations (4 rows)
    - Columns: augmentation on/off (2 columns)
    Each cell contains a 3×3 confusion matrix for the voting model.
    """
    setup_plot_style()

    # Create 4x2 grid with larger figure size for better visibility
    fig, axes = plt.subplots(4, 2, figsize=(TWO_COL_WIDTH_INCH * 1.6, TWO_COL_WIDTH_INCH * 2.2))

    # Row labels: (spec, poly) combinations
    row_labels = []
    for spec in [False, True]:
        for poly in [False, True]:
            s = "Spec+" if spec else "Spec-"
            p = "Poly+" if poly else "Poly-"
            row_labels.append(f"{s} / {p}")

    # Column labels: augmentation combinations
    col_labels = ["Aug-", "Aug+"]

    # Initialize a dictionary to store matrices by position
    matrix_dict = {}
    for _, result_row in results_df.iterrows():
        spec = result_row["remove_specialized"]
        poly = result_row["apply_polymerization_filter"]
        aug = result_row["use_augmentation"]

        # Calculate grid position
        row_idx = int(spec) * 2 + int(poly)
        col_idx = int(aug)

        # Convert confusion matrix to numpy array (handle string format from CSV)
        cm_raw = result_row["confusion_matrix"]
        if isinstance(cm_raw, str):
            import ast

            cm_raw = ast.literal_eval(cm_raw)
        cm = np.array(cm_raw, dtype=int)
        if cm.size > 0 and cm.sum() > 0:
            matrix_dict[(row_idx, col_idx)] = cm

    # Plot each combination
    for row_idx in range(4):
        for col_idx in range(2):
            ax = axes[row_idx, col_idx]

            if (row_idx, col_idx) not in matrix_dict:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=8,
                )
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                cm = matrix_dict[(row_idx, col_idx)]

                # Normalize confusion matrix for better visualization
                cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
                cm_norm = np.nan_to_num(cm_norm)

                disp = ConfusionMatrixDisplay(
                    confusion_matrix=cm_norm, display_labels=[get_class_label(i) for i in range(3)]
                )
                disp.plot(
                    cmap=CONFUSION_MATRIX_CONFIG.get("cmap", "Blues"),
                    ax=ax,
                    values_format=".2f",
                    im_kw={"vmin": 0, "vmax": 1},
                    text_kw={"fontsize": 7},
                )

                # Remove colorbar from individual subplots
                if disp.im_ is not None and disp.im_.colorbar is not None:
                    disp.im_.colorbar.remove()

                # Add counts as text overlay (smaller font, below normalized values)
                for i in range(3):
                    for j in range(3):
                        count = cm[i, j]
                        if count > 0:
                            ax.text(
                                j,
                                i + 0.35,
                                f"({count})",
                                ha="center",
                                va="center",
                                fontsize=5,
                                color="gray",
                                alpha=0.7,
                            )

            # Set labels only on outer edges
            if row_idx == 3:
                ax.set_xlabel(col_labels[col_idx], fontsize=9, fontweight="bold")
            else:
                ax.set_xlabel("")
                ax.set_xticklabels([])

            if col_idx == 0:
                ax.set_ylabel(row_labels[row_idx], fontsize=9, fontweight="bold")
            else:
                ax.set_ylabel("")
                ax.set_yticklabels([])

            ax.tick_params(labelsize=7)
            ax.grid(False)

    # Add overall title
    fig.suptitle(
        "Confusion Matrices: Voting Model (All Filter Combinations)",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    # Add colorbar for the whole figure (shared across all subplots)
    if len(matrix_dict) > 0:
        # Use the first subplot's image for colorbar
        first_ax = None
        for row_idx in range(4):
            for col_idx in range(2):
                if (row_idx, col_idx) in matrix_dict and len(axes[row_idx, col_idx].images) > 0:
                    first_ax = axes[row_idx, col_idx]
                    break
            if first_ax is not None:
                break

        if first_ax is not None and len(first_ax.images) > 0:
            im = first_ax.images[0]
            cbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
            cbar.set_label("Normalized Count", fontsize=10)
            cbar.ax.tick_params(labelsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    for ext in ["png", "pdf"]:
        path = os.path.join(plots_dir, f"confusion_matrix_grid_4x2.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  ✓ Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def plot_sweep_results(results_df, plots_dir):
    """Generate all plots from sweep results."""
    setup_plot_style()
    os.makedirs(plots_dir, exist_ok=True)

    # Backward compatibility: older sweeps may contain extra dimensions
    # (e.g., negative-data variants). Collapse to (spec, poly, aug).
    key_cols = ["remove_specialized", "apply_polymerization_filter", "use_augmentation"]
    extra_cols = [c for c in results_df.columns if c not in key_cols]
    if results_df.duplicated(subset=key_cols).any():
        agg_map = {}
        for c in results_df.columns:
            if c in key_cols:
                continue
            if c == "confusion_matrix":
                agg_map[c] = "first"
            elif c == "run_name":
                agg_map[c] = "first"
            elif pd.api.types.is_numeric_dtype(results_df[c]):
                agg_map[c] = "mean"
            else:
                agg_map[c] = "first"
        results_df = results_df.groupby(key_cols, as_index=False).agg(agg_map)

    # Round all numeric metrics to 2 decimal places BEFORE plotting (0.5 always rounds up)
    results_df = results_df.copy()
    print(f"\n  Rounding values to 2 decimal places:")
    for metric_col in ["macro_accuracy", "macro_precision", "coverage"]:
        if metric_col in results_df.columns:
            sample_before = results_df[metric_col].dropna().head(5).tolist()
            results_df[metric_col] = results_df[metric_col].apply(
                lambda x: round_up_half(x, decimals=2)
            )
            sample_after = results_df[metric_col].dropna().head(5).tolist()
            print(f"    {metric_col}:")
            for b, a in zip(sample_before, sample_after):
                print(f"      {b} -> {a}")

    print("\n  Generating heatmap grids …")
    plot_heatmap_grid(results_df, plots_dir, "macro_accuracy", "Macro Accuracy")
    plot_heatmap_grid(results_df, plots_dir, "macro_precision", "Macro Precision")

    print("\n  Generating combined heatmaps …")
    plot_combined_heatmap(results_df, plots_dir, "macro_accuracy", "Macro Accuracy")
    plot_combined_heatmap(results_df, plots_dir, "macro_precision", "Macro Precision")

    print("\n  Generating coverage heatmaps …")
    plot_coverage_heatmap(results_df, plots_dir)

    print("\n  Generating combined accuracy+coverage heatmap …")
    plot_accuracy_and_coverage_side_by_side(results_df, plots_dir)

    print("\n  Generating sorted bar charts …")
    plot_sorted_bar(results_df, plots_dir, "macro_accuracy", "Macro Accuracy")
    plot_sorted_bar(results_df, plots_dir, "macro_precision", "Macro Precision")

    print("\n  Generating 4x2 confusion matrix grid …")
    plot_confusion_matrix_grid(results_df, plots_dir)


# ---------------------------------------------------------------------------
# Standalone main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    print("=" * 60)
    print("FILTER SWEEP — PLOT GENERATION")
    print("=" * 60)
    print(f"  Results: {args.results_path}")
    print(f"  Plots:   {args.plots_dir}")

    if not os.path.exists(args.results_path):
        print(f"\nError: Results file not found at {args.results_path}")
        print("Run sweep_filters.py first.")
        sys.exit(1)

    results_df = pd.read_csv(args.results_path)

    # Parse stringified dicts/lists if needed
    for col in ["confusion_matrix"]:
        if col in results_df.columns and isinstance(results_df[col].iloc[0], str):
            results_df[col] = results_df[col].apply(ast.literal_eval)

    print(f"  Loaded {len(results_df)} configurations")

    plot_sweep_results(results_df, args.plots_dir)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
