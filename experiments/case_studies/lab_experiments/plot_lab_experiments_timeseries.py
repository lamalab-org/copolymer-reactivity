#!/usr/bin/env python3
"""
Plot lab-experiment time series for the AN/VP system.

Creates a single figure with three subplots:
  1) time vs conversion (two lines: AN and VP)
  2) time vs average conversion with std bands (two lines: VA and BA)
  3) time vs average conversion with std bands (two lines: 1-octene and styrene)

Plot styling/colors are taken from `copol_prediction/analysis/plot_config.py`.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------------

# PROJECT_ROOT: go up 4 levels from this file:
# experiments/case_studies/lab_experiments/plot_lab_experiments_timeseries.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, str(Path(PROJECT_ROOT)))

from copol_prediction.analysis import plot_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    data_dir = Path(__file__).with_name("kinetics_data")
    default_csv_an_vp = data_dir / "an_vp.csv"
    default_csv_ba_va = data_dir / "ba_va.csv"
    default_csv_sty_oct = data_dir / "styrene_octene.csv"
    default_out = Path(__file__).with_name("kinetics_copolymerizations")

    p = argparse.ArgumentParser()
    p.add_argument("--csv-an-vp", type=Path, default=default_csv_an_vp, help="CSV for AN/VP time series.")
    p.add_argument("--csv-ba-va", type=Path, default=default_csv_ba_va, help="CSV for BA/VA time series.")
    p.add_argument(
        "--csv-sty-oct",
        type=Path,
        default=default_csv_sty_oct,
        help="CSV for styrene/1-octene time series (same column schema as BA/VA).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=default_out,
        help="Output path (suffix optional). Will write both .png and .pdf.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    plot_config.setup_plot_style()

    df_an_vp = pd.read_csv(args.csv_an_vp)
    req_an_vp = {"time_min", "conversion_an", "conversion_vp"}
    missing_an_vp = sorted(req_an_vp - set(df_an_vp.columns))
    if missing_an_vp:
        raise SystemExit(
            f"Missing required columns in {args.csv_an_vp}: {missing_an_vp}. "
            "Expected columns: time_min, conversion_an, conversion_vp"
        )

    df_ba_va = pd.read_csv(args.csv_ba_va)
    req_ba_va = {
        "time_min",
        "avg_conversion_va",
        "std_conversion_va",
        "avg_conversion_ba",
        "std_conversion_ba",
    }
    missing_ba_va = sorted(req_ba_va - set(df_ba_va.columns))
    if missing_ba_va:
        raise SystemExit(
            f"Missing required columns in {args.csv_ba_va}: {missing_ba_va}. "
            "Expected columns: time_min, avg_conversion_va, std_conversion_va, avg_conversion_ba, std_conversion_ba"
        )

    df_sty_oct = pd.read_csv(args.csv_sty_oct)
    missing_sty_oct = sorted(req_ba_va - set(df_sty_oct.columns))
    if missing_sty_oct:
        raise SystemExit(
            f"Missing required columns in {args.csv_sty_oct}: {missing_sty_oct}. "
            "Expected columns: time_min, avg_conversion_va, std_conversion_va, avg_conversion_ba, std_conversion_ba"
        )

    # Use the same two colors across all lab subplots
    color_1 = plot_config.LAB_SERIES_COLORS["series_1"]
    color_2 = plot_config.LAB_SERIES_COLORS["series_2"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        sharey=False,
        figsize=(plot_config.TWO_COL_WIDTH_INCH, 0.75 * plot_config.TWO_COL_GOLDEN_RATIO_HEIGHT_INCH),
        constrained_layout=True,
    )

    def panel_label(ax: plt.Axes, label: str) -> None:
        ax.text(
            0.0,
            1.06,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    def legend_below(ax: plt.Axes, ncol: int = 2) -> None:
        ax.legend(
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.25),
            ncol=ncol,
            borderaxespad=0.0,
        )

    # 1) time vs conversion (requested)
    ax0 = axes[0]
    panel_label(ax0, "A")
    t0 = df_an_vp["time_min"]
    conv_an = df_an_vp["conversion_an"]
    conv_vp = df_an_vp["conversion_vp"]
    ax0.plot(t0, conv_an, label="Acrylonitrile", color=color_1, linewidth=2)
    ax0.plot(t0, conv_vp, label="N-vinyl-2-pyrrolidone", color=color_2, linewidth=2)
    ax0.set_xlabel("Time (min)")
    ax0.set_ylabel("Conversion")
    legend_below(ax0, ncol=1)

    # 2) time vs average conversion with std bands (VA/BA)
    ax1 = axes[1]
    panel_label(ax1, "B")
    t1 = df_ba_va["time_min"]
    mean_va = df_ba_va["avg_conversion_va"]
    std_va = df_ba_va["std_conversion_va"]
    mean_ba = df_ba_va["avg_conversion_ba"]
    std_ba = df_ba_va["std_conversion_ba"]

    ax1.plot(t1, mean_va, label="Vinyl acetate", color=color_1, linewidth=2)
    ax1.fill_between(t1, mean_va - std_va, mean_va + std_va, color=color_1, alpha=0.2, linewidth=0)

    ax1.plot(t1, mean_ba, label="Butyl acrylate", color=color_2, linewidth=2)
    ax1.fill_between(t1, mean_ba - std_ba, mean_ba + std_ba, color=color_2, alpha=0.2, linewidth=0)

    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel("Conversion")
    legend_below(ax1, ncol=1)

    # 3) time vs average conversion with std bands (1-octene/styrene)
    ax2 = axes[2]
    panel_label(ax2, "C")
    t2 = df_sty_oct["time_min"]
    mean_oct = df_sty_oct["avg_conversion_va"]
    std_oct = df_sty_oct["std_conversion_va"]
    mean_sty = df_sty_oct["avg_conversion_ba"]
    std_sty = df_sty_oct["std_conversion_ba"]

    ax2.plot(t2, mean_oct, label="1-octene", color=color_1, linewidth=2)
    ax2.fill_between(t2, mean_oct - std_oct, mean_oct + std_oct, color=color_1, alpha=0.2, linewidth=0)

    ax2.plot(t2, mean_sty, label="Styrene", color=color_2, linewidth=2)
    ax2.fill_between(t2, mean_sty - std_sty, mean_sty + std_sty, color=color_2, alpha=0.2, linewidth=0)

    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel("Conversion")
    legend_below(ax2, ncol=1)

    out_base = args.out
    if out_base.suffix.lower() in {".png", ".pdf"}:
        out_base = out_base.with_suffix("")

    out_png = out_base.with_suffix(".png")
    out_pdf = out_base.with_suffix(".pdf")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    main()

