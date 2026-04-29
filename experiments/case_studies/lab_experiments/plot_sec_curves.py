#!/usr/bin/env python3
"""
Plot SEC curves from TXT files.

Expected data format: two numeric columns per line:
  time  intensity

Batch mode:
  - scans `Experimental_data/` for SEC *.txt files
  - expects exactly 3 files total (e.g. MWH-017, MWH-018, MWH-022)
  - plots the three curves side-by-side in one figure
  - saves PNG + PDF into `output/sec_curves/`

Only the x-axis is shown/labeled (Elution time), y-axis is hidden.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


# ---------------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------------

# PROJECT_ROOT: go up 4 levels from this file:
# experiments/case_studies/lab_experiments/plot_sec_curves.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, str(Path(PROJECT_ROOT)))

from copol_prediction.analysis import plot_config  # noqa: E402


SAMPLE_DISPLAY_NAMES: dict[str, str] = {
    "MWH-017": r"Poly(acrylonitrile-co-$\mathit{N}$-vinyl-2-pyrrolidone)",
    # user sometimes refers to 019; keep both to be safe
    "MWH-018": "Poly(butyl acrylate-co-vinyl acetate)",
    "MWH-019": "Poly(butyl acrylate-co-vinyl acetate)",
    "MWH-022": "Poly(styrene-co-1-octene)",
}


def _infer_mwh_tag(text: str) -> str | None:
    """
    Extract MWH tag from a string.
    Accepts: MWH-017, MWH_017, MWH017, etc. Returns normalized 'MWH-017'.
    """
    # Don't require a word-boundary after the digits because filenames often continue with "_..."
    m = re.search(r"\bMWH[-_ ]?(\d{1,4})", text, flags=re.IGNORECASE)
    if not m:
        return None
    num = int(m.group(1))
    return f"MWH-{num:03d}"


def _iter_xy_numeric(lines: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        parts = line.replace(",", ".").split()
        if len(parts) < 2:
            continue
        try:
            x = float(parts[0])
            y = float(parts[1])
        except ValueError:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError("No numeric x/y pairs found in file.")
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    order = np.argsort(x_arr)
    return x_arr[order], y_arr[order]


def read_sec_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return _iter_xy_numeric(f)


def _style_sec_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel("Elution time", labelpad=8)

    ax.set_ylabel("RI signal", labelpad=8)
    # Keep 0..1 normalization, but add a little whitespace around the trace.
    ax.set_ylim(-0.03, 1.03)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.tick_params(axis="y", which="both", left=True, right=False, labelleft=True, pad=4)

    # Show axis lines (spines) for x and y, hide top/right.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    ax.margins(x=0.02, y=0.10)
    ax.tick_params(axis="x", pad=4)


def _set_ylim_with_padding(ax: plt.Axes, y: np.ndarray, pad_frac: float = 0.06) -> None:
    # Not used anymore (we normalize to 0..1)
    return


def _normalize_0_1(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 2:
        return y
    y_min = float(np.min(y[finite]))
    y_max = float(np.max(y[finite]))
    if not (np.isfinite(y_min) and np.isfinite(y_max)) or y_max == y_min:
        return y
    y_norm = (y - y_min) / (y_max - y_min)
    return y_norm


def _collect_sec_files(data_dir: Path) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for p in sorted(data_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".txt":
            continue
        if "sec" not in p.name.lower():
            continue
        tag = _infer_mwh_tag(p.name) or _infer_mwh_tag(str(p))
        if not tag:
            continue
        groups.setdefault(tag, []).append(p)
    return groups


def _sec_variant_label(path: Path) -> str:
    """
    Extract the variant label between the MWH tag and the SEC device token.
    Examples:
      MWH_017_c_SEC9.txt   -> "c"
      MWH_017_cd_SEC9.txt  -> "cd"
    Falls back to the filename stem if parsing fails.
    """
    stem = path.stem
    m = re.search(r"mwh[-_ ]?\d+[-_ ]*([^_ ]+?)\s*[-_ ]*sec\d+", stem, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    return stem


def _tag_number(tag: str) -> int | None:
    m = re.search(r"mwh-(\d+)", tag, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _inj_variant_label(path: Path) -> str | None:
    """
    Extract variant from injection info filename.
    Example: '... MWH_022_cd - 1.TXT' -> 'cd'
    """
    stem = path.stem
    m = re.search(r"mwh[-_ ]?\d+[-_ ]*([a-z]{1,5})\b", stem, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).lower()


def _parse_mn_d_from_inj_file(path: Path, max_lines: int = 250) -> tuple[float | None, float | None]:
    mn = None
    dispersity = None
    mn_re = re.compile(r"^\s*Mn:\s*([0-9.]+E[+\-]?\d+)", flags=re.IGNORECASE)
    d_re = re.compile(r"^\s*D:\s*([0-9.]+E[+\-]?\d+)", flags=re.IGNORECASE)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for _ in range(max_lines):
            line = f.readline()
            if not line:
                break
            if mn is None:
                m1 = mn_re.search(line)
                if m1:
                    try:
                        mn = float(m1.group(1))
                    except ValueError:
                        mn = None
            if dispersity is None:
                m2 = d_re.search(line)
                if m2:
                    try:
                        dispersity = float(m2.group(1))
                    except ValueError:
                        dispersity = None
            if mn is not None and dispersity is not None:
                break
    return mn, dispersity


def _collect_inj_info(data_dir: Path) -> dict[tuple[str, str | None], tuple[float | None, float | None]]:
    """
    Map (tag, variant) -> (Mn, D) from injection info files.
    """
    info: dict[tuple[str, str | None], tuple[float | None, float | None]] = {}
    for p in sorted(data_dir.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".txt":
            continue
        name_l = p.name.lower()
        if "inj" not in name_l:
            continue
        tag = _infer_mwh_tag(p.name) or _infer_mwh_tag(str(p))
        if not tag:
            continue
        variant = _inj_variant_label(p)
        info[(tag, variant)] = _parse_mn_d_from_inj_file(p)
    return info


def _format_sci_tex(value: float, sigfigs: int = 3) -> str:
    """
    Format number as LaTeX-friendly scientific notation, e.g. 1.33×10^{4}.
    """
    if not np.isfinite(value) or value == 0:
        return f"{value:.{max(sigfigs-1,0)}g}"
    exp = int(np.floor(np.log10(abs(value))))
    mant = value / (10**exp)
    # Keep mantissa readable (sigfigs total)
    mant_str = f"{mant:.{max(sigfigs-1,0)}g}"
    return rf"{mant_str}\times10^{{{exp}}}"


def _parse_args() -> argparse.Namespace:
    default_data_dir = Path(__file__).with_name("Experimental_data")
    default_out_dir = Path(__file__).with_name("output") / "sec_curves"

    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing SEC *.TXT files. Default: lab_experiments/Experimental_data",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help="Directory to write outputs into. Default: lab_experiments/output/sec_curves",
    )
    p.add_argument("--file", type=Path, default=None, help="Optional single-run: path to one SEC *.TXT file.")
    p.add_argument(
        "--expect-total",
        type=int,
        default=3,
        help="Expected total number of SEC files in batch mode (default: 3).",
    )
    p.add_argument(
        "--out-base",
        type=str,
        default="sec_curves_overview",
        help="Base name (without extension) for batch-mode output files.",
    )
    return p.parse_args()


def _sort_sec_paths(paths: list[Path]) -> list[Path]:
    return sorted(
        paths,
        key=lambda p: (
            (_infer_mwh_tag(p.name) is None, _tag_number(_infer_mwh_tag(p.name) or "MWH-999") or 999),
            _sec_variant_label(p).lower(),
            p.name.lower(),
        ),
    )


def _plot_overview(
    paths: list[Path],
    out_dir: Path,
    out_base_name: str,
    expect_total: int,
    inj_info: dict[tuple[str, str | None], tuple[float | None, float | None]] | None = None,
) -> None:
    paths = _sort_sec_paths(paths)
    if len(paths) != expect_total:
        found = ", ".join(p.name for p in paths) if paths else "<none>"
        raise SystemExit(f"Expected exactly {expect_total} SEC files total, but found {len(paths)}: {found}")
    n = len(paths)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n,
        sharey=False,
        figsize=(plot_config.TWO_COL_WIDTH_INCH, 0.58 * plot_config.TWO_COL_GOLDEN_RATIO_HEIGHT_INCH),
        constrained_layout=False,
    )
    if n == 1:
        axes = [axes]  # type: ignore[list-item]

    color = plot_config.NEUTRAL_COLORS.get("text", "#000000")

    for idx, (ax, p) in enumerate(zip(axes, paths, strict=False), start=1):
        x, y = read_sec_xy(p)
        y = _normalize_0_1(y)
        ax.plot(x, y, color=color, linewidth=0.6, alpha=1.0)
        tag = _infer_mwh_tag(p.name) or "MWH"
        variant = _sec_variant_label(p)
        display = SAMPLE_DISPLAY_NAMES.get(tag, tag)
        # Force 2-line titles to avoid overlap in narrow panels.
        if "-co-" in display:
            display_wrapped = display.replace("-co-", "-co-\n", 1)
        else:
            display_wrapped = display.replace("Poly(", "Poly(\n", 1) if display.startswith("Poly(") else display
        # Title should be only the polymer name (no c/cd/etc.)
        panel = chr(ord("A") + (idx - 1)) if 1 <= idx <= 26 else str(idx)
        label = f"{panel}: {display_wrapped}"
        _style_sec_axis(ax, label)

        # Add a bit of whitespace between the curve and the axes by expanding x-limits.
        if np.isfinite(x).sum() >= 2:
            x_min = float(np.min(x[np.isfinite(x)]))
            x_max = float(np.max(x[np.isfinite(x)]))
            if np.isfinite(x_min) and np.isfinite(x_max) and x_max != x_min:
                x_pad = 0.02 * (x_max - x_min)
                ax.set_xlim(x_min - x_pad, x_max + x_pad)

        # For MWH-022: show x ticks as integers (no decimals)
        if tag == "MWH-022":
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        if inj_info is not None:
            key_exact = (tag, (variant.lower() if variant else None))
            key_tag_only = (tag, None)
            mn, disp = inj_info.get(key_exact, inj_info.get(key_tag_only, (None, None)))
            lines: list[str] = []
            if mn is not None and np.isfinite(mn):
                lines.append(rf"$M_n$ = ${_format_sci_tex(mn)}$ g/mol")
            if disp is not None and np.isfinite(disp):
                lines.append(rf"$Đ$ = {disp:.3f}")
            if lines:
                ax.text(
                    0.02,
                    -0.30,
                    "\n".join(lines),
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=9,
                    clip_on=False,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
                )

    # Make room for the boxes below the x-axis and long two-line titles.
    fig.subplots_adjust(left=0.06, right=0.98, top=0.87, bottom=0.21, wspace=0.28)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_base = out_dir / out_base_name
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    plot_config.setup_plot_style()

    if args.file is not None:
        # single-file quick check
        _plot_overview([args.file], args.out_dir, out_base_name="sec_curve_single", expect_total=1)
        return

    groups = _collect_sec_files(args.data_dir)
    if not groups:
        raise SystemExit(f"No SEC *.TXT files found in {args.data_dir}.")

    all_paths: list[Path] = []
    for _tag, paths in sorted(groups.items()):
        all_paths.extend(paths)

    # Build output name that includes the sample IDs
    tags = sorted({(_infer_mwh_tag(p.name) or "MWH") for p in all_paths}, key=lambda t: _tag_number(t) or 999)
    tag_suffix = "_".join(tags)
    out_name = f"{args.out_base}_{tag_suffix}" if tag_suffix else args.out_base

    inj_info = _collect_inj_info(args.data_dir)
    _plot_overview(all_paths, args.out_dir, out_base_name=out_name, expect_total=args.expect_total, inj_info=inj_info)


if __name__ == "__main__":
    main()

