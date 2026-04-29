#!/usr/bin/env python3
"""
Plot 1H and 13C NMR spectra exported as SPECMAN_ASCII(ACD) text files.

Supported input formats:
- Classic SPECMAN_ASCII: two columns per data line: <ppm> <intensity> (header ignored)
- Bruker-style ascii export `ascii-spec.txt`: columns are (idx, intensity, ..., ppm)

Outputs a single figure with two subplots side-by-side (1H left, 13C right),
with only the x-axis labeled ("ppm") and the y-axis hidden.
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
from matplotlib import transforms


# ---------------------------------------------------------------------------
# Paths & imports
# ---------------------------------------------------------------------------

# PROJECT_ROOT: go up 4 levels from this file:
# experiments/case_studies/lab_experiments/plot_nmr_spectra.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, str(Path(PROJECT_ROOT)))

from copol_prediction.analysis import plot_config  # noqa: E402


H1_INTEGRALS: dict[str, list[tuple[float, float, float]]] = {
    # (ppm_start, ppm_end, integral_value)
    "MWH-018": [
        (4.73, 4.93, 1.000),
        (3.81, 4.17, 14.229),
    ],
    "MWH-022": [
        (0.76, 2.36, 3.515),
        (6.30, 6.689, 2.000),
        (6.90, 7.26, 2.969),
    ],
    "MWH-017": [
        (2.10, 1.55, 6.646),
        (4.13, 4.39, 0.783),
    ],
}


def _infer_sample_tag(*paths: Path) -> str:
    """
    Infer a short sample tag from filenames (e.g. "MWH-017").
    Falls back to "sample" if nothing useful is found.
    """
    joined = " ".join(p.name for p in paths if p is not None)
    m = re.search(r"\bmwh[-_ ]?(\d{1,4})", joined, flags=re.IGNORECASE)
    if m:
        try:
            num = int(m.group(1))
            return f"MWH-{num:03d}"
        except ValueError:
            pass
    return "sample"


def _iter_xy_from_speclines(lines: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        # Fast reject typical header lines like: "Key = Value"
        if "=" in line and ("\t" not in line and " " not in line):
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


def _looks_like_ascii_spec(first_lines: list[str]) -> bool:
    """
    Heuristic for Bruker-style ascii export:
    - First line is a sample name
    - Data lines look like: "1, 30196736, 8450.7014, 16.895321"
      (index, intensity, something, ppm)
    """
    nonempty = [ln.strip() for ln in first_lines if ln and ln.strip()]
    if len(nonempty) < 2:
        return False
    # If the second non-empty line has >=3 commas, it's a strong signal.
    return nonempty[1].count(",") >= 3


def _iter_xy_from_ascii_spec(lines: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse ascii-spec.txt where:
    - column 2: intensity
    - column 4: shift in ppm
    Columns are comma-separated, sometimes with spaces.
    """
    xs: list[float] = []
    ys: list[float] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        # First line is typically just the sample name (non-numeric)
        if "," not in line:
            continue

        parts = [p for p in re.split(r"[,\s]+", line) if p]
        # Expect: idx, intensity, something, ppm
        if len(parts) < 4:
            continue
        try:
            y = float(parts[1].replace(",", "."))
            x = float(parts[3].replace(",", "."))
        except ValueError:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        xs.append(x)
        ys.append(y)

    if not xs:
        raise ValueError("No numeric x/y pairs found in ascii-spec file.")

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    order = np.argsort(x_arr)
    return x_arr[order], y_arr[order]


def read_integrals_table(path: Path) -> list[tuple[float, float, float]]:
    """
    Parse Bruker 'integrals.txt' like:
      Number   Integrated Region     Integral
        1      4.436      4.037         1.00000
    Returns list of (ppm_start, ppm_end, integral_value).
    """
    integrals: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = line.replace(",", ".").split()
            if len(parts) < 4:
                continue
            try:
                # parts[0] is an integer index; ignore
                x1 = float(parts[1])
                x2 = float(parts[2])
                val = float(parts[3])
            except ValueError:
                continue
            if not (np.isfinite(x1) and np.isfinite(x2) and np.isfinite(val)):
                continue
            integrals.append((x1, x2, val))
    return integrals


def _override_selected_integrals(
    selected: list[tuple[float, float, float]],
    measured: list[tuple[float, float, float]],
    *,
    tol_ppm: float = 0.18,
) -> list[tuple[float, float, float]]:
    """
    Keep the selected ppm windows, but replace their values from the measured list.
    Matching is done by comparing window endpoints (order-insensitive).
    """
    if not selected or not measured:
        return selected

    out: list[tuple[float, float, float]] = []
    for a, b, old_val in selected:
        lo_s, hi_s = (a, b) if a <= b else (b, a)
        best = None
        best_score = float("inf")
        for x1, x2, val in measured:
            lo_m, hi_m = (x1, x2) if x1 <= x2 else (x2, x1)
            score = abs(lo_s - lo_m) + abs(hi_s - hi_m)
            if score < best_score:
                best_score = score
                best = val
        if best is not None and best_score <= 2.0 * tol_ppm:
            out.append((a, b, float(best)))
        else:
            out.append((a, b, old_val))
    return out


def read_nmr_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        # Peek a few lines to decide which parser to use
        peek = []
        for _ in range(6):
            ln = f.readline()
            if not ln:
                break
            peek.append(ln)
        f.seek(0)

        if path.name.lower() == "ascii-spec.txt" or _looks_like_ascii_spec(peek):
            return _iter_xy_from_ascii_spec(f)
        return _iter_xy_from_speclines(f)


def _mask_xrange(x: np.ndarray, xmin: float, xmax: float) -> np.ndarray:
    lo, hi = (xmin, xmax) if xmin <= xmax else (xmax, xmin)
    return (x >= lo) & (x <= hi)


def _set_ylim_with_padding(ax: plt.Axes, y: np.ndarray, pad_frac: float = 0.06) -> None:
    y = y[np.isfinite(y)]
    if y.size < 2:
        return
    y_low = float(np.min(y))
    y_high = float(np.max(y))
    if not (np.isfinite(y_low) and np.isfinite(y_high)) or y_low == y_high:
        return
    pad = pad_frac * (y_high - y_low)
    ax.set_ylim(y_low - pad, y_high + pad)

def _draw_integral_brackets(
    ax: plt.Axes,
    integrals: list[tuple[float, float, float]],
    *,
    color: str = "#b20404",
) -> None:
    """
    Draw NMR-style integral brackets below the x-axis.
    Coordinates: x in data (ppm), y in axes fraction (can be negative).
    """
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    # Draw just above the x-axis, small height.
    # "Open upward": bottom horizontal with two legs going up (like a shallow ∪).
    y_bottom = 0.090
    y_top = 0.125
    # Number should sit below the bracket but still above the axis line.
    y_text = 0.008

    for x1, x2, val in integrals:
        lo = min(x1, x2)
        hi = max(x1, x2)
        # Bracket open upward
        ax.plot(
            [lo, lo, hi, hi],
            [y_top, y_bottom, y_bottom, y_top],
            transform=trans,
            clip_on=False,
            color=color,
            linewidth=1.0,
        )
        ax.text(
            (lo + hi) / 2.0,
            y_text,
            f"{val:.2f}",
            transform=trans,
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=color,
            clip_on=False,
        )


def _style_nmr_axis(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, pad=8)
    ax.set_xlabel("ppm", labelpad=6)

    # Hide y-axis (no ticks, no label) but keep trace visible.
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)

    # Clean spines: typical NMR plots show only bottom axis.
    for spine in ("left", "right", "top"):
        ax.spines[spine].set_visible(False)

    # Add a bit of breathing room between axes and trace.
    ax.margins(x=0.02, y=0.10)
    ax.tick_params(axis="x", pad=4)


def _detect_nucleus_from_filename(path: Path) -> str | None:
    name = path.name.lower()
    # Don't treat integral tables as spectra
    if "integral" in name:
        return None
    # New naming convention: <sample>_1H_ascii-spec.txt / <sample>_13C_ascii-spec.txt
    if "_13c_" in name:
        return "13C"
    if "_1h_" in name:
        return "1H"
    if "13c" in name:
        return "13C"
    if re.search(r"(^|[^0-9])1h([^a-z0-9]|$)", name):
        return "1H"
    return None


def _detect_nucleus_from_header(path: Path, max_lines: int = 80) -> str | None:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for _ in range(max_lines):
                line = f.readline()
                if not line:
                    break
                if line.startswith("Nucleus"):
                    # e.g. "Nucleus = 1H"
                    parts = line.split("=")
                    if len(parts) >= 2:
                        nuc = parts[1].strip()
                        if nuc in {"1H", "13C"}:
                            return nuc
    except OSError:
        return None
    return None


def _detect_nucleus(path: Path) -> str | None:
    return _detect_nucleus_from_filename(path) or _detect_nucleus_from_header(path)


def _collect_nmr_pairs(data_dir: Path) -> dict[str, dict[str, Path]]:
    """
    Returns mapping: sample_tag -> {"1H": path, "13C": path}
    """
    pairs: dict[str, dict[str, Path]] = {}
    # Recursive scan: new lab export folders often have nested structure.
    for p in sorted(data_dir.rglob("*.txt")):
        name_l = p.name.lower()
        if "integral" in name_l:
            continue
        # Prefer the new export naming to avoid accidentally picking unrelated txt files.
        if "ascii-spec" not in name_l:
            continue
        tag = _infer_sample_tag(p)
        nucleus = _detect_nucleus(p)
        if tag == "sample" or nucleus is None:
            continue
        pairs.setdefault(tag, {})
        # If multiple candidates exist, prefer the strict pattern first.
        existing = pairs[tag].get(nucleus)
        if existing is None:
            pairs[tag][nucleus] = p
            continue
        # Replace a less specific match with the stricter one.
        strict_pat = f"_{nucleus.lower()}_ascii-spec.txt"
        if strict_pat in p.name.lower() and strict_pat not in existing.name.lower():
            pairs[tag][nucleus] = p
    return pairs


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    default_data_dir = Path(__file__).with_name("Experimental_data")
    default_out_dir = Path(__file__).with_name("output") / "nmr_spectra"

    p.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Directory containing NMR txt files (SPECMAN_ASCII). Default: lab_experiments/Experimental_data",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=default_out_dir,
        help="Directory to write outputs into. Default: lab_experiments/output/nmr_spectra",
    )
    p.add_argument("--h1", type=Path, default=None, help="Optional single-run: path to 1H NMR txt.")
    p.add_argument("--c13", type=Path, default=None, help="Optional single-run: path to 13C NMR txt.")
    p.add_argument(
        "--h1-integrals",
        type=Path,
        default=None,
        help=(
            "Optional single-run: path to 1H integrals table (e.g. integrals.txt). "
            "If omitted, will look for 'integrals.txt' next to --h1."
        ),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Optional single-run: output path base (suffix optional). "
            "If omitted, auto-named as nmr_spectra_<MWH-###> inside --out-dir."
        ),
    )
    p.add_argument(
        "--invert-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invert x-axis so ppm decreases left-to-right (typical NMR convention).",
    )
    p.add_argument(
        "--h1-range",
        type=float,
        nargs=2,
        default=(5.0, 0.0),
        metavar=("XMAX", "XMIN"),
        help="Displayed 1H ppm range as 'XMAX XMIN'. Default: 5 0",
    )
    p.add_argument(
        "--c13-range",
        type=float,
        nargs=2,
        default=(200.0, 0.0),
        metavar=("XMAX", "XMIN"),
        help="Displayed 13C ppm range as 'XMAX XMIN'. Default: 200 0",
    )
    p.add_argument(
        "--c13-y-clip-upper-quantile",
        type=float,
        default=0.999,
        help=(
            "Clip 13C y-axis upper limit to this quantile (0-1) "
            "to intentionally cut the tallest peak and expand smaller peaks. Default: 0.999"
        ),
    )
    p.add_argument(
        "--c13-y-clip-lower-quantile",
        type=float,
        default=0.01,
        help="Set 13C y-axis lower limit to this quantile (0-1). Default: 0.01",
    )
    return p.parse_args()


def _plot_pair(
    *,
    tag: str,
    h1_path: Path,
    c13_path: Path,
    out_base: Path,
    invert_x: bool,
    h1_range: tuple[float, float],
    c13_range: tuple[float, float],
    c13_y_clip_lower_quantile: float,
    c13_y_clip_upper_quantile: float,
    h1_integrals_path: Path | None = None,
) -> None:
    x_h, y_h = read_nmr_xy(h1_path)
    x_c, y_c = read_nmr_xy(c13_path)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        sharey=False,
        figsize=(plot_config.TWO_COL_WIDTH_INCH, 0.55 * plot_config.TWO_COL_GOLDEN_RATIO_HEIGHT_INCH),
        constrained_layout=False,
    )

    ax_h, ax_c = axes
    color = plot_config.NEUTRAL_COLORS.get("text", "#000000")

    h1_xmin, h1_xmax = sorted(h1_range)
    c13_xmin, c13_xmax = sorted(c13_range)

    ax_h.plot(x_h, y_h, color=color, linewidth=0.6, alpha=1.0)
    _style_nmr_axis(ax_h, r"$^1$H NMR")
    ax_h.set_xlim(h1_xmin, h1_xmax)
    if invert_x:
        ax_h.invert_xaxis()
    has_integrals = tag in H1_INTEGRALS
    y_h_win = y_h[_mask_xrange(x_h, h1_xmin, h1_xmax)]
    if has_integrals:
        # Increase distance from x-axis by expanding the lower y-limit more strongly.
        y_fin = y_h_win[np.isfinite(y_h_win)]
        if y_fin.size >= 2:
            y_low = float(np.min(y_fin))
            y_high = float(np.max(y_fin))
            if np.isfinite(y_low) and np.isfinite(y_high) and y_low != y_high:
                yr = y_high - y_low
                ax_h.set_ylim(y_low - 0.22 * yr, y_high + 0.06 * yr)
            else:
                _set_ylim_with_padding(ax_h, y_h_win, pad_frac=0.10)
        else:
            _set_ylim_with_padding(ax_h, y_h_win, pad_frac=0.10)
    else:
        _set_ylim_with_padding(ax_h, y_h_win, pad_frac=0.06)
    if has_integrals:
        selected = H1_INTEGRALS[tag]
        integ_path = h1_integrals_path
        if integ_path is None:
            cand = h1_path.parent / "integrals.txt"
            if cand.exists():
                integ_path = cand
            else:
                # Common alternative naming: <sample>_1H_integrals.txt (or similar)
                matches = sorted(h1_path.parent.glob("*integral*.txt"))
                if matches:
                    integ_path = matches[0]
        if integ_path is not None and integ_path.exists():
            measured = read_integrals_table(integ_path)
            selected = _override_selected_integrals(selected, measured)
        _draw_integral_brackets(ax_h, selected)

    ax_c.plot(x_c, y_c, color=color, linewidth=0.6, alpha=1.0)
    _style_nmr_axis(ax_c, r"$^{13}$C NMR")
    ax_c.set_xlim(c13_xmin, c13_xmax)
    if invert_x:
        ax_c.invert_xaxis()

    # Intentionally clip the tallest 13C peak to improve visibility of smaller peaks
    q_low = float(np.clip(c13_y_clip_lower_quantile, 0.0, 1.0))
    q_high = float(np.clip(c13_y_clip_upper_quantile, 0.0, 1.0))
    if q_low > q_high:
        q_low, q_high = q_high, q_low
    y_low = float(np.quantile(y_c, q_low))
    y_high = float(np.quantile(y_c, q_high))
    if np.isfinite(y_low) and np.isfinite(y_high) and y_low != y_high:
        pad = 0.06 * (y_high - y_low)
        ax_c.set_ylim(y_low - pad, y_high + pad)

    # Extra bottom space for 1H integral annotations
    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.20, wspace=0.20)

    out_base = out_base.with_suffix("")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    plot_config.setup_plot_style()

    # Single-run mode (explicit paths)
    if args.h1 is not None or args.c13 is not None:
        if args.h1 is None or args.c13 is None:
            raise SystemExit("For single-run mode, provide both --h1 and --c13.")
        tag = _infer_sample_tag(args.h1, args.c13)
        out_base = args.out or (args.out_dir / f"nmr_spectra_{tag}")
        _plot_pair(
            tag=tag,
            h1_path=args.h1,
            c13_path=args.c13,
            out_base=out_base,
            invert_x=args.invert_x,
            h1_range=tuple(args.h1_range),
            c13_range=tuple(args.c13_range),
            c13_y_clip_lower_quantile=args.c13_y_clip_lower_quantile,
            c13_y_clip_upper_quantile=args.c13_y_clip_upper_quantile,
            h1_integrals_path=args.h1_integrals,
        )
        return

    # Batch mode: scan data directory for all MWH-### pairs
    pairs = _collect_nmr_pairs(args.data_dir)
    if not pairs:
        raise SystemExit(f"No NMR pairs found in {args.data_dir}. Expected *.txt with MWH-### and 1H/13C.")

    for tag, nuc_map in sorted(pairs.items()):
        if "1H" not in nuc_map or "13C" not in nuc_map:
            continue
        out_base = args.out_dir / f"nmr_spectra_{tag}"

        # Sample-specific 1H windows (requested)
        h1_range = tuple(args.h1_range)
        if tag == "MWH-018":
            h1_range = (7.5, 0.0)
        elif tag == "MWH-022":
            h1_range = (7.5, 0.0)

        _plot_pair(
            tag=tag,
            h1_path=nuc_map["1H"],
            c13_path=nuc_map["13C"],
            out_base=out_base,
            invert_x=args.invert_x,
            h1_range=h1_range,
            c13_range=tuple(args.c13_range),
            c13_y_clip_lower_quantile=args.c13_y_clip_lower_quantile,
            c13_y_clip_upper_quantile=args.c13_y_clip_upper_quantile,
        )


if __name__ == "__main__":
    main()

