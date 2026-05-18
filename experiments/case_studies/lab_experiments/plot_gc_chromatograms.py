#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# PROJECT_ROOT: go up 4 levels from this file:
# experiments/case_studies/lab_experiments/plot_gc_chromatograms.py
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.insert(0, str(Path(PROJECT_ROOT)))

from copol_prediction.analysis import plot_config  # noqa: E402


@dataclass(frozen=True)
class Chromatogram:
    x_min: float
    x_max: float
    interval_msec: int | None
    n_points: int | None
    x: np.ndarray
    y: np.ndarray


def _to_float(s: str) -> float:
    return float(s.strip().replace(",", "."))


def _to_int(s: str) -> int:
    return int(float(s.strip().replace(",", ".")))


def read_chromatogram_ch1(path: Path, *, max_header_lines: int = 2000) -> Chromatogram:
    """
    Parse the [Chromatogram (Ch1)] block from LabSolutions ASCII export.

    Expected block header:
        [Chromatogram (Ch1)]
        Interval(msec)    40
        # of Points       18945
        Start Time(min)   0,000
        End Time(min)     12,630
        R.Time (min)      Intensity
        <x>               <y>
        ...
    """
    in_block = False
    interval_msec: int | None = None
    n_points: int | None = None
    x_min: float | None = None
    x_max: float | None = None

    xs: list[float] = []
    ys: list[float] = []
    expecting_xy = False

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for i in range(max_header_lines):
            line = f.readline()
            if not line:
                break
            line = line.rstrip("\n")

            if not in_block:
                if line.strip() == "[Chromatogram (Ch1)]":
                    in_block = True
                continue

            if (
                line.startswith("[")
                and line.endswith("]")
                and line.strip() != "[Chromatogram (Ch1)]"
            ):
                # Another section started without us seeing data; stop.
                break

            if not expecting_xy:
                if not line.strip():
                    continue
                if "\t" not in line:
                    continue

                key, val = line.split("\t", 1)
                key = key.strip()
                val = val.strip()

                if key == "Interval(msec)":
                    interval_msec = _to_int(val)
                elif key == "# of Points":
                    n_points = _to_int(val)
                elif key == "Start Time(min)":
                    x_min = _to_float(val)
                elif key == "End Time(min)":
                    x_max = _to_float(val)
                elif key.startswith("R.Time"):
                    expecting_xy = True
                continue

            # XY rows
            if not line.strip():
                break
            if line.startswith("[") and line.endswith("]"):
                break

            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                x = _to_float(parts[0])
                y = float(parts[1].strip().replace(",", "."))
            except ValueError:
                continue

            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)

        # If we reached max_header_lines without finishing, continue streaming until block ends.
        if in_block and expecting_xy and len(xs) == 0:
            # unlikely, but keep behavior sane
            pass
        if in_block and expecting_xy and (line := f.readline()):
            # Continue reading remaining file for the XY rows (fast path)
            while line:
                s = line.rstrip("\n").strip()
                if not s:
                    break
                if s.startswith("[") and s.endswith("]"):
                    break
                parts = s.split("\t")
                if len(parts) >= 2:
                    try:
                        x = _to_float(parts[0])
                        y = float(parts[1].strip().replace(",", "."))
                    except ValueError:
                        line = f.readline()
                        continue
                    if np.isfinite(x) and np.isfinite(y):
                        xs.append(x)
                        ys.append(y)
                line = f.readline()

    if not in_block:
        raise ValueError(f"No [Chromatogram (Ch1)] block found in {path}")
    if not xs:
        raise ValueError(f"No chromatogram points parsed from {path}")

    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)

    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]

    if x_min is None:
        x_min = float(np.min(x_arr))
    if x_max is None:
        x_max = float(np.max(x_arr))

    # Best-effort consistency check
    if n_points is not None and n_points != int(x_arr.size):
        # Some exports can include extra/truncated points; don't hard-fail.
        pass

    return Chromatogram(
        x_min=float(x_min),
        x_max=float(x_max),
        interval_msec=interval_msec,
        n_points=n_points,
        x=x_arr,
        y=y_arr,
    )


def _style_gc_axis(ax: plt.Axes) -> None:
    ax.set_xlabel("Retention time (min)", labelpad=6)
    ax.set_ylabel("Intensity", labelpad=6)

    # Similar cleanliness to the NMR plots: no top/right spines.
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", pad=4)

    # Hide y-axis ticks/labels but keep the y-axis label ("Intensity").
    ax.set_yticks([])
    ax.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax.margins(x=0.0, y=0.04)


def plot_one(
    *,
    chrom: Chromatogram,
    title: str,
    out_base: Path,
    y_lim: tuple[float, float] | None = None,
    x_lim: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=(
            plot_config.TWO_COL_WIDTH_INCH,
            1.15 * plot_config.ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
        ),
        constrained_layout=True,
    )

    color = plot_config.NEUTRAL_COLORS.get("text", "#000000")
    ax.plot(chrom.x, chrom.y, color=color, linewidth=0.6, alpha=1.0)
    if x_lim is not None:
        ax.set_xlim(*x_lim)
    else:
        ax.set_xlim(chrom.x_min, chrom.x_max)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    _style_gc_axis(ax)

    out_base = out_base.with_suffix("")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"))
    fig.savefig(out_base.with_suffix(".pdf"))
    plt.close(fig)


def _default_files() -> list[Path]:
    base = Path(__file__).with_name("Experimental_data") / "GC_data"
    return [
        base / "MWH_017_ASCII" / "MWH_017_gc_t2h.txt",
        base / "MWH_018_ASCII" / "MWH_018_gc_t240m_01.txt",
        base / "MWH_022_ASCII" / "MWH_022_gc_t480m_01.txt",
    ]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--files",
        type=Path,
        nargs="*",
        default=None,
        help="GC ASCII files to plot (defaults to the three requested example files).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).with_name("output") / "gc_chromatograms",
        help="Output directory for plots (png+pdf).",
    )
    p.add_argument(
        "--y-upper-quantile",
        type=float,
        default=0.995,
        help=(
            "Clip y-axis upper limit to this quantile (0-1) to hide the tallest peak "
            "and reveal smaller peaks. Default: 0.995"
        ),
    )
    p.add_argument(
        "--y-lower-quantile",
        type=float,
        default=0.01,
        help="Set y-axis lower limit to this quantile (0-1). Default: 0.01",
    )
    p.add_argument(
        "--x-end-min",
        type=float,
        default=10.0,
        help="Plot x-axis from 0 to this retention time (min). Default: 10",
    )
    args = p.parse_args()

    plot_config.setup_plot_style()

    files = args.files if args.files else _default_files()
    out_dir: Path = args.out_dir

    for fp in files:
        if not fp.exists():
            raise SystemExit(f"File not found: {fp}")
        chrom = read_chromatogram_ch1(fp)
        title = fp.stem
        out_base = out_dir / f"gc_{fp.stem}"
        # Clip y-limits to improve visibility of smaller peaks while keeping raw values.
        y_lim: tuple[float, float] | None = None
        y = chrom.y
        finite = np.isfinite(y)
        if np.any(finite):
            q_low = float(np.clip(args.y_lower_quantile, 0.0, 1.0))
            q_high = float(np.clip(args.y_upper_quantile, 0.0, 1.0))
            if q_low > q_high:
                q_low, q_high = q_high, q_low
            y_low = float(np.quantile(y[finite], q_low))
            y_high = float(np.quantile(y[finite], q_high))
            if np.isfinite(y_low) and np.isfinite(y_high) and y_low != y_high:
                pad = 0.06 * (y_high - y_low)
                y_lim = (y_low - pad, y_high + pad)

        plot_one(
            chrom=chrom,
            title=title,
            out_base=out_base,
            y_lim=y_lim,
            x_lim=(0.0, float(args.x_end_min)),
        )


if __name__ == "__main__":
    main()
