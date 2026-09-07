#!/usr/bin/env python3
"""
Generate a LaTeX table of GC conditions and retention times for the lab polymers.

Columns:
  Polymer | GC temp. program | Monomer 1 | Monomer 2 | Ret. time M1 | Ret. time M2 | Ret. time Trioxane

Outputs a .tex file (and optionally prints to stdout).
"""

from __future__ import annotations

import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

SAMPLE_ORDER: list[str] = ["MWH-017", "MWH-018", "MWH-027"]

SAMPLE_DATA: dict[str, dict] = {
    "MWH-017": {
        "polymer": r"\shortstack[l]{Poly(acrylonitrile-\textit{co}-\\\textit{N}-vinyl-2-pyrrolidone)}",
        "monomer1": "Acrylonitrile",
        "monomer2": r"\textit{N}-Vinyl-2-pyrrolidone",
        "temp_program": r"60--200\,\textdegree C",
        "rt_m1": 1.34,
        "rt_m2": 6.40,
        "rt_trioxane": 1.90,
    },
    "MWH-018": {
        "polymer": r"\shortstack[l]{Poly(butyl acrylate-\textit{co}-\\vinyl acetate)}",
        "monomer1": "Butyl acrylate",
        "monomer2": "Vinyl acetate",
        "temp_program": r"60--200\,\textdegree C",
        "rt_m1": 3.93,
        "rt_m2": 1.45,
        "rt_trioxane": 1.90,
    },
    "MWH-027": {
        "polymer": r"\shortstack[l]{Poly(acrylonitrile-\textit{co}-\\ethyl methacrylate)}",
        "monomer1": "Acrylonitrile",
        "monomer2": "Ethyl methacrylate",
        "temp_program": r"60--200\,\textdegree C",
        "rt_m1": 1.34,
        "rt_m2": 2.70,
        "rt_trioxane": 1.90,
    },
}

DEFAULT_OUT_DIR = Path(__file__).with_name("output")
DEFAULT_OUT_PATH = DEFAULT_OUT_DIR / "gc_retention_times_table.tex"


def _fmt_rt(value: float) -> str:
    return rf"${value:.2f}$"


def make_table() -> str:
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    # Columns: polymer | temp | M1 | M2 | rt_M1 | rt_M2 | rt_trioxane
    lines.append(r"\begin{tabular}{@{} l l l l ccc @{}}")
    lines.append(r"\toprule")
    lines.append(
        r"Polymer & "
        r"\shortstack[l]{Temp.\\program} & "
        r"Monomer 1 & "
        r"Monomer 2 & "
        r"\multicolumn{3}{c}{Retention time (min)} \\"
    )
    lines.append(r"\cmidrule(lr){5-7}")
    lines.append(r" &  &  &  & " r"Monomer 1 & Monomer 2 & Trioxane \\")
    lines.append(r"\midrule")

    for tag in SAMPLE_ORDER:
        d = SAMPLE_DATA[tag]
        row = [
            d["polymer"],
            d["temp_program"],
            d["monomer1"],
            d["monomer2"],
            _fmt_rt(d["rt_m1"]),
            _fmt_rt(d["rt_m2"]),
            _fmt_rt(d["rt_trioxane"]),
        ]
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{GC temperature program and retention times of the two monomers and "
        r"the internal standard trioxane for each copolymerization system.}"
    )
    lines.append(r"\label{tab:gc-retention-times}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help="Output .tex path. Default: lab_experiments/output/gc_retention_times_table.tex",
    )
    p.add_argument(
        "--stdout",
        action="store_true",
        help="Also print the LaTeX table to stdout.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    latex = make_table()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(latex, encoding="utf-8")
    if args.stdout:
        print(latex)


if __name__ == "__main__":
    main()
