#!/usr/bin/env python3
"""
Generate a LaTeX table skeleton for lab experiment case studies.

Rows: the three lab polymers (MWH-017, MWH-018, MWH-022).
Columns:
  - Monomer 1 name
  - Monomer 2 name
  - Ratio monomer1:monomer2 (subcolumns: from GC, from NMR)
  - Molar mass Mn
  - Degree of polymerization (subcolumns: M1 from GC, M1 from NMR, M2 from GC, M2 from NMR)

All cells are left empty except the monomer names (and the polymer tag in the first column).

This script prints LaTeX to stdout by default and can optionally write to a file.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

SAMPLE_ORDER: list[str] = ["MWH-017", "MWH-018", "MWH-027"]

# Names taken from the lab case study definitions (see lab_experiments_case_study.py)
# Monomer names hard-coded (incl. manual line breaks where needed).
# These values are LaTeX strings and are inserted as-is into the table.
SAMPLE_TO_MONOMERS_LATEX: dict[str, tuple[str, str]] = {
    # Use \shortstack (built-in LaTeX) to avoid requiring extra packages like `makecell`.
    "MWH-017": (r"acrylonitrile", r"\shortstack[l]{N-vinyl-2-\\pyrrolidone}"),
    "MWH-018": (r"butyl acrylate", r"vinyl acetate"),
    "MWH-027": (r"acrylonitrile", r"\shortstack[l]{ethyl\\methacrylate}"),
}

# Molar masses (g/mol) provided by you.
MONOMER_MOLAR_MASS_G_PER_MOL: dict[str, float] = {
    "styrene": 104.15,
    "octene": 112.24,
    "butyl acrylate": 128.17,
    "vinyl acetate": 86.09,
    "vinyl pyrrolidone": 111.14,  # N-vinyl-2-pyrrolidone
    "acrylonitrile": 53.06,
    "ethyl methacrylate": 114.14,
}

# Keys for the DP calculation: (monomer1_key, monomer2_key)
SAMPLE_TO_MONOMER_KEYS: dict[str, tuple[str, str]] = {
    "MWH-017": ("acrylonitrile", "vinyl pyrrolidone"),
    "MWH-018": ("butyl acrylate", "vinyl acetate"),
    "MWH-027": ("acrylonitrile", "ethyl methacrylate"),
}

# Values hard-coded from your updated LaTeX table.
# LaTeX strings are inserted as-is (use math mode where appropriate).
SAMPLE_TO_VALUES_LATEX: dict[str, dict[str, str]] = {
    "MWH-017": {
        "ratio_gc": r"$1:0.79$",
        "ratio_nmr": r"$1:0.74$",
        "mn": r"$1.8 \times 10^{5}$",
    },
    "MWH-018": {
        "ratio_gc": r"$1:0.14$",
        "ratio_nmr": r"$1:0.15$",
        "mn": r"$8.5 \times 10^{4}$",
    },
    "MWH-027": {
        "ratio_gc": r"$1:1.6$",
        "ratio_nmr": r"$1:1.7$",
        "mn": r"$6.1 \times 10^{4}$",
    },
}

DEFAULT_OUT_DIR = Path(__file__).with_name("output")
DEFAULT_OUT_PATH = DEFAULT_OUT_DIR / "lab_experiments_analysis_table.tex"


def _parse_ratio_m2_over_m1(ratio_latex: str) -> float | None:
    """
    Parse a ratio like "$1:0.14$" -> 0.14 (interpreted as M2/M1).
    """
    s = (ratio_latex or "").strip()
    if not s:
        return None
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = s.replace(" ", "")
    m = re.match(r"^1:(\d+(?:\.\d+)?)$", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _parse_mn_g_per_mol(mn_latex: str) -> float | None:
    """
    Parse Mn LaTeX like "$8.5 \\times 10^{4}$" -> 85000.
    """
    s = (mn_latex or "").strip()
    if not s:
        return None
    if s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = s.replace(" ", "")
    m = re.match(r"^(\d+(?:\.\d+)?)\\times10\^\{(-?\d+)\}$", s)
    if m:
        try:
            a = float(m.group(1))
            p = int(m.group(2))
            return float(a * (10**p))
        except Exception:
            return None
    try:
        return float(s)
    except Exception:
        return None


def _compute_dp(
    mn: float | None, mm1: float, mm2: float, r_m2_over_m1: float | None
) -> tuple[float | None, float | None]:
    """
    Solve:
      Mn = x*MM1 + (r*x)*MM2   with r = M2/M1
      DP_M1 = x, DP_M2 = r*x
    """
    if mn is None or r_m2_over_m1 is None:
        return (None, None)
    denom = mm1 + r_m2_over_m1 * mm2
    if denom <= 0:
        return (None, None)
    x = mn / denom
    return (x, r_m2_over_m1 * x)


def _format_dp(dp: float | None) -> str:
    if dp is None or not (dp > 0):
        return ""
    return rf"${dp:.1f}$"


def _latex_escape(text: str) -> str:
    # Minimal escaping for typical chemical/common names.
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def make_table() -> str:
    lines: list[str] = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    # Keep it simple and compact: smaller font + reduced padding.
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")
    # Trim outer whitespace with @{} ... @{}
    lines.append(r"\begin{tabular}{@{} l l l cc c cccc @{} }")
    lines.append(r"\toprule")
    lines.append(
        r"Polymer & Monomer 1 & Monomer 2 & \multicolumn{2}{c}{Ratio $M_1\!:\!M_2$} & $M_n$ & \multicolumn{4}{c}{Degree of polymerization} \\"
    )
    lines.append(
        r" &  &  & from GC & from NMR &  & $M_1$ (GC) & $M_1$ (NMR) & $M_2$ (GC) & $M_2$ (NMR) \\"
    )
    lines.append(r"\midrule")

    empty = ""
    for tag in SAMPLE_ORDER:
        m1, m2 = SAMPLE_TO_MONOMERS_LATEX.get(tag, ("", ""))
        vals = SAMPLE_TO_VALUES_LATEX.get(tag, {})
        ratio_gc = vals.get("ratio_gc", empty)
        ratio_nmr = vals.get("ratio_nmr", empty)
        mn = vals.get("mn", empty)

        # Compute DP values from ratio + Mn (separately for GC and NMR ratios).
        mk1, mk2 = SAMPLE_TO_MONOMER_KEYS.get(tag, ("", ""))
        mm1 = MONOMER_MOLAR_MASS_G_PER_MOL.get(mk1)
        mm2 = MONOMER_MOLAR_MASS_G_PER_MOL.get(mk2)
        mn_num = _parse_mn_g_per_mol(mn)
        r_gc = _parse_ratio_m2_over_m1(ratio_gc)
        r_nmr = _parse_ratio_m2_over_m1(ratio_nmr)
        dp1_gc = dp2_gc = dp1_nmr = dp2_nmr = None
        if mm1 is not None and mm2 is not None:
            dp1_gc, dp2_gc = _compute_dp(mn_num, mm1, mm2, r_gc)
            dp1_nmr, dp2_nmr = _compute_dp(mn_num, mm1, mm2, r_nmr)
        row = [
            _latex_escape(tag),
            m1,
            m2,
            ratio_gc,  # Ratio from GC
            ratio_nmr,  # Ratio from NMR
            mn,  # Mn
            _format_dp(dp1_gc),  # DP M1 from GC
            _format_dp(dp1_nmr),  # DP M1 from NMR
            _format_dp(dp2_gc),  # DP M2 from GC
            _format_dp(dp2_nmr),  # DP M2 from NMR
        ]
        lines.append(" & ".join(row) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\caption{Summary of lab experiment polymers and analysis results.}")
    lines.append(r"\label{tab:lab-experiments-summary}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=(
            "Output .tex path. Default: lab_experiments/output/lab_experiments_analysis_table.tex "
            "(relative to this script)."
        ),
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
