#!/usr/bin/env python3
"""Regenerate `paper_dataset/grouped_by_unique_monomer_pairs.csv`.

This is the canonical "one row per unique reaction" view of the dataset —
3,791 rows, 1,206 publications — that the paper cites and the public Zenodo
record archives. The frozen file is committed at
``copol_prediction/paper_dataset/grouped_by_unique_monomer_pairs.csv``;
this script is the recipe so anyone can rebuild it from any snapshot of
``processed_data.csv``.

The recipe (originally implemented as
``analyze_monomer_pairs()`` in the now-deleted
``copol_prediction/analysis/data_analysis.py``):

  1. Drop rows with NaN ``monomer1_smiles`` or ``monomer2_smiles``.
  2. Deduplicate by ``reaction_id`` — the canonical reaction key. Multiple
     measurement rows per reaction (different temperatures, methods, etc.)
     collapse to one.
  3. Add an order-insensitive ``monomer_pair_key`` and a contiguous
     integer ``group_id`` for the unique monomer pair the row represents.

Usage::

    # Default: rebuild from the live dataset and write next to it.
    python copol_prediction/build_grouped_dataset.py

    # Or supply an explicit source (e.g. the frozen paper snapshot).
    python copol_prediction/build_grouped_dataset.py \\
        --source copol_prediction/paper_dataset/processed_data.csv \\
        --output copol_prediction/paper_dataset/grouped_by_unique_monomer_pairs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "copol_prediction" / "processed_data.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT / "copol_prediction" / "paper_dataset" / "grouped_by_unique_monomer_pairs.csv"
)


def build(source: Path) -> pd.DataFrame:
    """Apply the dedup-and-tag recipe to a `processed_data.csv` snapshot."""
    df = pd.read_csv(source)
    df = df.dropna(subset=["monomer1_smiles", "monomer2_smiles"]).copy()
    df = df.drop_duplicates(subset="reaction_id")

    # Order-insensitive monomer-pair key (matches the historical artifact).
    df["monomer_pair_key"] = df.apply(
        lambda row: tuple(sorted([row["monomer1_smiles"], row["monomer2_smiles"]])),
        axis=1,
    )
    pair_to_id = {pair: idx for idx, pair in enumerate(df["monomer_pair_key"].unique())}
    df["group_id"] = df["monomer_pair_key"].map(pair_to_id)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Source processed_data.csv (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path for grouped CSV (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    print(f"Reading: {args.source}")
    grouped = build(args.source)

    n_papers = grouped["PDF_name"].nunique() if "PDF_name" in grouped.columns else None
    print(f"Grouped: {len(grouped):,} unique reactions", end="")
    if n_papers is not None:
        print(f" across {n_papers:,} publications")
    else:
        print()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(args.output, index=False)
    print(f"Wrote:   {args.output}")


if __name__ == "__main__":
    main()
