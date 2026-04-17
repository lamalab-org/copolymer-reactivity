#!/usr/bin/env python3
"""
Reaction conditions comparison: With vs Without reaction condition features.

Both variants use the **voting model** (XGBoost + Lookup must agree).
The only difference is which features the XGBoost component is trained on:
  1. All features  (including temperature, solvent props, embeddings)
  2. Without reaction conditions  (monomer descriptors only)

The Lookup model is identical in both cases (Tanimoto similarity, no
reaction-condition features).  It uses an *expanding* nearest-neighbor
strategy: it adds neighbors group-by-group (same similarity level) until
one class reaches a strict majority.  If no majority is reached after
all neighbors, it abstains (-1) and the XGBoost prediction is used.

Uses the same training configurations as the final model:
  - No specialized filter applied
  - No augmentation (Gaussian sampling disabled)
  - NO negative data (matching final model config)
- Evaluation on a reduced TEST set focused on condition-variation within the
  same monomer pair (see `select_condition_variation_subset` below).

Usage:
    python run_comparison.py [--n-iter 25] [--output-dir results]
    python run_comparison.py --plot-only
"""

import os
import sys
import json
import argparse
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix as sk_confusion_matrix,
)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))

sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, '..'))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'copol_prediction'))

from copolpredictor import (
    model_training,
    prediction_utils,
    data_augmentation,
)
from utils import load_data_split
from copol_prediction.analysis.analyze_model import (
    compute_fingerprints_for_smiles,
)

try:
    from copol_prediction.analysis.plot_config import (
        setup_plot_style,
        COMPARISON_COLORS,
        TWO_COL_WIDTH_INCH,
    )
except ImportError:
    def setup_plot_style():
        pass
    COMPARISON_COLORS = {'original': '#3A3B73', 'filtered': '#e27f07'}
    TWO_COL_WIDTH_INCH = 7

try:
    _STYLE_PATH = os.path.join(_PROJECT_ROOT, 'copol_prediction', 'analysis',
                                'lamalab.mplstyle')
    if os.path.exists(_STYLE_PATH):
        plt.style.use(_STYLE_PATH)
except Exception:
    pass

# Reaction condition features to exclude in the "no conditions" variant
REACTION_CONDITION_FEATURES = [
    'temperature',
    'polytype_emb_1', 'polytype_emb_2',
    'method_emb_1', 'method_emb_2',
    'solvent_logP', 'solvent_TPSA',
    'solvent_HBD', 'solvent_FractionCSP3',
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Reaction conditions comparison: with vs without"
    )
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--n-iter", type=int, default=25,
                        help="Hyperparameter search iterations per model")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training, re-plot from saved results JSON")
    parser.add_argument(
        "--use-augmentation",
        action="store_true",
        help=(
            "Enable Gaussian augmentation during XGBoost training. "
            "Default is OFF to match paper figures and avoid confounding."
        ),
    )
    parser.add_argument(
        "--eval-split",
        choices=["val", "test"],
        default="test",
        help="Which split to evaluate on (default: test).",
    )
    parser.add_argument(
        "--subset",
        choices=["all", "condition_variation"],
        default="condition_variation",
        help=(
            "Which subset to evaluate on. "
            "'condition_variation' keeps only monomer pairs that appear with "
            "multiple (r1,r2) AND multiple reaction conditions. (default)"
        ),
    )
    parser.add_argument(
        "--print-sample",
        type=int,
        default=100,
        help=(
            "Print N example evaluation rows after subsetting (default: 100). "
            "Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--print-pairs",
        type=int,
        default=10,
        help=(
            "Print up to N monomer-pair groups from the evaluation subset, "
            "showing multiple rows per pair to verify varying (r1,r2)/conditions "
            "(default: 10). Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-rows-per-pair",
        type=int,
        default=12,
        help="Max rows to print per monomer pair group (default: 12).",
    )
    parser.add_argument(
        "--print-pair-class-stats",
        action="store_true",
        help=(
            "Print per-monomer-pair class counts for the special subset "
            "(counts of r_product_class within each pair)."
        ),
    )
    parser.add_argument(
        "--pair-class-stats-max",
        type=int,
        default=300,
        help=(
            "Max number of pair rows to print for pair-class stats (default: 300). "
            "If exceeded, output is truncated; use --pair-class-stats-csv to save all."
        ),
    )
    parser.add_argument(
        "--pair-class-stats-csv",
        action="store_true",
        help="Save per-pair class count table to CSV in the output dir.",
    )
    parser.add_argument(
        "--no-top-pairs-plot",
        action="store_true",
        help="Disable the top-10 monomer-pair scatter plot for the special subset.",
    )
    parser.add_argument(
        "--print-no-temp-solvent-change",
        action="store_true",
        help=(
            "For the special subset: print rows where class changes occur without any "
            "change in temperature or solvent (same monomer pair, same temp+solvent, "
            "different r_product_class)."
        ),
    )
    parser.add_argument(
        "--no-temp-solvent-max-rows",
        type=int,
        default=120,
        help="Max rows to print for the no-temp/solvent-change analysis (default: 120).",
    )
    parser.add_argument(
        "--no-temp-solvent-csv",
        action="store_true",
        help="Save the no-temp/solvent-change rows to CSV in the output dir.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed for sampling printed rows (default: 42).",
    )
    parser.add_argument(
        "--sample-csv",
        action="store_true",
        help="Also save the printed sample rows to CSV in the output dir.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def filter_valid_smiles(df):
    """Remove rows where any SMILES column cannot be parsed by RDKit."""
    from rdkit import Chem
    smiles_cols = ['monomer1_smiles', 'monomer2_smiles', 'solvent_smiles']
    n_before = len(df)
    mask = pd.Series(True, index=df.index)
    for col in smiles_cols:
        if col in df.columns:
            valid = df[col].apply(
                lambda s: Chem.MolFromSmiles(str(s)) is not None
                if pd.notna(s) else False
            )
            mask &= valid
    df_out = df[mask].reset_index(drop=True)
    n_removed = n_before - len(df_out)
    if n_removed > 0:
        print(f"    Filtered {n_removed} rows with invalid SMILES "
              f"({len(df_out)} remaining)")
    return df_out


def load_data():
    """Load central train/val/test split + negative data.
    
    Uses the same configuration as the final model:
    - No specialized filter applied to training data
    - Negative data available but NOT used (final model: no negative data)
    - Evaluation on validation set (not test set)
    """
    copol_dir = os.path.join(_PROJECT_ROOT, 'copol_prediction')
    split_dir = os.path.join(copol_dir, 'artifacts', 'data_splits')

    df_train, df_val, df_test = load_data_split.load_train_val_test_split(split_dir=split_dir)
    load_data_split.print_split_info(split_dir=split_dir)

    neg_path = os.path.join(copol_dir, 'filter', 'artificial_datapoints',
                            'processed_combined_augmented.csv')
    df_neg = None
    if os.path.exists(neg_path):
        df_neg = pd.read_csv(neg_path)
        if 'Class' in df_neg.columns:
            df_neg = df_neg.rename(columns={'Class': 'r_product_class'})
        df_neg['r_product_class'] = df_neg['r_product_class'].astype(int)
        if 'reaction_id' not in df_neg.columns:
            df_neg['reaction_id'] = [f"neg_{i}" for i in range(len(df_neg))]
        print(f"  Negative data available: {len(df_neg)} samples (NOT used, matching final model config)")

    print("  Validating SMILES …")
    df_train = filter_valid_smiles(df_train)
    df_val = filter_valid_smiles(df_val)
    df_test = filter_valid_smiles(df_test)
    if df_neg is not None:
        df_neg = filter_valid_smiles(df_neg)

    return df_train, df_val, df_test, df_neg


def _canonical_monomer_pair_key(df: pd.DataFrame) -> pd.Series:
    """
    Canonical, order-invariant key for monomer pairs based on SMILES.
    """
    m1 = df["monomer1_smiles"].astype(str)
    m2 = df["monomer2_smiles"].astype(str)
    a = np.minimum(m1, m2)
    b = np.maximum(m1, m2)
    return a + "||" + b


def select_condition_variation_subset(df_eval: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where the *same monomer pair* appears with:
      - multiple distinct (r1,r2) values AND
      - multiple distinct reaction conditions AND
      - at least two different target classes across those rows

    Additionally excludes rows where class changes occur without any change
    in temperature or solvent (same pair + same temp+solvent).

    This targets exactly the scenario where reaction conditions should matter.
    """
    required = [
        "monomer1_smiles",
        "monomer2_smiles",
        "constant_1",
        "constant_2",
        "r_product_class",
    ]
    missing = [c for c in required if c not in df_eval.columns]
    if missing:
        raise ValueError(f"Missing required columns for subset selection: {missing}")

    df = df_eval.copy()
    df["_pair_key"] = _canonical_monomer_pair_key(df)

    # Reactivity ratio signature
    df["_r_sig"] = (
        df["constant_1"].round(6).astype(str) + "," + df["constant_2"].round(6).astype(str)
    )

    # Reaction-conditions signature (use columns if present; ignore if missing)
    cond_cols = [
        "temperature",
        "solvent_smiles",
        "polymerization_type",
        "method",
    ]
    present = [c for c in cond_cols if c in df.columns]
    if present:
        df["_cond_sig"] = df[present].astype(str).agg("|".join, axis=1)
    else:
        # If we cannot define conditions, the subset is empty by definition
        return df.iloc[0:0].drop(columns=["_pair_key", "_r_sig"], errors="ignore")

    g = df.groupby("_pair_key", dropna=False)
    n_r = g["_r_sig"].nunique()
    n_c = g["_cond_sig"].nunique()
    n_y = g["r_product_class"].nunique()
    keep_keys = n_r[(n_r >= 2) & (n_c >= 2) & (n_y >= 2)].index

    out = df[df["_pair_key"].isin(keep_keys)].reset_index(drop=True)
    out = out.drop(columns=["_pair_key", "_r_sig", "_cond_sig"], errors="ignore")
    # Exclude "no temp/solvent change" cases
    exclude_mask = mask_class_change_no_temp_solvent_change(out)
    return out[~exclude_mask].reset_index(drop=True)


def condition_variation_mask(df_eval: pd.DataFrame) -> pd.Series:
    """
    Boolean mask for rows belonging to monomer pairs that have:
      - multiple distinct (r1,r2) AND
      - multiple distinct reaction conditions
      - at least two different target classes
    """
    required = [
        "monomer1_smiles",
        "monomer2_smiles",
        "constant_1",
        "constant_2",
        "r_product_class",
    ]
    missing = [c for c in required if c not in df_eval.columns]
    if missing:
        raise ValueError(f"Missing required columns for subset selection: {missing}")

    df = df_eval.copy()
    df["_pair_key"] = _canonical_monomer_pair_key(df)
    df["_r_sig"] = (
        df["constant_1"].round(6).astype(str) + "," + df["constant_2"].round(6).astype(str)
    )

    cond_cols = [
        "temperature",
        "solvent_smiles",
        "polymerization_type",
        "method",
    ]
    present = [c for c in cond_cols if c in df.columns]
    if present:
        df["_cond_sig"] = df[present].astype(str).agg("|".join, axis=1)
    else:
        return pd.Series(False, index=df_eval.index)

    g = df.groupby("_pair_key", dropna=False)
    n_r = g["_r_sig"].nunique()
    n_c = g["_cond_sig"].nunique()
    n_y = g["r_product_class"].nunique()
    keep_keys = set(n_r[(n_r >= 2) & (n_c >= 2) & (n_y >= 2)].index)
    base_mask = df["_pair_key"].isin(keep_keys)
    # Exclude rows where class changes happen without temp/solvent change
    exclude = mask_class_change_no_temp_solvent_change(df_eval)
    return base_mask & (~exclude)


def _print_condition_variation_pairs(
    df_eval: pd.DataFrame,
    *,
    n_pairs: int,
    max_rows_per_pair: int,
    seed: int,
) -> None:
    """
    Print groups of rows for the same monomer pair to verify:
      - same monomer pair
      - varying (r1,r2) and conditions
    """
    if n_pairs <= 0 or len(df_eval) == 0:
        return

    # Build pair + signatures (same as subset logic)
    df = df_eval.copy()
    df["_pair_key"] = _canonical_monomer_pair_key(df)
    df["_r_sig"] = (
        df["constant_1"].round(6).astype(str) + "," + df["constant_2"].round(6).astype(str)
    )
    cond_cols = ["temperature", "solvent_smiles", "polymerization_type", "method"]
    present = [c for c in cond_cols if c in df.columns]
    if present:
        df["_cond_sig"] = df[present].astype(str).agg("|".join, axis=1)
    else:
        df["_cond_sig"] = "NA"

    g = df.groupby("_pair_key", dropna=False)
    summary = (
        g.agg(
            n_rows=("r_product_class", "size"),
            n_r=(" _r_sig".strip(), "nunique") if False else ("_r_sig", "nunique"),
            n_cond=("_cond_sig", "nunique"),
            n_class=("r_product_class", "nunique"),
        )
        .reset_index()
        .sort_values(["n_class", "n_r", "n_cond", "n_rows"], ascending=False)
    )

    # Pick a diverse set from the top by shuffling within the eligible pool
    eligible = summary[(summary["n_r"] >= 2) & (summary["n_cond"] >= 2) & (summary["n_class"] >= 2)].copy()
    if len(eligible) == 0:
        print("  No monomer-pair groups with >=2 (r1,r2), >=2 condition signatures, and >=2 classes.")
        return
    rng = np.random.default_rng(int(seed))
    # sample from top 200 to keep output interesting but stable
    top_pool = eligible.head(min(200, len(eligible))).copy()
    pick = top_pool.sample(n=min(int(n_pairs), len(top_pool)), random_state=int(seed))

    cols_pref = [
        "reaction_id",
        "monomer1_smiles",
        "monomer2_smiles",
        "constant_1",
        "constant_2",
        "temperature",
        "solvent_smiles",
        "polymerization_type",
        "method",
        "r_product_class",
    ]
    cols = [c for c in cols_pref if c in df_eval.columns]

    print("\n  --- Condition-variation monomer-pair examples ---")
    for _, row in pick.iterrows():
        key = row["_pair_key"]
        df_pair = df[df["_pair_key"] == key].copy()
        # Sort to make differences easier to see
        sort_cols = [c for c in ["constant_1", "constant_2", "temperature", "solvent_smiles", "method"] if c in df_pair.columns]
        if sort_cols:
            df_pair = df_pair.sort_values(sort_cols).reset_index(drop=True)

        head = df_pair.head(int(max_rows_per_pair))
        if cols:
            head = head[cols]

        print(f"\n  Pair group: n_rows={int(row['n_rows'])}, n_r={int(row['n_r'])}, n_cond={int(row['n_cond'])}")
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 140,
            "display.max_colwidth", 60,
        ):
            print(head.to_string(index=False))
        if len(df_pair) > int(max_rows_per_pair):
            print(f"  ... ({len(df_pair) - int(max_rows_per_pair)} more rows not shown)")
    print("\n  --- end pair examples ---\n")


def _pair_class_stats_table(df_eval_special: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-(canonical) monomer-pair table with counts of each class.
    """
    df = df_eval_special.copy()
    df["_pair_key"] = _canonical_monomer_pair_key(df)
    df["_r_sig"] = (
        df["constant_1"].round(6).astype(str) + "," + df["constant_2"].round(6).astype(str)
    )
    cond_cols = ["temperature", "solvent_smiles", "polymerization_type", "method"]
    present = [c for c in cond_cols if c in df.columns]
    df["_cond_sig"] = df[present].astype(str).agg("|".join, axis=1) if present else "NA"

    # class counts
    counts = (
        df.pivot_table(
            index="_pair_key",
            columns="r_product_class",
            values="reaction_id" if "reaction_id" in df.columns else "constant_1",
            aggfunc="count",
            fill_value=0,
        )
        .rename(columns={0: "n_class0", 1: "n_class1", 2: "n_class2"})
        .reset_index()
    )
    for col in ["n_class0", "n_class1", "n_class2"]:
        if col not in counts.columns:
            counts[col] = 0

    g = df.groupby("_pair_key", dropna=False)
    meta = g.agg(
        n_rows=("r_product_class", "size"),
        n_r=("_r_sig", "nunique"),
        n_cond=("_cond_sig", "nunique"),
        n_class=("r_product_class", "nunique"),
    ).reset_index()

    out = meta.merge(counts, on="_pair_key", how="left")
    out = out.sort_values(["n_rows", "n_class", "n_r", "n_cond"], ascending=False).reset_index(drop=True)
    return out


def _voting_keep_mask(xgb_pred: np.ndarray, lookup_pred: np.ndarray) -> np.ndarray:
    """
    Keep samples where voting produces a prediction:
      - agree (xgb == lookup) OR
      - lookup abstains (-1) => use xgb
    Disagreeing cases are excluded (consistent with evaluate_voting()).
    """
    abstain = (lookup_pred == -1)
    agree = (xgb_pred == lookup_pred)
    return agree | abstain


def _voting_output(xgb_pred: np.ndarray, lookup_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (y_pred, keep_mask) for the voting scheme:
      - agree (xgb == lookup) => keep, pred = xgb
      - lookup abstains (-1)  => keep, pred = xgb
      - disagree              => drop (keep=False), pred = -1
    """
    abstain = (lookup_pred == -1)
    agree = (xgb_pred == lookup_pred)
    keep = agree | abstain
    y_pred = np.full(len(xgb_pred), -1, dtype=int)
    y_pred[keep] = xgb_pred[keep].astype(int)
    return y_pred, keep


def plot_top_pairs_true_vs_pred(
    *,
    df_special: pd.DataFrame,
    pair_keys_special: np.ndarray,
    y_true_special: np.ndarray,
    pred_with: np.ndarray,
    pred_without: np.ndarray,
    lookup_pred_special: np.ndarray,
    output_dir: str,
    top_n_pairs: int = 10,
) -> None:
    """
    Scatter plot for the top-N monomer pairs (by n_rows in special subset):
    x = monomer pair (top-N), y = class {0,1,2}.
    For each datapoint, plot three markers (slightly offset in x):
      - True class (gray)
      - Pred with conditions (red)
      - Pred without conditions (blue)
    """
    if len(df_special) == 0:
        print("  Top-pairs plot: special subset is empty; skipping.")
        return

    # Count rows per pair and pick top-N
    counts = pd.Series(pair_keys_special).value_counts()
    top_keys = counts.head(int(top_n_pairs)).index.tolist()
    if not top_keys:
        print("  Top-pairs plot: no pairs found; skipping.")
        return

    # Voting outputs + keep masks per model
    y_pred_with, keep_with = _voting_output(pred_with, lookup_pred_special)
    y_pred_without, keep_without = _voting_output(pred_without, lookup_pred_special)

    # Only plot points where we have a voting output for BOTH variants,
    # so each datapoint has three comparable markers.
    keep = keep_with & keep_without

    fig_w = TWO_COL_WIDTH_INCH
    fig_h = TWO_COL_WIDTH_INCH * 0.55
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # Visual settings
    x_off_true = -0.22
    x_off_with = 0.0
    x_off_wo = 0.22
    s = 14
    a = 0.55

    color_true = "#7a7a7a"
    color_with = "#661124"
    color_wo = "#143D60"

    # Plot per pair
    xticks = []
    xticklabels = []
    for i, key in enumerate(top_keys):
        m = (pair_keys_special == key) & keep
        if not np.any(m):
            continue

        yt = y_true_special[m].astype(int)
        yw = y_pred_with[m].astype(int)
        ywo = y_pred_without[m].astype(int)

        # small deterministic jitter to reduce overplotting
        rng = np.random.default_rng(12345 + i)
        jitter = rng.normal(0.0, 0.03, size=len(yt))

        x_base = float(i)
        ax.scatter(x_base + x_off_true + jitter, yt, s=s, alpha=a, color=color_true, label="True" if i == 0 else None)
        ax.scatter(x_base + x_off_with + jitter, yw, s=s, alpha=a, color=color_with, label="With cond." if i == 0 else None)
        ax.scatter(x_base + x_off_wo + jitter, ywo, s=s, alpha=a, color=color_wo, label="Without cond." if i == 0 else None)

        xticks.append(i)
        # Short label: first 10 chars of each SMILES in canonical key
        a_sm, b_sm = key.split("||", 1) if "||" in key else (key, "")
        xticklabels.append(f"{a_sm[:10]}… / {b_sm[:10]}…\n(n={int(counts[key])})")

    ax.set_title("Top monomer pairs in special test subset", fontsize=10)
    ax.set_xlabel("Monomer pair (top by n rows)", fontsize=9)
    ax.set_ylabel("Class", fontsize=9)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["Class 0", "Class 1", "Class 2"], fontsize=8)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=7, rotation=0, ha="center")
    ax.set_ylim(-0.4, 2.4)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out = os.path.join(output_dir, f"top_pairs_true_vs_pred.{ext}")
        plt.savefig(out, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  ✓ Saved {out}")
    plt.close(fig)

    # Also print a table for these top-N pairs (debugging / paper QA)
    try:
        df = df_special.copy()
        df["_pair_key"] = pair_keys_special
        df["_keep_both"] = keep
        df["_y_true"] = y_true_special.astype(int)
        df["_pred_with"] = y_pred_with.astype(int)
        df["_pred_without"] = y_pred_without.astype(int)

        df_top = df[df["_pair_key"].isin(top_keys) & df["_keep_both"]].copy()
        if len(df_top) == 0:
            print("  Top-pairs table: no rows (after keep mask); skipping print.")
            return

        # rank + counts for readability
        rank_map = {k: i + 1 for i, k in enumerate(top_keys)}
        df_top["pair_rank"] = df_top["_pair_key"].map(rank_map).astype(int)
        df_top["pair_n_rows"] = df_top["_pair_key"].map(counts.to_dict()).astype(int)

        cols_pref = [
            "pair_rank",
            "pair_n_rows",
            "reaction_id",
            "monomer1_smiles",
            "monomer2_smiles",
            "temperature",
            "solvent_smiles",
            "constant_1",
            "constant_2",
            "r_product_class",
            "_y_true",
            "_pred_with",
            "_pred_without",
        ]
        cols = [c for c in cols_pref if c in df_top.columns]
        df_show = df_top[cols].rename(
            columns={
                "r_product_class": "true_class",
                "_y_true": "true_class_check",
                "_pred_with": "pred_with_cond",
                "_pred_without": "pred_without_cond",
            }
        )
        df_show = df_show.sort_values(["pair_rank", "temperature", "solvent_smiles"], ascending=True)

        print("\n  --- Top pairs (table): true vs predictions ---")
        with pd.option_context(
            "display.max_rows", None,
            "display.max_columns", None,
            "display.width", 180,
            "display.max_colwidth", 80,
        ):
            print(df_show.to_string(index=False))
        print("  --- end top pairs table ---\n")
    except Exception as e:
        print(f"  Top-pairs table: failed to print due to: {e}")


def rows_with_class_change_no_temp_solvent_change(df_special: pd.DataFrame) -> pd.DataFrame:
    """
    Return rows from the special subset where (within the same canonical monomer pair
    and same temperature+solvent) the target class varies.

    This identifies cases where class changes cannot be explained by temperature or solvent.
    """
    required = [
        "monomer1_smiles",
        "monomer2_smiles",
        "r_product_class",
        "temperature",
        "solvent_smiles",
    ]
    missing = [c for c in required if c not in df_special.columns]
    if missing:
        raise ValueError(f"Missing required columns for no-temp/solvent-change analysis: {missing}")

    df = df_special.copy()
    df["_pair_key"] = _canonical_monomer_pair_key(df)
    # group key: same monomer pair AND same temp+solvent
    df["_pair_temp_solvent"] = (
        df["_pair_key"].astype(str)
        + "||T="
        + df["temperature"].astype(str)
        + "||S="
        + df["solvent_smiles"].astype(str)
    )

    g = df.groupby("_pair_temp_solvent", dropna=False)
    n_class = g["r_product_class"].nunique()
    keep_groups = set(n_class[n_class >= 2].index)
    out = df[df["_pair_temp_solvent"].isin(keep_groups)].copy()
    return out.drop(columns=["_pair_key", "_pair_temp_solvent"], errors="ignore")


def mask_class_change_no_temp_solvent_change(df_eval: pd.DataFrame) -> pd.Series:
    """
    Boolean mask over df_eval identifying rows where (within the same canonical
    monomer pair and same temperature+solvent) the target class varies.
    """
    required = [
        "monomer1_smiles",
        "monomer2_smiles",
        "r_product_class",
        "temperature",
        "solvent_smiles",
    ]
    missing = [c for c in required if c not in df_eval.columns]
    if missing:
        raise ValueError(f"Missing required columns for no-temp/solvent-change mask: {missing}")

    df = df_eval.copy()
    df["_pair_key"] = _canonical_monomer_pair_key(df)
    df["_pair_temp_solvent"] = (
        df["_pair_key"].astype(str)
        + "||T="
        + df["temperature"].astype(str)
        + "||S="
        + df["solvent_smiles"].astype(str)
    )
    g = df.groupby("_pair_temp_solvent", dropna=False)
    n_class = g["r_product_class"].nunique()
    keep_groups = set(n_class[n_class >= 2].index)
    return df["_pair_temp_solvent"].isin(keep_groups)


# ---------------------------------------------------------------------------
# Lookup with expanding nearest-neighbor majority vote
# ---------------------------------------------------------------------------
def compute_lookup_predictions_expanding(df_test, df_train, y_train,
                                          fp_dict):
    """Lookup predictions by expanding nearest neighbors until majority.

    For each test point the algorithm:
      1. Computes Tanimoto similarity to every training point.
      2. Groups training points by unique similarity levels (descending).
      3. Adds group after group (same-similarity neighbors) until one
         class has a *strict* majority among all neighbors seen so far.
      4. If no majority is reached after all neighbors, abstains (-1).

    The Lookup is purely SMILES-based (no reaction-condition features).

    Returns:
        lookup_pred : np.ndarray of int  (-1 = abstain)
        lookup_sim  : np.ndarray of float  (similarity of the deciding group)
    """
    from rdkit.Chem import DataStructs

    required = ['monomer1_smiles', 'monomer2_smiles', 'solvent_smiles']
    for col in required:
        if col not in df_test.columns or col not in df_train.columns:
            raise ValueError(f"Required column '{col}' missing")

    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    # Precompute training fingerprints
    train_mon1_fps = [fp_dict.get(sm) for sm in df_train['monomer1_smiles']]
    train_mon2_fps = [fp_dict.get(sm) for sm in df_train['monomer2_smiles']]
    train_solv_fps = [fp_dict.get(sm) for sm in df_train['solvent_smiles']]
    n_train = len(df_train)

    predictions = []
    similarities = []

    for test_pos, (test_idx, test_row) in enumerate(df_test.iterrows()):
        fp_m1 = fp_dict.get(test_row['monomer1_smiles'])
        fp_m2 = fp_dict.get(test_row['monomer2_smiles'])
        fp_sv = fp_dict.get(test_row['solvent_smiles'])

        if fp_m1 is None or fp_m2 is None or fp_sv is None:
            predictions.append(y_train[0] if n_train > 0 else 0)
            similarities.append(0.0)
            continue

        # Monomer similarity (direct + flipped, take best per training point)
        m1_d = np.array(DataStructs.BulkTanimotoSimilarity(fp_m1, train_mon1_fps))
        m2_d = np.array(DataStructs.BulkTanimotoSimilarity(fp_m2, train_mon2_fps))
        m1_f = np.array(DataStructs.BulkTanimotoSimilarity(fp_m1, train_mon2_fps))
        m2_f = np.array(DataStructs.BulkTanimotoSimilarity(fp_m2, train_mon1_fps))
        mon_sim = np.maximum((m1_d + m2_d) / 2.0, (m1_f + m2_f) / 2.0)

        # Solvent similarity
        solv_sim = np.array(DataStructs.BulkTanimotoSimilarity(fp_sv, train_solv_fps))

        # Combined similarity
        combined = (mon_sim + solv_sim) / 2.0

        # Sort training points by descending similarity
        order = np.argsort(-combined)
        sorted_sims = combined[order]
        sorted_labels = y_train[order]

        # Walk through unique similarity levels, expanding the voter pool
        pred_class = -1
        deciding_sim = 0.0
        counts = Counter()
        tol = 1e-10
        i = 0
        n = len(sorted_sims)

        while i < n:
            # Collect the next group of equal-similarity neighbors
            current_sim = sorted_sims[i]
            if np.isnan(current_sim):
                break
            while i < n and abs(sorted_sims[i] - current_sim) < tol:
                counts[int(sorted_labels[i])] += 1
                i += 1

            # Check for a strict majority
            top = counts.most_common(2)
            if len(top) == 1 or top[0][1] > top[1][1]:
                pred_class = top[0][0]
                deciding_sim = float(current_sim)
                break

        predictions.append(int(pred_class))
        similarities.append(deciding_sim)

        if (test_pos + 1) % 500 == 0:
            print(f"    Lookup: processed {test_pos + 1}/{len(df_test)}")

    return np.array(predictions), np.array(similarities)


# ---------------------------------------------------------------------------
# XGBoost training
# ---------------------------------------------------------------------------
PARAM_GRID = {
    'n_estimators': [500, 600, 700],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.04, 0.05, 0.06],
    'subsample': [0.85, 0.9, 0.95],
    'colsample_bytree': [0.85, 0.9, 1.0],
    'reg_alpha': [0.0, 0.1, 0.3],
    'reg_lambda': [1.0, 1.5, 2.0],
    'min_child_weight': [2, 3, 5],
    'gamma': [0.3, 0.5, 0.7],
}


def train_xgboost(df_train, features, config):
    """Train XGBoost with final model configurations and return (model, cv_score).
    
    Applies:
    - Specialized filter (if configured)
    - Augmentation (if configured)
    - NO negative data (matching final model config)
    """
    df_train_processed = df_train.copy()
    
    # Apply specialized filter (only to training data)
    if config.get('remove_specialized', False):
        if 'specialized_filter' in df_train_processed.columns:
            before = len(df_train_processed)
            df_train_processed = df_train_processed[
                df_train_processed['specialized_filter'] != 'specialized'
            ].reset_index(drop=True)
            print(f"    Applied specialized filter: {before} -> {len(df_train_processed)} samples")
        else:
            print("    Warning: 'specialized_filter' column not found")
    
    # Apply augmentation (if configured)
    if config.get('use_augmentation', False):
        original_len = len(df_train_processed)
        df_train_processed = data_augmentation.augment_with_gaussian_samples(
            df_train_processed,
            num_samples=config.get('augmentation_samples', 5),
            std_factor=0.3,
            random_state=config['random_state'],
        )
        print(f"    Applied augmentation: {original_len} -> {len(df_train_processed)} samples")
    
    # Note: Negative data is NOT added (matching final model config)
    
    X = df_train_processed[features]
    y = df_train_processed['r_product_class'].astype(int).values
    groups = df_train_processed['reaction_id'].astype(str).values

    class_weights = model_training.calculate_class_weights(y)

    result = model_training.train_xgboost_with_cv(
        X_train=X, y_train=y, groups=groups,
        param_grid=PARAM_GRID, n_iter=config['n_iter'],
        cv=5, random_state=config['random_state'],
        class_weights=class_weights, n_jobs=-1,
    )
    model = model_training.train_final_model(
        X_train=X, y_train=y,
        params=result['best_params'],
        class_weights=class_weights,
        random_state=config['random_state'],
    )
    return model, result['best_score']


# ---------------------------------------------------------------------------
# Voting evaluation (Lookup abstain = use XGBoost)
# ---------------------------------------------------------------------------
def evaluate_voting(xgb_pred, lookup_pred, y_true, label):
    """Voting: where Lookup agrees with XGBoost, use that.
    Where Lookup abstains (-1), use XGBoost directly.
    Where they disagree, exclude the sample.

    Returns metrics dict.
    """
    abstain = (lookup_pred == -1)
    agree = (xgb_pred == lookup_pred)
    use_xgb_fallback = abstain  # Lookup tied → take XGBoost
    voted = agree | use_xgb_fallback  # keep these samples

    n_total = len(y_true)
    n_agree = int(agree.sum())
    n_abstain = int(abstain.sum())
    n_voted = int(voted.sum())
    coverage = n_voted / n_total

    y_true_v = y_true[voted]
    y_pred_v = xgb_pred[voted]  # for agree, xgb == lookup; for abstain, use xgb

    macro_acc = balanced_accuracy_score(y_true_v, y_pred_v)
    macro_prec = precision_score(y_true_v, y_pred_v, average='macro',
                                 zero_division=0)
    per_cls_acc = recall_score(y_true_v, y_pred_v, labels=[0, 1, 2],
                               average=None, zero_division=0)
    per_cls_prec = precision_score(y_true_v, y_pred_v, labels=[0, 1, 2],
                                   average=None, zero_division=0)
    cm = sk_confusion_matrix(y_true_v, y_pred_v, labels=[0, 1, 2])

    print(f"  [{label}]  Macro Acc: {macro_acc:.4f}  |  Macro Prec: {macro_prec:.4f}")
    print(f"    Agree: {n_agree}  |  Lookup abstain (→XGBoost): {n_abstain}  "
          f"|  Disagree (removed): {n_total - n_voted}  |  Coverage: {coverage:.1%}")

    return {
        'label': label,
        'macro_accuracy': macro_acc,
        'macro_precision': macro_prec,
        'coverage': coverage,
        'n_agree': n_agree,
        'n_abstain': n_abstain,
        'n_voted': n_voted,
        'n_total': n_total,
        'per_class_acc': per_cls_acc.tolist(),
        'per_class_prec': per_cls_prec.tolist(),
        'confusion_matrix': cm.tolist(),
    }


def evaluate_voting_within_pair(
    xgb_pred: np.ndarray,
    lookup_pred: np.ndarray,
    y_true: np.ndarray,
    pair_keys: np.ndarray,
    label: str,
) -> dict:
    """
    Evaluate voting model and compute the within-pair balanced accuracy:

      BalancedAcc_pair = mean_k recall_k   (over classes present in that pair)
      Final score      = mean over pairs (BalancedAcc_pair)

    The per-pair score is computed on the *kept/voted* samples only
    (i.e., after excluding disagreeing cases), consistent with other metrics.
    """
    base = evaluate_voting(xgb_pred, lookup_pred, y_true, label)

    abstain = (lookup_pred == -1)
    agree = (xgb_pred == lookup_pred)
    voted = agree | abstain

    y_true_v = y_true[voted]
    y_pred_v = xgb_pred[voted]
    pair_v = pair_keys[voted]

    # per-pair balanced accuracy (mean recall over classes present in the pair)
    per_pair_scores = []
    for key in pd.unique(pair_v):
        m = pair_v == key
        yt = y_true_v[m]
        yp = y_pred_v[m]
        classes_present = np.unique(yt)
        rec = recall_score(yt, yp, labels=classes_present, average=None, zero_division=0)
        per_pair_scores.append(float(np.mean(rec)) if len(rec) else 0.0)

    per_pair_scores = np.asarray(per_pair_scores, dtype=float)
    pair_score = float(np.mean(per_pair_scores)) if per_pair_scores.size else 0.0
    base.update(
        {
            "pair_balanced_accuracy": pair_score,
            "n_pairs": int(len(per_pair_scores)),
            "pair_balanced_accuracy_std": float(np.std(per_pair_scores)) if per_pair_scores.size else 0.0,
        }
    )
    print(f"    Within-pair balanced accuracy (mean over pairs): {pair_score:.4f}  (n_pairs={base['n_pairs']})")
    return base


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def create_comparison_plot(res_full, res_no_cond, output_dir):
    """3-panel comparison with shared legend below."""
    setup_plot_style()

    c1 = COMPARISON_COLORS.get('original', '#3A3B73')
    c2 = COMPARISON_COLORS.get('filtered', '#e27f07')
    alpha = 0.75
    w = 0.25
    pad = 0.4

    height = TWO_COL_WIDTH_INCH * 0.42
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3,
                                         figsize=(TWO_COL_WIDTH_INCH, height))

    # --- (a) Macro metrics ---
    labels_a = ['Within-pair\nBal. Acc', 'Macro\nPrecision']
    v_full = [res_full.get('pair_balanced_accuracy', res_full['macro_accuracy']), res_full['macro_precision']]
    v_no = [res_no_cond.get('pair_balanced_accuracy', res_no_cond['macro_accuracy']), res_no_cond['macro_precision']]
    x = np.arange(len(labels_a))

    b1 = ax1.bar(x - w / 2, v_full, w, label='With Reaction\nConditions',
                 color=c1, alpha=alpha)
    b2 = ax1.bar(x + w / 2, v_no, w, label='Without Reaction\nConditions',
                 color=c2, alpha=alpha)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                     f'{h:.2f}', ha='center', va='bottom', fontsize=6)
    ax1.set_ylabel('Score', fontsize=8)
    ax1.set_title('a', fontsize=10, loc='left', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_a, fontsize=7)
    ax1.set_xlim(x[0] - pad, x[-1] + pad)
    ax1.set_ylim(0, 1.08)
    ax1.tick_params(labelsize=6)
    ax1.grid(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # --- (b) Per-class accuracy ---
    classes = ['Class 0\n(Alternating)', 'Class 1\n(Block-like)',
               'Class 2\n(Homopolymer)']
    x2 = np.arange(3)
    ax2.bar(x2 - w / 2, res_full['per_class_acc'], w, color=c1, alpha=alpha)
    ax2.bar(x2 + w / 2, res_no_cond['per_class_acc'], w, color=c2, alpha=alpha)
    for vals, off in [(res_full['per_class_acc'], -w / 2),
                      (res_no_cond['per_class_acc'], w / 2)]:
        for i, v in enumerate(vals):
            ax2.text(i + off, v + 0.03, f'{v:.2f}',
                     ha='center', va='bottom', fontsize=5)
    ax2.set_ylabel('Accuracy', fontsize=8)
    ax2.set_title('b', fontsize=10, loc='left', fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(classes, fontsize=6)
    ax2.set_xlim(x2[0] - pad, x2[-1] + pad)
    ax2.set_ylim(0, 1.08)
    ax2.tick_params(labelsize=6)
    ax2.grid(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # --- (c) Coverage (% of test data predicted) ---
    x3 = np.arange(2)
    labels_c = ['With\nConditions', 'Without\nConditions']
    cov_full = res_full['coverage'] * 100
    cov_no = res_no_cond['coverage'] * 100
    covs = [cov_full, cov_no]
    ax3.bar(x3, covs, width=w, color=[c1, c2], alpha=alpha)
    for i, pct in enumerate(covs):
        ax3.text(i, pct + max(covs) * 0.02, f'{pct:.1f}%',
                 ha='center', va='bottom', fontsize=7)
    ax3.set_ylabel('Coverage (%)', fontsize=8)
    ax3.set_title('c', fontsize=10, loc='left', fontweight='bold')
    ax3.set_xticks(x3)
    ax3.set_xticklabels(labels_c, fontsize=7)
    ax3.set_xlim(x3[0] - pad, x3[-1] + pad)
    ax3.set_ylim(0, 108)
    ax3.tick_params(labelsize=6)
    ax3.grid(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Shared legend
    handles = [b1[0], b2[0]]
    leg_labels = ['With Reaction Conditions', 'Without Reaction Conditions']
    fig.legend(handles, leg_labels, loc='lower center', ncol=2,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    for ext in ['png', 'pdf']:
        path = os.path.join(output_dir, f'reaction_conditions_comparison.{ext}')
        plt.savefig(path, dpi=300 if ext == 'png' else None,
                    bbox_inches='tight')
        print(f"  ✓ Saved {path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    config = {
        'n_iter': args.n_iter,
        'random_state': args.random_state,
        'eval_split': args.eval_split,
        'subset': args.subset,
    }

    output_dir = os.path.join(_SCRIPT_DIR, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Plot-only mode
    # ------------------------------------------------------------------
    if args.plot_only:
        json_path = os.path.join(output_dir, 'comparison_results.json')
        if not os.path.exists(json_path):
            print(f"Error: {json_path} not found. Run without --plot-only first.")
            sys.exit(1)
        print("=" * 60)
        print("PLOT-ONLY MODE")
        print("=" * 60)
        with open(json_path) as f:
            saved = json.load(f)
        create_comparison_plot(
            saved['with_conditions'], saved['without_conditions'],
            output_dir,
        )
        print("Done.")
        return

    print("=" * 60)
    print("REACTION CONDITIONS COMPARISON — VOTING MODEL")
    print("  With vs Without Reaction Condition Features")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data …")
    df_train, df_val, df_test, df_neg = load_data()
    
    # Load final model configuration
    final_model_path = os.path.join(_PROJECT_ROOT, 'copol_prediction', 'artifacts', 'model_bundle')
    training_config = {}
    try:
        meta_path = os.path.join(final_model_path, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
                raw_cfg = meta.get('training_config', {}) or {}

                # Bundle metadata uses *_used and specialized_removed_from_training.
                # This experiment script uses remove_specialized/use_augmentation/add_negative_data flags.
                aug_used = bool(raw_cfg.get('augmentation_used', False))
                neg_used = bool(raw_cfg.get('negative_data_used', False))
                specialized_removed = bool(raw_cfg.get('specialized_removed_from_training', False))

                training_config = {
                    'use_augmentation': aug_used,
                    'add_negative_data': neg_used,
                    'remove_specialized': specialized_removed,
                }

                print("  Loaded training config from final model:")
                print(f"    remove_specialized: {training_config['remove_specialized']}")
                print(f"    use_augmentation: {training_config['use_augmentation']}")
                print(f"    add_negative_data: {training_config['add_negative_data']}")
    except Exception as e:
        print(f"  Warning: Could not load final model config: {e}")
        print("  Using defaults for THIS experiment: remove_specialized=False, use_augmentation=False, add_negative_data=False")
        training_config = {
            'remove_specialized': False,
            'use_augmentation': False,
            'add_negative_data': False,
        }

    # Hard default: augmentation OFF unless explicitly enabled via CLI.
    # This prevents experiments from silently inheriting augmentation settings.
    training_config['use_augmentation'] = bool(args.use_augmentation)
    
    # Prepare lookup pool (apply specialized filter, NO negative data)
    df_lookup_pool = df_train.copy()
    if training_config.get('remove_specialized', False):
        if 'specialized_filter' in df_lookup_pool.columns:
            before = len(df_lookup_pool)
            df_lookup_pool = df_lookup_pool[
                df_lookup_pool['specialized_filter'] != 'specialized'
            ].reset_index(drop=True)
            print(f"  Lookup pool: Applied specialized filter ({before} -> {len(df_lookup_pool)} samples)")
        else:
            print("  Lookup pool: Warning: 'specialized_filter' column not found; cannot apply specialized filter")
    
    # Note: Negative data is NOT added to lookup pool (matching final model config)
    print(f"  Lookup pool: {len(df_lookup_pool)} samples (train only, no negative data)")

    # Choose eval split (normal) + optional special subset
    df_eval_all = df_val if args.eval_split == "val" else df_test
    if "r_product_class" not in df_eval_all.columns:
        raise ValueError("Missing r_product_class in evaluation split.")

    print(f"  Evaluation split (normal): {args.eval_split} ({len(df_eval_all)} rows)")

    special_mask = None
    df_eval_special = None
    if args.subset == "condition_variation":
        special_mask_raw = condition_variation_mask(df_eval_all) | False
        # (condition_variation_mask already excludes no-temp/solvent-change rows)
        special_mask = special_mask_raw
        df_eval_special_raw = df_eval_all[special_mask].copy()
        # Ensure each reaction_id appears only once in the special subset
        if "reaction_id" in df_eval_special_raw.columns:
            before = len(df_eval_special_raw)
            df_eval_special_raw = df_eval_special_raw.drop_duplicates(subset=["reaction_id"], keep="first")
            after = len(df_eval_special_raw)
            print(f"  Special subset: dropped duplicate reaction_id rows ({before} -> {after})")
        df_eval_special = df_eval_special_raw.reset_index(drop=True)
        n_excluded_nts = int(mask_class_change_no_temp_solvent_change(df_eval_all).sum())
        print(f"  Evaluation split (special subset): condition_variation ({len(df_eval_special)} rows)")
        print(f"  Excluded (class change w/o temp+solvent change): {n_excluded_nts} rows (from eval split)")
    else:
        print("  Special subset disabled (--subset all).")

    # Feature sets
    all_features = [c for c in prediction_utils.feature_columns
                    if c in df_train.columns]
    no_cond_features = [f for f in all_features
                        if f not in REACTION_CONDITION_FEATURES]
    print(f"  All features:              {len(all_features)}")
    print(f"  Without reaction cond.:    {len(no_cond_features)}")
    print(f"  Removed features:          "
          f"{[f for f in REACTION_CONDITION_FEATURES if f in all_features]}")

    # ------------------------------------------------------------------
    # 2. Lookup predictions (shared, with majority vote)
    # ------------------------------------------------------------------
    print("\n[2/5] Computing Lookup predictions (expanding neighbor vote) …")
    smiles_cols = ['monomer1_smiles', 'monomer2_smiles', 'solvent_smiles']
    all_smiles = set()
    for data in [df_lookup_pool, df_eval_all]:
        for col in smiles_cols:
            if col in data.columns:
                all_smiles.update(data[col].dropna().unique())
    fp_dict = compute_fingerprints_for_smiles(list(all_smiles))
    n_valid = sum(1 for v in fp_dict.values() if v is not None)
    print(f"  Fingerprint cache: {n_valid}/{len(all_smiles)} SMILES")

    y_lookup_pool = df_lookup_pool['r_product_class'].astype(int).values
    lookup_pred, lookup_sim = compute_lookup_predictions_expanding(
        df_eval_all, df_lookup_pool, y_lookup_pool, fp_dict,
    )
    n_abstain = int((lookup_pred == -1).sum())
    print(f"  Lookup predictions: {len(lookup_pred)}  "
          f"(abstained on {n_abstain} — no majority found)")

    # Update config with final model settings
    config.update({
        'remove_specialized': training_config.get('remove_specialized', True),
        'use_augmentation': training_config.get('use_augmentation', False),
        'add_negative_data': training_config.get('add_negative_data', False),
        'augmentation_samples': 5 if training_config.get('use_augmentation', False) else 0,
    })
    
    print(f"\n  Training configuration:")
    print(f"    remove_specialized: {config['remove_specialized']}")
    print(f"    use_augmentation: {config['use_augmentation']}")
    print(f"    add_negative_data: {config['add_negative_data']}")
    print(f"    augmentation_samples: {config['augmentation_samples']}")

    # ------------------------------------------------------------------
    # 3. Train XGBoost — with reaction conditions
    # ------------------------------------------------------------------
    print("\n[3/5] Training XGBoost WITH reaction conditions …")
    xgb_full, cv_full = train_xgboost(df_train, all_features, config)
    xgb_pred_full = xgb_full.predict(df_eval_all[all_features])
    print(f"  CV score: {cv_full:.4f}")

    # ------------------------------------------------------------------
    # 4. Train XGBoost — without reaction conditions
    # ------------------------------------------------------------------
    print("\n[4/5] Training XGBoost WITHOUT reaction conditions …")
    xgb_no_cond, cv_no_cond = train_xgboost(df_train, no_cond_features, config)
    xgb_pred_no_cond = xgb_no_cond.predict(df_eval_all[no_cond_features])
    print(f"  CV score: {cv_no_cond:.4f}")

    # ------------------------------------------------------------------
    # 5. Voting evaluation
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating voting models …")
    # Normal evaluation (full chosen split)
    y_true_all = df_eval_all["r_product_class"].astype(int).values
    pair_keys_all = _canonical_monomer_pair_key(df_eval_all).values
    res_full_all = evaluate_voting_within_pair(
        xgb_pred_full, lookup_pred, y_true_all, pair_keys_all, "With Conditions (normal)"
    )
    res_no_cond_all = evaluate_voting_within_pair(
        xgb_pred_no_cond, lookup_pred, y_true_all, pair_keys_all, "Without Conditions (normal)"
    )

    # Special evaluation (subset) — this is what we will use for paper plotting
    res_full = None
    res_no_cond = None
    if special_mask is not None:
        idx = np.where(special_mask.values)[0]
        # Align idx with the de-duplicated special subset (reaction_id unique)
        if "reaction_id" in df_eval_all.columns:
            df_idx = df_eval_all.iloc[idx]
            idx = df_idx.drop_duplicates(subset=["reaction_id"], keep="first").index.values
        y_true = y_true_all[idx]
        pair_keys_all = _canonical_monomer_pair_key(df_eval_all).values
        pair_keys = pair_keys_all[idx]
        res_full = evaluate_voting_within_pair(
            xgb_pred_full[idx], lookup_pred[idx], y_true, pair_keys, "With Conditions (special)"
        )
        res_no_cond = evaluate_voting_within_pair(
            xgb_pred_no_cond[idx], lookup_pred[idx], y_true, pair_keys, "Without Conditions (special)"
        )

        # Per-pair class statistics for the special subset
        if args.print_pair_class_stats:
            stats = _pair_class_stats_table(df_eval_special)
            n_total_pairs = len(stats)
            n_print = min(int(args.pair_class_stats_max), n_total_pairs)
            print("\n  --- Special-subset per-pair class counts ---")
            print(f"  Pairs in special subset: {n_total_pairs}")
            with pd.option_context(
                "display.max_rows", None,
                "display.max_columns", None,
                "display.width", 160,
                "display.max_colwidth", 80,
            ):
                print(stats.head(n_print).to_string(index=False))
            if n_total_pairs > n_print:
                print(f"  ... ({n_total_pairs - n_print} more pairs not shown; increase --pair-class-stats-max or use --pair-class-stats-csv)")
            print("  --- end per-pair class counts ---\n")

            if args.pair_class_stats_csv:
                stats_path = os.path.join(output_dir, "pair_class_counts_special.csv")
                stats.to_csv(stats_path, index=False)
                print(f"  ✓ Saved per-pair class counts CSV: {stats_path}")

        # Print grouped examples + sample rows from the special subset
        if args.print_pairs and args.print_pairs > 0:
            _print_condition_variation_pairs(
                df_eval_special,
                n_pairs=int(args.print_pairs),
                max_rows_per_pair=int(args.max_rows_per_pair),
                seed=int(args.sample_seed),
            )

        if args.print_sample and args.print_sample > 0:
            n = min(int(args.print_sample), len(df_eval_special))
            if n == 0:
                print("  Sample rows (special subset): none (empty set)")
            else:
                cols_pref = [
                    "reaction_id",
                    "monomer1_smiles",
                    "monomer2_smiles",
                    "constant_1",
                    "constant_2",
                    "temperature",
                    "solvent_smiles",
                    "polymerization_type",
                    "method",
                    "r_product_class",
                ]
                cols = [c for c in cols_pref if c in df_eval_special.columns]
                sample_df = df_eval_special.sample(n=n, random_state=int(args.sample_seed)).copy()
                if cols:
                    sample_df = sample_df[cols]

                print("\n  --- Special-subset sample rows ---")
                with pd.option_context(
                    "display.max_rows", None,
                    "display.max_columns", None,
                    "display.width", 140,
                    "display.max_colwidth", 60,
                ):
                    print(sample_df.to_string(index=False))
                print("  --- end special sample ---\n")

                if args.sample_csv:
                    sample_path = os.path.join(output_dir, "evaluation_sample_special.csv")
                    sample_df.to_csv(sample_path, index=False)
                    print(f"  ✓ Saved special sample CSV: {sample_path}")

        # Plot: top-10 pairs with true vs predicted class markers
        if not args.no_top_pairs_plot:
            plot_top_pairs_true_vs_pred(
                df_special=df_eval_special,
                pair_keys_special=pair_keys,
                y_true_special=y_true,
                pred_with=xgb_pred_full[idx],
                pred_without=xgb_pred_no_cond[idx],
                lookup_pred_special=lookup_pred[idx],
                output_dir=output_dir,
                top_n_pairs=10,
            )

        # Analysis: class changes with no temperature/solvent change
        if args.print_no_temp_solvent_change:
            df_nts = rows_with_class_change_no_temp_solvent_change(df_eval_special)
            print("\n  --- Special subset: class changes without temp/solvent change ---")
            print(f"  Rows matching criterion: {len(df_nts)} / {len(df_eval_special)}")
            if len(df_nts) > 0:
                cols_pref = [
                    "reaction_id",
                    "monomer1_smiles",
                    "monomer2_smiles",
                    "constant_1",
                    "constant_2",
                    "temperature",
                    "solvent_smiles",
                    "polymerization_type",
                    "method",
                    "r_product_class",
                ]
                cols = [c for c in cols_pref if c in df_nts.columns]
                show = df_nts.copy()
                if cols:
                    show = show[cols]
                # deterministic sample for printing
                n_show = min(int(args.no_temp_solvent_max_rows), len(show))
                show = show.sample(n=n_show, random_state=int(args.sample_seed)) if len(show) > n_show else show
                with pd.option_context(
                    "display.max_rows", None,
                    "display.max_columns", None,
                    "display.width", 160,
                    "display.max_colwidth", 80,
                ):
                    print(show.to_string(index=False))
                if args.no_temp_solvent_csv:
                    out_path = os.path.join(output_dir, "special_subset_class_change_no_temp_solvent.csv")
                    df_nts.to_csv(out_path, index=False)
                    print(f"  ✓ Saved CSV: {out_path}")
            else:
                print("  (none found)")
            print("  --- end no-temp/solvent-change analysis ---\n")
    else:
        # If no special subset requested, use normal results as primary
        res_full, res_no_cond = res_full_all, res_no_cond_all

    # ---- Comparison table ----
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'With Cond.':>12} {'Without Cond.':>14} {'Δ':>10}")
    print("-" * 64)

    for key, label in [('pair_balanced_accuracy', 'Within-pair balanced acc'),
                       ('macro_accuracy', 'Macro Accuracy'),
                       ('macro_precision', 'Macro Precision'),
                       ('coverage', 'Coverage')]:
        w_ = res_full[key]
        wo = res_no_cond[key]
        d = wo - w_
        if key == 'coverage':
            print(f"{label:<25} {w_:>12.1%} {wo:>14.1%} {d:>+10.1%}")
        else:
            print(f"{label:<25} {w_:>12.4f} {wo:>14.4f} {d:>+10.4f}")

    print(f"{'CV score (XGBoost)':<25} {cv_full:>12.4f} {cv_no_cond:>14.4f} "
          f"{cv_no_cond - cv_full:>+10.4f}")
    print(f"{'Num features (XGBoost)':<25} {len(all_features):>12d} "
          f"{len(no_cond_features):>14d}")

    print("\n" + "-" * 64)
    print("PER-CLASS ACCURACY (RECALL)")
    print("-" * 64)
    cls_names = ['Class 0 (Alternating)', 'Class 1 (Block-like)',
                 'Class 2 (Homopolymer)']
    for i, name in enumerate(cls_names):
        w_ = res_full['per_class_acc'][i]
        wo = res_no_cond['per_class_acc'][i]
        print(f"  {name:<25} {w_:>10.4f} {wo:>12.4f} {wo - w_:>+10.4f}")
    print("=" * 70)

    # ---- Save results ----
    results_json = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'with_conditions': {
            **res_full,
            'cv_score': cv_full,
            'n_features': len(all_features),
            'features': all_features,
        },
        'without_conditions': {
            **res_no_cond,
            'cv_score': cv_no_cond,
            'n_features': len(no_cond_features),
            'features': no_cond_features,
        },
        'normal_eval': {
            'with_conditions': res_full_all,
            'without_conditions': res_no_cond_all,
            'n_rows': int(len(df_eval_all)),
        },
    }
    json_path = os.path.join(output_dir, 'comparison_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\n  Results: {json_path}")

    # ---- Plot ----
    print("\nCreating comparison plot …")
    create_comparison_plot(res_full, res_no_cond, output_dir)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
