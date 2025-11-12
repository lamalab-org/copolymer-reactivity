"""
Holdout set management utilities for copolymerization prediction.

This module creates/loads persistent group-based holdout sets
WITHOUT applying train-only filters.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def get_or_create_holdout_groups(
    base_df,
    group_col="reaction_id",
    test_groups_path="artifacts/test_ids.csv",
    test_size=0.2,
    random_state=42,
):
    """
    Create or load a persistent global hold-out at the GROUP level.

    Behavior:
    - If a saved list of test groups exists, intersect with current groups.
      If intersection is empty or tiny, fall back to creating a new split.
    - Otherwise, create a new grouped split with approx. `test_size`, save it, and return.

    Returns:
        pd.Series of unique group IDs for the holdout set (dtype=str)
    """
    if group_col not in base_df.columns:
        raise ValueError(f"'{group_col}' not found in base_df; cannot build grouped hold-out.")

    os.makedirs(os.path.dirname(test_groups_path), exist_ok=True)

    # Current groups as strings (robust to type changes across runs)
    current_groups = pd.Series(base_df[group_col].astype(str).unique())

    # 1) Try to load persisted IDs
    if os.path.exists(test_groups_path):
        saved = pd.read_csv(test_groups_path)
        if group_col in saved.columns:
            saved_ids = saved[group_col].astype(str)
            inter = saved_ids[saved_ids.isin(current_groups)]
            if len(inter) > 0:
                return inter.reset_index(drop=True)
            else:
                print("  [holdout_utils] Saved test IDs had no overlap with current data → regenerating.")

    # 2) Create a fresh grouped split hitting ~test_size
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    # Use a stable index (no filtering here—base_df is already the minimal “holdout-safe” base)
    idx_train, idx_test = next(splitter.split(base_df, groups=base_df[group_col].astype(str)))
    holdout_groups = pd.Series(base_df.iloc[idx_test][group_col].astype(str).unique())

    # 3) Persist for reproducibility
    pd.DataFrame({group_col: holdout_groups}).to_csv(test_groups_path, index=False)

    return holdout_groups.reset_index(drop=True)


def make_base_dataset_for_holdout(df):
    """
    Build the base dataset used ONLY to determine group-level holdout IDs.

    IMPORTANT:
    - Do NOT apply train-only filters here (e.g., removing 'specialized').
    - Keep only minimal sanity checks needed for target creation.

    Returns:
        base_df (minimal-cleaned copy of df)
    """
    base = df.copy()

    # Minimal, non-controversial checks (not "filters" by business logic):
    # - r1r2 must exist and be non-negative so we can derive targets uniformly later.
    base = base[base['r1r2'].notna()]
    base = base[base['r1r2'] >= 0]

    # DO NOT drop 'specialized' here – that’s a train-only filter.

    # Create 3-class target for consistency (harmless; not a filter)
    bins = [-np.inf, 1, 25, np.inf]
    labels = [0, 1, 2]
    base['r_product_class'] = pd.cut(base['r1r2'], bins=bins, labels=labels, right=False).astype(int)

    # Extreme override (same as train/holdout target logic; not a filter)
    if {'constant_1', 'constant_2'}.issubset(base.columns):
        extreme_mask = (
            ((base['constant_1'] <= 0.1) & (base['constant_2'] > 25)) |
            ((base['constant_2'] <= 0.1) & (base['constant_1'] > 25))
        )
        base.loc[extreme_mask, 'r_product_class'] = 2

    if 'reaction_id' not in base.columns:
        raise ValueError("reaction_id is required for grouped hold-out")

    return base


def split_train_holdout(df, holdout_groups, group_col='reaction_id'):
    """
    Split dataframe into train and holdout sets based on group IDs.
    No additional filtering; this just partitions by group membership.
    """
    holdout_set = set(pd.Series(holdout_groups).astype(str))
    df_holdout = df[df[group_col].astype(str).isin(holdout_set)].reset_index(drop=True)
    df_train = df[~df[group_col].astype(str).isin(holdout_set)].reset_index(drop=True)
    return df_train, df_holdout
