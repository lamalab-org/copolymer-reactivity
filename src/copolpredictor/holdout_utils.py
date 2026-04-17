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

    # Create 3-class target based on individual reactivity ratios (r1 = constant_1, r2 = constant_2)
    # 0: alternating         (r1 < 1 and r2 < 1)
    # 1: gradient            (rest)
    # 2: symmetric_blocky    (r1 > 1 and r2 > 1 and 0.5 < r1/r2 < 2)
    if {'constant_1', 'constant_2'}.issubset(base.columns):
        r1 = base['constant_1']
        r2 = base['constant_2']

        mask_alt = (r1 < 1) & (r2 < 1)
        ratio = r1 / r2
        mask_sym = (r1 > 1) & (r2 > 1) & (ratio > 0.5) & (ratio < 2)

        # Default: gradient (1)
        base['r_product_class'] = 1
        base.loc[mask_sym, 'r_product_class'] = 2
        base.loc[mask_alt, 'r_product_class'] = 0
    else:
        raise ValueError("Required columns 'constant_1' and 'constant_2' not found for class definition.")

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


def get_or_create_train_val_test_groups(
    base_df,
    group_col="reaction_id",
    test_groups_path="artifacts/test_ids.csv",
    val_groups_path="artifacts/val_ids.csv",
    test_size=0.2,
    val_size=0.1,
    random_state=42,
):
    """
    Create or load persistent train/validation/test splits at the GROUP level.
    
    This creates three splits:
    - Train: ~(1 - test_size - val_size) of groups
    - Validation: ~val_size of groups
    - Test: ~test_size of groups
    
    Behavior:
    - If saved lists exist, load and intersect with current groups.
    - Otherwise, create new grouped splits and save them.
    
    Returns:
        tuple: (train_groups, val_groups, test_groups) as pd.Series
    """
    if group_col not in base_df.columns:
        raise ValueError(f"'{group_col}' not found in base_df; cannot build grouped splits.")
    
    os.makedirs(os.path.dirname(test_groups_path), exist_ok=True)
    if val_groups_path != test_groups_path:
        os.makedirs(os.path.dirname(val_groups_path), exist_ok=True)
    
    # Current groups as strings (robust to type changes across runs)
    current_groups = pd.Series(base_df[group_col].astype(str).unique())
    
    # 1) Try to load persisted IDs
    test_groups = None
    val_groups = None
    
    if os.path.exists(test_groups_path):
        saved_test = pd.read_csv(test_groups_path)
        if group_col in saved_test.columns:
            saved_test_ids = saved_test[group_col].astype(str)
            inter_test = saved_test_ids[saved_test_ids.isin(current_groups)]
            if len(inter_test) > 0:
                test_groups = inter_test.reset_index(drop=True)
    
    if os.path.exists(val_groups_path):
        saved_val = pd.read_csv(val_groups_path)
        if group_col in saved_val.columns:
            saved_val_ids = saved_val[group_col].astype(str)
            inter_val = saved_val_ids[saved_val_ids.isin(current_groups)]
            if len(inter_val) > 0:
                val_groups = inter_val.reset_index(drop=True)
    
    # If both exist and don't overlap, use them
    if test_groups is not None and val_groups is not None:
        # Check for overlap
        test_set = set(test_groups)
        val_set = set(val_groups)
        if len(test_set & val_set) == 0:
            # Both are valid and don't overlap, return them
            train_set = set(current_groups) - test_set - val_set
            train_groups = pd.Series(list(train_set)).reset_index(drop=True)
            return train_groups, val_groups, test_groups
        else:
            print("  [holdout_utils] Saved val/test IDs overlap → regenerating.")
    
    # 2) Create fresh splits
    # First split: separate test from rest
    splitter1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_rest, idx_test = next(splitter1.split(base_df, groups=base_df[group_col].astype(str)))
    
    # Second split: separate val from train (from the rest)
    df_rest = base_df.iloc[idx_rest].reset_index(drop=True)
    # Calculate val_size relative to the rest (not the full dataset)
    val_size_relative = val_size / (1 - test_size)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_size_relative, random_state=random_state + 1)
    idx_train, idx_val = next(splitter2.split(df_rest, groups=df_rest[group_col].astype(str)))
    
    # Extract group IDs
    test_groups = pd.Series(base_df.iloc[idx_test][group_col].astype(str).unique())
    val_groups = pd.Series(df_rest.iloc[idx_val][group_col].astype(str).unique())
    train_groups = pd.Series(df_rest.iloc[idx_train][group_col].astype(str).unique())
    
    # 3) Persist for reproducibility
    pd.DataFrame({group_col: test_groups}).to_csv(test_groups_path, index=False)
    pd.DataFrame({group_col: val_groups}).to_csv(val_groups_path, index=False)
    
    return train_groups.reset_index(drop=True), val_groups.reset_index(drop=True), test_groups.reset_index(drop=True)


def split_train_val_test(df, train_groups, val_groups, test_groups, group_col='reaction_id'):
    """
    Split dataframe into train, validation, and test sets based on group IDs.
    No additional filtering; this just partitions by group membership.
    
    Returns:
        tuple: (df_train, df_val, df_test)
    """
    train_set = set(pd.Series(train_groups).astype(str))
    val_set = set(pd.Series(val_groups).astype(str))
    test_set = set(pd.Series(test_groups).astype(str))
    
    df_train = df[df[group_col].astype(str).isin(train_set)].reset_index(drop=True)
    df_val = df[df[group_col].astype(str).isin(val_set)].reset_index(drop=True)
    df_test = df[df[group_col].astype(str).isin(test_set)].reset_index(drop=True)
    
    return df_train, df_val, df_test
