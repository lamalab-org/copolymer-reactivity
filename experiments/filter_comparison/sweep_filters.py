#!/usr/bin/env python3
"""
Filter sweep for copolymerization reactivity prediction (Voting Model).

Tests combinations of data filters with the voting model (XGBoost + Lookup).
For each configuration, only samples where XGBoost and Lookup agree are
evaluated ("voting").

Search space (16 combinations = 2 x 2 x 2 x 2):
  - remove_specialized:          [False, True]
  - apply_polymerization_filter: [False, True]
  - use_augmentation:            [False, True]  (XGBoost training only)
  
Note: The previous "negative data" training toggle has been removed. We now
only sweep the three filters above (8 combinations total).

Caching strategy (avoids redundant training):
  - 8 unique XGBoost models  (spec x poly x aug)
  - 4 unique Lookup pred sets (spec x poly)
  - 8 voting evaluations using cached predictions

Uses the central train/test split from copol_prediction/artifacts/data_splits/.

Usage:
    python sweep_filters.py [--output-dir DIR] [--plots-dir DIR]
"""

import os
import sys
import json
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import shutil
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
    data_augmentation,
    model_training,
    prediction_utils,
)
from utils import load_data_split
from copol_prediction.analysis.analyze_model import (
    compute_naive_baseline_predictions_with_similarity,
    compute_fingerprints_for_smiles,
)
from copol_prediction.mayo_lewis_classification import classify_reactivity_curve

try:
    from copol_prediction.analysis.plot_config import setup_plot_style, HEATMAP_CMAP
except ImportError:
    def setup_plot_style():
        pass
    HEATMAP_CMAP = 'Blues'

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Note: Negative data is only applied to XGBoost training (like augmentation),
# NOT to Lookup pool. This simplifies to a boolean flag.


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep filter combinations for the voting model"
    )
    parser.add_argument("--output-dir", type=str,
                        default="experiments/filter_comparison/output/voting_sweep",
                        help="Directory to save result JSON/CSV")
    parser.add_argument("--plots-dir", type=str,
                        default="experiments/filter_comparison/output/voting_sweep",
                        help="Directory to save plots")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-iter", type=int, default=25,
                        help="Hyperparameter search iterations per model")
    parser.add_argument("--augmentation-samples", type=int, default=5)
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training, re-plot from saved results CSV")
    return parser.parse_args()


def _safe_move_tree(src_dir: str, dst_dir: str):
    """Move all files from src_dir into dst_dir, then delete src_dir if empty."""
    if not os.path.isdir(src_dir):
        return
    os.makedirs(dst_dir, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        rel_root = os.path.relpath(root, src_dir)
        out_root = dst_dir if rel_root == '.' else os.path.join(dst_dir, rel_root)
        os.makedirs(out_root, exist_ok=True)
        for fn in files:
            src_path = os.path.join(root, fn)
            dst_path = os.path.join(out_root, fn)
            if os.path.exists(dst_path):
                os.remove(dst_path)
            shutil.move(src_path, dst_path)
    # Remove empty directories
    for root, dirs, files in os.walk(src_dir, topdown=False):
        if not dirs and not files:
            try:
                os.rmdir(root)
            except OSError:
                pass


def migrate_legacy_outputs(output_dir: str, plots_dir: str):
    """Ensure all sweep outputs live under experiments/filter_comparison/output/."""
    # Legacy locations used previously in this repo
    legacy_results_dir = os.path.join(_PROJECT_ROOT, 'artifacts', 'experiments_voting')
    legacy_plots_dir = os.path.join(_PROJECT_ROOT, 'output', 'voting_sweep')

    _safe_move_tree(legacy_results_dir, output_dir)
    _safe_move_tree(legacy_plots_dir, plots_dir)


# ---------------------------------------------------------------------------
# Data loading helpers
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
        print(f"  Filtered {n_removed} rows with invalid SMILES "
              f"({len(df_out)} remaining)")
    return df_out


def load_base_data():
    """Load base train/validation split."""
    copol_dir = os.path.join(_PROJECT_ROOT, 'copol_prediction')
    split_dir = os.path.join(copol_dir, 'artifacts', 'data_splits')

    df_train, df_val, _ = load_data_split.load_train_val_test_split(split_dir=split_dir)
    load_data_split.print_split_info(split_dir=split_dir)

    # Filter out rows with unparseable SMILES to prevent RDKit segfaults
    print("Validating SMILES …")
    df_train = filter_valid_smiles(df_train)
    df_val = filter_valid_smiles(df_val)
    return df_train, df_val


def apply_cleaning_filters(df, remove_specialized, apply_poly_filter,
                           set_name="Data"):
    """Apply data-quality filters (specialized removal, poly-type filter)."""
    w = df.copy()
    t0 = len(w)

    w = w[w['r1r2'].notna() & (w['r1r2'] >= 0)]
    if len(w) < t0:
        print(f"    [{set_name}] r1r2 valid: {t0} -> {len(w)}")

    if remove_specialized:
        for spec_path in [
            os.path.join(_PROJECT_ROOT,
                         "copol_prediction/filter/llm_specialized_filter/classified_output.csv"),
        ]:
            if 'llm_specialized_filter' not in w.columns and os.path.exists(spec_path):
                df_spec = pd.read_csv(spec_path)
                if {'specialized_filter', 'reaction_id'}.issubset(df_spec.columns):
                    df_spec = df_spec[['reaction_id', 'specialized_filter']].rename(
                        columns={'specialized_filter': 'llm_specialized_filter'})
                    w = w.merge(df_spec, on='reaction_id', how='left')
                    break
        if 'llm_specialized_filter' in w.columns:
            t1 = len(w)
            w = w[w['llm_specialized_filter'] != 'specialized']
            if len(w) < t1:
                print(f"    [{set_name}] specialized: {t1} -> {len(w)}")

    if apply_poly_filter and 'polymerization_type' in w.columns:
        t2 = len(w)
        w = w[w['polymerization_type'].notna() & (w['polymerization_type'] != "")]
        if len(w) < t2:
            print(f"    [{set_name}] poly filter: {t2} -> {len(w)}")

    if 'r_product_class' not in w.columns:
        if {'constant_1', 'constant_2'}.issubset(w.columns):
            def _class_from_row(row):
                res = classify_reactivity_curve(float(row['constant_1']), float(row['constant_2']))
                return res['class_id']

            w['r_product_class'] = w.apply(_class_from_row, axis=1).astype(int)
        else:
            raise ValueError("Required columns 'constant_1' and 'constant_2' not found for class definition.")

    return w


def remove_nan_rows(df, features):
    """Drop rows with NaN in features or target."""
    X = df[features]
    y = df['r_product_class'].astype(int)
    mask = ~(X.isna().any(axis=1) | y.isna())
    return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training helper
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


def train_xgboost_model(df_train, features, config):
    """Train an XGBoost model on the given data.

    Returns (model, cv_score, best_params).
    """
    X_train = df_train[features]
    y_train = df_train['r_product_class'].astype(int).values
    groups = df_train['reaction_id'].astype(str).values

    class_weights = model_training.calculate_class_weights(y_train)

    train_result = model_training.train_xgboost_with_cv(
        X_train=X_train, y_train=y_train, groups=groups,
        param_grid=PARAM_GRID, n_iter=config['n_iter'],
        cv=5, random_state=config['random_state'],
        class_weights=class_weights, n_jobs=-1,
    )

    final_model = model_training.train_final_model(
        X_train=X_train, y_train=y_train,
        params=train_result['best_params'],
        class_weights=class_weights,
        random_state=config['random_state'],
    )

    return final_model, train_result['best_score'], train_result['best_params']


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    config = {
        'output_dir': args.output_dir,
        'plots_dir': args.plots_dir,
        'random_state': args.random_state,
        'n_iter': args.n_iter,
        'augmentation_samples': args.augmentation_samples,
    }

    os.makedirs(config['output_dir'], exist_ok=True)
    os.makedirs(config['plots_dir'], exist_ok=True)
    migrate_legacy_outputs(config['output_dir'], config['plots_dir'])

    # ------------------------------------------------------------------
    # Plot-only mode
    # ------------------------------------------------------------------
    if args.plot_only:
        csv_path = os.path.join(config['output_dir'], 'sweep_results.csv')
        if not os.path.exists(csv_path):
            # Backward compatibility: some runs wrote to legacy locations.
            legacy_csv = os.path.join(_PROJECT_ROOT, 'artifacts', 'experiments_voting', 'sweep_results.csv')
            if os.path.exists(legacy_csv):
                os.makedirs(config['output_dir'], exist_ok=True)
                shutil.copy2(legacy_csv, csv_path)
            else:
                print(f"Error: {csv_path} not found. Run without --plot-only first.")
                sys.exit(1)
        print("=" * 60)
        print("PLOT-ONLY MODE")
        print("=" * 60)
        results_df = pd.read_csv(csv_path)
        print(f"  Loaded {len(results_df)} configurations from {csv_path}")
        try:
            from plot_sweep_results import plot_sweep_results
            plot_sweep_results(results_df, config['plots_dir'])
        except ImportError:
            print("  Error: Could not import plot_sweep_results.")
        print("\nDone.")
        return

    print("=" * 60)
    print("FILTER SWEEP — VOTING MODEL (XGBoost + Lookup)")
    print("=" * 60)
    print(f"  Output dir:  {config['output_dir']}")
    print(f"  Plots dir:   {config['plots_dir']}")
    print(f"  HP iters:    {config['n_iter']}")
    print(f"  Random seed: {config['random_state']}")

    # ------------------------------------------------------------------
    # 1. Load base data
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("LOADING DATA")
    print("=" * 60)

    df_train_base, df_val_base = load_base_data()

    # ------------------------------------------------------------------
    # 2. Pre-compute fingerprints for all SMILES (used by Lookup)
    # ------------------------------------------------------------------
    print("\nPre-computing fingerprint cache …")
    smiles_cols = ['monomer1_smiles', 'monomer2_smiles', 'solvent_smiles']
    all_smiles = set()
    for data in [df_train_base, df_val_base]:
        for col in smiles_cols:
            if col in data.columns:
                all_smiles.update(data[col].dropna().unique())
    fp_dict = compute_fingerprints_for_smiles(list(all_smiles))
    n_valid = sum(1 for v in fp_dict.values() if v is not None)
    print(f"  Cached fingerprints for {n_valid}/{len(all_smiles)} unique SMILES")

    # ------------------------------------------------------------------
    # 3. Prepare cleaned data for each (spec, poly) combo
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PREPARING CLEANED DATA VARIANTS")
    print("=" * 60)

    features_map = {}      # (spec, poly) -> features list
    cleaned_cache = {}      # (spec, poly) -> (df_train_clean, df_test_clean)

    for spec in [False, True]:
        for poly in [False, True]:
            key = (spec, poly)
            print(f"\n  Cleaning variant spec={spec} poly={poly} …")
            df_tr = apply_cleaning_filters(df_train_base, spec, poly, "Train")
            df_te = apply_cleaning_filters(df_val_base, spec, poly, "Validation")

            feats = [c for c in prediction_utils.feature_columns if c in df_tr.columns]
            df_tr = remove_nan_rows(df_tr, feats)
            df_te = remove_nan_rows(df_te, feats)

            print(f"    Train: {len(df_tr)} | Validation: {len(df_te)} | Features: {len(feats)}")

            features_map[key] = feats
            cleaned_cache[key] = (df_tr, df_te)

    # ------------------------------------------------------------------
    # 4. Train all unique XGBoost models
    #    Unique key: (spec, poly, aug)  -> 8 models
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST MODELS (cached)")
    print("=" * 60)

    xgb_cache = {}   # key -> (model, cv_score, n_train)
    xgb_keys = set()
    for spec, poly, aug in itertools.product([False, True], [False, True], [False, True]):
        xgb_keys.add((spec, poly, aug))

    xgb_keys = sorted(xgb_keys)
    for i, (spec, poly, aug) in enumerate(xgb_keys, 1):
        key_str = f"spec={int(spec)} poly={int(poly)} aug={int(aug)}"
        print(f"\n[XGBoost {i}/{len(xgb_keys)}] {key_str}")

        df_tr, _ = cleaned_cache[(spec, poly)]
        feats = features_map[(spec, poly)]

        df_train_xgb = df_tr.copy()

        if aug:
            original_len = len(df_train_xgb)
            df_train_xgb = data_augmentation.augment_with_gaussian_samples(
                df_train_xgb,
                num_samples=config['augmentation_samples'],
                std_factor=0.3,
                random_state=config['random_state'],
            )
            print(f"  Augmentation: {original_len} -> {len(df_train_xgb)}")

        class_counts = pd.Series(
            df_train_xgb['r_product_class'].astype(int).values
        ).value_counts().sort_index()
        print(f"  Train size: {len(df_train_xgb)}  |  Classes: {dict(class_counts)}")

        model, cv_score, best_params = train_xgboost_model(df_train_xgb, feats, config)
        print(f"  CV score: {cv_score:.4f}")

        xgb_cache[(spec, poly, aug)] = (model, cv_score, len(df_train_xgb))

    # ------------------------------------------------------------------
    # 5. Compute all unique Lookup prediction sets
    #    Unique key: (spec, poly, neg_in_lookup)  -> 8 sets
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("COMPUTING LOOKUP PREDICTIONS (cached)")
    print("=" * 60)

    lookup_cache = {}  # key -> (lookup_pred, n_train_lookup)
    # Note: Negative data is NOT added to Lookup pool (only to XGBoost training)
    lookup_keys = []
    for spec, poly in itertools.product([False, True], [False, True]):
        lookup_keys.append((spec, poly))

    lookup_keys = sorted(lookup_keys)
    for i, (spec, poly) in enumerate(lookup_keys, 1):
        key_str = f"spec={int(spec)} poly={int(poly)}"
        print(f"\n[Lookup {i}/{len(lookup_keys)}] {key_str}")

        df_tr, df_te = cleaned_cache[(spec, poly)]
        feats = features_map[(spec, poly)]

        # Lookup pool: NO negative data (only original training data)
        df_train_lookup = df_tr.copy()

        y_train_lu = df_train_lookup['r_product_class'].astype(int).values
        print(f"  Lookup pool: {len(df_train_lookup)} samples (no negative data)")

        lu_pred, _ = compute_naive_baseline_predictions_with_similarity(
            df_te, df_train_lookup, y_train_lu, feats, fp_dict=fp_dict
        )
        lookup_cache[(spec, poly)] = (lu_pred, len(df_train_lookup))

    # ------------------------------------------------------------------
    # 6. Evaluate all voting combinations (8)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATING VOTING COMBINATIONS")
    print("=" * 60)

    search_space = list(itertools.product(
        [False, True],      # remove_specialized
        [False, True],      # apply_polymerization_filter
        [False, True],      # use_augmentation
    ))

    results = []
    for idx, (spec, poly, aug) in enumerate(search_space, 1):
        run_name = (f"spec{int(spec)}_poly{int(poly)}"
                    f"_aug{int(aug)}")
        print(f"\n[{idx}/{len(search_space)}] {run_name}")

        # Retrieve cached data
        _, df_te = cleaned_cache[(spec, poly)]
        feats = features_map[(spec, poly)]
        model, cv_score, n_train_xgb = xgb_cache[(spec, poly, aug)]
        # Lookup pool: NO negative data (only spec/poly filters)
        lu_pred, n_train_lookup = lookup_cache[(spec, poly)]

        # XGBoost predictions
        X_test = df_te[feats]
        y_test = df_te['r_product_class'].astype(int).values
        xgb_pred = model.predict(X_test)

        # Voting
        models_agree = (xgb_pred == lu_pred)
        n_agree = int(models_agree.sum())
        n_total = len(y_test)
        coverage = n_agree / n_total

        y_true_v = y_test[models_agree]
        y_pred_v = xgb_pred[models_agree]

        if len(y_true_v) == 0:
            print(f"  WARNING: No samples where models agree — skipping")
            continue

        macro_acc = balanced_accuracy_score(y_true_v, y_pred_v)
        macro_prec = precision_score(y_true_v, y_pred_v, average='macro',
                                     zero_division=0)
        per_cls_acc = recall_score(y_true_v, y_pred_v, labels=[0, 1, 2],
                                   average=None, zero_division=0)
        cm = sk_confusion_matrix(y_true_v, y_pred_v, labels=[0, 1, 2])

        print(f"  Macro Acc: {macro_acc:.4f}  |  Macro Prec: {macro_prec:.4f}  "
              f"|  Coverage: {coverage:.1%}  |  Voting: {n_agree}/{n_total}")

        results.append({
            'run_name': run_name,
            'remove_specialized': spec,
            'apply_polymerization_filter': poly,
            'use_augmentation': aug,
            'cv_score': cv_score,
            'macro_accuracy': macro_acc,
            'macro_precision': macro_prec,
            'coverage': coverage,
            'per_class_acc_0': float(per_cls_acc[0]),
            'per_class_acc_1': float(per_cls_acc[1]),
            'per_class_acc_2': float(per_cls_acc[2]),
            'confusion_matrix': cm.tolist(),
            'n_train_xgb': n_train_xgb,
            'n_train_lookup': n_train_lookup,
            'n_test': n_total,
            'n_voting': n_agree,
        })

    # ------------------------------------------------------------------
    # 7. Save results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)

    if not results:
        print("No successful runs!")
        sys.exit(1)

    results_df = pd.DataFrame(results)

    csv_path = os.path.join(config['output_dir'], 'sweep_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"  CSV: {csv_path}")

    json_path = os.path.join(config['output_dir'], 'sweep_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  JSON: {json_path}")

    # ------------------------------------------------------------------
    # 8. Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    results_sorted = results_df.sort_values('macro_accuracy', ascending=False)

    print(f"\n{'Run':<40} {'Macro Acc':>10} {'Macro Prec':>11} "
          f"{'Coverage':>9} {'Voting':>7}")
    print("-" * 82)
    for _, r in results_sorted.head(10).iterrows():
        print(f"{r['run_name']:<40} {r['macro_accuracy']:>10.4f} "
              f"{r['macro_precision']:>11.4f} {r['coverage']:>9.1%} "
              f"{r['n_voting']:>7d}")

    best = results_sorted.iloc[0]
    print(f"\nBest: {best['run_name']}")
    print(f"  Macro Accuracy:  {best['macro_accuracy']:.4f}")
    print(f"  Macro Precision: {best['macro_precision']:.4f}")
    print(f"  Coverage:        {best['coverage']:.1%}")

    # ------------------------------------------------------------------
    # 9. Generate plots
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("CREATING PLOTS")
    print("=" * 60)

    try:
        from plot_sweep_results import plot_sweep_results
        plot_sweep_results(results_df, config['plots_dir'])
    except ImportError:
        print("  Warning: Could not import plot_sweep_results.")
        print("  Run plot_sweep_results.py separately to generate plots.")

    print("\n" + "=" * 60)
    print("SWEEP COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
