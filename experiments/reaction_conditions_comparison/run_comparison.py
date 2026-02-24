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
  - Specialized filter applied to training data (and lookup pool)
  - Augmentation enabled (Gaussian sampling)
  - NO negative data (matching final model config)
  - Evaluation on validation set (not test set)

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
    - Specialized filter applied to training data only
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
    labels_a = ['Macro\nAccuracy', 'Macro\nPrecision']
    v_full = [res_full['macro_accuracy'], res_full['macro_precision']]
    v_no = [res_no_cond['macro_accuracy'], res_no_cond['macro_precision']]
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
                training_config = meta.get('training_config', {})
                print(f"  Loaded training config from final model:")
                print(f"    remove_specialized: {training_config.get('remove_specialized', False)}")
                print(f"    use_augmentation: {training_config.get('use_augmentation', False)}")
                print(f"    add_negative_data: {training_config.get('add_negative_data', False)}")
    except Exception as e:
        print(f"  Warning: Could not load final model config: {e}")
        print(f"  Using defaults: remove_specialized=True, use_augmentation=True, add_negative_data=False")
        training_config = {
            'remove_specialized': True,
            'use_augmentation': True,
            'add_negative_data': False,
        }
    
    # Prepare lookup pool (apply specialized filter, NO negative data)
    df_lookup_pool = df_train.copy()
    if training_config.get('remove_specialized', False):
        if 'specialized_filter' in df_lookup_pool.columns:
            before = len(df_lookup_pool)
            df_lookup_pool = df_lookup_pool[
                df_lookup_pool['specialized_filter'] != 'specialized'
            ].reset_index(drop=True)
            print(f"  Lookup pool: Applied specialized filter ({before} -> {len(df_lookup_pool)} samples)")
    
    # Note: Negative data is NOT added to lookup pool (matching final model config)
    print(f"  Lookup pool: {len(df_lookup_pool)} samples (train only, no negative data)")

    # Evaluate on validation set (not test set)
    y_true = df_val['r_product_class'].astype(int).values

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
    for data in [df_lookup_pool, df_val]:
        for col in smiles_cols:
            if col in data.columns:
                all_smiles.update(data[col].dropna().unique())
    fp_dict = compute_fingerprints_for_smiles(list(all_smiles))
    n_valid = sum(1 for v in fp_dict.values() if v is not None)
    print(f"  Fingerprint cache: {n_valid}/{len(all_smiles)} SMILES")

    y_lookup_pool = df_lookup_pool['r_product_class'].astype(int).values
    lookup_pred, lookup_sim = compute_lookup_predictions_expanding(
        df_val, df_lookup_pool, y_lookup_pool, fp_dict,
    )
    n_abstain = int((lookup_pred == -1).sum())
    print(f"  Lookup predictions: {len(lookup_pred)}  "
          f"(abstained on {n_abstain} — no majority found)")

    # Update config with final model settings
    config.update({
        'remove_specialized': training_config.get('remove_specialized', True),
        'use_augmentation': training_config.get('use_augmentation', True),
        'add_negative_data': training_config.get('add_negative_data', False),
        'augmentation_samples': 5,
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
    xgb_pred_full = xgb_full.predict(df_val[all_features])
    print(f"  CV score: {cv_full:.4f}")

    # ------------------------------------------------------------------
    # 4. Train XGBoost — without reaction conditions
    # ------------------------------------------------------------------
    print("\n[4/5] Training XGBoost WITHOUT reaction conditions …")
    xgb_no_cond, cv_no_cond = train_xgboost(df_train, no_cond_features, config)
    xgb_pred_no_cond = xgb_no_cond.predict(df_val[no_cond_features])
    print(f"  CV score: {cv_no_cond:.4f}")

    # ------------------------------------------------------------------
    # 5. Voting evaluation
    # ------------------------------------------------------------------
    print("\n[5/5] Evaluating voting models …")
    res_full = evaluate_voting(xgb_pred_full, lookup_pred, y_true,
                               "With Conditions")
    res_no_cond = evaluate_voting(xgb_pred_no_cond, lookup_pred, y_true,
                                  "Without Conditions")

    # ---- Comparison table ----
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"\n{'Metric':<25} {'With Cond.':>12} {'Without Cond.':>14} {'Δ':>10}")
    print("-" * 64)

    for key, label in [('macro_accuracy', 'Macro Accuracy'),
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
