#!/usr/bin/env python3
"""
Model analysis script for copolymerization prediction.

Generates various analysis plots for trained models.

Usage:
    python analyze_model.py [--all] [--combined] [--confusion] [--confidence] [--features] [--calibration]
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure copol_prediction/ is on sys.path when run as a script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WORKSPACE_ROOT = os.path.dirname(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from copolpredictor.prediction_utils import create_grouped_kfold_splits

try:
    # When imported as package: copol_prediction.analysis.analyze_model
    from ..mayo_lewis_classification import classify_reactivity_curve
except ImportError:
    # When run as a script directly
    try:
        from mayo_lewis_classification import classify_reactivity_curve
    except ImportError:
        from copol_prediction.mayo_lewis_classification import classify_reactivity_curve

from utils.load_data_split import load_train_val_test_split

from copolpredictor.inference import CopolymerPredictor

try:
    # When imported as a package module (e.g. from experiments/)
    from .plot_config import (
        CALIBRATION_CONFIG,
        CLASS_COLORS,
        CLASS_LABELS,
        COMPARISON_COLORS,
        CONFIDENCE_PLOT_CONFIG,
        CONFUSION_MATRIX_CONFIG,
        ERROR_ANALYSIS_CONFIG,
        FEATURE_IMPORTANCE_CONFIG,
        HIGHLIGHT_COLORS,
        ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
        ONE_COL_WIDTH_INCH,
        SEQUENTIAL_COLORS,
        TWO_COL_GOLDEN_RATIO_HEIGHT_INCH,
        TWO_COL_WIDTH_INCH,
        get_class_color,
        get_class_label,
        setup_plot_style,
    )
except ImportError:
    # When run as a script from within copol_prediction/analysis
    from plot_config import (
        CALIBRATION_CONFIG,
        CLASS_COLORS,
        CLASS_LABELS,
        COMPARISON_COLORS,
        CONFIDENCE_PLOT_CONFIG,
        CONFUSION_MATRIX_CONFIG,
        ERROR_ANALYSIS_CONFIG,
        FEATURE_IMPORTANCE_CONFIG,
        HIGHLIGHT_COLORS,
        ONE_COL_GOLDEN_RATIO_HEIGHT_INCH,
        ONE_COL_WIDTH_INCH,
        SEQUENTIAL_COLORS,
        TWO_COL_GOLDEN_RATIO_HEIGHT_INCH,
        TWO_COL_WIDTH_INCH,
        get_class_color,
        get_class_label,
        setup_plot_style,
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Analyze voting model on test set")
    parser.add_argument(
        "--model-path", default="../artifacts/model_bundle", help="Path to model bundle"
    )
    parser.add_argument(
        "--output-dir", default="../output/analysis", help="Output directory for plots"
    )

    # Plot selection
    parser.add_argument("--all", action="store_true", help="Generate all plots")
    parser.add_argument(
        "--combined", action="store_true", help="Combined confusion matrix and confidence plot"
    )
    parser.add_argument("--confusion", action="store_true", help="Confusion matrix")
    parser.add_argument("--confidence", action="store_true", help="Confidence distribution")
    parser.add_argument("--features", action="store_true", help="Feature importance")
    parser.add_argument("--calibration", action="store_true", help="Calibration curve")
    parser.add_argument(
        "--calibration-compare",
        action="store_true",
        help="Compare uncalibrated vs Platt vs Isotonic calibration on validation set",
    )
    parser.add_argument("--errors", action="store_true", help="Error analysis by class")
    parser.add_argument("--confidence-vs-r1r2", action="store_true", help="Confidence vs r1r2 plot")
    parser.add_argument("--filtering", action="store_true", help="Confidence filtering analysis")
    parser.add_argument(
        "--min-retention",
        type=float,
        default=0.7,
        help="Minimum retention rate for filtering (default: 0.7)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for combined plot (default: 0.7)",
    )
    parser.add_argument(
        "--no-latex-table",
        dest="latex_table",
        action="store_false",
        default=True,
        help="Skip LaTeX performance table generation (default: table is created)",
    )
    parser.add_argument(
        "--n-folds", type=int, default=5, help="Number of CV folds for error bars (default: 5)"
    )

    return parser.parse_args()


def setup_style():
    """Setup matplotlib style."""
    setup_plot_style()  # Load lamalab.mplstyle and set color scheme


def calculate_tanimoto_similarity(fp1, fp2) -> float:
    """Calculate Tanimoto similarity between two fingerprints."""
    try:
        if fp1 is None or fp2 is None:
            return 0.0
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    except Exception:
        return 0.0


def load_train_with_negative_data(split_dir):
    """Load training data and append negative data for the lookup pool.

    Returns (df_train_with_neg, df_test).
    Note: Uses test set from train/val/test split (validation set not used here).
    """
    df_train, df_val, df_test = load_train_val_test_split(split_dir=split_dir)
    project_root = os.path.dirname(os.path.dirname(split_dir))  # up from artifacts/data_splits
    neg_path = os.path.join(
        project_root, "filter", "artificial_datapoints", "processed_combined_augmented.csv"
    )
    if os.path.exists(neg_path):
        df_neg = pd.read_csv(neg_path)
        if "Class" in df_neg.columns:
            df_neg = df_neg.rename(columns={"Class": "r_product_class"})
        df_neg["r_product_class"] = df_neg["r_product_class"].astype(int)
        df_train = pd.concat([df_train, df_neg], ignore_index=True)
        print(
            f"  ✓ Added {len(df_neg)} negative data points to lookup pool ({len(df_train)} total)"
        )
    else:
        print(f"  ⚠ Warning: Negative data not found at {neg_path}")
    return df_train, df_test


def compute_fingerprints_for_smiles(smiles_list, radius: int = 2, n_bits: int = 2048):
    """
    Compute fingerprints for a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings
        radius: Morgan fingerprint radius
        n_bits: Number of bits in fingerprint

    Returns:
        Dictionary mapping SMILES to fingerprints (None for invalid SMILES)
    """
    fp_dict = {}
    for smiles in smiles_list:
        if smiles in fp_dict:
            continue  # Already computed

        try:
            if pd.isna(smiles) or not smiles:
                fp_dict[smiles] = None
                continue

            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                fp_dict[smiles] = None
            else:
                fp_dict[smiles] = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        except Exception:
            fp_dict[smiles] = None

    return fp_dict


def compute_naive_baseline_predictions(df_test, df_train, y_train, feature_cols=None, fp_dict=None):
    """
    Compute naive baseline predictions by finding the most similar training point
    for each test point using Tanimoto similarity on SMILES.

    Falls mehrere Punkte die gleiche maximale Tanimoto-Ähnlichkeit haben,
    werden Features zur Unterscheidung verwendet.

    Args:
        df_test: Test DataFrame (must contain monomer1_smiles, monomer2_smiles, solvent_smiles)
        df_train: Training DataFrame (must contain monomer1_smiles, monomer2_smiles, solvent_smiles)
        y_train: Training labels (numpy array or Series)
        feature_cols: Optional list of feature columns to use for tie-breaking

    Returns:
        numpy array: Baseline predictions for test set
    """
    # Convert y_train to numpy array if needed
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    # Check required columns
    required_cols = ["monomer1_smiles", "monomer2_smiles", "solvent_smiles"]
    for col in required_cols:
        if col not in df_test.columns or col not in df_train.columns:
            raise ValueError(f"Required column '{col}' not found in dataframes")

    # Collect all unique SMILES (only if we don't get a cache)
    if fp_dict is None:
        print("  Computing fingerprints for unique SMILES...")
        unique_monomer1 = set(df_test["monomer1_smiles"].dropna().unique()) | set(
            df_train["monomer1_smiles"].dropna().unique()
        )
        unique_monomer2 = set(df_test["monomer2_smiles"].dropna().unique()) | set(
            df_train["monomer2_smiles"].dropna().unique()
        )
        unique_solvents = set(df_test["solvent_smiles"].dropna().unique()) | set(
            df_train["solvent_smiles"].dropna().unique()
        )

        # Compute fingerprints once for all unique SMILES
        all_unique_smiles = list(unique_monomer1 | unique_monomer2 | unique_solvents)
        fp_dict = compute_fingerprints_for_smiles(all_unique_smiles)
        print(
            f"  Computed fingerprints for {len([v for v in fp_dict.values() if v is not None])} valid SMILES"
        )

    # Precompute fingerprint lists for all training points to use BulkTanimotoSimilarity
    # Filter out None values to avoid segmentation faults with BulkTanimotoSimilarity
    train_mon1_fps = []
    train_mon2_fps = []
    train_solv_fps = []
    valid_train_indices = []

    for idx, (mon1, mon2, solv) in enumerate(
        zip(df_train["monomer1_smiles"], df_train["monomer2_smiles"], df_train["solvent_smiles"])
    ):
        fp1 = fp_dict.get(mon1)
        fp2 = fp_dict.get(mon2)
        fps = fp_dict.get(solv)

        # Only include training points with all valid fingerprints
        if fp1 is not None and fp2 is not None and fps is not None:
            train_mon1_fps.append(fp1)
            train_mon2_fps.append(fp2)
            train_solv_fps.append(fps)
            valid_train_indices.append(idx)

    n_train = len(df_train)
    n_valid_train = len(valid_train_indices)

    if n_valid_train == 0:
        raise ValueError("No valid fingerprints found in training data!")

    baseline_predictions = []

    # Process each test point
    for test_pos, (test_idx, test_row) in enumerate(df_test.iterrows()):
        test_mon1 = test_row["monomer1_smiles"]
        test_mon2 = test_row["monomer2_smiles"]
        test_solv = test_row["solvent_smiles"]

        # Get fingerprints for test point
        test_mon1_fp = fp_dict.get(test_mon1)
        test_mon2_fp = fp_dict.get(test_mon2)
        test_solv_fp = fp_dict.get(test_solv)

        # Fallback: if any test fingerprint is missing, just use first valid training point
        if test_mon1_fp is None or test_mon2_fp is None or test_solv_fp is None:
            best_positions = [valid_train_indices[0] if valid_train_indices else 0]
        else:
            # Calculate similarities in bulk for efficiency (only with valid fingerprints)
            # Monomer direct: (test_mon1 vs train_mon1, test_mon2 vs train_mon2)
            mon1_direct = np.array(DataStructs.BulkTanimotoSimilarity(test_mon1_fp, train_mon1_fps))
            mon2_direct = np.array(DataStructs.BulkTanimotoSimilarity(test_mon2_fp, train_mon2_fps))
            mon_sim_direct = (mon1_direct + mon2_direct) / 2.0

            # Monomer flipped: (test_mon1 vs train_mon2, test_mon2 vs train_mon1)
            mon1_flipped = np.array(
                DataStructs.BulkTanimotoSimilarity(test_mon1_fp, train_mon2_fps)
            )
            mon2_flipped = np.array(
                DataStructs.BulkTanimotoSimilarity(test_mon2_fp, train_mon1_fps)
            )
            mon_sim_flipped = (mon1_flipped + mon2_flipped) / 2.0

            # Take best monomer similarity per training point
            mon_similarity = np.maximum(mon_sim_direct, mon_sim_flipped)

            # Solvent similarity
            solv_similarity = np.array(
                DataStructs.BulkTanimotoSimilarity(test_solv_fp, train_solv_fps)
            )

            # Combined similarity (average of monomer and solvent)
            combined_similarity = (mon_similarity + solv_similarity) / 2.0

            # Track best similarity
            best_similarity = combined_similarity.max()
            if np.isnan(best_similarity):
                # Fallback if all similarities are NaN
                best_positions = [valid_train_indices[0] if valid_train_indices else 0]
            else:
                tol = 1e-10
                # Get indices in filtered list
                filtered_best_positions = np.where(
                    np.abs(combined_similarity - best_similarity) < tol
                )[0].tolist()
                if not filtered_best_positions:
                    # Fallback to argmax
                    filtered_best_positions = [int(np.nanargmax(combined_similarity))]
                # Map back to original DataFrame indices
                best_positions = [valid_train_indices[i] for i in filtered_best_positions]

        # If multiple points have the same similarity, use features to break tie
        if len(best_positions) > 1 and feature_cols is not None:
            # Get feature values for test point
            try:
                test_features = df_test.loc[test_idx, feature_cols].values

                # Convert to numeric and handle NaN
                test_features_numeric = pd.to_numeric(test_features, errors="coerce").astype(float)
                has_nan_test = pd.isna(test_features_numeric).any()

                if has_nan_test:
                    # If test has NaN, just take first best match
                    best_pos = best_positions[0]
                else:
                    # Find closest in feature space among tied points
                    best_pos = best_positions[0]
                    min_feature_dist = np.inf

                    for pos in best_positions:
                        try:
                            train_features = df_train.iloc[pos][feature_cols].values

                            # Convert to numeric and handle NaN
                            train_features_numeric = pd.to_numeric(
                                train_features, errors="coerce"
                            ).astype(float)
                            has_nan_train = pd.isna(train_features_numeric).any()

                            # Skip if train has NaN
                            if has_nan_train:
                                continue

                            # Calculate Euclidean distance in feature space
                            feature_dist = np.linalg.norm(
                                test_features_numeric - train_features_numeric
                            )

                            if feature_dist < min_feature_dist:
                                min_feature_dist = feature_dist
                                best_pos = pos
                        except Exception:
                            # Skip this train point if there's an error
                            continue
            except Exception:
                # If feature extraction fails, just take first best match
                best_pos = best_positions[0]
        else:
            # Just take first best match
            best_pos = best_positions[0]

        # Safety: ensure position is within bounds
        if not isinstance(best_pos, (int, np.integer)) or best_pos < 0 or best_pos >= n_train:
            best_pos = 0

        baseline_predictions.append(y_train[best_pos])

        # Optional progress logging for long runs
        if (test_pos + 1) % 500 == 0:
            print(f"  Baseline: processed {test_pos + 1}/{len(df_test)} test points")

    return np.array(baseline_predictions)


def compute_naive_baseline_predictions_with_similarity(
    df_test, df_train, y_train, feature_cols=None, fp_dict=None
):
    """
    Compute naive baseline predictions AND the corresponding similarity score
    (combined monomer+solvent similarity) for each test point.

    This mirrors compute_naive_baseline_predictions but additionally returns
    the best similarity value used for the chosen neighbor.

    Args:
        df_test: Test DataFrame (must contain monomer1_smiles, monomer2_smiles, solvent_smiles)
        df_train: Training DataFrame (must contain monomer1_smiles, monomer2_smiles, solvent_smiles)
        y_train: Training labels (numpy array or Series)
        feature_cols: Optional list of feature columns to use for tie-breaking
        fp_dict: Optional precomputed fingerprint dictionary {smiles: fp}

    Returns:
        Tuple (baseline_predictions, similarities) where:
            - baseline_predictions: np.ndarray of predicted classes
            - similarities: np.ndarray of best combined similarity per test point
    """
    # Convert y_train to numpy array if needed
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    # Check required columns
    required_cols = ["monomer1_smiles", "monomer2_smiles", "solvent_smiles"]
    for col in required_cols:
        if col not in df_test.columns or col not in df_train.columns:
            raise ValueError(f"Required column '{col}' not found in dataframes")

    # Collect all unique SMILES (only if we don't get a cache)
    if fp_dict is None:
        print("  Computing fingerprints for unique SMILES...")
        unique_monomer1 = set(df_test["monomer1_smiles"].dropna().unique()) | set(
            df_train["monomer1_smiles"].dropna().unique()
        )
        unique_monomer2 = set(df_test["monomer2_smiles"].dropna().unique()) | set(
            df_train["monomer2_smiles"].dropna().unique()
        )
        unique_solvents = set(df_test["solvent_smiles"].dropna().unique()) | set(
            df_train["solvent_smiles"].dropna().unique()
        )

        # Compute fingerprints once for all unique SMILES
        all_unique_smiles = list(unique_monomer1 | unique_monomer2 | unique_solvents)
        fp_dict = compute_fingerprints_for_smiles(all_unique_smiles)
        print(
            f"  Computed fingerprints for {len([v for v in fp_dict.values() if v is not None])} valid SMILES"
        )

    # Precompute fingerprint lists for all training points to use BulkTanimotoSimilarity
    # Filter out None values to avoid segmentation faults with BulkTanimotoSimilarity
    train_mon1_fps = []
    train_mon2_fps = []
    train_solv_fps = []
    valid_train_indices = []

    for idx, (mon1, mon2, solv) in enumerate(
        zip(df_train["monomer1_smiles"], df_train["monomer2_smiles"], df_train["solvent_smiles"])
    ):
        fp1 = fp_dict.get(mon1)
        fp2 = fp_dict.get(mon2)
        fps = fp_dict.get(solv)

        # Only include training points with all valid fingerprints
        if fp1 is not None and fp2 is not None and fps is not None:
            train_mon1_fps.append(fp1)
            train_mon2_fps.append(fp2)
            train_solv_fps.append(fps)
            valid_train_indices.append(idx)

    n_train = len(df_train)
    n_valid_train = len(valid_train_indices)

    if n_valid_train == 0:
        raise ValueError("No valid fingerprints found in training data!")

    baseline_predictions = []
    similarities = []

    # Process each test point
    for test_pos, (test_idx, test_row) in enumerate(df_test.iterrows()):
        test_mon1 = test_row["monomer1_smiles"]
        test_mon2 = test_row["monomer2_smiles"]
        test_solv = test_row["solvent_smiles"]

        # Get fingerprints for test point
        test_mon1_fp = fp_dict.get(test_mon1)
        test_mon2_fp = fp_dict.get(test_mon2)
        test_solv_fp = fp_dict.get(test_solv)

        # Fallback: if any test fingerprint is missing, just use first valid training point
        if test_mon1_fp is None or test_mon2_fp is None or test_solv_fp is None:
            best_positions = [valid_train_indices[0] if valid_train_indices else 0]
            best_similarity = 0.0
        else:
            # Calculate similarities in bulk for efficiency (only with valid fingerprints)
            # Monomer direct: (test_mon1 vs train_mon1, test_mon2 vs train_mon2)
            mon1_direct = np.array(DataStructs.BulkTanimotoSimilarity(test_mon1_fp, train_mon1_fps))
            mon2_direct = np.array(DataStructs.BulkTanimotoSimilarity(test_mon2_fp, train_mon2_fps))
            mon_sim_direct = (mon1_direct + mon2_direct) / 2.0

            # Monomer flipped: (test_mon1 vs train_mon2, test_mon2 vs train_mon1)
            mon1_flipped = np.array(
                DataStructs.BulkTanimotoSimilarity(test_mon1_fp, train_mon2_fps)
            )
            mon2_flipped = np.array(
                DataStructs.BulkTanimotoSimilarity(test_mon2_fp, train_mon1_fps)
            )
            mon_sim_flipped = (mon1_flipped + mon2_flipped) / 2.0

            # Take best monomer similarity per training point
            mon_similarity = np.maximum(mon_sim_direct, mon_sim_flipped)

            # Solvent similarity
            solv_similarity = np.array(
                DataStructs.BulkTanimotoSimilarity(test_solv_fp, train_solv_fps)
            )

            # Combined similarity (average of monomer and solvent)
            combined_similarity = (mon_similarity + solv_similarity) / 2.0

            # Track best similarity
            best_similarity = combined_similarity.max()
            if np.isnan(best_similarity):
                # Fallback if all similarities are NaN
                best_positions = [valid_train_indices[0] if valid_train_indices else 0]
                best_similarity = 0.0
            else:
                tol = 1e-10
                # Get indices in filtered list
                filtered_best_positions = np.where(
                    np.abs(combined_similarity - best_similarity) < tol
                )[0].tolist()
                if not filtered_best_positions:
                    # Fallback to argmax
                    filtered_best_positions = [int(np.nanargmax(combined_similarity))]
                # Map back to original DataFrame indices
                best_positions = [valid_train_indices[i] for i in filtered_best_positions]

        # If multiple points have the same similarity, use features to break tie
        if len(best_positions) > 1 and feature_cols is not None:
            # Get feature values for test point
            try:
                test_features = df_test.loc[test_idx, feature_cols].values

                # Convert to numeric and handle NaN
                test_features_numeric = pd.to_numeric(test_features, errors="coerce").astype(float)
                has_nan_test = pd.isna(test_features_numeric).any()

                if has_nan_test:
                    # If test has NaN, just take first best match
                    best_pos = best_positions[0]
                else:
                    # Find closest in feature space among tied points
                    best_pos = best_positions[0]
                    min_feature_dist = np.inf

                    for pos in best_positions:
                        try:
                            train_features = df_train.iloc[pos][feature_cols].values

                            # Convert to numeric and handle NaN
                            train_features_numeric = pd.to_numeric(
                                train_features, errors="coerce"
                            ).astype(float)
                            has_nan_train = pd.isna(train_features_numeric).any()

                            # Skip if train has NaN
                            if has_nan_train:
                                continue

                            # Calculate Euclidean distance in feature space
                            feature_dist = np.linalg.norm(
                                test_features_numeric - train_features_numeric
                            )

                            if feature_dist < min_feature_dist:
                                min_feature_dist = feature_dist
                                best_pos = pos
                        except Exception:
                            # Skip this train point if there's an error
                            continue
            except Exception:
                # If feature extraction fails, just take first best match
                best_pos = best_positions[0]
        else:
            # Just take first best match
            best_pos = best_positions[0]

        # Safety: ensure position is within bounds
        if not isinstance(best_pos, (int, np.integer)) or best_pos < 0 or best_pos >= n_train:
            best_pos = 0

        baseline_predictions.append(y_train[best_pos])
        similarities.append(float(best_similarity))

        # Optional progress logging for long runs
        if (test_pos + 1) % 500 == 0:
            print(f"  Lookup model: processed {test_pos + 1}/{len(df_test)} test points")

    return np.array(baseline_predictions), np.array(similarities)


def plot_confusion_matrix_and_confidence(
    y_true,
    y_pred,
    confidence_scores,
    correct_mask,
    output_dir,
    suffix="",
    confidence_threshold=0.7,
    normalize=False,
):
    """Plot combined figure: (a) confusion matrix, (b) conf-filtered confusion matrix, (c) confidence distribution.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidence_scores: Confidence scores
        correct_mask: Boolean mask indicating correct predictions
        output_dir: Output directory for plots
        suffix: Suffix for filename
        confidence_threshold: Confidence threshold for subplot (b). None = no filtering.
        normalize: If True, normalize confusion matrices by row (true label).
    """
    norm_label = "normalized " if normalize else ""
    # Store original data before filtering
    y_true_original = np.array(y_true).copy()
    y_pred_original = np.array(y_pred).copy()
    confidence_scores_original = np.array(confidence_scores).copy()
    correct_mask_original = np.array(correct_mask).copy()
    original_count = len(y_true)

    print(
        f"  Confidence score range (original): {confidence_scores_original.min():.3f} - {confidence_scores_original.max():.3f}"
    )

    # Filter by confidence threshold if specified and > 0
    if confidence_threshold is not None and confidence_threshold > 0:
        threshold_mask = confidence_scores >= confidence_threshold
        y_true_filtered = y_true[threshold_mask]
        y_pred_filtered = y_pred[threshold_mask]
        print(
            f"Generating {norm_label}combined plot{' (' + suffix + ')' if suffix else ''} (threshold: {confidence_threshold:.3f})..."
        )
        print(
            f"  Filtered: {original_count} → {len(y_true_filtered)} samples ({len(y_true_filtered)/original_count*100:.1f}% retained)"
        )
    else:
        y_true_filtered = y_true
        y_pred_filtered = y_pred
        print(f"Generating {norm_label}combined plot{' (' + suffix + ')' if suffix else ''}...")

    # Create figure with 3 subplots: (a) CM all, (b) CM filtered, (c) confidence dist
    height = TWO_COL_WIDTH_INCH * (5 / 14) * 1.2
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(TWO_COL_WIDTH_INCH * 1.5, height))

    # Compute confusion matrices
    cm_all = confusion_matrix(y_true_original, y_pred_original, labels=[0, 1, 2])
    if confidence_threshold is not None and confidence_threshold > 0:
        cm_filtered = confusion_matrix(y_true_filtered, y_pred_filtered, labels=[0, 1, 2])
    else:
        cm_filtered = cm_all

    if normalize:
        # Normalize by row (true label) to get proportions
        cm_all_plot = cm_all.astype(float) / cm_all.sum(axis=1, keepdims=True)
        cm_all_plot = np.nan_to_num(cm_all_plot)
        cm_filtered_plot = cm_filtered.astype(float) / cm_filtered.sum(axis=1, keepdims=True)
        cm_filtered_plot = np.nan_to_num(cm_filtered_plot)
        vmin, vmax_shared = 0, 1
        values_fmt = ".2f"
    else:
        cm_all_plot = cm_all
        cm_filtered_plot = cm_filtered
        vmin, vmax_shared = 0, 600
        values_fmt = CONFUSION_MATRIX_CONFIG["values_format"]

    # Subplot 1: Confusion Matrix (All data)
    disp_all = ConfusionMatrixDisplay(
        confusion_matrix=cm_all_plot, display_labels=[get_class_label(i) for i in range(3)]
    )
    im_all = disp_all.plot(
        cmap=CONFUSION_MATRIX_CONFIG["cmap"],
        ax=ax1,
        values_format=values_fmt,
        im_kw={"vmin": vmin, "vmax": vmax_shared},
        text_kw={"fontsize": 10},
    )
    ax1.set_title("a", fontsize=12, loc="left", fontweight="bold")
    ax1.set_xlabel(ax1.get_xlabel(), fontsize=10)
    ax1.set_ylabel(ax1.get_ylabel(), fontsize=10)
    ax1.tick_params(labelsize=8)
    ax1.grid(False)
    if im_all.im_ is not None:
        cbar_all = im_all.im_.colorbar
        if cbar_all is not None:
            cbar_all.ax.tick_params(labelsize=8)

    # Subplot 2: Confusion Matrix (Filtered data)
    disp_filtered = ConfusionMatrixDisplay(
        confusion_matrix=cm_filtered_plot, display_labels=[get_class_label(i) for i in range(3)]
    )
    im_filtered = disp_filtered.plot(
        cmap=CONFUSION_MATRIX_CONFIG["cmap"],
        ax=ax2,
        values_format=values_fmt,
        im_kw={"vmin": vmin, "vmax": vmax_shared},
        text_kw={"fontsize": 10},
    )
    ax2.set_title("b", fontsize=12, loc="left", fontweight="bold")
    ax2.set_xlabel(ax2.get_xlabel(), fontsize=10)
    ax2.set_ylabel(ax2.get_ylabel(), fontsize=10)
    ax2.tick_params(labelsize=8)
    ax2.grid(False)
    if im_filtered.im_ is not None:
        cbar_filtered = im_filtered.im_.colorbar
        if cbar_filtered is not None:
            cbar_filtered.ax.tick_params(labelsize=8)

    # Subplot 3: Confidence Distribution (All data, 0-1)
    correct_conf_all = confidence_scores_original[correct_mask_original]
    incorrect_conf_all = confidence_scores_original[~correct_mask_original]

    bins = np.linspace(0, 1, CONFIDENCE_PLOT_CONFIG["bins"] + 1)
    ax3.hist(
        correct_conf_all,
        bins=bins,
        alpha=CONFIDENCE_PLOT_CONFIG["alpha"],
        label="Correct",
        color=COMPARISON_COLORS["correct"],
        edgecolor=CONFIDENCE_PLOT_CONFIG["edgecolor"],
    )
    ax3.hist(
        incorrect_conf_all,
        bins=bins,
        alpha=CONFIDENCE_PLOT_CONFIG["alpha"],
        label="Incorrect",
        color=COMPARISON_COLORS["incorrect"],
        edgecolor=CONFIDENCE_PLOT_CONFIG["edgecolor"],
    )
    ax3.set_xlabel("Confidence Score", fontsize=12)
    ax3.set_ylabel("Count", fontsize=12)
    ax3.set_title("c", fontsize=14, loc="left", fontweight="bold")
    ax3.set_xlim(0, 1)
    ax3.legend(fontsize=10)
    ax3.grid(False)
    ax3.tick_params(labelsize=10)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save
    norm_suffix = "_normalized" if normalize else ""
    threshold_suffix = (
        f"_threshold_{confidence_threshold:.3f}"
        if (confidence_threshold is not None and confidence_threshold > 0)
        else ""
    )
    base = f'confusion_and_confidence{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}{norm_suffix}{threshold_suffix}'
    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"{base}.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
        print(f"  ✓ Saved {ext.upper()} to {path}")

    plt.close()


def plot_confusion_matrix(y_true, y_pred, output_dir, suffix=""):
    """Plot confusion matrix."""
    print(f"Generating confusion matrix{' (' + suffix + ')' if suffix else ''}...")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_INCH, 3))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=[get_class_label(i) for i in range(3)]
    )
    disp.plot(
        cmap=CONFUSION_MATRIX_CONFIG["cmap"],
        ax=ax,
        values_format=CONFUSION_MATRIX_CONFIG["values_format"],
        im_kw={"vmin": 0, "vmax": 2500},
    )

    title = "Confusion Matrix" + (" - " + suffix if suffix else "")
    plt.title(title, fontsize=14, pad=20)
    plt.tight_layout()

    filename = f'confusion_matrix{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.png'
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved to {path}")

    # Also save normalized version
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_INCH, 3))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_norm, display_labels=[get_class_label(i) for i in range(3)]
    )
    disp.plot(cmap=CONFUSION_MATRIX_CONFIG["cmap"], ax=ax, values_format=".2f")

    title_norm = "Normalized Confusion Matrix" + (" - " + suffix if suffix else "")
    plt.title(title_norm, fontsize=14, pad=20)
    plt.tight_layout()

    filename_norm = f'confusion_matrix_normalized{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.png'
    path_norm = os.path.join(output_dir, filename_norm)
    plt.savefig(path_norm, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved to {path_norm}")


def plot_confidence_distribution(confidence_scores, correct_mask, output_dir, suffix=""):
    """Plot confidence score distribution."""
    print(f"Generating confidence distribution plot{' (' + suffix + ')' if suffix else ''}...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TWO_COL_WIDTH_INCH, 3.5))

    # Overall distribution
    ax1.hist(
        confidence_scores,
        bins=CONFIDENCE_PLOT_CONFIG["bins"],
        edgecolor=CONFIDENCE_PLOT_CONFIG["edgecolor"],
        alpha=CONFIDENCE_PLOT_CONFIG["alpha"],
    )
    ax1.axvline(
        confidence_scores.mean(),
        color=HIGHLIGHT_COLORS["mean"],
        linestyle="--",
        label=f"Mean: {confidence_scores.mean():.3f}",
    )
    ax1.set_xlabel("Confidence Score", fontsize=12)
    ax1.set_ylabel("Count", fontsize=12)
    ax1.set_title("Overall Confidence Distribution", fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Correct vs incorrect
    correct_conf = confidence_scores[correct_mask]
    incorrect_conf = confidence_scores[~correct_mask]

    ax2.hist(
        correct_conf,
        bins=CONFIDENCE_PLOT_CONFIG["bins"],
        alpha=CONFIDENCE_PLOT_CONFIG["alpha"],
        label="Correct",
        color=COMPARISON_COLORS["correct"],
        edgecolor=CONFIDENCE_PLOT_CONFIG["edgecolor"],
    )
    ax2.hist(
        incorrect_conf,
        bins=CONFIDENCE_PLOT_CONFIG["bins"],
        alpha=CONFIDENCE_PLOT_CONFIG["alpha"],
        label="Incorrect",
        color=COMPARISON_COLORS["incorrect"],
        edgecolor=CONFIDENCE_PLOT_CONFIG["edgecolor"],
    )
    ax2.set_xlabel("Confidence Score", fontsize=12)
    ax2.set_ylabel("Count", fontsize=12)
    ax2.set_title("Confidence: Correct vs Incorrect", fontsize=13)
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()

    filename = (
        f'confidence_distribution{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.png'
    )
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved to {path}")

    # Print statistics
    print(f"  Mean confidence (correct): {correct_conf.mean():.3f}")
    print(f"  Mean confidence (incorrect): {incorrect_conf.mean():.3f}")


def format_feature_name(name):
    """Format feature name for display."""
    # Special replacements with numbered suffixes (specific ones first)
    name = name.replace("polytype_emb_1", "polymerization type emb. 1")
    name = name.replace("polytype_emb_2", "polymerization type emb. 2")
    name = name.replace("method_emb_1", "polymerization method emb. 1")
    name = name.replace("method_emb_2", "polymerization method emb. 2")
    # General cases without numbers
    name = name.replace("polytype_emb", "polymerization type emb.")
    name = name.replace("method_emb", "polymerization method emb.")

    # Delta HOMO-LUMO formatting
    if "delta_HOMO_LUMO" in name or "delta_homo_lumo" in name:
        # Replace delta with symbol
        name = name.replace("delta_HOMO_LUMO", "Δ HOMO-LUMO")
        name = name.replace("delta_homo_lumo", "Δ HOMO-LUMO")
        # Replace AA, AB, BA, BB with 1-1, 1-2, 2-1, 2-2
        name = name.replace("_AA", " 1-1")
        name = name.replace("_AB", " 1-2")
        name = name.replace("_BA", " 2-1")
        name = name.replace("_BB", " 2-2")

    # Replace remaining underscores with spaces
    name = name.replace("_", " ")

    return name


def plot_feature_importance(predictor, output_dir, top_n=20):
    """Plot feature importance from model."""
    print("Generating feature importance plot...")

    importance_df = predictor.get_feature_importance()
    top_features = importance_df.head(top_n)

    # Format feature names
    formatted_names = [format_feature_name(name) for name in top_features["feature"]]

    # Use TWO_COL width, dynamic height based on number of features
    height = max(4, top_n * 0.2)
    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, height))

    ax.barh(
        range(len(top_features)),
        top_features["importance"],
        color=FEATURE_IMPORTANCE_CONFIG["color"],
    )
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(formatted_names, fontsize=7)
    ax.set_xlabel("Importance", fontsize=9)
    ax.tick_params(axis="x", labelsize=7)
    ax.invert_yaxis()
    ax.grid(False)
    # Remove top and right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    # Save as PNG
    path_png = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved PNG to {path_png}")

    # Save as PDF
    path_pdf = os.path.join(output_dir, "feature_importance.pdf")
    plt.savefig(path_pdf, bbox_inches="tight")
    print(f"  ✓ Saved PDF to {path_pdf}")

    plt.close()

    # Save to CSV
    csv_path = os.path.join(output_dir, "feature_importance.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved CSV to {csv_path}")


def plot_calibration_curve_multiclass(y_true, y_proba, output_dir, suffix=""):
    """Plot calibration curves for each class."""
    print(f"Generating calibration curves{' (' + suffix + ')' if suffix else ''}...")

    # Original was (15, 5), ratio 3:1. With width 7, height = 7/3 ≈ 2.33, let's use 3 for better visibility
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL_WIDTH_INCH, 3))

    class_names = [get_class_label(i, style="long") for i in range(3)]

    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        # Binary indicator for this class
        y_binary = (y_true == i).astype(int)
        y_prob_class = y_proba[:, i]

        # Calculate calibration curve (quantile bins: same number of samples per bin)
        n_bins = CALIBRATION_CONFIG.get("n_bins", 5)
        strategy = CALIBRATION_CONFIG.get("strategy", "quantile")
        prob_true, prob_pred = calibration_curve(
            y_binary, y_prob_class, n_bins=n_bins, strategy=strategy
        )

        # Plot
        ax.plot(
            prob_pred,
            prob_true,
            marker=CALIBRATION_CONFIG["marker"],
            linewidth=1.5,
            markersize=4,
            color=get_class_color(i),
            label="Model",
        )
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

        ax.set_xlabel("Mean Predicted Probability", fontsize=8)
        ax.set_ylabel("Fraction of Positives", fontsize=8)
        ax.set_title(class_name, fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
        ax.grid(False)
        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save as PNG
    filename_png = (
        f'calibration_curves{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.png'
    )
    path_png = os.path.join(output_dir, filename_png)
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved PNG to {path_png}")

    # Save as PDF
    filename_pdf = (
        f'calibration_curves{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.pdf'
    )
    path_pdf = os.path.join(output_dir, filename_pdf)
    plt.savefig(path_pdf, bbox_inches="tight")
    print(f"  ✓ Saved PDF to {path_pdf}")

    plt.close()


def _fit_ovr_calibrators(y_true, y_proba, method="sigmoid"):
    """
    Fit one-vs-rest calibrators for each class.

    Args:
        y_true: True multiclass labels
        y_proba: Uncalibrated probability matrix (n_samples, n_classes)
        method: "sigmoid" (Platt scaling) or "isotonic"

    Returns:
        List of fitted per-class calibrators
    """
    n_classes = y_proba.shape[1]
    calibrators = []
    for cls in range(n_classes):
        y_binary = (y_true == cls).astype(int)
        p_cls = y_proba[:, cls]

        if method == "sigmoid":
            cal = LogisticRegression(max_iter=1000)
            cal.fit(p_cls.reshape(-1, 1), y_binary)
        elif method == "isotonic":
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(p_cls, y_binary)
        else:
            raise ValueError(f"Unknown method '{method}'")
        calibrators.append(cal)
    return calibrators


def _apply_ovr_calibrators(y_proba, calibrators):
    """
    Apply one-vs-rest calibrators and renormalize probabilities per sample.
    """
    calibrated_cols = []
    for cls, cal in enumerate(calibrators):
        p_cls = y_proba[:, cls]
        if hasattr(cal, "predict_proba"):
            p_cal = cal.predict_proba(p_cls.reshape(-1, 1))[:, 1]
        else:
            p_cal = cal.predict(p_cls)
        calibrated_cols.append(np.clip(p_cal, 1e-9, 1.0))

    p_mat = np.vstack(calibrated_cols).T
    row_sums = p_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return p_mat / row_sums


def _brier_per_class(y_true, y_proba):
    """
    Compute per-class Brier score for multiclass probabilities.
    """
    n_classes = y_proba.shape[1]
    scores = []
    for cls in range(n_classes):
        y_binary = (y_true == cls).astype(float)
        p_cls = y_proba[:, cls]
        scores.append(float(np.mean((p_cls - y_binary) ** 2)))
    return np.array(scores)


def _classification_error_per_class(y_true, y_pred, n_classes=3):
    """
    Per-class classification error = 1 - recall(class).
    """
    errors = []
    for cls in range(n_classes):
        mask = y_true == cls
        if np.sum(mask) == 0:
            errors.append(np.nan)
        else:
            cls_acc = np.mean(y_pred[mask] == y_true[mask])
            errors.append(float(1.0 - cls_acc))
    return np.array(errors)


def _confidence_from_proba(y_proba):
    """
    Confidence from calibrated probabilities:
    use predicted-class probability (equivalent to max probability).
    """
    return np.max(y_proba, axis=1)


def compare_calibration_methods_on_validation(df_val, predictor, output_dir, suffix="val"):
    """
    Compare uncalibrated probabilities vs Platt vs Isotonic on validation set.

    Generates:
    - calibration comparison curves per class
    - per-class Brier error bar chart
    """
    print("\n" + "=" * 60)
    print("CALIBRATION COMPARISON (validation set)")
    print("=" * 60)

    X_val = df_val[predictor.features]
    if "r_product_class" not in df_val.columns:
        raise ValueError("Validation set must contain 'r_product_class'")
    y_val = df_val["r_product_class"].astype(int).values

    # Base model probabilities
    y_proba_raw = predictor.predict_proba(X_val)

    # Fit OVR calibrators on validation set (quick first-pass comparison)
    platt_cals = _fit_ovr_calibrators(y_val, y_proba_raw, method="sigmoid")
    iso_cals = _fit_ovr_calibrators(y_val, y_proba_raw, method="isotonic")

    y_proba_platt = _apply_ovr_calibrators(y_proba_raw, platt_cals)
    y_proba_iso = _apply_ovr_calibrators(y_proba_raw, iso_cals)

    methods = [
        ("Uncalibrated", y_proba_raw, "#5A5A5A"),
        ("Platt", y_proba_platt, "#661124"),
        ("Isotonic", y_proba_iso, "#143D60"),
    ]

    # --- Calibration curves (3 subplots, one per class) ---
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL_WIDTH_INCH, 3))
    n_bins = CALIBRATION_CONFIG.get("n_bins", 5)
    strategy = CALIBRATION_CONFIG.get("strategy", "quantile")

    for cls, ax in enumerate(axes):
        y_binary = (y_val == cls).astype(int)
        for name, proba, color in methods:
            prob_true, prob_pred = calibration_curve(
                y_binary, proba[:, cls], n_bins=n_bins, strategy=strategy
            )
            ax.plot(
                prob_pred,
                prob_true,
                marker="o",
                linewidth=1.4,
                markersize=3.5,
                color=color,
                label=name,
            )

        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
        ax.set_title(get_class_label(cls, style="long"), fontsize=9)
        ax.set_xlabel("Mean Predicted Probability", fontsize=8)
        ax.set_ylabel("Fraction of Positives", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        ncol=3,
        fontsize=7,
        bbox_to_anchor=(0.5, -0.02),
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    curves_path = os.path.join(output_dir, f"calibration_comparison_{suffix}.png")
    plt.savefig(curves_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved calibration comparison curves: {curves_path}")

    # --- Per-class Brier error comparison ---
    brier_raw = _brier_per_class(y_val, y_proba_raw)
    brier_platt = _brier_per_class(y_val, y_proba_platt)
    brier_iso = _brier_per_class(y_val, y_proba_iso)

    x = np.arange(3)
    width = 0.26
    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH))
    ax.bar(x - width, brier_raw, width, label="Uncalibrated", color="#5A5A5A")
    ax.bar(x, brier_platt, width, label="Platt", color="#661124")
    ax.bar(x + width, brier_iso, width, label="Isotonic", color="#143D60")

    ax.set_xticks(x)
    ax.set_xticklabels([get_class_label(i, style="short") for i in range(3)], fontsize=8)
    ax.set_ylabel("Brier Score (lower is better)", fontsize=8)
    ax.set_title("Per-class calibration error (validation)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    brier_path = os.path.join(output_dir, f"calibration_brier_per_class_{suffix}.png")
    plt.savefig(brier_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved per-class Brier plot: {brier_path}")

    # --- Per-class classification error comparison ---
    y_pred_raw = np.argmax(y_proba_raw, axis=1)
    y_pred_platt = np.argmax(y_proba_platt, axis=1)
    y_pred_iso = np.argmax(y_proba_iso, axis=1)

    err_raw = _classification_error_per_class(y_val, y_pred_raw, n_classes=3)
    err_platt = _classification_error_per_class(y_val, y_pred_platt, n_classes=3)
    err_iso = _classification_error_per_class(y_val, y_pred_iso, n_classes=3)

    fig, ax = plt.subplots(figsize=(ONE_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH))
    ax.bar(x - width, err_raw, width, label="Uncalibrated", color="#5A5A5A")
    ax.bar(x, err_platt, width, label="Platt", color="#661124")
    ax.bar(x + width, err_iso, width, label="Isotonic", color="#143D60")

    ax.set_xticks(x)
    ax.set_xticklabels([get_class_label(i, style="short") for i in range(3)], fontsize=8)
    ax.set_ylabel("Classification Error (1 - recall)", fontsize=8)
    ax.set_title("Per-class prediction error (validation)", fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    class_err_path = os.path.join(output_dir, f"calibration_class_error_per_class_{suffix}.png")
    plt.savefig(class_err_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved per-class classification error plot: {class_err_path}")

    # --- Error-analysis-by-class histograms (correct vs incorrect) ---
    conf_raw = _confidence_from_proba(y_proba_raw)
    conf_platt = _confidence_from_proba(y_proba_platt)
    conf_iso = _confidence_from_proba(y_proba_iso)

    plot_error_analysis_by_class(
        y_val, y_pred_raw, conf_raw, output_dir, suffix=f"{suffix}_uncalibrated"
    )
    plot_error_analysis_by_class(
        y_val, y_pred_platt, conf_platt, output_dir, suffix=f"{suffix}_platt"
    )
    plot_error_analysis_by_class(
        y_val, y_pred_iso, conf_iso, output_dir, suffix=f"{suffix}_isotonic"
    )

    # Print compact numeric summary
    print("\nPer-class Brier scores (validation):")
    for cls in range(3):
        print(
            f"  Class {cls}: uncal={brier_raw[cls]:.4f}, "
            f"platt={brier_platt[cls]:.4f}, isotonic={brier_iso[cls]:.4f}"
        )
    print("\nPer-class classification error (validation):")
    for cls in range(3):
        print(
            f"  Class {cls}: uncal={err_raw[cls]:.4f}, "
            f"platt={err_platt[cls]:.4f}, isotonic={err_iso[cls]:.4f}"
        )


def plot_baseline_comparison(y_true, y_pred_model, baseline_pred, output_dir, suffix=""):
    """
    Create a separate plot comparing model and baseline performance.
    Also prints detailed statistics.

    Args:
        y_true: True labels
        y_pred_model: Model predictions
        baseline_pred: Baseline predictions
        output_dir: Output directory for plots
        suffix: Suffix for filename
    """
    print(f"\n{'='*60}")
    print("BASELINE COMPARISON STATISTICS")
    print("=" * 60)

    # Calculate overall metrics
    model_acc = accuracy_score(y_true, y_pred_model)
    baseline_acc = accuracy_score(y_true, baseline_pred)

    model_precision = precision_score(y_true, y_pred_model, average="weighted", zero_division=0)
    baseline_precision = precision_score(y_true, baseline_pred, average="weighted", zero_division=0)

    model_recall = recall_score(y_true, y_pred_model, average="weighted", zero_division=0)
    baseline_recall = recall_score(y_true, baseline_pred, average="weighted", zero_division=0)

    model_f1 = f1_score(y_true, y_pred_model, average="weighted", zero_division=0)
    baseline_f1 = f1_score(y_true, baseline_pred, average="weighted", zero_division=0)

    # Calculate macro-averaged metrics
    model_precision_macro = precision_score(y_true, y_pred_model, average="macro", zero_division=0)
    baseline_precision_macro = precision_score(
        y_true, baseline_pred, average="macro", zero_division=0
    )

    model_recall_macro = recall_score(y_true, y_pred_model, average="macro", zero_division=0)
    baseline_recall_macro = recall_score(y_true, baseline_pred, average="macro", zero_division=0)

    model_f1_macro = f1_score(y_true, y_pred_model, average="macro", zero_division=0)
    baseline_f1_macro = f1_score(y_true, baseline_pred, average="macro", zero_division=0)

    model_acc_macro = balanced_accuracy_score(y_true, y_pred_model)
    baseline_acc_macro = balanced_accuracy_score(y_true, baseline_pred)

    # Print overall statistics
    print("\nOverall Metrics (Weighted):")
    print(f"{'Metric':<20} {'Model':<15} {'Baseline':<15} {'Difference':<15}")
    print("-" * 65)
    print(
        f"{'Accuracy':<20} {model_acc:<15.4f} {baseline_acc:<15.4f} {model_acc-baseline_acc:+<15.4f}"
    )
    print(
        f"{'Precision (weighted)':<20} {model_precision:<15.4f} {baseline_precision:<15.4f} {model_precision-baseline_precision:+<15.4f}"
    )
    print(
        f"{'Recall (weighted)':<20} {model_recall:<15.4f} {baseline_recall:<15.4f} {model_recall-baseline_recall:+<15.4f}"
    )
    print(
        f"{'F1-score (weighted)':<20} {model_f1:<15.4f} {baseline_f1:<15.4f} {model_f1-baseline_f1:+<15.4f}"
    )

    # Print macro-averaged statistics
    print("\nOverall Metrics (Macro):")
    print(f"{'Metric':<20} {'Model':<15} {'Baseline':<15} {'Difference':<15}")
    print("-" * 65)
    print(
        f"{'Accuracy (macro)':<20} {model_acc_macro:<15.4f} {baseline_acc_macro:<15.4f} {model_acc_macro-baseline_acc_macro:+<15.4f}"
    )
    print(
        f"{'Precision (macro)':<20} {model_precision_macro:<15.4f} {baseline_precision_macro:<15.4f} {model_precision_macro-baseline_precision_macro:+<15.4f}"
    )
    print(
        f"{'Recall (macro)':<20} {model_recall_macro:<15.4f} {baseline_recall_macro:<15.4f} {model_recall_macro-baseline_recall_macro:+<15.4f}"
    )
    print(
        f"{'F1-score (macro)':<20} {model_f1_macro:<15.4f} {baseline_f1_macro:<15.4f} {model_f1_macro-baseline_f1_macro:+<15.4f}"
    )

    # Calculate per-class metrics
    print("\nPer-Class Precision:")
    print(f"{'Class':<10} {'Model':<15} {'Baseline':<15} {'Difference':<15}")
    print("-" * 55)

    model_precision_per_class = []
    baseline_precision_per_class = []
    class_names = [get_class_label(i) for i in range(3)]

    for cls in range(3):
        # Model precision for this class
        model_pred_cls_mask = y_pred_model == cls
        if model_pred_cls_mask.sum() > 0:
            model_tp = ((y_pred_model == cls) & (y_true == cls)).sum()
            model_prec = model_tp / model_pred_cls_mask.sum()
            model_precision_per_class.append(model_prec)
        else:
            model_precision_per_class.append(0.0)

        # Baseline precision for this class
        baseline_pred_cls_mask = baseline_pred == cls
        if baseline_pred_cls_mask.sum() > 0:
            baseline_tp = ((baseline_pred == cls) & (y_true == cls)).sum()
            baseline_prec = baseline_tp / baseline_pred_cls_mask.sum()
            baseline_precision_per_class.append(baseline_prec)
        else:
            baseline_precision_per_class.append(0.0)

        print(
            f"{class_names[cls]:<10} {model_precision_per_class[cls]:<15.4f} {baseline_precision_per_class[cls]:<15.4f} {model_precision_per_class[cls]-baseline_precision_per_class[cls]:+<15.4f}"
        )

    # Calculate per-class recall
    print("\nPer-Class Recall:")
    print(f"{'Class':<10} {'Model':<15} {'Baseline':<15} {'Difference':<15}")
    print("-" * 55)

    model_recall_per_class = []
    baseline_recall_per_class = []

    for cls in range(3):
        # Model recall for this class
        cls_mask = y_true == cls
        if cls_mask.sum() > 0:
            model_tp = ((y_pred_model == cls) & (y_true == cls)).sum()
            model_rec = model_tp / cls_mask.sum()
            model_recall_per_class.append(model_rec)
        else:
            model_recall_per_class.append(0.0)

        # Baseline recall for this class
        if cls_mask.sum() > 0:
            baseline_tp = ((baseline_pred == cls) & (y_true == cls)).sum()
            baseline_rec = baseline_tp / cls_mask.sum()
            baseline_recall_per_class.append(baseline_rec)
        else:
            baseline_recall_per_class.append(0.0)

        print(
            f"{class_names[cls]:<10} {model_recall_per_class[cls]:<15.4f} {baseline_recall_per_class[cls]:<15.4f} {model_recall_per_class[cls]-baseline_recall_per_class[cls]:+<15.4f}"
        )

    # Calculate per-class F1
    print("\nPer-Class F1-score:")
    print(f"{'Class':<10} {'Model':<15} {'Baseline':<15} {'Difference':<15}")
    print("-" * 55)

    model_f1_per_class = []
    baseline_f1_per_class = []

    for cls in range(3):
        # Model F1 for this class
        if model_precision_per_class[cls] + model_recall_per_class[cls] > 0:
            model_f1_cls = (
                2
                * (model_precision_per_class[cls] * model_recall_per_class[cls])
                / (model_precision_per_class[cls] + model_recall_per_class[cls])
            )
            model_f1_per_class.append(model_f1_cls)
        else:
            model_f1_per_class.append(0.0)

        # Baseline F1 for this class
        if baseline_precision_per_class[cls] + baseline_recall_per_class[cls] > 0:
            baseline_f1_cls = (
                2
                * (baseline_precision_per_class[cls] * baseline_recall_per_class[cls])
                / (baseline_precision_per_class[cls] + baseline_recall_per_class[cls])
            )
            baseline_f1_per_class.append(baseline_f1_cls)
        else:
            baseline_f1_per_class.append(0.0)

        print(
            f"{class_names[cls]:<10} {model_f1_per_class[cls]:<15.4f} {baseline_f1_per_class[cls]:<15.4f} {model_f1_per_class[cls]-baseline_f1_per_class[cls]:+<15.4f}"
        )

    # Calculate per-class accuracy
    print("\nPer-Class Accuracy:")
    print(f"{'Class':<10} {'Model':<15} {'Baseline':<15} {'Difference':<15}")
    print("-" * 55)

    model_acc_per_class = []
    baseline_acc_per_class = []

    for cls in range(3):
        # Model accuracy for this class (correct predictions / total samples of this class)
        cls_mask = y_true == cls
        if cls_mask.sum() > 0:
            model_acc_cls = (y_pred_model[cls_mask] == y_true[cls_mask]).mean()
            model_acc_per_class.append(model_acc_cls)
        else:
            model_acc_per_class.append(0.0)

        # Baseline accuracy for this class
        if cls_mask.sum() > 0:
            baseline_acc_cls = (baseline_pred[cls_mask] == y_true[cls_mask]).mean()
            baseline_acc_per_class.append(baseline_acc_cls)
        else:
            baseline_acc_per_class.append(0.0)

        print(
            f"{class_names[cls]:<10} {model_acc_per_class[cls]:<15.4f} {baseline_acc_per_class[cls]:<15.4f} {model_acc_per_class[cls]-baseline_acc_per_class[cls]:+<15.4f}"
        )

    print("=" * 60)

    # Create confusion matrices
    cm_model = confusion_matrix(y_true, y_pred_model, labels=[0, 1, 2])
    cm_baseline = confusion_matrix(y_true, baseline_pred, labels=[0, 1, 2])

    # Create plot with confusion matrices and metrics
    print(f"\nGenerating baseline comparison plot{' (' + suffix + ')' if suffix else ''}...")

    fig = plt.figure(figsize=(TWO_COL_WIDTH_INCH * 1.5, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Top row: Confusion matrices
    ax_cm_model = fig.add_subplot(gs[0, 0])
    ax_cm_baseline = fig.add_subplot(gs[0, 1])
    ax_cm_diff = fig.add_subplot(gs[0, 2])

    # Bottom row: Metrics
    ax_prec = fig.add_subplot(gs[1, 0])
    ax_rec = fig.add_subplot(gs[1, 1])
    ax_f1 = fig.add_subplot(gs[1, 2])

    axes_metrics = [ax_prec, ax_rec, ax_f1]

    # Confusion Matrix: Model
    vmax_model = max(2500, cm_model.max() * 1.1)
    disp_model = ConfusionMatrixDisplay(confusion_matrix=cm_model, display_labels=class_names)
    im_model = disp_model.plot(
        cmap=CONFUSION_MATRIX_CONFIG["cmap"],
        ax=ax_cm_model,
        values_format=CONFUSION_MATRIX_CONFIG["values_format"],
        im_kw={"vmin": 0, "vmax": vmax_model},
        text_kw={"fontsize": 8},
    )
    ax_cm_model.set_title("Model Confusion Matrix", fontsize=10, fontweight="bold")
    ax_cm_model.set_xlabel(ax_cm_model.get_xlabel(), fontsize=9)
    ax_cm_model.set_ylabel(ax_cm_model.get_ylabel(), fontsize=9)
    ax_cm_model.tick_params(labelsize=7)
    ax_cm_model.grid(False)
    if im_model.im_ is not None:
        cbar_model = im_model.im_.colorbar
        if cbar_model is not None:
            cbar_model.ax.tick_params(labelsize=7)

    # Confusion Matrix: Baseline
    vmax_baseline = max(2500, cm_baseline.max() * 1.1)
    disp_baseline = ConfusionMatrixDisplay(confusion_matrix=cm_baseline, display_labels=class_names)
    im_baseline = disp_baseline.plot(
        cmap=CONFUSION_MATRIX_CONFIG["cmap"],
        ax=ax_cm_baseline,
        values_format=CONFUSION_MATRIX_CONFIG["values_format"],
        im_kw={"vmin": 0, "vmax": vmax_baseline},
        text_kw={"fontsize": 8},
    )
    ax_cm_baseline.set_title("Baseline Confusion Matrix", fontsize=10, fontweight="bold")
    ax_cm_baseline.set_xlabel(ax_cm_baseline.get_xlabel(), fontsize=9)
    ax_cm_baseline.set_ylabel(ax_cm_baseline.get_ylabel(), fontsize=9)
    ax_cm_baseline.tick_params(labelsize=7)
    ax_cm_baseline.grid(False)
    if im_baseline.im_ is not None:
        cbar_baseline = im_baseline.im_.colorbar
        if cbar_baseline is not None:
            cbar_baseline.ax.tick_params(labelsize=7)

    # Confusion Matrix: Difference (Model - Baseline)
    cm_diff = cm_model.astype(float) - cm_baseline.astype(float)
    vmax_diff = max(abs(cm_diff.min()), abs(cm_diff.max())) * 1.1
    disp_diff = ConfusionMatrixDisplay(confusion_matrix=cm_diff, display_labels=class_names)
    im_diff = disp_diff.plot(
        cmap="RdBu_r",
        ax=ax_cm_diff,
        values_format=".0f",
        im_kw={"vmin": -vmax_diff, "vmax": vmax_diff},
        text_kw={"fontsize": 8},
    )
    ax_cm_diff.set_title("Difference (Model - Baseline)", fontsize=10, fontweight="bold")
    ax_cm_diff.set_xlabel(ax_cm_diff.get_xlabel(), fontsize=9)
    ax_cm_diff.set_ylabel(ax_cm_diff.get_ylabel(), fontsize=9)
    ax_cm_diff.tick_params(labelsize=7)
    ax_cm_diff.grid(False)
    if im_diff.im_ is not None:
        cbar_diff = im_diff.im_.colorbar
        if cbar_diff is not None:
            cbar_diff.ax.tick_params(labelsize=7)

    x = np.arange(len(class_names))
    width = 0.35

    # Plot 1: Precision
    ax1 = axes_metrics[0]
    bars1 = ax1.bar(
        x - width / 2,
        model_precision_per_class,
        width,
        label="Model",
        color=COMPARISON_COLORS.get("correct", "#009688"),
        alpha=0.7,
    )
    bars2 = ax1.bar(
        x + width / 2,
        baseline_precision_per_class,
        width,
        label="Naive Baseline",
        color=COMPARISON_COLORS.get("incorrect", "#e91e63"),
        alpha=0.7,
    )
    ax1.set_xlabel("Class", fontsize=10)
    ax1.set_ylabel("Precision", fontsize=10)
    ax1.set_title("Precision per Class", fontsize=11, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, fontsize=9)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0, 1)
    ax1.grid(False, axis="y")
    ax1.tick_params(labelsize=8)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Plot 2: Recall
    ax2 = axes_metrics[1]
    bars3 = ax2.bar(
        x - width / 2,
        model_recall_per_class,
        width,
        label="Model",
        color=COMPARISON_COLORS.get("correct", "#009688"),
        alpha=0.7,
    )
    bars4 = ax2.bar(
        x + width / 2,
        baseline_recall_per_class,
        width,
        label="Naive Baseline",
        color=COMPARISON_COLORS.get("incorrect", "#e91e63"),
        alpha=0.7,
    )
    ax2.set_xlabel("Class", fontsize=10)
    ax2.set_ylabel("Recall", fontsize=10)
    ax2.set_title("Recall per Class", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, fontsize=9)
    ax2.legend(fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.grid(False, axis="y")
    ax2.tick_params(labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Add value labels
    for bars in [bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # Plot 3: F1-score
    ax3 = axes_metrics[2]
    bars5 = ax3.bar(
        x - width / 2,
        model_f1_per_class,
        width,
        label="Model",
        color=COMPARISON_COLORS.get("correct", "#009688"),
        alpha=0.7,
    )
    bars6 = ax3.bar(
        x + width / 2,
        baseline_f1_per_class,
        width,
        label="Naive Baseline",
        color=COMPARISON_COLORS.get("incorrect", "#e91e63"),
        alpha=0.7,
    )
    ax3.set_xlabel("Class", fontsize=10)
    ax3.set_ylabel("F1-score", fontsize=10)
    ax3.set_title("F1-score per Class", fontsize=11, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(class_names, fontsize=9)
    ax3.legend(fontsize=9)
    ax3.set_ylim(0, 1)
    ax3.grid(False, axis="y")
    ax3.tick_params(labelsize=8)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # Add value labels
    for bars in [bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()

    # Save as PNG
    filename_png = (
        f'baseline_comparison{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.png'
    )
    path_png = os.path.join(output_dir, filename_png)
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved PNG to {path_png}")

    # Save as PDF
    filename_pdf = (
        f'baseline_comparison{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.pdf'
    )
    path_pdf = os.path.join(output_dir, filename_pdf)
    plt.savefig(path_pdf, bbox_inches="tight")
    print(f"  ✓ Saved PDF to {path_pdf}")

    plt.close()

    # Create separate figure for accuracy comparison
    fig2, ax_acc = plt.subplots(1, 1, figsize=(ONE_COL_WIDTH_INCH, 3))

    bars_acc1 = ax_acc.bar(
        x - width / 2,
        model_acc_per_class,
        width,
        label="Model",
        color=COMPARISON_COLORS.get("correct", "#009688"),
        alpha=0.7,
    )
    bars_acc2 = ax_acc.bar(
        x + width / 2,
        baseline_acc_per_class,
        width,
        label="Naive Baseline",
        color=COMPARISON_COLORS.get("incorrect", "#e91e63"),
        alpha=0.7,
    )
    ax_acc.set_xlabel("Class", fontsize=10)
    ax_acc.set_ylabel("Accuracy", fontsize=10)
    ax_acc.set_title("Accuracy per Class", fontsize=11, fontweight="bold")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(class_names, fontsize=9)
    ax_acc.legend(fontsize=9)
    ax_acc.set_ylim(0, 1)
    ax_acc.grid(False, axis="y")
    ax_acc.tick_params(labelsize=8)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)

    # Add value labels
    for bars in [bars_acc1, bars_acc2]:
        for bar in bars:
            height = bar.get_height()
            ax_acc.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()

    # Save accuracy plot
    filename_acc_png = f'baseline_comparison_accuracy{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.png'
    path_acc_png = os.path.join(output_dir, filename_acc_png)
    plt.savefig(path_acc_png, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved accuracy plot PNG to {path_acc_png}")

    filename_acc_pdf = f'baseline_comparison_accuracy{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.pdf'
    path_acc_pdf = os.path.join(output_dir, filename_acc_pdf)
    plt.savefig(path_acc_pdf, bbox_inches="tight")
    print(f"  ✓ Saved accuracy plot PDF to {path_acc_pdf}")

    plt.close(fig2)


def plot_error_analysis_by_class(y_true, y_pred, confidence_scores, output_dir, suffix=""):
    """Analyze errors by true class."""
    print(f"Generating error analysis{' (' + suffix + ')' if suffix else ''}...")

    # Original was (15, 5), ratio 3:1. With width 7, use height 3 for better visibility
    fig, axes = plt.subplots(1, 3, figsize=(TWO_COL_WIDTH_INCH, 3))

    class_names = [get_class_label(i, style="long") for i in range(3)]

    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        mask = y_true == i
        correct = y_true[mask] == y_pred[mask]
        conf = confidence_scores[mask]

        # Plot confidence for correct vs incorrect
        correct_conf = conf[correct]
        incorrect_conf = conf[~correct]

        ax.hist(
            correct_conf,
            bins=ERROR_ANALYSIS_CONFIG["bins"],
            alpha=ERROR_ANALYSIS_CONFIG["alpha"],
            label=f"Correct ({len(correct_conf)})",
            color=COMPARISON_COLORS["correct"],
            edgecolor=ERROR_ANALYSIS_CONFIG["edgecolor"],
        )
        ax.hist(
            incorrect_conf,
            bins=ERROR_ANALYSIS_CONFIG["bins"],
            alpha=ERROR_ANALYSIS_CONFIG["alpha"],
            label=f"Incorrect ({len(incorrect_conf)})",
            color=COMPARISON_COLORS["incorrect"],
            edgecolor=ERROR_ANALYSIS_CONFIG["edgecolor"],
        )

        ax.set_xlabel("Confidence Score", fontsize=8)
        ax.set_ylabel("Count", fontsize=8)
        ax.set_title(class_name, fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=6)
        ax.grid(False)
        # Remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()

    # Save as PNG
    filename_png = (
        f'error_analysis_by_class{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.png'
    )
    path_png = os.path.join(output_dir, filename_png)
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved PNG to {path_png}")

    # Save as PDF
    filename_pdf = (
        f'error_analysis_by_class{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.pdf'
    )
    path_pdf = os.path.join(output_dir, filename_pdf)
    plt.savefig(path_pdf, bbox_inches="tight")
    print(f"  ✓ Saved PDF to {path_pdf}")

    plt.close()


def plot_confidence_vs_r1r2(df, predictions, confidence_scores, output_dir, suffix=""):
    """Plot confidence vs r1r2 value."""
    print(f"Generating confidence vs r1r2 plot{' (' + suffix + ')' if suffix else ''}...")

    # Create plot data
    plot_df = pd.DataFrame(
        {"r1r2": df["r1r2"], "confidence": confidence_scores, "predicted_class": predictions}
    )

    # Filter extreme values for better visualization
    plot_df = plot_df[(plot_df["r1r2"] > 0.01) & (plot_df["r1r2"] < 100)]

    plt.figure(figsize=(TWO_COL_WIDTH_INCH, 3))

    # Scatter plot
    for cls in [0, 1, 2]:
        mask = plot_df["predicted_class"] == cls
        plt.scatter(
            plot_df.loc[mask, "r1r2"],
            plot_df.loc[mask, "confidence"],
            alpha=0.5,
            s=20,
            c=get_class_color(cls),
            label=get_class_label(cls, style="short"),
            edgecolors="none",
        )

    # Class boundaries (r1*r2 product)
    plt.axvline(1, color="gray", linestyle="--", linewidth=1.5, label="Class boundaries")
    plt.axvline(25, color="gray", linestyle="--", linewidth=1.5)

    # Moving average
    plot_df_sorted = plot_df.sort_values("r1r2")
    window_size = max(50, len(plot_df) // 50)
    rolling_mean = plot_df_sorted["confidence"].rolling(window=window_size, center=True).mean()
    plt.plot(
        plot_df_sorted["r1r2"],
        rolling_mean,
        "r-",
        linewidth=2,
        label=f"Rolling mean (n={window_size})",
    )

    plt.xlabel("r1×r2", fontsize=12)
    plt.ylabel("Confidence Score", fontsize=12)
    title = "Prediction Confidence vs r-Product" + (" - " + suffix if suffix else "")
    plt.title(title, fontsize=14)
    plt.xlim(0, 50)
    plt.ylim(0, 1.05)
    plt.legend(loc="best")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    filename = f'confidence_vs_r1r2{("_" + suffix.lower().replace(" ", "_")) if suffix else ""}.png'
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved to {path}")


def print_classification_report(y_true, y_pred):
    """Print classification metrics."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    # Calculate macro accuracy (balanced accuracy = mean of per-class recalls)
    macro_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Macro Accuracy: {macro_acc:.3f}\n")

    report = classification_report(
        y_true, y_pred, target_names=[get_class_label(i, style="long") for i in range(3)], digits=3
    )
    print(report)


def find_optimal_threshold_per_class(y_true, y_pred, confidence_scores, min_retention=0.7):
    """
    Find optimal confidence threshold for each class.

    Keeps at least min_retention (70%) of predictions per class,
    but removes incorrect predictions with high confidence if they
    outnumber correct ones at that confidence level.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidence_scores: Confidence scores
        min_retention: Minimum fraction of predictions to keep per class

    Returns:
        Dictionary with threshold per class and filtered indices
    """
    thresholds = {}
    filtered_indices = []

    for cls in [0, 1, 2]:
        # Get predictions for this class
        class_mask = y_pred == cls
        class_indices = np.where(class_mask)[0]

        if len(class_indices) == 0:
            continue

        # Sort by confidence (descending)
        class_conf = confidence_scores[class_indices]
        class_true = y_true[class_indices]
        class_pred = y_pred[class_indices]

        sorted_idx = np.argsort(class_conf)[::-1]
        sorted_conf = class_conf[sorted_idx]
        sorted_true = class_true[sorted_idx]
        sorted_pred = class_pred[sorted_idx]

        # Calculate number to keep (at least min_retention)
        min_keep = int(len(class_indices) * min_retention)

        # Find optimal threshold
        best_threshold = 0.0
        best_idx = len(class_indices)

        for i in range(min_keep, len(class_indices)):
            # Count correct and incorrect up to this point
            correct = (sorted_true[:i] == sorted_pred[:i]).sum()
            incorrect = i - correct

            # If incorrect > correct at this confidence level, cut here
            if incorrect > correct and i >= min_keep:
                best_threshold = sorted_conf[i - 1]
                best_idx = i
                break
        else:
            # Keep all if no good cutoff found
            best_threshold = sorted_conf[-1] if len(sorted_conf) > 0 else 0.0
            best_idx = len(class_indices)

        thresholds[cls] = best_threshold

        # Add indices to keep
        keep_mask = class_conf >= best_threshold
        filtered_indices.extend(class_indices[keep_mask])

    return thresholds, np.array(filtered_indices)


def _sweep_confidence_thresholds(
    y_true, y_pred, confidence_scores, n_points: int = 101
) -> pd.DataFrame:
    """
    Sweep a *global* confidence threshold t in [0,1] and compute:
      - coverage: fraction retained
      - balanced_accuracy (macro acc) on retained
      - per-class accuracy on retained (equals recall per class on retained set)
      - retained counts per class (by true label)
    """
    from sklearn.metrics import balanced_accuracy_score

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    conf = np.asarray(confidence_scores).astype(float)

    thresholds = np.linspace(0.0, 1.0, max(2, int(n_points)))
    rows = []
    n_total = len(y_true)

    for t in thresholds:
        keep = conf >= t
        n_keep = int(keep.sum())
        if n_keep == 0:
            rows.append(
                {
                    "threshold": float(t),
                    "coverage": 0.0,
                    "n_retained": 0,
                    "balanced_accuracy": np.nan,
                    "acc_class_0": np.nan,
                    "acc_class_1": np.nan,
                    "acc_class_2": np.nan,
                    "n_true_0": 0,
                    "n_true_1": 0,
                    "n_true_2": 0,
                }
            )
            continue

        yt = y_true[keep]
        yp = y_pred[keep]
        bal_acc = float(balanced_accuracy_score(yt, yp))

        per_cls_acc = {}
        per_cls_n = {}
        for cls in [0, 1, 2]:
            m = yt == cls
            per_cls_n[cls] = int(m.sum())
            per_cls_acc[cls] = float(np.mean(yp[m] == yt[m])) if m.sum() else np.nan

        rows.append(
            {
                "threshold": float(t),
                "coverage": float(n_keep / n_total) if n_total else 0.0,
                "n_retained": int(n_keep),
                "balanced_accuracy": bal_acc,
                "acc_class_0": per_cls_acc[0],
                "acc_class_1": per_cls_acc[1],
                "acc_class_2": per_cls_acc[2],
                "n_true_0": per_cls_n[0],
                "n_true_1": per_cls_n[1],
                "n_true_2": per_cls_n[2],
            }
        )

    return pd.DataFrame(rows)


def _plot_threshold_sweep_metrics(df_sweep: pd.DataFrame, output_dir: str):
    """
    Plot macro/balanced accuracy and per-class accuracy vs global threshold.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH))
    ax.plot(
        df_sweep["threshold"],
        df_sweep["balanced_accuracy"],
        label="Macro acc (balanced)",
        color="#000000",
        linewidth=1.8,
    )
    ax.plot(
        df_sweep["threshold"],
        df_sweep["acc_class_0"],
        label=get_class_label(0, style="short"),
        color=get_class_color(0),
        linewidth=1.4,
    )
    ax.plot(
        df_sweep["threshold"],
        df_sweep["acc_class_1"],
        label=get_class_label(1, style="short"),
        color=get_class_color(1),
        linewidth=1.4,
    )
    ax.plot(
        df_sweep["threshold"],
        df_sweep["acc_class_2"],
        label=get_class_label(2, style="short"),
        color=get_class_color(2),
        linewidth=1.4,
    )

    ax.set_xlabel("Confidence threshold (keep if conf ≥ t)", fontsize=8)
    ax.set_ylabel("Accuracy on retained subset", fontsize=8)
    ax.set_title("Accuracy vs confidence threshold", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"confidence_threshold_sweep_metrics.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(
        f"  ✓ Saved threshold sweep metrics plot to {os.path.join(output_dir, 'confidence_threshold_sweep_metrics.png')}"
    )


def _plot_threshold_sweep_coverage(df_sweep: pd.DataFrame, output_dir: str):
    """
    Plot coverage (retained fraction) vs global threshold.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH))
    ax.plot(df_sweep["threshold"], df_sweep["coverage"], color="#143D60", linewidth=1.8)

    ax.set_xlabel("Confidence threshold (keep if conf ≥ t)", fontsize=8)
    ax.set_ylabel("Coverage (retained fraction)", fontsize=8)
    ax.set_title("Coverage vs confidence threshold", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"confidence_threshold_sweep_coverage.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(
        f"  ✓ Saved threshold sweep coverage plot to {os.path.join(output_dir, 'confidence_threshold_sweep_coverage.png')}"
    )


def _plot_threshold_sweep_combined(df_sweep: pd.DataFrame, output_dir: str):
    """
    Single figure with two subplots:
      - Left: coverage vs threshold
      - Right: macro & per-class accuracy vs threshold
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax_cov, ax_met) = plt.subplots(
        1, 2, figsize=(TWO_COL_WIDTH_INCH, ONE_COL_GOLDEN_RATIO_HEIGHT_INCH)
    )

    # Left: coverage
    ax_cov.plot(df_sweep["threshold"], df_sweep["coverage"], color="#143D60", linewidth=1.8)
    ax_cov.set_xlabel("Confidence threshold (keep if conf ≥ t)", fontsize=8)
    ax_cov.set_ylabel("Coverage", fontsize=8)
    ax_cov.set_title("Coverage", fontsize=9)
    ax_cov.set_xlim(0, 1)
    ax_cov.set_ylim(0, 1)
    ax_cov.tick_params(labelsize=7)
    ax_cov.spines["top"].set_visible(False)
    ax_cov.spines["right"].set_visible(False)
    ax_cov.grid(True, axis="y", alpha=0.25, linestyle="--")

    # Right: accuracies
    ax_met.plot(
        df_sweep["threshold"],
        df_sweep["balanced_accuracy"],
        label="Macro acc (balanced)",
        color="#000000",
        linewidth=1.8,
    )
    ax_met.plot(
        df_sweep["threshold"],
        df_sweep["acc_class_0"],
        label=get_class_label(0, style="short"),
        color=get_class_color(0),
        linewidth=1.4,
    )
    ax_met.plot(
        df_sweep["threshold"],
        df_sweep["acc_class_1"],
        label=get_class_label(1, style="short"),
        color=get_class_color(1),
        linewidth=1.4,
    )
    ax_met.plot(
        df_sweep["threshold"],
        df_sweep["acc_class_2"],
        label=get_class_label(2, style="short"),
        color=get_class_color(2),
        linewidth=1.4,
    )
    ax_met.set_xlabel("Confidence threshold (keep if conf ≥ t)", fontsize=8)
    ax_met.set_ylabel("Accuracy on retained subset", fontsize=8)
    ax_met.set_title("Accuracy", fontsize=9)
    ax_met.set_xlim(0, 1)
    ax_met.set_ylim(0, 1)
    ax_met.tick_params(labelsize=7)
    ax_met.spines["top"].set_visible(False)
    ax_met.spines["right"].set_visible(False)
    ax_met.grid(True, axis="y", alpha=0.25, linestyle="--")
    ax_met.legend(frameon=False, fontsize=7, ncol=1, loc="lower left")

    plt.tight_layout()

    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"confidence_threshold_sweep_combined.{ext}")
        plt.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(
        f"  ✓ Saved threshold sweep combined plot to {os.path.join(output_dir, 'confidence_threshold_sweep_combined.png')}"
    )


def analyze_confidence_filtering(y_true, y_pred, confidence_scores, output_dir, min_retention=0.7):
    """
    Perform dynamic confidence filtering analysis.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        confidence_scores: Confidence scores
        output_dir: Output directory
        min_retention: Minimum retention rate per class
    """
    print("Generating confidence filtering analysis...")

    # -------------------------
    # Global threshold sweep (diagnostic)
    # -------------------------
    try:
        sweep_df = _sweep_confidence_thresholds(y_true, y_pred, confidence_scores)
        _plot_threshold_sweep_combined(sweep_df, output_dir)
        sweep_csv = os.path.join(output_dir, "confidence_threshold_sweep.csv")
        sweep_df.to_csv(sweep_csv, index=False)
        print(f"  ✓ Saved threshold sweep CSV to {sweep_csv}")
    except Exception as e:
        print(f"  ⚠ Threshold sweep failed: {e}")

    # Find optimal thresholds
    thresholds, filtered_indices = find_optimal_threshold_per_class(
        y_true, y_pred, confidence_scores, min_retention
    )

    # Filter data
    y_true_filtered = y_true[filtered_indices]
    y_pred_filtered = y_pred[filtered_indices]
    conf_filtered = confidence_scores[filtered_indices]

    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    # Original metrics
    orig_acc = accuracy_score(y_true, y_pred)
    orig_f1 = f1_score(y_true, y_pred, average="weighted")

    # Filtered metrics
    filt_acc = accuracy_score(y_true_filtered, y_pred_filtered)
    filt_f1 = f1_score(y_true_filtered, y_pred_filtered, average="weighted")

    # Per-class statistics
    class_stats = []
    for cls in [0, 1, 2]:
        orig_mask = y_pred == cls
        filt_mask = y_pred_filtered == cls

        orig_count = orig_mask.sum()
        filt_count = filt_mask.sum()
        retention = filt_count / orig_count if orig_count > 0 else 0

        orig_acc_cls = accuracy_score(y_true[orig_mask], y_pred[orig_mask]) if orig_count > 0 else 0
        filt_acc_cls = (
            accuracy_score(y_true_filtered[filt_mask], y_pred_filtered[filt_mask])
            if filt_count > 0
            else 0
        )

        class_stats.append(
            {
                "class": cls,
                "threshold": thresholds.get(cls, 0.0),
                "original_count": orig_count,
                "filtered_count": filt_count,
                "retention_rate": retention,
                "original_accuracy": orig_acc_cls,
                "filtered_accuracy": filt_acc_cls,
                "accuracy_gain": filt_acc_cls - orig_acc_cls,
            }
        )

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(TWO_COL_WIDTH_INCH, 6))

    # 1. Threshold and retention per class
    ax1 = axes[0, 0]
    class_names = [get_class_label(i) for i in range(3)]
    thresholds_list = [s["threshold"] for s in class_stats]
    retention_list = [s["retention_rate"] for s in class_stats]

    x = np.arange(len(class_names))
    width = 0.35

    ax1_twin = ax1.twinx()
    color1 = SEQUENTIAL_COLORS[0]
    color2 = SEQUENTIAL_COLORS[1]
    bars1 = ax1.bar(
        x - width / 2, thresholds_list, width, label="Threshold", color=color1, alpha=0.7
    )
    bars2 = ax1_twin.bar(
        x + width / 2, retention_list, width, label="Retention", color=color2, alpha=0.7
    )

    ax1.set_xlabel("Class")
    ax1.set_ylabel("Confidence Threshold", color=color1)
    ax1_twin.set_ylabel("Retention Rate", color=color2)
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names)
    ax1.set_title("Threshold and Retention per Class")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1_twin.tick_params(axis="y", labelcolor=color2)
    ax1.grid(alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, thresholds_list)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{val:.3f}",
            ha="center",
            va="center",
            fontsize=9,
        )
    for i, (bar, val) in enumerate(zip(bars2, retention_list)):
        ax1_twin.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() / 2,
            f"{val:.1%}",
            ha="center",
            va="center",
            fontsize=9,
        )

    # 2. Accuracy comparison
    ax2 = axes[0, 1]
    orig_acc_list = [s["original_accuracy"] for s in class_stats]
    filt_acc_list = [s["filtered_accuracy"] for s in class_stats]

    x = np.arange(len(class_names))
    width = 0.35

    ax2.bar(
        x - width / 2,
        orig_acc_list,
        width,
        label="Original",
        color=COMPARISON_COLORS["original"],
        alpha=0.7,
    )
    ax2.bar(
        x + width / 2,
        filt_acc_list,
        width,
        label="Filtered",
        color=COMPARISON_COLORS["filtered"],
        alpha=0.7,
    )

    ax2.set_xlabel("Class")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy: Original vs Filtered")
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names)
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1)

    # Add value labels
    for i, v in enumerate(orig_acc_list):
        ax2.text(i - width / 2, v + 0.02, f"{v:.2%}", ha="center", fontsize=9)
    for i, v in enumerate(filt_acc_list):
        ax2.text(i + width / 2, v + 0.02, f"{v:.2%}", ha="center", fontsize=9)

    # 3. Overall metrics comparison
    ax3 = axes[1, 0]
    metrics = ["Accuracy", "F1 (weighted)"]
    orig_metrics = [orig_acc, orig_f1]
    filt_metrics = [filt_acc, filt_f1]

    x = np.arange(len(metrics))
    width = 0.35

    ax3.bar(
        x - width / 2,
        orig_metrics,
        width,
        label="Original",
        color=COMPARISON_COLORS["original"],
        alpha=0.7,
    )
    ax3.bar(
        x + width / 2,
        filt_metrics,
        width,
        label="Filtered",
        color=COMPARISON_COLORS["filtered"],
        alpha=0.7,
    )

    ax3.set_ylabel("Score")
    ax3.set_title("Overall Metrics Comparison")
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_ylim(0, 1)

    # Add value labels
    for i, v in enumerate(orig_metrics):
        ax3.text(i - width / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)
    for i, v in enumerate(filt_metrics):
        ax3.text(i + width / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=10)

    # 4. Sample count comparison
    ax4 = axes[1, 1]
    orig_counts = [s["original_count"] for s in class_stats]
    filt_counts = [s["filtered_count"] for s in class_stats]

    x = np.arange(len(class_names))
    width = 0.35

    ax4.bar(
        x - width / 2,
        orig_counts,
        width,
        label="Original",
        color=COMPARISON_COLORS["original"],
        alpha=0.7,
    )
    ax4.bar(
        x + width / 2,
        filt_counts,
        width,
        label="Filtered",
        color=COMPARISON_COLORS["filtered"],
        alpha=0.7,
    )

    ax4.set_xlabel("Class")
    ax4.set_ylabel("Sample Count")
    ax4.set_title("Sample Count per Class")
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_names)
    ax4.legend()
    ax4.grid(alpha=0.3)

    # Add value labels
    for i, v in enumerate(orig_counts):
        ax4.text(i - width / 2, v + max(orig_counts) * 0.02, f"{v}", ha="center", fontsize=9)
    for i, v in enumerate(filt_counts):
        ax4.text(i + width / 2, v + max(orig_counts) * 0.02, f"{v}", ha="center", fontsize=9)

    plt.tight_layout()

    path = os.path.join(output_dir, "confidence_filtering_analysis.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved to {path}")

    # Save detailed report
    report_path = os.path.join(output_dir, "confidence_filtering_report.txt")
    with open(report_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("CONFIDENCE FILTERING ANALYSIS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Minimum retention rate: {min_retention:.1%}\n")
        f.write(
            f"Total samples: {len(y_true)} → {len(y_true_filtered)} ({len(y_true_filtered)/len(y_true):.1%})\n\n"
        )

        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Metric':<20} {'Original':<15} {'Filtered':<15} {'Gain':<15}\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Accuracy':<20} {orig_acc:<15.4f} {filt_acc:<15.4f} {filt_acc-orig_acc:+<15.4f}\n"
        )
        f.write(
            f"{'F1 (weighted)':<20} {orig_f1:<15.4f} {filt_f1:<15.4f} {filt_f1-orig_f1:+<15.4f}\n\n"
        )

        f.write("PER-CLASS STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(
            f"{'Class':<10} {'Threshold':<12} {'Orig Count':<12} {'Filt Count':<12} {'Retention':<12} {'Orig Acc':<10} {'Filt Acc':<10} {'Gain':<10}\n"
        )
        f.write("-" * 80 + "\n")

        for stats in class_stats:
            f.write(
                f"{stats['class']:<10} "
                f"{stats['threshold']:<12.4f} "
                f"{stats['original_count']:<12d} "
                f"{stats['filtered_count']:<12d} "
                f"{stats['retention_rate']:<12.1%} "
                f"{stats['original_accuracy']:<10.3f} "
                f"{stats['filtered_accuracy']:<10.3f} "
                f"{stats['accuracy_gain']:+<10.3f}\n"
            )

        f.write("\n" + "=" * 80 + "\n")
        f.write("FILTERED CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")

        report = classification_report(
            y_true_filtered,
            y_pred_filtered,
            target_names=[get_class_label(i, style="long") for i in range(3)],
            digits=4,
        )
        f.write(report)

    print(f"  ✓ Saved report to {report_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("CONFIDENCE FILTERING SUMMARY")
    print("=" * 60)
    print(
        f"Retention: {len(y_true_filtered)}/{len(y_true)} ({len(y_true_filtered)/len(y_true):.1%})"
    )
    print(f"Accuracy:  {orig_acc:.4f} → {filt_acc:.4f} (Δ {filt_acc-orig_acc:+.4f})")
    print(f"F1:        {orig_f1:.4f} → {filt_f1:.4f} (Δ {filt_f1-orig_f1:+.4f})")
    print("\nPer-class thresholds:")
    for stats in class_stats:
        print(
            f"  Class {stats['class']}: {stats['threshold']:.4f} "
            f"(retention: {stats['retention_rate']:.1%}, "
            f"acc gain: {stats['accuracy_gain']:+.3f})"
        )
    print("=" * 60)


def calculate_metrics_with_cv(df, predictor, n_folds=5, id_column="reaction_id"):
    """
    Calculate metrics with group-based cross-validation.
    Uses grouped KFold to ensure no data leakage (e.g., normal and flipped
    variants of the same reaction stay together).

    Args:
        df: DataFrame with features and labels (must contain id_column)
        predictor: Trained predictor
        n_folds: Number of CV folds
        id_column: Column name for grouping (default: 'reaction_id')

    Returns:
        Dictionary with mean and std for each metric
    """
    # Storage for CV metrics
    metrics_list = {
        "accuracy": [],
        "accuracy_macro": [],
        "precision_macro": [],
        "recall_macro": [],
        "f1_macro": [],
        "precision_weighted": [],
        "recall_weighted": [],
        "f1_weighted": [],
    }

    # Create group-based folds
    splits = create_grouped_kfold_splits(df, n_splits=n_folds, id_column=id_column)

    # Cross-validation
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        # Get test fold data
        df_test_fold = df.iloc[test_idx]
        X_test_fold = df_test_fold[predictor.features]
        y_test_fold = df_test_fold["r_product_class"].astype(int).values

        # Make predictions
        y_pred_fold = predictor.model.predict(X_test_fold)

        # Calculate metrics
        metrics_list["accuracy"].append(accuracy_score(y_test_fold, y_pred_fold))
        metrics_list["accuracy_macro"].append(balanced_accuracy_score(y_test_fold, y_pred_fold))
        metrics_list["precision_macro"].append(
            precision_score(y_test_fold, y_pred_fold, average="macro", zero_division=0)
        )
        metrics_list["recall_macro"].append(
            recall_score(y_test_fold, y_pred_fold, average="macro", zero_division=0)
        )
        metrics_list["f1_macro"].append(
            f1_score(y_test_fold, y_pred_fold, average="macro", zero_division=0)
        )
        metrics_list["precision_weighted"].append(
            precision_score(y_test_fold, y_pred_fold, average="weighted", zero_division=0)
        )
        metrics_list["recall_weighted"].append(
            recall_score(y_test_fold, y_pred_fold, average="weighted", zero_division=0)
        )
        metrics_list["f1_weighted"].append(
            f1_score(y_test_fold, y_pred_fold, average="weighted", zero_division=0)
        )

    # Calculate mean and std
    results = {}
    for metric_name, values in metrics_list.items():
        results[metric_name] = {
            "mean": np.mean(values),
            "std": np.std(values, ddof=1),  # Sample standard deviation
        }

    return results


def create_latex_performance_table(predictor, output_dir, n_folds=5, confidence_threshold=0.7):
    """
    Create LaTeX performance table with Train, Test (voting), and Filtered Test columns.

    The test column reflects the voting model (only where XGBoost and Lookup agree).
    The filtered-test column additionally applies the confidence threshold.

    Args:
        predictor: Trained predictor
        output_dir: Output directory
        n_folds: Number of CV folds for error bars
        confidence_threshold: Confidence threshold for the filtered column
    """
    print("\n" + "=" * 60)
    print("GENERATING LATEX PERFORMANCE TABLE")
    print("=" * 60)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        split_dir = os.path.join(project_root, "artifacts", "data_splits")

        print("Loading train/test splits (incl. negative data for lookup)...")
        df_train, df_test = load_train_with_negative_data(split_dir=split_dir)
        print(f"  Train (with neg): {len(df_train)} samples")
        print(f"  Test:  {len(df_test)} samples")

        # --- Compute voting on test set ---
        required_smiles_cols = ["monomer1_smiles", "monomer2_smiles", "solvent_smiles"]
        y_train_labels = df_train["r_product_class"].astype(int).values

        print("  Computing lookup predictions for voting (incl. negative data)...")
        lookup_pred, _ = compute_naive_baseline_predictions_with_similarity(
            df_test, df_train, y_train_labels, predictor.features
        )

        X_test = df_test[predictor.features]
        results = predictor.predict_with_confidence(X_test)
        xgb_pred = results["predictions"]
        confidence = results["confidence"]

        models_agree = xgb_pred == lookup_pred
        voting_mask = models_agree
        filtered_mask = models_agree & (confidence >= confidence_threshold)

        df_test_voting = df_test[voting_mask].reset_index(drop=True)
        df_test_filtered = df_test[filtered_mask].reset_index(drop=True)

        print(f"  Test (voting): {len(df_test_voting)} samples")
        print(f"  Test (filtered t={confidence_threshold}): {len(df_test_filtered)} samples")

        # --- Calculate metrics with CV ---
        print(f"Calculating metrics with {n_folds}-fold group-based CV...")
        print("  Train set...")
        train_m = calculate_metrics_with_cv(df_train, predictor, n_folds=n_folds)
        print("  Test set (voting)...")
        test_m = calculate_metrics_with_cv(df_test_voting, predictor, n_folds=n_folds)
        print("  Test set (filtered)...")
        filtered_m = calculate_metrics_with_cv(df_test_filtered, predictor, n_folds=n_folds)

        def fmt(m, key):
            return f"${m[key]['mean']:.2f} \\pm {m[key]['std']:.2f}$"

        # --- Write LaTeX ---
        latex_file = os.path.join(output_dir, "performance_table.tex")
        with open(latex_file, "w") as f:
            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write(
                "\\caption{Overall performance of the voting model (XGBoost + Lookup). "
                "Error bars are standard deviations from "
                f"{n_folds}-fold group-based cross-validation.}}\n"
            )
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\toprule\n")
            f.write(
                "\\textbf{Metric} & \\textbf{Train} & \\textbf{Test} & \\textbf{Filtered Test} \\\\\n"
            )
            f.write("\\midrule\n")
            f.write(
                f"Samples & {len(df_train)} & {len(df_test_voting)} & {len(df_test_filtered)} \\\\[3pt]\n"
            )

            f.write(
                f"Accuracy (macro) & {fmt(train_m, 'accuracy_macro')} & {fmt(test_m, 'accuracy_macro')} & {fmt(filtered_m, 'accuracy_macro')} \\\\\n"
            )
            f.write(
                f"Accuracy (weighted) & {fmt(train_m, 'accuracy')} & {fmt(test_m, 'accuracy')} & {fmt(filtered_m, 'accuracy')} \\\\[3pt]\n"
            )

            f.write(
                f"Precision (macro) & {fmt(train_m, 'precision_macro')} & {fmt(test_m, 'precision_macro')} & {fmt(filtered_m, 'precision_macro')} \\\\\n"
            )
            f.write(
                f"Precision (weighted) & {fmt(train_m, 'precision_weighted')} & {fmt(test_m, 'precision_weighted')} & {fmt(filtered_m, 'precision_weighted')} \\\\\n"
            )

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:model_performance}\n")
            f.write("\\end{table}\n")

        print(f"\n✓ LaTeX table saved to: {latex_file}")
        return latex_file

    except FileNotFoundError as e:
        print(f"✗ Error creating LaTeX table: {e}")
        print("  Note: Train/test splits are required.")
        return None
    except Exception as e:
        print(f"✗ Error creating LaTeX table: {e}")
        import traceback

        traceback.print_exc()
        return None


def create_latex_per_class_table(predictor, output_dir, confidence_threshold=0.7):
    """
    Create per-class LaTeX table with Train, Test (voting), and Filtered Test columns.

    Shows per-class precision plus summary rows for macro/weighted accuracy and precision.
    """
    print("\n" + "=" * 60)
    print("GENERATING PER-CLASS LATEX TABLE")
    print("=" * 60)

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        split_dir = os.path.join(project_root, "artifacts", "data_splits")

        df_train, df_test = load_train_with_negative_data(split_dir=split_dir)
        y_train_labels = df_train["r_product_class"].astype(int).values

        # --- Compute voting on test set ---
        print("  Computing lookup predictions for voting (incl. negative data)...")
        lookup_pred, _ = compute_naive_baseline_predictions_with_similarity(
            df_test, df_train, y_train_labels, predictor.features
        )

        X_test = df_test[predictor.features]
        results_test = predictor.predict_with_confidence(X_test)
        xgb_pred_test = results_test["predictions"]
        confidence_test = results_test["confidence"]

        models_agree = xgb_pred_test == lookup_pred
        filtered_mask = models_agree & (confidence_test >= confidence_threshold)

        # --- Prepare datasets ---
        X_train = df_train[predictor.features]
        y_true_train = df_train["r_product_class"].astype(int).values
        y_pred_train = predictor.model.predict(X_train)

        y_true_test = df_test["r_product_class"].astype(int).values
        y_true_voting = y_true_test[models_agree]
        y_pred_voting = xgb_pred_test[models_agree]
        y_true_filtered = y_true_test[filtered_mask]
        y_pred_filtered = xgb_pred_test[filtered_mask]

        datasets = [
            ("Train", y_true_train, y_pred_train),
            ("Test", y_true_voting, y_pred_voting),
            ("Filtered Test", y_true_filtered, y_pred_filtered),
        ]

        # --- Compute per-class metrics: macro accuracy (= recall) and precision ---
        class_names = {0: "0 (Alternating)", 1: "1 (Random)", 2: "2 (Gradient)"}
        per_class_prec = {}
        per_class_acc = {}
        for name, yt, yp in datasets:
            per_class_prec[name] = precision_score(
                yt, yp, labels=[0, 1, 2], average=None, zero_division=0
            )
            per_class_acc[name] = recall_score(
                yt, yp, labels=[0, 1, 2], average=None, zero_division=0
            )

        def fmt(val):
            return f"${val:.2f}$"

        col_names = ["Train", "Test", "Filtered Test"]

        # --- Write LaTeX ---
        latex_file = os.path.join(output_dir, "per_class_performance_table.tex")
        with open(latex_file, "w") as f:
            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write(
                "\\caption{Per-class accuracy and precision of the voting model "
                "(XGBoost + Lookup).}\n"
            )
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\toprule\n")
            f.write(
                "\\textbf{Class} & \\textbf{Train} & \\textbf{Test} & \\textbf{Filtered Test} \\\\\n"
            )

            # Accuracy per class = fraction of true class X samples correctly predicted
            f.write("\\midrule\n")
            f.write("\\multicolumn{4}{l}{\\textit{Accuracy (per class)}} \\\\\n")
            for cls in [0, 1, 2]:
                vals = [fmt(per_class_acc[name][cls]) for name in col_names]
                f.write(f"\\quad {class_names[cls]} & {vals[0]} & {vals[1]} & {vals[2]} \\\\\n")

            # Precision per class
            f.write("\\midrule\n")
            f.write("\\multicolumn{4}{l}{\\textit{Precision (per class)}} \\\\\n")
            for cls in [0, 1, 2]:
                vals = [fmt(per_class_prec[name][cls]) for name in col_names]
                f.write(f"\\quad {class_names[cls]} & {vals[0]} & {vals[1]} & {vals[2]} \\\\\n")

            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\label{tab:class_performance}\n")
            f.write("\\end{table}\n")

        print(f"\n✓ Per-class LaTeX table saved to: {latex_file}")
        return latex_file

    except Exception as e:
        print(f"✗ Error creating per-class LaTeX table: {e}")
        import traceback

        traceback.print_exc()
        return None


def generate_plots_for_dataset(df, predictor, args, suffix=""):
    """Generate analysis plots for the voting model on the test set.

    The final model is a Voting model (XGBoost + Lookup): only samples
    where both models agree are considered predicted.  All plots are
    generated on this voting-filtered base set.  The combined plot
    additionally shows a confidence-filtered variant (threshold 0.7).
    """
    # Prepare features
    try:
        X = df[predictor.features]

        if "r_product_class" in df.columns:
            y_true = df["r_product_class"].astype(int).values
        else:
            if {"constant_1", "constant_2"}.issubset(df.columns):

                def _class_from_row(row):
                    res = classify_reactivity_curve(
                        float(row["constant_1"]), float(row["constant_2"])
                    )
                    return res["class_id"]

                y_true = df.apply(_class_from_row, axis=1).astype(int).values
            else:
                bins = [-np.inf, 1, 25, np.inf]
                labels = [0, 1, 2]
                y_true = (
                    pd.cut(df["r1r2"], bins=bins, labels=labels, right=False).astype(int).values
                )
    except Exception as e:
        print(f"  ✗ Error preparing features: {e}")
        return

    # Make XGBoost predictions
    try:
        results = predictor.predict_with_confidence(X)
        y_pred = results["predictions"]
        y_proba = results["probabilities"]
        confidence = results["confidence"]
    except Exception as e:
        print(f"  ✗ Error making predictions: {e}")
        return

    # ------- Load training data & compute Lookup predictions for voting -------
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        split_dir = os.path.join(project_root, "artifacts", "data_splits")

        df_train_split, _ = load_train_with_negative_data(split_dir=split_dir)

        required_smiles_cols = ["monomer1_smiles", "monomer2_smiles", "solvent_smiles"]
        if not all(col in df_train_split.columns for col in required_smiles_cols):
            raise ValueError("Training data missing SMILES columns")
        if not all(col in df.columns for col in required_smiles_cols):
            raise ValueError("Test data missing SMILES columns")

        y_train = df_train_split["r_product_class"].astype(int).values
        print("  Computing lookup predictions for voting model (incl. negative data)...")
        lookup_pred, lookup_sim = compute_naive_baseline_predictions_with_similarity(
            df, df_train_split, y_train, predictor.features
        )
        models_agree = y_pred == lookup_pred
    except Exception as e:
        print(f"  ✗ Could not compute voting predictions: {e}")
        print("  Cannot run voting model analysis without lookup predictions.")
        return

    # ------- Apply voting filter (base dataset = models agree) -------
    n_total = len(models_agree)
    n_agree = models_agree.sum()
    print(
        f"\n  Voting model: {n_agree}/{n_total} samples where models agree "
        f"({n_agree / n_total * 100:.1f}%)"
    )

    y_true = y_true[models_agree]
    y_pred = y_pred[models_agree]
    y_proba = y_proba[models_agree]
    confidence = confidence[models_agree]
    correct_mask = y_pred == y_true
    df = df[models_agree].reset_index(drop=True)

    print(f"  Voting model accuracy: {correct_mask.mean():.3f}")
    print_classification_report(y_true, y_pred)

    # Determine which plots to generate
    generate_all = args.all or not any(
        [
            args.combined,
            args.confusion,
            args.confidence,
            args.features,
            args.calibration,
            args.errors,
            args.confidence_vs_r1r2,
            args.filtering,
        ]
    )

    # ------- Generate plots (all on voting-filtered base set) -------
    print(f"\nGenerating plots{' (' + suffix + ')' if suffix else ''}...")

    if generate_all or args.combined:
        plot_confusion_matrix_and_confidence(
            y_true,
            y_pred,
            confidence,
            correct_mask,
            args.output_dir,
            suffix,
            confidence_threshold=args.confidence_threshold,
            normalize=False,
        )
        plot_confusion_matrix_and_confidence(
            y_true,
            y_pred,
            confidence,
            correct_mask,
            args.output_dir,
            suffix,
            confidence_threshold=args.confidence_threshold,
            normalize=True,
        )

    if generate_all or args.confidence:
        plot_confidence_distribution(confidence, correct_mask, args.output_dir, suffix)

    if (generate_all or args.features) and not suffix:
        plot_feature_importance(predictor, args.output_dir)

    if generate_all or args.calibration:
        plot_calibration_curve_multiclass(y_true, y_proba, args.output_dir, suffix)

    if generate_all or args.errors:
        plot_error_analysis_by_class(y_true, y_pred, confidence, args.output_dir, suffix)

    if (generate_all or args.confidence_vs_r1r2) and "r1r2" in df.columns:
        plot_confidence_vs_r1r2(df, y_pred, confidence, args.output_dir, suffix)

    if (generate_all or args.filtering) and not suffix:
        analyze_confidence_filtering(
            y_true, y_pred, confidence, args.output_dir, args.min_retention
        )


def main():
    """Main analysis pipeline.

    Loads the global train/validation/test split and evaluates the voting model
    (XGBoost + Lookup, threshold 0.7) on the test set.
    Plots are generated twice: unfiltered and voting-filtered.
    """
    args = parse_args()

    # Setup
    setup_style()
    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve relative paths robustly (works whether called from repo root or from within `copol_prediction/analysis/`).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if args.model_path and not os.path.isabs(args.model_path):
        # First interpret relative to this script (copol_prediction/analysis).
        candidate = os.path.normpath(os.path.join(script_dir, args.model_path))
        if os.path.exists(candidate):
            args.model_path = candidate
        else:
            # Then interpret relative to project root (copol_prediction/).
            candidate2 = os.path.normpath(os.path.join(project_root, args.model_path))
            if os.path.exists(candidate2):
                args.model_path = candidate2

    print("=" * 60)
    print("MODEL ANALYSIS  (test set, voting model)")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Confidence threshold: {args.confidence_threshold}")

    # Load model
    print("\nLoading model...")
    try:
        predictor = CopolymerPredictor(args.model_path)
        print(f"  ✓ Model loaded ({len(predictor.features)} features)")
    except Exception as e:
        print(f"  ✗ Error loading model: {e}")
        sys.exit(1)

    # Generate LaTeX tables if requested
    if args.latex_table:
        create_latex_performance_table(
            predictor,
            args.output_dir,
            n_folds=args.n_folds,
            confidence_threshold=args.confidence_threshold,
        )
        create_latex_per_class_table(
            predictor, args.output_dir, confidence_threshold=args.confidence_threshold
        )

    # Load test set from the global train/validation/test split
    print("\nLoading test set from train/validation/test split...")
    try:
        split_dir = os.path.join(project_root, "artifacts", "data_splits")
        _, df_val, df_test = load_train_val_test_split(split_dir=split_dir)
        print(f"  ✓ Validation set loaded ({len(df_val)} samples)")
        print(f"  ✓ Test set loaded ({len(df_test)} samples)")
    except Exception as e:
        print(f"  ✗ Error loading test set: {e}")
        sys.exit(1)

    # Optional: calibration method comparison on validation set
    if args.calibration_compare:
        try:
            compare_calibration_methods_on_validation(
                df_val, predictor, args.output_dir, suffix="validation"
            )
        except Exception as e:
            print(f"  ✗ Calibration comparison failed: {e}")

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    generate_plots_for_dataset(df_test, predictor, args, suffix="")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"\nAll plots saved to: {args.output_dir}/")
    print("\nGenerated plots:")
    for file in sorted(os.listdir(args.output_dir)):
        if file.endswith((".png", ".pdf")):
            print(f"  - {file}")


if __name__ == "__main__":
    main()
