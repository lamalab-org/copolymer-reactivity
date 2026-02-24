#!/usr/bin/env python3
"""
Feature selection based on SHAP importance analysis.

Loads SHAP results and selects features using various strategies:
- Top-N: Select top N feature groups
- Threshold: Select features above a SHAP importance threshold
- Percentile: Select top X% of features
- Cumulative: Select features until cumulative importance reaches threshold

Usage:
  python select_features_from_shap.py --shap-results results/shap_importance_detailed.csv [--strategy top-n --n 20]
"""

import os
import sys
import argparse
import pandas as pd
import json

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, '../..'))
sys.path.insert(0, os.path.join(_script_dir, '../../copol_prediction'))

from src.copolpredictor import prediction_utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select features based on SHAP importance analysis"
    )
    parser.add_argument(
        "--shap-results",
        type=str,
        required=True,
        help="Path to SHAP results CSV (shap_importance_detailed.csv)",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["top-n", "threshold", "percentile", "cumulative"],
        default="top-n",
        help="Selection strategy (default: top-n)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="For top-n strategy: number of top features to select (default: 20)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="For threshold strategy: minimum SHAP importance (default: 0.05)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=75,
        help="For percentile strategy: top X percentile to select (default: 75)",
    )
    parser.add_argument(
        "--cumulative-threshold",
        type=float,
        default=0.9,
        help="For cumulative strategy: cumulative importance threshold (default: 0.9 = 90%%)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for selected features JSON (default: same dir as SHAP results)",
    )
    parser.add_argument(
        "--expand-groups",
        action="store_true",
        help="Expand feature groups to individual features (default: keep groups as-is)",
    )
    return parser.parse_args()


def load_shap_results(shap_results_path):
    """Load SHAP results CSV."""
    if not os.path.exists(shap_results_path):
        raise FileNotFoundError(f"SHAP results not found: {shap_results_path}")
    
    df = pd.read_csv(shap_results_path)
    print(f"Loaded {len(df)} feature groups from {shap_results_path}")
    return df


def select_top_n(df, n):
    """Select top N feature groups by importance."""
    selected = df.head(n).copy()
    print(f"Selected top {n} feature groups")
    return selected


def select_by_threshold(df, threshold):
    """Select feature groups with importance >= threshold."""
    selected = df[df["importance_mean"] >= threshold].copy()
    print(f"Selected {len(selected)} feature groups with importance >= {threshold}")
    return selected


def select_by_percentile(df, percentile):
    """Select top X percentile of feature groups."""
    threshold = df["importance_mean"].quantile((100 - percentile) / 100)
    selected = df[df["importance_mean"] >= threshold].copy()
    print(f"Selected {len(selected)} feature groups (top {percentile}th percentile, threshold={threshold:.6f})")
    return selected


def select_by_cumulative(df, cumulative_threshold):
    """Select feature groups until cumulative importance reaches threshold."""
    df_sorted = df.sort_values("importance_mean", ascending=False).copy()
    df_sorted["cumulative_importance"] = df_sorted["importance_mean"].cumsum()
    total_importance = df_sorted["importance_mean"].sum()
    target_importance = cumulative_threshold * total_importance
    
    selected = df_sorted[df_sorted["cumulative_importance"] <= target_importance].copy()
    if len(selected) == 0:
        # At least select the top feature
        selected = df_sorted.head(1).copy()
    
    actual_cumulative = selected["importance_mean"].sum() / total_importance
    print(f"Selected {len(selected)} feature groups (cumulative importance: {actual_cumulative:.1%})")
    return selected


def expand_feature_groups(selected_df):
    """Expand feature groups to individual features."""
    all_features = set()
    for _, row in selected_df.iterrows():
        features_str = row["features"]
        if pd.notna(features_str):
            # Features are stored as "feature1|feature2|..."
            features = features_str.split("|")
            all_features.update(features)
    return sorted(list(all_features))


def get_selected_features(selected_df, expand_groups=False):
    """Get list of selected features (either groups or expanded individual features)."""
    if expand_groups:
        features = expand_feature_groups(selected_df)
        print(f"Expanded to {len(features)} individual features")
    else:
        # Return group labels (first feature in each group)
        features = selected_df["group_label"].tolist()
        print(f"Keeping {len(features)} feature groups")
    return features


def save_selected_features(features, selected_df, output_path, metadata=None, expand_groups=False):
    """Save selected features to JSON file."""
    # Include group information
    group_info = []
    for _, row in selected_df.iterrows():
        group_label = row["group_label"]
        importance = row["importance_mean"]
        features_str = row["features"]
        
        group_data = {
            "group_label": group_label,
            "importance_mean": float(importance),
            "importance_std": float(row.get("importance_std", 0)),
            "n_features_in_group": row.get("n_features", 1),
        }
        
        if pd.notna(features_str):
            group_data["features"] = features_str.split("|")
        else:
            group_data["features"] = []
        
        group_info.append(group_data)
    
    output = {
        "selected_features": features,
        "n_features": len(features),
        "n_groups": len(selected_df),
        "groups": group_info,
        "expand_groups": expand_groups,
        "metadata": metadata or {},
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✓ Saved selected features to {output_path}")


def main():
    args = parse_args()
    
    # Load SHAP results
    print("=" * 60)
    print("FEATURE SELECTION FROM SHAP ANALYSIS")
    print("=" * 60)
    df = load_shap_results(args.shap_results)
    
    # Apply selection strategy
    print(f"\nStrategy: {args.strategy}")
    if args.strategy == "top-n":
        selected_df = select_top_n(df, args.n)
    elif args.strategy == "threshold":
        selected_df = select_by_threshold(df, args.threshold)
    elif args.strategy == "percentile":
        selected_df = select_by_percentile(df, args.percentile)
    elif args.strategy == "cumulative":
        selected_df = select_by_cumulative(df, args.cumulative_threshold)
    
    # Print detailed summary with group members
    print(f"\nSelected feature groups:")
    print("-" * 60)
    for i, (_, row) in enumerate(selected_df.iterrows(), 1):
        group_label = row["group_label"]
        importance = row["importance_mean"]
        features_str = row["features"]
        
        if pd.notna(features_str):
            # Parse features from "feature1|feature2|..." format
            features_list = features_str.split("|")
            n_features = len(features_list)
            
            if n_features == 1:
                print(f"  {i:2d}. {group_label:<45} (importance: {importance:.6f})")
                print(f"      └─ Feature: {features_list[0]}")
            else:
                print(f"  {i:2d}. {group_label:<45} (importance: {importance:.6f}, {n_features} features)")
                for j, feat in enumerate(features_list, 1):
                    prefix = "      ├─" if j < n_features else "      └─"
                    print(f"      {prefix} {feat}")
        else:
            print(f"  {i:2d}. {group_label:<45} (importance: {importance:.6f})")
            print(f"      └─ (features not available)")
    
    # Get selected features
    selected_features = get_selected_features(selected_df, expand_groups=args.expand_groups)
    
    # Print summary of individual features if expanded
    if args.expand_groups:
        print(f"\nExpanded individual features ({len(selected_features)} total):")
        for i, feat in enumerate(selected_features[:20], 1):
            print(f"  {i:2d}. {feat}")
        if len(selected_features) > 20:
            print(f"  ... and {len(selected_features) - 20} more")
    
    # Compare with original feature set
    original_features = prediction_utils.feature_columns_all
    print(f"\nOriginal feature set: {len(original_features)} features")
    print(f"Selected features: {len(selected_features)} features")
    print(f"Reduction: {len(original_features) - len(selected_features)} features ({100 * (1 - len(selected_features)/len(original_features)):.1f}% reduction)")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        shap_dir = os.path.dirname(args.shap_results)
        output_path = os.path.join(shap_dir, "selected_features.json")
    
    metadata = {
        "strategy": args.strategy,
        "n_selected": len(selected_features),
        "n_original": len(original_features),
        "shap_results_file": args.shap_results,
    }
    if args.strategy == "top-n":
        metadata["n"] = args.n
    elif args.strategy == "threshold":
        metadata["threshold"] = args.threshold
    elif args.strategy == "percentile":
        metadata["percentile"] = args.percentile
    elif args.strategy == "cumulative":
        metadata["cumulative_threshold"] = args.cumulative_threshold
    
    save_selected_features(selected_features, selected_df, output_path, metadata, expand_groups=args.expand_groups)
    
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"\nTo use these features, load the JSON file:")
    print(f"  import json")
    print(f"  with open('{output_path}') as f:")
    print(f"      data = json.load(f)")
    print(f"      features = data['selected_features']")


if __name__ == "__main__":
    main()
