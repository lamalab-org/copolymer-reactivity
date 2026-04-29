#!/usr/bin/env python3
"""
Create central train/validation/test split used by ALL scripts.

This ensures all models and experiments use identical data splits:
- train_final_model.py
- sweep_filters.py
- All experiment scripts

Run this ONCE before any training/experiments.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

from copolpredictor import data_processing, holdout_utils, prediction_utils

try:
    # When run as a package module: python -m copol_prediction.create_data_split
    from .mayo_lewis_classification import classify_reactivity_curve
except ImportError:
    # When run as a script from within copol_prediction/: python create_data_split.py
    from mayo_lewis_classification import classify_reactivity_curve

# Feature columns for NaN checking — must stay in sync with the trained model.
# Always derived from prediction_utils.feature_columns so there's a single source of truth.
FEATURE_COLUMNS = prediction_utils.feature_columns


def parse_args():
    parser = argparse.ArgumentParser(description="Create central train/validation/test split")
    parser.add_argument('--remove-specialized', action='store_true',
                       help='Remove datapoints with specialized_filter=specialized')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data for test set (default: 0.2)')
    parser.add_argument('--val-size', type=float, default=0.1,
                       help='Fraction of data for validation set (default: 0.1)')
    return parser.parse_args()


def create_split(remove_specialized=False, test_size=0.2, val_size=0.1):
    """
    Create train/validation/test split and save to artifacts/data_splits/
    """
    print("="*70)
    print("CREATING CENTRAL TRAIN/VALIDATION/TEST SPLIT")
    print("="*70)
    print("\nThis split will be used by all training scripts and experiments")
    print("to ensure fair comparison.\n")
    print(f"Split sizes: Train ~{1-test_size-val_size:.1%}, Validation ~{val_size:.1%}, Test ~{test_size:.1%}\n")
    
    # Load processed data
    processed_path = "output/processed_data.csv"
    
    if not os.path.exists(processed_path):
        print("Processed data not found. Processing from scratch...")
        df = data_processing.load_and_preprocess_data(
            "../data_extraction/artifacts/datasets/extracted_reactions.csv"
        )
        if df is None or len(df) == 0:
            print("Error: No data available after preprocessing")
            sys.exit(1)
        df.to_csv(processed_path, index=False)
        print(f"Saved processed data to: {processed_path}")
    else:
        print(f"Loading processed data from: {processed_path}")
        df = pd.read_csv(processed_path)
    
    print(f"Total samples: {len(df)}")
    
    # Load and merge specialized_filter classifications
    print("\nLoading specialized_filter classifications...")
    specialized_path = "filter/llm_specialized_filter/classified_output.csv"
    
    if os.path.exists(specialized_path):
        df_classified = pd.read_csv(specialized_path)
        print(f"Loaded classified data: {len(df_classified)} rows")
        
        # Check if llm_specialized_filter column exists
        if 'llm_specialized_filter' in df_classified.columns and 'original_source' in df_classified.columns:
            # Normalize DOIs for matching
            def normalize_doi(doi_str):
                if pd.isna(doi_str):
                    return None
                doi_str = str(doi_str).strip().lower()
                for prefix in ['https://doi.org/', 'http://doi.org/', 'doi:', 'doi ']:
                    if doi_str.startswith(prefix.lower()):
                        doi_str = doi_str[len(prefix):]
                return doi_str.strip()
            
            # Create lookup dict from classified data
            doi_to_classification = {}
            for _, row in df_classified.iterrows():
                doi_norm = normalize_doi(row['original_source'])
                if doi_norm:
                    doi_to_classification[doi_norm] = row['llm_specialized_filter']
            
            print(f"Created lookup with {len(doi_to_classification)} DOI classifications")
            
            # Apply to main dataframe
            if 'original_source' in df.columns:
                def get_classification(doi_str):
                    doi_norm = normalize_doi(doi_str)
                    if doi_norm and doi_norm in doi_to_classification:
                        return doi_to_classification[doi_norm]
                    return 'unclear'
                
                df['specialized_filter'] = df['original_source'].apply(get_classification)
                
                # Count classifications
                print("\nSpecialized filter distribution:")
                for val, count in df['specialized_filter'].value_counts().items():
                    pct = count / len(df) * 100
                    print(f"  {val}: {count:4d} ({pct:5.1f}%)")
            else:
                print("Warning: 'original_source' column not found in data")
                df['specialized_filter'] = 'unclear'
        else:
            print(f"Warning: Required columns not found in {specialized_path}")
            print(f"Available columns: {df_classified.columns.tolist()[:5]}...")
            df['specialized_filter'] = 'unclear'
    else:
        print(f"Warning: {specialized_path} not found, setting all to 'unclear'")
        df['specialized_filter'] = 'unclear'
    
    # Apply basic filters
    print("\nApplying basic filters...")
    df_filtered = df.copy()
    df_filtered = df_filtered[df_filtered['r1r2'].notna()]
    df_filtered = df_filtered[df_filtered['r1r2'] >= 0]
    print(f"After basic filtering: {len(df_filtered)} samples")
    
    # Create target classes based on individual reactivity ratios (r1 = constant_1, r2 = constant_2)
    # using the Mayo–Lewis curve-based classification (alternating / gradient / random-to-blocky).
    if {'constant_1', 'constant_2'}.issubset(df_filtered.columns):
        def _class_from_row(row):
            res = classify_reactivity_curve(float(row['constant_1']), float(row['constant_2']))
            return res['class_id']

        df_filtered['r_product_class'] = df_filtered.apply(_class_from_row, axis=1).astype(int)
    else:
        raise ValueError(
            "Required columns 'constant_1' and 'constant_2' not found for class definition."
        )
    
    print("\nClass distribution:")
    class_counts = df_filtered['r_product_class'].value_counts().sort_index()
    for cls, count in class_counts.items():
        pct = count / len(df_filtered) * 100
        print(f"  Class {cls}: {count:4d} ({pct:5.2f}%)")
    
    # Remove NaN rows BEFORE split (important for consistent split!)
    print("\nRemoving NaN rows before split...")
    available_features = [c for c in FEATURE_COLUMNS if c in df_filtered.columns]
    print(f"Checking {len(available_features)} features for NaN")
    
    X_all = df_filtered[available_features]
    y_all = df_filtered['r_product_class'].astype(int)
    mask = ~(pd.isna(X_all).any(axis=1) | pd.isna(y_all))
    df_clean = df_filtered[mask].reset_index(drop=True)
    print(f"After NaN removal: {len(df_clean)} samples (removed {len(df_filtered) - len(df_clean)})")
    
    # Create base dataset for holdout splitting
    base_df = holdout_utils.make_base_dataset_for_holdout(df_clean)
    
    # Get or create train/val/test groups
    test_path = "artifacts/test_ids.csv"
    val_path = "artifacts/val_ids.csv"
    train_groups, val_groups, test_groups = holdout_utils.get_or_create_train_val_test_groups(
        base_df, 
        group_col='reaction_id',
        test_groups_path=test_path,
        val_groups_path=val_path,
        test_size=test_size,
        val_size=val_size
    )
    
    print(f"\nSplit groups saved to:")
    print(f"  Test: {test_path}")
    print(f"  Validation: {val_path}")
    
    # Split into train, validation, and test (using df_clean, not df_filtered!)
    df_train, df_val, df_test = holdout_utils.split_train_val_test(
        df_clean, train_groups, val_groups, test_groups, group_col='reaction_id'
    )
    
    print(f"\nInitial split:")
    print(f"  Train:      {len(df_train):4d} samples ({df_train['reaction_id'].nunique()} groups)")
    print(f"  Validation: {len(df_val):4d} samples ({df_val['reaction_id'].nunique()} groups)")
    print(f"  Test:       {len(df_test):4d} samples ({df_test['reaction_id'].nunique()} groups)")
    
    # Validate: No overlapping reaction_ids between any sets
    train_ids = set(df_train['reaction_id'].astype(str).unique())
    val_ids = set(df_val['reaction_id'].astype(str).unique())
    test_ids = set(df_test['reaction_id'].astype(str).unique())
    
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    
    if len(train_val_overlap) > 0 or len(train_test_overlap) > 0 or len(val_test_overlap) > 0:
        print(f"\n  ERROR: Overlapping reaction_ids detected!")
        if len(train_val_overlap) > 0:
            print(f"    Train-Val overlap: {len(train_val_overlap)} reaction_ids")
        if len(train_test_overlap) > 0:
            print(f"    Train-Test overlap: {len(train_test_overlap)} reaction_ids")
        if len(val_test_overlap) > 0:
            print(f"    Val-Test overlap: {len(val_test_overlap)} reaction_ids")
        raise ValueError("Splits have overlapping reaction_ids! This should not happen.")
    else:
        print(f"\n  ✓ Validation: No overlapping reaction_ids between any sets")
        print(f"    Total unique reaction_ids: {len(train_ids) + len(val_ids) + len(test_ids)}")
    
    # Note: Specialized filter is NOT applied to validation or test sets
    # Validation and test sets should remain unchanged to ensure fair evaluation
    # If remove_specialized is True, it would only affect training data filtering
    # (which is handled separately in training scripts)
    if remove_specialized:
        print("\n⚠️  Note: --remove-specialized flag is set, but specialized datapoints")
        print("   are NOT removed from validation or test sets (only affects training).")
    
    print(f"\nFinal split:")
    print(f"  Train:      {len(df_train):4d} samples ({df_train['reaction_id'].nunique()} groups)")
    print(f"  Validation: {len(df_val):4d} samples ({df_val['reaction_id'].nunique()} groups)")
    print(f"  Test:       {len(df_test):4d} samples ({df_test['reaction_id'].nunique()} groups)")
    
    print("\nTrain class distribution:")
    train_class_counts = df_train['r_product_class'].value_counts().sort_index()
    for cls, count in train_class_counts.items():
        pct = count / len(df_train) * 100
        print(f"  Class {cls}: {count:4d} ({pct:5.2f}%)")
    
    print("\nValidation class distribution:")
    val_class_counts = df_val['r_product_class'].value_counts().sort_index()
    for cls, count in val_class_counts.items():
        pct = count / len(df_val) * 100
        print(f"  Class {cls}: {count:4d} ({pct:5.2f}%)")
    
    print("\nTest class distribution:")
    test_class_counts = df_test['r_product_class'].value_counts().sort_index()
    for cls, count in test_class_counts.items():
        pct = count / len(df_test) * 100
        print(f"  Class {cls}: {count:4d} ({pct:5.2f}%)")
    
    # Save splits
    output_dir = "artifacts/data_splits"
    os.makedirs(output_dir, exist_ok=True)
    
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print(f"\n✓ Saved train split to:      {train_path}")
    print(f"✓ Saved validation split to: {val_path}")
    print(f"✓ Saved test split to:        {test_path}")
    
    # Save split metadata
    filters_applied = ['r1r2 >= 0', 'r1r2 not null', 'removed rows with NaN features']
    # Note: specialized_filter is NOT applied to validation or test sets
    
    split_info = {
        'total_samples': len(df_clean),
        'train_samples': len(df_train),
        'val_samples': len(df_val),
        'test_samples': len(df_test),
        'train_groups': int(df_train['reaction_id'].nunique()),
        'val_groups': int(df_val['reaction_id'].nunique()),
        'test_groups': int(df_test['reaction_id'].nunique()),
        'train_class_counts': {int(k): int(v) for k, v in train_class_counts.items()},
        'val_class_counts': {int(k): int(v) for k, v in val_class_counts.items()},
        'test_class_counts': {int(k): int(v) for k, v in test_class_counts.items()},
        'test_ids_path': test_path,
        'val_ids_path': val_path,
        'test_size_ratio': len(df_test) / len(df_clean),
        'val_size_ratio': len(df_val) / len(df_clean),
        'filters_applied': filters_applied,
        'remove_specialized_from_test': False,  # Never applied to test/validation
    }
    
    info_path = os.path.join(output_dir, 'split_info.json')
    with open(info_path, 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"✓ Saved split info to: {info_path}")
    
    print("\n" + "="*70)
    print("SPLIT CREATION COMPLETE")
    print("="*70)
    print("\nThese splits should now be used by:")
    print("  - train_final_model.py")
    print("  - sweep_filters.py")
    print("  - All experiment scripts")
    print("\nThis ensures all models are trained and evaluated on identical data!")


if __name__ == "__main__":
    args = parse_args()
    create_split(
        remove_specialized=args.remove_specialized,
        test_size=args.test_size,
        val_size=args.val_size
    )
