"""
Utility functions to load the central train/test split.

Use these functions in all training scripts to ensure consistent data splits.
"""

import os
import pandas as pd


def load_train_test_split(split_dir="artifacts/data_splits"):
    """
    Load the central train/test split.
    
    Returns:
        tuple: (df_train, df_test)
    
    Raises:
        FileNotFoundError: If split files don't exist
    """
    train_path = os.path.join(split_dir, 'train.csv')
    test_path = os.path.join(split_dir, 'test.csv')
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Train split not found at {train_path}\n"
            f"Run: python create_data_split.py"
        )
    
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test split not found at {test_path}\n"
            f"Run: python create_data_split.py"
        )
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    return df_train, df_test


def get_split_info(split_dir="artifacts/data_splits"):
    """
    Get metadata about the train/test split.
    
    Returns:
        dict: Split information
    """
    import json
    
    info_path = os.path.join(split_dir, 'split_info.json')
    
    if not os.path.exists(info_path):
        return None
    
    with open(info_path, 'r') as f:
        return json.load(f)


def print_split_info(split_dir="artifacts/data_splits"):
    """Print information about the current split."""
    info = get_split_info(split_dir)
    
    if info is None:
        print("No split info available")
        return
    
    print("="*60)
    print("CURRENT DATA SPLIT INFO")
    print("="*60)
    print(f"Total samples: {info['total_samples']}")
    print(f"Train samples: {info['train_samples']} ({info['train_samples']/info['total_samples']*100:.1f}%)")
    print(f"Test samples:  {info['test_samples']} ({info['test_samples']/info['total_samples']*100:.1f}%)")
    print(f"Train groups:  {info['train_groups']}")
    print(f"Test groups:   {info['test_groups']}")
    
    # Show filters
    if 'filters_applied' in info:
        print(f"\nFilters applied: {', '.join(info['filters_applied'])}")
    if info.get('remove_specialized_from_test', False) or info.get('remove_specialized', False):
        print("  ⚠️  Specialized reactions removed from TEST SET only")
    
    print("\nTrain class distribution:")
    for cls, count in sorted(info['train_class_counts'].items()):
        pct = count / info['train_samples'] * 100
        print(f"  Class {cls}: {count:4d} ({pct:5.2f}%)")
    print("\nTest class distribution:")
    for cls, count in sorted(info['test_class_counts'].items()):
        pct = count / info['test_samples'] * 100
        print(f"  Class {cls}: {count:4d} ({pct:5.2f}%)")
    print("="*60)

