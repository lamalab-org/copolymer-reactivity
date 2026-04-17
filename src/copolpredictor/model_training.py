"""
Model training utilities for copolymerization prediction.

This module provides functions for training XGBoost classifiers with
hyperparameter search, cross-validation, and class weight handling.
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
import joblib
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, GroupKFold
from sklearn.metrics import make_scorer, f1_score


def calculate_class_weights(y_train):
    """
    Calculate balanced class weights.
    
    Args:
        y_train: Training labels
        
    Returns:
        Dictionary mapping class labels to weights
    """
    class_counts = pd.Series(y_train).value_counts().sort_index()
    total_samples = len(y_train)
    n_classes = class_counts.shape[0]
    
    class_weights = {
        int(cls): round(total_samples / (n_classes * count), 4)
        for cls, count in class_counts.items()
    }
    
    return class_weights


def create_sample_weights(y_train, class_weights):
    """
    Create sample weights array based on class weights.
    
    Args:
        y_train: Training labels
        class_weights: Dictionary of class weights
        
    Returns:
        Array of sample weights
    """
    return np.array([class_weights[int(label)] for label in y_train])


def train_xgboost_with_cv(
    X_train,
    y_train,
    groups=None,
    param_grid=None,
    n_iter=10,
    cv=5,
    random_state=42,
    class_weights=None,
    n_jobs=-1
):
    """
    Train XGBoost classifier with cross-validation and hyperparameter search.
    
    Args:
        X_train: Training features
        y_train: Training labels
        groups: Group labels for GroupKFold
        param_grid: Parameter distributions for RandomizedSearchCV
        n_iter: Number of iterations for random search
        cv: Number of CV folds or CV splitter
        random_state: Random seed
        class_weights: Dictionary of class weights
        n_jobs: Number of parallel jobs
        
    Returns:
        Dictionary with best model, parameters, and CV results
    """
    # Default parameter grid
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 300, 600, 800],
            'max_depth': [3, 5, 8],
            'learning_rate': [0.04, 0.05, 0.06, 0.07],
            'subsample': [0.8, 0.9, 0.95],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 0.6],
            'reg_lambda': [1, 1.5, 2, 3],
            'min_child_weight': [2, 3, 5, 7],
            'gamma': [0.5, 0.6]
        }
    
    # Determine number of classes
    n_classes = len(np.unique(y_train))
    
    # Base model
    base_model = xgb.XGBClassifier(
        objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
        num_class=n_classes if n_classes > 2 else None,
        eval_metric='mlogloss',
        random_state=random_state
    )
    
    # Calculate class weights if not provided
    if class_weights is None:
        class_weights = calculate_class_weights(y_train)
    
    # Create sample weights
    sample_weights = create_sample_weights(y_train, class_weights)
    
    # Setup CV
    if isinstance(cv, int):
        if groups is not None:
            cv_splitter = GroupKFold(n_splits=cv)
            cv_splits = list(cv_splitter.split(X_train, y_train, groups=groups))
        else:
            cv_splits = cv
    else:
        cv_splits = cv
    
    # Scorer
    scorer = make_scorer(f1_score, average='weighted')
    
    # Random search
    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scorer,
        cv=cv_splits,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=1
    )
    
    # Fit
    search.fit(X_train, y_train, sample_weight=sample_weights)
    
    return {
        'best_model': search.best_estimator_,
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'cv_results': search.cv_results_,
        'class_weights': class_weights
    }


def train_final_model(
    X_train,
    y_train,
    params,
    class_weights=None,
    random_state=42
):
    """
    Train final model with given parameters on full training set.
    
    Args:
        X_train: Training features
        y_train: Training labels
        params: Model hyperparameters
        class_weights: Dictionary of class weights
        random_state: Random seed
        
    Returns:
        Trained XGBoost model
    """
    # Determine number of classes
    n_classes = len(np.unique(y_train))
    
    # Create model
    model = xgb.XGBClassifier(
        objective='multi:softprob' if n_classes > 2 else 'binary:logistic',
        num_class=n_classes if n_classes > 2 else None,
        eval_metric='mlogloss',
        random_state=random_state,
        **params
    )
    
    # Calculate class weights if not provided
    if class_weights is None:
        class_weights = calculate_class_weights(y_train)
    
    # Create sample weights
    sample_weights = create_sample_weights(y_train, class_weights)
    
    # Train
    model.fit(X_train, y_train, sample_weight=sample_weights)
    
    return model


def save_model_bundle(
    model,
    feature_list,
    class_labels,
    out_dir="artifacts/model_bundle",
    metadata=None
):
    """
    Save trained model bundle with metadata.
    
    Args:
        model: Trained model
        feature_list: List of feature names
        class_labels: List/tuple of class labels
        out_dir: Output directory
        metadata: Optional additional metadata dictionary
        
    Returns:
        Path to saved bundle directory
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Save model
    joblib.dump(model, f"{out_dir}/model.joblib")
    
    # Try to save native booster format
    try:
        model.get_booster().save_model(f"{out_dir}/model.xgb.json")
    except Exception as e:
        print(f"[WARN] Booster save failed: {e}")
    
    # Create metadata
    meta = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "feature_columns": list(feature_list),
        "class_labels": list(class_labels),
        "task": "multiclass_r_product_class",
        "n_features": len(feature_list),
        "n_classes": len(class_labels)
    }
    
    # Add custom metadata
    if metadata:
        meta.update(metadata)
    
    # Save metadata
    with open(f"{out_dir}/meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Model bundle saved to {out_dir}")
    return out_dir


def load_model_bundle(bundle_dir="artifacts/model_bundle"):
    """
    Load trained model bundle.
    
    Args:
        bundle_dir: Directory containing the model bundle
        
    Returns:
        Dictionary with model, features, labels, and metadata
    """
    # Load model
    model = joblib.load(f"{bundle_dir}/model.joblib")
    
    # Load metadata
    with open(f"{bundle_dir}/meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    
    bundle = {
        'model': model,
        'features': meta['feature_columns'],
        'class_labels': meta['class_labels'],
        'metadata': meta
    }

    # Optional: probability calibration payload (fit on validation set)
    cal_path = os.path.join(bundle_dir, "calibration.joblib")
    if os.path.exists(cal_path):
        try:
            bundle["calibration"] = joblib.load(cal_path)
        except Exception as e:
            print(f"[WARN] Failed to load calibration from {cal_path}: {e}")

    return bundle


