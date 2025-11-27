"""
Model calibration utilities for copolymerization prediction.

This module provides tools for calibrating trained classifiers using
sigmoid (Platt scaling) or isotonic regression methods.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class CalibratedModel:
    """
    A wrapper for calibrated classification models.
    
    Provides predict_proba and predict methods with calibrated probabilities.
    """
    
    def __init__(self, base_model, calibrator):
        """
        Initialize calibrated model.
        
        Args:
            base_model: Trained classifier with predict_proba method
            calibrator: Fitted calibrator (LogisticRegression or IsotonicRegression)
        """
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X):
        """
        Predict calibrated probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of shape (n_samples, n_classes) with calibrated probabilities
        """
        raw_proba = self.base_model.predict_proba(X)[:, 1]

        # Handle sigmoid vs isotonic calibrator
        if hasattr(self.calibrator, "predict_proba"):
            calibrated = self.calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
        else:
            calibrated = self.calibrator.predict(raw_proba)

        # Return as proper two-column probability matrix: [P(class 0), P(class 1)]
        calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
        return np.vstack([1 - calibrated, calibrated]).T

    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted class labels
        """
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def calibrate_model_with_weights(
    model,
    X_calib,
    y_calib,
    class_weight_dict=None,
    method="sigmoid",
    calibrator_kwargs=None
):
    """
    Calibrate a classifier using sigmoid (Platt scaling) or isotonic regression.

    Args:
        model: A trained classifier supporting predict_proba()
        X_calib: Calibration feature matrix
        y_calib: Binary labels for calibration set
        class_weight_dict: Optional dict, e.g., {0: 3.0, 1: 1.0}
        method: 'sigmoid' (Platt scaling) or 'isotonic'
        calibrator_kwargs: Optional kwargs for the calibrator

    Returns:
        CalibratedModel: A wrapper with predict_proba and predict
    """

    # Step 1: Predict raw probabilities (class 1)
    raw_proba = model.predict_proba(X_calib)[:, 1]

    # Step 2: Sample weights based on class weights
    if class_weight_dict is not None:
        sample_weights = np.array([class_weight_dict[y] for y in y_calib])
    else:
        sample_weights = None

    # Step 3: Fit calibrator
    if method == "sigmoid":
        calibrator_args = calibrator_kwargs or {}
        calibrator = LogisticRegression(**calibrator_args)
        calibrator.fit(raw_proba.reshape(-1, 1), y_calib, sample_weight=sample_weights)

    elif method == "isotonic":
        # IsotonicRegression doesn't accept 'solver' or many LogisticRegression kwargs
        # Only safe kwargs like y_min, y_max, increasing etc.
        calibrator_args = calibrator_kwargs or {}
        calibrator = IsotonicRegression(out_of_bounds="clip", **calibrator_args)
        calibrator.fit(raw_proba, y_calib, sample_weight=sample_weights)

    else:
        raise ValueError(f"Unknown calibration method '{method}'. Use 'sigmoid' or 'isotonic'.")

    return CalibratedModel(model, calibrator)


def calculate_prediction_confidence(model, X_test):
    """
    Calculate prediction confidence using entropy-based uncertainty.
    Lower entropy = higher confidence
    
    Args:
        model: Trained model with predict_proba method
        X_test: Test feature matrix
        
    Returns:
        Array of confidence scores (0 to 1)
    """
    # Get probabilities for all classes
    y_pred_proba = model.predict_proba(X_test)

    # Avoid log(0) by clipping probabilities
    epsilon = 1e-15
    y_pred_proba = np.clip(y_pred_proba, epsilon, 1 - epsilon)

    # Calculate entropy: H = -sum(p * log(p))
    entropy = -np.sum(y_pred_proba * np.log(y_pred_proba), axis=1)

    # Maximum entropy
    max_entropy = np.log(y_pred_proba.shape[1])

    # Convert entropy to confidence: confidence = 1 - (entropy / max_entropy)
    confidence = 1 - (entropy / max_entropy)

    return confidence


