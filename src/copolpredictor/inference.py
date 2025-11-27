"""
Inference module for copolymerization prediction.

This module provides a high-level interface for making predictions
with trained models.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Union, Optional

from .model_training import load_model_bundle


class CopolymerPredictor:
    """
    High-level predictor for copolymerization reactivity.
    
    This class provides a simple interface for loading a trained model
    and making predictions on new data.
    """
    
    def __init__(self, model_bundle_path: str = "artifacts/model_bundle"):
        """
        Initialize predictor with trained model.
        
        Args:
            model_bundle_path: Path to model bundle directory
        """
        self.bundle_path = model_bundle_path
        self.bundle = None
        self.model = None
        self.features = None
        self.class_labels = None
        self.metadata = None
        
        self._load_model()
    
    def _load_model(self):
        """Load model bundle."""
        if not os.path.exists(self.bundle_path):
            raise FileNotFoundError(f"Model bundle not found at: {self.bundle_path}")
        
        self.bundle = load_model_bundle(self.bundle_path)
        self.model = self.bundle['model']
        self.features = self.bundle['features']
        self.class_labels = self.bundle['class_labels']
        self.metadata = self.bundle['metadata']
        
        print(f"✓ Loaded model from {self.bundle_path}")
        print(f"  - Features: {len(self.features)}")
        print(f"  - Classes: {self.class_labels}")
        print(f"  - Created: {self.metadata.get('created_at', 'unknown')}")
    
    def predict(self, X: Union[pd.DataFrame, dict, np.ndarray]) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features as DataFrame, dict, or array
            
        Returns:
            Array of predicted class labels
        """
        X_processed = self._prepare_input(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: Union[pd.DataFrame, dict, np.ndarray]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Input features as DataFrame, dict, or array
            
        Returns:
            Array of shape (n_samples, n_classes) with class probabilities
        """
        X_processed = self._prepare_input(X)
        return self.model.predict_proba(X_processed)
    
    def predict_with_confidence(
        self,
        X: Union[pd.DataFrame, dict, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Predict with confidence scores.
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with predictions, probabilities, and confidence scores
        """
        X_processed = self._prepare_input(X)
        
        # Get predictions
        y_pred = self.model.predict(X_processed)
        y_proba = self.model.predict_proba(X_processed)
        
        # Calculate confidence (entropy-based)
        epsilon = 1e-15
        y_proba_clipped = np.clip(y_proba, epsilon, 1 - epsilon)
        entropy = -np.sum(y_proba_clipped * np.log(y_proba_clipped), axis=1)
        max_entropy = np.log(len(self.class_labels))
        confidence = 1 - (entropy / max_entropy)
        
        return {
            'predictions': y_pred,
            'probabilities': y_proba,
            'confidence': confidence,
            'class_labels': self.class_labels
        }
    
    def predict_r_product_range(
        self,
        X: Union[pd.DataFrame, dict, np.ndarray]
    ) -> List[str]:
        """
        Predict r-product range category with human-readable labels.
        
        Args:
            X: Input features
            
        Returns:
            List of category labels
        """
        y_pred = self.predict(X)
        
        # Map class indices to range labels
        range_labels = {
            0: "< 1 (Strong alternating tendency)",
            1: "1-25 (Random to weak block)",
            2: "> 25 (Strong block tendency)"
        }
        
        return [range_labels.get(int(pred), "Unknown") for pred in y_pred]
    
    def _prepare_input(self, X: Union[pd.DataFrame, dict, np.ndarray]) -> pd.DataFrame:
        """
        Prepare input data for prediction.
        
        Args:
            X: Input features in various formats
            
        Returns:
            DataFrame with correct feature order
        """
        # Convert to DataFrame if needed
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        elif isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.features)
        elif not isinstance(X, pd.DataFrame):
            raise ValueError(f"Unsupported input type: {type(X)}")
        
        # Check for missing features
        missing_features = set(self.features) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select and order features
        return X[self.features]
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importances from the model.
        
        Args:
            top_n: Number of top features to return (None = all)
            
        Returns:
            DataFrame with features and importances
        """
        if not hasattr(self.model, 'feature_importances_'):
            raise AttributeError("Model does not have feature importances")
        
        importance_df = pd.DataFrame({
            'feature': self.features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        if top_n:
            importance_df = importance_df.head(top_n)
        
        return importance_df


def predict_from_smiles(
    monomer1_smiles: str,
    monomer2_smiles: str,
    temperature: float,
    solvent_smiles: str,
    polymerization_type: str,
    method: str,
    model_path: str = "artifacts/model_bundle",
    features_base_path: str = "copol_prediction/output/molecule_properties"
) -> Dict:
    """
    Make prediction directly from SMILES and reaction conditions.
    
    This is a convenience function that handles feature extraction
    and prediction in one step.
    
    Args:
        monomer1_smiles: SMILES string for first monomer
        monomer2_smiles: SMILES string for second monomer
        temperature: Reaction temperature
        solvent_smiles: SMILES string for solvent
        polymerization_type: Type of polymerization
        method: Polymerization method
        model_path: Path to model bundle
        features_base_path: Path to molecular properties directory
        
    Returns:
        Dictionary with prediction results
    """
    # This would need to be implemented with actual feature calculation
    # For now, this is a placeholder showing the intended interface
    raise NotImplementedError(
        "Feature calculation from SMILES needs to be integrated. "
        "Please use monomer_feature_calculation.py first, then use "
        "CopolymerPredictor.predict() with the calculated features."
    )


def batch_predict(
    input_csv: str,
    output_csv: str,
    model_path: str = "artifacts/model_bundle"
) -> pd.DataFrame:
    """
    Make predictions on a batch of samples from CSV.
    
    Args:
        input_csv: Path to input CSV with features
        output_csv: Path to save predictions
        model_path: Path to model bundle
        
    Returns:
        DataFrame with predictions
    """
    # Load data
    df = pd.read_csv(input_csv)
    
    # Initialize predictor
    predictor = CopolymerPredictor(model_path)
    
    # Make predictions
    results = predictor.predict_with_confidence(df)
    
    # Add predictions to dataframe
    df['predicted_class'] = results['predictions']
    df['confidence'] = results['confidence']
    
    # Add probabilities
    for i, label in enumerate(results['class_labels']):
        df[f'proba_class_{label}'] = results['probabilities'][:, i]
    
    # Save
    df.to_csv(output_csv, index=False)
    print(f"✓ Predictions saved to {output_csv}")
    
    return df


