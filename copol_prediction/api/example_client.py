#!/usr/bin/env python3
"""
Example client for the Copolymerization Prediction API.

This script demonstrates how to use the API from Python code.
"""

import requests
from typing import Dict, List
import json


class CopolymerAPIClient:
    """Client for interacting with the Copolymerization Prediction API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict:
        """Check if API is healthy."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        response = requests.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()
    
    def get_required_features(self) -> List[str]:
        """Get list of required features."""
        response = requests.get(f"{self.base_url}/features")
        response.raise_for_status()
        return response.json()['required_features']
    
    def predict(self, features: Dict[str, float]) -> Dict:
        """
        Make a single prediction.
        
        Args:
            features: Dictionary with feature values
            
        Returns:
            Prediction results
        """
        response = requests.post(
            f"{self.base_url}/predict",
            json={"features": features}
        )
        response.raise_for_status()
        return response.json()
    
    def predict_batch(self, samples: List[Dict[str, float]]) -> Dict:
        """
        Make batch predictions.
        
        Args:
            samples: List of feature dictionaries
            
        Returns:
            Batch prediction results
        """
        response = requests.post(
            f"{self.base_url}/predict/batch",
            json={"samples": samples}
        )
        response.raise_for_status()
        return response.json()


def example_single_prediction():
    """Example: Make a single prediction."""
    print("\n" + "="*60)
    print("Example: Single Prediction")
    print("="*60)
    
    # Initialize client
    client = CopolymerAPIClient()
    
    # Check if API is healthy
    health = client.health_check()
    print(f"API Status: {health['status']}")
    
    # Example features
    features = {
        "fukui_radical_max_1": 0.15,
        "fukui_radical_max_2": 0.18,
        "delta_HOMO_LUMO_AA": -5.2,
        "delta_HOMO_LUMO_AB": -4.8,
        "delta_HOMO_LUMO_BB": -5.5,
        "delta_HOMO_LUMO_BA": -4.9,
        "temperature": 60.0,
        "polytype_emb_1": 0.23,
        "polytype_emb_2": -0.15,
        "method_emb_1": 0.45,
        "method_emb_2": -0.32,
        "solvent_logP": 2.1,
        "solvent_TPSA": 20.5,
        "solvent_HBD": 0.0,
        "solvent_FractionCSP3": 0.67
    }
    
    # Make prediction
    result = client.predict(features)
    
    print(f"\nPrediction Results:")
    print(f"  Predicted Class: {result['predicted_class']}")
    print(f"  R-Product Range: {result['r_product_range']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"\n  Class Probabilities:")
    for cls, prob in result['class_probabilities'].items():
        print(f"    {cls}: {prob:.2%}")


def example_batch_prediction():
    """Example: Make batch predictions."""
    print("\n" + "="*60)
    print("Example: Batch Prediction")
    print("="*60)
    
    # Initialize client
    client = CopolymerAPIClient()
    
    # Multiple samples
    samples = [
        {
            "fukui_radical_max_1": 0.15,
            "fukui_radical_max_2": 0.18,
            "delta_HOMO_LUMO_AA": -5.2,
            "delta_HOMO_LUMO_AB": -4.8,
            "delta_HOMO_LUMO_BB": -5.5,
            "delta_HOMO_LUMO_BA": -4.9,
            "temperature": 60.0,
            "polytype_emb_1": 0.23,
            "polytype_emb_2": -0.15,
            "method_emb_1": 0.45,
            "method_emb_2": -0.32,
            "solvent_logP": 2.1,
            "solvent_TPSA": 20.5,
            "solvent_HBD": 0.0,
            "solvent_FractionCSP3": 0.67
        },
        {
            "fukui_radical_max_1": 0.20,
            "fukui_radical_max_2": 0.22,
            "delta_HOMO_LUMO_AA": -4.8,
            "delta_HOMO_LUMO_AB": -4.5,
            "delta_HOMO_LUMO_BB": -5.0,
            "delta_HOMO_LUMO_BA": -4.7,
            "temperature": 80.0,
            "polytype_emb_1": 0.30,
            "polytype_emb_2": -0.20,
            "method_emb_1": 0.50,
            "method_emb_2": -0.35,
            "solvent_logP": 1.8,
            "solvent_TPSA": 25.0,
            "solvent_HBD": 1.0,
            "solvent_FractionCSP3": 0.50
        }
    ]
    
    # Make batch prediction
    results = client.predict_batch(samples)
    
    print(f"\nBatch Prediction Results:")
    print(f"  Total Samples: {results['total_samples']}")
    
    for i, pred in enumerate(results['predictions'], 1):
        print(f"\n  Sample {i}:")
        print(f"    Class: {pred['predicted_class']}")
        print(f"    Range: {pred['r_product_range']}")
        print(f"    Confidence: {pred['confidence']:.2%}")


def example_model_info():
    """Example: Get model information."""
    print("\n" + "="*60)
    print("Example: Model Information")
    print("="*60)
    
    # Initialize client
    client = CopolymerAPIClient()
    
    # Get model info
    info = client.get_model_info()
    
    print(f"\nModel Information:")
    print(f"  Version: {info['model_version']}")
    print(f"  Number of Features: {info['n_features']}")
    print(f"  Class Labels: {info['class_labels']}")
    print(f"  Created: {info['created_at']}")
    
    # Get required features
    features = client.get_required_features()
    print(f"\n  Required Features ({len(features)}):")
    for i, feature in enumerate(features, 1):
        print(f"    {i:2d}. {feature}")


def example_error_handling():
    """Example: Handle errors gracefully."""
    print("\n" + "="*60)
    print("Example: Error Handling")
    print("="*60)
    
    # Initialize client
    client = CopolymerAPIClient()
    
    # Try prediction with missing features
    incomplete_features = {
        "fukui_radical_max_1": 0.15,
        # Missing all other features!
    }
    
    try:
        result = client.predict(incomplete_features)
        print("Unexpected success!")
    except requests.exceptions.HTTPError as e:
        print(f"\n✓ Correctly caught error:")
        print(f"  Status Code: {e.response.status_code}")
        print(f"  Error Detail: {e.response.json().get('detail', 'N/A')}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("COPOLYMERIZATION API CLIENT EXAMPLES")
    print("="*60)
    
    try:
        # Check if API is running
        response = requests.get("http://localhost:8000/health", timeout=5)
        response.raise_for_status()
    except Exception as e:
        print("\n✗ API is not running!")
        print("  Please start the API first:")
        print("    cd copol_prediction/api")
        print("    python app.py")
        return
    
    # Run examples
    example_model_info()
    example_single_prediction()
    example_batch_prediction()
    example_error_handling()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()

