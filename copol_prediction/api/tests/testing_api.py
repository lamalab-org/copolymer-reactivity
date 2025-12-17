#!/usr/bin/env python3
"""
Test script for the Copolymerization Prediction API.

This script tests all API endpoints to ensure they work correctly.

Usage:
    python testing_api.py [--url http://localhost:8000]
"""

import argparse
import sys
import requests
from typing import Dict, Any
from urllib.parse import quote


def test_health(base_url: str) -> bool:
    """Test health check endpoint."""
    print("\n" + "="*60)
    print("Testing: Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/health")
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Response: {data}")
        
        if data.get('model_loaded'):
            print("✓ Model is loaded")
            return True
        else:
            print("✗ Model is NOT loaded")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_model_info(base_url: str) -> bool:
    """Test model info endpoint."""
    print("\n" + "="*60)
    print("Testing: Model Info")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/model/info")
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Model Version: {data['model_version']}")
        print(f"✓ Number of Features: {data['n_features']}")
        print(f"✓ Class Labels: {data['class_labels']}")
        print(f"✓ Created At: {data['created_at']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_features(base_url: str) -> Dict[str, Any]:
    """Test features endpoint and return feature list."""
    print("\n" + "="*60)
    print("Testing: Features List")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/features")
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Number of Features: {data['n_features']}")
        print(f"✓ Required Features:")
        for i, feature in enumerate(data['required_features'], 1):
            print(f"   {i:2d}. {feature}")
        
        return data
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return {}


def test_predict(base_url: str) -> bool:
    """Test single prediction endpoint."""
    print("\n" + "="*60)
    print("Testing: Single Prediction")
    print("="*60)
    
    # Example feature values
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
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"features": features}
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Predicted Class: {data['predicted_class']}")
        print(f"✓ R-Product Range: {data['r_product_range']}")
        print(f"✓ Confidence: {data['confidence']:.3f}")
        print(f"✓ Class Probabilities:")
        for cls, prob in data['class_probabilities'].items():
            print(f"   {cls}: {prob:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False


def test_batch_predict(base_url: str) -> bool:
    """Test batch prediction endpoint."""
    print("\n" + "="*60)
    print("Testing: Batch Prediction")
    print("="*60)
    
    # Example batch of 2 samples
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
    
    try:
        response = requests.post(
            f"{base_url}/predict/batch",
            json={"samples": samples}
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Total Samples: {data['total_samples']}")
        print(f"✓ Predictions:")
        
        for i, pred in enumerate(data['predictions'], 1):
            print(f"\n   Sample {i}:")
            print(f"   - Class: {pred['predicted_class']}")
            print(f"   - Range: {pred['r_product_range']}")
            print(f"   - Confidence: {pred['confidence']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False


def test_error_handling(base_url: str) -> bool:
    """Test error handling with invalid input."""
    print("\n" + "="*60)
    print("Testing: Error Handling")
    print("="*60)
    
    # Test with missing features
    invalid_features = {
        "fukui_radical_max_1": 0.15,
        # Missing all other features
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"features": invalid_features}
        )
        
        if response.status_code == 400:
            print(f"✓ Correctly returned 400 Bad Request for invalid input")
            print(f"✓ Error message: {response.json().get('detail', 'N/A')}")
            return True
        else:
            print(f"✗ Expected 400 status code, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_preprocess_solvent(base_url: str) -> bool:
    """Test solvent preprocessing endpoint."""
    print("\n" + "="*60)
    print("Testing: Preprocess Solvent")
    print("="*60)
    
    try:
        response = requests.post(
            f"{base_url}/preprocess/solvent",
            json={"solvent_name": "toluene"}
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Solvent Name: {data['solvent_name']}")
        print(f"✓ SMILES: {data['solvent_smiles']}")
        print(f"✓ Success: {data['success']}")
        
        if data['success']:
            print(f"✓ Features:")
            for key, value in data['features'].items():
                print(f"   {key}: {value}")
        else:
            print(f"⚠ Error: {data.get('error', 'N/A')}")
        
        return data['success']
        
    except Exception as e:
        print(f"✗ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False


def test_preprocess_monomer(base_url: str) -> bool:
    """Test monomer preprocessing endpoint."""
    print("\n" + "="*60)
    print("Testing: Preprocess Monomer")
    print("="*60)
    
    try:
        # Test with a common monomer (likely in cache)
        response = requests.post(
            f"{base_url}/preprocess/monomer",
            json={"monomer_name": "styrene"}
        )
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Monomer Name: {data['monomer_name']}")
        print(f"✓ SMILES: {data['monomer_smiles']}")
        print(f"✓ Success: {data['success']}")
        print(f"✓ From Cache: {data['from_cache']}")
        
        if data['success']:
            print(f"✓ Features:")
            for key, value in data['features'].items():
                print(f"   {key}: {value}")
        else:
            print(f"⚠ Error: {data.get('error', 'N/A')}")
        
        return data['success']
        
    except Exception as e:
        print(f"✗ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False


def test_embeddings_method(base_url: str) -> bool:
    """Test get method embeddings endpoint."""
    print("\n" + "="*60)
    print("Testing: Get Method Embeddings")
    print("="*60)
    
    try:
        # Use a known example method that exists in embeddings JSON
        method_name = "solvent"
        encoded_method = quote(method_name, safe='')
        response = requests.get(f"{base_url}/embeddings/method/{encoded_method}")
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Method: {method_name}")
        print(f"✓ PCA_1: {data.get('pca_1')}")
        print(f"✓ PCA_2: {data.get('pca_2')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False


def test_embeddings_polytype(base_url: str) -> bool:
    """Test get polytype embeddings endpoint."""
    print("\n" + "="*60)
    print("Testing: Get Polytype Embeddings")
    print("="*60)
    
    try:
        # Use a known example polytype that exists in embeddings JSON
        polytype_name = "free radical"
        encoded_polytype = quote(polytype_name, safe='')
        response = requests.get(f"{base_url}/embeddings/polytype/{encoded_polytype}")
        response.raise_for_status()
        data = response.json()
        
        print(f"✓ Status: {response.status_code}")
        print(f"✓ Polytype: {polytype_name}")
        print(f"✓ PCA_1: {data.get('pca_1')}")
        print(f"✓ PCA_2: {data.get('pca_2')}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"   Response: {e.response.text}")
        return False


def main():
    """Run all API tests."""
    parser = argparse.ArgumentParser(description="Test Copolymerization API")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the API"
    )
    args = parser.parse_args()
    
    base_url = args.url.rstrip('/')
    
    print("\n" + "="*60)
    print("COPOLYMERIZATION API TEST SUITE")
    print("="*60)
    print(f"Testing API at: {base_url}")
    
    # Check if API is reachable
    try:
        response = requests.get(base_url)
        print(f"✓ API is reachable")
    except Exception as e:
        print(f"✗ Cannot reach API at {base_url}")
        print(f"   Error: {e}")
        print("\nMake sure the API is running:")
        print("   cd copol_prediction/api")
        print("   python app.py")
        sys.exit(1)
    
    # Run tests
    results = {
        "Health Check": test_health(base_url),
        "Model Info": test_model_info(base_url),
        "Features List": test_features(base_url) != {},
        "Single Prediction": test_predict(base_url),
        "Batch Prediction": test_batch_predict(base_url),
        "Error Handling": test_error_handling(base_url),
        "Preprocess Solvent": test_preprocess_solvent(base_url),
        "Preprocess Monomer": test_preprocess_monomer(base_url),
        "Get Method Embeddings": test_embeddings_method(base_url),
        "Get Polytype Embeddings": test_embeddings_polytype(base_url),
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "-"*60)
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    print("="*60)
    
    if failed > 0:
        print("\n⚠️  Some tests failed. Please check the API.")
        sys.exit(1)
    else:
        print("\n✓ All tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

