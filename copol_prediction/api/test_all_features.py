#!/usr/bin/env python3
"""
Comprehensive test script for all API features.

This script tests:
1. Health check
2. Model info
3. Preprocessing (solvent, monomer, all)
4. Predictions (single and batch)
5. Nearest neighbors (baseline lookup)
6. Reaction optimization (3x3 grid)
7. DOI checking
8. Embeddings

Usage:
    python test_all_features.py [--url URL]
"""

import argparse
import requests
import json
import sys
from typing import Dict, Any


def test_health_check(base_url: str) -> bool:
    """Test health check endpoint."""
    print("\n" + "="*60)
    print("1. HEALTH CHECK")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Status: {data['status']}")
            print(f"✓ Model loaded: {data['model_loaded']}")
            return True
        else:
            print(f"✗ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_model_info(base_url: str) -> bool:
    """Test model info endpoint."""
    print("\n" + "="*60)
    print("2. MODEL INFO")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/model/info")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Model version: {data['model_version']}")
            print(f"✓ Features: {data['n_features']}")
            print(f"✓ Class labels: {data['class_labels']}")
            return True
        else:
            print(f"✗ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_preprocess_solvent(base_url: str) -> bool:
    """Test solvent preprocessing."""
    print("\n" + "="*60)
    print("3. PREPROCESS SOLVENT")
    print("="*60)
    
    try:
        response = requests.post(
            f"{base_url}/preprocess/solvent",
            json={"solvent_smiles": "CCO"}  # Ethanol
        )
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Success: {data['success']}")
            print(f"✓ Features calculated: {len([k for k, v in data['features'].items() if v is not None])}")
            return True
        else:
            print(f"✗ Error: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_preprocess_all(base_url: str) -> Dict[str, Any]:
    """Test combined preprocessing with all features."""
    print("\n" + "="*60)
    print("4. PREPROCESS ALL (with nearest neighbors)")
    print("="*60)
    
    data = {
        "monomer1_smiles": "C=CC1=CC=CC=C1",  # Styrene
        "monomer2_smiles": "C=C(C)C(=O)OCCO",  # 2-hydroxyethyl methacrylate
        "solvent_smiles": "CCO",  # Ethanol
        "method": "solvent",
        "polytype": "free radical",
        "temperature": 60.0
    }
    
    try:
        response = requests.post(f"{base_url}/preprocess_all", json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success: {result['success']}")
            print(f"✓ Features: {len(result['features'])}")
            
            # Check nearest neighbors
            if result.get('nearest_neighbors'):
                print(f"✓ Nearest neighbors: {len(result['nearest_neighbors'])} found")
                if len(result['nearest_neighbors']) > 0:
                    nn = result['nearest_neighbors'][0]
                    print(f"  Top match: {nn['monomer1_name']} + {nn['monomer2_name']} "
                          f"(similarity: {nn['similarity']:.3f})")
            else:
                print("⚠ Nearest neighbors: Not available")

            return result
        else:
            print(f"✗ Error: {response.status_code}")
            print(response.text)
            return {}
    except Exception as e:
        print(f"✗ Error: {e}")
        return {}


def test_predict(base_url: str, features: Dict[str, float]) -> bool:
    """Test single prediction."""
    print("\n" + "="*60)
    print("5. PREDICT (Single)")
    print("="*60)
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json={"features": features}
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Predicted class: {result['predicted_class']} ({result['predicted_class_name']})")
            print(f"✓ Confidence: {result['confidence']:.4f}")
            return True
        else:
            print(f"✗ Error: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_predict_batch(base_url: str, features: Dict[str, float]) -> bool:
    """Test batch prediction."""
    print("\n" + "="*60)
    print("6. PREDICT BATCH")
    print("="*60)
    
    try:
        response = requests.post(
            f"{base_url}/predict/batch",
            json={
                "samples": [features, features]  # Two identical samples
            }
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Total samples: {result['total_samples']}")
            print(f"✓ Predictions: {len(result['predictions'])}")
            return True
        else:
            print(f"✗ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_optimize_reaction(base_url: str) -> bool:
    """Test reaction optimization."""
    print("\n" + "="*60)
    print("7. OPTIMIZE REACTION (3x3 Grid)")
    print("="*60)
    
    data = {
        "monomer1_smiles": "C=CC1=CC=CC=C1",
        "monomer2_smiles": "C=C(C)C(=O)OCCO",
        "solvent_smiles": "CCO",
        "method": "solvent",
        "polytype": "free radical",
        "temperature": 60.0,
        "temperature_step": 20.0,
        "n_solvents": 3
    }
    
    try:
        response = requests.post(f"{base_url}/optimize_reaction", json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"✓ Success: {result['success']}")
            print(f"✓ Predictions: {len(result['predictions'])}")
            print(f"✓ Base temperature: {result['base_temperature']}°C")
            print(f"✓ Temperature step: {result['temperature_step']}°C")

            if result['predictions']:
                best = max(result['predictions'], key=lambda x: x['confidence'])
                print(f"\n  Best prediction:")
                print(f"    Temperature: {best['temperature']}°C")
                print(f"    Solvent: {best['solvent_name']}")
                print(f"    Class: {best['predicted_class']}")
                print(f"    Confidence: {best['confidence']:.4f}")

            return True
        else:
            print(f"✗ Error: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_check_doi(base_url: str) -> bool:
    """Test DOI checking."""
    print("\n" + "="*60)
    print("8. CHECK DOI")
    print("="*60)
    
    try:
        response = requests.post(
            f"{base_url}/check_doi",
            json={"doi": "10.1016/0014-3057(84)90010-7"}
        )
        if response.status_code == 200:
            result = response.json()
            print(f"✓ DOI exists: {result['exists']}")
            print(f"✓ Normalized DOI: {result['normalized_doi']}")
            return True
        else:
            print(f"✗ Error: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_embeddings(base_url: str) -> bool:
    """Test embeddings endpoints."""
    print("\n" + "="*60)
    print("9. EMBEDDINGS")
    print("="*60)
    
    try:
        # Test method embedding
        response = requests.get(f"{base_url}/embeddings/method/solvent")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Method embedding: {list(data.keys())}")
        else:
            print(f"✗ Method embedding error: {response.status_code}")
            return False
        
        # Test polytype embedding
        response = requests.get(f"{base_url}/embeddings/polytype/free radical")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Polytype embedding: {list(data.keys())}")
            return True
        else:
            print(f"✗ Polytype embedding error: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    """Run all tests."""
    parser = argparse.ArgumentParser(description="Test all API features")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)"
    )
    args = parser.parse_args()
    
    base_url = args.url
    
    print("="*60)
    print("COMPREHENSIVE API TEST SUITE")
    print("="*60)
    print(f"\nTesting API at: {base_url}")
    print("\nMake sure the API is running!")
    print("Start with: python app.py")
    
    results = {}
    
    # Test 1: Health check
    results['health'] = test_health_check(base_url)
    
    # Test 2: Model info
    results['model_info'] = test_model_info(base_url)
    
    # Test 3: Preprocess solvent
    results['preprocess_solvent'] = test_preprocess_solvent(base_url)
    
    # Test 4: Preprocess all (get features for later tests)
    preprocess_result = test_preprocess_all(base_url)
    results['preprocess_all'] = bool(preprocess_result.get('success', False))
    features = preprocess_result.get('features', {})
    
    # Test 5: Predict
    if features:
        results['predict'] = test_predict(base_url, features)
        results['predict_batch'] = test_predict_batch(base_url, features)
    else:
        print("\n⚠ Skipping prediction tests (no features available)")
        results['predict'] = False
        results['predict_batch'] = False

    # Test 7: Optimize reaction
    results['optimize_reaction'] = test_optimize_reaction(base_url)

    # Test 8: Check DOI
    results['check_doi'] = test_check_doi(base_url)

    # Test 9: Embeddings
    results['embeddings'] = test_embeddings(base_url)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Error: Could not connect to API")
        print("   Make sure the API is running at the specified URL!")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
