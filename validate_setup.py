#!/usr/bin/env python3
"""
Validate the Insurance Fraud Detection pipeline.
Run this script to ensure everything is working correctly.
"""

import os
import sys
from pathlib import Path

def check_files():
    """Check if required files exist."""
    print("=" * 60)
    print("CHECKING PROJECT FILES")
    print("=" * 60)
    
    required_files = [
        'preprocessing.py',
        'model_training.py',
        'app.py',
        'templates/index.html',
        'insurance_claims.csv'
    ]
    
    all_exist = True
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False
    
    return all_exist

def check_artifacts():
    """Check if model artifacts exist."""
    print("\n" + "=" * 60)
    print("CHECKING MODEL ARTIFACTS")
    print("=" * 60)
    
    artifact_files = [
        'models/best_model.pkl',
        'models/std_scaler.pkl',
        'models/model_features.pkl'
    ]
    
    all_exist = True
    for file_path in artifact_files:
        exists = os.path.exists(file_path)
        status = "✓" if exists else "✗"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Model artifacts not found!")
        print("Run: python model_training.py")
    
    return all_exist

def test_loading():
    """Test loading model artifacts."""
    print("\n" + "=" * 60)
    print("TESTING ARTIFACT LOADING")
    print("=" * 60)
    
    try:
        import joblib
        import numpy as np
        
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/std_scaler.pkl')
        features = joblib.load('models/model_features.pkl')
        
        print(f"✓ Model loaded: {type(model).__name__}")
        print(f"✓ Scaler loaded: {type(scaler).__name__}")
        print(f"✓ Features loaded: {len(features)} features")
        print(f"\nFeatures: {features}")
        
        # Test prediction
        dummy_input = np.array([[1.0] * len(features)])
        scaled = scaler.transform(dummy_input)
        prediction = model.predict(scaled)
        probability = model.predict_proba(scaled)
        
        print(f"\n✓ Test prediction successful!")
        print(f"  Dummy prediction: {prediction[0]}")
        print(f"  Probabilities: [{probability[0][0]:.4f}, {probability[0][1]:.4f}]")
        
        return True
        
    except FileNotFoundError:
        print("✗ Model artifacts not found")
        print("Run: python model_training.py")
        return False
    except Exception as e:
        print(f"✗ Error loading artifacts: {e}")
        return False

def test_imports():
    """Test if required packages are installed."""
    print("\n" + "=" * 60)
    print("TESTING PACKAGE IMPORTS")
    print("=" * 60)
    
    packages = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', 'sklearn'),
        ('flask', 'Flask'),
        ('joblib', 'joblib')
    ]
    
    all_imported = True
    for package_name, import_name in packages:
        try:
            if import_name == 'sklearn':
                import sklearn
            elif import_name == 'Flask':
                from flask import Flask
            else:
                __import__(package_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} - NOT INSTALLED")
            all_imported = False
    
    return all_imported

def main():
    """Run all validation checks."""
    print("\n" + "=" * 70)
    print(" INSURANCE FRAUD DETECTION - VALIDATION SCRIPT")
    print("=" * 70)
    
    # Run checks
    files_ok = check_files()
    imports_ok = test_imports()
    artifacts_ok = check_artifacts()
    
    if artifacts_ok:
        loading_ok = test_loading()
    else:
        loading_ok = False
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    results = [
        ("Project files", files_ok),
        ("Package imports", imports_ok),
        ("Model artifacts", artifacts_ok),
        ("Artifact loading", loading_ok)
    ]
    
    all_passed = all(result[1] for result in results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} - {name}")
    
    print("=" * 60)
    
    if all_passed:
        print("\n✅ All checks passed!")
        print("\nYou can now:")
        print("  1. Train model: python model_training.py")
        print("  2. Start server: python app.py")
        print("  3. Open browser: http://localhost:5000")
    else:
        print("\n❌ Some checks failed!")
        print("\nPlease fix the issues above before proceeding.")
        sys.exit(1)

if __name__ == "__main__":
    main()
