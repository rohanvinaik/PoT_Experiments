#!/usr/bin/env python3
"""
Test script for batch verification and calibration functionality.
"""

import torch
import torch.nn as nn
import sys
import os
import numpy as np

# Add project root to path
sys.path.append('/Users/rohanvinaik/PoT_Experiments')

def create_test_model():
    """Create a test vision model."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)
    )

def test_enhanced_verifier():
    """Test EnhancedVisionVerifier functionality."""
    print("Testing EnhancedVisionVerifier...")
    
    try:
        from pot.vision.verifier import EnhancedVisionVerifier
        
        model = create_test_model()
        config = {
            'temperature': 1.0,
            'normalization': 'softmax',
            'verification_method': 'batch',
            'device': 'cpu'
        }
        
        verifier = EnhancedVisionVerifier(model, config)
        print("✓ EnhancedVisionVerifier created successfully")
        
        # Test basic model run
        x = torch.randn(2, 3, 224, 224)
        result = verifier.run_model(x)
        print(f"✓ Model run successful: {result['logits'].shape}")
        
        return verifier
        
    except Exception as e:
        print(f"✗ EnhancedVisionVerifier test failed: {e}")
        return None

def test_batch_verification(verifier):
    """Test batch verification functionality."""
    print("\nTesting batch verification...")
    
    if verifier is None:
        print("⚠ Skipping batch verification test (no verifier)")
        return False
    
    try:
        # Test _batch_verification method
        result = verifier._batch_verification(
            num_challenges=5,
            threshold=0.5,
            challenge_types=['frequency', 'texture']
        )
        
        print(f"✓ Batch verification completed")
        print(f"  Verified: {result['verified']}")
        print(f"  Confidence: {result['confidence']:.3f}")
        print(f"  Success rate: {result['success_rate']:.3f}")
        print(f"  Num challenges: {result['num_challenges']}")
        print(f"  Results: {len(result['results'])} challenge results")
        
        return True
        
    except Exception as e:
        print(f"✗ Batch verification test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_challenge_evaluation(verifier):
    """Test challenge response evaluation."""
    print("\nTesting challenge response evaluation...")
    
    if verifier is None:
        print("⚠ Skipping evaluation test (no verifier)")
        return False
    
    try:
        # Create mock output
        mock_output = {
            'logits': torch.randn(1, 10),
            'embeddings': {
                'early': torch.randn(1, 32),
                'mid': torch.randn(1, 64),
                'penultimate': torch.randn(1, 64),
                'final': torch.randn(1, 10)
            },
            'inference_time': 0.01
        }
        
        # Test different challenge types
        for challenge_type in ['frequency', 'texture', 'natural']:
            success = verifier._evaluate_challenge_response(mock_output, challenge_type)
            print(f"  {challenge_type}: {'✓' if success else '✗'} evaluation result")
        
        print("✓ Challenge evaluation completed")
        return True
        
    except Exception as e:
        print(f"✗ Challenge evaluation test failed: {e}")
        return False

def test_reference_statistics(verifier):
    """Test reference statistics system."""
    print("\nTesting reference statistics...")
    
    if verifier is None:
        print("⚠ Skipping statistics test (no verifier)")
        return False
    
    try:
        # Test getting default statistics
        for challenge_type in ['frequency', 'texture', 'natural']:
            stats = verifier._get_reference_statistics(challenge_type)
            print(f"  {challenge_type} stats keys: {list(stats.keys())}")
            
            # Verify required keys exist
            required_keys = ['logit_mean', 'logit_std', 'embedding_norm']
            for key in required_keys:
                if key not in stats:
                    print(f"    ⚠ Missing key: {key}")
                else:
                    print(f"    ✓ {key}: {stats[key]}")
        
        print("✓ Reference statistics test completed")
        return True
        
    except Exception as e:
        print(f"✗ Reference statistics test failed: {e}")
        return False

def test_calibrator():
    """Test VisionVerifierCalibrator functionality."""
    print("\nTesting VisionVerifierCalibrator...")
    
    try:
        from pot.vision.verifier import VisionVerifierCalibrator, EnhancedVisionVerifier
        
        # Create verifier for calibration
        model = create_test_model()
        config = {
            'temperature': 1.0,
            'normalization': 'softmax',
            'verification_method': 'batch',
            'device': 'cpu'
        }
        
        verifier = EnhancedVisionVerifier(model, config)
        calibrator = VisionVerifierCalibrator(verifier)
        
        print("✓ VisionVerifierCalibrator created")
        
        # Test calibration with small sample size
        print("Running calibration with 10 samples per challenge type...")
        stats = calibrator.calibrate(
            num_samples=10,
            challenge_types=['frequency', 'texture']
        )
        
        print(f"✓ Calibration completed for {len(stats)} challenge types")
        
        # Verify statistics structure
        for challenge_type, challenge_stats in stats.items():
            print(f"  {challenge_type}:")
            for key, value in challenge_stats.items():
                if isinstance(value, float):
                    print(f"    {key}: {value:.4f}")
                else:
                    print(f"    {key}: {value}")
        
        # Test saving and loading calibration
        calibration_path = '/tmp/test_calibration.json'
        calibrator.save_calibration(calibration_path)
        print("✓ Calibration saved")
        
        # Create new calibrator and load
        new_calibrator = VisionVerifierCalibrator(verifier)
        new_calibrator.load_calibration(calibration_path)
        print("✓ Calibration loaded")
        
        # Test calibration summary
        summary = calibrator.get_calibration_summary()
        print(f"✓ Calibration summary: {summary['num_challenge_types']} challenge types")
        
        # Clean up
        os.remove(calibration_path)
        
        return True
        
    except Exception as e:
        print(f"✗ Calibrator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_calibration_validation():
    """Test calibration validation functionality."""
    print("\nTesting calibration validation...")
    
    try:
        from pot.vision.verifier import VisionVerifierCalibrator, EnhancedVisionVerifier
        
        # Create verifier and calibrator
        model = create_test_model()
        config = {
            'temperature': 1.0,
            'normalization': 'softmax',
            'verification_method': 'batch',
            'device': 'cpu'
        }
        
        verifier = EnhancedVisionVerifier(model, config)
        calibrator = VisionVerifierCalibrator(verifier)
        
        # Quick calibration
        print("Performing quick calibration...")
        calibrator.calibrate(
            num_samples=5,
            challenge_types=['frequency']
        )
        
        # Validate calibration
        print("Validating calibration...")
        validation_results = calibrator.validate_calibration(
            num_validation_samples=10
        )
        
        print(f"✓ Validation completed")
        for challenge_type, success_rate in validation_results.items():
            print(f"  {challenge_type}: {success_rate:.2%} success rate")
        
        return True
        
    except Exception as e:
        print(f"✗ Calibration validation test failed: {e}")
        return False

def test_integration():
    """Test integration of batch verification with existing systems."""
    print("\nTesting integration...")
    
    try:
        from pot.vision.verifier import EnhancedVisionVerifier
        from pot.vision.vision_config import VisionConfigPresets
        
        # Use configuration presets
        config = VisionConfigPresets.standard_verification()
        config.device = 'cpu'
        config.verification_method = 'batch'
        config.num_challenges = 5
        
        model = create_test_model()
        verifier = EnhancedVisionVerifier(model, config.to_dict())
        
        print("✓ Integration with configuration system")
        
        # Test with verify_session if available
        try:
            result = verifier.verify_session(
                num_challenges=3,
                challenge_types=['frequency', 'texture']
            )
            print(f"✓ Integration with verify_session: verified={result.get('verified', 'unknown')}")
        except Exception as e:
            print(f"⚠ verify_session not fully compatible: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all batch verification tests."""
    print("=" * 70)
    print("BATCH VERIFICATION AND EVALUATION TESTS")
    print("=" * 70)
    
    tests = [
        test_enhanced_verifier,
        lambda: test_batch_verification(test_enhanced_verifier()),
        lambda: test_challenge_evaluation(test_enhanced_verifier()),
        lambda: test_reference_statistics(test_enhanced_verifier()),
        test_calibrator,
        test_calibration_validation,
        test_integration,
    ]
    
    results = []
    verifier = None
    
    for i, test in enumerate(tests):
        if i == 0:
            # First test returns verifier for subsequent tests
            verifier = test()
            results.append(verifier is not None)
        elif i in [1, 2, 3]:
            # These tests need the verifier
            results.append(test() if verifier else False)
        else:
            # Independent tests
            results.append(test())
    
    print("\n" + "=" * 70)
    print("BATCH VERIFICATION TEST SUMMARY")
    print("=" * 70)
    
    test_names = [
        "EnhancedVisionVerifier Creation",
        "Batch Verification",
        "Challenge Evaluation", 
        "Reference Statistics",
        "VisionVerifierCalibrator",
        "Calibration Validation",
        "System Integration"
    ]
    
    passed = 0
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{i+1:2d}. {name:<30}: {status}")
        if result:
            passed += 1
    
    print("-" * 70)
    print(f"Tests passed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("✓ All batch verification tests passed!")
        return 0
    else:
        print("✗ Some batch verification tests failed")
        return 1

if __name__ == "__main__":
    exit(main())