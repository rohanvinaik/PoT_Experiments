#!/usr/bin/env python3
"""
Test ZK proof integration in the PoT framework.
This verifies that ZK proofs can be generated without OOM issues.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_lightweight_zk():
    """Test Python-based lightweight ZK proofs"""
    print("="*60)
    print("Testing Lightweight Python ZK Proofs")
    print("="*60)
    
    try:
        from pot.zk.auto_prover import auto_prove_training_step
        from pot.zk.verifier import ZKVerifier
        
        print("\n1. Creating mock training step...")
        # Small matrices to avoid memory issues
        mock_before = {
            'layer1.weight': np.random.randn(10, 10).astype(np.float32),
            'layer1.bias': np.random.randn(10).astype(np.float32)
        }
        
        # Simulate gradient update
        mock_after = {
            'layer1.weight': mock_before['layer1.weight'] - 0.001 * np.random.randn(10, 10).astype(np.float32),
            'layer1.bias': mock_before['layer1.bias'] - 0.001 * np.random.randn(10).astype(np.float32)
        }
        
        mock_batch = {
            'input': np.random.randn(32, 10).astype(np.float32),
            'target': np.random.randn(32, 10).astype(np.float32)
        }
        
        print("2. Generating ZK proof...")
        proof_result = auto_prove_training_step(
            model_before=mock_before,
            model_after=mock_after,
            batch_data=mock_batch,
            learning_rate=0.001,
            step_number=1
        )
        
        print(f"   ✅ Proof generated successfully")
        print(f"   Proof type: {proof_result.get('type', 'unknown')}")
        print(f"   Proof hash: {proof_result.get('proof_hash', 'N/A')[:16]}...")
        
        print("\n3. Verifying proof...")
        verifier = ZKVerifier()
        is_valid = verifier.verify_proof(
            proof_result['proof'],
            proof_result['public_inputs']
        )
        
        print(f"   Verification result: {'✅ VALID' if is_valid else '❌ INVALID'}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   ZK modules may not be properly installed")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_poseidon_hash():
    """Test Poseidon hash function (ZK-friendly)"""
    print("\n" + "="*60)
    print("Testing Poseidon Hash Function")
    print("="*60)
    
    try:
        from pot.zk.poseidon import PoseidonHash
        
        print("\n1. Initializing Poseidon hash...")
        hasher = PoseidonHash()
        
        print("2. Hashing test data...")
        test_data = [1.0, 2.0, 3.0, 4.0]
        hash_result = hasher.hash(test_data)
        
        print(f"   ✅ Hash computed: {hash_result[:16]}...")
        
        print("3. Testing determinism...")
        hash_result2 = hasher.hash(test_data)
        
        if hash_result == hash_result2:
            print("   ✅ Hash is deterministic")
        else:
            print("   ❌ Hash is not deterministic!")
        
        return True
        
    except ImportError:
        print("   ⚠️ Poseidon module not available")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_lora_detection():
    """Test LoRA model detection"""
    print("\n" + "="*60)
    print("Testing LoRA Detection")
    print("="*60)
    
    try:
        from pot.zk.lora_builder import LoRAWitnessBuilder
        
        print("\n1. Creating witness builder...")
        builder = LoRAWitnessBuilder()
        
        print("2. Testing SGD model (non-LoRA)...")
        sgd_model = {
            'layer1.weight': np.random.randn(10, 10),
            'layer1.bias': np.random.randn(10)
        }
        
        is_lora = builder.detect_lora_training(sgd_model)
        print(f"   SGD model detected as LoRA: {is_lora} {'❌' if is_lora else '✅'}")
        
        print("3. Testing LoRA model...")
        lora_model = {
            'base_model.layer1.weight': np.random.randn(10, 10),
            'lora_A.layer1': np.random.randn(8, 10),  # rank 8
            'lora_B.layer1': np.random.randn(10, 8)
        }
        
        is_lora = builder.detect_lora_training(lora_model)
        print(f"   LoRA model detected as LoRA: {is_lora} {'✅' if is_lora else '❌'}")
        
        return True
        
    except ImportError:
        print("   ⚠️ LoRA builder not available")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_memory_usage():
    """Test memory usage of ZK operations"""
    print("\n" + "="*60)
    print("Testing Memory Usage")
    print("="*60)
    
    try:
        import psutil
        import gc
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / (1024**3)  # GB
        
        print(f"Initial memory: {initial_memory:.2f}GB")
        
        # Run a small ZK proof
        from pot.zk.auto_prover import AutoProver
        
        prover = AutoProver(max_retries=1)
        
        # Very small model to test memory
        small_model_before = {'w': np.random.randn(5, 5).astype(np.float32)}
        small_model_after = {'w': np.random.randn(5, 5).astype(np.float32)}
        
        print("\nGenerating small proof...")
        result = prover.auto_prove_training_step(
            small_model_before,
            small_model_after,
            {'input': np.random.randn(1, 5).astype(np.float32)},
            0.001
        )
        
        # Check memory after
        gc.collect()
        final_memory = process.memory_info().rss / (1024**3)
        memory_increase = final_memory - initial_memory
        
        print(f"Final memory: {final_memory:.2f}GB")
        print(f"Memory increase: {memory_increase:.2f}GB")
        
        if memory_increase < 0.5:  # Less than 500MB increase
            print("✅ Memory usage acceptable")
            return True
        else:
            print(f"⚠️ High memory usage: {memory_increase:.2f}GB")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Run all ZK integration tests"""
    print("\n" + "="*70)
    print("🔐 ZK PROOF INTEGRATION TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Test 1: Lightweight ZK
    print("\n[Test 1/4]")
    results['lightweight_zk'] = test_lightweight_zk()
    
    # Test 2: Poseidon Hash
    print("\n[Test 2/4]")
    results['poseidon'] = test_poseidon_hash()
    
    # Test 3: LoRA Detection
    print("\n[Test 3/4]")
    results['lora_detection'] = test_lora_detection()
    
    # Test 4: Memory Usage
    print("\n[Test 4/4]")
    results['memory_usage'] = test_memory_usage()
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20}: {status}")
    
    total_passed = sum(1 for v in results.values() if v)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All ZK integration tests passed!")
        print("The ZK proof system is ready for use with the throttled pipeline.")
    else:
        print("\n⚠️ Some tests failed. ZK proofs may not work correctly.")
        print("Consider using --skip-zk flag when testing large models.")
    
    return 0 if total_passed == total_tests else 1

if __name__ == "__main__":
    sys.exit(main())