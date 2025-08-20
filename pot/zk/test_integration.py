"""
Integration test for Python-Rust ZK prover interoperability.
"""

import sys
import numpy as np
from pathlib import Path
import hashlib
import subprocess
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))

from pot.zk.zk_types import SGDStepStatement, SGDStepWitness
from prover import SGDZKProver, ProverConfig
from verifier import SGDZKVerifier, VerifierConfig
from builder import ZKWitnessBuilder


def test_end_to_end_proof():
    """Test complete proof generation and verification pipeline."""
    
    print("🧪 Testing ZK Proof Integration")
    print("=" * 50)
    
    # 1. Create mock model weights (16x4 matrix)
    print("\n1️⃣ Creating mock model weights...")
    np.random.seed(42)
    
    weights_before = {
        "layer1": np.random.randn(16, 4) * 0.1
    }
    
    # Simulate SGD update
    learning_rate = 0.01
    gradients = np.random.randn(16, 4) * 0.01
    
    weights_after = {
        "layer1": weights_before["layer1"] - learning_rate * gradients
    }
    
    print(f"   ✓ Created 16x4 weight matrix")
    print(f"   ✓ Learning rate: {learning_rate}")
    
    # 2. Create batch data
    print("\n2️⃣ Creating training batch...")
    batch_data = {
        "inputs": np.random.randn(1, 16),  # 1 sample, 16 features
        "targets": np.random.randn(1, 4)   # 1 sample, 4 outputs
    }
    print(f"   ✓ Batch size: 1")
    print(f"   ✓ Input features: 16")
    print(f"   ✓ Output features: 4")
    
    # 3. Build witness using Merkle tree
    print("\n3️⃣ Building witness with Merkle trees...")
    builder = ZKWitnessBuilder()
    
    # Extract first 64 weights (entire 16x4 matrix)
    layer_indices = [("layer1", i) for i in range(64)]
    
    witness_data = builder.extract_sgd_update_witness(
        weights_before=weights_before,
        weights_after=weights_after,
        batch_data=batch_data,
        learning_rate=learning_rate,
        layer_indices=layer_indices
    )
    
    print(f"   ✓ Extracted {len(witness_data['weights_before'])} weight values")
    print(f"   ✓ Generated Merkle root for W_t: {witness_data['w_t_root'].hex()[:16]}...")
    print(f"   ✓ Generated Merkle root for W_t+1: {witness_data['w_t1_root'].hex()[:16]}...")
    print(f"   ✓ Generated batch commitment: {witness_data['batch_root'].hex()[:16]}...")
    
    # 4. Create SGDStepStatement and SGDStepWitness
    print("\n4️⃣ Creating statement and witness...")
    
    # Create public statement
    statement = SGDStepStatement(
        W_t_root=witness_data['w_t_root'],
        batch_root=witness_data['batch_root'],
        hparams_hash=hashlib.sha256(b"hyperparams").digest(),
        W_t1_root=witness_data['w_t1_root'],
        step_nonce=123,
        step_number=456,
        epoch=1
    )
    
    # Create private witness
    witness = SGDStepWitness(
        weights_before=witness_data['weights_before'],
        weights_after=witness_data['weights_after'],
        batch_inputs=witness_data['batch_inputs'],
        batch_targets=witness_data['batch_targets'],
        learning_rate=learning_rate,
        loss_value=witness_data['loss_value']
    )
    
    print(f"   ✓ Statement created with step #{statement.step_number}")
    print(f"   ✓ Witness created with loss value: {witness.loss_value:.4f}")
    
    # 5. Check if Rust binaries exist
    print("\n5️⃣ Checking Rust binaries...")
    prover_binary = Path(__file__).parent / "prover_halo2/target/release/prove_sgd_stdin"
    verifier_binary = Path(__file__).parent / "prover_halo2/target/release/verify_sgd_stdin"
    
    if not prover_binary.exists() or not verifier_binary.exists():
        print("   ⚠️ Rust binaries not found. Building them now...")
        build_cmd = f"cd {Path(__file__).parent / 'prover_halo2'} && cargo build --release --bin prove_sgd_stdin --bin verify_sgd_stdin"
        result = subprocess.run(build_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"   ❌ Build failed: {result.stderr}")
            return False
        print("   ✓ Rust binaries built successfully")
    else:
        print("   ✓ Rust binaries found")
    
    # 6. Generate proof
    print("\n6️⃣ Generating ZK proof...")
    prover_config = ProverConfig(params_k=10)  # Use smaller k for testing
    prover = SGDZKProver(prover_config)
    
    start_time = time.time()
    try:
        proof = prover.prove_sgd_step(statement, witness)
        proof_time = time.time() - start_time
        print(f"   ✓ Proof generated in {proof_time:.2f} seconds")
        print(f"   ✓ Proof size: {len(proof)} bytes")
    except Exception as e:
        print(f"   ❌ Proof generation failed: {e}")
        return False
    
    # 7. Verify proof
    print("\n7️⃣ Verifying ZK proof...")
    verifier_config = VerifierConfig(params_k=10)
    verifier = SGDZKVerifier(verifier_config)
    
    start_time = time.time()
    try:
        is_valid = verifier.verify_sgd_step(statement, proof)
        verify_time = time.time() - start_time
        
        if is_valid:
            print(f"   ✅ Proof verified successfully in {verify_time:.2f} seconds")
        else:
            print(f"   ❌ Proof verification failed")
            return False
    except Exception as e:
        print(f"   ❌ Verification error: {e}")
        return False
    
    # 8. Test tampering detection
    print("\n8️⃣ Testing tampering detection...")
    
    # Create tampered statement (different root)
    tampered_statement = SGDStepStatement(
        W_t_root=hashlib.sha256(b"tampered").digest(),  # Wrong root!
        batch_root=statement.batch_root,
        hparams_hash=statement.hparams_hash,
        W_t1_root=statement.W_t1_root,
        step_nonce=statement.step_nonce,
        step_number=statement.step_number,
        epoch=statement.epoch
    )
    
    try:
        is_valid_tampered = verifier.verify_sgd_step(tampered_statement, proof)
        if not is_valid_tampered:
            print(f"   ✅ Tampered statement correctly rejected")
        else:
            print(f"   ❌ Tampered statement incorrectly accepted!")
            return False
    except Exception as e:
        print(f"   ✅ Tampered statement rejected with error: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("✅ All tests passed!")
    print("\nSummary:")
    print(f"  • Model: 16x4 weight matrix")
    print(f"  • Batch: 1 sample with 16 inputs, 4 outputs")
    print(f"  • Proof size: {len(proof)} bytes")
    print(f"  • Proof generation: {proof_time:.2f}s")
    print(f"  • Verification: {verify_time:.2f}s")
    print(f"  • Tampering detection: ✓")
    
    return True


def test_convenience_functions():
    """Test the convenience functions."""
    print("\n🧪 Testing Convenience Functions")
    print("=" * 50)
    
    from prover import prove_sgd_step
    from verifier import verify_sgd_step
    
    # Create simple test data
    statement = SGDStepStatement(
        W_t_root=b"root1" * 8,
        batch_root=b"root2" * 8,
        hparams_hash=b"hash3" * 8,
        W_t1_root=b"root4" * 8,
        step_nonce=1,
        step_number=2,
        epoch=3
    )
    
    witness = SGDStepWitness(
        weights_before=[0.1] * 64,
        weights_after=[0.09] * 64,
        batch_inputs=[0.5] * 16,
        batch_targets=[1.0] * 4,
        learning_rate=0.01,
        loss_value=0.5
    )
    
    try:
        # Test proving
        config = ProverConfig(params_k=10)
        proof = prove_sgd_step(statement, witness, config)
        print(f"   ✓ prove_sgd_step() generated {len(proof)} byte proof")
        
        # Test verification
        is_valid = verify_sgd_step(statement, proof, VerifierConfig(params_k=10))
        if is_valid:
            print(f"   ✓ verify_sgd_step() validated proof")
        else:
            print(f"   ❌ verify_sgd_step() failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    print("\n✅ Convenience functions working correctly")
    return True


if __name__ == "__main__":
    # Run tests
    success = test_end_to_end_proof()
    
    if success:
        success = test_convenience_functions()
    
    if success:
        print("\n🎉 All integration tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed")
        exit(1)