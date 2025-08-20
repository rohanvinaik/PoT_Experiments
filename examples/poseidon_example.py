"""
Example demonstrating Poseidon hash integration with training provenance.

This example shows:
1. Using Poseidon for ZK-friendly commitments
2. Dual commitment schemes (SHA-256 + Poseidon)
3. Performance comparison
4. Compatibility verification
"""

import sys
import time
import hashlib
import numpy as np
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).parent.parent))

from pot.prototypes.training_provenance_auditor import TrainingProvenanceAuditor
from pot.zk.poseidon import (
    PoseidonHash, 
    poseidon_hash,
    poseidon_merkle_root,
    PoseidonMerkleTree
)
from pot.zk.field_arithmetic import FieldElement
from pot.zk.commitments import DualCommitment


def benchmark_hash_functions():
    """Benchmark Poseidon vs SHA-256."""
    print("=" * 60)
    print("Hash Function Benchmark")
    print("=" * 60)
    
    # Test data
    small_data = b"Hello, World!"
    medium_data = b"x" * 1024  # 1KB
    large_data = b"y" * 1024 * 10  # 10KB
    
    test_cases = [
        ("Small (13B)", small_data),
        ("Medium (1KB)", medium_data),
        ("Large (10KB)", large_data)
    ]
    
    for name, data in test_cases:
        print(f"\n{name}:")
        
        # SHA-256
        start = time.time()
        for _ in range(1000):
            sha_hash = hashlib.sha256(data).digest()
        sha_time = time.time() - start
        
        # Poseidon
        start = time.time()
        for _ in range(1000):
            pos_hash = poseidon_hash(data)
        pos_time = time.time() - start
        
        print(f"  SHA-256:  {sha_time:.4f}s for 1000 iterations")
        print(f"  Poseidon: {pos_time:.4f}s for 1000 iterations")
        print(f"  Ratio:    {pos_time/sha_time:.2f}x")


def demonstrate_field_arithmetic():
    """Demonstrate field arithmetic operations."""
    print("\n" + "=" * 60)
    print("Field Arithmetic Examples")
    print("=" * 60)
    
    # Create field elements
    a = FieldElement(42)
    b = FieldElement(17)
    
    print(f"\nBasic operations:")
    print(f"  a = {a.value}")
    print(f"  b = {b.value}")
    print(f"  a + b = {(a + b).value}")
    print(f"  a * b = {(a * b).value}")
    print(f"  a - b = {(a - b).value}")
    print(f"  a / b = {(a / b).value}")
    
    # Modular arithmetic
    large = FieldElement(FieldElement.MODULUS - 1)
    print(f"\nModular arithmetic:")
    print(f"  (p-1) + 1 = {(large + FieldElement.one()).value}")
    print(f"  (p-1) + 2 = {(large + FieldElement(2)).value}")
    
    # Field element from bytes
    bytes_data = b"Hello, Field!"
    fe = FieldElement.from_bytes(hashlib.sha256(bytes_data).digest())
    print(f"\nFrom bytes:")
    print(f"  Input: '{bytes_data.decode()}'")
    print(f"  Field element: 0x{fe.value:064x}"[:50] + "...")


def demonstrate_merkle_trees():
    """Compare SHA-256 and Poseidon Merkle trees."""
    print("\n" + "=" * 60)
    print("Merkle Tree Comparison")
    print("=" * 60)
    
    # Create test data
    leaves = [f"Transaction_{i}".encode() for i in range(8)]
    
    # SHA-256 Merkle tree
    print("\nSHA-256 Merkle Tree:")
    from pot.prototypes.training_provenance_auditor import compute_merkle_root
    sha_root = compute_merkle_root(leaves)
    print(f"  Root: {sha_root.hex()[:32]}...")
    
    # Poseidon Merkle tree
    print("\nPoseidon Merkle Tree:")
    pos_tree = PoseidonMerkleTree(leaves)
    pos_root = pos_tree.root()
    print(f"  Root: {pos_root.hex()[:32]}...")
    
    # Generate and verify proof
    print("\nMerkle Proof (Poseidon):")
    proof = pos_tree.proof(0)
    print(f"  Proof length: {len(proof)} elements")
    print(f"  Proof size: {sum(len(p) for p in proof)} bytes")
    
    is_valid = pos_tree.verify(leaves[0], 0, proof)
    print(f"  Verification: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Invalid proof
    is_invalid = pos_tree.verify(b"wrong_leaf", 0, proof)
    print(f"  Wrong leaf: {'✗ Invalid' if not is_invalid else '✓ Rejected'}")


def demonstrate_dual_commitment():
    """Demonstrate dual commitment scheme."""
    print("\n" + "=" * 60)
    print("Dual Commitment Scheme")
    print("=" * 60)
    
    # Create dual commitment
    dual = DualCommitment()
    
    # Create model weights
    weights = np.random.randn(256, 256).astype(np.float32)
    print(f"\nModel weights: {weights.shape} = {weights.size} parameters")
    
    # Commit to weights
    commitment = dual.commit_tensor(weights)
    
    print(f"\nDual commitment:")
    print(f"  SHA-256 root:  {commitment['sha256_root'][:32]}...")
    print(f"  Poseidon root: {commitment['poseidon_root'][:32]}...")
    print(f"  Tensor size:   {len(commitment['tensor_data'])} bytes")
    
    # Verify consistency
    is_consistent = dual.verify_consistency(commitment)
    print(f"  Consistency:   {'✓ Valid' if is_consistent else '✗ Invalid'}")
    
    # Training batch commitment
    batch_inputs = np.random.randn(32, 256).astype(np.float32)
    batch_targets = np.random.randn(32, 10).astype(np.float32)
    
    batch_commitment = dual.commit_batch(batch_inputs, batch_targets)
    print(f"\nBatch commitment:")
    print(f"  SHA-256 root:  {batch_commitment['sha256_root'][:32]}...")
    print(f"  Poseidon root: {batch_commitment['poseidon_root'][:32]}...")
    print(f"  Batch shapes:  inputs={batch_commitment['inputs_shape']}, targets={batch_commitment['targets_shape']}")


def demonstrate_training_auditor():
    """Demonstrate TrainingProvenanceAuditor with Poseidon."""
    print("\n" + "=" * 60)
    print("Training Auditor with Poseidon")
    print("=" * 60)
    
    # Create auditors with different hash functions
    auditor_sha = TrainingProvenanceAuditor(
        model_id="model_sha256",
        hash_function="sha256"
    )
    
    auditor_pos = TrainingProvenanceAuditor(
        model_id="model_poseidon",
        hash_function="poseidon"
    )
    
    print(f"\nAuditor configurations:")
    print(f"  SHA-256:  {auditor_sha.model_id} using {auditor_sha.hash_function}")
    print(f"  Poseidon: {auditor_pos.model_id} using {auditor_pos.hash_function}")
    
    # Log training events
    metrics = {'loss': 0.5, 'accuracy': 0.92}
    
    print(f"\nLogging training events...")
    event_sha = auditor_sha.log_training_event(epoch=1, metrics=metrics)
    event_pos = auditor_pos.log_training_event(epoch=1, metrics=metrics)
    
    print(f"  SHA-256 event hash:  {event_sha.event_hash[:32]}...")
    print(f"  Poseidon event hash: {event_pos.event_hash[:32]}...")
    
    # Compute Merkle roots
    data_blocks = [
        b"checkpoint_1",
        b"checkpoint_2",
        b"checkpoint_3",
        b"checkpoint_4"
    ]
    
    sha_root = auditor_sha._compute_merkle_root(data_blocks)
    pos_root = auditor_pos._compute_merkle_root(data_blocks)
    
    print(f"\nMerkle roots for checkpoints:")
    print(f"  SHA-256:  {sha_root.hex()[:32]}...")
    print(f"  Poseidon: {pos_root.hex()[:32]}...")
    
    # Performance comparison
    print(f"\nPerformance (1000 iterations):")
    
    start = time.time()
    for _ in range(1000):
        _ = auditor_sha._compute_merkle_root(data_blocks)
    sha_time = time.time() - start
    
    start = time.time()
    for _ in range(1000):
        _ = auditor_pos._compute_merkle_root(data_blocks)
    pos_time = time.time() - start
    
    print(f"  SHA-256:  {sha_time:.4f}s")
    print(f"  Poseidon: {pos_time:.4f}s")
    print(f"  Ratio:    {pos_time/sha_time:.2f}x")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 60)
    print("POSEIDON HASH INTEGRATION DEMONSTRATION")
    print("=" * 60)
    
    # Run benchmarks
    benchmark_hash_functions()
    
    # Demonstrate field arithmetic
    demonstrate_field_arithmetic()
    
    # Demonstrate Merkle trees
    demonstrate_merkle_trees()
    
    # Demonstrate dual commitment
    demonstrate_dual_commitment()
    
    # Demonstrate training auditor
    demonstrate_training_auditor()
    
    print("\n" + "=" * 60)
    print("✅ All demonstrations completed successfully!")
    print("=" * 60)
    
    print("\nKey Takeaways:")
    print("1. Poseidon provides ZK-friendly hashing for circuit optimization")
    print("2. Field arithmetic enables native operations in ZK circuits")
    print("3. Dual commitments maintain compatibility while enabling ZK proofs")
    print("4. TrainingProvenanceAuditor supports both SHA-256 and Poseidon")
    print("5. Performance trade-offs exist but are acceptable for ZK use cases")


if __name__ == "__main__":
    main()