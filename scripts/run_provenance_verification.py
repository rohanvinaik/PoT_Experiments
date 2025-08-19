#!/usr/bin/env python3
"""
Provenance Verification Test
"""

import os
import sys
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("PROVENANCE VERIFICATION")
print("=" * 60)

try:
    from pot.security.provenance_auditor import ProvenanceAuditor
    
    print("Testing provenance with Merkle tree...")
    
    # Create test training data
    training_events = [
        {"epoch": 0, "loss": 2.5, "accuracy": 0.45, "timestamp": time.time()},
        {"epoch": 1, "loss": 1.8, "accuracy": 0.62, "timestamp": time.time() + 1},
        {"epoch": 2, "loss": 1.2, "accuracy": 0.78, "timestamp": time.time() + 2},
        {"epoch": 3, "loss": 0.9, "accuracy": 0.85, "timestamp": time.time() + 3},
        {"epoch": 4, "loss": 0.7, "accuracy": 0.89, "timestamp": time.time() + 4}
    ]
    
    # Initialize auditor
    auditor = ProvenanceAuditor()
    
    # Add events
    for event in training_events:
        auditor.add_training_event(event)
    
    # Get Merkle root
    merkle_root = auditor.get_merkle_root()
    print(f"\nMerkle root: {merkle_root[:32]}...")
    
    # Test verification for multiple events
    verification_results = []
    
    for idx in [0, 2, 4]:  # Test first, middle, and last
        proof = auditor.get_merkle_proof(idx)
        is_valid = auditor.verify_merkle_proof(
            training_events[idx],
            proof,
            merkle_root
        )
        
        print(f"\nEvent {idx} (epoch {training_events[idx]['epoch']}):")
        print(f"  Loss: {training_events[idx]['loss']:.2f}")
        print(f"  Accuracy: {training_events[idx]['accuracy']:.2f}")
        print(f"  Proof nodes: {len(proof)}")
        print(f"  Verification: {'✓ VALID' if is_valid else '✗ INVALID'}")
        
        verification_results.append({
            "event_idx": idx,
            "epoch": training_events[idx]["epoch"],
            "verified": is_valid,
            "proof_length": len(proof)
        })
    
    # Save results
    Path("experimental_results").mkdir(exist_ok=True)
    output = {
        "test": "provenance_verification",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "merkle_root": merkle_root,
        "total_events": len(training_events),
        "verifications": verification_results,
        "all_valid": all(v["verified"] for v in verification_results)
    }
    
    with open("experimental_results/provenance_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: experimental_results/provenance_results.json")
    
    if output["all_valid"]:
        print("\n✅ Provenance verification completed - all proofs valid")
    else:
        print("\n❌ Provenance verification failed - some proofs invalid")
        sys.exit(1)
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)