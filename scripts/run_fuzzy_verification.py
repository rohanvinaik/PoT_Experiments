#!/usr/bin/env python3
"""
Fuzzy Hash Verification Test
"""

import os
import sys
import json
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("FUZZY HASH VERIFICATION")
print("=" * 60)

try:
    from pot.security.fuzzy_hash_verifier import FuzzyHashVerifier, HashAlgorithm
    
    # Test data
    reference = "The quick brown fox jumps over the lazy dog"
    similar = "The quick brown fox jumps over the lazy dogs"
    different = "Hello world, this is a completely different text"
    
    results = {}
    
    # Test each algorithm
    for algo in [HashAlgorithm.SHA256, HashAlgorithm.TLSH, HashAlgorithm.SSDEEP]:
        print(f"\nTesting {algo.value}:")
        
        try:
            verifier = FuzzyHashVerifier(algorithm=algo)
            
            # Compute hashes
            ref_hash = verifier.compute_hash(reference.encode())
            sim_hash = verifier.compute_hash(similar.encode())
            diff_hash = verifier.compute_hash(different.encode())
            
            # Compute similarities
            sim_score = verifier.compute_similarity(ref_hash, sim_hash)
            diff_score = verifier.compute_similarity(ref_hash, diff_hash)
            
            if algo == HashAlgorithm.SHA256:
                print(f"  Type: exact hash (not fuzzy)")
            else:
                print(f"  Type: fuzzy hash")
            
            print(f"  Similar text score: {sim_score:.3f}")
            print(f"  Different text score: {diff_score:.3f}")
            
            results[algo.value] = {
                "similar_score": sim_score,
                "different_score": diff_score,
                "is_fuzzy": algo != HashAlgorithm.SHA256
            }
            
        except ImportError:
            print(f"  ⚠️ {algo.value} library not available")
            results[algo.value] = {"error": "library not available"}
    
    # Save results
    Path("experimental_results").mkdir(exist_ok=True)
    output = {
        "test": "fuzzy_hash_verification",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "algorithms_tested": list(results.keys()),
        "results": results
    }
    
    with open("experimental_results/fuzzy_hash_results.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: experimental_results/fuzzy_hash_results.json")
    print("\n✅ Fuzzy hash verification completed")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)