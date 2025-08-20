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
    reference = "The quick brown fox jumps over the lazy dog for fuzzy hash testing with sufficient length for TLSH"
    similar = "The quick brown fox jumps over the lazy dogs for fuzzy hash testing with sufficient length for TLSH"
    different = "Hello world, this is a completely different text used for testing fuzzy hash algorithms with enough data"
    
    results = {}
    
    # Create single verifier instance
    verifier = FuzzyHashVerifier()
    
    # Test each algorithm
    for algo in ["tlsh", "ssdeep", "sha256"]:
        print(f"\nTesting {algo}:")
        
        try:
            # Generate hashes with specific algorithm
            ref_hash = verifier.generate_fuzzy_hash(reference.encode(), algorithm=algo)
            sim_hash = verifier.generate_fuzzy_hash(similar.encode(), algorithm=algo)
            diff_hash = verifier.generate_fuzzy_hash(different.encode(), algorithm=algo)
            
            # Compute similarities
            sim_score = verifier.compare(ref_hash, sim_hash)
            diff_score = verifier.compare(ref_hash, diff_hash)
            
            print(f"  Algorithm: {ref_hash['algorithm']}")
            print(f"  Is fuzzy: {ref_hash['is_fuzzy']}")
            print(f"  Similar text score: {sim_score:.3f}")
            print(f"  Different text score: {diff_score:.3f}")
            
            results[algo] = {
                "algorithm_label": ref_hash['algorithm'],
                "is_fuzzy": ref_hash['is_fuzzy'],
                "similar_score": sim_score,
                "different_score": diff_score
            }
            
        except Exception as e:
            print(f"  ⚠️ {algo} failed: {e}")
            results[algo] = {"error": str(e)}
    
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