#!/usr/bin/env python3
"""
Basic usage example for semantic verification module.
Demonstrates creating and using a concept library for semantic verification.
"""

import torch
import numpy as np
from pathlib import Path

# Import semantic verification components
from pot.semantic import (
    ConceptLibrary,
    SemanticMatcher,
    ConceptVector,
    SemanticLibrary,
    save_concept_library,
    load_concept_library
)


def create_concept_library_example():
    """Example of creating and populating a concept library."""
    print("=" * 60)
    print("Creating Concept Library")
    print("=" * 60)
    
    # Create a concept library with Gaussian method
    library = ConceptLibrary(dim=128, method='gaussian')
    print(f"Created library with dimension={library.dim}, method={library.method}")
    
    # Generate synthetic training data for concepts
    print("\nAdding concepts from training data...")
    
    # Concept 1: "Normal behavior" - centered around origin
    normal_embeddings = torch.randn(100, 128) * 0.5
    library.add_concept('normal_behavior', normal_embeddings)
    print("  ✓ Added 'normal_behavior' concept (100 samples)")
    
    # Concept 2: "Class A" - shifted positive
    class_a_embeddings = torch.randn(80, 128) * 0.5 + torch.ones(128) * 1.0
    library.add_concept('class_a', class_a_embeddings)
    print("  ✓ Added 'class_a' concept (80 samples)")
    
    # Concept 3: "Class B" - shifted negative
    class_b_embeddings = torch.randn(80, 128) * 0.5 - torch.ones(128) * 1.0
    library.add_concept('class_b', class_b_embeddings)
    print("  ✓ Added 'class_b' concept (80 samples)")
    
    # Get library summary
    summary = library.get_summary()
    print(f"\nLibrary Summary:")
    print(f"  - Total concepts: {summary['n_concepts']}")
    print(f"  - Total samples: {summary['total_samples']}")
    print(f"  - Average samples per concept: {summary['avg_samples_per_concept']:.1f}")
    
    return library


def semantic_matching_example(library):
    """Example of using semantic matching with the concept library."""
    print("\n" + "=" * 60)
    print("Semantic Matching")
    print("=" * 60)
    
    # Create a semantic matcher
    matcher = SemanticMatcher(library=library, threshold=0.7)
    print(f"Created semantic matcher with threshold={matcher.threshold}")
    
    # Test matching with different embeddings
    print("\nTesting semantic matching...")
    
    # Test 1: Embedding similar to normal behavior
    test_normal = torch.randn(128) * 0.5 + torch.randn(128) * 0.1
    matches_normal = matcher.match_to_library(test_normal)
    print("\nTest 1 - Normal-like embedding:")
    for concept, score in list(matches_normal.items())[:3]:
        print(f"  {concept}: {score:.3f}")
    
    # Test 2: Embedding similar to class A
    test_class_a = torch.randn(128) * 0.5 + torch.ones(128) * 1.0
    matches_class_a = matcher.match_to_library(test_class_a)
    print("\nTest 2 - Class A-like embedding:")
    for concept, score in list(matches_class_a.items())[:3]:
        print(f"  {concept}: {score:.3f}")
    
    # Test 3: Anomalous embedding (far from all concepts)
    test_anomaly = torch.randn(128) * 2.0 + torch.ones(128) * 5.0
    matches_anomaly = matcher.match_to_library(test_anomaly)
    print("\nTest 3 - Anomalous embedding:")
    for concept, score in list(matches_anomaly.items())[:3]:
        print(f"  {concept}: {score:.3f}")
    
    # Find best match for each test
    print("\nBest matches:")
    for name, embedding in [("Normal", test_normal), ("Class A", test_class_a), ("Anomaly", test_anomaly)]:
        best_concept, best_score = matcher.get_best_match(embedding, min_similarity=0.5)
        if best_concept:
            print(f"  {name}: {best_concept} (score: {best_score:.3f})")
        else:
            print(f"  {name}: No match above threshold (best: {best_score:.3f})")
    
    return matcher


def drift_detection_example(library, matcher):
    """Example of detecting semantic drift."""
    print("\n" + "=" * 60)
    print("Semantic Drift Detection")
    print("=" * 60)
    
    # Generate embeddings that gradually drift from normal behavior
    print("\nSimulating gradual drift from 'normal_behavior'...")
    
    drift_results = []
    for drift_level in [0.0, 0.5, 1.0, 2.0, 3.0]:
        # Generate embeddings with increasing drift
        drifted_embeddings = torch.randn(30, 128) * 0.5 + torch.ones(128) * drift_level
        
        # Compute drift score
        drift_score = matcher.compute_semantic_drift(drifted_embeddings, 'normal_behavior')
        drift_results.append((drift_level, drift_score))
        
        print(f"  Drift level {drift_level:.1f}: score = {drift_score:.3f}")
    
    # Analyze drift trend
    print("\nDrift Analysis:")
    if drift_results[-1][1] > drift_results[0][1] * 1.5:
        print("  ⚠️  Significant drift detected!")
    else:
        print("  ✓ Drift within acceptable range")


def clustering_example(matcher):
    """Example of clustering outputs."""
    print("\n" + "=" * 60)
    print("Output Clustering")
    print("=" * 60)
    
    # Generate mixed embeddings from different distributions
    print("\nGenerating mixed embeddings from 3 distributions...")
    embeddings = []
    
    # Group 1: Normal-like
    for _ in range(20):
        embeddings.append(torch.randn(128) * 0.5)
    
    # Group 2: Class A-like
    for _ in range(20):
        embeddings.append(torch.randn(128) * 0.5 + torch.ones(128) * 1.0)
    
    # Group 3: Class B-like
    for _ in range(20):
        embeddings.append(torch.randn(128) * 0.5 - torch.ones(128) * 1.0)
    
    # Perform clustering
    labels = matcher.cluster_outputs(embeddings, n_clusters=3)
    
    # Analyze clusters
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(f"\nFound {len(unique_labels)} clusters:")
    for label, count in zip(unique_labels, counts):
        print(f"  Cluster {label}: {count} samples")
    
    # Check cluster purity (simplified)
    expected_labels = [0] * 20 + [1] * 20 + [2] * 20
    matches = sum(1 for i in range(60) if labels[i] == labels[expected_labels[i]])
    purity = matches / 60
    print(f"\nClustering quality (simplified): {purity:.1%}")


def save_load_example(library):
    """Example of saving and loading a concept library."""
    print("\n" + "=" * 60)
    print("Persistence Example")
    print("=" * 60)
    
    # Save library
    save_path = Path("example_concept_library.pt")
    print(f"\nSaving library to {save_path}...")
    library.save(str(save_path))
    print("  ✓ Library saved")
    
    # Create new library and load
    print("\nLoading library into new instance...")
    library2 = ConceptLibrary(dim=64, method='hypervector')  # Different initial params
    library2.load(str(save_path))
    print("  ✓ Library loaded")
    
    # Verify loaded correctly
    print("\nVerifying loaded library:")
    print(f"  - Dimension: {library2.dim}")
    print(f"  - Method: {library2.method}")
    print(f"  - Concepts: {library2.list_concepts()}")
    
    # Clean up
    save_path.unlink()
    print(f"\n  ✓ Cleaned up {save_path}")
    
    return library2


def hypervector_example():
    """Example using hypervector representation."""
    print("\n" + "=" * 60)
    print("Hypervector Representation")
    print("=" * 60)
    
    # Create library with hypervector method
    library = ConceptLibrary(dim=512, method='hypervector')
    print(f"Created hypervector library with dimension={library.dim}")
    
    # Add concepts
    print("\nAdding hypervector concepts...")
    
    # Generate embeddings
    concept_embeddings = torch.randn(50, 512)
    library.add_concept('hypervector_concept', concept_embeddings)
    
    # Get the hypervector
    hypervector = library.get_concept_vector('hypervector_concept')
    
    # Analyze hypervector properties
    unique_values = torch.unique(hypervector)
    print(f"\nHypervector properties:")
    print(f"  - Unique values: {unique_values.tolist()}")
    print(f"  - Positive elements: {torch.sum(hypervector > 0).item()}")
    print(f"  - Zero elements: {torch.sum(hypervector == 0).item()}")
    print(f"  - Negative elements: {torch.sum(hypervector < 0).item()}")
    
    # Test Hamming similarity
    matcher = SemanticMatcher(library=library)
    
    # Create test hypervector
    test_hv = torch.sign(torch.randn(512))
    similarity = matcher.compute_similarity(test_hv, 'hypervector_concept', method='hamming')
    print(f"\nHamming similarity with random hypervector: {similarity:.3f}")


def coverage_analysis_example(library):
    """Example of analyzing concept coverage."""
    print("\n" + "=" * 60)
    print("Coverage Analysis")
    print("=" * 60)
    
    matcher = SemanticMatcher(library=library, threshold=0.6)
    
    # Generate test embeddings
    print("\nGenerating test embeddings...")
    
    # In-distribution embeddings (should be covered)
    in_dist_embeddings = []
    for _ in range(10):
        # Mix of normal and class embeddings
        if np.random.rand() > 0.5:
            in_dist_embeddings.append(torch.randn(128) * 0.5)
        else:
            in_dist_embeddings.append(torch.randn(128) * 0.5 + torch.ones(128) * np.random.choice([-1, 1]))
    
    # Out-of-distribution embeddings (may not be covered)
    out_dist_embeddings = []
    for _ in range(10):
        out_dist_embeddings.append(torch.randn(128) * 3.0 + torch.ones(128) * np.random.randn() * 5)
    
    # Compute coverage
    in_dist_tensor = torch.stack(in_dist_embeddings)
    out_dist_tensor = torch.stack(out_dist_embeddings)
    
    in_coverage = matcher.compute_coverage(in_dist_tensor)
    out_coverage = matcher.compute_coverage(out_dist_tensor)
    
    print(f"\nCoverage Analysis:")
    print(f"  - In-distribution coverage: {in_coverage:.1%}")
    print(f"  - Out-of-distribution coverage: {out_coverage:.1%}")
    
    if in_coverage > out_coverage:
        print("  ✓ Library correctly covers in-distribution better than out-of-distribution")
    else:
        print("  ⚠️  Coverage analysis may need adjustment")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print(" " * 20 + "SEMANTIC VERIFICATION EXAMPLES")
    print("=" * 70)
    
    # 1. Create concept library
    library = create_concept_library_example()
    
    # 2. Semantic matching
    matcher = semantic_matching_example(library)
    
    # 3. Drift detection
    drift_detection_example(library, matcher)
    
    # 4. Clustering
    clustering_example(matcher)
    
    # 5. Save/Load
    loaded_library = save_load_example(library)
    
    # 6. Hypervector example
    hypervector_example()
    
    # 7. Coverage analysis
    coverage_analysis_example(library)
    
    print("\n" + "=" * 70)
    print(" " * 25 + "EXAMPLES COMPLETE")
    print("=" * 70)
    print("\nThese examples demonstrate the core functionality of the semantic")
    print("verification module. For integration with verifiers, see integration_example.py")


if __name__ == "__main__":
    main()