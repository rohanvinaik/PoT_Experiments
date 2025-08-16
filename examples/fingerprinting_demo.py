#!/usr/bin/env python3
"""
Behavioral Fingerprinting Demo

This script demonstrates the key features of the behavioral fingerprinting system
for neural network verification, as described in Paper §2.2.

The demo covers:
1. Basic fingerprinting of a vision model
2. Jacobian analysis for sensitivity detection
3. Comparing fingerprints between models
4. Using fingerprints for quick verification

Requirements:
- PyTorch
- torchvision (for pretrained models)
- numpy
"""

import sys
import torch
import torch.nn as nn
import numpy as np
from typing import List
import time

# Add parent directory to path for imports
sys.path.append('..')

from pot.core.fingerprint import (
    FingerprintConfig,
    fingerprint_run,
    compare_fingerprints,
    fingerprint_distance,
    is_behavioral_match,
    compute_jacobian_sketch,
    compare_jacobian_sketches
)


# ============================================================================
# Demo 1: Basic Fingerprinting of a Vision Model
# ============================================================================

def demo_basic_fingerprinting():
    """Demonstrate basic IO fingerprinting on a simple vision model."""
    print("\n" + "="*60)
    print("Demo 1: Basic Vision Model Fingerprinting")
    print("="*60)
    
    # Create a simple CNN model
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.fc = nn.Linear(32 * 8 * 8, 10)
            
        def forward(self, x):
            x = self.pool(torch.relu(self.conv1(x)))
            x = self.pool(torch.relu(self.conv2(x)))
            x = x.view(-1, 32 * 8 * 8)
            return self.fc(x)
    
    model = SimpleCNN()
    model.eval()
    
    # Generate challenge inputs (32x32 RGB images)
    print("\n1. Generating challenge inputs...")
    challenges = []
    torch.manual_seed(42)
    for i in range(5):
        # Create diverse challenges: noise, gradients, patterns
        if i % 3 == 0:
            # Random noise
            challenge = torch.randn(1, 3, 32, 32)
        elif i % 3 == 1:
            # Gradient pattern
            x = torch.linspace(-1, 1, 32)
            y = torch.linspace(-1, 1, 32)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            challenge = torch.stack([xx, yy, xx+yy]).unsqueeze(0)
        else:
            # Checkerboard pattern
            pattern = torch.zeros(32, 32)
            pattern[::4, ::4] = 1
            pattern[2::4, 2::4] = 1
            challenge = pattern.repeat(1, 3, 1, 1)
        
        challenges.append(challenge)
    
    print(f"  ✓ Generated {len(challenges)} diverse challenge inputs")
    
    # Configure fingerprinting (IO only for speed)
    print("\n2. Configuring fingerprinting...")
    config = FingerprintConfig.for_vision_model(
        compute_jacobian=False,  # Start with IO only
        include_timing=True,
        memory_efficient=False
    )
    print(f"  ✓ Config: precision={config.canonicalize_precision}, "
          f"batch_size={config.batch_size}")
    
    # Compute fingerprint
    print("\n3. Computing model fingerprint...")
    start_time = time.time()
    
    # Create a wrapper function for the model
    def model_wrapper(x):
        with torch.no_grad():
            return model(x).numpy()
    
    fingerprint = fingerprint_run(model_wrapper, challenges, config)
    elapsed = time.time() - start_time
    
    print(f"  ✓ Fingerprint computed in {elapsed:.3f}s")
    print(f"  - IO Hash: {fingerprint.io_hash[:32]}...")
    print(f"  - Num outputs: {len(fingerprint.raw_outputs)}")
    if fingerprint.timing_info:
        avg_time = np.mean(fingerprint.timing_info)
        print(f"  - Avg time per challenge: {avg_time*1000:.2f}ms")
    
    # Verify determinism
    print("\n4. Verifying determinism...")
    fingerprint2 = fingerprint_run(model_wrapper, challenges, config)
    
    if fingerprint.io_hash == fingerprint2.io_hash:
        print("  ✓ Fingerprints are deterministic (hashes match)")
    else:
        print("  ✗ Warning: Fingerprints don't match (non-deterministic)")
    
    return model, challenges, fingerprint


# ============================================================================
# Demo 2: Jacobian Analysis for Sensitivity Detection
# ============================================================================

def demo_jacobian_analysis():
    """Demonstrate Jacobian sketching for sensitivity analysis."""
    print("\n" + "="*60)
    print("Demo 2: Jacobian Analysis for Sensitivity Detection")
    print("="*60)
    
    # Use the same simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.ReLU(),
        nn.Linear(10, 5)
    )
    model.eval()
    
    print("\n1. Computing Jacobian for sensitivity analysis...")
    
    # Single input for Jacobian analysis
    input_tensor = torch.randn(1, 10)
    
    # Compute different types of Jacobian sketches
    print("\n2. Computing different Jacobian sketch types...")
    
    # Sign sketch (most robust)
    sign_sketch = compute_jacobian_sketch(
        model, input_tensor,
        method='sign',
        epsilon=1e-6,
        delta=1e-3
    )
    print(f"  ✓ Sign sketch computed: {len(sign_sketch)} bytes")
    
    # Magnitude sketch (captures scale)
    magnitude_sketch = compute_jacobian_sketch(
        model, input_tensor,
        method='magnitude',
        num_bins=8
    )
    print(f"  ✓ Magnitude sketch computed: {len(magnitude_sketch)} bytes")
    
    # Analyze sensitivity to perturbations
    print("\n3. Testing sensitivity to input perturbations...")
    
    # Small perturbation
    perturbed_input = input_tensor + 0.01 * torch.randn_like(input_tensor)
    perturbed_sketch = compute_jacobian_sketch(
        model, perturbed_input,
        method='sign'
    )
    
    # Compare sketches
    similarity = compare_jacobian_sketches(sign_sketch, perturbed_sketch, method='hamming')
    print(f"  - Similarity after small perturbation: {similarity:.3f}")
    
    # Large perturbation
    random_input = torch.randn_like(input_tensor)
    random_sketch = compute_jacobian_sketch(
        model, random_input,
        method='sign'
    )
    
    similarity2 = compare_jacobian_sketches(sign_sketch, random_sketch, method='hamming')
    print(f"  - Similarity with random input: {similarity2:.3f}")
    
    # Interpretation
    print("\n4. Interpretation:")
    if similarity > 0.9:
        print("  ✓ Model shows stable gradients (robust to small perturbations)")
    elif similarity > 0.7:
        print("  ~ Model shows moderate gradient stability")
    else:
        print("  ! Model shows high gradient sensitivity")
    
    return model, sign_sketch


# ============================================================================
# Demo 3: Comparing Fingerprints Between Models
# ============================================================================

def demo_fingerprint_comparison():
    """Demonstrate fingerprint comparison between different models."""
    print("\n" + "="*60)
    print("Demo 3: Comparing Fingerprints Between Models")
    print("="*60)
    
    # Create three models with different relationships
    torch.manual_seed(42)
    
    # Model 1: Original
    model1 = nn.Sequential(
        nn.Linear(20, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Model 2: Same architecture, different weights
    model2 = nn.Sequential(
        nn.Linear(20, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    
    # Model 3: Fine-tuned version of model1 (small changes)
    model3 = nn.Sequential(
        nn.Linear(20, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    )
    model3.load_state_dict(model1.state_dict())
    # Apply small perturbation to simulate fine-tuning
    with torch.no_grad():
        for param in model3.parameters():
            param.add_(0.001 * torch.randn_like(param))
    
    # Generate common challenges
    challenges = [torch.randn(1, 20) for _ in range(10)]
    
    # Configure with both IO and Jacobian
    config = FingerprintConfig()
    config.compute_jacobian = True
    config.jacobian_sketch_type = 'sign'
    config.output_type = 'logits'
    config.canonicalize_precision = 6
    
    print("\n1. Computing fingerprints for all models...")
    
    # Compute fingerprints
    def create_wrapper(model):
        def wrapper(x):
            with torch.no_grad():
                return model(x).numpy()
        return wrapper
    
    fp1 = fingerprint_run(create_wrapper(model1), challenges, config)
    fp2 = fingerprint_run(create_wrapper(model2), challenges, config)
    fp3 = fingerprint_run(create_wrapper(model3), challenges, config)
    
    print("  ✓ All fingerprints computed")
    
    # Compare fingerprints
    print("\n2. Pairwise fingerprint comparisons:")
    
    # Model 1 vs Model 2 (different models)
    sim_1_2 = compare_fingerprints(fp1, fp2)
    dist_1_2 = fingerprint_distance(fp1, fp2, metric='combined')
    print(f"  - Model1 vs Model2 (different): similarity={sim_1_2:.3f}, distance={dist_1_2:.3f}")
    
    # Model 1 vs Model 3 (fine-tuned)
    sim_1_3 = compare_fingerprints(fp1, fp3)
    dist_1_3 = fingerprint_distance(fp1, fp3, metric='combined')
    print(f"  - Model1 vs Model3 (fine-tuned): similarity={sim_1_3:.3f}, distance={dist_1_3:.3f}")
    
    # Self-similarity check
    sim_1_1 = compare_fingerprints(fp1, fp1)
    print(f"  - Model1 vs Model1 (self): similarity={sim_1_1:.3f}")
    
    # Analyze different components
    print("\n3. Component-wise analysis:")
    
    # IO distance only
    io_dist_1_2 = fingerprint_distance(fp1, fp2, metric='io')
    io_dist_1_3 = fingerprint_distance(fp1, fp3, metric='io')
    print(f"  - IO distance: Model1-Model2={io_dist_1_2:.3f}, Model1-Model3={io_dist_1_3:.3f}")
    
    # Jacobian distance only
    if fp1.jacobian_sketch and fp2.jacobian_sketch:
        jac_dist_1_2 = fingerprint_distance(fp1, fp2, metric='jacobian')
        jac_dist_1_3 = fingerprint_distance(fp1, fp3, metric='jacobian')
        print(f"  - Jacobian distance: Model1-Model2={jac_dist_1_2:.3f}, Model1-Model3={jac_dist_1_3:.3f}")
    
    # Interpretation
    print("\n4. Interpretation:")
    if sim_1_3 > 0.9:
        print("  ✓ Fine-tuned model detected as very similar (as expected)")
    if sim_1_2 < 0.5:
        print("  ✓ Different models show low similarity (as expected)")
    
    return fp1, fp2, fp3


# ============================================================================
# Demo 4: Quick Verification Using Fingerprints
# ============================================================================

def demo_quick_verification():
    """Demonstrate using fingerprints for quick model verification."""
    print("\n" + "="*60)
    print("Demo 4: Quick Verification Using Fingerprints")
    print("="*60)
    
    # Create reference model and candidate models
    torch.manual_seed(42)
    reference_model = nn.Linear(10, 5)
    
    # Candidate 1: Same model
    candidate1 = nn.Linear(10, 5)
    candidate1.load_state_dict(reference_model.state_dict())
    
    # Candidate 2: Slightly modified
    candidate2 = nn.Linear(10, 5)
    candidate2.load_state_dict(reference_model.state_dict())
    with torch.no_grad():
        candidate2.weight.data += 0.0001  # Very small change
    
    # Candidate 3: Different model
    candidate3 = nn.Linear(10, 5)
    
    # Quick verification challenges (fewer for speed)
    quick_challenges = [torch.randn(1, 10) for _ in range(3)]
    
    # Configure for quick verification
    quick_config = FingerprintConfig()
    quick_config.compute_jacobian = False  # IO only for speed
    quick_config.canonicalize_precision = 4  # Lower precision for speed
    
    print("\n1. Computing reference fingerprint...")
    def create_wrapper(model):
        def wrapper(x):
            with torch.no_grad():
                return model(x).numpy()
        return wrapper
    
    ref_fp = fingerprint_run(create_wrapper(reference_model), quick_challenges, quick_config)
    print(f"  ✓ Reference fingerprint: {ref_fp.io_hash[:16]}...")
    
    print("\n2. Quick verification of candidates:")
    
    # Test each candidate
    candidates = [
        ("Identical copy", candidate1),
        ("Slightly modified", candidate2),
        ("Different model", candidate3)
    ]
    
    for name, candidate in candidates:
        start = time.time()
        
        # Compute fingerprint
        cand_fp = fingerprint_run(create_wrapper(candidate), quick_challenges, quick_config)
        
        # Quick verification with different thresholds
        is_exact = cand_fp.io_hash == ref_fp.io_hash
        is_match_high = is_behavioral_match(ref_fp, cand_fp, threshold=0.95)
        is_match_low = is_behavioral_match(ref_fp, cand_fp, threshold=0.8)
        
        elapsed = (time.time() - start) * 1000  # Convert to ms
        
        print(f"\n  {name}:")
        print(f"    - Time: {elapsed:.1f}ms")
        print(f"    - Exact match: {is_exact}")
        print(f"    - Match (95% threshold): {is_match_high}")
        print(f"    - Match (80% threshold): {is_match_low}")
        
        similarity = compare_fingerprints(ref_fp, cand_fp)
        print(f"    - Similarity: {similarity:.3f}")
    
    # Demonstrate batch verification
    print("\n3. Batch verification (checking multiple models at once):")
    
    from pot.core.fingerprint import batch_compare_fingerprints
    
    # Compute all fingerprints
    all_fps = [ref_fp]
    for _, model in candidates:
        fp = fingerprint_run(create_wrapper(model), quick_challenges, quick_config)
        all_fps.append(fp)
    
    # Compare all against reference
    similarities = batch_compare_fingerprints(all_fps[1:], reference=ref_fp)
    
    print("\n  Batch results (similarity to reference):")
    for i, (name, _) in enumerate(candidates):
        print(f"    - {name}: {similarities[i]:.3f}")
    
    # Find best match
    from pot.core.fingerprint import find_closest_match
    best_idx, best_sim = find_closest_match(ref_fp, all_fps[1:])
    print(f"\n  Best match: {candidates[best_idx][0]} (similarity: {best_sim:.3f})")
    
    return ref_fp, all_fps


# ============================================================================
# Main Demo Runner
# ============================================================================

def main():
    """Run all demos."""
    print("="*60)
    print("Behavioral Fingerprinting System Demo")
    print("Reference: Paper §2.2 - Behavioral Verification")
    print("="*60)
    
    try:
        # Demo 1: Basic fingerprinting
        model, challenges, fp = demo_basic_fingerprinting()
        
        # Demo 2: Jacobian analysis
        jacobian_model, sketch = demo_jacobian_analysis()
        
        # Demo 3: Model comparison
        fp1, fp2, fp3 = demo_fingerprint_comparison()
        
        # Demo 4: Quick verification
        ref_fp, all_fps = demo_quick_verification()
        
        # Summary
        print("\n" + "="*60)
        print("Demo Summary")
        print("="*60)
        print("\nKey Takeaways:")
        print("1. IO fingerprints provide fast, deterministic model identification")
        print("2. Jacobian sketches capture model sensitivity and gradient structure")
        print("3. Fingerprint comparison can detect fine-tuning and model relationships")
        print("4. Quick verification enables real-time model authentication")
        print("\nRecommended Usage:")
        print("- Use IO fingerprints for quick identity checks (< 100ms)")
        print("- Add Jacobian analysis for security-critical verification")
        print("- Tune thresholds based on your security requirements")
        print("- Use batch processing for verifying multiple models efficiently")
        
        print("\n✅ All demos completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())