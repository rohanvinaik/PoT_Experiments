"""Demo of sequential verification with visualization."""

import numpy as np
from pot.core.sequential import sequential_verify

def demo_sequential_verification():
    """Demonstrate sequential verification with different scenarios."""
    print("Sequential Verification Orchestrator Demo")
    print("=" * 50)
    
    # Common parameters
    tau = 0.1
    alpha = 0.05
    beta = 0.05
    n_max = 500
    
    print(f"Parameters: tau={tau}, alpha={alpha}, beta={beta}, n_max={n_max}")
    print()
    
    # Scenario 1: Genuine model (low distances)
    print("Scenario 1: Genuine Model (should accept)")
    print("-" * 40)
    np.random.seed(42)
    genuine_stream = (np.random.beta(2, 20) for _ in range(n_max))  # Mean ~0.09
    
    decision, trail = sequential_verify(genuine_stream, tau, alpha, beta, n_max)
    
    print(f"Decision: {decision['type']}")
    print(f"Stopping time: {decision['stopping_time']} samples")
    print(f"Final mean: {decision['final_mean']:.4f}")
    print(f"Confidence interval: [{decision['confidence_interval'][0]:.4f}, {decision['confidence_interval'][1]:.4f}]")
    
    # Show convergence
    if len(trail) > 0:
        print("\nConvergence trajectory (every 10 samples):")
        print("  t  | Mean  | Radius | CI Lower | CI Upper")
        print("-----|-------|--------|----------|----------")
        for i in range(0, len(trail), max(1, len(trail)//10)):
            t, mean, r_alpha, _ = trail[i]
            ci_lower = max(0, mean - r_alpha)
            ci_upper = min(1, mean + r_alpha)
            print(f"{t:4d} | {mean:.3f} | {r_alpha:.4f} | {ci_lower:.4f} | {ci_upper:.4f}")
    
    print()
    
    # Scenario 2: Adversarial model (high distances)
    print("Scenario 2: Adversarial Model (should reject)")
    print("-" * 40)
    np.random.seed(43)
    adversarial_stream = (np.random.beta(20, 2) for _ in range(n_max))  # Mean ~0.91
    
    decision, trail = sequential_verify(adversarial_stream, tau, alpha, beta, n_max)
    
    print(f"Decision: {decision['type']}")
    print(f"Stopping time: {decision['stopping_time']} samples")
    print(f"Final mean: {decision['final_mean']:.4f}")
    print(f"Confidence interval: [{decision['confidence_interval'][0]:.4f}, {decision['confidence_interval'][1]:.4f}]")
    print()
    
    # Scenario 3: Borderline case
    print("Scenario 3: Borderline Model (near threshold)")
    print("-" * 40)
    np.random.seed(44)
    # Generate values close to threshold with some variance
    borderline_stream = (tau + np.random.normal(0, 0.02) for _ in range(n_max))
    
    decision, trail = sequential_verify(borderline_stream, tau, alpha, beta, n_max)
    
    print(f"Decision: {decision['type']}")
    print(f"Stopping time: {decision['stopping_time']} samples")
    print(f"Final mean: {decision['final_mean']:.4f}")
    if 'forced_stop' in decision and decision['forced_stop']:
        print("Note: Decision was forced at n_max")
    print()
    
    # Show how radii shrink over time
    print("Radius Evolution (Scenario 3):")
    print("-" * 40)
    if len(trail) >= 5:
        sample_points = [1, 5, 10, 25, 50, 100, 200, len(trail)]
        sample_points = [sp for sp in sample_points if sp <= len(trail)]
        
        print("Samples | Mean  | Alpha Radius | Beta Radius")
        print("--------|-------|--------------|------------")
        for sp in sample_points:
            t, mean, r_alpha, r_beta = trail[sp-1]
            print(f"{t:7d} | {mean:.3f} | {r_alpha:11.4f} | {r_beta:10.4f}")
    
    print()
    print("Key Observations:")
    print("- Genuine models (low Z) accept quickly")
    print("- Adversarial models (high Z) reject quickly")
    print("- Borderline cases take longer or hit n_max")
    print("- Radii shrink as sqrt(log(t)/t), providing tighter bounds over time")
    print("- Early stopping saves computation while maintaining error guarantees")


if __name__ == "__main__":
    demo_sequential_verification()