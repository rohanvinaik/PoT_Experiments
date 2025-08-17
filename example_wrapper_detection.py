#!/usr/bin/env python3
"""
Example demonstrating the comprehensive wrapper detection system.
"""

import numpy as np
import matplotlib.pyplot as plt
from pot.core.wrapper_detection import (
    WrapperDetector, 
    AdaptiveThresholdManager, 
    AdversarySimulator,
    kolmogorov_smirnov_test,
    anderson_darling_test,
    wasserstein_distance
)

def create_legitimate_baseline():
    """Create baseline statistics from legitimate model."""
    # Legitimate model has consistent, fast response times
    response_times = np.random.exponential(0.03, 2000)  # ~30ms average
    
    # Legitimate responses have consistent statistical properties
    responses = []
    for _ in range(500):
        response = np.random.multivariate_normal(
            mean=[0.0, 1.0, -0.5], 
            cov=[[1.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        responses.append(response)
    
    return {
        'response_times': response_times.tolist(),
        'responses': responses
    }

def simulate_wrapper_attack():
    """Simulate a sophisticated wrapper attack."""
    
    def genuine_model(x):
        # Genuine model with consistent behavior
        return np.random.multivariate_normal(
            mean=[0.0, 1.0, -0.5], 
            cov=[[1.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
    
    def fake_model(x):
        # Fake model with different statistical properties
        return np.random.multivariate_normal(
            mean=[0.5, 0.0, 1.0], 
            cov=[[2.0, 0.0, 0.5], [0.0, 0.5, 0.0], [0.5, 0.0, 2.0]]
        )
    
    # Create wrapper adversary
    adversary = AdversarySimulator(genuine_model, 'wrapper')
    
    # Override fake model
    adversary._fake_model = fake_model
    
    # Generate attack sequence
    attack_data = adversary.generate_attack_sequence(n_requests=200, challenge_ratio=0.15)
    
    return attack_data

def demonstrate_statistical_tests():
    """Demonstrate statistical test functions."""
    print("=== Statistical Tests Demonstration ===")
    
    # Create two different distributions
    dist1 = np.random.normal(0, 1, 500)
    dist2 = np.random.normal(0.5, 1.5, 500)
    
    # Run tests
    ks_stat, ks_p = kolmogorov_smirnov_test(dist1, dist2)
    ad_stat, ad_p = anderson_darling_test(dist1, dist2) 
    w_dist = wasserstein_distance(dist1, dist2)
    
    print(f"Kolmogorov-Smirnov Test:")
    print(f"  Statistic: {ks_stat:.4f}")
    print(f"  P-value: {ks_p:.4f}")
    print(f"  Significant: {ks_p < 0.05}")
    
    print(f"\nAnderson-Darling Test:")
    print(f"  Statistic: {ad_stat:.4f}")
    print(f"  P-value: {ad_p:.4f}")
    print(f"  Significant: {ad_p < 0.05}")
    
    print(f"\nWasserstein Distance: {w_dist:.4f}")
    print()

def demonstrate_wrapper_detection():
    """Demonstrate comprehensive wrapper detection."""
    print("=== Wrapper Detection Demonstration ===")
    
    # Create legitimate baseline
    baseline_stats = create_legitimate_baseline()
    print(f"Created baseline with {len(baseline_stats['response_times'])} timing samples")
    print(f"and {len(baseline_stats['responses'])} response samples")
    
    # Initialize detector
    detector = WrapperDetector(baseline_stats)
    print("Initialized WrapperDetector")
    
    # Test with legitimate traffic
    print("\n--- Testing Legitimate Traffic ---")
    legit_times = np.random.exponential(0.03, 100).tolist()
    legit_responses = []
    for _ in range(100):
        response = np.random.multivariate_normal(
            mean=[0.0, 1.0, -0.5], 
            cov=[[1.0, 0.1, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        legit_responses.append(response)
    
    legit_result = detector.detect_wrapper(legit_times, legit_responses)
    print(f"Legitimate Detection Result:")
    print(f"  Is Wrapped: {legit_result['is_wrapped']}")
    print(f"  Confidence: {legit_result['confidence']:.3f}")
    print(f"  Timing Anomaly: {legit_result['evidence']['timing_anomaly']:.3f}")
    print(f"  ECDF Deviation: {legit_result['evidence']['ecdf_deviation']:.3f}")
    print(f"  Behavioral Drift: {legit_result['evidence']['behavioral_drift']:.3f}")
    
    # Test with wrapper attack
    print("\n--- Testing Wrapper Attack ---")
    attack_data = simulate_wrapper_attack()
    
    attack_result = detector.detect_wrapper(
        attack_data['timings'],
        attack_data['responses']
    )
    
    print(f"Attack Detection Result:")
    print(f"  Is Wrapped: {attack_result['is_wrapped']}")
    print(f"  Confidence: {attack_result['confidence']:.3f}")
    print(f"  Timing Anomaly: {attack_result['evidence']['timing_anomaly']:.3f}")
    print(f"  ECDF Deviation: {attack_result['evidence']['ecdf_deviation']:.3f}")
    print(f"  Behavioral Drift: {attack_result['evidence']['behavioral_drift']:.3f}")
    
    # Show statistical test results
    if 'statistical_tests' in attack_result['evidence']:
        print(f"  Statistical Tests:")
        for test_name, test_results in attack_result['evidence']['statistical_tests'].items():
            print(f"    {test_name}:")
            for metric, value in test_results.items():
                print(f"      {metric}: {value:.4f}")
    
    return legit_result, attack_result

def demonstrate_adaptive_thresholds():
    """Demonstrate adaptive threshold management."""
    print("\n=== Adaptive Threshold Management ===")
    
    # Initialize manager
    manager = AdaptiveThresholdManager()
    print("Initialized AdaptiveThresholdManager")
    
    # Simulate labeled training data
    n_samples = 100
    legitimate_scores = {
        'timing_anomaly': np.random.beta(2, 8, n_samples//2),  # Low scores for legitimate
        'behavioral_drift': np.random.beta(2, 8, n_samples//2)
    }
    
    attack_scores = {
        'timing_anomaly': np.random.beta(6, 2, n_samples//2),  # High scores for attacks
        'behavioral_drift': np.random.beta(6, 2, n_samples//2)
    }
    
    # Combine and create labels
    all_scores = {
        'timing_anomaly': np.concatenate([legitimate_scores['timing_anomaly'], 
                                        attack_scores['timing_anomaly']]),
        'behavioral_drift': np.concatenate([legitimate_scores['behavioral_drift'], 
                                          attack_scores['behavioral_drift']])
    }
    
    labels = np.concatenate([np.zeros(n_samples//2), np.ones(n_samples//2)])
    
    # Update thresholds
    manager.update_thresholds(all_scores, labels)
    
    print("Updated thresholds based on labeled data:")
    thresholds = manager.get_all_thresholds()
    for name, threshold in thresholds.items():
        print(f"  {name}: {threshold:.3f}")
    
    # Test threshold optimization
    for target_far in [0.01, 0.05, 0.10]:
        timing_threshold = manager.get_optimal_threshold('timing_anomaly', target_far)
        print(f"Optimal timing threshold for FAR={target_far}: {timing_threshold:.3f}")

def main():
    """Run complete wrapper detection demonstration."""
    print("Wrapper Detection System Demonstration")
    print("="*50)
    
    # Test statistical functions
    demonstrate_statistical_tests()
    
    # Test wrapper detection
    legit_result, attack_result = demonstrate_wrapper_detection()
    
    # Test adaptive thresholds
    demonstrate_adaptive_thresholds()
    
    print("\n=== Summary ===")
    print(f"Legitimate traffic detected as wrapped: {legit_result['is_wrapped']}")
    print(f"Attack traffic detected as wrapped: {attack_result['is_wrapped']}")
    
    if attack_result['is_wrapped'] and not legit_result['is_wrapped']:
        print("✓ Detection system working correctly!")
    else:
        print("⚠ Detection system may need tuning")

if __name__ == "__main__":
    main()