#!/usr/bin/env python3
"""
Comprehensive demonstration of defense mechanisms for PoT verification.

This script showcases the adaptive verification, input filtering, and randomized
defense systems, both individually and as an integrated defense framework.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
import time

from pot.core.defenses import (
    AdaptiveVerifier,
    InputFilter, 
    RandomizedDefense,
    IntegratedDefenseSystem,
    MockBaseVerifier,
    DefenseConfig,
    create_defense_config
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_model(num_classes: int = 10) -> nn.Module:
    """Create a simple test model."""
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(64, 32),
        nn.ReLU(), 
        nn.Linear(32, num_classes)
    )


def create_test_data(num_samples: int = 100, input_dim: int = 32) -> torch.Tensor:
    """Create synthetic test data."""
    return torch.randn(num_samples, input_dim) * 0.5


def create_adversarial_data(clean_data: torch.Tensor, epsilon: float = 0.3) -> torch.Tensor:
    """Create simple adversarial examples."""
    noise = torch.randn_like(clean_data) * epsilon
    return clean_data + noise


def demonstrate_adaptive_verifier():
    """Demonstrate adaptive verification capabilities."""
    print("=== Adaptive Verifier Demonstration ===")
    
    # Setup
    base_verifier = MockBaseVerifier(base_confidence=0.8)
    config = create_defense_config('adaptive')
    
    adaptive_verifier = AdaptiveVerifier(base_verifier, config)
    model = create_test_model()
    
    print(f"Initialized adaptive verifier with config: {config.defense_type}")
    
    # Test normal verification
    print("\n--- Normal Verification ---")
    clean_input = torch.randn(32) * 0.3
    result = adaptive_verifier.verify(clean_input, model)
    
    print(f"Clean input verification:")
    print(f"  Verified: {result['verified']}")
    print(f"  Original confidence: {result['original_confidence']:.3f}")
    print(f"  Adapted confidence: {result['adapted_confidence']:.3f}")
    print(f"  Attack indicators: {len(result['attack_indicators'])}")
    
    # Test with suspicious input
    print("\n--- Suspicious Input Verification ---")
    suspicious_input = torch.randn(32) * 2.0  # Higher magnitude
    result = adaptive_verifier.verify(suspicious_input, model)
    
    print(f"Suspicious input verification:")
    print(f"  Verified: {result['verified']}")
    print(f"  Original confidence: {result['original_confidence']:.3f}")
    print(f"  Adapted confidence: {result['adapted_confidence']:.3f}")
    print(f"  Attack indicators: {result['attack_indicators']}")
    
    # Simulate attack observations for adaptation
    print("\n--- Attack Adaptation ---")
    attack_observations = {
        'strength': 0.8,
        'detection_confidence': 0.9,
        'characteristics': {'mean_magnitude': 2.5, 'std_value': 1.2},
        'success': True
    }
    
    adaptive_verifier.adapt_to_attack('perturbation', attack_observations)
    
    # Test after adaptation
    result_adapted = adaptive_verifier.verify(suspicious_input, model)
    print(f"After adaptation:")
    print(f"  Adapted confidence: {result_adapted['adapted_confidence']:.3f}")
    print(f"  Thresholds updated: {len(adaptive_verifier.thresholds)}")
    
    # Add defense layer
    print("\n--- Adding Defense Layer ---")
    adaptive_verifier.add_defense_layer('input_filtering', noise_threshold=0.5)
    
    result_with_defense = adaptive_verifier.verify(suspicious_input, model)
    print(f"With defense layer:")
    print(f"  Defense layers active: {result_with_defense['adaptation_metadata']['defense_layers_active']}")
    print(f"  Adapted confidence: {result_with_defense['adapted_confidence']:.3f}")
    
    # Get adaptation summary
    summary = adaptive_verifier.get_adaptation_summary()
    print(f"\n--- Adaptation Summary ---")
    print(f"  Total verifications: {summary['verification_stats']['total_verifications']}")
    print(f"  Adaptations made: {summary['verification_stats']['adaptations_made']}")
    print(f"  Attack patterns learned: {len(summary['attack_patterns'])}")
    print(f"  Defense layers: {summary['defense_layers']}")
    
    return adaptive_verifier


def demonstrate_input_filter():
    """Demonstrate input filtering capabilities."""
    print("\n=== Input Filter Demonstration ===")
    
    # Setup
    config = create_defense_config('filter')
    input_filter = InputFilter(config)
    
    print(f"Initialized input filter with detection methods: {config.parameters['detection_methods']}")
    
    # Create training data
    print("\n--- Filter Calibration ---")
    clean_samples = create_test_data(200, 32)
    adversarial_samples = create_adversarial_data(clean_samples[:50], epsilon=0.4)
    
    input_filter.calibrate(clean_samples, adversarial_samples)
    print("Filter calibrated with clean and adversarial samples")
    
    # Test detection on clean input
    print("\n--- Clean Input Detection ---")
    clean_input = torch.randn(32) * 0.2
    detection_result = input_filter.detect_adversarial(clean_input)
    
    print(f"Clean input detection:")
    print(f"  Is adversarial: {detection_result['is_adversarial']}")
    print(f"  Combined score: {detection_result['combined_score']:.3f}")
    print(f"  Confidence: {detection_result['confidence']:.3f}")
    print(f"  Detection methods: {list(detection_result['detection_results'].keys())}")
    
    # Test detection on adversarial input
    print("\n--- Adversarial Input Detection ---")
    adv_input = create_adversarial_data(clean_input.unsqueeze(0), epsilon=0.5).squeeze(0)
    detection_result = input_filter.detect_adversarial(adv_input)
    
    print(f"Adversarial input detection:")
    print(f"  Is adversarial: {detection_result['is_adversarial']}")
    print(f"  Combined score: {detection_result['combined_score']:.3f}")
    print(f"  Confidence: {detection_result['confidence']:.3f}")
    
    # Test input sanitization
    print("\n--- Input Sanitization ---")
    sanitization_result = input_filter.sanitize_input(adv_input)
    
    print(f"Input sanitization:")
    print(f"  Methods applied: {sanitization_result['methods_applied']}")
    print(f"  L2 change: {sanitization_result['l2_change']:.3f}")
    print(f"  L∞ change: {sanitization_result['linf_change']:.3f}")
    
    # Test after sanitization
    sanitized_detection = input_filter.detect_adversarial(sanitization_result['sanitized_input'])
    print(f"  Detection after sanitization: {sanitized_detection['is_adversarial']}")
    print(f"  Score after sanitization: {sanitized_detection['combined_score']:.3f}")
    
    # Test distribution validation
    print("\n--- Distribution Validation ---")
    validation_result = input_filter.validate_input_distribution(adv_input)
    
    print(f"Distribution validation:")
    print(f"  Is valid: {validation_result['is_valid']}")
    print(f"  Validation score: {validation_result['validation_score']:.3f}")
    print(f"  Violations: {validation_result['num_violations']}")
    
    # Get filter statistics
    stats = input_filter.get_filter_statistics()
    print(f"\n--- Filter Statistics ---")
    print(f"  Total processed: {stats['total_processed']}")
    print(f"  Detection rate: {stats['detection_rate']:.3f}")
    print(f"  Sanitization rate: {stats['sanitization_rate']:.3f}")
    print(f"  Calibrated: {stats['calibrated']}")
    
    return input_filter


def demonstrate_randomized_defense():
    """Demonstrate randomized defense mechanisms."""
    print("\n=== Randomized Defense Demonstration ===")
    
    # Setup
    config = create_defense_config('randomized')
    randomized_defense = RandomizedDefense(config)
    model = create_test_model()
    
    print(f"Initialized randomized defense with config: {config.defense_type}")
    
    # Test random smoothing
    print("\n--- Random Smoothing ---")
    test_input = torch.randn(1, 32) * 0.3
    
    smoothing_result = randomized_defense.random_smoothing(
        model=model,
        input_data=test_input,
        noise_level=0.1,
        n_samples=50
    )
    
    print(f"Random smoothing:")
    print(f"  Noise level: {smoothing_result['noise_level']}")
    print(f"  Samples: {smoothing_result['n_samples']}")
    print(f"  Overall confidence: {smoothing_result['overall_confidence']:.3f}")
    print(f"  Overall consensus: {smoothing_result['overall_consensus']:.3f}")
    
    # Test with different threat levels
    print("\n--- Adaptive Smoothing ---")
    for threat_level in [0.2, 0.5, 0.8]:
        adaptive_result = randomized_defense.adaptive_smoothing(
            model=model,
            input_data=test_input,
            threat_level=threat_level
        )
        
        print(f"Threat level {threat_level}:")
        print(f"  Adapted noise: {adaptive_result['adaptation_metadata']['adapted_noise_level']:.3f}")
        print(f"  Adapted samples: {adaptive_result['adaptation_metadata']['adapted_samples']}")
        print(f"  Confidence: {adaptive_result['overall_confidence']:.3f}")
    
    # Test stochastic verification
    print("\n--- Stochastic Verification ---")
    base_verifier = MockBaseVerifier(base_confidence=0.7)
    
    stochastic_result = randomized_defense.stochastic_verification(
        verifier=base_verifier,
        input_data=test_input.squeeze(0),
        model=model,
        n_trials=10
    )
    
    print(f"Stochastic verification:")
    print(f"  Final decision: {stochastic_result['final_decision']}")
    print(f"  Consensus strength: {stochastic_result['consensus_strength']:.3f}")
    print(f"  Overall confidence: {stochastic_result['overall_confidence']:.3f}")
    print(f"  Successful trials: {stochastic_result['successful_trials']}")
    
    # Test ensemble verification
    print("\n--- Ensemble Verification ---")
    verifiers = [
        MockBaseVerifier(base_confidence=0.8),
        MockBaseVerifier(base_confidence=0.7),
        MockBaseVerifier(base_confidence=0.75)
    ]
    
    ensemble_result = randomized_defense.ensemble_randomized_verification(
        verifiers=verifiers,
        input_data=test_input.squeeze(0),
        model=model
    )
    
    print(f"Ensemble verification:")
    print(f"  Ensemble decision: {ensemble_result['ensemble_decision']}")
    print(f"  Ensemble confidence: {ensemble_result['ensemble_confidence']:.3f}")
    print(f"  Consensus strength: {ensemble_result['consensus_strength']:.3f}")
    print(f"  Positive votes: {ensemble_result['voting_summary']['positive_votes']}")
    print(f"  Total verifiers: {ensemble_result['voting_summary']['total_verifiers']}")
    
    # Get defense statistics
    stats = randomized_defense.get_defense_statistics()
    print(f"\n--- Randomized Defense Statistics ---")
    print(f"  Smoothing applications: {stats['smoothing_applications']}")
    print(f"  Stochastic verifications: {stats['stochastic_verifications']}")
    print(f"  Consensus rate: {stats['consensus_rate']:.3f}")
    print(f"  Average noise level: {stats['average_noise_level']:.3f}")
    
    return randomized_defense


def demonstrate_integrated_system():
    """Demonstrate the integrated defense system."""
    print("\n=== Integrated Defense System Demonstration ===")
    
    # Setup
    base_verifier = MockBaseVerifier(base_confidence=0.8)
    
    defense_configs = {
        'adaptive': create_defense_config('adaptive'),
        'filter': create_defense_config('filter'),
        'random': create_defense_config('randomized')
    }
    
    integrated_system = IntegratedDefenseSystem(base_verifier, defense_configs)
    model = create_test_model()
    
    print("Initialized integrated defense system with all components")
    
    # Train the system
    print("\n--- System Training ---")
    clean_samples = create_test_data(100, 32)
    adversarial_samples = create_adversarial_data(clean_samples[:30], epsilon=0.4)
    
    integrated_system.train_defenses(clean_samples, adversarial_samples)
    print("Defense system trained and calibrated")
    
    # Test with clean input
    print("\n--- Clean Input Processing ---")
    clean_input = torch.randn(32) * 0.2
    
    clean_result = integrated_system.comprehensive_defense(
        input_data=clean_input,
        model=model
    )
    
    print(f"Clean input defense:")
    print(f"  Final decision: {clean_result['final_decision']['verified']}")
    print(f"  Defense confidence: {clean_result['defense_confidence']:.3f}")
    print(f"  Threat level: {clean_result['threat_level']:.3f}")
    print(f"  Processing time: {clean_result['processing_time']:.3f}s")
    
    defense_summary = clean_result['defense_summary']
    print(f"  Defense activated: {defense_summary['defense_activated']}")
    print(f"  Threats found: {defense_summary['threats_found']}")
    print(f"  Final status: {defense_summary['final_status']}")
    
    # Test with adversarial input
    print("\n--- Adversarial Input Processing ---")
    adv_input = create_adversarial_data(clean_input.unsqueeze(0), epsilon=0.6).squeeze(0)
    
    adv_result = integrated_system.comprehensive_defense(
        input_data=adv_input,
        model=model,
        threat_level=0.8  # High threat level
    )
    
    print(f"Adversarial input defense:")
    print(f"  Final decision: {adv_result['final_decision']['verified']}")
    print(f"  Defense confidence: {adv_result['defense_confidence']:.3f}")
    print(f"  Threat level: {adv_result['threat_level']:.3f}")
    print(f"  Processing time: {adv_result['processing_time']:.3f}s")
    
    defense_summary = adv_result['defense_summary']
    print(f"  Defense activated: {defense_summary['defense_activated']}")
    print(f"  Threats found: {defense_summary['threats_found']}")
    print(f"  Final status: {defense_summary['final_status']}")
    
    # Test pipeline results details
    print("\n--- Pipeline Analysis ---")
    pipeline = adv_result['pipeline_results']
    
    if 'input_filtering' in pipeline:
        filter_result = pipeline['input_filtering']
        print(f"  Input filtering:")
        print(f"    Adversarial detected: {filter_result['detection']['is_adversarial']}")
        print(f"    Detection score: {filter_result['detection']['combined_score']:.3f}")
        print(f"    Filtering applied: {filter_result['filtering_applied']}")
        
    if 'randomized_defense' in pipeline:
        random_result = pipeline['randomized_defense']
        print(f"  Randomized defense:")
        print(f"    Overall confidence: {random_result['overall_confidence']:.3f}")
        print(f"    Threat level used: {random_result['adaptation_metadata']['threat_level']:.3f}")
        
    if 'adaptive_verification' in pipeline:
        adaptive_result = pipeline['adaptive_verification']
        print(f"  Adaptive verification:")
        print(f"    Verified: {adaptive_result['verified']}")
        print(f"    Adapted confidence: {adaptive_result['adapted_confidence']:.3f}")
        print(f"    Attack indicators: {len(adaptive_result['attack_indicators'])}")
    
    # Get system status
    status = integrated_system.get_system_status()
    print(f"\n--- System Status ---")
    print(f"  Total inputs processed: {status['system_stats']['total_inputs_processed']}")
    print(f"  Threats detected: {status['system_stats']['threats_detected']}")
    print(f"  Successful defenses: {status['system_stats']['successful_defenses']}")
    print(f"  Pipeline failures: {status['system_stats']['pipeline_failures']}")
    
    return integrated_system


def demonstrate_defense_effectiveness():
    """Demonstrate defense effectiveness against various attack types."""
    print("\n=== Defense Effectiveness Analysis ===")
    
    # Setup integrated system
    base_verifier = MockBaseVerifier(base_confidence=0.8)
    integrated_system = IntegratedDefenseSystem(base_verifier)
    model = create_test_model()
    
    # Train system
    clean_samples = create_test_data(150, 32)
    adversarial_samples = create_adversarial_data(clean_samples[:50], epsilon=0.3)
    integrated_system.train_defenses(clean_samples, adversarial_samples)
    
    # Test different attack types
    attack_types = [
        ('clean', lambda x: x),
        ('noise_low', lambda x: x + torch.randn_like(x) * 0.1),
        ('noise_medium', lambda x: x + torch.randn_like(x) * 0.3),
        ('noise_high', lambda x: x + torch.randn_like(x) * 0.6),
        ('scaled', lambda x: x * 2.0),
        ('shifted', lambda x: x + 1.0)
    ]
    
    results = {}
    
    for attack_name, attack_func in attack_types:
        print(f"\n--- Testing {attack_name} attack ---")
        
        # Generate test samples
        test_samples = create_test_data(20, 32)
        attack_results = []
        
        for sample in test_samples:
            attacked_sample = attack_func(sample)
            
            result = integrated_system.comprehensive_defense(
                input_data=attacked_sample,
                model=model
            )
            
            attack_results.append(result)
            
        # Analyze results
        verified_count = sum(1 for r in attack_results if r['final_decision']['verified'])
        avg_confidence = np.mean([r['defense_confidence'] for r in attack_results])
        avg_threat_level = np.mean([r['threat_level'] for r in attack_results])
        avg_processing_time = np.mean([r['processing_time'] for r in attack_results])
        
        results[attack_name] = {
            'verification_rate': verified_count / len(attack_results),
            'avg_confidence': avg_confidence,
            'avg_threat_level': avg_threat_level,
            'avg_processing_time': avg_processing_time
        }
        
        print(f"  Verification rate: {results[attack_name]['verification_rate']:.2%}")
        print(f"  Average confidence: {results[attack_name]['avg_confidence']:.3f}")
        print(f"  Average threat level: {results[attack_name]['avg_threat_level']:.3f}")
        print(f"  Average processing time: {results[attack_name]['avg_processing_time']:.3f}s")
    
    # Summary
    print(f"\n--- Effectiveness Summary ---")
    print("Attack Type        | Verification Rate | Avg Confidence | Threat Level")
    print("-" * 70)
    for attack_name, stats in results.items():
        print(f"{attack_name:18} | {stats['verification_rate']:15.2%} | {stats['avg_confidence']:12.3f} | {stats['avg_threat_level']:10.3f}")
    
    return results


def demonstrate_performance_analysis():
    """Demonstrate performance characteristics of defense system."""
    print("\n=== Performance Analysis ===")
    
    # Setup
    base_verifier = MockBaseVerifier(base_confidence=0.8)
    integrated_system = IntegratedDefenseSystem(base_verifier)
    model = create_test_model()
    
    # Train system
    clean_samples = create_test_data(100, 32)
    integrated_system.train_defenses(clean_samples)
    
    # Performance test with different input sizes
    input_sizes = [16, 32, 64, 128]
    batch_sizes = [1, 5, 10, 20]
    
    print("\nPerformance vs Input Size:")
    print("Input Size | Processing Time (ms) | Memory Usage")
    print("-" * 50)
    
    for input_size in input_sizes:
        test_input = torch.randn(input_size) * 0.3
        
        # Measure processing time
        start_time = time.time()
        result = integrated_system.comprehensive_defense(test_input, model)
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        print(f"{input_size:10d} | {processing_time:18.2f} | Memory tracking N/A")
    
    print("\nPerformance vs Threat Level:")
    print("Threat Level | Processing Time (ms) | Defenses Activated")
    print("-" * 55)
    
    test_input = torch.randn(32) * 0.3
    for threat_level in [0.2, 0.4, 0.6, 0.8]:
        start_time = time.time()
        result = integrated_system.comprehensive_defense(
            test_input, model, threat_level=threat_level
        )
        processing_time = (time.time() - start_time) * 1000
        
        defenses_activated = len(result['defense_summary']['defense_activated'])
        
        print(f"{threat_level:12.1f} | {processing_time:18.2f} | {defenses_activated:17d}")
    
    # Throughput analysis
    print("\nThroughput Analysis:")
    num_samples = 50
    test_inputs = [torch.randn(32) * 0.3 for _ in range(num_samples)]
    
    start_time = time.time()
    for test_input in test_inputs:
        integrated_system.comprehensive_defense(test_input, model)
    total_time = time.time() - start_time
    
    throughput = num_samples / total_time
    avg_time_per_sample = total_time / num_samples * 1000
    
    print(f"  Samples processed: {num_samples}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {throughput:.1f} samples/second")
    print(f"  Average time per sample: {avg_time_per_sample:.2f}ms")
    
    return {
        'throughput': throughput,
        'avg_time_per_sample': avg_time_per_sample,
        'total_time': total_time
    }


def main():
    """Run comprehensive defense system demonstration."""
    print("Comprehensive Defense Mechanisms Demonstration")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Demonstrate individual components
        adaptive_verifier = demonstrate_adaptive_verifier()
        input_filter = demonstrate_input_filter()
        randomized_defense = demonstrate_randomized_defense()
        
        # Demonstrate integrated system
        integrated_system = demonstrate_integrated_system()
        
        # Effectiveness analysis
        effectiveness_results = demonstrate_defense_effectiveness()
        
        # Performance analysis
        performance_results = demonstrate_performance_analysis()
        
        print("\n" + "=" * 60)
        print("Defense System Summary:")
        print("✓ AdaptiveVerifier - Dynamic threshold adjustment and attack learning")
        print("✓ InputFilter - Multi-method adversarial detection and sanitization")
        print("✓ RandomizedDefense - Smoothing and stochastic verification")
        print("✓ IntegratedDefenseSystem - Comprehensive multi-layered protection")
        print("✓ Defense Effectiveness - Tested against multiple attack types")
        print("✓ Performance Analysis - Scalability and throughput evaluation")
        
        print(f"\nKey Performance Metrics:")
        print(f"  Throughput: {performance_results['throughput']:.1f} samples/second")
        print(f"  Average processing time: {performance_results['avg_time_per_sample']:.2f}ms")
        
        print("\nThe comprehensive defense system is ready for production PoT protection!")
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()