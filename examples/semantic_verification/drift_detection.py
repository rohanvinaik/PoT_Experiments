#!/usr/bin/env python3
"""
Drift detection example demonstrating semantic drift monitoring.
Shows how to detect and analyze behavioral drift in model outputs over time.
"""

import torch
import numpy as np
import time
from typing import List, Tuple

# Import semantic and behavioral fingerprinting components
from pot.semantic import (
    ConceptLibrary,
    SemanticMatcher,
    BehavioralFingerprint,
    ContinuousMonitor,
    create_behavioral_monitor
)


def simulate_model_outputs(phase: str, n_outputs: int = 20) -> List[torch.Tensor]:
    """
    Simulate model outputs for different phases of operation.
    
    Args:
        phase: Operation phase ('normal', 'drift_start', 'drifted', 'anomaly')
        n_outputs: Number of outputs to generate
    
    Returns:
        List of simulated output tensors
    """
    outputs = []
    dim = 128
    
    for i in range(n_outputs):
        if phase == 'normal':
            # Normal behavior - centered around origin
            output = torch.randn(dim) * 0.5
        
        elif phase == 'drift_start':
            # Beginning of drift - gradual shift
            drift_factor = i / n_outputs  # Gradual increase
            output = torch.randn(dim) * 0.5 + torch.ones(dim) * drift_factor * 0.5
        
        elif phase == 'drifted':
            # Fully drifted - shifted distribution
            output = torch.randn(dim) * 0.5 + torch.ones(dim) * 2.0
        
        elif phase == 'anomaly':
            # Anomalous behavior - very different
            output = torch.randn(dim) * 2.0 + torch.ones(dim) * np.random.uniform(-3, 3)
        
        else:
            output = torch.randn(dim)
        
        outputs.append(output)
    
    return outputs


def semantic_drift_detection_example():
    """Demonstrate semantic drift detection using concept library."""
    print("=" * 70)
    print("SEMANTIC DRIFT DETECTION")
    print("=" * 70)
    
    # Create concept library with reference concepts
    print("\n1. Setting up concept library...")
    library = ConceptLibrary(dim=128, method='gaussian')
    
    # Add reference concept from normal behavior
    normal_embeddings = torch.stack(simulate_model_outputs('normal', 100))
    library.add_concept('normal_behavior', normal_embeddings)
    print(f"   ✓ Added 'normal_behavior' concept (100 samples)")
    
    # Add other known concepts
    for i, shift in enumerate([1.0, -1.0]):
        shifted_embeddings = torch.randn(50, 128) * 0.5 + torch.ones(128) * shift
        library.add_concept(f'known_pattern_{i}', shifted_embeddings)
        print(f"   ✓ Added 'known_pattern_{i}' concept")
    
    # Create semantic matcher
    matcher = SemanticMatcher(library=library, threshold=0.7)
    
    # Simulate different phases and detect drift
    print("\n2. Simulating model behavior over time...")
    print("-" * 50)
    
    phases = [
        ('normal', 30),
        ('drift_start', 30),
        ('drifted', 30),
        ('normal', 20),  # Return to normal
        ('anomaly', 10)
    ]
    
    all_drift_scores = []
    
    for phase_name, n_samples in phases:
        print(f"\nPhase: {phase_name.upper()}")
        
        # Generate outputs for this phase
        outputs = simulate_model_outputs(phase_name, n_samples)
        embeddings = torch.stack(outputs)
        
        # Compute drift from normal behavior
        drift_score = matcher.compute_semantic_drift(embeddings, 'normal_behavior')
        all_drift_scores.append((phase_name, drift_score))
        
        print(f"   Drift score: {drift_score:.3f}")
        
        # Check matches to known concepts
        avg_embedding = torch.mean(embeddings, dim=0)
        matches = matcher.match_to_library(avg_embedding)
        
        print("   Top matches:")
        for concept, score in list(matches.items())[:2]:
            print(f"     - {concept}: {score:.3f}")
        
        # Alert if significant drift
        if drift_score > 0.3:
            print(f"   ⚠️  WARNING: Significant drift detected (score={drift_score:.3f})")
    
    # Summary
    print("\n" + "=" * 50)
    print("DRIFT ANALYSIS SUMMARY:")
    for phase, score in all_drift_scores:
        bar = "█" * int(score * 20)
        print(f"  {phase:12s}: {bar:20s} {score:.3f}")


def behavioral_fingerprint_example():
    """Demonstrate behavioral fingerprinting for continuous monitoring."""
    print("\n" + "=" * 70)
    print("BEHAVIORAL FINGERPRINTING")
    print("=" * 70)
    
    # Create behavioral fingerprint system
    print("\n1. Setting up behavioral fingerprint system...")
    fingerprint = BehavioralFingerprint(
        window_size=30,
        fingerprint_dim=64,
        decay_factor=0.95
    )
    print(f"   ✓ Window size: {fingerprint.window_size}")
    print(f"   ✓ Fingerprint dimension: {fingerprint.fingerprint_dim}")
    print(f"   ✓ Decay factor: {fingerprint.decay_factor}")
    
    # Phase 1: Establish normal behavior
    print("\n2. Establishing normal behavior baseline...")
    normal_outputs = simulate_model_outputs('normal', 30)
    
    for i, output in enumerate(normal_outputs):
        fingerprint.update(output, metadata={'phase': 'normal', 'index': i})
    
    # Compute and set reference fingerprint
    reference_fp = fingerprint.compute_fingerprint()
    fingerprint.set_reference(reference_fp, threshold=0.7)
    print(f"   ✓ Reference fingerprint set (norm={torch.norm(reference_fp).item():.3f})")
    
    # Phase 2: Monitor for anomalies
    print("\n3. Monitoring for anomalies...")
    print("-" * 50)
    
    test_phases = [
        ('normal', 10, "Normal operation"),
        ('drift_start', 15, "Gradual drift beginning"),
        ('drifted', 10, "Significant drift"),
        ('anomaly', 5, "Anomalous behavior")
    ]
    
    anomaly_results = []
    
    for phase_name, n_samples, description in test_phases:
        print(f"\n{description}:")
        outputs = simulate_model_outputs(phase_name, n_samples)
        
        phase_anomalies = []
        for i, output in enumerate(outputs):
            fingerprint.update(output)
            
            # Check for anomaly every 5 updates
            if (i + 1) % 5 == 0:
                current_fp = fingerprint.compute_fingerprint()
                is_anomaly, score = fingerprint.detect_anomaly(current_fp)
                phase_anomalies.append((is_anomaly, score))
                
                if is_anomaly:
                    print(f"   Sample {i+1}: ⚠️  Anomaly detected (score={score:.3f})")
                else:
                    print(f"   Sample {i+1}: ✓ Normal (score={score:.3f})")
        
        anomaly_results.append((phase_name, phase_anomalies))
    
    # Phase 3: Drift detection
    print("\n4. Detecting behavioral drift...")
    has_drift, drift_score = fingerprint.detect_drift(window=10)
    
    if has_drift:
        print(f"   ⚠️  Drift detected: score={drift_score:.3f}")
    else:
        print(f"   ✓ No significant drift: score={drift_score:.3f}")
    
    # Get statistics
    stats = fingerprint.get_statistics()
    print(f"\n5. Fingerprint Statistics:")
    print(f"   - Total updates: {stats['n_updates']}")
    print(f"   - Current observations: {stats['n_observations']}")
    print(f"   - History size: {stats['history_size']}")
    if 'fingerprint_diversity' in stats:
        print(f"   - Fingerprint diversity: {stats['fingerprint_diversity']:.3f}")


def continuous_monitoring_example():
    """Demonstrate continuous monitoring with semantic integration."""
    print("\n" + "=" * 70)
    print("CONTINUOUS MONITORING WITH SEMANTIC ANALYSIS")
    print("=" * 70)
    
    # Create semantic library
    print("\n1. Setting up semantic library...")
    library = ConceptLibrary(dim=64, method='gaussian')
    
    # Add known behavior patterns
    for i, name in enumerate(['normal', 'warning', 'critical']):
        embeddings = torch.randn(50, 64) * 0.5 + torch.ones(64) * i
        library.add_concept(f'{name}_pattern', embeddings)
    print(f"   ✓ Added {len(library.list_concepts())} behavior patterns")
    
    # Create behavioral monitor with semantic integration
    print("\n2. Creating continuous monitor...")
    monitor = create_behavioral_monitor(
        window_size=20,
        fingerprint_dim=64,
        semantic_library=library
    )
    print(f"   ✓ Monitor created with semantic matching")
    
    # Simulate continuous operation
    print("\n3. Simulating continuous model operation...")
    print("-" * 50)
    
    operation_sequence = [
        ('normal', 25, 0.1),      # Normal for a while
        ('drift_start', 15, 0.2),  # Start drifting
        ('drifted', 20, 0.5),      # Significant drift
        ('anomaly', 10, 1.0),      # Anomalous period
        ('normal', 15, 0.1)        # Recovery
    ]
    
    alert_count = 0
    drift_count = 0
    
    for phase, n_samples, expected_severity in operation_sequence:
        print(f"\n[{phase.upper()}] Processing {n_samples} samples...")
        
        outputs = simulate_model_outputs(phase, n_samples)
        
        for i, output in enumerate(outputs):
            result = monitor.process_output(output, metadata={'phase': phase, 'sample': i})
            
            # Check for anomalies
            if 'anomaly_check' in result:
                if result['anomaly_check']['is_anomaly']:
                    alert_count += 1
                    score = result['anomaly_check']['score']
                    print(f"   Sample {i+1}: ⚠️  ALERT - Anomaly detected (score={score:.3f})")
                    
                    # Check semantic matches
                    if 'semantic_matches' in result:
                        top_match = list(result['semantic_matches'].items())[0]
                        print(f"              Closest pattern: {top_match[0]} ({top_match[1]:.3f})")
            
            # Check for drift
            if 'drift_check' in result:
                if result['drift_check']['has_drift']:
                    drift_count += 1
                    score = result['drift_check']['score']
                    print(f"   📊 DRIFT detected (score={score:.3f})")
    
    # Get final summary
    print("\n4. Monitoring Summary:")
    print("=" * 50)
    summary = monitor.get_monitoring_summary()
    
    print(f"   Total checks: {summary['n_checks']}")
    print(f"   Anomalies detected: {summary['n_anomalies']}")
    print(f"   Drifts detected: {summary['n_drifts']}")
    
    if summary['n_anomalies'] > 0 and 'anomaly_stats' in summary:
        print(f"\n   Anomaly Statistics:")
        print(f"     - Mean score: {summary['anomaly_stats']['mean_score']:.3f}")
        print(f"     - Max score: {summary['anomaly_stats']['max_score']:.3f}")
        print(f"     - Frequency: {summary['anomaly_stats']['frequency']:.1%}")
    
    if 'fingerprint_stats' in summary:
        fp_stats = summary['fingerprint_stats']
        if 'fingerprint_diversity' in fp_stats:
            print(f"\n   Behavioral Diversity: {fp_stats['fingerprint_diversity']:.3f}")


def visualize_drift_timeline():
    """Create a simple text-based visualization of drift over time."""
    print("\n" + "=" * 70)
    print("DRIFT TIMELINE VISUALIZATION")
    print("=" * 70)
    
    # Create monitor
    monitor = create_behavioral_monitor(window_size=15, fingerprint_dim=32)
    
    # Simulate timeline
    timeline = []
    phases = [
        ('normal', 20, '✓'),
        ('drift_start', 15, '~'),
        ('drifted', 20, '!'),
        ('anomaly', 10, '⚠'),
        ('normal', 20, '✓')
    ]
    
    print("\nProcessing timeline...")
    for phase, duration, symbol in phases:
        outputs = simulate_model_outputs(phase, duration)
        
        for output in outputs:
            result = monitor.process_output(output)
            
            # Determine status
            if 'anomaly_check' in result and result['anomaly_check']['is_anomaly']:
                timeline.append('⚠')
            elif 'drift_check' in result and result['drift_check']['has_drift']:
                timeline.append('!')
            else:
                timeline.append(symbol)
    
    # Display timeline
    print("\nDrift Timeline:")
    print("-" * 70)
    
    # Legend
    print("Legend: ✓=Normal  ~=Drift Starting  !=Drifted  ⚠=Anomaly")
    print()
    
    # Timeline in rows of 50
    for i in range(0, len(timeline), 50):
        row = timeline[i:i+50]
        print(f"[{i:3d}-{i+len(row)-1:3d}] {''.join(row)}")
    
    # Statistics
    normal_count = timeline.count('✓')
    drift_count = timeline.count('~') + timeline.count('!')
    anomaly_count = timeline.count('⚠')
    
    print("\nTimeline Statistics:")
    print(f"  Normal: {normal_count}/{len(timeline)} ({normal_count/len(timeline)*100:.1f}%)")
    print(f"  Drift: {drift_count}/{len(timeline)} ({drift_count/len(timeline)*100:.1f}%)")
    print(f"  Anomaly: {anomaly_count}/{len(timeline)} ({anomaly_count/len(timeline)*100:.1f}%)")


def adaptive_threshold_example():
    """Demonstrate adaptive threshold adjustment based on drift detection."""
    print("\n" + "=" * 70)
    print("ADAPTIVE THRESHOLD ADJUSTMENT")
    print("=" * 70)
    
    # Create fingerprint system
    fingerprint = BehavioralFingerprint(window_size=20, fingerprint_dim=32)
    
    # Initial threshold
    current_threshold = 0.8
    print(f"\n1. Initial anomaly threshold: {current_threshold:.2f}")
    
    # Track false positive rate
    false_positives = 0
    true_positives = 0
    total_checks = 0
    
    print("\n2. Monitoring and adjusting thresholds...")
    print("-" * 50)
    
    # Simulate different phases with known labels
    test_data = [
        ('normal', 30, False),  # Not anomalous
        ('drift_start', 20, False),  # Mild drift, not anomalous
        ('anomaly', 15, True),  # True anomaly
        ('normal', 25, False),  # Back to normal
    ]
    
    for phase, n_samples, is_true_anomaly in test_data:
        print(f"\n[{phase.upper()}] - True anomaly: {is_true_anomaly}")
        
        outputs = simulate_model_outputs(phase, n_samples)
        
        for output in outputs:
            fingerprint.update(output)
            
            # Set reference if first phase
            if total_checks == 0:
                ref_fp = fingerprint.compute_fingerprint()
                fingerprint.set_reference(ref_fp, current_threshold)
            
            # Check for anomaly
            if total_checks > 0 and total_checks % 5 == 0:
                current_fp = fingerprint.compute_fingerprint()
                is_detected, score = fingerprint.detect_anomaly(current_fp, current_threshold)
                
                # Update statistics
                if is_detected and not is_true_anomaly:
                    false_positives += 1
                elif is_detected and is_true_anomaly:
                    true_positives += 1
                
                # Adaptive threshold adjustment
                if false_positives > 3:
                    # Too many false positives, increase threshold
                    current_threshold = min(0.95, current_threshold + 0.05)
                    fingerprint.reference_threshold = current_threshold
                    false_positives = 0  # Reset counter
                    print(f"   📊 Threshold increased to {current_threshold:.2f} (reducing sensitivity)")
                
                elif total_checks > 50 and true_positives == 0 and is_true_anomaly:
                    # Missing true anomalies, decrease threshold
                    current_threshold = max(0.5, current_threshold - 0.05)
                    fingerprint.reference_threshold = current_threshold
                    print(f"   📊 Threshold decreased to {current_threshold:.2f} (increasing sensitivity)")
            
            total_checks += 1
    
    print(f"\n3. Final Statistics:")
    print(f"   - Final threshold: {current_threshold:.2f}")
    print(f"   - True positives: {true_positives}")
    print(f"   - False positives: {false_positives}")
    print(f"   - Total checks: {total_checks}")


def main():
    """Run all drift detection examples."""
    print("\n" + "=" * 80)
    print(" " * 25 + "DRIFT DETECTION AND MONITORING EXAMPLES")
    print("=" * 80)
    
    # 1. Semantic drift detection
    semantic_drift_detection_example()
    
    # 2. Behavioral fingerprinting
    behavioral_fingerprint_example()
    
    # 3. Continuous monitoring
    continuous_monitoring_example()
    
    # 4. Drift timeline visualization
    visualize_drift_timeline()
    
    # 5. Adaptive thresholds
    adaptive_threshold_example()
    
    print("\n" + "=" * 80)
    print(" " * 30 + "EXAMPLES COMPLETE")
    print("=" * 80)
    print("\nThese examples demonstrate various approaches to detecting and")
    print("monitoring semantic and behavioral drift in model outputs.")
    print("\nKey takeaways:")
    print("  • Semantic drift tracks deviation from known concept distributions")
    print("  • Behavioral fingerprinting captures temporal patterns")
    print("  • Continuous monitoring enables real-time anomaly detection")
    print("  • Adaptive thresholds can reduce false positives/negatives")


if __name__ == "__main__":
    main()