#!/usr/bin/env python3
"""
Sequential Decision Logic Demo for PoT Framework

This script demonstrates the Sequential Probability Ratio Test (SPRT) implementation
with comprehensive examples showing:
- Different configuration presets and their effects
- Early stopping criteria in action
- Confidence evolution and decision boundaries
- Performance analysis across multiple experiments
- Visualization tools for understanding SPRT behavior
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import time
import json
from pathlib import Path

from pot.experiments.sequential_decision import (
    SequentialDecisionMaker, SPRTConfig, DecisionOutcome, StoppingReason,
    SequentialVisualizer, PAPER_CONFIGS, create_sequential_decision_maker,
    run_sequential_experiment, analyze_sequential_performance, save_sequential_results
)

def demo_basic_sprt():
    """Demonstrate basic SPRT functionality."""
    print("🎯 Basic SPRT Demo")
    print("=" * 40)
    
    # Create decision maker with standard configuration
    decision_maker = create_sequential_decision_maker("standard")
    config = decision_maker.config
    
    print(f"Configuration:")
    print(f"  α (Type I error): {config.alpha}")
    print(f"  β (Type II error): {config.beta}")
    print(f"  H0 (null hypothesis): {config.h0}")
    print(f"  H1 (alternative hypothesis): {config.h1}")
    print(f"  Max queries: {config.max_queries}")
    print(f"  Decision boundaries: [{decision_maker.boundary_reject:.3f}, {decision_maker.boundary_accept:.3f}]")
    
    print(f"\n🔄 Running sequential observations...")
    
    # Simulate observations
    observations = [True, True, False, True, True, True, False, True, True]
    
    for i, obs in enumerate(observations):
        decision = decision_maker.update(obs, {"observation_id": i})
        confidence = decision_maker.get_confidence()
        state = decision_maker.get_current_state()
        
        print(f"  Obs {i+1}: {'Pass' if obs else 'Fail'} → "
              f"LLR={state['log_likelihood_ratio']:.3f}, "
              f"Confidence={confidence:.3f}, "
              f"Decision={decision.value}")
        
        if decision != DecisionOutcome.CONTINUE:
            print(f"  🏁 Final decision: {decision.value}")
            print(f"  📊 Total observations: {state['observations']}")
            print(f"  ⏱️  Total time: {state['elapsed_time']:.4f}s")
            break
    
    return decision_maker.final_result

def demo_configuration_presets():
    """Demonstrate different configuration presets."""
    print("\n🔧 Configuration Presets Demo")
    print("=" * 40)
    
    test_observations = [True, True, False, True, True, True, True]
    
    for preset_name in ["fast", "standard", "conservative", "research"]:
        print(f"\n📋 {preset_name.upper()} Preset:")
        
        decision_maker = create_sequential_decision_maker(preset_name)
        config = decision_maker.config
        
        print(f"  Error rates: α={config.alpha}, β={config.beta}")
        print(f"  Hypotheses: H0={config.h0}, H1={config.h1}")
        print(f"  Boundaries: [{decision_maker.boundary_reject:.3f}, {decision_maker.boundary_accept:.3f}]")
        print(f"  Max queries: {config.max_queries}")
        
        # Run experiment
        result = run_sequential_experiment(decision_maker, test_observations)
        
        print(f"  Result: {result.final_decision.value} after {result.total_observations} observations")
        print(f"  Confidence: {result.final_confidence:.3f}")
        print(f"  Stopping reason: {result.stopping_reason.value}")

def demo_early_stopping():
    """Demonstrate different early stopping criteria."""
    print("\n⏰ Early Stopping Criteria Demo")
    print("=" * 40)
    
    # Demo 1: Max queries stopping
    print("\n🔢 Max Queries Stopping:")
    config1 = SPRTConfig(alpha=0.01, beta=0.01, h0=0.5, h1=0.6, max_queries=3)
    decision_maker1 = SequentialDecisionMaker(config1)
    
    # Alternating observations that won't reach decision boundary quickly
    for i in range(5):
        obs = i % 2 == 0
        decision = decision_maker1.update(obs)
        print(f"  Obs {i+1}: {'Pass' if obs else 'Fail'} → {decision.value}")
        if decision != DecisionOutcome.CONTINUE:
            break
    
    # Demo 2: Confidence threshold stopping
    print("\n📊 Confidence Threshold Stopping:")
    config2 = SPRTConfig(alpha=0.05, beta=0.05, h0=0.5, h1=0.95, 
                        confidence_threshold=0.85, max_queries=20)
    decision_maker2 = SequentialDecisionMaker(config2)
    
    # Many passes to build confidence quickly
    for i in range(5):
        decision = decision_maker2.update(True)
        confidence = decision_maker2.get_confidence()
        print(f"  Pass {i+1}: Confidence={confidence:.3f} → {decision.value}")
        if decision != DecisionOutcome.CONTINUE:
            break
    
    # Demo 3: Time limit stopping
    print("\n⏱️  Time Limit Stopping:")
    config3 = SPRTConfig(alpha=0.05, beta=0.05, h0=0.5, h1=0.9, 
                        time_limit=0.001, max_queries=100)  # Very short time limit
    decision_maker3 = SequentialDecisionMaker(config3)
    
    time.sleep(0.002)  # Wait to exceed time limit
    decision = decision_maker3.update(True)
    print(f"  After delay: {decision.value}")

def demo_confidence_evolution():
    """Demonstrate confidence evolution patterns."""
    print("\n📈 Confidence Evolution Demo")
    print("=" * 40)
    
    decision_maker = create_sequential_decision_maker("standard")
    
    # Pattern 1: All passes (should increase confidence)
    print("\n✅ All Passes Pattern:")
    decision_maker.reset()
    for i in range(6):
        decision = decision_maker.update(True)
        confidence = decision_maker.get_confidence()
        llr = decision_maker.log_likelihood_ratio
        print(f"  Pass {i+1}: LLR={llr:.3f}, Confidence={confidence:.3f}")
        if decision != DecisionOutcome.CONTINUE:
            break
    
    # Pattern 2: All fails (should decrease confidence)
    print("\n❌ All Fails Pattern:")
    decision_maker.reset()
    for i in range(6):
        decision = decision_maker.update(False)
        confidence = decision_maker.get_confidence()
        llr = decision_maker.log_likelihood_ratio
        print(f"  Fail {i+1}: LLR={llr:.3f}, Confidence={confidence:.3f}")
        if decision != DecisionOutcome.CONTINUE:
            break
    
    # Pattern 3: Mixed observations
    print("\n🔄 Mixed Pattern:")
    decision_maker.reset()
    mixed_pattern = [True, True, False, True, False, True, True, False]
    for i, obs in enumerate(mixed_pattern):
        decision = decision_maker.update(obs)
        confidence = decision_maker.get_confidence()
        llr = decision_maker.log_likelihood_ratio
        print(f"  {'Pass' if obs else 'Fail'} {i+1}: LLR={llr:.3f}, Confidence={confidence:.3f}")
        if decision != DecisionOutcome.CONTINUE:
            break

def demo_performance_comparison():
    """Demonstrate performance comparison across configurations."""
    print("\n⚡ Performance Comparison Demo")
    print("=" * 40)
    
    # Generate test scenarios
    scenarios = {
        "mostly_legitimate": [True] * 8 + [False] * 2,
        "mostly_illegitimate": [False] * 8 + [True] * 2,
        "mixed_evidence": [True, False] * 5,
        "strong_legitimate": [True] * 10,
        "strong_illegitimate": [False] * 10
    }
    
    presets = ["fast", "standard", "conservative"]
    
    print(f"{'Scenario':<20} {'Preset':<12} {'Decision':<8} {'Obs':<4} {'Conf':<6} {'Time':<6}")
    print("-" * 70)
    
    for scenario_name, observations in scenarios.items():
        for preset in presets:
            decision_maker = create_sequential_decision_maker(preset)
            result = run_sequential_experiment(decision_maker, observations)
            
            print(f"{scenario_name:<20} {preset:<12} {result.final_decision.value:<8} "
                  f"{result.total_observations:<4} {result.final_confidence:<6.3f} "
                  f"{result.total_time:<6.4f}")

def demo_efficiency_analysis():
    """Demonstrate efficiency gains from sequential testing."""
    print("\n📊 Efficiency Analysis Demo")
    print("=" * 40)
    
    # Compare SPRT vs fixed-sample testing
    n_experiments = 50
    fixed_sample_size = 20
    
    # Generate random experiments
    np.random.seed(42)
    results = []
    
    for i in range(n_experiments):
        # Generate observations with some legitimate models (70% pass rate)
        # and some illegitimate models (30% pass rate)
        if i < n_experiments // 2:
            # Legitimate model
            observations = np.random.random(fixed_sample_size) < 0.7
        else:
            # Illegitimate model
            observations = np.random.random(fixed_sample_size) < 0.3
        
        decision_maker = create_sequential_decision_maker("standard")
        result = run_sequential_experiment(decision_maker, observations.tolist())
        results.append(result)
    
    # Analyze efficiency
    analysis = analyze_sequential_performance(results)
    
    print(f"📈 Efficiency Results:")
    print(f"  Total experiments: {analysis['total_experiments']}")
    print(f"  Average observations used: {analysis['average_observations']:.1f}")
    print(f"  Fixed sample size: {fixed_sample_size}")
    print(f"  Efficiency gain: {(1 - analysis['average_observations']/fixed_sample_size)*100:.1f}%")
    print(f"  Time savings: {analysis['average_time']:.4f}s avg per experiment")
    
    print(f"\n📊 Decision Distribution:")
    for decision, count in analysis['decision_counts'].items():
        if count > 0:
            percentage = (count / analysis['total_experiments']) * 100
            print(f"  {decision}: {count} ({percentage:.1f}%)")
    
    print(f"\n🛑 Stopping Reasons:")
    for reason, count in analysis['stopping_reason_counts'].items():
        if count > 0:
            percentage = (count / analysis['total_experiments']) * 100
            print(f"  {reason}: {count} ({percentage:.1f}%)")
    
    return results

def demo_visualizations():
    """Demonstrate visualization capabilities."""
    print("\n🎨 Visualization Demo")
    print("=" * 40)
    
    # Create output directory
    output_dir = Path("sequential_demo_plots")
    output_dir.mkdir(exist_ok=True)
    
    visualizer = SequentialVisualizer(output_dir)
    
    # 1. Decision boundary plot
    print("📊 Creating decision boundary plot...")
    config = PAPER_CONFIGS["standard"]
    boundary_plot = visualizer.plot_decision_boundary(config)
    print(f"  ✅ Saved: {boundary_plot}")
    
    # 2. Confidence evolution plot
    print("📈 Creating confidence evolution plot...")
    decision_maker = create_sequential_decision_maker("standard")
    observations = [True, True, False, True, True, True, False, True, True, True]
    result = run_sequential_experiment(decision_maker, observations)
    
    confidence_plot = visualizer.plot_confidence_evolution(result)
    print(f"  ✅ Saved: {confidence_plot}")
    
    # 3. ROC curves (using previous results)
    print("📊 Creating ROC curves...")
    # Generate some results with ground truth for ROC analysis
    roc_results = []
    np.random.seed(123)
    
    for i in range(30):
        ground_truth = i % 2 == 0  # Alternate ground truth
        
        if ground_truth:
            # Legitimate model - higher pass rate
            observations = (np.random.random(10) < 0.8).tolist()
        else:
            # Illegitimate model - lower pass rate
            observations = (np.random.random(10) < 0.3).tolist()
        
        decision_maker = create_sequential_decision_maker("standard")
        result = run_sequential_experiment(decision_maker, observations)
        result.metadata["ground_truth"] = ground_truth
        roc_results.append(result)
    
    roc_plot = visualizer.plot_roc_curves(roc_results)
    print(f"  ✅ Saved: {roc_plot}")
    
    return output_dir

def demo_integration_with_pot():
    """Demonstrate integration with PoT framework concepts."""
    print("\n🔗 PoT Framework Integration Demo")
    print("=" * 40)
    
    # Simulate PoT verification process
    print("🧪 Simulating PoT Verification Process:")
    
    # Create decision maker with PoT-like parameters
    config = SPRTConfig(
        alpha=0.001,        # Very low false positive rate (similar to PoT's alpha)
        beta=0.001,         # Very low false negative rate (similar to PoT's beta) 
        h0=0.01,           # Low null hypothesis (similar to PoT's tau_id)
        h1=0.9,            # High legitimate model performance
        max_queries=100,
        confidence_threshold=0.99
    )
    
    decision_maker = SequentialDecisionMaker(config)
    
    print(f"  Configuration similar to PoT paper:")
    print(f"    α = {config.alpha} (false positive rate)")
    print(f"    β = {config.beta} (false negative rate)")
    print(f"    H0 = {config.h0} (illegitimate model probability)")
    print(f"    H1 = {config.h1} (legitimate model probability)")
    
    # Simulate challenge-response verification
    challenge_families = ["vision:freq", "vision:texture", "lm:templates"]
    
    for family in challenge_families:
        print(f"\n  🎯 Challenge Family: {family}")
        
        # Simulate challenges for this family
        for i in range(5):
            # Simulate challenge result (legitimate model should pass most)
            challenge_passed = np.random.random() > 0.2  # 80% pass rate
            
            challenge_metadata = {
                "challenge_family": family,
                "challenge_id": f"{family}_{i}",
                "difficulty": 0.5
            }
            
            decision = decision_maker.update(challenge_passed, challenge_metadata)
            confidence = decision_maker.get_confidence()
            
            print(f"    Challenge {i+1}: {'✅ Pass' if challenge_passed else '❌ Fail'} "
                  f"(Confidence: {confidence:.3f})")
            
            if decision != DecisionOutcome.CONTINUE:
                print(f"    🏁 Verification complete: {decision.value}")
                print(f"    📊 Total challenges: {len(decision_maker.observations)}")
                print(f"    🎯 Final confidence: {confidence:.4f}")
                break
        
        if decision_maker.is_complete:
            break
    
    return decision_maker.final_result

def main():
    """Run comprehensive sequential decision demonstration."""
    print("🧪 Sequential Decision Logic Demo")
    print("=" * 60)
    print("Comprehensive demonstration of SPRT implementation with")
    print("early stopping, confidence tracking, and PoT integration.\n")
    
    try:
        # Run all demonstrations
        basic_result = demo_basic_sprt()
        demo_configuration_presets()
        demo_early_stopping()
        demo_confidence_evolution()
        demo_performance_comparison()
        efficiency_results = demo_efficiency_analysis()
        plots_dir = demo_visualizations()
        pot_result = demo_integration_with_pot()
        
        # Save comprehensive results
        print("\n💾 Saving Results")
        print("=" * 40)
        
        all_results = [basic_result, pot_result] + efficiency_results
        output_file = Path("sequential_decision_demo_results.json")
        saved_file = save_sequential_results(all_results, output_file)
        
        print(f"✅ Results saved to: {saved_file}")
        print(f"📊 Plots saved to: {plots_dir}")
        
        # Final summary
        print(f"\n🎉 Demo completed successfully!")
        print(f"\n✅ Key Features Demonstrated:")
        print(f"   • Sequential Probability Ratio Test (SPRT) mechanics")
        print(f"   • Multiple configuration presets (fast, standard, conservative, research)")
        print(f"   • Early stopping criteria (max queries, confidence, time limits)")
        print(f"   • Confidence evolution and decision boundaries")
        print(f"   • Performance analysis and efficiency gains")
        print(f"   • Comprehensive visualization tools")
        print(f"   • Integration with PoT framework concepts")
        print(f"   • Real-time decision tracking and metadata")
        
        # Performance summary
        if efficiency_results:
            analysis = analyze_sequential_performance(efficiency_results)
            print(f"\n📊 Performance Summary:")
            print(f"   • Average observations: {analysis['average_observations']:.1f}")
            print(f"   • Efficiency gain: {analysis['efficiency_gain']}")
            print(f"   • Average confidence: {analysis['average_confidence']:.3f}")
            print(f"   • Success rate: {(analysis['decision_counts'].get('accept', 0) + analysis['decision_counts'].get('reject', 0)) / len(efficiency_results) * 100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)