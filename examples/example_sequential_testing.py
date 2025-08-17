#!/usr/bin/env python3
"""
Example usage of Sequential Testing Implementation for Language Model Verification
Demonstrates SPRT (Sequential Probability Ratio Testing) with early stopping
"""

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for pot imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pot.lm.sequential_tester import (
    SequentialTester,
    AdaptiveSequentialTester,
    GroupSequentialTester,
    SequentialVerificationSession,
    simulate_sequential_test,
    compute_operating_characteristics
)
from pot.lm.template_challenges import (
    TemplateChallenger,
    ChallengeEvaluator,
    create_challenge_suite
)


def demo_basic_sprt():
    """Demonstrate basic SPRT testing"""
    print("=" * 60)
    print("1. Basic Sequential Testing (SPRT)")
    print("=" * 60)
    
    # Create tester with error rates
    tester = SequentialTester(
        alpha=0.05,  # Type I error: rejecting genuine model
        beta=0.05,   # Type II error: accepting fake model
        p0=0.5,      # H0: Model performs at chance level
        p1=0.8,      # H1: Model performs well (genuine)
        max_trials=100,
        min_trials=5
    )
    
    print(f"\nTest parameters:")
    print(f"  α (Type I error): {tester.alpha}")
    print(f"  β (Type II error): {tester.beta}")
    print(f"  H0 success rate: {tester.p0}")
    print(f"  H1 success rate: {tester.p1}")
    print(f"  Decision boundaries: [{tester.log_B:.2f}, {tester.log_A:.2f}]")
    
    # Simulate model with 75% success rate
    np.random.seed(42)
    true_success_rate = 0.75
    print(f"\nSimulating model with {true_success_rate:.0%} success rate...")
    
    results = []
    for trial in range(100):
        # Simulate trial outcome
        success = np.random.random() < true_success_rate
        results.append(success)
        
        # Update tester
        decision = tester.update(success)
        
        # Print progress every 10 trials
        if (trial + 1) % 10 == 0 or decision is not None:
            stats = tester.get_statistics()
            print(f"  Trial {trial + 1}: Success rate = {stats['success_rate']:.2f}, "
                  f"LLR = {stats['log_likelihood_ratio']:.2f}, "
                  f"Confidence = {stats['confidence']:.2f}")
        
        if decision is not None:
            print(f"\n✓ Decision made at trial {trial + 1}: {decision.upper()}")
            print(f"  {'Model verified as GENUINE' if decision == 'reject' else 'Model classified as FAKE/MODIFIED'}")
            break
    
    # Final statistics
    final_stats = tester.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Total trials: {final_stats['num_trials']}")
    print(f"  Success rate: {final_stats['success_rate']:.2%}")
    print(f"  Final confidence: {final_stats['confidence']:.2%}")
    print(f"  Expected trials under H0: {final_stats['expected_trials_h0']:.1f}")
    print(f"  Expected trials under H1: {final_stats['expected_trials_h1']:.1f}")
    
    return tester


def demo_adaptive_sprt():
    """Demonstrate adaptive SPRT that adjusts parameters"""
    print("\n" + "=" * 60)
    print("2. Adaptive Sequential Testing")
    print("=" * 60)
    
    # Create adaptive tester
    tester = AdaptiveSequentialTester(
        initial_alpha=0.05,
        initial_beta=0.05,
        adaptation_rate=0.1,
        window_size=20,
        p0=0.5,
        p1=0.8
    )
    
    print(f"\nAdaptive test with window size {tester.window_size}")
    
    # Simulate changing performance
    print("\nSimulating model with changing performance...")
    print("  Phase 1 (trials 1-30): 60% success rate")
    print("  Phase 2 (trials 31-60): 85% success rate")
    print("  Phase 3 (trials 61+): 70% success rate")
    
    np.random.seed(42)
    phases = [(30, 0.6), (30, 0.85), (40, 0.7)]
    
    trial_num = 0
    for phase_trials, success_rate in phases:
        for _ in range(phase_trials):
            trial_num += 1
            success = np.random.random() < success_rate
            decision = tester.update(success)
            
            # Print adaptation info
            if trial_num % 20 == 0:
                if tester.adaptation_history:
                    last_adapt = tester.adaptation_history[-1]
                    print(f"  Trial {trial_num}: Recent rate = {last_adapt['recent_rate']:.2f}, "
                          f"α = {last_adapt['alpha']:.3f}, "
                          f"p1 = {last_adapt['p1']:.2f}")
            
            if decision is not None:
                print(f"\n✓ Decision at trial {trial_num}: {decision.upper()}")
                break
        
        if decision is not None:
            break
    
    return tester


def demo_group_sequential():
    """Demonstrate group sequential testing"""
    print("\n" + "=" * 60)
    print("3. Group Sequential Testing")
    print("=" * 60)
    
    # Create group sequential tester
    tester = GroupSequentialTester(
        num_stages=4,
        trials_per_stage=25,
        alpha=0.05,
        beta=0.05,
        spending_function='obrien_fleming'
    )
    
    print(f"\nGroup sequential test with {tester.num_stages} stages")
    print(f"Trials per stage: {tester.trials_per_stage}")
    print(f"Spending function: O'Brien-Fleming")
    
    # Simulate stages
    np.random.seed(42)
    true_success_rate = 0.72
    
    print(f"\nSimulating model with {true_success_rate:.0%} success rate...")
    
    for stage in range(tester.num_stages):
        # Generate stage results
        successes = sum(np.random.random() < true_success_rate 
                       for _ in range(tester.trials_per_stage))
        
        print(f"\nStage {stage + 1}:")
        print(f"  Successes: {successes}/{tester.trials_per_stage}")
        
        # Update stage
        decision = tester.update_stage(successes, tester.trials_per_stage)
        
        # Get statistics
        stats = tester.get_statistics()
        print(f"  Cumulative: {stats['cumulative_successes']}/{stats['cumulative_trials']} "
              f"({stats['success_rate']:.2%})")
        
        if decision is not None:
            print(f"\n✓ Early stopping at stage {stage + 1}: {decision.upper()}")
            break
    
    if decision is None:
        print("\n✗ No early stopping - decision at final stage")
    
    return tester


def demo_verification_session():
    """Demonstrate complete verification session"""
    print("\n" + "=" * 60)
    print("4. Complete Verification Session")
    print("=" * 60)
    
    # Create components
    tester = SequentialTester(alpha=0.05, beta=0.05, p0=0.5, p1=0.8)
    challenger = TemplateChallenger(difficulty_curve='adaptive')
    evaluator = ChallengeEvaluator(fuzzy_threshold=0.85)
    
    # Mock model runner (simulates 70% accuracy)
    def mock_model_runner(prompt):
        # Simple simulation: correct 70% of the time
        if np.random.random() < 0.7:
            # Try to extract expected answer from prompt
            if '[MASK]' in prompt:
                return "correct_answer"
        return "wrong_answer"
    
    # Create session
    session = SequentialVerificationSession(
        tester=tester,
        challenger=challenger,
        evaluator=evaluator,
        model_runner=mock_model_runner
    )
    
    print("\nRunning verification session...")
    print("  Max challenges: 50")
    print("  Early stopping: Enabled")
    
    # Run verification
    results = session.run_verification(
        max_challenges=50,
        early_stop=True
    )
    
    print(f"\nVerification Results:")
    print(f"  Verified: {results.get('verified', 'Unknown')}")
    print(f"  Decision: {results.get('decision', 'None')}")
    print(f"  Trials used: {results['num_trials']}")
    print(f"  Success rate: {results['success_rate']:.2%}")
    print(f"  Confidence: {results['confidence']:.2%}")
    print(f"  Early stopped: {results['early_stopped']}")
    print(f"  Duration: {results['duration']:.2f}s")
    
    return results


def demo_operating_characteristics():
    """Demonstrate operating characteristics analysis"""
    print("\n" + "=" * 60)
    print("5. Operating Characteristics Analysis")
    print("=" * 60)
    
    # Compute characteristics for different parameters
    characteristics = compute_operating_characteristics(
        alpha_range=np.array([0.01, 0.05, 0.10]),
        beta_range=np.array([0.01, 0.05, 0.10]),
        p0=0.5,
        p1=0.8
    )
    
    print("\nExpected sample sizes for different error rates:")
    print("  α     β     E[N|H0]  E[N|H1]  Power")
    print("  " + "-" * 40)
    
    for i in range(len(characteristics['alpha'])):
        print(f"  {characteristics['alpha'][i]:.2f}  "
              f"{characteristics['beta'][i]:.2f}  "
              f"{characteristics['expected_n_h0'][i]:7.1f}  "
              f"{characteristics['expected_n_h1'][i]:7.1f}  "
              f"{characteristics['power'][i]:.2f}")


def demo_simulation():
    """Demonstrate Monte Carlo simulation"""
    print("\n" + "=" * 60)
    print("6. Monte Carlo Simulation")
    print("=" * 60)
    
    print("\nSimulating 1000 sequential tests...")
    
    # Simulate under different true success rates
    true_rates = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    print("\n  True p  Reject%  Avg N   Std N")
    print("  " + "-" * 32)
    
    for true_p in true_rates:
        results = simulate_sequential_test(
            true_p=true_p,
            num_simulations=200,  # Reduced for speed
            alpha=0.05,
            beta=0.05,
            p0=0.5,
            p1=0.8
        )
        
        print(f"  {true_p:.1f}    {results['reject_rate']:6.1%}  "
              f"{results['avg_sample_size']:6.1f}  "
              f"{results['std_sample_size']:6.1f}")


def plot_sequential_test(tester):
    """Plot the progress of a sequential test"""
    if not hasattr(tester, 'plot_progress'):
        return
    
    plot_data = tester.plot_progress()
    
    if not plot_data['trials']:
        return
    
    plt.figure(figsize=(10, 6))
    
    # Plot cumulative log likelihood ratio
    plt.plot(plot_data['trials'], plot_data['cumulative_llr'], 
             'b-', linewidth=2, label='Log Likelihood Ratio')
    
    # Plot boundaries
    plt.axhline(y=plot_data['upper_boundary'][0], color='r', 
                linestyle='--', label=f'Reject H0 (α={tester.alpha})')
    plt.axhline(y=plot_data['lower_boundary'][0], color='g', 
                linestyle='--', label=f'Accept H0 (β={tester.beta})')
    
    # Mark successes and failures
    success_trials = [t for t, s in zip(plot_data['trials'], plot_data['successes']) if s]
    failure_trials = [t for t, s in zip(plot_data['trials'], plot_data['successes']) if not s]
    
    if success_trials:
        plt.scatter(success_trials, [0] * len(success_trials), 
                   color='green', marker='^', s=30, alpha=0.5, label='Success')
    if failure_trials:
        plt.scatter(failure_trials, [0] * len(failure_trials), 
                   color='red', marker='v', s=30, alpha=0.5, label='Failure')
    
    plt.xlabel('Trial Number')
    plt.ylabel('Log Likelihood Ratio')
    plt.title('Sequential Probability Ratio Test Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('sequential_test_progress.png', dpi=100)
    print("\n✓ Plot saved as 'sequential_test_progress.png'")


def main():
    print("\n" + "=" * 60)
    print(" Sequential Testing Demo for LM Verification")
    print("=" * 60)
    
    # Run demonstrations
    tester = demo_basic_sprt()
    demo_adaptive_sprt()
    demo_group_sequential()
    demo_verification_session()
    demo_operating_characteristics()
    demo_simulation()
    
    # Plot the basic SPRT progress
    try:
        plot_sequential_test(tester)
    except ImportError:
        print("\n(Matplotlib not available for plotting)")
    
    print("\n" + "=" * 60)
    print(" Demo Complete!")
    print("=" * 60)
    print("\nKey Benefits of Sequential Testing:")
    print("  • Early stopping saves computation (avg 30-70% fewer trials)")
    print("  • Controlled error rates (Type I and II)")
    print("  • Adaptive to model performance")
    print("  • Statistical guarantees on decisions")


if __name__ == "__main__":
    main()