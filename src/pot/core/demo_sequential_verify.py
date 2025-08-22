#!/usr/bin/env python3
"""
Comprehensive demo of sequential verification with visualization.

Demonstrates:
1. Three test scenarios (H0, H1, borderline)
2. Trajectory plotting with confidence bounds
3. Anytime validity properties
4. Type I and Type II error rate validation
"""

import numpy as np
from typing import Iterator, List, Tuple
import time
from dataclasses import dataclass

# Try to import matplotlib, but make it optional
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except (ImportError, AttributeError) as e:
    # AttributeError can occur if there are issues with dependencies
    HAS_MATPLOTLIB = False

# Correct import from pot.core.sequential
from pot.core.sequential import sequential_verify, SPRTResult


@dataclass
class ScenarioResult:
    """Result from a test scenario."""
    name: str
    true_mean: float
    result: SPRTResult
    trajectory: List[Tuple[int, float, float, float]]


def generate_data_stream(mean: float, variance: float, n_max: int, seed: int = None) -> Iterator[float]:
    """
    Generate synthetic data stream with specified mean and variance.
    
    Args:
        mean: True mean of the distribution
        variance: Variance of the distribution
        n_max: Maximum number of samples
        seed: Random seed for reproducibility
    
    Yields:
        Values in [0,1] with specified statistics
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Use Beta distribution to ensure [0,1] bounds
    # Convert mean and variance to alpha, beta parameters
    if variance > 0:
        # Solve for alpha, beta given mean and variance constraints
        alpha = mean * (mean * (1 - mean) / variance - 1)
        beta = (1 - mean) * (mean * (1 - mean) / variance - 1)
        
        # Ensure valid parameters
        alpha = max(0.1, alpha)
        beta = max(0.1, beta)
    else:
        # Zero variance - return constant
        alpha = beta = 1000  # Large values give concentrated distribution
    
    for _ in range(n_max):
        yield np.clip(np.random.beta(alpha, beta), 0, 1)


def run_scenario(name: str, true_mean: float, tau: float = 0.5, 
                 alpha: float = 0.05, beta: float = 0.05,
                 max_samples: int = 500, seed: int = None) -> ScenarioResult:
    """
    Run a single test scenario.
    
    Args:
        name: Scenario name
        true_mean: True mean of the data stream
        tau: Decision threshold
        alpha: Type I error rate
        beta: Type II error rate
        max_samples: Maximum samples before forced decision
        seed: Random seed
    
    Returns:
        ScenarioResult with test outcome
    """
    print(f"\n{name}")
    print("-" * 50)
    print(f"True mean: {true_mean:.3f}, Threshold: {tau:.3f}")
    
    # Generate data stream
    stream = generate_data_stream(
        mean=true_mean,
        variance=0.01,  # Small variance for clearer demonstration
        n_max=max_samples,
        seed=seed
    )
    
    # Run sequential verification
    start_time = time.time()
    result = sequential_verify(
        stream=stream,
        tau=tau,
        alpha=alpha,
        beta=beta,
        max_samples=max_samples,
        compute_p_value=True
    )
    elapsed = time.time() - start_time
    
    # Print results
    print(f"Decision: {result.decision} at n={result.stopped_at}")
    print(f"Final mean: {result.final_mean:.4f}")
    print(f"Final variance: {result.final_variance:.6f}")
    print(f"Confidence interval: [{result.confidence_interval[0]:.4f}, "
          f"{result.confidence_interval[1]:.4f}]")
    print(f"P-value: {result.p_value:.6f}" if result.p_value else "P-value: N/A")
    print(f"Time: {elapsed:.3f}s")
    
    if result.forced_stop:
        print("⚠️  Decision was forced at max_samples")
    
    # Extract trajectory for plotting
    trajectory = []
    for i, state in enumerate(result.trajectory, 1):
        # Compute radius at this point
        from pot.core.boundaries import CSState, eb_radius
        cs_state = CSState()
        cs_state.n = state.n
        cs_state.mean = state.mean
        cs_state.M2 = state.M2
        radius = eb_radius(cs_state, alpha)
        trajectory.append((i, state.mean, radius, state.variance))
    
    return ScenarioResult(name, true_mean, result, trajectory)


def plot_trajectories(scenarios: List[ScenarioResult], tau: float = 0.5):
    """
    Plot trajectories for all scenarios showing running mean and confidence bounds.
    
    Args:
        scenarios: List of scenario results
        tau: Decision threshold
    """
    if not HAS_MATPLOTLIB:
        print("\n⚠️  Matplotlib not available - skipping plots")
        return None
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for ax, scenario in zip(axes, scenarios):
        if not scenario.trajectory:
            continue
            
        # Extract data
        ts = [t[0] for t in scenario.trajectory]
        means = [t[1] for t in scenario.trajectory]
        radii = [t[2] for t in scenario.trajectory]
        
        # Compute confidence bounds
        lower_bounds = [max(0, m - r) for m, r in zip(means, radii)]
        upper_bounds = [min(1, m + r) for m, r in zip(means, radii)]
        
        # Plot trajectory
        ax.plot(ts, means, 'b-', label='Running mean', linewidth=2)
        ax.fill_between(ts, lower_bounds, upper_bounds, 
                        alpha=0.3, color='blue', label='Confidence bounds')
        
        # Plot threshold
        ax.axhline(y=tau, color='r', linestyle='--', label=f'Threshold (τ={tau})')
        
        # Plot true mean
        ax.axhline(y=scenario.true_mean, color='g', linestyle=':', 
                  label=f'True mean ({scenario.true_mean:.2f})')
        
        # Mark stopping point
        if scenario.result.stopped_at < len(ts):
            ax.axvline(x=scenario.result.stopped_at, color='orange', 
                      linestyle='-', alpha=0.5, label='Stopping point')
        
        # Labels
        ax.set_xlabel('Sample number')
        ax.set_ylabel('Value')
        ax.set_title(f'{scenario.name}\nDecision: {scenario.result.decision}')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.suptitle('Sequential Verification Trajectories', y=1.02, fontsize=14)
    return fig


def demonstrate_anytime_validity(tau: float = 0.5, n_runs: int = 100):
    """
    Demonstrate that stopping at any point maintains error guarantees.
    
    Args:
        tau: Decision threshold
        n_runs: Number of simulation runs
    """
    print("\n" + "=" * 60)
    print("ANYTIME VALIDITY DEMONSTRATION")
    print("=" * 60)
    
    alpha = 0.05
    beta = 0.05
    max_samples = 200
    
    # Test at different stopping times
    stop_times = [10, 25, 50, 100, 200]
    
    for stop_time in stop_times:
        h0_errors = 0  # Type I errors (false rejection when true mean < tau)
        h1_errors = 0  # Type II errors (false acceptance when true mean > tau)
        h0_count = 0
        h1_count = 0
        
        for run in range(n_runs):
            # Test H0 case (true mean < tau)
            true_mean_h0 = 0.4
            stream_h0 = generate_data_stream(
                mean=true_mean_h0, 
                variance=0.02,
                n_max=stop_time,
                seed=1000 + run
            )
            
            result_h0 = sequential_verify(
                stream=stream_h0,
                tau=tau,
                alpha=alpha,
                beta=beta,
                max_samples=stop_time
            )
            
            if result_h0.decision == 'H1':  # False rejection
                h0_errors += 1
            if result_h0.decision in ['H0', 'H1']:  # Made a decision
                h0_count += 1
            
            # Test H1 case (true mean > tau)
            true_mean_h1 = 0.6
            stream_h1 = generate_data_stream(
                mean=true_mean_h1,
                variance=0.02,
                n_max=stop_time,
                seed=2000 + run
            )
            
            result_h1 = sequential_verify(
                stream=stream_h1,
                tau=tau,
                alpha=alpha,
                beta=beta,
                max_samples=stop_time
            )
            
            if result_h1.decision == 'H0':  # False acceptance
                h1_errors += 1
            if result_h1.decision in ['H0', 'H1']:  # Made a decision
                h1_count += 1
        
        # Calculate error rates (only for runs that made decisions)
        type_i_rate = h0_errors / max(1, h0_count) if h0_count > 0 else 0
        type_ii_rate = h1_errors / max(1, h1_count) if h1_count > 0 else 0
        
        print(f"\nStop time: {stop_time:3d} samples")
        print(f"  Decisions made: H0={h0_count}/{n_runs}, H1={h1_count}/{n_runs}")
        print(f"  Type I error rate:  {type_i_rate:.3f} (target ≤ {alpha:.3f})")
        print(f"  Type II error rate: {type_ii_rate:.3f} (target ≤ {beta:.3f})")
        
        # Check if error rates are within bounds (with some tolerance for simulation)
        tolerance = 2 * np.sqrt(alpha * (1 - alpha) / n_runs)  # 2 std errors
        if type_i_rate <= alpha + tolerance:
            print(f"  ✓ Type I error controlled")
        else:
            print(f"  ⚠️  Type I error exceeds target")
        
        if type_ii_rate <= beta + tolerance:
            print(f"  ✓ Type II error controlled")
        else:
            print(f"  ⚠️  Type II error exceeds target")
    
    print("\n" + "=" * 60)
    print("Key Observation: Error rates remain controlled regardless")
    print("of stopping time, demonstrating anytime validity.")
    print("=" * 60)


def verify_error_rates(tau: float = 0.5, n_simulations: int = 1000):
    """
    Verify Type I and Type II error rates over multiple runs.
    
    Args:
        tau: Decision threshold
        n_simulations: Number of simulation runs
    """
    print("\n" + "=" * 60)
    print("ERROR RATE VERIFICATION")
    print("=" * 60)
    
    alpha = 0.05
    beta = 0.05
    max_samples = 500
    
    # Track errors
    type_i_errors = 0
    type_ii_errors = 0
    
    # Track stopping times
    h0_stop_times = []
    h1_stop_times = []
    
    print(f"Running {n_simulations} simulations...")
    print(f"Target: Type I ≤ {alpha:.3f}, Type II ≤ {beta:.3f}")
    
    for i in range(n_simulations):
        if (i + 1) % 200 == 0:
            print(f"  Progress: {i+1}/{n_simulations}")
        
        # H0 scenario: true mean < tau
        true_mean_h0 = tau - 0.1  # 0.4 if tau=0.5
        stream_h0 = generate_data_stream(
            mean=true_mean_h0,
            variance=0.02,
            n_max=max_samples,
            seed=10000 + i
        )
        
        result_h0 = sequential_verify(
            stream=stream_h0,
            tau=tau,
            alpha=alpha,
            beta=beta,
            max_samples=max_samples
        )
        
        if result_h0.decision == 'H1':
            type_i_errors += 1
        h0_stop_times.append(result_h0.stopped_at)
        
        # H1 scenario: true mean > tau
        true_mean_h1 = tau + 0.1  # 0.6 if tau=0.5
        stream_h1 = generate_data_stream(
            mean=true_mean_h1,
            variance=0.02,
            n_max=max_samples,
            seed=20000 + i
        )
        
        result_h1 = sequential_verify(
            stream=stream_h1,
            tau=tau,
            alpha=alpha,
            beta=beta,
            max_samples=max_samples
        )
        
        if result_h1.decision == 'H0':
            type_ii_errors += 1
        h1_stop_times.append(result_h1.stopped_at)
    
    # Calculate observed error rates
    observed_type_i = type_i_errors / n_simulations
    observed_type_ii = type_ii_errors / n_simulations
    
    # Calculate confidence intervals for error rates
    se_type_i = np.sqrt(observed_type_i * (1 - observed_type_i) / n_simulations)
    se_type_ii = np.sqrt(observed_type_ii * (1 - observed_type_ii) / n_simulations)
    
    print("\n" + "=" * 60)
    print("RESULTS:")
    print(f"Type I error rate:  {observed_type_i:.4f} ± {1.96*se_type_i:.4f}")
    print(f"  Target: ≤ {alpha:.3f}")
    print(f"  Status: {'✓ PASS' if observed_type_i <= alpha else '✗ FAIL'}")
    
    print(f"\nType II error rate: {observed_type_ii:.4f} ± {1.96*se_type_ii:.4f}")
    print(f"  Target: ≤ {beta:.3f}")
    print(f"  Status: {'✓ PASS' if observed_type_ii <= beta else '✗ FAIL'}")
    
    print(f"\nMean stopping times:")
    print(f"  H0 scenarios: {np.mean(h0_stop_times):.1f} ± {np.std(h0_stop_times):.1f}")
    print(f"  H1 scenarios: {np.mean(h1_stop_times):.1f} ± {np.std(h1_stop_times):.1f}")
    
    print("=" * 60)


def main():
    """Main demo function."""
    print("=" * 60)
    print("SEQUENTIAL VERIFICATION DEMO")
    print("=" * 60)
    
    # Parameters
    tau = 0.5
    alpha = 0.05
    beta = 0.05
    max_samples = 500
    
    print(f"\nParameters:")
    print(f"  Threshold (τ): {tau}")
    print(f"  Type I error (α): {alpha}")
    print(f"  Type II error (β): {beta}")
    print(f"  Max samples: {max_samples}")
    
    # Run three test scenarios
    scenarios = []
    
    # Scenario 1: H0 case (mean < tau)
    scenarios.append(run_scenario(
        name="Scenario 1: H0 Case (mean=0.4 < τ=0.5)",
        true_mean=0.4,
        tau=tau,
        alpha=alpha,
        beta=beta,
        max_samples=max_samples,
        seed=42
    ))
    
    # Scenario 2: H1 case (mean > tau)
    scenarios.append(run_scenario(
        name="Scenario 2: H1 Case (mean=0.6 > τ=0.5)",
        true_mean=0.6,
        tau=tau,
        alpha=alpha,
        beta=beta,
        max_samples=max_samples,
        seed=43
    ))
    
    # Scenario 3: Borderline case (mean ≈ tau)
    scenarios.append(run_scenario(
        name="Scenario 3: Borderline (mean=0.5 ≈ τ=0.5)",
        true_mean=0.5,
        tau=tau,
        alpha=alpha,
        beta=beta,
        max_samples=max_samples,
        seed=44
    ))
    
    # Plot trajectories
    if HAS_MATPLOTLIB:
        try:
            fig = plot_trajectories(scenarios, tau)
            if fig:
                plt.savefig('sequential_verify_trajectories.png', dpi=150, bbox_inches='tight')
                print("\n✓ Trajectory plot saved to 'sequential_verify_trajectories.png'")
                # plt.show()  # Comment out for non-interactive environments
        except Exception as e:
            print(f"\n⚠️  Could not create plots: {e}")
    else:
        print("\n⚠️  Matplotlib not available - skipping plots")
    
    # Demonstrate anytime validity
    demonstrate_anytime_validity(tau=tau, n_runs=100)
    
    # Verify error rates
    verify_error_rates(tau=tau, n_simulations=1000)
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nKey Takeaways:")
    print("1. Sequential testing provides early stopping for clear cases")
    print("2. Confidence bounds shrink over time as √(log(log(t))/t)")
    print("3. Error rates remain controlled at ANY stopping time")
    print("4. Borderline cases may require more samples or hit max_samples")
    print("5. P-values remain valid despite optional stopping")


if __name__ == "__main__":
    main()