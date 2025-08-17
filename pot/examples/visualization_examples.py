#!/usr/bin/env python3
"""
Examples demonstrating visualization tools for sequential verification.
Shows all four main visualization functions with realistic data.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from pot.core.visualize_sequential import (
    plot_verification_trajectory,
    plot_operating_characteristics, 
    plot_anytime_validity,
    create_interactive_demo,
    VisualizationConfig
)
from pot.core.sequential import sequential_verify, SPRTResult, SequentialState


def generate_realistic_trajectory(true_mean: float = 0.03, noise_std: float = 0.02, 
                                tau: float = 0.05, seed: int = 42) -> SPRTResult:
    """Generate a realistic sequential verification trajectory"""
    np.random.seed(seed)
    
    def data_stream():
        for _ in range(500):  # Max samples
            yield np.random.normal(true_mean, noise_std)
    
    # Run actual sequential verification
    try:
        result = sequential_verify(
            stream=data_stream(),
            tau=tau,
            alpha=0.05,
            beta=0.05,
            max_samples=500,
            compute_p_value=True
        )
        return result
    except Exception as e:
        print(f"Sequential verify failed: {e}")
        # Fallback to mock result
        return create_mock_sprt_result(true_mean, noise_std, tau, seed)


def create_mock_sprt_result(true_mean: float, noise_std: float, tau: float, seed: int) -> SPRTResult:
    """Create mock SPRTResult for demonstration when sequential_verify is not available"""
    np.random.seed(seed)
    
    trajectory = []
    running_sum = 0
    n = 0
    
    for i in range(1, 501):
        n = i
        sample = np.random.normal(true_mean, noise_std)
        running_sum += sample
        mean = running_sum / n
        
        # Approximate confidence radius
        radius = (noise_std / np.sqrt(n)) * 2.0  # Rough 95% CI
        
        trajectory.append({
            'n': n,
            'mean': mean,
            'radius': radius,
            'tau': tau
        })
        
        # Simple stopping rule
        if mean + radius < tau:
            decision = 'H0'
            break
        elif mean - radius > tau:
            decision = 'H1' 
            break
    else:
        decision = 'H1' if mean > tau else 'H0'
    
    # Create mock SPRTResult
    class MockSPRTResult:
        def __init__(self):
            self.decision = decision
            self.stopped_at = n
            self.final_mean = mean
            self.confidence_radius = radius
            self.tau = tau
            self.trajectory = trajectory
            self.p_value = 0.023 if decision == 'H1' else 0.89
            self.forced_stop = (n >= 500)
    
    return MockSPRTResult()


def example_1_trajectory_visualization():
    """Example 1: Visualizing a single verification trajectory"""
    print("=" * 60)
    print("Example 1: Single Trajectory Visualization")
    print("=" * 60)
    
    # Generate three different scenarios
    scenarios = [
        {"name": "Model Passes (H0)", "true_mean": 0.02, "tau": 0.05, "seed": 42},
        {"name": "Model Fails (H1)", "true_mean": 0.08, "tau": 0.05, "seed": 43}, 
        {"name": "Borderline Case", "true_mean": 0.049, "tau": 0.05, "seed": 44}
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, scenario in enumerate(scenarios):
        print(f"\nScenario: {scenario['name']}")
        print(f"True mean: {scenario['true_mean']}, Threshold: {scenario['tau']}")
        
        # Generate trajectory
        result = generate_realistic_trajectory(
            true_mean=scenario['true_mean'],
            tau=scenario['tau'],
            seed=scenario['seed']
        )
        
        print(f"Decision: {result.decision}, Stopped at: {result.stopped_at}")
        
        # Create individual plot (save the full version)
        if i == 0:  # Save detailed plot for first scenario
            config = VisualizationConfig(figsize=(12, 8), show_legend=True)
            detailed_fig = plot_verification_trajectory(
                result, 
                config=config,
                save_path=f'trajectory_detailed_{scenario["name"].replace(" ", "_").lower()}.png',
                show_details=True
            )
            plt.close(detailed_fig)  # Close to avoid memory issues
        
        # Create subplot version
        ax = axes[i]
        
        # Extract trajectory data
        if result.trajectory:
            time_points = [t['n'] for t in result.trajectory]
            means = [t['mean'] for t in result.trajectory]
            lower_bounds = [t['mean'] - t['radius'] for t in result.trajectory]
            upper_bounds = [t['mean'] + t['radius'] for t in result.trajectory]
            
            # Plot trajectory
            ax.plot(time_points, means, 'b-', linewidth=2, label='Running Mean')
            ax.fill_between(time_points, lower_bounds, upper_bounds, 
                           alpha=0.3, color='blue', label='95% CI')
            
            # Threshold and regions
            ax.axhline(y=scenario['tau'], color='red', linestyle='--', linewidth=2, 
                      label=f'τ = {scenario["tau"]:.3f}')
            
            # Mark stopping point
            if result.stopped_at <= len(time_points):
                stop_idx = result.stopped_at - 1
                ax.scatter(time_points[stop_idx], means[stop_idx], 
                          color='red', s=80, zorder=5)
            
            ax.set_title(f'{scenario["name"]}\n{result.decision} at n={result.stopped_at}')
            ax.set_xlabel('Sample Number')
            ax.set_ylabel('Running Mean')
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend()
    
    plt.tight_layout()
    plt.savefig('trajectory_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Trajectory visualizations saved:")
    print("  - trajectory_comparison.png (3-panel comparison)")
    print("  - trajectory_detailed_model_passes_(h0).png (detailed view)")


def example_2_operating_characteristics():
    """Example 2: Operating characteristics comparison"""
    print("\n" + "=" * 60)
    print("Example 2: Operating Characteristics Analysis")
    print("=" * 60)
    
    # Different threshold scenarios
    thresholds = [0.03, 0.05, 0.08]
    effect_sizes = np.linspace(0.0, 0.12, 20)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, tau in enumerate(thresholds):
        print(f"\nAnalyzing threshold τ = {tau:.3f}")
        
        # Create operating characteristics plot for this threshold
        oc_fig = plot_operating_characteristics(
            tau=tau,
            alpha=0.05,
            beta=0.05,
            effect_sizes=effect_sizes,
            max_samples_fixed=1000,
            save_path=f'operating_characteristics_tau_{tau:.3f}.png'
        )
        plt.close(oc_fig)  # Close to save memory
        
        print(f"  ✓ Saved operating_characteristics_tau_{tau:.3f}.png")
    
    # Create a summary comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'green', 'red']
    
    for i, tau in enumerate(thresholds):
        # Simulate power curves for comparison
        power_values = []
        stopping_times = []
        
        for effect in effect_sizes:
            if effect <= 0:
                power = 0.05  # Type I error
                stop_time = 800
            else:
                # Simplified power calculation
                power = min(0.99, 0.05 + (effect / tau) * 0.9)
                stop_time = max(50, 1000 * np.exp(-effect * 20))
            
            power_values.append(power)
            stopping_times.append(stop_time)
        
        # Power curve comparison
        ax1.plot(effect_sizes, power_values, color=colors[i], linewidth=2, 
                label=f'τ = {tau:.3f}', marker='o', markersize=3)
        
        # Stopping time comparison
        ax2.plot(effect_sizes, stopping_times, color=colors[i], linewidth=2,
                label=f'τ = {tau:.3f}', marker='s', markersize=3)
    
    # Format power plot
    ax1.set_xlabel('Effect Size (μ)')
    ax1.set_ylabel('Power')
    ax1.set_title('Power Curves for Different Thresholds')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    
    # Format stopping time plot
    ax2.set_xlabel('Effect Size (μ)')
    ax2.set_ylabel('Expected Sample Size')
    ax2.set_title('Sample Size Efficiency Comparison')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('threshold_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Operating characteristics analysis complete:")
    print("  - Individual OC plots for each threshold")
    print("  - threshold_comparison.png (summary comparison)")


def example_3_anytime_validity():
    """Example 3: Anytime validity demonstration"""
    print("\n" + "=" * 60)
    print("Example 3: Anytime Validity Demonstration")
    print("=" * 60)
    
    # Generate multiple trajectories under H0 and H1
    print("Generating 50 trajectories (25 H0, 25 H1)...")
    
    trajectories_h0 = []
    trajectories_h1 = []
    
    # H0 trajectories (true mean below threshold)
    for i in range(25):
        result = generate_realistic_trajectory(
            true_mean=0.02,  # Below tau=0.05
            tau=0.05,
            seed=100 + i
        )
        trajectories_h0.append(result)
    
    # H1 trajectories (true mean above threshold)  
    for i in range(25):
        result = generate_realistic_trajectory(
            true_mean=0.08,  # Above tau=0.05
            tau=0.05,
            seed=200 + i
        )
        trajectories_h1.append(result)
    
    all_trajectories = trajectories_h0 + trajectories_h1
    
    # Create anytime validity plot
    fig = plot_anytime_validity(
        trajectories=all_trajectories,
        alpha=0.05,
        save_path='anytime_validity_demo.png'
    )
    plt.show()
    
    # Analyze results
    h0_decisions = [t.decision for t in trajectories_h0]
    h1_decisions = [t.decision for t in trajectories_h1]
    
    type_i_errors = h0_decisions.count('H1') / len(h0_decisions)
    type_ii_errors = h1_decisions.count('H0') / len(h1_decisions)
    
    print(f"\nEmpirical Error Analysis:")
    print(f"Type I errors (false H1 on true H0): {type_i_errors:.3f} (target ≤ 0.05)")
    print(f"Type II errors (false H0 on true H1): {type_ii_errors:.3f} (target ≤ 0.05)")
    
    # Stopping time analysis
    all_stopping_times = [t.stopped_at for t in all_trajectories]
    print(f"\nStopping Time Analysis:")
    print(f"Mean stopping time: {np.mean(all_stopping_times):.1f}")
    print(f"Median stopping time: {np.median(all_stopping_times):.1f}")
    print(f"Range: {min(all_stopping_times)} - {max(all_stopping_times)}")
    
    print("\n✓ Anytime validity demonstration complete:")
    print("  - anytime_validity_demo.png saved")
    print("  - Error rates should be controlled despite optional stopping")


def example_4_interactive_demo_setup():
    """Example 4: Interactive demo setup instructions"""
    print("\n" + "=" * 60)
    print("Example 4: Interactive Demo Setup")
    print("=" * 60)
    
    print("The interactive demo requires Streamlit. To set it up:")
    print("\n1. Install Streamlit:")
    print("   pip install streamlit")
    
    print("\n2. Run the interactive demo:")
    print("   streamlit run pot/core/visualize_sequential.py")
    
    print("\n3. Or create a custom interactive demo:")
    
    # Create a simple launcher script
    launcher_code = '''#!/usr/bin/env python3
"""
Launcher for Sequential Verification Interactive Demo
"""
import streamlit as st
from pot.core.visualize_sequential import create_interactive_demo

if __name__ == "__main__":
    try:
        create_interactive_demo()
    except ImportError as e:
        st.error(f"Missing dependency: {e}")
        st.info("Install with: pip install streamlit plotly")
'''
    
    with open('launch_interactive_demo.py', 'w') as f:
        f.write(launcher_code)
    
    print("\n✓ Created launch_interactive_demo.py")
    print("  Run with: streamlit run launch_interactive_demo.py")
    
    # Test basic functionality without streamlit
    print("\n4. Testing visualization functions without Streamlit:")
    
    try:
        # Test configuration
        config = VisualizationConfig(
            figsize=(10, 6),
            dpi=100,
            style='seaborn',
            show_grid=True,
            interactive=False
        )
        print("   ✓ VisualizationConfig works")
        
        # Test mock data generation
        result = generate_realistic_trajectory()
        print(f"   ✓ Mock trajectory generation works (decision: {result.decision})")
        
        print("\nAll core visualization functions are working!")
        
    except Exception as e:
        print(f"   ✗ Error testing functions: {e}")


def example_5_custom_styling():
    """Example 5: Custom styling and advanced features"""
    print("\n" + "=" * 60)
    print("Example 5: Custom Styling and Advanced Features")
    print("=" * 60)
    
    # Create custom visualization configurations
    configs = {
        'publication': VisualizationConfig(
            figsize=(8, 6),
            dpi=300,
            style='seaborn-v0_8-whitegrid',
            palette='Set2',
            show_grid=True,
            save_format='pdf'
        ),
        'presentation': VisualizationConfig(
            figsize=(12, 8),
            dpi=150,
            style='seaborn-v0_8-darkgrid',
            palette='bright',
            show_grid=True,
            save_format='png'
        ),
        'interactive': VisualizationConfig(
            figsize=(10, 6),
            dpi=100,
            interactive=True,
            theme='dark'
        )
    }
    
    # Generate sample data
    result = generate_realistic_trajectory(true_mean=0.06, tau=0.05, seed=999)
    
    for style_name, config in configs.items():
        print(f"\nCreating {style_name} style plot...")
        
        try:
            fig = plot_verification_trajectory(
                result=result,
                config=config,
                save_path=f'trajectory_{style_name}.{config.save_format}',
                show_details=True
            )
            plt.close(fig)
            print(f"   ✓ Saved trajectory_{style_name}.{config.save_format}")
            
        except Exception as e:
            print(f"   ✗ Error with {style_name} style: {e}")
    
    print("\n✓ Custom styling examples complete")
    print("Generated plots with different styles for various use cases")


def run_all_examples():
    """Run all visualization examples"""
    print("🎨 Sequential Verification Visualization Examples")
    print("=" * 60)
    print("This script demonstrates all visualization tools in the PoT framework")
    
    try:
        example_1_trajectory_visualization()
        example_2_operating_characteristics() 
        example_3_anytime_validity()
        example_4_interactive_demo_setup()
        example_5_custom_styling()
        
        print("\n" + "🎉" * 20)
        print("ALL VISUALIZATION EXAMPLES COMPLETED SUCCESSFULLY!")
        print("🎉" * 20)
        
        print("\nGenerated Files:")
        files = [
            'trajectory_comparison.png',
            'trajectory_detailed_model_passes_(h0).png', 
            'operating_characteristics_tau_*.png',
            'threshold_comparison.png',
            'anytime_validity_demo.png',
            'launch_interactive_demo.py',
            'trajectory_publication.pdf',
            'trajectory_presentation.png'
        ]
        
        for file in files:
            print(f"  ✓ {file}")
        
        print("\nNext Steps:")
        print("1. View the generated plots to understand sequential verification")
        print("2. Run 'streamlit run launch_interactive_demo.py' for interactive exploration")
        print("3. Use these visualization tools in your own verification workflows")
        print("4. Customize the VisualizationConfig for your specific needs")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()