#!/usr/bin/env python3
"""
Visualization Tools for Sequential Verification Processes
Implements comprehensive plotting and interactive demos for PoT sequential testing.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

# Handle optional dependencies gracefully
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from .sequential import SPRTResult, sequential_verify, power_analysis
from .boundaries import CSState, eb_radius


# Configure plotting style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


@dataclass
class VisualizationConfig:
    """Configuration for visualization appearance and behavior."""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    style: str = 'seaborn'
    palette: str = 'husl'
    show_grid: bool = True
    show_legend: bool = True
    save_format: str = 'png'
    interactive: bool = False
    theme: str = 'light'  # 'light' or 'dark'


def plot_verification_trajectory(
    result: SPRTResult,
    config: Optional[VisualizationConfig] = None,
    save_path: Optional[str] = None,
    show_details: bool = True
) -> plt.Figure:
    """
    Plot sequential verification trajectory with confidence bounds.
    
    Args:
        result: SPRTResult from sequential_verify()
        config: Visualization configuration
        save_path: Optional path to save the plot
        show_details: Whether to show detailed annotations
        
    Returns:
        matplotlib Figure object
    """
    if config is None:
        config = VisualizationConfig()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=config.figsize, dpi=config.dpi)
    
    # Extract trajectory data
    if not result.trajectory:
        raise ValueError("SPRTResult must contain trajectory data for visualization")
    
    # Handle both dict and SequentialState trajectory formats
    if isinstance(result.trajectory[0], dict):
        # Dictionary format (legacy or mock)
        time_points = [t['n'] for t in result.trajectory]
        means = [t['mean'] for t in result.trajectory]
        lower_bounds = [t['mean'] - t['radius'] for t in result.trajectory]
        upper_bounds = [t['mean'] + t['radius'] for t in result.trajectory]
    else:
        # SequentialState format (new)
        time_points = [t.n for t in result.trajectory]
        means = [t.mean for t in result.trajectory]
        
        # Compute confidence bounds using EB radius
        from .boundaries import CSState, eb_radius
        lower_bounds = []
        upper_bounds = []
        
        for state in result.trajectory:
            # Convert to CSState for EB radius computation
            cs_state = CSState()
            cs_state.n = state.n
            cs_state.mean = state.mean
            cs_state.M2 = state.M2
            
            # Compute radius (use alpha=0.05 as default)
            alpha = 0.05
            try:
                radius = eb_radius(cs_state, alpha)
            except:
                # Fallback to simple radius if EB fails
                radius = (state.variance ** 0.5) / (state.n ** 0.5) * 1.96 if state.n > 0 else 0.1
            
            lower_bounds.append(state.mean - radius)
            upper_bounds.append(state.mean + radius)
    
    # Get tau from the first trajectory point or result
    tau = getattr(result, 'tau', None)
    if tau is None and result.trajectory:
        # Try to infer tau from trajectory metadata
        tau = result.trajectory[0].get('tau', 0.05)
    
    # Main trajectory plot
    ax1.plot(time_points, means, 'b-', linewidth=2, label='Running Mean', alpha=0.8)
    ax1.fill_between(time_points, lower_bounds, upper_bounds, 
                     alpha=0.3, color='blue', label='Confidence Bounds')
    
    # Tau threshold line
    ax1.axhline(y=tau, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold τ = {tau:.3f}')
    
    # Decision regions
    y_max = max(max(upper_bounds), tau * 1.2)
    y_min = min(min(lower_bounds), 0)
    
    # H0 acceptance region (below tau)
    ax1.fill_between([0, max(time_points)], [y_min, y_min], [tau, tau], 
                     alpha=0.1, color='green', label='H₀ Region (Accept)')
    
    # H1 rejection region (above tau)  
    ax1.fill_between([0, max(time_points)], [tau, tau], [y_max, y_max], 
                     alpha=0.1, color='red', label='H₁ Region (Reject)')
    
    # Mark stopping point
    if result.stopped_at and result.stopped_at <= len(time_points):
        stop_idx = result.stopped_at - 1
        ax1.scatter(time_points[stop_idx], means[stop_idx], 
                   color='red', s=100, zorder=5, 
                   label=f'Stop at n={result.stopped_at}')
    
    # Formatting
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Running Mean Distance')
    ax1.set_title(f'Sequential Verification Trajectory\nDecision: {result.decision}, '
                 f'Stopped at n={result.stopped_at}')
    ax1.grid(config.show_grid, alpha=0.3)
    if config.show_legend:
        ax1.legend(loc='best')
    
    # Confidence bound width over time
    widths = [u - l for u, l in zip(upper_bounds, lower_bounds)]
    ax2.plot(time_points, widths, 'g-', linewidth=2, label='Confidence Width')
    ax2.set_xlabel('Sample Number')
    ax2.set_ylabel('Confidence Bound Width')
    ax2.set_title('Confidence Bound Evolution')
    ax2.grid(config.show_grid, alpha=0.3)
    if config.show_legend:
        ax2.legend()
    
    # Add detailed annotations if requested
    if show_details:
        # Add decision annotation
        decision_color = 'green' if result.decision == 'H0' else 'red'
        ax1.text(0.02, 0.98, f'Decision: {result.decision}', 
                transform=ax1.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=decision_color, alpha=0.3),
                verticalalignment='top')
        
        # Add efficiency annotation
        if result.forced_stop:
            efficiency_text = f'Forced stop at max samples'
        else:
            max_possible = len(time_points) if time_points else result.stopped_at
            efficiency = 100 * (1 - result.stopped_at / max_possible) if max_possible > 0 else 0
            efficiency_text = f'Early stop: {efficiency:.1f}% fewer samples'
        
        ax1.text(0.02, 0.88, efficiency_text,
                transform=ax1.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                verticalalignment='top')
        
        # Add p-value if available
        if result.p_value is not None:
            ax1.text(0.02, 0.78, f'P-value: {result.p_value:.4f}',
                    transform=ax1.transAxes, fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5),
                    verticalalignment='top')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format=config.save_format, dpi=config.dpi, 
                   bbox_inches='tight')
    
    return fig


def plot_operating_characteristics(
    tau: float = 0.05,
    alpha: float = 0.05,
    beta: float = 0.05,
    effect_sizes: Optional[List[float]] = None,
    max_samples_fixed: int = 1000,
    config: Optional[VisualizationConfig] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot operating characteristics comparing sequential vs fixed-sample testing.
    
    Args:
        tau: Decision threshold
        alpha: Type I error rate
        beta: Type II error rate  
        effect_sizes: List of effect sizes to analyze
        max_samples_fixed: Sample size for fixed-sample comparison
        config: Visualization configuration
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    if config is None:
        config = VisualizationConfig()
    
    if effect_sizes is None:
        effect_sizes = np.linspace(0.0, 0.15, 20)
    
    # Compute power analysis
    try:
        power_result = power_analysis(
            tau=tau,
            alpha=alpha,
            beta=beta,
            effect_sizes=effect_sizes,
            n_simulations=500,  # Reduced for faster plotting
            max_samples=max_samples_fixed
        )
    except Exception as e:
        # Fallback to simulated data if power_analysis fails
        warnings.warn(f"Power analysis failed: {e}. Using simulated data.")
        power_result = _simulate_power_analysis(effect_sizes, tau, alpha, beta, max_samples_fixed)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=config.dpi)
    
    # Extract data from power analysis
    if hasattr(power_result, 'power_curve'):
        # Real power analysis result
        effect_vals = [e for e, _ in power_result.power_curve]
        power_vals = [p for _, p in power_result.power_curve]
        stopping_times = [power_result.expected_stopping_times.get(e, max_samples_fixed) 
                         for e in effect_vals]
    else:
        # Simulated result
        effect_vals = power_result['effect_sizes']
        power_vals = power_result['power_values']
        stopping_times = power_result['stopping_times']
    
    # 1. Power Curves
    ax1.plot(effect_vals, power_vals, 'b-', linewidth=2, marker='o', 
            label='Sequential Test', markersize=4)
    
    # Fixed-sample power (theoretical)
    fixed_power = [_compute_fixed_power(e, tau, alpha, max_samples_fixed) for e in effect_vals]
    ax1.plot(effect_vals, fixed_power, 'r--', linewidth=2, marker='s', 
            label=f'Fixed Sample (n={max_samples_fixed})', markersize=4)
    
    ax1.axhline(y=1-beta, color='gray', linestyle=':', alpha=0.7, 
               label=f'Target Power (1-β = {1-beta:.2f})')
    ax1.set_xlabel('Effect Size (μ)')
    ax1.set_ylabel('Power')
    ax1.set_title('Power Curves Comparison')
    ax1.grid(config.show_grid, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    
    # 2. Expected Stopping Times
    ax2.plot(effect_vals, stopping_times, 'g-', linewidth=2, marker='o', 
            label='Sequential Test', markersize=4)
    ax2.axhline(y=max_samples_fixed, color='red', linestyle='--', linewidth=2,
               label=f'Fixed Sample (n={max_samples_fixed})')
    
    ax2.set_xlabel('Effect Size (μ)')
    ax2.set_ylabel('Expected Sample Size')
    ax2.set_title('Sample Size Efficiency')
    ax2.grid(config.show_grid, alpha=0.3)
    ax2.legend()
    
    # 3. Efficiency Ratio
    efficiency = [100 * (1 - st / max_samples_fixed) for st in stopping_times]
    ax3.bar(range(len(effect_vals)), efficiency, alpha=0.7, color='skyblue')
    ax3.set_xlabel('Effect Size Index')
    ax3.set_ylabel('Sample Size Reduction (%)')
    ax3.set_title('Sequential Testing Efficiency')
    ax3.grid(config.show_grid, alpha=0.3, axis='y')
    
    # Add effect size labels on x-axis
    ax3.set_xticks(range(0, len(effect_vals), max(1, len(effect_vals)//10)))
    ax3.set_xticklabels([f'{effect_vals[i]:.3f}' 
                        for i in range(0, len(effect_vals), max(1, len(effect_vals)//10))],
                       rotation=45)
    
    # 4. Operating Characteristic Curves
    if hasattr(power_result, 'oc_curves') and power_result.oc_curves:
        # Plot Type I error curve if available
        if 'type_i' in power_result.oc_curves:
            type_i_curve = power_result.oc_curves['type_i']
            ax4.plot([x for x, _ in type_i_curve], [y for _, y in type_i_curve], 
                    'r-', label='Type I Error', linewidth=2)
        
        # Plot Type II error curve if available  
        if 'type_ii' in power_result.oc_curves:
            type_ii_curve = power_result.oc_curves['type_ii']
            ax4.plot([x for x, _ in type_ii_curve], [y for _, y in type_ii_curve], 
                    'b-', label='Type II Error', linewidth=2)
        
        ax4.axhline(y=alpha, color='red', linestyle=':', alpha=0.7, label=f'Target α = {alpha}')
        ax4.axhline(y=beta, color='blue', linestyle=':', alpha=0.7, label=f'Target β = {beta}')
        
        ax4.set_xlabel('Effect Size')
        ax4.set_ylabel('Error Rate')
        ax4.set_title('Operating Characteristic Curves')
        ax4.grid(config.show_grid, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(0, max(alpha, beta) * 2)
    else:
        # Simplified OC plot
        ax4.plot(effect_vals, [alpha] * len(effect_vals), 'r-', 
                label=f'Type I Error ≈ {alpha}', linewidth=2)
        type_ii_errors = [max(0.01, beta * (1 - p)) for p in power_vals]
        ax4.plot(effect_vals, type_ii_errors, 'b-', 
                label='Type II Error (approx)', linewidth=2)
        
        ax4.set_xlabel('Effect Size')
        ax4.set_ylabel('Error Rate')
        ax4.set_title('Error Rates vs Effect Size')
        ax4.grid(config.show_grid, alpha=0.3)
        ax4.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format=config.save_format, dpi=config.dpi, 
                   bbox_inches='tight')
    
    return fig


def plot_anytime_validity(
    trajectories: List[SPRTResult],
    alpha: float = 0.05,
    config: Optional[VisualizationConfig] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Demonstrate anytime validity across multiple trajectories.
    
    Args:
        trajectories: List of SPRTResult objects from multiple runs
        alpha: Significance level for error rate analysis
        config: Visualization configuration
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    if config is None:
        config = VisualizationConfig()
    
    if not trajectories:
        raise ValueError("At least one trajectory is required")
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12), dpi=config.dpi)
    
    # 1. Multiple trajectory visualization
    max_length = max(len(t.trajectory) if t.trajectory else t.stopped_at for t in trajectories)
    
    for i, result in enumerate(trajectories[:20]):  # Limit to 20 for readability
        if not result.trajectory:
            continue
        
        # Handle both dict and SequentialState trajectory formats
        if isinstance(result.trajectory[0], dict):
            time_points = [t['n'] for t in result.trajectory]
            means = [t['mean'] for t in result.trajectory]
        else:
            time_points = [t.n for t in result.trajectory]
            means = [t.mean for t in result.trajectory]
        
        alpha_line = 0.3 if len(trajectories) > 10 else 0.6
        color = f'C{i % 10}'
        ax1.plot(time_points, means, color=color, alpha=alpha_line, linewidth=1)
    
    # Add tau line if available
    if trajectories and hasattr(trajectories[0], 'tau'):
        tau = trajectories[0].tau
    else:
        tau = 0.05  # Default
    
    ax1.axhline(y=tau, color='red', linestyle='--', linewidth=2, 
               label=f'Threshold τ = {tau:.3f}')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Running Mean')
    ax1.set_title(f'Multiple Sequential Trajectories (n={len(trajectories)})')
    ax1.grid(config.show_grid, alpha=0.3)
    ax1.legend()
    
    # 2. Stopping time distribution
    stopping_times = [t.stopped_at for t in trajectories if t.stopped_at]
    if stopping_times:
        ax2.hist(stopping_times, bins=min(30, len(set(stopping_times))), 
                alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=np.mean(stopping_times), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean = {np.mean(stopping_times):.1f}')
        ax2.axvline(x=np.median(stopping_times), color='green', linestyle='--', 
                   linewidth=2, label=f'Median = {np.median(stopping_times):.1f}')
        ax2.set_xlabel('Stopping Time')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Stopping Times')
        ax2.grid(config.show_grid, alpha=0.3)
        ax2.legend()
    
    # 3. Empirical error rates at different stopping times
    decisions = [t.decision for t in trajectories]
    h0_decisions = sum(1 for d in decisions if d == 'H0')
    h1_decisions = sum(1 for d in decisions if d == 'H1')
    
    # Compute empirical error rates (assuming ground truth)
    # For demonstration, assume first half are true H0, second half are true H1
    n_total = len(trajectories)
    n_h0_true = n_total // 2
    n_h1_true = n_total - n_h0_true
    
    type_i_errors = []
    type_ii_errors = []
    
    # Analyze error rates at different stopping times
    unique_stops = sorted(set(stopping_times)) if stopping_times else [10, 20, 50, 100]
    
    for stop_time in unique_stops[:20]:  # Limit for readability
        # Count errors for trajectories stopped at or before this time
        relevant_trajectories = [t for t in trajectories if t.stopped_at <= stop_time]
        if not relevant_trajectories:
            continue
            
        # Simulate Type I errors (false rejections of true H0)
        h0_trajectories = relevant_trajectories[:len(relevant_trajectories)//2]
        type_i = sum(1 for t in h0_trajectories if t.decision == 'H1') / max(1, len(h0_trajectories))
        type_i_errors.append(type_i)
        
        # Simulate Type II errors (false acceptances of true H1)
        h1_trajectories = relevant_trajectories[len(relevant_trajectories)//2:]
        type_ii = sum(1 for t in h1_trajectories if t.decision == 'H0') / max(1, len(h1_trajectories))
        type_ii_errors.append(type_ii)
    
    if type_i_errors and type_ii_errors:
        ax3.plot(unique_stops[:len(type_i_errors)], type_i_errors, 'r-o', 
                label='Type I Error Rate', linewidth=2, markersize=4)
        ax3.plot(unique_stops[:len(type_ii_errors)], type_ii_errors, 'b-s', 
                label='Type II Error Rate', linewidth=2, markersize=4)
        ax3.axhline(y=alpha, color='red', linestyle=':', alpha=0.7, 
                   label=f'Target α = {alpha}')
        ax3.axhline(y=alpha, color='blue', linestyle=':', alpha=0.7, 
                   label=f'Target β = {alpha}')  # Assuming β = α
        
        ax3.set_xlabel('Maximum Stopping Time')
        ax3.set_ylabel('Empirical Error Rate')
        ax3.set_title('Error Rate Control Across Stopping Times')
        ax3.grid(config.show_grid, alpha=0.3)
        ax3.legend()
        ax3.set_ylim(0, max(alpha * 3, 0.1))
    
    # 4. Anytime validity demonstration
    # Show how confidence bounds maintain coverage regardless of when we stop
    coverage_rates = []
    stop_times_analyzed = []
    
    # Analyze coverage at different stopping points
    for stop_time in range(5, max_length + 1, max(1, max_length // 20)):
        # Count how many trajectories have coverage at this stopping time
        covered = 0
        total = 0
        
        for result in trajectories:
            if not result.trajectory or len(result.trajectory) < stop_time:
                continue
                
            traj_point = result.trajectory[stop_time - 1]
            
            # Handle both dict and SequentialState formats
            if isinstance(traj_point, dict):
                mean = traj_point['mean']
                radius = traj_point['radius']
            else:
                mean = traj_point.mean
                # Compute radius for SequentialState
                from .boundaries import CSState, eb_radius
                cs_state = CSState()
                cs_state.n = traj_point.n
                cs_state.mean = traj_point.mean
                cs_state.M2 = traj_point.M2
                
                try:
                    radius = eb_radius(cs_state, alpha)
                except:
                    radius = (traj_point.variance ** 0.5) / (traj_point.n ** 0.5) * 1.96 if traj_point.n > 0 else 0.1
            
            # Assume true parameter is tau for coverage calculation
            true_param = tau
            if mean - radius <= true_param <= mean + radius:
                covered += 1
            total += 1
        
        if total > 0:
            coverage_rates.append(covered / total)
            stop_times_analyzed.append(stop_time)
    
    if coverage_rates:
        ax4.plot(stop_times_analyzed, coverage_rates, 'g-o', linewidth=2, markersize=4,
                label='Empirical Coverage')
        ax4.axhline(y=1-alpha, color='red', linestyle='--', linewidth=2,
                   label=f'Target Coverage = {1-alpha:.2f}')
        
        ax4.set_xlabel('Stopping Time')
        ax4.set_ylabel('Coverage Rate')
        ax4.set_title('Anytime Validity: Coverage Across Stopping Times')
        ax4.grid(config.show_grid, alpha=0.3)
        ax4.legend()
        ax4.set_ylim(0.8, 1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, format=config.save_format, dpi=config.dpi, 
                   bbox_inches='tight')
    
    return fig


def create_interactive_demo(
    port: int = 8501,
    debug: bool = False
) -> None:
    """
    Create interactive Streamlit demo for sequential testing.
    
    Args:
        port: Port number for Streamlit app
        debug: Enable debug mode
        
    Note:
        Requires streamlit to be installed. Run with: streamlit run this_file.py
    """
    if not STREAMLIT_AVAILABLE:
        raise ImportError("Streamlit is required for interactive demo. Install with: pip install streamlit")
    
    st.set_page_config(
        page_title="Sequential Verification Demo",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🔍 Sequential Verification Interactive Demo")
    st.markdown("""
    This interactive demo demonstrates sequential hypothesis testing for model verification.
    Adjust the parameters below to see how they affect the verification process.
    """)
    
    # Sidebar controls
    st.sidebar.header("📋 Test Parameters")
    
    # Basic parameters
    tau = st.sidebar.slider("Threshold (τ)", 0.01, 0.20, 0.05, 0.01)
    alpha = st.sidebar.slider("Type I Error (α)", 0.001, 0.10, 0.05, 0.001)
    beta = st.sidebar.slider("Type II Error (β)", 0.001, 0.10, 0.05, 0.001)
    max_samples = st.sidebar.slider("Max Samples", 50, 1000, 500, 50)
    
    # Data generation parameters
    st.sidebar.subheader("🎲 Data Generation")
    true_mean = st.sidebar.slider("True Mean (μ)", 0.0, 0.15, 0.03, 0.01)
    noise_std = st.sidebar.slider("Noise Std (σ)", 0.01, 0.10, 0.02, 0.01)
    seed = st.sidebar.slider("Random Seed", 1, 1000, 42, 1)
    
    # Visualization options
    st.sidebar.subheader("🎨 Visualization")
    show_confidence = st.sidebar.checkbox("Show Confidence Bounds", True)
    show_regions = st.sidebar.checkbox("Show Decision Regions", True)
    show_details = st.sidebar.checkbox("Show Details", True)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📈 Real-time Sequential Test")
        
        # Generate test data
        np.random.seed(seed)
        
        def generate_data_stream():
            """Generate synthetic data stream"""
            for i in range(max_samples):
                yield np.random.normal(true_mean, noise_std)
        
        # Run sequential test (simplified for demo)
        try:
            # For demo purposes, we'll simulate the sequential test
            result = _simulate_sequential_test(
                true_mean=true_mean,
                noise_std=noise_std,
                tau=tau,
                alpha=alpha,
                beta=beta,
                max_samples=max_samples,
                seed=seed
            )
            
            # Create the plot
            fig = _plot_streamlit_trajectory(
                result, tau, show_confidence, show_regions, show_details
            )
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error running sequential test: {e}")
            st.info("This is a simplified demo. Full functionality requires the complete PoT framework.")
    
    with col2:
        st.header("📊 Test Results")
        
        if 'result' in locals():
            # Decision
            decision_color = "green" if result['decision'] == 'H0' else "red"
            st.markdown(f"**Decision:** <span style='color: {decision_color}'>{result['decision']}</span>", 
                       unsafe_allow_html=True)
            
            # Key metrics
            st.metric("Stopping Time", f"{result['stopped_at']}")
            st.metric("Final Mean", f"{result['final_mean']:.4f}")
            st.metric("Efficiency", f"{100 * (1 - result['stopped_at']/max_samples):.1f}%")
            
            # Parameter summary
            st.subheader("📋 Parameters")
            st.write(f"**Threshold (τ):** {tau:.3f}")
            st.write(f"**Type I Error (α):** {alpha:.3f}")
            st.write(f"**Type II Error (β):** {beta:.3f}")
            st.write(f"**True Mean (μ):** {true_mean:.3f}")
            st.write(f"**Noise (σ):** {noise_std:.3f}")
            
            # Interpretation
            st.subheader("🎯 Interpretation")
            if result['decision'] == 'H0':
                st.success("✅ Model passes verification (H₀ accepted)")
                st.write("The observed distances are consistent with the reference model.")
            else:
                st.error("❌ Model fails verification (H₁ accepted)")
                st.write("The observed distances suggest the model differs from the reference.")
        
        # Educational information
        st.subheader("📚 About Sequential Testing")
        st.markdown("""
        **Key Benefits:**
        - 🚀 **Efficiency**: Stop early when decision is clear
        - 📈 **Anytime Validity**: Valid at any stopping time
        - ⚖️ **Error Control**: Maintains Type I/II error rates
        
        **How it works:**
        1. Collect samples sequentially
        2. Update running statistics
        3. Check stopping condition at each step
        4. Stop when confidence bounds exclude/include threshold
        """)
    
    # Additional demos
    if st.checkbox("🔬 Show Advanced Features"):
        st.header("🔬 Advanced Features Demo")
        
        # Power analysis
        st.subheader("📊 Power Analysis")
        if st.button("Run Power Analysis"):
            effect_sizes = np.linspace(0.0, 0.10, 10)
            try:
                fig_power = _plot_streamlit_power_analysis(tau, alpha, beta, effect_sizes)
                st.pyplot(fig_power)
            except Exception as e:
                st.error(f"Power analysis error: {e}")
        
        # Multiple trajectories
        st.subheader("🔄 Multiple Runs")
        n_runs = st.slider("Number of Runs", 5, 50, 10)
        if st.button("Run Multiple Tests"):
            try:
                trajectories = []
                progress_bar = st.progress(0)
                for i in range(n_runs):
                    result_i = _simulate_sequential_test(
                        true_mean, noise_std, tau, alpha, beta, max_samples, seed + i
                    )
                    trajectories.append(result_i)
                    progress_bar.progress((i + 1) / n_runs)
                
                fig_multiple = _plot_streamlit_multiple_trajectories(trajectories, tau)
                st.pyplot(fig_multiple)
                
                # Summary statistics
                decisions = [t['decision'] for t in trajectories]
                stopping_times = [t['stopped_at'] for t in trajectories]
                
                st.write(f"**H₀ decisions:** {decisions.count('H0')}/{n_runs}")
                st.write(f"**H₁ decisions:** {decisions.count('H1')}/{n_runs}")
                st.write(f"**Average stopping time:** {np.mean(stopping_times):.1f}")
                
            except Exception as e:
                st.error(f"Multiple runs error: {e}")


# Helper functions for visualization

def _simulate_power_analysis(effect_sizes, tau, alpha, beta, max_samples):
    """Simulate power analysis when the real function is not available"""
    power_values = []
    stopping_times = []
    
    for effect in effect_sizes:
        # Simplified power calculation
        if effect <= 0:
            power = alpha  # Type I error rate
            avg_stop = max_samples * 0.8
        else:
            # Approximate power based on effect size
            z_score = effect / (0.02 / np.sqrt(100))  # Rough approximation
            power = min(0.99, max(alpha, 1 - beta + (effect / tau) * (1 - alpha - beta)))
            avg_stop = max(10, max_samples * (0.2 + 0.6 * np.exp(-effect * 50)))
        
        power_values.append(power)
        stopping_times.append(avg_stop)
    
    return {
        'effect_sizes': effect_sizes,
        'power_values': power_values,
        'stopping_times': stopping_times
    }


def _compute_fixed_power(effect_size, tau, alpha, n_fixed):
    """Compute theoretical power for fixed-sample test"""
    if effect_size <= 0:
        return alpha
    
    # Simplified power calculation for one-sample t-test
    std_error = 0.02 / np.sqrt(n_fixed)  # Rough approximation
    z_score = effect_size / std_error
    
    # Approximate power using normal approximation
    from scipy import stats as scipy_stats
    power = 1 - scipy_stats.norm.cdf(scipy_stats.norm.ppf(1 - alpha) - z_score)
    return min(0.99, max(alpha, power))


def _simulate_sequential_test(true_mean, noise_std, tau, alpha, beta, max_samples, seed):
    """Simulate a sequential test for the interactive demo"""
    np.random.seed(seed)
    
    # Simple sequential test simulation
    trajectory = []
    running_sum = 0
    n = 0
    
    for i in range(max_samples):
        n += 1
        sample = np.random.normal(true_mean, noise_std)
        running_sum += sample
        mean = running_sum / n
        
        # Simple confidence radius (not exact EB)
        std_error = noise_std / np.sqrt(n)
        radius = 1.96 * std_error  # Approximate 95% CI
        
        trajectory.append({
            'n': n,
            'mean': mean,
            'radius': radius
        })
        
        # Simple stopping rule
        if mean + radius < tau:
            decision = 'H0'
            break
        elif mean - radius > tau:
            decision = 'H1'
            break
    else:
        # Forced stop
        decision = 'H0' if mean < tau else 'H1'
    
    return {
        'decision': decision,
        'stopped_at': n,
        'final_mean': mean,
        'trajectory': trajectory
    }


def _plot_streamlit_trajectory(result, tau, show_confidence, show_regions, show_details):
    """Create trajectory plot for Streamlit"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    trajectory = result['trajectory']
    time_points = [t['n'] for t in trajectory]
    means = [t['mean'] for t in trajectory]
    
    # Main trajectory
    ax.plot(time_points, means, 'b-', linewidth=2, label='Running Mean')
    
    if show_confidence:
        lower_bounds = [t['mean'] - t['radius'] for t in trajectory]
        upper_bounds = [t['mean'] + t['radius'] for t in trajectory]
        ax.fill_between(time_points, lower_bounds, upper_bounds, 
                       alpha=0.3, color='blue', label='Confidence Bounds')
    
    # Threshold
    ax.axhline(y=tau, color='red', linestyle='--', linewidth=2, 
              label=f'Threshold τ = {tau:.3f}')
    
    if show_regions:
        y_max = max(means) * 1.2
        y_min = min(means) * 0.8
        ax.fill_between([0, max(time_points)], [y_min, y_min], [tau, tau], 
                       alpha=0.1, color='green', label='H₀ Region')
        ax.fill_between([0, max(time_points)], [tau, tau], [y_max, y_max], 
                       alpha=0.1, color='red', label='H₁ Region')
    
    # Mark stopping point
    ax.scatter(result['stopped_at'], result['final_mean'], 
              color='red', s=100, zorder=5, label=f'Stop at n={result["stopped_at"]}')
    
    ax.set_xlabel('Sample Number')
    ax.set_ylabel('Running Mean Distance')
    ax.set_title(f'Sequential Test: {result["decision"]} at n={result["stopped_at"]}')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig


def _plot_streamlit_power_analysis(tau, alpha, beta, effect_sizes):
    """Create power analysis plot for Streamlit"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Simulate power analysis
    power_data = _simulate_power_analysis(effect_sizes, tau, alpha, beta, 500)
    
    # Power curve
    ax1.plot(power_data['effect_sizes'], power_data['power_values'], 
            'b-o', linewidth=2, label='Sequential Test')
    ax1.axhline(y=1-beta, color='gray', linestyle=':', 
               label=f'Target Power = {1-beta:.2f}')
    ax1.set_xlabel('Effect Size')
    ax1.set_ylabel('Power')
    ax1.set_title('Power Curve')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Stopping times
    ax2.plot(power_data['effect_sizes'], power_data['stopping_times'], 
            'g-o', linewidth=2)
    ax2.set_xlabel('Effect Size')
    ax2.set_ylabel('Expected Sample Size')
    ax2.set_title('Sample Size Efficiency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def _plot_streamlit_multiple_trajectories(trajectories, tau):
    """Create multiple trajectories plot for Streamlit"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Multiple trajectories
    for i, result in enumerate(trajectories):
        trajectory = result['trajectory']
        time_points = [t['n'] for t in trajectory]
        means = [t['mean'] for t in trajectory]
        color = 'green' if result['decision'] == 'H0' else 'red'
        ax1.plot(time_points, means, color=color, alpha=0.5, linewidth=1)
    
    ax1.axhline(y=tau, color='black', linestyle='--', linewidth=2, 
               label=f'Threshold τ = {tau:.3f}')
    ax1.set_xlabel('Sample Number')
    ax1.set_ylabel('Running Mean')
    ax1.set_title(f'Multiple Sequential Tests (n={len(trajectories)})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Stopping time histogram
    stopping_times = [t['stopped_at'] for t in trajectories]
    ax2.hist(stopping_times, bins=min(15, len(set(stopping_times))), 
            alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(x=np.mean(stopping_times), color='red', linestyle='--', 
               label=f'Mean = {np.mean(stopping_times):.1f}')
    ax2.set_xlabel('Stopping Time')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Stopping Time Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    return fig


# Example usage functions

def demo_trajectory_plot():
    """Demonstrate trajectory plotting with example data"""
    print("Demo: Sequential Verification Trajectory")
    print("=" * 50)
    
    # Create example SPRT result
    from .sequential import SequentialState
    
    # Simulate trajectory data
    trajectory = []
    state = SequentialState()
    np.random.seed(42)
    
    for i in range(1, 51):
        # Simulate sample
        sample = np.random.normal(0.03, 0.02)
        state.n = i
        state.mean = (state.mean * (i-1) + sample) / i
        
        # Simple radius calculation
        radius = 0.02 / np.sqrt(i)
        
        trajectory.append({
            'n': i,
            'mean': state.mean,
            'radius': radius
        })
        
        # Check stopping condition
        if state.mean + radius < 0.05:
            decision = 'H0'
            break
        elif state.mean - radius > 0.05:
            decision = 'H1'
            break
    else:
        decision = 'H1' if state.mean > 0.05 else 'H0'
    
    # Create mock SPRTResult
    class MockSPRTResult:
        def __init__(self):
            self.decision = decision
            self.stopped_at = i
            self.trajectory = trajectory
            self.tau = 0.05
            self.p_value = 0.023
            self.forced_stop = False
    
    result = MockSPRTResult()
    
    # Plot trajectory
    fig = plot_verification_trajectory(result, save_path='demo_trajectory.png')
    plt.show()
    print(f"Decision: {result.decision}, Stopped at: {result.stopped_at}")


def demo_operating_characteristics():
    """Demonstrate operating characteristics plotting"""
    print("\nDemo: Operating Characteristics")
    print("=" * 50)
    
    fig = plot_operating_characteristics(
        tau=0.05,
        alpha=0.05,
        beta=0.05,
        effect_sizes=np.linspace(0.0, 0.12, 15),
        save_path='demo_oc.png'
    )
    plt.show()
    print("Operating characteristics plot generated")


def demo_anytime_validity():
    """Demonstrate anytime validity with multiple trajectories"""
    print("\nDemo: Anytime Validity")
    print("=" * 50)
    
    # Generate multiple mock trajectories
    trajectories = []
    np.random.seed(123)
    
    for run in range(20):
        trajectory = []
        state = SequentialState()
        
        for i in range(1, 101):
            sample = np.random.normal(0.03 + run * 0.001, 0.02)
            state.n = i
            state.mean = (state.mean * (i-1) + sample) / i
            radius = 0.02 / np.sqrt(i)
            
            trajectory.append({
                'n': i,
                'mean': state.mean,
                'radius': radius
            })
            
            if state.mean + radius < 0.05:
                decision = 'H0'
                break
            elif state.mean - radius > 0.05:
                decision = 'H1'
                break
        else:
            decision = 'H1' if state.mean > 0.05 else 'H0'
        
        class MockSPRTResult:
            def __init__(self):
                self.decision = decision
                self.stopped_at = i
                self.trajectory = trajectory
                self.tau = 0.05
        
        trajectories.append(MockSPRTResult())
    
    fig = plot_anytime_validity(trajectories, save_path='demo_anytime.png')
    plt.show()
    print(f"Anytime validity plot generated with {len(trajectories)} trajectories")


if __name__ == "__main__":
    # Run all demos
    demo_trajectory_plot()
    demo_operating_characteristics()
    demo_anytime_validity()
    
    print("\n" + "=" * 50)
    print("All visualization demos completed!")
    print("Generated files: demo_trajectory.png, demo_oc.png, demo_anytime.png")
    print("\nTo run interactive demo:")
    print("streamlit run pot/core/visualize_sequential.py")