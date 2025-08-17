"""
Example: Track semantic evolution over training.

This example demonstrates how to track the evolution of semantic
representations during model training or over time.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from pot.semantic import (
    TopographicalEvolutionTracker,
    ConceptLibrary,
    SemanticMatcher,
    create_topographical_semantic_system,
    analyze_semantic_evolution
)
from pot.semantic.topography_utils import (
    track_cluster_evolution,
    compute_cluster_stability
)
from pot.semantic.topography_visualizer import (
    plot_evolution_trajectory,
    plot_drift_metrics_dashboard,
    create_evolution_animation,
    plot_cluster_evolution_heatmap
)


def simulate_training_evolution(n_epochs=10, n_samples=100, dim=64):
    """
    Simulate the evolution of embeddings during training.
    
    Args:
        n_epochs: Number of training epochs
        n_samples: Number of samples per epoch
        dim: Embedding dimension
    
    Returns:
        List of embedding snapshots and timestamps
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    snapshots = []
    timestamps = []
    
    # Initial random embeddings
    base_embeddings = torch.randn(n_samples, dim)
    
    for epoch in range(n_epochs):
        # Simulate training progress - embeddings become more structured
        noise_scale = 1.0 / (epoch + 1)  # Decreasing noise
        
        # Add structure: clusters emerge over time
        if epoch < 3:
            # Early training: mostly noise
            embeddings = base_embeddings + torch.randn(n_samples, dim) * noise_scale
        elif epoch < 6:
            # Mid training: clusters start forming
            cluster_centers = torch.randn(3, dim) * 2
            labels = np.random.choice(3, n_samples)
            embeddings = torch.zeros(n_samples, dim)
            for i in range(n_samples):
                embeddings[i] = cluster_centers[labels[i]] + torch.randn(dim) * noise_scale
        else:
            # Late training: well-defined clusters
            cluster_centers = torch.randn(3, dim) * 3
            labels = np.random.choice(3, n_samples)
            embeddings = torch.zeros(n_samples, dim)
            for i in range(n_samples):
                embeddings[i] = cluster_centers[labels[i]] + torch.randn(dim) * noise_scale * 0.5
        
        snapshots.append(embeddings)
        timestamps.append(float(epoch))
        
        print(f"Epoch {epoch + 1}/{n_epochs}: Generated snapshot")
    
    return snapshots, timestamps


def track_evolution_basic(snapshots, timestamps):
    """
    Basic evolution tracking example.
    
    Args:
        snapshots: List of embedding snapshots
        timestamps: List of timestamps
    """
    print("\n" + "="*50)
    print("Basic Evolution Tracking")
    print("="*50)
    
    # Create evolution tracker
    tracker = TopographicalEvolutionTracker()
    
    # Add snapshots
    for i, (snapshot, timestamp) in enumerate(zip(snapshots, timestamps)):
        # Convert to numpy if needed
        if isinstance(snapshot, torch.Tensor):
            snapshot = snapshot.detach().numpy()
        
        # Add snapshot with metrics computation
        tracker.add_snapshot(
            snapshot,
            timestamp=timestamp,
            compute_metrics=True
        )
        print(f"Added snapshot at t={timestamp:.1f}")
    
    # Compute drift metrics
    drift_metrics = tracker.compute_drift_metrics()
    
    print("\nDrift Metrics:")
    print("-" * 30)
    for key, value in drift_metrics.items():
        if isinstance(value, float):
            print(f"{key:<20}: {value:.4f}")
        elif isinstance(value, list) and len(value) > 0:
            print(f"{key:<20}: {len(value)} values, mean={np.mean(value):.4f}")
    
    # Detect regime changes
    print("\nRegime Change Detection:")
    print("-" * 30)
    
    changes_gradient = tracker.detect_regime_changes(method='gradient', threshold=0.5)
    changes_variance = tracker.detect_regime_changes(method='variance', threshold=2.0)
    
    print(f"Gradient method: {len(changes_gradient)} changes detected")
    if changes_gradient:
        print(f"  At indices: {changes_gradient}")
    
    print(f"Variance method: {len(changes_variance)} changes detected")
    if changes_variance:
        print(f"  At indices: {changes_variance}")
    
    return tracker


def track_cluster_evolution_example(snapshots, timestamps):
    """
    Track how clusters evolve over time.
    
    Args:
        snapshots: List of embedding snapshots
        timestamps: List of timestamps
    """
    print("\n" + "="*50)
    print("Cluster Evolution Tracking")
    print("="*50)
    
    # Project snapshots to 2D for clustering
    from pot.semantic import TopographicalProjector
    
    projector = TopographicalProjector('pca')
    projected_snapshots = []
    
    for snapshot in snapshots:
        projected = projector.project_latents(snapshot)
        projected_snapshots.append(projected)
    
    # Track cluster evolution
    evolution = track_cluster_evolution(
        projected_snapshots,
        method='kmeans',
        n_clusters=3
    )
    
    print(f"Tracked {evolution['n_snapshots']} snapshots")
    print(f"Mean number of clusters: {evolution['mean_n_clusters']:.2f}")
    print(f"Cluster variance: {evolution['cluster_variance']:.2f}")
    
    if 'mean_stability' in evolution:
        print(f"Mean stability: {evolution['mean_stability']:.3f}")
    
    # Analyze transitions
    print("\nCluster Transitions:")
    print("-" * 30)
    
    for i, transition in enumerate(evolution['transitions']):
        n_splits = np.sum(transition['n_splits'])
        n_merges = np.sum(transition['n_merges'])
        print(f"t={timestamps[i]:.1f} → t={timestamps[i+1]:.1f}: "
              f"{n_splits} splits, {n_merges} merges")
    
    return evolution, projected_snapshots


def visualize_evolution(tracker, snapshots, timestamps):
    """
    Create visualizations of the evolution.
    
    Args:
        tracker: TopographicalEvolutionTracker with data
        snapshots: Original embedding snapshots
        timestamps: Timestamps
    """
    print("\n" + "="*50)
    print("Creating Evolution Visualizations")
    print("="*50)
    
    # Get projected snapshots from tracker
    projected_snapshots = []
    for snapshot_data in tracker.snapshots:
        projected_snapshots.append(snapshot_data['data'])
    
    # 1. Evolution trajectory plot
    print("Creating evolution trajectory plot...")
    traj_fig = plot_evolution_trajectory(
        projected_snapshots,
        timestamps=timestamps,
        title="Embedding Evolution Trajectory"
    )
    
    # 2. Drift metrics dashboard
    print("Creating drift metrics dashboard...")
    drift_metrics = tracker.compute_drift_metrics()
    drift_fig = plot_drift_metrics_dashboard(
        drift_metrics,
        timestamps=timestamps,
        title="Evolution Drift Metrics"
    )
    
    # 3. Cluster evolution heatmap
    print("Creating cluster evolution heatmap...")
    evolution, _ = track_cluster_evolution_example(snapshots, timestamps)
    
    if evolution['cluster_labels']:
        cluster_fig = plot_cluster_evolution_heatmap(
            evolution['cluster_labels'],
            timestamps=timestamps,
            title="Cluster Evolution Heatmap"
        )
    else:
        cluster_fig = None
    
    return traj_fig, drift_fig, cluster_fig


def semantic_evolution_with_concepts(snapshots, timestamps):
    """
    Track evolution relative to concept library.
    
    Args:
        snapshots: Embedding snapshots
        timestamps: Timestamps
    """
    print("\n" + "="*50)
    print("Semantic Evolution with Concept Library")
    print("="*50)
    
    # Create semantic system
    dim = snapshots[0].shape[1]
    library, matcher, projector = create_topographical_semantic_system(
        dim=dim,
        projection_method='pca'
    )
    
    # Add reference concepts (from first snapshot)
    print("Building concept library from initial snapshot...")
    initial_snapshot = snapshots[0]
    
    # Create 3 concepts from clusters in initial data
    n_per_concept = len(initial_snapshot) // 3
    for i in range(3):
        start_idx = i * n_per_concept
        end_idx = start_idx + n_per_concept
        concept_embeddings = initial_snapshot[start_idx:end_idx]
        library.add_concept(f'concept_{i}', concept_embeddings)
    
    print(f"Added {len(library.concepts)} concepts to library")
    
    # Analyze evolution
    print("Analyzing semantic evolution...")
    
    # Convert snapshots to list of lists for analysis
    embeddings_history = []
    for snapshot in snapshots:
        embeddings_list = [snapshot[i] for i in range(len(snapshot))]
        embeddings_history.append(embeddings_list)
    
    result = analyze_semantic_evolution(
        embeddings_history,
        library,
        timestamps=timestamps,
        method='pca'
    )
    
    print("\nEvolution Analysis Results:")
    print("-" * 30)
    print(f"Number of snapshots: {result['n_snapshots']}")
    print(f"Total drift: {result['total_drift']:.4f}")
    print(f"Stability: {result['stability']:.4f}")
    
    if result['regime_changes']:
        print(f"Regime changes detected: {len(result['regime_changes'])}")
    
    # Track semantic trajectory for a subset of embeddings
    print("\nTracking semantic trajectories...")
    
    # Take first embedding from each snapshot
    trajectory_embeddings = [snapshot[0] for snapshot in snapshots]
    trajectory_result = matcher.track_semantic_trajectory(
        trajectory_embeddings,
        timestamps=timestamps,
        projection_method='pca'
    )
    
    print(f"Trajectory length: {trajectory_result['trajectory_length']}")
    print(f"Total distance: {trajectory_result['total_distance']:.3f}")
    print(f"Mean velocity: {trajectory_result['mean_velocity']:.3f}")
    
    if trajectory_result['concept_visits']:
        print(f"Concepts visited: {len(trajectory_result['concept_visits'])}")
        for visit in trajectory_result['concept_visits'][:3]:
            print(f"  - {visit['concept']} at t={visit['timestamp']:.1f}")
    
    return result, trajectory_result


def create_evolution_animation_example(snapshots, timestamps):
    """
    Create an animated visualization of evolution.
    
    Args:
        snapshots: Embedding snapshots
        timestamps: Timestamps
    """
    print("\n" + "="*50)
    print("Creating Evolution Animation")
    print("="*50)
    
    # Project snapshots
    from pot.semantic import TopographicalProjector
    
    projector = TopographicalProjector('pca')
    projected_snapshots = []
    
    for snapshot in snapshots:
        projected = projector.project_latents(snapshot)
        projected_snapshots.append(projected)
    
    # Create animation (saves as HTML file)
    print("Generating animation...")
    animation_path = "evolution_animation.html"
    
    try:
        fig = create_evolution_animation(
            projected_snapshots,
            timestamps=timestamps,
            title="Embedding Evolution Animation",
            save_path=animation_path
        )
        print(f"Animation saved to: {animation_path}")
    except Exception as e:
        print(f"Could not create animation: {e}")
        print("(Animation requires plotly to be installed)")
    
    return projected_snapshots


def main():
    """Main example execution."""
    print("="*60)
    print("Semantic Evolution Tracking Example")
    print("="*60)
    
    # Simulate training evolution
    print("\nSimulating training evolution...")
    snapshots, timestamps = simulate_training_evolution(
        n_epochs=8,
        n_samples=150,
        dim=64
    )
    print(f"Generated {len(snapshots)} snapshots")
    
    # Basic evolution tracking
    tracker = track_evolution_basic(snapshots, timestamps)
    
    # Cluster evolution tracking
    evolution, projected_snapshots = track_cluster_evolution_example(
        snapshots, timestamps
    )
    
    # Create visualizations
    traj_fig, drift_fig, cluster_fig = visualize_evolution(
        tracker, snapshots, timestamps
    )
    
    # Semantic evolution with concepts
    semantic_result, trajectory_result = semantic_evolution_with_concepts(
        snapshots, timestamps
    )
    
    # Create animation (optional)
    # create_evolution_animation_example(snapshots, timestamps)
    
    # Show plots
    plt.show()
    
    print("\n" + "="*60)
    print("Evolution tracking example completed!")
    print("="*60)


if __name__ == "__main__":
    main()