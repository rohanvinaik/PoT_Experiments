"""
Example: Compare different projection methods.

This example demonstrates how to compare different dimensionality
reduction methods for topographical visualization and analysis.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import time
from scipy.spatial.distance import pdist

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from pot.semantic import (
    TopographicalProjector,
    ConceptLibrary,
    create_topographical_semantic_system
)
from pot.semantic.topography_utils import (
    prepare_latents_for_projection,
    compute_trustworthiness,
    compute_continuity,
    compute_shepard_correlation,
    compute_neighborhood_preservation,
    compute_stress_metrics,
    estimate_intrinsic_dimension
)
from pot.semantic.topography_visualizer import (
    compare_projections,
    plot_distance_preservation
)


def create_test_datasets():
    """
    Create different types of test datasets for comparison.
    
    Returns:
        Dictionary of datasets with different characteristics
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    datasets = {}
    
    # 1. Swiss Roll (manifold data)
    n_samples = 1000
    t = 1.5 * np.pi * (1 + 2 * np.random.rand(n_samples))
    height = 20 * np.random.rand(n_samples)
    X = np.zeros((n_samples, 3))
    X[:, 0] = t * np.cos(t)
    X[:, 1] = height
    X[:, 2] = t * np.sin(t)
    # Add noise dimensions
    noise = np.random.randn(n_samples, 47) * 0.1
    swiss_roll = np.hstack([X, noise])
    datasets['Swiss Roll'] = torch.tensor(swiss_roll, dtype=torch.float32)
    
    # 2. Clustered data
    n_clusters = 5
    n_per_cluster = 200
    dim = 50
    clusters = []
    labels = []
    for i in range(n_clusters):
        center = torch.randn(dim) * 3
        cluster = torch.randn(n_per_cluster, dim) * 0.5 + center
        clusters.append(cluster)
        labels.extend([i] * n_per_cluster)
    datasets['Clustered'] = torch.cat(clusters)
    datasets['Clustered_labels'] = np.array(labels)
    
    # 3. High-dimensional Gaussian
    datasets['Gaussian'] = torch.randn(1000, 50)
    
    # 4. Sparse data (many zeros)
    sparse_data = torch.randn(1000, 50)
    mask = torch.rand(1000, 50) > 0.7  # 70% zeros
    sparse_data[mask] = 0
    datasets['Sparse'] = sparse_data
    
    # 5. Linear subspace
    # Data lies in a lower-dimensional linear subspace
    basis = torch.randn(10, 50)  # 10-dimensional subspace in 50D
    coefficients = torch.randn(1000, 10)
    linear_data = coefficients @ basis
    linear_data += torch.randn(1000, 50) * 0.01  # Small noise
    datasets['Linear Subspace'] = linear_data
    
    return datasets


def compare_methods_on_dataset(data, dataset_name, labels=None):
    """
    Compare projection methods on a single dataset.
    
    Args:
        data: Input data tensor
        dataset_name: Name of the dataset
        labels: Optional labels for visualization
    
    Returns:
        Dictionary of results
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Shape: {data.shape}")
    print(f"{'='*60}")
    
    # Estimate intrinsic dimension
    data_np = prepare_latents_for_projection(data)
    intrinsic_dim = estimate_intrinsic_dimension(data_np, method='mle')
    print(f"Estimated intrinsic dimension: {intrinsic_dim:.2f}")
    
    results = {}
    methods = ['pca', 'umap', 'tsne']
    
    # For t-SNE, use a subset if data is large
    subset_size = min(500, len(data))
    if len(data) > subset_size:
        subset_indices = np.random.choice(len(data), subset_size, replace=False)
        data_subset = data[subset_indices]
        labels_subset = labels[subset_indices] if labels is not None else None
    else:
        data_subset = data
        labels_subset = labels
        subset_indices = np.arange(len(data))
    
    data_subset_np = prepare_latents_for_projection(data_subset)
    
    for method in methods:
        print(f"\n{method.upper()} Projection:")
        print("-" * 40)
        
        # Time the projection
        start_time = time.time()
        
        # Create projector
        if method == 'tsne':
            # Use subset for t-SNE
            projector = TopographicalProjector(method)
            projected = projector.project_latents(data_subset)
            elapsed_time = time.time() - start_time
            
            # Store subset results
            results[method] = {
                'projection': projected,
                'time': elapsed_time,
                'subset_indices': subset_indices,
                'data': data_subset_np
            }
        else:
            # Use full data for other methods
            projector = TopographicalProjector(method)
            projected = projector.project_latents(data)
            elapsed_time = time.time() - start_time
            
            # For comparison, also get subset projection
            projected_subset = projected[subset_indices] if len(data) > subset_size else projected
            
            results[method] = {
                'projection': projected_subset,
                'time': elapsed_time,
                'subset_indices': subset_indices,
                'data': data_subset_np
            }
        
        print(f"Projection time: {elapsed_time:.2f} seconds")
        
        # Compute quality metrics
        metrics = compute_neighborhood_preservation(
            results[method]['data'],
            results[method]['projection'],
            k=10
        )
        
        stress = compute_stress_metrics(
            results[method]['data'],
            results[method]['projection'],
            normalized=True
        )
        
        results[method]['metrics'] = {**metrics, **stress}
        
        # Print key metrics
        print(f"Trustworthiness: {metrics['trustworthiness']:.3f}")
        print(f"Continuity: {metrics['continuity']:.3f}")
        print(f"Kruskal Stress: {stress['kruskal_stress_1']:.3f}")
        print(f"Shepard Correlation: {stress['shepard_correlation']:.3f}")
    
    return results, labels_subset


def create_comparison_table(all_results):
    """
    Create a comparison table of all methods across datasets.
    
    Args:
        all_results: Dictionary of results for all datasets
    """
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS SUMMARY")
    print("="*80)
    
    # Metrics to compare
    metrics_to_show = [
        'trustworthiness',
        'continuity',
        'shepard_correlation',
        'kruskal_stress_1'
    ]
    
    methods = ['pca', 'umap', 'tsne']
    
    for metric in metrics_to_show:
        print(f"\n{metric.upper().replace('_', ' ')}:")
        print("-" * 60)
        print(f"{'Dataset':<20} {'PCA':<15} {'UMAP':<15} {'t-SNE':<15}")
        print("-" * 60)
        
        for dataset_name, results in all_results.items():
            if dataset_name.endswith('_labels'):
                continue
            row = f"{dataset_name:<20}"
            for method in methods:
                if method in results:
                    value = results[method]['metrics'].get(metric, 0)
                    row += f" {value:<14.3f}"
                else:
                    row += f" {'N/A':<14}"
            print(row)
    
    # Timing comparison
    print(f"\nPROJECTION TIME (seconds):")
    print("-" * 60)
    print(f"{'Dataset':<20} {'PCA':<15} {'UMAP':<15} {'t-SNE':<15}")
    print("-" * 60)
    
    for dataset_name, results in all_results.items():
        if dataset_name.endswith('_labels'):
            continue
        row = f"{dataset_name:<20}"
        for method in methods:
            if method in results:
                time_val = results[method]['time']
                row += f" {time_val:<14.3f}"
            else:
                row += f" {'N/A':<14}"
        print(row)


def visualize_all_comparisons(all_results):
    """
    Create visualizations comparing all methods.
    
    Args:
        all_results: Dictionary of results
    """
    # Create a figure with subplots for each dataset
    n_datasets = len([k for k in all_results.keys() if not k.endswith('_labels')])
    fig, axes = plt.subplots(n_datasets, 3, figsize=(15, 5 * n_datasets))
    
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    dataset_idx = 0
    for dataset_name, results in all_results.items():
        if dataset_name.endswith('_labels'):
            continue
        
        # Get labels if available
        labels = all_results.get(f"{dataset_name}_labels", None)
        if labels is not None and 'subset_indices' in results.get('pca', {}):
            labels = labels[results['pca']['subset_indices']]
        
        for method_idx, method in enumerate(['pca', 'umap', 'tsne']):
            if method in results:
                ax = axes[dataset_idx, method_idx]
                projection = results[method]['projection']
                
                if labels is not None:
                    scatter = ax.scatter(
                        projection[:, 0],
                        projection[:, 1],
                        c=labels,
                        cmap='tab10',
                        alpha=0.6,
                        s=10
                    )
                else:
                    ax.scatter(
                        projection[:, 0],
                        projection[:, 1],
                        alpha=0.6,
                        s=10
                    )
                
                ax.set_title(f"{dataset_name} - {method.upper()}")
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
                ax.grid(True, alpha=0.3)
        
        dataset_idx += 1
    
    plt.tight_layout()
    return fig


def analyze_distance_preservation(data, projections):
    """
    Analyze how well distances are preserved.
    
    Args:
        data: Original high-dimensional data
        projections: Dictionary of projections
    """
    print("\n" + "="*60)
    print("Distance Preservation Analysis")
    print("="*60)
    
    # Compute original distances
    original_distances = pdist(data)
    
    for method_name, projection in projections.items():
        print(f"\n{method_name.upper()}:")
        
        # Compute projected distances
        projected_distances = pdist(projection)
        
        # Shepard correlation
        corr = compute_shepard_correlation(original_distances, projected_distances)
        print(f"Shepard correlation: {corr:.3f}")
        
        # Create distance preservation plot
        fig = plot_distance_preservation(
            original_distances,
            projected_distances,
            method_name=method_name.upper(),
            sample_size=1000  # Sample for visualization
        )
    
    return fig


def main():
    """Main example execution."""
    print("="*80)
    print("COMPARATIVE ANALYSIS OF PROJECTION METHODS")
    print("="*80)
    
    # Create test datasets
    print("\nCreating test datasets...")
    datasets = create_test_datasets()
    print(f"Created {len([k for k in datasets.keys() if not k.endswith('_labels')])} datasets")
    
    # Run comparison on each dataset
    all_results = {}
    
    for dataset_name, data in datasets.items():
        if dataset_name.endswith('_labels'):
            all_results[dataset_name] = data
            continue
        
        labels = datasets.get(f"{dataset_name}_labels", None)
        results, labels_subset = compare_methods_on_dataset(
            data, dataset_name, labels
        )
        all_results[dataset_name] = results
        if labels_subset is not None:
            all_results[f"{dataset_name}_labels_subset"] = labels_subset
    
    # Create comparison table
    create_comparison_table(all_results)
    
    # Create visualizations
    print("\n" + "="*60)
    print("Creating Visualizations")
    print("="*60)
    
    comparison_fig = visualize_all_comparisons(all_results)
    
    # Analyze distance preservation for one dataset
    print("\nAnalyzing distance preservation for Clustered dataset...")
    clustered_results = all_results['Clustered']
    projections_dict = {
        method: results['projection'] 
        for method, results in clustered_results.items()
    }
    
    distance_fig = analyze_distance_preservation(
        clustered_results['pca']['data'],  # Use subset data
        projections_dict
    )
    
    # Show all plots
    plt.show()
    
    print("\n" + "="*80)
    print("Comparative analysis completed!")
    print("="*80)
    
    # Print recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 40)
    print("• PCA: Fast, preserves global structure, good for linear relationships")
    print("• UMAP: Good balance of speed and quality, preserves both local and global structure")
    print("• t-SNE: Best for visualization, preserves local structure, but slow for large datasets")
    print("\nChoose based on your specific needs:")
    print("- Real-time analysis: PCA")
    print("- General purpose: UMAP")
    print("- Publication visualizations: t-SNE")


if __name__ == "__main__":
    main()