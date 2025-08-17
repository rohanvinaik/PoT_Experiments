"""
Example: Project embeddings using UMAP/t-SNE/SOM.

This example demonstrates how to use different projection methods
to visualize high-dimensional embeddings in 2D/3D space.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

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
    compute_shepard_correlation
)
from pot.semantic.topography_visualizer import (
    plot_projection,
    compare_projections
)


def generate_synthetic_embeddings(n_samples=200, dim=128, n_clusters=3):
    """
    Generate synthetic embeddings with cluster structure.
    
    Args:
        n_samples: Number of samples
        dim: Embedding dimension
        n_clusters: Number of clusters
    
    Returns:
        Tuple of (embeddings, labels)
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    embeddings = []
    labels = []
    
    samples_per_cluster = n_samples // n_clusters
    
    for i in range(n_clusters):
        # Create cluster with different center
        center = torch.randn(dim) * 2
        cluster = torch.randn(samples_per_cluster, dim) + center
        embeddings.append(cluster)
        labels.extend([i] * samples_per_cluster)
    
    embeddings = torch.cat(embeddings)
    labels = np.array(labels)
    
    return embeddings, labels


def project_with_umap(embeddings, labels=None):
    """
    Project embeddings using UMAP.
    
    Args:
        embeddings: High-dimensional embeddings
        labels: Optional cluster labels for coloring
    
    Returns:
        2D projected data
    """
    print("\n" + "="*50)
    print("UMAP Projection")
    print("="*50)
    
    # Create projector with UMAP
    projector = TopographicalProjector(
        method='umap',
        n_neighbors=15,
        min_dist=0.1,
        n_components=2
    )
    
    # Project embeddings
    projected = projector.project_latents(embeddings)
    
    print(f"Original shape: {embeddings.shape}")
    print(f"Projected shape: {projected.shape}")
    
    # Visualize
    fig = plot_projection(
        projected,
        labels=labels,
        title="UMAP Projection",
        method='scatter'
    )
    
    # Compute quality metrics
    embeddings_np = prepare_latents_for_projection(embeddings)
    trust = compute_trustworthiness(embeddings_np, projected, n_neighbors=10)
    cont = compute_continuity(embeddings_np, projected, n_neighbors=10)
    
    print(f"Trustworthiness: {trust:.3f}")
    print(f"Continuity: {cont:.3f}")
    
    return projected, fig


def project_with_tsne(embeddings, labels=None):
    """
    Project embeddings using t-SNE.
    
    Args:
        embeddings: High-dimensional embeddings
        labels: Optional cluster labels for coloring
    
    Returns:
        2D projected data
    """
    print("\n" + "="*50)
    print("t-SNE Projection")
    print("="*50)
    
    # Create projector with t-SNE
    projector = TopographicalProjector(
        method='tsne',
        perplexity=30,
        learning_rate='auto',
        n_components=2
    )
    
    # Use subset for t-SNE (it's slow)
    subset_size = min(500, len(embeddings))
    subset_indices = np.random.choice(len(embeddings), subset_size, replace=False)
    embeddings_subset = embeddings[subset_indices]
    labels_subset = labels[subset_indices] if labels is not None else None
    
    # Project embeddings
    projected = projector.project_latents(embeddings_subset)
    
    print(f"Original shape: {embeddings_subset.shape}")
    print(f"Projected shape: {projected.shape}")
    
    # Visualize
    fig = plot_projection(
        projected,
        labels=labels_subset,
        title="t-SNE Projection",
        method='scatter'
    )
    
    # Compute quality metrics
    embeddings_np = prepare_latents_for_projection(embeddings_subset)
    trust = compute_trustworthiness(embeddings_np, projected, n_neighbors=10)
    cont = compute_continuity(embeddings_np, projected, n_neighbors=10)
    
    print(f"Trustworthiness: {trust:.3f}")
    print(f"Continuity: {cont:.3f}")
    
    return projected, fig


def project_with_som(embeddings, labels=None):
    """
    Project embeddings using Self-Organizing Map.
    
    Args:
        embeddings: High-dimensional embeddings
        labels: Optional cluster labels for coloring
    
    Returns:
        2D projected data
    """
    print("\n" + "="*50)
    print("SOM Projection")
    print("="*50)
    
    from pot.semantic.topography import SOMProjector
    
    # Create SOM projector
    projector = SOMProjector(
        grid_size=(20, 20),
        learning_rate=0.5,
        sigma=1.0
    )
    
    # Prepare data
    embeddings_np = prepare_latents_for_projection(embeddings)
    
    # Train SOM
    print("Training SOM...")
    projector.train(embeddings_np, num_iteration=100)
    
    # Project embeddings
    projected = projector.project(embeddings_np)
    
    print(f"Original shape: {embeddings.shape}")
    print(f"Projected shape: {projected.shape}")
    print(f"Quantization error: {projector.quantization_error(embeddings_np):.3f}")
    print(f"Topographic error: {projector.topographic_error(embeddings_np):.3f}")
    
    # Visualize
    fig = plot_projection(
        projected,
        labels=labels,
        title="SOM Projection",
        method='scatter'
    )
    
    # Compute quality metrics
    trust = compute_trustworthiness(embeddings_np, projected, n_neighbors=10)
    cont = compute_continuity(embeddings_np, projected, n_neighbors=10)
    
    print(f"Trustworthiness: {trust:.3f}")
    print(f"Continuity: {cont:.3f}")
    
    return projected, fig


def project_with_pca(embeddings, labels=None):
    """
    Project embeddings using PCA (baseline).
    
    Args:
        embeddings: High-dimensional embeddings
        labels: Optional cluster labels for coloring
    
    Returns:
        2D projected data
    """
    print("\n" + "="*50)
    print("PCA Projection (Baseline)")
    print("="*50)
    
    # Create projector with PCA
    projector = TopographicalProjector(
        method='pca',
        n_components=2,
        whiten=False
    )
    
    # Project embeddings
    projected = projector.project_latents(embeddings)
    
    print(f"Original shape: {embeddings.shape}")
    print(f"Projected shape: {projected.shape}")
    
    # Visualize
    fig = plot_projection(
        projected,
        labels=labels,
        title="PCA Projection",
        method='scatter'
    )
    
    # Compute quality metrics
    embeddings_np = prepare_latents_for_projection(embeddings)
    trust = compute_trustworthiness(embeddings_np, projected, n_neighbors=10)
    cont = compute_continuity(embeddings_np, projected, n_neighbors=10)
    
    print(f"Trustworthiness: {trust:.3f}")
    print(f"Continuity: {cont:.3f}")
    
    return projected, fig


def compare_all_methods(embeddings, labels=None):
    """
    Compare all projection methods side by side.
    
    Args:
        embeddings: High-dimensional embeddings
        labels: Optional cluster labels
    """
    print("\n" + "="*50)
    print("Comparing All Projection Methods")
    print("="*50)
    
    # Prepare data
    embeddings_np = prepare_latents_for_projection(embeddings)
    
    # Project with each method
    projections = {}
    
    # PCA
    pca_proj = TopographicalProjector('pca')
    projections['PCA'] = pca_proj.project_latents(embeddings)
    
    # UMAP
    umap_proj = TopographicalProjector('umap')
    projections['UMAP'] = umap_proj.project_latents(embeddings)
    
    # t-SNE (subset)
    subset_size = min(500, len(embeddings))
    subset_indices = np.random.choice(len(embeddings), subset_size, replace=False)
    tsne_proj = TopographicalProjector('tsne')
    tsne_result = tsne_proj.project_latents(embeddings[subset_indices])
    
    # For comparison, we'll use the subset for all
    projections_subset = {
        'PCA': projections['PCA'][subset_indices],
        'UMAP': projections['UMAP'][subset_indices],
        't-SNE': tsne_result
    }
    
    labels_subset = labels[subset_indices] if labels is not None else None
    embeddings_subset = embeddings_np[subset_indices]
    
    # Compare projections
    fig = compare_projections(
        list(projections_subset.values()),
        method_names=list(projections_subset.keys()),
        labels=labels_subset,
        title="Projection Method Comparison"
    )
    
    # Compute metrics for each
    print("\nQuality Metrics Comparison:")
    print("-" * 40)
    print(f"{'Method':<10} {'Trustworthiness':<15} {'Continuity':<15}")
    print("-" * 40)
    
    for method_name, projected in projections_subset.items():
        trust = compute_trustworthiness(embeddings_subset, projected, n_neighbors=10)
        cont = compute_continuity(embeddings_subset, projected, n_neighbors=10)
        print(f"{method_name:<10} {trust:<15.3f} {cont:<15.3f}")
    
    return fig


def main():
    """Main example execution."""
    print("="*60)
    print("Topographical Projection Example")
    print("="*60)
    
    # Generate synthetic embeddings
    print("\nGenerating synthetic embeddings...")
    embeddings, labels = generate_synthetic_embeddings(
        n_samples=300,
        dim=128,
        n_clusters=3
    )
    print(f"Generated {len(embeddings)} embeddings with {len(np.unique(labels))} clusters")
    
    # Test each projection method
    pca_proj, pca_fig = project_with_pca(embeddings, labels)
    umap_proj, umap_fig = project_with_umap(embeddings, labels)
    tsne_proj, tsne_fig = project_with_tsne(embeddings, labels)
    som_proj, som_fig = project_with_som(embeddings, labels)
    
    # Compare all methods
    comparison_fig = compare_all_methods(embeddings, labels)
    
    # Show all plots
    plt.show()
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()