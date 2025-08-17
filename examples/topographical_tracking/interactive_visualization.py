"""
Example: Create interactive topographical visualizations.

This example demonstrates how to create interactive visualizations
using plotly for exploring topographical projections.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from pot.semantic import (
    ConceptLibrary,
    SemanticMatcher,
    TopographicalProjector,
    create_topographical_semantic_system,
    visualize_semantic_landscape
)
from pot.semantic.topography_visualizer import (
    create_interactive_plot,
    create_interactive_embedding,
    create_som_dashboard,
    create_dashboard,
    create_evolution_animation_interactive
)
from pot.semantic.topography import SOMProjector
from pot.semantic.topography_utils import prepare_latents_for_projection


def create_sample_data():
    """
    Create sample data with meaningful structure.
    
    Returns:
        Tuple of (embeddings, labels, metadata)
    """
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create embeddings with clear structure
    n_samples = 500
    dim = 100
    
    # Create 4 distinct clusters
    clusters = []
    labels = []
    metadata = []
    
    cluster_names = ['Animals', 'Vehicles', 'Electronics', 'Food']
    cluster_items = {
        'Animals': ['dog', 'cat', 'bird', 'fish', 'lion', 'tiger', 'bear', 'wolf'],
        'Vehicles': ['car', 'truck', 'bike', 'plane', 'boat', 'train', 'bus', 'subway'],
        'Electronics': ['phone', 'laptop', 'tablet', 'TV', 'camera', 'speaker', 'headphones', 'watch'],
        'Food': ['apple', 'banana', 'pizza', 'burger', 'salad', 'pasta', 'sushi', 'cake']
    }
    
    samples_per_cluster = n_samples // len(cluster_names)
    
    for i, cluster_name in enumerate(cluster_names):
        # Create cluster center
        center = torch.randn(dim) * 3
        
        # Create samples around center
        for j in range(samples_per_cluster):
            # Add some variance within cluster
            sample = center + torch.randn(dim) * 0.8
            clusters.append(sample)
            labels.append(i)
            
            # Create metadata
            item_idx = j % len(cluster_items[cluster_name])
            item_name = cluster_items[cluster_name][item_idx]
            metadata.append({
                'cluster': cluster_name,
                'item': item_name,
                'cluster_id': i,
                'sample_id': len(clusters) - 1,
                'confidence': np.random.rand()
            })
    
    embeddings = torch.stack(clusters)
    labels = np.array(labels)
    
    return embeddings, labels, metadata


def basic_interactive_plot(embeddings, labels, metadata):
    """
    Create a basic interactive scatter plot.
    
    Args:
        embeddings: High-dimensional embeddings
        labels: Cluster labels
        metadata: Additional metadata for hover information
    """
    print("\n" + "="*50)
    print("Creating Basic Interactive Plot")
    print("="*50)
    
    # Project embeddings
    projector = TopographicalProjector('umap')
    projected = projector.project_latents(embeddings)
    
    # Prepare hover text
    hover_texts = []
    for meta in metadata:
        hover_text = (
            f"Cluster: {meta['cluster']}<br>"
            f"Item: {meta['item']}<br>"
            f"Confidence: {meta['confidence']:.2f}"
        )
        hover_texts.append(hover_text)
    
    # Create interactive plot
    fig = create_interactive_plot(
        projected,
        labels=labels,
        hover_texts=hover_texts,
        title="Interactive Embedding Visualization",
        color_scale='Viridis'
    )
    
    # Save to HTML
    output_path = "interactive_basic.html"
    fig.write_html(output_path)
    print(f"Saved to: {output_path}")
    
    return fig


def semantic_landscape_interactive(embeddings, labels, metadata):
    """
    Create an interactive semantic landscape.
    
    Args:
        embeddings: Embeddings
        labels: Labels
        metadata: Metadata
    """
    print("\n" + "="*50)
    print("Creating Interactive Semantic Landscape")
    print("="*50)
    
    # Create semantic system
    dim = embeddings.shape[1]
    library, matcher, projector = create_topographical_semantic_system(
        dim=dim,
        projection_method='umap'
    )
    
    # Add concepts to library
    cluster_names = ['Animals', 'Vehicles', 'Electronics', 'Food']
    for i, name in enumerate(cluster_names):
        cluster_mask = labels == i
        cluster_embeddings = embeddings[cluster_mask][:20]  # Use subset
        library.add_concept(name, cluster_embeddings)
    
    print(f"Added {len(library.concepts)} concepts to library")
    
    # Create interactive landscape
    fig = visualize_semantic_landscape(
        library,
        method='umap',
        interactive=True
    )
    
    if fig:
        output_path = "semantic_landscape.html"
        fig.write_html(output_path)
        print(f"Saved to: {output_path}")
    
    return fig


def som_interactive_dashboard(embeddings, labels):
    """
    Create an interactive SOM dashboard.
    
    Args:
        embeddings: Embeddings
        labels: Labels
    """
    print("\n" + "="*50)
    print("Creating Interactive SOM Dashboard")
    print("="*50)
    
    # Prepare data
    data_np = prepare_latents_for_projection(embeddings)
    
    # Train SOM
    som_projector = SOMProjector(grid_size=(15, 15))
    print("Training SOM...")
    som_projector.train(data_np[:300], num_iteration=100)  # Use subset for speed
    
    # Project data
    projected = som_projector.project(data_np[:300])
    
    # Get SOM components for visualization
    if hasattr(som_projector, 'som') and som_projector.som is not None:
        # Create dashboard
        try:
            fig = create_som_dashboard(
                som_projector.som,
                data_np[:300],
                labels[:300] if labels is not None else None,
                title="Interactive SOM Dashboard"
            )
            
            if fig:
                output_path = "som_dashboard.html"
                fig.write_html(output_path)
                print(f"Saved to: {output_path}")
                return fig
        except Exception as e:
            print(f"Could not create SOM dashboard: {e}")
    
    return None


def evolution_animation(embeddings, labels):
    """
    Create an animated evolution visualization.
    
    Args:
        embeddings: Embeddings
        labels: Labels
    """
    print("\n" + "="*50)
    print("Creating Evolution Animation")
    print("="*50)
    
    # Simulate evolution by gradually revealing clusters
    n_frames = 10
    n_samples = len(embeddings)
    samples_per_frame = n_samples // n_frames
    
    snapshots = []
    timestamps = []
    
    # Project all data first
    projector = TopographicalProjector('pca')  # Use PCA for speed
    
    for frame in range(n_frames):
        # Take progressively more samples
        n_current = (frame + 1) * samples_per_frame
        subset = embeddings[:n_current]
        
        # Project subset
        projected = projector.project_latents(subset)
        snapshots.append(projected)
        timestamps.append(float(frame))
        
        print(f"Frame {frame + 1}/{n_frames}: {n_current} samples")
    
    # Create animation
    try:
        fig = create_evolution_animation_interactive(
            snapshots,
            timestamps=timestamps,
            labels=[labels[:len(s)] for s in snapshots],
            title="Embedding Evolution Animation"
        )
        
        if fig:
            output_path = "evolution_animation.html"
            fig.write_html(output_path)
            print(f"Saved to: {output_path}")
            return fig
    except Exception as e:
        print(f"Could not create animation: {e}")
    
    return None


def comprehensive_dashboard(embeddings, labels, metadata):
    """
    Create a comprehensive interactive dashboard.
    
    Args:
        embeddings: Embeddings
        labels: Labels  
        metadata: Metadata
    """
    print("\n" + "="*50)
    print("Creating Comprehensive Dashboard")
    print("="*50)
    
    # Project with multiple methods
    methods = ['pca', 'umap']
    projections = {}
    
    for method in methods:
        print(f"Projecting with {method.upper()}...")
        projector = TopographicalProjector(method)
        
        # Use subset for UMAP if needed
        if method == 'umap':
            subset = embeddings[:300]
            projected = projector.project_latents(subset)
            projections[method] = projected
        else:
            projected = projector.project_latents(embeddings)
            projections[method] = projected[:300]  # Match subset size
    
    # Prepare data for dashboard
    data_dict = {
        'projections': projections,
        'labels': labels[:300],
        'metadata': metadata[:300]
    }
    
    # Create dashboard
    try:
        fig = create_dashboard(
            data_dict,
            title="Topographical Analysis Dashboard"
        )
        
        if fig:
            output_path = "comprehensive_dashboard.html"
            # Note: create_dashboard returns matplotlib figure
            # For a true interactive dashboard, you'd need a web framework
            print(f"Dashboard created (matplotlib version)")
            return fig
    except Exception as e:
        print(f"Could not create dashboard: {e}")
    
    return None


def create_3d_visualization(embeddings, labels, metadata):
    """
    Create a 3D interactive visualization.
    
    Args:
        embeddings: Embeddings
        labels: Labels
        metadata: Metadata
    """
    print("\n" + "="*50)
    print("Creating 3D Interactive Visualization")
    print("="*50)
    
    # Project to 3D
    projector = TopographicalProjector(
        method='pca',
        n_components=3
    )
    projected_3d = projector.project_latents(embeddings)
    
    # Create 3D plot using plotly
    try:
        import plotly.graph_objects as go
        
        # Prepare hover text
        hover_texts = [
            f"Cluster: {m['cluster']}<br>Item: {m['item']}"
            for m in metadata
        ]
        
        # Create 3D scatter
        fig = go.Figure(data=[go.Scatter3d(
            x=projected_3d[:, 0],
            y=projected_3d[:, 1],
            z=projected_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=labels,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Cluster"),
                opacity=0.8
            ),
            text=hover_texts,
            hovertemplate='%{text}<br>X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}'
        )])
        
        fig.update_layout(
            title="3D Embedding Visualization",
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3"
            ),
            width=900,
            height=700
        )
        
        output_path = "visualization_3d.html"
        fig.write_html(output_path)
        print(f"Saved to: {output_path}")
        
        return fig
        
    except ImportError:
        print("Plotly not installed. Cannot create 3D visualization.")
        return None


def main():
    """Main example execution."""
    print("="*60)
    print("Interactive Topographical Visualization Example")
    print("="*60)
    
    # Create sample data
    print("\nCreating sample data...")
    embeddings, labels, metadata = create_sample_data()
    print(f"Created {len(embeddings)} embeddings with {len(np.unique(labels))} clusters")
    
    # Create various interactive visualizations
    
    # 1. Basic interactive plot
    basic_fig = basic_interactive_plot(embeddings, labels, metadata)
    
    # 2. Semantic landscape
    landscape_fig = semantic_landscape_interactive(embeddings, labels, metadata)
    
    # 3. SOM dashboard
    som_fig = som_interactive_dashboard(embeddings, labels)
    
    # 4. Evolution animation
    evolution_fig = evolution_animation(embeddings, labels)
    
    # 5. 3D visualization
    viz_3d = create_3d_visualization(embeddings, labels, metadata)
    
    # 6. Comprehensive dashboard
    dashboard_fig = comprehensive_dashboard(embeddings, labels, metadata)
    
    print("\n" + "="*60)
    print("Interactive visualizations created!")
    print("="*60)
    print("\nGenerated files:")
    print("  - interactive_basic.html: Basic interactive scatter plot")
    print("  - semantic_landscape.html: Interactive concept landscape")
    print("  - som_dashboard.html: Self-Organizing Map dashboard")
    print("  - evolution_animation.html: Animated evolution")
    print("  - visualization_3d.html: 3D interactive plot")
    print("\nOpen these HTML files in a web browser to interact with them.")
    
    # Show matplotlib figures if any
    plt.show()


if __name__ == "__main__":
    main()