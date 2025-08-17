"""
Visualization tools for topographical analysis.
Provides interactive and static visualizations for manifold embeddings,
topological structures, and concept space exploration.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
import matplotlib.figure
import seaborn as sns
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

logger = logging.getLogger(__name__)

# Handle optional plotly dependency
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.info("Plotly not available. Interactive visualizations will be limited.")


def plot_projection(projected_data: np.ndarray, 
                   labels: Optional[np.ndarray] = None,
                   method: str = 'scatter',
                   title: Optional[str] = None,
                   figsize: Tuple[int, int] = (10, 8),
                   alpha: float = 0.7,
                   point_size: int = 50,
                   cmap: str = 'viridis',
                   save_path: Optional[str] = None) -> matplotlib.figure.Figure:
    """
    Create 2D/3D scatter plot of projected data.
    
    Args:
        projected_data: Projected points (n_samples, 2 or 3)
        labels: Optional labels for coloring
        method: Plot method ('scatter', 'hexbin', 'contour')
        title: Plot title
        figsize: Figure size
        alpha: Point transparency
        point_size: Size of scatter points
        cmap: Colormap
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    n_dims = projected_data.shape[1] if len(projected_data.shape) > 1 else 1
    
    if n_dims == 2:
        ax = fig.add_subplot(111)
        
        if method == 'scatter':
            if labels is not None:
                scatter = ax.scatter(projected_data[:, 0], projected_data[:, 1],
                                   c=labels, s=point_size, alpha=alpha,
                                   cmap=cmap, edgecolors='black', linewidth=0.5)
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(projected_data[:, 0], projected_data[:, 1],
                          s=point_size, alpha=alpha, edgecolors='black', linewidth=0.5)
                          
        elif method == 'hexbin':
            hexbin = ax.hexbin(projected_data[:, 0], projected_data[:, 1],
                              gridsize=30, cmap=cmap, alpha=alpha)
            plt.colorbar(hexbin, ax=ax)
            
        elif method == 'contour':
            # Create density contour
            from scipy.stats import gaussian_kde
            xy = projected_data.T
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = projected_data[idx, 0], projected_data[idx, 1], z[idx]
            
            contour = ax.tricontourf(x, y, z, levels=20, cmap=cmap, alpha=alpha)
            plt.colorbar(contour, ax=ax)
            
            if labels is not None:
                scatter = ax.scatter(projected_data[:, 0], projected_data[:, 1],
                                   c=labels, s=10, alpha=0.5, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        
    elif n_dims == 3:
        ax = fig.add_subplot(111, projection='3d')
        
        if labels is not None:
            scatter = ax.scatter(projected_data[:, 0], projected_data[:, 1], projected_data[:, 2],
                               c=labels, s=point_size, alpha=alpha, cmap=cmap)
            # Create a mappable for colorbar
            mappable = plt.cm.ScalarMappable(cmap=cmap)
            mappable.set_array(labels)
            plt.colorbar(mappable, ax=ax)
        else:
            ax.scatter(projected_data[:, 0], projected_data[:, 1], projected_data[:, 2],
                      s=point_size, alpha=alpha)
        
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
    
    else:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f"Cannot visualize {n_dims}D data directly", 
                ha='center', va='center', transform=ax.transAxes)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Projected Data ({method.capitalize()} plot)")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_density_map(projected_data: np.ndarray,
                    bandwidth: float = 0.1,
                    resolution: int = 100,
                    title: Optional[str] = None,
                    figsize: Tuple[int, int] = (10, 8),
                    cmap: str = 'YlOrRd',
                    show_points: bool = True,
                    save_path: Optional[str] = None) -> matplotlib.figure.Figure:
    """
    Create density heatmap of projected space.
    
    Args:
        projected_data: Projected points (n_samples, 2)
        bandwidth: KDE bandwidth parameter
        resolution: Grid resolution for density estimation
        title: Plot title
        figsize: Figure size
        cmap: Colormap for density
        show_points: Whether to overlay actual points
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    if projected_data.shape[1] != 2:
        raise ValueError("Density map requires 2D data")
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grid for density estimation
    x_min, x_max = projected_data[:, 0].min(), projected_data[:, 0].max()
    y_min, y_max = projected_data[:, 1].min(), projected_data[:, 1].max()
    
    # Add margin
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min - x_margin, x_max + x_margin, resolution),
        np.linspace(y_min - y_margin, y_max + y_margin, resolution)
    )
    
    # Compute KDE
    from sklearn.neighbors import KernelDensity
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(projected_data)
    
    # Evaluate density on grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density).reshape(xx.shape)
    
    # Plot density
    im = ax.contourf(xx, yy, density, levels=20, cmap=cmap, alpha=0.9)
    plt.colorbar(im, ax=ax, label="Density")
    
    # Add contour lines
    ax.contour(xx, yy, density, levels=10, colors='black', alpha=0.2, linewidths=0.5)
    
    # Overlay points if requested
    if show_points:
        ax.scatter(projected_data[:, 0], projected_data[:, 1],
                  c='black', s=5, alpha=0.3, zorder=5)
    
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Density Map (bandwidth={bandwidth:.3f})")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_som_grid_extended(som_model, data: np.ndarray,
                          plot_type: str = 'umatrix',
                          title: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 8),
                          cmap: str = 'RdBu_r',
                          save_path: Optional[str] = None) -> matplotlib.figure.Figure:
    """
    Visualize SOM grid with various plot types.
    
    Args:
        som_model: Trained SOM model (SOMProjector instance)
        data: Input data used for training
        plot_type: Type of visualization ('umatrix', 'hits', 'components', 'clusters')
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if plot_type == 'umatrix':
        # U-matrix visualization
        u_matrix = som_model.get_u_matrix()
        im = ax.imshow(u_matrix, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, label="Distance")
        
        if not title:
            title = "SOM U-Matrix"
            
    elif plot_type == 'hits':
        # Hit map visualization
        hit_map = som_model.get_hit_map(data)
        im = ax.imshow(hit_map, cmap='YlOrRd', aspect='auto')
        plt.colorbar(im, ax=ax, label="Hits")
        
        # Add text annotations for counts
        for i in range(hit_map.shape[0]):
            for j in range(hit_map.shape[1]):
                if hit_map[i, j] > 0:
                    ax.text(j, i, str(int(hit_map[i, j])),
                           ha='center', va='center',
                           color='white' if hit_map[i, j] > hit_map.max()/2 else 'black',
                           fontsize=8)
        
        if not title:
            title = "SOM Hit Map"
            
    elif plot_type == 'components':
        # Component planes - show first component
        component_planes = som_model.get_component_planes()
        if len(component_planes) > 0:
            im = ax.imshow(component_planes[0], cmap='viridis', aspect='auto')
            plt.colorbar(im, ax=ax, label="Component 0")
        
        if not title:
            title = "SOM Component Plane (First Component)"
            
    elif plot_type == 'clusters':
        # Cluster visualization using K-means on SOM weights
        from sklearn.cluster import KMeans
        
        weights = som_model.weights.reshape(-1, som_model.weights.shape[-1])
        n_clusters = min(5, len(weights) // 4)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(weights)
        cluster_grid = cluster_labels.reshape(som_model.x_dim, som_model.y_dim)
        
        im = ax.imshow(cluster_grid, cmap='tab10', aspect='auto')
        plt.colorbar(im, ax=ax, label="Cluster")
        
        if not title:
            title = f"SOM Clusters (K={n_clusters})"
    
    else:
        ax.text(0.5, 0.5, f"Unknown plot type: {plot_type}",
               ha='center', va='center', transform=ax.transAxes)
        if not title:
            title = "SOM Grid"
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, som_model.y_dim, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, som_model.x_dim, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_interactive_plot(projected_data: np.ndarray,
                          hover_data: Optional[Dict] = None,
                          color_by: Optional[Union[str, np.ndarray]] = None,
                          title: Optional[str] = None,
                          point_size: int = 8,
                          save_path: Optional[str] = None) -> Optional[Any]:
    """
    Create interactive Plotly visualization.
    
    Args:
        projected_data: Projected points (n_samples, 2 or 3)
        hover_data: Dictionary of additional data to show on hover
        color_by: Column name or array for coloring points
        title: Plot title
        point_size: Size of points
        save_path: Path to save HTML
        
    Returns:
        Plotly figure or None if not available
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available. Cannot create interactive plot.")
        return None
    
    n_dims = projected_data.shape[1]
    
    # Prepare hover text
    hover_text = []
    for i in range(len(projected_data)):
        text = f"Index: {i}"
        if hover_data:
            for key, values in hover_data.items():
                if i < len(values):
                    if isinstance(values[i], (int, float)):
                        text += f"<br>{key}: {values[i]:.3f}"
                    else:
                        text += f"<br>{key}: {values[i]}"
        hover_text.append(text)
    
    # Determine colors
    if color_by is not None:
        if isinstance(color_by, str) and hover_data and color_by in hover_data:
            colors = hover_data[color_by]
        elif isinstance(color_by, (list, np.ndarray)):
            colors = color_by
        else:
            colors = np.arange(len(projected_data))
    else:
        colors = np.arange(len(projected_data))
    
    if n_dims == 2:
        fig = go.Figure(data=[
            go.Scatter(
                x=projected_data[:, 0],
                y=projected_data[:, 1],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=colors,
                    colorscale='Viridis',
                    showscale=True,
                    line=dict(width=0.5, color='black')
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=title or "Interactive 2D Projection",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            hovermode='closest',
            width=800,
            height=600
        )
        
    elif n_dims == 3:
        fig = go.Figure(data=[
            go.Scatter3d(
                x=projected_data[:, 0],
                y=projected_data[:, 1],
                z=projected_data[:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=colors,
                    colorscale='Viridis',
                    showscale=True,
                    line=dict(width=0.5, color='black')
                ),
                text=hover_text,
                hovertemplate='%{text}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=title or "Interactive 3D Projection",
            scene=dict(
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                zaxis_title="Component 3"
            ),
            hovermode='closest',
            width=800,
            height=600
        )
    else:
        logger.error(f"Cannot create interactive plot for {n_dims}D data")
        return None
    
    # Add customization
    fig.update_layout(
        template='plotly_white',
        font=dict(size=12),
        showlegend=False
    )
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved interactive plot to {save_path}")
    
    return fig


def create_evolution_animation_interactive(snapshots: List[np.ndarray],
                                         timestamps: List[float],
                                         title: Optional[str] = None,
                                         save_path: Optional[str] = None) -> Optional[Any]:
    """
    Animate topographical evolution over time using Plotly.
    
    Args:
        snapshots: List of projected snapshots
        timestamps: List of timestamps
        title: Animation title
        save_path: Path to save HTML
        
    Returns:
        Plotly figure with animation or None
    """
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available. Cannot create animation.")
        return None
    
    if len(snapshots) < 2:
        logger.warning("Need at least 2 snapshots for animation")
        return None
    
    # Prepare frames
    frames = []
    for i, (snapshot, timestamp) in enumerate(zip(snapshots, timestamps)):
        frame_data = go.Scatter(
            x=snapshot[:, 0],
            y=snapshot[:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=np.arange(len(snapshot)),
                colorscale='Viridis',
                showscale=True
            ),
            name=f"t={timestamp:.2f}"
        )
        
        frames.append(go.Frame(
            data=[frame_data],
            name=str(i),
            traces=[0]
        ))
    
    # Initial frame
    fig = go.Figure(
        data=[frames[0].data[0]],
        frames=frames
    )
    
    # Add slider and buttons
    fig.update_layout(
        title=title or "Topographical Evolution Animation",
        xaxis_title="Component 1",
        yaxis_title="Component 2",
        hovermode='closest',
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play",
                         method="animate",
                         args=[None, {"frame": {"duration": 100, "redraw": True},
                                     "fromcurrent": True,
                                     "transition": {"duration": 50}}]),
                    dict(label="Pause",
                         method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate",
                                       "transition": {"duration": 0}}])
                ],
                x=0.1,
                y=1.15
            )
        ],
        sliders=[dict(
            active=0,
            steps=[dict(
                args=[[str(i)],
                     {"frame": {"duration": 100, "redraw": True},
                      "mode": "immediate",
                      "transition": {"duration": 50}}],
                label=f"t={timestamps[i]:.1f}",
                method="animate"
            ) for i in range(len(frames))],
            x=0.1,
            len=0.9,
            y=0,
            xanchor="left",
            yanchor="top",
            currentvalue=dict(
                font=dict(size=16),
                prefix="Time: ",
                visible=True,
                xanchor="right"
            )
        )]
    )
    
    # Set axis ranges
    all_points = np.vstack(snapshots)
    fig.update_xaxes(range=[all_points[:, 0].min() - 1, all_points[:, 0].max() + 1])
    fig.update_yaxes(range=[all_points[:, 1].min() - 1, all_points[:, 1].max() + 1])
    
    if save_path:
        fig.write_html(save_path)
        logger.info(f"Saved animation to {save_path}")
    
    return fig


def compare_projections(data: Union[torch.Tensor, np.ndarray],
                       methods: List[str] = ['umap', 'tsne', 'som'],
                       title: Optional[str] = None,
                       figsize: Tuple[int, int] = (15, 5),
                       save_path: Optional[str] = None) -> matplotlib.figure.Figure:
    """
    Side-by-side comparison of different projection methods.
    
    Args:
        data: Input data to project
        methods: List of projection methods to compare
        title: Overall title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure with comparison
    """
    from ..topography import TopographicalProjector
    
    # Convert data if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()
    
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    
    projections = {}
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        try:
            # Create projector
            projector = TopographicalProjector(method=method)
            
            # Project data
            projected = projector.project_latents(data)
            projections[method] = projected
            
            # Plot
            scatter = ax.scatter(projected[:, 0], projected[:, 1],
                               c=np.arange(len(projected)), 
                               cmap='viridis', s=30, alpha=0.7,
                               edgecolors='black', linewidth=0.5)
            
            ax.set_title(f"{method.upper()}")
            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")
            ax.grid(True, alpha=0.3)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Error with {method}:\n{str(e)[:50]}",
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{method.upper()} (Failed)")
    
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle("Projection Method Comparison")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_embedding(embedding: np.ndarray,
                  labels: Optional[np.ndarray] = None,
                  colors: Optional[np.ndarray] = None,
                  title: str = "Topographical Embedding",
                  figsize: Tuple[int, int] = (10, 8),
                  point_size: int = 50,
                  alpha: float = 0.7,
                  cmap: str = "viridis",
                  show_legend: bool = True,
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot 2D or 3D embedding with optional labels/colors.
    
    Args:
        embedding: Embedded points (n_samples, 2 or 3)
        labels: Optional labels for points
        colors: Optional colors for points
        title: Plot title
        figsize: Figure size
        point_size: Size of scatter points
        alpha: Transparency
        cmap: Colormap
        show_legend: Whether to show legend
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    n_dims = embedding.shape[1]
    
    if n_dims == 2:
        ax = fig.add_subplot(111)
        
        if colors is not None:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                               c=colors, s=point_size, alpha=alpha,
                               cmap=cmap, edgecolors='black', linewidth=0.5)
            if isinstance(colors[0], (int, float)):
                plt.colorbar(scatter, ax=ax)
        elif labels is not None:
            unique_labels = np.unique(labels)
            colors_list = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                          c=[colors_list[i]], s=point_size, alpha=alpha,
                          label=str(label), edgecolors='black', linewidth=0.5)
            
            if show_legend and len(unique_labels) <= 20:
                ax.legend(loc='best', framealpha=0.9)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1],
                      s=point_size, alpha=alpha, edgecolors='black', linewidth=0.5)
        
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        
    elif n_dims == 3:
        ax = fig.add_subplot(111, projection='3d')
        
        if colors is not None:
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                               c=colors, s=point_size, alpha=alpha, cmap=cmap)
            if isinstance(colors[0], (int, float)):
                plt.colorbar(scatter, ax=ax)
        elif labels is not None:
            unique_labels = np.unique(labels)
            colors_list = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                          c=[colors_list[i]], s=point_size, alpha=alpha, label=str(label))
            
            if show_legend and len(unique_labels) <= 20:
                ax.legend(loc='best', framealpha=0.9)
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2],
                      s=point_size, alpha=alpha)
        
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
    
    else:
        raise ValueError(f"Can only plot 2D or 3D embeddings, got {n_dims}D")
    
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_neighborhood_graph(embedding: np.ndarray,
                          adjacency: np.ndarray,
                          labels: Optional[np.ndarray] = None,
                          title: str = "Neighborhood Graph",
                          figsize: Tuple[int, int] = (10, 8),
                          node_size: int = 50,
                          edge_alpha: float = 0.3,
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot neighborhood graph structure.
    
    Args:
        embedding: 2D embedded points
        adjacency: Adjacency matrix
        labels: Optional node labels
        title: Plot title
        figsize: Figure size
        node_size: Size of nodes
        edge_alpha: Edge transparency
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges
    edges = []
    for i in range(adjacency.shape[0]):
        for j in range(i+1, adjacency.shape[1]):
            if adjacency[i, j] > 0:
                edges.append([embedding[i], embedding[j]])
    
    if edges:
        lc = LineCollection(edges, alpha=edge_alpha, colors='gray', linewidths=0.5)
        ax.add_collection(lc)
    
    # Draw nodes
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                      c=[colors[i]], s=node_size, label=str(label),
                      edgecolors='black', linewidth=0.5, zorder=5)
        
        if len(unique_labels) <= 10:
            ax.legend(loc='best')
    else:
        ax.scatter(embedding[:, 0], embedding[:, 1], s=node_size,
                  edgecolors='black', linewidth=0.5, zorder=5)
    
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_quality_heatmap(quality_scores: np.ndarray,
                        embedding: np.ndarray,
                        title: str = "Embedding Quality",
                        figsize: Tuple[int, int] = (10, 8),
                        resolution: int = 50,
                        cmap: str = "RdYlGn",
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot quality scores as heatmap overlay.
    
    Args:
        quality_scores: Per-point quality scores
        embedding: 2D embedded points
        title: Plot title
        figsize: Figure size
        resolution: Grid resolution
        cmap: Colormap
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    from scipy.interpolate import griddata
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grid
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    
    # Add margin
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    
    xi = np.linspace(x_min - margin_x, x_max + margin_x, resolution)
    yi = np.linspace(y_min - margin_y, y_max + margin_y, resolution)
    xi, yi = np.meshgrid(xi, yi)
    
    # Interpolate quality scores
    zi = griddata(embedding, quality_scores, (xi, yi), method='cubic', fill_value=np.nan)
    
    # Plot heatmap
    im = ax.contourf(xi, yi, zi, levels=20, cmap=cmap, alpha=0.8)
    plt.colorbar(im, ax=ax, label="Quality Score")
    
    # Overlay points
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1],
                        c=quality_scores, s=30, cmap=cmap,
                        edgecolors='black', linewidth=0.5, zorder=5)
    
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_persistence_diagram(persistence: List[np.ndarray],
                           title: str = "Persistence Diagram",
                           figsize: Tuple[int, int] = (10, 8),
                           save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot topological persistence diagram.
    
    Args:
        persistence: List of persistence diagrams by dimension
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, len(persistence), figsize=figsize)
    
    if len(persistence) == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for dim, (ax, diagram) in enumerate(zip(axes, persistence)):
        if len(diagram) > 0:
            # Plot persistence points
            ax.scatter(diagram[:, 0], diagram[:, 1],
                      c=colors[dim % len(colors)], s=50, alpha=0.7)
            
            # Plot diagonal
            max_val = max(diagram[:, 0].max(), diagram[:, 1].max()) if len(diagram) > 0 else 1
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
            
            # Plot infinite persistence
            inf_mask = np.isinf(diagram[:, 1])
            if np.any(inf_mask):
                ax.scatter(diagram[inf_mask, 0],
                          np.ones(inf_mask.sum()) * max_val * 1.1,
                          c=colors[dim % len(colors)], s=50, marker='^')
        
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
        ax.set_title(f"H{dim}")
        ax.set_aspect('equal')
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_som_grid(som_weights: np.ndarray,
                 grid_size: Tuple[int, int],
                 title: str = "SOM Grid",
                 figsize: Tuple[int, int] = (10, 8),
                 save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot Self-Organizing Map grid.
    
    Args:
        som_weights: SOM weight vectors
        grid_size: Grid dimensions (x, y)
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x_size, y_size = grid_size
    
    # Create hexagonal or rectangular grid
    for i in range(x_size):
        for j in range(y_size):
            idx = i * y_size + j
            if idx < len(som_weights):
                # Position in grid
                if j % 2 == 0:
                    x = i
                else:
                    x = i + 0.5
                y = j * np.sqrt(3) / 2
                
                # Color based on weight vector
                color = plt.cm.viridis(som_weights[idx, :3].mean())
                
                # Draw hexagon
                hexagon = Polygon([
                    (x - 0.5, y),
                    (x - 0.25, y + 0.433),
                    (x + 0.25, y + 0.433),
                    (x + 0.5, y),
                    (x + 0.25, y - 0.433),
                    (x - 0.25, y - 0.433)
                ], facecolor=color, edgecolor='black', linewidth=0.5)
                
                ax.add_patch(hexagon)
    
    ax.set_xlim(-1, x_size)
    ax.set_ylim(-1, y_size * np.sqrt(3) / 2)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_concept_trajectory(embeddings_over_time: List[np.ndarray],
                          concept_indices: List[int],
                          time_labels: Optional[List[str]] = None,
                          title: str = "Concept Trajectory",
                          figsize: Tuple[int, int] = (12, 8),
                          save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot trajectory of concepts over time.
    
    Args:
        embeddings_over_time: List of embeddings at different time points
        concept_indices: Indices of concepts to track
        time_labels: Labels for time points
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_timepoints = len(embeddings_over_time)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(concept_indices)))
    
    for idx, concept_idx in enumerate(concept_indices):
        trajectory = []
        
        for t, embedding in enumerate(embeddings_over_time):
            if concept_idx < len(embedding):
                trajectory.append(embedding[concept_idx])
        
        if trajectory:
            trajectory = np.array(trajectory)
            
            # Plot trajectory
            ax.plot(trajectory[:, 0], trajectory[:, 1],
                   'o-', color=colors[idx], alpha=0.7,
                   label=f"Concept {concept_idx}")
            
            # Mark start and end
            ax.scatter(trajectory[0, 0], trajectory[0, 1],
                      s=100, color=colors[idx], marker='s', zorder=5)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1],
                      s=100, color=colors[idx], marker='^', zorder=5)
            
            # Add time labels if provided
            if time_labels:
                for t, point in enumerate(trajectory):
                    if t % max(1, len(trajectory) // 5) == 0:  # Show every 5th label
                        ax.annotate(time_labels[t], point,
                                  fontsize=8, alpha=0.7)
    
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_interactive_embedding(embedding: np.ndarray,
                                labels: Optional[np.ndarray] = None,
                                hover_data: Optional[Dict[str, np.ndarray]] = None,
                                title: str = "Interactive Topographical Embedding",
                                save_path: Optional[str] = None) -> Any:
    """
    Create interactive embedding visualization using Plotly.
    
    Args:
        embedding: Embedded points
        labels: Optional labels
        hover_data: Additional data to show on hover
        title: Plot title
        save_path: Path to save HTML
        
    Returns:
        Plotly figure
    """
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        
        n_dims = embedding.shape[1]
        
        # Prepare hover text
        hover_text = []
        for i in range(len(embedding)):
            text = f"Index: {i}"
            if labels is not None:
                text += f"<br>Label: {labels[i]}"
            if hover_data:
                for key, values in hover_data.items():
                    if i < len(values):
                        text += f"<br>{key}: {values[i]:.3f}"
            hover_text.append(text)
        
        if n_dims == 2:
            if labels is not None:
                fig = go.Figure(data=[
                    go.Scatter(
                        x=embedding[labels == label, 0],
                        y=embedding[labels == label, 1],
                        mode='markers',
                        name=str(label),
                        text=[hover_text[i] for i in np.where(labels == label)[0]],
                        hovertemplate='%{text}<extra></extra>',
                        marker=dict(size=8)
                    )
                    for label in np.unique(labels)
                ])
            else:
                fig = go.Figure(data=[
                    go.Scatter(
                        x=embedding[:, 0],
                        y=embedding[:, 1],
                        mode='markers',
                        text=hover_text,
                        hovertemplate='%{text}<extra></extra>',
                        marker=dict(size=8, color=np.arange(len(embedding)),
                                  colorscale='Viridis')
                    )
                ])
            
            fig.update_layout(
                title=title,
                xaxis_title="Component 1",
                yaxis_title="Component 2",
                hovermode='closest'
            )
            
        elif n_dims == 3:
            if labels is not None:
                fig = go.Figure(data=[
                    go.Scatter3d(
                        x=embedding[labels == label, 0],
                        y=embedding[labels == label, 1],
                        z=embedding[labels == label, 2],
                        mode='markers',
                        name=str(label),
                        text=[hover_text[i] for i in np.where(labels == label)[0]],
                        hovertemplate='%{text}<extra></extra>',
                        marker=dict(size=5)
                    )
                    for label in np.unique(labels)
                ])
            else:
                fig = go.Figure(data=[
                    go.Scatter3d(
                        x=embedding[:, 0],
                        y=embedding[:, 1],
                        z=embedding[:, 2],
                        mode='markers',
                        text=hover_text,
                        hovertemplate='%{text}<extra></extra>',
                        marker=dict(size=5, color=np.arange(len(embedding)),
                                  colorscale='Viridis')
                    )
                ])
            
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title="Component 1",
                    yaxis_title="Component 2",
                    zaxis_title="Component 3"
                ),
                hovermode='closest'
            )
        
        else:
            raise ValueError(f"Can only create 2D or 3D interactive plots, got {n_dims}D")
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
        
    except ImportError:
        logger.warning("Plotly not installed. Cannot create interactive visualization.")
        return None


def plot_distance_preservation(original_distances: np.ndarray,
                              embedded_distances: np.ndarray,
                              title: str = "Distance Preservation",
                              figsize: Tuple[int, int] = (10, 8),
                              sample_size: Optional[int] = 5000,
                              save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distance preservation scatter plot (Shepard diagram).
    
    Args:
        original_distances: Distances in original space
        embedded_distances: Distances in embedded space
        title: Plot title
        figsize: Figure size
        sample_size: Number of distance pairs to plot
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Sample if too many points
    if sample_size and len(original_distances) > sample_size:
        indices = np.random.choice(len(original_distances), sample_size, replace=False)
        orig_sample = original_distances[indices]
        embed_sample = embedded_distances[indices]
    else:
        orig_sample = original_distances
        embed_sample = embedded_distances
    
    # Shepard diagram
    ax1.scatter(orig_sample, embed_sample, alpha=0.3, s=1)
    ax1.plot([0, orig_sample.max()], [0, orig_sample.max()], 'r--', alpha=0.5)
    ax1.set_xlabel("Original Distance")
    ax1.set_ylabel("Embedded Distance")
    ax1.set_title("Shepard Diagram")
    
    # Residuals
    residuals = embed_sample - orig_sample
    ax2.scatter(orig_sample, residuals, alpha=0.3, s=1)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.set_xlabel("Original Distance")
    ax2.set_ylabel("Residual")
    ax2.set_title("Residual Plot")
    
    # Add correlation
    from scipy.stats import spearmanr
    corr, _ = spearmanr(orig_sample, embed_sample)
    fig.suptitle(f"{title} (Spearman ρ = {corr:.3f})")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_manifold_density(embedding: np.ndarray,
                         title: str = "Manifold Density",
                         figsize: Tuple[int, int] = (10, 8),
                         bandwidth: float = 0.5,
                         resolution: int = 100,
                         cmap: str = "YlOrRd",
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot density estimation on manifold.
    
    Args:
        embedding: 2D embedded points
        title: Plot title
        figsize: Figure size
        bandwidth: KDE bandwidth
        resolution: Grid resolution
        cmap: Colormap
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    from sklearn.neighbors import KernelDensity
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Fit KDE
    kde = KernelDensity(bandwidth=bandwidth)
    kde.fit(embedding)
    
    # Create grid
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    
    margin_x = (x_max - x_min) * 0.1
    margin_y = (y_max - y_min) * 0.1
    
    xx, yy = np.meshgrid(
        np.linspace(x_min - margin_x, x_max + margin_x, resolution),
        np.linspace(y_min - margin_y, y_max + margin_y, resolution)
    )
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Compute density
    log_density = kde.score_samples(grid_points)
    density = np.exp(log_density).reshape(xx.shape)
    
    # Plot density
    im = ax.contourf(xx, yy, density, levels=20, cmap=cmap, alpha=0.8)
    plt.colorbar(im, ax=ax, label="Density")
    
    # Overlay points
    ax.scatter(embedding[:, 0], embedding[:, 1],
              c='black', s=10, alpha=0.5, zorder=5)
    
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_som_u_matrix(u_matrix: np.ndarray,
                     title: str = "SOM U-Matrix",
                     figsize: Tuple[int, int] = (10, 8),
                     cmap: str = "RdBu_r",
                     show_values: bool = False,
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot SOM U-matrix showing cluster boundaries.
    
    Args:
        u_matrix: U-matrix from SOM
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        show_values: Whether to show values in cells
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(u_matrix, cmap=cmap, aspect='auto')
    plt.colorbar(im, ax=ax, label="Average Distance")
    
    # Add grid lines
    ax.set_xticks(np.arange(u_matrix.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(u_matrix.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    # Show values if requested
    if show_values and u_matrix.size < 100:  # Only for small matrices
        for i in range(u_matrix.shape[0]):
            for j in range(u_matrix.shape[1]):
                text = ax.text(j, i, f'{u_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white",
                             fontsize=8)
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_som_component_planes(component_planes: np.ndarray,
                             feature_names: Optional[List[str]] = None,
                             title: str = "SOM Component Planes",
                             figsize: Tuple[int, int] = (15, 10),
                             cmap: str = "viridis",
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot SOM component planes for each feature.
    
    Args:
        component_planes: Component planes (n_features, x_dim, y_dim)
        feature_names: Names of features
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    n_features = component_planes.shape[0]
    
    # Determine grid layout
    n_cols = min(4, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_features == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(n_features):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col] if n_rows > 1 else axes[col]
        
        im = ax.imshow(component_planes[idx], cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax)
        
        if feature_names and idx < len(feature_names):
            ax.set_title(f"Feature {idx}: {feature_names[idx]}")
        else:
            ax.set_title(f"Feature {idx}")
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    
    # Hide unused subplots
    for idx in range(n_features, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_som_hit_map(hit_map: np.ndarray,
                    title: str = "SOM Hit Map",
                    figsize: Tuple[int, int] = (10, 8),
                    cmap: str = "YlOrRd",
                    show_counts: bool = True,
                    save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot SOM hit map showing data distribution.
    
    Args:
        hit_map: Hit map with counts
        title: Plot title
        figsize: Figure size
        cmap: Colormap
        show_counts: Whether to show count values
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(hit_map, cmap=cmap, aspect='auto', interpolation='nearest')
    plt.colorbar(im, ax=ax, label="Number of Hits")
    
    # Add grid
    ax.set_xticks(np.arange(hit_map.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(hit_map.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    # Show counts if requested
    if show_counts and hit_map.size < 100:
        for i in range(hit_map.shape[0]):
            for j in range(hit_map.shape[1]):
                count = int(hit_map[i, j])
                if count > 0:
                    color = "white" if count > hit_map.max() / 2 else "black"
                    ax.text(j, i, str(count), ha="center", va="center",
                           color=color, fontsize=8)
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_som_clusters(som_coords: np.ndarray,
                     labels: Optional[np.ndarray] = None,
                     grid_size: Tuple[int, int] = (10, 10),
                     title: str = "SOM Cluster Map",
                     figsize: Tuple[int, int] = (10, 8),
                     save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot data points on SOM grid with cluster labels.
    
    Args:
        som_coords: 2D coordinates from SOM
        labels: Cluster labels for coloring
        grid_size: SOM grid dimensions
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw grid
    for i in range(grid_size[0] + 1):
        ax.axvline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
    for j in range(grid_size[1] + 1):
        ax.axhline(j - 0.5, color='gray', linewidth=0.5, alpha=0.5)
    
    # Plot points
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(som_coords[mask, 1], som_coords[mask, 0],
                      c=[colors[idx]], s=50, alpha=0.7,
                      label=f"Cluster {label}", edgecolors='black', linewidth=0.5)
        
        if len(unique_labels) <= 10:
            ax.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    else:
        ax.scatter(som_coords[:, 1], som_coords[:, 0],
                  s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    ax.set_xlim(-1, grid_size[1])
    ax.set_ylim(-1, grid_size[0])
    ax.set_title(title)
    ax.set_xlabel("SOM X")
    ax.set_ylabel("SOM Y")
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_som_dashboard(som_projector,
                        data: np.ndarray,
                        labels: Optional[np.ndarray] = None,
                        feature_names: Optional[List[str]] = None,
                        save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive SOM visualization dashboard.
    
    Args:
        som_projector: Trained SOMProjector instance
        data: Input data used for training
        labels: Optional labels for data points
        feature_names: Optional feature names
        save_path: Path to save dashboard
        
    Returns:
        Dashboard figure
    """
    fig = plt.figure(figsize=(20, 15))
    
    # U-Matrix
    ax1 = plt.subplot(3, 3, 1)
    u_matrix = som_projector.get_u_matrix()
    im1 = ax1.imshow(u_matrix, cmap='RdBu_r', aspect='auto')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title("U-Matrix (Cluster Boundaries)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    
    # Hit Map
    ax2 = plt.subplot(3, 3, 2)
    hit_map = som_projector.get_hit_map(data)
    im2 = ax2.imshow(hit_map, cmap='YlOrRd', aspect='auto')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title("Hit Map (Data Distribution)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    
    # Data on Grid
    ax3 = plt.subplot(3, 3, 3)
    coords = som_projector.project_to_2d(data)
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for idx, label in enumerate(unique_labels):
            mask = labels == label
            ax3.scatter(coords[mask, 0], coords[mask, 1],
                       c=[colors[idx]], s=20, alpha=0.5, label=str(label))
        if len(unique_labels) <= 10:
            ax3.legend(loc='best', fontsize=8)
    else:
        ax3.scatter(coords[:, 0], coords[:, 1], s=20, alpha=0.5)
    ax3.set_title("Data Points on SOM Grid")
    ax3.set_xlabel("X")
    ax3.set_ylabel("Y")
    ax3.grid(True, alpha=0.3)
    
    # Component Planes (first 6)
    component_planes = som_projector.get_component_planes()
    n_components_to_show = min(6, component_planes.shape[0])
    
    for i in range(n_components_to_show):
        ax = plt.subplot(3, 3, 4 + i)
        im = ax.imshow(component_planes[i], cmap='viridis', aspect='auto')
        plt.colorbar(im, ax=ax)
        
        if feature_names and i < len(feature_names):
            ax.set_title(f"Component: {feature_names[i][:15]}")
        else:
            ax.set_title(f"Component {i}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
    
    # Fill remaining subplots with metrics
    if n_components_to_show < 6:
        ax_info = plt.subplot(3, 3, 4 + n_components_to_show)
        ax_info.axis('off')
        
        # Calculate metrics
        qe = som_projector.get_quantization_error(data)
        te = som_projector.get_topographic_error(data)
        
        info_text = f"SOM Metrics:\n\n"
        info_text += f"Grid Size: {som_projector.x_dim}x{som_projector.y_dim}\n"
        info_text += f"Input Dim: {som_projector.input_len}\n"
        info_text += f"Samples: {len(data)}\n\n"
        info_text += f"Quantization Error: {qe:.4f}\n"
        info_text += f"Topographic Error: {te:.4f}\n"
        
        ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                    fontsize=11, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle("SOM Analysis Dashboard", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_evolution_trajectory(evolution_tracker,
                            point_indices: Optional[List[int]] = None,
                            title: str = "Evolution Trajectory",
                            figsize: Tuple[int, int] = (12, 8),
                            show_timestamps: bool = True,
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot evolution trajectory over time.
    
    Args:
        evolution_tracker: TopographicalEvolutionTracker instance
        point_indices: Specific points to track (None = centroid)
        title: Plot title
        figsize: Figure size
        show_timestamps: Whether to show timestamp labels
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Get trajectory
    trajectory = evolution_tracker.compute_trajectory(point_indices)
    
    if len(trajectory) == 0:
        ax1.text(0.5, 0.5, "No trajectory data", ha='center', va='center')
        ax2.text(0.5, 0.5, "No trajectory data", ha='center', va='center')
        return fig
    
    # Plot trajectory path
    if point_indices is None:
        # Single centroid trajectory
        ax1.plot(trajectory[:, 0], trajectory[:, 1], 'o-', alpha=0.7, linewidth=2)
        
        # Mark start and end
        ax1.scatter(trajectory[0, 0], trajectory[0, 1], s=100, c='green', 
                   marker='s', label='Start', zorder=5)
        ax1.scatter(trajectory[-1, 0], trajectory[-1, 1], s=100, c='red', 
                   marker='^', label='End', zorder=5)
        
        # Add timestamps if requested
        if show_timestamps and len(evolution_tracker.timestamps) > 0:
            n_labels = min(5, len(trajectory))
            step = max(1, len(trajectory) // n_labels)
            for i in range(0, len(trajectory), step):
                if i < len(evolution_tracker.timestamps):
                    ax1.annotate(f"t={evolution_tracker.timestamps[i]:.1f}",
                               (trajectory[i, 0], trajectory[i, 1]),
                               fontsize=8, alpha=0.7)
    else:
        # Multiple point trajectories
        n_points = trajectory.shape[1] if len(trajectory.shape) > 2 else 1
        colors = plt.cm.rainbow(np.linspace(0, 1, n_points))
        
        for j in range(n_points):
            if len(trajectory.shape) > 2:
                traj = trajectory[:, j, :]
            else:
                traj = trajectory
            
            ax1.plot(traj[:, 0], traj[:, 1], 'o-', alpha=0.5, 
                    color=colors[j], label=f"Point {point_indices[j]}")
    
    ax1.set_title("Trajectory in Projected Space")
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot drift metrics over time
    if len(evolution_tracker.timestamps) > 1:
        timestamps = evolution_tracker.timestamps
        
        # Compute drift at each step
        drifts = []
        for i in range(1, len(timestamps)):
            metrics = evolution_tracker.compute_drift_metrics(
                window_start=max(0, i-10), window_end=i+1
            )
            drifts.append(metrics['centroid_shift'])
        
        ax2.plot(timestamps[1:], drifts, 'o-', alpha=0.7)
        ax2.set_title("Drift Over Time")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Centroid Shift")
        ax2.grid(True, alpha=0.3)
        
        # Mark regime changes
        regime_changes = evolution_tracker.detect_regime_changes()
        for idx in regime_changes:
            if idx < len(timestamps):
                ax2.axvline(timestamps[idx], color='red', linestyle='--', 
                          alpha=0.5, label='Regime Change' if idx == regime_changes[0] else '')
        
        if regime_changes:
            ax2.legend()
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_drift_metrics_dashboard(evolution_tracker,
                                title: str = "Drift Metrics Dashboard",
                                figsize: Tuple[int, int] = (15, 10),
                                save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comprehensive drift metrics dashboard.
    
    Args:
        evolution_tracker: TopographicalEvolutionTracker instance
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    
    # Get all drift metrics
    metrics = evolution_tracker.compute_drift_metrics()
    
    # Metric values bar chart
    ax1 = plt.subplot(2, 3, 1)
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    bars = ax1.bar(metric_names, metric_values)
    ax1.set_title("Current Drift Metrics")
    ax1.set_ylabel("Value")
    ax1.tick_params(axis='x', rotation=45)
    
    # Color bars by severity
    for bar, val in zip(bars, metric_values):
        if abs(val) > 1.0:
            bar.set_color('red')
        elif abs(val) > 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    # Stability score over time
    ax2 = plt.subplot(2, 3, 2)
    if len(evolution_tracker.timestamps) > 1:
        stability_scores = []
        for i in range(1, len(evolution_tracker.timestamps)):
            score = evolution_tracker.compute_stability_score(window=min(i, 10))
            stability_scores.append(score)
        
        ax2.plot(evolution_tracker.timestamps[1:], stability_scores, 'o-', alpha=0.7)
        ax2.set_title("Stability Score Over Time")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Stability (0-1)")
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0.5, color='red', linestyle='--', alpha=0.3)
    
    # Cluster evolution
    ax3 = plt.subplot(2, 3, 3)
    cluster_evolution = evolution_tracker.get_cluster_evolution(n_clusters=3)
    if cluster_evolution['cluster_sizes']:
        cluster_sizes = np.array(cluster_evolution['cluster_sizes'])
        times = cluster_evolution['timestamps'][:len(cluster_sizes)]
        
        for i in range(cluster_sizes.shape[1]):
            ax3.plot(times, cluster_sizes[:, i], 'o-', alpha=0.7, label=f"Cluster {i}")
        
        ax3.set_title("Cluster Size Evolution")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Size")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Path length accumulation
    ax4 = plt.subplot(2, 3, 4)
    if len(evolution_tracker.timestamps) > 1:
        path_lengths = []
        for i in range(2, len(evolution_tracker.timestamps) + 1):
            # Compute path length up to this point
            tracker_subset = evolution_tracker
            trajectory = tracker_subset.compute_trajectory()[:i]
            if len(trajectory) > 1:
                length = 0
                for j in range(1, len(trajectory)):
                    length += np.linalg.norm(trajectory[j] - trajectory[j-1])
                path_lengths.append(length)
            else:
                path_lengths.append(0)
        
        ax4.plot(evolution_tracker.timestamps[1:], path_lengths, 'o-', alpha=0.7)
        ax4.set_title("Cumulative Path Length")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Total Distance")
        ax4.grid(True, alpha=0.3)
    
    # Regime changes
    ax5 = plt.subplot(2, 3, 5)
    regime_changes = evolution_tracker.detect_regime_changes()
    
    if regime_changes and len(evolution_tracker.timestamps) > 0:
        # Plot timeline with regime changes marked
        ax5.scatter(evolution_tracker.timestamps, 
                   np.zeros(len(evolution_tracker.timestamps)), 
                   alpha=0.3, s=50)
        
        for idx in regime_changes:
            if idx < len(evolution_tracker.timestamps):
                ax5.scatter(evolution_tracker.timestamps[idx], 0, 
                          color='red', s=200, marker='x', linewidth=3)
                ax5.annotate(f"Change {idx}", 
                           (evolution_tracker.timestamps[idx], 0),
                           xytext=(0, 20), textcoords='offset points',
                           ha='center', fontsize=8)
        
        ax5.set_title("Regime Change Detection")
        ax5.set_xlabel("Time")
        ax5.set_ylim(-0.5, 0.5)
        ax5.set_yticks([])
    else:
        ax5.text(0.5, 0.5, "No regime changes detected", 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title("Regime Change Detection")
    
    # Summary statistics
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    summary_text = f"Evolution Summary\n\n"
    summary_text += f"Total Snapshots: {len(evolution_tracker.snapshots)}\n"
    summary_text += f"Time Range: {evolution_tracker.timestamps[0]:.1f} - {evolution_tracker.timestamps[-1]:.1f}\n" if evolution_tracker.timestamps else ""
    summary_text += f"Total Drift: {metrics['centroid_shift']:.3f}\n"
    summary_text += f"Avg Velocity: {metrics['velocity']:.3f}\n"
    summary_text += f"Stability Score: {evolution_tracker.compute_stability_score():.3f}\n"
    summary_text += f"Regime Changes: {len(regime_changes)}\n"
    
    outliers = evolution_tracker.identify_outlier_movements()
    summary_text += f"Outlier Movements: {sum(len(v) for v in outliers.values())}\n"
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_evolution_animation(evolution_tracker,
                             interval: int = 100,
                             save_path: Optional[str] = None):
    """
    Create animated visualization of evolution.
    
    Args:
        evolution_tracker: TopographicalEvolutionTracker instance
        interval: Animation interval in milliseconds
        save_path: Path to save animation (requires ffmpeg)
        
    Returns:
        Animation object or None
    """
    try:
        from matplotlib.animation import FuncAnimation
        import matplotlib.animation as animation
    except ImportError:
        logger.warning("Animation support not available")
        return None
    
    if len(evolution_tracker.projected_snapshots) < 2:
        logger.warning("Need at least 2 snapshots for animation")
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Initialize plots
    scatter = ax1.scatter([], [], alpha=0.6)
    trajectory_line, = ax1.plot([], [], 'r-', alpha=0.3, linewidth=1)
    time_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, 
                         verticalalignment='top')
    
    # Set axis limits
    all_points = np.vstack(evolution_tracker.projected_snapshots)
    ax1.set_xlim(all_points[:, 0].min() - 1, all_points[:, 0].max() + 1)
    ax1.set_ylim(all_points[:, 1].min() - 1, all_points[:, 1].max() + 1)
    ax1.set_xlabel("Component 1")
    ax1.set_ylabel("Component 2")
    ax1.set_title("Evolution Animation")
    ax1.grid(True, alpha=0.3)
    
    # Drift plot
    drift_line, = ax2.plot([], [], 'b-', alpha=0.7)
    ax2.set_xlim(evolution_tracker.timestamps[0], evolution_tracker.timestamps[-1])
    ax2.set_ylim(0, 2)  # Will be adjusted
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Drift")
    ax2.set_title("Cumulative Drift")
    ax2.grid(True, alpha=0.3)
    
    # Animation data
    trajectory_history = []
    drift_history = []
    
    def init():
        scatter.set_offsets(np.empty((0, 2)))
        trajectory_line.set_data([], [])
        drift_line.set_data([], [])
        time_text.set_text('')
        return scatter, trajectory_line, drift_line, time_text
    
    def animate(frame):
        # Update scatter plot
        current_points = evolution_tracker.projected_snapshots[frame]
        scatter.set_offsets(current_points)
        
        # Update trajectory
        centroid = np.mean(current_points, axis=0)
        trajectory_history.append(centroid)
        if len(trajectory_history) > 1:
            traj = np.array(trajectory_history)
            trajectory_line.set_data(traj[:, 0], traj[:, 1])
        
        # Update drift
        if frame > 0:
            metrics = evolution_tracker.compute_drift_metrics(
                window_start=0, window_end=frame+1
            )
            drift_history.append(metrics['centroid_shift'])
            times = evolution_tracker.timestamps[:len(drift_history)]
            drift_line.set_data(times, drift_history)
            
            # Adjust y-axis if needed
            if drift_history:
                max_drift = max(drift_history)
                ax2.set_ylim(0, max_drift * 1.2)
        
        # Update time text
        if frame < len(evolution_tracker.timestamps):
            time_text.set_text(f"Time: {evolution_tracker.timestamps[frame]:.2f}")
        
        return scatter, trajectory_line, drift_line, time_text
    
    anim = FuncAnimation(fig, animate, init_func=init,
                        frames=len(evolution_tracker.projected_snapshots),
                        interval=interval, blit=True, repeat=True)
    
    if save_path:
        try:
            if save_path.endswith('.gif'):
                writer = animation.PillowWriter(fps=1000/interval)
            else:
                writer = animation.FFMpegWriter(fps=1000/interval)
            anim.save(save_path, writer=writer)
            logger.info(f"Saved animation to {save_path}")
        except:
            logger.warning("Could not save animation. Ensure ffmpeg is installed.")
    
    return anim


def plot_cluster_evolution_heatmap(evolution_tracker,
                                  n_clusters: int = 5,
                                  title: str = "Cluster Evolution Heatmap",
                                  figsize: Tuple[int, int] = (12, 6),
                                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot heatmap of cluster evolution over time.
    
    Args:
        evolution_tracker: TopographicalEvolutionTracker instance
        n_clusters: Number of clusters
        title: Plot title
        figsize: Figure size
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Get cluster evolution
    evolution = evolution_tracker.get_cluster_evolution(n_clusters)
    
    if evolution['cluster_sizes']:
        # Cluster sizes heatmap
        sizes = np.array(evolution['cluster_sizes']).T
        im1 = ax1.imshow(sizes, aspect='auto', cmap='YlOrRd', interpolation='nearest')
        plt.colorbar(im1, ax=ax1, label="Cluster Size")
        
        ax1.set_xlabel("Time Index")
        ax1.set_ylabel("Cluster ID")
        ax1.set_title("Cluster Size Evolution")
        
        # Add time labels
        if len(evolution['timestamps']) > 0:
            n_ticks = min(10, len(evolution['timestamps']))
            tick_indices = np.linspace(0, len(evolution['timestamps'])-1, n_ticks, dtype=int)
            ax1.set_xticks(tick_indices)
            ax1.set_xticklabels([f"{evolution['timestamps'][i]:.1f}" for i in tick_indices], 
                               rotation=45)
        
        # Cluster stability
        if evolution['cluster_stability']:
            ax2.plot(evolution['timestamps'][:len(evolution['cluster_stability'])],
                    evolution['cluster_stability'], 'o-', alpha=0.7)
            ax2.set_xlabel("Time")
            ax2.set_ylabel("Stability Score")
            ax2.set_title("Cluster Stability Over Time")
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1.1)
    else:
        ax1.text(0.5, 0.5, "No cluster data available", 
                ha='center', va='center', transform=ax1.transAxes)
        ax2.text(0.5, 0.5, "No stability data available", 
                ha='center', va='center', transform=ax2.transAxes)
    
    fig.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_dashboard(analyzer,
                    labels: Optional[np.ndarray] = None,
                    save_path: Optional[str] = None):
    """
    Create comprehensive visualization dashboard.
    
    Args:
        analyzer: Fitted TopographicalAnalyzer
        labels: Optional labels for points
        save_path: Path to save dashboard
        
    Returns:
        Dashboard figure or application
    """
    try:
        import streamlit as st
        
        # This would be a Streamlit app
        # For now, create a static matplotlib dashboard
        
        fig = plt.figure(figsize=(20, 12))
        
        # Main embedding
        ax1 = plt.subplot(2, 3, 1)
        if analyzer.embedding:
            embedding = analyzer.embedding.embedded_vectors
            if labels is not None:
                unique_labels = np.unique(labels)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    ax1.scatter(embedding[mask, 0], embedding[mask, 1],
                              c=[colors[i]], label=str(label), alpha=0.7)
                ax1.legend()
            else:
                ax1.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7)
        ax1.set_title("Topographical Embedding")
        ax1.set_xlabel("Component 1")
        ax1.set_ylabel("Component 2")
        
        # Quality metrics
        ax2 = plt.subplot(2, 3, 2)
        if analyzer.embedding and analyzer.embedding.quality_metrics:
            metrics = analyzer.embedding.quality_metrics
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            bars = ax2.bar(range(len(metric_names)), metric_values)
            ax2.set_xticks(range(len(metric_names)))
            ax2.set_xticklabels(metric_names, rotation=45, ha='right')
            ax2.set_title("Quality Metrics")
            ax2.set_ylabel("Score")
            
            # Color bars by value
            for bar, val in zip(bars, metric_values):
                if val > 0.8:
                    bar.set_color('green')
                elif val > 0.6:
                    bar.set_color('yellow')
                else:
                    bar.set_color('red')
        
        # Neighborhood graph
        ax3 = plt.subplot(2, 3, 3)
        if analyzer.structure and analyzer.structure.adjacency_matrix is not None:
            adj = analyzer.structure.adjacency_matrix
            edges = []
            for i in range(adj.shape[0]):
                for j in range(i+1, adj.shape[1]):
                    if adj[i, j] > 0:
                        edges.append([embedding[i], embedding[j]])
            
            if edges:
                from matplotlib.collections import LineCollection
                lc = LineCollection(edges[:100], alpha=0.2, colors='gray')  # Limit edges
                ax3.add_collection(lc)
            
            ax3.scatter(embedding[:, 0], embedding[:, 1], s=20, alpha=0.7)
            ax3.set_title("Neighborhood Structure")
            ax3.set_xlabel("Component 1")
            ax3.set_ylabel("Component 2")
            ax3.autoscale()
        
        # Local dimensions
        ax4 = plt.subplot(2, 3, 4)
        if analyzer.structure and analyzer.structure.local_dimensions is not None:
            local_dims = analyzer.structure.local_dimensions[~np.isnan(analyzer.structure.local_dimensions)]
            if len(local_dims) > 0:
                ax4.hist(local_dims, bins=30, alpha=0.7, edgecolor='black')
                ax4.axvline(np.median(local_dims), color='red', linestyle='--',
                          label=f'Median: {np.median(local_dims):.2f}')
                ax4.set_title("Local Intrinsic Dimensions")
                ax4.set_xlabel("Dimension")
                ax4.set_ylabel("Count")
                ax4.legend()
        
        # Distance preservation
        ax5 = plt.subplot(2, 3, 5)
        if analyzer.embedding:
            from scipy.spatial.distance import pdist
            orig_dists = pdist(analyzer.embedding.original_vectors)
            embed_dists = pdist(analyzer.embedding.embedded_vectors)
            
            # Sample for visualization
            sample_idx = np.random.choice(len(orig_dists), min(1000, len(orig_dists)), replace=False)
            ax5.scatter(orig_dists[sample_idx], embed_dists[sample_idx], alpha=0.3, s=1)
            ax5.plot([0, orig_dists[sample_idx].max()], 
                    [0, orig_dists[sample_idx].max()], 'r--', alpha=0.5)
            ax5.set_title("Distance Preservation")
            ax5.set_xlabel("Original Distance")
            ax5.set_ylabel("Embedded Distance")
        
        # Method info
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        if analyzer.config:
            info_text = f"Method: {analyzer.config.method.value}\n"
            info_text += f"Components: {analyzer.config.n_components}\n"
            info_text += f"Neighbors: {analyzer.config.n_neighbors}\n"
            info_text += f"Metric: {analyzer.config.metric}\n"
            
            if analyzer.embedding:
                info_text += f"\nSamples: {analyzer.embedding.n_samples}\n"
                
                if analyzer.embedding.quality_metrics:
                    trust = analyzer.embedding.quality_metrics.get('trustworthiness', 'N/A')
                    info_text += f"Trustworthiness: {trust:.3f}\n" if isinstance(trust, float) else f"Trustworthiness: {trust}\n"
            
            ax6.text(0.1, 0.9, info_text, transform=ax6.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            ax6.set_title("Configuration")
        
        plt.suptitle("Topographical Analysis Dashboard", fontsize=16)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
        
    except ImportError:
        logger.warning("Dashboard dependencies not available")
        return None