"""
Semantic verification utilities for PoT.
Performance optimized with caching, GPU acceleration, and sparse representations.
"""

import numpy as np
import torch
from typing import Optional, Tuple, Union, List, Dict, Any
from pathlib import Path

from .types import (
    ConceptVector,
    SemanticMatchResult,
    SemanticLibrary,
    MatchingConfig,
    SemanticDistance
)

from .library import (
    ConceptLibrary,
    ConceptVectorManager,
    load_concept_library,
    save_concept_library
)

from .match import (
    SemanticMatcher,
    LRUCache,
    set_device,
    get_device,
    compute_semantic_distance,
    semantic_similarity_score,
    batch_semantic_matching
)

# Optimized implementations
try:
    from .library_optimized import (
        MemoryMappedConceptLibrary,
        SparseHypervector,
        IncrementalStats,
        create_optimized_library
    )
    OPTIMIZED_AVAILABLE = True
except ImportError:
    OPTIMIZED_AVAILABLE = False

from .utils import (
    normalize_concept_vector,
    compute_concept_centroid,
    cluster_concepts,
    validate_semantic_config,
    extract_embeddings_from_logits,
    normalize_embeddings,
    create_random_hypervector,
    bind_hypervectors,
    bundle_hypervectors,
    reduce_dimensionality,
    compute_embedding_statistics,
    plot_concept_space,
    visualize_drift
)

from .config import (
    SemanticVerificationConfig,
    load_semantic_config,
    create_semantic_components,
    integrate_with_verifier,
    get_default_config
)

from .behavioral_fingerprint import (
    BehavioralFingerprint,
    BehaviorSnapshot,
    FingerprintHistory,
    ContinuousMonitor,
    create_behavioral_monitor
)

# Topographical learning components
from .topography import (
    TopographicalMethod,
    NeighborhoodMetric,
    TopographicalConfig,
    TopographicalEmbedding,
    TopologicalStructure,
    TopographicalMapper,
    UMAPMapper,
    TSNEMapper,
    SOMMapper,
    SOMProjector,
    TopographicalAnalyzer,
    TopographicalProjector,
    TopographicalEvolutionTracker,
    ConceptSpaceNavigator,
    create_topographical_analyzer,
    compare_topographies
)

from .topography_utils import (
    compute_distance_matrix,
    compute_neighborhood_preservation,
    compute_stress_metrics,
    estimate_intrinsic_dimension,
    compute_geodesic_distances,
    compute_persistence_diagram,
    compute_bottleneck_distance,
    compute_wasserstein_distance,
    subsample_data,
    align_embeddings,
    compute_local_quality,
    detect_outliers_topological,
    interpolate_manifold,
    prepare_latents_for_projection,
    compute_intrinsic_dimension,
    select_optimal_parameters,
    compute_trustworthiness,
    compute_continuity,
    compute_shepard_correlation,
    identify_clusters_in_projection,
    track_cluster_evolution,
    compute_cluster_transitions,
    compute_cluster_stability
)

from .topography_visualizer import (
    plot_projection,
    plot_density_map,
    plot_som_grid_extended,
    create_interactive_plot,
    create_evolution_animation_interactive,
    compare_projections,
    plot_embedding,
    plot_neighborhood_graph,
    plot_quality_heatmap,
    plot_persistence_diagram,
    plot_som_grid,
    plot_concept_trajectory,
    create_interactive_embedding,
    plot_distance_preservation,
    plot_manifold_density,
    plot_som_u_matrix,
    plot_som_component_planes,
    plot_som_hit_map,
    plot_som_clusters,
    create_som_dashboard,
    plot_evolution_trajectory,
    plot_drift_metrics_dashboard,
    create_evolution_animation,
    plot_cluster_evolution_heatmap,
    create_dashboard
)

# Unified interface for topographical semantic verification
def create_topographical_semantic_system(library_path: Optional[str] = None,
                                        projection_method: str = 'umap',
                                        **kwargs) -> Tuple['ConceptLibrary', 'SemanticMatcher', 'TopographicalProjector']:
    """
    Create a unified topographical semantic verification system.
    
    Args:
        library_path: Optional path to existing concept library
        projection_method: Default projection method ('umap', 'tsne', 'som', 'pca')
        **kwargs: Additional configuration options
        
    Returns:
        Tuple of (ConceptLibrary, SemanticMatcher, TopographicalProjector)
    """
    # Create concept library
    if library_path and Path(library_path).exists():
        library = ConceptLibrary(library_path=library_path)
    else:
        dim = kwargs.get('dim', 768)
        method = kwargs.get('concept_method', 'gaussian')
        library = ConceptLibrary(dim=dim, method=method)
    
    # Create semantic matcher
    threshold = kwargs.get('threshold', 0.8)
    matcher = SemanticMatcher(library, threshold=threshold)
    
    # Create topographical projector
    projector_config = {
        'default_method': projection_method,
        'cache_projections': kwargs.get('cache_projections', True)
    }
    projector = TopographicalProjector(projector_config)
    
    # Add topographical view to library
    library.add_topographical_view(projector)
    
    return library, matcher, projector


def visualize_semantic_landscape(library: 'ConceptLibrary',
                                method: str = 'umap',
                                interactive: bool = False,
                                save_path: Optional[str] = None) -> Union['matplotlib.figure.Figure', Any]:
    """
    Visualize the semantic landscape of a concept library.
    
    Args:
        library: ConceptLibrary to visualize
        method: Projection method ('umap', 'tsne', 'som', 'pca')
        interactive: Whether to create interactive visualization
        save_path: Optional path to save visualization
        
    Returns:
        matplotlib Figure or plotly figure object
    """
    if interactive:
        from .topography_visualizer import create_interactive_plot
        
        positions = library.get_concept_positions(method=method)
        if not positions:
            return None
        
        names = list(positions.keys())
        positions_array = np.array([positions[name] for name in names])
        
        fig = create_interactive_plot(
            positions_array,
            labels=names,
            title=f"Interactive Semantic Landscape ({method.upper()})"
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    else:
        return library.visualize_concept_space(method=method, save_path=save_path)


def analyze_semantic_evolution(embeddings_history: List[List[torch.Tensor]],
                              library: 'ConceptLibrary',
                              timestamps: Optional[List[float]] = None,
                              method: str = 'umap') -> Dict[str, Any]:
    """
    Analyze the evolution of semantic representations over time.
    
    Args:
        embeddings_history: List of embedding sets at different time points
        library: ConceptLibrary for reference
        timestamps: Optional timestamps for each snapshot
        method: Projection method
        
    Returns:
        Dictionary with evolution metrics and analysis
    """
    from .topography import TopographicalEvolutionTracker
    from .topography_utils import track_cluster_evolution
    
    # Create evolution tracker
    tracker = TopographicalEvolutionTracker()
    
    # Project each snapshot
    projector_config = {'default_method': method}
    projector = TopographicalProjector(projector_config)
    
    snapshots = []
    for embeddings in embeddings_history:
        if isinstance(embeddings[0], torch.Tensor):
            embeddings_tensor = torch.stack(embeddings)
        else:
            embeddings_tensor = torch.tensor(np.vstack(embeddings))
        
        projected = projector.project_latents(embeddings_tensor)
        snapshots.append(projected)
        
        # Add to tracker
        timestamp = timestamps[len(snapshots)-1] if timestamps else len(snapshots)-1
        tracker.add_snapshot(projected, timestamp=timestamp, compute_metrics=True)
    
    # Analyze evolution
    drift_metrics = tracker.compute_drift_metrics()
    cluster_evolution = track_cluster_evolution(snapshots)
    
    # Identify regime changes
    regime_changes = tracker.detect_regime_changes(method='gradient')
    
    return {
        'drift_metrics': drift_metrics,
        'cluster_evolution': cluster_evolution,
        'regime_changes': regime_changes,
        'n_snapshots': len(snapshots),
        'total_drift': drift_metrics.get('cumulative_drift', 0.0),
        'stability': cluster_evolution.get('mean_stability', 0.0)
    }


from pathlib import Path

__all__ = [
    # Types
    'ConceptVector',
    'SemanticMatchResult', 
    'SemanticLibrary',
    'MatchingConfig',
    'SemanticDistance',
    
    # Library management
    'ConceptLibrary',
    'ConceptVectorManager',
    'load_concept_library',
    'save_concept_library',
    
    # Matching and scoring
    'SemanticMatcher',
    'LRUCache',
    'set_device',
    'get_device',
    'compute_semantic_distance',
    'semantic_similarity_score',
    'batch_semantic_matching',
    
    # Utilities
    'normalize_concept_vector',
    'compute_concept_centroid',
    'cluster_concepts',
    'validate_semantic_config',
    'extract_embeddings_from_logits',
    'normalize_embeddings',
    'create_random_hypervector',
    'bind_hypervectors',
    'bundle_hypervectors',
    'reduce_dimensionality',
    'compute_embedding_statistics',
    'plot_concept_space',
    'visualize_drift',
    
    # Configuration
    'SemanticVerificationConfig',
    'load_semantic_config',
    'create_semantic_components',
    'integrate_with_verifier',
    'get_default_config',
    
    # Behavioral Fingerprinting
    'BehavioralFingerprint',
    'BehaviorSnapshot',
    'FingerprintHistory',
    'ContinuousMonitor',
    'create_behavioral_monitor',
    
    # Topographical learning
    'TopographicalMethod',
    'NeighborhoodMetric',
    'TopographicalConfig',
    'TopographicalEmbedding',
    'TopologicalStructure',
    'TopographicalMapper',
    'UMAPMapper',
    'TSNEMapper',
    'SOMMapper',
    'SOMProjector',
    'TopographicalAnalyzer',
    'TopographicalProjector',
    'TopographicalEvolutionTracker',
    'ConceptSpaceNavigator',
    'create_topographical_analyzer',
    'compare_topographies',
    
    # Topography utilities
    'compute_distance_matrix',
    'compute_neighborhood_preservation',
    'compute_stress_metrics',
    'estimate_intrinsic_dimension',
    'compute_geodesic_distances',
    'compute_persistence_diagram',
    'compute_bottleneck_distance',
    'compute_wasserstein_distance',
    'subsample_data',
    'align_embeddings',
    'compute_local_quality',
    'detect_outliers_topological',
    'interpolate_manifold',
    'prepare_latents_for_projection',
    'compute_intrinsic_dimension',
    'select_optimal_parameters',
    'compute_trustworthiness',
    'compute_continuity',
    'compute_shepard_correlation',
    'identify_clusters_in_projection',
    'track_cluster_evolution',
    'compute_cluster_transitions',
    'compute_cluster_stability',
    
    # Topography visualization
    'plot_projection',
    'plot_density_map',
    'plot_som_grid_extended',
    'create_interactive_plot',
    'create_evolution_animation_interactive',
    'compare_projections',
    'plot_embedding',
    'plot_neighborhood_graph',
    'plot_quality_heatmap',
    'plot_persistence_diagram',
    'plot_som_grid',
    'plot_concept_trajectory',
    'create_interactive_embedding',
    'plot_distance_preservation',
    'plot_manifold_density',
    'plot_som_u_matrix',
    'plot_som_component_planes',
    'plot_som_hit_map',
    'plot_som_clusters',
    'create_som_dashboard',
    'plot_evolution_trajectory',
    'plot_drift_metrics_dashboard',
    'create_evolution_animation',
    'plot_cluster_evolution_heatmap',
    'create_dashboard',
    
    # Unified topographical interface
    'create_topographical_semantic_system',
    'visualize_semantic_landscape',
    'analyze_semantic_evolution',
    
    # Optimization flag
    'OPTIMIZED_AVAILABLE'
]

# Add optimized components if available
if OPTIMIZED_AVAILABLE:
    __all__.extend([
        'MemoryMappedConceptLibrary',
        'SparseHypervector',
        'IncrementalStats',
        'create_optimized_library'
    ])