"""
Semantic verification utilities for PoT.
Performance optimized with caching, GPU acceleration, and sparse representations.
"""

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
    interpolate_manifold
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