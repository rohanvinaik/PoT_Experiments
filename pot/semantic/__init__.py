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