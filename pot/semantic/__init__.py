"""Semantic verification utilities for PoT."""

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
    compute_semantic_distance,
    semantic_similarity_score,
    batch_semantic_matching
)

from .utils import (
    normalize_concept_vector,
    compute_concept_centroid,
    cluster_concepts,
    validate_semantic_config,
    extract_embeddings_from_logits,
    normalize_embeddings,
    create_random_hypervector,
    hypervector_binding,
    hypervector_bundling,
    compute_embedding_statistics,
    reduce_embedding_dimension,
    plot_concept_space,
    visualize_concept_drift
)

from .config import (
    SemanticVerificationConfig,
    load_semantic_config,
    create_semantic_components,
    integrate_with_verifier,
    get_default_config
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
    'hypervector_binding',
    'hypervector_bundling',
    'compute_embedding_statistics',
    'reduce_embedding_dimension',
    'plot_concept_space',
    'visualize_concept_drift',
    
    # Configuration
    'SemanticVerificationConfig',
    'load_semantic_config',
    'create_semantic_components',
    'integrate_with_verifier',
    'get_default_config'
]