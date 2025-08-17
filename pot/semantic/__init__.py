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
    validate_semantic_config
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
    'validate_semantic_config'
]