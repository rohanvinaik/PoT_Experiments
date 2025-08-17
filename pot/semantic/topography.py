"""
Topographical learning module for semantic verification in PoT.
Implements advanced manifold learning and topological structure preservation
for concept space analysis and model behavior verification.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import logging
from pathlib import Path
import json
import pickle
from abc import ABC, abstractmethod

# Import existing semantic types
from .types import ConceptVector, SemanticDistance
from .library import ConceptLibrary
from .utils import normalize_embeddings, compute_embedding_statistics

logger = logging.getLogger(__name__)


class TopographicalMethod(Enum):
    """Enumeration of supported topographical learning methods."""
    UMAP = "umap"
    TSNE = "tsne"
    SOM = "som"
    ISOMAP = "isomap"
    LLE = "lle"  # Locally Linear Embedding
    SPECTRAL = "spectral"
    MDS = "mds"  # Multidimensional Scaling
    DIFFUSION_MAP = "diffusion_map"


class NeighborhoodMetric(Enum):
    """Metrics for measuring neighborhood preservation."""
    TRUSTWORTHINESS = "trustworthiness"
    CONTINUITY = "continuity"
    LOCAL_CONTINUITY = "local_continuity"
    SAMMON_STRESS = "sammon_stress"
    SHEPARD_CORRELATION = "shepard_correlation"


@dataclass
class TopographicalConfig:
    """
    Configuration for topographical learning and analysis.
    
    Attributes:
        method: Topographical learning method to use
        n_components: Target dimensionality
        n_neighbors: Number of neighbors for local methods
        metric: Distance metric for embeddings
        min_dist: Minimum distance for UMAP
        learning_rate: Learning rate for iterative methods
        perplexity: Perplexity for t-SNE
        grid_size: Grid dimensions for SOM
        topology: Topology type for SOM ('rectangular' or 'hexagonal')
        random_state: Random seed for reproducibility
        preserve_density: Whether to preserve density information
        angular_rp_forest: Use angular random projection forest (UMAP)
        verbose: Verbosity level
    """
    method: TopographicalMethod = TopographicalMethod.UMAP
    n_components: int = 2
    n_neighbors: int = 15
    metric: str = "euclidean"
    min_dist: float = 0.1
    learning_rate: float = 1.0
    perplexity: float = 30.0
    grid_size: Tuple[int, int] = (10, 10)
    topology: str = "rectangular"
    random_state: Optional[int] = 42
    preserve_density: bool = False
    angular_rp_forest: bool = True
    verbose: int = 0
    
    @classmethod
    def for_high_dimensional(cls, n_samples: int) -> 'TopographicalConfig':
        """Create config optimized for high-dimensional data."""
        return cls(
            method=TopographicalMethod.UMAP,
            n_neighbors=min(15, n_samples // 10),
            min_dist=0.05,
            metric="cosine",
            preserve_density=True
        )
    
    @classmethod
    def for_visualization(cls) -> 'TopographicalConfig':
        """Create config optimized for visualization."""
        return cls(
            method=TopographicalMethod.TSNE,
            n_components=2,
            perplexity=30.0,
            learning_rate=200.0
        )
    
    @classmethod
    def for_clustering(cls) -> 'TopographicalConfig':
        """Create config optimized for clustering analysis."""
        return cls(
            method=TopographicalMethod.SOM,
            grid_size=(20, 20),
            topology="hexagonal"
        )


@dataclass
class TopographicalEmbedding:
    """
    Result of topographical embedding.
    
    Attributes:
        original_vectors: Original high-dimensional vectors
        embedded_vectors: Low-dimensional embeddings
        method: Method used for embedding
        config: Configuration used
        metadata: Additional metadata
        quality_metrics: Quality metrics for the embedding
        model: Trained embedding model (if applicable)
    """
    original_vectors: np.ndarray
    embedded_vectors: np.ndarray
    method: TopographicalMethod
    config: TopographicalConfig
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    model: Optional[Any] = None
    
    @property
    def n_samples(self) -> int:
        """Number of samples in the embedding."""
        return self.embedded_vectors.shape[0]
    
    @property
    def n_components(self) -> int:
        """Number of components in the embedding."""
        return self.embedded_vectors.shape[1]
    
    def compute_reconstruction_error(self) -> float:
        """Compute reconstruction error if applicable."""
        if self.model is None:
            return np.nan
        
        try:
            # Attempt to reconstruct if model supports it
            if hasattr(self.model, 'inverse_transform'):
                reconstructed = self.model.inverse_transform(self.embedded_vectors)
                error = np.mean((self.original_vectors - reconstructed) ** 2)
                return float(error)
        except:
            pass
        
        return np.nan


@dataclass
class TopologicalStructure:
    """
    Represents topological structure of concept space.
    
    Attributes:
        adjacency_matrix: Adjacency matrix of the graph
        connected_components: List of connected components
        persistence_diagram: Persistence diagram for TDA
        betti_numbers: Betti numbers (topological invariants)
        geodesic_distances: Geodesic distances between points
        local_dimensions: Local intrinsic dimensions
    """
    adjacency_matrix: Optional[np.ndarray] = None
    connected_components: List[List[int]] = field(default_factory=list)
    persistence_diagram: Optional[np.ndarray] = None
    betti_numbers: List[int] = field(default_factory=list)
    geodesic_distances: Optional[np.ndarray] = None
    local_dimensions: Optional[np.ndarray] = None


class TopographicalMapper(ABC):
    """Abstract base class for topographical mapping methods."""
    
    @abstractmethod
    def fit(self, data: np.ndarray) -> 'TopographicalMapper':
        """Fit the mapper to data."""
        pass
    
    @abstractmethod
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data to embedded space."""
        pass
    
    @abstractmethod
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        pass


class UMAPMapper(TopographicalMapper):
    """UMAP-based topographical mapper."""
    
    def __init__(self, config: TopographicalConfig):
        self.config = config
        self.model = None
    
    def fit(self, data: np.ndarray) -> 'UMAPMapper':
        """Fit UMAP model to data."""
        try:
            import umap
            
            self.model = umap.UMAP(
                n_components=self.config.n_components,
                n_neighbors=self.config.n_neighbors,
                min_dist=self.config.min_dist,
                metric=self.config.metric,
                random_state=self.config.random_state,
                densmap=self.config.preserve_density,
                angular_rp_forest=self.config.angular_rp_forest,
                verbose=self.config.verbose
            )
            self.model.fit(data)
            
        except ImportError:
            raise ImportError("UMAP not installed. Run: pip install umap-learn")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data using fitted UMAP model."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.transform(data)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data in one step."""
        self.fit(data)
        return self.model.embedding_


class TSNEMapper(TopographicalMapper):
    """t-SNE based topographical mapper."""
    
    def __init__(self, config: TopographicalConfig):
        self.config = config
        self.model = None
        self.embedding_ = None
    
    def fit(self, data: np.ndarray) -> 'TSNEMapper':
        """Fit t-SNE model (not separate from transform)."""
        # t-SNE doesn't support separate fit/transform
        logger.warning("t-SNE doesn't support separate fit(). Use fit_transform().")
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform not supported for t-SNE."""
        raise NotImplementedError("t-SNE doesn't support transform on new data.")
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data using t-SNE."""
        try:
            from sklearn.manifold import TSNE
            
            self.model = TSNE(
                n_components=self.config.n_components,
                perplexity=self.config.perplexity,
                learning_rate=self.config.learning_rate,
                random_state=self.config.random_state,
                metric=self.config.metric,
                verbose=self.config.verbose
            )
            self.embedding_ = self.model.fit_transform(data)
            return self.embedding_
            
        except ImportError:
            raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")


class SOMMapper(TopographicalMapper):
    """Self-Organizing Map based topographical mapper."""
    
    def __init__(self, config: TopographicalConfig):
        self.config = config
        self.model = None
    
    def fit(self, data: np.ndarray) -> 'SOMMapper':
        """Fit SOM to data."""
        try:
            from minisom import MiniSom
            
            x_size, y_size = self.config.grid_size
            input_dim = data.shape[1]
            
            self.model = MiniSom(
                x_size, y_size, input_dim,
                sigma=1.0,
                learning_rate=self.config.learning_rate,
                topology=self.config.topology,
                random_seed=self.config.random_state
            )
            
            self.model.random_weights_init(data)
            self.model.train_batch(data, num_iteration=100)
            
        except ImportError:
            raise ImportError("MiniSom not installed. Run: pip install minisom")
        
        return self
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Transform data to SOM grid coordinates."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get winning nodes for each sample
        embeddings = []
        for sample in data:
            winner = self.model.winner(sample)
            embeddings.append(winner)
        
        return np.array(embeddings)
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform data using SOM."""
        self.fit(data)
        return self.transform(data)


class TopographicalAnalyzer:
    """
    Main class for topographical analysis of concept spaces.
    
    Provides methods for:
    - Manifold learning and dimensionality reduction
    - Topological structure analysis
    - Neighborhood preservation metrics
    - Concept drift detection in topological space
    """
    
    def __init__(self, config: Optional[TopographicalConfig] = None):
        """Initialize analyzer with configuration."""
        self.config = config or TopographicalConfig()
        self.mapper: Optional[TopographicalMapper] = None
        self.embedding: Optional[TopographicalEmbedding] = None
        self.structure: Optional[TopologicalStructure] = None
        
    def create_mapper(self) -> TopographicalMapper:
        """Create appropriate mapper based on configuration."""
        if self.config.method == TopographicalMethod.UMAP:
            return UMAPMapper(self.config)
        elif self.config.method == TopographicalMethod.TSNE:
            return TSNEMapper(self.config)
        elif self.config.method == TopographicalMethod.SOM:
            return SOMMapper(self.config)
        else:
            raise NotImplementedError(f"Method {self.config.method} not implemented")
    
    def fit_transform(self, 
                     vectors: Union[np.ndarray, List[ConceptVector]],
                     compute_metrics: bool = True) -> TopographicalEmbedding:
        """
        Fit topographical model and transform vectors.
        
        Args:
            vectors: Input vectors (high-dimensional)
            compute_metrics: Whether to compute quality metrics
            
        Returns:
            TopographicalEmbedding object
        """
        # Convert ConceptVectors to numpy array if needed
        if isinstance(vectors, list) and len(vectors) > 0:
            if isinstance(vectors[0], ConceptVector):
                data = np.array([cv.vector for cv in vectors])
            else:
                data = np.array(vectors)
        else:
            data = np.asarray(vectors)
        
        # Normalize data if needed
        if self.config.metric in ["cosine", "correlation"]:
            data = normalize_embeddings(data, method="l2")
        
        # Create and apply mapper
        self.mapper = self.create_mapper()
        embedded = self.mapper.fit_transform(data)
        
        # Create embedding object
        self.embedding = TopographicalEmbedding(
            original_vectors=data,
            embedded_vectors=embedded,
            method=self.config.method,
            config=self.config,
            model=self.mapper.model if hasattr(self.mapper, 'model') else None
        )
        
        # Compute quality metrics if requested
        if compute_metrics:
            self.embedding.quality_metrics = self.compute_quality_metrics()
        
        return self.embedding
    
    def compute_quality_metrics(self, k: int = 10) -> Dict[str, float]:
        """
        Compute quality metrics for the embedding.
        
        Args:
            k: Number of neighbors for local metrics
            
        Returns:
            Dictionary of quality metrics
        """
        if self.embedding is None:
            raise ValueError("No embedding available. Call fit_transform() first.")
        
        metrics = {}
        
        # Compute trustworthiness and continuity
        try:
            from sklearn.manifold import trustworthiness
            metrics['trustworthiness'] = float(trustworthiness(
                self.embedding.original_vectors,
                self.embedding.embedded_vectors,
                n_neighbors=k
            ))
        except:
            pass
        
        # Compute stress (sum of squared distance differences)
        try:
            from scipy.spatial.distance import pdist, squareform
            orig_dists = pdist(self.embedding.original_vectors)
            embed_dists = pdist(self.embedding.embedded_vectors)
            stress = np.sum((orig_dists - embed_dists) ** 2)
            metrics['stress'] = float(stress)
            
            # Normalized stress
            metrics['normalized_stress'] = float(stress / np.sum(orig_dists ** 2))
            
            # Correlation between distances
            correlation = np.corrcoef(orig_dists, embed_dists)[0, 1]
            metrics['distance_correlation'] = float(correlation)
        except:
            pass
        
        # Reconstruction error if applicable
        recon_error = self.embedding.compute_reconstruction_error()
        if not np.isnan(recon_error):
            metrics['reconstruction_error'] = recon_error
        
        return metrics
    
    def analyze_topology(self, 
                        k_neighbors: int = 10,
                        compute_geodesics: bool = False) -> TopologicalStructure:
        """
        Analyze topological structure of the embedding.
        
        Args:
            k_neighbors: Number of neighbors for graph construction
            compute_geodesics: Whether to compute geodesic distances
            
        Returns:
            TopologicalStructure object
        """
        if self.embedding is None:
            raise ValueError("No embedding available. Call fit_transform() first.")
        
        self.structure = TopologicalStructure()
        
        # Build k-NN graph
        try:
            from sklearn.neighbors import kneighbors_graph
            adjacency = kneighbors_graph(
                self.embedding.embedded_vectors,
                n_neighbors=k_neighbors,
                mode='connectivity'
            )
            self.structure.adjacency_matrix = adjacency.toarray()
        except:
            logger.warning("Could not compute adjacency matrix")
        
        # Find connected components
        try:
            from scipy.sparse.csgraph import connected_components
            n_components, labels = connected_components(
                self.structure.adjacency_matrix,
                directed=False
            )
            
            components = [[] for _ in range(n_components)]
            for idx, label in enumerate(labels):
                components[label].append(idx)
            self.structure.connected_components = components
        except:
            logger.warning("Could not compute connected components")
        
        # Compute geodesic distances if requested
        if compute_geodesics and self.structure.adjacency_matrix is not None:
            try:
                from sklearn.manifold import Isomap
                iso = Isomap(n_neighbors=k_neighbors)
                iso.fit(self.embedding.embedded_vectors)
                self.structure.geodesic_distances = iso.dist_matrix_
            except:
                logger.warning("Could not compute geodesic distances")
        
        # Estimate local intrinsic dimensions
        try:
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=k_neighbors+1)
            nbrs.fit(self.embedding.embedded_vectors)
            distances, _ = nbrs.kneighbors(self.embedding.embedded_vectors)
            
            # MLE estimator for local dimension
            local_dims = []
            for i in range(len(distances)):
                dists = distances[i, 1:]  # Exclude self
                if dists[-1] > 0:
                    dim = len(dists) / np.sum(np.log(dists[-1] / dists[:-1]))
                    local_dims.append(dim)
                else:
                    local_dims.append(np.nan)
            
            self.structure.local_dimensions = np.array(local_dims)
        except:
            logger.warning("Could not compute local dimensions")
        
        return self.structure
    
    def detect_concept_drift(self,
                           reference_embedding: TopographicalEmbedding,
                           test_embedding: TopographicalEmbedding,
                           method: str = "procrustes") -> Dict[str, Any]:
        """
        Detect concept drift in topological space.
        
        Args:
            reference_embedding: Reference embedding
            test_embedding: Test embedding to compare
            method: Method for drift detection ('procrustes', 'wasserstein', 'mmd')
            
        Returns:
            Dictionary with drift metrics
        """
        drift_info = {
            'method': method,
            'drift_detected': False,
            'drift_score': 0.0,
            'p_value': None
        }
        
        ref_vecs = reference_embedding.embedded_vectors
        test_vecs = test_embedding.embedded_vectors
        
        if method == "procrustes":
            # Procrustes analysis for shape comparison
            try:
                from scipy.spatial import procrustes
                _, _, disparity = procrustes(ref_vecs, test_vecs)
                drift_info['drift_score'] = float(disparity)
                drift_info['drift_detected'] = disparity > 0.1  # Threshold
            except:
                logger.warning("Could not perform Procrustes analysis")
                
        elif method == "wasserstein":
            # Wasserstein distance between distributions
            try:
                from scipy.stats import wasserstein_distance
                # Compute 1D projections and compare
                drift_scores = []
                for dim in range(ref_vecs.shape[1]):
                    dist = wasserstein_distance(
                        ref_vecs[:, dim],
                        test_vecs[:, dim]
                    )
                    drift_scores.append(dist)
                
                drift_info['drift_score'] = float(np.mean(drift_scores))
                drift_info['drift_detected'] = drift_info['drift_score'] > 0.5
            except:
                logger.warning("Could not compute Wasserstein distance")
                
        elif method == "mmd":
            # Maximum Mean Discrepancy
            try:
                def gaussian_kernel(x, y, sigma=1.0):
                    """Gaussian RBF kernel."""
                    dist = np.sum((x[:, None] - y[None, :]) ** 2, axis=2)
                    return np.exp(-dist / (2 * sigma ** 2))
                
                K_xx = gaussian_kernel(ref_vecs, ref_vecs)
                K_yy = gaussian_kernel(test_vecs, test_vecs)
                K_xy = gaussian_kernel(ref_vecs, test_vecs)
                
                mmd = np.mean(K_xx) + np.mean(K_yy) - 2 * np.mean(K_xy)
                drift_info['drift_score'] = float(mmd)
                drift_info['drift_detected'] = mmd > 0.05
            except:
                logger.warning("Could not compute MMD")
        
        return drift_info
    
    def find_concept_clusters(self,
                            min_cluster_size: int = 5,
                            method: str = "dbscan") -> Dict[int, List[int]]:
        """
        Find concept clusters in topological space.
        
        Args:
            min_cluster_size: Minimum cluster size
            method: Clustering method ('dbscan', 'hdbscan', 'spectral')
            
        Returns:
            Dictionary mapping cluster ID to sample indices
        """
        if self.embedding is None:
            raise ValueError("No embedding available. Call fit_transform() first.")
        
        clusters = {}
        
        if method == "dbscan":
            try:
                from sklearn.cluster import DBSCAN
                clustering = DBSCAN(min_samples=min_cluster_size, eps=0.5)
                labels = clustering.fit_predict(self.embedding.embedded_vectors)
                
                for label in np.unique(labels):
                    if label != -1:  # Ignore noise
                        clusters[int(label)] = np.where(labels == label)[0].tolist()
            except:
                logger.warning("Could not perform DBSCAN clustering")
                
        elif method == "hdbscan":
            try:
                import hdbscan
                clustering = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                labels = clustering.fit_predict(self.embedding.embedded_vectors)
                
                for label in np.unique(labels):
                    if label != -1:
                        clusters[int(label)] = np.where(labels == label)[0].tolist()
            except:
                logger.warning("HDBSCAN not available")
                
        elif method == "spectral":
            try:
                from sklearn.cluster import SpectralClustering
                n_clusters = max(2, len(self.embedding.embedded_vectors) // 20)
                clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors')
                labels = clustering.fit_predict(self.embedding.embedded_vectors)
                
                for label in np.unique(labels):
                    cluster_indices = np.where(labels == label)[0].tolist()
                    if len(cluster_indices) >= min_cluster_size:
                        clusters[int(label)] = cluster_indices
            except:
                logger.warning("Could not perform spectral clustering")
        
        return clusters
    
    def save(self, path: Union[str, Path]) -> None:
        """Save analyzer state to file."""
        path = Path(path)
        state = {
            'config': self.config,
            'embedding': self.embedding,
            'structure': self.structure
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved topographical analyzer to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """Load analyzer state from file."""
        path = Path(path)
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.config = state['config']
        self.embedding = state['embedding']
        self.structure = state.get('structure')
        
        logger.info(f"Loaded topographical analyzer from {path}")


class TopographicalProjector:
    """
    High-performance topographical projector for latent space visualization.
    Implements multiple projection methods with automatic parameter selection.
    """
    
    def __init__(self, method: str = 'umap', **kwargs):
        """
        Initialize projector with specified method.
        
        Args:
            method: Projection method ('umap', 'tsne', 'som')
            **kwargs: Method-specific parameters:
                UMAP: n_neighbors, min_dist, n_components, metric
                t-SNE: perplexity, learning_rate, n_components
                SOM: grid_size, learning_rate, topology
        """
        self.method = method.lower()
        self.params = kwargs
        self.fitted_model = None
        self.is_fitted = False
        
        # Validate method
        valid_methods = ['umap', 'tsne', 'som']
        if self.method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}, got {self.method}")
        
        # Set default parameters based on method
        self._set_default_params()
        
        # Update with user-provided parameters
        self.params.update(kwargs)
        
        logger.info(f"Initialized TopographicalProjector with method: {self.method}")
    
    def _set_default_params(self):
        """Set default parameters for each projection method."""
        if self.method == 'umap':
            self.params = {
                'n_neighbors': 15,
                'min_dist': 0.1,
                'n_components': 2,
                'metric': 'euclidean',
                'random_state': 42,
                'verbose': False
            }
        elif self.method == 'tsne':
            self.params = {
                'perplexity': 30.0,
                'learning_rate': 'auto',
                'n_components': 2,
                'n_iter': 1000,
                'random_state': 42,
                'verbose': 0
            }
        elif self.method == 'som':
            self.params = {
                'grid_size': (10, 10),
                'learning_rate': 0.5,
                'topology': 'rectangular',
                'sigma': 1.0,
                'num_iteration': 100,
                'random_seed': 42
            }
    
    def _auto_adjust_params(self, n_samples: int, n_features: int):
        """Automatically adjust parameters based on data size."""
        if self.method == 'umap':
            # Adjust n_neighbors based on sample size
            max_neighbors = min(n_samples - 1, 100)
            if 'n_neighbors' not in self.params or self.params['n_neighbors'] > max_neighbors:
                self.params['n_neighbors'] = min(15, max_neighbors)
            
            # Adjust min_dist for very small datasets
            if n_samples < 50:
                self.params['min_dist'] = max(0.01, self.params.get('min_dist', 0.1))
                
        elif self.method == 'tsne':
            # Adjust perplexity based on sample size
            max_perplexity = (n_samples - 1) / 3
            if 'perplexity' not in self.params or self.params['perplexity'] > max_perplexity:
                self.params['perplexity'] = min(30.0, max(5.0, max_perplexity))
            
            # Auto learning rate if not specified
            if self.params.get('learning_rate') == 'auto':
                self.params['learning_rate'] = max(n_samples / 12, 200.0)
                
        elif self.method == 'som':
            # Adjust grid size based on sample size
            if n_samples < 100:
                self.params['grid_size'] = (5, 5)
            elif n_samples < 500:
                self.params['grid_size'] = (10, 10)
            else:
                grid_dim = min(30, int(np.sqrt(n_samples / 5)))
                self.params['grid_size'] = (grid_dim, grid_dim)
    
    def _validate_input(self, latents: torch.Tensor) -> np.ndarray:
        """Validate and convert input latents to numpy array."""
        if not isinstance(latents, (torch.Tensor, np.ndarray)):
            raise TypeError("Latents must be torch.Tensor or numpy.ndarray")
        
        # Convert to numpy if needed
        if isinstance(latents, torch.Tensor):
            if latents.is_cuda:
                latents = latents.cpu()
            latents = latents.detach().numpy()
        
        # Ensure 2D array
        if latents.ndim == 1:
            latents = latents.reshape(-1, 1)
        elif latents.ndim > 2:
            # Flatten to 2D
            latents = latents.reshape(latents.shape[0], -1)
        
        # Check for NaN or Inf
        if np.any(~np.isfinite(latents)):
            logger.warning("Input contains NaN or Inf values, replacing with 0")
            latents = np.nan_to_num(latents, nan=0.0, posinf=0.0, neginf=0.0)
        
        return latents
    
    def project_latents(self, latents: torch.Tensor, 
                       method: Optional[str] = None,
                       return_model: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Any]]:
        """
        Project high-dimensional latents to 2D/3D space.
        
        Args:
            latents: High-dimensional latent vectors (n_samples, n_features)
            method: Override projection method (optional)
            return_model: Whether to return the fitted model
            
        Returns:
            Projected points in low-dimensional space (n_samples, n_components)
            Optionally returns (projected_points, fitted_model)
        """
        # Use override method if provided
        current_method = method.lower() if method else self.method
        
        # Validate and convert input
        latents_np = self._validate_input(latents)
        n_samples, n_features = latents_np.shape
        
        # Auto-adjust parameters
        self._auto_adjust_params(n_samples, n_features)
        
        logger.info(f"Projecting {n_samples} samples with {n_features} features using {current_method}")
        
        # Select projection method
        if current_method == 'umap':
            projected, model = self._project_umap(latents_np)
        elif current_method == 'tsne':
            projected, model = self._project_tsne(latents_np)
        elif current_method == 'som':
            projected, model = self._project_som(latents_np)
        else:
            raise ValueError(f"Unknown projection method: {current_method}")
        
        # Store fitted model
        self.fitted_model = model
        self.is_fitted = True
        
        if return_model:
            return projected, model
        return projected
    
    def _project_umap(self, latents: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Project using UMAP (Uniform Manifold Approximation and Projection).
        
        Args:
            latents: Input data (n_samples, n_features)
            
        Returns:
            Tuple of (projected_points, fitted_model)
        """
        try:
            import umap
        except ImportError:
            raise ImportError("UMAP not installed. Run: pip install umap-learn")
        
        # Create UMAP model with current parameters
        model = umap.UMAP(
            n_neighbors=self.params.get('n_neighbors', 15),
            min_dist=self.params.get('min_dist', 0.1),
            n_components=self.params.get('n_components', 2),
            metric=self.params.get('metric', 'euclidean'),
            random_state=self.params.get('random_state', 42),
            verbose=self.params.get('verbose', False)
        )
        
        # Fit and transform
        try:
            projected = model.fit_transform(latents)
            logger.info(f"UMAP projection complete: {latents.shape} -> {projected.shape}")
        except Exception as e:
            logger.error(f"UMAP projection failed: {e}")
            # Fallback to PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.params.get('n_components', 2))
            projected = pca.fit_transform(latents)
            model = pca
            logger.warning("Fell back to PCA due to UMAP failure")
        
        return projected, model
    
    def _project_tsne(self, latents: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Project using t-SNE (t-Distributed Stochastic Neighbor Embedding).
        
        Args:
            latents: Input data (n_samples, n_features)
            
        Returns:
            Tuple of (projected_points, fitted_model)
        """
        try:
            from sklearn.manifold import TSNE
        except ImportError:
            raise ImportError("scikit-learn not installed. Run: pip install scikit-learn")
        
        # Create t-SNE model with current parameters
        model = TSNE(
            n_components=self.params.get('n_components', 2),
            perplexity=self.params.get('perplexity', 30.0),
            learning_rate=self.params.get('learning_rate', 'auto'),
            n_iter=self.params.get('n_iter', 1000),
            random_state=self.params.get('random_state', 42),
            verbose=self.params.get('verbose', 0)
        )
        
        # Fit and transform
        try:
            projected = model.fit_transform(latents)
            logger.info(f"t-SNE projection complete: {latents.shape} -> {projected.shape}")
        except Exception as e:
            logger.error(f"t-SNE projection failed: {e}")
            # Fallback to PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.params.get('n_components', 2))
            projected = pca.fit_transform(latents)
            model = pca
            logger.warning("Fell back to PCA due to t-SNE failure")
        
        return projected, model
    
    def _project_som(self, latents: np.ndarray) -> Tuple[np.ndarray, Any]:
        """
        Project using SOM (Self-Organizing Map).
        
        Args:
            latents: Input data (n_samples, n_features)
            
        Returns:
            Tuple of (projected_points, fitted_model)
        """
        try:
            from minisom import MiniSom
        except ImportError:
            raise ImportError("MiniSom not installed. Run: pip install minisom")
        
        # Get SOM parameters
        grid_size = self.params.get('grid_size', (10, 10))
        x_size, y_size = grid_size
        input_dim = latents.shape[1]
        
        # Create SOM model
        model = MiniSom(
            x_size, y_size, input_dim,
            sigma=self.params.get('sigma', 1.0),
            learning_rate=self.params.get('learning_rate', 0.5),
            topology=self.params.get('topology', 'rectangular'),
            random_seed=self.params.get('random_seed', 42)
        )
        
        # Initialize and train
        try:
            model.random_weights_init(latents)
            model.train_batch(latents, self.params.get('num_iteration', 100))
            
            # Get 2D coordinates for each sample
            projected = np.zeros((latents.shape[0], 2))
            for i, sample in enumerate(latents):
                winner = model.winner(sample)
                # Add small random jitter to avoid overlapping points
                jitter = np.random.normal(0, 0.1, 2)
                projected[i] = np.array(winner) + jitter
            
            logger.info(f"SOM projection complete: {latents.shape} -> {projected.shape}")
            
        except Exception as e:
            logger.error(f"SOM projection failed: {e}")
            # Fallback to grid positions
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            projected = pca.fit_transform(latents)
            model = pca
            logger.warning("Fell back to PCA due to SOM failure")
        
        return projected, model
    
    def fit_transform(self, latents: torch.Tensor) -> np.ndarray:
        """
        Fit projection model and transform latents.
        
        Args:
            latents: High-dimensional latent vectors
            
        Returns:
            Projected points in low-dimensional space
        """
        return self.project_latents(latents, return_model=False)
    
    def transform(self, latents: torch.Tensor) -> np.ndarray:
        """
        Transform new latents using fitted model.
        
        Args:
            latents: New high-dimensional latent vectors
            
        Returns:
            Projected points in low-dimensional space
            
        Raises:
            RuntimeError: If model hasn't been fitted yet
        """
        if not self.is_fitted or self.fitted_model is None:
            raise RuntimeError("Model not fitted. Call fit_transform() or project_latents() first.")
        
        # Validate input
        latents_np = self._validate_input(latents)
        
        # Transform based on method
        if self.method == 'umap':
            if hasattr(self.fitted_model, 'transform'):
                projected = self.fitted_model.transform(latents_np)
            else:
                # Fallback for models that don't support transform
                logger.warning("Model doesn't support transform, refitting")
                return self.fit_transform(latents)
                
        elif self.method == 'tsne':
            # t-SNE doesn't support transform on new data
            logger.warning("t-SNE doesn't support transform on new data, refitting")
            return self.fit_transform(latents)
            
        elif self.method == 'som':
            # Use fitted SOM to get coordinates
            if hasattr(self.fitted_model, 'winner'):
                projected = np.zeros((latents_np.shape[0], 2))
                for i, sample in enumerate(latents_np):
                    winner = self.fitted_model.winner(sample)
                    jitter = np.random.normal(0, 0.1, 2)
                    projected[i] = np.array(winner) + jitter
            else:
                # Fallback
                logger.warning("SOM model doesn't support transform, refitting")
                return self.fit_transform(latents)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return projected
    
    def save_model(self, path: str):
        """Save fitted projection model to file."""
        import pickle
        if not self.is_fitted:
            raise RuntimeError("No fitted model to save")
        
        with open(path, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'params': self.params,
                'model': self.fitted_model
            }, f)
        logger.info(f"Saved projection model to {path}")
    
    def load_model(self, path: str):
        """Load fitted projection model from file."""
        import pickle
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.method = data['method']
        self.params = data['params']
        self.fitted_model = data['model']
        self.is_fitted = True
        logger.info(f"Loaded projection model from {path}")


class SOMProjector:
    """
    Self-Organizing Map projector for topographical learning.
    Supports MiniSom library with custom fallback implementation.
    """
    
    def __init__(self, x_dim: int = 10, y_dim: int = 10, 
                 input_len: int = None, sigma: float = 1.0,
                 learning_rate: float = 0.5, topology: str = 'rectangular',
                 activation_distance: str = 'euclidean', random_seed: int = 42):
        """
        Initialize SOM with grid dimensions and parameters.
        
        Args:
            x_dim: Width of the SOM grid
            y_dim: Height of the SOM grid
            input_len: Dimensionality of input vectors
            sigma: Spread of the neighborhood function
            learning_rate: Initial learning rate
            topology: Grid topology ('rectangular' or 'hexagonal')
            activation_distance: Distance metric for activation
            random_seed: Random seed for reproducibility
        """
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.topology = topology
        self.activation_distance = activation_distance
        self.random_seed = random_seed
        
        self.som = None
        self.is_trained = False
        self.use_minisom = True
        self.weights = None
        
        # Try to use MiniSom if available
        try:
            from minisom import MiniSom
            self._minisom_available = True
            logger.info("Using MiniSom library for SOM implementation")
        except ImportError:
            self._minisom_available = False
            logger.info("MiniSom not available, using custom implementation")
            self.use_minisom = False
    
    def _init_som(self, input_len: int):
        """Initialize the SOM implementation."""
        self.input_len = input_len
        
        if self._minisom_available and self.use_minisom:
            from minisom import MiniSom
            self.som = MiniSom(
                x=self.x_dim,
                y=self.y_dim,
                input_len=input_len,
                sigma=self.sigma,
                learning_rate=self.learning_rate,
                topology=self.topology,
                activation_distance=self.activation_distance,
                random_seed=self.random_seed
            )
        else:
            # Use custom implementation
            self._init_custom_som(input_len)
    
    def _init_custom_som(self, input_len: int):
        """Initialize custom SOM implementation."""
        np.random.seed(self.random_seed)
        
        # Initialize weight vectors randomly
        self.weights = np.random.randn(self.x_dim, self.y_dim, input_len)
        
        # Normalize weights
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                norm = np.linalg.norm(self.weights[i, j])
                if norm > 0:
                    self.weights[i, j] /= norm
        
        # Create grid coordinates
        self.grid_coords = np.array([[i, j] for i in range(self.x_dim) 
                                     for j in range(self.y_dim)])
        
        logger.info(f"Initialized custom SOM with shape ({self.x_dim}, {self.y_dim}, {input_len})")
    
    def _validate_input(self, data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Validate and convert input data."""
        if isinstance(data, torch.Tensor):
            if data.is_cuda:
                data = data.cpu()
            data = data.detach().numpy()
        
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        # Handle NaN/Inf
        if np.any(~np.isfinite(data)):
            logger.warning("Input contains NaN/Inf, replacing with 0")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        return data
    
    def train(self, data: torch.Tensor, num_epochs: int = 100, verbose: bool = False):
        """
        Train SOM on input data.
        
        Args:
            data: Input data tensor (n_samples, n_features)
            num_epochs: Number of training epochs
            verbose: Whether to print progress
        """
        data_np = self._validate_input(data)
        n_samples, n_features = data_np.shape
        
        # Initialize SOM if not done
        if self.som is None and self.weights is None:
            self._init_som(n_features)
        
        if self._minisom_available and self.use_minisom:
            # Train using MiniSom
            self.som.random_weights_init(data_np)
            
            if verbose:
                from tqdm import tqdm
                for epoch in tqdm(range(num_epochs), desc="Training SOM"):
                    self.som.train_random(data_np, 1)
            else:
                self.som.train_batch(data_np, num_epochs)
            
            self.weights = self.som.get_weights()
        else:
            # Train using custom implementation
            self._train_custom_som(data_np, num_epochs, verbose)
        
        self.is_trained = True
        logger.info(f"SOM training completed after {num_epochs} epochs")
    
    def _train_custom_som(self, data: np.ndarray, num_epochs: int, verbose: bool):
        """Custom SOM training implementation."""
        n_samples = len(data)
        
        initial_learning_rate = self.learning_rate
        initial_sigma = self.sigma
        
        for epoch in range(num_epochs):
            # Decay learning rate and neighborhood radius
            decay = 1 - epoch / num_epochs
            current_lr = initial_learning_rate * decay
            current_sigma = initial_sigma * decay
            
            # Random order for each epoch
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                sample = data[idx]
                
                # Find BMU (Best Matching Unit)
                bmu_idx = self._find_bmu_custom(sample)
                bmu_x, bmu_y = bmu_idx // self.y_dim, bmu_idx % self.y_dim
                
                # Update weights
                for i in range(self.x_dim):
                    for j in range(self.y_dim):
                        # Calculate distance in grid space
                        grid_dist = np.sqrt((i - bmu_x)**2 + (j - bmu_y)**2)
                        
                        # Neighborhood function (Gaussian)
                        if current_sigma > 0:
                            h = np.exp(-(grid_dist**2) / (2 * current_sigma**2))
                        else:
                            h = 1.0 if grid_dist == 0 else 0.0
                        
                        # Update weight
                        self.weights[i, j] += current_lr * h * (sample - self.weights[i, j])
            
            if verbose and (epoch + 1) % 10 == 0:
                qe = self.get_quantization_error(data)
                logger.info(f"Epoch {epoch+1}/{num_epochs}, QE: {qe:.4f}")
    
    def _find_bmu_custom(self, sample: np.ndarray) -> int:
        """Find Best Matching Unit for custom implementation."""
        distances = np.zeros((self.x_dim, self.y_dim))
        
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                distances[i, j] = np.linalg.norm(sample - self.weights[i, j])
        
        # Return flattened index
        return np.argmin(distances)
    
    def get_winner_coordinates(self, data: torch.Tensor) -> np.ndarray:
        """
        Get 2D grid coordinates for each input.
        
        Args:
            data: Input data tensor
            
        Returns:
            Array of (x, y) coordinates for each input
        """
        if not self.is_trained:
            raise RuntimeError("SOM not trained. Call train() first.")
        
        data_np = self._validate_input(data)
        coordinates = []
        
        if self._minisom_available and self.use_minisom:
            for sample in data_np:
                winner = self.som.winner(sample)
                coordinates.append(winner)
        else:
            for sample in data_np:
                bmu_idx = self._find_bmu_custom(sample)
                x, y = bmu_idx // self.y_dim, bmu_idx % self.y_dim
                coordinates.append([x, y])
        
        return np.array(coordinates)
    
    def get_quantization_error(self, data: Optional[Union[torch.Tensor, np.ndarray]] = None) -> float:
        """
        Compute average quantization error.
        
        Args:
            data: Input data (uses training data if None)
            
        Returns:
            Average distance between inputs and their BMUs
        """
        if not self.is_trained:
            raise RuntimeError("SOM not trained. Call train() first.")
        
        if data is not None:
            data_np = self._validate_input(data)
        else:
            raise ValueError("Data must be provided for quantization error calculation")
        
        if self._minisom_available and self.use_minisom:
            return float(self.som.quantization_error(data_np))
        else:
            # Custom implementation
            total_error = 0.0
            for sample in data_np:
                bmu_idx = self._find_bmu_custom(sample)
                bmu_x, bmu_y = bmu_idx // self.y_dim, bmu_idx % self.y_dim
                error = np.linalg.norm(sample - self.weights[bmu_x, bmu_y])
                total_error += error
            
            return total_error / len(data_np)
    
    def get_topographic_error(self, data: torch.Tensor) -> float:
        """
        Compute topographic preservation error.
        
        Measures how well the SOM preserves the topology of the input space.
        Error is the proportion of samples for which first and second BMUs are not adjacent.
        
        Args:
            data: Input data tensor
            
        Returns:
            Topographic error (0 = perfect preservation, 1 = no preservation)
        """
        if not self.is_trained:
            raise RuntimeError("SOM not trained. Call train() first.")
        
        data_np = self._validate_input(data)
        
        if self._minisom_available and self.use_minisom:
            return float(self.som.topographic_error(data_np))
        else:
            # Custom implementation
            errors = 0
            
            for sample in data_np:
                # Find first and second BMUs
                distances = np.zeros((self.x_dim, self.y_dim))
                for i in range(self.x_dim):
                    for j in range(self.y_dim):
                        distances[i, j] = np.linalg.norm(sample - self.weights[i, j])
                
                # Get sorted indices
                flat_distances = distances.flatten()
                sorted_indices = np.argsort(flat_distances)
                
                # Get coordinates of first and second BMUs
                bmu1_idx = sorted_indices[0]
                bmu2_idx = sorted_indices[1]
                
                x1, y1 = bmu1_idx // self.y_dim, bmu1_idx % self.y_dim
                x2, y2 = bmu2_idx // self.y_dim, bmu2_idx % self.y_dim
                
                # Check if they are adjacent
                grid_dist = abs(x1 - x2) + abs(y1 - y2)
                if grid_dist > 1:
                    errors += 1
            
            return errors / len(data_np)
    
    def get_u_matrix(self) -> np.ndarray:
        """
        Compute U-matrix (unified distance matrix).
        
        Shows the average distance between each node and its neighbors.
        High values indicate cluster boundaries.
        
        Returns:
            U-matrix of shape (x_dim, y_dim)
        """
        if not self.is_trained:
            raise RuntimeError("SOM not trained. Call train() first.")
        
        if self._minisom_available and self.use_minisom:
            # Use MiniSom's distance_map if available
            if hasattr(self.som, 'distance_map'):
                return self.som.distance_map()
        
        # Custom implementation
        u_matrix = np.zeros((self.x_dim, self.y_dim))
        
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                distances = []
                
                # Check all neighbors
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        
                        ni, nj = i + di, j + dj
                        
                        # Check boundaries
                        if 0 <= ni < self.x_dim and 0 <= nj < self.y_dim:
                            dist = np.linalg.norm(self.weights[i, j] - self.weights[ni, nj])
                            distances.append(dist)
                
                # Average distance to neighbors
                if distances:
                    u_matrix[i, j] = np.mean(distances)
        
        return u_matrix
    
    def get_component_planes(self) -> np.ndarray:
        """
        Get component planes showing the distribution of each feature.
        
        Returns:
            Array of shape (n_features, x_dim, y_dim)
        """
        if not self.is_trained:
            raise RuntimeError("SOM not trained. Call train() first.")
        
        if self.weights is None:
            if self._minisom_available and self.use_minisom:
                self.weights = self.som.get_weights()
            else:
                raise RuntimeError("Weights not available")
        
        # Reshape weights to get component planes
        return self.weights.transpose(2, 0, 1)
    
    def get_hit_map(self, data: torch.Tensor) -> np.ndarray:
        """
        Get hit map showing data distribution on the SOM grid.
        
        Args:
            data: Input data tensor
            
        Returns:
            Hit map of shape (x_dim, y_dim) with counts
        """
        if not self.is_trained:
            raise RuntimeError("SOM not trained. Call train() first.")
        
        data_np = self._validate_input(data)
        hit_map = np.zeros((self.x_dim, self.y_dim))
        
        # Count hits for each neuron
        coordinates = self.get_winner_coordinates(data_np)
        for x, y in coordinates:
            hit_map[int(x), int(y)] += 1
        
        return hit_map
    
    def project_to_2d(self, data: torch.Tensor, add_jitter: bool = True) -> np.ndarray:
        """
        Project data to 2D coordinates with optional jitter.
        
        Args:
            data: Input data tensor
            add_jitter: Add small random offset to avoid overlaps
            
        Returns:
            2D coordinates for visualization
        """
        coordinates = self.get_winner_coordinates(data)
        
        if add_jitter:
            # Add small random jitter
            jitter = np.random.normal(0, 0.1, coordinates.shape)
            coordinates = coordinates.astype(float) + jitter
        
        return coordinates
    
    def fallback_kmeans_grid(self, data: torch.Tensor) -> np.ndarray:
        """
        Fallback using KMeans + grid arrangement if SOM fails.
        
        Args:
            data: Input data tensor
            
        Returns:
            2D grid coordinates
        """
        from sklearn.cluster import KMeans
        
        data_np = self._validate_input(data)
        n_clusters = self.x_dim * self.y_dim
        
        # Perform KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed)
        labels = kmeans.fit_predict(data_np)
        
        # Arrange clusters on grid
        coordinates = np.zeros((len(data_np), 2))
        for i, label in enumerate(labels):
            x = label // self.y_dim
            y = label % self.y_dim
            # Add jitter
            coordinates[i] = [x + np.random.normal(0, 0.1), 
                            y + np.random.normal(0, 0.1)]
        
        logger.info("Used KMeans fallback for grid arrangement")
        return coordinates
    
    def save(self, path: str):
        """Save SOM model to file."""
        import pickle
        
        state = {
            'x_dim': self.x_dim,
            'y_dim': self.y_dim,
            'input_len': self.input_len,
            'weights': self.weights if self.weights is not None else 
                      (self.som.get_weights() if self.som else None),
            'is_trained': self.is_trained,
            'params': {
                'sigma': self.sigma,
                'learning_rate': self.learning_rate,
                'topology': self.topology,
                'activation_distance': self.activation_distance,
                'random_seed': self.random_seed
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved SOM model to {path}")
    
    def load(self, path: str):
        """Load SOM model from file."""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.x_dim = state['x_dim']
        self.y_dim = state['y_dim']
        self.input_len = state['input_len']
        self.weights = state['weights']
        self.is_trained = state['is_trained']
        
        # Restore parameters
        params = state.get('params', {})
        self.sigma = params.get('sigma', 1.0)
        self.learning_rate = params.get('learning_rate', 0.5)
        self.topology = params.get('topology', 'rectangular')
        self.activation_distance = params.get('activation_distance', 'euclidean')
        self.random_seed = params.get('random_seed', 42)
        
        # Reinitialize with loaded weights if using MiniSom
        if self._minisom_available and self.weights is not None:
            self._init_som(self.input_len)
            if self.som:
                # Set weights directly if MiniSom supports it
                if hasattr(self.som, '_weights'):
                    self.som._weights = self.weights
        
        logger.info(f"Loaded SOM model from {path}")


class TopographicalEvolutionTracker:
    """
    Track topographical changes and evolution over time.
    Monitors drift, trajectories, and regime changes in latent space projections.
    """
    
    def __init__(self, projector: TopographicalProjector,
                 window_size: int = 100,
                 memory_limit: int = 1000):
        """
        Initialize tracker with projection method.
        
        Args:
            projector: TopographicalProjector instance
            window_size: Size of sliding window for drift computation
            memory_limit: Maximum number of snapshots to keep in memory
        """
        self.projector = projector
        self.window_size = window_size
        self.memory_limit = memory_limit
        
        # Storage for snapshots
        self.snapshots = []
        self.projected_snapshots = []
        self.timestamps = []
        self.metadata = []
        
        # Cached computations
        self._trajectory_cache = None
        self._drift_cache = {}
        self._regime_changes = None
        
        logger.info(f"Initialized evolution tracker with window_size={window_size}")
    
    def add_snapshot(self, latents: torch.Tensor, 
                    timestamp: float, metadata: Dict = None):
        """
        Add new latent snapshot for tracking.
        
        Args:
            latents: Latent vectors to track
            timestamp: Time of snapshot
            metadata: Optional metadata for snapshot
        """
        # Project latents
        projected = self.projector.project_latents(latents)
        
        # Store snapshot
        self.snapshots.append(latents)
        self.projected_snapshots.append(projected)
        self.timestamps.append(timestamp)
        self.metadata.append(metadata or {})
        
        # Enforce memory limit
        if len(self.snapshots) > self.memory_limit:
            self.snapshots.pop(0)
            self.projected_snapshots.pop(0)
            self.timestamps.pop(0)
            self.metadata.pop(0)
        
        # Invalidate caches
        self._trajectory_cache = None
        self._drift_cache = {}
        self._regime_changes = None
        
        logger.debug(f"Added snapshot at t={timestamp}, total snapshots: {len(self.snapshots)}")
    
    def compute_trajectory(self, points: Optional[List[int]] = None) -> np.ndarray:
        """
        Compute movement trajectory in projected space.
        
        Args:
            points: Specific point indices to track (None = track centroids)
            
        Returns:
            Array of trajectory points (n_snapshots, n_points, 2)
        """
        if len(self.projected_snapshots) < 2:
            logger.warning("Need at least 2 snapshots for trajectory")
            return np.array([])
        
        if points is None:
            # Track centroids
            trajectory = []
            for projected in self.projected_snapshots:
                centroid = np.mean(projected, axis=0)
                trajectory.append(centroid)
            return np.array(trajectory)
        else:
            # Track specific points
            trajectory = []
            for projected in self.projected_snapshots:
                if max(points) < len(projected):
                    trajectory.append(projected[points])
                else:
                    # Handle missing points
                    valid_points = [p for p in points if p < len(projected)]
                    if valid_points:
                        trajectory.append(projected[valid_points])
            
            if trajectory:
                return np.array(trajectory)
            return np.array([])
    
    def compute_drift_metrics(self, window_start: Optional[int] = None,
                            window_end: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate various drift metrics.
        
        Args:
            window_start: Start index of window (None = use last window_size)
            window_end: End index of window
            
        Returns:
            Dictionary with drift metrics:
            - centroid_shift: distance between centroids
            - spread_change: change in data spread
            - density_shift: KL divergence of density estimates
            - trajectory_length: total path length
            - velocity: average movement speed
        """
        if len(self.projected_snapshots) < 2:
            return {
                'centroid_shift': 0.0,
                'spread_change': 0.0,
                'density_shift': 0.0,
                'trajectory_length': 0.0,
                'velocity': 0.0
            }
        
        # Determine window
        if window_start is None:
            window_start = max(0, len(self.projected_snapshots) - self.window_size)
        if window_end is None:
            window_end = len(self.projected_snapshots)
        
        # Get snapshots in window
        window_snapshots = self.projected_snapshots[window_start:window_end]
        window_timestamps = self.timestamps[window_start:window_end]
        
        if len(window_snapshots) < 2:
            return {
                'centroid_shift': 0.0,
                'spread_change': 0.0,
                'density_shift': 0.0,
                'trajectory_length': 0.0,
                'velocity': 0.0
            }
        
        metrics = {}
        
        # Centroid shift
        first_centroid = np.mean(window_snapshots[0], axis=0)
        last_centroid = np.mean(window_snapshots[-1], axis=0)
        metrics['centroid_shift'] = float(np.linalg.norm(last_centroid - first_centroid))
        
        # Spread change (using standard deviation)
        first_spread = np.std(window_snapshots[0])
        last_spread = np.std(window_snapshots[-1])
        metrics['spread_change'] = float(abs(last_spread - first_spread))
        
        # Density shift (KL divergence)
        try:
            from sklearn.neighbors import KernelDensity
            
            kde_first = KernelDensity(bandwidth=0.5)
            kde_last = KernelDensity(bandwidth=0.5)
            
            kde_first.fit(window_snapshots[0])
            kde_last.fit(window_snapshots[-1])
            
            # Sample points for KL divergence estimation
            n_samples = min(100, len(window_snapshots[0]))
            sample_indices = np.random.choice(len(window_snapshots[0]), n_samples, replace=False)
            sample_points = window_snapshots[0][sample_indices]
            
            log_p = kde_first.score_samples(sample_points)
            log_q = kde_last.score_samples(sample_points)
            
            # KL divergence approximation
            kl_div = np.mean(np.exp(log_p) * (log_p - log_q))
            metrics['density_shift'] = float(abs(kl_div))
        except:
            metrics['density_shift'] = 0.0
        
        # Trajectory length
        trajectory = self.compute_trajectory()
        if len(trajectory) > 1:
            path_length = 0.0
            for i in range(1, len(trajectory)):
                path_length += np.linalg.norm(trajectory[i] - trajectory[i-1])
            metrics['trajectory_length'] = float(path_length)
        else:
            metrics['trajectory_length'] = 0.0
        
        # Velocity (average speed)
        if len(window_timestamps) > 1:
            time_diff = window_timestamps[-1] - window_timestamps[0]
            if time_diff > 0:
                metrics['velocity'] = metrics['trajectory_length'] / time_diff
            else:
                metrics['velocity'] = 0.0
        else:
            metrics['velocity'] = 0.0
        
        return metrics
    
    def detect_regime_changes(self, threshold: float = 0.1,
                            method: str = 'gradient') -> List[int]:
        """
        Detect significant changes in topographical structure.
        
        Args:
            threshold: Threshold for detecting changes
            method: Detection method ('gradient', 'variance', 'clustering')
            
        Returns:
            List of snapshot indices where regime changes occur
        """
        if len(self.projected_snapshots) < 3:
            return []
        
        regime_changes = []
        
        if method == 'gradient':
            # Detect based on gradient of drift metrics
            gradients = []
            
            for i in range(1, len(self.projected_snapshots) - 1):
                # Compute local drift
                prev_centroid = np.mean(self.projected_snapshots[i-1], axis=0)
                curr_centroid = np.mean(self.projected_snapshots[i], axis=0)
                next_centroid = np.mean(self.projected_snapshots[i+1], axis=0)
                
                # Compute gradient
                grad1 = np.linalg.norm(curr_centroid - prev_centroid)
                grad2 = np.linalg.norm(next_centroid - curr_centroid)
                gradient_change = abs(grad2 - grad1)
                gradients.append(gradient_change)
            
            # Find peaks in gradient
            mean_grad = np.mean(gradients)
            std_grad = np.std(gradients)
            threshold_value = mean_grad + threshold * std_grad
            
            for i, grad in enumerate(gradients):
                if grad > threshold_value:
                    regime_changes.append(i + 1)
        
        elif method == 'variance':
            # Detect based on variance changes
            variances = []
            
            for snapshot in self.projected_snapshots:
                var = np.var(snapshot)
                variances.append(var)
            
            # Detect sudden changes in variance
            for i in range(1, len(variances)):
                var_change = abs(variances[i] - variances[i-1]) / (variances[i-1] + 1e-10)
                if var_change > threshold:
                    regime_changes.append(i)
        
        elif method == 'clustering':
            # Detect based on clustering structure changes
            from sklearn.cluster import KMeans
            
            n_clusters = 3  # Fixed number for comparison
            prev_labels = None
            
            for i, snapshot in enumerate(self.projected_snapshots):
                if len(snapshot) < n_clusters:
                    continue
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(snapshot)
                
                if prev_labels is not None:
                    # Compare clustering similarity (adjusted Rand index)
                    from sklearn.metrics import adjusted_rand_score
                    
                    # Need same-length arrays
                    min_len = min(len(labels), len(prev_labels))
                    if min_len > 0:
                        similarity = adjusted_rand_score(labels[:min_len], prev_labels[:min_len])
                        
                        if similarity < (1 - threshold):
                            regime_changes.append(i)
                
                prev_labels = labels
        
        self._regime_changes = regime_changes
        return regime_changes
    
    def compute_stability_score(self, window: Optional[int] = None) -> float:
        """
        Compute stability score for recent snapshots.
        
        Args:
            window: Number of recent snapshots to consider
            
        Returns:
            Stability score (0 = unstable, 1 = stable)
        """
        if window is None:
            window = min(self.window_size, len(self.projected_snapshots))
        
        if len(self.projected_snapshots) < 2:
            return 1.0
        
        # Get recent snapshots
        recent = self.projected_snapshots[-window:]
        
        # Compute pairwise distances between consecutive snapshots
        distances = []
        for i in range(1, len(recent)):
            # Use Procrustes distance for alignment-invariant comparison
            from scipy.spatial import procrustes
            _, _, disparity = procrustes(recent[i-1], recent[i])
            distances.append(disparity)
        
        if not distances:
            return 1.0
        
        # Convert to stability score (lower distance = higher stability)
        mean_distance = np.mean(distances)
        stability = np.exp(-mean_distance)  # Exponential decay
        
        return float(stability)
    
    def get_cluster_evolution(self, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Track cluster evolution over time.
        
        Args:
            n_clusters: Number of clusters to track
            
        Returns:
            Dictionary with cluster evolution information
        """
        from sklearn.cluster import KMeans
        
        evolution = {
            'cluster_centers': [],
            'cluster_sizes': [],
            'cluster_stability': [],
            'timestamps': self.timestamps.copy()
        }
        
        prev_centers = None
        
        for snapshot in self.projected_snapshots:
            if len(snapshot) < n_clusters:
                continue
            
            # Cluster current snapshot
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(snapshot)
            centers = kmeans.cluster_centers_
            
            # Compute cluster sizes
            sizes = [np.sum(labels == i) for i in range(n_clusters)]
            
            evolution['cluster_centers'].append(centers)
            evolution['cluster_sizes'].append(sizes)
            
            # Compute stability (how much centers moved)
            if prev_centers is not None:
                # Match clusters between snapshots
                from scipy.optimize import linear_sum_assignment
                
                # Cost matrix (distances between centers)
                cost_matrix = np.zeros((n_clusters, n_clusters))
                for i in range(n_clusters):
                    for j in range(n_clusters):
                        cost_matrix[i, j] = np.linalg.norm(centers[i] - prev_centers[j])
                
                # Find optimal matching
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                
                # Compute stability as inverse of movement
                total_movement = cost_matrix[row_ind, col_ind].sum()
                stability = 1.0 / (1.0 + total_movement)
                evolution['cluster_stability'].append(float(stability))
            else:
                evolution['cluster_stability'].append(1.0)
            
            prev_centers = centers
        
        return evolution
    
    def compute_path_length(self, point_indices: Optional[List[int]] = None) -> float:
        """
        Compute total path length in projected space.
        
        Args:
            point_indices: Specific points to track (None = centroid)
            
        Returns:
            Total path length
        """
        trajectory = self.compute_trajectory(point_indices)
        
        if len(trajectory) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(trajectory)):
            segment_length = np.linalg.norm(trajectory[i] - trajectory[i-1])
            total_length += segment_length
        
        return float(total_length)
    
    def identify_outlier_movements(self, threshold: float = 2.0) -> Dict[int, List[int]]:
        """
        Identify points with unusual movement patterns.
        
        Args:
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Dictionary mapping snapshot index to outlier point indices
        """
        if len(self.projected_snapshots) < 2:
            return {}
        
        outliers = {}
        
        for i in range(1, len(self.projected_snapshots)):
            prev = self.projected_snapshots[i-1]
            curr = self.projected_snapshots[i]
            
            # Compute movements for points that exist in both
            min_points = min(len(prev), len(curr))
            movements = []
            
            for j in range(min_points):
                movement = np.linalg.norm(curr[j] - prev[j])
                movements.append(movement)
            
            if movements:
                movements = np.array(movements)
                mean_movement = np.mean(movements)
                std_movement = np.std(movements)
                
                if std_movement > 0:
                    z_scores = (movements - mean_movement) / std_movement
                    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
                    
                    if len(outlier_indices) > 0:
                        outliers[i] = outlier_indices.tolist()
        
        return outliers
    
    def export_evolution_data(self) -> Dict[str, Any]:
        """
        Export evolution data for external analysis or visualization.
        
        Returns:
            Dictionary with all evolution data
        """
        return {
            'timestamps': self.timestamps.copy(),
            'projected_snapshots': [s.tolist() for s in self.projected_snapshots],
            'metadata': self.metadata.copy(),
            'drift_metrics': self.compute_drift_metrics(),
            'regime_changes': self.detect_regime_changes(),
            'stability_score': self.compute_stability_score(),
            'cluster_evolution': self.get_cluster_evolution(),
            'outlier_movements': self.identify_outlier_movements()
        }
    
    def save_state(self, path: str):
        """Save tracker state to file."""
        import pickle
        
        state = {
            'snapshots': self.snapshots,
            'projected_snapshots': self.projected_snapshots,
            'timestamps': self.timestamps,
            'metadata': self.metadata,
            'window_size': self.window_size,
            'memory_limit': self.memory_limit
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved evolution tracker state to {path}")
    
    def load_state(self, path: str):
        """Load tracker state from file."""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.snapshots = state['snapshots']
        self.projected_snapshots = state['projected_snapshots']
        self.timestamps = state['timestamps']
        self.metadata = state['metadata']
        self.window_size = state.get('window_size', 100)
        self.memory_limit = state.get('memory_limit', 1000)
        
        # Invalidate caches
        self._trajectory_cache = None
        self._drift_cache = {}
        self._regime_changes = None
        
        logger.info(f"Loaded evolution tracker state from {path}")


class ConceptSpaceNavigator:
    """
    Navigate and explore concept space using topographical representations.
    
    Provides methods for:
    - Finding similar concepts
    - Path finding between concepts
    - Concept interpolation
    - Boundary detection
    """
    
    def __init__(self, analyzer: TopographicalAnalyzer):
        """Initialize navigator with a fitted analyzer."""
        if analyzer.embedding is None:
            raise ValueError("Analyzer must have a fitted embedding")
        self.analyzer = analyzer
    
    def find_nearest_concepts(self,
                             query_point: np.ndarray,
                             k: int = 5,
                             return_distances: bool = True) -> Union[List[int], Tuple[List[int], np.ndarray]]:
        """
        Find k nearest concepts to a query point in embedded space.
        
        Args:
            query_point: Query point in embedded space
            k: Number of neighbors to find
            return_distances: Whether to return distances
            
        Returns:
            Indices of nearest concepts (and distances if requested)
        """
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(self.analyzer.embedding.embedded_vectors)
        
        distances, indices = nbrs.kneighbors([query_point])
        
        if return_distances:
            return indices[0].tolist(), distances[0]
        return indices[0].tolist()
    
    def find_path(self,
                 start_idx: int,
                 end_idx: int,
                 method: str = "geodesic") -> List[int]:
        """
        Find path between two concepts in topological space.
        
        Args:
            start_idx: Starting concept index
            end_idx: Ending concept index
            method: Path finding method ('geodesic', 'euclidean')
            
        Returns:
            List of indices forming the path
        """
        if self.analyzer.structure is None:
            self.analyzer.analyze_topology()
        
        if method == "geodesic" and self.analyzer.structure.adjacency_matrix is not None:
            # Use shortest path on graph
            try:
                from scipy.sparse.csgraph import shortest_path
                dist_matrix, predecessors = shortest_path(
                    self.analyzer.structure.adjacency_matrix,
                    directed=False,
                    return_predecessors=True
                )
                
                # Reconstruct path
                path = []
                current = end_idx
                while current != start_idx and current != -9999:
                    path.append(current)
                    current = predecessors[start_idx, current]
                
                if current == start_idx:
                    path.append(start_idx)
                    return list(reversed(path))
            except:
                logger.warning("Could not compute geodesic path")
        
        # Fallback to simple linear interpolation
        n_steps = 10
        start_vec = self.analyzer.embedding.embedded_vectors[start_idx]
        end_vec = self.analyzer.embedding.embedded_vectors[end_idx]
        
        path = [start_idx]
        for i in range(1, n_steps):
            alpha = i / n_steps
            interp_point = (1 - alpha) * start_vec + alpha * end_vec
            nearest_idx = self.find_nearest_concepts(interp_point, k=1, return_distances=False)[0]
            if nearest_idx not in path:
                path.append(nearest_idx)
        
        if end_idx not in path:
            path.append(end_idx)
        
        return path
    
    def interpolate_concepts(self,
                           indices: List[int],
                           weights: Optional[List[float]] = None) -> np.ndarray:
        """
        Interpolate between multiple concepts.
        
        Args:
            indices: Concept indices to interpolate
            weights: Weights for each concept (default: uniform)
            
        Returns:
            Interpolated point in embedded space
        """
        if weights is None:
            weights = np.ones(len(indices)) / len(indices)
        else:
            weights = np.array(weights) / np.sum(weights)
        
        vectors = self.analyzer.embedding.embedded_vectors[indices]
        interpolated = np.sum(vectors * weights[:, None], axis=0)
        
        return interpolated
    
    def find_boundaries(self,
                       resolution: int = 50,
                       method: str = "gradient") -> np.ndarray:
        """
        Find concept boundaries in embedded space.
        
        Args:
            resolution: Grid resolution for boundary detection
            method: Boundary detection method
            
        Returns:
            Binary mask indicating boundaries
        """
        # Create grid over embedded space
        embedded = self.analyzer.embedding.embedded_vectors
        x_min, x_max = embedded[:, 0].min(), embedded[:, 0].max()
        y_min, y_max = embedded[:, 1].min(), embedded[:, 1].max()
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # Compute density at each grid point
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(bandwidth=0.5)
        kde.fit(embedded)
        log_density = kde.score_samples(grid_points)
        density = np.exp(log_density).reshape(xx.shape)
        
        if method == "gradient":
            # Compute gradient magnitude
            gy, gx = np.gradient(density)
            gradient_mag = np.sqrt(gx**2 + gy**2)
            
            # Threshold to find boundaries
            threshold = np.percentile(gradient_mag, 90)
            boundaries = gradient_mag > threshold
        else:
            # Simple threshold on density
            threshold = np.percentile(density, 10)
            boundaries = density < threshold
        
        return boundaries


def create_topographical_analyzer(method: str = "umap",
                                 n_components: int = 2,
                                 **kwargs) -> TopographicalAnalyzer:
    """
    Convenience function to create a topographical analyzer.
    
    Args:
        method: Topographical method name
        n_components: Number of components
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured TopographicalAnalyzer
    """
    method_enum = TopographicalMethod(method.lower())
    config = TopographicalConfig(
        method=method_enum,
        n_components=n_components,
        **kwargs
    )
    return TopographicalAnalyzer(config)


def compare_topographies(embedding1: TopographicalEmbedding,
                        embedding2: TopographicalEmbedding,
                        metric: str = "procrustes") -> float:
    """
    Compare two topographical embeddings.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        metric: Comparison metric
        
    Returns:
        Similarity score (0-1, higher is more similar)
    """
    if metric == "procrustes":
        from scipy.spatial import procrustes
        _, _, disparity = procrustes(
            embedding1.embedded_vectors,
            embedding2.embedded_vectors
        )
        return 1.0 - min(1.0, disparity)
    
    elif metric == "correlation":
        # Correlation between pairwise distances
        from scipy.spatial.distance import pdist
        dist1 = pdist(embedding1.embedded_vectors)
        dist2 = pdist(embedding2.embedded_vectors)
        corr = np.corrcoef(dist1, dist2)[0, 1]
        return max(0, corr)
    
    else:
        raise ValueError(f"Unknown metric: {metric}")