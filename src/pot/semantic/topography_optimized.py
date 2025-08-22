"""
Optimized topographical learning implementations for large-scale data.
Includes incremental learning, batch processing, and performance optimizations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from collections import OrderedDict, deque
import logging
import time
import hashlib
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# Check for GPU acceleration libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    from cuml.manifold import UMAP as cuUMAP
    from cuml.manifold import TSNE as cuTSNE
    from cuml.decomposition import PCA as cuPCA
    RAPIDS_AVAILABLE = True
except ImportError:
    RAPIDS_AVAILABLE = False


class IncrementalUMAP:
    """
    UMAP that can be updated with new data incrementally.
    Uses online learning techniques to update the embedding.
    """
    
    def __init__(self, n_components: int = 2, n_neighbors: int = 15,
                 min_dist: float = 0.1, metric: str = 'euclidean',
                 update_strategy: str = 'append', use_gpu: bool = False):
        """
        Initialize incremental UMAP.
        
        Args:
            n_components: Number of dimensions for projection
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance parameter
            metric: Distance metric to use
            update_strategy: How to update ('append', 'merge', 'weighted')
            use_gpu: Whether to use GPU acceleration if available
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.update_strategy = update_strategy
        self.use_gpu = use_gpu and RAPIDS_AVAILABLE
        
        self.embeddings_history = []
        self.data_history = []
        self.model = None
        self.is_fitted = False
        
        # For weighted updates
        self.sample_weights = []
        self.total_samples = 0
        
        logger.info(f"Initialized IncrementalUMAP (GPU: {self.use_gpu})")
    
    def fit(self, data: np.ndarray, sample_weight: Optional[np.ndarray] = None):
        """
        Initial fit of the UMAP model.
        
        Args:
            data: Initial data to fit
            sample_weight: Optional weights for samples
        """
        if self.use_gpu and RAPIDS_AVAILABLE:
            self.model = cuUMAP(
                n_components=self.n_components,
                n_neighbors=min(self.n_neighbors, len(data) - 1),
                min_dist=self.min_dist,
                metric=self.metric
            )
        else:
            try:
                from umap import UMAP
                self.model = UMAP(
                    n_components=self.n_components,
                    n_neighbors=min(self.n_neighbors, len(data) - 1),
                    min_dist=self.min_dist,
                    metric=self.metric,
                    random_state=42
                )
            except ImportError:
                raise ImportError("UMAP not installed. Install with: pip install umap-learn")
        
        # Fit initial model
        embedding = self.model.fit_transform(data)
        
        # Store history
        self.data_history.append(data)
        self.embeddings_history.append(embedding)
        if sample_weight is not None:
            self.sample_weights.append(sample_weight)
        else:
            self.sample_weights.append(np.ones(len(data)))
        
        self.total_samples = len(data)
        self.is_fitted = True
        
        logger.info(f"Fitted initial UMAP with {len(data)} samples")
        return embedding
    
    def partial_fit(self, new_data: np.ndarray, 
                   sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Incrementally update UMAP with new data.
        
        Args:
            new_data: New data points to add
            sample_weight: Optional weights for new samples
            
        Returns:
            Updated embeddings for all data
        """
        if not self.is_fitted:
            return self.fit(new_data, sample_weight)
        
        if self.update_strategy == 'append':
            return self._update_append(new_data, sample_weight)
        elif self.update_strategy == 'merge':
            return self._update_merge(new_data, sample_weight)
        elif self.update_strategy == 'weighted':
            return self._update_weighted(new_data, sample_weight)
        else:
            raise ValueError(f"Unknown update strategy: {self.update_strategy}")
    
    def _update_append(self, new_data: np.ndarray, 
                      sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Append new data and refit on combined dataset."""
        # Combine old and new data
        combined_data = np.vstack(self.data_history + [new_data])
        
        # Limit history size for memory
        max_history = 10000
        if len(combined_data) > max_history:
            # Keep most recent data
            combined_data = combined_data[-max_history:]
        
        # Refit model
        if hasattr(self.model, 'transform'):
            # Try to use transform for new points (faster)
            new_embedding = self.model.transform(new_data)
            
            # Update with weighted average
            if self.embeddings_history:
                old_embedding = np.vstack(self.embeddings_history)
                combined_embedding = np.vstack([old_embedding, new_embedding])
            else:
                combined_embedding = new_embedding
        else:
            # Full refit
            combined_embedding = self.model.fit_transform(combined_data)
        
        # Update history
        self.data_history.append(new_data)
        self.embeddings_history.append(new_embedding if 'new_embedding' in locals() 
                                      else combined_embedding[-len(new_data):])
        self.total_samples += len(new_data)
        
        logger.info(f"Updated UMAP with {len(new_data)} new samples (total: {self.total_samples})")
        return combined_embedding
    
    def _update_merge(self, new_data: np.ndarray,
                     sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Merge new data using landmark points."""
        # Select landmarks from existing data
        n_landmarks = min(100, len(self.data_history[0]))
        landmark_indices = np.random.choice(len(self.data_history[0]), 
                                          n_landmarks, replace=False)
        
        landmarks = self.data_history[0][landmark_indices]
        
        # Combine landmarks with new data
        combined = np.vstack([landmarks, new_data])
        
        # Fit on combined
        embedding = self.model.fit_transform(combined)
        
        # Separate embeddings
        landmark_embedding = embedding[:n_landmarks]
        new_embedding = embedding[n_landmarks:]
        
        # Interpolate old embeddings based on landmark movement
        old_embedding = self.embeddings_history[0]
        updated_old = self._interpolate_embeddings(
            old_embedding, landmark_indices, landmark_embedding
        )
        
        # Combine all embeddings
        combined_embedding = np.vstack([updated_old, new_embedding])
        
        # Update history
        self.data_history = [np.vstack([self.data_history[0], new_data])]
        self.embeddings_history = [combined_embedding]
        self.total_samples += len(new_data)
        
        return combined_embedding
    
    def _update_weighted(self, new_data: np.ndarray,
                        sample_weight: Optional[np.ndarray] = None) -> np.ndarray:
        """Update using weighted combination of old and new."""
        # Weight decay for old samples
        decay_factor = 0.9
        
        # Update weights
        for i in range(len(self.sample_weights)):
            self.sample_weights[i] *= decay_factor
        
        if sample_weight is not None:
            self.sample_weights.append(sample_weight)
        else:
            self.sample_weights.append(np.ones(len(new_data)))
        
        # Combine data with weights
        all_data = np.vstack(self.data_history + [new_data])
        all_weights = np.concatenate(self.sample_weights)
        
        # Weighted resampling for efficiency
        n_samples = min(5000, len(all_data))
        probabilities = all_weights / all_weights.sum()
        sample_indices = np.random.choice(len(all_data), n_samples, 
                                        p=probabilities, replace=True)
        
        sampled_data = all_data[sample_indices]
        
        # Refit on sampled data
        embedding = self.model.fit_transform(sampled_data)
        
        # Project all data
        full_embedding = self.model.transform(all_data)
        
        # Update history
        self.data_history.append(new_data)
        self.embeddings_history = [full_embedding]
        self.total_samples = len(all_data)
        
        return full_embedding
    
    def _interpolate_embeddings(self, old_embeddings: np.ndarray,
                               landmark_indices: np.ndarray,
                               landmark_new_positions: np.ndarray) -> np.ndarray:
        """Interpolate embeddings based on landmark movement."""
        from scipy.spatial import KDTree
        
        # Build KD-tree of old landmark positions
        old_landmarks = old_embeddings[landmark_indices]
        tree = KDTree(old_landmarks)
        
        # For each old point, find nearest landmarks
        updated = np.zeros_like(old_embeddings)
        
        for i, point in enumerate(old_embeddings):
            if i in landmark_indices:
                # Landmark - use new position
                landmark_idx = np.where(landmark_indices == i)[0][0]
                updated[i] = landmark_new_positions[landmark_idx]
            else:
                # Interpolate based on nearest landmarks
                distances, indices = tree.query(point, k=min(5, len(landmark_indices)))
                
                # Weighted average of landmark movements
                weights = 1.0 / (distances + 1e-10)
                weights /= weights.sum()
                
                movement = np.zeros(self.n_components)
                for w, idx in zip(weights, indices):
                    old_pos = old_landmarks[idx]
                    new_pos = landmark_new_positions[idx]
                    movement += w * (new_pos - old_pos)
                
                updated[i] = point + movement
        
        return updated


class OnlineSOM:
    """
    Self-Organizing Map with online learning capabilities.
    Can be updated incrementally with new data.
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (10, 10),
                 input_dim: Optional[int] = None,
                 learning_rate: float = 0.5,
                 sigma: float = 1.0,
                 decay_function: str = 'exponential'):
        """
        Initialize online SOM.
        
        Args:
            grid_size: Size of the SOM grid
            input_dim: Input dimension (set on first fit)
            learning_rate: Initial learning rate
            sigma: Initial neighborhood radius
            decay_function: How to decay learning rate ('exponential', 'linear')
        """
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.initial_learning_rate = learning_rate
        self.initial_sigma = sigma
        self.decay_function = decay_function
        
        self.weights = None
        self.iteration = 0
        self.is_initialized = False
        
        # For batch updates
        self.batch_data = []
        self.batch_size = 100
        
        logger.info(f"Initialized OnlineSOM with grid {grid_size}")
    
    def initialize_weights(self, data: np.ndarray):
        """Initialize SOM weights from data."""
        self.input_dim = data.shape[1]
        n_neurons = self.grid_size[0] * self.grid_size[1]
        
        # Initialize weights using PCA
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            pca.fit(data)
            
            # Create grid in PCA space
            x = np.linspace(-3, 3, self.grid_size[0])
            y = np.linspace(-3, 3, self.grid_size[1])
            xx, yy = np.meshgrid(x, y)
            grid_points = np.column_stack([xx.ravel(), yy.ravel()])
            
            # Transform back to data space
            if hasattr(pca, 'inverse_transform'):
                self.weights = pca.inverse_transform(grid_points)
            else:
                # Fallback to random initialization
                self.weights = np.random.randn(n_neurons, self.input_dim)
        except:
            # Random initialization
            self.weights = np.random.randn(n_neurons, self.input_dim)
        
        self.weights = self.weights.reshape(self.grid_size[0], 
                                           self.grid_size[1], 
                                           self.input_dim)
        self.is_initialized = True
    
    def partial_fit(self, data: np.ndarray, epochs: int = 1):
        """
        Update SOM with new data.
        
        Args:
            data: New data points
            epochs: Number of epochs to train
        """
        if not self.is_initialized:
            self.initialize_weights(data)
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(len(data))
            
            for idx in indices:
                self._update_weights(data[idx])
                self.iteration += 1
        
        logger.info(f"Updated SOM with {len(data)} samples")
    
    def _update_weights(self, sample: np.ndarray):
        """Update weights for a single sample."""
        # Find BMU (Best Matching Unit)
        bmu = self._find_bmu(sample)
        
        # Calculate current learning rate and sigma
        learning_rate = self._decay_parameter(
            self.initial_learning_rate, self.iteration
        )
        sigma = self._decay_parameter(
            self.initial_sigma, self.iteration
        )
        
        # Update weights
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                # Calculate distance to BMU
                dist = np.sqrt((i - bmu[0])**2 + (j - bmu[1])**2)
                
                # Calculate neighborhood function
                if dist <= sigma * 3:  # Only update within 3*sigma
                    influence = np.exp(-(dist**2) / (2 * sigma**2))
                    
                    # Update weight
                    self.weights[i, j] += (learning_rate * influence * 
                                          (sample - self.weights[i, j]))
    
    def _find_bmu(self, sample: np.ndarray) -> Tuple[int, int]:
        """Find Best Matching Unit for a sample."""
        # Vectorized distance computation
        distances = np.sum((self.weights - sample)**2, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx
    
    def _decay_parameter(self, initial_value: float, iteration: int) -> float:
        """Decay parameter over time."""
        if self.decay_function == 'exponential':
            decay_rate = 0.01
            return initial_value * np.exp(-decay_rate * iteration / 1000)
        elif self.decay_function == 'linear':
            min_value = initial_value * 0.01
            decay_rate = (initial_value - min_value) / 10000
            return max(min_value, initial_value - decay_rate * iteration)
        else:
            return initial_value
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """Project data onto SOM grid."""
        projections = []
        
        for sample in data:
            bmu = self._find_bmu(sample)
            projections.append(bmu)
        
        return np.array(projections)
    
    def batch_update(self, data: np.ndarray):
        """
        Batch update for efficiency.
        
        Args:
            data: New batch of data
        """
        self.batch_data.append(data)
        
        # Process when batch is full
        total_size = sum(len(d) for d in self.batch_data)
        if total_size >= self.batch_size:
            combined_data = np.vstack(self.batch_data)
            self.partial_fit(combined_data)
            self.batch_data = []


def project_latents_batched(latents: torch.Tensor,
                           batch_size: int = 1000,
                           method: str = 'umap',
                           use_gpu: bool = False,
                           **kwargs) -> np.ndarray:
    """
    Process large datasets in batches for memory efficiency.
    
    Args:
        latents: Input latent vectors
        batch_size: Size of batches to process
        method: Projection method ('umap', 'tsne', 'pca')
        use_gpu: Whether to use GPU if available
        **kwargs: Additional parameters for projection method
        
    Returns:
        Projected data
    """
    # Convert to numpy if needed
    if isinstance(latents, torch.Tensor):
        latents_np = latents.detach().cpu().numpy()
    else:
        latents_np = latents
    
    n_samples = len(latents_np)
    
    # For small datasets, process directly
    if n_samples <= batch_size * 2:
        return _project_single_batch(latents_np, method, use_gpu, **kwargs)
    
    logger.info(f"Processing {n_samples} samples in batches of {batch_size}")
    
    # Process in batches
    projections = []
    
    # First, fit on a subset to get the projection model
    subset_size = min(batch_size * 2, n_samples)
    subset_indices = np.random.choice(n_samples, subset_size, replace=False)
    subset_data = latents_np[subset_indices]
    
    # Fit model on subset
    model, subset_projection = _fit_projection_model(
        subset_data, method, use_gpu, **kwargs
    )
    
    # Store subset projections
    projection_dict = {idx: proj for idx, proj in 
                      zip(subset_indices, subset_projection)}
    
    # Transform remaining data in batches
    remaining_indices = [i for i in range(n_samples) if i not in subset_indices]
    
    for i in range(0, len(remaining_indices), batch_size):
        batch_indices = remaining_indices[i:i + batch_size]
        batch_data = latents_np[batch_indices]
        
        # Transform batch
        if hasattr(model, 'transform'):
            batch_projection = model.transform(batch_data)
        else:
            # For methods without transform, use approximation
            batch_projection = _approximate_projection(
                batch_data, subset_data, subset_projection, method
            )
        
        # Store projections
        for idx, proj in zip(batch_indices, batch_projection):
            projection_dict[idx] = proj
        
        logger.info(f"Processed batch {i//batch_size + 1}/{len(remaining_indices)//batch_size + 1}")
    
    # Assemble final projections in correct order
    final_projections = np.array([projection_dict[i] for i in range(n_samples)])
    
    return final_projections


def _project_single_batch(data: np.ndarray, method: str, 
                         use_gpu: bool, **kwargs) -> np.ndarray:
    """Project a single batch of data."""
    if use_gpu and RAPIDS_AVAILABLE:
        return _project_gpu(data, method, **kwargs)
    else:
        return _project_cpu(data, method, **kwargs)


def _project_gpu(data: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """Project using GPU acceleration with RAPIDS."""
    if method == 'umap':
        model = cuUMAP(
            n_components=kwargs.get('n_components', 2),
            n_neighbors=min(kwargs.get('n_neighbors', 15), len(data) - 1),
            min_dist=kwargs.get('min_dist', 0.1)
        )
    elif method == 'tsne':
        model = cuTSNE(
            n_components=kwargs.get('n_components', 2),
            perplexity=min(kwargs.get('perplexity', 30), len(data) // 4)
        )
    elif method == 'pca':
        model = cuPCA(
            n_components=kwargs.get('n_components', 2)
        )
    else:
        raise ValueError(f"GPU acceleration not available for {method}")
    
    return model.fit_transform(data)


def _project_cpu(data: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """Project using CPU."""
    if method == 'umap':
        from umap import UMAP
        model = UMAP(
            n_components=kwargs.get('n_components', 2),
            n_neighbors=min(kwargs.get('n_neighbors', 15), len(data) - 1),
            min_dist=kwargs.get('min_dist', 0.1),
            random_state=42
        )
        return model.fit_transform(data)
    
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        model = TSNE(
            n_components=kwargs.get('n_components', 2),
            perplexity=min(kwargs.get('perplexity', 30), len(data) // 4),
            random_state=42
        )
        return model.fit_transform(data)
    
    elif method == 'pca':
        from sklearn.decomposition import PCA
        model = PCA(
            n_components=kwargs.get('n_components', 2),
            random_state=42
        )
        return model.fit_transform(data)
    
    else:
        raise ValueError(f"Unknown method: {method}")


def _fit_projection_model(data: np.ndarray, method: str, 
                         use_gpu: bool, **kwargs) -> Tuple[Any, np.ndarray]:
    """Fit projection model and return model with projections."""
    if use_gpu and RAPIDS_AVAILABLE:
        if method == 'umap':
            model = cuUMAP(
                n_components=kwargs.get('n_components', 2),
                n_neighbors=min(kwargs.get('n_neighbors', 15), len(data) - 1),
                min_dist=kwargs.get('min_dist', 0.1)
            )
        elif method == 'tsne':
            model = cuTSNE(
                n_components=kwargs.get('n_components', 2),
                perplexity=min(kwargs.get('perplexity', 30), len(data) // 4)
            )
        elif method == 'pca':
            model = cuPCA(n_components=kwargs.get('n_components', 2))
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        if method == 'umap':
            from umap import UMAP
            model = UMAP(
                n_components=kwargs.get('n_components', 2),
                n_neighbors=min(kwargs.get('n_neighbors', 15), len(data) - 1),
                min_dist=kwargs.get('min_dist', 0.1),
                random_state=42
            )
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            model = TSNE(
                n_components=kwargs.get('n_components', 2),
                perplexity=min(kwargs.get('perplexity', 30), len(data) // 4),
                random_state=42
            )
        elif method == 'pca':
            from sklearn.decomposition import PCA
            model = PCA(
                n_components=kwargs.get('n_components', 2),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    projection = model.fit_transform(data)
    return model, projection


def _approximate_projection(new_data: np.ndarray, 
                           reference_data: np.ndarray,
                           reference_projection: np.ndarray,
                           method: str) -> np.ndarray:
    """Approximate projection for new data based on reference."""
    from sklearn.neighbors import KNeighborsRegressor
    
    # Use k-NN regression to approximate projection
    k = min(10, len(reference_data))
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn.fit(reference_data, reference_projection)
    
    return knn.predict(new_data)


class CachedProjector:
    """
    Cache projection results to avoid recomputation.
    Uses LRU cache with content-based hashing.
    """
    
    def __init__(self, base_projector: Any, cache_size: int = 1000,
                 cache_dir: Optional[Path] = None):
        """
        Initialize cached projector.
        
        Args:
            base_projector: Base projector to wrap
            cache_size: Maximum number of cached results
            cache_dir: Optional directory for persistent cache
        """
        self.base_projector = base_projector
        self.cache_size = cache_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # In-memory LRU cache
        self.cache = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Create cache directory if needed
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_persistent_cache()
        
        logger.info(f"Initialized CachedProjector with size {cache_size}")
    
    def project_latents(self, latents: Union[torch.Tensor, np.ndarray],
                       **kwargs) -> np.ndarray:
        """
        Project latents with caching.
        
        Args:
            latents: Input data
            **kwargs: Additional parameters
            
        Returns:
            Projected data
        """
        # Generate cache key
        cache_key = self._generate_cache_key(latents, **kwargs)
        
        # Check cache
        if cache_key in self.cache:
            self.cache_hits += 1
            # Move to end (most recently used)
            self.cache.move_to_end(cache_key)
            logger.debug(f"Cache hit (rate: {self.get_hit_rate():.2%})")
            return self.cache[cache_key].copy()
        
        # Check persistent cache
        if self.cache_dir:
            persistent_result = self._check_persistent_cache(cache_key)
            if persistent_result is not None:
                self.cache_hits += 1
                self._update_cache(cache_key, persistent_result)
                return persistent_result.copy()
        
        # Cache miss - compute projection
        self.cache_misses += 1
        logger.debug(f"Cache miss (rate: {self.get_hit_rate():.2%})")
        
        start_time = time.time()
        result = self.base_projector.project_latents(latents, **kwargs)
        compute_time = time.time() - start_time
        
        # Update cache
        self._update_cache(cache_key, result)
        
        # Save to persistent cache
        if self.cache_dir:
            self._save_to_persistent_cache(cache_key, result)
        
        logger.info(f"Computed projection in {compute_time:.2f}s")
        
        return result
    
    def _generate_cache_key(self, latents: Union[torch.Tensor, np.ndarray],
                           **kwargs) -> str:
        """Generate hash-based cache key."""
        # Convert to numpy for consistent hashing
        if isinstance(latents, torch.Tensor):
            data_bytes = latents.detach().cpu().numpy().tobytes()
        else:
            data_bytes = latents.tobytes()
        
        # Hash data and parameters
        hasher = hashlib.sha256()
        hasher.update(data_bytes)
        
        # Add parameters to hash
        param_str = str(sorted(kwargs.items()))
        hasher.update(param_str.encode())
        
        return hasher.hexdigest()[:16]
    
    def _update_cache(self, key: str, value: np.ndarray):
        """Update LRU cache."""
        # Add to cache
        self.cache[key] = value.copy()
        
        # Evict oldest if cache is full
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
    
    def _check_persistent_cache(self, key: str) -> Optional[np.ndarray]:
        """Check persistent cache on disk."""
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except:
                # Corrupted cache file
                cache_file.unlink()
        return None
    
    def _save_to_persistent_cache(self, key: str, value: np.ndarray):
        """Save to persistent cache on disk."""
        cache_file = self.cache_dir / f"{key}.npy"
        try:
            np.save(cache_file, value)
        except Exception as e:
            logger.warning(f"Failed to save to persistent cache: {e}")
    
    def _load_persistent_cache(self):
        """Load persistent cache metadata."""
        # Load most recent files into memory cache
        cache_files = sorted(
            self.cache_dir.glob("*.npy"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:self.cache_size // 2]
        
        for cache_file in cache_files:
            key = cache_file.stem
            try:
                value = np.load(cache_file)
                self.cache[key] = value
            except:
                # Remove corrupted file
                cache_file.unlink()
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0
    
    def clear_cache(self):
        """Clear all caches."""
        self.cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0
        
        if self.cache_dir:
            for cache_file in self.cache_dir.glob("*.npy"):
                cache_file.unlink()
        
        logger.info("Cleared all caches")


class ApproximateProjector:
    """
    Fast approximate projections using various techniques.
    """
    
    def __init__(self, method: str = 'pca_umap', 
                 initial_dim: int = 50,
                 use_annoy: bool = True):
        """
        Initialize approximate projector.
        
        Args:
            method: Approximation method
            initial_dim: Dimension for initial reduction
            use_annoy: Whether to use Annoy for approximate nearest neighbors
        """
        self.method = method
        self.initial_dim = initial_dim
        self.use_annoy = use_annoy
        
        self.pca_model = None
        self.projection_model = None
        
        logger.info(f"Initialized ApproximateProjector with method {method}")
    
    def fit_transform(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Fit and transform data with approximations.
        
        Args:
            data: Input data
            **kwargs: Additional parameters
            
        Returns:
            Projected data
        """
        n_samples, n_features = data.shape
        
        # Step 1: Initial dimensionality reduction with PCA
        if n_features > self.initial_dim and n_samples > self.initial_dim:
            from sklearn.decomposition import PCA
            
            self.pca_model = PCA(n_components=self.initial_dim, random_state=42)
            data_reduced = self.pca_model.fit_transform(data)
            logger.info(f"Reduced dimensions from {n_features} to {self.initial_dim}")
        else:
            data_reduced = data
        
        # Step 2: Apply main projection method
        if 'umap' in self.method:
            projection = self._approximate_umap(data_reduced, **kwargs)
        elif 'tsne' in self.method:
            projection = self._approximate_tsne(data_reduced, **kwargs)
        else:
            projection = data_reduced[:, :2]  # Just take first 2 dimensions
        
        return projection
    
    def _approximate_umap(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Approximate UMAP with optimizations."""
        try:
            from umap import UMAP
            
            # Use approximate nearest neighbors
            if self.use_annoy:
                try:
                    from annoy import AnnoyIndex
                    
                    # Build Annoy index
                    dim = data.shape[1]
                    metric = kwargs.get('metric', 'euclidean')
                    annoy_metric = 'euclidean' if metric == 'euclidean' else 'angular'
                    
                    index = AnnoyIndex(dim, annoy_metric)
                    for i, vec in enumerate(data):
                        index.add_item(i, vec)
                    index.build(10)  # 10 trees
                    
                    # Use with UMAP (if supported)
                    kwargs['metric'] = 'precomputed'
                    
                    # Compute distance matrix using Annoy
                    n_neighbors = kwargs.get('n_neighbors', 15)
                    distances = np.zeros((len(data), len(data)))
                    
                    for i in range(len(data)):
                        neighbors, dists = index.get_nns_by_item(
                            i, min(n_neighbors * 2, len(data)), 
                            include_distances=True
                        )
                        for j, d in zip(neighbors, dists):
                            distances[i, j] = d
                            distances[j, i] = d
                    
                    data = distances
                    logger.info("Using Annoy for approximate nearest neighbors")
                    
                except ImportError:
                    logger.info("Annoy not available, using exact nearest neighbors")
            
            # Configure UMAP for speed
            model = UMAP(
                n_components=kwargs.get('n_components', 2),
                n_neighbors=min(kwargs.get('n_neighbors', 15), len(data) - 1),
                min_dist=kwargs.get('min_dist', 0.1),
                n_epochs=kwargs.get('n_epochs', 200),  # Fewer epochs for speed
                init='random',  # Faster than spectral
                random_state=42
            )
            
            return model.fit_transform(data)
            
        except ImportError:
            logger.warning("UMAP not available, falling back to PCA")
            return data[:, :2]
    
    def _approximate_tsne(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Approximate t-SNE with optimizations."""
        try:
            from sklearn.manifold import TSNE
            
            # Use Barnes-Hut approximation
            model = TSNE(
                n_components=kwargs.get('n_components', 2),
                perplexity=min(kwargs.get('perplexity', 30), len(data) // 4),
                method='barnes_hut',  # O(n log n) instead of O(n^2)
                angle=kwargs.get('angle', 0.5),  # Trade accuracy for speed
                n_iter=kwargs.get('n_iter', 250),  # Fewer iterations
                random_state=42
            )
            
            return model.fit_transform(data)
            
        except ImportError:
            logger.warning("t-SNE not available, falling back to PCA")
            return data[:, :2]
    
    def transform(self, new_data: np.ndarray) -> np.ndarray:
        """
        Transform new data using fitted models.
        
        Args:
            new_data: New data to transform
            
        Returns:
            Projected data
        """
        # Apply PCA if fitted
        if self.pca_model is not None:
            new_data = self.pca_model.transform(new_data)
        
        # Use nearest neighbor approximation
        if self.projection_model is not None and hasattr(self.projection_model, 'transform'):
            return self.projection_model.transform(new_data)
        else:
            # Fallback to simple interpolation
            from sklearn.neighbors import KNeighborsRegressor
            
            if hasattr(self, 'training_data') and hasattr(self, 'training_projection'):
                knn = KNeighborsRegressor(n_neighbors=5)
                knn.fit(self.training_data, self.training_projection)
                return knn.predict(new_data)
            else:
                return new_data[:, :2]


def optimize_projection_pipeline(data: Union[torch.Tensor, np.ndarray],
                                method: str = 'auto',
                                target_dim: int = 2,
                                max_samples: int = 10000,
                                use_gpu: bool = None) -> np.ndarray:
    """
    Optimized projection pipeline with automatic method selection.
    
    Args:
        data: Input data
        method: Projection method or 'auto' for automatic selection
        target_dim: Target dimensionality
        max_samples: Maximum samples to use for fitting
        use_gpu: Whether to use GPU (None for auto-detect)
        
    Returns:
        Projected data
    """
    # Convert to numpy
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = data
    
    n_samples, n_features = data_np.shape
    
    # Auto-detect GPU usage
    if use_gpu is None:
        use_gpu = RAPIDS_AVAILABLE and n_samples > 5000
    
    # Auto-select method
    if method == 'auto':
        if n_samples < 1000:
            method = 'pca' if n_features > 50 else 'umap'
        elif n_samples < 5000:
            method = 'umap'
        else:
            method = 'pca_umap'  # PCA followed by UMAP
    
    logger.info(f"Using method: {method}, GPU: {use_gpu}")
    
    # Downsample if needed
    if n_samples > max_samples:
        sample_indices = np.random.choice(n_samples, max_samples, replace=False)
        fitting_data = data_np[sample_indices]
        logger.info(f"Downsampled from {n_samples} to {max_samples} for fitting")
    else:
        fitting_data = data_np
        sample_indices = None
    
    # Apply projection
    if method == 'pca_umap':
        # Two-stage projection
        projector = ApproximateProjector(
            method='pca_umap',
            initial_dim=min(50, n_features // 2)
        )
        
        if sample_indices is not None:
            # Fit on subset, transform all
            fitting_projection = projector.fit_transform(fitting_data)
            
            # Store for transform
            projector.training_data = fitting_data
            projector.training_projection = fitting_projection
            
            # Transform all data
            projection = projector.transform(data_np)
        else:
            projection = projector.fit_transform(fitting_data)
    
    elif use_gpu and RAPIDS_AVAILABLE:
        projection = project_latents_batched(
            data_np,
            batch_size=min(5000, n_samples // 2),
            method=method.replace('pca_', ''),
            use_gpu=True,
            n_components=target_dim
        )
    
    else:
        projection = project_latents_batched(
            data_np,
            batch_size=min(2000, n_samples // 2),
            method=method.replace('pca_', ''),
            use_gpu=False,
            n_components=target_dim
        )
    
    return projection


# Export main classes and functions
__all__ = [
    'IncrementalUMAP',
    'OnlineSOM',
    'project_latents_batched',
    'CachedProjector',
    'ApproximateProjector',
    'optimize_projection_pipeline'
]