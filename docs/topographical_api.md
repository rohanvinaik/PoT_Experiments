# Topographical Learning API Documentation

This document provides comprehensive API documentation for the topographical learning module in the PoT framework.

## Core Classes

### TopographicalProjector

**Location**: `pot.semantic.topography.TopographicalProjector`

The main interface for dimensionality reduction and topographical analysis.

```python
class TopographicalProjector:
    """
    Universal projector supporting multiple dimensionality reduction methods.
    Provides a unified interface for UMAP, t-SNE, PCA, and SOM projections.
    """
    
    def __init__(self, method: Union[str, TopographicalMethod] = 'umap', **kwargs)
    """
    Initialize projector with specified method.
    
    Args:
        method: Projection method ('umap', 'tsne', 'pca', 'som')
        **kwargs: Method-specific parameters
        
    Raises:
        ValueError: If method is not supported
        ImportError: If required dependencies are missing
    """
    
    def project_latents(self, latents: torch.Tensor, **kwargs) -> np.ndarray
    """
    Project high-dimensional data to lower dimensions.
    
    Args:
        latents: Input tensor of shape (n_samples, n_features)
        **kwargs: Additional projection parameters
        
    Returns:
        np.ndarray: Projected data of shape (n_samples, n_components)
        
    Raises:
        ValueError: If input dimensions are invalid
        RuntimeError: If projection fails
    """
```

**Usage Examples**:

```python
# Basic usage
projector = TopographicalProjector(method='umap')
projected = projector.project_latents(embeddings)

# With custom parameters
projector = TopographicalProjector(
    method='umap',
    n_neighbors=30,
    min_dist=0.01,
    n_components=3
)

# Method comparison
methods = ['pca', 'umap', 'tsne']
results = {}
for method in methods:
    proj = TopographicalProjector(method)
    results[method] = proj.project_latents(embeddings)
```

### TopographicalEvolutionTracker

**Location**: `pot.semantic.topography.TopographicalEvolutionTracker`

Tracks changes in embedding spaces over time.

```python
class TopographicalEvolutionTracker:
    """
    Tracks evolution of topographical structures over time.
    Detects drift, regime changes, and cluster evolution.
    """
    
    def add_snapshot(self, projection: np.ndarray, timestamp: float,
                    compute_metrics: bool = True) -> None
    """
    Add a new projection snapshot.
    
    Args:
        projection: 2D projection array of shape (n_samples, 2)
        timestamp: Timestamp for this snapshot
        compute_metrics: Whether to compute quality metrics
        
    Raises:
        ValueError: If projection shape is invalid
    """
    
    def compute_drift_metrics(self) -> Dict[str, Any]
    """
    Compute comprehensive drift metrics.
    
    Returns:
        Dict containing:
        - cumulative_drift: Total drift accumulated
        - centroid_shift: Movement of data center
        - spread_change: Changes in data spread
        - density_shift: Changes in local density
        
    Raises:
        RuntimeError: If insufficient snapshots available
    """
    
    def detect_regime_changes(self, method: str = 'gradient') -> List[int]
    """
    Detect regime changes in the evolution.
    
    Args:
        method: Detection method ('gradient', 'variance', 'clustering')
        
    Returns:
        List of snapshot indices where regime changes occurred
        
    Raises:
        ValueError: If method is not supported
    """
```

**Usage Examples**:

```python
tracker = TopographicalEvolutionTracker()

# Add snapshots over time
for t, embeddings in enumerate(embedding_history):
    projection = projector.project_latents(embeddings)
    tracker.add_snapshot(projection, timestamp=float(t))

# Analyze evolution
drift = tracker.compute_drift_metrics()
changes = tracker.detect_regime_changes()

print(f"Total drift: {drift['cumulative_drift']:.3f}")
print(f"Regime changes at: {changes}")
```

## Performance-Optimized Classes

### IncrementalUMAP

**Location**: `pot.semantic.topography_optimized.IncrementalUMAP`

UMAP implementation supporting incremental learning.

```python
class IncrementalUMAP:
    """
    Incremental UMAP for streaming data and online learning scenarios.
    Supports multiple update strategies for different use cases.
    """
    
    def __init__(self, n_components: int = 2, n_neighbors: int = 15,
                 min_dist: float = 0.1, metric: str = 'euclidean',
                 update_strategy: str = 'append', use_gpu: bool = False)
    """
    Initialize incremental UMAP.
    
    Args:
        n_components: Number of output dimensions (default: 2)
        n_neighbors: Number of neighbors for graph construction (default: 15)
        min_dist: Minimum distance in embedding space (default: 0.1)
        metric: Distance metric ('euclidean', 'cosine', etc.)
        update_strategy: How to handle new data ('append', 'merge', 'weighted')
        use_gpu: Use GPU acceleration if available (default: False)
        
    Raises:
        ImportError: If UMAP or RAPIDS dependencies missing for GPU
    """
    
    def fit(self, X: np.ndarray) -> np.ndarray
    """
    Initial fit on data.
    
    Args:
        X: Training data of shape (n_samples, n_features)
        
    Returns:
        Initial embedding of shape (n_samples, n_components)
        
    Raises:
        ValueError: If input data is invalid
    """
    
    def partial_fit(self, X_new: np.ndarray, 
                   sample_weight: Optional[np.ndarray] = None) -> np.ndarray
    """
    Update embedding with new data.
    
    Args:
        X_new: New data to incorporate of shape (n_new_samples, n_features)
        sample_weight: Optional weights for new samples
        
    Returns:
        Updated embedding including new points
        
    Raises:
        RuntimeError: If not fitted yet
        ValueError: If feature dimensions don't match
    """
```

**Update Strategies**:
- `'append'`: Simply add new points to existing embedding
- `'merge'`: Merge new points using graph connectivity
- `'weighted'`: Use sample weights to balance old vs new data

**Usage Examples**:

```python
# Basic incremental learning
inc_umap = IncrementalUMAP(update_strategy='append')
initial_embedding = inc_umap.fit(initial_data)

# Add new data streams
for new_batch in data_stream:
    updated_embedding = inc_umap.partial_fit(new_batch)

# Access embedding history
all_embeddings = inc_umap.embeddings_history
evolution_metrics = inc_umap.evolution_metrics
```

### OnlineSOM

**Location**: `pot.semantic.topography_optimized.OnlineSOM`

Self-Organizing Map with online learning capabilities.

```python
class OnlineSOM:
    """
    Online Self-Organizing Map for topological data representation.
    Supports streaming data with adaptive learning rates.
    """
    
    def __init__(self, grid_size: Tuple[int, int] = (10, 10),
                 learning_rate: float = 0.5, sigma: float = 1.0,
                 decay_function: str = 'exponential')
    """
    Initialize online SOM.
    
    Args:
        grid_size: SOM grid dimensions (width, height)
        learning_rate: Initial learning rate (default: 0.5)
        sigma: Initial neighborhood radius (default: 1.0)
        decay_function: Learning rate decay ('exponential', 'linear', 'inverse_time')
        
    Raises:
        ValueError: If grid_size is invalid
    """
    
    def partial_fit(self, X: np.ndarray, epochs: int = 1) -> None
    """
    Train SOM on new data batch.
    
    Args:
        X: Training data of shape (n_samples, n_features)
        epochs: Number of training epochs for this batch
        
    Raises:
        ValueError: If input shape is incompatible
    """
    
    def transform(self, X: np.ndarray) -> np.ndarray
    """
    Transform data to 2D SOM coordinates.
    
    Args:
        X: Data to transform of shape (n_samples, n_features)
        
    Returns:
        2D coordinates of shape (n_samples, 2)
        
    Raises:
        RuntimeError: If SOM not trained yet
    """
```

**Decay Functions**:
- `'exponential'`: learning_rate * exp(-epoch / time_constant)
- `'linear'`: learning_rate * (1 - epoch / max_epochs)
- `'inverse_time'`: learning_rate / (1 + epoch / time_constant)

### CachedProjector

**Location**: `pot.semantic.topography_optimized.CachedProjector`

Caching wrapper for expensive projections.

```python
class CachedProjector:
    """
    Caching wrapper for topographical projectors.
    Uses LRU cache with optional persistent storage.
    """
    
    def __init__(self, base_projector: Any, cache_size: int = 1000,
                 cache_dir: Optional[Path] = None)
    """
    Initialize cached projector.
    
    Args:
        base_projector: Underlying projector to wrap
        cache_size: Maximum number of cached projections
        cache_dir: Directory for persistent cache storage
        
    Raises:
        OSError: If cache directory cannot be created
    """
    
    def project_latents(self, latents: torch.Tensor, **kwargs) -> np.ndarray
    """
    Project with caching.
    
    Args:
        latents: Input embeddings to project
        **kwargs: Additional projection parameters
        
    Returns:
        Projected data (from cache or computed)
        
    Notes:
        Cache key based on content hash and parameters
    """
    
    def get_cache_stats(self) -> Dict[str, int]
    """
    Get cache performance statistics.
    
    Returns:
        Dict with 'hits', 'misses', 'size', 'max_size'
    """
    
    def clear_cache(self) -> None
    """Clear all cached projections."""
```

## Utility Functions

### Batch Processing

```python
def project_latents_batched(latents: torch.Tensor,
                           method: str = 'umap',
                           batch_size: int = 1000,
                           use_gpu: bool = False,
                           **kwargs) -> np.ndarray
"""
Memory-efficient batch processing for large datasets.

Args:
    latents: Input tensor of shape (n_samples, n_features)
    method: Projection method ('umap', 'tsne', 'pca')
    batch_size: Number of samples per batch
    use_gpu: Use GPU acceleration if available
    **kwargs: Method-specific parameters

Returns:
    Projected data of shape (n_samples, n_components)

Raises:
    ValueError: If batch_size <= 0 or method unsupported
    RuntimeError: If GPU requested but unavailable

Examples:
    # Process large dataset in batches
    large_data = torch.randn(100000, 768)
    projected = project_latents_batched(
        large_data,
        method='umap',
        batch_size=5000,
        use_gpu=True
    )
    
    # With custom parameters
    projected = project_latents_batched(
        data,
        method='umap',
        batch_size=2000,
        n_neighbors=30,
        min_dist=0.01
    )
"""
```

### Quality Assessment

```python
def compute_trustworthiness(X_high: np.ndarray, X_low: np.ndarray,
                           n_neighbors: int = 10) -> float
"""
Compute trustworthiness metric for projection quality.

Args:
    X_high: High-dimensional data of shape (n_samples, n_features)
    X_low: Low-dimensional projection of shape (n_samples, n_components)
    n_neighbors: Number of neighbors to consider

Returns:
    Trustworthiness score between 0.0 and 1.0 (higher is better)

Raises:
    ValueError: If shapes don't match or n_neighbors invalid

Notes:
    Measures preservation of local neighborhoods from high to low dimensions.
    Score > 0.9: Excellent, 0.7-0.9: Good, < 0.7: Poor
"""

def compute_continuity(X_high: np.ndarray, X_low: np.ndarray,
                      n_neighbors: int = 10) -> float
"""
Compute continuity metric for projection quality.

Args:
    X_high: High-dimensional data of shape (n_samples, n_features)
    X_low: Low-dimensional projection of shape (n_samples, n_components)
    n_neighbors: Number of neighbors to consider

Returns:
    Continuity score between 0.0 and 1.0 (higher is better)

Notes:
    Measures preservation of local neighborhoods from low to high dimensions.
    Complementary to trustworthiness metric.
"""

def compute_stress_metrics(X_high: np.ndarray, X_low: np.ndarray,
                          normalized: bool = True) -> Dict[str, float]
"""
Compute various stress metrics for distance preservation.

Args:
    X_high: High-dimensional data
    X_low: Low-dimensional projection
    normalized: Whether to normalize stress values

Returns:
    Dict containing:
    - kruskal_stress_1: Kruskal's stress formula 1
    - kruskal_stress_2: Kruskal's stress formula 2
    - sammon_stress: Sammon's stress
    - shepard_correlation: Shepard diagram correlation

Notes:
    Lower stress values indicate better distance preservation.
    Kruskal stress < 0.1: Excellent, 0.1-0.2: Good, > 0.2: Poor
"""
```

### Parameter Optimization

```python
def select_optimal_parameters(data: np.ndarray, method: str,
                             optimize_for: str = 'balanced') -> Dict[str, Any]
"""
Automatically select optimal parameters for projection methods.

Args:
    data: Input data for parameter estimation
    method: Projection method ('umap', 'tsne', 'som', 'pca')
    optimize_for: Optimization target ('speed', 'quality', 'balanced')

Returns:
    Dict of optimal parameters for the method

Raises:
    ValueError: If method or optimize_for not supported

Examples:
    # Optimize for speed
    fast_params = select_optimal_parameters(data, 'umap', 'speed')
    
    # Optimize for quality
    quality_params = select_optimal_parameters(data, 'umap', 'quality')
    
    # Use optimal parameters
    projector = TopographicalProjector('umap', **fast_params)
"""
```

## Visualization Functions

### Static Plotting

```python
def plot_projection(projection: np.ndarray,
                   labels: Optional[np.ndarray] = None,
                   title: str = "Projection",
                   color_palette: str = 'husl',
                   point_size: int = 20,
                   alpha: float = 0.7,
                   save_path: Optional[str] = None) -> 'matplotlib.figure.Figure'
"""
Create static scatter plot of 2D projection.

Args:
    projection: 2D projected data of shape (n_samples, 2)
    labels: Optional cluster labels for coloring
    title: Plot title
    color_palette: Seaborn color palette name
    point_size: Size of scatter points
    alpha: Point transparency
    save_path: Optional path to save figure

Returns:
    Matplotlib figure object

Examples:
    # Basic plot
    fig = plot_projection(projected_data)
    
    # With cluster labels
    fig = plot_projection(projected_data, labels=cluster_labels)
    
    # Customized appearance
    fig = plot_projection(
        projected_data,
        title="UMAP Projection",
        color_palette='viridis',
        point_size=30,
        save_path='projection.png'
    )
"""
```

### Interactive Visualization

```python
def create_interactive_plot(projection: np.ndarray,
                           labels: Optional[np.ndarray] = None,
                           hover_texts: Optional[List[str]] = None,
                           title: str = "Interactive Projection") -> 'plotly.graph_objects.Figure'
"""
Create interactive scatter plot using Plotly.

Args:
    projection: 2D projected data
    labels: Optional labels for coloring
    hover_texts: Optional hover information for each point
    title: Plot title

Returns:
    Plotly figure object with zoom, pan, selection capabilities

Requires:
    plotly package

Examples:
    # Basic interactive plot
    fig = create_interactive_plot(projected_data)
    fig.show()
    
    # With hover information
    hover_info = [f"Sample {i}" for i in range(len(projected_data))]
    fig = create_interactive_plot(
        projected_data,
        labels=cluster_labels,
        hover_texts=hover_info
    )
    
    # Save as HTML
    fig.write_html('interactive_plot.html')
"""
```

### Animation

```python
def create_evolution_animation(snapshots: List[np.ndarray],
                              timestamps: List[float],
                              title: str = "Evolution Animation",
                              frame_duration: int = 500) -> 'plotly.graph_objects.Figure'
"""
Create animated visualization of embedding evolution.

Args:
    snapshots: List of 2D projections over time
    timestamps: Corresponding timestamps
    title: Animation title
    frame_duration: Duration of each frame in milliseconds

Returns:
    Plotly figure with animation controls

Examples:
    # Create evolution animation
    animation = create_evolution_animation(
        snapshots=projection_history,
        timestamps=time_points,
        title="Semantic Space Evolution"
    )
    animation.show()
    
    # Save as HTML with controls
    animation.write_html('evolution.html')
"""
```

## Integration Functions

### Semantic System Integration

```python
def create_topographical_semantic_system(
    library_path: Optional[str] = None,
    projection_method: str = 'umap',
    **kwargs
) -> Tuple['ConceptLibrary', 'SemanticMatcher', 'TopographicalProjector']
"""
Create integrated topographical semantic verification system.

Args:
    library_path: Optional path to existing concept library
    projection_method: Default projection method
    **kwargs: Additional configuration options

Returns:
    Tuple of (ConceptLibrary, SemanticMatcher, TopographicalProjector)

Examples:
    # Create new system
    library, matcher, projector = create_topographical_semantic_system(
        dim=768,
        projection_method='umap'
    )
    
    # Load existing library
    library, matcher, projector = create_topographical_semantic_system(
        library_path='existing_library.pt',
        projection_method='tsne'
    )
"""

def analyze_semantic_evolution(embeddings_history: List[List[torch.Tensor]],
                              library: 'ConceptLibrary',
                              timestamps: Optional[List[float]] = None,
                              method: str = 'umap') -> Dict[str, Any]
"""
Analyze evolution of semantic representations over time.

Args:
    embeddings_history: List of embedding sets at different time points
    library: ConceptLibrary for reference
    timestamps: Optional timestamps for each snapshot
    method: Projection method for analysis

Returns:
    Dict with evolution metrics:
    - drift_metrics: Quantitative drift measurements
    - cluster_evolution: Cluster formation/dissolution
    - regime_changes: Detected change points
    - n_snapshots: Number of analyzed snapshots
    - total_drift: Cumulative drift score
    - stability: Overall stability measure

Examples:
    # Analyze evolution
    result = analyze_semantic_evolution(
        embeddings_history=model_embeddings_over_time,
        library=concept_library,
        timestamps=training_epochs,
        method='umap'
    )
    
    print(f"Total drift: {result['total_drift']:.3f}")
    print(f"Stability: {result['stability']:.3f}")
"""
```

## Configuration

### Method Parameters

**UMAP Parameters**:
- `n_neighbors` (5-100): Number of nearest neighbors. Higher values preserve more global structure.
- `min_dist` (0.0-1.0): Minimum distance between points in embedding. Lower values create tighter clusters.
- `metric`: Distance metric ('euclidean', 'cosine', 'manhattan', etc.)
- `n_epochs`: Number of optimization iterations (auto-selected based on data size)

**t-SNE Parameters**:
- `perplexity` (5-50): Effective number of nearest neighbors. Should be smaller than number of points.
- `learning_rate` (10-1000): Optimization learning rate. Higher values for larger datasets.
- `n_iter` (250-5000): Number of optimization iterations.
- `early_exaggeration` (4-50): Factor for early optimization phase.

**PCA Parameters**:
- `n_components`: Number of principal components to keep
- `whiten`: Whether to whiten the components (equal variance)
- `svd_solver`: SVD algorithm ('auto', 'full', 'arpack', 'randomized')

**SOM Parameters**:
- `grid_size`: Tuple of (width, height) for SOM grid
- `learning_rate`: Initial learning rate
- `sigma`: Initial neighborhood radius
- `decay_function`: How learning rate decays ('exponential', 'linear', 'inverse_time')

### Performance Optimization

**GPU Acceleration**:
- Requires RAPIDS cuML: `conda install -c rapidsai cuml`
- Set `use_gpu=True` in relevant functions
- 5-10x speedup for large datasets

**Memory Management**:
- Use batch processing for datasets > 10,000 samples
- Enable caching for repeated projections
- Consider PCA preprocessing for high-dimensional data

**Quality vs Speed Tradeoffs**:
- PCA: Fastest, linear relationships only
- UMAP: Good balance of speed and quality
- t-SNE: Highest quality for local structure, slowest

## Error Handling

Common exceptions and their meanings:

**ValueError**:
- Invalid input dimensions
- Unsupported method or parameters
- Incompatible data types

**RuntimeError**:
- Projection algorithm failed to converge
- GPU operations failed
- Insufficient memory

**ImportError**:
- Missing optional dependencies (UMAP, plotly, RAPIDS)
- Version compatibility issues

**Best Practices**:
1. Always check data shape and type before projection
2. Start with PCA for quick exploration
3. Use appropriate n_neighbors/perplexity for dataset size
4. Monitor memory usage for large datasets
5. Validate projection quality with metrics
6. Cache expensive projections for reuse

## Performance Guidelines

**Dataset Size Recommendations**:
- < 1,000 samples: Any method, t-SNE for best quality
- 1,000 - 10,000 samples: UMAP recommended
- 10,000 - 100,000 samples: UMAP with batching or PCA
- > 100,000 samples: PCA or GPU-accelerated UMAP

**Memory Requirements** (approximate):
- PCA: O(n_features * n_components)
- UMAP: O(n_samples * n_neighbors)
- t-SNE: O(n_samples²)
- SOM: O(grid_width * grid_height * n_features)

**Computational Complexity**:
- PCA: O(n_samples * n_features * n_components)
- UMAP: O(n_samples * log(n_samples) * n_neighbors)
- t-SNE: O(n_samples² * n_components)
- SOM: O(n_samples * n_epochs * grid_size)