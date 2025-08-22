# Topographical Learning for Semantic Verification

This module provides advanced topographical learning capabilities for visualizing and analyzing high-dimensional semantic spaces in the PoT (Proof-of-Training) framework.

## Overview

Topographical learning projects high-dimensional embeddings into lower-dimensional spaces (typically 2D or 3D) while preserving important structural relationships. This enables:

- **Visual Analysis**: Intuitive exploration of semantic spaces
- **Evolution Tracking**: Monitoring how representations change over time  
- **Quality Assessment**: Quantitative evaluation of projection methods
- **Cluster Discovery**: Identifying natural groupings in data
- **Drift Detection**: Detecting distributional shifts and concept drift

## Quick Start

```python
from pot.semantic import (
    TopographicalProjector,
    create_topographical_semantic_system,
    analyze_semantic_evolution
)
import torch

# Basic projection
projector = TopographicalProjector(method='umap')
embeddings = torch.randn(1000, 768)  # Your high-dimensional data
projected = projector.project_latents(embeddings)

# Integrated semantic system
library, matcher, projector = create_topographical_semantic_system(
    dim=768,
    projection_method='umap'
)

# Add concepts and visualize
library.add_concept('concept_a', torch.randn(100, 768))
library.add_concept('concept_b', torch.randn(100, 768))
positions = library.get_concept_positions(method='umap')
```

## Projection Methods

### 1. UMAP (Uniform Manifold Approximation and Projection)

**Best for**: General-purpose dimensionality reduction, preserving both local and global structure.

**Strengths**:
- Excellent balance of speed and quality
- Preserves both local neighborhoods and global structure
- Scales well to large datasets
- Robust to hyperparameter choices

**Weaknesses**:
- Can be slower than PCA for very large datasets
- May create false clusters in uniform data

**Key Parameters**:
```python
TopographicalProjector(
    method='umap',
    n_neighbors=15,      # More neighbors = more global structure
    min_dist=0.1,        # Smaller = tighter clusters
    metric='euclidean'   # Distance metric
)
```

**Parameter Tuning Guidelines**:
- `n_neighbors`: 5-100. Start with 15. Increase for more global structure, decrease for local details.
- `min_dist`: 0.0-1.0. Start with 0.1. Decrease for tighter clusters, increase for more spread.
- `metric`: Use 'cosine' for normalized vectors, 'euclidean' for general data.

**When to Use**:
- ✅ General-purpose visualization (recommended default)
- ✅ Large datasets (1000+ samples)
- ✅ Mixed local/global structure analysis
- ✅ When you need reproducible, stable results

### 2. t-SNE (t-Distributed Stochastic Neighbor Embedding)

**Best for**: Revealing fine-grained local cluster structure, publication-quality visualizations.

**Strengths**:
- Excellent at revealing local cluster structure
- Creates visually appealing, well-separated clusters
- Good for exploratory data analysis
- Strong theoretical foundation

**Weaknesses**:
- Computationally expensive (O(N²) complexity)
- Poor preservation of global structure
- Sensitive to hyperparameters
- Non-deterministic (random initialization effects)

**Key Parameters**:
```python
TopographicalProjector(
    method='tsne',
    perplexity=30,           # Effective number of neighbors
    learning_rate=200,       # Optimization step size
    n_iter=1000             # Number of optimization steps
)
```

**Parameter Tuning Guidelines**:
- `perplexity`: 5-50. Start with 30. Increase for larger datasets, decrease for small/sparse data.
- `learning_rate`: 10-1000. Start with 200. Increase if points cluster too much, decrease if too spread.
- `n_iter`: 250-5000. Start with 1000. Increase for better convergence, decrease for speed.

**When to Use**:
- ✅ Small to medium datasets (< 10,000 samples)
- ✅ When local cluster structure is most important
- ✅ Publication-quality visualizations
- ✅ Exploratory analysis of complex data

**Avoid When**:
- ❌ Very large datasets (> 10,000 samples)
- ❌ When global structure matters
- ❌ Time-critical applications

### 3. PCA (Principal Component Analysis)

**Best for**: Fast visualization, understanding global structure, baseline comparisons.

**Strengths**:
- Extremely fast and scalable
- Deterministic and reproducible
- Preserves global variance structure
- Well-understood mathematical properties
- Good preprocessing step for other methods

**Weaknesses**:
- Only captures linear relationships
- May miss important nonlinear structure
- Often produces less visually appealing clusters

**Key Parameters**:
```python
TopographicalProjector(
    method='pca',
    n_components=2,          # Number of output dimensions
    whiten=False            # Whether to normalize components
)
```

**Parameter Tuning Guidelines**:
- `n_components`: Usually 2 or 3 for visualization
- `whiten`: Set to True if you want equal variance in all components

**When to Use**:
- ✅ Very large datasets (> 100,000 samples)
- ✅ Real-time/interactive applications
- ✅ As preprocessing for other methods
- ✅ When linear structure is sufficient
- ✅ Baseline comparisons

**Ideal for**:
- Quick data exploration
- Preprocessing before UMAP/t-SNE
- When computational resources are limited

### 4. SOM (Self-Organizing Map)

**Best for**: Topographic mapping, preserving neighborhood relationships, interpretable clustering.

**Strengths**:
- Preserves topological relationships
- Provides interpretable grid structure
- Good for understanding data distribution
- Can handle streaming/online data

**Weaknesses**:
- Requires grid size selection
- Less flexible than other methods
- May distort complex manifolds

**Key Parameters**:
```python
from pot.semantic.topography import SOMProjector

som = SOMProjector(
    grid_size=(10, 10),      # Grid dimensions
    sigma=1.0,               # Neighborhood radius
    learning_rate=0.5        # Learning rate
)
```

**Parameter Tuning Guidelines**:
- `grid_size`: Start with (10, 10). Larger grids for more detail, smaller for overview.
- `sigma`: Start with 1.0. Larger values create smoother maps.
- `learning_rate`: 0.1-1.0. Start with 0.5.

**When to Use**:
- ✅ When topological preservation is crucial
- ✅ Interpretable clustering needed
- ✅ Online/streaming data
- ✅ Understanding data distribution patterns

## Quality Assessment

### Trustworthiness
Measures how well local neighborhoods are preserved from high to low dimensions.

- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: 
  - > 0.9: Excellent neighborhood preservation
  - 0.7-0.9: Good preservation
  - < 0.7: Poor preservation, consider different method/parameters

### Continuity  
Measures how well the low-dimensional neighborhoods reflect high-dimensional structure.

- **Range**: 0.0 to 1.0 (higher is better)
- **Interpretation**: Similar to trustworthiness but measures reverse direction

### Kruskal Stress
Measures how well distances are preserved between high and low dimensions.

- **Range**: 0.0 to 1.0 (lower is better)
- **Interpretation**:
  - < 0.1: Excellent distance preservation
  - 0.1-0.2: Good preservation
  - > 0.2: Poor preservation

### Shepard Correlation
Correlation between high-dimensional and low-dimensional distances.

- **Range**: -1.0 to 1.0 (higher is better)
- **Interpretation**:
  - > 0.8: Strong distance correlation
  - 0.5-0.8: Moderate correlation
  - < 0.5: Weak correlation

## Method Selection Guide

### Decision Tree

```
Dataset Size:
├── < 1,000 samples
│   ├── Local structure important? → t-SNE
│   └── Speed important? → PCA
├── 1,000 - 10,000 samples
│   ├── Balanced view needed? → UMAP (recommended)
│   ├── Fine local structure? → t-SNE  
│   └── Very fast needed? → PCA
└── > 10,000 samples
    ├── High quality needed? → UMAP with batching
    ├── Real-time required? → PCA
    └── Topological structure? → SOM
```

### Data Type Considerations

**Text Embeddings (BERT, etc.)**:
- Use UMAP with `metric='cosine'`
- Consider PCA preprocessing if dim > 1000

**Image Features (ResNet, etc.)**:
- UMAP with `metric='euclidean'` works well
- t-SNE for detailed cluster analysis
- PCA for quick overview

**Categorical Embeddings**:
- UMAP with `metric='hamming'` or `metric='jaccard'`
- Consider specialized distance metrics

**Time Series**:
- UMAP with dynamic time warping distance
- SOM for temporal pattern analysis

## Performance Optimization

### For Large Datasets

```python
# Use batched processing
from pot.semantic.topography_optimized import project_latents_batched

projected = project_latents_batched(
    embeddings,
    method='umap',
    batch_size=2000,
    use_gpu=True  # If RAPIDS available
)
```

### GPU Acceleration

```python
# Enable GPU acceleration (requires RAPIDS cuML)
projector = TopographicalProjector(method='umap')
projected = project_latents_batched(
    embeddings,
    method='umap',
    use_gpu=True
)
```

### Caching for Repeated Analysis

```python
from pot.semantic.topography_optimized import CachedProjector

base_projector = TopographicalProjector('umap')
cached_projector = CachedProjector(
    base_projector,
    cache_size=100,
    cache_dir='./projections_cache'
)
```

### Approximation for Speed

```python
from pot.semantic.topography_optimized import ApproximateProjector

# PCA preprocessing + approximate nearest neighbors
projector = ApproximateProjector(
    method='pca_umap',
    pca_components=50,
    approximate_nn=True
)
```

## Incremental/Online Learning

### Incremental UMAP

For streaming data or when you need to add new points to existing projections:

```python
from pot.semantic.topography_optimized import IncrementalUMAP

# Initialize
inc_umap = IncrementalUMAP(
    n_components=2,
    update_strategy='append'  # or 'merge', 'weighted'
)

# Initial fit
projection = inc_umap.fit(initial_data)

# Add new data incrementally
for new_batch in data_stream:
    new_projection = inc_umap.partial_fit(new_batch)
```

### Online SOM

For continuous learning scenarios:

```python
from pot.semantic.topography_optimized import OnlineSOM

# Initialize
online_som = OnlineSOM(
    grid_size=(15, 15),
    decay_function='exponential'
)

# Process data in batches
for batch in data_batches:
    online_som.partial_fit(batch, epochs=1)

# Get current projection
projection = online_som.transform(current_data)
```

## Evolution Tracking

Monitor how your embedding space changes over time:

```python
from pot.semantic import TopographicalEvolutionTracker

tracker = TopographicalEvolutionTracker()

# Add snapshots over time
for timestamp, embeddings in evolution_data:
    snapshot = projector.project_latents(embeddings)
    tracker.add_snapshot(
        snapshot,
        timestamp=timestamp,
        compute_metrics=True
    )

# Analyze evolution
drift_metrics = tracker.compute_drift_metrics()
regime_changes = tracker.detect_regime_changes(method='gradient')

print(f\"Cumulative drift: {drift_metrics['cumulative_drift']:.3f}\")
print(f\"Regime changes at: {regime_changes}\")
```

## Visualization Best Practices

### Static Plots

```python
from pot.semantic.topography_visualizer import plot_projection

fig = plot_projection(
    projection,
    labels=cluster_labels,
    title=\"UMAP Projection\",
    color_palette='husl',
    point_size=20,
    alpha=0.7
)
```

### Interactive Exploration

```python
from pot.semantic.topography_visualizer import create_interactive_plot

# Requires plotly
fig = create_interactive_plot(
    projection,
    labels=labels,
    hover_texts=hover_info,
    title=\"Interactive Embedding Explorer\"
)
fig.show()
```

### Animation for Evolution

```python
from pot.semantic.topography_visualizer import create_evolution_animation

# Create animation showing evolution over time
animation = create_evolution_animation(
    snapshots_list,
    timestamps,
    title=\"Semantic Space Evolution\"
)
```

## Integration with Semantic Verification

### Complete System

```python
# Create integrated system
library, matcher, projector = create_topographical_semantic_system(
    dim=768,
    projection_method='umap',
    cache_projections=True
)

# Add concepts
library.add_concept('positive', positive_embeddings)
library.add_concept('negative', negative_embeddings)

# Get spatial relationships
positions = library.get_concept_positions(method='umap')
distances = library.compute_concept_distances()

# Track semantic trajectories
trajectory = matcher.track_semantic_trajectory(
    embeddings_sequence,
    projection_method='umap',
    smooth=True
)
```

### Drift Detection in Production

```python
# Set up continuous monitoring
monitor = library.create_drift_monitor(
    baseline_embeddings=reference_data,
    drift_threshold=0.3,
    window_size=100
)

# In production loop
for new_embedding in embedding_stream:
    drift_score = monitor.update(new_embedding)
    if drift_score > 0.3:
        print(f\"Drift detected: {drift_score:.3f}\")
        # Trigger retraining or investigation
```

## Troubleshooting

### Common Issues

**1. Poor Clustering Quality**
- Try different methods: UMAP → t-SNE → PCA
- Adjust hyperparameters (especially `n_neighbors` for UMAP, `perplexity` for t-SNE)
- Check data preprocessing (normalization, scaling)
- Consider PCA preprocessing for high-dimensional data

**2. Slow Performance**
- Use PCA for quick exploration
- Enable GPU acceleration if available
- Use batched processing for large datasets
- Consider approximation methods
- Cache repeated projections

**3. Unstable Results**
- Set random seeds for reproducibility
- Use UMAP instead of t-SNE for more stable results
- Increase number of optimization iterations
- Check for data quality issues

**4. Memory Issues**
- Use batched processing
- Reduce batch size
- Use PCA preprocessing to reduce dimensionality
- Enable automatic garbage collection

### Performance Benchmarking

Use the built-in benchmarking suite:

```bash
python benchmarks/topography_benchmarks.py
```

This will generate comprehensive performance reports comparing all methods on your hardware.

## Advanced Usage

### Custom Distance Metrics

```python
from sklearn.metrics import pairwise_distances

def custom_distance(X, Y):
    # Your custom distance function
    return pairwise_distances(X, Y, metric='your_metric')

projector = TopographicalProjector(
    method='umap',
    metric=custom_distance
)
```

### Pipeline Composition

```python
# Create multi-stage pipelines
pipeline = [
    ('pca', {'n_components': 50}),
    ('umap', {'n_neighbors': 15, 'min_dist': 0.1})
]

from pot.semantic.topography_optimized import optimize_projection_pipeline
result = optimize_projection_pipeline(embeddings, pipeline=pipeline)
```

### Quality-Speed Tradeoffs

```python
# Automatic optimization for your use case
from pot.semantic.topography_utils import select_optimal_parameters

# Optimize for speed
fast_params = select_optimal_parameters(
    data, 
    method='umap',
    optimize_for='speed'
)

# Optimize for quality  
quality_params = select_optimal_parameters(
    data,
    method='umap', 
    optimize_for='quality'
)
```

## References and Further Reading

### Academic Papers
- **UMAP**: McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426
- **t-SNE**: Van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9(11)
- **SOM**: Kohonen, T. (1982). Self-organized formation of topologically correct feature maps. Biological Cybernetics, 43(1), 59-69

### Implementation Details
- **UMAP Implementation**: Uses `umap-learn` library with RAPIDS cuML for GPU acceleration
- **t-SNE Implementation**: Uses `scikit-learn` with OpenMP parallelization
- **SOM Implementation**: Custom implementation optimized for online learning

### Configuration
See `configs/topographical.yaml` for complete configuration options and detailed parameter descriptions.