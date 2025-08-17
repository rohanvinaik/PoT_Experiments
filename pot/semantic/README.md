# Semantic Verification Module

Advanced semantic analysis for neural network verification in the Proof-of-Training framework.

## Overview

The semantic verification module provides concept-based behavioral analysis that complements traditional distance metrics. It captures and analyzes the semantic meaning of model outputs, enabling detection of subtle behavioral changes that preserve numerical similarity but alter conceptual understanding.

## Architecture

```
pot/semantic/
├── __init__.py              # Module exports and API
├── types.py                 # Core data structures
├── library.py               # ConceptLibrary implementation
├── match.py                 # SemanticMatcher with similarity metrics
├── utils.py                 # Utility functions (embeddings, hypervectors)
├── config.py                # Configuration management
├── behavioral_fingerprint.py # Temporal pattern analysis
└── README.md                # This file
```

## Core Components

### 1. ConceptLibrary

Statistical representation of behavioral concepts using Gaussian distributions or hypervectors.

```python
from pot.semantic import ConceptLibrary

# Gaussian method for continuous embeddings
library = ConceptLibrary(dim=768, method='gaussian')

# Add concepts from training data
library.add_concept('normal_behavior', training_embeddings)
library.add_concept('edge_case', edge_case_embeddings)

# Hypervector method for discrete/binary representations
hv_library = ConceptLibrary(dim=512, method='hypervector')
```

**Key Features:**
- Dual representation methods (Gaussian/hypervector)
- Incremental concept updates
- Statistical modeling with covariance regularization
- Torch-based persistence

### 2. SemanticMatcher

Multi-metric similarity analysis and drift detection.

```python
from pot.semantic import SemanticMatcher

matcher = SemanticMatcher(library=library, threshold=0.7)

# Compute similarity to specific concept
similarity = matcher.compute_similarity(
    embedding, 
    concept='normal_behavior',
    method='cosine'  # or 'euclidean', 'mahalanobis', 'hamming'
)

# Match to entire library
matches = matcher.match_to_library(embedding)
best_concept, score = matcher.get_best_match(embedding)

# Detect semantic drift
drift_score = matcher.compute_semantic_drift(
    new_embeddings, 
    reference_concept='normal_behavior'
)
```

**Similarity Metrics:**
- **Cosine**: Direction-based similarity (scale-invariant)
- **Euclidean**: Absolute distance in embedding space
- **Mahalanobis**: Distribution-aware distance using covariance
- **Hamming**: Binary distance for hypervectors

### 3. BehavioralFingerprint

Temporal pattern capture with sliding windows and decay factors.

```python
from pot.semantic import BehavioralFingerprint

fingerprint = BehavioralFingerprint(
    window_size=50,
    fingerprint_dim=128,
    decay_factor=0.95
)

# Update with model outputs
for output in model_outputs:
    fingerprint.update(output, metadata={'timestamp': time.time()})

# Compute fingerprint
fp = fingerprint.compute_fingerprint(normalize=True)

# Detect anomalies
is_anomaly, score = fingerprint.detect_anomaly(fp, threshold=0.9)

# Detect drift
has_drift, drift_score = fingerprint.detect_drift(window=10)
```

**Key Features:**
- Sliding window observations
- Temporal decay weighting
- PCA/SVD dimensionality reduction
- Reference fingerprint comparison
- Alert callback system

### 4. ContinuousMonitor

Real-time monitoring with integrated semantic analysis.

```python
from pot.semantic import create_behavioral_monitor

monitor = create_behavioral_monitor(
    window_size=100,
    fingerprint_dim=128,
    semantic_library=library,
    alert_threshold=0.9,
    check_interval=10
)

# Process outputs in real-time
for output in output_stream:
    result = monitor.process_output(output)
    
    if 'anomaly_check' in result:
        if result['anomaly_check']['is_anomaly']:
            handle_anomaly(result)
    
    if 'drift_check' in result:
        if result['drift_check']['has_drift']:
            handle_drift(result)

# Get monitoring summary
summary = monitor.get_monitoring_summary()
```

## Integration with Verifiers

### Language Model Verifier

```python
from pot.lm.verifier import LMVerifier
from pot.semantic import ConceptLibrary

# Create semantic library from training
library = ConceptLibrary(dim=768, method='gaussian')
library.add_concept('formal_language', formal_embeddings)
library.add_concept('informal_language', informal_embeddings)

# Enhanced LM verification
verifier = LMVerifier(
    reference_model=ref_model,
    semantic_library=library,
    semantic_weight=0.3  # 30% semantic, 70% distance
)

result = verifier.verify(test_model, challenges)
print(f"Combined score: {result.combined_score:.3f}")
```

### Vision Model Verifier

```python
from pot.vision.verifier import VisionVerifier
from pot.semantic import ConceptLibrary

# Create visual concept library
library = ConceptLibrary(dim=512, method='gaussian')
library.add_concept('object_class_a', class_a_features)
library.add_concept('background', background_features)

# Enhanced vision verification
verifier = VisionVerifier(
    reference_model=ref_model,
    semantic_library=library,
    semantic_weight=0.3
)

result = verifier.verify(test_model, challenges)
```

## Configuration Management

```python
from pot.semantic import SemanticVerificationConfig

config = SemanticVerificationConfig(
    enabled=True,
    semantic_weight=0.4,
    library_method='gaussian',
    library_dimension=768,
    matching_threshold=0.7,
    matching_primary_method='cosine',
    lm_enabled=True,
    lm_semantic_weight=0.35,
    vision_enabled=True,
    vision_semantic_weight=0.3
)

# Validate configuration
config.validate()

# Create components from config
library, matcher = create_semantic_components(config)

# Integrate with existing verifier
integrate_with_verifier('lm', verifier, config)
```

## Hypervector Operations

High-dimensional computing with binary/ternary vectors.

```python
from pot.semantic.utils import (
    generate_random_hypervector,
    bind_hypervectors,
    bundle_hypervectors,
    compute_hamming_similarity
)

# Generate random hypervectors
hv1 = generate_random_hypervector(dim=10000, method='ternary')
hv2 = generate_random_hypervector(dim=10000, method='ternary')

# Binding (multiplication-like)
bound = bind_hypervectors(hv1, hv2)

# Bundling (addition-like)
bundle = bundle_hypervectors([hv1, hv2, bound])

# Similarity
similarity = compute_hamming_similarity(hv1, hv2)
```

## Drift Detection Methods

### Statistical Tests

```python
# Kolmogorov-Smirnov test for distribution shift
ks_statistic, ks_pvalue = matcher.compute_semantic_drift(
    new_embeddings, 
    reference_concept,
    method='ks'
)

# Wasserstein distance for distribution divergence
wasserstein_dist = matcher.compute_semantic_drift(
    new_embeddings,
    reference_concept, 
    method='wasserstein'
)
```

### Temporal Analysis

```python
# Sliding window drift detection
fingerprint = BehavioralFingerprint(window_size=50)

for output in output_stream:
    fingerprint.update(output)
    
    # Check every N updates
    if fingerprint.n_updates % 10 == 0:
        has_drift, score = fingerprint.detect_drift(
            window=20,  # Compare last 20 fingerprints
            threshold=0.3
        )
        
        if has_drift:
            print(f"Drift detected: {score:.3f}")
```

## Performance Optimization

### Memory Management

```python
# Limit concept library size
library = ConceptLibrary(dim=768, max_samples_per_concept=1000)

# Limit fingerprint history
fingerprint = BehavioralFingerprint(
    window_size=100,  # Sliding window
    history_max_size=500  # History limit
)

# Use PCA for dimensionality reduction
fingerprint = BehavioralFingerprint(
    fingerprint_dim=32,  # Reduced dimension
    use_pca=True,
    pca_components=32
)
```

### Computation Optimization

```python
# Batch processing
embeddings = extract_embeddings_batch(inputs, batch_size=32)

# Cached similarity computation
matcher = SemanticMatcher(library, cache_size=100)

# Approximate methods for large-scale
matcher.compute_similarity(embedding, concept, method='approximate_cosine')
```

## Advanced Usage

### Adaptive Thresholds

```python
monitor = ContinuousMonitor(
    fingerprint=fingerprint,
    adaptive_threshold=True,
    initial_threshold=0.8,
    adaptation_rate=0.01
)

# Threshold adjusts based on false positive rate
for output in output_stream:
    result = monitor.process_output(output)
    
    if result.get('threshold_adjusted'):
        print(f"New threshold: {result['new_threshold']:.3f}")
```

### Multi-Modal Concepts

```python
# Combined text and image concepts
text_embeddings = extract_text_embeddings(texts)
image_embeddings = extract_image_embeddings(images)

# Concatenate or average
combined = torch.cat([text_embeddings, image_embeddings], dim=-1)

library.add_concept('multimodal_concept', combined)
```

### Concept Clustering

```python
# Cluster model outputs to discover concepts
outputs = collect_model_outputs(model, inputs)
labels = matcher.cluster_outputs(outputs, n_clusters=5)

# Create concepts from clusters
for cluster_id in range(5):
    cluster_outputs = outputs[labels == cluster_id]
    library.add_concept(f'discovered_{cluster_id}', cluster_outputs)
```

## Testing

Run tests with:

```bash
# Unit tests
python -m pytest tests/test_semantic/ -v

# Specific component tests
python -m pytest tests/test_semantic/test_library.py -v
python -m pytest tests/test_semantic/test_match.py -v
python -m pytest tests/test_semantic/test_behavioral_fingerprint.py -v
python -m pytest tests/test_semantic/test_integration.py -v
```

## Examples

Complete working examples in `examples/semantic_verification/`:

- **basic_usage.py**: Creating and using concept libraries
- **integration_example.py**: Integration with LM/Vision verifiers
- **drift_detection.py**: Continuous monitoring and drift detection

## Performance Benchmarks

| Operation | Small (d=128) | Medium (d=768) | Large (d=2048) |
|-----------|---------------|----------------|----------------|
| Add concept (100 samples) | 5ms | 25ms | 120ms |
| Compute similarity | 0.5ms | 2ms | 8ms |
| Match to library (10 concepts) | 5ms | 20ms | 80ms |
| Drift detection (100 samples) | 10ms | 50ms | 200ms |
| Fingerprint update | 1ms | 3ms | 10ms |
| Fingerprint computation | 5ms | 15ms | 50ms |

## Best Practices

1. **Concept Library Design**
   - Use representative training data for concepts
   - Maintain balanced sample sizes across concepts
   - Regularly update concepts with new data
   - Use appropriate method (Gaussian vs hypervector)

2. **Similarity Metrics**
   - Cosine for direction-based similarity
   - Euclidean for absolute position
   - Mahalanobis for distribution-aware comparison
   - Hamming for discrete/binary features

3. **Drift Detection**
   - Set appropriate window sizes for your use case
   - Use multiple detection methods for robustness
   - Implement gradual threshold adaptation
   - Log all drift events for analysis

4. **Performance**
   - Batch operations when possible
   - Use dimensionality reduction for large embeddings
   - Cache frequently accessed similarities
   - Monitor memory usage with large histories

5. **Integration**
   - Start with low semantic weights (0.1-0.3)
   - Validate on held-out test sets
   - Monitor false positive/negative rates
   - Adjust thresholds based on deployment needs

## Troubleshooting

### Common Issues

**High memory usage:**
- Reduce window_size and history_max_size
- Enable PCA dimensionality reduction
- Limit samples per concept
- Use sparse representations for hypervectors

**Slow performance:**
- Batch embedding extraction
- Reduce fingerprint dimension
- Use approximate similarity methods
- Enable caching in matcher

**Poor drift detection:**
- Increase window size for more stable estimates
- Adjust detection threshold
- Use multiple drift metrics
- Ensure sufficient concept coverage

**Integration errors:**
- Verify dimension compatibility
- Check semantic_library is not None
- Ensure backward compatibility flags
- Validate configuration before use

## Future Enhancements

- [ ] GPU acceleration for large-scale operations
- [ ] Online learning for concept updates
- [ ] Hierarchical concept organization
- [ ] Cross-modal concept alignment
- [ ] Explainable semantic decisions
- [ ] Distributed monitoring systems
- [ ] Advanced anomaly detection algorithms
- [ ] Concept evolution tracking

## References

1. Kanerva, P. (2009). Hyperdimensional computing
2. Ge, R., et al. (2015). Gaussian mixture embeddings
3. Rabanser, S., et al. (2019). Failing loudly: Detecting dataset shift
4. Lipton, Z., et al. (2018). Detecting and correcting for label shift

## License

Part of the PoT Experiments framework under MIT License.