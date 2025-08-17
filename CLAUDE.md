# Claude Instructions for PoT Experiments

Proof-of-Training (PoT) framework for behavioral verification of neural networks using cryptographic techniques and machine learning.

## Project Structure
```
PoT_Experiments/
├── pot/                     # Core implementation
│   ├── core/               # Framework fundamentals
│   ├── vision/             # Vision model verification
│   ├── lm/                 # Language model verification
│   ├── semantic/           # Semantic verification (NEW 2025-08-17)
│   ├── security/           # Security and verification protocols
│   ├── audit/              # Audit infrastructure
│   ├── eval/               # Evaluation utilities
│   └── prototypes/         # Experimental features
├── configs/                # YAML configurations
├── scripts/                # Experiment runners
├── outputs/                # Results (auto-created)
└── verification_reports/   # Compliance reports
```

## Core Components

### 1. Core Framework (`pot/core/`)
- **Challenge Generation** (`challenge.py`): KDF-based deterministic challenges for vision:freq, vision:texture, lm:templates
- **PRF Module** (`prf.py`): HMAC-SHA256 NIST SP 800-108 pseudorandom functions
- **Boundaries** (`boundaries.py`): EB confidence sequences with Welford's algorithm (2025-08-16)
  - Formula: r_t(α) = sqrt(2σ²log(log(t)/α)/t) + c*log(log(t)/α)/t
- **Sequential Testing** (`sequential.py`): Anytime-valid verification with numerical stability (2025-08-16)
  - `sequential_verify()`: Main function with trajectory recording and p-values
  - Advanced features: mixture testing, adaptive thresholds, multi-armed verification
- **Behavioral Fingerprinting** (`fingerprint.py`): IO hashing + Jacobian sketching
  - Performance: <100ms IO-only, ~500ms with Jacobian
  - Factory configs: `FingerprintConfig.for_vision_model()`, `.for_language_model()`
- **Canonicalization** (`canonicalize.py`): Robust normalization for NaN/Inf handling
- **Statistics/Logging/Governance**: Testing utilities, JSONL logging, session management

### 2. Security Components (`pot/security/`)
- **Proof of Training** (`proof_of_training.py`): Complete 6-step verification protocol (2025-08-17)
  - **Expected Ranges**: Behavioral validation against calibrated reference ranges
  - Profiles: quick (~1s), standard (~5s), comprehensive (~30s)
- **Fuzzy Hash Verifier**: SSDeep/TLSH approximate matching
- **Token Space Normalizer**: Cross-tokenizer LM verification
- **Integrated Verification**: Multi-method (exact, fuzzy, statistical) protocols
- **Leakage Tracking**: Challenge reuse policy with ρ monitoring

### 3. Audit Infrastructure (`pot/audit/`) (2025-08-17)
- **Commit-Reveal Protocol** (`commit_reveal.py`): SHA256 cryptographic audit trails with atomic writes
- **Audit Schema** (`schema.py`): Auto-detection validation with legacy compatibility
- **Cryptographic Utilities** (`crypto_utils.py`): Advanced primitives (NEW 2025-08-17)
  - Salt generation, hash chains, timestamp proofs (LOCAL/RFC3161/OpenTimestamps)
  - Commitment aggregation via Merkle trees, ZK proofs, PBKDF2 key derivation
- **Audit Trail Query** (`query.py`): Analysis system with anomaly detection (NEW 2025-08-17)
  - Multi-dimensional querying (10,000+ records), integrity verification
  - Streamlit dashboard with real-time monitoring and export capabilities

### 3.1. Semantic Verification (`pot/semantic/`) (NEW 2025-08-17)
- **ConceptLibrary** (`library.py`): Foundation for concept vector management
  - **Statistical Modeling**: Gaussian (μ, Σ) and hypervector representations
  - **Dual Method Support**: Traditional statistics and modern hypervector approaches
  - **Tensor Operations**: PyTorch integration with efficient torch.save/load persistence
  - **Integrity Checking**: Hash-based validation and comprehensive error handling
- **Semantic Matching** (`match.py`): Multi-metric concept vector comparison
  - **Distance Metrics**: Cosine, Euclidean, Manhattan, Jaccard, Jensen-Shannon
  - **Batch Processing**: Efficient similarity scoring and candidate matching
  - **Fallback Support**: Robust matching with multiple metric attempts
- **Type System** (`types.py`): Core data structures for semantic operations
  - **ConceptVector**: Metadata-rich vector representations with integrity checks
  - **SemanticMatchResult**: Comprehensive matching results with confidence scoring
  - **MatchingConfig**: Flexible configuration for semantic comparison operations
- **Utilities** (`utils.py`): Helper functions for semantic analysis
  - **Vector Operations**: Normalization (L2, L1, max, z-score), centroid computation
  - **Clustering**: K-means and DBSCAN with statistical analysis
  - **Dimensionality Reduction**: PCA for visualization and analysis
  - **Outlier Detection**: Z-score and IQR-based anomaly identification

### 3.2. Topographical Learning (`pot/semantic/topography*`) (NEW 2025-08-17)
- **Projection Methods** (`topography.py`): Multi-method dimensionality reduction for semantic space visualization
  - **UMAP**: Uniform manifold approximation (recommended default) - preserves local + global structure
  - **t-SNE**: Local cluster structure emphasis - excellent for fine-grained analysis
  - **PCA**: Fast linear projection - ideal for quick exploration and preprocessing
  - **SOM**: Self-organizing maps - topological preservation and interpretable clustering
- **Performance Optimization** (`topography_optimized.py`): High-performance implementations (NEW 2025-08-17)
  - **IncrementalUMAP**: Online learning with append/merge/weighted update strategies
  - **OnlineSOM**: Streaming SOM with exponential/linear decay functions
  - **BatchedProjection**: Memory-efficient processing for large datasets (>100k samples)
  - **CachedProjector**: LRU + persistent disk caching with content-based hashing
  - **ApproximateProjector**: PCA preprocessing + approximate nearest neighbors for speed
  - **GPU Acceleration**: RAPIDS cuML integration for 5-10x speedup when available
- **Quality Assessment** (`topography_utils.py`): Projection evaluation and optimization
  - **Metrics**: Trustworthiness, continuity, Kruskal stress, Shepard correlation
  - **Parameter Selection**: Automatic optimization for speed/quality tradeoffs
  - **Cluster Analysis**: K-means, DBSCAN, Gaussian mixture identification
  - **Outlier Detection**: Topological anomaly identification
- **Evolution Tracking** (`topography.py`): Temporal analysis of semantic spaces
  - **Drift Detection**: Centroid shift, density change, regime change detection
  - **Trajectory Tracking**: Semantic path analysis with velocity/acceleration
  - **Cluster Evolution**: Merge/split detection, stability analysis
  - **Snapshot Management**: Efficient storage and comparison of temporal embeddings
- **Visualization** (`topography_visualizer.py`): Interactive and static plotting
  - **Static Plots**: Matplotlib-based with customizable styling and export
  - **Interactive Exploration**: Plotly-based with hover, zoom, selection
  - **SOM Visualization**: U-matrix, component planes, hit maps, cluster overlays
  - **Animation**: Evolution videos with smooth transitions and regime highlighting
  - **Dashboard Creation**: Multi-panel analysis with quality metrics integration

### 3.3. Blockchain Infrastructure (`pot/prototypes/`) (2025-08-17)
- **BlockchainClient**: Multi-chain support (Ethereum, Polygon, BSC, Arbitrum, Optimism)
  - Gas optimization: 60-80% savings via Merkle batching, EIP-1559 support
  - Production features: retry logic, context managers, thread-safe operations
- **MockBlockchainClient**: Zero-cost testing with full API compatibility

### 4. Vision Components (`pot/vision/`) (2025-08-16)
- **VisionVerifier**: Integrated fingerprinting + sequential testing
  - Sequential modes: 'legacy' SPRT or 'enhanced' EB-based (90% evaluation reduction)
  - Challenges: sine gratings, textures (Perlin/Gabor/checkerboard)
  - Distance metrics: cosine, L2, L1 with augmentation robustness
- **Vision Models/Probes**: Base classes and pattern generation utilities

### 5. Language Model Components (`pot/lm/`) (2025-08-16)
- **LMVerifier**: Text-aware fingerprinting + sequential testing
  - Template challenges, fuzzy/exact/edit/embedding distance metrics
  - Handles variable-length outputs with conservative text thresholds
- **Fuzzy Hashing**: Token-level matching with cross-tokenizer normalization
- **LM Models**: Base classes with deterministic generation control

### 6. Evaluation/Prototypes/Scripts
- **Evaluation** (`pot/eval/`): ROC/DET curves, FAR/FRR analysis, plotting utilities
- **Merkle Trees** (`pot/prototypes/`): Complete implementation for training provenance (2025-08-17)
  - Core functions: `build_merkle_tree()`, `generate_merkle_proof()`, `verify_merkle_proof()`
  - O(log n) proof size, handles millions of events, integration with blockchain storage
- **Scripts**: Experiments (E1-E7), attack simulations, baselines, demos
- **Configs**: YAML files for CIFAR-10, ImageNet, small/large LMs

## Functionality Overview

### Challenge Families
- **vision:freq**: Sine gratings (frequency, orientation, phase, contrast)
- **vision:texture**: Perlin noise, Gabor filters, checkerboard patterns  
- **lm:templates**: Text generation with slots (subject, verb, object, etc.)

### Verification Profiles & Security Levels
- **quick**: 1 challenge, ~1s, 70-80% confidence (low security, development)
- **standard**: 3-5 challenges, ~5s, 85-90% confidence (medium security, staging)
- **comprehensive**: All challenges, ~30s, 95%+ confidence (high security, production)

### Cryptographic Protocols (2025-08-17)
- **Commit-Reveal**: SHA256 tamper-evident workflow with automatic salt generation
- **6-Step Verification**: Pre-commitment → Challenges → Execution → Testing → Validation → Audit

## Enhanced Verification Protocol (2025-08-16)

**6-Step Cryptographic Verification Protocol:**
1. **Commit-Reveal**: SHA256 pre-verification commitment with automatic salt
2. **PRF Challenges**: NIST SP 800-108 deterministic challenge generation  
3. **EB Bounds**: Empirical Bernstein confidence sequences with Welford's algorithm
4. **Sequential Testing**: Anytime-valid verification with early stopping (up to 90% reduction)
   - Advanced: mixture testing, adaptive thresholds, multi-armed verification
   - Visualization: trajectory plots, operating characteristics, interactive demos
5. **Leakage Tracking**: Challenge reuse policy with ρ monitoring

### Enhanced CLI
```bash
python scripts/run_verify_enhanced.py --config configs/vision_cifar10.yaml \
  --alpha 0.001 --beta 0.001 --tau-id 0.01 --boundary EB --rho-max 0.3
```

### Test Suite (2025-08-17)
**Core Tests:** boundaries, PRF, sequential verification, fingerprinting
**Security Tests:** audit trails, leakage tracking, crypto utilities, query system

```

**Documentation**: See docs/statistical_verification.md (theory), examples/sequential_analysis.ipynb (tutorials)

### Blockchain Client (2025-08-17)
**Networks**: Ethereum, Polygon, BSC, Arbitrum, Optimism + local development  
**Features**: 60-80% gas savings via Merkle batching, EIP-1559 support, automatic retry logic  
**Operations**: `store_commitment()`, `batch_store_commitments()`, `verify_commitment_onchain()`

### Expected Ranges Verification (2025-08-17)
**Components**: ExpectedRanges, RangeCalibrator, ValidationReport  
**Features**: Auto-calibration from reference models, attack detection, continuous monitoring with statistical validation

### Merkle Trees for Training Provenance (2025-08-17)
```python
from pot.prototypes.training_provenance_auditor import build_merkle_tree, generate_merkle_proof, verify_merkle_proof

# Build tree and generate proofs (O(log n) proof size)
tree = build_merkle_tree([b"epoch_0", b"epoch_1", b"epoch_2"])
proof = generate_merkle_proof(tree, 1)
is_valid = verify_merkle_proof(leaf_hash, proof, tree.hash)

# Training provenance auditor
auditor = TrainingProvenanceAuditor(model_id="resnet50_cifar10")
for epoch in range(5):
    auditor.log_training_event(epoch=epoch, metrics={'loss': 1.0/(epoch+1)})
proof_data = auditor.generate_training_proof(start_epoch=0, end_epoch=4)
```

### Merkle Tree Performance
**Scale**: Handles millions of events efficiently  
**Complexity**: O(n) construction, O(log n) proofs, O(1) storage for root  
**Example**: 1M events → ~20 hash proof size, <1s verification

### Cryptographic Utilities (2025-08-17)
**Primitives**: Secure salt generation, hash chains, timestamp proofs, ZK proofs, PBKDF2 key derivation  
**Features**: Auto-fallback implementations, thread-safe operations, multiple hash algorithms

### Complete Integrated Protocol (2025-08-17)
**6-Step Workflow**: Pre-commitment → Challenge Generation → Model Execution → Statistical Testing → Range Validation → Audit Trail  
**Components**: SessionConfig, VerificationReport, Expected Ranges with multi-modal support  
**Use Cases**: Deployment verification, continuous monitoring, security scanning, compliance reporting

## Quick Start
```bash
pip install -r requirements.txt

# Run experiment E1
python scripts/run_generate_reference.py --config configs/vision_cifar10.yaml
python scripts/run_grid.py --config configs/vision_cifar10.yaml --exp E1

# Test suite
bash run_all_quick.sh  # Smoke test
bash run_all.sh        # Full test suite

# Semantic verification example
python -c "
from pot.semantic import ConceptLibrary
import torch
lib = ConceptLibrary(dim=128, method='gaussian')
embeddings = torch.randn(20, 128)
lib.add_concept('test_concept', embeddings)
print('Semantic library created with', len(lib.list_concepts()), 'concepts')
"

# Topographical learning example
python -c "
from pot.semantic import TopographicalProjector
import torch
projector = TopographicalProjector(method='umap')
embeddings = torch.randn(500, 128)
projected = projector.project_latents(embeddings)
print(f'Projected {embeddings.shape} to {projected.shape} using UMAP')
"

# Performance benchmarking
python benchmarks/topography_benchmarks.py
```

## Guidelines

### Code Requirements
1. **Determinism**: Seeded generators, `torch.use_deterministic_algorithms(True)`, reproducible fingerprints
2. **Structure**: Core in `pot/`, security in `pot/security/`, configs in `configs/`, scripts in `scripts/`
3. **Testing**: `bash run_all_quick.sh` for smoke tests, `bash run_all.sh` for full validation

### Fingerprinting Best Practices
1. **Config**: Use factory methods, enable Jacobian for security (2-5x overhead), adjust precision for model type
2. **Performance**: IO-only for quick checks (<100ms), cache reference fingerprints, batch GPU challenges
3. **Integration**: Use as pre-filter, set thresholds (0.95+ high security, 0.8-0.9 development)
4. **Edge Cases**: Handles NaN/Inf, variable outputs, model failures gracefully

### Experiment Protocol
1. **Setup**: Use `configs/` directory, check EXPERIMENTS.md, output to `outputs/`
2. **Profiles**: quick (1s), standard (5s), comprehensive (30s)
3. **Security**: low (70%, dev), medium (85%, staging), high (95%, production)

## Common Tasks

**Add Model Type**: Extend enum in `proof_of_training.py`, add challenges in `challenge.py`, create config  
**Attack Simulation**: `python scripts/run_attack.py --config configs/vision_cifar10.yaml --attack targeted_finetune`  
**Compliance**: Use `ProofOfTraining(config).perform_verification()` with 'comprehensive' profile  
**Semantic Verification**: Build concept library with `ConceptLibrary(dim, method)`, add concepts from embeddings, perform matching with `SemanticMatcher`  
**Topographical Analysis**: Use `TopographicalProjector(method)` for visualization, `project_latents_batched()` for large datasets, `IncrementalUMAP` for streaming data  
**Performance Benchmarking**: Run `python benchmarks/topography_benchmarks.py` for speed/quality analysis across methods and dataset sizes

### Commit-Reveal Protocol (2025-08-17)
```python
from pot.audit.commit_reveal import compute_commitment, verify_reveal

# Pre-verification commitment
commitment = compute_commitment({'model_id': 'resnet50_v1', 'params': {'alpha': 0.01}})

# Post-verification reveal and validation
is_valid = verify_reveal(commitment, verification_results, salt)
```

### Audit Schema Validation (2025-08-17)
**Functions**: `validate_audit_record()`, `create_enhanced_audit_record()`, `sanitize_for_audit()`  
**Features**: Auto-detection, timestamp validation, legacy compatibility, comprehensive error reporting

## Debugging & Best Practices

**Debug**: Check JSONL logs in `outputs/`, verify `PYTHONHASHSEED=0`, reduce batch sizes for memory issues  
**Practice**: Commit config snapshots, use structured logging, report confidence intervals, proper error handling

## Documentation

**Structure**: CLAUDE.md (framework overview), README.md (quick start), EXPERIMENTS.md (protocols), AGENTS.md (automation)  
**Theory**: docs/statistical_verification.md, examples/sequential_analysis.ipynb  
**Guidelines**: Update docs with code changes, include paper references, comprehensive docstrings

## Remember

Research framework for Proof-of-Training validation. Maintain separation: core framework (`pot/`) vs security extensions (`pot/security/`) vs semantic verification (`pot/semantic/`).
