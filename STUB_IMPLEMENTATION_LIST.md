# Comprehensive List of Stubs to Implement

## Executive Summary
Total stubs identified: **55** across various categories
- **Critical Security**: 4 stubs requiring immediate attention
- **Core Functionality**: 8 stubs with NotImplementedError or TODO markers
- **Placeholder Logic**: 38 instances of placeholder code
- **Pass-only Functions**: 5 functions with no implementation

---

## 🔴 PRIORITY 1: Critical Security Stubs

### 1. Cryptographic Hash Functions (`pot/security/fuzzy_hash_verifier.py`)
**Location**: Lines 110-115  
**Functions**:
- `FuzzyHasher.generate_hash()` - Generate cryptographic hash
- `FuzzyHasher.compare()` - Compare hashes securely

**Implementation Needed**:
```python
def generate_hash(self, data: bytes) -> str:
    # Implement using SHA256 or BLAKE2b
    # Add salt for additional security
    # Return hex digest
```

### 2. Blockchain Integration (`pot/security/training_provenance_auditor.py`)
**Location**: Lines 297-307  
**Functions**:
- `BlockchainClient.store_hash()` - Store hash on blockchain
- `BlockchainClient.retrieve_hash()` - Retrieve hash from blockchain
- `BlockchainClient.verify_hash()` - Verify hash integrity

**Implementation Needed**:
```python
def store_hash(self, hash_value: str, metadata: dict) -> str:
    # Implement blockchain storage (e.g., Ethereum smart contract)
    # Return transaction ID
```

### 3. Commitment Verification (`pot/core/governance.py`)
**Location**: Line 67  
**Function**: `verify_commitment()`

**Current State**: Returns `True` as placeholder  
**Implementation Needed**:
```python
def verify_commitment(commitment: str, secret: str, metadata: Dict) -> bool:
    # Implement proper cryptographic commitment verification
    # Use timing-safe comparison
    # Validate against salt and metadata
```

### 4. Training Event Hash Calculation (`pot/security/training_provenance_auditor.py`)
**Location**: Line 70  
**Function**: `TrainingEvent.calculate_hash()`

**Implementation Needed**:
```python
def calculate_hash(self) -> str:
    # Implement deterministic hash of training event
    # Include all relevant fields
    # Ensure reproducibility
```

---

## 🟡 PRIORITY 2: Core Functionality Stubs

### 5. Vision Model Feature Extraction (`pot/vision/models.py`)
**Location**: Line 17  
**Function**: `VisionModel.get_features()`

**Current State**: Raises `NotImplementedError`  
**Implementation Needed**:
```python
def get_features(self, x: torch.Tensor) -> torch.Tensor:
    # Extract features from penultimate layer
    # Handle different model architectures
    # Return normalized feature vector
```

### 6. Lightweight Fingerprinting (`pot/eval/baselines.py`)
**Location**: Lines 7-9  
**Function**: `lightweight_fingerprint()`

**Current State**: Pass-only with TODO comment  
**Implementation Needed**:
```python
def lightweight_fingerprint(model_outputs):
    # Compute statistical fingerprint
    # Include mean, std, quantiles
    # Generate compact hash representation
```

### 7. DET Curve Plotting (`pot/eval/plots.py`)
**Location**: Lines 16-18  
**Function**: `plot_det_curve()`

**Current State**: Pass-only with TODO comment  
**Implementation Needed**:
```python
def plot_det_curve(y_true, y_scores, save_path=None):
    # Implement Detection Error Tradeoff curve
    # Use probit scale for axes
    # Return matplotlib figure
```

### 8. Vision Probe Generation (`pot/vision/probes.py`)
**Location**: Lines 3-9  
**Functions**:
- `render_sine_grating()` - Generate sine grating patterns
- `render_texture()` - Generate texture patterns

**Implementation Needed**:
```python
def render_sine_grating(size=(224, 224), frequency=10, angle=0):
    # Generate deterministic sine grating
    # Support rotation and frequency control
    # Return as numpy array

def render_texture(size=(224, 224), texture_type='noise'):
    # Generate various texture types
    # Ensure deterministic generation
    # Support Perlin noise, checkerboard, etc.
```

---

## 🟢 PRIORITY 3: Placeholder Logic

### 9. Attack Implementations (`pot/core/attacks.py`)
**Location**: Lines 18-71  
**Functions with placeholder logic**:
- `targeted_finetune()` - Line 18: "Placeholder for actual fine-tuning logic"
- `limited_distillation()` - Line 33: "Placeholder for actual distillation logic"
- `wrapper_attack()` - Line 47: "Placeholder for wrapper attack"
- `extraction_attack()` - Line 61: "Placeholder for extraction attack"

### 10. Model Response Simulation (`pot/core/coverage_separation.py`)
**Location**: Various  
**Description**: Placeholder for model response simulation in optimization

### 11. Output Distance Computation (`pot/lm/verifier.py`)
**Location**: Line 114  
**Function**: `LMVerifier.compute_output_distance()`
**Issue**: Low complexity, needs proper implementation

### 12. Time Tolerance Verification (`pot/lm/verifier.py`)
**Location**: Line 284  
**Function**: `LMVerifier.verify_with_time_tolerance()`
**Issue**: Low complexity, needs proper implementation

### 13. Wrapper Detection (`pot/core/wrapper_detection.py`)
**Location**: Various  
**Description**: Multiple placeholder implementations for detection logic

### 14. Model Transformation (`pot/vision/models.py`)
**Location**: Lines 58-67  
**Functions**:
- `apply_quantization()` - Placeholder for quantization
- `apply_pruning()` - Placeholder for pruning

---

## 🔵 PRIORITY 4: Script Placeholders

### 15. Distance Computation (`scripts/run_verify.py`)
**Description**: Placeholder for actual distance computation from model outputs

### 16. Attack Cost Metrics (`scripts/run_attack.py`)
**Description**: Placeholder cost calculations for attacks

### 17. Grid Search Simulation (`scripts/run_grid.py`)
**Description**: Placeholder distance simulation and AUROC calculation

### 18. Coverage Embedding (`scripts/run_coverage.py`)
**Description**: Random projection placeholder for actual model embeddings

### 19. Baseline Verification (`scripts/run_baselines.py`)
**Function**: `BaselineMethod.verify()`
**Location**: Line 31  
**Issue**: Abstract method needs implementation in subclasses

---

## Implementation Roadmap

### Phase 1: Security Critical (Week 1)
- [ ] Implement cryptographic hash functions
- [ ] Add commitment verification
- [ ] Complete training event hashing
- [ ] Add timing-safe comparisons

### Phase 2: Core Functions (Week 2)
- [ ] Implement vision model feature extraction
- [ ] Add lightweight fingerprinting
- [ ] Complete DET curve plotting
- [ ] Implement vision probe generation

### Phase 3: Attack Suite (Week 3)
- [ ] Implement targeted fine-tuning
- [ ] Add knowledge distillation
- [ ] Complete wrapper attack
- [ ] Add extraction attack

### Phase 4: Verification & Testing (Week 4)
- [ ] Implement LM distance computation
- [ ] Add time tolerance verification
- [ ] Complete wrapper detection
- [ ] Implement model transformations

### Phase 5: Scripts & Integration (Week 5)
- [ ] Update all script placeholders
- [ ] Add proper cost metrics
- [ ] Implement real embeddings
- [ ] Complete baseline methods

---

## Testing Requirements

For each implemented stub:
1. Unit tests with >80% coverage
2. Integration tests with other components
3. Performance benchmarks
4. Security audit for crypto functions
5. Documentation with examples

---

## Notes

- **Blockchain Integration**: Consider using mock implementation for testing
- **Cryptographic Functions**: Use established libraries (e.g., cryptography, hashlib)
- **Model Features**: Support multiple architectures (ResNet, ViT, BERT, etc.)
- **Attack Implementations**: Follow paper specifications exactly
- **Placeholder Data**: Replace all synthetic data with realistic distributions

---

*Generated from TailChasingFixer analysis and manual code review*  
*Last updated: 2025-08-15*