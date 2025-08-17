# Claude Instructions for PoT Experiments

## Project Overview

This is the Proof-of-Training (PoT) Experiments repository, which implements a comprehensive framework for behavioral verification of neural networks. The system combines cryptographic techniques with machine learning to verify model identity and training integrity.

## Project Structure

```
PoT_Experiments/
├── pot/                      # Core implementation
│   ├── core/                # Core utilities (challenges, stats, governance)
│   ├── vision/              # Vision model components
│   ├── lm/                  # Language model components
│   ├── eval/                # Evaluation metrics and plotting
│   └── security/            # Advanced security components
├── configs/                 # YAML configuration files
├── scripts/                 # Experiment runner scripts
├── outputs/                 # Experiment results (auto-created)
└── verification_reports/    # Compliance reports
```

## Key Components

### 1. Core Framework (`pot/core/`)
- **Challenge Generation** (`challenge.py`): KDF-based deterministic challenge creation
  - Supports vision:freq, vision:texture, lm:templates families
  - Challenge dataclass with unique IDs via xxhash
  - Model-specific challenges via optional model_id
- **PRF Module** (`prf.py`): HMAC-SHA256 based pseudorandom functions
  - `prf_derive_key`: Domain-separated key derivation
  - `prf_bytes`: NIST SP 800-108 counter mode PRF
  - `prf_integers`, `prf_floats`, `prf_choice`: Typed random generation
  - `prf_expand`: Hybrid HMAC/xxhash for performance
- **Boundaries** (`boundaries.py`): Confidence sequences with EB radius (UPDATED 2025-08-16)
  - `CSState`: Online confidence sequence state tracking with Welford's algorithm
  - `eb_radius(state, alpha, c=1.0)`: Empirical Bernstein bounds computation
    - Formula: r_t(α) = sqrt(2 * σ²_t * log(log(t) / α) / t) + c * log(log(t) / α) / t
    - Configurable constant c for bias term tuning (default 1.0)
    - Anytime-valid with log-log correction factor
  - `eb_confidence_interval(mean, variance, n, alpha, c=1.0)`: Confidence bounds
    - Returns (lower, upper) tuple clipped to [0,1]
    - Uses EB radius for anytime-valid intervals
  - `log_log_correction(t, alpha)`: Log-log correction for anytime validity
    - Handles edge cases for small t (t < e)
    - Returns log(log(max(e, t)) / α)
- **Sequential Testing** (`sequential.py`): Anytime-valid sequential verification (UPDATED 2025-08-16)
  - `SequentialState`: State tracking with Welford's algorithm for numerical stability
    - Maintains running mean, variance, sum_x, sum_x2, and M2
    - Online updates with `update(x)` method
  - `SPRTResult`: Complete test result with trajectory and audit trail
    - Contains decision, stopped_at, final statistics, confidence bounds
    - Full trajectory for analysis and visualization
    - Optional anytime-valid p-value computation
  - **Numerical Stability Helpers** (NEW):
    - `welford_update(state, new_value)`: Numerically stable mean/variance updates
      - Uses Welford's online algorithm to avoid catastrophic cancellation
      - Maintains precision for very long sequences
    - `compute_empirical_variance(state, bessel_correction)`: Robust variance estimation
      - Handles n=1 case appropriately
      - Ensures non-negative results
    - `check_stopping_condition(state, tau, alpha)`: EB-based stopping decisions
      - Returns (should_stop, decision) tuple
      - Uses confidence intervals to determine H0/H1
    - `compute_anytime_p_value(state, tau)`: Martingale-based p-values
      - Remains valid despite optional stopping
      - Uses law of iterated logarithm for correction
  - `sequential_verify(stream, tau, alpha, beta, max_samples, compute_p_value)`: Main function
    - Now uses numerical stability helpers internally
    - Optional anytime-valid p-value computation
    - Returns complete SPRTResult with audit trail
  - `SequentialTester`: Legacy SPRT implementation
  - Asymmetric error control (α, β)
- **Behavioral Fingerprinting** (`fingerprint.py`): Comprehensive model fingerprinting system
  - `FingerprintConfig`: Configuration with parameter tuning guidelines
  - `FingerprintResult`: Dataclass storing IO hash, Jacobian sketch, timing
  - `fingerprint_run`: Main function to compute behavioral fingerprints
  - **IO Fingerprinting**: Hash-based signatures of canonicalized outputs
  - **Jacobian Analysis**: Gradient structure capture via sketching
    - `jacobian_sign_hash`: Sign pattern compression (most robust)
    - `jacobian_magnitude_sketch`: Scale-preserving sketch
    - `compute_jacobian_sketch`: Integrated sketch computation
  - **Comparison Utilities**:
    - `compare_fingerprints`: Weighted similarity scoring
    - `fingerprint_distance`: Multiple distance metrics
    - `is_behavioral_match`: Binary decision with threshold
    - `batch_compare_fingerprints`: Efficient batch processing
    - `find_closest_match`: Best match from candidates
  - **Canonicalization Integration**: Ensures reproducible comparisons
  - **Performance**: Sub-100ms for IO, ~500ms with Jacobian
- **Canonicalization** (`canonicalize.py`): Normalization for robust comparison
  - `canonicalize_number`: Numeric precision normalization
  - `canonicalize_text`: Text/token normalization
  - `canonicalize_logits`: Handle NaN/Inf in neural network outputs
  - `canonicalize_model_output`: Auto-detect and process various output types
  - `canonicalize_batch_outputs`: Batch processing with deterministic ordering
- **Statistics** (`stats.py`): Statistical testing utilities
  - `empirical_bernstein_bound`: Confidence intervals
  - `t_statistic`: Test statistics computation
  - FAR/FRR calculation utilities
- **Wrapper Detection** (`wrapper_detection.py`): Attack detection
  - Statistical anomaly detection in responses
  - Timing analysis for wrapper identification
- **Governance** (`governance.py`): Session and nonce management
  - `new_session_nonce`: Generate unique session identifiers
  - Policy enforcement utilities
- **Logging** (`logging.py`): Structured logging for experiments
  - `StructuredLogger`: JSONL format logging
  - Experiment tracking and metrics recording
- **Cost Tracking** (`cost_tracker.py`): API usage monitoring
  - Token counting for LLM APIs
  - Cost estimation and budgeting

### 2. Security Components (`pot/security/`)
- **Proof of Training** (`proof_of_training.py`): Main verification system
  - `ProofOfTraining`: Complete verification protocol
  - Model registration and fingerprinting
  - Verification profiles (quick, standard, comprehensive)
  - Cryptographic proof generation
- **Fuzzy Hash Verifier** (`fuzzy_hash_verifier.py`): Approximate matching
  - SSDeep and TLSH algorithm support
  - `ChallengeVector`: Challenge generation for verification
  - Similarity threshold-based verification
- **Token Space Normalizer** (`token_space_normalizer.py`): LM tokenization handling
  - Handles BPE, WordPiece, SentencePiece tokenizers
  - Token-invariant hashing for cross-tokenizer verification
  - Stochastic decoding control for determinism
- **Integrated Verification** (`integrated_verification.py`): Combined protocols
  - Multi-method verification (exact, fuzzy, statistical)
  - Cross-platform verification support
  - Batch verification capabilities
- **Leakage Tracking** (`leakage.py`): Challenge reuse policy
  - `ReusePolicy`: Enforce maximum challenge uses
  - `LeakageAuditor`: Track and report leakage ratio (ρ)
  - Session-based tracking with JSON persistence

### 3. Audit Infrastructure (`pot/audit/`)
- **Commit-Reveal Protocol** (`commit_reveal.py`): Tamper-evident trails
  - `make_commitment`: Generate cryptographic commitments
  - `verify_commitment`: Validate audit trails
  - HMAC-based commitment scheme
- **Audit Schema** (`schema.py`): Structured audit records
  - JSON schema definitions for audit logs
  - Compliance documentation format
  - Verification result serialization

### 4. Vision Components (`pot/vision/`) (UPDATED 2025-08-16)
- **Vision Verifier** (`verifier.py`): Vision model verification with fingerprinting and sequential testing
  - `VisionVerifier`: Main verification class with integrated fingerprinting and sequential verification
    - `use_fingerprinting`: Enable behavioral fingerprinting
    - `fingerprint_config`: Configuration for fingerprint computation
    - `use_sequential`: Enable sequential testing for early stopping
    - `sequential_mode`: 'legacy' (old SPRT) or 'enhanced' (new EB-based)
    - `compute_reference_fingerprint`: Generate reference model fingerprint
    - `compute_fingerprint_similarity`: Compare fingerprints
  - `VisionVerificationResult`: Enhanced with fingerprint and sequential fields
    - `fingerprint`: FingerprintResult object
    - `fingerprint_match`: Similarity score to reference
    - `sequential_result`: SPRTResult with trajectory and p-values
  - Sine grating and texture challenge generation
  - Perceptual distance computation (cosine, L2, L1)
  - Augmentation-based robustness testing
  - `BatchVisionVerifier`: Multi-model verification with fingerprinting
  - **Sequential Integration** (NEW):
    - Enhanced mode uses EB-based sequential verification
    - Early stopping reduces evaluations by up to 90%
    - Complete trajectory recording for audit
    - Anytime-valid p-values and confidence intervals
  - **Fingerprint Integration**:
    - Automatic fingerprint computation during verification
    - Early rejection based on fingerprint mismatch
    - Fingerprint metadata in verification results
- **Vision Models** (`models.py`): Model wrappers and utilities
  - `VisionModel`: Base class for vision models
  - Feature extraction interfaces
  - Model loading and checkpoint management
- **Vision Probes** (`probes.py`): Challenge generation
  - Frequency-based patterns (sine gratings)
  - Texture synthesis (Perlin noise, Gabor filters)
  - Geometric patterns (checkerboards)

### 5. Language Model Components (`pot/lm/`) (UPDATED 2025-08-16)
- **LM Verifier** (`verifier.py`): Language model verification with fingerprinting and sequential testing
  - `LMVerifier`: Main verification class with integrated fingerprinting and sequential verification
    - `use_fingerprinting`: Enable behavioral fingerprinting for text
    - `fingerprint_config`: LM-specific fingerprint configuration
    - `use_sequential`: Enable sequential testing for early stopping
    - `sequential_mode`: 'legacy' (old SPRT) or 'enhanced' (new EB-based)
    - `compute_reference_fingerprint`: Generate fingerprint from prompts
    - `compute_fingerprint_similarity`: Text-aware similarity comparison
  - `LMVerificationResult`: Enhanced with fingerprint and sequential fields
    - `fingerprint`: FingerprintResult for text outputs
    - `fingerprint_match`: Similarity using fuzzy text matching
    - `sequential_result`: SPRTResult with trajectory and p-values
  - Template-based challenge generation
  - Output distance computation (fuzzy, exact, edit, embedding)
  - Time-tolerance verification for model drift
  - `BatchLMVerifier`: Multi-model verification with fingerprinting
  - Adaptive verification with dynamic challenge counts
  - **Sequential Integration** (NEW):
    - Enhanced mode uses EB-based sequential verification
    - Early stopping with text-specific distance metrics
    - Handles variable-length outputs gracefully
    - Maintains fuzzy similarity tracking alongside sequential tests
  - **Fingerprint Integration**:
    - Text canonicalization for consistent hashing
    - Fuzzy matching for fingerprint comparison
    - Conservative thresholds for text variability
- **Fuzzy Hashing** (`fuzzy_hash.py`): Token-level matching
  - `TokenSpaceNormalizer`: Cross-tokenizer normalization
  - `NGramFuzzyHasher`: N-gram based fuzzy matching
  - `AdvancedFuzzyHasher`: SSDeep/TLSH integration
- **LM Models** (`models.py`): Model interfaces
  - `LM`: Base class for language models
  - Tokenizer integration
  - Deterministic generation control

### 6. Evaluation Components (`pot/eval/`)
- **Metrics**: ROC, DET curves, FAR/FRR analysis
- **Baselines**: Reference implementations for comparison
- **Plotting**: Visualization utilities for results

### 7. Experiment Scripts and Demos
- **Core Experiments** (`scripts/`):
  - `run_generate_reference.py`: Create reference models
  - `run_grid.py`: Grid search experiments (E1-E7)
  - `run_verify.py`: Basic verification runner
  - `run_verify_enhanced.py`: Enhanced verification with full protocol
  - `run_plots.py`: Generate ROC/DET curves and visualizations
- **Demonstration Scripts** (`pot/core/`) (UPDATED 2025-08-16):
  - `demo_sequential_verify.py`: Complete sequential verification demo
    - Three test scenarios: H0 (mean<tau), H1 (mean>tau), borderline (mean≈tau)
    - Trajectory plotting with confidence bounds
    - Anytime validity demonstration across different stopping times
    - Type I/II error rate verification over 1000+ simulations
    - Shows early stopping saves computation while maintaining guarantees
- **Attack Simulations**:
  - `run_attack.py`: Basic attack scenarios
  - `run_attack_realistic.py`: Realistic adversarial scenarios
  - `run_attack_simulator.py`: Comprehensive attack simulation
- **Advanced Features**:
  - `run_baselines.py`: Baseline method comparisons
  - `run_coverage.py`: Coverage analysis for challenges
  - `run_api_verify.py`: API-based verification
  - `ablation_scaling.py`: Scaling and ablation studies
  - `audit_log_demo.py`: Audit system demonstration

### 8. Configuration Files (`configs/`)
- `vision_cifar10.yaml`: CIFAR-10 vision experiments
- `vision_imagenet.yaml`: ImageNet vision experiments
- `lm_small.yaml`: Small language model experiments
- `lm_large.yaml`: Large language model experiments
- Model-specific configurations with hyperparameters

## Complete Functionality Overview

### Supported Challenge Families
1. **vision:freq**: Sine grating patterns
   - Parameters: frequency (cycles/degree), orientation, phase, contrast
   - Use: Testing frequency response and orientation selectivity

2. **vision:texture**: Complex texture patterns
   - Types: Perlin noise, Gabor filters, checkerboard
   - Parameters: octaves, wavelength, square_size, etc.
   - Use: Testing response to naturalistic and synthetic textures

3. **lm:templates**: Template-based text generation
   - Slots: subject, verb, object, adjective, adverb
   - Use: Testing consistent language understanding

### Verification Profiles
- **quick**: 1 challenge, ~1 second, 70-80% confidence
- **standard**: 3-5 challenges, ~5 seconds, 85-90% confidence
- **comprehensive**: All challenges, ~30 seconds, 95%+ confidence

### Security Levels
- **low**: 70% threshold, development use
- **medium**: 85% threshold, staging use
- **high**: 95% threshold, production use

## Enhanced Verification Protocol

### Protocol Features (Updated 2025-08-16)

The POT system now includes a complete cryptographic verification protocol:

1. **Commit-Reveal Protocol** (`pot/audit/commit_reveal.py`)
   - Pre-verification commitment generation
   - Post-verification reveal with proof
   - Tamper-evident audit trails

2. **PRF-Based Challenge Generation** (`pot/core/prf.py`)
   - NIST SP 800-108 counter mode construction
   - Deterministic pseudorandom generation
   - Cryptographically secure challenge derivation

3. **Confidence Sequences** (`pot/core/boundaries.py`)
   - Anytime-valid sequential testing
   - Empirical Bernstein (EB) bounds
   - Welford's algorithm for online statistics

4. **Sequential Verification** (`pot/core/sequential.py`) - Complete anytime-valid testing suite
   - **Core Sequential Testing**:
     - Early stopping with dual radius computation
     - Asymmetric error control (α, β)
     - Optional stopping for efficiency
   - **Advanced Sequential Features** (NEW as of 2025-08-16):
     - `mixture_sequential_test`: Combine multiple test statistics using mixture martingales
     - `adaptive_tau_selection`: Dynamic threshold adjustment based on observed variance
     - `multi_armed_sequential_verify`: Test multiple hypotheses with family-wise error control
     - `power_analysis`: Compute operating characteristics and sample size recommendations
     - `confidence_sequences`: Time-uniform confidence sequences for continuous monitoring

5. **Leakage Tracking** (`pot/security/leakage.py`)
   - Challenge reuse policy enforcement
   - Leakage ratio (ρ) calculation
   - Session-based tracking with persistence

### Enhanced CLI (`scripts/run_verify_enhanced.py`)

```bash
# Strict verification with tight bounds
python scripts/run_verify_enhanced.py \
    --config configs/vision_cifar10.yaml \
    --alpha 0.001 --beta 0.001 --tau-id 0.01 \
    --n-max 1000 --boundary EB \
    --master-key $(openssl rand -hex 32) \
    --reuse-u 5 --rho-max 0.3 \
    --outdir outputs/verify/strict
```

### Test Suite (UPDATED 2025-08-16)

Comprehensive unit tests are available in `pot/core/` and `tests/`:

**Core Framework Tests**:
- `pot/core/test_boundaries.py`: Confidence sequence tests with FPR validation
  - Welford's algorithm correctness
  - EB radius computation and properties
  - Sequential decision making (one-sided and two-sided)
  - Adaptive threshold computation
  - Variance bounds for [0,1] values
- `pot/core/test_prf.py`: PRF determinism and uniformity tests
  - Cryptographic PRF validation
  - Challenge generation determinism
  - Output uniformity verification
  - Edge case handling
- `pot/core/test_sequential_verify.py`: Complete sequential verification tests (NEW)
  - Type I error control (false positives < α)
  - Type II error control (false negatives < β)
  - Borderline behavior at decision boundary
  - Anytime validity across different stopping strategies
  - Numerical stability with 100k+ samples
  - Trajectory recording and audit trails
  - Anytime-valid p-value computation
  - Integration testing with performance benchmarks
- `pot/core/test_fingerprint.py`: Behavioral fingerprinting tests
  - Determinism and reproducibility
  - Sensitivity to model changes
  - Jacobian computation correctness
  - Canonicalization integration
  - Edge cases (NaN/Inf, failures, variable outputs)
  - Performance characteristics

**Security Component Tests**:
- `test_audit.py`: Commit-reveal protocol tests
- `test_reuse.py`: Leakage tracking and policy tests
- `test_equivalence.py`: Transform equivalence tests

Run all tests:
```bash
# Run specific test suites
python -m pot.core.test_sequential_verify  # Sequential verification
python -m pot.core.test_boundaries         # Confidence sequences
python -m pot.core.test_prf               # PRF and challenges
python -m pot.core.test_fingerprint       # Behavioral fingerprinting

# Run all pytest tests
pytest tests/ -v
```

## API Usage Examples

### Basic Verification
```python
from pot.security.proof_of_training import ProofOfTraining

pot = ProofOfTraining({
    'verification_type': 'fuzzy',
    'model_type': 'vision',
    'security_level': 'high'
})

result = pot.perform_verification(model, model_id, 'standard')
print(f"Verified: {result.verified} (confidence: {result.confidence:.2%})")
```

### Behavioral Fingerprinting
```python
from pot.core.fingerprint import (
    FingerprintConfig, 
    fingerprint_run,
    compare_fingerprints,
    is_behavioral_match
)

# Configure fingerprinting for vision model
config = FingerprintConfig.for_vision_model(
    compute_jacobian=True,  # Enable gradient analysis
    include_timing=True,
    memory_efficient=False
)

# Compute fingerprint
fingerprint = fingerprint_run(model, challenges, config)
print(f"IO Hash: {fingerprint.io_hash[:32]}...")

# Compare fingerprints
similarity = compare_fingerprints(fp1, fp2)
is_match = is_behavioral_match(fp1, fp2, threshold=0.95)

# Batch comparison
from pot.core.fingerprint import batch_compare_fingerprints
similarities = batch_compare_fingerprints(fingerprints, reference=ref_fp)
```

### Integrated Verification with Fingerprinting and Sequential Testing (UPDATED 2025-08-16)
```python
from pot.vision.verifier import VisionVerifier
from pot.lm.verifier import LMVerifier

# Vision model with fingerprinting and enhanced sequential testing
vision_verifier = VisionVerifier(
    reference_model=ref_model,
    use_fingerprinting=True,
    use_sequential=True,
    sequential_mode='enhanced',  # NEW: Use EB-based sequential verification
    fingerprint_config=FingerprintConfig.for_vision_model()
)
result = vision_verifier.verify(
    model, 
    challenges,
    tolerance=0.05,
    alpha=0.01,  # Type I error rate
    beta=0.01    # Type II error rate
)
print(f"Fingerprint match: {result.fingerprint_match:.3f}")
if result.sequential_result:
    print(f"Early stopping at: {result.sequential_result.stopped_at}")
    print(f"P-value: {result.sequential_result.p_value:.6f}")
    print(f"Trajectory length: {len(result.sequential_result.trajectory)}")

# Language model with fingerprinting and sequential testing
lm_verifier = LMVerifier(
    reference_model=ref_lm,
    use_fingerprinting=True,
    use_sequential=True,
    sequential_mode='enhanced',
    fingerprint_config=FingerprintConfig.for_language_model()
)
result = lm_verifier.verify(model, prompts, alpha=0.01, beta=0.01)
print(f"Decision: {result.accepted}, stopped at {result.n_challenges} challenges")
```

### Challenge Generation
```python
from pot.core.challenge import ChallengeConfig, generate_challenges

config = ChallengeConfig(
    master_key_hex='deadbeef' * 8,
    session_nonce_hex='cafebabe' * 4,
    n=10,
    family='vision:texture',
    params={'texture_types': ['perlin', 'gabor', 'checkerboard']},
    model_id='resnet50_v1'
)

challenges = generate_challenges(config)
```

### Sequential Verification (UPDATED 2025-08-16)
```python
from pot.core.sequential import sequential_verify

def distance_stream():
    for challenge in challenges:
        yield compute_distance(model, challenge)

result = sequential_verify(
    stream=distance_stream(),
    tau=0.05,
    alpha=0.01,
    beta=0.01,
    max_samples=500,
    compute_p_value=True  # Optional: compute anytime-valid p-value
)

print(f"Decision: {result.decision}")
print(f"Stopped at: {result.stopped_at} samples")
print(f"Final mean: {result.final_mean:.3f} ± {result.confidence_radius:.3f}")
if result.p_value is not None:
    print(f"Anytime-valid p-value: {result.p_value:.4f}")
```

### Advanced Sequential Testing (NEW 2025-08-16)
```python
from pot.core.sequential import (
    mixture_sequential_test, adaptive_tau_selection, 
    multi_armed_sequential_verify, power_analysis, confidence_sequences
)

# 1. Mixture Sequential Testing - Combine multiple test statistics for robust decisions
streams = [mean_distances, median_distances, trimmed_mean_distances]
mixture_result = mixture_sequential_test(
    streams=streams,
    weights=[0.5, 0.3, 0.2],  # Weighted combination
    tau=0.05,
    alpha=0.01,
    combination_method='weighted_average'  # or 'fisher', 'min_p'
)
print(f"Mixture decision: {mixture_result.decision}")
print(f"Combined statistic: {mixture_result.final_combined_statistic:.3f}")

# 2. Adaptive Tau Selection - Dynamic threshold adjustment based on variance
adaptive_result = adaptive_tau_selection(
    stream=distance_stream(),
    initial_tau=0.05,
    adaptation_rate=0.1,
    min_tau=0.01,
    max_tau=0.2,
    union_bound_correction=True  # Maintains validity
)
print(f"Adaptive decision: {adaptive_result.decision}")
print(f"Final tau: {adaptive_result.final_tau:.4f}")

# 3. Multi-Armed Testing - Multiple hypotheses with family-wise error control
streams = {'model_A': stream_A, 'model_B': stream_B, 'model_C': stream_C}
hypotheses = {'model_A': 0.03, 'model_B': 0.05, 'model_C': 0.07}
multi_result = multi_armed_sequential_verify(
    streams=streams,
    hypotheses=hypotheses,
    alpha=0.05,  # Family-wise error rate
    correction_method='bonferroni'  # or 'holm'
)
print(f"Decisions: {multi_result.decisions}")
print(f"FWER controlled: {multi_result.fwer_controlled}")

# 4. Power Analysis - Operating characteristics and sample size planning
power_result = power_analysis(
    tau=0.05,
    alpha=0.05,
    beta=0.05,
    effect_sizes=[0.0, 0.02, 0.05, 0.1, 0.2],
    n_simulations=1000
)
print(f"Power curve: {power_result.power_curve}")
print(f"Expected stopping times: {power_result.expected_stopping_times}")
print(f"Recommended sample size: {power_result.sample_size_recommendation}")

# 5. Confidence Sequences - Time-uniform bounds for continuous monitoring
conf_seq = confidence_sequences(
    stream=distance_stream(),
    alpha=0.05,
    method='eb',  # Empirical Bernstein bounds
    return_all=True
)
final_mean = conf_seq.means[-1]
final_lower = conf_seq.lower_bounds[-1]
final_upper = conf_seq.upper_bounds[-1]
print(f"Final estimate: {final_mean:.3f}")
print(f"95% confidence interval: [{final_lower:.3f}, {final_upper:.3f}]")
print(f"Valid at any stopping time: {conf_seq.is_valid}")
```

## Running Experiments

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run core separation experiment (E1)
python scripts/run_generate_reference.py --config configs/vision_cifar10.yaml
python scripts/run_grid.py --config configs/vision_cifar10.yaml --exp E1
python scripts/run_plots.py --exp_dir outputs/vision_cifar10/E1 --plot_type roc
```

### Complete Test Suite
```bash
# Run all tests including security components
bash run_all.sh

# Faster smoke test (skips heavy benchmarks)
bash run_all_quick.sh

# Python-only unit tests
pytest -q
```

## Important Guidelines

### When Modifying Code

1. **Maintain Determinism**: 
   - Always use seeded random generators
   - Set `torch.use_deterministic_algorithms(True)`
   - For LMs: `do_sample=False, temperature=0.0, top_k=1`
   - Fingerprints must be reproducible: same model + challenges = same fingerprint

2. **Preserve Structure**:
   - Keep core framework in `pot/`
   - Security extensions in `pot/security/`
   - Configs in `configs/`
   - Scripts in `scripts/`
   - Fingerprinting utilities in `pot/core/fingerprint.py`

3. **Testing** (UPDATED 2025-08-16):
   - Run sequential verification tests: `python -m pot.core.test_sequential_verify`
   - Run confidence sequence tests: `python -m pot.core.test_boundaries`
   - Run PRF tests: `python -m pot.core.test_prf`
   - Run fingerprint tests: `python -m pot.core.test_fingerprint`
   - Run component tests: `python pot/security/test_*.py`
   - Run integration: `python pot/security/proof_of_training.py`
   - Full validation: `bash run_all.sh`
   - Quick validation: `bash run_all_quick.sh`

### When Using Behavioral Fingerprinting

1. **Configuration Selection**:
   - Use factory methods: `FingerprintConfig.for_vision_model()` or `.for_language_model()`
   - Enable Jacobian only for security-critical verification (adds 2-5x overhead)
   - Adjust `canonicalize_precision` based on model precision (6 for float32, 3-4 for quantized)

2. **Performance Optimization**:
   - IO fingerprinting alone for quick checks (<100ms)
   - Use `memory_efficient=True` for models >1GB
   - Batch challenges when possible for GPU models
   - Cache reference fingerprints to avoid recomputation

3. **Integration Best Practices**:
   - Always compute reference fingerprint once and reuse
   - Use fingerprinting as pre-filter before expensive statistical tests
   - Set appropriate thresholds: 0.95+ for high security, 0.8-0.9 for development
   - Log fingerprint mismatches for audit trails

4. **Handling Edge Cases**:
   - Fingerprints handle NaN/Inf via canonicalization
   - Variable-length outputs supported for LMs
   - Model failures on some inputs won't crash fingerprinting
   - Empty challenge lists return valid (empty) fingerprints

### When Running Experiments

1. **Follow Experimental Protocol**:
   - Use configs from `configs/` directory
   - Check EXPERIMENTS.md for detailed protocols
   - Ensure outputs go to `outputs/` directory

2. **Verification Depth Levels**:
   - `quick`: ~1 second, 1 challenge, lower confidence
   - `standard`: ~5 seconds, 3-5 challenges, medium confidence  
   - `comprehensive`: ~30 seconds, all challenges, highest confidence

3. **Security Levels**:
   - `low`: 70% threshold, development use
   - `medium`: 85% threshold, staging use
   - `high`: 95% threshold, production use

## Common Tasks

### Adding a New Model Type
1. Extend `ModelType` enum in `pot/security/proof_of_training.py`
2. Add challenge generation in `pot/core/challenge.py`
3. Create config in `configs/`
4. Update `ChallengeLibrary` class

### Running Attack Simulations
```bash
python scripts/run_attack.py --config configs/vision_cifar10.yaml \
    --attack targeted_finetune --rho 0.25
```

### Generating Compliance Reports
```python
from pot.security.proof_of_training import ProofOfTraining

pot = ProofOfTraining(config)
result = pot.perform_verification(model, model_id, 'comprehensive')
proof = pot.generate_verification_proof(result)
```

## Debugging Tips

1. **Check Logs**: Outputs are in JSONL format in `outputs/` directory
2. **Verify Seeds**: Ensure `PYTHONHASHSEED=0` is set
3. **Memory Issues**: Reduce batch sizes or challenge counts
4. **Import Errors**: Check optional dependencies (ssdeep, tlsh) and ensure a compatible
   `torchvision` build if vision tests are required

## Best Practices

1. **Always commit config snapshots** with experiment results
2. **Use structured logging** via `pot.core.logging.StructuredLogger`
3. **Report confidence intervals** not just point estimates
4. **Implement proper error handling** in production code
5. **Document any deviations** from the experimental protocol

## Documentation Guidelines

**CRITICAL REQUIREMENT FOR AI AGENTS - MANDATORY DOCUMENTATION PROTOCOL**

### ⚠️ ABSOLUTE REQUIREMENT ⚠️

**AS AN AI AGENT, YOU MUST UPDATE DOCUMENTATION FILES IMMEDIATELY AFTER ANY CODE CHANGES**

This is NOT optional. Failure to update documentation is considered an incomplete task.

### Required Documentation Updates

When making ANY changes to the codebase:

1. **MANDATORY: Update CLAUDE.md (this file)** IMMEDIATELY after implementing:
   - ANY new functions, classes, or modules - document them in the appropriate section
   - ANY changes to existing components - update their descriptions
   - ANY new parameters or configuration options - add to relevant sections
   - ANY algorithm implementations - include formulas and paper references
   - ANY performance characteristics - document complexity and benchmarks
   - Mark updates with date: (UPDATED YYYY-MM-DD)
   
2. **MANDATORY: Update AGENTS.md** IMMEDIATELY after implementing:
   - ANY new API endpoints or interfaces
   - ANY changes to verification workflows
   - ANY new integration patterns or examples
   - ANY performance benchmarks or thresholds
   - ANY user-facing functionality changes
   - Mark sections with update dates

3. **MANDATORY: Update README.md** when:
   - Adding new challenge families
   - Creating new verification modes
   - Changing installation or setup requirements
   - Adding major user-facing features
   - Modifying command-line interfaces

### Code Documentation Requirements

1. **Add comprehensive docstrings** to all new functions and classes:
   - Include cryptographic properties for security-related functions
   - Specify parameter ranges and types clearly
   - Provide usage examples in docstrings
   - Reference relevant paper sections (e.g., §2.3 for challenge generation)

2. **Add inline comments** that:
   - Reference specific paper sections for algorithms
   - Explain non-obvious implementation choices
   - Mark security-critical code sections
   - Note performance vs security trade-offs

3. **Update requirements.txt** when adding dependencies:
   - Include version pins for reproducibility
   - Add comments explaining why each dependency is needed
   - Group related dependencies together

### Documentation Update Checklist

**THIS CHECKLIST IS MANDATORY - YOU MUST COMPLETE ALL APPLICABLE ITEMS**

After implementing ANY feature or change, verify you have:
- [ ] Updated CLAUDE.md with ALL implementation details, formulas, and parameters
- [ ] Updated AGENTS.md with ALL integration instructions and API changes
- [ ] Updated README.md if ANY user-facing changes were made
- [ ] Added comprehensive docstrings with paper references to ALL new functions
- [ ] Added inline comments explaining ALL complex logic
- [ ] Updated requirements.txt if ANY dependencies were added
- [ ] Added the feature to the "Complete Functionality Overview" section
- [ ] Marked ALL updated sections with (UPDATED YYYY-MM-DD)
- [ ] Verified that examples still work with the changes
- [ ] Ensured backward compatibility or documented breaking changes

### Enforcement

**AI AGENTS MUST**:
1. Check this checklist BEFORE considering any task complete
2. Update documentation IMMEDIATELY after code changes (not at the end)
3. Include update dates in documentation
4. Ensure examples and usage patterns are current
5. NEVER skip documentation updates - they are PART of the implementation

## Contact & Support

- Check EXPERIMENTS.md for detailed experimental protocols
- Review AGENTS.md for integration instructions
- See README.md for general project overview

## Remember

This is a research framework for validating Proof-of-Training systems. The security components provide cryptographic verification of model behavior, while the experimental framework tests robustness, efficiency, and attack resistance. Always maintain the separation between the core framework (`pot/`) and security extensions (`pot/security/`).