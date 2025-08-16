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
- **Boundaries** (`boundaries.py`): Confidence sequences with EB radius
  - `CSState`: Online confidence sequence state tracking
  - `eb_radius`: Empirical Bernstein bounds computation
  - Welford's algorithm for streaming statistics
- **Sequential Testing** (`sequential.py`): Anytime-valid sequential verification
  - `SequentialTester`: SPRT-based early stopping
  - `sequential_verify`: Complete verification with audit trail
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

### 4. Vision Components (`pot/vision/`)
- **Vision Verifier** (`verifier.py`): Vision model verification with fingerprinting
  - `VisionVerifier`: Main verification class with integrated fingerprinting
    - `use_fingerprinting`: Enable behavioral fingerprinting
    - `fingerprint_config`: Configuration for fingerprint computation
    - `compute_reference_fingerprint`: Generate reference model fingerprint
    - `compute_fingerprint_similarity`: Compare fingerprints
  - `VisionVerificationResult`: Enhanced with fingerprint fields
    - `fingerprint`: FingerprintResult object
    - `fingerprint_match`: Similarity score to reference
  - Sine grating and texture challenge generation
  - Perceptual distance computation (cosine, L2, L1)
  - Augmentation-based robustness testing
  - `BatchVisionVerifier`: Multi-model verification with fingerprinting
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

### 5. Language Model Components (`pot/lm/`)
- **LM Verifier** (`verifier.py`): Language model verification with fingerprinting
  - `LMVerifier`: Main verification class with integrated fingerprinting
    - `use_fingerprinting`: Enable behavioral fingerprinting for text
    - `fingerprint_config`: LM-specific fingerprint configuration
    - `compute_reference_fingerprint`: Generate fingerprint from prompts
    - `compute_fingerprint_similarity`: Text-aware similarity comparison
  - `LMVerificationResult`: Enhanced with fingerprint fields
    - `fingerprint`: FingerprintResult for text outputs
    - `fingerprint_match`: Similarity using fuzzy text matching
  - Template-based challenge generation
  - Output distance computation (fuzzy, exact, edit, embedding)
  - Time-tolerance verification for model drift
  - `BatchLMVerifier`: Multi-model verification with fingerprinting
  - Adaptive verification with dynamic challenge counts
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

### 7. Experiment Scripts (`scripts/`)
- **Core Experiments**:
  - `run_generate_reference.py`: Create reference models
  - `run_grid.py`: Grid search experiments (E1-E7)
  - `run_verify.py`: Basic verification runner
  - `run_verify_enhanced.py`: Enhanced verification with full protocol
  - `run_plots.py`: Generate ROC/DET curves and visualizations
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

4. **Sequential Verification** (`pot/core/sequential.py`)
   - Early stopping with dual radius computation
   - Asymmetric error control (α, β)
   - Optional stopping for efficiency

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

### Test Suite

Comprehensive unit tests are available in `tests/`:
- `test_boundaries.py`: Confidence sequence tests with FPR validation
- `test_audit.py`: Commit-reveal protocol tests
- `test_prf.py`: PRF determinism and uniformity tests
- `test_reuse.py`: Leakage tracking and policy tests
- `test_equivalence.py`: Transform equivalence tests

Run all tests:
```bash
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

### Integrated Verification with Fingerprinting
```python
from pot.vision.verifier import VisionVerifier
from pot.lm.verifier import LMVerifier

# Vision model with fingerprinting
vision_verifier = VisionVerifier(
    reference_model=ref_model,
    use_fingerprinting=True,
    fingerprint_config=FingerprintConfig.for_vision_model()
)
result = vision_verifier.verify(model, challenges)
print(f"Fingerprint match: {result.fingerprint_match:.3f}")

# Language model with fingerprinting
lm_verifier = LMVerifier(
    reference_model=ref_lm,
    use_fingerprinting=True,
    fingerprint_config=FingerprintConfig.for_language_model()
)
result = lm_verifier.verify(model, prompts)
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

### Sequential Verification
```python
from pot.core.sequential import sequential_verify

def distance_stream():
    for challenge in challenges:
        yield compute_distance(model, challenge)

decision, trail = sequential_verify(
    stream=distance_stream(),
    tau=0.05,
    alpha=0.01,
    beta=0.01,
    n_max=500
)
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

3. **Testing**:
   - Run fingerprint tests: `python -m pot.core.test_fingerprint`
   - Run component tests: `python pot/security/test_*.py`
   - Run integration: `python pot/security/proof_of_training.py`
   - Full validation: `bash run_all.sh`

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

When adding substantial functionality to the codebase:

1. **Add comprehensive docstrings** to all new functions and classes:
   - Include cryptographic properties for security-related functions
   - Specify parameter ranges and types clearly
   - Provide usage examples in docstrings
   - Reference relevant paper sections (e.g., §2.3 for challenge generation)

2. **Update this CLAUDE.md file** with:
   - New module descriptions and their purposes
   - Important implementation details
   - Usage patterns and best practices
   - Any new dependencies or requirements

3. **Update AGENTS.md** if the functionality affects:
   - Integration patterns with external systems
   - API interfaces or protocols
   - Agent-based verification workflows

4. **Add inline comments** that:
   - Reference specific paper sections for algorithms
   - Explain non-obvious implementation choices
   - Mark security-critical code sections
   - Note performance vs security trade-offs

5. **Update requirements.txt** when adding dependencies:
   - Include version pins for reproducibility
   - Add comments explaining why each dependency is needed
   - Group related dependencies together

6. **Maintain the README.md** by:
   - Adding new challenge families to the documentation
   - Updating usage examples for new features
   - Documenting any new verification profiles or modes

## Contact & Support

- Check EXPERIMENTS.md for detailed experimental protocols
- Review AGENTS.md for integration instructions
- See README.md for general project overview

## Remember

This is a research framework for validating Proof-of-Training systems. The security components provide cryptographic verification of model behavior, while the experimental framework tests robustness, efficiency, and attack resistance. Always maintain the separation between the core framework (`pot/`) and security extensions (`pot/security/`).