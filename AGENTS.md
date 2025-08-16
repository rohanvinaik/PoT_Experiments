# Agent Instructions for Proof-of-Training System

## Overview

This document provides instructions for AI agents and developers on how to use the Proof-of-Training (PoT) verification system. The system provides cryptographic verification of neural network training and model identity through multiple verification techniques.

## Complete Project Structure

The PoT system provides comprehensive model verification through multiple components:

### 1. Core Framework (`pot/core/`)
- **Challenge Generation** (`challenge.py`): KDF-based deterministic challenge creation
  - Supports vision:freq, vision:texture, lm:templates families
  - Model-specific challenges via model_id parameter
- **PRF Functions** (`prf.py`): Cryptographic pseudorandom functions
  - HMAC-SHA256 for security, xxhash for performance
  - NIST SP 800-108 counter mode construction
- **Statistical Testing** (`stats.py`, `boundaries.py`, `sequential.py`) (UPDATED 2025-08-16)
  - **Empirical Bernstein bounds** (`boundaries.py`):
    - `eb_radius(state, alpha, c=1.0)`: Anytime-valid confidence radius
    - `eb_confidence_interval(mean, variance, n, alpha, c=1.0)`: Confidence bounds
    - `log_log_correction(t, alpha)`: Correction factor for anytime validity
  - SPRT-based sequential testing with early stopping
  - Asymmetric error control (α, β)
- **Behavioral Fingerprinting** (`fingerprint.py`): Fast model identification
  - IO fingerprinting: Hash-based signatures of canonicalized outputs
  - Jacobian sketching: Compressed gradient structure analysis
  - Comparison utilities for similarity scoring and batch processing
  - Sub-100ms verification for identity checks
- **Canonicalization** (`canonicalize.py`): Robust output comparison
  - Handles NaN/Inf values, text normalization, numeric precision
- **Wrapper Detection** (`wrapper_detection.py`): Attack detection
- **Governance** (`governance.py`): Session and policy management
- **Utilities**: Logging, cost tracking, JSON encoding

### 2. Security Components (`pot/security/`)
- **ProofOfTraining** (`proof_of_training.py`): Main verification system
  - Three verification profiles: quick, standard, comprehensive
  - Three security levels: low (70%), medium (85%), high (95%)
- **FuzzyHashVerifier** (`fuzzy_hash_verifier.py`): Approximate matching
  - SSDeep and TLSH algorithm support
- **TokenSpaceNormalizer** (`token_space_normalizer.py`): LM tokenization
  - Cross-tokenizer verification support
- **IntegratedVerification** (`integrated_verification.py`): Combined protocols
- **LeakageTracking** (`leakage.py`): Challenge reuse policy (ρ ≤ ρ_max)

### 3. Audit Infrastructure (`pot/audit/`)
- **CommitReveal** (`commit_reveal.py`): Tamper-evident verification
- **Schema** (`schema.py`): Structured audit records with JSON schemas

### 4. Vision Components (`pot/vision/`)
- **VisionVerifier** (`verifier.py`): Vision model verification with fingerprinting
  - Integrated behavioral fingerprinting support
  - Sine grating and texture challenge generation
  - Perceptual distance metrics (cosine, L2, L1)
  - Augmentation-based robustness testing
  - Early rejection based on fingerprint mismatch
- **Models** (`models.py`): Vision model interfaces and loading
- **Probes** (`probes.py`): Visual challenge generation utilities

### 5. Language Model Components (`pot/lm/`)
- **LMVerifier** (`verifier.py`): Language model verification with fingerprinting
  - Integrated behavioral fingerprinting for text outputs
  - Template-based challenge generation
  - Multiple distance metrics (fuzzy, exact, edit, embedding)
  - Time-tolerance verification for drift handling
  - Text canonicalization for consistent fingerprints
- **FuzzyHash** (`fuzzy_hash.py`): Token-level fuzzy matching
- **Models** (`models.py`): LM interfaces and generation control

### 6. Evaluation (`pot/eval/`)
- Metrics computation (ROC, DET, FAR/FRR)
- Baseline implementations
- Visualization utilities

### 7. Experiment Scripts (`scripts/`)
- **Core**: `run_generate_reference.py`, `run_grid.py`, `run_verify.py`
- **Enhanced**: `run_verify_enhanced.py` (full protocol with all features)
- **Attacks**: `run_attack.py`, `run_attack_realistic.py`, `run_attack_simulator.py`
- **Analysis**: `run_plots.py`, `run_baselines.py`, `run_coverage.py`
- **API**: `run_api_verify.py`, `api_verification.py`

### 8. Configurations (`configs/`)
- Vision: `vision_cifar10.yaml`, `vision_imagenet.yaml`
- Language: `lm_small.yaml`, `lm_large.yaml`
- Custom model configurations

## Behavioral Fingerprinting Integration

### Overview
Behavioral fingerprinting provides fast, deterministic model verification that complements statistical methods. It's ideal for:
- Quick identity checks before expensive verification
- Detecting fine-tuning or model modifications
- Batch verification of multiple models
- Creating audit trails

### Key Components
1. **IO Fingerprinting**: Creates hash-based signatures from model outputs
2. **Jacobian Sketching**: Compresses gradient information for deeper analysis
3. **Comparison Utilities**: Tools for similarity scoring and batch processing

### Integration Pattern
```python
from pot.core.fingerprint import FingerprintConfig, fingerprint_run, is_behavioral_match
from pot.vision.verifier import VisionVerifier

# Step 1: Quick fingerprint check
config = FingerprintConfig.for_vision_model(compute_jacobian=False)
fp_candidate = fingerprint_run(model, challenges, config)

if not is_behavioral_match(fp_reference, fp_candidate, threshold=0.9):
    return "Model rejected - fingerprint mismatch"

# Step 2: Full statistical verification if fingerprints match
verifier = VisionVerifier(reference_model, use_fingerprinting=True)
result = verifier.verify(model, challenges)
```

### Configuration Guidelines
- **Vision Models**: Enable Jacobian, use magnitude sketch, precision 6
- **Language Models**: Skip Jacobian (expensive), focus on IO hash, precision 5
- **Large Models (>1GB)**: Use `memory_efficient=True`, skip Jacobian
- **Security-Critical**: Enable both IO and Jacobian, threshold 0.95+

### Performance Benchmarks
- IO Fingerprinting: ~10-50ms for 10 challenges
- Jacobian Sketching: ~100-500ms additional
- Batch Verification: Linear scaling with model count
- Memory: O(n_challenges × output_dim)

## Demonstrations (UPDATED 2025-08-16)

### Sequential Verification Demo

Run the comprehensive sequential verification demo:

```bash
python -m pot.core.demo_sequential_verify
```

This demonstrates:
- Three test scenarios (H0, H1, borderline cases)
- Trajectory visualization with confidence bounds
- Anytime validity across different stopping times
- Type I/II error rate validation over 1000 simulations
- Early stopping efficiency while maintaining guarantees

Output includes:
- Decision points and stopping times for each scenario
- Confidence intervals and p-values
- Error rate verification showing control at target levels
- Mean stopping times showing efficiency gains

## Quick Start Guide

### Installation
```bash
git clone https://github.com/rohanvinaik/PoT_Experiments.git
cd PoT_Experiments
pip install -r requirements.txt
```

### Test Suite
```bash
# Full experimental validation (~10 minutes)
bash run_all.sh

# Quick validation (~2 minutes)
bash run_all_quick.sh

# Unit tests only (~1 minute)
pytest -q
```

### 1. Basic Model Verification

To run the core experiments (E1-E7), follow the experimental protocol:

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiment E1 (core separation)
python scripts/run_generate_reference.py --config configs/vision_cifar10.yaml
python scripts/run_grid.py --config configs/vision_cifar10.yaml --exp E1
python scripts/run_plots.py --exp_dir outputs/vision_cifar10/E1 --plot_type roc
```

### 2. Enhanced Verification Protocol (Complete) (UPDATED 2025-08-16)

The system provides a complete cryptographic verification protocol with all features:

```python
from pot.core.sequential import (
    sequential_verify, 
    SequentialState, 
    SPRTResult,
    welford_update,
    compute_empirical_variance,
    check_stopping_condition,
    compute_anytime_p_value
)
from pot.core.boundaries import CSState, eb_radius, eb_confidence_interval
from pot.core.prf import prf_derive_key
from pot.audit import make_commitment, verify_commitment

# Setup PRF-based challenge generation
master_key = os.urandom(32)
nonce = os.urandom(32)
derived_key = prf_derive_key(master_key, "challenge:vision", nonce)

# Create commitment before verification
data_to_commit = serialize_for_commit(challenge_ids, ranges, context)
commitment = make_commitment(master_key, nonce, data_to_commit)

# Option 1: Use sequential verification with early stopping
def distance_stream():
    for challenge in challenges:
        yield compute_distance(model, challenge)

# NEW: Returns SPRTResult with complete audit trail and p-values
result = sequential_verify(
    stream=distance_stream(),
    tau=0.05,           # Identity threshold
    alpha=0.01,         # Type I error bound
    beta=0.01,          # Type II error bound  
    max_samples=500,    # Maximum challenges
    compute_p_value=True  # NEW: Compute anytime-valid p-value
)

# Access result details
print(f"Decision: {result.decision} at n={result.stopped_at}")
print(f"Final mean: {result.final_mean:.3f}, variance: {result.final_variance:.3f}")
print(f"CI: {result.confidence_interval}")
print(f"P-value: {result.p_value:.6f}")  # NEW: Anytime-valid p-value
print(f"Trajectory length: {len(result.trajectory)}")

# Option 2: Manual confidence sequence tracking with EB bounds
state = CSState()
for distance in distance_stream():
    state.update(distance)
    
    # Get anytime-valid confidence interval
    radius = eb_radius(state, alpha=0.05, c=1.0)  # c can be tuned
    lower, upper = eb_confidence_interval(
        state.mean, state.variance, state.n, alpha=0.05
    )
    
    # Make decision based on confidence interval
    if upper < 0.05:  # Entire CI below threshold
        decision = "accept_identity"
        break
    elif lower > 0.05:  # Entire CI above threshold  
        decision = "reject_identity"
        break

# Option 3: Use numerical stability helpers directly (NEW)
state = SequentialState()
for value in distance_stream():
    # Numerically stable update
    state = welford_update(state, value)
    
    # Compute variance with Bessel correction
    variance = compute_empirical_variance(state, bessel_correction=True)
    
    # Check stopping condition
    should_stop, decision = check_stopping_condition(state, tau=0.05, alpha=0.01)
    
    if should_stop:
        # Compute anytime-valid p-value
        p_value = compute_anytime_p_value(state, tau=0.05)
        print(f"Decision: {decision}, p-value: {p_value:.6f}")
        break

# Verify commitment after
is_valid = verify_commitment(master_key, nonce, data_to_commit, commitment)
```

### 3. Production Model Verification

For production deployment with full security:

```python
from pot.security.proof_of_training import ProofOfTraining

# Initialize with appropriate settings
config = {
    'verification_type': 'fuzzy',  # Use fuzzy matching for robustness
    'model_type': 'generic',       # Works with any model
    'security_level': 'medium'     # Balance between security and speed
}

pot = ProofOfTraining(config)

# Register your model
model_id = pot.register_model(
    model,
    architecture="your_architecture",
    parameter_count=1000000
)

# Perform verification
result = pot.perform_verification(model, model_id, 'standard')
print(f"Model verified: {result.verified} (confidence: {result.confidence:.2%})")
```

### 3. Model-Specific Configurations

#### For Vision Models (CNNs, ViTs)
```python
config = {
    'verification_type': 'fuzzy',
    'model_type': 'vision',
    'security_level': 'high'
}
```

#### For Language Models (GPT, BERT, T5)
```python
config = {
    'verification_type': 'statistical',  # Better for stochastic outputs
    'model_type': 'language',
    'security_level': 'high'
}
```

#### For Multimodal Models (CLIP, DALL-E style)
```python
config = {
    'verification_type': 'fuzzy',
    'model_type': 'multimodal',
    'security_level': 'high'
}
```

## Experimental Framework Usage

### Complete Experiment Suite (E1-E7)

The framework validates PoT through comprehensive experiments:

#### E1: Core Separation - Identity vs Different Models
```bash
python scripts/run_generate_reference.py --config configs/vision_cifar10.yaml
python scripts/run_grid.py --config configs/vision_cifar10.yaml --exp E1
python scripts/run_plots.py --exp_dir outputs/vision_cifar10/E1 --plot_type roc
# Expected: AUROC ≈ 0.99, FAR ~0.4%, FRR ~0-1.2%
```

#### E2: Leakage Robustness - Challenge Reuse Attack
```bash
python scripts/run_attack.py --config configs/lm_small.yaml \
    --attack targeted_finetune --rho 0.25
# Tests robustness to ρ=25% challenge leakage
```

#### E3: Non-IID Drift - Distribution Shift
```bash
python scripts/run_verify.py --config configs/vision_cifar10.yaml \
    --challenge_family vision:freq --n 256 --drift_params drift.yaml
# Tests robustness to training/deployment drift
```

#### E4: Adversarial Attacks - Wrapper and Routing
```bash
python scripts/run_attack.py --config configs/vision_cifar10.yaml --attack wrapper
python scripts/run_attack_realistic.py --config configs/lm_small.yaml
# Tests detection of wrapper attacks and routing
```

#### E5: Sequential Testing - Early Stopping
```bash
python scripts/run_verify.py --config configs/lm_small.yaml \
    --challenge_family lm:templates --n 512 --seq eb
# Expected: 2-3 queries average with early stopping
```

#### E6: Baseline Comparison
```bash
python scripts/run_baselines.py --config configs/vision_cifar10.yaml
# Compares against FBI, adversarial trajectories, fixed-n aggregations
```

#### E7: Ablation Studies
```bash
python scripts/ablation_scaling.py --config configs/vision_cifar10.yaml
# Tests sequential rules, τ calibration, score clipping, challenge families
```

### Generate Visualizations
```bash
# ROC curves
python scripts/run_plots.py --exp_dir outputs/vision_cifar10/E1 --plot_type roc

# DET curves
python scripts/run_plots.py --exp_dir outputs/vision_cifar10/E1 --plot_type det

# FAR/FRR analysis
python scripts/run_plots.py --exp_dir outputs/vision_cifar10/E1 --plot_type far_frr
```

### Challenge Generation

```python
from pot.core.challenge import ChallengeConfig, generate_challenges
from pot.core.governance import new_session_nonce

config = ChallengeConfig(
    master_key_hex="0" * 64,  # Use proper key in production
    session_nonce_hex=new_session_nonce(),
    n=256,
    family="vision:freq",
    params={"freq_range": [0.5, 8.0], "contrast_range": [0.3, 1.0]}
)

challenges = generate_challenges(config)
```

## Security Component Usage

### 1. Fuzzy Hash Verification

Use when you need to verify models with minor output variations:

```python
from pot.security.fuzzy_hash_verifier import FuzzyHashVerifier, ChallengeVector

# Initialize verifier
verifier = FuzzyHashVerifier(similarity_threshold=0.85)

# Generate challenge
challenge = ChallengeVector(
    dimension=1000,
    topology='complex',  # Options: 'complex', 'sparse', 'normal'
    seed=42
)

# Generate and verify hash
model_output = model(challenge.vector)
hash_output = verifier.generate_fuzzy_hash(model_output)

# Store reference for later verification
verifier.store_reference_hash('model_v1', model_output)

# Later: verify against stored reference
is_valid = verifier.verify_against_stored('model_v1', new_output)
```

### 2. Training Provenance Tracking

Use to track and verify training history:

```python
from pot.prototypes.training_provenance_auditor import TrainingProvenanceAuditor, EventType

# Initialize auditor
auditor = TrainingProvenanceAuditor(model_id="your_model_id")

# During training: log events
for epoch in range(num_epochs):
    # ... training code ...
    
    auditor.log_training_event(
        epoch=epoch,
        metrics={
            'loss': loss_value,
            'accuracy': accuracy,
            'gradient_norm': grad_norm,
            'learning_rate': lr
        },
        checkpoint_hash=hash_of_checkpoint,
        event_type=EventType.EPOCH_END
    )

# Generate cryptographic proof
proof = auditor.generate_training_proof(
    start_epoch=0,
    end_epoch=num_epochs-1,
    proof_type=ProofType.MERKLE  # or ProofType.ZERO_KNOWLEDGE
)

# Embed provenance in model
model_with_provenance = auditor.embed_provenance(model_state_dict)
```

### 3. Leakage Tracking and Reuse Policy

Control challenge reuse to prevent adversarial learning:

```python
from pot.security.leakage import ReusePolicy, LeakageAuditor

# Initialize policy
policy = ReusePolicy(
    u=5,           # Max 5 uses per challenge
    rho_max=0.3,   # Max 30% leakage ratio
    persistence_path="reuse_state.json"
)

# Start session
session_id = "verification_001"
policy.start_session(session_id)

# Check if challenges are safe to use
is_safe, observed_rho = policy.check_leakage_threshold(challenge_ids)
if not is_safe:
    print(f"Warning: Leakage ratio {observed_rho:.2%} exceeds threshold")

# Record challenge uses
for cid in challenge_ids:
    policy.record_use(cid, session_id)

# End session and get stats
stats = policy.end_session(session_id)
print(f"Session complete: {stats.total_challenges} challenges, "
      f"{stats.leaked_challenges} reused (ρ={stats.observed_rho:.2%})")
```

### 4. Token Space Normalization (Language Models)

Use for handling tokenization differences:

```python
from pot.security.token_space_normalizer import TokenSpaceNormalizer, StochasticDecodingController

# Initialize components
normalizer = TokenSpaceNormalizer(TokenizerType.BPE, vocab_size=50000)
controller = StochasticDecodingController(seed=42)

# Force deterministic generation
controller.set_deterministic_mode(temperature=0.0, top_k=1)

# Generate normalized hash for text
text = "Your test prompt"
invariant_hash = normalizer.compute_token_invariant_hash(text)

# Generate controlled variants for verification
variants = controller.generate_verification_response(
    model, 
    challenge_text,
    num_variants=5
)

# Check semantic consistency
similarity = controller.compute_semantic_similarity(variants)
```

## Verification Depth Levels

### Quick Verification (`'quick'`)
- **Use Case**: Rapid sanity checks, CI/CD pipelines
- **Time**: ~1 second
- **Challenges**: 1 challenge test
- **Confidence**: Lower (70-80%)

### Standard Verification (`'standard'`)
- **Use Case**: Regular deployment checks
- **Time**: ~5 seconds
- **Challenges**: 3-5 challenges
- **Confidence**: Medium (85-90%)

### Comprehensive Verification (`'comprehensive'`)
- **Use Case**: Critical deployments, regulatory compliance
- **Time**: ~30 seconds
- **Challenges**: All available challenges
- **Includes**: Training provenance verification
- **Confidence**: High (95%+)

## Advanced Features

### Batch Verification

For verifying multiple models at once:

```python
models = [model1, model2, model3]
model_ids = [id1, id2, id3]

results = pot.batch_verify(models, model_ids, 'standard')

for result in results:
    print(f"Model {result.model_id}: {result.verified}")
```

### Incremental Verification During Training

Monitor model integrity during training:

```python
for epoch in range(num_epochs):
    # Training step
    train_one_epoch(model)
    
    # Incremental verification every 10 epochs
    if epoch % 10 == 0:
        passed = pot.incremental_verify(
            model, 
            model_id,
            epoch,
            metrics={'loss': current_loss}
        )
        
        if not passed:
            print(f"Warning: Verification failed at epoch {epoch}")
```

### Cross-Platform Verification

Verify models using only their outputs (no model access needed):

```python
# Collect outputs from deployed model
outputs = {
    'challenge_1': model_output_1,
    'challenge_2': model_output_2,
    'challenge_3': model_output_3
}

# Verify against registered model
result = pot.cross_platform_verify(outputs, model_id)
```

### Cryptographic Proof Generation

Generate verifiable proofs for third-party verification:

```python
# After verification
proof_bytes = pot.generate_verification_proof(verification_result)

# Share proof with third party
# They can verify without model access:
verification = pot.verify_proof(proof_bytes)
print(f"Proof valid: {verification['valid']}")
```

## Security Considerations

### Security Levels

- **Low**: 70% threshold, 3 challenges, suitable for development
- **Medium**: 85% threshold, 5 challenges, suitable for staging
- **High**: 95% threshold, 10+ challenges, suitable for production

### Best Practices

1. **Always use deterministic mode for verification**:
   - Set random seeds
   - Disable dropout
   - Use temperature=0 for language models

2. **Store reference challenges securely**:
   - Use the blockchain integration for immutable storage
   - Keep master keys separate from challenge data

3. **Regular re-verification**:
   - Set up continuous monitoring for deployed models
   - Re-verify after updates or migrations

4. **Use appropriate verification type**:
   - `exact`: Only when outputs are fully deterministic
   - `fuzzy`: For most neural networks with floating-point operations
   - `statistical`: For language models and stochastic systems

## Troubleshooting

### Common Issues

1. **Low verification confidence**:
   - Increase number of challenges
   - Use higher security level
   - Check for model modifications

2. **Verification failures with correct model**:
   - Ensure deterministic mode is enabled
   - Check for preprocessing differences
   - Verify hardware compatibility

3. **Slow verification**:
   - Use 'quick' or 'standard' depth
   - Enable caching
   - Consider batch verification

## Integration Examples

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pot.security.proof_of_training import ProofOfTraining

app = FastAPI()
pot = ProofOfTraining(config)

@app.post("/verify_model")
async def verify_model(model_id: str):
    model = load_model(model_id)
    result = pot.perform_verification(model, model_id, 'quick')
    
    if not result.verified:
        raise HTTPException(status_code=403, detail="Model verification failed")
    
    return {
        "verified": result.verified,
        "confidence": result.confidence,
        "proof": pot.generate_verification_proof(result).decode()
    }
```

### PyTorch Lightning Integration

```python
import pytorch_lightning as pl
from pot.security.proof_of_training import ProofOfTraining

class VerifiedModel(pl.LightningModule):
    def __init__(self, model, pot_config):
        super().__init__()
        self.model = model
        self.pot = ProofOfTraining(pot_config)
        self.model_id = None
        
    def on_train_start(self):
        # Register model at training start
        self.model_id = self.pot.register_model(
            self.model,
            architecture=self.model.__class__.__name__
        )
    
    def on_epoch_end(self):
        # Incremental verification
        self.pot.incremental_verify(
            self.model,
            self.model_id,
            self.current_epoch,
            self.trainer.logged_metrics
        )
```

## API Reference

### Main Classes

- `ProofOfTraining`: Main verification system
- `FuzzyHashVerifier`: Fuzzy matching for outputs
- `TrainingProvenanceAuditor`: Training history tracking
- `TokenSpaceNormalizer`: Token normalization for LLMs
- `StochasticDecodingController`: Control random generation
- `ChallengeVector`: Generate verification challenges
- `ChallengeLibrary`: Pre-built challenges for different models

### Key Methods

- `register_model()`: Register a model for verification
- `perform_verification()`: Execute verification protocol
- `generate_verification_proof()`: Create cryptographic proof
- `batch_verify()`: Verify multiple models
- `incremental_verify()`: Verify during training
- `cross_platform_verify()`: Verify with outputs only

## Complete Module Reference

### Core Modules (`pot/core/`) (UPDATED 2025-08-16)
- `challenge.py`: Challenge generation with KDF pattern
- `prf.py`: Cryptographic PRF functions
- `boundaries.py`: Confidence sequences and EB bounds
  - `eb_radius()`: Anytime-valid EB radius computation
  - `eb_confidence_interval()`: Confidence bounds using EB
  - `log_log_correction()`: Correction factor for anytime validity
- `sequential.py`: Sequential testing with early stopping (UPDATED 2025-08-16)
  - `SequentialState`: Online state tracking with Welford's algorithm
  - `SPRTResult`: Complete test result with trajectory and p-values
  - `sequential_verify()`: Main verification with EB bounds
  - **Numerical Stability Helpers**:
    - `welford_update()`: Numerically stable online updates
    - `compute_empirical_variance()`: Robust variance computation
    - `check_stopping_condition()`: EB-based stopping logic
    - `compute_anytime_p_value()`: Martingale-based p-values
- `fingerprint.py`: Model behavioral fingerprinting
- `canonicalize.py`: Output normalization
- `stats.py`: Statistical utilities
- `wrapper_detection.py`: Attack detection
- `governance.py`: Session management
- `logging.py`: Structured logging
- `cost_tracker.py`: API cost monitoring

### Security Modules (`pot/security/`)
- `proof_of_training.py`: Main verification system
- `fuzzy_hash_verifier.py`: Approximate matching
- `token_space_normalizer.py`: Tokenization handling
- `integrated_verification.py`: Combined protocols
- `leakage.py`: Challenge reuse tracking

### Audit Modules (`pot/audit/`)
- `commit_reveal.py`: Cryptographic commitments
- `schema.py`: Audit record schemas

### Vision Modules (`pot/vision/`)
- `verifier.py`: Vision verification
- `models.py`: Model interfaces
- `probes.py`: Visual challenges
- `datasets.py`: Data loading

### LM Modules (`pot/lm/`)
- `verifier.py`: LM verification
- `fuzzy_hash.py`: Token matching
- `models.py`: LM interfaces
- `probes.py`: Text challenges

### Key Scripts (`scripts/`)
- `run_verify_enhanced.py`: Full protocol CLI
- `run_grid.py`: Experiment runner
- `run_attack.py`: Attack simulations
- `run_baselines.py`: Baseline comparisons
- `run_api_verify.py`: API verification

### Test Files
- `tests/test_*.py`: Integration tests
- `pot/*/test_*.py`: Unit tests
- Total: 100+ test cases covering all components

### Documentation
- `README.md`: Project overview and quick start
- `EXPERIMENTS.md`: Detailed experimental protocols
- `CLAUDE.md`: Implementation details for Claude
- `AGENTS.md`: This file - API and integration guide

## Available Challenge Families

### Vision Challenges
1. **vision:freq**: Sine grating patterns
   - Parameters: frequency [0.1-10 cycles/degree], orientation [0-180°], phase [0-2π], contrast [0-1]
   - Generated via: `generate_vision_freq_challenges()`

2. **vision:texture**: Complex textures
   - Perlin noise: octaves [1-5], persistence [0.3-0.7], scale [0.01-0.1]
   - Gabor filters: wavelength [5-30px], orientation [0-180°], sigma [3-15]
   - Checkerboard: square_size [8-32px], contrast [0.3-1], rotation [0-45°]
   - Generated via: `generate_vision_texture_challenges()`

### Language Model Challenges
1. **lm:templates**: Template-based prompts
   - Slots: {subject}, {verb}, {object}, {adjective}, {adverb}
   - Example: "The {adjective} {subject} {verb} the {object}"
   - Generated via: `generate_lm_templates_challenges()`

## Support

For issues or questions:
1. Check test files: `python pot/security/test_*.py`
2. Run demos: `python pot/security/proof_of_training.py`
3. See experiments: `EXPERIMENTS.md`
4. Implementation details: `CLAUDE.md`
5. GitHub issues: https://github.com/rohanvinaik/PoT_Experiments

## Updates and Maintenance

The system is designed to be extensible. To add new:
- **Verification methods**: Extend `VerificationType` enum
- **Model types**: Extend `ModelType` enum and `ChallengeLibrary`
- **Hash algorithms**: Add to `FuzzyHashVerifier`
- **Proof types**: Extend `ProofType` in `TrainingProvenanceAuditor`

## Documentation Requirements for Contributors

**⚠️ CRITICAL REQUIREMENT FOR AI AGENTS - MANDATORY DOCUMENTATION PROTOCOL ⚠️**

### ABSOLUTE REQUIREMENT: Documentation MUST Be Updated

**FAILURE TO UPDATE DOCUMENTATION = INCOMPLETE TASK**

When you make ANY changes to the PoT codebase, you are REQUIRED to:

1. **MANDATORY: Update AGENTS.md (this file)** IMMEDIATELY after:
   - ANY changes to API interfaces or endpoints
   - ANY new integration patterns or workflows
   - ANY performance changes or new benchmarks
   - ANY new usage examples or configuration options
   - ANY user-facing functionality changes
   - Mark with: (UPDATED YYYY-MM-DD)

2. **MANDATORY: Update CLAUDE.md** IMMEDIATELY after:
   - ANY new functions, classes, or modules
   - ANY algorithm implementations (include formulas)
   - ANY changes to core components
   - ANY security features or protocols
   - ANY implementation details affecting future work
   - Include paper references (e.g., §2.4)

3. **MANDATORY: Update README.md** when:
   - Adding new challenge families
   - Creating new verification modes
   - Changing dependencies or setup
   - Adding major user features
   - Modifying CLI interfaces

### 1. Code Documentation

- **Docstrings**: Add detailed docstrings to all functions and classes
  - Include cryptographic properties for security functions
  - Specify parameter ranges and types
  - Add usage examples
  - Reference paper sections (e.g., §2.3 for challenge generation)

- **Inline Comments**: Add comments that:
  - Reference specific paper sections for algorithms
  - Explain non-obvious implementation choices
  - Mark security-critical sections
  - Note performance vs security trade-offs

### 2. File Updates

When adding new features, update these files:

- **AGENTS.md** (this file): Add integration examples and API documentation
- **CLAUDE.md**: Update with implementation details and best practices
- **README.md**: Add new challenge families or verification modes
- **requirements.txt**: Document new dependencies with version pins

### Documentation Verification Checklist

**THIS IS MANDATORY - ALL ITEMS MUST BE CHECKED**

Before ANY task can be considered complete, you MUST verify:
- [ ] AGENTS.md updated with ALL integration details and examples
- [ ] CLAUDE.md updated with ALL implementation details and formulas
- [ ] README.md updated if ANY user-facing changes
- [ ] ALL new functions have comprehensive docstrings with paper refs
- [ ] ALL complex logic has explanatory inline comments
- [ ] ALL dependencies added to requirements.txt with exact versions
- [ ] ALL examples tested and verified to work
- [ ] ALL sections marked with (UPDATED YYYY-MM-DD)
- [ ] Backward compatibility verified or breaking changes documented
- [ ] Performance implications documented if applicable

**AI AGENT ENFORCEMENT RULES**:
1. You CANNOT mark a task complete without updating docs
2. Documentation updates must happen IMMEDIATELY after code changes
3. Each function MUST have a docstring explaining its purpose
4. Each docstring MUST include parameter types and return values
5. Complex algorithms MUST reference paper sections (e.g., §2.4)

### 3. Challenge Family Extensions

When adding new challenge families:
1. Implement generation function in `pot/core/challenge.py`
2. Register in the `generate_challenges()` dispatcher
3. Update verifiers to use new challenges
4. Add unit tests in `tests/`
5. Document in README.md under "Supported Challenge Families"

### 4. Testing Requirements

- Add unit tests for all new functions
- Include integration tests for new workflows
- Ensure tests are deterministic with fixed seeds
- Document test coverage in pull requests

### 5. Security Considerations

For security-related changes:
- Document threat model assumptions
- Explain cryptographic choices
- Note any security/performance trade-offs
- Update security level recommendations if needed