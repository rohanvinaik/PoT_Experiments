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
- **Statistical Testing** (`stats.py`, `boundaries.py`, `sequential.py`)
  - Empirical Bernstein bounds, SPRT, confidence sequences
  - Early stopping with asymmetric error control
- **Fingerprinting** (`fingerprint.py`): Behavioral model identification
- **Canonicalization** (`canonicalize.py`): Robust output comparison
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
- **VisionVerifier** (`verifier.py`): Vision model verification
  - Sine grating and texture challenge generation
  - Perceptual distance metrics (cosine, L2, L1)
  - Augmentation-based robustness testing
- **Models** (`models.py`): Vision model interfaces and loading
- **Probes** (`probes.py`): Visual challenge generation utilities

### 5. Language Model Components (`pot/lm/`)
- **LMVerifier** (`verifier.py`): Language model verification
  - Template-based challenge generation
  - Multiple distance metrics (fuzzy, exact, edit, embedding)
  - Time-tolerance verification for drift handling
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

### 2. Enhanced Verification Protocol (Complete)

The system provides a complete cryptographic verification protocol with all features:

```python
from pot.core.sequential import sequential_verify
from pot.core.boundaries import CSState, eb_radius
from pot.core.prf import prf_derive_key
from pot.audit import make_commitment, verify_commitment

# Setup PRF-based challenge generation
master_key = os.urandom(32)
nonce = os.urandom(32)
derived_key = prf_derive_key(master_key, "challenge:vision", nonce)

# Create commitment before verification
data_to_commit = serialize_for_commit(challenge_ids, ranges, context)
commitment = make_commitment(master_key, nonce, data_to_commit)

# Run sequential verification with early stopping
def distance_stream():
    for challenge in challenges:
        yield compute_distance(model, challenge)

decision, trail = sequential_verify(
    stream=distance_stream(),
    tau=0.05,      # Identity threshold
    alpha=0.01,    # Type I error bound
    beta=0.01,     # Type II error bound
    n_max=500      # Maximum challenges
)

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

### Core Modules (`pot/core/`)
- `challenge.py`: Challenge generation with KDF pattern
- `prf.py`: Cryptographic PRF functions
- `boundaries.py`: Confidence sequences and EB bounds
- `sequential.py`: Sequential testing with early stopping
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

When adding substantial functionality to the PoT codebase, please ensure comprehensive documentation:

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