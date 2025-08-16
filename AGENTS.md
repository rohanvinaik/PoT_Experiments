# Agent Instructions for Proof-of-Training System

## Overview

This document provides instructions for AI agents and developers on how to use the Proof-of-Training (PoT) verification system. The system provides cryptographic verification of neural network training and model identity through multiple verification techniques.

## Project Structure

The PoT system consists of two main components:

1. **Core Experimental Framework** (`pot/`): Implements the research experiments for validating PoT concepts
   - `pot/core/`: Challenge generation, fingerprinting, statistics
   - `pot/vision/`: Vision model experiments
   - `pot/lm/`: Language model experiments
   - `pot/eval/`: Evaluation metrics and baselines

2. **Security Components** (`pot/security/`): Prototype verification components
   - Fuzzy hash verification
   - Token space normalization
   - Integrated proof-of-training system
   - Training provenance auditing (experimental, see `pot/prototypes/`)

## Quick Start Guide

### Test Suite

```bash
# Full experimental validation
bash run_all.sh

# Faster smoke tests for iteration
bash run_all_quick.sh

# Python-only unit tests
pytest -q
```

### 1. Running Experiments

To run the core experiments (E1-E7), follow the experimental protocol:

```bash
# Install dependencies
pip install -r requirements.txt

# Run experiment E1 (core separation)
python scripts/run_generate_reference.py --config configs/vision_cifar10.yaml
python scripts/run_grid.py --config configs/vision_cifar10.yaml --exp E1
python scripts/run_plots.py --exp_dir outputs/vision_cifar10/E1 --plot_type roc
```

### 2. Basic Model Verification

For production verification using security components:

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

### Running Core Experiments

The experimental framework includes 7 key experiments (E1-E7):

```bash
# E1: Separation vs Query Budget
python scripts/run_grid.py --config configs/vision_cifar10.yaml --exp E1

# E2: Leakage Ablation
python scripts/run_attack.py --config configs/lm_small.yaml --attack targeted_finetune --rho 0.25

# E3: Non-IID Drift (configure drift parameters in config)
python scripts/run_verify.py --config configs/vision_cifar10.yaml --challenge_family vision:freq --n 256

# E4: Adversarial Attacks
python scripts/run_attack.py --config configs/vision_cifar10.yaml --attack wrapper

# E5: Sequential Testing (enabled in config)
python scripts/run_verify.py --config configs/lm_small.yaml --challenge_family lm:templates --n 512

# Generate all plots
python scripts/run_plots.py --exp_dir outputs/vision_cifar10/E1 --plot_type roc
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

### 3. Token Space Normalization (Language Models)

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

## File Locations

- **Experimental Framework**: `pot/core/`, `pot/vision/`, `pot/lm/`, `pot/eval/`
- **Security Components**: `pot/security/`
- **Prototypes**: `pot/prototypes/`
- **Configurations**: `configs/`
- **Experiment Scripts**: `scripts/`
- **Tests**: `pot/security/test_*.py`
- **Documentation**: `README.md`, `EXPERIMENTS.md`, `CLAUDE.md`, `AGENTS.md`

## Support

For issues or questions:
1. Check the test files for usage examples: `python pot/security/test_*.py`
2. Run the integrated demo: `python pot/security/proof_of_training.py`
3. Run experiments: See `EXPERIMENTS.md`
4. Check `CLAUDE.md` for detailed instructions
5. Open an issue on GitHub

## Updates and Maintenance

The system is designed to be extensible. To add new:
- **Verification methods**: Extend `VerificationType` enum
- **Model types**: Extend `ModelType` enum and `ChallengeLibrary`
- **Hash algorithms**: Add to `FuzzyHashVerifier`
- **Proof types**: Extend `ProofType` in `TrainingProvenanceAuditor`