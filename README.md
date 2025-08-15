# Proof-of-Training (PoT) Experiments

A comprehensive implementation of Proof-of-Training verification systems for neural networks, providing cryptographic verification of model training and identity.

## Overview

This repository contains a complete Proof-of-Training framework with two main components:

1. **Experimental Framework** (`pot/`): Research implementation for validating PoT concepts through systematic experiments (E1-E7)
2. **Security Components** (`pot/security/`): Production-ready verification components including fuzzy hashing, provenance tracking, and token normalization

## Project Structure

```
PoT_Experiments/
├── pot/                          # Core experimental framework
│   ├── core/                    # Challenge generation, stats, governance
│   ├── vision/                  # Vision model experiments
│   ├── lm/                      # Language model experiments
│   ├── eval/                    # Evaluation metrics and plotting
│   └── security/                # Production security components
│       ├── fuzzy_hash_verifier.py
│       ├── training_provenance_auditor.py
│       ├── token_space_normalizer.py
│       └── proof_of_training.py
├── configs/                     # Experiment configurations
│   ├── vision_cifar10.yaml
│   ├── vision_imagenet_sub.yaml
│   └── lm_small.yaml
├── scripts/                     # Experiment runner scripts
│   ├── run_generate_reference.py
│   ├── run_verify.py
│   ├── run_attack.py
│   ├── run_grid.py
│   └── run_plots.py
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
├── run_all.sh                 # Complete test suite runner
├── EXPERIMENTS.md             # Detailed experimental protocols
├── CLAUDE.md                  # Instructions for Claude AI
└── AGENTS.md                  # Instructions for AI agents

```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/PoT_Experiments.git
cd PoT_Experiments

# Install dependencies
pip install -r requirements.txt
```

### Running Core Experiments

```bash
# E1: Separation vs Query Budget
python scripts/run_generate_reference.py --config configs/vision_cifar10.yaml
python scripts/run_grid.py --config configs/vision_cifar10.yaml --exp E1
python scripts/run_plots.py --exp_dir outputs/vision_cifar10/E1 --plot_type roc

# E2: Leakage Ablation
python scripts/run_attack.py --config configs/lm_small.yaml --attack targeted_finetune --rho 0.25

# Run complete test suite
bash run_all.sh
```

### Using Security Components

```python
from pot.security.proof_of_training import ProofOfTraining

# Initialize system
config = {
    'verification_type': 'fuzzy',
    'model_type': 'vision',
    'security_level': 'high'
}

pot = ProofOfTraining(config)

# Register and verify model
model_id = pot.register_model(model, architecture="resnet50")
result = pot.perform_verification(model, model_id, 'comprehensive')
print(f"Verified: {result.verified}, Confidence: {result.confidence:.2%}")
```

## Security Components

### 1. Fuzzy Hash Verifier (`fuzzy_hash_verifier.py`)
- **Purpose**: Enables approximate matching of model outputs while maintaining security
- **Features**:
  - Multiple hash algorithms (SSDeep, TLSH, SHA256)
  - Configurable similarity thresholds
  - Batch verification support
  - Reference hash storage and retrieval
- **Usage**:
```python
from pot.security.fuzzy_hash_verifier import FuzzyHashVerifier, ChallengeVector

verifier = FuzzyHashVerifier(similarity_threshold=0.85)
challenge = ChallengeVector(dimension=1000, topology='complex')
hash_output = verifier.generate_fuzzy_hash(model_output)
result = verifier.verify_fuzzy(candidate_hash, reference_hash)
```

### 2. Training Provenance Auditor (`training_provenance_auditor.py`)
- **Purpose**: Captures and verifies complete training history without storing full data
- **Features**:
  - Merkle tree construction for training events
  - Zero-knowledge proofs of training progression
  - Blockchain integration for immutable storage
  - Compression for efficient storage
- **Usage**:
```python
from pot.security.training_provenance_auditor import TrainingProvenanceAuditor, ProofType

auditor = TrainingProvenanceAuditor(model_id="model_001")
auditor.log_training_event(epoch=1, metrics={'loss': 0.5})
proof = auditor.generate_training_proof(0, 10, ProofType.MERKLE)
```

### 3. Token Space Normalizer (`token_space_normalizer.py`)
- **Purpose**: Handles tokenization variability and stochastic decoding in language models
- **Features**:
  - Support for multiple tokenizers (BPE, WordPiece, SentencePiece)
  - Deterministic output control
  - Semantic similarity computation
  - Multilingual support
- **Usage**:
```python
from pot.security.token_space_normalizer import TokenSpaceNormalizer, StochasticDecodingController, TokenizerType

normalizer = TokenSpaceNormalizer(TokenizerType.BPE)
controller = StochasticDecodingController(seed=42)
controller.set_deterministic_mode(temperature=0.0)
```

### 4. Integrated Proof-of-Training System (`proof_of_training.py`)
- **Purpose**: Complete verification protocol combining all components
- **Features**:
  - Model registration and fingerprinting
  - Adaptive challenge generation
  - Multi-level verification (quick/standard/comprehensive)
  - Cryptographic proof generation
  - Batch and incremental verification
- **Usage**:
```python
from pot.security.proof_of_training import ProofOfTraining

config = {
    'verification_type': 'fuzzy',
    'model_type': 'language',
    'security_level': 'high'
}

pot = ProofOfTraining(config)
model_id = pot.register_model(model, architecture="transformer")
result = pot.perform_verification(model, model_id, 'comprehensive')
proof = pot.generate_verification_proof(result)
```

## Installation

### Requirements
```bash
pip install numpy
pip install torch  # Optional, for PyTorch models
pip install transformers  # Optional, for language models

# Optional fuzzy hashing libraries
pip install python-ssdeep  # For SSDeep algorithm
pip install python-tlsh  # For TLSH algorithm
```

### Clone Repository
```bash
git clone https://github.com/yourusername/PoT_Experiments.git
cd PoT_Experiments
```

## Quick Start

### Basic Verification Flow
```python
from proof_of_training import ProofOfTraining

# Initialize system
pot = ProofOfTraining({
    'verification_type': 'fuzzy',
    'model_type': 'generic',
    'security_level': 'medium'
})

# Register your model
model_id = pot.register_model(
    model,
    architecture="resnet50",
    parameter_count=25_000_000
)

# Perform verification
result = pot.perform_verification(model, model_id, 'standard')
print(f"Verified: {result.verified}")
print(f"Confidence: {result.confidence:.2%}")

# Generate cryptographic proof
proof = pot.generate_verification_proof(result)
```

## Verification Types

### 1. Quick Verification
- Single challenge test
- Fuzzy matching
- ~1 second runtime
- Good for rapid checks

### 2. Standard Verification  
- Multiple challenges
- Statistical analysis
- ~5 seconds runtime
- Balanced accuracy/speed

### 3. Comprehensive Verification
- Full challenge suite
- Provenance checking
- Training history verification
- ~30 seconds runtime
- Maximum security

## Model Types Supported

- **Vision Models**: CNNs, Vision Transformers, etc.
- **Language Models**: GPT, BERT, T5, etc.
- **Multimodal Models**: CLIP, DALL-E style models
- **Generic Models**: Any neural network architecture

## Security Levels

- **Low**: 70% confidence threshold, fewer challenges
- **Medium**: 85% confidence threshold, standard challenges
- **High**: 95% confidence threshold, extensive challenges

## Advanced Features

### Batch Verification
```python
models = [model1, model2, model3]
model_ids = [id1, id2, id3]
results = pot.batch_verify(models, model_ids)
```

### Incremental Verification During Training
```python
for epoch in range(100):
    train_model(model)
    pot.incremental_verify(model, model_id, epoch, metrics)
```

### Cross-Platform Verification
```python
# Verify using only model outputs (offline)
outputs = {'challenge_1': output1, 'challenge_2': output2}
result = pot.cross_platform_verify(outputs, model_id)
```

## Testing

Run the test suites:
```bash
# Security component tests
python pot/security/test_fuzzy_verifier.py
python pot/security/test_provenance_auditor.py
python pot/security/test_token_normalizer.py

# Integrated demo
python pot/security/proof_of_training.py

# Full experimental validation
bash run_all.sh
```

## Experiments

The framework includes 7 core experiments (E1-E7) for validating PoT concepts:

- **E1**: Separation vs Query Budget - Test model discrimination with varying challenge sizes
- **E2**: Leakage Ablation - Robustness to challenge leakage
- **E3**: Non-IID Drift - Stability under distribution shifts
- **E4**: Adversarial Attacks - Resistance to active attacks
- **E5**: Sequential Testing - Early stopping with SPRT/e-values
- **E6**: Baseline Comparisons - Compare against naive methods
- **E7**: Ablation Studies - Component contribution analysis

See `EXPERIMENTS.md` for detailed protocols and expected results.

## Key Innovations

1. **Fuzzy Hashing**: Allows verification despite minor output variations
2. **Merkle Trees**: Efficient cryptographic proofs of training history
3. **Token Normalization**: Handles tokenization differences in language models
4. **Zero-Knowledge Proofs**: Verify training without revealing sensitive data
5. **Adaptive Challenges**: Automatically adjusts to model architecture

## Use Cases

- **Model Authentication**: Verify model identity before deployment
- **Regulatory Compliance**: Prove training methodology for regulated industries
- **IP Protection**: Cryptographically prove model ownership
- **Quality Assurance**: Ensure models meet training standards
- **Federated Learning**: Verify contributions from distributed training

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

MIT License - See LICENSE file for details

## Citation

If you use this code in your research, please cite:
```bibtex
@software{pot_experiments,
  title = {Proof-of-Training Experiments},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/PoT_Experiments}
}
```

## Contact

For questions or collaboration opportunities, please open an issue on GitHub.

## Acknowledgments

This implementation integrates concepts from cryptographic verification, blockchain technology, and neural network security research.