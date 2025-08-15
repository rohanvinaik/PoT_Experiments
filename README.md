# Proof-of-Training (PoT) Experiments

A comprehensive implementation of Proof-of-Training verification systems for neural networks, providing cryptographic verification of model training and identity.

## Overview

This repository contains a complete Proof-of-Training framework that combines multiple verification techniques to ensure model authenticity and training integrity. The system is designed to work with various model types including vision, language, and multimodal models.

## Core Components

### 1. Fuzzy Hash Verifier (`fuzzy_hash_verifier.py`)
- **Purpose**: Enables approximate matching of model outputs while maintaining security
- **Features**:
  - Multiple hash algorithms (SSDeep, TLSH, SHA256)
  - Configurable similarity thresholds
  - Batch verification support
  - Reference hash storage and retrieval
- **Usage**:
```python
from fuzzy_hash_verifier import FuzzyHashVerifier, ChallengeVector

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
from training_provenance_auditor import TrainingProvenanceAuditor

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
from token_space_normalizer import TokenSpaceNormalizer, StochasticDecodingController

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
from proof_of_training import ProofOfTraining

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
python test_fuzzy_verifier.py
python test_provenance_auditor.py
python test_token_normalizer.py
```

Run the integrated demo:
```bash
python proof_of_training.py
```

## Architecture

```
PoT_Experiments/
├── fuzzy_hash_verifier.py      # Fuzzy hashing for approximate matching
├── training_provenance_auditor.py  # Training history tracking
├── token_space_normalizer.py   # Token normalization for LLMs
├── proof_of_training.py        # Integrated verification system
├── test_*.py                   # Test suites
└── README.md                   # This file
```

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