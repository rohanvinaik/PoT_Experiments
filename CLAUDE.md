# PoT Experiments - Claude Instructions

Proof-of-Training (PoT) framework for behavioral verification of neural networks using cryptographic techniques.

## Project Structure
```
PoT_Experiments/
├── pot/
│   ├── core/          # Framework fundamentals (challenges, PRF, fingerprinting, sequential testing)
│   ├── vision/        # Vision model verification & attacks
│   ├── lm/            # Language model verification  
│   ├── semantic/      # Semantic verification & topographical learning
│   ├── security/      # Security protocols & verification
│   ├── audit/         # Cryptographic audit infrastructure
│   └── eval/          # Evaluation & metrics
├── configs/           # YAML experiment configurations
├── scripts/           # Experiment runners
└── outputs/           # Results directory
```

## Core Components

### 1. Verification Framework (`pot/core/`)
- **Challenge Generation**: KDF-based deterministic challenges (vision:freq, vision:texture, lm:templates)
- **Sequential Testing**: Empirical Bernstein bounds with early stopping (90% reduction)
- **Behavioral Fingerprinting**: IO hashing + optional Jacobian sketching (<100ms IO, ~500ms w/Jacobian)
- **Attack Suites** (`attack_suites.py`): Comprehensive attack orchestration
  - StandardAttackSuite: Basic attacks (distillation, compression, fine-tuning)
  - AdaptiveAttackSuite: Evolutionary algorithms, defense observation
  - ComprehensiveAttackSuite: Full spectrum evaluation
- **Defense Mechanisms** (`defenses.py`): Multi-layer protection
  - AdaptiveVerifier: Dynamic threshold adjustment, attack pattern learning
  - InputFilter: Adversarial detection & sanitization
  - RandomizedDefense: Smoothing & stochastic verification
  - IntegratedDefenseSystem: Orchestrated defense deployment

### 2. Vision Components (`pot/vision/`)
- **VisionVerifier**: Integrated fingerprinting + sequential testing
- **Vision Attacks** (`attacks.py`): 
  - AdversarialPatchAttack: Optimized patch generation (PGD/Adam/Momentum)
  - UniversalPerturbationAttack: Transferable perturbations
  - VisionModelExtraction: Jacobian/prediction-based extraction
  - BackdoorAttack: Injection & detection (activation clustering, neural cleanse)

### 3. Language Models (`pot/lm/`)
- **LMVerifier**: Text-aware fingerprinting, template challenges
- **Fuzzy Hashing**: Token-level matching with cross-tokenizer normalization

### 4. Security Layer (`pot/security/`)
- **ProofOfTraining**: 6-step cryptographic verification protocol
  - Profiles: quick (~1s), standard (~5s), comprehensive (~30s)
  - Security levels: low (70%), medium (85%), high (95%)
- **Expected Ranges**: Behavioral validation with auto-calibration
- **Leakage Tracking**: Challenge reuse monitoring (ρ policy)

### 5. Audit Infrastructure (`pot/audit/`)
- **Commit-Reveal Protocol**: SHA256 tamper-evident trails
- **Cryptographic Utils**: Merkle trees, timestamp proofs, ZK proofs
- **Query System**: 10,000+ record analysis with anomaly detection

### 6. Semantic Verification (`pot/semantic/`)
- **ConceptLibrary**: Gaussian/hypervector concept management
- **SemanticMatcher**: Multi-metric similarity (cosine, Euclidean, JS divergence)
- **Topographical Learning**: UMAP, t-SNE, PCA, SOM projections
  - IncrementalUMAP: Online learning for streaming data
  - GPU acceleration via RAPIDS (5-10x speedup)
  - Evolution tracking & drift detection

## Key APIs

### Basic Verification
```python
from pot.security.proof_of_training import ProofOfTraining

pot = ProofOfTraining(config)
result = pot.perform_verification(model, model_id, 'standard')
```

### Attack Execution
```python
from pot.core.attack_suites import StandardAttackSuite

suite = StandardAttackSuite()
results = suite.run_attack_suite(model, data_loader, device='cuda')
```

### Defense Deployment
```python
from pot.core.defenses import IntegratedDefenseSystem

defense = IntegratedDefenseSystem(config)
result = defense.comprehensive_defense(input_data, model, threat_level=0.7)
```

### Vision Attacks
```python
from pot.vision.attacks import execute_vision_attack

result = execute_vision_attack(
    attack_type='adversarial_patch',
    config={'patch_size': 32, 'epsilon': 0.03},
    model=model, 
    data_loader=loader
)
```

## Verification Profiles

| Profile | Challenges | Time | Confidence | Use Case |
|---------|------------|------|------------|----------|
| quick | 1 | ~1s | 70-80% | Development |
| standard | 3-5 | ~5s | 85-90% | Staging |
| comprehensive | All | ~30s | 95%+ | Production |

## Testing
```bash
bash run_all_quick.sh   # Smoke tests
bash run_all.sh         # Full suite
python -m pot.core.test_sequential_verify  # Sequential tests
```

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run verification
python scripts/run_verify_enhanced.py --config configs/vision_cifar10.yaml \
  --alpha 0.001 --beta 0.001 --tau-id 0.01

# Run attacks
python scripts/run_attack.py --config configs/vision_cifar10.yaml \
  --attack targeted_finetune

# Deploy defenses
python example_defenses.py
```

## Best Practices
1. **Determinism**: Always seed generators, use `torch.use_deterministic_algorithms(True)`
2. **Fingerprinting**: Use factory configs, cache references, batch GPU operations
3. **Security**: Apply defense-in-depth, monitor leakage, use comprehensive profile for production
4. **Performance**: IO-only fingerprinting for speed, enable Jacobian for security
5. **Logging**: Use structured JSONL logs, report confidence intervals

## Common Tasks

| Task | Command/Code |
|------|-------------|
| Add new attack | Extend `AttackConfig` in `attack_suites.py` |
| Custom defense | Subclass `BaseDefense` in `defenses.py` |
| Vision attack | Use `execute_vision_attack()` from `vision/attacks.py` |
| Semantic analysis | Create `ConceptLibrary`, use `SemanticMatcher` |
| Audit trail | Use commit-reveal protocol in `audit/commit_reveal.py` |

## Module Dependencies
- Core: PyTorch, NumPy, SciPy
- Semantic: UMAP, scikit-learn, optional RAPIDS
- Security: cryptography, hashlib
- Vision: torchvision, PIL

## Key Files
- `pot/core/challenge.py`: Challenge generation
- `pot/core/sequential.py`: Statistical verification
- `pot/core/fingerprint.py`: Behavioral fingerprinting
- `pot/core/attack_suites.py`: Attack orchestration
- `pot/core/defenses.py`: Defense mechanisms
- `pot/vision/attacks.py`: Vision-specific attacks
- `pot/security/proof_of_training.py`: Main verification protocol

Research framework - maintain separation between core (`pot/`), security (`pot/security/`), and semantic (`pot/semantic/`) components.