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

### 1. Core Framework (`pot/`)
- **Challenge Generation**: KDF-based deterministic challenge creation
- **PRF Module** (`pot/core/prf.py`): HMAC-SHA256 based pseudorandom functions
- **Boundaries** (`pot/core/boundaries.py`): Confidence sequences with EB radius
- **Sequential Testing** (`pot/core/sequential.py`): Anytime-valid sequential verification
- **Fingerprinting**: Behavioral fingerprints via I/O hashing and Jacobian analysis
- **Canonicalization**: Numeric and text normalization for robust comparison
- **Statistics**: FAR/FRR calculation, confidence bounds, sequential testing

### 2. Security Components (`pot/security/`)
- **Fuzzy Hash Verifier**: Approximate matching with SSDeep/TLSH
- **Training Provenance Auditor**: Merkle tree-based training history
- **Token Space Normalizer**: Handle tokenization differences in LMs
- **Integrated PoT System**: Complete verification protocol
- **Leakage Tracking**: Challenge reuse policy enforcement (`pot/security/leakage.py`)

### 3. Audit Infrastructure (`pot/audit/`)
- **Commit-Reveal Protocol**: Tamper-evident verification trails
- **Audit Records**: JSON-based compliance documentation

### 4. Experiments (`scripts/` and `configs/`)
- E1-E7: Comprehensive experimental validation
- Grid search over model variants and challenge sizes
- Attack simulations and robustness testing

## Enhanced Verification Protocol

### New Protocol Features (Added 2025-08-16)

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

2. **Preserve Structure**:
   - Keep core framework in `pot/`
   - Security extensions in `pot/security/`
   - Configs in `configs/`
   - Scripts in `scripts/`

3. **Testing**:
   - Run component tests: `python pot/security/test_*.py`
   - Run integration: `python pot/security/proof_of_training.py`
   - Full validation: `bash run_all.sh`

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