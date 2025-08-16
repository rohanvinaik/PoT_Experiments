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
- **Fingerprinting**: Behavioral fingerprints via I/O hashing and Jacobian analysis
- **Canonicalization**: Numeric and text normalization for robust comparison
- **Statistics**: FAR/FRR calculation, confidence bounds, sequential testing

### 2. Security Components (`pot/security/`)
- **Fuzzy Hash Verifier**: Approximate matching with SSDeep/TLSH
- **Training Provenance Auditor**: Merkle tree-based training history
- **Token Space Normalizer**: Handle tokenization differences in LMs
- **Integrated PoT System**: Complete verification protocol

### 3. Experiments (`scripts/` and `configs/`)
- E1-E7: Comprehensive experimental validation
- Grid search over model variants and challenge sizes
- Attack simulations and robustness testing

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

## Contact & Support

- Check EXPERIMENTS.md for detailed experimental protocols
- Review AGENTS.md for integration instructions
- See README.md for general project overview

## Remember

This is a research framework for validating Proof-of-Training systems. The security components provide cryptographic verification of model behavior, while the experimental framework tests robustness, efficiency, and attack resistance. Always maintain the separation between the core framework (`pot/`) and security extensions (`pot/security/`).