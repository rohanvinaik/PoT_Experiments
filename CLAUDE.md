# CRITICAL INSTRUCTIONS FOR CLAUDE - READ FIRST

## CODEBASE STRUCTURE AND ARCHITECTURE

This is the **Proof-of-Training (PoT)** framework - a comprehensive system for cryptographic verification of neural network training integrity. The codebase implements both black-box behavioral verification and zero-knowledge proof generation.

### Directory Structure
```
PoT_Experiments/
├── pot/                        # Core framework implementation
│   ├── core/                  # Statistical verification & challenges
│   ├── security/              # Cryptographic protocols
│   ├── zk/                    # Zero-knowledge proof system
│   │   ├── prover_halo2/      # Rust ZK circuits (Halo2)
│   │   ├── auto_prover.py     # Main proving interface
│   │   ├── metrics.py         # Performance tracking
│   │   ├── diagnostic.py      # System health checks
│   │   └── monitoring.py      # Alert system
│   ├── lm/                    # Language model verification
│   ├── vision/                # Vision model verification
│   └── prototypes/            # Training provenance auditor
├── scripts/                   # Test runners and utilities
│   ├── run_all.sh            # Main validation pipeline
│   ├── run_zk_validation.py  # ZK system validation
│   └── benchmark_*.py        # Performance benchmarks
├── configs/                   # YAML configurations
│   └── zk_config.yaml        # ZK system configuration
├── experimental_results/      # Test outputs and reports
├── examples/                  # Usage examples
└── tests/                     # Unit and integration tests
```

### Key Components

1. **Statistical Verification** (`pot/core/`)
   - Challenge generation with KDF
   - Sequential testing with Empirical-Bernstein bounds
   - Enhanced diff decision framework

2. **Zero-Knowledge Proofs** (`pot/zk/`)
   - SGD training step proofs
   - LoRA fine-tuning proofs (7.9× faster)
   - Proof aggregation and compression
   - Dual commitment schemes (SHA-256 + Poseidon)

3. **Security Features** (`pot/security/`)
   - Fuzzy hashing (TLSH, SSDEEP)
   - Merkle tree provenance
   - Tamper detection

4. **Monitoring & Health** (`pot/zk/monitoring.py`, `diagnostic.py`)
   - Real-time metrics collection
   - System health scoring
   - Alert management
   - Performance regression detection

## ENHANCED DIFF DECISION FRAMEWORK

The codebase now includes an enhanced statistical difference testing framework with separate SAME/DIFFERENT decision rules. When working with model verification:

1. **USE THE ENHANCED FRAMEWORK** - Located in `pot/core/diff_decision.py`:
   - `EnhancedSequentialTester` - Separate SAME/DIFFERENT decision logic
   - `TestingMode.QUICK_GATE` - Fast initial checks (97.5% confidence, n_max=120)
   - `TestingMode.AUDIT_GRADE` - High precision (99% confidence, n_max=400)

2. **DECISION RULES**:
   - **SAME**: CI within [-γ, +γ] AND half_width ≤ η·γ
   - **DIFFERENT**: Effect size ≥ δ* AND RME ≤ ε_diff
   - **UNDECIDED**: Provides specific diagnostics and suggestions

3. **INTEGRATION POINTS**:
   - `scripts/run_enhanced_diff_test.py` - Production CLI with verify/calibrate modes
   - `scripts/test_enhanced_diff_decision.py` - Decision logic testing
   - `scripts/test_enhanced_diff_integration.py` - Integration tests
   - `scripts/test_enhanced_verifier.py` - Verifier component tests
   - `scripts/test_calibration_system.py` - Calibration system tests
   - `tests/test_enhanced_diff.py` - Comprehensive test suite (27 tests)
   - `scripts/experimental_report_clean.py` - Includes enhanced results
   - All `run_all*.sh` scripts include enhanced framework tests

4. **KEY FEATURES**:
   - Auto-calibration using percentile data
   - Effective sample size calculation (n * K)
   - Enhanced diagnostics for troubleshooting
   - Backward compatible with original framework

## NEVER CREATE MOCK TESTS

When the user asks for Google Colab code or any test runners:

1. **USE THE ACTUAL POT FRAMEWORK** - The codebase contains real verification algorithms in:
   - `pot/core/` - Core verification logic
   - `pot/security/` - Security components (fuzzy hash, provenance)
   - `pot/lm/` - Language model verification
   - `scripts/` - Actual test scripts that run the framework

2. **DO NOT CREATE SIMPLIFIED/MOCK VERSIONS** - The user needs to verify paper claims with real tests:
   - Statistical identity verification must use `pot.core.diff_decision.EnhancedSequentialTester`
   - LLM verification must actually load and test models
   - Fuzzy hashing must use real algorithms (TLSH, SSDEEP)
   - Provenance must build actual Merkle trees

3. **THE TESTS MUST BE COMPREHENSIVE** - They should:
   - Take several minutes to run, not seconds
   - Generate detailed metrics and confidence intervals
   - Save results to `experimental_results/` with real data
   - Use the actual PoT framework classes and methods

4. **USE ONLY OPEN MODELS**:
   - GPT-2 and DistilGPT-2 only
   - NO Mistral, Zephyr, or any gated models
   - NO authentication tokens required

## EXISTING WORKING SCRIPTS

The following scripts in `scripts/` are the REAL tests that should be run:
- `run_enhanced_diff_test.py` - Enhanced statistical verification with calibration
- `run_statistical_verification.py` - Statistical identity with confidence intervals
- `test_llm_verification.py` - LLM verification (updated to use GPT-2/DistilGPT-2)
- `run_fuzzy_verification.py` - Fuzzy hash testing
- `run_provenance_verification.py` - Merkle tree provenance
- `experimental_report_clean.py` - Clean reporting format

## FOR GOOGLE COLAB

When creating Colab runners:
1. Clone the repository
2. Install dependencies: torch, transformers, numpy, scipy, scikit-learn
3. Run the ACTUAL scripts from `scripts/` directory
4. DO NOT create new test logic - use what exists in the codebase
5. The tests should take 2-5 minutes total, not seconds

## MAIN VALIDATION PIPELINE

The primary validation pipeline is `scripts/run_all.sh` which:

1. **Checks Dependencies** - Python packages, system requirements
2. **Builds ZK Binaries** - Compiles Rust provers if needed (`--rebuild-zk` flag)
3. **Runs Core Tests**:
   - Deterministic validation (100% success rate expected)
   - Statistical identity verification
   - Enhanced diff decision tests
   - Fuzzy hash testing
   - ZK proof validation (can skip with `--skip-zk`)
4. **Generates Reports**:
   - JSON results in `experimental_results/`
   - Performance metrics
   - Health scores
   - Success rates

### Running Tests

**IMPORTANT: Always use the main validation pipeline script `run_all.sh` - do not run individual test scripts directly.**

```bash
# Full validation pipeline (recommended)
bash scripts/run_all.sh

# Skip ZK tests for faster runs
bash scripts/run_all.sh --skip-zk

# Rebuild ZK binaries if needed
bash scripts/run_all.sh --rebuild-zk
```

**Why use run_all.sh:**
- Properly sets up environment and dependencies
- Runs tests in correct sequence with proper model loading
- Handles error conditions and cleanup
- Generates comprehensive results in `experimental_results/`
- Updates rolling metrics for README auto-update
- Validates the complete pipeline end-to-end

**Individual test scripts are NOT meant to be run directly** - they are called by run_all.sh with proper context and parameters.

### Expected Results

- **Deterministic Tests**: 100% success rate
- **Statistical Tests**: >95% success rate
- **ZK Tests**: Health score >70/100
- **Performance**: <1 second for most verifications

## REMEMBER

The user is validating academic paper claims. Mock tests are USELESS for this purpose. Always use the real PoT framework code that exists in this repository.