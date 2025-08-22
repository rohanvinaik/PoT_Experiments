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

4. **MODEL USAGE GUIDELINES**:
   - **Preferred**: GPT-2, DistilGPT-2, Pythia models (no auth required)
   - **Advanced**: Mistral, Llama-2/3, Zephyr (require HF authentication)
   - **Download as needed**: Use HuggingFace CLI for any required models
   - See "DOWNLOADING MODELS FROM HUGGINGFACE" section for setup

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

## POT_VERIFIER: CLEAN SKELETON IMPLEMENTATION

### Overview

A minimal, production-ready scaffold of the PoT verifier with:
- **HMAC challenge generation** (pre-commitment)
- **Empirical-Bernstein (EB) confidence sequences**
- **SAME/DIFFERENT decision rules** with early stopping
- **Transcript logging + evidence bundles** (ZIP)
- **Modular design** for easy extension

### Quick Start

```bash
# 1. Install base dependencies
pip install -e .

# 2. (Optional) Install HuggingFace support
pip install ".[hf]"

# 3. Run with echo models (no dependencies)
python -m scripts.run_diff --config configs/example_api.yaml

# 4. Run with local HF models (requires transformers)
python -m scripts.run_diff --config configs/example_local.yaml
```

### Directory Structure

```
pot_verifier/
├── core/           # Statistical testing and decision logic
├── lm/             # Model interfaces (HF, API, echo)
└── logging/        # Audit trail generation
scripts/
└── run_diff.py     # Main runner
configs/            # YAML configurations
```

### Extension Points

- **Prompt family** (`core/challenges.py`): Replace `iter_prompt_from_seed` with real templates
- **Scoring** (`core/scoring.py`): Swap difflib for token-level distance or fuzzy hash
- **HF loader** (`lm/hf_local.py`): Add sharded loading for large models
- **API client** (`lm/api_client.py`): Implement real deterministic clients
- **Security**: Add weight hashing (local) or TEE attestation (API)

## MAIN VALIDATION PIPELINE (MANIFEST-DRIVEN)

The primary validation pipeline now uses a **manifest-driven runner** for reproducibility:

```bash
# Run experiments defined in a YAML manifest
bash scripts/run_all.sh manifests/neurips_demo.yaml

# Or specify custom output directory
bash scripts/run_all.sh manifests/neurips_demo.yaml runs/custom_output
```

This new pipeline:
1. **Reads YAML manifest** - Defines experiments, models, and parameters
2. **Runs sequential tests** - Using EnhancedSequentialTester with EB bounds
3. **Generates evidence** - Transcripts, summaries, metrics, and bundles
4. **Bootstrap analysis** - Optional power analysis from transcripts

### Running Experiments

#### Individual Experiments

```bash
# Run a specific experiment from manifest
python tools/pot_runner.py run --manifest manifests/neurips_demo.yaml --id exp_002

# Run bootstrap power analysis
python tools/pot_runner.py power --run runs/neurips_demo --B 1000

# Create evidence bundle for an experiment
python tools/pot_runner.py bundle --run runs/neurips_demo/exp_001
```

#### Legacy Scripts (Still Available)

The following scripts from the original pipeline are still available for direct use:

```bash
# Enhanced diff decision tests
python scripts/run_enhanced_diff_test.py \
    --ref-model gpt2 --cand-model distilgpt2 --mode audit

# Security verification
python scripts/run_security_tests_simple.py

# ZK proof validation
python scripts/run_zk_validation.py
```

### Model Loading Pipeline for Custom Tests

**IMPORTANT: There is an existing model loading pipeline for testing custom model pairs!**

```bash
# Easy model selection pipeline (scans your LLM models directory)
python scripts/run_pipeline_with_models.py --models-dir /Users/rohanvinaik/LLM_Models

# Automated testing with presets
python scripts/run_pipeline_with_models.py --auto-pairs small --test-mode enhanced --non-interactive

# Size fraud detection (built-in test)
python scripts/test_size_fraud_detection.py
```

**Model Loading Features:**
- Automatically scans `/Users/rohanvinaik/LLM_Models` for valid models
- Categorizes by size (Small <1B, Medium 1B-7B, Large 7B+)
- Supports interactive model pair selection
- Built-in test scenarios (size fraud, distillation, instruction-tuning)
- Generates comprehensive reports in `experimental_results/`

**Preset Options:**
- `--auto-pairs small`: Test small model pairs (GPT-2 variants)
- `--auto-pairs large`: Test large model pairs (7B+ models)
- `--auto-pairs mixed`: Test cross-size comparisons
- `--auto-pairs base-ft`: Test base vs fine-tuned models

**Test Modes:**
- `quick`: Fast statistical tests only
- `statistical`: Statistical identity verification
- `enhanced`: Enhanced diff decision framework
- `comprehensive`: Full pipeline with ZK proofs (default)

### Expected Results

- **Deterministic Tests**: 100% success rate
- **Statistical Tests**: 
  - >95% success rate for clear cases
  - Proper SAME/DIFFERENT decisions with 97.5-99% confidence
  - Effect size detection for behavioral differences
- **Security Tests**:
  - Config Hash: 100% discrimination accuracy
  - TLSH Fuzzy Hash: 80%+ accuracy, similarity scores
  - Tokenizer: Correctly identifies architecture incompatibilities
- **ZK Tests**: Health score >70/100
- **Performance**: 
  - Small models (GPT-2, DistilGPT2): 1-2 seconds per query
  - Medium models (Pythia): ~1 second per query  
  - Large models (7B+): 8-10 seconds per query

## CRITICAL EXECUTION REQUIREMENTS

### Prerequisites & Environment Setup

**REQUIRED before running any tests:**

1. **Working Directory**: Always execute from `/Users/rohanvinaik/PoT_Experiments`
2. **Python Environment**: Python 3.11.8 with required packages
3. **Model Directory**: Models must be in `/Users/rohanvinaik/LLM_Models` 
4. **Rust Toolchain**: Version 1.88.0+ for ZK circuits (installed and working)

### Dependency Installation

```bash
# Core Python dependencies (REQUIRED)
pip install torch>=2.2.0 transformers>=4.36.2 numpy scipy scikit-learn

# Optional but recommended
pip install tlsh  # For fuzzy hashing (TLSH)
# Note: SSDeep is optional (warnings are normal)
```

### Model Access Requirements

**Models that WORK without authentication:**
- `gpt2`, `distilgpt2`, `gpt2-medium` (HuggingFace)
- `EleutherAI/pythia-70m`, `EleutherAI/pythia-160m`
- `microsoft/DialoGPT-medium`, `microsoft/DialoGPT-large`
- Models in `/Users/rohanvinaik/LLM_Models/` directory

**Models that require authentication (but can be downloaded):**
- Mistral, Zephyr, Llama-2, Llama-3 (gated models)
- Requires HuggingFace token setup (see downloading section below)

## DOWNLOADING MODELS FROM HUGGINGFACE

### Setting Up HuggingFace Authentication (for gated models)

**For gated models like Llama, Mistral, etc.:**

```bash
# Install HuggingFace CLI
pip install --upgrade huggingface_hub

# Login with your token (get from https://huggingface.co/settings/tokens)
huggingface-cli login
# Enter your token when prompted

# Verify login
huggingface-cli whoami
```

### Model Downloading Procedures

**1. Direct download to local model directory:**

```bash
# Navigate to model storage directory
cd /Users/rohanvinaik/LLM_Models

# Download open models (no auth required)
huggingface-cli download gpt2 --local-dir gpt2 --local-dir-use-symlinks False
huggingface-cli download distilgpt2 --local-dir distilgpt2 --local-dir-use-symlinks False
huggingface-cli download EleutherAI/pythia-70m --local-dir pythia-70m --local-dir-use-symlinks False
huggingface-cli download EleutherAI/pythia-160m --local-dir pythia-160m --local-dir-use-symlinks False

# Download gated models (requires authentication)
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir llama-2-7b-chat-hf --local-dir-use-symlinks False
huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct --local-dir llama-3-8b-instruct --local-dir-use-symlinks False
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --local-dir mistral-7b-instruct --local-dir-use-symlinks False
```

**2. Using Python API for programmatic downloads:**

```python
from huggingface_hub import snapshot_download
import os

# Set local directory
models_dir = "/Users/rohanvinaik/LLM_Models"

# Download function with progress
def download_model(model_name, local_name=None):
    if local_name is None:
        local_name = model_name.split('/')[-1]
    
    local_path = os.path.join(models_dir, local_name)
    print(f"Downloading {model_name} to {local_path}")
    
    snapshot_download(
        repo_id=model_name,
        local_dir=local_path,
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print(f"✅ Downloaded {model_name}")

# Example downloads
download_model("gpt2")
download_model("distilgpt2") 
download_model("EleutherAI/pythia-70m", "pythia-70m")
download_model("meta-llama/Llama-2-7b-chat-hf", "llama-2-7b-chat-hf")
```

**3. Batch download script:**

```bash
#!/bin/bash
# save as download_models.sh

MODEL_DIR="/Users/rohanvinaik/LLM_Models"
cd "$MODEL_DIR"

# Open models (no auth required)
echo "Downloading open models..."
huggingface-cli download gpt2 --local-dir gpt2 --local-dir-use-symlinks False
huggingface-cli download distilgpt2 --local-dir distilgpt2 --local-dir-use-symlinks False
huggingface-cli download EleutherAI/pythia-70m --local-dir pythia-70m --local-dir-use-symlinks False
huggingface-cli download EleutherAI/pythia-160m --local-dir pythia-160m --local-dir-use-symlinks False

# Gated models (requires authentication)
echo "Downloading gated models (requires HF login)..."
huggingface-cli download meta-llama/Llama-2-7b-chat-hf --local-dir llama-2-7b-chat-hf --local-dir-use-symlinks False
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.2 --local-dir mistral-7b-instruct --local-dir-use-symlinks False

echo "All models downloaded to $MODEL_DIR"
```

### Model Organization and Naming

**Recommended directory structure:**
```
/Users/rohanvinaik/LLM_Models/
├── gpt2/                          # 117M params, Base GPT-2
├── distilgpt2/                    # 82M params, Distilled GPT-2
├── gpt2-medium/                   # 345M params, Medium GPT-2
├── pythia-70m/                    # 70M params, Pythia small
├── pythia-160m/                   # 160M params, Pythia medium
├── llama-2-7b-chat-hf/           # 7B params, Llama-2 chat
├── llama-3-8b-instruct/          # 8B params, Llama-3 instruct
├── mistral-7b-instruct/          # 7B params, Mistral instruct
└── ...
```

**Naming conventions for the pipeline:**
- Use descriptive folder names that include model size
- Avoid spaces and special characters in folder names
- Include variant information (base, chat, instruct) in name
- The pipeline will auto-detect and categorize by parameter count

### Verification of Downloaded Models

**Check model integrity after download:**

```bash
# Verify model files exist
ls -la /Users/rohanvinaik/LLM_Models/gpt2/
# Should contain: config.json, pytorch_model.bin, tokenizer.json, etc.

# Test model loading
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_path = '/Users/rohanvinaik/LLM_Models/gpt2'
print(f'Testing {model_path}...')
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    print(f'✅ {model_path} loads successfully')
    print(f'   Model size: {model.num_parameters():,} parameters')
except Exception as e:
    print(f'❌ Error loading {model_path}: {e}')
"
```

### Integration with PoT Pipeline

**Using downloaded models with the verification pipeline:**

```bash
# Scan downloaded models
python scripts/run_pipeline_with_models.py --models-dir /Users/rohanvinaik/LLM_Models

# Test specific downloaded model pairs
python scripts/run_enhanced_diff_test.py \
  --ref-model /Users/rohanvinaik/LLM_Models/llama-2-7b-chat-hf \
  --cand-model /Users/rohanvinaik/LLM_Models/mistral-7b-instruct \
  --mode audit --verbose

# Size fraud detection with downloaded models
python scripts/run_pipeline_with_models.py --auto-pairs mixed --test-mode enhanced
```

### Storage Management

**Disk space considerations:**
- Small models (70M-345M): ~500MB each
- Medium models (1B-3B): ~2-6GB each  
- Large models (7B+): ~13-28GB each
- Plan for 50-100GB storage for comprehensive model collection

**Cleanup old models:**
```bash
# Remove specific model
rm -rf /Users/rohanvinaik/LLM_Models/model_name

# Check disk usage
du -sh /Users/rohanvinaik/LLM_Models/*
```

### Troubleshooting Model Downloads

**Common download issues:**

1. **Authentication errors for gated models:**
   ```bash
   # Re-login to HuggingFace
   huggingface-cli logout
   huggingface-cli login
   ```

2. **Download interrupted/corrupted:**
   ```bash
   # Resume interrupted download
   huggingface-cli download model_name --local-dir dir_name --resume-download
   ```

3. **Network timeout errors:**
   ```bash
   # Use smaller batch sizes or retry
   export HF_HUB_OFFLINE=0
   huggingface-cli download model_name --local-dir dir_name --max-workers 1
   ```

4. **Disk space errors:**
   ```bash
   # Check available space
   df -h /Users/rohanvinaik/LLM_Models
   # Clean up if needed
   ```

### Creating Your Own Manifest

Create a YAML file in `manifests/` with your experiment configuration:

```yaml
run_id: "my-experiment"
hmac_key_hex: "0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f0f"

global:
  n_challenges: 32
  out_root: "runs/my_experiment"
  mode: "AUDIT"  # QUICK, AUDIT, or EXTENDED

experiments:
  - id: exp_001
    ref:
      type: hf_local
      model_path: /Users/rohanvinaik/LLM_Models/gpt2
    cand:
      type: hf_local
      model_path: /Users/rohanvinaik/LLM_Models/distilgpt2
```

### Output Structure

After running experiments, find results in:
```
runs/neurips_demo/
├── exp_001/
│   ├── transcript.ndjson      # Step-by-step verification log
│   ├── summary.json           # Final verdict and statistics
│   ├── metrics.json           # RSS/CPU/IO measurements
│   ├── manifest.json          # Experiment configuration
│   └── evidence_bundle.zip    # Complete audit package
└── batch_summary.json         # Overview of all experiments
```

### Pipeline Features

**1. Model Support:**
- **Local HuggingFace models**: Automatically loads from `/Users/rohanvinaik/LLM_Models/`
- **Echo stubs**: For testing without models
- **API endpoints**: Extensible for remote model verification

**2. Verification Modes:**
- **QUICK**: 10-120 queries, 97.5% confidence
- **AUDIT**: 30-400 queries, 99% confidence (default)
- **EXTENDED**: 50-800 queries, 99.9% confidence

**3. Evidence Generation:**
- Cryptographic challenge generation via HMAC-SHA256
- Complete transcripts with inputs/outputs
- Performance metrics (memory, CPU, I/O)
- Reproducible evidence bundles for review

### Integration with Existing Pipeline

The new runner seamlessly integrates with the existing PoT framework:
- Uses `pot.core.diff_decision.EnhancedSequentialTester` when available
- Falls back to minimal implementation if imports fail
- Compatible with all existing model verification features

### Example Experiments in neurips_demo.yaml

The included `manifests/neurips_demo.yaml` demonstrates various verification scenarios:

1. **exp_001**: Echo test for pipeline validation
2. **exp_002**: GPT-2 vs DistilGPT-2 (distillation detection)
3. **exp_003**: Pythia-70M self-consistency check
4. **exp_004**: Pythia-70M vs Pythia-160M (size difference detection)
5. **exp_005**: API endpoint comparison example
6. **exp_006**: Robustness test with wrapper attack

To run specific experiments:
```bash
# Run just the GPT-2 vs DistilGPT-2 comparison
python tools/pot_runner.py run --manifest manifests/neurips_demo.yaml --id exp_002

# Run with custom output directory
bash scripts/run_all.sh manifests/neurips_demo.yaml runs/custom_output
```

### Error Handling & Common Issues

**Normal Warnings (IGNORE these):**
```
FuzzyHashVerifier not available  # SSDeep not installed
TokenSpaceNormalizer not available  # Optional component
```

**Fatal Errors to Address:**
- `ModuleNotFoundError`: Missing dependencies, install with pip
- `CUDA out of memory`: Use `--skip-zk` flag or smaller models
- `No such file or directory`: Check working directory and model paths
- `Permission denied`: Check file permissions on scripts

### Debugging Failed Tests

**If run_all.sh fails:**
1. Check `experimental_results/run_all_YYYYMMDD_HHMMSS.log` for details
2. Try `bash scripts/run_all.sh --skip-zk` to isolate ZK issues
3. Verify models exist: `ls /Users/rohanvinaik/LLM_Models/`
4. Check disk space: `df -h` (needs >2GB free)

**If individual scripts fail:**
- NEVER run them directly, always use the pipeline
- Check if the script requires specific model paths
- Verify the script exists in `scripts/` directory

### Step-by-Step Execution Guide

**For Full Pipeline Validation:**
```bash
# 1. Navigate to correct directory
cd /Users/rohanvinaik/PoT_Experiments

# 2. Verify environment
python --version  # Should be 3.11.8
which python     # Should point to correct environment

# 3. Run full validation (15-20 minutes)
bash scripts/run_all.sh

# 4. Check results
ls experimental_results/  # Should contain new JSON files
tail experimental_results/run_all_*.log  # Check for errors
```

**For Quick Testing:**
```bash
# Fast test without ZK proofs (2-5 minutes)
cd /Users/rohanvinaik/PoT_Experiments
bash scripts/run_all.sh --skip-zk
```

**For Custom Model Testing:**
```bash
# Test specific model pairs
cd /Users/rohanvinaik/PoT_Experiments
python scripts/run_pipeline_with_models.py --auto-pairs small --non-interactive

# Size fraud detection
python scripts/test_size_fraud_detection.py
```

### Output Validation

**Successful run should produce:**
- Exit code 0 (no errors)
- New files in `experimental_results/` with timestamp
- Updated `experimental_results/rolling_metrics.json`
- Console output showing "✅" success indicators
- ZK proofs generated and verified (if not using --skip-zk)

**Success indicators to look for:**
```
✅ Standard deterministic validation passed
✅ Enhanced diff decision tests passed  
✅ ZK system pre-flight checks passed
✅ All ZK prover binaries ready
```

### File Permissions & System Requirements

**Required permissions:**
- Read/write access to `/Users/rohanvinaik/PoT_Experiments/`
- Execute permissions on `scripts/*.sh` files
- Network access for downloading models (if needed)

**System requirements:**
- **Memory**: 16GB RAM minimum (for 7B models)
- **Storage**: 10GB free space for models and results
- **CPU**: Multi-core recommended (M2 Pro or equivalent)

### Recovery from Common Failures

**If tests hang or timeout:**
```bash
# Kill hung processes
pkill -f python
pkill -f rust

# Clean temporary files
rm -rf /tmp/pot_*

# Restart with clean state
bash scripts/run_all.sh
```

**If ZK tests fail:**
```bash
# Rebuild ZK binaries
bash scripts/run_all.sh --rebuild-zk

# Or skip ZK if hardware incompatible
bash scripts/run_all.sh --skip-zk
```

## TROUBLESHOOTING GUIDE

### Quick Diagnostic Commands

**Before asking for help, run these diagnostic commands:**

```bash
# Check working directory
pwd  # Should be /Users/rohanvinaik/PoT_Experiments

# Check Python environment  
python --version && which python

# Check required packages
python -c "import torch, transformers, numpy, scipy; print('✅ Core deps OK')"

# Check model directory
ls -la /Users/rohanvinaik/LLM_Models/ | head -5

# Check script permissions
ls -la scripts/run_all.sh

# Check recent log files
ls -lt experimental_results/*.log | head -3
```

### Most Common Issues & Solutions

**1. "No such file or directory" when running scripts**
```bash
# Solution: Verify working directory
cd /Users/rohanvinaik/PoT_Experiments
ls scripts/run_all.sh  # Should exist
```

**2. "ModuleNotFoundError: No module named 'pot'"**
```bash
# Solution: Run from correct directory
cd /Users/rohanvinaik/PoT_Experiments
python -c "import pot; print('✅ pot module found')"
```

**3. "Permission denied" errors**
```bash
# Solution: Fix script permissions
chmod +x scripts/*.sh
chmod +x scripts/*.py
```

**4. ZK tests fail with "cargo not found"**
```bash
# Solution: Check Rust installation
which cargo  # Should return a path
cargo --version  # Should be 1.88.0+

# If missing: install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

**5. Models not found or authentication errors**
```bash
# Solution: Use only open models
# AVOID: mistral, zephyr, llama-2 (require auth)
# USE: gpt2, distilgpt2, pythia models

# Test model access
python -c "from transformers import AutoTokenizer; tok = AutoTokenizer.from_pretrained('gpt2'); print('✅ GPT-2 accessible')"
```

**6. Out of memory errors**
```bash
# Solution: Use smaller models or skip ZK
bash scripts/run_all.sh --skip-zk

# For model testing, use small models only
python scripts/test_size_fraud_detection.py  # Uses Pythia-70M/160M
```

### Emergency Reset Procedure

**If everything is broken and you need to start fresh:**

```bash
# 1. Clean all processes
pkill -f python
pkill -f cargo
pkill -f rust

# 2. Clean temporary files
rm -rf /tmp/pot_*
rm -rf experimental_results/temp_*

# 3. Reset to working directory
cd /Users/rohanvinaik/PoT_Experiments

# 4. Verify core dependencies
python -c "import torch, transformers; print('Dependencies OK')"

# 5. Test with minimal command
bash scripts/run_all.sh --skip-zk
```

### When to Use Different Commands

**Use `bash scripts/run_all.sh` when:**
- You want to validate the complete framework
- You have 15-20 minutes for full testing
- You want to reproduce paper results
- ZK circuits are working (Rust/cargo available)

**Use `bash scripts/run_all.sh --skip-zk` when:**
- ZK tests are failing due to hardware/Rust issues
- You want faster testing (5-10 minutes)
- You only need statistical verification
- Memory constraints prevent ZK proof generation

**Use `python scripts/test_size_fraud_detection.py` when:**
- You want to test specific model pairs (Pythia)
- You need a quick functional test (2-3 minutes)
- You want to validate size fraud detection specifically

**Use `python scripts/run_pipeline_with_models.py` when:**
- You have custom models in `/Users/rohanvinaik/LLM_Models/`
- You want interactive model selection
- You need to test specific model combinations

## REMEMBER

The user is validating academic paper claims. Mock tests are USELESS for this purpose. Always use the real PoT framework code that exists in this repository.

**FINAL CHECKLIST for LLM instances:**
- [ ] Working directory: `/Users/rohanvinaik/PoT_Experiments`
- [ ] Python 3.11.8 with torch, transformers, numpy, scipy
- [ ] Models accessible (gpt2, distilgpt2, pythia)
- [ ] Script permissions set (`chmod +x scripts/*`)
- [ ] Use `bash scripts/run_all.sh` for full validation
- [ ] Check logs in `experimental_results/` if errors occur
- [ ] Use `--skip-zk` flag if ZK tests fail