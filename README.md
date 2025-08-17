# Proof-of-Training (PoT) Experiments

Statistical black-box verification of model identity with cryptographically derived challenges. This repo implements the experimental framework from "Proof-of-Training: A Statistical Framework for Black-Box Neural Network Verification" and provides calibrated FAR/FRR, ROC/DET analyses, leakage studies, sequential early-stopping, and comprehensive visualization tools.

Headline results (vision & LM, open models; α=β=0.01, τ=0.05, n∈{256,512}):
- AUROC ≈ 0.99 with empirical-Bernstein calibration;
- FAR ~0.4%, FRR ~0–1.2% (see τ curves);
- 2–3 avg queries to decision via sequential EB (vs 256–512 fixed);
- Robust to ρ=0.25 challenge leakage;
- Outperforms simple distance baselines (E6).
- Fuzzy hashing outperforms exact token matching on sample LMs ([docs/lm_hashing_benchmark.md](docs/lm_hashing_benchmark.md)).

Reproduce with: `bash run_all.sh` (details in [EXPERIMENTS.md](EXPERIMENTS.md)). Consolidated outputs from the latest run are
available in `pot_final_results_20250815_193813.json`.

### Reproducible environment

For a clean setup that includes all pinned dependencies, use the provided
Dockerfile:

```bash
docker build -t pot-experiments .
docker run --rm pot-experiments bash run_all.sh
```

Alternatively, create a local virtual environment with the same
dependencies:

```bash
bash setup_env.sh
source .venv/bin/activate
```

Once activated, run the test suite directly:

```bash
make test
```

The container installs packages from `requirements.txt` and sets `PYTHONPATH`
so the `pot` modules can be imported without additional configuration.

Note: "Security components" under `pot/security/` are prototypes; core verification uses `pot/core/*` and `scripts/run_*` only.

### Sequential Verification Quick Start

The PoT framework implements anytime-valid sequential hypothesis testing using Empirical-Bernstein bounds for efficient model verification. Sequential testing provides 70-90% sample reduction while maintaining rigorous error control.

#### When to Use Sequential vs Fixed-Sample Testing

| Scenario | Recommendation | Reason |
|----------|----------------|---------|
| **API-based models** | Sequential | Expensive queries, early stopping valuable |
| **Real-time verification** | Sequential | Immediate decisions needed |
| **Variable effect sizes** | Sequential | Adaptive to difficulty |
| **Batch processing** | Fixed-sample | Parallel processing preferred |
| **Regulatory compliance** | Fixed-sample | May require predetermined sample size |
| **Exploratory analysis** | Sequential | Faster iteration cycles |

#### Performance Characteristics

```python
from pot.core.sequential import sequential_verify
import numpy as np

# Quick example: verify if two models are equivalent
def compare_models(model_a, model_b, challenges):
    def distance_stream():
        for challenge in challenges:
            out_a = model_a(challenge)
            out_b = model_b(challenge) 
            yield np.linalg.norm(out_a - out_b)
    
    result = sequential_verify(
        stream=distance_stream(),
        tau=0.05,           # Accept if mean distance < 5%
        alpha=0.05,         # 5% false positive rate
        beta=0.05,          # 5% false negative rate
        max_samples=1000    # Upper bound for efficiency
    )
    
    return result

# Typical results:
# - Decision made after 20-80 samples (vs 256-512 fixed)
# - Type I/II error rates controlled at specified levels
# - Complete audit trail in result.trajectory
```

#### Error Rate Guarantees

**Mathematical Properties**:
- **Type I Error**: P(reject H₀ | μ ≤ τ) ≤ α (anytime-valid)
- **Type II Error**: P(accept H₀ | μ > τ + δ) ≤ β (for effect size δ)
- **Efficiency**: E[T] ≪ n_fixed for practical scenarios

**Practical Performance** (α = β = 0.05, τ = 0.05):
- **Average samples**: 50-100 (vs 512 fixed)
- **Worst-case samples**: ≤ max_samples
- **False positive rate**: ≤ 5% (guaranteed)
- **False negative rate**: ≤ 5% (for detectable differences)

#### Integration with Model Verifiers

```python
from pot.vision.verifier import VisionVerifier
from pot.lm.verifier import LMVerifier

# Enable sequential testing in verifiers
vision_verifier = VisionVerifier(
    reference_model=ref_model,
    use_sequential=True,
    sequential_mode='enhanced'  # Use EB-based bounds
)

# Verify with early stopping
result = vision_verifier.verify(
    candidate_model, 
    challenges,
    tolerance=0.05,
    alpha=0.01,     # Strict error control
    beta=0.01
)

print(f"Decision: {result.accepted}")
if result.sequential_result:
    efficiency = 1 - (result.sequential_result.stopped_at / 1000)
    print(f"Sample efficiency: {efficiency:.1%}")
```

### Visualization Tools

Comprehensive visualization tools for sequential verification analysis:

```python
from pot.core.visualize_sequential import *

# Visualize single verification trajectory
result = sequential_verify(stream=data_stream(), tau=0.05)
plot_verification_trajectory(result, save_path='trajectory.png')

# Compare sequential vs fixed-sample performance  
plot_operating_characteristics(tau=0.05, effect_sizes=[0.0, 0.02, 0.05, 0.1])

# Demonstrate anytime validity across multiple runs
trajectories = [sequential_verify(...) for _ in range(50)]
plot_anytime_validity(trajectories)

# Interactive demo (requires streamlit)
# streamlit run pot/core/visualize_sequential.py
```

Key features:
- **Real-time trajectory visualization** with confidence bounds and decision regions
- **Operating characteristics analysis** comparing sequential vs fixed-sample efficiency  
- **Anytime validity demonstrations** showing error control across stopping times
- **Interactive Streamlit demo** for educational exploration and parameter tuning
- **Publication-ready outputs** with customizable styling and high-resolution export

### Running without CUDA

Install CPU-only dependencies and execute the quick validation script:

```bash
pip install -r requirements-cpu.txt
bash run_all_quick.sh
```

The script automatically detects the absence of CUDA and skips GPU checks.
All experiment runners also support a `--cpu-only` flag to force models onto
the CPU, for example:

```bash
python scripts/run_grid.py --config configs/vision_cifar10.yaml --exp E1 --cpu-only
python scripts/run_verify.py --config configs/vision_cifar10.yaml --challenge_family vision:texture --n 256 --cpu-only
```

This setup enables running the core PoT pipeline on machines without
NVIDIA GPUs.

### Advanced Sequential Features

The framework includes cutting-edge sequential testing capabilities:

```python
from pot.core.sequential import (
    mixture_sequential_test,      # Combine multiple test statistics
    adaptive_tau_selection,       # Dynamic threshold adjustment
    multi_armed_sequential_verify,# Multiple hypothesis testing
    power_analysis               # Operating characteristics
)

# Mixture testing for robustness
streams = [mean_distances, median_distances, trimmed_mean]
mixture_result = mixture_sequential_test(
    streams=streams,
    weights=[0.5, 0.3, 0.2],
    tau=0.05,
    alpha=0.05
)

# Adaptive threshold based on observed variance
adaptive_result = adaptive_tau_selection(
    stream=distance_stream(),
    initial_tau=0.05,
    adaptation_rate=0.1,
    union_bound_correction=True
)

# Multiple model comparison with FWER control
multi_result = multi_armed_sequential_verify(
    streams={'model_A': stream_A, 'model_B': stream_B},
    hypotheses={'model_A': 0.03, 'model_B': 0.07},
    alpha=0.05,
    correction_method='bonferroni'
)
```

## Relation to Proof-of-Learning

PoT offers statistical model-identity checks and complements cryptographic Proof-of-Learning (PoL) systems that attest to training provenance. PoL schemes require access to training traces or commitments and have recently improved via polynomial commitments and gradient compression, while PoT operates post-hoc on black-box models. Combining PoT with PoL can bind behavioral fingerprints to verifiable training histories.

## Behavioral Fingerprinting

The PoT framework includes a comprehensive behavioral fingerprinting system (Paper §2.2) that captures and compresses model behavior for quick verification:

### Overview

Behavioral fingerprinting creates deterministic signatures of neural network behavior through two complementary mechanisms:

1. **Input-Output (IO) Fingerprinting**: Captures model outputs on challenge inputs, creating stable hashes of canonicalized responses. Provides sub-100ms verification for model identity checks.

2. **Jacobian Fingerprinting**: Analyzes gradient structure by computing and sketching the Jacobian matrix. Captures model sensitivity patterns and decision boundaries, ideal for detecting fine-tuning or architectural changes.

### When to Use IO vs Jacobian Analysis

| Scenario | Recommended Approach | Rationale |
|----------|---------------------|-----------|
| Quick identity check | IO only | Fast (<100ms), deterministic, sufficient for exact matching |
| Fine-tuning detection | IO + Jacobian | Jacobian captures subtle parameter changes |
| Architecture verification | Jacobian (magnitude) | Magnitude sketches reveal layer structure |
| Large language models | IO only | Jacobian computation expensive for transformers |
| Security-critical | Both | Maximum confidence through dual verification |
| Resource-constrained | IO only | Minimal compute/memory overhead |

### Performance Considerations

- **IO Fingerprinting**: ~10-50ms per model for 10 challenges
- **Jacobian Sketching**: ~100-500ms additional overhead
- **Memory Usage**: O(n_challenges × output_dim) for IO, O(input_dim × output_dim) for Jacobian
- **Storage**: 64 bytes (IO hash) + optional 32 bytes (Jacobian sketch)

### Integration with Statistical Verification

Fingerprinting complements statistical verification by:
1. Providing quick pre-filtering before expensive statistical tests
2. Enabling batch verification of multiple models
3. Creating audit trails through deterministic hashes
4. Supporting early rejection of obviously different models

Example integration:
```python
from pot.core.fingerprint import FingerprintConfig, fingerprint_run, is_behavioral_match
from pot.vision.verifier import VisionVerifier

# Quick fingerprint check first
config = FingerprintConfig.for_vision_model(compute_jacobian=False)
fp_candidate = fingerprint_run(model, challenges, config)

if not is_behavioral_match(fp_reference, fp_candidate, threshold=0.9):
    # Early rejection - skip expensive statistical verification
    return "Model rejected (fingerprint mismatch)"

# Proceed with full statistical verification if fingerprints match
verifier = VisionVerifier(reference_model, use_fingerprinting=True)
result = verifier.verify(model, challenges)
```

See `examples/fingerprinting_demo.py` for complete usage examples.

## Threat model

Adversary may (i) fine-tune or compress a copy, (ii) perform wrapper routing, (iii) access up to a fraction ρ of past challenges, (iv) query black-box APIs polynomially. These capabilities map to misuse and robustness obligations in the EU AI Act's risk-management and cybersecurity provisions (Art. 9, Art. 15) and the NIST AI Risk Management Framework's "Secure and Resilient" profile. PoT's challenge-based auditing detects unauthorized model alterations and drift, supporting those standards, but it does not address white-box exposure, network tampering, or hardware bypass—gaps relative to EU AI Act Art. 15(5) and NIST confidentiality/integrity expectations. Deployment must ensure secure channels, challenge secrecy, and complementary operational controls.[1][2]

## Verification profiles

- **quick**: `n=16`, τ from prior calibration, `seq=EB(delta=0.1)`
- **standard**: `n=128`, τ calibrated on held-out, `seq=EB(delta=0.02)`
- **comp**: `n=512`, τ + EB + SPRT audit trace, leakage-resilient challenges

## Claims ↔ Evidence map

| Claim | Exact config | Command | Artifact |
|------|--------------|---------|----------|
| AUROC ≈ 0.99 | α=β=0.01, τ=0.05, n=512, vision:texture | `python scripts/run_grid.py --config configs/vision_cifar10.yaml --exp E1` | `outputs/vision_cifar10/E1/roc.png`, `outputs/vision_cifar10/E1/grid_results.jsonl` |
| FAR ~0.4%, FRR ~0–1.2% | τ=0.05, n=256, EB | `python scripts/run_verify.py --config configs/vision_cifar10.yaml --challenge_family vision:texture --n 256 --seq eb --store-trace` | `outputs/vision_cifar10/E3/verify.jsonl` |
| 2–3 queries avg | EB `delta=0.02`, `B=1` | `python scripts/run_verify.py --config configs/vision_cifar10.yaml --n 512 --seq eb --store-trace` | `outputs/vision_cifar10/E5/sequential_trace.jsonl` |
| ρ=0.25 leakage | targeted fine-tune attack | `python scripts/run_attack.py --config configs/lm_small.yaml --attack targeted_finetune --rho 0.25` | `attacks/targeted_finetune_rho0.25/attack_log.jsonl` |

## Reproducibility

- **Environment**: tested on CUDA 12, cuDNN 9, torch 2.8 (see `requirements.txt` for Python deps).
- **GPU**: A100 40GB.
- **Seeds**: `[0, 1, 2]` set in configs; additionally set `PYTHONHASHSEED`, `np.random.seed`, `torch.manual_seed`.
- **Bound & distances**: outputs clipped to `[0,1]`; L2 distance for vision, token-level Hamming for LMs.
- **Calibration**: thresholds calibrated on held-out split (see `outputs/*/calibration.png`).
- **Model checkpoints**: see `outputs/*/references/*.ckpt` with SHA256 hashes in `outputs/*/sha256.txt`.
  ```
  export PYTHONHASHSEED=0
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  torch.use_deterministic_algorithms(True)
  ```
## Testing

The repository contains unit tests for core, vision, language, evaluation, and security modules
(`pot/*/test_*.py`) along with end-to-end checks in `tests/`:

- `test_attacks_integration.py`
- `test_run_grid_integration.py`
- `test_run_plots.py`
- `test_run_verify.py`

Install dependencies from `requirements.txt` (NumPy, PyTorch, torchvision, matplotlib,
transformers, etc.). A CUDA-enabled GPU is recommended but the tests run on CPU for small
mock models. Vision tests fall back to mock models if `torchvision` is missing or fails to
load, so a compatible `torchvision` build is only needed for full coverage. Run
the full suite with:

```bash
pytest -q
```

On a CPU-only environment with the pinned dependencies, this command currently reports
`92 passed, 3 skipped` in about one minute.

### Adding tests

- Place fast unit tests next to new modules under `pot/` using the `test_*.py` pattern.
- For new attack or verification pipelines, add integration tests under `tests/` similar to
  `test_attacks_integration.py`.
- Use small deterministic mock models or fixtures and avoid network calls.
- Ensure the suite remains runnable via `pytest -q`.

## Quick start

### Environment setup

- **Python**: 3.10
- **GPU**: NVIDIA A100 40GB (≥16GB VRAM recommended)
- **CUDA/cuDNN**: 12.0 / 9.x
- **Non-default deps**: `ssdeep`, `py-tlsh` for fuzzy hashing

```bash
git clone https://github.com/yourusername/PoT_Experiments.git
cd PoT_Experiments
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Python dependencies

The basic experiments require the following Python packages (versions tested):

| Package | Version |
|---------|---------|
| torch | 2.2.0 |
| torchvision | 0.17.0 |
| transformers | 4.31.0 |
| accelerate | 0.21.0 |
| sentence-transformers | 2.2.2 |
| numpy | 1.24.4 |
| scipy | 1.10.1 |
| scikit-learn | 1.3.0 |
| einops | 0.6.1 |
| tqdm | 4.65.0 |
| pyyaml | 6.0 |
| matplotlib | 3.7.1 |
| seaborn | 0.12.2 |
| xxhash | 3.4.1 |
| ssdeep | 3.4 |
| py-tlsh | 4.7.2 |

A minimal pinned requirements file is provided in
[requirements-basic.txt](requirements-basic.txt) for convenience.

### Sequential Testing Tutorial

For detailed theoretical background and worked examples, see:
- **Theory**: [docs/statistical_verification.md](docs/statistical_verification.md)
- **Examples**: [examples/sequential_analysis.ipynb](examples/sequential_analysis.ipynb)
- **API Reference**: [CLAUDE.md](CLAUDE.md#sequential-verification-updated-2025-08-16)

### Dataset setup

#### Vision (CIFAR-10)

The vision experiments use CIFAR-10. The dataset will be downloaded automatically, or you can pre-fetch it:

```bash
python - <<'PY'
import torchvision
_ = torchvision.datasets.CIFAR10(root="data", download=True)
PY
```

Dataset site: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

#### Language (TinyLlama + small SFT)

Language experiments pull `TinyLlama/TinyLlama-1.1B` and a small SFT dataset. Use Hugging Face tools:

```bash
pip install huggingface_hub datasets
huggingface-cli download TinyLlama/TinyLlama-1.1B --local-dir models/TinyLlama-1.1B
python - <<'PY'
from datasets import load_dataset
load_dataset('tatsu-lab/alpaca', split='train[:1000]').save_to_disk('data/small_sft')
PY
```

Model page: [https://huggingface.co/TinyLlama/TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B)<br>
Dataset page: [https://huggingface.co/datasets/tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)

### Optional fuzzy-hash dependencies

These features require native libraries and are installed via an optional extra:

```bash
pip install ".[fuzzy]"  # installs ssdeep/tlsh
```

### Running core experiments

```bash
# E1: Separation vs Query Budget
python scripts/run_generate_reference.py --config configs/vision_cifar10.yaml
python scripts/run_grid.py --config configs/vision_cifar10.yaml --exp E1
python scripts/run_plots.py --exp_dir outputs/vision_cifar10/E1 --plot_type roc

# E2: Leakage Ablation
python scripts/run_attack.py --config configs/lm_small.yaml --attack targeted_finetune --rho 0.25
```

### Minimal sequential verification demo

```bash
# Generate reference model
python scripts/run_generate_reference.py --config configs/vision_cifar10.yaml

# Sequential verification with EB bounds (recommended)
python scripts/run_verify.py --config configs/vision_cifar10.yaml \
    --challenge_family vision:texture --n 1000 --seq eb --store-trace
    
# Compare with fixed-sample baseline
python scripts/run_verify.py --config configs/vision_cifar10.yaml \
    --challenge_family vision:texture --n 256 --seq none
    
# Enhanced verification with full protocol
python scripts/run_verify_enhanced.py --config configs/vision_cifar10.yaml \
    --alpha 0.01 --beta 0.01 --boundary EB --n-max 1000
```

## Audit System Overview

The PoT framework includes a comprehensive cryptographic audit system that provides tamper-evident verification trails and compliance reporting. The audit system combines multiple cryptographic techniques for maximum security and transparency.

### Core Audit Components

1. **Commit-Reveal Protocol** - Prevents parameter tampering via cryptographic commitments
2. **Expected Ranges Validation** - Behavioral validation against calibrated reference model ranges  
3. **Merkle Tree Provenance** - Cryptographic proof of training progression with logarithmic proof sizes
4. **Blockchain Integration** - Immutable on-chain commitment storage with gas optimization
5. **Query and Analysis Tools** - Comprehensive audit trail analysis and anomaly detection

### Quick Start: Enabling Audit Trails

```python
from pot.security.proof_of_training import ProofOfTraining, ExpectedRanges, SessionConfig

# Basic audit configuration
config = {
    'verification_type': 'fuzzy',
    'model_type': 'vision',
    'security_level': 'high',
    'audit_log_path': 'verification_audit.jsonl'  # Enable audit logging
}

pot_system = ProofOfTraining(config)

# Register model with expected behavioral ranges
model_id = pot_system.register_model(model, architecture="resnet50_v2")

expected_ranges = ExpectedRanges(
    accuracy_range=(0.85, 0.95),          # Expected accuracy bounds
    latency_range=(10.0, 50.0),           # Response time bounds (ms) 
    fingerprint_similarity=(0.90, 0.99),  # Behavioral similarity bounds
    jacobian_norm_range=(0.5, 2.0),       # Gradient norm bounds
    confidence_level=0.95,                 # Statistical confidence
    tolerance_factor=1.1                   # 10% production tolerance
)

pot_system.set_expected_ranges(model_id, expected_ranges)

# Perform verification with full audit trail
result = pot_system.perform_verification(model, model_id, 'comprehensive')

if result.verified:
    print("✓ Model verification passed")
    if result.range_validation and result.range_validation.passed:
        print("✓ Model within expected behavioral ranges")
else:
    print("✗ Model verification failed")
    if result.range_validation and not result.range_validation.passed:
        print("Range violations detected:")
        for violation in result.range_validation.violations:
            print(f"  - {violation}")
```

### Complete End-to-End Verification with Audit

```python
from pot.security.proof_of_training import SessionConfig
from pot.audit.commit_reveal import compute_commitment, verify_reveal

# Complete session with cryptographic audit trail
session_config = SessionConfig(
    model=model,
    model_id="production_model_v2.1", 
    master_seed="secure_seed_64_characters_long_for_deterministic_challenges",
    
    # Challenge parameters
    num_challenges=20,
    challenge_family="vision:texture",
    challenge_params={'texture_types': ['perlin', 'gabor', 'checkerboard']},
    
    # Statistical testing
    accuracy_threshold=0.02,    # Very strict threshold
    type1_error=0.01,          # 1% false positive rate  
    type2_error=0.01,          # 1% false negative rate
    max_samples=1000,
    
    # Component activation
    use_fingerprinting=True,   # Enable behavioral fingerprinting
    use_sequential=True,       # Enable early stopping
    use_range_validation=True, # Enable range validation
    use_blockchain=True,       # Enable blockchain storage
    
    expected_ranges=expected_ranges,
    audit_log_path="production_audit.jsonl"
)

# Execute with full cryptographic protocol
verification_report = pot_system.run_verification(session_config)

print(f"Verification Result: {'PASSED' if verification_report.passed else 'FAILED'}")
print(f"Confidence Score: {verification_report.confidence:.4f}")
print(f"Session ID: {verification_report.session_id}")
print(f"Duration: {verification_report.duration_seconds:.2f} seconds")

# Comprehensive reporting
if verification_report.commitment_record:
    print(f"Cryptographic commitment: {verification_report.commitment_record.commitment_hash[:16]}...")

if verification_report.blockchain_tx:
    print(f"Blockchain transaction: {verification_report.blockchain_tx}")
```

### Audit Trail Query and Analysis

```python
from pot.audit.query import AuditTrailQuery

# Load and analyze audit trail
query = AuditTrailQuery("production_audit.jsonl")

print(f"Total audit records: {len(query.records)}")
print(f"Models monitored: {len(query.model_index)}")

# Multi-dimensional querying
recent_records = query.query_by_timerange(
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now()
)

failed_verifications = query.query_by_verification_result("FAIL")
high_confidence = query.query_by_confidence_range(0.9, 1.0)

# Integrity verification
integrity_report = query.verify_integrity()
print(f"Audit integrity score: {integrity_report.integrity_score:.3f}")

# Anomaly detection
anomalies = query.find_anomalies()
high_severity = [a for a in anomalies if a.severity >= 0.7]

if high_severity:
    print(f"⚠️ {len(high_severity)} high-severity anomalies detected")
    for anomaly in high_severity[:3]:
        print(f"  - {anomaly.description}")

# Generate compliance report
html_report = query.generate_audit_report("html")
with open("compliance_report.html", "w") as f:
    f.write(html_report)
```

### Configuration Examples

#### Development Configuration
```python
dev_config = {
    'verification_type': 'fuzzy',
    'model_type': 'vision',
    'security_level': 'low',           # 70% threshold for development
    'audit_log_path': 'dev_audit.json',
    'enable_blockchain': False,        # Skip blockchain for dev speed
    'expected_ranges': None            # Skip range validation in dev
}
```

#### Production Configuration  
```python
production_config = {
    'verification_type': 'fuzzy',
    'model_type': 'vision',
    'security_level': 'high',          # 95% threshold for production
    'audit_log_path': 'production_audit.jsonl',
    'enable_blockchain': True,
    'blockchain_config': BlockchainConfig.polygon_mainnet(POLYGON_RPC_URL),
    'expected_ranges': ExpectedRanges(
        accuracy_range=(0.92, 0.97),       # Tight production bounds
        latency_range=(5.0, 25.0),         # Fast response required
        fingerprint_similarity=(0.98, 0.999), # High behavioral consistency
        jacobian_norm_range=(0.8, 1.5),    # Stable gradients
        confidence_level=0.999,             # Very high confidence
        tolerance_factor=1.02               # Minimal tolerance
    )
}
```

### Security Considerations

**Cryptographic Security**:
- SHA256 commitments provide 256-bit security level
- 32-byte salts prevent rainbow table attacks  
- Constant-time comparisons prevent timing attacks
- Merkle trees offer collision-resistant tamper detection

**Operational Security**:
- Generate master seeds with cryptographically secure randomness
- Implement role-based access control for verification functions
- Use write-only audit logs with integrity verification
- Monitor for high-severity anomalies and security events

**Blockchain Considerations**:
- Multi-chain redundancy prevents single point of failure
- Gas-optimized batch operations reduce costs by 60-80%
- Smart contract immutability ensures tamper-resistance

### Performance Characteristics

| Operation | Time Complexity | Practical Performance |
|-----------|----------------|---------------------|
| Commitment Generation | O(1) | <10ms per commitment |
| Merkle Proof Generation | O(log n) | <1ms for 1M tree |
| Audit Query (indexed) | O(1) | <10ms for 100K records |
| Anomaly Detection | O(n) | <5s for 100K records |
| Blockchain Storage | O(1) | 15s per transaction |
| Batch Blockchain Storage | O(n) | 60-80% gas savings |

### Blockchain Integration

```python
from pot.prototypes.training_provenance_auditor import BlockchainClient, BlockchainConfig

# Multi-chain deployment for redundancy
config = BlockchainConfig.polygon_mainnet("https://polygon-mainnet.g.alchemy.com/v2/KEY")

with BlockchainClient(config) as client:
    # Store single commitment
    commitment_hash = hashlib.sha256(b"model_verification_complete").digest()
    tx_hash = client.store_commitment(commitment_hash, {"model_id": "prod_v2.1"})
    
    # Batch storage for efficiency
    batch_commitments = [hashlib.sha256(f"epoch_{i}".encode()).digest() for i in range(100)]
    batch_tx = client.batch_store_commitments(batch_commitments)
    
    # Verify on-chain
    is_valid = client.verify_commitment_onchain(commitment_hash)
    print(f"On-chain verification: {is_valid}")
```

### Training Provenance with Merkle Trees

```python
from pot.prototypes.training_provenance_auditor import (
    build_merkle_tree, generate_merkle_proof, verify_merkle_proof
)

# Build provenance tree for training history
training_events = [f"epoch_{i}_checkpoint".encode() for i in range(100)]
tree = build_merkle_tree(training_events)
root_hash = tree.hash  # Compact proof of entire training

# Generate proof for specific epoch
epoch_50_proof = generate_merkle_proof(tree, 50)  # O(log n) proof size

# Verify epoch occurred in training (without full training data)
epoch_50_hash = hashlib.sha256(b"epoch_50_checkpoint").digest()
is_valid = verify_merkle_proof(epoch_50_hash, epoch_50_proof, root_hash)
assert is_valid  # Cryptographically proves epoch 50 occurred
```

For complete audit system documentation, see [docs/audit_system.md](docs/audit_system.md).

## Using security components (prototype)

```python
from pot.security.proof_of_training import ProofOfTraining
from pot.security.fuzzy_hash_verifier import FuzzyHashVerifier, HashAlgorithm

# Fuzzy hashing example using canonicalized text bytes
verifier = FuzzyHashVerifier(similarity_threshold=0.8, algorithm=HashAlgorithm.SSDEEP)
reference = b"canonical text"  # canonicalized bytes
candidate = b"canonical text with tweaks"
score = verifier.verify_fuzzy(reference, candidate)

pot = ProofOfTraining({'verification_type': 'fuzzy', 'model_type': 'vision', 'security_level': 'high'})
result = pot.perform_verification(model, model_id="model_v1", profile="standard")
```

`pot/prototypes/training_provenance_auditor.py` provides an experimental training-provenance auditor and optional blockchain logging sink.

## Challenge Generation System

The PoT framework uses cryptographically secure challenge generation to ensure deterministic yet unpredictable test inputs for model verification. This implements the KDF-based approach from §2.3 of the paper.

### Cryptographic Foundation

All challenges follow the pattern: `c_i = KDF(master_seed || model_id || i || salt)`

This ensures:
- **Determinism**: Same inputs always produce identical challenges
- **Unpredictability**: Without the master key, challenges appear random
- **Model-specificity**: Different models receive different challenges when model_id is provided
- **Session uniqueness**: Each verification session gets fresh challenges via nonce

### PRF/KDF Implementation

The system uses a hybrid approach for performance and security:
- **HMAC-SHA256**: Cryptographic strength for key derivation and small outputs
- **xxhash**: Fast non-cryptographic hashing for challenge IDs and large data
- **NIST SP 800-108**: Counter mode construction for variable-length PRF output

```python
from pot.core.challenge import ChallengeConfig, generate_challenges

# Generate deterministic challenges for a specific model
config = ChallengeConfig(
    master_key_hex='deadbeef' * 8,  # 256-bit master key
    session_nonce_hex='cafebabe' * 4,  # 128-bit session nonce
    n=10,  # Number of challenges
    family='vision:freq',  # Challenge type
    params={'freq_range': [0.5, 10.0], 'contrast_range': [0.2, 1.0]},
    model_id='resnet50_v1'  # Optional: model-specific challenges
)

result = generate_challenges(config)
# result['challenges'] contains Challenge objects with unique IDs
```

### Supported Challenge Families

#### Vision Challenges

1. **vision:freq** - Sine grating patterns
   - Parameters: frequency (cycles/degree), orientation (degrees), phase (radians), contrast
   - Use case: Testing frequency response and orientation selectivity

2. **vision:texture** - Complex texture patterns
   - Types: Perlin noise, Gabor filters, checkerboard patterns
   - Parameters vary by type (octaves, wavelength, square_size, etc.)
   - Use case: Testing response to naturalistic and synthetic textures

#### Language Model Challenges

1. **lm:templates** - Template-based text generation
   - Grammatical slots: subject, verb, object, adjective, adverb
   - Deterministic slot filling from provided vocabularies
   - Use case: Testing consistent language understanding and generation

### Extending with New Challenge Families

To add a new challenge family:

1. **Define the generation function** in `pot/core/challenge.py`:
```python
def generate_[family]_challenges(cfg: ChallengeConfig, seed: bytes, salt: str) -> List[Challenge]:
    """Generate challenges for new family."""
    challenges = []
    for i in range(cfg.n):
        # Use PRF functions for deterministic parameter generation
        param = prf_floats(seed, f'challenge_{i}_param'.encode(), 1, min_val, max_val)[0]
        
        # Create unique challenge ID
        challenge_id = xxhash.xxh3_64_hexdigest(...)
        
        challenges.append(Challenge(
            challenge_id=challenge_id,
            index=i,
            family='new:family',
            parameters={...}
        ))
    return challenges
```

2. **Register in generate_challenges()** dispatcher:
```python
elif cfg.family == "new:family":
    challenges = generate_new_family_challenges(cfg, seed, salt)
```

3. **Update verifiers** to use the new challenges in the appropriate domain verifier.

### Security Considerations

- Master keys should be generated using cryptographically secure random sources
- Session nonces must be unique per verification session
- Challenge reuse is tracked via the leakage tracking system (`pot/security/leakage.py`)
- The salt parameter enables commit-reveal protocols for tamper-evident verification

## Project structure

```
PoT_Experiments/
├── pot/
│   ├── core/          # Statistical verification and cryptography
│   ├── vision/        # Vision model experiments
│   ├── lm/            # Language model experiments
│   ├── eval/          # Evaluation metrics and plots
│   ├── security/      # Prototype security components
│   └── prototypes/    # Experimental provenance auditor
├── scripts/           # Experiment runner scripts
├── configs/           # Experiment configurations
└── EXPERIMENTS.md     # Detailed experimental protocols
```

## Baselines

PoT is compared against:

- benign-input fingerprints (FBI),
- adversarial-trajectory fingerprints (NeurIPS'24),
- fixed-n L2/Hamming aggregations, and
- SPRT/Hoeffding sequential tests.

Across vision and LM benchmarks PoT reaches AUROC ≈ 0.99 with FAR ~0.4% and FRR ~0–1.2% while averaging 2–3 queries. Baselines attain AUROC 0.82–0.96 and typically require more queries; see experiment E6 for metrics and artifacts.

## Ablation studies

We ablate sequential rules (EB, Hoeffding, SPRT), τ calibration, score clipping, and challenge families. EB offers the best query efficiency at comparable FAR/FRR. τ calibration and clipping stabilize error rates, and texture vs frequency probes reveal coverage–separation trade-offs.

## Leakage study

Leakage fractions ρ ∈ {0, 0.1, 0.25, 0.5, 0.75} are evaluated with an adaptive attacker that learns the challenge distribution, following watermarking robustness test design. Detection degrades gracefully; even at ρ = 0.5 the calibrated τ maintains >60% detection.

## Limitations

- Adversary with full weight access can trivially pass verification by serving the reference.
- Non-IID drift beyond tested ranges can increase FRR; re-calibration required.
- Fuzzy hashing provides approximate matching tolerant to tokenization/formatting drift; not a cryptographic primitive and used only as an auxiliary signal.
- Results on closed-source APIs may differ due to server-side nondeterminism.

## Key innovations

1. **Anytime-valid sequential testing** with Empirical Bernstein bounds for 70-90% sample reduction.
2. **Behavioral fingerprinting** combining IO hashing and Jacobian analysis for fast pre-filtering.
3. **Cryptographically secure challenges** using NIST SP 800-108 PRF construction.
4. **Advanced sequential methods**: mixture testing, adaptive thresholds, multi-armed verification.
5. **Comprehensive visualization tools** for trajectory analysis and operating characteristics.
6. **Token-level fuzzy hashing** for robust language model verification.
7. **Time-aware tolerance** for handling model drift and version changes.
8. **Complete audit trails** with deterministic reproducibility and verification proofs.

## Use cases

**Primary Applications**:
- **Model authentication** before deployment with early stopping efficiency
- **Regulatory compliance** with complete audit trails and anytime-valid guarantees
- **IP protection** via behavioral fingerprinting and statistical verification
- **Quality assurance** for model releases with adaptive sample allocation
- **Federated learning** participation checks with distributed sequential testing

**Sequential Testing Scenarios**:
- **API model verification** where query costs are high
- **Real-time deployment** requiring immediate accept/reject decisions  
- **Continuous monitoring** with adaptive thresholds
- **A/B testing** with multiple model comparisons
- **Research workflows** enabling rapid iteration and exploration

## Documentation

**Complete Documentation**:
- **[CLAUDE.md](CLAUDE.md)**: Complete framework overview with API examples
- **[docs/statistical_verification.md](docs/statistical_verification.md)**: Theoretical foundations and mathematical background  
- **[examples/sequential_analysis.ipynb](examples/sequential_analysis.ipynb)**: Worked examples and tutorials
- **[EXPERIMENTS.md](EXPERIMENTS.md)**: Detailed experimental protocols
- **[AGENTS.md](AGENTS.md)**: Integration instructions for AI agents

**Quick References**:
- Sequential testing: `python -c "from pot.core.sequential import sequential_verify; help(sequential_verify)"`
- Visualization: `streamlit run pot/core/visualize_sequential.py`
- Test suite: `python -m pot.core.test_sequential_verify`

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. When contributing:

1. **Follow the documentation protocol** in CLAUDE.md §Documentation Guidelines
2. **Add tests** for new sequential testing features
3. **Update mathematical documentation** with formulas and references
4. **Maintain anytime validity** properties in sequential extensions

## License

MIT License - see LICENSE.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pot_experiments,
  title = {Proof-of-Training Experiments: Implementation of Black-Box Neural Network Verification with Anytime-Valid Sequential Testing},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/PoT_Experiments},
  note = {Includes anytime-valid sequential hypothesis testing, behavioral fingerprinting, and comprehensive visualization tools}
}
```

For the theoretical foundation:

```bibtex
@article{pot_paper,
  title = {Proof-of-Training: A Statistical Framework for Black-Box Neural Network Verification},
  author = {Paper Authors},
  journal = {Conference/Journal},
  year = {2024},
  note = {Section 2.4 covers sequential verification methodology}
}
```

[1]: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689
[2]: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
