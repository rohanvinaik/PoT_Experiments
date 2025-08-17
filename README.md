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

### Visualization Tools

The framework includes comprehensive visualization tools for sequential verification analysis:

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

### Minimal vision demo (ResNet18 vs seed-variant)

```bash
python scripts/run_generate_reference.py --config configs/vision_cifar10.yaml
python scripts/run_verify.py --config configs/vision_cifar10.yaml --challenge_family vision:texture --n 256 --seq eb --store-trace
```

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

1. Empirical Bernstein bounds for tighter confidence intervals.
2. Sequential probability ratio tests for early stopping.
3. Token-level fuzzy hashing for tokenization drift.
4. Cryptographically derived challenges with epoch-based key rotation.
5. Coverage–separation trade-off analysis for challenge design.
6. Wrapper attack detection via timing and consistency checks.
7. Time-aware tolerance for version drift.

## Use cases

- Model authentication before deployment
- Regulatory compliance and audit trails
- IP protection via statistical verification
- Quality assurance for model releases
- Federated learning participation checks

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

MIT License - see LICENSE.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{pot_experiments,
  title = {Proof-of-Training Experiments: Implementation of Black-Box Neural Network Verification},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/PoT_Experiments}
}
```

[1]: https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689
[2]: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
