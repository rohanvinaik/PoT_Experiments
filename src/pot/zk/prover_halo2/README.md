# PoT ZK Prover (Halo2)

A zero-knowledge proof system for verifying SGD training steps using Halo2, designed to integrate with the Proof-of-Training (PoT) framework.

## Overview

This crate implements a ZK circuit that proves correct execution of SGD training steps without revealing the private model weights, gradients, or training data. The circuit verifies:

1. **Merkle Inclusion**: Weights are properly committed under W_t_root
2. **Batch Inclusion**: Training batch is committed under batch_root  
3. **SGD Update**: Correct computation of w_new = w_old - lr * grad
4. **Result Commitment**: Updated weights hash to W_t1_root

## Architecture

### Core Components

- **`src/poseidon.rs`**: Poseidon hash implementation compatible with BN254 field
- **`src/fixed_point.rs`**: Fixed-point arithmetic with 2^16 scale for decimal operations
- **`src/circuit.rs`**: Main SGDCircuit implementation with constraint system
- **`src/lib.rs`**: High-level API for proving and verification

### Binary Targets

- **`prove_sgd`**: CLI tool for generating ZK proofs
- **`verify_sgd`**: CLI tool for verifying ZK proofs

## Installation

```bash
# Build the library
cargo build --release

# Build with Python bindings (optional)
cargo build --release --features python-bindings

# Run tests
cargo test
```

## Usage

### Command Line Interface

#### Generate a Proof

```bash
cargo run --bin prove_sgd -- \
  --public-inputs public_inputs.json \
  --witness witness_data.json \
  --output proof.json \
  --params-k 17 \
  --max-weights 1024 \
  --verbose
```

#### Verify a Proof

```bash
cargo run --bin verify_sgd -- \
  --proof proof.json \
  --params-k 17 \
  --verbose
```

### Input Format

#### Public Inputs (`public_inputs.json`)
```json
{
  "w_t_root": "0x1234567890abcdef...",
  "batch_root": "0xfedcba0987654321...",
  "hparams_hash": "0x1111111111111111...",
  "w_t1_root": "0x2222222222222222...",
  "step_nonce": 123,
  "step_number": 456,
  "epoch": 1
}
```

#### Witness Data (`witness_data.json`)
```json
{
  "weights_before": [1.0, 2.0, 3.0],
  "weights_after": [0.9, 1.9, 2.9],
  "batch_inputs": [0.5, 0.6],
  "batch_targets": [1.0, 0.0],
  "gradients": [0.1, 0.1, 0.1],
  "learning_rate": 0.01,
  "loss_value": 0.5
}
```

### Rust API

```rust
use pot_zk_prover::{SGDProver, ProvingSystemParams};

// Create prover with default parameters
let prover = SGDProver::new()?;

// Generate proof from JSON
let proof_json = prover.prove_from_json(
    &public_inputs_json,
    &witness_json
)?;

// Verify proof
let is_valid = prover.verify_from_json(&proof_json)?;
println!("Proof valid: {}", is_valid);
```

### Python Integration (Optional)

When built with `python-bindings` feature:

```python
import pot_zk_prover

prover = pot_zk_prover.PySGDProver()
proof = prover.prove(public_inputs_json, witness_json)
is_valid = prover.verify(proof)
```

## Circuit Details

### Fixed-Point Arithmetic

The circuit uses fixed-point representation with scale 2^16 = 65536 to handle decimal numbers in the field:

- **Integer**: `value * 65536`
- **Float**: `(value * 65536.0).round()`
- **Recovery**: `fixed_value / 65536.0`

### Poseidon Hash

Compatible with BN254 field parameters, using the same configuration as `halo2_gadgets::poseidon`:

- **Field**: Pallas base field
- **Width**: 3 (rate 2, capacity 1)
- **Rounds**: Full rounds + partial rounds as per Poseidon specification

### Circuit Constraints

The circuit enforces:

1. **Merkle Proofs**: Verify inclusion of weights and batch data
2. **SGD Updates**: Constrain `w_new = w_old - lr * grad` for each weight
3. **Hash Consistency**: Ensure computed hashes match public commitments
4. **Range Checks**: Validate that values are within reasonable bounds

### Security Considerations

- **Soundness**: Invalid SGD computations cannot produce valid proofs
- **Zero-Knowledge**: Private weights and gradients remain hidden
- **Completeness**: Valid SGD steps always produce valid proofs
- **Replay Protection**: Step nonces prevent proof reuse

## Performance

### Circuit Size
- **Constraints**: ~50K (depending on parameters)
- **Public Inputs**: 3 field elements
- **Private Inputs**: Variable (based on model size)

### Timing (Rough Estimates)
- **Setup**: 10-30 seconds (one-time, cacheable)
- **Proving**: 5-15 seconds (depends on model size)
- **Verification**: 10-50 milliseconds

### Memory Usage
- **Setup**: ~1-4 GB RAM
- **Proving**: ~2-8 GB RAM
- **Verification**: ~100-500 MB RAM

## Configuration Parameters

### Circuit Parameters
```rust
pub struct SGDCircuitParams {
    pub max_weights: usize,      // Default: 1024
    pub max_merkle_depth: usize, // Default: 20
    pub max_batch_size: usize,   // Default: 256
}
```

### Proving Parameters
```rust
pub struct ProvingSystemParams {
    pub circuit_params: SGDCircuitParams,
    pub kzg_params_k: u32,      // Default: 17 (2^17 = 131K constraints)
}
```

## Integration with PoT Framework

This Halo2 implementation is designed to replace the mock Poseidon hasher in the Python PoT framework:

1. **Python Side**: `pot/zk/commitments.py` calls Rust functions
2. **Rust Side**: This crate provides the actual ZK proving functionality
3. **Compatibility**: Hash outputs match between Python and Rust implementations

### Integration Steps

1. Build this crate with Python bindings
2. Install the Python module: `pip install .`
3. Update `pot/zk/commitments.py` to use real Poseidon implementation
4. Replace mock proof generation with actual Halo2 proving

## Development

### Testing
```bash
# Run all tests
cargo test

# Run with verbose output
cargo test -- --nocapture

# Test specific module
cargo test poseidon

# Run integration tests
cargo test --test integration
```

### Benchmarking
```bash
# Run with benchmark timing
cargo run --bin prove_sgd -- ... --benchmark
cargo run --bin verify_sgd -- ... --benchmark
```

### Debugging
```bash
# Enable debug output
RUST_LOG=debug cargo run --bin prove_sgd -- ...

# Verbose circuit debugging
cargo test circuit --features debug-circuit
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce `kzg_params_k` or `max_weights`
2. **Slow Performance**: Use release build and consider smaller parameters
3. **Invalid Proofs**: Check that public inputs match witness data
4. **Setup Errors**: Ensure sufficient RAM and valid parameters

### Error Messages

- **"Circuit synthesis error"**: Invalid witness data or constraint violations
- **"Proof generation error"**: Setup parameters too small or memory issues
- **"Proof verification error"**: Invalid proof or mismatched parameters
- **"Invalid input"**: Malformed JSON or out-of-range values

## Contributing

1. Follow Rust conventions and run `cargo fmt`
2. Add tests for new functionality
3. Update documentation for API changes
4. Ensure compatibility with Python integration

## License

MIT License - see LICENSE file for details.