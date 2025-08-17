# Provenance Integration with Training Pipeline

This document explains how to integrate blockchain-based provenance recording with your training pipeline using the PoT framework.

## Overview

The provenance integration system provides:

- **Training Checkpoint Recording**: Automatic logging of model states, metrics, and fingerprints during training
- **Validation Recording**: Verification results with validator signatures and confidence scores
- **Blockchain Integration**: Optional tamper-evident storage on Ethereum/Polygon networks
- **Proof Generation**: Complete proof-of-training artifacts with Merkle tree verification
- **CLI Tools**: Command-line interface for provenance operations

## Quick Start

### 1. Basic Setup

```python
from pot.core.provenance_integration import ProvenanceRecorder, ProvenanceConfig

# Initialize recorder
config = ProvenanceConfig(
    enabled=True,
    blockchain_enabled=False,  # Start with local storage
    fingerprint_checkpoints=True
)
recorder = ProvenanceRecorder(config)
```

### 2. Record Training Progress

```python
# During training loop
checkpoint_id = recorder.record_training_checkpoint(
    model_hash=get_model_hash(model),
    metrics={"loss": 0.1, "accuracy": 0.95},
    epoch=epoch,
    model_id="my_model",
    model_fn=lambda x: model(x)  # For fingerprinting
)
```

### 3. Record Validation Results

```python
# After validation
validation_id = recorder.record_validation(
    model_hash=get_model_hash(model),
    validator_id="validation_suite_v1",
    validation_result={"accuracy": 0.95, "confidence": 0.9},
    model_id="my_model"
)
```

### 4. Generate Proof of Training

```python
# Generate complete proof
proof = recorder.generate_proof_of_training("my_model")

# Verify proof integrity
is_valid = recorder.verify_training_provenance(proof)
```

## Configuration Options

### ProvenanceConfig

```python
@dataclass
class ProvenanceConfig:
    enabled: bool = False                    # Enable/disable provenance recording
    blockchain_enabled: bool = False         # Enable blockchain storage
    local_storage_path: str = "./provenance_records.json"
    batch_size: int = 10                    # Batch size for Merkle trees
    auto_verify: bool = True                # Auto-verify proofs
    fingerprint_checkpoints: bool = True    # Generate model fingerprints
    record_challenges: bool = True          # Record challenge hashes
    client_config: Optional[Dict] = None    # Blockchain client config
```

### Environment Variables

For blockchain integration:

```bash
# Blockchain Configuration
export RPC_URL="https://polygon-rpc.com"
export PRIVATE_KEY="0x..."  # Keep secure!
export CONTRACT_ADDRESS="0x..."
export BLOCKCHAIN_CLIENT_TYPE="auto"  # "auto", "web3", "local"

# Optional
export GAS_PRICE_GWEI="30"
export CONFIRMATION_BLOCKS="2"
export FORCE_LOCAL_BLOCKCHAIN="false"
```

## Integration Patterns

### 1. Decorator Pattern

```python
from pot.core.provenance_integration import integrate_with_training_loop

@integrate_with_training_loop(recorder, model_fn, "my_model")
def train_epoch(model, data_loader, optimizer):
    # Your training code
    return {
        "epoch": epoch,
        "model_hash": get_model_hash(model),
        "metrics": {"loss": loss, "accuracy": acc}
    }
```

### 2. Manual Integration

```python
def training_loop():
    for epoch in range(num_epochs):
        # Training step
        model, metrics = train_epoch(model, data_loader)
        
        # Record checkpoint
        if recorder.config.enabled:
            recorder.record_training_checkpoint(
                model_hash=get_model_hash(model),
                metrics=metrics,
                epoch=epoch,
                model_id=model_id
            )
        
        # Validation step
        if epoch % validation_frequency == 0:
            val_results = validate_model(model, val_loader)
            
            # Record validation
            if recorder.config.enabled:
                recorder.record_validation(
                    model_hash=get_model_hash(model),
                    validator_id="official_validator",
                    validation_result=val_results,
                    model_id=model_id
                )
```

### 3. Context Manager Pattern

```python
class ProvenanceContext:
    def __init__(self, recorder, model_id):
        self.recorder = recorder
        self.model_id = model_id
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not exc_type:  # No exception
            # Generate final proof
            self.proof = self.recorder.generate_proof_of_training(self.model_id)

# Usage
with ProvenanceContext(recorder, "my_model") as ctx:
    # Training code
    pass
# Proof automatically generated
```

## CLI Usage

### Initialize Provenance

```bash
# Local storage only
python scripts/provenance_cli.py init

# With blockchain integration
python scripts/provenance_cli.py init --blockchain --fingerprint
```

### Record Training Data

```bash
# Record checkpoint
python scripts/provenance_cli.py checkpoint \
    --model-hash 0x1234567890abcdef... \
    --epoch 10 \
    --metrics '{"loss": 0.1, "accuracy": 0.95}' \
    --model-id my_model

# Record validation
python scripts/provenance_cli.py validate \
    --model-hash 0x1234567890abcdef... \
    --validator official_suite \
    --result '{"accuracy": 0.95, "confidence": 0.9}' \
    --model-id my_model
```

### Generate and Verify Proofs

```bash
# Generate proof
python scripts/provenance_cli.py proof \
    --model-id my_model \
    --output my_model_proof.json

# Verify proof
python scripts/provenance_cli.py verify \
    --proof my_model_proof.json

# View training history
python scripts/provenance_cli.py history --model-id my_model
```

### Test Blockchain Connection

```bash
# Test local client
python scripts/provenance_cli.py test

# Test blockchain client
python scripts/provenance_cli.py test --blockchain
```

## Data Structures

### Training Checkpoint

```python
@dataclass
class TrainingCheckpoint:
    model_hash: str              # SHA256 hash of model parameters
    epoch: int                   # Training epoch number
    metrics: Dict[str, float]    # Training metrics
    timestamp: str               # ISO timestamp
    model_id: str                # Model identifier
    checkpoint_id: str           # Unique checkpoint ID
    fingerprint_hash: Optional[str]  # Model fingerprint
    validation_hash: Optional[str]   # Associated validation
```

### Validation Record

```python
@dataclass
class ValidationRecord:
    model_hash: str              # SHA256 hash of validated model
    validator_id: str            # Validator identifier
    validation_result: Dict[str, Any]  # Validation results
    timestamp: str               # ISO timestamp
    model_id: str                # Model identifier
    validation_id: str           # Unique validation ID
    confidence_score: Optional[float]  # Confidence level
    challenge_hash: Optional[str]      # Challenge configuration hash
```

### Provenance Proof

```python
@dataclass
class ProvenanceProof:
    model_id: str                         # Model identifier
    final_model_hash: str                 # Final model hash
    training_chain: List[TrainingCheckpoint]  # Training history
    validation_chain: List[ValidationRecord]  # Validation history
    merkle_root: str                      # Merkle tree root hash
    blockchain_transactions: List[str]     # Blockchain transaction IDs
    proof_timestamp: str                  # Proof generation time
    verification_metadata: Dict[str, Any] # Verification metadata
    signature_hash: str                   # Proof signature
```

## Merkle Tree Verification

The system uses Merkle trees for efficient batch verification:

```python
# Build tree from checkpoint and validation IDs
hashes = [cp.checkpoint_id for cp in checkpoints] + [vr.validation_id for vr in validations]
merkle_tree = MerkleTree(hashes)

# Get proof for specific record
proof_path = merkle_tree.get_proof(index)

# Verify record
is_valid = merkle_tree.verify_proof(record_hash, index, proof_path)
```

## Blockchain Integration

### Supported Networks

- **Ethereum Mainnet**: Production deployments
- **Polygon**: Lower cost alternative
- **Testnets**: Goerli, Mumbai for testing

### Smart Contract Integration

The system integrates with deployed PoT contracts:

```solidity
// Store hash with metadata
function storeHash(string hash, string metadata) returns (uint256 transactionId)

// Retrieve stored data
function getHash(uint256 transactionId) returns (string, string, uint256, uint256, address)

// Verify hash matches stored value
function verifyHash(string hash, uint256 transactionId) returns (bool)
```

### Cost Optimization

- **Batch Operations**: Use Merkle trees to reduce transaction costs
- **Local Fallback**: Automatic fallback to local storage if blockchain unavailable
- **Selective Recording**: Configure which events to record on-chain

## Best Practices

### 1. Model Hashing

```python
def get_model_hash(model):
    """Generate deterministic hash of model parameters."""
    param_bytes = b""
    for param in sorted(model.named_parameters()):
        param_bytes += param[1].data.cpu().numpy().tobytes()
    return hashlib.sha256(param_bytes).hexdigest()
```

### 2. Secure Configuration

```python
# Use environment variables for sensitive data
config = ProvenanceConfig(
    blockchain_enabled=os.getenv("ENABLE_BLOCKCHAIN", "false").lower() == "true",
    client_config={
        "rpc_url": os.getenv("RPC_URL"),
        "private_key": os.getenv("PRIVATE_KEY"),  # Keep secure!
        "contract_address": os.getenv("CONTRACT_ADDRESS")
    }
)
```

### 3. Error Handling

```python
try:
    checkpoint_id = recorder.record_training_checkpoint(...)
    logger.info(f"Checkpoint recorded: {checkpoint_id}")
except Exception as e:
    logger.error(f"Failed to record checkpoint: {e}")
    # Continue training even if provenance fails
```

### 4. Performance Considerations

- Use `io_only=True` for faster fingerprinting
- Record checkpoints at reasonable intervals (not every batch)
- Enable blockchain recording selectively for important milestones

## Troubleshooting

### Common Issues

1. **Blockchain Connection Failures**
   ```bash
   # Test connection
   python scripts/provenance_cli.py test --blockchain
   
   # Check environment variables
   echo $RPC_URL $CONTRACT_ADDRESS
   ```

2. **Fingerprinting Errors**
   ```python
   # Disable fingerprinting if causing issues
   config.fingerprint_checkpoints = False
   ```

3. **Storage Permissions**
   ```bash
   # Check write permissions for storage path
   touch ./provenance_records.json
   ```

4. **Import Errors**
   ```bash
   # Ensure PoT dependencies are installed
   pip install -r requirements.txt
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed logging
recorder = ProvenanceRecorder(config)
```

## Examples

Complete examples are available in:

- `examples/training_with_provenance.py`: Full training loop integration
- `scripts/provenance_cli.py`: Command-line tools
- `pot/security/test_blockchain_client.py`: Test examples

## Integration Checklist

- [ ] Configure environment variables for blockchain (if needed)
- [ ] Initialize ProvenanceRecorder with appropriate configuration
- [ ] Add checkpoint recording to training loop
- [ ] Add validation recording after model evaluation
- [ ] Implement model hashing function
- [ ] Test provenance recording with sample data
- [ ] Generate and verify proof-of-training
- [ ] Set up CLI tools for operational use
- [ ] Configure error handling and logging
- [ ] Document model-specific integration details