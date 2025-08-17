# PoT Audit System Documentation

The Proof-of-Training (PoT) Audit System provides comprehensive cryptographic verification and tamper-evident audit trails for neural network model verification. This document covers the complete audit infrastructure including commit-reveal protocols, Merkle tree provenance, blockchain integration, and query/analysis tools.

## Table of Contents

1. [Overview](#overview)
2. [Commit-Reveal Protocol](#commit-reveal-protocol)
3. [Expected Ranges Validation](#expected-ranges-validation)
4. [Merkle Tree Provenance System](#merkle-tree-provenance-system)
5. [Blockchain Integration](#blockchain-integration)
6. [Query and Analysis Tools](#query-and-analysis-tools)
7. [Configuration Examples](#configuration-examples)
8. [Security Considerations](#security-considerations)
9. [Performance Characteristics](#performance-characteristics)
10. [Integration Patterns](#integration-patterns)

## Overview

The PoT Audit System ensures verification integrity through multiple cryptographic mechanisms:

- **Commit-Reveal Protocol**: Prevents parameter tampering and ensures verification reproducibility
- **Expected Ranges Validation**: Behavioral validation against calibrated reference model ranges
- **Merkle Tree Provenance**: Cryptographic proof of training progression and model lineage
- **Blockchain Integration**: Immutable on-chain commitment storage with gas optimization
- **Audit Trail Query System**: Comprehensive analysis and anomaly detection for audit trails

### Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    PoT Audit System                        │
├─────────────────────────────────────────────────────────────┤
│  Commit-Reveal    │  Expected Ranges  │  Merkle Trees      │
│  Protocol         │  Validation       │  Provenance        │
│                   │                   │                    │
│  • SHA256 commits │  • Range calib.   │  • Training proof  │
│  • Salt security  │  • Tolerance      │  • Log(n) proofs   │
│  • Tamper detect  │  • Statistics     │  • Immutable       │
├─────────────────────────────────────────────────────────────┤
│  Blockchain       │  Audit Query      │  Crypto Utils      │
│  Integration      │  System           │  Foundation        │
│                   │                   │                    │
│  • Multi-chain    │  • Anomaly detect │  • Secure RNG      │
│  • Gas optimize   │  • Integrity      │  • Hash chains     │
│  • Merkle batch   │  • Reporting      │  • ZK proofs       │
└─────────────────────────────────────────────────────────────┘
```

## Commit-Reveal Protocol

The commit-reveal protocol ensures verification parameters cannot be modified after commitment, providing tamper-evident verification workflows.

### Protocol Flow

1. **Pre-Verification Commitment**
   ```python
   verification_params = {
       'model_id': 'production_model_v2.1',
       'challenges': challenge_list,
       'parameters': {'alpha': 0.01, 'tau': 0.05},
       'session_id': generate_session_id()
   }
   
   commitment = compute_commitment(verification_params)
   write_commitment_record(commitment, 'audit_logs/pre_verification.json')
   ```

2. **Verification Execution**
   ```python
   # Parameters are now committed and cannot be changed
   verification_result = perform_verification(model, verification_params)
   ```

3. **Post-Verification Reveal**
   ```python
   reveal_data = {
       'verification_result': verification_result,
       'actual_challenges_used': result.challenges_used,
       'completion_metadata': result.metadata
   }
   
   # Verify consistency with original commitment
   salt = bytes.fromhex(commitment.salt)
   is_valid = verify_reveal(commitment, reveal_data, salt)
   ```

### Commitment Record Structure

```json
{
  "commitment_hash": "a1b2c3...64-char-hex-hash",
  "salt": "d4e5f6...64-char-hex-salt", 
  "timestamp": "2025-08-17T10:30:45.123Z",
  "version": "1.0"
}
```

### Security Properties

- **Cryptographic Binding**: SHA256-based commitments with 256-bit security
- **Salt Protection**: 32-byte cryptographically secure salts prevent rainbow table attacks
- **Temporal Integrity**: Timestamp validation prevents replay attacks
- **Deterministic Serialization**: Canonical JSON ensures reproducible hashing

### Implementation Example

```python
from pot.audit.commit_reveal import compute_commitment, verify_reveal

# Create commitment with automatic salt generation
data = {'model_id': 'test_model', 'confidence_threshold': 0.95}
commitment = compute_commitment(data)

# Later verification
salt = bytes.fromhex(commitment.salt)
is_valid = verify_reveal(commitment, data, salt)
assert is_valid  # Should be True for unmodified data
```

## Expected Ranges Validation

Expected ranges validation provides behavioral validation by comparing model performance against calibrated reference ranges.

### Range Components

```python
@dataclass
class ExpectedRanges:
    accuracy_range: Tuple[float, float]          # Model accuracy bounds
    latency_range: Tuple[float, float]           # Response time bounds (ms)
    fingerprint_similarity: Tuple[float, float]  # Behavioral similarity bounds
    jacobian_norm_range: Tuple[float, float]     # Gradient norm bounds
    confidence_level: float = 0.95              # Statistical confidence
    tolerance_factor: float = 1.1               # Production robustness margin
```

### Range Calibration Process

1. **Reference Model Setup**
   ```python
   calibrator = RangeCalibrator(
       confidence_level=0.99,      # High confidence for production
       percentile_margin=0.02      # Tight ranges (2% margin)
   )
   ```

2. **Calibration Execution**
   ```python
   # Use production test suite for realistic calibration
   test_suite = load_production_test_data()
   
   calibrated_ranges = calibrator.calibrate(
       reference_model=reference_model,
       test_suite=test_suite,
       model_type=ModelType.VISION,
       num_runs=50  # More runs = better statistics
   )
   ```

3. **Range Validation**
   ```python
   pot_system.set_expected_ranges(model_id, calibrated_ranges)
   
   result = pot_system.perform_verification(model, model_id, 'comprehensive')
   
   if result.range_validation:
       if result.range_validation.passed:
           print("✓ Model within expected behavioral ranges")
       else:
           print("⚠ Range violations detected:")
           for violation in result.range_validation.violations:
               print(f"  - {violation}")
   ```

### Validation Report Structure

```python
@dataclass 
class ValidationReport:
    passed: bool                           # Overall validation result
    confidence: float                      # Validation confidence score
    violations: List[str]                  # Specific violations detected
    range_scores: Dict[str, float]         # Per-metric scoring
    statistical_significance: Optional[float]  # P-value if applicable
```

### Advanced Usage

```python
# Production deployment with strict ranges
production_ranges = ExpectedRanges(
    accuracy_range=(0.92, 0.97),          # Tight accuracy bounds
    latency_range=(5.0, 25.0),            # Fast response required
    fingerprint_similarity=(0.98, 0.999), # High behavioral consistency
    jacobian_norm_range=(0.8, 1.5),       # Stable gradients
    confidence_level=0.999,                # Very high confidence
    tolerance_factor=1.02                  # Minimal tolerance
)

# Continuous monitoring with lenient ranges
monitoring_ranges = ExpectedRanges(
    accuracy_range=(0.88, 0.99),          # Allow some drift
    latency_range=(4.0, 35.0),            # More latency tolerance  
    fingerprint_similarity=(0.95, 1.0),   # Some behavioral variation OK
    jacobian_norm_range=(0.7, 1.8),       # Gradient tolerance
    confidence_level=0.95,                 # Standard confidence
    tolerance_factor=1.1                   # 10% tolerance
)
```

## Merkle Tree Provenance System

Merkle trees provide cryptographic proof of training progression with logarithmic proof sizes and tamper-evident properties.

### Core Operations

1. **Tree Construction**
   ```python
   from pot.prototypes.training_provenance_auditor import build_merkle_tree
   
   training_events = [
       b"epoch_0_checkpoint",
       b"epoch_1_checkpoint", 
       b"epoch_2_checkpoint",
       b"model_completed"
   ]
   
   tree = build_merkle_tree(training_events)
   root_hash = tree.hash  # Compact representation of entire training history
   ```

2. **Proof Generation**
   ```python
   # Generate proof that epoch 1 occurred in training
   proof = generate_merkle_proof(tree, 1)  # O(log n) size proof
   
   # Proof contains path from leaf to root
   print(f"Proof size: {len(proof)} steps for {len(training_events)} events")
   ```

3. **Proof Verification**
   ```python
   # Verifier only needs root hash and proof (not full training data)
   epoch_1_hash = hashlib.sha256(b"epoch_1_checkpoint").digest()
   is_valid = verify_merkle_proof(epoch_1_hash, proof, root_hash)
   
   assert is_valid  # Cryptographically proves epoch 1 occurred
   ```

### Training Provenance Integration

```python
from pot.prototypes.training_provenance_auditor import TrainingProvenanceAuditor

# Initialize auditor for model training
auditor = TrainingProvenanceAuditor(
    model_id="production_resnet50_v2",
    blockchain_client=blockchain_client  # Optional: store on-chain
)

# Log training events with automatic Merkle tree construction
for epoch in range(100):
    auditor.log_training_event(
        epoch=epoch,
        metrics={'loss': compute_loss(), 'accuracy': compute_accuracy()},
        checkpoint_hash=save_checkpoint(),
        event_type=auditor.EventType.EPOCH_END
    )

# Generate training completion proof
training_proof = auditor.generate_training_proof(
    start_epoch=0,
    end_epoch=99,
    proof_type=auditor.ProofType.MERKLE
)

# Embed provenance in model for distribution
model_with_provenance = auditor.embed_provenance(model_state_dict)
```

### Scalability Properties

- **Construction**: O(n) time complexity for n training events
- **Proof Size**: O(log n) - scales to millions of events with small proofs
- **Verification**: O(log n) - fast verification regardless of training size
- **Storage**: O(1) root hash storage, O(log n) per individual proof

### Advanced Merkle Features

```python
# Large-scale training with millions of events
large_training_events = [f"event_{i}".encode() for i in range(1_000_000)]

# Efficient tree construction
tree = build_merkle_tree(large_training_events)

# Proof size remains logarithmic
proof_500k = generate_merkle_proof(tree, 500_000)
print(f"Proof for 1M tree: {len(proof_500k)} hashes")  # ~20 hashes

# Batch verification for audit
random_indices = random.sample(range(1_000_000), 100)
for idx in random_indices:
    proof = generate_merkle_proof(tree, idx)
    event_hash = hashlib.sha256(f"event_{idx}".encode()).digest()
    assert verify_merkle_proof(event_hash, proof, tree.hash)
```

## Blockchain Integration

The blockchain integration provides immutable on-chain storage for commitments with multi-chain support and gas optimization.

### Supported Chains

- **Ethereum Mainnet**: 3 confirmation blocks, optimized gas settings
- **Polygon**: 5 confirmation blocks, fast finality
- **Binance Smart Chain**: 3 confirmation blocks
- **Arbitrum One**: 1 confirmation block (instant finality)
- **Optimism**: 1 confirmation block
- **Local Development**: Ganache/Hardhat support

### Basic Operations

```python
from pot.prototypes.training_provenance_auditor import BlockchainClient, BlockchainConfig

# Initialize client for production chain
config = BlockchainConfig.polygon_mainnet("https://polygon-mainnet.g.alchemy.com/v2/KEY")
client = BlockchainClient(config)

# Store single commitment
with client as bc:
    commitment_hash = hashlib.sha256(b"training_complete").digest()
    metadata = {"model_id": "prod_model_v1", "training_epochs": 100}
    
    tx_hash = bc.store_commitment(commitment_hash, metadata)
    print(f"Stored on-chain: {tx_hash}")
    
    # Verify commitment exists on-chain
    is_valid = bc.verify_commitment_onchain(commitment_hash)
    assert is_valid
```

### Gas-Optimized Batch Operations

```python
# Batch storage for 60-80% gas savings
training_commitments = []
for epoch in range(100):
    epoch_data = {"epoch": epoch, "loss": 1.0/(epoch+1)}
    commitment = hashlib.sha256(str(epoch_data).encode()).digest()
    training_commitments.append(commitment)

# Store all 100 commitments in single transaction
batch_tx = client.batch_store_commitments(training_commitments)

# Retrieve batch with Merkle proofs
batch_record = client.get_batch_commitment(batch_tx)
print(f"Merkle root: {batch_record.merkle_root}")
print(f"Individual proofs: {len(batch_record.proofs)}")

# Verify individual commitment in batch
commitment_to_verify = training_commitments[50]
proof = batch_record.proofs[commitment_to_verify.hex()]
root_hash = bytes.fromhex(batch_record.merkle_root)

# Verify without needing full batch data
leaf_hash = hashlib.sha256(commitment_to_verify).digest()
is_valid = verify_merkle_proof(leaf_hash, proof, root_hash)
assert is_valid
```

### Multi-Chain Deployment

```python
# Deploy across multiple chains for redundancy
chains = {
    "ethereum": BlockchainConfig.ethereum_mainnet(ETH_RPC_URL),
    "polygon": BlockchainConfig.polygon_mainnet(POLYGON_RPC_URL),
    "arbitrum": BlockchainConfig.arbitrum_mainnet(ARB_RPC_URL)
}

commitment_hash = hashlib.sha256(b"critical_model_verification").digest()
tx_hashes = {}

for chain_name, config in chains.items():
    with BlockchainClient(config) as client:
        tx_hash = client.store_commitment(commitment_hash, {"chain": chain_name})
        tx_hashes[chain_name] = tx_hash
        print(f"{chain_name}: {tx_hash}")
```

### Gas Cost Analysis

```python
# Compare individual vs batch storage costs
individual_gas = sum(
    client.estimate_gas_cost("store_commitment")["gas_needed"]
    for _ in range(100)
)

batch_gas = client.estimate_gas_cost("batch_store_commitments")["gas_needed"]
savings = (individual_gas - batch_gas) / individual_gas * 100

print(f"Gas savings: {savings:.1f}% ({individual_gas} vs {batch_gas})")
```

## Query and Analysis Tools

The audit trail query system provides comprehensive analysis capabilities for security monitoring and compliance reporting.

### Basic Querying

```python
from pot.audit.query import AuditTrailQuery

# Load audit trail from file or directory
query = AuditTrailQuery("audit_logs/verification_trail.jsonl")

# Multi-dimensional querying
model_a_records = query.query_by_model("production_model_A")
recent_records = query.query_by_timerange(
    start_time=datetime.now() - timedelta(days=7),
    end_time=datetime.now()
)
high_conf_records = query.query_by_confidence_range(0.9, 1.0)
failed_verifications = query.query_by_verification_result("FAIL")
```

### Integrity Verification

```python
# Comprehensive integrity analysis
integrity_report = query.verify_integrity()

print(f"Integrity Status: {integrity_report.status.value}")
print(f"Integrity Score: {integrity_report.integrity_score:.3f}")
print(f"Valid Records: {integrity_report.valid_records}/{integrity_report.total_records}")
print(f"Hash Chain Valid: {integrity_report.hash_chain_valid}")

# Commitment verification results
cv = integrity_report.commitment_verification
print(f"Commitments - Verified: {cv['verified']}, Failed: {cv['failed']}")

if integrity_report.anomalies_detected:
    print("Integrity Anomalies:")
    for anomaly in integrity_report.anomalies_detected:
        print(f"  - {anomaly.description} (severity: {anomaly.severity:.2f})")
```

### Advanced Anomaly Detection

```python
# Statistical anomaly detection
anomalies = query.find_anomalies()

# Categorize by severity
high_severity = [a for a in anomalies if a.severity >= 0.7]
medium_severity = [a for a in anomalies if 0.3 <= a.severity < 0.7]

print(f"High severity anomalies: {len(high_severity)}")
print(f"Medium severity anomalies: {len(medium_severity)}")

# Analyze anomaly patterns
from collections import Counter
anomaly_types = Counter([a.anomaly_type.value for a in anomalies])
for atype, count in anomaly_types.most_common():
    print(f"{atype}: {count} occurrences")
```

### Report Generation

```python
# Generate reports in multiple formats
json_report = query.generate_audit_report("json")
markdown_report = query.generate_audit_report("markdown") 
html_report = query.generate_audit_report("html")

# Parse JSON report for programmatic analysis
import json
report_data = json.loads(json_report)

summary = report_data['summary_statistics']
print(f"Total records: {summary['total_records']}")
print(f"Unique models: {summary['unique_models']}")
print(f"Average confidence: {summary['average_confidence']}")
print(f"Pass rate: {summary['pass_rate']}%")
```

### Interactive Dashboard

```python
from pot.audit.query import AuditDashboard

# Create interactive dashboard
dashboard = AuditDashboard(query)

# Generate Streamlit app code
dashboard_app_code = """
import streamlit as st
from pot.audit.query import AuditTrailQuery, AuditDashboard

# Load audit data
query = AuditTrailQuery("audit_logs/")
dashboard = AuditDashboard(query)

# Create dashboard interface
dashboard.create_streamlit_app()
"""

# Run with: streamlit run dashboard_app.py
```

### Real-World Usage Scenarios

```python
# Scenario 1: Security incident investigation
def investigate_security_incident(query, incident_time, time_window_hours=6):
    start_time = incident_time - timedelta(hours=time_window_hours//2)
    end_time = incident_time + timedelta(hours=time_window_hours//2)
    
    incident_records = query.query_by_timerange(start_time, end_time)
    anomalies = query.find_anomalies()
    
    incident_anomalies = [
        a for a in anomalies 
        if start_time <= a.timestamp <= end_time
    ]
    
    return {
        'total_records': len(incident_records),
        'anomalies_detected': len(incident_anomalies),
        'affected_models': set(r['model_id'] for r in incident_records),
        'high_severity_count': len([a for a in incident_anomalies if a.severity >= 0.7])
    }

# Scenario 2: Compliance monitoring
def generate_compliance_report(query, period_days=30):
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=period_days)
    
    period_records = query.query_by_timerange(start_time, end_time)
    integrity_report = query.verify_integrity()
    
    return {
        'period': f"{period_days} days",
        'verification_events': len(period_records),
        'data_integrity_score': integrity_report.integrity_score,
        'models_monitored': len(query.model_index),
        'compliance_status': 'COMPLIANT' if integrity_report.integrity_score > 0.95 else 'NON_COMPLIANT'
    }
```

## Configuration Examples

### Basic Development Setup

```python
# Development configuration with minimal security
dev_config = {
    'verification_type': 'fuzzy',
    'model_type': 'vision',
    'security_level': 'low',           # 70% threshold
    'audit_log_path': 'dev_audit.json',
    'enable_blockchain': False,        # Skip blockchain for dev speed
    'expected_ranges': None            # Skip range validation in dev
}

pot_system = ProofOfTraining(dev_config)
```

### Production Configuration

```python
# Production configuration with full security
production_config = {
    'verification_type': 'fuzzy',
    'model_type': 'vision', 
    'security_level': 'high',          # 95% threshold
    'audit_log_path': 'production_audit.jsonl',
    'enable_blockchain': True,
    'blockchain_config': BlockchainConfig.polygon_mainnet(POLYGON_RPC_URL),
    'expected_ranges': ExpectedRanges(
        accuracy_range=(0.92, 0.97),
        latency_range=(5.0, 25.0),
        fingerprint_similarity=(0.98, 0.999),
        jacobian_norm_range=(0.8, 1.5),
        confidence_level=0.999,
        tolerance_factor=1.02
    )
}

pot_system = ProofOfTraining(production_config)
```

### Complete Session Configuration

```python
# Comprehensive session setup for critical verification
session_config = SessionConfig(
    model=production_model,
    model_id="critical_model_v3.2",
    master_seed=generate_secure_seed(64),
    
    # Challenge parameters
    num_challenges=50,                 # More challenges for higher confidence
    challenge_family="vision:texture", # Complex challenge type
    challenge_params={'texture_types': ['perlin', 'gabor', 'checkerboard']},
    
    # Statistical testing
    accuracy_threshold=0.02,           # Very strict threshold
    type1_error=0.001,                # 0.1% false positive rate
    type2_error=0.001,                # 0.1% false negative rate
    max_samples=2000,                 # Allow extensive testing
    
    # Component activation
    use_fingerprinting=True,
    use_sequential=True,
    use_range_validation=True,
    use_blockchain=True,
    
    # Security settings
    expected_ranges=strict_production_ranges,
    audit_log_path="critical_verifications.jsonl",
    blockchain_config=multi_chain_config
)

# Execute with full audit trail
verification_report = pot_system.run_verification(session_config)
```

## Security Considerations

### Cryptographic Security

1. **Commitment Security**
   - SHA256 provides 256-bit security level
   - 32-byte salts prevent rainbow table attacks
   - Constant-time comparison prevents timing attacks

2. **Merkle Tree Security**
   - Collision-resistant hash function (SHA256)
   - Avalanche effect: small changes cause large hash differences
   - Logarithmic proof size prevents denial-of-service via large proofs

3. **Blockchain Security**
   - Multi-chain redundancy prevents single point of failure
   - Gas optimization maintains economic feasibility
   - Smart contract immutability ensures tamper-resistance

### Operational Security

1. **Key Management**
   ```python
   # Generate cryptographically secure seeds
   master_seed = secrets.token_hex(32)  # 256-bit entropy
   
   # Derive session keys from master seed
   session_key = derive_key_from_password(master_seed, session_salt)
   ```

2. **Access Control**
   ```python
   # Implement role-based access to verification functions
   @require_permission("model_verification")
   def perform_critical_verification(model, model_id):
       return pot_system.perform_verification(model, model_id, 'comprehensive')
   ```

3. **Audit Trail Protection**
   ```python
   # Write-only audit logs with integrity verification
   audit_file_permissions = 0o640  # Read-write owner, read group, no others
   write_audit_record(record, audit_file, permissions=audit_file_permissions)
   ```

### Attack Mitigation

1. **Replay Attack Prevention**
   - Timestamp validation in commitments
   - Nonce-based challenge generation
   - Session-specific salts

2. **Parameter Tampering Prevention**
   - Cryptographic commit-reveal protocol
   - Deterministic serialization
   - Integrity verification

3. **Model Substitution Detection**
   - Behavioral fingerprinting
   - Expected ranges validation
   - Continuous monitoring

### Security Monitoring

```python
# Implement security monitoring for anomaly detection
def monitor_security_events(query_system):
    recent_anomalies = query_system.find_anomalies()
    
    security_events = [
        a for a in recent_anomalies 
        if a.severity >= 0.7 and a.anomaly_type in [
            AnomalyType.SECURITY_BREACH,
            AnomalyType.TAMPERING_DETECTED,
            AnomalyType.UNUSUAL_BEHAVIOR
        ]
    ]
    
    if security_events:
        alert_security_team(security_events)
        trigger_incident_response()
    
    return len(security_events)
```

## Performance Characteristics

### Scalability Metrics

| Operation | Time Complexity | Space Complexity | Practical Performance |
|-----------|----------------|------------------|---------------------|
| Commitment Generation | O(1) | O(1) | <10ms per commitment |
| Reveal Verification | O(1) | O(1) | <5ms per verification |
| Merkle Tree Construction | O(n) | O(n) | 1M events in ~2s |
| Merkle Proof Generation | O(log n) | O(log n) | <1ms for 1M tree |
| Merkle Proof Verification | O(log n) | O(1) | <0.5ms per proof |
| Blockchain Storage | O(1) | O(1) | 15s per transaction |
| Batch Blockchain Storage | O(n) | O(log n) | 60-80% gas savings |
| Audit Query (indexed) | O(1) | O(n) | <10ms for 100K records |
| Anomaly Detection | O(n) | O(1) | <5s for 100K records |

### Memory Usage

```python
# Memory-efficient large-scale processing
def process_large_audit_trail(audit_file_path):
    # Stream processing for large files
    query = AuditTrailQuery(audit_file_path)
    
    # Memory usage scales with number of unique models, not total records
    print(f"Records: {len(query.records)}")
    print(f"Memory footprint: ~{len(query.model_index) * 100}KB")
    
    # Efficient batch processing
    for batch in query.batch_process(batch_size=1000):
        process_batch(batch)  # Process in chunks
```

### Performance Optimization

```python
# Optimized configuration for high-throughput environments
optimized_config = {
    'verification_type': 'fuzzy',      # Faster than exact matching
    'security_level': 'medium',       # Balance security vs speed
    'use_sequential': True,           # Early stopping saves computation
    'fingerprint_config': FingerprintConfig(
        compute_jacobian=False,       # Skip expensive gradient computation
        memory_efficient=True,        # Reduce memory usage
        canonicalize_precision=4      # Lower precision for speed
    ),
    'blockchain_batch_size': 100,     # Batch operations for efficiency
    'audit_buffer_size': 1000        # Buffer writes for performance
}
```

## Integration Patterns

### Microservice Integration

```python
# RESTful API for distributed verification
from flask import Flask, request, jsonify

app = Flask(__name__)
pot_system = ProofOfTraining(production_config)

@app.route('/verify', methods=['POST'])
def verify_model():
    data = request.json
    model_id = data['model_id']
    verification_depth = data.get('depth', 'standard')
    
    # Load model from model registry
    model = load_model_from_registry(model_id)
    
    # Perform verification with full audit trail
    result = pot_system.perform_verification(model, model_id, verification_depth)
    
    return jsonify({
        'verified': result.verified,
        'confidence': result.confidence,
        'session_id': result.session_id,
        'audit_trail_id': result.audit_trail_id
    })

@app.route('/audit/query', methods=['GET'])
def query_audit_trail():
    query_params = request.args
    query_system = AuditTrailQuery(audit_log_path)
    
    if 'model_id' in query_params:
        records = query_system.query_by_model(query_params['model_id'])
    elif 'start_time' in query_params and 'end_time' in query_params:
        start = datetime.fromisoformat(query_params['start_time'])
        end = datetime.fromisoformat(query_params['end_time'])
        records = query_system.query_by_timerange(start, end)
    else:
        records = query_system.records[:100]  # Recent records
    
    return jsonify({'records': records, 'count': len(records)})
```

### CI/CD Pipeline Integration

```yaml
# GitHub Actions workflow for model verification
name: Model Verification Pipeline
on:
  push:
    paths: ['models/**']

jobs:
  verify-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Setup PoT Environment
        run: |
          pip install -r requirements.txt
          export POT_MASTER_SEED=${{ secrets.POT_MASTER_SEED }}
          export BLOCKCHAIN_RPC_URL=${{ secrets.BLOCKCHAIN_RPC_URL }}
      
      - name: Run Model Verification
        run: |
          python scripts/run_verify_enhanced.py \
            --config configs/production.yaml \
            --model-path models/latest.pt \
            --security-level high \
            --audit-output verification_audit.json \
            --blockchain-store true
      
      - name: Generate Audit Report
        run: |
          python -c "
          from pot.audit.query import AuditTrailQuery
          query = AuditTrailQuery('verification_audit.json')
          report = query.generate_audit_report('html')
          with open('audit_report.html', 'w') as f:
              f.write(report)
          "
      
      - name: Upload Audit Artifacts
        uses: actions/upload-artifact@v2
        with:
          name: verification-audit
          path: |
            verification_audit.json
            audit_report.html
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pot-verification-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: pot-verification
  template:
    metadata:
      labels:
        app: pot-verification
    spec:
      containers:
      - name: pot-service
        image: pot-verification:latest
        env:
        - name: POT_CONFIG_PATH
          value: "/config/production.yaml"
        - name: BLOCKCHAIN_RPC_URL
          valueFrom:
            secretKeyRef:
              name: blockchain-config
              key: rpc-url
        - name: POT_MASTER_SEED
          valueFrom:
            secretKeyRef:
              name: pot-secrets
              key: master-seed
        volumeMounts:
        - name: config-volume
          mountPath: /config
        - name: audit-storage
          mountPath: /audit
        ports:
        - containerPort: 8080
          name: http
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: config-volume
        configMap:
          name: pot-config
      - name: audit-storage
        persistentVolumeClaim:
          claimName: audit-storage-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: pot-verification-service
spec:
  selector:
    app: pot-verification
  ports:
  - port: 80
    targetPort: 8080
    name: http
  type: LoadBalancer
```

### Database Integration

```python
# SQLAlchemy models for audit data persistence
from sqlalchemy import create_engine, Column, String, DateTime, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class AuditRecord(Base):
    __tablename__ = 'audit_records'
    
    id = Column(String, primary_key=True)
    model_id = Column(String, index=True)
    verification_decision = Column(String, index=True)
    confidence = Column(Float)
    timestamp = Column(DateTime, index=True)
    commitment_hash = Column(String)
    salt = Column(String)
    metadata = Column(Text)  # JSON-encoded metadata
    integrity_verified = Column(Boolean, default=False)

# Database-backed audit trail query
class DatabaseAuditQuery:
    def __init__(self, database_url):
        self.engine = create_engine(database_url)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def query_by_model(self, model_id):
        return self.session.query(AuditRecord).filter(
            AuditRecord.model_id == model_id
        ).all()
    
    def query_by_timerange(self, start_time, end_time):
        return self.session.query(AuditRecord).filter(
            AuditRecord.timestamp.between(start_time, end_time)
        ).all()
    
    def store_audit_record(self, record_data):
        audit_record = AuditRecord(**record_data)
        self.session.add(audit_record)
        self.session.commit()
        return audit_record.id
```

---

## Conclusion

The PoT Audit System provides production-ready cryptographic verification infrastructure with comprehensive security guarantees. The modular design allows incremental adoption while maintaining backward compatibility and high performance at scale.

For implementation guidance, see the comprehensive test suites in `pot/audit/test_*.py` and integration examples in `pot/security/test_integrated.py`.

### Quick Start Checklist

- [ ] Choose appropriate security level for your use case
- [ ] Configure expected ranges for your model types
- [ ] Set up audit trail storage (file-based or database)
- [ ] Configure blockchain integration if required
- [ ] Implement monitoring and alerting for anomalies
- [ ] Test end-to-end verification workflow
- [ ] Deploy with appropriate security controls

### Support and Resources

- **API Documentation**: See function docstrings in source code
- **Test Examples**: Comprehensive test suites in `pot/audit/` and `pot/security/`
- **Performance Benchmarks**: Run `test_performance_characteristics()` functions
- **Security Analysis**: Review security considerations section above
- **Integration Patterns**: Use examples provided for your deployment environment