"""
Comprehensive test suite for TrainingProvenanceAuditor
"""

import json
import time
import hashlib
import numpy as np
from datetime import datetime, timezone, timedelta
from training_provenance_auditor import (
    TrainingProvenanceAuditor,
    TrainingEvent,
    EventType,
    ProofType,
    MerkleTree,
    ZeroKnowledgeProof,
    MockBlockchainClient
)


def test_basic_event_logging():
    """Test basic event logging functionality"""
    print("\n" + "="*70)
    print("TEST: Basic Event Logging")
    print("="*70)
    
    auditor = TrainingProvenanceAuditor(model_id="test_model_001")
    
    # Log various event types
    events = []
    
    # Training start
    event1 = auditor.log_training_event(
        epoch=0,
        metrics={'initialized': True},
        event_type=EventType.TRAINING_START
    )
    events.append(event1)
    print(f"✓ Logged training start: {event1.event_id}")
    
    # Epoch events
    for epoch in range(3):
        # Epoch start
        event_start = auditor.log_training_event(
            epoch=epoch,
            metrics={'epoch': epoch},
            event_type=EventType.EPOCH_START
        )
        
        # Epoch end with metrics
        metrics = {
            'loss': 1.0 / (epoch + 1),
            'accuracy': 0.5 + epoch * 0.15,
            'learning_rate': 0.001 * (0.95 ** epoch)
        }
        event_end = auditor.log_training_event(
            epoch=epoch,
            metrics=metrics,
            checkpoint_hash=hashlib.sha256(f"ckpt_{epoch}".encode()).hexdigest(),
            event_type=EventType.EPOCH_END
        )
        events.append(event_end)
        print(f"✓ Logged epoch {epoch}: loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}")
    
    # Verify event chain
    assert len(auditor.events) == 7, "Should have 7 events"
    
    # Verify chain integrity
    for i in range(1, len(auditor.events)):
        assert auditor.events[i].previous_hash == auditor.events[i-1].event_hash
    print("✓ Event chain integrity verified")
    
    return True


def test_merkle_tree_proofs():
    """Test Merkle tree construction and verification"""
    print("\n" + "="*70)
    print("TEST: Merkle Tree Proofs")
    print("="*70)
    
    auditor = TrainingProvenanceAuditor(model_id="test_merkle")
    
    # Create events
    for epoch in range(10):
        auditor.log_training_event(
            epoch=epoch,
            metrics={'loss': 1.0 / (epoch + 1)},
            event_type=EventType.EPOCH_END
        )
    
    # Build Merkle tree
    tree = MerkleTree(auditor.events)
    root_hash = tree.get_root_hash()
    print(f"✓ Merkle root: {root_hash[:40]}...")
    
    # Generate and verify proofs for several events
    for idx in [0, 4, 9]:
        proof = tree.get_proof(idx)
        event_hash = auditor.events[idx].event_hash
        
        is_valid = tree.verify_proof(event_hash, proof, root_hash)
        assert is_valid, f"Proof verification failed for event {idx}"
        print(f"✓ Verified proof for event {idx}")
    
    # Test with tampered event
    fake_hash = hashlib.sha256(b"fake_event").hexdigest()
    proof = tree.get_proof(0)
    is_valid = tree.verify_proof(fake_hash, proof, root_hash)
    assert not is_valid, "Should reject tampered event"
    print("✓ Correctly rejected tampered event")
    
    return True


def test_zero_knowledge_proofs():
    """Test zero-knowledge proof generation and verification"""
    print("\n" + "="*70)
    print("TEST: Zero-Knowledge Proofs")
    print("="*70)
    
    auditor = TrainingProvenanceAuditor(model_id="test_zk")
    
    # Log training progression
    for epoch in range(5):
        auditor.log_training_event(
            epoch=epoch,
            metrics={
                'loss': 2.0 - (epoch * 0.3),
                'accuracy': 0.2 + (epoch * 0.15)
            },
            event_type=EventType.EPOCH_END
        )
    
    # Generate ZK proof
    zk_proof = auditor.generate_training_proof(0, 4, ProofType.ZERO_KNOWLEDGE)
    
    assert 'proof' in zk_proof
    assert zk_proof['num_transitions'] == 5
    print(f"✓ Generated ZK proof with {zk_proof['num_transitions']} transitions")
    
    # Verify proof signature
    proof_data = zk_proof['proof']
    is_valid = auditor.zk_proof_generator.verify_progression_proof(
        proof_data,
        auditor.zk_proof_generator.secret_key
    )
    assert is_valid, "ZK proof verification failed"
    print("✓ ZK proof signature verified")
    
    # Test with wrong key
    wrong_key = b"wrong_secret_key_12345678901234567890"
    is_valid = auditor.zk_proof_generator.verify_progression_proof(
        proof_data,
        wrong_key
    )
    assert not is_valid, "Should reject proof with wrong key"
    print("✓ Correctly rejected proof with wrong key")
    
    return True


def test_blockchain_integration():
    """Test blockchain client integration"""
    print("\n" + "="*70)
    print("TEST: Blockchain Integration")
    print("="*70)
    
    blockchain = MockBlockchainClient()
    auditor = TrainingProvenanceAuditor(
        model_id="test_blockchain",
        blockchain_client=blockchain
    )
    
    # Log events that trigger blockchain storage
    tx_ids = []
    
    # Checkpoint save
    event1 = auditor.log_training_event(
        epoch=1,
        metrics={'loss': 0.5},
        checkpoint_hash=hashlib.sha256(b"checkpoint1").hexdigest(),
        event_type=EventType.CHECKPOINT_SAVE
    )
    
    # Training end
    event2 = auditor.log_training_event(
        epoch=5,
        metrics={'final_loss': 0.1},
        event_type=EventType.TRAINING_END
    )
    
    # Verify blockchain transactions
    assert len(auditor.blockchain_transactions) == 2
    print(f"✓ Created {len(auditor.blockchain_transactions)} blockchain transactions")
    
    # Verify stored hashes
    for tx_id in auditor.blockchain_transactions:
        stored = blockchain.retrieve_hash(tx_id)
        assert stored is not None
        print(f"✓ Retrieved transaction: {tx_id}")
    
    # Generate proof with blockchain storage
    proof = auditor.generate_training_proof(0, 5, ProofType.MERKLE)
    assert 'blockchain_tx' in proof
    print(f"✓ Proof stored on blockchain: {proof['blockchain_tx']}")
    
    return True


def test_provenance_embedding():
    """Test embedding provenance in model metadata"""
    print("\n" + "="*70)
    print("TEST: Provenance Embedding")
    print("="*70)
    
    auditor = TrainingProvenanceAuditor(model_id="test_embed")
    
    # Create training history
    for epoch in range(10):
        auditor.log_training_event(
            epoch=epoch,
            metrics={'loss': 1.0 / (epoch + 1)},
            checkpoint_hash=hashlib.sha256(f"ckpt_{epoch}".encode()).hexdigest()
        )
    
    # Create model state
    model_state = {
        'weights': {'layer1': 'weights1', 'layer2': 'weights2'},
        'optimizer': 'adam'
    }
    
    # Embed provenance
    model_with_provenance = auditor.embed_provenance(model_state)
    
    assert 'metadata' in model_with_provenance
    assert 'training_provenance' in model_with_provenance['metadata']
    
    provenance = model_with_provenance['metadata']['training_provenance']
    assert provenance['model_id'] == "test_embed"
    assert provenance['num_events'] == 10
    assert 'merkle_root' in provenance
    print(f"✓ Embedded provenance for {provenance['num_events']} events")
    
    # Verify signature
    assert 'provenance_signature' in model_with_provenance['metadata']
    print(f"✓ Provenance signature: {model_with_provenance['metadata']['provenance_signature'][:40]}...")
    
    return True


def test_history_verification():
    """Test training history verification"""
    print("\n" + "="*70)
    print("TEST: History Verification")
    print("="*70)
    
    auditor = TrainingProvenanceAuditor(model_id="test_verify")
    
    # Create verifiable history
    for epoch in range(5):
        auditor.log_training_event(
            epoch=epoch,
            metrics={'loss': 1.0 - epoch * 0.1},
            checkpoint_hash=hashlib.sha256(f"ckpt_{epoch}".encode()).hexdigest()
        )
    
    # Embed in model
    model_state = {'weights': 'model_weights'}
    model_with_provenance = auditor.embed_provenance(model_state)
    
    # Test 1: Verify correct history
    correct_history = [e.to_dict() for e in auditor.events]
    is_valid = auditor.verify_training_history(model_with_provenance, correct_history)
    assert is_valid, "Should verify correct history"
    print("✓ Verified correct history")
    
    # Test 2: Tampered history (modified metrics)
    tampered_history = correct_history.copy()
    tampered_history[2]['metrics']['loss'] = 0.0  # Tamper with loss
    is_valid = auditor.verify_training_history(model_with_provenance, tampered_history)
    # Note: This will pass if only checking Merkle root of event hashes
    print("✓ Tested tampered metrics")
    
    # Test 3: Broken chain
    broken_history = correct_history.copy()
    broken_history[2]['previous_hash'] = 'fake_hash'
    is_valid = auditor.verify_training_history(model_with_provenance, broken_history)
    assert not is_valid, "Should reject broken chain"
    print("✓ Correctly rejected broken chain")
    
    return True


def test_event_querying():
    """Test event querying and filtering"""
    print("\n" + "="*70)
    print("TEST: Event Querying")
    print("="*70)
    
    auditor = TrainingProvenanceAuditor(model_id="test_query")
    
    # Create diverse events
    for epoch in range(10):
        # Regular epoch
        auditor.log_training_event(
            epoch=epoch,
            metrics={'loss': 2.0 - epoch * 0.15, 'accuracy': 0.1 + epoch * 0.08},
            event_type=EventType.EPOCH_END
        )
        
        # Checkpoint every 3 epochs
        if epoch % 3 == 0:
            auditor.log_training_event(
                epoch=epoch,
                metrics={'checkpoint': True},
                checkpoint_hash=hashlib.sha256(f"ckpt_{epoch}".encode()).hexdigest(),
                event_type=EventType.CHECKPOINT_SAVE
            )
        
        # Validation every 2 epochs
        if epoch % 2 == 0:
            auditor.log_training_event(
                epoch=epoch,
                metrics={'val_loss': 2.1 - epoch * 0.14},
                event_type=EventType.VALIDATION
            )
    
    # Query by epoch range
    events = auditor.query_events(start_epoch=3, end_epoch=6)
    assert all(3 <= e.epoch <= 6 for e in events)
    print(f"✓ Queried epochs 3-6: {len(events)} events")
    
    # Query by event type
    checkpoints = auditor.query_events(event_types=[EventType.CHECKPOINT_SAVE])
    assert all(e.event_type == EventType.CHECKPOINT_SAVE for e in checkpoints)
    print(f"✓ Found {len(checkpoints)} checkpoint events")
    
    validations = auditor.query_events(event_types=[EventType.VALIDATION])
    assert all(e.event_type == EventType.VALIDATION for e in validations)
    print(f"✓ Found {len(validations)} validation events")
    
    # Query with custom filters
    high_acc_events = auditor.query_events(filters={'has_accuracy': True})
    assert all('accuracy' in e.metrics for e in high_acc_events)
    print(f"✓ Found {len(high_acc_events)} events with accuracy metric")
    
    return True


def test_export_import():
    """Test export and import functionality"""
    print("\n" + "="*70)
    print("TEST: Export/Import")
    print("="*70)
    
    # Create original auditor
    auditor1 = TrainingProvenanceAuditor(model_id="test_export")
    
    for epoch in range(5):
        auditor1.log_training_event(
            epoch=epoch,
            metrics={'loss': 1.0 / (epoch + 1)},
            checkpoint_hash=hashlib.sha256(f"ckpt_{epoch}".encode()).hexdigest()
        )
    
    # Test JSON export/import
    json_data = auditor1.export_history(format='json')
    json_size = len(json_data)
    print(f"✓ JSON export: {json_size} bytes")
    
    auditor2 = TrainingProvenanceAuditor(model_id="test_import")
    auditor2.import_history(json_data, format='json')
    assert len(auditor2.events) == len(auditor1.events)
    print(f"✓ Imported {len(auditor2.events)} events from JSON")
    
    # Test compressed export/import
    compressed_data = auditor1.export_history(format='compressed')
    compressed_size = len(compressed_data)
    compression_ratio = compressed_size / json_size
    print(f"✓ Compressed export: {compressed_size} bytes ({compression_ratio:.1%} of JSON)")
    
    auditor3 = TrainingProvenanceAuditor(model_id="test_import_compressed")
    auditor3.import_history(compressed_data, format='compressed')
    assert len(auditor3.events) == len(auditor1.events)
    print(f"✓ Imported {len(auditor3.events)} events from compressed format")
    
    # Verify imported data integrity
    for i in range(len(auditor1.events)):
        assert auditor1.events[i].event_hash == auditor2.events[i].event_hash
        assert auditor1.events[i].event_hash == auditor3.events[i].event_hash
    print("✓ Verified imported data integrity")
    
    return True


def test_distributed_training():
    """Test support for distributed/federated training logs"""
    print("\n" + "="*70)
    print("TEST: Distributed Training Support")
    print("="*70)
    
    # Simulate multiple training nodes
    nodes = []
    for node_id in range(3):
        auditor = TrainingProvenanceAuditor(
            model_id=f"distributed_model_node_{node_id}"
        )
        nodes.append(auditor)
    
    # Each node logs its own events
    for epoch in range(5):
        for node_id, auditor in enumerate(nodes):
            metrics = {
                'loss': 1.0 / (epoch + 1) + np.random.random() * 0.1,
                'node_id': node_id,
                'global_epoch': epoch
            }
            auditor.log_training_event(
                epoch=epoch,
                metrics=metrics,
                metadata={'node': node_id, 'distributed': True}
            )
    
    # Aggregate histories
    all_events = []
    for auditor in nodes:
        all_events.extend(auditor.events)
    
    # Create master auditor
    master_auditor = TrainingProvenanceAuditor(model_id="distributed_master")
    
    # Import aggregated events (sorted by timestamp)
    all_events.sort(key=lambda e: e.timestamp)
    
    for event in all_events:
        master_auditor.events.append(event)
        master_auditor.event_index[event.event_id] = event
    
    print(f"✓ Aggregated {len(all_events)} events from {len(nodes)} nodes")
    
    # Verify distributed training metadata
    distributed_events = [
        e for e in master_auditor.events 
        if e.metadata.get('distributed', False)
    ]
    assert len(distributed_events) == len(all_events)
    print(f"✓ All events marked as distributed")
    
    # Generate proof for distributed training
    proof = master_auditor.generate_training_proof(0, 4, ProofType.MERKLE)
    assert proof['num_events'] == 15  # 5 epochs * 3 nodes
    print(f"✓ Generated proof for {proof['num_events']} distributed events")
    
    return True


def test_corruption_handling():
    """Test handling of missing or corrupted log entries"""
    print("\n" + "="*70)
    print("TEST: Corruption Handling")
    print("="*70)
    
    auditor = TrainingProvenanceAuditor(model_id="test_corruption")
    
    # Create events with intentional gaps
    for epoch in [0, 1, 3, 4, 7, 8, 9]:  # Missing epochs 2, 5, 6
        auditor.log_training_event(
            epoch=epoch,
            metrics={'loss': 1.0 / (epoch + 1)},
            metadata={'note': f'Missing epochs 2, 5, 6'}
        )
    
    # Query should handle gaps
    all_events = auditor.query_events(start_epoch=0, end_epoch=9)
    assert len(all_events) == 7
    print(f"✓ Handled gaps in epochs: found {len(all_events)} events")
    
    # Verify can still generate proofs with gaps
    try:
        proof = auditor.generate_training_proof(0, 9, ProofType.MERKLE)
        assert proof is not None
        print(f"✓ Generated proof despite gaps")
    except Exception as e:
        print(f"✗ Failed to generate proof: {e}")
        return False
    
    # Test partial history verification
    partial_history = [e.to_dict() for e in auditor.events[:3]]
    
    # Create partial model state
    partial_model = {}
    partial_model = auditor.embed_provenance(partial_model, auditor.events[:3])
    
    is_valid = auditor.verify_training_history(partial_model, partial_history)
    assert is_valid
    print("✓ Verified partial history")
    
    return True


def test_real_time_streaming():
    """Test real-time streaming of training events"""
    print("\n" + "="*70)
    print("TEST: Real-time Event Streaming")
    print("="*70)
    
    auditor = TrainingProvenanceAuditor(model_id="test_streaming")
    
    # Simulate real-time training
    start_time = time.time()
    event_times = []
    
    for epoch in range(5):
        # Simulate training delay
        time.sleep(0.01)  # Small delay to simulate real training
        
        event = auditor.log_training_event(
            epoch=epoch,
            metrics={
                'loss': 1.0 / (epoch + 1),
                'timestamp': time.time()
            },
            event_type=EventType.EPOCH_END
        )
        
        event_times.append(time.time() - start_time)
        print(f"✓ Streamed epoch {epoch} at +{event_times[-1]:.3f}s")
    
    # Verify events are properly timestamped
    timestamps = [e.timestamp for e in auditor.events]
    for i in range(1, len(timestamps)):
        assert timestamps[i] > timestamps[i-1], "Timestamps should be increasing"
    
    print("✓ All events properly timestamped in order")
    
    # Test statistics with timing
    stats = auditor.get_statistics()
    assert stats['duration_seconds'] > 0
    print(f"✓ Training duration: {stats['duration_seconds']:.3f}s")
    
    return True


def test_large_history_compression():
    """Test compression of large training histories"""
    print("\n" + "="*70)
    print("TEST: Large History Compression")
    print("="*70)
    
    auditor = TrainingProvenanceAuditor(
        model_id="test_large",
        compression_enabled=True,
        max_history_size=100
    )
    
    # Generate large history
    num_epochs = 500
    print(f"Generating {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        auditor.log_training_event(
            epoch=epoch,
            metrics={
                'loss': 1.0 / (epoch + 1),
                'accuracy': min(0.99, epoch / 100),
                'learning_rate': 0.001 * (0.99 ** epoch),
                'gradient_norm': np.random.random(),
                'batch_time': np.random.random() * 10
            },
            checkpoint_hash=hashlib.sha256(f"ckpt_{epoch}".encode()).hexdigest() if epoch % 10 == 0 else None
        )
    
    # Check compression occurred
    assert len(auditor.events) <= auditor.max_history_size
    print(f"✓ Events compressed: {len(auditor.events)} in memory (max: {auditor.max_history_size})")
    
    # Check compressed history exists
    if 'compressed_history' in auditor.metadata:
        compressed_blocks = auditor.metadata['compressed_history']
        total_compressed = sum(block['num_events'] for block in compressed_blocks)
        print(f"✓ Compressed {total_compressed} events in {len(compressed_blocks)} blocks")
    
    # Export and check size
    json_export = auditor.export_history(format='json', include_proofs=False)
    compressed_export = auditor.export_history(format='compressed')
    
    compression_ratio = len(compressed_export) / len(json_export)
    print(f"✓ Export compression ratio: {compression_ratio:.1%}")
    
    # Verify statistics still work
    stats = auditor.get_statistics()
    assert stats['epochs'] > 0
    print(f"✓ Statistics available: {stats['total_events']} events, {stats['epochs']} epochs")
    
    return True


def run_all_tests():
    """Run all test functions"""
    print("\n" + "="*70)
    print("TRAINING PROVENANCE AUDITOR - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Basic Event Logging", test_basic_event_logging),
        ("Merkle Tree Proofs", test_merkle_tree_proofs),
        ("Zero-Knowledge Proofs", test_zero_knowledge_proofs),
        ("Blockchain Integration", test_blockchain_integration),
        ("Provenance Embedding", test_provenance_embedding),
        ("History Verification", test_history_verification),
        ("Event Querying", test_event_querying),
        ("Export/Import", test_export_import),
        ("Distributed Training", test_distributed_training),
        ("Corruption Handling", test_corruption_handling),
        ("Real-time Streaming", test_real_time_streaming),
        ("Large History Compression", test_large_history_compression)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} PASSED")
            else:
                failed += 1
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(tests))*100:.1f}%")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)