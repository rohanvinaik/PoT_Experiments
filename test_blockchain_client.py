#!/usr/bin/env python3
"""
Comprehensive test suite for blockchain client implementation.

Tests both MockBlockchainClient and BlockchainClient functionality including:
- Connection management and configuration
- Commitment storage and retrieval
- Batch operations with Merkle tree optimization
- Error handling and retry logic
- Gas estimation and optimization
- Multi-chain support
"""

import sys
import os
import time
import hashlib
import json
from typing import List, Dict, Any

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pot.prototypes.training_provenance_auditor import (
    BlockchainClient, MockBlockchainClient, BlockchainConfig, ChainType,
    CommitmentRecord, BatchCommitmentRecord, BlockchainError, 
    ConnectionError, TransactionError, ContractError,
    compute_merkle_root, build_merkle_tree, generate_merkle_proof, verify_merkle_proof
)


def test_blockchain_config():
    """Test BlockchainConfig creation and predefined configurations"""
    print("Testing BlockchainConfig...")
    
    # Test custom configuration
    config = BlockchainConfig(
        chain_type=ChainType.ETHEREUM,
        rpc_url="https://mainnet.infura.io/v3/test",
        chain_id=1,
        gas_limit=300000
    )
    
    assert config.chain_type == ChainType.ETHEREUM
    assert config.chain_id == 1
    assert config.gas_limit == 300000
    assert config.confirmation_blocks == 1  # Default value
    
    # Test predefined configurations
    eth_config = BlockchainConfig.ethereum_mainnet("https://mainnet.infura.io/v3/test")
    assert eth_config.chain_type == ChainType.ETHEREUM
    assert eth_config.chain_id == 1
    assert eth_config.confirmation_blocks == 3
    
    polygon_config = BlockchainConfig.polygon_mainnet("https://polygon-rpc.com")
    assert polygon_config.chain_type == ChainType.POLYGON
    assert polygon_config.chain_id == 137
    assert polygon_config.confirmation_blocks == 5
    
    local_config = BlockchainConfig.local_ganache()
    assert local_config.chain_type == ChainType.LOCAL
    assert local_config.chain_id == 1337
    
    print("✓ BlockchainConfig tests passed")


def test_mock_blockchain_client_basic():
    """Test basic MockBlockchainClient functionality"""
    print("Testing MockBlockchainClient basic operations...")
    
    # Initialize mock client
    client = MockBlockchainClient()
    
    # Test connection
    assert client.connect() == True
    assert client._connection_established == True
    
    # Test balance
    balance = client.get_balance()
    assert balance == 10.0
    
    # Test gas estimation
    gas_estimate = client.estimate_gas_cost("store_commitment")
    assert "gas_needed" in gas_estimate
    assert gas_estimate["gas_needed"] == 75000
    
    print("✓ MockBlockchainClient basic tests passed")


def test_mock_blockchain_commitment_storage():
    """Test commitment storage and retrieval with MockBlockchainClient"""
    print("Testing MockBlockchainClient commitment storage...")
    
    client = MockBlockchainClient()
    
    # Test single commitment storage
    commitment_hash = hashlib.sha256(b"test_commitment_data").digest()
    metadata = {"model_id": "test_model", "epoch": 5}
    
    tx_hash = client.store_commitment(commitment_hash, metadata)
    assert tx_hash.startswith("0x")
    assert len(tx_hash) == 66  # 0x + 64 hex chars
    
    # Test commitment retrieval
    record = client.retrieve_commitment(tx_hash)
    assert record is not None
    assert record.commitment_hash == commitment_hash.hex()
    assert record.metadata == metadata
    assert record.tx_hash == tx_hash
    assert record.gas_used == 75000
    
    # Test commitment verification
    assert client.verify_commitment_onchain(commitment_hash) == True
    
    # Test non-existent commitment
    fake_hash = hashlib.sha256(b"fake_commitment").digest()
    assert client.verify_commitment_onchain(fake_hash) == False
    
    print("✓ MockBlockchainClient commitment storage tests passed")


def test_mock_blockchain_batch_operations():
    """Test batch commitment operations with Merkle tree optimization"""
    print("Testing MockBlockchainClient batch operations...")
    
    client = MockBlockchainClient()
    
    # Generate multiple commitments
    commitments = []
    for i in range(10):
        commitment = hashlib.sha256(f"commitment_{i}".encode()).digest()
        commitments.append(commitment)
    
    # Test batch storage
    batch_tx_hash = client.batch_store_commitments(commitments)
    assert batch_tx_hash.startswith("0x")
    
    # Test batch retrieval
    batch_record = client.get_batch_commitment(batch_tx_hash)
    assert batch_record is not None
    assert len(batch_record.commitment_hashes) == 10
    assert batch_record.merkle_root is not None
    
    # Verify Merkle root matches our computation
    expected_root = compute_merkle_root(commitments)
    assert batch_record.merkle_root == expected_root.hex()
    
    # Test that Merkle proofs are generated and stored
    assert len(batch_record.proofs) == 10
    
    # Verify a proof
    first_commitment = commitments[0]
    proof = batch_record.proofs[first_commitment.hex()]
    root_hash = bytes.fromhex(batch_record.merkle_root)
    
    # Note: Merkle tree expects hashed leaf data for verification
    leaf_hash = hashlib.sha256(first_commitment).digest()
    assert verify_merkle_proof(leaf_hash, proof, root_hash)
    
    # Test empty batch handling
    try:
        client.batch_store_commitments([])
        assert False, "Should raise ValueError for empty batch"
    except ValueError as e:
        assert "empty" in str(e).lower()
    
    print("✓ MockBlockchainClient batch operations tests passed")


def test_mock_blockchain_legacy_compatibility():
    """Test legacy method compatibility"""
    print("Testing MockBlockchainClient legacy compatibility...")
    
    client = MockBlockchainClient()
    
    # Test legacy store_hash method
    hash_value = "deadbeef" * 8  # 64 char hex string
    metadata = {"test": "data"}
    
    tx_hash = client.store_hash(hash_value, metadata)
    assert tx_hash.startswith("0x")
    
    # Test legacy retrieve_hash method
    retrieved = client.retrieve_hash(tx_hash)
    assert retrieved is not None
    assert retrieved["hash"] == hash_value
    assert retrieved["metadata"] == metadata
    
    # Test legacy verify_hash method
    assert client.verify_hash(hash_value, tx_hash) == True
    assert client.verify_hash("different_hash", tx_hash) == False
    
    print("✓ MockBlockchainClient legacy compatibility tests passed")


def test_blockchain_client_configuration():
    """Test BlockchainClient configuration without actual connection"""
    print("Testing BlockchainClient configuration...")
    
    # Test initialization with different configurations
    config = BlockchainConfig.local_ganache()
    client = BlockchainClient(config)
    
    assert client.config.chain_type == ChainType.LOCAL
    assert client.config.chain_id == 1337
    assert client._connection_established == False
    
    # Test default ABI is loaded
    assert len(client._default_abi) == 4
    
    # Test ABI contains expected functions
    function_names = [item["name"] for item in client._default_abi if item["type"] == "function"]
    expected_functions = ["storeCommitment", "storeBatchCommitments", "getCommitment", "verifyCommitment"]
    
    for expected in expected_functions:
        assert expected in function_names
    
    print("✓ BlockchainClient configuration tests passed")


def test_gas_optimization():
    """Test gas optimization utilities"""
    print("Testing gas optimization...")
    
    from pot.prototypes.training_provenance_auditor import gas_price_oracle
    
    # Test gas price oracle with mock Web3 object
    class MockWeb3:
        def __init__(self):
            self.eth = MockEth()
            self.chain_id = 1
        
        def from_wei(self, wei_value, unit):
            if unit == 'gwei':
                return wei_value / 1000000000
            return wei_value / 1000000000000000000
    
    class MockEth:
        def __init__(self):
            self.gas_price = 20000000000  # 20 Gwei in Wei
    
    mock_web3 = MockWeb3()
    
    # Test different gas strategies
    for strategy in ['slow', 'standard', 'fast', 'fastest']:
        gas_info = gas_price_oracle(mock_web3, strategy)
        assert 'gas_price' in gas_info
        assert 'max_fee_per_gas' in gas_info
        assert 'max_priority_fee_per_gas' in gas_info
        assert gas_info['gas_price'] > 0
    
    # Test fallback on error
    class FailingWeb3:
        def __init__(self):
            self.eth = None  # Will cause error
    
    failing_web3 = FailingWeb3()
    fallback_gas = gas_price_oracle(failing_web3)
    assert fallback_gas['gas_price'] == 20  # Default fallback
    
    print("✓ Gas optimization tests passed")


def test_error_handling():
    """Test error handling and custom exceptions"""
    print("Testing error handling...")
    
    # Test custom exceptions
    try:
        raise BlockchainError("Test blockchain error")
    except BlockchainError as e:
        assert str(e) == "Test blockchain error"
    
    try:
        raise ConnectionError("Test connection error")
    except ConnectionError as e:
        assert str(e) == "Test connection error"
    
    try:
        raise TransactionError("Test transaction error")
    except TransactionError as e:
        assert str(e) == "Test transaction error"
    
    try:
        raise ContractError("Test contract error")
    except ContractError as e:
        assert str(e) == "Test contract error"
    
    # Test retry decorator behavior by mocking
    from pot.prototypes.training_provenance_auditor import retry_on_failure
    
    call_count = 0
    
    @retry_on_failure(max_attempts=3, delay=0.01)  # Fast retry for testing
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Temporary failure")
        return "success"
    
    result = failing_function()
    assert result == "success"
    assert call_count == 3  # Should have tried 3 times
    
    # Test function that always fails
    call_count_2 = 0
    
    @retry_on_failure(max_attempts=2, delay=0.01)
    def always_failing_function():
        nonlocal call_count_2
        call_count_2 += 1
        raise Exception("Always fails")
    
    try:
        always_failing_function()
        assert False, "Should have raised exception"
    except Exception as e:
        assert str(e) == "Always fails"
        assert call_count_2 == 2  # Should have tried 2 times
    
    print("✓ Error handling tests passed")


def test_context_manager():
    """Test context manager functionality"""
    print("Testing context manager...")
    
    # Test MockBlockchainClient context manager
    with MockBlockchainClient() as client:
        assert client._connection_established == True
        
        # Test operations within context
        commitment = hashlib.sha256(b"context_test").digest()
        tx_hash = client.store_commitment(commitment, {"test": "context"})
        assert tx_hash is not None
    
    # Client should be disconnected after context
    assert client._connection_established == False
    
    print("✓ Context manager tests passed")


def test_integration_scenario():
    """Test realistic integration scenario"""
    print("Testing integration scenario...")
    
    # Simulate training provenance scenario
    client = MockBlockchainClient()
    
    # Training events over multiple epochs
    training_commitments = []
    training_metadata = []
    
    for epoch in range(5):
        # Generate commitment for each epoch
        event_data = {
            "epoch": epoch,
            "loss": 1.0 / (epoch + 1),
            "accuracy": 0.5 + epoch * 0.1,
            "timestamp": time.time() + epoch
        }
        
        event_bytes = json.dumps(event_data, sort_keys=True).encode()
        commitment = hashlib.sha256(event_bytes).digest()
        
        training_commitments.append(commitment)
        training_metadata.append(event_data)
    
    # Store individual commitments
    individual_tx_hashes = []
    for i, (commitment, metadata) in enumerate(zip(training_commitments, training_metadata)):
        tx_hash = client.store_commitment(commitment, metadata)
        individual_tx_hashes.append(tx_hash)
        
        # Verify each commitment
        assert client.verify_commitment_onchain(commitment)
    
    # Store as batch for efficiency
    batch_tx_hash = client.batch_store_commitments(training_commitments)
    batch_record = client.get_batch_commitment(batch_tx_hash)
    
    # Verify batch integrity
    assert len(batch_record.commitment_hashes) == 5
    
    # Verify individual commitments can be proven from batch
    for i, commitment in enumerate(training_commitments):
        proof = batch_record.proofs[commitment.hex()]
        root_hash = bytes.fromhex(batch_record.merkle_root)
        
        # Verify proof (Merkle tree expects hashed leaf data for verification)
        leaf_hash = hashlib.sha256(commitment).digest()
        assert verify_merkle_proof(leaf_hash, proof, root_hash)
    
    # Calculate gas savings
    individual_gas = sum(
        client.estimate_gas_cost("store_commitment")["gas_needed"] 
        for _ in training_commitments
    )
    
    batch_gas = client.estimate_gas_cost("batch_store_commitments")["gas_needed"]
    
    print(f"    Individual storage gas: {individual_gas}")
    print(f"    Batch storage gas: {batch_gas}")
    print(f"    Gas savings: {individual_gas - batch_gas} ({100*(individual_gas-batch_gas)/individual_gas:.1f}%)")
    
    assert batch_gas < individual_gas, "Batch operations should be more gas efficient"
    
    print("✓ Integration scenario tests passed")


def test_performance_benchmarks():
    """Test performance characteristics"""
    print("Testing performance benchmarks...")
    
    client = MockBlockchainClient()
    
    # Test single commitment performance
    commitment = hashlib.sha256(b"performance_test").digest()
    metadata = {"benchmark": True}
    
    start_time = time.time()
    tx_hash = client.store_commitment(commitment, metadata)
    single_time = time.time() - start_time
    
    print(f"    Single commitment storage: {single_time*1000:.2f}ms")
    
    # Test batch commitment performance
    batch_commitments = [
        hashlib.sha256(f"batch_commitment_{i}".encode()).digest()
        for i in range(100)
    ]
    
    start_time = time.time()
    batch_tx_hash = client.batch_store_commitments(batch_commitments)
    batch_time = time.time() - start_time
    
    print(f"    Batch commitment storage (100 items): {batch_time*1000:.2f}ms")
    print(f"    Time per commitment in batch: {batch_time*1000/100:.2f}ms")
    
    # Test retrieval performance
    start_time = time.time()
    retrieved = client.retrieve_commitment(tx_hash)
    retrieval_time = time.time() - start_time
    
    print(f"    Commitment retrieval: {retrieval_time*1000:.2f}ms")
    
    # Performance assertions
    assert single_time < 0.1, "Single commitment storage should be fast"
    assert batch_time < 1.0, "Batch storage should complete within 1 second"
    assert retrieval_time < 0.01, "Retrieval should be very fast"
    
    print("✓ Performance benchmark tests passed")


def run_all_tests():
    """Run all blockchain client tests"""
    print("=" * 70)
    print("COMPREHENSIVE BLOCKCHAIN CLIENT TEST SUITE")
    print("=" * 70)
    
    test_functions = [
        test_blockchain_config,
        test_mock_blockchain_client_basic,
        test_mock_blockchain_commitment_storage,
        test_mock_blockchain_batch_operations,
        test_mock_blockchain_legacy_compatibility,
        test_blockchain_client_configuration,
        test_gas_optimization,
        test_error_handling,
        test_context_manager,
        test_integration_scenario,
        test_performance_benchmarks
    ]
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    print(f"Total tests: {passed + failed}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        print("\nBlockchain client implementation ready for production!")
    else:
        print(f"❌ {failed} tests failed")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)