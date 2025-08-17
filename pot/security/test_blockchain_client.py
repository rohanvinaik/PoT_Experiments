"""
Comprehensive tests for blockchain client implementations.

Tests the LocalBlockchainClient, Web3BlockchainClient (mocked), and factory
functionality with proper fallback behavior and error handling.
"""

import os
import json
import tempfile
import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime, timezone

# Import blockchain clients
from .blockchain_client import BlockchainClient, StorageError, RetrievalError, VerificationError
from .local_blockchain_client import LocalBlockchainClient
from .blockchain_factory import (
    get_blockchain_client, 
    test_blockchain_connection,
    BlockchainConfig,
    BlockchainClientFactory
)

# Conditional Web3 imports
try:
    from .web3_blockchain_client import Web3BlockchainClient, Web3ConnectionError
    from web3 import Web3
    from web3.exceptions import Web3Exception
    WEB3_AVAILABLE = True
except ImportError:
    Web3BlockchainClient = None
    Web3ConnectionError = Exception
    Web3 = None
    Web3Exception = Exception
    WEB3_AVAILABLE = False


# Test Fixtures

@pytest.fixture
def temp_json_file():
    """Create a temporary JSON file for local client testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def sample_hash_data():
    """Sample hash and metadata for testing."""
    return {
        "hash": "0x1234567890abcdef" * 4,  # 64 character hex string
        "metadata": {
            "model_id": "test_model_v1",
            "epoch": 42,
            "accuracy": 0.95,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "test_run": True
        }
    }


@pytest.fixture
def multiple_hash_data():
    """Multiple hash records for batch testing."""
    return [
        {
            "hash": f"0x{i:064x}",
            "metadata": {"test_id": i, "batch": "test_batch_1"}
        }
        for i in range(1, 6)
    ]


@pytest.fixture
def local_client_config(temp_json_file):
    """Configuration for local blockchain client."""
    return {
        "storage_path": temp_json_file,
        "create_dirs": True
    }


@pytest.fixture
def mock_web3():
    """Mock Web3 instance for testing."""
    if not WEB3_AVAILABLE:
        pytest.skip("Web3 not available")
    
    mock_w3 = Mock()
    mock_w3.is_connected.return_value = True
    mock_w3.eth.chain_id = 137  # Polygon
    mock_w3.eth.block_number = 12345678
    mock_w3.eth.gas_price = 30000000000  # 30 Gwei
    mock_w3.to_wei.return_value = 30000000000
    mock_w3.from_wei.return_value = 30.0
    
    # Mock account balance
    mock_w3.eth.get_balance.return_value = 1000000000000000000  # 1 ETH
    
    # Mock transaction receipt
    mock_receipt = Mock()
    mock_receipt.status = 1
    mock_receipt.blockNumber = 12345679
    mock_receipt.gasUsed = 80000
    mock_receipt.transactionHash = "0xabcdef123456"
    
    mock_w3.eth.wait_for_transaction_receipt.return_value = mock_receipt
    mock_w3.eth.send_raw_transaction.return_value = "0xabcdef123456"
    mock_w3.eth.get_transaction_count.return_value = 42
    
    # Mock block
    mock_block = Mock()
    mock_block.number = 12345679
    mock_block.timestamp = 1640995200  # 2022-01-01
    mock_w3.eth.get_block.return_value = mock_block
    
    return mock_w3


@pytest.fixture(autouse=True)
def clear_factory_cache():
    """Clear factory cache before each test."""
    BlockchainClientFactory.clear_cache()
    yield
    BlockchainClientFactory.clear_cache()


# LocalBlockchainClient Tests

class TestLocalBlockchainClient:
    """Test suite for LocalBlockchainClient."""
    
    def test_initialization(self, local_client_config):
        """Test local client initialization."""
        client = LocalBlockchainClient("test_client", local_client_config)
        assert client.client_id == "test_client"
        assert client._initialized
        assert client.storage_path.exists()
    
    def test_store_hash(self, local_client_config, sample_hash_data):
        """Test hash storage in local client."""
        client = LocalBlockchainClient("test_client", local_client_config)
        
        tx_id = client.store_hash(sample_hash_data["hash"], sample_hash_data["metadata"])
        
        assert tx_id is not None
        assert isinstance(tx_id, str)
        assert len(tx_id) == 36  # UUID format
    
    def test_retrieve_hash(self, local_client_config, sample_hash_data):
        """Test hash retrieval from local client."""
        client = LocalBlockchainClient("test_client", local_client_config)
        
        # Store hash first
        tx_id = client.store_hash(sample_hash_data["hash"], sample_hash_data["metadata"])
        
        # Retrieve hash
        retrieved = client.retrieve_hash(tx_id)
        
        assert retrieved["hash_value"] == sample_hash_data["hash"]
        assert retrieved["metadata"] == sample_hash_data["metadata"]
        assert retrieved["transaction_id"] == tx_id
        assert "storage_timestamp" in retrieved
    
    def test_verify_hash(self, local_client_config, sample_hash_data):
        """Test hash verification in local client."""
        client = LocalBlockchainClient("test_client", local_client_config)
        
        # Store hash
        tx_id = client.store_hash(sample_hash_data["hash"], sample_hash_data["metadata"])
        
        # Verify correct hash
        assert client.verify_hash(sample_hash_data["hash"], tx_id)
        
        # Verify incorrect hash
        wrong_hash = "0x" + "f" * 64
        assert not client.verify_hash(wrong_hash, tx_id)
    
    def test_get_all_transactions(self, local_client_config, multiple_hash_data):
        """Test retrieving all transactions."""
        client = LocalBlockchainClient("test_client", local_client_config)
        
        # Store multiple hashes
        tx_ids = []
        for data in multiple_hash_data:
            tx_id = client.store_hash(data["hash"], data["metadata"])
            tx_ids.append(tx_id)
        
        # Retrieve all transactions
        all_transactions = client.get_all_transactions()
        
        assert len(all_transactions) == len(multiple_hash_data)
        
        # Check that all stored transactions are present
        retrieved_tx_ids = {tx["transaction_id"] for tx in all_transactions}
        assert set(tx_ids) == retrieved_tx_ids
    
    def test_transaction_not_found(self, local_client_config):
        """Test retrieval of non-existent transaction."""
        client = LocalBlockchainClient("test_client", local_client_config)
        
        with pytest.raises(RetrievalError, match="Transaction ID not found"):
            client.retrieve_hash("non-existent-uuid")
    
    def test_invalid_hash_format(self, local_client_config):
        """Test storing invalid hash format."""
        client = LocalBlockchainClient("test_client", local_client_config)
        
        with pytest.raises(StorageError, match="Invalid hash format"):
            client.store_hash("not-a-hex-hash", {"test": True})
    
    def test_thread_safety(self, local_client_config, sample_hash_data):
        """Test thread-safe operations."""
        import threading
        import concurrent.futures
        
        client = LocalBlockchainClient("test_client", local_client_config)
        results = []
        
        def store_hash_worker(i):
            hash_val = f"0x{i:064x}"
            metadata = {"worker_id": i}
            return client.store_hash(hash_val, metadata)
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(store_hash_worker, i) for i in range(10)]
            results = [future.result() for future in futures]
        
        # Verify all operations succeeded
        assert len(results) == 10
        assert len(set(results)) == 10  # All unique transaction IDs


# Web3BlockchainClient Tests (Mocked)

@pytest.mark.skipif(not WEB3_AVAILABLE, reason="Web3 not available")
class TestWeb3BlockchainClient:
    """Test suite for Web3BlockchainClient with mocked Web3."""
    
    def test_initialization_success(self, mock_web3):
        """Test successful Web3 client initialization."""
        with patch('pot.security.web3_blockchain_client.Web3') as mock_web3_class:
            mock_web3_class.return_value = mock_web3
            
            config = {
                "rpc_url": "https://polygon-rpc.com",
                "private_key": "0x" + "1" * 64,
                "contract_address": "0x" + "a" * 40,
            }
            
            with patch('pot.security.web3_blockchain_client.Account') as mock_account:
                mock_account.from_key.return_value.address = "0x" + "b" * 40
                
                # Mock contract code check
                mock_web3.eth.get_code.return_value = b'contract_code'
                
                client = Web3BlockchainClient("test_client", config)
                assert client.client_id == "test_client"
                assert client._initialized
    
    def test_connection_check(self, mock_web3):
        """Test connection checking."""
        with patch('pot.security.web3_blockchain_client.Web3') as mock_web3_class:
            mock_web3_class.return_value = mock_web3
            
            config = {
                "rpc_url": "https://polygon-rpc.com",
                "private_key": "0x" + "1" * 64,
                "contract_address": "0x" + "a" * 40,
            }
            
            with patch('pot.security.web3_blockchain_client.Account') as mock_account:
                mock_account.from_key.return_value.address = "0x" + "b" * 40
                mock_web3.eth.get_code.return_value = b'contract_code'
                
                client = Web3BlockchainClient("test_client", config)
                
                # Test successful connection
                assert client.check_connection()
                
                # Test failed connection
                mock_web3.is_connected.return_value = False
                assert not client.check_connection()
    
    def test_store_hash_mocked(self, mock_web3, sample_hash_data):
        """Test hash storage with mocked Web3."""
        with patch('pot.security.web3_blockchain_client.Web3') as mock_web3_class:
            mock_web3_class.return_value = mock_web3
            
            config = {
                "rpc_url": "https://polygon-rpc.com",
                "private_key": "0x" + "1" * 64,
                "contract_address": "0x" + "a" * 40,
            }
            
            with patch('pot.security.web3_blockchain_client.Account') as mock_account:
                mock_account_obj = Mock()
                mock_account_obj.address = "0x" + "b" * 40
                mock_account_obj.sign_transaction.return_value.rawTransaction = b'signed_tx'
                mock_account.from_key.return_value = mock_account_obj
                
                mock_web3.eth.get_code.return_value = b'contract_code'
                
                # Mock contract
                mock_contract = Mock()
                mock_function = Mock()
                mock_function.build_transaction.return_value = {"gas": 100000}
                mock_function.estimate_gas.return_value = 80000
                mock_contract.functions.storeHash.return_value = mock_function
                mock_web3.eth.contract.return_value = mock_contract
                
                client = Web3BlockchainClient("test_client", config)
                
                tx_id = client.store_hash(sample_hash_data["hash"], sample_hash_data["metadata"])
                
                assert tx_id == "0xabcdef123456"
    
    def test_missing_web3_dependency(self):
        """Test behavior when Web3 is not available."""
        with patch('pot.security.web3_blockchain_client.WEB3_AVAILABLE', False):
            config = {
                "rpc_url": "https://polygon-rpc.com",
                "private_key": "0x" + "1" * 64,
                "contract_address": "0x" + "a" * 40,
            }
            
            with pytest.raises(ImportError, match="web3 library not installed"):
                Web3BlockchainClient("test_client", config)


# Factory Tests

class TestBlockchainFactory:
    """Test suite for blockchain client factory."""
    
    def test_auto_fallback_to_local(self):
        """Test automatic fallback to local client."""
        # Clear environment variables
        with patch.dict(os.environ, {}, clear=True):
            client = get_blockchain_client()
            assert isinstance(client, LocalBlockchainClient)
    
    def test_explicit_local_client(self):
        """Test explicit local client creation."""
        client = get_blockchain_client(client_type="local")
        assert isinstance(client, LocalBlockchainClient)
    
    def test_force_local_override(self):
        """Test force_local parameter."""
        # Even with Web3 config, should return local client
        with patch.dict(os.environ, {
            "RPC_URL": "https://test.com",
            "PRIVATE_KEY": "0x" + "1" * 64,
            "CONTRACT_ADDRESS": "0x" + "a" * 40,
        }):
            client = get_blockchain_client(force_local=True)
            assert isinstance(client, LocalBlockchainClient)
    
    @pytest.mark.skipif(not WEB3_AVAILABLE, reason="Web3 not available")
    def test_web3_client_creation_with_config(self, mock_web3):
        """Test Web3 client creation with proper configuration."""
        with patch('pot.security.web3_blockchain_client.Web3') as mock_web3_class:
            mock_web3_class.return_value = mock_web3
            
            with patch('pot.security.web3_blockchain_client.Account') as mock_account:
                mock_account.from_key.return_value.address = "0x" + "b" * 40
                mock_web3.eth.get_code.return_value = b'contract_code'
                
                with patch.dict(os.environ, {
                    "RPC_URL": "https://polygon-rpc.com",
                    "PRIVATE_KEY": "0x" + "1" * 64,
                    "CONTRACT_ADDRESS": "0x" + "a" * 40,
                }):
                    client = get_blockchain_client(client_type="web3")
                    assert isinstance(client, Web3BlockchainClient)
    
    def test_client_caching(self):
        """Test client instance caching."""
        client1 = get_blockchain_client(client_id="cache_test")
        client2 = get_blockchain_client(client_id="cache_test")
        
        # Should be the same instance
        assert client1 is client2
        
        # Force new instance
        client3 = get_blockchain_client(client_id="cache_test", force_new=True)
        assert client3 is not client1
    
    def test_configuration_loading(self, temp_json_file):
        """Test configuration loading from file."""
        config_data = {
            "client_type": "local",
            "local_storage_path": temp_json_file,
            "client_id": "config_test"
        }
        
        with open(temp_json_file, 'w') as f:
            json.dump(config_data, f)
        
        client = get_blockchain_client(config_file=temp_json_file)
        assert client.client_id == "config_test"
        assert isinstance(client, LocalBlockchainClient)


# Connection Testing

class TestConnectionTesting:
    """Test suite for connection testing functionality."""
    
    def test_local_client_connection_test(self, temp_json_file):
        """Test connection testing with local client."""
        config = {"storage_path": temp_json_file}
        client = LocalBlockchainClient("test_client", config)
        
        success, results = test_blockchain_connection(client)
        
        assert success
        assert results["client_type"] == "LocalBlockchainClient"
        assert results["tests"]["connection"]["success"]
        assert results["tests"]["store_hash"]["success"]
        assert results["tests"]["retrieve_hash"]["success"]
        assert results["tests"]["verify_hash"]["success"]
    
    def test_connection_test_with_factory(self):
        """Test connection testing using factory."""
        success, results = test_blockchain_connection()
        
        # Should succeed with local client fallback
        assert success
        assert "LocalBlockchainClient" in results["client_type"]


# Hash Verification Workflows

class TestHashVerificationWorkflows:
    """Test suite for complete hash verification workflows."""
    
    def test_complete_verification_workflow(self, sample_hash_data):
        """Test complete hash verification workflow."""
        client = get_blockchain_client(force_local=True)
        
        # Step 1: Store hash
        tx_id = client.store_hash(sample_hash_data["hash"], sample_hash_data["metadata"])
        assert tx_id is not None
        
        # Step 2: Retrieve and verify structure
        retrieved = client.retrieve_hash(tx_id)
        assert retrieved["hash_value"] == sample_hash_data["hash"]
        assert retrieved["metadata"] == sample_hash_data["metadata"]
        
        # Step 3: Verify hash authenticity
        assert client.verify_hash(sample_hash_data["hash"], tx_id)
        
        # Step 4: Test negative verification
        wrong_hash = "0x" + "f" * 64
        assert not client.verify_hash(wrong_hash, tx_id)
    
    def test_batch_verification_workflow(self, multiple_hash_data):
        """Test batch hash verification workflow."""
        client = get_blockchain_client(force_local=True)
        
        # Store multiple hashes
        tx_ids = []
        for data in multiple_hash_data:
            tx_id = client.store_hash(data["hash"], data["metadata"])
            tx_ids.append((tx_id, data))
        
        # Verify all hashes
        for tx_id, data in tx_ids:
            assert client.verify_hash(data["hash"], tx_id)
        
        # Test cross-verification (should fail)
        for i, (tx_id, _) in enumerate(tx_ids):
            wrong_data = multiple_hash_data[(i + 1) % len(multiple_hash_data)]
            assert not client.verify_hash(wrong_data["hash"], tx_id)


# Integration Tests

class TestIntegration:
    """Integration tests for full PoT verification flow."""
    
    def test_full_pot_verification_flow(self, sample_hash_data):
        """Test complete PoT verification flow integration."""
        # This simulates how the PoT framework would use blockchain clients
        
        # Step 1: Get client (automatic selection)
        client = get_blockchain_client()
        
        # Step 2: Test connection
        success, test_results = test_blockchain_connection(client)
        assert success
        
        # Step 3: Store verification hash (simulating model verification)
        model_verification_hash = sample_hash_data["hash"]
        verification_metadata = {
            **sample_hash_data["metadata"],
            "verification_type": "model_fingerprint",
            "pot_version": "1.0.0",
            "network": test_results["client_type"]
        }
        
        verification_tx_id = client.store_hash(model_verification_hash, verification_metadata)
        
        # Step 4: Later verification (simulating audit)
        audit_retrieved = client.retrieve_hash(verification_tx_id)
        assert audit_retrieved["hash_value"] == model_verification_hash
        assert audit_retrieved["metadata"]["verification_type"] == "model_fingerprint"
        
        # Step 5: Cryptographic verification
        is_authentic = client.verify_hash(model_verification_hash, verification_tx_id)
        assert is_authentic
        
        # Step 6: Test tamper detection
        tampered_hash = model_verification_hash[:-1] + "f"
        is_tampered = client.verify_hash(tampered_hash, verification_tx_id)
        assert not is_tampered
    
    def test_factory_fallback_integration(self):
        """Test factory fallback behavior in realistic scenarios."""
        # Clear cache to ensure fresh testing
        BlockchainClientFactory.clear_cache()
        
        # Scenario 1: No Web3 configuration - should use local
        with patch.dict(os.environ, {}, clear=True):
            client1 = get_blockchain_client()
            assert isinstance(client1, LocalBlockchainClient)
        
        # Scenario 2: Partial Web3 configuration - should fallback to local
        with patch.dict(os.environ, {"RPC_URL": "https://test.com"}, clear=True):
            client2 = get_blockchain_client()
            assert isinstance(client2, LocalBlockchainClient)
        
        # Scenario 3: Force local even with full config
        with patch.dict(os.environ, {
            "RPC_URL": "https://test.com",
            "PRIVATE_KEY": "0x" + "1" * 64,
            "CONTRACT_ADDRESS": "0x" + "a" * 40,
            "FORCE_LOCAL_BLOCKCHAIN": "true"
        }, clear=True):
            client3 = get_blockchain_client()
            assert isinstance(client3, LocalBlockchainClient)
    
    def test_configuration_validation_flow(self):
        """Test configuration validation in realistic scenarios."""
        # Test various configuration scenarios
        configs = [
            # Valid local config
            {"client_type": "local", "local_storage_path": "./test.json"},
            
            # Auto config with no Web3 setup
            {"client_type": "auto"},
            
            # Force local override
            {"force_local": True, "client_type": "web3"},
        ]
        
        for config in configs:
            try:
                client = get_blockchain_client(**config)
                assert isinstance(client, BlockchainClient)
                
                # Test basic operations
                success, _ = test_blockchain_connection(client)
                assert success
                
            except Exception as e:
                pytest.fail(f"Configuration {config} failed: {str(e)}")


# Error Handling Tests

class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    def test_invalid_hash_formats(self):
        """Test handling of invalid hash formats."""
        client = get_blockchain_client(force_local=True)
        
        invalid_hashes = [
            "",  # Empty hash
            "not-hex",  # Non-hex string
            "0x123",  # Too short
            "123",  # Missing 0x prefix but valid hex
        ]
        
        for invalid_hash in invalid_hashes:
            with pytest.raises((StorageError, VerificationError)):
                client.store_hash(invalid_hash, {"test": True})
    
    def test_storage_permission_errors(self):
        """Test handling of storage permission errors."""
        # Try to use an invalid storage path
        config = {"storage_path": "/root/no_permission.json"}
        
        with pytest.raises(StorageError):
            client = LocalBlockchainClient("test_client", config)
    
    def test_corrupted_storage_recovery(self, temp_json_file):
        """Test recovery from corrupted storage."""
        # Create corrupted JSON file
        with open(temp_json_file, 'w') as f:
            f.write("{ invalid json")
        
        config = {"storage_path": temp_json_file}
        
        with pytest.raises(StorageError):
            LocalBlockchainClient("test_client", config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])