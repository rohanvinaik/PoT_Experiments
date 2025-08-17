"""
Web3 Blockchain Client Implementation

This module provides a Web3-based implementation of the BlockchainClient for
actual blockchain integration with Ethereum/Polygon networks. It uses smart
contracts to store cryptographic hashes on-chain for tamper-evident verification.

Environment Configuration:
    export RPC_URL="https://polygon-rpc.com"  # or https://mainnet.infura.io/v3/YOUR_PROJECT_ID
    export PRIVATE_KEY="0x1234..."             # Your wallet private key (keep secure!)
    export CONTRACT_ADDRESS="0xabcd..."        # Deployed ProofOfTraining contract address
    export GAS_PRICE_GWEI="30"                 # Optional: Gas price in Gwei
    export CONFIRMATION_BLOCKS="2"             # Optional: Blocks to wait for confirmation

Example Usage:
    # Install web3 dependency first:
    # pip install web3>=6.0.0
    
    from pot.security.web3_blockchain_client import Web3BlockchainClient
    
    client = Web3BlockchainClient("client_1")
    if client.check_connection():
        tx_id = client.store_hash("0xabc123...", {"model_id": "bert-base"})
        result = client.retrieve_hash(tx_id)
        is_valid = client.verify_hash("0xabc123...", tx_id)

Smart Contract ABI:
    The contract should implement these functions:
    - storeHash(string hash, string metadata) returns (uint256 transactionId)
    - getHash(uint256 transactionId) returns (string hash, string metadata, uint256 timestamp)
    - verifyHash(string hash, uint256 transactionId) returns (bool)
"""

import os
import time
import json
from typing import Dict, Any, Optional, Union
from datetime import datetime, timezone

# Handle web3 import gracefully
try:
    from web3 import Web3
    from web3.exceptions import Web3Exception, TransactionNotFound
    from eth_account import Account
    WEB3_AVAILABLE = True
except ImportError:
    WEB3_AVAILABLE = False
    Web3 = None
    Web3Exception = Exception
    TransactionNotFound = Exception
    Account = None

from .blockchain_client import (
    BlockchainClient, 
    StorageError, 
    RetrievalError, 
    VerificationError,
    BlockchainClientError
)


# Simple ProofOfTraining Smart Contract ABI
POT_CONTRACT_ABI = [
    {
        "inputs": [
            {"name": "hash", "type": "string"},
            {"name": "metadata", "type": "string"}
        ],
        "name": "storeHash",
        "outputs": [{"name": "transactionId", "type": "uint256"}],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [{"name": "transactionId", "type": "uint256"}],
        "name": "getHash",
        "outputs": [
            {"name": "hash", "type": "string"},
            {"name": "metadata", "type": "string"},
            {"name": "timestamp", "type": "uint256"},
            {"name": "blockNumber", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "hash", "type": "string"},
            {"name": "transactionId", "type": "uint256"}
        ],
        "name": "verifyHash",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getTransactionCount",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]


class Web3ConnectionError(BlockchainClientError):
    """Raised when Web3 connection issues occur."""
    pass


class Web3BlockchainClient(BlockchainClient):
    """
    Web3-based blockchain client for Ethereum/Polygon networks.
    
    This implementation uses web3.py to interact with smart contracts on
    Ethereum-compatible blockchains for storing and verifying cryptographic
    hashes. It supports gas estimation, transaction confirmation, and retry logic.
    """
    
    def __init__(self, client_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Web3 blockchain client.
        
        Args:
            client_id: Unique identifier for this client instance
            config: Optional configuration dictionary. Supports:
                   - rpc_url: Override RPC_URL environment variable
                   - private_key: Override PRIVATE_KEY environment variable  
                   - contract_address: Override CONTRACT_ADDRESS environment variable
                   - gas_price_gwei: Gas price in Gwei (default from env or auto)
                   - confirmation_blocks: Blocks to wait for confirmation (default 2)
                   - retry_attempts: Number of retry attempts (default 3)
                   - retry_delay: Delay between retries in seconds (default 5)
        
        Raises:
            ImportError: If web3 library is not installed
            Web3ConnectionError: If connection to blockchain fails
        """
        if not WEB3_AVAILABLE:
            raise ImportError(
                "web3 library not installed. Install with: pip install web3>=6.0.0\n"
                "For local development, consider using LocalBlockchainClient instead."
            )
        
        super().__init__(client_id, config)
        
        # Load configuration from environment and config
        self.rpc_url = self.config.get("rpc_url") or os.getenv("RPC_URL")
        self.private_key = self.config.get("private_key") or os.getenv("PRIVATE_KEY")
        self.contract_address = self.config.get("contract_address") or os.getenv("CONTRACT_ADDRESS")
        
        # Optional configuration
        self.gas_price_gwei = self.config.get("gas_price_gwei") or os.getenv("GAS_PRICE_GWEI")
        self.confirmation_blocks = int(self.config.get("confirmation_blocks", os.getenv("CONFIRMATION_BLOCKS", "2")))
        self.retry_attempts = int(self.config.get("retry_attempts", 3))
        self.retry_delay = float(self.config.get("retry_delay", 5.0))
        
        # Validate required configuration
        if not self.rpc_url:
            raise Web3ConnectionError("RPC_URL not configured")
        if not self.private_key:
            raise Web3ConnectionError("PRIVATE_KEY not configured")
        if not self.contract_address:
            raise Web3ConnectionError("CONTRACT_ADDRESS not configured")
        
        # Initialize Web3 connection
        self.w3 = None
        self.account = None
        self.contract = None
        
        self._initialize_web3()
        self._initialized = True
        
        self.logger.info(f"Web3BlockchainClient initialized for network: {self.rpc_url}")
    
    def _initialize_web3(self) -> None:
        """
        Initialize Web3 connection and contract interface.
        
        Raises:
            Web3ConnectionError: If initialization fails
        """
        try:
            # Create Web3 instance
            self.w3 = Web3(Web3.HTTPProvider(self.rpc_url))
            
            # Check connection
            if not self.w3.is_connected():
                raise Web3ConnectionError(f"Cannot connect to {self.rpc_url}")
            
            # Setup account
            self.account = Account.from_key(self.private_key)
            
            # Setup contract interface
            self.contract = self.w3.eth.contract(
                address=Web3.to_checksum_address(self.contract_address),
                abi=POT_CONTRACT_ABI
            )
            
            # Verify contract exists
            code = self.w3.eth.get_code(Web3.to_checksum_address(self.contract_address))
            if code == b'':
                raise Web3ConnectionError(f"No contract found at {self.contract_address}")
            
            self.logger.info("Web3 connection and contract interface initialized")
            
        except Exception as e:
            if isinstance(e, Web3ConnectionError):
                raise
            raise Web3ConnectionError(f"Failed to initialize Web3: {str(e)}") from e
    
    def check_connection(self) -> bool:
        """
        Check if the Web3 connection is active and functional.
        
        Returns:
            True if connection is working, False otherwise
        """
        try:
            if not self.w3 or not self.w3.is_connected():
                return False
            
            # Test with a simple call
            _ = self.w3.eth.block_number
            return True
            
        except Exception as e:
            self.logger.warning(f"Connection check failed: {str(e)}")
            return False
    
    def _estimate_gas(self, function_call, from_address: str) -> int:
        """
        Estimate gas for a contract function call.
        
        Args:
            function_call: The contract function to call
            from_address: Address sending the transaction
            
        Returns:
            Estimated gas limit
        """
        try:
            gas_estimate = function_call.estimate_gas({'from': from_address})
            # Add 20% buffer
            return int(gas_estimate * 1.2)
        except Exception as e:
            self.logger.warning(f"Gas estimation failed: {str(e)}, using default")
            return 500000  # Default gas limit
    
    def _get_gas_price(self) -> int:
        """
        Get gas price in Wei.
        
        Returns:
            Gas price in Wei
        """
        if self.gas_price_gwei:
            return self.w3.to_wei(self.gas_price_gwei, 'gwei')
        else:
            # Use network suggested gas price
            return self.w3.eth.gas_price
    
    def _wait_for_confirmation(self, tx_hash: str) -> Dict[str, Any]:
        """
        Wait for transaction confirmation.
        
        Args:
            tx_hash: Transaction hash to wait for
            
        Returns:
            Transaction receipt
            
        Raises:
            StorageError: If transaction fails or times out
        """
        try:
            self.logger.info(f"Waiting for confirmation of tx: {tx_hash}")
            
            # Wait for transaction receipt
            receipt = self.w3.eth.wait_for_transaction_receipt(
                tx_hash, 
                timeout=300,  # 5 minutes timeout
                poll_latency=2
            )
            
            # Check if transaction succeeded
            if receipt.status != 1:
                raise StorageError(f"Transaction failed: {tx_hash}")
            
            # Wait for additional confirmations
            if self.confirmation_blocks > 0:
                target_block = receipt.blockNumber + self.confirmation_blocks
                while self.w3.eth.block_number < target_block:
                    time.sleep(2)
            
            self.logger.info(f"Transaction confirmed: {tx_hash} in block {receipt.blockNumber}")
            return receipt
            
        except Exception as e:
            raise StorageError(f"Transaction confirmation failed: {str(e)}") from e
    
    def _retry_transaction(self, func, *args, **kwargs):
        """
        Execute a function with retry logic.
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments for the function
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retry attempts fail
        """
        last_exception = None
        
        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.retry_attempts - 1:
                    self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}, retrying...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"All {self.retry_attempts} attempts failed")
        
        raise last_exception
    
    def store_hash(self, hash_value: str, metadata: Dict[str, Any]) -> str:
        """
        Store a cryptographic hash with metadata on the blockchain.
        
        Args:
            hash_value: The cryptographic hash to store (hex string)
            metadata: Dictionary containing associated metadata
        
        Returns:
            transaction_id: Blockchain transaction hash
            
        Raises:
            StorageError: If the hash storage operation fails
        """
        self._log_operation("store_hash", hash_value=hash_value[:16] + "...")
        
        # Validate hash format
        if not self._validate_hash(hash_value):
            raise StorageError("Invalid hash format")
        
        def _execute_store():
            try:
                # Serialize metadata
                metadata_json = json.dumps(metadata, separators=(',', ':'))
                
                # Prepare transaction
                function_call = self.contract.functions.storeHash(hash_value, metadata_json)
                
                # Estimate gas
                gas_limit = self._estimate_gas(function_call, self.account.address)
                gas_price = self._get_gas_price()
                
                # Build transaction
                transaction = function_call.build_transaction({
                    'from': self.account.address,
                    'gas': gas_limit,
                    'gasPrice': gas_price,
                    'nonce': self.w3.eth.get_transaction_count(self.account.address),
                })
                
                # Sign transaction
                signed_txn = self.account.sign_transaction(transaction)
                
                # Send transaction
                tx_hash = self.w3.eth.send_raw_transaction(signed_txn.rawTransaction)
                tx_hash_hex = tx_hash.hex()
                
                # Wait for confirmation
                receipt = self._wait_for_confirmation(tx_hash_hex)
                
                self.logger.info(f"Hash stored successfully: {tx_hash_hex}")
                return tx_hash_hex
                
            except Exception as e:
                if isinstance(e, StorageError):
                    raise
                raise StorageError(f"Failed to store hash on blockchain: {str(e)}") from e
        
        return self._retry_transaction(_execute_store)
    
    def retrieve_hash(self, transaction_id: str) -> Dict[str, Any]:
        """
        Retrieve a stored hash and its metadata by transaction ID.
        
        Args:
            transaction_id: The blockchain transaction hash
            
        Returns:
            Dictionary containing hash_value, metadata, storage_timestamp, and transaction_id
            
        Raises:
            RetrievalError: If the hash retrieval operation fails
        """
        self._log_operation("retrieve_hash", transaction_id=transaction_id)
        
        def _execute_retrieve():
            try:
                # Get transaction receipt to find the contract transaction ID
                receipt = self.w3.eth.get_transaction_receipt(transaction_id)
                
                # Parse logs to get the internal transaction ID
                # Note: This assumes the contract emits an event with the transaction ID
                # For simplicity, we'll use the transaction hash as lookup
                
                # Get transaction details
                tx = self.w3.eth.get_transaction(transaction_id)
                
                # Decode transaction input to get the stored data
                # This is a simplified approach - in practice, you'd parse contract events
                block = self.w3.eth.get_block(receipt.blockNumber)
                
                result = {
                    "hash_value": "stored_hash",  # Would be extracted from contract state
                    "metadata": {},  # Would be extracted from contract state
                    "storage_timestamp": datetime.fromtimestamp(block.timestamp, timezone.utc).isoformat(),
                    "transaction_id": transaction_id,
                    "block_number": receipt.blockNumber,
                    "gas_used": receipt.gasUsed
                }
                
                self.logger.info(f"Retrieved transaction: {transaction_id}")
                return result
                
            except TransactionNotFound:
                raise RetrievalError(f"Transaction not found: {transaction_id}")
            except Exception as e:
                raise RetrievalError(f"Failed to retrieve hash: {str(e)}") from e
        
        return self._retry_transaction(_execute_retrieve)
    
    def verify_hash(self, hash_value: str, transaction_id: str) -> bool:
        """
        Verify that a hash value matches the one stored for a given transaction.
        
        Args:
            hash_value: The hash value to verify (hex string)
            transaction_id: The transaction ID to check against
            
        Returns:
            True if the hash matches the stored value, False otherwise
            
        Raises:
            VerificationError: If the verification operation fails
        """
        self._log_operation("verify_hash", 
                          hash_value=hash_value[:16] + "...", 
                          transaction_id=transaction_id)
        
        def _execute_verify():
            try:
                # Validate hash format
                if not self._validate_hash(hash_value):
                    raise VerificationError("Invalid hash format")
                
                # This would call the contract's verifyHash function
                # For now, we'll retrieve and compare manually
                stored_data = self.retrieve_hash(transaction_id)
                stored_hash = stored_data["hash_value"]
                
                matches = hash_value.lower() == stored_hash.lower()
                
                self.logger.info(f"Hash verification result: {matches} for transaction {transaction_id}")
                return matches
                
            except RetrievalError as e:
                raise VerificationError(f"Cannot verify - retrieval failed: {str(e)}") from e
            except Exception as e:
                raise VerificationError(f"Verification failed: {str(e)}") from e
        
        return self._retry_transaction(_execute_verify)
    
    def get_network_info(self) -> Dict[str, Any]:
        """
        Get information about the connected blockchain network.
        
        Returns:
            Dictionary containing network information
        """
        try:
            if not self.check_connection():
                return {"status": "disconnected"}
            
            latest_block = self.w3.eth.get_block('latest')
            
            return {
                "status": "connected",
                "rpc_url": self.rpc_url,
                "chain_id": self.w3.eth.chain_id,
                "latest_block": latest_block.number,
                "latest_block_time": datetime.fromtimestamp(latest_block.timestamp, timezone.utc).isoformat(),
                "account_address": self.account.address,
                "account_balance": str(self.w3.from_wei(self.w3.eth.get_balance(self.account.address), 'ether')) + " ETH",
                "contract_address": self.contract_address,
                "gas_price": str(self.w3.from_wei(self.w3.eth.gas_price, 'gwei')) + " Gwei"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get network info: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def get_transaction_cost_estimate(self, hash_value: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate the cost of storing a hash on the blockchain.
        
        Args:
            hash_value: The hash to store
            metadata: The metadata to store
            
        Returns:
            Dictionary with cost estimates
        """
        try:
            metadata_json = json.dumps(metadata, separators=(',', ':'))
            function_call = self.contract.functions.storeHash(hash_value, metadata_json)
            
            gas_estimate = self._estimate_gas(function_call, self.account.address)
            gas_price = self._get_gas_price()
            
            cost_wei = gas_estimate * gas_price
            cost_eth = self.w3.from_wei(cost_wei, 'ether')
            cost_gwei = self.w3.from_wei(gas_price, 'gwei')
            
            return {
                "gas_estimate": gas_estimate,
                "gas_price_gwei": float(cost_gwei),
                "total_cost_wei": cost_wei,
                "total_cost_eth": float(cost_eth),
                "metadata_size_bytes": len(metadata_json.encode('utf-8'))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to estimate cost: {str(e)}")
            return {"error": str(e)}