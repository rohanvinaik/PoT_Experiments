"""
Local JSON-based Blockchain Client Implementation

This module provides a local JSON file-based implementation of the BlockchainClient
abstract base class. It stores cryptographic hashes and metadata in a local JSON file
with thread-safe operations for testing and development purposes.
"""

import json
import uuid
import fcntl
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading

from .blockchain_client import (
    BlockchainClient, 
    StorageError, 
    RetrievalError, 
    VerificationError,
    BlockchainClientError
)


class LocalBlockchainClient(BlockchainClient):
    """
    Local JSON file-based implementation of BlockchainClient.
    
    This implementation stores all blockchain operations in a local JSON file
    with thread-safe file locking. It's designed for testing, development, and
    scenarios where a full blockchain infrastructure is not available.
    
    The JSON structure follows the specified format with transactions stored
    by their unique UUIDs, including hash values, metadata, timestamps, and
    block numbers (set to null for local storage).
    """
    
    def __init__(self, client_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the local blockchain client.
        
        Args:
            client_id: Unique identifier for this client instance
            config: Optional configuration dictionary. Supports:
                   - storage_path: Path to JSON file (default: ./provenance_log.json)
                   - create_dirs: Whether to create parent directories (default: True)
        """
        super().__init__(client_id, config)
        
        # Configure storage path
        default_path = "./provenance_log.json"
        self.storage_path = Path(self.config.get("storage_path", default_path))
        self.create_dirs = self.config.get("create_dirs", True)
        
        # Thread safety
        self._file_lock = threading.Lock()
        
        # Initialize storage
        self._initialize_storage()
        self._initialized = True
        
        self.logger.info(f"LocalBlockchainClient initialized with storage at {self.storage_path}")
    
    def _initialize_storage(self) -> None:
        """
        Initialize the JSON storage file if it doesn't exist.
        
        Raises:
            StorageError: If storage initialization fails
        """
        try:
            # Create parent directories if needed
            if self.create_dirs:
                self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Initialize JSON file if it doesn't exist
            if not self.storage_path.exists():
                initial_data = {
                    "transactions": {},
                    "metadata": {
                        "created": datetime.now(timezone.utc).isoformat(),
                        "client_id": self.client_id,
                        "version": "1.0"
                    }
                }
                self._write_json_file(initial_data)
                self.logger.info("Created new provenance log file")
            else:
                # Validate existing file structure
                self._validate_storage_file()
                
        except Exception as e:
            raise StorageError(f"Failed to initialize storage: {str(e)}") from e
    
    def _validate_storage_file(self) -> None:
        """
        Validate that the existing storage file has the correct structure.
        
        Raises:
            StorageError: If the file structure is invalid
        """
        try:
            data = self._read_json_file()
            if not isinstance(data, dict) or "transactions" not in data:
                raise StorageError("Invalid storage file structure")
        except json.JSONDecodeError as e:
            raise StorageError(f"Corrupted JSON file: {str(e)}") from e
    
    def _read_json_file(self) -> Dict[str, Any]:
        """
        Thread-safe read of the JSON storage file.
        
        Returns:
            The parsed JSON data
            
        Raises:
            StorageError: If reading fails
        """
        try:
            with self._file_lock:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    # Acquire file lock for reading
                    fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                    try:
                        return json.load(f)
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except FileNotFoundError:
            raise StorageError("Storage file not found")
        except json.JSONDecodeError as e:
            raise StorageError(f"Failed to parse JSON: {str(e)}") from e
        except Exception as e:
            raise StorageError(f"Failed to read storage file: {str(e)}") from e
    
    def _write_json_file(self, data: Dict[str, Any]) -> None:
        """
        Thread-safe write to the JSON storage file.
        
        Args:
            data: The data to write
            
        Raises:
            StorageError: If writing fails
        """
        try:
            with self._file_lock:
                # Write to temporary file first for atomic operation
                temp_path = self.storage_path.with_suffix('.tmp')
                with open(temp_path, 'w', encoding='utf-8') as f:
                    # Acquire exclusive file lock
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                    try:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                        f.flush()
                        os.fsync(f.fileno())  # Force write to disk
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                
                # Atomic move
                temp_path.replace(self.storage_path)
                
        except Exception as e:
            # Clean up temporary file if it exists
            temp_path = self.storage_path.with_suffix('.tmp')
            if temp_path.exists():
                temp_path.unlink()
            raise StorageError(f"Failed to write storage file: {str(e)}") from e
    
    def store_hash(self, hash_value: str, metadata: Dict[str, Any]) -> str:
        """
        Store a cryptographic hash with metadata in the local JSON file.
        
        Args:
            hash_value: The cryptographic hash to store (hex string)
            metadata: Dictionary containing associated metadata
        
        Returns:
            transaction_id: Unique UUID identifier for the storage transaction
            
        Raises:
            StorageError: If the hash storage operation fails
        """
        self._log_operation("store_hash", hash_value=hash_value[:16] + "...")
        
        # Validate hash format
        if not self._validate_hash(hash_value):
            raise StorageError("Invalid hash format")
        
        try:
            # Generate unique transaction ID
            transaction_id = str(uuid.uuid4())
            
            # Read current data
            data = self._read_json_file()
            
            # Create transaction record
            transaction_record = {
                "hash": hash_value,
                "metadata": metadata.copy(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "block_number": None,
                "client_id": self.client_id
            }
            
            # Store transaction
            data["transactions"][transaction_id] = transaction_record
            
            # Update file metadata
            if "metadata" not in data:
                data["metadata"] = {}
            data["metadata"]["last_updated"] = datetime.now(timezone.utc).isoformat()
            data["metadata"]["transaction_count"] = len(data["transactions"])
            
            # Write back to file
            self._write_json_file(data)
            
            self.logger.info(f"Stored hash with transaction ID: {transaction_id}")
            return transaction_id
            
        except Exception as e:
            if isinstance(e, StorageError):
                raise
            raise StorageError(f"Failed to store hash: {str(e)}") from e
    
    def retrieve_hash(self, transaction_id: str) -> Dict[str, Any]:
        """
        Retrieve a stored hash and its metadata by transaction ID.
        
        Args:
            transaction_id: The unique transaction identifier
            
        Returns:
            Dictionary containing hash_value, metadata, storage_timestamp, and transaction_id
            
        Raises:
            RetrievalError: If the hash retrieval operation fails
        """
        self._log_operation("retrieve_hash", transaction_id=transaction_id)
        
        try:
            data = self._read_json_file()
            
            if transaction_id not in data["transactions"]:
                raise RetrievalError(f"Transaction ID not found: {transaction_id}")
            
            transaction = data["transactions"][transaction_id]
            
            result = {
                "hash_value": transaction["hash"],
                "metadata": transaction["metadata"],
                "storage_timestamp": transaction["timestamp"],
                "transaction_id": transaction_id,
                "client_id": transaction.get("client_id", "unknown")
            }
            
            self.logger.info(f"Retrieved transaction: {transaction_id}")
            return result
            
        except RetrievalError:
            raise
        except Exception as e:
            raise RetrievalError(f"Failed to retrieve hash: {str(e)}") from e
    
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
        
        try:
            # Validate hash format
            if not self._validate_hash(hash_value):
                raise VerificationError("Invalid hash format")
            
            # Retrieve stored transaction
            stored_data = self.retrieve_hash(transaction_id)
            stored_hash = stored_data["hash_value"]
            
            # Compare hashes
            matches = hash_value.lower() == stored_hash.lower()
            
            self.logger.info(f"Hash verification result: {matches} for transaction {transaction_id}")
            return matches
            
        except RetrievalError as e:
            raise VerificationError(f"Cannot verify - retrieval failed: {str(e)}") from e
        except Exception as e:
            raise VerificationError(f"Verification failed: {str(e)}") from e
    
    def get_all_transactions(self) -> List[Dict[str, Any]]:
        """
        Retrieve the complete provenance history of all transactions.
        
        Returns:
            List of all transaction records with their IDs included
            
        Raises:
            RetrievalError: If reading the transaction history fails
        """
        self._log_operation("get_all_transactions")
        
        try:
            data = self._read_json_file()
            transactions = []
            
            for tx_id, tx_data in data["transactions"].items():
                transaction_record = {
                    "transaction_id": tx_id,
                    "hash_value": tx_data["hash"],
                    "metadata": tx_data["metadata"],
                    "storage_timestamp": tx_data["timestamp"],
                    "block_number": tx_data.get("block_number"),
                    "client_id": tx_data.get("client_id", "unknown")
                }
                transactions.append(transaction_record)
            
            # Sort by timestamp (newest first)
            transactions.sort(key=lambda x: x["storage_timestamp"], reverse=True)
            
            self.logger.info(f"Retrieved {len(transactions)} transactions")
            return transactions
            
        except Exception as e:
            raise RetrievalError(f"Failed to retrieve transaction history: {str(e)}") from e
    
    def get_transaction_count(self) -> int:
        """
        Get the total number of stored transactions.
        
        Returns:
            The number of transactions in storage
        """
        try:
            data = self._read_json_file()
            return len(data["transactions"])
        except Exception as e:
            self.logger.error(f"Failed to get transaction count: {str(e)}")
            return 0
    
    def clear_all_transactions(self) -> None:
        """
        Clear all stored transactions (for testing purposes).
        
        WARNING: This operation is irreversible.
        
        Raises:
            StorageError: If clearing fails
        """
        self._log_operation("clear_all_transactions")
        
        try:
            data = self._read_json_file()
            data["transactions"] = {}
            data["metadata"]["last_cleared"] = datetime.now(timezone.utc).isoformat()
            data["metadata"]["transaction_count"] = 0
            
            self._write_json_file(data)
            self.logger.warning("All transactions cleared from storage")
            
        except Exception as e:
            raise StorageError(f"Failed to clear transactions: {str(e)}") from e