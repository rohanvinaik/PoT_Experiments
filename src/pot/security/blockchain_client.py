"""
Blockchain Client Abstract Base Class for PoT Framework

This module provides an abstract base class for blockchain clients that can store
and verify cryptographic hashes on blockchain or distributed ledger systems.
Supports both on-chain storage and local storage backends for verification.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime


class BlockchainClientError(Exception):
    """Base exception for blockchain client operations."""
    pass


class StorageError(BlockchainClientError):
    """Raised when hash storage operations fail."""
    pass


class RetrievalError(BlockchainClientError):
    """Raised when hash retrieval operations fail."""
    pass


class VerificationError(BlockchainClientError):
    """Raised when hash verification operations fail."""
    pass


class BlockchainClient(ABC):
    """
    Abstract base class for blockchain clients supporting hash storage and verification.
    
    This class defines the interface for storing cryptographic hashes with metadata
    on blockchain or distributed storage systems. Implementations can support both
    on-chain storage (actual blockchain transactions) and local storage backends
    for testing and development.
    
    The client is designed to support the Proof-of-Training framework's need for
    tamper-evident storage of verification hashes and metadata.
    """
    
    def __init__(self, client_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the blockchain client.
        
        Args:
            client_id: Unique identifier for this client instance
            config: Optional configuration dictionary for the specific implementation
        """
        self.client_id = client_id
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._initialized = False
        
    @abstractmethod
    def store_hash(self, hash_value: str, metadata: Dict[str, Any]) -> str:
        """
        Store a cryptographic hash with associated metadata.
        
        This method stores a hash value along with metadata on the blockchain
        or storage backend. The hash is typically a SHA256 or similar cryptographic
        hash representing verification data from the PoT framework.
        
        Args:
            hash_value: The cryptographic hash to store (hex string)
            metadata: Dictionary containing associated metadata such as:
                     - timestamp: When the hash was generated
                     - model_id: Identifier of the verified model
                     - challenge_type: Type of verification challenge
                     - confidence: Verification confidence score
                     - any other relevant verification context
        
        Returns:
            transaction_id: Unique identifier for the storage transaction
            
        Raises:
            StorageError: If the hash storage operation fails
            BlockchainClientError: For general client errors
        """
        pass
    
    @abstractmethod
    def retrieve_hash(self, transaction_id: str) -> Dict[str, Any]:
        """
        Retrieve a stored hash and its metadata by transaction ID.
        
        Args:
            transaction_id: The unique transaction identifier returned by store_hash
            
        Returns:
            Dictionary containing:
                - hash_value: The original stored hash
                - metadata: The original metadata dictionary
                - storage_timestamp: When the hash was stored
                - transaction_id: The transaction identifier
                
        Raises:
            RetrievalError: If the hash retrieval operation fails
            BlockchainClientError: For general client errors
        """
        pass
    
    @abstractmethod
    def verify_hash(self, hash_value: str, transaction_id: str) -> bool:
        """
        Verify that a hash value matches the one stored for a given transaction.
        
        This method performs integrity verification by comparing a provided hash
        value with the one stored in the blockchain/storage system for the given
        transaction ID.
        
        Args:
            hash_value: The hash value to verify (hex string)
            transaction_id: The transaction ID to check against
            
        Returns:
            True if the hash matches the stored value, False otherwise
            
        Raises:
            VerificationError: If the verification operation fails
            BlockchainClientError: For general client errors
        """
        pass
    
    def get_client_info(self) -> Dict[str, Any]:
        """
        Get information about this client instance.
        
        Returns:
            Dictionary containing client metadata
        """
        return {
            "client_id": self.client_id,
            "client_type": self.__class__.__name__,
            "initialized": self._initialized,
            "config_keys": list(self.config.keys())
        }
    
    def _validate_hash(self, hash_value: str) -> bool:
        """
        Validate that a hash value is properly formatted.
        
        Args:
            hash_value: The hash to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(hash_value, str):
            return False
        
        # Check if it's a valid hexadecimal string
        try:
            int(hash_value, 16)
            return True
        except ValueError:
            return False
    
    def _log_operation(self, operation: str, **kwargs) -> None:
        """
        Log a blockchain operation with relevant context.
        
        Args:
            operation: The operation being performed
            **kwargs: Additional context to log
        """
        log_data = {
            "client_id": self.client_id,
            "operation": operation,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        self.logger.info(f"Blockchain operation: {operation}", extra=log_data)