"""
Blockchain Client Factory and Configuration Management

This module provides factory functions and configuration management for blockchain
clients in the PoT framework. It handles automatic fallback between Web3 and
local storage clients based on availability and configuration.
"""

import os
import json
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone

# Import blockchain clients
from .blockchain_client import BlockchainClient, BlockchainClientError
from .local_blockchain_client import LocalBlockchainClient

# Try to import Web3 client
try:
    from .web3_blockchain_client import Web3BlockchainClient, Web3ConnectionError
    WEB3_CLIENT_AVAILABLE = True
except ImportError:
    Web3BlockchainClient = None
    Web3ConnectionError = Exception
    WEB3_CLIENT_AVAILABLE = False


@dataclass
class BlockchainConfig:
    """
    Configuration class for blockchain client settings.
    
    Loads settings from environment variables, configuration files, or defaults.
    Provides validation and fallback logic for different deployment scenarios.
    """
    
    # Web3 Configuration
    rpc_url: Optional[str] = None
    private_key: Optional[str] = None
    contract_address: Optional[str] = None
    gas_price_gwei: Optional[str] = None
    confirmation_blocks: int = 2
    retry_attempts: int = 3
    retry_delay: float = 5.0
    
    # Local Configuration
    local_storage_path: str = "./provenance_log.json"
    create_dirs: bool = True
    
    # Factory Configuration
    client_type: str = "auto"  # "auto", "web3", "local"
    client_id: str = "pot_client"
    force_local: bool = False
    cache_client: bool = True
    
    # Logging Configuration
    log_fallbacks: bool = True
    log_level: str = "INFO"
    
    # Additional settings
    extra_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_environment(cls, config_file: Optional[str] = None) -> "BlockchainConfig":
        """
        Create configuration from environment variables and optional config file.
        
        Args:
            config_file: Optional path to JSON configuration file
            
        Returns:
            BlockchainConfig instance with loaded settings
        """
        config = cls()
        
        # Load from config file first (if provided)
        if config_file and Path(config_file).exists():
            config._load_from_file(config_file)
        
        # Override with environment variables
        config._load_from_environment()
        
        # Validate configuration
        config._validate()
        
        return config
    
    def _load_from_file(self, config_file: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            # Update fields that exist in the file
            for key, value in file_config.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    self.extra_config[key] = value
                    
        except Exception as e:
            logging.warning(f"Failed to load config file {config_file}: {str(e)}")
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        env_mapping = {
            "rpc_url": "RPC_URL",
            "private_key": "PRIVATE_KEY", 
            "contract_address": "CONTRACT_ADDRESS",
            "gas_price_gwei": "GAS_PRICE_GWEI",
            "confirmation_blocks": "CONFIRMATION_BLOCKS",
            "retry_attempts": "RETRY_ATTEMPTS",
            "retry_delay": "RETRY_DELAY",
            "local_storage_path": "LOCAL_STORAGE_PATH",
            "client_type": "BLOCKCHAIN_CLIENT_TYPE",
            "client_id": "BLOCKCHAIN_CLIENT_ID",
            "force_local": "FORCE_LOCAL_BLOCKCHAIN",
            "log_level": "BLOCKCHAIN_LOG_LEVEL"
        }
        
        for attr, env_var in env_mapping.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Type conversion
                if attr in ["confirmation_blocks", "retry_attempts"]:
                    setattr(self, attr, int(env_value))
                elif attr == "retry_delay":
                    setattr(self, attr, float(env_value))
                elif attr in ["create_dirs", "force_local", "cache_client", "log_fallbacks"]:
                    setattr(self, attr, env_value.lower() in ("true", "1", "yes"))
                else:
                    setattr(self, attr, env_value)
    
    def _validate(self) -> None:
        """Validate configuration completeness and consistency."""
        # Check for force local override
        if self.force_local:
            self.client_type = "local"
            return
        
        # Check Web3 configuration completeness
        if self.client_type in ["auto", "web3"]:
            web3_complete = all([
                self.rpc_url,
                self.private_key,
                self.contract_address
            ])
            
            if not web3_complete and self.client_type == "web3":
                raise ValueError(
                    "Web3 configuration incomplete. Required: RPC_URL, PRIVATE_KEY, CONTRACT_ADDRESS"
                )
    
    def is_web3_configured(self) -> bool:
        """Check if Web3 configuration is complete."""
        return all([
            self.rpc_url,
            self.private_key,
            self.contract_address,
            not self.force_local
        ])
    
    def get_web3_config(self) -> Dict[str, Any]:
        """Get Web3-specific configuration dictionary."""
        return {
            "rpc_url": self.rpc_url,
            "private_key": self.private_key,
            "contract_address": self.contract_address,
            "gas_price_gwei": self.gas_price_gwei,
            "confirmation_blocks": self.confirmation_blocks,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
            **self.extra_config
        }
    
    def get_local_config(self) -> Dict[str, Any]:
        """Get local client configuration dictionary."""
        return {
            "storage_path": self.local_storage_path,
            "create_dirs": self.create_dirs,
            **self.extra_config
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)."""
        return {
            "rpc_url": self.rpc_url,
            "contract_address": self.contract_address,
            "gas_price_gwei": self.gas_price_gwei,
            "confirmation_blocks": self.confirmation_blocks,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
            "local_storage_path": self.local_storage_path,
            "client_type": self.client_type,
            "client_id": self.client_id,
            "force_local": self.force_local,
            "web3_configured": self.is_web3_configured(),
            "web3_available": WEB3_CLIENT_AVAILABLE
        }


class BlockchainClientFactory:
    """
    Factory class for managing blockchain client instances with caching.
    """
    
    _instance_lock = threading.Lock()
    _cached_clients: Dict[str, BlockchainClient] = {}
    _logger = logging.getLogger(__name__ + ".Factory")
    
    @classmethod
    def get_client(
        cls, 
        config: Optional[BlockchainConfig] = None,
        force_new: bool = False
    ) -> BlockchainClient:
        """
        Get a blockchain client instance with automatic fallback logic.
        
        Args:
            config: Optional configuration object. If None, loads from environment.
            force_new: If True, creates new instance instead of using cached one.
            
        Returns:
            BlockchainClient instance (Web3 or Local)
        """
        if config is None:
            config = BlockchainConfig.from_environment()
        
        cache_key = f"{config.client_id}_{config.client_type}"
        
        # Check cache first (unless forcing new instance)
        if not force_new and config.cache_client and cache_key in cls._cached_clients:
            cls._logger.debug(f"Returning cached client: {cache_key}")
            return cls._cached_clients[cache_key]
        
        with cls._instance_lock:
            # Double-check cache after acquiring lock
            if not force_new and config.cache_client and cache_key in cls._cached_clients:
                return cls._cached_clients[cache_key]
            
            client = cls._create_client(config)
            
            # Cache the client if caching is enabled
            if config.cache_client:
                cls._cached_clients[cache_key] = client
                cls._logger.debug(f"Cached new client: {cache_key}")
            
            return client
    
    @classmethod
    def _create_client(cls, config: BlockchainConfig) -> BlockchainClient:
        """
        Create a new blockchain client based on configuration.
        
        Args:
            config: Configuration object
            
        Returns:
            Appropriate blockchain client instance
        """
        cls._logger.info(f"Creating blockchain client: type={config.client_type}")
        
        # Handle explicit local client request
        if config.client_type == "local" or config.force_local:
            return cls._create_local_client(config)
        
        # Handle explicit Web3 client request
        if config.client_type == "web3":
            if not WEB3_CLIENT_AVAILABLE:
                raise BlockchainClientError(
                    "Web3 client requested but web3 library not available. "
                    "Install with: pip install web3>=6.0.0"
                )
            return cls._create_web3_client(config)
        
        # Handle automatic selection
        if config.client_type == "auto":
            return cls._create_auto_client(config)
        
        # Unknown client type
        raise ValueError(f"Unknown client type: {config.client_type}")
    
    @classmethod
    def _create_auto_client(cls, config: BlockchainConfig) -> BlockchainClient:
        """
        Automatically select the best available client.
        
        Args:
            config: Configuration object
            
        Returns:
            Best available blockchain client
        """
        # Try Web3 client first if available and configured
        if WEB3_CLIENT_AVAILABLE and config.is_web3_configured():
            try:
                client = cls._create_web3_client(config)
                
                # Test connection
                if hasattr(client, 'check_connection') and client.check_connection():
                    cls._logger.info("Successfully created Web3 blockchain client")
                    return client
                else:
                    if config.log_fallbacks:
                        cls._logger.warning(
                            "Web3 client created but connection test failed, falling back to local client"
                        )
                    
            except Exception as e:
                if config.log_fallbacks:
                    cls._logger.warning(
                        f"Failed to create Web3 client ({str(e)}), falling back to local client"
                    )
        
        elif WEB3_CLIENT_AVAILABLE and not config.is_web3_configured():
            if config.log_fallbacks:
                cls._logger.info(
                    "Web3 available but not configured (missing RPC_URL, PRIVATE_KEY, or CONTRACT_ADDRESS), "
                    "using local client"
                )
        
        elif not WEB3_CLIENT_AVAILABLE:
            if config.log_fallbacks:
                cls._logger.info(
                    "Web3 library not available, using local client. "
                    "Install with: pip install web3>=6.0.0"
                )
        
        # Fall back to local client
        return cls._create_local_client(config)
    
    @classmethod
    def _create_web3_client(cls, config: BlockchainConfig) -> BlockchainClient:
        """Create Web3 blockchain client."""
        if not WEB3_CLIENT_AVAILABLE:
            raise BlockchainClientError("Web3 client not available")
        
        web3_config = config.get_web3_config()
        return Web3BlockchainClient(config.client_id, web3_config)
    
    @classmethod
    def _create_local_client(cls, config: BlockchainConfig) -> BlockchainClient:
        """Create local blockchain client."""
        local_config = config.get_local_config()
        return LocalBlockchainClient(config.client_id, local_config)
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached client instances."""
        with cls._instance_lock:
            cls._cached_clients.clear()
            cls._logger.info("Cleared blockchain client cache")
    
    @classmethod
    def get_cached_clients(cls) -> Dict[str, str]:
        """Get information about cached clients."""
        return {
            key: type(client).__name__ 
            for key, client in cls._cached_clients.items()
        }


def get_blockchain_client(
    config_file: Optional[str] = None,
    force_new: bool = False,
    **kwargs
) -> BlockchainClient:
    """
    Factory function to get a blockchain client with automatic fallback.
    
    This is the main entry point for getting blockchain clients in the PoT framework.
    It automatically handles Web3 availability checking, configuration validation,
    and fallback to local storage when needed.
    
    Args:
        config_file: Optional path to JSON configuration file
        force_new: If True, creates new instance instead of using cached one
        **kwargs: Additional configuration parameters to override
        
    Returns:
        BlockchainClient instance (Web3BlockchainClient or LocalBlockchainClient)
        
    Example:
        # Automatic configuration from environment
        client = get_blockchain_client()
        
        # With custom configuration file
        client = get_blockchain_client(config_file="blockchain_config.json")
        
        # Force local client
        client = get_blockchain_client(force_local=True)
    """
    # Load configuration
    config = BlockchainConfig.from_environment(config_file)
    
    # Override with any provided kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            config.extra_config[key] = value
    
    return BlockchainClientFactory.get_client(config, force_new)


def test_blockchain_connection(
    client: Optional[BlockchainClient] = None,
    config_file: Optional[str] = None
) -> Tuple[bool, Dict[str, Any]]:
    """
    Test blockchain connectivity and perform basic operations.
    
    Args:
        client: Optional client to test. If None, creates one using factory.
        config_file: Optional configuration file path
        
    Returns:
        Tuple of (success: bool, results: Dict) containing test results
    """
    logger = logging.getLogger(__name__ + ".ConnectionTest")
    
    if client is None:
        try:
            client = get_blockchain_client(config_file)
        except Exception as e:
            return False, {
                "error": f"Failed to create client: {str(e)}",
                "client_type": "unknown",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    results = {
        "client_type": type(client).__name__,
        "client_id": client.client_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tests": {}
    }
    
    logger.info(f"Testing blockchain connection with {results['client_type']}")
    
    try:
        # Test 1: Check connection (for Web3 clients)
        if hasattr(client, 'check_connection'):
            connection_ok = client.check_connection()
            results["tests"]["connection"] = {
                "success": connection_ok,
                "details": "Connection check successful" if connection_ok else "Connection failed"
            }
            
            if not connection_ok:
                logger.warning("Connection check failed")
                return False, results
        else:
            results["tests"]["connection"] = {
                "success": True,
                "details": "Local client - no connection test needed"
            }
        
        # Test 2: Store hash operation
        test_hash = "0x1234567890abcdef" * 4  # 64 character hex string
        test_metadata = {
            "test": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_id": "test_model"
        }
        
        try:
            tx_id = client.store_hash(test_hash, test_metadata)
            results["tests"]["store_hash"] = {
                "success": True,
                "transaction_id": tx_id,
                "details": "Hash stored successfully"
            }
            logger.info(f"Test hash stored with transaction ID: {tx_id}")
        except Exception as e:
            results["tests"]["store_hash"] = {
                "success": False,
                "error": str(e),
                "details": "Failed to store test hash"
            }
            logger.error(f"Store hash test failed: {str(e)}")
            return False, results
        
        # Test 3: Retrieve hash operation
        try:
            retrieved_data = client.retrieve_hash(tx_id)
            results["tests"]["retrieve_hash"] = {
                "success": True,
                "retrieved_hash": retrieved_data.get("hash_value"),
                "details": "Hash retrieved successfully"
            }
            logger.info("Test hash retrieved successfully")
        except Exception as e:
            results["tests"]["retrieve_hash"] = {
                "success": False,
                "error": str(e),
                "details": "Failed to retrieve test hash"
            }
            logger.error(f"Retrieve hash test failed: {str(e)}")
            return False, results
        
        # Test 4: Verify hash operation
        try:
            is_valid = client.verify_hash(test_hash, tx_id)
            results["tests"]["verify_hash"] = {
                "success": is_valid,
                "details": "Hash verification successful" if is_valid else "Hash verification failed"
            }
            
            if is_valid:
                logger.info("Test hash verification successful")
            else:
                logger.warning("Test hash verification failed")
                return False, results
                
        except Exception as e:
            results["tests"]["verify_hash"] = {
                "success": False,
                "error": str(e),
                "details": "Failed to verify test hash"
            }
            logger.error(f"Verify hash test failed: {str(e)}")
            return False, results
        
        # All tests passed
        results["overall_success"] = True
        logger.info("All blockchain connection tests passed")
        return True, results
        
    except Exception as e:
        results["overall_success"] = False
        results["error"] = str(e)
        logger.error(f"Blockchain connection test failed: {str(e)}")
        return False, results


def get_client_info(client: Optional[BlockchainClient] = None) -> Dict[str, Any]:
    """
    Get detailed information about a blockchain client.
    
    Args:
        client: Client to inspect. If None, creates one using factory.
        
    Returns:
        Dictionary containing client information
    """
    if client is None:
        client = get_blockchain_client()
    
    info = client.get_client_info()
    
    # Add factory-specific information
    info.update({
        "factory_cache": BlockchainClientFactory.get_cached_clients(),
        "web3_available": WEB3_CLIENT_AVAILABLE,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    # Add network info for Web3 clients
    if hasattr(client, 'get_network_info'):
        try:
            info["network_info"] = client.get_network_info()
        except Exception as e:
            info["network_info"] = {"error": str(e)}
    
    return info