"""
Prototype Training Provenance Auditor

Experimental module for logging training events and exploring provenance ideas.
Includes placeholder blockchain and zero-knowledge integrations; not production-ready.
"""

import hashlib
import json
import time
import zlib
import pickle
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
import struct
import hmac
import secrets
import base64
from collections import OrderedDict
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import shared interfaces
from ..core.interfaces import (
    EventType as BaseEventType,
    ProofType as BaseProofType,
    MerkleNode as BaseMerkleNode,
    IMerkleTree,
    IProvenanceAuditor,
    BasicMerkleTree,
    create_merkle_tree,
    TrainingEvent as BaseTrainingEvent
)

# Use the base enums
EventType = BaseEventType
ProofType = BaseProofType


class ChainType(Enum):
    """Supported blockchain networks"""
    ETHEREUM = "ethereum"
    POLYGON = "polygon"
    BSC = "bsc"
    ARBITRUM = "arbitrum"
    OPTIMISM = "optimism"
    LOCAL = "local"


@dataclass
class BlockchainConfig:
    """Configuration for blockchain client connections"""
    chain_type: ChainType
    rpc_url: str
    chain_id: int
    private_key: Optional[str] = None
    account_address: Optional[str] = None
    contract_address: Optional[str] = None
    contract_abi: Optional[List[Dict]] = None
    gas_limit: int = 500000
    gas_price_gwei: Optional[float] = None
    max_fee_per_gas_gwei: Optional[float] = None
    max_priority_fee_per_gas_gwei: Optional[float] = None
    confirmation_blocks: int = 1
    timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    batch_size: int = 100
    
    # Predefined chain configurations
    @classmethod
    def ethereum_mainnet(cls, rpc_url: str, private_key: str = None) -> 'BlockchainConfig':
        return cls(
            chain_type=ChainType.ETHEREUM,
            rpc_url=rpc_url,
            chain_id=1,
            private_key=private_key,
            gas_limit=500000,
            confirmation_blocks=3
        )
    
    @classmethod
    def polygon_mainnet(cls, rpc_url: str, private_key: str = None) -> 'BlockchainConfig':
        return cls(
            chain_type=ChainType.POLYGON,
            rpc_url=rpc_url,
            chain_id=137,
            private_key=private_key,
            gas_limit=500000,
            confirmation_blocks=5
        )
    
    @classmethod
    def local_ganache(cls, rpc_url: str = "http://127.0.0.1:8545") -> 'BlockchainConfig':
        return cls(
            chain_type=ChainType.LOCAL,
            rpc_url=rpc_url,
            chain_id=1337,
            gas_limit=6721975,
            confirmation_blocks=1
        )


@dataclass
class CommitmentRecord:
    """Record for blockchain commitment storage"""
    commitment_hash: str
    metadata: Dict[str, Any]
    tx_hash: str
    block_number: int
    timestamp: str
    gas_used: Optional[int] = None
    confirmations: int = 0


@dataclass
class BatchCommitmentRecord:
    """Record for batch commitment storage using Merkle root"""
    merkle_root: str
    commitment_hashes: List[str]
    tx_hash: str
    block_number: int
    timestamp: str
    gas_used: Optional[int] = None
    proofs: Dict[str, List[Tuple[str, bool]]] = field(default_factory=dict)


class BlockchainError(Exception):
    """Base exception for blockchain operations"""
    pass


class ConnectionError(BlockchainError):
    """Raised when blockchain connection fails"""
    pass


class TransactionError(BlockchainError):
    """Raised when transaction fails"""
    pass


class ContractError(BlockchainError):
    """Raised when smart contract interaction fails"""
    pass


def retry_on_failure(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Decorator for retrying failed blockchain operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time:.2f}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"All {max_attempts} attempts failed. Last error: {e}")
                        
            raise last_exception
        return wrapper
    return decorator


def gas_price_oracle(web3, strategy: str = "fast") -> Dict[str, int]:
    """Get current gas prices from the network"""
    try:
        # Try to get gas price from network
        if hasattr(web3.eth, 'gas_price'):
            base_price = web3.eth.gas_price
        else:
            base_price = web3.eth.get_block('latest')['baseFeePerGas']
        
        # Convert to Gwei for easier handling
        base_gwei = web3.from_wei(base_price, 'gwei')
        
        # Define strategies
        strategies = {
            'slow': {'multiplier': 1.0, 'priority': 1},
            'standard': {'multiplier': 1.2, 'priority': 2},
            'fast': {'multiplier': 1.5, 'priority': 3},
            'fastest': {'multiplier': 2.0, 'priority': 5}
        }
        
        config = strategies.get(strategy, strategies['fast'])
        
        return {
            'gas_price': int(base_gwei * config['multiplier']),
            'max_fee_per_gas': int(base_gwei * config['multiplier']),
            'max_priority_fee_per_gas': config['priority']
        }
    except Exception as e:
        logger.warning(f"Failed to get dynamic gas price: {e}. Using defaults.")
        return {
            'gas_price': 20,  # 20 Gwei default
            'max_fee_per_gas': 20,
            'max_priority_fee_per_gas': 2
        }


@dataclass
class TrainingEvent:
    """Represents a single training event"""
    event_id: str
    event_type: EventType
    epoch: int
    timestamp: datetime
    metrics: Dict[str, Any]
    checkpoint_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    previous_hash: Optional[str] = None
    event_hash: Optional[str] = None
    
    def __post_init__(self):
        if self.event_hash is None:
            self.event_hash = self.calculate_hash()
    
    def calculate_hash(self) -> str:
        """Calculate hash of the event"""
        event_data = {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'epoch': self.epoch,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics,
            'checkpoint_hash': self.checkpoint_hash,
            'metadata': self.metadata,
            'previous_hash': self.previous_hash
        }
        
        event_bytes = json.dumps(event_data, sort_keys=True).encode()
        return hashlib.sha256(event_bytes).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'epoch': self.epoch,
            'timestamp': self.timestamp.isoformat(),
            'metrics': self.metrics,
            'checkpoint_hash': self.checkpoint_hash,
            'metadata': self.metadata,
            'previous_hash': self.previous_hash,
            'event_hash': self.event_hash
        }


# Use the base MerkleNode and extend it with additional functionality
class MerkleNode(BaseMerkleNode):
    """Extended Merkle node with SHA256 hashing for provenance tracking."""
    
    def __init__(self, data: Optional[bytes] = None, 
                 left: Optional['MerkleNode'] = None,
                 right: Optional['MerkleNode'] = None):
        """
        Initialize a Merkle tree node.
        
        Args:
            data: Raw data for leaf nodes (None for internal nodes)
            left: Left child node (None for leaf nodes)
            right: Right child node (None for leaf nodes)
        """
        # Compute hash first
        hash_str = self._compute_hash(data, left, right)
        # Initialize base class with hash string
        super().__init__(hash=hash_str, data=data, left=left, right=right)
        # Store hash as bytes for backward compatibility
        self.hash = bytes.fromhex(hash_str) if hash_str else b''
    
    def _compute_hash(self, data: Optional[bytes], 
                     left: Optional['MerkleNode'], 
                     right: Optional['MerkleNode']) -> str:
        """
        Compute SHA256 hash for this node.
        
        For leaf nodes: hash the data directly
        For internal nodes: hash the concatenation of child hashes
        
        Returns:
            SHA256 hash as hex string
        """
        if data is not None:
            # Leaf node: hash the data
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif isinstance(data, bytes):
                data_bytes = data
            else:
                data_bytes = str(data).encode('utf-8')
            return hashlib.sha256(data_bytes).hexdigest()
        elif left and right:
            # Internal node: hash concatenated child hashes
            left_hash = left.hash if isinstance(left.hash, bytes) else bytes.fromhex(left.hash)
            right_hash = right.hash if isinstance(right.hash, bytes) else bytes.fromhex(right.hash)
            return hashlib.sha256(left_hash + right_hash).hexdigest()
        else:
            # Empty node
            return b''
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.left is None and self.right is None
    
    def get_hex_hash(self) -> str:
        """Get hexadecimal representation of hash."""
        return self.hash.hex()


def build_merkle_tree(data_blocks: List[bytes], hash_func=None) -> MerkleNode:
    """
    Build a complete Merkle tree from leaf data blocks.
    
    Constructs a binary tree where each leaf contains a data block
    and each internal node contains the hash of its children.
    Handles odd number of nodes by duplicating the last node.
    
    Args:
        data_blocks: List of data blocks as bytes
        
    Returns:
        Root node of the constructed Merkle tree
        
    Raises:
        ValueError: If data_blocks is empty
    """
    if not data_blocks:
        raise ValueError("Cannot build Merkle tree from empty data blocks")
    
    # Special case: single block tree
    if len(data_blocks) == 1:
        return MerkleNode(data=data_blocks[0])
    
    # Create leaf nodes - pad to make even number if needed
    leaf_nodes = [MerkleNode(data=block) for block in data_blocks]
    
    # If odd number of blocks, duplicate the last block
    if len(leaf_nodes) % 2 == 1:
        leaf_nodes.append(MerkleNode(data=data_blocks[-1]))
    
    current_level = leaf_nodes
    
    # Build tree bottom-up
    while len(current_level) > 1:
        next_level = []
        
        # If odd number of nodes, duplicate the last one
        if len(current_level) % 2 == 1:
            current_level.append(current_level[-1])
        
        # Process pairs of nodes
        for i in range(0, len(current_level), 2):
            left = current_level[i]
            right = current_level[i + 1]  # Should always exist after padding
            
            # Create parent node
            parent = MerkleNode(left=left, right=right)
            next_level.append(parent)
        
        current_level = next_level
    
    return current_level[0]


def compute_merkle_root(data_blocks: List[bytes]) -> bytes:
    """
    Compute the Merkle root hash for a list of data blocks.
    
    Args:
        data_blocks: List of data blocks as bytes
        
    Returns:
        Root hash as bytes
        
    Raises:
        ValueError: If data_blocks is empty
    """
    if not data_blocks:
        raise ValueError("Cannot compute Merkle root from empty data blocks")
    
    tree = build_merkle_tree(data_blocks)
    return tree.hash


def generate_merkle_proof(tree: MerkleNode, index: int) -> List[Tuple[bytes, bool]]:
    """
    Generate a Merkle proof for a leaf at the given index.
    
    The proof consists of sibling hashes along the path from leaf to root,
    with boolean indicators for whether each sibling is on the left (False) 
    or right (True) side.
    
    Args:
        tree: Root node of the Merkle tree
        index: Index of the leaf to prove (0-based)
        
    Returns:
        List of (sibling_hash, is_right) tuples representing the proof path
        
    Raises:
        ValueError: If index is invalid or tree structure is incorrect
    """
    # For single leaf tree, no proof needed
    if tree.is_leaf():
        if index != 0:
            raise ValueError(f"Invalid leaf index {index}, tree has 1 leaf")
        return []
    
    # Collect all leaf nodes in order
    leaves = _collect_leaves(tree)
    
    # Check if index is valid
    if index < 0 or index >= len(leaves):
        raise ValueError(f"Invalid leaf index {index}, tree has {len(leaves)} leaves")
    
    proof = []
    current_index = index
    current_node = tree
    
    # Traverse from root to leaf, collecting sibling hashes
    while not current_node.is_leaf():
        # Calculate the size of left subtree
        left_subtree_size = _get_subtree_size(current_node.left)
        
        if current_index < left_subtree_size:
            # Target is in left subtree
            sibling_hash = current_node.right.hash
            is_right = True  # Sibling is on the right
            current_node = current_node.left
        else:
            # Target is in right subtree
            sibling_hash = current_node.left.hash
            is_right = False  # Sibling is on the left
            current_node = current_node.right
            current_index -= left_subtree_size
        
        proof.append((sibling_hash, is_right))
    
    # Reverse the proof to go from leaf to root
    proof.reverse()
    
    return proof


def verify_merkle_proof(leaf_hash: bytes, proof: List[Tuple[bytes, bool]], root_hash: bytes) -> bool:
    """
    Verify that a leaf hash is part of a Merkle tree with the given root.
    
    Reconstructs the path from leaf to root using the proof and checks
    if the computed root matches the expected root hash.
    
    Args:
        leaf_hash: Hash of the leaf to verify
        proof: List of (sibling_hash, is_right) tuples from generate_merkle_proof
        root_hash: Expected root hash of the tree
        
    Returns:
        True if the proof is valid, False otherwise
    """
    current_hash = leaf_hash
    
    # Traverse the proof path, computing hashes along the way
    for sibling_hash, is_right in proof:
        if is_right:
            # Sibling is on the right, so current hash goes on the left
            combined = current_hash + sibling_hash
        else:
            # Sibling is on the left, so current hash goes on the right
            combined = sibling_hash + current_hash
        
        # Compute hash of combined data
        current_hash = hashlib.sha256(combined).digest()
    
    # Check if computed root matches expected root
    return current_hash == root_hash


def _collect_leaves(node: MerkleNode) -> List[MerkleNode]:
    """
    Collect all leaf nodes from a Merkle tree in left-to-right order.
    
    Args:
        node: Root node of the tree
        
    Returns:
        List of leaf nodes in order
    """
    if node.is_leaf():
        return [node]
    
    leaves = []
    if node.left:
        leaves.extend(_collect_leaves(node.left))
    if node.right:
        leaves.extend(_collect_leaves(node.right))
    
    return leaves


def _get_subtree_size(node: MerkleNode) -> int:
    """
    Get the number of leaf nodes in a subtree.
    
    Args:
        node: Root of the subtree
        
    Returns:
        Number of leaf nodes
    """
    if node.is_leaf():
        return 1
    
    size = 0
    if node.left:
        size += _get_subtree_size(node.left)
    if node.right:
        size += _get_subtree_size(node.right)
    
    return size


class MerkleTree:
    """Merkle tree for efficient verification of training events"""
    
    def __init__(self, events: List[TrainingEvent]):
        self.events = events
        self.leaves = []
        self.root = self._build_tree()
    
    def _build_tree(self) -> Optional[MerkleNode]:
        """Build Merkle tree from events"""
        if not self.events:
            return None
        
        # Create leaf nodes
        self.leaves = [
            MerkleNode(data=event.event_hash)
            for event in self.events
        ]
        
        # Build tree bottom-up
        current_level = self.leaves.copy()
        
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    # Duplicate last node if odd number
                    right = left
                
                # Create parent node with left and right children
                parent = MerkleNode(left=left, right=right)
                next_level.append(parent)
            
            current_level = next_level
        
        return current_level[0] if current_level else None
    
    def get_root_hash(self) -> Optional[str]:
        """Get root hash of the tree"""
        return self.root.hash.hex() if self.root else None
    
    def get_proof(self, event_index: int) -> List[Tuple[str, str]]:
        """Get Merkle proof for an event"""
        if event_index >= len(self.events):
            raise ValueError(f"Invalid event index: {event_index}")
        
        proof = []
        current_index = event_index
        current_level = self.leaves.copy()
        
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                if i == current_index or i + 1 == current_index:
                    # This pair contains our target
                    sibling_index = i + 1 if i == current_index else i

                    if sibling_index < len(current_level):
                        sibling = current_level[sibling_index]
                    else:
                        # Duplicate the node if no sibling (odd count)
                        sibling = current_level[i]
                    position = 'right' if i == current_index else 'left'
                    proof.append((sibling.hash.hex(), position))

                    # Update index for next level
                    current_index = i // 2
                
                # Build next level
                if i + 1 < len(current_level):
                    left_child = current_level[i]
                    right_child = current_level[i + 1]
                else:
                    left_child = current_level[i]
                    right_child = current_level[i]  # Duplicate for odd number of nodes
                
                parent_node = MerkleNode(left=left_child, right=right_child)
                next_level.append(parent_node)
            
            current_level = next_level
        
        return proof
    
    def verify_proof(self, event_hash: str, proof: List[Tuple[str, str]], 
                    root_hash: str) -> bool:
        """Verify a Merkle proof"""
        # The event_hash is a hex string, but we need to start with the 
        # same hash that the leaf node has, which is the hash of the hex string
        current_hash_bytes = hashlib.sha256(event_hash.encode()).digest()
        
        for sibling_hash, position in proof:
            sibling_hash_bytes = bytes.fromhex(sibling_hash)
            
            if position == 'right':
                combined_bytes = current_hash_bytes + sibling_hash_bytes
            else:
                combined_bytes = sibling_hash_bytes + current_hash_bytes
            
            current_hash_bytes = hashlib.sha256(combined_bytes).digest()
        
        # Convert back to hex for comparison
        current_hash_hex = current_hash_bytes.hex()
        return current_hash_hex == root_hash


class ZeroKnowledgeProof:
    """Zero-knowledge proof generator for training progression"""
    
    def __init__(self, secret_key: bytes = None):
        self.secret_key = secret_key or secrets.token_bytes(32)
    
    def generate_commitment(self, data: Any) -> Tuple[str, str]:
        """Generate commitment for data"""
        # Serialize data
        data_bytes = pickle.dumps(data)
        
        # Generate random nonce
        nonce = secrets.token_bytes(32)
        
        # Create commitment
        commitment_data = data_bytes + nonce
        commitment = hashlib.sha256(commitment_data).hexdigest()
        
        # Store nonce for later revelation
        nonce_hex = nonce.hex()
        
        return commitment, nonce_hex
    
    def generate_progression_proof(self, start_state: Dict, end_state: Dict,
                                  transitions: List[Dict]) -> Dict[str, Any]:
        """Generate ZK proof of training progression"""
        # Create commitments for states
        start_commitment, start_nonce = self.generate_commitment(start_state)
        end_commitment, end_nonce = self.generate_commitment(end_state)
        
        # Create transition commitments
        transition_commitments = []
        for transition in transitions:
            comm, nonce = self.generate_commitment(transition)
            transition_commitments.append({
                'commitment': comm,
                'epoch': transition.get('epoch', -1)
            })
        
        # Generate proof statement
        proof_statement = {
            'start_commitment': start_commitment,
            'end_commitment': end_commitment,
            'num_transitions': len(transitions),
            'transition_commitments': transition_commitments,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Sign the proof
        proof_bytes = json.dumps(proof_statement, sort_keys=True).encode()
        signature = hmac.new(self.secret_key, proof_bytes, hashlib.sha256).hexdigest()
        
        return {
            'proof': proof_statement,
            'signature': signature,
            'verification_data': {
                'start_nonce': start_nonce,
                'end_nonce': end_nonce
            }
        }
    
    def verify_progression_proof(self, proof_data: Dict, secret_key: bytes) -> bool:
        """Verify a progression proof"""
        proof = proof_data['proof']
        signature = proof_data['signature']
        
        # Verify signature
        proof_bytes = json.dumps(proof, sort_keys=True).encode()
        expected_signature = hmac.new(secret_key, proof_bytes, hashlib.sha256).hexdigest()
        
        return signature == expected_signature


class BlockchainClient:
    """Advanced blockchain client for on-chain commitment storage with multi-chain support"""

    def __init__(self, config: BlockchainConfig):
        """
        Initialize blockchain client with comprehensive configuration.
        
        Args:
            config: BlockchainConfig containing connection and transaction parameters
        """
        self.config = config
        self.web3 = None
        self.contract = None
        self.account = None
        self._lock = threading.Lock()
        self._connection_established = False
        
        # Default smart contract ABI for commitment storage
        self._default_abi = [
            {
                "inputs": [
                    {"name": "commitmentHash", "type": "bytes32"},
                    {"name": "metadata", "type": "string"}
                ],
                "name": "storeCommitment",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "merkleRoot", "type": "bytes32"},
                    {"name": "commitmentCount", "type": "uint256"}
                ],
                "name": "storeBatchCommitments",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "commitmentHash", "type": "bytes32"}
                ],
                "name": "getCommitment",
                "outputs": [
                    {"name": "exists", "type": "bool"},
                    {"name": "blockNumber", "type": "uint256"},
                    {"name": "timestamp", "type": "uint256"},
                    {"name": "metadata", "type": "string"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"name": "commitmentHash", "type": "bytes32"}
                ],
                "name": "verifyCommitment",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        logger.info(f"Initialized BlockchainClient for {config.chain_type.value} (Chain ID: {config.chain_id})")

    @retry_on_failure(max_attempts=3, delay=2.0)
    def connect(self) -> bool:
        """
        Establish blockchain connection with retry logic.
        
        Returns:
            True if connection successful, False otherwise
            
        Raises:
            ConnectionError: If unable to connect after retries
        """
        try:
            from web3 import Web3
            from eth_account import Account
            
            # Create Web3 instance with appropriate provider
            if self.config.rpc_url.startswith('ws'):
                from web3.providers import WebsocketProvider
                provider = WebsocketProvider(self.config.rpc_url)
            else:
                from web3.providers import HTTPProvider
                provider = HTTPProvider(self.config.rpc_url)
            
            self.web3 = Web3(provider)
            
            # Test connection
            if not self.web3.is_connected():
                raise ConnectionError(f"Failed to connect to {self.config.rpc_url}")
            
            # Verify chain ID
            actual_chain_id = self.web3.eth.chain_id
            if actual_chain_id != self.config.chain_id:
                logger.warning(f"Chain ID mismatch: expected {self.config.chain_id}, got {actual_chain_id}")
            
            # Setup account
            if self.config.private_key:
                self.account = Account.from_key(self.config.private_key)
                self.web3.eth.default_account = self.account.address
                logger.info(f"Using account: {self.account.address}")
            elif self.config.account_address:
                self.web3.eth.default_account = self.config.account_address
                logger.info(f"Using provided account: {self.config.account_address}")
            else:
                # Try to use first available account (for local development)
                accounts = self.web3.eth.accounts
                if accounts:
                    self.web3.eth.default_account = accounts[0]
                    logger.info(f"Using first available account: {accounts[0]}")
                else:
                    raise ConnectionError("No account available for transactions")
            
            # Setup smart contract if configured
            if self.config.contract_address:
                contract_abi = self.config.contract_abi or self._default_abi
                self.contract = self.web3.eth.contract(
                    address=self.config.contract_address,
                    abi=contract_abi
                )
                logger.info(f"Connected to contract at {self.config.contract_address}")
            
            self._connection_established = True
            logger.info(f"Successfully connected to {self.config.chain_type.value} blockchain")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to blockchain: {e}")
            raise ConnectionError(f"Blockchain connection failed: {e}")

    def _ensure_connected(self):
        """Ensure blockchain connection is established"""
        if not self._connection_established:
            self.connect()

    def _get_gas_params(self) -> Dict[str, Any]:
        """Get optimized gas parameters for transactions"""
        self._ensure_connected()
        
        gas_params = {'gas': self.config.gas_limit}
        
        try:
            # Use configured gas prices if available
            if self.config.gas_price_gwei:
                gas_params['gasPrice'] = self.web3.to_wei(self.config.gas_price_gwei, 'gwei')
            elif self.config.max_fee_per_gas_gwei and self.config.max_priority_fee_per_gas_gwei:
                # EIP-1559 transaction
                gas_params['maxFeePerGas'] = self.web3.to_wei(self.config.max_fee_per_gas_gwei, 'gwei')
                gas_params['maxPriorityFeePerGas'] = self.web3.to_wei(self.config.max_priority_fee_per_gas_gwei, 'gwei')
            else:
                # Use dynamic gas pricing
                gas_info = gas_price_oracle(self.web3)
                if 'maxFeePerGas' in gas_info and self.web3.eth.chain_id != 1337:  # Skip EIP-1559 for local
                    gas_params['maxFeePerGas'] = self.web3.to_wei(gas_info['max_fee_per_gas'], 'gwei')
                    gas_params['maxPriorityFeePerGas'] = self.web3.to_wei(gas_info['max_priority_fee_per_gas'], 'gwei')
                else:
                    gas_params['gasPrice'] = self.web3.to_wei(gas_info['gas_price'], 'gwei')
                    
        except Exception as e:
            logger.warning(f"Failed to get dynamic gas pricing: {e}. Using default.")
            gas_params['gasPrice'] = self.web3.to_wei(20, 'gwei')  # 20 Gwei fallback
        
        return gas_params

    @retry_on_failure(max_attempts=3, delay=1.0)
    def store_commitment(self, commitment_hash: bytes, metadata: Dict[str, Any]) -> str:
        """
        Store commitment on blockchain.
        
        Args:
            commitment_hash: 32-byte commitment hash
            metadata: Additional metadata to store
            
        Returns:
            Transaction hash
            
        Raises:
            TransactionError: If transaction fails
        """
        self._ensure_connected()
        
        try:
            commitment_hex = commitment_hash.hex() if isinstance(commitment_hash, bytes) else commitment_hash
            metadata_json = json.dumps(metadata, sort_keys=True)
            
            if self.contract:
                # Use smart contract
                return self._store_commitment_contract(commitment_hash, metadata_json)
            else:
                # Store in transaction data
                return self._store_commitment_data(commitment_hex, metadata)
                
        except Exception as e:
            logger.error(f"Failed to store commitment: {e}")
            raise TransactionError(f"Commitment storage failed: {e}")

    def _store_commitment_contract(self, commitment_hash: bytes, metadata_json: str) -> str:
        """Store commitment using smart contract"""
        try:
            # Build transaction
            gas_params = self._get_gas_params()
            
            # Get nonce
            nonce = self.web3.eth.get_transaction_count(self.web3.eth.default_account)
            
            # Build contract function call
            function_call = self.contract.functions.storeCommitment(
                commitment_hash,
                metadata_json
            )
            
            # Build transaction
            transaction = function_call.build_transaction({
                'from': self.web3.eth.default_account,
                'nonce': nonce,
                **gas_params
            })
            
            # Sign and send transaction
            if self.account:
                signed_txn = self.web3.eth.account.sign_transaction(transaction, self.config.private_key)
                tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            else:
                tx_hash = self.web3.eth.send_transaction(transaction)
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(
                tx_hash, 
                timeout=self.config.timeout_seconds
            )
            
            logger.info(f"Commitment stored via contract: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            raise TransactionError(f"Contract transaction failed: {e}")

    def _store_commitment_data(self, commitment_hex: str, metadata: Dict[str, Any]) -> str:
        """Store commitment in transaction data field"""
        try:
            # Prepare payload
            payload = {
                "commitment": commitment_hex,
                "metadata": metadata,
                "timestamp": datetime.utcnow().isoformat() + 'Z',
                "version": "1.0"
            }
            
            payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
            
            # Build transaction
            gas_params = self._get_gas_params()
            nonce = self.web3.eth.get_transaction_count(self.web3.eth.default_account)
            
            transaction = {
                'from': self.web3.eth.default_account,
                'to': self.web3.eth.default_account,  # Self-transaction
                'value': 0,
                'data': payload_bytes,
                'nonce': nonce,
                **gas_params
            }
            
            # Sign and send
            if self.account:
                signed_txn = self.web3.eth.account.sign_transaction(transaction, self.config.private_key)
                tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            else:
                tx_hash = self.web3.eth.send_transaction(transaction)
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(
                tx_hash,
                timeout=self.config.timeout_seconds
            )
            
            logger.info(f"Commitment stored in transaction data: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            raise TransactionError(f"Data transaction failed: {e}")

    @retry_on_failure(max_attempts=2, delay=0.5)
    def retrieve_commitment(self, tx_hash: str) -> Optional[CommitmentRecord]:
        """
        Retrieve commitment from blockchain.
        
        Args:
            tx_hash: Transaction hash containing the commitment
            
        Returns:
            CommitmentRecord if found, None otherwise
        """
        self._ensure_connected()
        
        try:
            # Get transaction and receipt
            tx = self.web3.eth.get_transaction(tx_hash)
            receipt = self.web3.eth.get_transaction_receipt(tx_hash)
            block = self.web3.eth.get_block(tx['blockNumber'])
            
            if self.contract and tx['to'] == self.config.contract_address:
                # Retrieve from contract logs
                return self._retrieve_from_contract_logs(receipt, block)
            else:
                # Retrieve from transaction data
                return self._retrieve_from_transaction_data(tx, receipt, block)
                
        except Exception as e:
            logger.error(f"Failed to retrieve commitment from {tx_hash}: {e}")
            return None

    def _retrieve_from_contract_logs(self, receipt, block) -> Optional[CommitmentRecord]:
        """Retrieve commitment from smart contract logs"""
        try:
            # Parse contract logs for commitment events
            for log in receipt['logs']:
                # This would need to be adapted based on actual contract events
                pass
            
            # For now, return basic info
            return CommitmentRecord(
                commitment_hash="",  # Would extract from logs
                metadata={},
                tx_hash=receipt['transactionHash'].hex(),
                block_number=receipt['blockNumber'],
                timestamp=datetime.fromtimestamp(block['timestamp'], timezone.utc).isoformat(),
                gas_used=receipt['gasUsed'],
                confirmations=self.web3.eth.block_number - receipt['blockNumber']
            )
            
        except Exception as e:
            logger.error(f"Failed to parse contract logs: {e}")
            return None

    def _retrieve_from_transaction_data(self, tx, receipt, block) -> Optional[CommitmentRecord]:
        """Retrieve commitment from transaction data field"""
        try:
            # Parse transaction data
            data_bytes = bytes(tx['input'])
            if not data_bytes:
                return None
                
            payload = json.loads(data_bytes.decode('utf-8'))
            
            return CommitmentRecord(
                commitment_hash=payload.get('commitment', ''),
                metadata=payload.get('metadata', {}),
                tx_hash=receipt['transactionHash'].hex(),
                block_number=receipt['blockNumber'],
                timestamp=datetime.fromtimestamp(block['timestamp'], timezone.utc).isoformat(),
                gas_used=receipt['gasUsed'],
                confirmations=self.web3.eth.block_number - receipt['blockNumber']
            )
            
        except Exception as e:
            logger.error(f"Failed to parse transaction data: {e}")
            return None

    @retry_on_failure(max_attempts=2, delay=0.5)
    def verify_commitment_onchain(self, commitment_hash: bytes) -> bool:
        """
        Verify commitment exists on blockchain.
        
        Args:
            commitment_hash: Commitment hash to verify
            
        Returns:
            True if commitment exists and is valid
        """
        self._ensure_connected()
        
        try:
            if self.contract:
                # Use smart contract verification
                result = self.contract.functions.verifyCommitment(commitment_hash).call()
                return bool(result)
            else:
                # For data-based storage, we'd need to search transaction history
                # This is not efficient for production use
                logger.warning("On-chain verification without smart contract is not efficient")
                return False
                
        except Exception as e:
            logger.error(f"Failed to verify commitment on-chain: {e}")
            return False

    @retry_on_failure(max_attempts=3, delay=2.0)
    def batch_store_commitments(self, commitments: List[bytes]) -> str:
        """
        Store multiple commitments efficiently using Merkle root.
        
        Args:
            commitments: List of commitment hashes to store
            
        Returns:
            Transaction hash for the batch operation
            
        Raises:
            TransactionError: If batch operation fails
        """
        self._ensure_connected()
        
        if not commitments:
            raise ValueError("Cannot store empty commitment list")
        
        try:
            # Build Merkle tree from commitments
            merkle_root_hash = compute_merkle_root(commitments)
            
            # Prepare metadata
            metadata = {
                "batch_size": len(commitments),
                "commitment_hashes": [c.hex() for c in commitments],
                "merkle_root": merkle_root_hash.hex(),
                "batch_timestamp": datetime.utcnow().isoformat() + 'Z'
            }
            
            if self.contract:
                # Use smart contract batch function
                return self._batch_store_contract(merkle_root_hash, len(commitments), metadata)
            else:
                # Store batch info in transaction data
                return self._batch_store_data(merkle_root_hash, metadata)
                
        except Exception as e:
            logger.error(f"Failed to batch store commitments: {e}")
            raise TransactionError(f"Batch commitment storage failed: {e}")

    def _batch_store_contract(self, merkle_root: bytes, count: int, metadata: Dict) -> str:
        """Store batch commitments using smart contract"""
        try:
            gas_params = self._get_gas_params()
            nonce = self.web3.eth.get_transaction_count(self.web3.eth.default_account)
            
            # Build contract function call
            function_call = self.contract.functions.storeBatchCommitments(
                merkle_root,
                count
            )
            
            transaction = function_call.build_transaction({
                'from': self.web3.eth.default_account,
                'nonce': nonce,
                **gas_params
            })
            
            # Sign and send
            if self.account:
                signed_txn = self.web3.eth.account.sign_transaction(transaction, self.config.private_key)
                tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            else:
                tx_hash = self.web3.eth.send_transaction(transaction)
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(
                tx_hash,
                timeout=self.config.timeout_seconds
            )
            
            logger.info(f"Batch commitments stored via contract: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            raise TransactionError(f"Batch contract transaction failed: {e}")

    def _batch_store_data(self, merkle_root: bytes, metadata: Dict) -> str:
        """Store batch commitments in transaction data"""
        try:
            payload = {
                "type": "batch_commitments",
                "merkle_root": merkle_root.hex(),
                "metadata": metadata,
                "version": "1.0"
            }
            
            payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
            
            gas_params = self._get_gas_params()
            nonce = self.web3.eth.get_transaction_count(self.web3.eth.default_account)
            
            transaction = {
                'from': self.web3.eth.default_account,
                'to': self.web3.eth.default_account,
                'value': 0,
                'data': payload_bytes,
                'nonce': nonce,
                **gas_params
            }
            
            # Sign and send
            if self.account:
                signed_txn = self.web3.eth.account.sign_transaction(transaction, self.config.private_key)
                tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            else:
                tx_hash = self.web3.eth.send_transaction(transaction)
            
            # Wait for confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(
                tx_hash,
                timeout=self.config.timeout_seconds
            )
            
            logger.info(f"Batch commitments stored in transaction data: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            raise TransactionError(f"Batch data transaction failed: {e}")

    def get_balance(self) -> float:
        """Get current account balance in ETH"""
        self._ensure_connected()
        
        try:
            balance_wei = self.web3.eth.get_balance(self.web3.eth.default_account)
            return self.web3.from_wei(balance_wei, 'ether')
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
            return 0.0

    def estimate_gas_cost(self, operation: str = "store_commitment") -> Dict[str, float]:
        """
        Estimate gas cost for different operations.
        
        Args:
            operation: Type of operation to estimate
            
        Returns:
            Dictionary with gas estimates in ETH and USD (if available)
        """
        self._ensure_connected()
        
        try:
            gas_params = self._get_gas_params()
            
            # Estimate gas usage based on operation
            gas_estimates = {
                "store_commitment": 100000,
                "batch_store_commitments": 150000,
                "verify_commitment": 50000
            }
            
            gas_needed = gas_estimates.get(operation, 100000)
            
            # Get gas price
            if 'gasPrice' in gas_params:
                gas_price = gas_params['gasPrice']
            else:
                gas_price = gas_params.get('maxFeePerGas', self.web3.to_wei(20, 'gwei'))
            
            cost_wei = gas_needed * gas_price
            cost_eth = self.web3.from_wei(cost_wei, 'ether')
            
            return {
                "gas_needed": gas_needed,
                "gas_price_gwei": self.web3.from_wei(gas_price, 'gwei'),
                "cost_eth": float(cost_eth),
                "cost_wei": cost_wei
            }
            
        except Exception as e:
            logger.error(f"Failed to estimate gas cost: {e}")
            return {"error": str(e)}

    def disconnect(self):
        """Clean up blockchain connection"""
        if self.web3 and hasattr(self.web3.provider, 'disconnect'):
            try:
                self.web3.provider.disconnect()
            except Exception as e:
                logger.warning(f"Error disconnecting: {e}")
        
        self._connection_established = False
        logger.info("Blockchain client disconnected")

    def __enter__(self):
        """Context manager entry"""
        if not self._connection_established:
            self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()


class MockBlockchainClient:
    """Mock blockchain client for testing and development"""
    
    def __init__(self, config: Optional[BlockchainConfig] = None):
        """Initialize mock client with optional config for compatibility"""
        self.config = config or BlockchainConfig.local_ganache()
        self.storage = {}
        self.batch_storage = {}
        self.transaction_counter = 0
        self.block_number = 1000000  # Start at reasonable block number
        self._connection_established = True
        
        logger.info("Initialized MockBlockchainClient for testing")
    
    def connect(self) -> bool:
        """Mock connection always succeeds"""
        self._connection_established = True
        logger.info("Mock blockchain client connected")
        return True
    
    def store_data(self, data_type: str, data: Dict[str, Any], metadata: Dict[str, Any] = None) -> str:
        """Store arbitrary data in mock storage"""
        tx_id = f"0x{self.transaction_counter:064x}"
        self.transaction_counter += 1
        self.block_number += 1
        
        self.storage[tx_id] = {
            'type': data_type,
            'data': data,
            'metadata': metadata or {},
            'block': self.block_number,
            'timestamp': time.time()
        }
        
        return tx_id
    
    def store_commitment(self, commitment_hash: bytes, metadata: Dict[str, Any]) -> str:
        """Store commitment in mock storage"""
        tx_id = f"0x{self.transaction_counter:064x}"
        self.transaction_counter += 1
        self.block_number += 1
        
        commitment_hex = commitment_hash.hex() if isinstance(commitment_hash, bytes) else commitment_hash
        
        record = CommitmentRecord(
            commitment_hash=commitment_hex,
            metadata=metadata,
            tx_hash=tx_id,
            block_number=self.block_number,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            gas_used=75000,  # Mock gas usage
            confirmations=1
        )
        
        self.storage[tx_id] = record
        logger.info(f"Stored commitment in mock blockchain: {tx_id}")
        return tx_id
    
    def retrieve_commitment(self, tx_hash: str) -> Optional[CommitmentRecord]:
        """Retrieve commitment from mock storage"""
        return self.storage.get(tx_hash)
    
    def verify_commitment_onchain(self, commitment_hash: bytes) -> bool:
        """Verify commitment exists in mock storage"""
        commitment_hex = commitment_hash.hex() if isinstance(commitment_hash, bytes) else commitment_hash
        
        for record in self.storage.values():
            if record.commitment_hash == commitment_hex:
                return True
        return False
    
    def batch_store_commitments(self, commitments: List[bytes]) -> str:
        """Store batch commitments using Merkle root in mock storage"""
        if not commitments:
            raise ValueError("Cannot store empty commitment list")
        
        tx_id = f"0x{self.transaction_counter:064x}"
        self.transaction_counter += 1
        self.block_number += 1
        
        # Build Merkle tree
        merkle_root_hash = compute_merkle_root(commitments)
        
        # Create batch record
        batch_record = BatchCommitmentRecord(
            merkle_root=merkle_root_hash.hex(),
            commitment_hashes=[c.hex() for c in commitments],
            tx_hash=tx_id,
            block_number=self.block_number,
            timestamp=datetime.utcnow().isoformat() + 'Z',
            gas_used=50000 + len(commitments) * 5000  # Mock gas scaling
        )
        
        # Generate Merkle proofs for each commitment
        tree = build_merkle_tree(commitments)
        for i, commitment in enumerate(commitments):
            proof = generate_merkle_proof(tree, i)
            batch_record.proofs[commitment.hex()] = proof
        
        self.batch_storage[tx_id] = batch_record
        logger.info(f"Stored batch commitments in mock blockchain: {tx_id} (count: {len(commitments)})")
        return tx_id
    
    def get_batch_commitment(self, tx_hash: str) -> Optional[BatchCommitmentRecord]:
        """Retrieve batch commitment record"""
        return self.batch_storage.get(tx_hash)
    
    def get_balance(self) -> float:
        """Return mock balance"""
        return 10.0  # 10 ETH for testing
    
    def estimate_gas_cost(self, operation: str = "store_commitment") -> Dict[str, float]:
        """Return mock gas estimates"""
        estimates = {
            "store_commitment": {"gas_needed": 75000, "cost_eth": 0.002},
            "batch_store_commitments": {"gas_needed": 150000, "cost_eth": 0.004},
            "verify_commitment": {"gas_needed": 25000, "cost_eth": 0.001}
        }
        
        return estimates.get(operation, estimates["store_commitment"])
    
    def disconnect(self):
        """Mock disconnection"""
        self._connection_established = False
        logger.info("Mock blockchain client disconnected")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()
    
    # Legacy compatibility methods
    def store_hash(self, hash_value: str, metadata: Dict) -> str:
        """Legacy method for compatibility"""
        if isinstance(hash_value, str):
            hash_bytes = bytes.fromhex(hash_value)
        else:
            hash_bytes = hash_value
        return self.store_commitment(hash_bytes, metadata)
    
    def retrieve_hash(self, transaction_id: str) -> Optional[Dict]:
        """Legacy method for compatibility"""
        record = self.retrieve_commitment(transaction_id)
        if record:
            return {
                "hash": record.commitment_hash,
                "metadata": record.metadata,
                "tx_hash": record.tx_hash,
                "block_number": record.block_number,
                "timestamp": record.timestamp
            }
        return None
    
    def verify_hash(self, hash_value: str, transaction_id: str) -> bool:
        """Legacy method for compatibility"""
        stored = self.retrieve_hash(transaction_id)
        return stored and stored['hash'] == hash_value


class TrainingProvenanceAuditor(IProvenanceAuditor):
    """
    Main class for training provenance auditing and verification
    
    Provides comprehensive tracking of training events with cryptographic
    proofs and blockchain integration.
    Implements the IProvenanceAuditor interface for compatibility with ZK modules.
    """
    
    def __init__(self, model_id: str, blockchain_client: Optional[BlockchainClient] = None,
                 compression_enabled: bool = True, max_history_size: int = 10000,
                 hash_function: str = "sha256", fail_on_zk_error: bool = False):
        """
        Initialize TrainingProvenanceAuditor
        
        Args:
            model_id: Unique identifier for the model
            blockchain_client: Optional blockchain client for immutable storage
            compression_enabled: Whether to compress large histories
            max_history_size: Maximum number of events to keep in memory
            hash_function: Hash function to use ("sha256" or "poseidon")
            fail_on_zk_error: Whether to raise exceptions on ZK witness failures
        """
        self.model_id = model_id
        self.blockchain_client = blockchain_client or MockBlockchainClient()
        self.compression_enabled = compression_enabled
        self.max_history_size = max_history_size
        self.hash_function = hash_function
        self.fail_on_zk_error = fail_on_zk_error
        
        # Set up hash function
        if hash_function == "poseidon":
            try:
                from ..zk.poseidon import poseidon_hash, poseidon_hash_two
                self._hash_func = poseidon_hash
                self._hash_two = poseidon_hash_two
            except ImportError:
                logger.warning("Poseidon not available, falling back to SHA-256")
                self.hash_function = "sha256"
                self._hash_func = lambda x: hashlib.sha256(x).digest()
                self._hash_two = lambda x, y: hashlib.sha256(x + y).digest()
        else:
            self._hash_func = lambda x: hashlib.sha256(x).digest()
            self._hash_two = lambda x, y: hashlib.sha256(x + y).digest()
        
        # Training history
        self.events: List[TrainingEvent] = []
        self.event_index: Dict[str, TrainingEvent] = {}
        
        # Merkle trees for epochs
        self.epoch_trees: Dict[int, MerkleTree] = {}
        self.master_tree: Optional[MerkleTree] = None
        self.merkle_trees: Dict[int, Dict] = {}  # For compatibility
        
        # Use Poseidon Merkle trees if requested
        self.use_poseidon_merkle = (hash_function == "poseidon")
        
        # ZK proof generator
        self.zk_proof_generator = ZeroKnowledgeProof()
        
        # Blockchain transactions
        self.blockchain_transactions: List[str] = []
        
        # Metadata
        self.metadata = {
            'model_id': model_id,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'version': '1.0.0',
            'total_events': 0,
            'last_epoch': -1
        }
        
        logger.info(f"TrainingProvenanceAuditor initialized for model: {model_id} with {hash_function} hashing")
    
    def _compute_merkle_root(self, data_blocks: List[bytes]) -> bytes:
        """
        Compute Merkle root using configured hash function.
        
        Args:
            data_blocks: List of data blocks
            
        Returns:
            Root hash as bytes
        """
        if self.use_poseidon_merkle:
            try:
                from ..zk.poseidon import poseidon_merkle_root
                return poseidon_merkle_root(data_blocks)
            except ImportError:
                logger.warning("Poseidon not available, using SHA-256")
        
        # Fall back to SHA-256
        return compute_merkle_root(data_blocks)
    
    def log_training_event(self, epoch: int, metrics: Dict[str, Any],
                          checkpoint_hash: Optional[str] = None,
                          timestamp: Optional[datetime] = None,
                          event_type: EventType = EventType.EPOCH_END,
                          metadata: Optional[Dict] = None) -> TrainingEvent:
        """
        Record a training event with cryptographic timestamp
        
        Args:
            epoch: Training epoch number
            metrics: Training metrics (loss, accuracy, etc.)
            checkpoint_hash: Hash of model checkpoint
            timestamp: Event timestamp (defaults to now)
            event_type: Type of training event
            metadata: Additional metadata
            
        Returns:
            Created TrainingEvent
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        # Generate event ID
        event_id = f"{self.model_id}_epoch{epoch}_{int(timestamp.timestamp())}"
        
        # Get previous hash for chain
        previous_hash = self.events[-1].event_hash if self.events else None
        
        # Create event
        event = TrainingEvent(
            event_id=event_id,
            event_type=event_type,
            epoch=epoch,
            timestamp=timestamp,
            metrics=metrics,
            checkpoint_hash=checkpoint_hash,
            metadata=metadata or {},
            previous_hash=previous_hash
        )
        
        # Store event
        self.events.append(event)
        self.event_index[event_id] = event
        
        # Update metadata
        self.metadata['total_events'] = len(self.events)
        # Ensure epoch comparison works with different types
        current_last = self.metadata['last_epoch']
        if not isinstance(current_last, int):
            if isinstance(current_last, str):
                try:
                    current_last = int(current_last)
                except ValueError:
                    current_last = -1
            else:
                current_last = -1
        # Ensure epoch is also int
        if not isinstance(epoch, int):
            epoch = int(epoch) if str(epoch).isdigit() else 0
        self.metadata['last_epoch'] = max(current_last, epoch)
        
        # Store on blockchain if configured
        if self.blockchain_client and event_type in [EventType.CHECKPOINT_SAVE, EventType.TRAINING_END]:
            tx_id = self.blockchain_client.store_hash(
                event.event_hash,
                {'event_id': event_id, 'epoch': epoch, 'type': event_type.value}
            )
            self.blockchain_transactions.append(tx_id)
            logger.info(f"Event stored on blockchain: {tx_id}")
        
        # Manage history size
        if len(self.events) > self.max_history_size:
            self._compress_old_events()
        
        logger.info(f"Logged training event: {event_id} (epoch {epoch})")
        
        return event
    
    def embed_provenance(self, model_state: Dict[str, Any], 
                        training_log: Optional[List[TrainingEvent]] = None) -> Dict[str, Any]:
        """
        Embed training history into model metadata
        
        Args:
            model_state: Current model state dictionary
            training_log: Training events to embed (defaults to all)
            
        Returns:
            Model state with embedded provenance
        """
        if training_log is None:
            training_log = self.events
        
        # Build Merkle tree for efficient verification
        merkle_tree = MerkleTree(training_log)
        root_hash = merkle_tree.get_root_hash()
        
        # Create provenance metadata
        provenance = {
            'model_id': self.model_id,
            'merkle_root': root_hash,
            'num_events': len(training_log),
            'first_event': training_log[0].to_dict() if training_log else None,
            'last_event': training_log[-1].to_dict() if training_log else None,
            'epoch_range': {
                'start': min(e.epoch for e in training_log) if training_log else None,
                'end': max(e.epoch for e in training_log) if training_log else None
            },
            'embedded_at': datetime.now(timezone.utc).isoformat(),
            'blockchain_txs': self.blockchain_transactions[-5:]  # Last 5 transactions
        }
        
        # Compress full history if enabled
        if self.compression_enabled and len(training_log) > 100:
            compressed = self._compress_events(training_log)
            provenance['compressed_history'] = compressed
            provenance['compression_ratio'] = len(compressed) / sum(
                len(json.dumps(e.to_dict())) for e in training_log
            )
        else:
            provenance['full_history'] = [e.to_dict() for e in training_log]
        
        # Embed in model state
        if 'metadata' not in model_state:
            model_state['metadata'] = {}
        
        model_state['metadata']['training_provenance'] = provenance
        
        # Sign the provenance
        provenance_bytes = json.dumps(provenance, sort_keys=True).encode()
        signature = hashlib.sha256(provenance_bytes).hexdigest()
        model_state['metadata']['provenance_signature'] = signature
        
        logger.info(f"Embedded provenance for {len(training_log)} events")
        
        return model_state
    
    def generate_training_proof(self, start_epoch: int, end_epoch: int,
                               proof_type: ProofType = ProofType.MERKLE) -> Dict[str, Any]:
        """
        Generate cryptographic proof of training progression
        
        Args:
            start_epoch: Starting epoch
            end_epoch: Ending epoch
            proof_type: Type of proof to generate
            
        Returns:
            Proof data dictionary
        """
        # Filter events for epoch range
        relevant_events = [
            e for e in self.events
            if start_epoch <= e.epoch <= end_epoch
        ]
        
        if not relevant_events:
            raise ValueError(f"No events found for epochs {start_epoch}-{end_epoch}")
        
        if proof_type == ProofType.MERKLE:
            # Generate Merkle proof
            tree = MerkleTree(relevant_events)
            root_hash = tree.get_root_hash()
            
            # Get proofs for boundary events
            start_proof = tree.get_proof(0)
            end_proof = tree.get_proof(len(relevant_events) - 1)
            
            proof_data = {
                'type': 'merkle',
                'root_hash': root_hash,
                'num_events': len(relevant_events),
                'start_epoch': start_epoch,
                'end_epoch': end_epoch,
                'start_event_hash': relevant_events[0].event_hash,
                'end_event_hash': relevant_events[-1].event_hash,
                'start_proof': start_proof,
                'end_proof': end_proof,
                'generated_at': datetime.now(timezone.utc).isoformat()
            }
            
        elif proof_type == ProofType.ZERO_KNOWLEDGE:
            # Generate ZK proof
            start_state = {
                'epoch': start_epoch,
                'metrics': relevant_events[0].metrics,
                'hash': relevant_events[0].event_hash
            }
            
            end_state = {
                'epoch': end_epoch,
                'metrics': relevant_events[-1].metrics,
                'hash': relevant_events[-1].event_hash
            }
            
            transitions = [
                {
                    'epoch': e.epoch,
                    'metrics_delta': self._calculate_metrics_delta(
                        relevant_events[i-1].metrics if i > 0 else {},
                        e.metrics
                    )
                }
                for i, e in enumerate(relevant_events)
            ]
            
            zk_proof = self.zk_proof_generator.generate_progression_proof(
                start_state, end_state, transitions
            )
            
            proof_data = {
                'type': 'zero_knowledge',
                'proof': zk_proof,
                'epoch_range': {'start': start_epoch, 'end': end_epoch},
                'num_transitions': len(transitions)
            }
            
        else:
            # Generate signature proof
            proof_content = {
                'model_id': self.model_id,
                'epoch_range': {'start': start_epoch, 'end': end_epoch},
                'events': [e.event_hash for e in relevant_events],
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            proof_bytes = json.dumps(proof_content, sort_keys=True).encode()
            signature = hmac.new(
                self.zk_proof_generator.secret_key,
                proof_bytes,
                hashlib.sha256
            ).hexdigest()
            
            proof_data = {
                'type': 'signature',
                'content': proof_content,
                'signature': signature
            }
        
        # Store proof on blockchain
        if self.blockchain_client:
            tx_id = self.blockchain_client.store_hash(
                hashlib.sha256(json.dumps(proof_data).encode()).hexdigest(),
                {'proof_type': proof_type.value, 'epochs': f"{start_epoch}-{end_epoch}"}
            )
            proof_data['blockchain_tx'] = tx_id
        
        logger.info(f"Generated {proof_type.value} proof for epochs {start_epoch}-{end_epoch}")
        
        return proof_data
    
    def verify_training_history(self, model: Any, claimed_history: List[Dict]) -> bool:
        """
        Verify model was trained according to claimed history
        
        Args:
            model: Model object or state dict
            claimed_history: Claimed training history
            
        Returns:
            True if history is verified
        """
        # Extract provenance from model
        if isinstance(model, dict):
            model_state = model
        else:
            model_state = getattr(model, 'state_dict', lambda: {})()
        
        if 'metadata' not in model_state or 'training_provenance' not in model_state['metadata']:
            logger.warning("No provenance metadata found in model")
            return False
        
        provenance = model_state['metadata']['training_provenance']
        
        # Verify signature
        expected_signature = model_state['metadata'].get('provenance_signature')
        if expected_signature:
            provenance_bytes = json.dumps(provenance, sort_keys=True).encode()
            actual_signature = hashlib.sha256(provenance_bytes).hexdigest()
            
            if expected_signature != actual_signature:
                logger.error("Provenance signature mismatch")
                return False
        
        # Verify Merkle root
        if 'merkle_root' in provenance:
            # Rebuild tree from claimed history
            claimed_events = [
                TrainingEvent(
                    event_id=h['event_id'],
                    event_type=EventType(h['event_type']),
                    epoch=h['epoch'],
                    timestamp=datetime.fromisoformat(h['timestamp']),
                    metrics=h['metrics'],
                    checkpoint_hash=h.get('checkpoint_hash'),
                    metadata=h.get('metadata', {}),
                    previous_hash=h.get('previous_hash'),
                    event_hash=h.get('event_hash')
                )
                for h in claimed_history
            ]
            
            claimed_tree = MerkleTree(claimed_events)
            claimed_root = claimed_tree.get_root_hash()
            
            if claimed_root != provenance['merkle_root']:
                logger.error(f"Merkle root mismatch: {claimed_root} != {provenance['merkle_root']}")
                return False
        
        # Verify blockchain hashes if present
        if self.blockchain_client and 'blockchain_txs' in provenance:
            for tx_id in provenance['blockchain_txs']:
                stored = self.blockchain_client.retrieve_hash(tx_id)
                if not stored:
                    logger.error(f"Blockchain transaction not found: {tx_id}")
                    return False
        
        # Verify event chain integrity
        previous_hash = None
        for event_dict in claimed_history:
            if previous_hash and event_dict.get('previous_hash') != previous_hash:
                logger.error(f"Event chain broken at {event_dict['event_id']}")
                return False
            previous_hash = event_dict.get('event_hash')
        
        logger.info("Training history verified successfully")
        return True
    
    def query_events(self, filters: Optional[Dict[str, Any]] = None,
                    start_epoch: Optional[int] = None,
                    end_epoch: Optional[int] = None,
                    event_types: Optional[List[EventType]] = None) -> List[TrainingEvent]:
        """
        Query training events with filters
        
        Args:
            filters: Custom filters for metrics/metadata
            start_epoch: Starting epoch (inclusive)
            end_epoch: Ending epoch (inclusive)
            event_types: Filter by event types
            
        Returns:
            Filtered list of events
        """
        results = self.events.copy()
        
        # Filter by epoch range
        if start_epoch is not None:
            results = [e for e in results if e.epoch >= start_epoch]
        
        if end_epoch is not None:
            results = [e for e in results if e.epoch <= end_epoch]
        
        # Filter by event types
        if event_types:
            results = [e for e in results if e.event_type in event_types]
        
        # Apply custom filters
        if filters:
            for key, value in filters.items():
                if key in ['min_loss', 'max_loss']:
                    # Metric filters
                    if key == 'min_loss':
                        results = [
                            e for e in results
                            if e.metrics.get('loss', float('inf')) >= value
                        ]
                    else:
                        results = [
                            e for e in results
                            if e.metrics.get('loss', 0) <= value
                        ]
                elif key.startswith('has_'):
                    # Check for presence of keys
                    metric_key = key[4:]  # Remove 'has_' prefix
                    results = [e for e in results if metric_key in e.metrics]
        
        return results
    
    def export_history(self, format: str = 'json',
                      include_proofs: bool = True) -> Union[str, bytes]:
        """
        Export training history in standard format
        
        Args:
            format: Export format ('json', 'protobuf', 'compressed')
            include_proofs: Whether to include cryptographic proofs
            
        Returns:
            Exported data
        """
        export_data = {
            'model_id': self.model_id,
            'metadata': self.metadata,
            'events': [e.to_dict() for e in self.events],
            'num_events': len(self.events),
            'exported_at': datetime.now(timezone.utc).isoformat()
        }
        
        if include_proofs and self.master_tree:
            export_data['merkle_root'] = self.master_tree.get_root_hash()
            export_data['blockchain_transactions'] = self.blockchain_transactions
        
        if format == 'json':
            return json.dumps(export_data, indent=2)
        
        elif format == 'protobuf':
            # Simplified protobuf-like binary format
            # In production, use actual protobuf
            return pickle.dumps(export_data)
        
        elif format == 'compressed':
            json_data = json.dumps(export_data).encode()
            return zlib.compress(json_data, level=9)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def import_history(self, data: Union[str, bytes], format: str = 'json') -> None:
        """
        Import training history from exported data
        
        Args:
            data: Exported data
            format: Data format
        """
        if format == 'json':
            import_data = json.loads(data)
        elif format == 'protobuf':
            import_data = pickle.loads(data)
        elif format == 'compressed':
            decompressed = zlib.decompress(data)
            import_data = json.loads(decompressed)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Clear existing data
        self.events.clear()
        self.event_index.clear()
        
        # Import events
        for event_dict in import_data['events']:
            event = TrainingEvent(
                event_id=event_dict['event_id'],
                event_type=EventType(event_dict['event_type']),
                epoch=event_dict['epoch'],
                timestamp=datetime.fromisoformat(event_dict['timestamp']),
                metrics=event_dict['metrics'],
                checkpoint_hash=event_dict.get('checkpoint_hash'),
                metadata=event_dict.get('metadata', {}),
                previous_hash=event_dict.get('previous_hash'),
                event_hash=event_dict.get('event_hash')
            )
            
            self.events.append(event)
            self.event_index[event.event_id] = event
        
        # Update metadata
        self.metadata = import_data['metadata']
        
        logger.info(f"Imported {len(self.events)} events")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics"""
        if not self.events:
            return {
                'total_events': 0,
                'epochs': 0,
                'duration': 0
            }
        
        # Calculate statistics
        epochs = set(e.epoch for e in self.events)
        start_time = min(e.timestamp for e in self.events)
        end_time = max(e.timestamp for e in self.events)
        duration = (end_time - start_time).total_seconds()
        
        # Metric statistics
        all_losses = [e.metrics.get('loss', 0) for e in self.events if 'loss' in e.metrics]
        all_accuracies = [e.metrics.get('accuracy', 0) for e in self.events if 'accuracy' in e.metrics]
        
        stats = {
            'total_events': len(self.events),
            'epochs': len(epochs),
            'duration_seconds': duration,
            'duration_hours': duration / 3600,
            'events_per_epoch': len(self.events) / len(epochs) if epochs else 0,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'blockchain_transactions': len(self.blockchain_transactions)
        }
        
        if all_losses:
            stats['loss_stats'] = {
                'min': min(all_losses),
                'max': max(all_losses),
                'mean': np.mean(all_losses),
                'std': np.std(all_losses),
                'final': all_losses[-1]
            }
        
        if all_accuracies:
            stats['accuracy_stats'] = {
                'min': min(all_accuracies),
                'max': max(all_accuracies),
                'mean': np.mean(all_accuracies),
                'std': np.std(all_accuracies),
                'final': all_accuracies[-1]
            }
        
        # Event type distribution
        event_types = {}
        for event in self.events:
            event_type = event.event_type.value
            event_types[event_type] = event_types.get(event_type, 0) + 1
        stats['event_types'] = event_types
        
        return stats
    
    def generate_zk_witness(self, training_step_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate ZK witness data for training step verification with comprehensive validation.
        
        This method automatically detects model type (SGD vs LoRA), extracts witness data,
        and generates dual commitments (SHA-256 + Poseidon) for ZK proof compatibility.
        
        Args:
            training_step_data: Complete training step information containing:
                - weights_before: Model weights before training step
                - weights_after: Model weights after training step  
                - batch_data: Training batch (inputs, targets, gradients, etc.)
                - hyperparameters: Training hyperparameters
                - step_info: Step metadata (epoch, step_number, etc.)
                
        Returns:
            Dictionary containing ZK witness data and dual commitments, or None on failure
        """
        from datetime import datetime, timezone
        import numpy as np
        
        try:
            # Import ZK modules with proper error handling
            from ..zk.witness import extract_sgd_witness, extract_lora_witness
            from ..zk.lora_builder import LoRAWitnessBuilder
            from ..zk.poseidon import PoseidonMerkleTree
            
            logger.info("Starting ZK witness generation with automatic model type detection")
            
            # Extract data from training_step_data
            weights_before = training_step_data.get('weights_before', {})
            weights_after = training_step_data.get('weights_after', {})
            batch_data = training_step_data.get('batch_data', {})
            hyperparameters = training_step_data.get('hyperparameters', {})
            step_info = training_step_data.get('step_info', {})
            
            # Validate required data
            validation_result = self._validate_training_step_data(training_step_data)
            if not validation_result['valid']:
                logger.error(f"Training step data validation failed: {validation_result['errors']}")
                if self.fail_on_zk_error:
                    raise ValueError(f"Invalid training step data: {validation_result['errors']}")
                return None
            
            # Detect model type using improved detection
            is_lora = self._detect_lora_model(weights_before, weights_after)
            model_type = "lora" if is_lora else "sgd"
            
            logger.info(f"Detected model type: {model_type}")
            
            # Extract witness based on detected model type
            if is_lora:
                witness = self._extract_lora_witness_safe(
                    weights_before, weights_after, batch_data, hyperparameters
                )
                statement = self._build_lora_statement(witness, step_info)
                proof_type = "lora_step"
            else:
                witness = self._extract_sgd_witness_safe(
                    weights_before, weights_after, batch_data, hyperparameters
                )
                statement = self._build_sgd_statement(witness, step_info)
                proof_type = "sgd_step"
            
            # Generate dual commitments (SHA-256 + Poseidon)
            dual_commitments = self._generate_dual_commitments(witness)
            
            # Create comprehensive witness record
            witness_record = {
                'witness_id': f"{self.model_id}_{proof_type}_{step_info.get('step_number', 0)}",
                'model_type': model_type,
                'proof_type': proof_type,
                'witness': witness,
                'statement': statement,
                'sha256_root': dual_commitments['sha256_root'],
                'poseidon_root': dual_commitments['poseidon_root'],
                'witness_metadata': {
                    'step_number': step_info.get('step_number', 0),
                    'epoch': step_info.get('epoch', 0),
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'model_id': self.model_id,
                    'witness_size_bytes': len(str(witness)),
                    'tensor_shapes': self._extract_tensor_shapes(witness),
                    'commitment_scheme': 'dual'
                },
                'validation': validation_result,
                'dual_commitments': dual_commitments,
                'generated_at': datetime.now(timezone.utc).isoformat(),
                'auditor_version': self.metadata.get('version', '1.0.0')
            }
            
            # Store witness record
            if not hasattr(self, 'witness_records'):
                self.witness_records = []
            self.witness_records.append(witness_record)
            
            # Log witness generation event
            self.log_training_event(
                epoch=step_info.get('epoch', 0),
                metrics={'witness_generated': True, 'model_type': model_type},
                event_type=EventType.CUSTOM,
                metadata={
                    'zk_witness_id': witness_record['witness_id'],
                    'commitment_scheme': 'dual',
                    'poseidon_available': dual_commitments['poseidon_available'],
                    'sha256_fallback': dual_commitments.get('sha256_fallback', False)
                }
            )
            
            logger.info(f"Successfully generated ZK witness: {witness_record['witness_id']}")
            
            return witness_record
            
        except ImportError as e:
            logger.error(f"ZK modules not available: {e}")
            return self._generate_sha256_only_witness(training_step_data)
        
        except Exception as e:
            logger.error(f"Failed to generate ZK witness: {e}")
            if self.fail_on_zk_error:
                raise
            return None
    
    def _validate_training_step_data(self, training_step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate training step data for ZK witness generation.
        
        Returns:
            Dictionary with validation results and error messages
        """
        errors = []
        warnings = []
        
        # Check required top-level keys
        required_keys = ['weights_before', 'weights_after', 'batch_data', 'hyperparameters', 'step_info']
        for key in required_keys:
            if key not in training_step_data:
                errors.append(f"Missing required key: {key}")
        
        if errors:
            return {'valid': False, 'errors': errors, 'warnings': warnings}
        
        weights_before = training_step_data['weights_before']
        weights_after = training_step_data['weights_after']
        batch_data = training_step_data['batch_data']
        hyperparameters = training_step_data['hyperparameters']
        
        # Validate weights structure
        if not isinstance(weights_before, dict) or not weights_before:
            errors.append("weights_before must be non-empty dictionary")
        if not isinstance(weights_after, dict) or not weights_after:
            errors.append("weights_after must be non-empty dictionary")
        
        # Check weight tensor dimensions
        if weights_before and weights_after:
            try:
                self._validate_tensor_dimensions(weights_before, weights_after)
            except ValueError as e:
                errors.append(f"Tensor dimension mismatch: {e}")
        
        # Validate batch data
        if not isinstance(batch_data, dict):
            errors.append("batch_data must be dictionary")
        else:
            # Check for required batch components
            if 'inputs' not in batch_data:
                warnings.append("Missing batch inputs - may affect witness quality")
            if 'targets' not in batch_data:
                warnings.append("Missing batch targets - may affect witness quality")
        
        # Validate hyperparameters
        if not isinstance(hyperparameters, dict):
            errors.append("hyperparameters must be dictionary")
        else:
            if 'learning_rate' not in hyperparameters:
                warnings.append("Missing learning_rate in hyperparameters")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'tensor_count': len(weights_before) if isinstance(weights_before, dict) else 0,
            'validated_at': datetime.now(timezone.utc).isoformat()
        }
    
    def _validate_tensor_dimensions(self, weights_before: Dict[str, Any], weights_after: Dict[str, Any]) -> None:
        """Validate that tensor dimensions match between before/after weights."""
        import numpy as np
        
        # Check that all keys match
        if set(weights_before.keys()) != set(weights_after.keys()):
            missing_before = set(weights_after.keys()) - set(weights_before.keys())
            missing_after = set(weights_before.keys()) - set(weights_after.keys())
            if missing_before:
                raise ValueError(f"Keys missing in weights_before: {missing_before}")
            if missing_after:
                raise ValueError(f"Keys missing in weights_after: {missing_after}")
        
        # Check tensor shapes
        for key in weights_before.keys():
            tensor_before = weights_before[key]
            tensor_after = weights_after[key]
            
            if hasattr(tensor_before, 'shape') and hasattr(tensor_after, 'shape'):
                if tensor_before.shape != tensor_after.shape:
                    raise ValueError(f"Shape mismatch for {key}: {tensor_before.shape} vs {tensor_after.shape}")
            elif isinstance(tensor_before, (list, tuple)) and isinstance(tensor_after, (list, tuple)):
                if len(tensor_before) != len(tensor_after):
                    raise ValueError(f"Length mismatch for {key}: {len(tensor_before)} vs {len(tensor_after)}")
    
    def _detect_lora_model(self, weights_before: Dict[str, Any], weights_after: Dict[str, Any]) -> bool:
        """
        Detect if the model uses LoRA fine-tuning by examining weight structures.
        
        Returns:
            True if LoRA model detected, False for standard SGD
        """
        try:
            from ..zk.lora_builder import LoRAWitnessBuilder
            
            # Use LoRA builder's detection logic
            builder = LoRAWitnessBuilder()
            return builder.detect_lora_training(weights_before) and builder.detect_lora_training(weights_after)
            
        except ImportError:
            logger.warning("LoRA builder not available, using fallback detection")
            return self._fallback_lora_detection(weights_before, weights_after)
    
    def _fallback_lora_detection(self, weights_before: Dict[str, Any], weights_after: Dict[str, Any]) -> bool:
        """Fallback LoRA detection when ZK modules unavailable."""
        # Look for LoRA-specific key patterns
        lora_patterns = ['lora_A', 'lora_B', 'lora_a', 'lora_b', 'adapter_A', 'adapter_B']
        
        all_keys = set(weights_before.keys()) | set(weights_after.keys())
        
        # Check for LoRA key patterns
        has_lora_keys = any(
            any(pattern in key.lower() for pattern in lora_patterns)
            for key in all_keys
        )
        
        if has_lora_keys:
            logger.info("Detected LoRA model based on key patterns")
            return True
        
        # Check for parameter efficiency (LoRA typically has much fewer trainable parameters)
        try:
            import numpy as np
            
            total_params_before = sum(
                np.prod(tensor.shape) if hasattr(tensor, 'shape') else len(tensor) if isinstance(tensor, (list, tuple)) else 1
                for tensor in weights_before.values()
            )
            
            # Check if any tensors changed (indicating training)
            changed_tensors = 0
            for key in weights_before.keys():
                if key in weights_after:
                    before_val = weights_before[key]
                    after_val = weights_after[key]
                    
                    if hasattr(before_val, 'shape') and hasattr(after_val, 'shape'):
                        if not np.allclose(before_val, after_val, rtol=1e-6):
                            changed_tensors += 1
            
            # If very few tensors changed relative to total, likely LoRA
            if changed_tensors > 0 and changed_tensors < len(weights_before) * 0.1:
                logger.info("Detected likely LoRA model based on parameter change patterns")
                return True
                
        except Exception as e:
            logger.debug(f"Parameter analysis failed: {e}")
        
        return False
    
    def _extract_sgd_witness_safe(self, weights_before: Dict[str, Any], weights_after: Dict[str, Any], 
                                 batch_data: Dict[str, Any], hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safely extract SGD witness with comprehensive error handling."""
        try:
            from ..zk.witness import extract_sgd_witness
            import numpy as np
            
            # Prepare data for witness extraction
            model_weights_before = {}
            model_weights_after = {}
            
            # Convert tensors to numpy arrays if needed
            for key, tensor in weights_before.items():
                if hasattr(tensor, 'detach'):  # PyTorch tensor
                    model_weights_before[key] = tensor.detach().cpu().numpy()
                elif isinstance(tensor, np.ndarray):
                    model_weights_before[key] = tensor
                else:
                    model_weights_before[key] = np.array(tensor)
            
            for key, tensor in weights_after.items():
                if hasattr(tensor, 'detach'):  # PyTorch tensor
                    model_weights_after[key] = tensor.detach().cpu().numpy()
                elif isinstance(tensor, np.ndarray):
                    model_weights_after[key] = tensor
                else:
                    model_weights_after[key] = np.array(tensor)
            
            # Extract batch components
            batch_inputs = batch_data.get('inputs', np.array([]))
            batch_targets = batch_data.get('targets', np.array([]))
            gradients = batch_data.get('gradients', {})
            loss_value = batch_data.get('loss', 0.0)
            
            if isinstance(batch_inputs, list):
                batch_inputs = np.array(batch_inputs)
            if isinstance(batch_targets, list):
                batch_targets = np.array(batch_targets)
            
            # Call witness extraction
            witness = extract_sgd_witness(
                model_weights_before=model_weights_before,
                model_weights_after=model_weights_after,
                batch_inputs=batch_inputs,
                batch_targets=batch_targets,
                hyperparameters=hyperparameters,
                gradients=gradients,
                loss_value=loss_value
            )
            
            logger.debug("Successfully extracted SGD witness")
            return witness
            
        except Exception as e:
            logger.error(f"SGD witness extraction failed: {e}")
            raise
    
    def _extract_lora_witness_safe(self, weights_before: Dict[str, Any], weights_after: Dict[str, Any],
                                  batch_data: Dict[str, Any], hyperparameters: Dict[str, Any]) -> Dict[str, Any]:
        """Safely extract LoRA witness with comprehensive error handling."""
        try:
            from ..zk.lora_builder import LoRAWitnessBuilder
            import numpy as np
            
            builder = LoRAWitnessBuilder()
            
            # Extract LoRA adapters
            adapters_before = builder.extract_lora_adapters(weights_before)
            adapters_after = builder.extract_lora_adapters(weights_after)
            
            if not adapters_before or not adapters_after:
                raise ValueError("Failed to extract LoRA adapters from model weights")
            
            # Build LoRA witness using the builder with learning rate
            learning_rate = hyperparameters.get('learning_rate', 0.001)
            witness = builder.build_lora_witness(
                adapters_before=adapters_before,
                adapters_after=adapters_after,
                batch_data=batch_data,
                learning_rate=learning_rate
            )
            
            logger.debug("Successfully extracted LoRA witness")
            return witness
            
        except Exception as e:
            logger.error(f"LoRA witness extraction failed: {e}")
            raise
    
    def _build_sgd_statement(self, witness: Dict[str, Any], step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build ZK statement for SGD proof."""
        return {
            'proof_type': 'sgd_step',
            'step_number': step_info.get('step_number', 0),
            'epoch': step_info.get('epoch', 0),
            'witness_hash': hash(str(witness)),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'statement_version': '1.0'
        }
    
    def _build_lora_statement(self, witness: Dict[str, Any], step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build ZK statement for LoRA proof."""
        return {
            'proof_type': 'lora_step',
            'step_number': step_info.get('step_number', 0),
            'epoch': step_info.get('epoch', 0),
            'witness_hash': hash(str(witness)),
            'rank': witness.get('rank', 'unknown'),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'statement_version': '1.0'
        }
    
    def _generate_dual_commitments(self, witness: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate dual commitments (SHA-256 + Poseidon) for witness data.
        
        Returns:
            Dictionary containing both commitment roots and availability info
        """
        import hashlib
        import json
        
        # Convert witness to dict if it's a dataclass
        if hasattr(witness, '__dataclass_fields__'):
            witness_dict = asdict(witness)
        else:
            witness_dict = witness
            
        # Generate SHA-256 commitment (always available)
        witness_str = json.dumps(witness_dict, sort_keys=True, default=str)
        sha256_root = hashlib.sha256(witness_str.encode()).hexdigest()
        
        poseidon_available = False
        poseidon_root = None
        sha256_fallback = False
        
        # Try to generate Poseidon commitment
        try:
            from ..zk.poseidon import PoseidonMerkleTree
            
            # Create Poseidon Merkle tree
            poseidon_tree = PoseidonMerkleTree()
            
            # Convert witness to field elements for Poseidon
            witness_elements = self._witness_to_field_elements(witness)
            poseidon_root = poseidon_tree.compute_root(witness_elements)
            poseidon_available = True
            
            logger.debug("Successfully generated Poseidon commitment")
            
        except ImportError:
            logger.warning("Poseidon commitment unavailable - using SHA-256 only")
            sha256_fallback = True
        except Exception as e:
            logger.warning(f"Poseidon commitment failed: {e} - falling back to SHA-256")
            sha256_fallback = True
        
        return {
            'sha256_root': sha256_root,
            'poseidon_root': poseidon_root,
            'poseidon_available': poseidon_available,
            'sha256_fallback': sha256_fallback,
            'supports_zk': poseidon_available,
            'commitment_scheme': 'dual' if poseidon_available else 'sha256_only'
        }
    
    def _witness_to_field_elements(self, witness: Dict[str, Any]) -> List[int]:
        """Convert witness data to field elements for Poseidon hashing."""
        elements = []
        
        import numpy as np
        
        # Convert witness to dict if it's a dataclass
        if hasattr(witness, '__dataclass_fields__'):
            witness_dict = asdict(witness)
        else:
            witness_dict = witness
            
        for key, value in witness_dict.items():
            if isinstance(value, (int, float)):
                # Convert to field element (mod prime)
                elements.append(int(abs(value) * 1000) % (2**251 - 1))  # BLS12-381 scalar field
            elif isinstance(value, np.ndarray):
                # Flatten and convert array elements
                flat_values = value.flatten()[:100]  # Limit size for efficiency
                for v in flat_values:
                    elements.append(int(abs(float(v)) * 1000) % (2**251 - 1))
            elif isinstance(value, (list, tuple)):
                # Convert list elements
                for i, v in enumerate(value[:50]):  # Limit size
                    if isinstance(v, (int, float)):
                        elements.append(int(abs(v) * 1000) % (2**251 - 1))
            elif isinstance(value, str):
                # Hash string to field element
                import hashlib
                hash_int = int(hashlib.sha256(value.encode()).hexdigest(), 16)
                elements.append(hash_int % (2**251 - 1))
        
        # Ensure we have at least one element
        if not elements:
            elements = [42]  # Default value
        
        return elements
    
    def _extract_tensor_shapes(self, witness: Dict[str, Any]) -> Dict[str, Any]:
        """Extract tensor shape information from witness data."""
        shapes = {}
        
        # Convert witness to dict if it's a dataclass
        if hasattr(witness, '__dataclass_fields__'):
            witness_dict = asdict(witness)
        else:
            witness_dict = witness
            
        for key, value in witness_dict.items():
            if hasattr(value, 'shape'):
                shapes[key] = list(value.shape)
            elif isinstance(value, (list, tuple)):
                shapes[key] = [len(value)]
            elif isinstance(value, dict):
                shapes[key] = f"dict_{len(value)}_keys"
            else:
                shapes[key] = "scalar"
        
        return shapes
    
    def _generate_sha256_only_witness(self, training_step_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate witness record with SHA-256 commitment only (fallback mode).
        
        This is used when ZK modules are unavailable but we still want to maintain
        audit trail compatibility.
        """
        import hashlib
        import json
        
        logger.warning("Generating SHA-256-only witness (ZK modules unavailable)")
        
        step_info = training_step_data.get('step_info', {})
        weights_before = training_step_data.get('weights_before', {})
        weights_after = training_step_data.get('weights_after', {})
        
        # Create simplified witness
        witness = {
            'weights_before_hash': hashlib.sha256(str(weights_before).encode()).hexdigest()[:16],
            'weights_after_hash': hashlib.sha256(str(weights_after).encode()).hexdigest()[:16],
            'step_number': step_info.get('step_number', 0),
            'epoch': step_info.get('epoch', 0),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'fallback_mode': True
        }
        
        # Generate SHA-256 root
        witness_str = json.dumps(witness, sort_keys=True)
        sha256_root = hashlib.sha256(witness_str.encode()).hexdigest()
        
        witness_record = {
            'witness_id': f"{self.model_id}_fallback_{step_info.get('step_number', 0)}",
            'model_type': 'unknown',
            'proof_type': 'sha256_only',
            'witness': witness,
            'statement': {'fallback_mode': True, 'limited_functionality': True},
            'sha256_root': sha256_root,
            'poseidon_root': None,
            'witness_metadata': {
                'step_number': step_info.get('step_number', 0),
                'epoch': step_info.get('epoch', 0),
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'model_id': self.model_id,
                'witness_size_bytes': len(str(witness)),
                'commitment_scheme': 'sha256_only',
                'fallback_mode': True
            },
            'dual_commitments': {
                'sha256_root': sha256_root,
                'poseidon_root': None,
                'poseidon_available': False,
                'sha256_fallback': True,
                'supports_zk': False,
                'commitment_scheme': 'sha256_only'
            },
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'auditor_version': self.metadata.get('version', '1.0.0')
        }
        
        logger.info(f"Generated SHA-256-only witness: {witness_record['witness_id']}")
        return witness_record

    def get_witness_records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve stored witness records with optional limit.
        
        Args:
            limit: Maximum number of records to return (None for all)
            
        Returns:
            List of witness records, most recent first
        """
        if not hasattr(self, 'witness_records'):
            return []
        
        records = sorted(
            self.witness_records, 
            key=lambda x: x['generated_at'], 
            reverse=True
        )
        
        if limit:
            records = records[:limit]
        
        return records
    
    def get_witness_record_by_id(self, witness_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific witness record by ID."""
        if not hasattr(self, 'witness_records'):
            return None
        
        for record in self.witness_records:
            if record.get('witness_id') == witness_id:
                return record
        
        return None
    
    def get_witness_statistics(self) -> Dict[str, Any]:
        """Get statistics about generated witness records."""
        if not hasattr(self, 'witness_records'):
            return {
                'total_witnesses': 0,
                'sgd_witnesses': 0,
                'lora_witnesses': 0,
                'poseidon_available_count': 0,
                'fallback_count': 0
            }
        
        stats = {
            'total_witnesses': len(self.witness_records),
            'sgd_witnesses': 0,
            'lora_witnesses': 0,
            'poseidon_available_count': 0,
            'fallback_count': 0,
            'sha256_only_count': 0,
            'dual_commitment_count': 0
        }
        
        for record in self.witness_records:
            model_type = record.get('model_type', 'unknown')
            if model_type == 'sgd':
                stats['sgd_witnesses'] += 1
            elif model_type == 'lora':
                stats['lora_witnesses'] += 1
                
            dual_commitments = record.get('dual_commitments', {})
            if dual_commitments.get('poseidon_available', False):
                stats['poseidon_available_count'] += 1
            if dual_commitments.get('sha256_fallback', False):
                stats['fallback_count'] += 1
            if dual_commitments.get('commitment_scheme') == 'sha256_only':
                stats['sha256_only_count'] += 1
            elif dual_commitments.get('commitment_scheme') == 'dual':
                stats['dual_commitment_count'] += 1
        
        return stats
    
    def validate_witness_merkle_paths(self, witness_id: str) -> Dict[str, Any]:
        """
        Validate Merkle paths for a specific witness record.
        
        Args:
            witness_id: ID of witness record to validate
            
        Returns:
            Validation results including path verification status
        """
        record = self.get_witness_record_by_id(witness_id)
        if not record:
            return {'valid': False, 'error': 'Witness record not found'}
        
        results = {
            'witness_id': witness_id,
            'sha256_valid': False,
            'poseidon_valid': False,
            'validation_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Validate SHA-256 commitment
        try:
            import hashlib
            import json
            
            witness = record.get('witness', {})
            witness_str = json.dumps(witness, sort_keys=True, default=str)
            expected_sha256 = hashlib.sha256(witness_str.encode()).hexdigest()
            actual_sha256 = record.get('sha256_root')
            
            results['sha256_valid'] = (expected_sha256 == actual_sha256)
            results['sha256_expected'] = expected_sha256
            results['sha256_actual'] = actual_sha256
            
        except Exception as e:
            results['sha256_error'] = str(e)
        
        # Validate Poseidon commitment if available
        dual_commitments = record.get('dual_commitments', {})
        if dual_commitments.get('poseidon_available', False):
            try:
                from ..zk.poseidon import PoseidonMerkleTree
                
                poseidon_tree = PoseidonMerkleTree()
                witness_elements = self._witness_to_field_elements(record.get('witness', {}))
                expected_poseidon = poseidon_tree.compute_root(witness_elements)
                actual_poseidon = record.get('poseidon_root')
                
                results['poseidon_valid'] = (expected_poseidon == actual_poseidon)
                results['poseidon_expected'] = expected_poseidon
                results['poseidon_actual'] = actual_poseidon
                
            except ImportError:
                results['poseidon_error'] = 'Poseidon module not available'
            except Exception as e:
                results['poseidon_error'] = str(e)
        
        results['overall_valid'] = results['sha256_valid'] and (
            not dual_commitments.get('poseidon_available', False) or 
            results.get('poseidon_valid', False)
        )
        
        return results

    def add_dual_commitment_support(self) -> None:
        """
        Add support for dual commitment schemes (SHA-256 + Poseidon)
        
        This method enhances the existing SHA-256 Merkle trees with
        ZK-friendly Poseidon commitments for seamless ZK proof integration.
        """
        try:
            from ..zk.commitments import create_zk_compatible_commitment
            
            logger.info("Adding dual commitment support to existing Merkle trees")
            
            # Enhance existing epoch trees with ZK compatibility
            for epoch, merkle_tree in self.epoch_trees.items():
                if hasattr(merkle_tree, 'root') and merkle_tree.root:
                    zk_commitment = create_zk_compatible_commitment(merkle_tree.root)
                    
                    # Store dual commitment metadata
                    if not hasattr(merkle_tree, 'zk_metadata'):
                        merkle_tree.zk_metadata = {}
                    
                    merkle_tree.zk_metadata.update({
                        "dual_commitment": zk_commitment,
                        "zk_compatible": True,
                        "enhanced_at": datetime.now(timezone.utc).isoformat()
                    })
                    
                    logger.info(f"Enhanced epoch {epoch} tree with ZK compatibility")
            
            # Enhance master tree if it exists
            if self.master_tree and hasattr(self.master_tree, 'root') and self.master_tree.root:
                zk_commitment = create_zk_compatible_commitment(self.master_tree.root)
                
                if not hasattr(self.master_tree, 'zk_metadata'):
                    self.master_tree.zk_metadata = {}
                
                self.master_tree.zk_metadata.update({
                    "dual_commitment": zk_commitment,
                    "zk_compatible": True,
                    "enhanced_at": datetime.now(timezone.utc).isoformat()
                })
                
                logger.info("Enhanced master tree with ZK compatibility")
            
            # Update auditor metadata
            self.metadata["zk_support"] = {
                "dual_commitments": True,
                "sha256_compatible": True,
                "poseidon_compatible": True,
                "enabled_at": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info("Successfully added dual commitment support")
            
        except ImportError:
            logger.warning("ZK module not available - dual commitment support not added")
        except Exception as e:
            logger.error(f"Failed to add dual commitment support: {e}")
    
    def _generate_mock_zk_witness(self, weights_before: Dict, weights_after: Dict, 
                                 batch_data: Dict, step_info: Dict, proof_type: str) -> Dict[str, Any]:
        """
        Generate mock ZK witness when ZK module is not available
        
        This ensures compatibility even when ZK dependencies are missing.
        """
        mock_witness = {
            "witness": {
                "type": "mock",
                "proof_type": proof_type,
                "note": "Mock witness - ZK module not available"
            },
            "statement": {
                "type": "mock",
                "step_number": step_info.get("step_number", 0),
                "epoch": step_info.get("epoch", 0)
            },
            "witness_record": {
                "witness_id": f"{self.model_id}_mock_{step_info.get('step_number', 0)}",
                "is_mock": True,
                "generated_at": datetime.now(timezone.utc).isoformat()
            },
            "compatibility": {
                "existing_merkle": True,
                "zk_friendly": False,
                "dual_commitment": False
            }
        }
        
        logger.warning("Generated mock ZK witness - install ZK module for full functionality")
        return mock_witness
    
    def _compress_events(self, events: List[TrainingEvent]) -> str:
        """Compress events for storage"""
        event_dicts = [e.to_dict() for e in events]
        json_data = json.dumps(event_dicts).encode()
        compressed = zlib.compress(json_data, level=9)
        return base64.b64encode(compressed).decode()
    
    def _decompress_events(self, compressed: str) -> List[TrainingEvent]:
        """Decompress stored events"""
        compressed_bytes = base64.b64decode(compressed)
        json_data = zlib.decompress(compressed_bytes)
        event_dicts = json.loads(json_data)
        
        events = []
        for event_dict in event_dicts:
            event = TrainingEvent(
                event_id=event_dict['event_id'],
                event_type=EventType(event_dict['event_type']),
                epoch=event_dict['epoch'],
                timestamp=datetime.fromisoformat(event_dict['timestamp']),
                metrics=event_dict['metrics'],
                checkpoint_hash=event_dict.get('checkpoint_hash'),
                metadata=event_dict.get('metadata', {}),
                previous_hash=event_dict.get('previous_hash'),
                event_hash=event_dict.get('event_hash')
            )
            events.append(event)
        
        return events
    
    def _compress_old_events(self):
        """Compress old events to save memory"""
        if len(self.events) <= self.max_history_size:
            return
        
        # Keep recent events in memory
        num_to_compress = len(self.events) - self.max_history_size
        events_to_compress = self.events[:num_to_compress]
        
        # Store compressed version
        compressed = self._compress_events(events_to_compress)
        
        # Update metadata
        if 'compressed_history' not in self.metadata:
            self.metadata['compressed_history'] = []
        
        self.metadata['compressed_history'].append({
            'num_events': len(events_to_compress),
            'epoch_range': {
                'start': min(e.epoch for e in events_to_compress),
                'end': max(e.epoch for e in events_to_compress)
            },
            'compressed_data': compressed
        })
        
        # Remove from main list
        self.events = self.events[num_to_compress:]
        
        logger.info(f"Compressed {num_to_compress} old events")
    
    def _calculate_metrics_delta(self, prev_metrics: Dict, curr_metrics: Dict) -> Dict:
        """Calculate metric changes between epochs"""
        delta = {}
        
        for key in curr_metrics:
            if key in prev_metrics:
                try:
                    delta[key] = curr_metrics[key] - prev_metrics[key]
                except (TypeError, KeyError):
                    delta[key] = None
            else:
                delta[key] = curr_metrics[key]
        
        return delta
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"TrainingProvenanceAuditor("
            f"model_id='{self.model_id}', "
            f"events={stats['total_events']}, "
            f"epochs={stats['epochs']}, "
            f"blockchain_txs={stats['blockchain_transactions']})"
        )


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Training Provenance Auditor - Example Usage")
    print("=" * 70)
    
    # Initialize auditor
    auditor = TrainingProvenanceAuditor(
        model_id="model_v1.0",
        blockchain_client=MockBlockchainClient()
    )
    
    # Simulate training process
    print("\nSimulating training process...")
    
    for epoch in range(5):
        # Log epoch start
        auditor.log_training_event(
            epoch=epoch,
            metrics={},
            event_type=EventType.EPOCH_START
        )
        
        # Simulate training metrics
        loss = 1.0 / (epoch + 1) + np.random.random() * 0.1
        accuracy = min(0.95, 0.5 + epoch * 0.1 + np.random.random() * 0.05)
        
        # Log epoch end with metrics
        event = auditor.log_training_event(
            epoch=epoch,
            metrics={
                'loss': loss,
                'accuracy': accuracy,
                'learning_rate': 0.001 * (0.9 ** epoch)
            },
            checkpoint_hash=hashlib.sha256(f"checkpoint_{epoch}".encode()).hexdigest(),
            event_type=EventType.EPOCH_END
        )
        
        print(f"Epoch {epoch}: loss={loss:.4f}, accuracy={accuracy:.4f}")
        
        # Save checkpoint every 2 epochs
        if epoch % 2 == 0:
            auditor.log_training_event(
                epoch=epoch,
                metrics={'checkpoint_saved': True},
                checkpoint_hash=hashlib.sha256(f"checkpoint_{epoch}".encode()).hexdigest(),
                event_type=EventType.CHECKPOINT_SAVE
            )
    
    # Log training end
    auditor.log_training_event(
        epoch=4,
        metrics={'final': True},
        event_type=EventType.TRAINING_END
    )
    
    # Generate proofs
    print("\n" + "=" * 70)
    print("Generating Training Proofs")
    print("=" * 70)
    
    # Merkle proof
    merkle_proof = auditor.generate_training_proof(0, 4, ProofType.MERKLE)
    print(f"\nMerkle proof generated:")
    print(f"  Root hash: {merkle_proof['root_hash'][:40]}...")
    print(f"  Events covered: {merkle_proof['num_events']}")
    
    # ZK proof
    zk_proof = auditor.generate_training_proof(0, 4, ProofType.ZERO_KNOWLEDGE)
    print(f"\nZK proof generated:")
    print(f"  Proof type: {zk_proof['type']}")
    print(f"  Transitions: {zk_proof['num_transitions']}")
    
    # Embed provenance in model
    print("\n" + "=" * 70)
    print("Embedding Provenance")
    print("=" * 70)
    
    model_state = {'weights': 'model_weights_here'}
    model_with_provenance = auditor.embed_provenance(model_state)
    
    provenance = model_with_provenance['metadata']['training_provenance']
    print(f"Provenance embedded:")
    print(f"  Model ID: {provenance['model_id']}")
    print(f"  Events: {provenance['num_events']}")
    print(f"  Epochs: {provenance['epoch_range']}")
    print(f"  Merkle root: {provenance['merkle_root'][:40]}...")
    
    # Verify training history
    print("\n" + "=" * 70)
    print("Verifying Training History")
    print("=" * 70)
    
    claimed_history = [e.to_dict() for e in auditor.events]
    is_valid = auditor.verify_training_history(model_with_provenance, claimed_history)
    print(f"History verification: {'PASSED' if is_valid else 'FAILED'}")
    
    # Query events
    print("\n" + "=" * 70)
    print("Querying Training Events")
    print("=" * 70)
    
    checkpoint_events = auditor.query_events(
        event_types=[EventType.CHECKPOINT_SAVE]
    )
    print(f"Checkpoint saves: {len(checkpoint_events)}")
    
    high_accuracy_events = auditor.query_events(
        filters={'has_accuracy': True}
    )
    print(f"Events with accuracy metric: {len(high_accuracy_events)}")
    
    # Export history
    print("\n" + "=" * 70)
    print("Exporting Training History")
    print("=" * 70)
    
    json_export = auditor.export_history(format='json')
    print(f"JSON export size: {len(json_export)} bytes")
    
    compressed_export = auditor.export_history(format='compressed')
    print(f"Compressed export size: {len(compressed_export)} bytes")
    print(f"Compression ratio: {len(compressed_export) / len(json_export):.2%}")
    
    # Get statistics
    print("\n" + "=" * 70)
    print("Training Statistics")
    print("=" * 70)
    
    stats = auditor.get_statistics()
    print(f"Total events: {stats['total_events']}")
    print(f"Epochs: {stats['epochs']}")
    print(f"Duration: {stats['duration_hours']:.2f} hours")
    print(f"Events per epoch: {stats['events_per_epoch']:.1f}")
    
    if 'loss_stats' in stats:
        print(f"\nLoss statistics:")
        print(f"  Initial: {stats['loss_stats']['max']:.4f}")
        print(f"  Final: {stats['loss_stats']['final']:.4f}")
        print(f"  Mean: {stats['loss_stats']['mean']:.4f}")
    
    if 'accuracy_stats' in stats:
        print(f"\nAccuracy statistics:")
        print(f"  Initial: {stats['accuracy_stats']['min']:.4f}")
        print(f"  Final: {stats['accuracy_stats']['final']:.4f}")
        print(f"  Mean: {stats['accuracy_stats']['mean']:.4f}")
    
    print(f"\nBlockchain transactions: {stats['blockchain_transactions']}")
    print(f"Event types: {stats['event_types']}")
    
    print("\n" + "=" * 70)
    print("Training Provenance Auditor - Complete")
    print("=" * 70)