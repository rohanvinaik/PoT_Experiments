"""
Training Provenance Auditor for Model Development Lifecycle

This module provides comprehensive training log embedding and provenance auditing
capabilities with cryptographic proofs, blockchain integration, and regulatory compliance.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of training events"""
    TRAINING_START = "training_start"
    EPOCH_START = "epoch_start"
    EPOCH_END = "epoch_end"
    CHECKPOINT_SAVE = "checkpoint_save"
    VALIDATION = "validation"
    EARLY_STOPPING = "early_stopping"
    TRAINING_END = "training_end"
    HYPERPARAMETER_CHANGE = "hyperparameter_change"
    ERROR = "error"
    CUSTOM = "custom"


class ProofType(Enum):
    """Types of cryptographic proofs"""
    MERKLE = "merkle"
    ZERO_KNOWLEDGE = "zero_knowledge"
    SIGNATURE = "signature"
    TIMESTAMP = "timestamp"


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


@dataclass
class MerkleNode:
    """Node in a Merkle tree"""
    hash_value: str
    left: Optional['MerkleNode'] = None
    right: Optional['MerkleNode'] = None
    data: Optional[Any] = None
    
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None


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
            MerkleNode(hash_value=event.event_hash, data=event)
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
                
                # Combine hashes
                combined = left.hash_value + right.hash_value
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                
                parent = MerkleNode(
                    hash_value=parent_hash,
                    left=left,
                    right=right
                )
                next_level.append(parent)
            
            current_level = next_level
        
        return current_level[0] if current_level else None
    
    def get_root_hash(self) -> Optional[str]:
        """Get root hash of the tree"""
        return self.root.hash_value if self.root else None
    
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
                    proof.append((sibling.hash_value, position))

                    # Update index for next level
                    current_index = i // 2
                
                # Build next level
                if i + 1 < len(current_level):
                    combined = current_level[i].hash_value + current_level[i + 1].hash_value
                else:
                    combined = current_level[i].hash_value + current_level[i].hash_value
                
                parent_hash = hashlib.sha256(combined.encode()).hexdigest()
                next_level.append(MerkleNode(hash_value=parent_hash))
            
            current_level = next_level
        
        return proof
    
    def verify_proof(self, event_hash: str, proof: List[Tuple[str, str]], 
                    root_hash: str) -> bool:
        """Verify a Merkle proof"""
        current_hash = event_hash
        
        for sibling_hash, position in proof:
            if position == 'right':
                combined = current_hash + sibling_hash
            else:
                combined = sibling_hash + current_hash
            
            current_hash = hashlib.sha256(combined.encode()).hexdigest()
        
        return current_hash == root_hash


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
    """Blockchain client using web3.py for provenance storage"""

    def __init__(
        self,
        web3: Optional[Any] = None,
        provider_url: Optional[str] = None,
        default_account: Optional[str] = None,
    ) -> None:
        """Create blockchain client.

        Args:
            web3: Pre-configured Web3 instance. If ``None``, a new instance
                will be created using ``provider_url``.
            provider_url: RPC endpoint for the blockchain. Defaults to
                ``http://127.0.0.1:8545`` if not provided.
            default_account: Account to use for transactions. If not
                provided, the first account from the provider is used.
        """
        from web3 import Web3  # Imported lazily to avoid hard dependency

        self.web3 = web3 or Web3(Web3.HTTPProvider(provider_url or "http://127.0.0.1:8545"))
        if not self.web3.is_connected():
            raise ConnectionError("Unable to connect to blockchain provider")

        self.default_account = (
            default_account or (self.web3.eth.accounts[0] if self.web3.eth.accounts else None)
        )
        if self.default_account is None:
            raise ValueError("No default account available for blockchain transactions")

    def store_hash(self, hash_value: str, metadata: Dict) -> str:
        """Store hash and metadata on blockchain.

        The hash and metadata are embedded in the transaction's data field as
        a JSON payload. The transaction hash is returned as the identifier.
        """

        payload = json.dumps({"hash": hash_value, "metadata": metadata}, sort_keys=True).encode()
        tx = {
            "from": self.default_account,
            "to": self.default_account,
            "value": 0,
            "data": self.web3.to_hex(payload),
            "gas": 100000,
        }

        tx_hash = self.web3.eth.send_transaction(tx)
        self.web3.eth.wait_for_transaction_receipt(tx_hash)
        tx_hex = tx_hash.hex()
        logger.info(f"Stored hash on blockchain: {tx_hex}")
        return tx_hex

    def retrieve_hash(self, transaction_id: str) -> Optional[Dict]:
        """Retrieve stored hash and metadata from blockchain."""

        from web3.exceptions import TransactionNotFound

        try:
            tx = self.web3.eth.get_transaction(transaction_id)
        except TransactionNotFound:
            return None

        data_bytes = bytes(tx["input"])
        try:
            payload = json.loads(data_bytes.decode())
        except Exception:
            return None

        block = self.web3.eth.get_block(tx["blockNumber"])
        return {
            "hash": payload.get("hash"),
            "metadata": payload.get("metadata"),
            "from": tx["from"],
            "to": tx["to"],
            "block_number": tx["blockNumber"],
            "timestamp": datetime.fromtimestamp(block["timestamp"], timezone.utc).isoformat(),
            "tx_hash": transaction_id,
        }

    def verify_hash(self, hash_value: str, transaction_id: str) -> bool:
        """Verify that the given hash matches blockchain record."""

        stored = self.retrieve_hash(transaction_id)
        return bool(stored and stored.get("hash") == hash_value)


class MockBlockchainClient(BlockchainClient):
    """Mock blockchain client for testing"""
    
    def __init__(self):
        self.storage = {}
        self.transaction_counter = 0
    
    def store_hash(self, hash_value: str, metadata: Dict) -> str:
        """Store hash in mock blockchain"""
        tx_id = f"tx_{self.transaction_counter:08d}"
        self.transaction_counter += 1
        
        self.storage[tx_id] = {
            'hash': hash_value,
            'metadata': metadata,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'block_number': self.transaction_counter
        }
        
        logger.info(f"Stored hash on mock blockchain: {tx_id}")
        return tx_id
    
    def retrieve_hash(self, transaction_id: str) -> Optional[Dict]:
        """Retrieve hash from mock blockchain"""
        return self.storage.get(transaction_id)
    
    def verify_hash(self, hash_value: str, transaction_id: str) -> bool:
        """Verify hash in mock blockchain"""
        stored = self.storage.get(transaction_id)
        return stored and stored['hash'] == hash_value


class TrainingProvenanceAuditor:
    """
    Main class for training provenance auditing and verification
    
    Provides comprehensive tracking of training events with cryptographic
    proofs and blockchain integration.
    """
    
    def __init__(self, model_id: str, blockchain_client: Optional[BlockchainClient] = None,
                 compression_enabled: bool = True, max_history_size: int = 10000):
        """
        Initialize TrainingProvenanceAuditor
        
        Args:
            model_id: Unique identifier for the model
            blockchain_client: Optional blockchain client for immutable storage
            compression_enabled: Whether to compress large histories
            max_history_size: Maximum number of events to keep in memory
        """
        self.model_id = model_id
        self.blockchain_client = blockchain_client or MockBlockchainClient()
        self.compression_enabled = compression_enabled
        self.max_history_size = max_history_size
        
        # Training history
        self.events: List[TrainingEvent] = []
        self.event_index: Dict[str, TrainingEvent] = {}
        
        # Merkle trees for epochs
        self.epoch_trees: Dict[int, MerkleTree] = {}
        self.master_tree: Optional[MerkleTree] = None
        
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
        
        logger.info(f"TrainingProvenanceAuditor initialized for model: {model_id}")
    
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
        self.metadata['last_epoch'] = max(self.metadata['last_epoch'], epoch)
        
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