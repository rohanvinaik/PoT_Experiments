"""
Provenance Integration for PoT Training Pipeline

This module integrates blockchain-based provenance recording with the PoT training
pipeline, providing tamper-evident logging of training checkpoints, validation
results, and complete proof-of-training generation.

Key Features:
- Training checkpoint recording with model hashes and metrics
- Validation result recording with verifier signatures
- Merkle tree construction for efficient batch verification
- Integration with existing PoT verification workflows
- Optional blockchain recording via configuration flags

Usage:
    # Initialize recorder
    recorder = ProvenanceRecorder(enabled=True)
    
    # Record training progress
    recorder.record_training_checkpoint(model_hash, metrics, epoch)
    recorder.record_validation(model_hash, validator_id, result)
    
    # Generate complete proof
    proof = recorder.generate_proof_of_training(model_id)
    
    # Verify proof integrity
    is_valid = recorder.verify_training_provenance(proof)
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from pathlib import Path
import numpy as np

# Import PoT core components
try:
    from .fingerprint import fingerprint_run, FingerprintConfig, FingerprintResult
    from .challenge import generate_challenges, ChallengeConfig
    from .sequential import sequential_verify, SPRTResult
    POT_CORE_AVAILABLE = True
except ImportError:
    POT_CORE_AVAILABLE = False
    logging.warning("PoT core components not available for provenance integration")

# Import blockchain clients
try:
    from ..security.blockchain_factory import get_blockchain_client, test_blockchain_connection
    from ..security.blockchain_client import BlockchainClient
    BLOCKCHAIN_AVAILABLE = True
except ImportError:
    BLOCKCHAIN_AVAILABLE = False
    logging.warning("Blockchain client not available for provenance recording")


@dataclass
class TrainingCheckpoint:
    """Training checkpoint record for provenance tracking."""
    model_hash: str
    epoch: int
    metrics: Dict[str, float]
    timestamp: str
    model_id: str
    checkpoint_id: str
    fingerprint_hash: Optional[str] = None
    validation_hash: Optional[str] = None


@dataclass
class ValidationRecord:
    """Validation result record for provenance tracking."""
    model_hash: str
    validator_id: str
    validation_result: Dict[str, Any]
    timestamp: str
    model_id: str
    validation_id: str
    confidence_score: Optional[float] = None
    challenge_hash: Optional[str] = None


@dataclass
class ProvenanceProof:
    """Complete proof-of-training with provenance chain."""
    model_id: str
    final_model_hash: str
    training_chain: List[TrainingCheckpoint]
    validation_chain: List[ValidationRecord]
    merkle_root: str
    blockchain_transactions: List[str]
    proof_timestamp: str
    verification_metadata: Dict[str, Any]
    signature_hash: str


class MerkleTree:
    """Simple Merkle tree implementation for batch provenance verification."""
    
    def __init__(self, data: List[str]):
        """
        Initialize Merkle tree with list of hash strings.
        
        Args:
            data: List of hash strings to build tree from
        """
        self.data = data
        self.tree = self._build_tree(data)
        self.root = self.tree[0] if self.tree else None
    
    def _hash_pair(self, left: str, right: str) -> str:
        """Hash a pair of values."""
        combined = left + right
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _build_tree(self, data: List[str]) -> List[str]:
        """Build Merkle tree from bottom up."""
        if not data:
            return []
        
        if len(data) == 1:
            return data
        
        # Make data length even by duplicating last element if needed
        if len(data) % 2 == 1:
            data = data + [data[-1]]
        
        # Build next level
        next_level = []
        for i in range(0, len(data), 2):
            next_level.append(self._hash_pair(data[i], data[i + 1]))
        
        # Recursively build upper levels
        upper_tree = self._build_tree(next_level)
        return upper_tree + next_level + data
    
    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        """
        Get Merkle proof for data at given index.
        
        Args:
            index: Index of data element
            
        Returns:
            List of (hash, position) tuples for verification path
        """
        if not self.data or index >= len(self.data):
            return []
        
        proof = []
        current_index = index
        level_start = len(self.tree) - len(self.data)
        
        # Work up the tree
        while level_start > 0:
            level_size = len(self.data) // (2 ** (len(self.tree) - level_start - len(self.data)))
            
            # Find sibling
            if current_index % 2 == 0:
                sibling_index = current_index + 1
                position = "right"
            else:
                sibling_index = current_index - 1
                position = "left"
            
            if sibling_index < level_size:
                sibling_hash = self.tree[level_start + sibling_index]
                proof.append((sibling_hash, position))
            
            current_index = current_index // 2
            level_start -= level_size // 2
        
        return proof
    
    def verify_proof(self, data_hash: str, index: int, proof: List[Tuple[str, str]]) -> bool:
        """
        Verify Merkle proof for given data.
        
        Args:
            data_hash: Hash of the data element
            index: Index of the data element
            proof: Merkle proof from get_proof()
            
        Returns:
            True if proof is valid
        """
        current_hash = data_hash
        current_index = index
        
        for sibling_hash, position in proof:
            if position == "left":
                current_hash = self._hash_pair(sibling_hash, current_hash)
            else:
                current_hash = self._hash_pair(current_hash, sibling_hash)
            current_index = current_index // 2
        
        return current_hash == self.root


@dataclass
class ProvenanceConfig:
    """Configuration for provenance recording."""
    enabled: bool = False
    blockchain_enabled: bool = False
    local_storage_path: str = "./provenance_records.json"
    batch_size: int = 10
    auto_verify: bool = True
    fingerprint_checkpoints: bool = True
    record_challenges: bool = True
    client_config: Optional[Dict[str, Any]] = None


class ProvenanceRecorder:
    """
    Main class for recording training provenance with blockchain integration.
    
    This class wraps blockchain clients and provides high-level methods for
    recording training checkpoints, validation results, and generating complete
    proof-of-training artifacts.
    """
    
    def __init__(self, config: Optional[ProvenanceConfig] = None):
        """
        Initialize provenance recorder.
        
        Args:
            config: Provenance configuration. If None, uses defaults.
        """
        self.config = config or ProvenanceConfig()
        self.logger = logging.getLogger(__name__)
        
        # Storage for provenance records
        self.training_checkpoints: List[TrainingCheckpoint] = []
        self.validation_records: List[ValidationRecord] = []
        self.blockchain_transactions: List[str] = []
        
        # Blockchain client (optional)
        self.blockchain_client: Optional[BlockchainClient] = None
        self._initialize_blockchain_client()
        
        # Local storage
        self.local_storage_path = Path(self.config.local_storage_path)
        
        self.logger.info(f"ProvenanceRecorder initialized (enabled={self.config.enabled})")
    
    def _initialize_blockchain_client(self) -> None:
        """Initialize blockchain client if enabled and available."""
        if not self.config.enabled or not self.config.blockchain_enabled:
            return
        
        if not BLOCKCHAIN_AVAILABLE:
            self.logger.warning("Blockchain integration requested but not available")
            return
        
        try:
            client_config = self.config.client_config or {}
            self.blockchain_client = get_blockchain_client(**client_config)
            
            # Test connection
            success, results = test_blockchain_connection(self.blockchain_client)
            if success:
                self.logger.info(f"Blockchain client initialized: {results['client_type']}")
            else:
                self.logger.warning("Blockchain connection test failed")
                self.blockchain_client = None
                
        except Exception as e:
            self.logger.error(f"Failed to initialize blockchain client: {str(e)}")
            self.blockchain_client = None
    
    def record_training_checkpoint(
        self, 
        model_hash: str, 
        metrics: Dict[str, float], 
        epoch: int,
        model_id: str = "default_model",
        model_fn: Optional[Callable] = None
    ) -> str:
        """
        Record a training checkpoint with optional fingerprinting.
        
        Args:
            model_hash: Hash of the model weights
            metrics: Training metrics (loss, accuracy, etc.)
            epoch: Training epoch number
            model_id: Identifier for the model being trained
            model_fn: Optional model function for fingerprinting
            
        Returns:
            Checkpoint ID for reference
        """
        if not self.config.enabled:
            return ""
        
        checkpoint_id = hashlib.sha256(
            f"{model_id}_{epoch}_{model_hash}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Generate fingerprint if requested and possible
        fingerprint_hash = None
        if self.config.fingerprint_checkpoints and model_fn and POT_CORE_AVAILABLE:
            try:
                fingerprint_hash = self._generate_model_fingerprint(model_fn, model_id)
            except Exception as e:
                self.logger.warning(f"Failed to generate fingerprint: {str(e)}")
        
        # Create checkpoint record
        checkpoint = TrainingCheckpoint(
            model_hash=model_hash,
            epoch=epoch,
            metrics=metrics,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_id=model_id,
            checkpoint_id=checkpoint_id,
            fingerprint_hash=fingerprint_hash
        )
        
        self.training_checkpoints.append(checkpoint)
        
        # Record to blockchain if available
        if self.blockchain_client:
            try:
                tx_id = self._record_to_blockchain(checkpoint, "training_checkpoint")
                self.blockchain_transactions.append(tx_id)
                self.logger.info(f"Recorded checkpoint to blockchain: {tx_id}")
            except Exception as e:
                self.logger.error(f"Failed to record checkpoint to blockchain: {str(e)}")
        
        # Save to local storage
        self._save_local_records()
        
        self.logger.info(f"Recorded training checkpoint: epoch {epoch}, checkpoint_id {checkpoint_id}")
        return checkpoint_id
    
    def record_validation(
        self, 
        model_hash: str, 
        validator_id: str, 
        validation_result: Dict[str, Any],
        model_id: str = "default_model",
        challenge_config: Optional[ChallengeConfig] = None
    ) -> str:
        """
        Record a validation result with optional challenge verification.
        
        Args:
            model_hash: Hash of the validated model
            validator_id: Identifier of the validator
            validation_result: Validation results and metrics
            model_id: Identifier for the model being validated
            challenge_config: Optional challenge configuration for verification
            
        Returns:
            Validation ID for reference
        """
        if not self.config.enabled:
            return ""
        
        validation_id = hashlib.sha256(
            f"{model_id}_{validator_id}_{model_hash}_{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Generate challenge hash if requested
        challenge_hash = None
        if self.config.record_challenges and challenge_config and POT_CORE_AVAILABLE:
            try:
                challenge_hash = self._generate_challenge_hash(challenge_config)
            except Exception as e:
                self.logger.warning(f"Failed to generate challenge hash: {str(e)}")
        
        # Create validation record
        validation = ValidationRecord(
            model_hash=model_hash,
            validator_id=validator_id,
            validation_result=validation_result,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_id=model_id,
            validation_id=validation_id,
            confidence_score=validation_result.get("confidence"),
            challenge_hash=challenge_hash
        )
        
        self.validation_records.append(validation)
        
        # Record to blockchain if available
        if self.blockchain_client:
            try:
                tx_id = self._record_to_blockchain(validation, "validation_record")
                self.blockchain_transactions.append(tx_id)
                self.logger.info(f"Recorded validation to blockchain: {tx_id}")
            except Exception as e:
                self.logger.error(f"Failed to record validation to blockchain: {str(e)}")
        
        # Save to local storage
        self._save_local_records()
        
        self.logger.info(f"Recorded validation: validator {validator_id}, validation_id {validation_id}")
        return validation_id
    
    def generate_proof_of_training(self, model_id: str) -> ProvenanceProof:
        """
        Generate complete proof-of-training with Merkle tree verification.
        
        Args:
            model_id: Identifier for the model
            
        Returns:
            Complete provenance proof
        """
        if not self.config.enabled:
            raise ValueError("Provenance recording not enabled")
        
        # Filter records for this model
        model_checkpoints = [cp for cp in self.training_checkpoints if cp.model_id == model_id]
        model_validations = [vr for vr in self.validation_records if vr.model_id == model_id]
        
        if not model_checkpoints:
            raise ValueError(f"No training checkpoints found for model {model_id}")
        
        # Get final model hash
        final_checkpoint = max(model_checkpoints, key=lambda x: x.epoch)
        final_model_hash = final_checkpoint.model_hash
        
        # Build Merkle tree for verification
        all_hashes = (
            [cp.checkpoint_id for cp in model_checkpoints] +
            [vr.validation_id for vr in model_validations]
        )
        merkle_tree = MerkleTree(all_hashes)
        
        # Create verification metadata
        verification_metadata = {
            "total_checkpoints": len(model_checkpoints),
            "total_validations": len(model_validations),
            "training_epochs": [cp.epoch for cp in model_checkpoints],
            "final_metrics": final_checkpoint.metrics,
            "validator_ids": list(set(vr.validator_id for vr in model_validations)),
            "blockchain_enabled": self.blockchain_client is not None,
            "merkle_tree_size": len(all_hashes)
        }
        
        # Generate signature hash
        proof_content = {
            "model_id": model_id,
            "final_model_hash": final_model_hash,
            "merkle_root": merkle_tree.root or "",
            "metadata": verification_metadata
        }
        signature_hash = hashlib.sha256(
            json.dumps(proof_content, sort_keys=True).encode()
        ).hexdigest()
        
        # Create proof
        proof = ProvenanceProof(
            model_id=model_id,
            final_model_hash=final_model_hash,
            training_chain=model_checkpoints,
            validation_chain=model_validations,
            merkle_root=merkle_tree.root or "",
            blockchain_transactions=self.blockchain_transactions.copy(),
            proof_timestamp=datetime.now(timezone.utc).isoformat(),
            verification_metadata=verification_metadata,
            signature_hash=signature_hash
        )
        
        self.logger.info(f"Generated proof-of-training for model {model_id}")
        return proof
    
    def verify_training_provenance(self, proof: ProvenanceProof) -> bool:
        """
        Verify the integrity of a training provenance proof.
        
        Args:
            proof: Provenance proof to verify
            
        Returns:
            True if proof is valid and complete
        """
        try:
            # Verify signature hash
            proof_content = {
                "model_id": proof.model_id,
                "final_model_hash": proof.final_model_hash,
                "merkle_root": proof.merkle_root,
                "metadata": proof.verification_metadata
            }
            expected_signature = hashlib.sha256(
                json.dumps(proof_content, sort_keys=True).encode()
            ).hexdigest()
            
            if expected_signature != proof.signature_hash:
                self.logger.error("Proof signature verification failed")
                return False
            
            # Verify Merkle tree
            all_hashes = (
                [cp.checkpoint_id for cp in proof.training_chain] +
                [vr.validation_id for vr in proof.validation_chain]
            )
            merkle_tree = MerkleTree(all_hashes)
            
            if merkle_tree.root != proof.merkle_root:
                self.logger.error("Merkle tree verification failed")
                return False
            
            # Verify blockchain transactions if available
            if self.blockchain_client and proof.blockchain_transactions:
                for tx_id in proof.blockchain_transactions:
                    try:
                        self.blockchain_client.retrieve_hash(tx_id)
                    except Exception as e:
                        self.logger.error(f"Blockchain transaction verification failed: {tx_id}")
                        return False
            
            # Verify training progression
            if not self._verify_training_progression(proof.training_chain):
                self.logger.error("Training progression verification failed")
                return False
            
            self.logger.info(f"Provenance proof verified successfully for model {proof.model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Proof verification failed: {str(e)}")
            return False
    
    def _generate_model_fingerprint(self, model_fn: Callable, model_id: str) -> str:
        """Generate fingerprint hash for model."""
        if not POT_CORE_AVAILABLE:
            return ""
        
        try:
            # Create basic fingerprint config
            config = FingerprintConfig(
                challenge_family="vision:freq",  # Default family
                io_only=True,  # Fast fingerprinting
                seed=42
            )
            
            # Generate fingerprint (this would need actual challenges in practice)
            result = fingerprint_run(model_fn, [], config)
            return result.io_hash
            
        except Exception as e:
            self.logger.warning(f"Fingerprint generation failed: {str(e)}")
            return ""
    
    def _generate_challenge_hash(self, challenge_config: ChallengeConfig) -> str:
        """Generate hash for challenge configuration."""
        config_str = json.dumps(asdict(challenge_config), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _record_to_blockchain(self, record: Union[TrainingCheckpoint, ValidationRecord], record_type: str) -> str:
        """Record data to blockchain."""
        if not self.blockchain_client:
            raise ValueError("Blockchain client not available")
        
        # Prepare metadata
        metadata = {
            "record_type": record_type,
            "timestamp": record.timestamp,
            "model_id": record.model_id,
            **asdict(record)
        }
        
        # Create hash of the record
        record_hash = hashlib.sha256(
            json.dumps(metadata, sort_keys=True).encode()
        ).hexdigest()
        
        # Store on blockchain
        return self.blockchain_client.store_hash(record_hash, metadata)
    
    def _save_local_records(self) -> None:
        """Save records to local storage."""
        try:
            # Ensure parent directory exists
            self.local_storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data
            data = {
                "training_checkpoints": [asdict(cp) for cp in self.training_checkpoints],
                "validation_records": [asdict(vr) for vr in self.validation_records],
                "blockchain_transactions": self.blockchain_transactions,
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            # Write to file
            with open(self.local_storage_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save local records: {str(e)}")
    
    def _verify_training_progression(self, checkpoints: List[TrainingCheckpoint]) -> bool:
        """Verify that training progression makes sense."""
        if len(checkpoints) < 2:
            return True
        
        # Sort by epoch
        sorted_checkpoints = sorted(checkpoints, key=lambda x: x.epoch)
        
        # Check epoch progression
        for i in range(1, len(sorted_checkpoints)):
            if sorted_checkpoints[i].epoch <= sorted_checkpoints[i-1].epoch:
                return False
        
        # Check that model hashes change (models should evolve)
        unique_hashes = set(cp.model_hash for cp in sorted_checkpoints)
        if len(unique_hashes) < len(sorted_checkpoints) // 2:
            # Too many identical hashes suggests no training progress
            return False
        
        return True
    
    def load_local_records(self) -> None:
        """Load records from local storage."""
        if not self.local_storage_path.exists():
            return
        
        try:
            with open(self.local_storage_path, 'r') as f:
                data = json.load(f)
            
            # Load checkpoints
            self.training_checkpoints = [
                TrainingCheckpoint(**cp) for cp in data.get("training_checkpoints", [])
            ]
            
            # Load validations
            self.validation_records = [
                ValidationRecord(**vr) for vr in data.get("validation_records", [])
            ]
            
            # Load blockchain transactions
            self.blockchain_transactions = data.get("blockchain_transactions", [])
            
            self.logger.info(f"Loaded {len(self.training_checkpoints)} checkpoints and {len(self.validation_records)} validations")
            
        except Exception as e:
            self.logger.error(f"Failed to load local records: {str(e)}")
    
    def get_model_history(self, model_id: str) -> Dict[str, Any]:
        """Get complete training and validation history for a model."""
        model_checkpoints = [cp for cp in self.training_checkpoints if cp.model_id == model_id]
        model_validations = [vr for vr in self.validation_records if vr.model_id == model_id]
        
        return {
            "model_id": model_id,
            "checkpoints": model_checkpoints,
            "validations": model_validations,
            "total_epochs": len(model_checkpoints),
            "total_validations": len(model_validations),
            "blockchain_transactions": len(self.blockchain_transactions)
        }
    
    def clear_records(self, model_id: Optional[str] = None) -> None:
        """Clear provenance records, optionally for specific model."""
        if model_id:
            self.training_checkpoints = [cp for cp in self.training_checkpoints if cp.model_id != model_id]
            self.validation_records = [vr for vr in self.validation_records if vr.model_id != model_id]
        else:
            self.training_checkpoints.clear()
            self.validation_records.clear()
            self.blockchain_transactions.clear()
        
        self._save_local_records()
        self.logger.info(f"Cleared records for model: {model_id or 'all models'}")


# Integration helper functions

def create_provenance_recorder(
    enabled: bool = True,
    blockchain_enabled: bool = False,
    **kwargs
) -> ProvenanceRecorder:
    """
    Factory function to create provenance recorder with common configurations.
    
    Args:
        enabled: Whether to enable provenance recording
        blockchain_enabled: Whether to enable blockchain integration
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured ProvenanceRecorder instance
    """
    config = ProvenanceConfig(
        enabled=enabled,
        blockchain_enabled=blockchain_enabled,
        **kwargs
    )
    return ProvenanceRecorder(config)


def integrate_with_training_loop(
    recorder: ProvenanceRecorder,
    model_fn: Callable,
    model_id: str = "training_model"
) -> Callable:
    """
    Decorator to integrate provenance recording with training loops.
    
    Args:
        recorder: ProvenanceRecorder instance
        model_fn: Model function for fingerprinting
        model_id: Identifier for the model
        
    Returns:
        Decorator function
    """
    def decorator(training_function: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Extract epoch and metrics from training function
            result = training_function(*args, **kwargs)
            
            # Record checkpoint if result contains required fields
            if isinstance(result, dict) and "epoch" in result and "model_hash" in result:
                recorder.record_training_checkpoint(
                    model_hash=result["model_hash"],
                    metrics=result.get("metrics", {}),
                    epoch=result["epoch"],
                    model_id=model_id,
                    model_fn=model_fn
                )
            
            return result
        return wrapper
    return decorator