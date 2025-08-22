"""
ZK Training Prover - Wrapper for automatic ZK proof generation during training

This module provides a high-level wrapper that automatically generates
zero-knowledge proofs during model training while maintaining dual
Merkle trees (SHA-256 and Poseidon).
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
import numpy as np

from pot.prototypes.training_provenance_auditor_zk import (
    TrainingProvenanceAuditorZK,
    ZKProofRecord
)
from pot.zk.zk_types import SGDStepStatement, SGDStepWitness
from pot.zk.builder import SimpleMerkleTree

logger = logging.getLogger(__name__)


def poseidon_hash(data: bytes) -> bytes:
    """
    Mock Poseidon hash function for testing
    In production, this would use the actual Poseidon implementation
    """
    # For now, use SHA-256 with a different prefix to simulate Poseidon
    return hashlib.sha256(b"POSEIDON:" + data).digest()


class ZKTrainingProver:
    """
    High-level wrapper for automatic ZK proof generation during training
    
    This class wraps TrainingProvenanceAuditor and automatically:
    - Generates ZK proofs during training steps
    - Maintains Poseidon Merkle trees alongside SHA-256
    - Provides simple API for training frameworks
    - Handles proof storage and verification
    """
    
    def __init__(self,
                 model_id: str,
                 proof_frequency: int = 10,
                 enable_blockchain: bool = False,
                 enable_dual_trees: bool = True,
                 zk_params_k: int = 10,
                 auto_verify: bool = True,
                 save_proofs_to_disk: bool = True,
                 proof_dir: Optional[Path] = None):
        """
        Initialize ZK Training Prover
        
        Args:
            model_id: Unique identifier for the model
            proof_frequency: Generate proof every N steps (0 = disabled)
            enable_blockchain: Whether to store proof hashes on blockchain
            enable_dual_trees: Maintain both SHA-256 and Poseidon trees
            zk_params_k: ZK circuit parameter size
            auto_verify: Automatically verify proofs after generation
            save_proofs_to_disk: Save proofs to disk for later verification
            proof_dir: Directory to save proofs (default: ./zk_proofs/{model_id})
        """
        self.model_id = model_id
        self.proof_frequency = proof_frequency
        self.enable_dual_trees = enable_dual_trees
        self.auto_verify = auto_verify
        self.save_proofs_to_disk = save_proofs_to_disk
        
        # Initialize the enhanced auditor
        from pot.prototypes.training_provenance_auditor import MockBlockchainClient
        blockchain_client = MockBlockchainClient() if enable_blockchain else None
        
        self.auditor = TrainingProvenanceAuditorZK(
            model_id=model_id,
            blockchain_client=blockchain_client,
            zk_enabled=(proof_frequency > 0),
            zk_proof_frequency=proof_frequency,
            dual_commitment_mode=enable_dual_trees,
            zk_params_k=zk_params_k
        )
        
        # Proof storage directory
        if save_proofs_to_disk:
            self.proof_dir = proof_dir or Path(f"./zk_proofs/{model_id}")
            self.proof_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.proof_dir = None
        
        # Poseidon Merkle trees
        if enable_dual_trees:
            self.poseidon_trees: Dict[int, SimpleMerkleTree] = {}
            self.poseidon_master_tree: Optional[SimpleMerkleTree] = None
        
        # Training state tracking
        self.current_epoch = 0
        self.current_step = 0
        self.total_steps = 0
        self.proof_count = 0
        self.verification_failures = 0
        
        # Performance metrics
        self.proof_generation_times: List[float] = []
        self.verification_times: List[float] = []
        
        logger.info(f"ZKTrainingProver initialized for model {model_id}")
        logger.info(f"Proof frequency: every {proof_frequency} steps")
        logger.info(f"Dual trees: {enable_dual_trees}, Auto-verify: {auto_verify}")
    
    def log_training_step(self,
                          epoch: int,
                          step: int,
                          loss: float,
                          accuracy: Optional[float] = None,
                          weights_before: Optional[Dict[str, np.ndarray]] = None,
                          weights_after: Optional[Dict[str, np.ndarray]] = None,
                          batch_data: Optional[Dict[str, np.ndarray]] = None,
                          learning_rate: float = 0.01,
                          additional_metrics: Optional[Dict[str, Any]] = None) -> Optional[ZKProofRecord]:
        """
        Log a training step with automatic ZK proof generation
        
        Args:
            epoch: Current epoch number
            step: Step number within epoch
            loss: Training loss value
            accuracy: Optional accuracy metric
            weights_before: Model weights before update (required for ZK proofs)
            weights_after: Model weights after update (required for ZK proofs)
            batch_data: Training batch (required for ZK proofs)
            learning_rate: Learning rate used
            additional_metrics: Additional metrics to log
            
        Returns:
            ZKProofRecord if proof was generated, None otherwise
        """
        self.current_epoch = epoch
        self.current_step = step
        self.total_steps += 1
        
        # Prepare metrics
        metrics = {'loss': loss}
        if accuracy is not None:
            metrics['accuracy'] = accuracy
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Check if we should generate a proof
        # Note: We check (total_steps + 1) because we haven't incremented yet
        should_generate_proof = (
            self.proof_frequency > 0 and
            (self.total_steps + 1) % self.proof_frequency == 0 and
            weights_before is not None and
            weights_after is not None and
            batch_data is not None
        )
        
        if should_generate_proof:
            # Log with ZK proof generation
            start_time = time.time()
            
            event, zk_proof = self.auditor.log_training_step_with_zk(
                epoch=epoch,
                step=step,
                metrics=metrics,
                weights_before=weights_before,
                weights_after=weights_after,
                batch_data=batch_data,
                learning_rate=learning_rate
            )
            
            proof_time = time.time() - start_time
            self.proof_generation_times.append(proof_time)
            
            if zk_proof:
                self.proof_count += 1
                
                # Update Poseidon tree if dual mode enabled
                if self.enable_dual_trees:
                    self._update_poseidon_tree(epoch, step, zk_proof)
                
                # Auto-verify if enabled
                if self.auto_verify:
                    verify_start = time.time()
                    is_valid = self._verify_proof(zk_proof)
                    verify_time = time.time() - verify_start
                    self.verification_times.append(verify_time)
                    
                    if not is_valid:
                        self.verification_failures += 1
                        logger.warning(f"Proof verification failed for step {step}")
                
                # Save proof to disk if enabled
                if self.save_proofs_to_disk and self.proof_dir:
                    self._save_proof_to_disk(zk_proof)
                
                logger.info(f"Generated ZK proof #{self.proof_count} in {proof_time:.2f}s")
                return zk_proof
        else:
            # Log without ZK proof
            from pot.prototypes.training_provenance_auditor import EventType
            event = self.auditor.log_training_event(
                epoch=epoch,
                metrics=metrics,
                event_type=EventType.CHECKPOINT_SAVE
            )
        
        return None
    
    def _update_poseidon_tree(self, epoch: int, step: int, proof: ZKProofRecord):
        """Update Poseidon Merkle tree with new proof"""
        if epoch not in self.poseidon_trees:
            self.poseidon_trees[epoch] = SimpleMerkleTree([], hash_function=poseidon_hash)
        
        # Add proof hash to epoch tree
        proof_data = json.dumps({
            'step': step,
            'proof_hash': proof.proof_hash,
            'timestamp': proof.timestamp.isoformat()
        }).encode()
        
        self.poseidon_trees[epoch].leaves.append(proof_data)
        
        # Update master tree with epoch roots
        epoch_roots = []
        for e in sorted(self.poseidon_trees.keys()):
            root = self.poseidon_trees[e].compute_root()
            epoch_roots.append(root)
        
        self.poseidon_master_tree = SimpleMerkleTree(epoch_roots, hash_function=poseidon_hash)
    
    def _verify_proof(self, proof: ZKProofRecord) -> bool:
        """Verify a ZK proof"""
        if not self.auditor.zk_verifier:
            return False
        
        try:
            return self.auditor.zk_verifier.verify_sgd_step(
                proof.statement,
                proof.proof_data
            )
        except Exception as e:
            logger.error(f"Proof verification error: {e}")
            return False
    
    def _save_proof_to_disk(self, proof: ZKProofRecord):
        """Save proof to disk for later verification"""
        proof_file = self.proof_dir / f"proof_epoch{proof.epoch}_step{proof.step_number}.json"
        
        proof_data = {
            'model_id': self.model_id,
            'epoch': proof.epoch,
            'step_number': proof.step_number,
            'proof_hash': proof.proof_hash,
            'proof_data': proof.proof_data.hex() if isinstance(proof.proof_data, bytes) else str(proof.proof_data),
            'timestamp': proof.timestamp.isoformat(),
            'verification_status': proof.verification_status,
            'blockchain_tx': proof.blockchain_tx,
            'statement': {
                'W_t_root': proof.statement.W_t_root.hex() if isinstance(proof.statement.W_t_root, bytes) else str(proof.statement.W_t_root),
                'batch_root': proof.statement.batch_root.hex() if isinstance(proof.statement.batch_root, bytes) else str(proof.statement.batch_root),
                'hparams_hash': proof.statement.hparams_hash.hex() if isinstance(proof.statement.hparams_hash, bytes) else str(proof.statement.hparams_hash),
                'W_t1_root': proof.statement.W_t1_root.hex() if isinstance(proof.statement.W_t1_root, bytes) else str(proof.statement.W_t1_root),
                'step_nonce': proof.statement.step_nonce,
                'step_number': proof.statement.step_number,
                'epoch': proof.statement.epoch
            }
        }
        
        with open(proof_file, 'w') as f:
            json.dump(proof_data, f, indent=2)
    
    def finish_epoch(self, epoch: int) -> Dict[str, Any]:
        """
        Finalize epoch and generate summary
        
        Returns:
            Summary statistics for the epoch
        """
        # Generate epoch commitment
        epoch_commitment = self.auditor.commit_to_blockchain(
            f"Epoch {epoch} completed with {self.current_step} steps"
        )
        
        # Get Poseidon root if dual mode
        poseidon_root = None
        if self.enable_dual_trees and epoch in self.poseidon_trees:
            poseidon_root = self.poseidon_trees[epoch].compute_root().hex()
        
        summary = {
            'epoch': epoch,
            'total_steps': self.current_step,
            'proofs_generated': self.proof_count,
            'verification_failures': self.verification_failures,
            'sha256_root': self.auditor.merkle_trees.get(epoch, {}).get('root', None),
            'poseidon_root': poseidon_root,
            'blockchain_commitment': epoch_commitment,
            'avg_proof_time': np.mean(self.proof_generation_times) if self.proof_generation_times else 0,
            'avg_verify_time': np.mean(self.verification_times) if self.verification_times else 0
        }
        
        logger.info(f"Epoch {epoch} summary: {summary}")
        return summary
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary
        
        Returns:
            Dictionary with training statistics and proof information
        """
        return {
            'model_id': self.model_id,
            'total_epochs': self.current_epoch,
            'total_steps': self.total_steps,
            'total_proofs': self.proof_count,
            'verification_failures': self.verification_failures,
            'proof_frequency': self.proof_frequency,
            'dual_trees_enabled': self.enable_dual_trees,
            'avg_proof_generation_time': np.mean(self.proof_generation_times) if self.proof_generation_times else 0,
            'avg_verification_time': np.mean(self.verification_times) if self.verification_times else 0,
            'sha256_master_root': self.auditor.master_tree.root if self.auditor.master_tree else None,
            'poseidon_master_root': self.poseidon_master_tree.compute_root().hex() if self.poseidon_master_tree else None,
            'proof_save_dir': str(self.proof_dir) if self.proof_dir else None
        }
    
    def verify_all_proofs(self) -> Tuple[int, int]:
        """
        Verify all generated proofs
        
        Returns:
            Tuple of (valid_count, invalid_count)
        """
        valid = 0
        invalid = 0
        
        for proof_record in self.auditor.zk_proofs:
            if self._verify_proof(proof_record):
                valid += 1
            else:
                invalid += 1
        
        logger.info(f"Proof verification complete: {valid} valid, {invalid} invalid")
        return valid, invalid
    
    def export_provenance(self, output_file: Path) -> Dict[str, Any]:
        """
        Export complete provenance data including ZK proofs
        
        Args:
            output_file: Path to save provenance data
            
        Returns:
            Exported provenance dictionary
        """
        provenance = self.auditor.generate_provenance_proof()
        
        # Add ZK proof information
        provenance['zk_proofs'] = {
            'total_proofs': self.proof_count,
            'proof_frequency': self.proof_frequency,
            'verification_failures': self.verification_failures,
            'dual_trees': self.enable_dual_trees,
            'proofs': []
        }
        
        # Include proof summaries
        for proof in self.auditor.zk_proofs[:100]:  # Limit to first 100 for size
            provenance['zk_proofs']['proofs'].append({
                'epoch': proof.epoch,
                'step': proof.step_number,
                'proof_hash': proof.proof_hash,
                'verified': proof.verification_status,
                'blockchain_tx': proof.blockchain_tx
            })
        
        # Add tree roots
        if self.enable_dual_trees:
            provenance['merkle_roots'] = {
                'sha256': self.auditor.master_tree.root if self.auditor.master_tree else None,
                'poseidon': self.poseidon_master_tree.compute_root().hex() if self.poseidon_master_tree else None
            }
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(provenance, f, indent=2, default=str)
        
        logger.info(f"Provenance exported to {output_file}")
        return provenance


class ZKTrainingCallback:
    """
    Training callback for automatic ZK proof generation
    
    Can be integrated with popular training frameworks like PyTorch Lightning,
    Keras, or HuggingFace Transformers.
    """
    
    def __init__(self, zk_prover: ZKTrainingProver):
        """
        Initialize callback with ZK prover
        
        Args:
            zk_prover: Configured ZKTrainingProver instance
        """
        self.zk_prover = zk_prover
        self.last_weights = None
    
    def on_train_batch_end(self,
                           epoch: int,
                           step: int,
                           batch: Dict[str, Any],
                           outputs: Dict[str, Any],
                           model: Any,
                           optimizer: Any) -> None:
        """
        Callback for end of training batch
        
        Args:
            epoch: Current epoch
            step: Current step
            batch: Training batch data
            outputs: Model outputs/loss
            model: Training model
            optimizer: Optimizer instance
        """
        # Extract current weights
        current_weights = self._extract_model_weights(model)
        
        # Extract batch data
        batch_data = {
            'inputs': batch.get('inputs', batch.get('x', None)),
            'targets': batch.get('targets', batch.get('y', None))
        }
        
        # Get learning rate
        learning_rate = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else 0.01
        
        # Log the training step
        self.zk_prover.log_training_step(
            epoch=epoch,
            step=step,
            loss=outputs.get('loss', 0.0),
            accuracy=outputs.get('accuracy', None),
            weights_before=self.last_weights,
            weights_after=current_weights,
            batch_data=batch_data if self.last_weights else None,
            learning_rate=learning_rate
        )
        
        # Update last weights
        self.last_weights = current_weights
    
    def on_epoch_end(self, epoch: int) -> None:
        """Callback for end of epoch"""
        summary = self.zk_prover.finish_epoch(epoch)
        logger.info(f"Epoch {epoch} completed: {summary}")
    
    def on_train_end(self) -> None:
        """Callback for end of training"""
        summary = self.zk_prover.get_training_summary()
        logger.info(f"Training completed: {summary}")
        
        # Verify all proofs
        valid, invalid = self.zk_prover.verify_all_proofs()
        logger.info(f"Final proof verification: {valid}/{valid+invalid} valid")
    
    def _extract_model_weights(self, model: Any) -> Dict[str, np.ndarray]:
        """Extract model weights as numpy arrays"""
        weights = {}
        
        # PyTorch model
        if hasattr(model, 'state_dict'):
            for name, param in model.state_dict().items():
                if 'weight' in name or 'bias' in name:
                    weights[name] = param.detach().cpu().numpy()
        
        # Keras/TensorFlow model
        elif hasattr(model, 'get_weights'):
            for i, w in enumerate(model.get_weights()):
                weights[f'layer_{i}'] = w
        
        # NumPy arrays
        elif isinstance(model, dict):
            weights = {k: v.copy() if isinstance(v, np.ndarray) else v
                      for k, v in model.items()}
        
        return weights