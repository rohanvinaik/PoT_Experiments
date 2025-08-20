"""
Enhanced Training Provenance Auditor with ZK Proof Support

This module extends the TrainingProvenanceAuditor with zero-knowledge proof
capabilities for SGD training step verification.
"""

import sys
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import original auditor
from pot.prototypes.training_provenance_auditor import (
    TrainingProvenanceAuditor,
    TrainingEvent,
    EventType,
    MerkleTree,
    MerkleNode,
    BlockchainClient,
    MockBlockchainClient
)

# Import ZK modules
from pot.zk.zk_types import SGDStepStatement, SGDStepWitness
from pot.zk.prover import SGDZKProver, ProverConfig
from pot.zk.verifier import SGDZKVerifier, VerifierConfig
from pot.zk.builder import ZKWitnessBuilder

logger = logging.getLogger(__name__)


@dataclass
class ZKProofRecord:
    """Record of a generated ZK proof"""
    step_number: int
    epoch: int
    proof_hash: str
    proof_data: bytes
    statement: SGDStepStatement
    timestamp: datetime
    verification_status: bool = False
    blockchain_tx: Optional[str] = None


class TrainingProvenanceAuditorZK(TrainingProvenanceAuditor):
    """
    Enhanced Training Provenance Auditor with Zero-Knowledge Proof Support
    
    Extends the base auditor with:
    - ZK proof generation for SGD steps
    - Dual commitment mode (SHA-256 and Poseidon)
    - Automatic proof generation during training
    - Blockchain storage of proof hashes
    """
    
    def __init__(self, 
                 model_id: str, 
                 blockchain_client: Optional[BlockchainClient] = None,
                 compression_enabled: bool = True, 
                 max_history_size: int = 10000,
                 zk_enabled: bool = True,
                 zk_proof_frequency: int = 10,
                 dual_commitment_mode: bool = True,
                 zk_params_k: int = 10):
        """
        Initialize enhanced auditor with ZK support
        
        Args:
            model_id: Unique identifier for the model
            blockchain_client: Optional blockchain client for immutable storage
            compression_enabled: Whether to compress large histories
            max_history_size: Maximum number of events to keep in memory
            zk_enabled: Whether to generate ZK proofs
            zk_proof_frequency: Generate proof every N steps (0 = never, 1 = every step)
            dual_commitment_mode: Maintain both SHA-256 and Poseidon Merkle trees
            zk_params_k: ZK circuit parameter size
        """
        super().__init__(model_id, blockchain_client, compression_enabled, max_history_size)
        
        # ZK configuration
        self.zk_enabled = zk_enabled
        self.zk_proof_frequency = zk_proof_frequency
        self.dual_commitment_mode = dual_commitment_mode
        
        # ZK components
        if self.zk_enabled:
            self.zk_prover = SGDZKProver(ProverConfig(params_k=zk_params_k))
            self.zk_verifier = SGDZKVerifier(VerifierConfig(params_k=zk_params_k))
            self.zk_witness_builder = ZKWitnessBuilder()
        else:
            self.zk_prover = None
            self.zk_verifier = None
            self.zk_witness_builder = None
        
        # ZK proof storage
        self.zk_proofs: List[ZKProofRecord] = []
        self.zk_proof_index: Dict[int, ZKProofRecord] = {}
        
        # Poseidon Merkle trees (for dual commitment mode)
        if self.dual_commitment_mode:
            self.poseidon_trees: Dict[int, Any] = {}  # Epoch -> Poseidon tree
            self.poseidon_master_tree: Optional[Any] = None
        
        # Track model weights for ZK proofs
        self.last_weights: Optional[Dict[str, np.ndarray]] = None
        self.current_weights: Optional[Dict[str, np.ndarray]] = None
        self.last_batch_data: Optional[Dict[str, np.ndarray]] = None
        self.last_learning_rate: float = 0.0
        
        # Step counter for proof frequency
        self.step_counter = 0
        
        logger.info(f"TrainingProvenanceAuditorZK initialized with ZK={'enabled' if zk_enabled else 'disabled'}")
    
    def log_training_step_with_zk(self,
                                  epoch: int,
                                  step: int,
                                  metrics: Dict[str, Any],
                                  weights_before: Dict[str, np.ndarray],
                                  weights_after: Dict[str, np.ndarray],
                                  batch_data: Dict[str, np.ndarray],
                                  learning_rate: float,
                                  checkpoint_hash: Optional[str] = None,
                                  timestamp: Optional[datetime] = None,
                                  metadata: Optional[Dict] = None) -> Tuple[TrainingEvent, Optional[ZKProofRecord]]:
        """
        Log training step with optional ZK proof generation
        
        Args:
            epoch: Training epoch number
            step: Step number within epoch
            metrics: Training metrics (loss, accuracy, etc.)
            weights_before: Model weights before SGD update
            weights_after: Model weights after SGD update
            batch_data: Training batch (inputs and targets)
            learning_rate: Learning rate used for update
            checkpoint_hash: Hash of model checkpoint
            timestamp: Event timestamp
            metadata: Additional metadata
            
        Returns:
            Tuple of (TrainingEvent, Optional[ZKProofRecord])
        """
        # Log the event normally
        event = self.log_training_event(
            epoch=epoch,
            metrics={**metrics, 'step': step},
            checkpoint_hash=checkpoint_hash,
            timestamp=timestamp,
            event_type=EventType.CHECKPOINT_SAVE,
            metadata=metadata
        )
        
        # Store weights and batch data
        self.last_weights = weights_before
        self.current_weights = weights_after
        self.last_batch_data = batch_data
        self.last_learning_rate = learning_rate
        
        # Generate ZK proof if enabled and frequency matches
        zk_proof_record = None
        self.step_counter += 1
        
        if (self.zk_enabled and 
            self.zk_proof_frequency > 0 and 
            self.step_counter % self.zk_proof_frequency == 0):
            
            try:
                zk_proof_record = self._generate_zk_proof(
                    epoch=epoch,
                    step=step,
                    event=event,
                    weights_before=weights_before,
                    weights_after=weights_after,
                    batch_data=batch_data,
                    learning_rate=learning_rate
                )
                
                if zk_proof_record:
                    # Store proof hash on blockchain
                    if self.blockchain_client:
                        tx_hash = self._store_proof_on_blockchain(zk_proof_record)
                        zk_proof_record.blockchain_tx = tx_hash
                    
                    logger.info(f"Generated ZK proof for step {step} in epoch {epoch}")
                    
            except Exception as e:
                logger.error(f"Failed to generate ZK proof: {e}")
        
        return event, zk_proof_record
    
    def _generate_zk_proof(self,
                          epoch: int,
                          step: int,
                          event: TrainingEvent,
                          weights_before: Dict[str, np.ndarray],
                          weights_after: Dict[str, np.ndarray],
                          batch_data: Dict[str, np.ndarray],
                          learning_rate: float) -> Optional[ZKProofRecord]:
        """
        Generate ZK proof for SGD step
        
        Returns:
            ZKProofRecord if successful, None otherwise
        """
        if not self.zk_witness_builder or not self.zk_prover:
            return None
        
        try:
            # Extract witness data
            witness_data = self.zk_witness_builder.extract_sgd_update_witness(
                weights_before=weights_before,
                weights_after=weights_after,
                batch_data=batch_data,
                learning_rate=learning_rate
            )
            
            # Create statement
            statement = SGDStepStatement(
                W_t_root=witness_data['w_t_root'],
                batch_root=witness_data['batch_root'],
                hparams_hash=hashlib.sha256(
                    json.dumps({'lr': learning_rate, 'epoch': epoch}).encode()
                ).digest(),
                W_t1_root=witness_data['w_t1_root'],
                step_nonce=step,
                step_number=self.step_counter,
                epoch=epoch
            )
            
            # Create witness
            witness = SGDStepWitness(
                weights_before=witness_data['weights_before'],
                weights_after=witness_data['weights_after'],
                batch_inputs=witness_data['batch_inputs'],
                batch_targets=witness_data['batch_targets'],
                learning_rate=learning_rate,
                loss_value=witness_data.get('loss_value', 0.0)
            )
            
            # Generate proof
            proof_data = self.zk_prover.prove_sgd_step(statement, witness)
            
            # Create proof record
            proof_record = ZKProofRecord(
                step_number=step,
                epoch=epoch,
                proof_hash=hashlib.sha256(proof_data).hexdigest(),
                proof_data=proof_data,
                statement=statement,
                timestamp=datetime.now(timezone.utc),
                verification_status=False
            )
            
            # Verify the proof immediately
            if self.zk_verifier:
                proof_record.verification_status = self.zk_verifier.verify_sgd_step(
                    statement, proof_data
                )
            
            # Store proof record
            self.zk_proofs.append(proof_record)
            self.zk_proof_index[step] = proof_record
            
            return proof_record
            
        except Exception as e:
            logger.error(f"ZK proof generation failed: {e}")
            return None
    
    def _store_proof_on_blockchain(self, proof_record: ZKProofRecord) -> Optional[str]:
        """
        Store proof hash on blockchain
        
        Args:
            proof_record: ZK proof record to store
            
        Returns:
            Transaction hash if successful
        """
        try:
            proof_data = {
                'model_id': self.model_id,
                'epoch': proof_record.epoch,
                'step': proof_record.step_number,
                'proof_hash': proof_record.proof_hash,
                'timestamp': proof_record.timestamp.isoformat(),
                'verified': proof_record.verification_status
            }
            
            tx_hash = self.blockchain_client.store_data(
                data_type='zk_proof',
                data=proof_data,
                metadata={'proof_size': len(proof_record.proof_data)}
            )
            
            self.blockchain_transactions.append(tx_hash)
            logger.info(f"Stored ZK proof on blockchain: {tx_hash}")
            
            return tx_hash
            
        except Exception as e:
            logger.error(f"Failed to store proof on blockchain: {e}")
            return None
    
    def generate_zk_training_proof(self, 
                                   start_epoch: int, 
                                   end_epoch: int) -> Dict[str, Any]:
        """
        Generate comprehensive ZK proof for training period
        
        Args:
            start_epoch: Starting epoch (inclusive)
            end_epoch: Ending epoch (inclusive)
            
        Returns:
            Dictionary containing proof summary
        """
        proofs_in_range = [
            p for p in self.zk_proofs 
            if start_epoch <= p.epoch <= end_epoch
        ]
        
        if not proofs_in_range:
            logger.warning(f"No ZK proofs found for epochs {start_epoch}-{end_epoch}")
            return {
                'status': 'no_proofs',
                'start_epoch': start_epoch,
                'end_epoch': end_epoch,
                'proof_count': 0
            }
        
        # Create summary
        summary = {
            'status': 'success',
            'start_epoch': start_epoch,
            'end_epoch': end_epoch,
            'proof_count': len(proofs_in_range),
            'verified_count': sum(1 for p in proofs_in_range if p.verification_status),
            'blockchain_txs': [p.blockchain_tx for p in proofs_in_range if p.blockchain_tx],
            'proof_hashes': [p.proof_hash for p in proofs_in_range],
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Create aggregate proof hash
        aggregate_data = ''.join([p.proof_hash for p in proofs_in_range])
        summary['aggregate_hash'] = hashlib.sha256(aggregate_data.encode()).hexdigest()
        
        # Store summary on blockchain if available
        if self.blockchain_client:
            tx_hash = self.blockchain_client.store_data(
                data_type='zk_training_proof',
                data=summary,
                metadata={'epochs': f"{start_epoch}-{end_epoch}"}
            )
            summary['blockchain_tx'] = tx_hash
        
        return summary
    
    def embed_provenance(self, model_path: str = None, 
                        include_zk_proofs: bool = True) -> Dict[str, Any]:
        """
        Enhanced embed_provenance with ZK proof data
        
        Args:
            model_path: Path to model file to embed provenance into
            include_zk_proofs: Whether to include ZK proof data
            
        Returns:
            Provenance metadata dictionary
        """
        # Get base provenance data
        base_provenance = super().generate_provenance_report()
        
        # Add ZK proof data if available
        if include_zk_proofs and self.zk_enabled:
            zk_data = {
                'zk_enabled': True,
                'proof_frequency': self.zk_proof_frequency,
                'dual_commitment': self.dual_commitment_mode,
                'total_proofs': len(self.zk_proofs),
                'verified_proofs': sum(1 for p in self.zk_proofs if p.verification_status),
                'proof_hashes': [p.proof_hash for p in self.zk_proofs[-10:]],  # Last 10
                'blockchain_txs': list(set([p.blockchain_tx for p in self.zk_proofs 
                                           if p.blockchain_tx]))
            }
            
            # Add Poseidon roots if in dual commitment mode
            if self.dual_commitment_mode and self.poseidon_master_tree:
                zk_data['poseidon_root'] = self._get_poseidon_root()
                zk_data['sha256_root'] = self.master_tree.get_root_hash() if self.master_tree else None
            
            base_provenance['zk_proofs'] = zk_data
        else:
            base_provenance['zk_proofs'] = {'zk_enabled': False}
        
        # Embed in model file if provided
        if model_path:
            self._embed_in_model_file(model_path, base_provenance)
        
        return base_provenance
    
    def _get_poseidon_root(self) -> Optional[str]:
        """Get Poseidon Merkle root if available"""
        # This would use actual Poseidon implementation
        # For now, return a mock value
        if self.poseidon_master_tree:
            return "poseidon_root_" + hashlib.sha256(
                str(self.step_counter).encode()
            ).hexdigest()[:16]
        return None
    
    def _embed_in_model_file(self, model_path: str, provenance_data: Dict[str, Any]):
        """Embed provenance data in model file metadata"""
        try:
            import pickle
            from pathlib import Path
            
            path = Path(model_path)
            if not path.exists():
                logger.error(f"Model file not found: {model_path}")
                return
            
            # For PyTorch models
            if path.suffix in ['.pt', '.pth']:
                import torch
                checkpoint = torch.load(model_path, map_location='cpu')
                if isinstance(checkpoint, dict):
                    checkpoint['provenance'] = provenance_data
                else:
                    checkpoint = {
                        'model': checkpoint,
                        'provenance': provenance_data
                    }
                torch.save(checkpoint, model_path)
                logger.info(f"Embedded provenance in PyTorch model: {model_path}")
            
            # For pickle files
            elif path.suffix == '.pkl':
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                wrapped = {
                    'model': model,
                    'provenance': provenance_data
                }
                
                with open(model_path, 'wb') as f:
                    pickle.dump(wrapped, f)
                logger.info(f"Embedded provenance in pickle model: {model_path}")
            
            else:
                # Save as separate JSON file
                json_path = path.with_suffix('.provenance.json')
                with open(json_path, 'w') as f:
                    json.dump(provenance_data, f, indent=2, default=str)
                logger.info(f"Saved provenance as separate file: {json_path}")
                
        except Exception as e:
            logger.error(f"Failed to embed provenance: {e}")
    
    def verify_zk_proof(self, proof_record: ZKProofRecord) -> bool:
        """
        Verify a ZK proof record
        
        Args:
            proof_record: ZK proof record to verify
            
        Returns:
            True if proof is valid
        """
        if not self.zk_verifier:
            return False
        
        try:
            is_valid = self.zk_verifier.verify_sgd_step(
                proof_record.statement,
                proof_record.proof_data
            )
            proof_record.verification_status = is_valid
            return is_valid
            
        except Exception as e:
            logger.error(f"Proof verification failed: {e}")
            return False
    
    def get_zk_proof_summary(self) -> Dict[str, Any]:
        """Get summary of all ZK proofs"""
        return {
            'total_proofs': len(self.zk_proofs),
            'verified': sum(1 for p in self.zk_proofs if p.verification_status),
            'on_blockchain': sum(1 for p in self.zk_proofs if p.blockchain_tx),
            'epochs_covered': list(set(p.epoch for p in self.zk_proofs)),
            'proof_frequency': self.zk_proof_frequency,
            'dual_commitment': self.dual_commitment_mode
        }