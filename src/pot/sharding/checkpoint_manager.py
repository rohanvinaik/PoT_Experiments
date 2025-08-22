"""
Checkpoint Manager for Fault-Tolerant Verification

Manages checkpoints to enable recovery from failures during long-running
verification of large models.
"""

import os
import json
import pickle
import time
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint:
    """Verification checkpoint"""
    checkpoint_id: str
    model_path: str
    shard_config: Dict[str, Any]
    challenges: List[str]
    processed_shards: List[int]
    responses: List[Any]
    metadata: Dict[str, Any]
    timestamp: float
    status: str  # 'in_progress', 'completed', 'failed'


class CheckpointManager:
    """
    Manages checkpoints for fault-tolerant sharded verification.
    
    Features:
    - Automatic checkpoint creation
    - Incremental updates
    - Recovery from failures
    - Checkpoint validation
    - Storage optimization
    """
    
    def __init__(self, checkpoint_dir: str = "/tmp/pot_checkpoints"):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_checkpoints: Dict[str, Checkpoint] = {}
        self.checkpoint_interval = 5  # Checkpoint every 5 shards
        self.max_checkpoints = 10  # Keep last 10 checkpoints
        self.compression_enabled = True
    
    def create_checkpoint(
        self,
        model_path: str,
        shard_config: Dict[str, Any],
        challenges: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Checkpoint:
        """
        Create a new checkpoint.
        
        Args:
            model_path: Path to model being verified
            shard_config: Sharding configuration
            challenges: Verification challenges
            metadata: Additional metadata
            
        Returns:
            Created checkpoint
        """
        # Generate checkpoint ID
        checkpoint_id = self._generate_checkpoint_id(model_path)
        
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            model_path=model_path,
            shard_config=shard_config,
            challenges=challenges,
            processed_shards=[],
            responses=[],
            metadata=metadata or {},
            timestamp=time.time(),
            status='in_progress'
        )
        
        # Save initial checkpoint
        self._save_checkpoint(checkpoint)
        self.active_checkpoints[checkpoint_id] = checkpoint
        
        logger.info(f"Created checkpoint {checkpoint_id}")
        return checkpoint
    
    def update_checkpoint(
        self,
        checkpoint: Checkpoint,
        shard_id: int,
        shard_responses: List[Any],
        force_save: bool = False
    ) -> None:
        """
        Update checkpoint with shard processing results.
        
        Args:
            checkpoint: Checkpoint to update
            shard_id: Processed shard ID
            shard_responses: Responses from shard
            force_save: Force immediate save
        """
        checkpoint.processed_shards.append(shard_id)
        checkpoint.responses.extend(shard_responses)
        checkpoint.metadata['last_update'] = time.time()
        
        # Save periodically or when forced
        should_save = (
            force_save or
            len(checkpoint.processed_shards) % self.checkpoint_interval == 0
        )
        
        if should_save:
            self._save_checkpoint(checkpoint)
            logger.debug(f"Updated checkpoint {checkpoint.checkpoint_id}")
    
    def complete_checkpoint(self, checkpoint: Checkpoint) -> None:
        """
        Mark checkpoint as completed.
        
        Args:
            checkpoint: Checkpoint to complete
        """
        checkpoint.status = 'completed'
        checkpoint.metadata['completion_time'] = time.time()
        self._save_checkpoint(checkpoint)
        
        # Remove from active
        if checkpoint.checkpoint_id in self.active_checkpoints:
            del self.active_checkpoints[checkpoint.checkpoint_id]
        
        logger.info(f"Completed checkpoint {checkpoint.checkpoint_id}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
    
    def recover_from_checkpoint(
        self,
        checkpoint_id: Optional[str] = None
    ) -> Optional[Checkpoint]:
        """
        Recover from a checkpoint.
        
        Args:
            checkpoint_id: Specific checkpoint to recover (latest if None)
            
        Returns:
            Recovered checkpoint or None
        """
        if checkpoint_id:
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.ckpt"
        else:
            # Find latest checkpoint
            checkpoints = list(self.checkpoint_dir.glob("*.ckpt"))
            if not checkpoints:
                logger.warning("No checkpoints found for recovery")
                return None
            
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            checkpoint_id = checkpoint_path.stem
        
        try:
            checkpoint = self._load_checkpoint(checkpoint_path)
            
            if checkpoint:
                logger.info(f"Recovered from checkpoint {checkpoint_id}")
                logger.info(f"  Processed shards: {len(checkpoint.processed_shards)}")
                logger.info(f"  Responses collected: {len(checkpoint.responses)}")
                
                # Mark as recovered
                checkpoint.metadata['recovered'] = True
                checkpoint.metadata['recovery_time'] = time.time()
                
                self.active_checkpoints[checkpoint_id] = checkpoint
                return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to recover checkpoint: {e}")
        
        return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        List all available checkpoints.
        
        Returns:
            List of checkpoint summaries
        """
        checkpoints = []
        
        for ckpt_file in self.checkpoint_dir.glob("*.ckpt"):
            try:
                # Load checkpoint metadata only
                checkpoint = self._load_checkpoint(ckpt_file, metadata_only=True)
                
                if checkpoint:
                    checkpoints.append({
                        'checkpoint_id': checkpoint.checkpoint_id,
                        'model_path': checkpoint.model_path,
                        'status': checkpoint.status,
                        'processed_shards': len(checkpoint.processed_shards),
                        'timestamp': checkpoint.timestamp,
                        'file_size_mb': ckpt_file.stat().st_size / 1024 / 1024
                    })
            except Exception as e:
                logger.warning(f"Could not load checkpoint {ckpt_file}: {e}")
        
        # Sort by timestamp
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        return checkpoints
    
    def validate_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """
        Validate checkpoint integrity.
        
        Args:
            checkpoint: Checkpoint to validate
            
        Returns:
            True if valid
        """
        try:
            # Check required fields
            if not checkpoint.checkpoint_id or not checkpoint.model_path:
                return False
            
            # Check shard processing consistency
            if checkpoint.processed_shards:
                expected_shards = set(range(max(checkpoint.processed_shards) + 1))
                actual_shards = set(checkpoint.processed_shards)
                
                if expected_shards != actual_shards:
                    logger.warning(f"Missing shards in checkpoint: "
                                 f"{expected_shards - actual_shards}")
                    return False
            
            # Verify responses match processed shards
            # This is simplified - actual implementation would be more thorough
            
            return True
            
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
            return False
    
    def _generate_checkpoint_id(self, model_path: str) -> str:
        """Generate unique checkpoint ID"""
        timestamp = int(time.time() * 1000)
        model_hash = hashlib.md5(model_path.encode()).hexdigest()[:8]
        return f"ckpt_{timestamp}_{model_hash}"
    
    def _save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Save checkpoint to disk"""
        checkpoint_path = self.checkpoint_dir / f"{checkpoint.checkpoint_id}.ckpt"
        temp_path = checkpoint_path.with_suffix('.tmp')
        
        try:
            # Save to temporary file first
            if self.compression_enabled:
                import gzip
                with gzip.open(temp_path, 'wb') as f:
                    pickle.dump(checkpoint, f)
            else:
                with open(temp_path, 'wb') as f:
                    pickle.dump(checkpoint, f)
            
            # Atomic rename
            temp_path.rename(checkpoint_path)
            
            # Also save metadata in JSON for easy inspection
            meta_path = checkpoint_path.with_suffix('.json')
            with open(meta_path, 'w') as f:
                json.dump({
                    'checkpoint_id': checkpoint.checkpoint_id,
                    'model_path': checkpoint.model_path,
                    'status': checkpoint.status,
                    'processed_shards': checkpoint.processed_shards,
                    'num_responses': len(checkpoint.responses),
                    'timestamp': checkpoint.timestamp,
                    'metadata': checkpoint.metadata
                }, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            if temp_path.exists():
                temp_path.unlink()
            raise
    
    def _load_checkpoint(
        self,
        checkpoint_path: Path,
        metadata_only: bool = False
    ) -> Optional[Checkpoint]:
        """Load checkpoint from disk"""
        try:
            if metadata_only:
                # Load JSON metadata
                meta_path = checkpoint_path.with_suffix('.json')
                if meta_path.exists():
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)
                    
                    # Create partial checkpoint with metadata
                    return Checkpoint(
                        checkpoint_id=meta['checkpoint_id'],
                        model_path=meta['model_path'],
                        shard_config={},
                        challenges=[],
                        processed_shards=meta['processed_shards'],
                        responses=[],
                        metadata=meta['metadata'],
                        timestamp=meta['timestamp'],
                        status=meta['status']
                    )
            
            # Load full checkpoint
            if self.compression_enabled:
                import gzip
                with gzip.open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
            else:
                with open(checkpoint_path, 'rb') as f:
                    checkpoint = pickle.load(f)
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints to save space"""
        checkpoints = list(self.checkpoint_dir.glob("*.ckpt"))
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by modification time
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        
        # Remove oldest checkpoints
        to_remove = checkpoints[:-self.max_checkpoints]
        
        for ckpt_path in to_remove:
            try:
                # Remove checkpoint and metadata
                ckpt_path.unlink()
                meta_path = ckpt_path.with_suffix('.json')
                if meta_path.exists():
                    meta_path.unlink()
                
                logger.debug(f"Removed old checkpoint {ckpt_path.stem}")
                
            except Exception as e:
                logger.warning(f"Could not remove checkpoint {ckpt_path}: {e}")
    
    def export_checkpoint(
        self,
        checkpoint: Checkpoint,
        export_path: str
    ) -> None:
        """
        Export checkpoint for archival or transfer.
        
        Args:
            checkpoint: Checkpoint to export
            export_path: Export destination
        """
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Export checkpoint data
        ckpt_export = export_dir / f"{checkpoint.checkpoint_id}_export.ckpt"
        with open(ckpt_export, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Export summary
        summary_path = export_dir / f"{checkpoint.checkpoint_id}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'checkpoint_id': checkpoint.checkpoint_id,
                'model_path': checkpoint.model_path,
                'status': checkpoint.status,
                'shard_config': checkpoint.shard_config,
                'num_challenges': len(checkpoint.challenges),
                'processed_shards': checkpoint.processed_shards,
                'num_responses': len(checkpoint.responses),
                'timestamp': checkpoint.timestamp,
                'metadata': checkpoint.metadata
            }, f, indent=2)
        
        logger.info(f"Exported checkpoint to {export_dir}")
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get checkpoint recovery statistics"""
        total_checkpoints = len(list(self.checkpoint_dir.glob("*.ckpt")))
        active_checkpoints = len(self.active_checkpoints)
        
        # Calculate storage usage
        total_size = sum(
            p.stat().st_size for p in self.checkpoint_dir.glob("*")
        )
        
        # Count recovered checkpoints
        recovered_count = sum(
            1 for ckpt in self.active_checkpoints.values()
            if ckpt.metadata.get('recovered', False)
        )
        
        return {
            'total_checkpoints': total_checkpoints,
            'active_checkpoints': active_checkpoints,
            'recovered_checkpoints': recovered_count,
            'storage_usage_mb': total_size / 1024 / 1024,
            'checkpoint_dir': str(self.checkpoint_dir)
        }