#!/usr/bin/env python3
"""
Complete ZK Training Example

Demonstrates end-to-end training with zero-knowledge proof generation
for both SGD and LoRA fine-tuning paths.
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# PoT imports
from pot.zk import (
    prove_sgd_step,
    verify_sgd_step,
    prove_lora_step,
    verify_lora_step,
    SGDStepStatement,
    LoRAStepStatement,
    get_zk_metrics_collector,
)
from pot.zk.monitoring import ZKSystemMonitor
from pot.prototypes.training_provenance_auditor import TrainingProvenanceAuditor


class SimpleModel(nn.Module):
    """Simple neural network for demonstration"""
    
    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, output_dim: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class LoRAAdapter:
    """LoRA adapter for efficient fine-tuning"""
    
    def __init__(self, base_dim: int, rank: int = 16, alpha: float = 32.0):
        self.rank = rank
        self.alpha = alpha
        self.base_dim = base_dim
        
        # Initialize LoRA matrices
        self.adapter_a = torch.randn(base_dim, rank) * 0.01
        self.adapter_b = torch.randn(rank, base_dim) * 0.01
    
    def compute_delta(self) -> torch.Tensor:
        """Compute LoRA weight update: ΔW = α × (B × A)"""
        return self.alpha * (self.adapter_b @ self.adapter_a.T)
    
    def apply_to_weights(self, base_weights: torch.Tensor) -> torch.Tensor:
        """Apply LoRA update to base weights"""
        return base_weights + self.compute_delta()


class ZKTrainingLoop:
    """Training loop with ZK proof generation"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.metrics_collector = get_zk_metrics_collector()
        self.provenance_auditor = TrainingProvenanceAuditor(
            model_id=config.get('model_id', 'zk_model'),
            hash_function='poseidon'  # ZK-friendly hash
        )
        
        # Track training state
        self.epoch = 0
        self.step = 0
        self.proofs = []
        
    def extract_weights(self) -> np.ndarray:
        """Extract model weights as numpy array"""
        weights = []
        for param in self.model.parameters():
            weights.append(param.detach().cpu().numpy().flatten())
        return np.concatenate(weights)
    
    def compute_weight_hash(self, weights: np.ndarray) -> str:
        """Compute hash of weights for statement"""
        # Simplified hash computation
        return f"0x{hash(weights.tobytes()) & ((1 << 256) - 1):064x}"
    
    def train_step_sgd(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """Single SGD training step with ZK proof"""
        inputs, targets = batch
        
        # Capture weights before training
        weights_before = self.extract_weights()
        weights_before_hash = self.compute_weight_hash(weights_before)
        
        # Forward pass
        self.model.train()
        outputs = self.model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # Backward pass
        self.model.zero_grad()
        loss.backward()
        
        # Capture gradients
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().cpu().numpy().flatten())
        gradients = np.concatenate(gradients) if gradients else np.array([])
        
        # Update weights
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['learning_rate'])
        optimizer.step()
        
        # Capture weights after training
        weights_after = self.extract_weights()
        weights_after_hash = self.compute_weight_hash(weights_after)
        
        # Build ZK statement
        statement = SGDStepStatement(
            model_id=self.config['model_id'],
            step_number=self.step,
            epoch=self.epoch,
            weights_before_hash=weights_before_hash,
            weights_after_hash=weights_after_hash,
            batch_hash=self.compute_weight_hash(inputs.numpy()),
            learning_rate=self.config['learning_rate'],
            batch_size=len(inputs),
        )
        
        # Generate ZK proof
        print(f"  Generating SGD proof for step {self.step}...")
        proof_start = time.time()
        
        try:
            proof = prove_sgd_step(
                statement=statement,
                weights_before=weights_before,
                weights_after=weights_after,
                batch={'inputs': inputs.numpy(), 'targets': targets.numpy()},
                gradients=gradients,
                learning_rate=self.config['learning_rate']
            )
            proof_time = (time.time() - proof_start) * 1000
            
            # Record metrics
            self.metrics_collector.record_proof_generation(
                proof_type='sgd',
                duration=proof_time,
                proof_size=len(str(proof)),
                success=True
            )
            
            # Verify proof
            is_valid = verify_sgd_step(statement, proof)
            
            print(f"    ✅ Proof generated in {proof_time:.1f}ms (valid: {is_valid})")
            
            # Store proof
            self.proofs.append({
                'step': self.step,
                'type': 'sgd',
                'proof': proof,
                'statement': statement,
                'time_ms': proof_time,
                'valid': is_valid
            })
            
        except Exception as e:
            print(f"    ❌ Proof generation failed: {e}")
            self.metrics_collector.record_proof_generation(
                proof_type='sgd',
                duration=0,
                proof_size=0,
                success=False
            )
        
        self.step += 1
        
        return {
            'loss': loss.item(),
            'proof_generated': len(self.proofs) > 0,
            'proof_time_ms': proof_time if 'proof_time' in locals() else None
        }
    
    def train_step_lora(self, batch: Tuple[torch.Tensor, torch.Tensor], 
                       lora_adapter: LoRAAdapter) -> Dict[str, Any]:
        """LoRA fine-tuning step with ZK proof"""
        inputs, targets = batch
        
        # Get base weights
        base_weights = self.extract_weights()
        base_weights_hash = self.compute_weight_hash(base_weights)
        
        # Forward pass with LoRA
        self.model.train()
        outputs = self.model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        
        # Update LoRA adapters (simplified)
        loss.backward()
        lora_adapter.adapter_a -= self.config['learning_rate'] * lora_adapter.adapter_a.grad
        lora_adapter.adapter_b -= self.config['learning_rate'] * lora_adapter.adapter_b.grad
        
        # Build LoRA statement
        statement = LoRAStepStatement(
            model_id=self.config['model_id'],
            step_number=self.step,
            epoch=self.epoch,
            base_weights_hash=base_weights_hash,
            adapter_a_hash=self.compute_weight_hash(lora_adapter.adapter_a.numpy()),
            adapter_b_hash=self.compute_weight_hash(lora_adapter.adapter_b.numpy()),
            rank=lora_adapter.rank,
            scale_factor=lora_adapter.alpha,
        )
        
        # Generate LoRA proof
        print(f"  Generating LoRA proof for step {self.step}...")
        proof_start = time.time()
        
        try:
            proof = prove_lora_step(
                statement=statement,
                base_weights=base_weights,
                adapter_a=lora_adapter.adapter_a.numpy(),
                adapter_b=lora_adapter.adapter_b.numpy(),
            )
            proof_time = (time.time() - proof_start) * 1000
            
            # Record metrics
            self.metrics_collector.record_proof_generation(
                proof_type='lora',
                duration=proof_time,
                proof_size=len(str(proof)),
                success=True
            )
            
            # Verify proof
            is_valid = verify_lora_step(statement, proof)
            
            print(f"    ✅ LoRA proof generated in {proof_time:.1f}ms (valid: {is_valid})")
            
            # Store proof
            self.proofs.append({
                'step': self.step,
                'type': 'lora',
                'proof': proof,
                'statement': statement,
                'time_ms': proof_time,
                'valid': is_valid
            })
            
        except Exception as e:
            print(f"    ❌ LoRA proof generation failed: {e}")
            self.metrics_collector.record_proof_generation(
                proof_type='lora',
                duration=0,
                proof_size=0,
                success=False
            )
        
        self.step += 1
        
        return {
            'loss': loss.item(),
            'proof_generated': len(self.proofs) > 0,
            'proof_time_ms': proof_time if 'proof_time' in locals() else None
        }
    
    def train_epoch(self, dataloader, mode: str = 'sgd', lora_adapter: Optional[LoRAAdapter] = None):
        """Train one epoch with ZK proofs"""
        print(f"\n🔄 Epoch {self.epoch + 1} ({mode.upper()} mode)")
        print("-" * 50)
        
        epoch_losses = []
        epoch_proof_times = []
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= self.config.get('max_batches_per_epoch', 5):
                break
            
            print(f"Batch {batch_idx + 1}/{min(len(dataloader), self.config.get('max_batches_per_epoch', 5))}")
            
            if mode == 'sgd':
                result = self.train_step_sgd((inputs, targets))
            elif mode == 'lora' and lora_adapter:
                result = self.train_step_lora((inputs, targets), lora_adapter)
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
            epoch_losses.append(result['loss'])
            if result['proof_time_ms']:
                epoch_proof_times.append(result['proof_time_ms'])
        
        # Log epoch metrics
        avg_loss = np.mean(epoch_losses)
        avg_proof_time = np.mean(epoch_proof_times) if epoch_proof_times else 0
        
        self.provenance_auditor.log_training_event(
            epoch=self.epoch,
            metrics={
                'avg_loss': avg_loss,
                'avg_proof_time_ms': avg_proof_time,
                'proofs_generated': len(epoch_proof_times),
                'mode': mode
            }
        )
        
        print(f"\n📊 Epoch Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Proofs Generated: {len(epoch_proof_times)}/{batch_idx + 1}")
        print(f"  Average Proof Time: {avg_proof_time:.1f}ms")
        
        self.epoch += 1
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        report = self.metrics_collector.generate_report()
        
        sgd_proofs = [p for p in self.proofs if p['type'] == 'sgd']
        lora_proofs = [p for p in self.proofs if p['type'] == 'lora']
        
        return {
            'total_proofs': len(self.proofs),
            'sgd_proofs': len(sgd_proofs),
            'lora_proofs': len(lora_proofs),
            'sgd_avg_time_ms': np.mean([p['time_ms'] for p in sgd_proofs]) if sgd_proofs else 0,
            'lora_avg_time_ms': np.mean([p['time_ms'] for p in lora_proofs]) if lora_proofs else 0,
            'success_rate': report.get('proof_success_rate', 0),
            'verification_rate': sum(1 for p in self.proofs if p['valid']) / len(self.proofs) if self.proofs else 0
        }


def create_dummy_dataloader(batch_size: int = 32, num_batches: int = 10):
    """Create dummy data for demonstration"""
    data = []
    for _ in range(num_batches):
        inputs = torch.randn(batch_size, 784)  # MNIST-like
        targets = torch.randint(0, 10, (batch_size,))
        data.append((inputs, targets))
    return data


def main():
    """Main training demonstration"""
    print("=" * 70)
    print("🚀 Complete ZK Training Example")
    print("=" * 70)
    
    # Configuration
    config = {
        'model_id': 'demo_model',
        'learning_rate': 0.01,
        'batch_size': 32,
        'max_batches_per_epoch': 3,  # Limit for demo
        'lora_rank': 16,
        'lora_alpha': 32.0,
        'zk_enabled': True,
    }
    
    # Initialize model and training
    model = SimpleModel()
    trainer = ZKTrainingLoop(model, config)
    
    # Create dummy data
    dataloader = create_dummy_dataloader(
        batch_size=config['batch_size'],
        num_batches=10
    )
    
    # Optional: Start monitoring
    monitor = None
    if config.get('enable_monitoring', False):
        monitor = ZKSystemMonitor(check_interval=60)
        monitor.start_monitoring()
        print("📊 Monitoring started\n")
    
    try:
        # Phase 1: SGD Training with ZK Proofs
        print("\n" + "=" * 70)
        print("📚 Phase 1: SGD Training with ZK Proofs")
        print("=" * 70)
        
        for epoch in range(2):
            trainer.train_epoch(dataloader, mode='sgd')
        
        sgd_metrics = trainer.get_metrics_summary()
        
        # Phase 2: LoRA Fine-tuning with ZK Proofs
        print("\n" + "=" * 70)
        print("🎯 Phase 2: LoRA Fine-tuning with ZK Proofs")
        print("=" * 70)
        
        # Initialize LoRA adapter
        lora_adapter = LoRAAdapter(
            base_dim=128,  # Hidden dimension
            rank=config['lora_rank'],
            alpha=config['lora_alpha']
        )
        
        for epoch in range(2):
            trainer.train_epoch(dataloader, mode='lora', lora_adapter=lora_adapter)
        
        # Final metrics
        print("\n" + "=" * 70)
        print("📈 Training Complete - Final Metrics")
        print("=" * 70)
        
        final_metrics = trainer.get_metrics_summary()
        
        print(f"\n🎯 Proof Generation Summary:")
        print(f"  Total Proofs: {final_metrics['total_proofs']}")
        print(f"  SGD Proofs: {final_metrics['sgd_proofs']} (avg: {final_metrics['sgd_avg_time_ms']:.1f}ms)")
        print(f"  LoRA Proofs: {final_metrics['lora_proofs']} (avg: {final_metrics['lora_avg_time_ms']:.1f}ms)")
        print(f"  Success Rate: {final_metrics['success_rate']:.1%}")
        print(f"  Verification Rate: {final_metrics['verification_rate']:.1%}")
        
        # Compare SGD vs LoRA efficiency
        if final_metrics['sgd_proofs'] > 0 and final_metrics['lora_proofs'] > 0:
            speedup = final_metrics['sgd_avg_time_ms'] / final_metrics['lora_avg_time_ms']
            print(f"\n⚡ LoRA Speedup: {speedup:.2f}x faster than SGD")
        
        # Save proofs
        output_dir = Path("experimental_results")
        output_dir.mkdir(exist_ok=True)
        
        proof_file = output_dir / f"zk_proofs_{int(time.time())}.json"
        with open(proof_file, 'w') as f:
            json.dump({
                'config': config,
                'metrics': final_metrics,
                'proofs': [
                    {
                        'step': p['step'],
                        'type': p['type'],
                        'time_ms': p['time_ms'],
                        'valid': p['valid']
                    }
                    for p in trainer.proofs
                ]
            }, f, indent=2)
        
        print(f"\n💾 Proofs saved to: {proof_file}")
        
        # Generate provenance report
        provenance_report = trainer.provenance_auditor.generate_audit_report()
        print(f"\n🔍 Provenance Audit:")
        print(f"  Model ID: {provenance_report['model_id']}")
        print(f"  Total Events: {len(provenance_report['training_events'])}")
        print(f"  Root Hash: {provenance_report['merkle_root'][:16]}...")
        
    finally:
        if monitor:
            monitor.stop_monitoring()
            print("\n📊 Monitoring stopped")
    
    print("\n" + "=" * 70)
    print("✅ ZK Training Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()