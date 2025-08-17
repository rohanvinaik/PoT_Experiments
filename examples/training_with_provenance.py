#!/usr/bin/env python3
"""
Example: Training with Provenance Recording

This example demonstrates how to integrate provenance recording with a typical
PyTorch training loop, showing how to record checkpoints, validation results,
and generate complete proof-of-training artifacts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from pot.core.provenance_integration import (
        ProvenanceRecorder, 
        ProvenanceConfig,
        integrate_with_training_loop
    )
    POT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PoT provenance integration not available: {e}")
    POT_AVAILABLE = False


class SimpleModel(nn.Module):
    """Simple neural network for demonstration."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def get_model_hash(model: nn.Module) -> str:
    """Generate hash of model parameters."""
    param_bytes = b""
    for param in model.parameters():
        param_bytes += param.data.cpu().numpy().tobytes()
    
    return hashlib.sha256(param_bytes).hexdigest()


def create_dummy_data(batch_size: int = 64, num_batches: int = 10):
    """Create dummy training data for demonstration."""
    for _ in range(num_batches):
        x = torch.randn(batch_size, 784)
        y = torch.randint(0, 10, (batch_size,))
        yield x, y


def train_epoch(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    criterion: nn.Module,
    epoch: int,
    recorder: Optional[ProvenanceRecorder] = None,
    model_id: str = "demo_model"
) -> Dict[str, Any]:
    """Train model for one epoch with optional provenance recording."""
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    correct = 0
    total = 0
    
    # Training loop
    for batch_idx, (data, target) in enumerate(create_dummy_data()):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Calculate accuracy
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
    
    # Calculate metrics
    avg_loss = total_loss / num_batches
    accuracy = 100.0 * correct / total
    
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "epoch": epoch
    }
    
    # Get model hash
    model_hash = get_model_hash(model)
    
    # Record checkpoint if provenance is enabled
    if recorder and POT_AVAILABLE:
        checkpoint_id = recorder.record_training_checkpoint(
            model_hash=model_hash,
            metrics=metrics,
            epoch=epoch,
            model_id=model_id,
            model_fn=lambda x: model(x)  # Model function for fingerprinting
        )
        print(f"  Recorded checkpoint: {checkpoint_id}")
    
    return {
        "epoch": epoch,
        "model_hash": model_hash,
        "metrics": metrics,
        "loss": avg_loss,
        "accuracy": accuracy
    }


def validate_model(
    model: nn.Module, 
    criterion: nn.Module,
    validator_id: str = "validation_suite",
    recorder: Optional[ProvenanceRecorder] = None,
    model_id: str = "demo_model"
) -> Dict[str, Any]:
    """Validate model with optional provenance recording."""
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in create_dummy_data(batch_size=32, num_batches=5):
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    # Calculate validation metrics
    validation_result = {
        "val_loss": total_loss / 5,
        "val_accuracy": 100.0 * correct / total,
        "confidence": min(1.0, correct / total + 0.1),  # Mock confidence
        "num_samples": total
    }
    
    # Get model hash
    model_hash = get_model_hash(model)
    
    # Record validation if provenance is enabled
    if recorder and POT_AVAILABLE:
        validation_id = recorder.record_validation(
            model_hash=model_hash,
            validator_id=validator_id,
            validation_result=validation_result,
            model_id=model_id
        )
        print(f"  Recorded validation: {validation_id}")
    
    return validation_result


def main():
    """Main training loop with provenance recording."""
    print("Training with Provenance Recording Demo")
    print("=" * 50)
    
    # Configuration
    model_id = "simple_mnist_demo"
    num_epochs = 5
    enable_provenance = True
    enable_blockchain = False  # Set to True to test blockchain integration
    
    # Initialize model and training components
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Optimizer: {optimizer.__class__.__name__}")
    print(f"Epochs: {num_epochs}")
    print()
    
    # Initialize provenance recorder
    recorder = None
    if enable_provenance and POT_AVAILABLE:
        config = ProvenanceConfig(
            enabled=True,
            blockchain_enabled=enable_blockchain,
            local_storage_path="./training_provenance.json",
            fingerprint_checkpoints=True,
            record_challenges=False  # Disable for this simple demo
        )
        recorder = ProvenanceRecorder(config)
        print("✓ Provenance recording enabled")
        
        if enable_blockchain and recorder.blockchain_client:
            print(f"✓ Blockchain integration active: {type(recorder.blockchain_client).__name__}")
        else:
            print("✓ Local provenance storage active")
        print()
    
    # Training loop
    print("Starting training...")
    training_history = []
    
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        
        # Train
        train_result = train_epoch(
            model, optimizer, criterion, epoch, recorder, model_id
        )
        training_history.append(train_result)
        
        print(f"  Loss: {train_result['loss']:.4f}, Accuracy: {train_result['accuracy']:.2f}%")
        
        # Validate every 2 epochs
        if epoch % 2 == 0:
            val_result = validate_model(
                model, criterion, f"validator_{epoch}", recorder, model_id
            )
            print(f"  Val Loss: {val_result['val_loss']:.4f}, Val Accuracy: {val_result['val_accuracy']:.2f}%")
        
        print()
    
    print("Training completed!")
    print()
    
    # Generate proof of training
    if recorder and POT_AVAILABLE:
        print("Generating proof of training...")
        try:
            proof = recorder.generate_proof_of_training(model_id)
            
            # Save proof to file
            proof_dict = {
                "model_id": proof.model_id,
                "final_model_hash": proof.final_model_hash,
                "training_chain": [vars(cp) for cp in proof.training_chain],
                "validation_chain": [vars(vr) for vr in proof.validation_chain],
                "merkle_root": proof.merkle_root,
                "blockchain_transactions": proof.blockchain_transactions,
                "proof_timestamp": proof.proof_timestamp,
                "verification_metadata": proof.verification_metadata,
                "signature_hash": proof.signature_hash
            }
            
            proof_file = f"proof_{model_id}.json"
            with open(proof_file, 'w') as f:
                json.dump(proof_dict, f, indent=2)
            
            print(f"✓ Proof generated: {proof_file}")
            print(f"  - Final model hash: {proof.final_model_hash[:16]}...")
            print(f"  - Training checkpoints: {proof.verification_metadata['total_checkpoints']}")
            print(f"  - Validation records: {proof.verification_metadata['total_validations']}")
            print(f"  - Merkle root: {proof.merkle_root[:16]}...")
            
            # Verify proof
            print("\nVerifying proof...")
            is_valid = recorder.verify_training_provenance(proof)
            print(f"✓ Proof verification: {'PASSED' if is_valid else 'FAILED'}")
            
        except Exception as e:
            print(f"Error generating proof: {e}")
    
    # Show training summary
    print("\nTraining Summary:")
    print(f"Model ID: {model_id}")
    print(f"Final model hash: {get_model_hash(model)[:16]}...")
    print(f"Epochs completed: {len(training_history)}")
    
    if training_history:
        final_metrics = training_history[-1]["metrics"]
        print(f"Final loss: {final_metrics['loss']:.4f}")
        print(f"Final accuracy: {final_metrics['accuracy']:.2f}%")
    
    if recorder:
        history = recorder.get_model_history(model_id)
        print(f"Provenance records: {history['total_epochs']} checkpoints, {history['total_validations']} validations")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main()