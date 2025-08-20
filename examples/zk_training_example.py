"""
Example: Zero-Knowledge Proof Generation During Training

This example demonstrates how to use the ZKTrainingProver to automatically
generate zero-knowledge proofs during model training.
"""

import sys
import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pot.zk.training_integration import ZKTrainingProver, ZKTrainingCallback
from pot.prototypes.training_provenance_auditor import MockBlockchainClient


class SimpleLinearModel:
    """Simple linear model for demonstration"""
    
    def __init__(self, input_dim: int = 16, output_dim: int = 4):
        """Initialize with random weights"""
        self.weights = {
            'layer1': np.random.randn(input_dim, output_dim) * 0.1
        }
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass"""
        return x @ self.weights['layer1']
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute MSE loss"""
        return np.mean((predictions - targets) ** 2)
    
    def compute_gradients(self, x: np.ndarray, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients using backpropagation"""
        batch_size = x.shape[0]
        error = 2 * (predictions - targets) / batch_size
        grad_w = x.T @ error
        return {'layer1': grad_w}
    
    def update_weights(self, gradients: Dict[str, np.ndarray], learning_rate: float):
        """Update weights using SGD"""
        for name, grad in gradients.items():
            self.weights[name] -= learning_rate * grad
    
    def get_state_dict(self) -> Dict[str, np.ndarray]:
        """Get current model state"""
        return {k: v.copy() for k, v in self.weights.items()}


def generate_synthetic_batch(batch_size: int, input_dim: int, output_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training batch"""
    x = np.random.randn(batch_size, input_dim)
    # Create targets with some pattern
    w_true = np.random.randn(input_dim, output_dim) * 0.5
    y = x @ w_true + np.random.randn(batch_size, output_dim) * 0.1
    return x, y


def train_with_zk_proofs():
    """
    Train a simple model with automatic ZK proof generation
    """
    print("=" * 60)
    print("Zero-Knowledge Proof Training Example")
    print("=" * 60)
    
    # Configuration
    input_dim = 16
    output_dim = 4
    batch_size = 1
    num_epochs = 3
    steps_per_epoch = 20
    learning_rate = 0.01
    proof_frequency = 5  # Generate proof every 5 steps
    
    # Initialize model
    model = SimpleLinearModel(input_dim, output_dim)
    
    # Initialize ZK Training Prover
    print("\n1. Initializing ZK Training Prover...")
    zk_prover = ZKTrainingProver(
        model_id="example_model_001",
        proof_frequency=proof_frequency,
        enable_blockchain=True,  # Use mock blockchain
        enable_dual_trees=True,  # Maintain both SHA-256 and Poseidon trees
        zk_params_k=10,
        auto_verify=True,  # Auto-verify proofs
        save_proofs_to_disk=True,
        proof_dir=Path("./zk_proofs_example")
    )
    print(f"   - Proof frequency: every {proof_frequency} steps")
    print(f"   - Dual commitment trees: Enabled")
    print(f"   - Auto-verification: Enabled")
    
    # Training loop
    print("\n2. Starting training with ZK proof generation...")
    print("-" * 60)
    
    total_loss = 0
    step_count = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        epoch_proofs = 0
        
        for step in range(steps_per_epoch):
            step_count += 1
            
            # Save weights before update
            weights_before = model.get_state_dict()
            
            # Generate batch
            x, y = generate_synthetic_batch(batch_size, input_dim, output_dim)
            batch_data = {'inputs': x, 'targets': y}
            
            # Forward pass
            predictions = model.forward(x)
            loss = model.compute_loss(predictions, y)
            epoch_loss += loss
            
            # Backward pass
            gradients = model.compute_gradients(x, predictions, y)
            
            # Update weights
            model.update_weights(gradients, learning_rate)
            weights_after = model.get_state_dict()
            
            # Log training step with potential ZK proof generation
            proof_record = zk_prover.log_training_step(
                epoch=epoch,
                step=step,
                loss=float(loss),
                accuracy=None,  # Not computed for regression
                weights_before=weights_before,
                weights_after=weights_after,
                batch_data=batch_data,
                learning_rate=learning_rate,
                additional_metrics={'gradient_norm': np.linalg.norm(gradients['layer1'])}
            )
            
            # Check if proof was generated
            if proof_record:
                epoch_proofs += 1
                print(f"   Step {step:3d}: Loss={loss:.4f}, ZK Proof Generated ✓")
            elif step % 5 == 0:  # Show more steps
                print(f"   Step {step:3d}: Loss={loss:.4f}")
        
        # Finish epoch
        avg_loss = epoch_loss / steps_per_epoch
        print(f"\n   Epoch Summary:")
        print(f"   - Average Loss: {avg_loss:.4f}")
        print(f"   - ZK Proofs Generated: {epoch_proofs}")
        
        epoch_summary = zk_prover.finish_epoch(epoch)
        if epoch_summary.get('poseidon_root'):
            print(f"   - Poseidon Root: {epoch_summary['poseidon_root'][:16]}...")
        if epoch_summary.get('blockchain_commitment'):
            print(f"   - Blockchain TX: {epoch_summary['blockchain_commitment']}")
    
    print("\n" + "=" * 60)
    print("3. Training Complete - Generating Summary")
    print("-" * 60)
    
    # Get training summary
    summary = zk_prover.get_training_summary()
    print(f"\nTraining Summary:")
    print(f"  - Model ID: {summary['model_id']}")
    print(f"  - Total Steps: {summary['total_steps']}")
    print(f"  - Total ZK Proofs: {summary['total_proofs']}")
    print(f"  - Verification Failures: {summary['verification_failures']}")
    print(f"  - Avg Proof Time: {summary['avg_proof_generation_time']:.3f}s")
    print(f"  - Avg Verify Time: {summary['avg_verification_time']:.3f}s")
    
    # Verify all proofs
    print("\n4. Verifying All Generated Proofs...")
    valid, invalid = zk_prover.verify_all_proofs()
    print(f"   Verification Results: {valid} valid, {invalid} invalid")
    print(f"   Success Rate: {100 * valid / (valid + invalid):.1f}%")
    
    # Export provenance
    print("\n5. Exporting Provenance Data...")
    provenance_file = Path("./zk_training_provenance.json")
    provenance = zk_prover.export_provenance(provenance_file)
    print(f"   Provenance exported to: {provenance_file}")
    print(f"   - SHA-256 Master Root: {provenance.get('merkle_roots', {}).get('sha256', 'N/A')}")
    print(f"   - Poseidon Master Root: {provenance.get('merkle_roots', {}).get('poseidon', 'N/A')[:16]}...")
    
    # Display proof storage info
    if zk_prover.proof_dir and zk_prover.proof_dir.exists():
        proof_files = list(zk_prover.proof_dir.glob("*.json"))
        print(f"\n6. ZK Proofs Saved to Disk:")
        print(f"   Directory: {zk_prover.proof_dir}")
        print(f"   Number of proof files: {len(proof_files)}")
        if proof_files:
            print(f"   Example: {proof_files[0].name}")
    
    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)
    
    return zk_prover


def demonstrate_proof_verification():
    """
    Demonstrate loading and verifying a saved proof
    """
    print("\n" + "=" * 60)
    print("Proof Verification Demonstration")
    print("=" * 60)
    
    proof_dir = Path("./zk_proofs_example")
    if not proof_dir.exists():
        print("No saved proofs found. Run training example first.")
        return
    
    proof_files = list(proof_dir.glob("*.json"))
    if not proof_files:
        print("No proof files found.")
        return
    
    # Load first proof
    proof_file = proof_files[0]
    print(f"\nLoading proof from: {proof_file.name}")
    
    with open(proof_file, 'r') as f:
        proof_data = json.load(f)
    
    print(f"\nProof Details:")
    print(f"  - Model ID: {proof_data['model_id']}")
    print(f"  - Epoch: {proof_data['epoch']}")
    print(f"  - Step: {proof_data['step_number']}")
    print(f"  - Timestamp: {proof_data['timestamp']}")
    print(f"  - Proof Hash: {proof_data['proof_hash'][:16]}...")
    print(f"  - Verification Status: {proof_data['verification_status']}")
    
    if proof_data.get('blockchain_tx'):
        print(f"  - Blockchain TX: {proof_data['blockchain_tx']}")
    
    # Display statement
    statement = proof_data['statement']
    print(f"\nStatement (Public Inputs):")
    print(f"  - W_t Root: {statement['W_t_root'][:16]}...")
    print(f"  - W_t+1 Root: {statement['W_t1_root'][:16]}...")
    print(f"  - Batch Root: {statement['batch_root'][:16]}...")
    print(f"  - Step Nonce: {statement['step_nonce']}")
    
    print("\n" + "=" * 60)


def demonstrate_callback_integration():
    """
    Demonstrate how to integrate ZK proofs with training callbacks
    """
    print("\n" + "=" * 60)
    print("Training Callback Integration Example")
    print("=" * 60)
    
    # Initialize components
    model = SimpleLinearModel(16, 4)
    zk_prover = ZKTrainingProver(
        model_id="callback_example",
        proof_frequency=3,
        enable_blockchain=False,
        enable_dual_trees=True
    )
    
    # Create callback
    callback = ZKTrainingCallback(zk_prover)
    
    print("\nSimulating training with callbacks...")
    
    # Simulate training loop
    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}")
        
        for step in range(10):
            # Generate batch
            x, y = generate_synthetic_batch(1, 16, 4)
            
            # Forward pass
            predictions = model.forward(x)
            loss = model.compute_loss(predictions, y)
            
            # Create mock optimizer
            class MockOptimizer:
                param_groups = [{'lr': 0.01}]
            
            # Simulate batch end callback
            callback.on_train_batch_end(
                epoch=epoch,
                step=step,
                batch={'inputs': x, 'targets': y},
                outputs={'loss': loss},
                model=model,
                optimizer=MockOptimizer()
            )
            
            # Update model (normally done by optimizer)
            gradients = model.compute_gradients(x, predictions, y)
            model.update_weights(gradients, 0.01)
            
            if step % 3 == 0:
                print(f"   Step {step}: Loss={loss:.4f}")
        
        # Epoch end callback
        callback.on_epoch_end(epoch)
    
    # Training end callback
    callback.on_train_end()
    
    print("\n" + "=" * 60)
    print("Callback Integration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Run main training example
    zk_prover = train_with_zk_proofs()
    
    # Demonstrate proof verification
    demonstrate_proof_verification()
    
    # Demonstrate callback integration
    demonstrate_callback_integration()
    
    print("\n✅ All examples completed successfully!")
    print("\nKey Takeaways:")
    print("1. ZK proofs can be automatically generated during training")
    print("2. Dual Merkle trees (SHA-256 and Poseidon) provide compatibility")
    print("3. Proofs are saved to disk for later verification")
    print("4. Integration with training frameworks is straightforward via callbacks")
    print("5. Blockchain storage provides immutable proof of training")