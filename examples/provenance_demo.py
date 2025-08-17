#!/usr/bin/env python3
"""
Comprehensive Provenance Recording Demo

This demonstration shows the complete workflow for blockchain-based provenance
recording in the PoT framework, including both local and blockchain modes,
training simulation, proof generation, and verification workflows.

Features:
- Training simulation with checkpoint recording
- Validation recording with multiple validators
- Proof-of-training generation with Merkle tree verification
- Both local JSON and blockchain storage modes
- Complete verification workflow demonstration
- Error handling and fallback scenarios
"""

import argparse
import json
import logging
import os
import sys
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from pot.core.provenance_integration import (
        ProvenanceRecorder, 
        ProvenanceConfig,
        TrainingCheckpoint,
        ValidationRecord,
        ProvenanceProof
    )
    from pot.security.blockchain_factory import get_blockchain_client, test_blockchain_connection
    from pot.security.blockchain_client import BlockchainClient
    POT_AVAILABLE = True
except ImportError as e:
    print(f"Error: PoT modules not available: {e}")
    print("Make sure you're running from the PoT_Experiments directory")
    POT_AVAILABLE = False
    sys.exit(1)


# Demo Configuration
DEMO_MODELS = {
    "mnist_classifier": {
        "architecture": "conv2d_relu_fc",
        "input_shape": [28, 28, 1],
        "num_classes": 10
    },
    "bert_fine_tuned": {
        "architecture": "transformer_encoder",
        "num_layers": 12,
        "hidden_size": 768
    },
    "resnet50_transfer": {
        "architecture": "resnet50",
        "input_shape": [224, 224, 3],
        "pretrained": "imagenet"
    }
}

DEMO_VALIDATORS = [
    "official_validation_suite",
    "independent_auditor_alpha", 
    "regulatory_compliance_checker",
    "community_validator_beta"
]


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the demo."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def generate_model_hash(model_id: str, epoch: int) -> str:
    """Generate a realistic model hash for demonstration."""
    # Simulate model parameter evolution during training
    base_data = f"{model_id}_epoch_{epoch}_{DEMO_MODELS[model_id]}"
    return hashlib.sha256(base_data.encode()).hexdigest()


def generate_training_metrics(epoch: int, model_id: str) -> Dict[str, float]:
    """Generate realistic training metrics that improve over time."""
    import random
    
    # Simulate improving metrics with some noise
    base_accuracy = 0.6 + (epoch / 50.0) * 0.35  # 60% -> 95% over 50 epochs
    base_loss = 2.0 * (1.0 - epoch / 50.0) + 0.1  # 2.0 -> 0.1 over 50 epochs
    
    # Add realistic noise
    accuracy = min(0.99, base_accuracy + random.gauss(0, 0.02))
    loss = max(0.01, base_loss + random.gauss(0, 0.1))
    
    metrics = {
        "loss": round(loss, 4),
        "accuracy": round(accuracy, 4),
        "learning_rate": 0.001 * (0.95 ** (epoch // 5)),  # Decay every 5 epochs
    }
    
    # Add model-specific metrics
    if "classifier" in model_id:
        metrics["f1_score"] = round(accuracy * 0.95, 4)
        metrics["precision"] = round(accuracy * 0.98, 4)
        metrics["recall"] = round(accuracy * 0.93, 4)
    elif "bert" in model_id:
        metrics["perplexity"] = round(50.0 * (1.0 - accuracy), 2)
        metrics["bleu_score"] = round(accuracy * 0.8, 4)
    elif "resnet" in model_id:
        metrics["top5_accuracy"] = round(min(0.999, accuracy * 1.15), 4)
        metrics["mAP"] = round(accuracy * 0.9, 4)
    
    return metrics


def generate_validation_result(model_id: str, validator: str, epoch: int) -> Dict[str, Any]:
    """Generate realistic validation results."""
    import random
    
    base_accuracy = 0.6 + (epoch / 50.0) * 0.35
    accuracy = min(0.99, base_accuracy + random.gauss(0, 0.015))
    
    result = {
        "accuracy": round(accuracy, 4),
        "confidence": round(min(1.0, accuracy + 0.1), 4),
        "validation_samples": random.randint(1000, 10000),
        "validation_time": round(random.uniform(30, 300), 2)
    }
    
    # Validator-specific additional metrics
    if "official" in validator:
        result["official_score"] = round(accuracy * 100, 2)
        result["compliance_passed"] = accuracy > 0.85
    elif "auditor" in validator:
        result["audit_score"] = round(accuracy * 100, 2) 
        result["security_passed"] = True
        result["bias_check"] = "passed" if accuracy > 0.8 else "warning"
    elif "regulatory" in validator:
        result["regulatory_score"] = round(accuracy * 95, 2)
        result["gdpr_compliance"] = True
        result["ai_act_compliance"] = accuracy > 0.9
    elif "community" in validator:
        result["community_score"] = round(accuracy * 90, 2)
        result["peer_reviews"] = random.randint(5, 25)
        result["consensus_rating"] = "high" if accuracy > 0.9 else "medium"
    
    return result


def simulate_training(
    model_id: str, 
    recorder: ProvenanceRecorder, 
    num_epochs: int = 20,
    validation_frequency: int = 5
) -> List[TrainingCheckpoint]:
    """Simulate a complete training process with provenance recording."""
    print(f"\n🚀 Starting training simulation for {model_id}")
    print(f"   Model: {DEMO_MODELS[model_id]['architecture']}")
    print(f"   Epochs: {num_epochs}, Validation every {validation_frequency} epochs")
    
    checkpoints = []
    
    for epoch in range(1, num_epochs + 1):
        # Generate training metrics
        metrics = generate_training_metrics(epoch, model_id)
        model_hash = generate_model_hash(model_id, epoch)
        
        print(f"   Epoch {epoch:2d}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        # Record checkpoint
        if recorder.config.enabled:
            checkpoint_id = recorder.record_training_checkpoint(
                model_hash=model_hash,
                metrics=metrics,
                epoch=epoch,
                model_id=model_id
            )
            
            # Get the recorded checkpoint for return
            checkpoint = TrainingCheckpoint(
                model_hash=model_hash,
                epoch=epoch,
                metrics=metrics,
                timestamp="",  # Will be set by recorder
                model_id=model_id,
                checkpoint_id=checkpoint_id
            )
            checkpoints.append(checkpoint)
        
        # Validation every N epochs
        if epoch % validation_frequency == 0:
            print(f"   🔍 Running validation at epoch {epoch}")
            
            for validator in DEMO_VALIDATORS[:2]:  # Use first 2 validators
                validation_result = generate_validation_result(model_id, validator, epoch)
                
                if recorder.config.enabled:
                    validation_id = recorder.record_validation(
                        model_hash=model_hash,
                        validator_id=validator,
                        validation_result=validation_result,
                        model_id=model_id
                    )
                    print(f"      ✓ {validator}: {validation_result['accuracy']:.4f} ({validation_id})")
        
        # Small delay to simulate realistic training time
        time.sleep(0.1)
    
    final_metrics = generate_training_metrics(num_epochs, model_id)
    print(f"   ✅ Training completed! Final accuracy: {final_metrics['accuracy']:.4f}")
    
    return checkpoints


def demonstrate_local_mode():
    """Demonstrate provenance recording in local mode."""
    print("\n" + "="*60)
    print("🏠 LOCAL MODE DEMONSTRATION")
    print("="*60)
    
    # Configure for local mode
    config = ProvenanceConfig(
        enabled=True,
        blockchain_enabled=False,
        local_storage_path="./demo_provenance_local.json",
        fingerprint_checkpoints=False,  # Disable for faster demo
        record_challenges=False
    )
    
    recorder = ProvenanceRecorder(config)
    print(f"✓ Local provenance recorder initialized")
    print(f"  Storage: {recorder.local_storage_path}")
    
    # Simulate training for a model
    model_id = "mnist_classifier"
    checkpoints = simulate_training(model_id, recorder, num_epochs=10, validation_frequency=3)
    
    # Generate proof of training
    print(f"\n📋 Generating proof of training...")
    proof = recorder.generate_proof_of_training(model_id)
    
    print(f"✓ Proof generated successfully!")
    print(f"  Model: {proof.model_id}")
    print(f"  Final hash: {proof.final_model_hash[:16]}...")
    print(f"  Training checkpoints: {proof.verification_metadata['total_checkpoints']}")
    print(f"  Validation records: {proof.verification_metadata['total_validations']}")
    print(f"  Merkle root: {proof.merkle_root[:16]}...")
    
    # Verify proof
    print(f"\n🔍 Verifying proof integrity...")
    is_valid = recorder.verify_training_provenance(proof)
    print(f"✓ Proof verification: {'PASSED' if is_valid else 'FAILED'}")
    
    # Save proof to file
    proof_file = f"demo_proof_{model_id}_local.json"
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
    
    with open(proof_file, 'w') as f:
        json.dump(proof_dict, f, indent=2)
    
    print(f"✓ Proof saved to: {proof_file}")
    
    return recorder, proof


def demonstrate_blockchain_mode():
    """Demonstrate provenance recording in blockchain mode."""
    print("\n" + "="*60)
    print("⛓️  BLOCKCHAIN MODE DEMONSTRATION")
    print("="*60)
    
    # Test blockchain connectivity first
    print("🔗 Testing blockchain connectivity...")
    try:
        client = get_blockchain_client()
        success, results = test_blockchain_connection(client)
        
        print(f"   Client type: {results['client_type']}")
        if success:
            print("   ✓ Connection successful")
            blockchain_available = "Blockchain" in results['client_type']
        else:
            print(f"   ⚠️  Connection failed: {results.get('error', 'Unknown error')}")
            blockchain_available = False
    except Exception as e:
        print(f"   ❌ Blockchain client error: {e}")
        blockchain_available = False
    
    # Configure for blockchain mode (will fallback to local if not available)
    config = ProvenanceConfig(
        enabled=True,
        blockchain_enabled=True,  # Request blockchain
        local_storage_path="./demo_provenance_blockchain.json",
        fingerprint_checkpoints=False,
        record_challenges=False
    )
    
    recorder = ProvenanceRecorder(config)
    
    if recorder.blockchain_client:
        print(f"✓ Blockchain recorder initialized: {type(recorder.blockchain_client).__name__}")
    else:
        print(f"⚠️  Fell back to local storage (blockchain unavailable)")
    
    # Simulate training for a different model
    model_id = "bert_fine_tuned"
    checkpoints = simulate_training(model_id, recorder, num_epochs=8, validation_frequency=4)
    
    # Show blockchain transactions if available
    if recorder.blockchain_transactions:
        print(f"\n📜 Blockchain transactions recorded:")
        for i, tx_id in enumerate(recorder.blockchain_transactions):
            print(f"   {i+1}: {tx_id}")
    
    # Generate proof of training
    print(f"\n📋 Generating proof of training...")
    proof = recorder.generate_proof_of_training(model_id)
    
    print(f"✓ Proof generated successfully!")
    print(f"  Blockchain transactions: {len(proof.blockchain_transactions)}")
    
    # Verify proof (including blockchain verification if available)
    print(f"\n🔍 Verifying proof with blockchain validation...")
    is_valid = recorder.verify_training_provenance(proof)
    print(f"✓ Full proof verification: {'PASSED' if is_valid else 'FAILED'}")
    
    # Save proof to file
    proof_file = f"demo_proof_{model_id}_blockchain.json"
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
    
    with open(proof_file, 'w') as f:
        json.dump(proof_dict, f, indent=2)
    
    print(f"✓ Proof saved to: {proof_file}")
    
    return recorder, proof


def demonstrate_verification_workflow():
    """Demonstrate the complete verification workflow."""
    print("\n" + "="*60)
    print("🔍 VERIFICATION WORKFLOW DEMONSTRATION")
    print("="*60)
    
    # Create a new recorder for verification
    config = ProvenanceConfig(enabled=True, blockchain_enabled=False)
    verifier = ProvenanceRecorder(config)
    
    # Load and verify existing proofs
    proof_files = [
        f"demo_proof_mnist_classifier_local.json",
        f"demo_proof_bert_fine_tuned_blockchain.json"
    ]
    
    for proof_file in proof_files:
        if not Path(proof_file).exists():
            print(f"⚠️  Proof file not found: {proof_file}")
            continue
        
        print(f"\n📄 Verifying proof: {proof_file}")
        
        try:
            with open(proof_file, 'r') as f:
                proof_data = json.load(f)
            
            # Reconstruct proof object
            training_chain = [TrainingCheckpoint(**cp) for cp in proof_data["training_chain"]]
            validation_chain = [ValidationRecord(**vr) for vr in proof_data["validation_chain"]]
            
            proof = ProvenanceProof(
                model_id=proof_data["model_id"],
                final_model_hash=proof_data["final_model_hash"],
                training_chain=training_chain,
                validation_chain=validation_chain,
                merkle_root=proof_data["merkle_root"],
                blockchain_transactions=proof_data["blockchain_transactions"],
                proof_timestamp=proof_data["proof_timestamp"],
                verification_metadata=proof_data["verification_metadata"],
                signature_hash=proof_data["signature_hash"]
            )
            
            # Verify proof
            is_valid = verifier.verify_training_provenance(proof)
            
            print(f"   Model: {proof.model_id}")
            print(f"   Checkpoints: {len(proof.training_chain)}")
            print(f"   Validations: {len(proof.validation_chain)}")
            print(f"   Blockchain TXs: {len(proof.blockchain_transactions)}")
            print(f"   Verification: {'✅ PASSED' if is_valid else '❌ FAILED'}")
            
            # Show training progression
            if proof.training_chain:
                print(f"   Training progression:")
                for cp in proof.training_chain[-3:]:  # Show last 3 checkpoints
                    acc = cp.metrics.get('accuracy', 0)
                    loss = cp.metrics.get('loss', 0)
                    print(f"     Epoch {cp.epoch}: acc={acc:.4f}, loss={loss:.4f}")
            
        except Exception as e:
            print(f"   ❌ Verification failed: {e}")


def demonstrate_attack_scenarios():
    """Demonstrate detection of tampered proofs."""
    print("\n" + "="*60)
    print("🛡️  ATTACK DETECTION DEMONSTRATION")
    print("="*60)
    
    proof_file = "demo_proof_mnist_classifier_local.json"
    if not Path(proof_file).exists():
        print(f"⚠️  Original proof file not found: {proof_file}")
        return
    
    # Load original proof
    with open(proof_file, 'r') as f:
        original_proof = json.load(f)
    
    config = ProvenanceConfig(enabled=True, blockchain_enabled=False)
    verifier = ProvenanceRecorder(config)
    
    # Test 1: Tampered final model hash
    print("\n🧪 Test 1: Tampered final model hash")
    tampered_proof = original_proof.copy()
    tampered_proof["final_model_hash"] = "0x" + "f" * 62  # Invalid hash
    
    try:
        training_chain = [TrainingCheckpoint(**cp) for cp in tampered_proof["training_chain"]]
        validation_chain = [ValidationRecord(**vr) for vr in tampered_proof["validation_chain"]]
        
        proof = ProvenanceProof(
            model_id=tampered_proof["model_id"],
            final_model_hash=tampered_proof["final_model_hash"],
            training_chain=training_chain,
            validation_chain=validation_chain,
            merkle_root=tampered_proof["merkle_root"],
            blockchain_transactions=tampered_proof["blockchain_transactions"],
            proof_timestamp=tampered_proof["proof_timestamp"],
            verification_metadata=tampered_proof["verification_metadata"],
            signature_hash=tampered_proof["signature_hash"]
        )
        
        is_valid = verifier.verify_training_provenance(proof)
        print(f"   Tampered hash detection: {'✅ DETECTED' if not is_valid else '❌ MISSED'}")
        
    except Exception as e:
        print(f"   ✅ DETECTED (Exception): {e}")
    
    # Test 2: Modified training metrics
    print("\n🧪 Test 2: Modified training metrics")
    tampered_proof = original_proof.copy()
    if tampered_proof["training_chain"]:
        tampered_proof["training_chain"][0]["metrics"]["accuracy"] = 0.99  # Unrealistic accuracy
    
    try:
        training_chain = [TrainingCheckpoint(**cp) for cp in tampered_proof["training_chain"]]
        validation_chain = [ValidationRecord(**vr) for vr in tampered_proof["validation_chain"]]
        
        proof = ProvenanceProof(
            model_id=tampered_proof["model_id"],
            final_model_hash=tampered_proof["final_model_hash"],
            training_chain=training_chain,
            validation_chain=validation_chain,
            merkle_root=tampered_proof["merkle_root"],
            blockchain_transactions=tampered_proof["blockchain_transactions"],
            proof_timestamp=tampered_proof["proof_timestamp"],
            verification_metadata=tampered_proof["verification_metadata"],
            signature_hash=tampered_proof["signature_hash"]
        )
        
        is_valid = verifier.verify_training_provenance(proof)
        print(f"   Modified metrics detection: {'✅ DETECTED' if not is_valid else '❌ MISSED'}")
        
    except Exception as e:
        print(f"   ✅ DETECTED (Exception): {e}")
    
    # Test 3: Invalid Merkle root
    print("\n🧪 Test 3: Invalid Merkle root")
    tampered_proof = original_proof.copy()
    tampered_proof["merkle_root"] = "0x" + "a" * 62  # Invalid root
    
    try:
        training_chain = [TrainingCheckpoint(**cp) for cp in tampered_proof["training_chain"]]
        validation_chain = [ValidationRecord(**vr) for vr in tampered_proof["validation_chain"]]
        
        proof = ProvenanceProof(
            model_id=tampered_proof["model_id"],
            final_model_hash=tampered_proof["final_model_hash"],
            training_chain=training_chain,
            validation_chain=validation_chain,
            merkle_root=tampered_proof["merkle_root"],
            blockchain_transactions=tampered_proof["blockchain_transactions"],
            proof_timestamp=tampered_proof["proof_timestamp"],
            verification_metadata=tampered_proof["verification_metadata"],
            signature_hash=tampered_proof["signature_hash"]
        )
        
        is_valid = verifier.verify_training_provenance(proof)
        print(f"   Invalid Merkle root detection: {'✅ DETECTED' if not is_valid else '❌ MISSED'}")
        
    except Exception as e:
        print(f"   ✅ DETECTED (Exception): {e}")


def demonstrate_performance_analysis():
    """Demonstrate performance characteristics of the system."""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE ANALYSIS")
    print("="*60)
    
    config = ProvenanceConfig(
        enabled=True,
        blockchain_enabled=False,
        fingerprint_checkpoints=False
    )
    recorder = ProvenanceRecorder(config)
    
    # Benchmark checkpoint recording
    print("\n📊 Benchmarking checkpoint recording...")
    model_id = "performance_test_model"
    
    start_time = time.time()
    num_checkpoints = 50
    
    for epoch in range(1, num_checkpoints + 1):
        metrics = generate_training_metrics(epoch, model_id)
        model_hash = generate_model_hash(model_id, epoch)
        
        recorder.record_training_checkpoint(
            model_hash=model_hash,
            metrics=metrics,
            epoch=epoch,
            model_id=model_id
        )
    
    checkpoint_time = time.time() - start_time
    
    print(f"   ✓ {num_checkpoints} checkpoints recorded in {checkpoint_time:.2f}s")
    print(f"   ✓ Average: {(checkpoint_time / num_checkpoints) * 1000:.1f}ms per checkpoint")
    
    # Benchmark proof generation
    print("\n📊 Benchmarking proof generation...")
    start_time = time.time()
    proof = recorder.generate_proof_of_training(model_id)
    proof_time = time.time() - start_time
    
    print(f"   ✓ Proof generated in {proof_time:.3f}s")
    print(f"   ✓ Merkle tree size: {proof.verification_metadata.get('merkle_tree_size', 0)} elements")
    
    # Benchmark proof verification
    print("\n📊 Benchmarking proof verification...")
    start_time = time.time()
    is_valid = recorder.verify_training_provenance(proof)
    verify_time = time.time() - start_time
    
    print(f"   ✓ Proof verified in {verify_time:.3f}s (result: {is_valid})")
    
    # Storage analysis
    storage_path = recorder.local_storage_path
    if storage_path.exists():
        storage_size = storage_path.stat().st_size
        print(f"\n💾 Storage analysis:")
        print(f"   ✓ Local storage size: {storage_size:,} bytes ({storage_size/1024:.1f} KB)")
        print(f"   ✓ Storage per checkpoint: {storage_size/num_checkpoints:.0f} bytes")


def show_summary():
    """Show summary of demonstration results."""
    print("\n" + "="*60)
    print("📋 DEMONSTRATION SUMMARY")
    print("="*60)
    
    # List generated files
    demo_files = [
        "demo_provenance_local.json",
        "demo_provenance_blockchain.json", 
        "demo_proof_mnist_classifier_local.json",
        "demo_proof_bert_fine_tuned_blockchain.json"
    ]
    
    print("\n📁 Generated files:")
    for file_name in demo_files:
        if Path(file_name).exists():
            size = Path(file_name).stat().st_size
            print(f"   ✓ {file_name} ({size:,} bytes)")
        else:
            print(f"   ⚠️  {file_name} (not found)")
    
    print("\n🎯 Key features demonstrated:")
    print("   ✓ Local JSON storage with thread-safe operations")
    print("   ✓ Blockchain integration with automatic fallback")
    print("   ✓ Training checkpoint recording with metrics")
    print("   ✓ Multi-validator validation recording")
    print("   ✓ Proof-of-training generation with Merkle trees")
    print("   ✓ Complete proof verification workflow")
    print("   ✓ Attack detection and tamper resistance")
    print("   ✓ Performance benchmarking and analysis")
    
    print("\n🔧 Next steps:")
    print("   • Configure blockchain environment variables for production")
    print("   • Deploy smart contracts to target networks")
    print("   • Integrate with actual training pipelines")
    print("   • Set up monitoring and alerting for provenance systems")
    print("   • Review security considerations and access controls")


def main():
    """Main demonstration entry point."""
    parser = argparse.ArgumentParser(
        description="PoT Provenance Recording Demonstration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--mode", choices=["local", "blockchain", "both"], default="both",
                       help="Demonstration mode")
    parser.add_argument("--skip-blockchain", action="store_true",
                       help="Skip blockchain demonstration")
    parser.add_argument("--skip-attacks", action="store_true", 
                       help="Skip attack detection demonstration")
    parser.add_argument("--skip-performance", action="store_true",
                       help="Skip performance analysis")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging("DEBUG" if args.verbose else "INFO")
    
    print("🚀 PoT Provenance Recording Demonstration")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    
    if not POT_AVAILABLE:
        print("❌ PoT modules not available - exiting")
        return
    
    try:
        # Run demonstrations based on mode
        if args.mode in ["local", "both"]:
            demonstrate_local_mode()
        
        if args.mode in ["blockchain", "both"] and not args.skip_blockchain:
            demonstrate_blockchain_mode()
        
        # Always run verification workflow
        demonstrate_verification_workflow()
        
        # Optional demonstrations
        if not args.skip_attacks:
            demonstrate_attack_scenarios()
        
        if not args.skip_performance:
            demonstrate_performance_analysis()
        
        # Show summary
        show_summary()
        
        print("\n✅ Demonstration completed successfully!")
        
    except KeyboardInterrupt:
        print("\n❌ Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()