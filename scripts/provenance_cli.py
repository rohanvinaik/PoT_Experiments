#!/usr/bin/env python3
"""
Provenance CLI Tool for PoT Framework

Command-line interface for managing training provenance records, generating
proofs, and verifying training integrity using blockchain integration.

Usage Examples:
    # Initialize provenance recording
    python provenance_cli.py init --blockchain --model-id my_model

    # Record training checkpoint
    python provenance_cli.py checkpoint --model-hash 0x123... --epoch 10 --metrics '{"loss": 0.1}'

    # Record validation
    python provenance_cli.py validate --model-hash 0x123... --validator alice --result '{"accuracy": 0.95}'

    # Generate proof of training
    python provenance_cli.py proof --model-id my_model --output proof.json

    # Verify proof
    python provenance_cli.py verify --proof proof.json

    # View training history
    python provenance_cli.py history --model-id my_model
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from pot.core.provenance_integration import (
        ProvenanceRecorder, 
        ProvenanceConfig,
        create_provenance_recorder
    )
    from pot.security.blockchain_factory import test_blockchain_connection
    POT_AVAILABLE = True
except ImportError as e:
    print(f"Error importing PoT modules: {e}")
    print("Make sure you're running from the PoT_Experiments directory")
    POT_AVAILABLE = False


def init_provenance(args) -> None:
    """Initialize provenance recording system."""
    print("Initializing provenance recording...")
    
    config = ProvenanceConfig(
        enabled=True,
        blockchain_enabled=args.blockchain,
        local_storage_path=args.storage_path,
        fingerprint_checkpoints=args.fingerprint,
        record_challenges=args.challenges
    )
    
    recorder = ProvenanceRecorder(config)
    
    if args.blockchain and recorder.blockchain_client:
        print("Testing blockchain connection...")
        success, results = test_blockchain_connection(recorder.blockchain_client)
        if success:
            print(f"✓ Blockchain connection successful: {results['client_type']}")
        else:
            print("✗ Blockchain connection failed")
            return
    
    print(f"✓ Provenance recording initialized")
    print(f"  - Storage path: {recorder.local_storage_path}")
    print(f"  - Blockchain enabled: {config.blockchain_enabled}")
    print(f"  - Fingerprinting enabled: {config.fingerprint_checkpoints}")


def record_checkpoint(args) -> None:
    """Record a training checkpoint."""
    recorder = create_provenance_recorder(
        enabled=True,
        blockchain_enabled=args.blockchain
    )
    
    # Parse metrics if provided
    metrics = {}
    if args.metrics:
        try:
            metrics = json.loads(args.metrics)
        except json.JSONDecodeError:
            print("Error: Invalid JSON format for metrics")
            return
    
    checkpoint_id = recorder.record_training_checkpoint(
        model_hash=args.model_hash,
        metrics=metrics,
        epoch=args.epoch,
        model_id=args.model_id
    )
    
    print(f"✓ Training checkpoint recorded: {checkpoint_id}")
    print(f"  - Model: {args.model_id}")
    print(f"  - Epoch: {args.epoch}")
    print(f"  - Hash: {args.model_hash[:16]}...")


def record_validation(args) -> None:
    """Record a validation result."""
    recorder = create_provenance_recorder(
        enabled=True,
        blockchain_enabled=args.blockchain
    )
    
    # Parse validation result
    try:
        validation_result = json.loads(args.result)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format for validation result")
        return
    
    validation_id = recorder.record_validation(
        model_hash=args.model_hash,
        validator_id=args.validator,
        validation_result=validation_result,
        model_id=args.model_id
    )
    
    print(f"✓ Validation recorded: {validation_id}")
    print(f"  - Model: {args.model_id}")
    print(f"  - Validator: {args.validator}")
    print(f"  - Hash: {args.model_hash[:16]}...")


def generate_proof(args) -> None:
    """Generate proof of training."""
    recorder = create_provenance_recorder(enabled=True)
    recorder.load_local_records()
    
    try:
        proof = recorder.generate_proof_of_training(args.model_id)
        
        # Save proof to file
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            # Convert dataclass to dict for JSON serialization
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
            json.dump(proof_dict, f, indent=2)
        
        print(f"✓ Proof of training generated: {output_path}")
        print(f"  - Model: {proof.model_id}")
        print(f"  - Final hash: {proof.final_model_hash[:16]}...")
        print(f"  - Checkpoints: {proof.verification_metadata['total_checkpoints']}")
        print(f"  - Validations: {proof.verification_metadata['total_validations']}")
        print(f"  - Merkle root: {proof.merkle_root[:16]}...")
        
    except ValueError as e:
        print(f"Error generating proof: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


def verify_proof(args) -> None:
    """Verify a proof of training."""
    try:
        with open(args.proof, 'r') as f:
            proof_dict = json.load(f)
        
        # Reconstruct proof object (simplified for CLI)
        print("Verifying proof of training...")
        print(f"  - Model: {proof_dict['model_id']}")
        print(f"  - Final hash: {proof_dict['final_model_hash'][:16]}...")
        print(f"  - Checkpoints: {proof_dict['verification_metadata']['total_checkpoints']}")
        print(f"  - Validations: {proof_dict['verification_metadata']['total_validations']}")
        
        # Basic verification (signature check)
        import hashlib
        proof_content = {
            "model_id": proof_dict["model_id"],
            "final_model_hash": proof_dict["final_model_hash"],
            "merkle_root": proof_dict["merkle_root"],
            "metadata": proof_dict["verification_metadata"]
        }
        expected_signature = hashlib.sha256(
            json.dumps(proof_content, sort_keys=True).encode()
        ).hexdigest()
        
        if expected_signature == proof_dict["signature_hash"]:
            print("✓ Proof signature verified")
        else:
            print("✗ Proof signature verification failed")
            return
        
        print("✓ Proof of training verified successfully")
        
    except FileNotFoundError:
        print(f"Error: Proof file not found: {args.proof}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in proof file")
    except Exception as e:
        print(f"Error verifying proof: {e}")


def show_history(args) -> None:
    """Show training history for a model."""
    recorder = create_provenance_recorder(enabled=True)
    recorder.load_local_records()
    
    history = recorder.get_model_history(args.model_id)
    
    print(f"Training History for Model: {args.model_id}")
    print("=" * 50)
    
    if not history["checkpoints"]:
        print("No training checkpoints found")
        return
    
    print(f"Total Epochs: {history['total_epochs']}")
    print(f"Total Validations: {history['total_validations']}")
    print(f"Blockchain Transactions: {history['blockchain_transactions']}")
    print()
    
    print("Training Checkpoints:")
    for cp in sorted(history["checkpoints"], key=lambda x: x.epoch):
        print(f"  Epoch {cp.epoch:3d}: {cp.model_hash[:16]}... (metrics: {cp.metrics})")
    
    if history["validations"]:
        print("\nValidation Records:")
        for vr in history["validations"]:
            result_summary = {k: v for k, v in vr.validation_result.items() if k in ["accuracy", "loss", "confidence"]}
            print(f"  {vr.validator_id}: {vr.model_hash[:16]}... (result: {result_summary})")


def test_connection(args) -> None:
    """Test blockchain connection."""
    print("Testing blockchain connection...")
    
    from pot.security.blockchain_factory import get_blockchain_client, test_blockchain_connection
    
    try:
        client = get_blockchain_client(force_local=not args.blockchain)
        success, results = test_blockchain_connection(client)
        
        print(f"Client Type: {results['client_type']}")
        print(f"Connection Status: {'✓ Success' if success else '✗ Failed'}")
        
        if success:
            for test_name, test_result in results.get("tests", {}).items():
                status = "✓" if test_result["success"] else "✗"
                print(f"  {test_name}: {status} {test_result['details']}")
        else:
            print(f"Error: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Connection test failed: {e}")


def main():
    """Main CLI entry point."""
    if not POT_AVAILABLE:
        print("Error: PoT modules not available")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(
        description="Provenance CLI for PoT Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize provenance recording")
    init_parser.add_argument("--blockchain", action="store_true", help="Enable blockchain integration")
    init_parser.add_argument("--storage-path", default="./provenance_records.json", help="Local storage path")
    init_parser.add_argument("--fingerprint", action="store_true", help="Enable fingerprinting")
    init_parser.add_argument("--challenges", action="store_true", help="Record challenge hashes")
    init_parser.set_defaults(func=init_provenance)
    
    # Checkpoint command
    checkpoint_parser = subparsers.add_parser("checkpoint", help="Record training checkpoint")
    checkpoint_parser.add_argument("--model-hash", required=True, help="Model hash")
    checkpoint_parser.add_argument("--epoch", type=int, required=True, help="Training epoch")
    checkpoint_parser.add_argument("--metrics", help="Training metrics (JSON format)")
    checkpoint_parser.add_argument("--model-id", default="default_model", help="Model identifier")
    checkpoint_parser.add_argument("--blockchain", action="store_true", help="Record to blockchain")
    checkpoint_parser.set_defaults(func=record_checkpoint)
    
    # Validation command
    validate_parser = subparsers.add_parser("validate", help="Record validation result")
    validate_parser.add_argument("--model-hash", required=True, help="Model hash")
    validate_parser.add_argument("--validator", required=True, help="Validator ID")
    validate_parser.add_argument("--result", required=True, help="Validation result (JSON format)")
    validate_parser.add_argument("--model-id", default="default_model", help="Model identifier")
    validate_parser.add_argument("--blockchain", action="store_true", help="Record to blockchain")
    validate_parser.set_defaults(func=record_validation)
    
    # Proof command
    proof_parser = subparsers.add_parser("proof", help="Generate proof of training")
    proof_parser.add_argument("--model-id", required=True, help="Model identifier")
    proof_parser.add_argument("--output", default="proof.json", help="Output file path")
    proof_parser.set_defaults(func=generate_proof)
    
    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify proof of training")
    verify_parser.add_argument("--proof", required=True, help="Proof file path")
    verify_parser.set_defaults(func=verify_proof)
    
    # History command
    history_parser = subparsers.add_parser("history", help="Show training history")
    history_parser.add_argument("--model-id", required=True, help="Model identifier")
    history_parser.set_defaults(func=show_history)
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Test blockchain connection")
    test_parser.add_argument("--blockchain", action="store_true", help="Test blockchain client")
    test_parser.set_defaults(func=test_connection)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()