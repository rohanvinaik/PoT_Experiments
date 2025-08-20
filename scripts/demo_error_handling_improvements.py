#!/usr/bin/env python3
"""
Demo of improved ZK error handling showing key improvements over hardcoded fallbacks.
"""

import sys
import logging
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

# Setup logging  
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent.parent))

from pot.zk.exceptions import (
    ProverNotFoundError, ProofGenerationError, InvalidModelStateError, RetryExhaustedError
)
from pot.zk.auto_prover import AutoProver
from pot.zk.auditor_integration import create_zk_integration, ProofFailureAction


def demo_old_vs_new_fallback():
    """Demonstrate the improvement from hardcoded fallback to proper error handling."""
    print("=" * 60)
    print("OLD vs NEW: Hardcoded Fallback Replacement")
    print("=" * 60)
    
    print("\n❌ OLD APPROACH (Before):")
    print("   - Hardcoded: return b'full_sgd_proof', 'full'")
    print("   - No error information")
    print("   - Silent failures masked")
    print("   - No retry logic")
    print("   - No configuration options")
    
    print("\n✅ NEW APPROACH (After):")
    print("   - Proper exception hierarchy")
    print("   - Detailed error messages with context")
    print("   - Configurable failure handling")
    print("   - Retry logic with exponential backoff")
    print("   - Integration with audit systems")
    print("   - Performance monitoring")


def demo_exception_hierarchy():
    """Demonstrate the comprehensive exception hierarchy."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE EXCEPTION HIERARCHY")
    print("=" * 60)
    
    exceptions_demo = [
        ("ProverNotFoundError", lambda: ProverNotFoundError("prove_lora_stdin", ["/usr/bin", "/opt/bin"])),
        ("ProofGenerationError", lambda: ProofGenerationError("Circuit constraint violation", "prove_sgd_stdin", 1, "Invalid witness")),
        ("InvalidModelStateError", lambda: InvalidModelStateError("Architecture mismatch", "LoRA", "SGD")),
        ("RetryExhaustedError", lambda: RetryExhaustedError("Proof generation", 3, ProofGenerationError("Last error"))),
    ]
    
    for name, creator in exceptions_demo:
        error = creator()
        print(f"\n{name}:")
        print(f"  📝 Message: {str(error)}")
        print(f"  🔍 Type: {type(error).__name__}")
        
        # Show additional attributes
        if hasattr(error, 'binary_name'):
            print(f"  🔧 Binary: {error.binary_name}")
        if hasattr(error, 'search_paths'):
            print(f"  📂 Paths: {error.search_paths}")
        if hasattr(error, 'attempts'):
            print(f"  🔁 Attempts: {error.attempts}")


def demo_model_type_detection():
    """Demonstrate automatic model type detection with error handling."""
    print("\n" + "=" * 60)
    print("INTELLIGENT MODEL TYPE DETECTION")
    print("=" * 60)
    
    prover = AutoProver()
    
    # Test cases with different model types
    test_cases = [
        ("Standard SGD Model", {'weights': np.random.randn(32, 32), 'bias': np.random.randn(32)}),
        ("LoRA Fine-tuned Model", {
            'lora_A.weight': np.random.randn(768, 16),
            'lora_B.weight': np.random.randn(16, 768),
            'base.weight': np.random.randn(768, 768)
        }),
        ("Invalid Empty Model", {}),
        ("Malformed LoRA Model", {'lora_A.weight': np.random.randn(100, 16)}),  # Missing lora_B
    ]
    
    for name, model in test_cases:
        print(f"\n{name}:")
        try:
            if name == "Invalid Empty Model":
                # This should fail
                model_type = prover.detect_model_type(model, model)
            elif name == "Malformed LoRA Model":
                # This should fail during adapter extraction
                model_type = prover.detect_model_type(model, model)
            else:
                model_type = prover.detect_model_type(model, model)
            
            print(f"  ✅ Detected: {model_type}")
            
            # Show compression info for LoRA
            if model_type == "lora" and 'lora_A.weight' in model:
                total_params = sum(np.prod(tensor.shape) for tensor in model.values())
                lora_params = np.prod(model['lora_A.weight'].shape) + np.prod(model['lora_B.weight'].shape)
                compression = total_params / lora_params if lora_params > 0 else 1
                print(f"  📊 Compression: {compression:.1f}x reduction")
                
        except InvalidModelStateError as e:
            print(f"  ❌ Invalid model: {str(e).split('(')[0].strip()}")
        except Exception as e:
            print(f"  ⚠️ Error: {e}")


def demo_retry_logic():
    """Demonstrate retry logic with exponential backoff."""
    print("\n" + "=" * 60)  
    print("RETRY LOGIC WITH EXPONENTIAL BACKOFF")
    print("=" * 60)
    
    print("\nSimulating transient failures...")
    
    # Create a function that fails a few times then succeeds
    attempt_count = 0
    def flaky_function():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count <= 2:
            raise ProofGenerationError(f"Transient failure #{attempt_count}")
        return {"success": True, "proof": b"recovered_proof", "proof_type": "sgd"}
    
    prover = AutoProver(max_retries=3, retry_delay=0.1, retry_exponential=True)
    
    import time
    start_time = time.time()
    
    try:
        result = prover._retry_with_backoff(flaky_function)
        elapsed = time.time() - start_time
        
        print(f"✅ Recovery successful after {attempt_count} attempts")
        print(f"⏱️ Total time: {elapsed:.2f}s (with exponential backoff)")
        print(f"📊 Proof type: {result['proof_type']}")
        print(f"💾 Proof size: {len(result['proof'])} bytes")
        
    except RetryExhaustedError as e:
        print(f"❌ All retries exhausted: {e}")


def demo_auditor_integration_modes():
    """Demonstrate different auditor integration modes."""
    print("\n" + "=" * 60)
    print("AUDITOR INTEGRATION MODES")
    print("=" * 60)
    
    modes = [
        ("Development", "continue", "Continue training on proof failures"),
        ("Production", "fail_fast", "Fail immediately on any proof error"),
        ("Research", "continue", "Log failures but keep training"),
        ("Disabled", "continue", "No proof generation (disabled)"),
    ]
    
    for mode_name, failure_action, description in modes:
        print(f"\n{mode_name} Mode:")
        print(f"  📝 Description: {description}")
        
        enabled = mode_name != "Disabled"
        integration = create_zk_integration(
            enabled=enabled,
            failure_action=failure_action,
            max_retries=1 if mode_name == "Development" else 0
        )
        
        print(f"  ⚙️ Enabled: {integration.config.enabled}")
        print(f"  🚨 Failure action: {integration.config.failure_action.value}")
        print(f"  🔁 Max retries: {integration.config.max_retries}")
        
        # Simulate proof generation
        if enabled:
            # Mock a proof generation attempt
            stats = {
                'total_attempts': 10,
                'successful_proofs': 8 if mode_name != "Production" else 10,
                'failed_proofs': 2 if mode_name != "Production" else 0,
            }
            stats['success_rate'] = stats['successful_proofs'] / stats['total_attempts']
            
            print(f"  📊 Simulated stats: {stats['success_rate']:.1%} success rate")
            
            if stats['failed_proofs'] > 0:
                if failure_action == "continue":
                    print(f"  ✅ Training continued despite {stats['failed_proofs']} proof failures")
                elif failure_action == "fail_fast": 
                    print(f"  ❌ Training would halt on first failure")


def demo_comprehensive_error_info():
    """Demonstrate comprehensive error information."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ERROR INFORMATION")
    print("=" * 60)
    
    # Create a realistic error scenario
    error = ProofGenerationError(
        "Halo2 circuit constraint violation at gate 42",
        binary_name="prove_lora_stdin",
        exit_code=2,
        stderr="Error: Invalid witness data\nConstraint failed: adapter_a[5] * adapter_b[5] != expected_product"
    )
    
    print("\nDetailed Error Information:")
    print(f"  🚨 Error: {error}")
    print(f"  🔧 Binary: {error.binary_name}")
    print(f"  💥 Exit code: {error.exit_code}")
    print(f"  📜 Stderr output:")
    for line in error.stderr.split('\n'):
        if line.strip():
            print(f"      {line}")
    
    print("\n🔍 This level of detail enables:")
    print("  - Precise debugging of circuit constraints")
    print("  - Binary-specific troubleshooting") 
    print("  - Automated error classification")
    print("  - Integration with monitoring systems")


def main():
    """Run all demonstrations."""
    print("🚀 IMPROVED ZK ERROR HANDLING DEMONSTRATION")
    print("Showing key improvements over hardcoded b'full_sgd_proof' fallbacks")
    
    demos = [
        demo_old_vs_new_fallback,
        demo_exception_hierarchy,
        demo_model_type_detection,
        demo_retry_logic,
        demo_auditor_integration_modes,
        demo_comprehensive_error_info,
    ]
    
    try:
        for demo in demos:
            demo()
        
        # Summary
        print("\n" + "=" * 60)
        print("🎉 SUMMARY OF IMPROVEMENTS")
        print("=" * 60)
        
        improvements = [
            "❌ Removed hardcoded b'full_sgd_proof' fallback",
            "✅ Added comprehensive exception hierarchy",
            "✅ Implemented intelligent model type detection",
            "✅ Added retry logic with exponential backoff",
            "✅ Created configurable failure handling modes",
            "✅ Integrated with audit systems",
            "✅ Added detailed error reporting",
            "✅ Enabled production-grade error handling",
        ]
        
        for improvement in improvements:
            print(f"  {improvement}")
        
        print(f"\n🏆 Result: Production-ready ZK proof system with proper error handling!")
        print(f"🔒 Benefits: Reliable, debuggable, and configurable proof generation")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())