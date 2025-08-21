#!/usr/bin/env python3
"""
Direct Mixtral model testing with proper resource management.
Tests the Mixtral-8x22B Base vs Instruct Q4 models.
"""

import os
import sys
import json
import hashlib
import time
import gc
import traceback
from pathlib import Path
from datetime import datetime

# Set resource limits BEFORE any imports
os.environ['OMP_NUM_THREADS'] = '6'
os.environ['MKL_NUM_THREADS'] = '6'
os.environ['TORCH_NUM_THREADS'] = '6'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_CACHE'] = '/tmp/transformers_cache'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU only

# Add parent directory for imports
sys.path.insert(0, '/Users/rohanvinaik/PoT_Experiments')

def check_models_exist():
    """Check if the Mixtral models exist."""
    models = {
        "base": "/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Base-Q4",
        "instruct": "/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Instruct-Q4"
    }
    
    print("=" * 60)
    print("CHECKING MODEL AVAILABILITY")
    print("=" * 60)
    
    for name, path in models.items():
        model_path = Path(path)
        if model_path.exists():
            # Check for config.json
            config_path = model_path / "config.json"
            if config_path.exists():
                print(f"✅ {name}: Found at {path}")
                # Get model size
                total_size = sum(f.stat().st_size for f in model_path.glob("**/*") if f.is_file())
                print(f"   Size: {total_size / (1024**3):.1f} GB")
            else:
                print(f"⚠️ {name}: Directory exists but no config.json")
        else:
            print(f"❌ {name}: Not found at {path}")
    
    return models

def compare_configs(models):
    """Compare model configurations without loading models."""
    print("\n" + "=" * 60)
    print("CONFIGURATION COMPARISON")
    print("=" * 60)
    
    configs = {}
    for name, path in models.items():
        config_path = Path(path) / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract key information
            configs[name] = {
                "architecture": config.get("architectures", ["Unknown"])[0],
                "model_type": config.get("model_type", "Unknown"),
                "hidden_size": config.get("hidden_size", 0),
                "num_hidden_layers": config.get("num_hidden_layers", 0),
                "num_attention_heads": config.get("num_attention_heads", 0),
                "num_key_value_heads": config.get("num_key_value_heads", 0),
                "num_local_experts": config.get("num_local_experts", 0),
                "num_experts_per_tok": config.get("num_experts_per_tok", 0),
                "vocab_size": config.get("vocab_size", 0),
                "max_position_embeddings": config.get("max_position_embeddings", 0),
                "rope_theta": config.get("rope_theta", 0),
            }
            
            # Compute config hash
            config_str = json.dumps(config, sort_keys=True)
            configs[name]["hash"] = hashlib.sha256(config_str.encode()).hexdigest()[:16]
            
            print(f"\n{name.upper()} Model Configuration:")
            for key, value in configs[name].items():
                if key != "hash":
                    print(f"  {key}: {value}")
    
    # Compare configurations
    if len(configs) == 2:
        base_config = configs.get("base", {})
        instruct_config = configs.get("instruct", {})
        
        print("\n" + "-" * 40)
        print("CONFIGURATION DIFFERENCES:")
        
        all_same = True
        for key in base_config:
            if key == "hash":
                continue
            if base_config[key] != instruct_config.get(key):
                print(f"  {key}: {base_config[key]} (base) vs {instruct_config.get(key)} (instruct)")
                all_same = False
        
        if all_same:
            print("  ✅ Configurations are IDENTICAL (same architecture)")
        else:
            print("  ⚠️ Configurations DIFFER")
        
        if base_config.get("hash") == instruct_config.get("hash"):
            print("  ✅ Config hashes MATCH - truly identical configs")
        else:
            print("  ❌ Config hashes DIFFER - models are different")
    
    return configs

def test_with_pot_framework(models):
    """Test using the actual PoT framework with minimal resources."""
    print("\n" + "=" * 60)
    print("POT FRAMEWORK VERIFICATION")
    print("=" * 60)
    
    try:
        # Import PoT components
        from pot.core.diff_decision import EnhancedSequentialTester, TestingMode
        
        print("✅ PoT framework imported successfully")
        
        # Create tester with minimal settings
        print("\nInitializing tester with minimal resources...")
        tester = EnhancedSequentialTester(
            mode=TestingMode.QUICK_GATE,
            n_max=5,  # Only 5 queries maximum
            n_min=3,  # Minimum 3 queries
            verbose=True
        )
        
        print(f"Configuration:")
        print(f"  Mode: QUICK_GATE")
        print(f"  Max queries: 5")
        print(f"  Confidence target: 97.5%")
        
        print("\n⚠️ WARNING: Loading 80GB quantized models...")
        print("This will use significant memory. Monitoring system...")
        
        # Memory check before
        import subprocess
        result = subprocess.run("top -l 1 -n 0 | grep PhysMem", 
                              shell=True, capture_output=True, text=True)
        print(f"Memory before: {result.stdout.strip()}")
        
        print("\nStarting verification (this may take several minutes)...")
        start_time = time.time()
        
        try:
            # Run the test with timeout
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Test timed out after 5 minutes")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(300)  # 5 minute timeout
            
            result = tester.test_models(
                models["base"],
                models["instruct"]
            )
            
            signal.alarm(0)  # Cancel alarm
            
            elapsed = time.time() - start_time
            
            print(f"\n" + "=" * 40)
            print("VERIFICATION RESULTS:")
            print("=" * 40)
            print(f"Decision: {result.decision}")
            print(f"Confidence: {result.confidence:.2%}")
            print(f"Samples used: {result.n_samples}")
            print(f"Time elapsed: {elapsed:.1f} seconds")
            
            if hasattr(result, 'effect_size'):
                print(f"Effect size: {result.effect_size:.3f}")
            if hasattr(result, 'mean_diff'):
                print(f"Mean difference: {result.mean_diff:.6f}")
            
            # Memory check after
            result_mem = subprocess.run("top -l 1 -n 0 | grep PhysMem", 
                                      shell=True, capture_output=True, text=True)
            print(f"\nMemory after: {result_mem.stdout.strip()}")
            
            return {
                "success": True,
                "decision": result.decision,
                "confidence": result.confidence,
                "n_samples": result.n_samples,
                "elapsed_time": elapsed
            }
            
        except TimeoutError as e:
            print(f"\n❌ Test timed out: {e}")
            return {"success": False, "error": "timeout", "message": str(e)}
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            traceback.print_exc()
            return {"success": False, "error": "execution", "message": str(e)}
            
    except ImportError as e:
        print(f"❌ Could not import PoT framework: {e}")
        return {"success": False, "error": "import", "message": str(e)}
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return {"success": False, "error": "unexpected", "message": str(e)}
    finally:
        # Force garbage collection
        gc.collect()

def generate_report(models, config_results, pot_results):
    """Generate final test report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"/Users/rohanvinaik/PoT_Experiments/experimental_results/mixtral_test_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "models": models,
        "configuration_analysis": config_results,
        "pot_verification": pot_results,
        "system_info": {
            "cpu_threads": os.environ.get('OMP_NUM_THREADS', 'default'),
            "tokenizers_parallel": os.environ.get('TOKENIZERS_PARALLELISM', 'true')
        }
    }
    
    # Save JSON report
    report_path = output_dir / "test_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Report saved to: {report_path}")
    print(f"\nModels tested:")
    print(f"  Base: Mixtral-8x22B-Base-Q4")
    print(f"  Instruct: Mixtral-8x22B-Instruct-Q4")
    
    if pot_results.get("success"):
        print(f"\n✅ Verification completed successfully")
        print(f"  Decision: {pot_results['decision']}")
        print(f"  Confidence: {pot_results['confidence']:.2%}")
        print(f"  Time: {pot_results['elapsed_time']:.1f}s")
    else:
        print(f"\n⚠️ Verification could not complete")
        print(f"  Reason: {pot_results.get('error', 'unknown')}")
    
    return output_dir

def main():
    """Main execution."""
    print("=" * 60)
    print("MIXTRAL-8x22B MODEL VERIFICATION")
    print("Base-Q4 vs Instruct-Q4")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Resource limits: 6 CPU threads, CPU-only mode")
    
    # Step 1: Check models
    models = check_models_exist()
    
    # Step 2: Compare configurations
    config_results = compare_configs(models)
    
    # Step 3: Ask user before running memory-intensive test
    print("\n" + "=" * 60)
    print("⚠️ MEMORY WARNING")
    print("=" * 60)
    print("The next test will attempt to load the 80GB models.")
    print("This may use up to 50GB of RAM.")
    print("Make sure to save any work and close unnecessary applications.")
    
    response = input("\nProceed with PoT framework verification? (y/n): ")
    
    if response.lower() == 'y':
        pot_results = test_with_pot_framework(models)
    else:
        print("Skipping PoT framework verification")
        pot_results = {"success": False, "error": "skipped", "message": "User skipped test"}
    
    # Step 4: Generate report
    output_dir = generate_report(models, config_results, pot_results)
    
    print("\n" + "=" * 60)
    print("✅ Testing completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()