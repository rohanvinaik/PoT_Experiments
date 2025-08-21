#!/usr/bin/env python3
import sys
import os
import gc
import json
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, "/Users/rohanvinaik/PoT_Experiments")

# Force CPU only and limit threads
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'

print("Running minimal statistical test...")
print("⚠️ Using only 3 queries to minimize memory usage")

try:
    from pot.core.diff_decision import EnhancedSequentialTester, TestingMode
    
    # Use the most conservative settings
    tester = EnhancedSequentialTester(
        mode=TestingMode.QUICK_GATE,
        n_max=3,  # Only 3 queries
        verbose=True
    )
    
    # Models
    model1 = "/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Base-Q4"
    model2 = "/Users/rohanvinaik/LLM_Models/Mixtral-8x22B-Instruct-Q4"
    
    print(f"Model 1: {Path(model1).name}")
    print(f"Model 2: {Path(model2).name}")
    
    # Note: This may fail if models are too large
    print("\n⚠️ Attempting to load models...")
    print("If this hangs, press Ctrl+C to skip")
    
    # We'll use a timeout approach
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Model loading timed out")
    
    # Set 60 second timeout for loading
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(60)
    
    try:
        # Try to run test
        result = tester.test_models(model1, model2)
        signal.alarm(0)  # Cancel alarm
        
        print(f"\nDecision: {result.decision}")
        print(f"Confidence: {result.confidence:.2%}")
        
        # Save result
        with open("$OUTPUT_DIR/statistical_result.json", "w") as f:
            json.dump({
                "decision": result.decision,
                "confidence": result.confidence,
                "n_samples": result.n_samples
            }, f, indent=2)
            
    except TimeoutError:
        print("\n❌ Model loading timed out - models too large")
        print("Skipping statistical test")
    except Exception as e:
        print(f"\n❌ Statistical test failed: {e}")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Framework not properly installed")
except Exception as e:
    print(f"❌ Unexpected error: {e}")

# Cleanup
gc.collect()
print("\n✅ Test completed")
