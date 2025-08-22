#!/usr/bin/env python3
"""
FULL ANALYTICAL PIPELINE FOR GOOGLE COLAB
==========================================
This runs the COMPLETE PoT framework tests, not mock versions.
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path

def setup_colab_environment():
    """Setup the complete Colab environment."""
    print("=" * 70)
    print("POT EXPERIMENTS - FULL ANALYTICAL PIPELINE")
    print("=" * 70)
    print("This runs the COMPLETE PoT framework tests")
    print("Using open models: GPT-2 and DistilGPT-2")
    print("=" * 70)
    
    # Ensure we're in the right directory
    if os.path.exists('/content'):
        os.chdir('/content')
    else:
        os.chdir(os.path.expanduser('~'))
    
    print(f"\n📍 Base directory: {os.getcwd()}")
    
    # Clean and clone
    print("\n📥 Setting up repository...")
    if os.path.exists('PoT_Experiments'):
        subprocess.run(['rm', '-rf', 'PoT_Experiments'], check=False)
    
    result = subprocess.run(
        ['git', 'clone', 'https://github.com/rohanvinaik/PoT_Experiments.git'],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"❌ Clone failed: {result.stderr}")
        return False
    
    os.chdir('PoT_Experiments')
    print(f"✅ Repository ready at: {os.getcwd()}")
    
    # Install ALL required dependencies
    print("\n📦 Installing complete dependencies...")
    packages = [
        'torch',
        'transformers',
        'numpy',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'pandas',
        'tqdm'
    ]
    
    for pkg in packages:
        print(f"  Installing {pkg}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)
    
    # Try to install fuzzy hash libraries
    print("  Installing fuzzy hash libraries...")
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'tlsh-py'], check=False)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ssdeep'], check=False)
    
    print("✅ Dependencies installed")
    
    return True

def run_full_test(script_path, test_name, timeout=300):
    """Run a full test from the actual codebase."""
    print(f"\n{'='*60}")
    print(f"Running: {test_name}")
    print(f"Script: {script_path}")
    print('='*60)
    
    if not os.path.exists(script_path):
        print(f"❌ Script not found: {script_path}")
        return False, "Script not found"
    
    try:
        # Set PYTHONPATH to include current directory
        env = os.environ.copy()
        env['PYTHONPATH'] = os.getcwd()
        
        # Run the actual test script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env
        )
        
        # Print full output
        if result.stdout:
            print(result.stdout)
        
        # Print errors if any (filter warnings)
        if result.stderr:
            error_lines = [l for l in result.stderr.split('\n') 
                          if l and 'Warning' not in l and 'FutureWarning' not in l]
            if error_lines:
                print("Errors:", '\n'.join(error_lines[:10]))  # Limit error output
        
        success = result.returncode == 0
        status = "PASS" if success else f"FAIL (code {result.returncode})"
        
        return success, status
        
    except subprocess.TimeoutExpired:
        print(f"❌ Test timed out after {timeout} seconds")
        return False, "TIMEOUT"
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, str(e)

def analyze_results():
    """Analyze the generated results files."""
    print("\n" + "=" * 70)
    print("📊 ANALYZING RESULTS")
    print("=" * 70)
    
    exp_dir = Path("experimental_results")
    if not exp_dir.exists():
        print("No results directory found")
        return
    
    # List all JSON files
    json_files = list(exp_dir.glob("*.json"))
    if not json_files:
        print("No result files found")
        return
    
    print(f"\nFound {len(json_files)} result files:")
    
    for jf in sorted(json_files, key=os.path.getmtime, reverse=True)[:10]:
        size_kb = jf.stat().st_size / 1024
        print(f"\n📄 {jf.name} ({size_kb:.1f} KB)")
        
        try:
            with open(jf) as f:
                data = json.load(f)
                
            # Show key metrics based on file type
            if 'statistical' in jf.name.lower():
                print("  Type: Statistical Identity Verification")
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, dict):
                            decision = value.get('decision', 'N/A')
                            n_used = value.get('n_used', 'N/A')
                            print(f"    {key}: decision={decision}, samples={n_used}")
            
            elif 'llm' in jf.name.lower():
                print("  Type: LLM Verification")
                if 'models' in data:
                    print(f"    Models tested: {data['models']}")
                if 'status' in data:
                    print(f"    Status: {data['status']}")
            
            elif 'fuzzy' in jf.name.lower():
                print("  Type: Fuzzy Hash Verification")
                if 'algorithms_tested' in data:
                    print(f"    Algorithms: {data['algorithms_tested']}")
            
            elif 'provenance' in jf.name.lower():
                print("  Type: Provenance Audit")
                if 'merkle_root' in data:
                    print(f"    Merkle root: {data['merkle_root'][:32]}...")
            
            elif 'report' in jf.name.lower():
                print("  Type: Clean Report")
                if 'verification_summary' in data:
                    summary = data['verification_summary']
                    print(f"    Overall: {summary.get('overall_status', 'N/A')}")
                    
        except Exception as e:
            print(f"    Error reading file: {e}")

def main():
    """Main execution of full pipeline."""
    
    # Setup environment
    if not setup_colab_environment():
        print("❌ Failed to setup environment")
        return 1
    
    # Check device
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n🖥️ Device: {device}")
        if device == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name()}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except:
        device = "cpu"
        print(f"\n🖥️ Device: {device}")
    
    # Create results directory
    Path("experimental_results").mkdir(exist_ok=True)
    
    # Define the FULL test suite
    test_suite = [
        ("scripts/run_statistical_verification.py", "Statistical Identity Verification", 120),
        ("scripts/test_llm_open_models_only.py", "LLM Verification (Open Models)", 300),
        ("scripts/run_fuzzy_verification.py", "Fuzzy Hash Verification", 120),
        ("scripts/run_provenance_verification.py", "Provenance Auditing", 120),
        ("scripts/experimental_report_clean.py", "Clean Reporting Format", 120)
    ]
    
    # Run the full test suite
    print("\n" + "=" * 70)
    print("🚀 RUNNING FULL TEST SUITE")
    print("=" * 70)
    
    results = {}
    start_time = time.time()
    
    for script, name, timeout in test_suite:
        test_start = time.time()
        success, status = run_full_test(script, name, timeout)
        test_time = time.time() - test_start
        
        results[name] = {
            "success": success,
            "status": status,
            "time": round(test_time, 2)
        }
        
        print(f"\n{'✅' if success else '❌'} {name}")
        print(f"   Status: {status}")
        print(f"   Time: {test_time:.1f}s")
    
    total_time = time.time() - start_time
    
    # Analyze results
    analyze_results()
    
    # Generate final summary
    print("\n" + "=" * 70)
    print("📈 FINAL SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)
    
    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success Rate: {(passed/total)*100:.0f}%")
    print(f"Total Time: {total_time:.1f}s")
    
    print("\nDetailed Results:")
    for test, info in results.items():
        status_icon = "✅" if info['success'] else "❌"
        print(f"  {status_icon} {test}")
        print(f"      Status: {info['status']}, Time: {info['time']}s")
    
    # Save comprehensive summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "environment": "Google Colab",
        "device": device,
        "pipeline": "FULL",
        "models_used": ["gpt2", "distilgpt2"],
        "total_tests": total,
        "passed": passed,
        "failed": total - passed,
        "success_rate": passed/total if total > 0 else 0,
        "total_time": round(total_time, 2),
        "test_results": results
    }
    
    summary_path = "experimental_results/full_pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📄 Full summary saved to: {summary_path}")
    
    # Final message
    print("\n" + "=" * 70)
    if passed == total:
        print("🎉 ALL TESTS PASSED!")
        print("The complete PoT analytical pipeline executed successfully.")
    elif passed > 0:
        print(f"⚠️ PARTIAL SUCCESS: {passed}/{total} tests passed")
        print("Review the output above for details on failures.")
    else:
        print("❌ All tests failed - check the output for errors")
    
    print("\n📝 Evidence for your paper:")
    print("  • Statistical identity verification with confidence intervals")
    print("  • LLM verification using GPT-2 and DistilGPT-2")
    print("  • Fuzzy hash algorithm comparisons")
    print("  • Merkle tree provenance proofs")
    print("  • Clean reporting format with all metrics")
    print("\nAll tests used open models - no authentication required!")
    print("=" * 70)
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())