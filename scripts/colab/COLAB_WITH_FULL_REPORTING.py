#!/usr/bin/env python3
"""
GOOGLE COLAB RUNNER WITH FULL REPORTING PIPELINE
=================================================
This actually uses the complete PoT reporting system.
Generates detailed JSON and text reports with all metrics.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

def main():
    print("=" * 70)
    print("POT FRAMEWORK - FULL ANALYTICAL PIPELINE WITH REPORTING")
    print("=" * 70)
    print("This runs the complete PoT framework including:")
    print("  • Statistical identity verification")
    print("  • LLM verification (GPT-2/DistilGPT-2)")
    print("  • Fuzzy hash verification")
    print("  • Provenance auditing")
    print("  • FULL REPORTING PIPELINE")
    print("=" * 70)
    
    # Setup environment
    if os.path.exists('/content'):
        os.chdir('/content')
    
    # Clean and clone
    print("\n📥 Setting up repository...")
    if os.path.exists('PoT_Experiments'):
        subprocess.run(['rm', '-rf', 'PoT_Experiments'], check=False)
    
    subprocess.run(['git', 'clone', 'https://github.com/rohanvinaik/PoT_Experiments.git'], check=True)
    os.chdir('PoT_Experiments')
    print(f"📍 Working directory: {os.getcwd()}")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    packages = ['torch', 'transformers', 'numpy', 'scipy', 'scikit-learn', 'matplotlib', 'pandas']
    for pkg in packages:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg], check=False)
    
    # Create results directory
    Path("experimental_results").mkdir(exist_ok=True)
    Path("test_results").mkdir(exist_ok=True)
    
    # Run the complete test suite WITH REPORTING
    print("\n" + "=" * 70)
    print("🚀 RUNNING COMPLETE TEST SUITE WITH REPORTING")
    print("=" * 70)
    
    # 1. Statistical Verification
    print("\n" + "="*60)
    print("1. STATISTICAL IDENTITY VERIFICATION")
    print("="*60)
    result = subprocess.run(
        [sys.executable, 'scripts/run_statistical_verification.py'],
        capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': os.getcwd()}
    )
    print(result.stdout)
    
    # 2. LLM Verification with open models
    print("\n" + "="*60)
    print("2. LLM VERIFICATION (GPT-2 vs DistilGPT-2)")
    print("="*60)
    result = subprocess.run(
        [sys.executable, 'scripts/test_llm_verification.py'],
        capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': os.getcwd()}
    )
    print(result.stdout)
    
    # 3. Fuzzy Hash Verification
    print("\n" + "="*60)
    print("3. FUZZY HASH VERIFICATION")
    print("="*60)
    result = subprocess.run(
        [sys.executable, 'scripts/run_fuzzy_verification.py'],
        capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': os.getcwd()}
    )
    print(result.stdout)
    
    # 4. Provenance Auditing
    print("\n" + "="*60)
    print("4. PROVENANCE AUDITING")
    print("="*60)
    result = subprocess.run(
        [sys.executable, 'scripts/run_provenance_verification.py'],
        capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': os.getcwd()}
    )
    print(result.stdout)
    
    # 5. GENERATE COMPREHENSIVE REPORT
    print("\n" + "="*60)
    print("5. GENERATING COMPREHENSIVE REPORT")
    print("="*60)
    result = subprocess.run(
        [sys.executable, 'scripts/experimental_report_clean.py'],
        capture_output=True, text=True,
        env={**os.environ, 'PYTHONPATH': os.getcwd()}
    )
    print(result.stdout)
    
    # 6. GENERATE VALIDATION REPORT
    print("\n" + "="*60)
    print("6. GENERATING VALIDATION REPORT")
    print("="*60)
    if os.path.exists('scripts/generate_validation_report.py'):
        result = subprocess.run(
            [sys.executable, 'scripts/generate_validation_report.py'],
            capture_output=True, text=True,
            env={**os.environ, 'PYTHONPATH': os.getcwd()}
        )
        print(result.stdout)
    
    # 7. VALIDATE RESULTS
    print("\n" + "="*60)
    print("7. VALIDATING RESULTS")
    print("="*60)
    if os.path.exists('scripts/validate_results.py'):
        result = subprocess.run(
            [sys.executable, 'scripts/validate_results.py'],
            capture_output=True, text=True,
            env={**os.environ, 'PYTHONPATH': os.getcwd()}
        )
        print(result.stdout)
    
    # 8. DISPLAY ALL GENERATED REPORTS
    print("\n" + "=" * 70)
    print("📊 GENERATED REPORTS AND RESULTS")
    print("=" * 70)
    
    # Show JSON results
    exp_dir = Path("experimental_results")
    if exp_dir.exists():
        print("\n📁 Experimental Results (JSON):")
        for jf in sorted(exp_dir.glob("*.json"), key=os.path.getmtime, reverse=True)[:10]:
            size_kb = jf.stat().st_size / 1024
            print(f"  • {jf.name} ({size_kb:.1f} KB)")
            
            # Show key metrics from each file
            try:
                with open(jf) as f:
                    data = json.load(f)
                    
                if 'statistical' in jf.name:
                    if isinstance(data, dict):
                        for key in list(data.keys())[:2]:
                            if 'decision' in str(data[key]):
                                print(f"    → {key}: {data[key].get('decision', 'N/A')}")
                
                elif 'summary' in jf.name or 'report' in jf.name:
                    if 'verification_summary' in data:
                        summary = data['verification_summary']
                        print(f"    → Overall: {summary.get('overall_status', 'N/A')}")
                        if 'statistical' in summary:
                            print(f"    → Statistical: {summary['statistical'].get('status', 'N/A')}")
                        if 'fuzzy_hash' in summary:
                            print(f"    → Fuzzy: {summary['fuzzy_hash'].get('status', 'N/A')}")
                        if 'provenance' in summary:
                            print(f"    → Provenance: {summary['provenance'].get('status', 'N/A')}")
            except:
                pass
    
    # Show text reports
    test_dir = Path("test_results")
    if test_dir.exists():
        print("\n📄 Text Reports:")
        for tf in sorted(test_dir.glob("*.txt"), key=os.path.getmtime, reverse=True)[:5]:
            size_kb = tf.stat().st_size / 1024
            print(f"  • {tf.name} ({size_kb:.1f} KB)")
    
    # Show markdown reports
    if test_dir.exists():
        print("\n📝 Markdown Reports:")
        for mf in sorted(test_dir.glob("*.md"), key=os.path.getmtime, reverse=True)[:5]:
            size_kb = mf.stat().st_size / 1024
            print(f"  • {mf.name} ({size_kb:.1f} KB)")
    
    # 9. FINAL SUMMARY WITH METRICS
    print("\n" + "=" * 70)
    print("📈 VERIFICATION METRICS SUMMARY")
    print("=" * 70)
    
    # Try to load and display the latest comprehensive report
    report_files = list(exp_dir.glob("*report*.json")) if exp_dir.exists() else []
    if report_files:
        latest_report = max(report_files, key=os.path.getmtime)
        print(f"\nLatest Report: {latest_report.name}")
        
        with open(latest_report) as f:
            report_data = json.load(f)
        
        # Display key metrics
        if 'statistical' in report_data:
            stat = report_data['statistical']
            print("\nStatistical Identity:")
            print(f"  Decision: {stat.get('decision', 'N/A')}")
            print(f"  Samples: {stat.get('n_used', 'N/A')}")
            print(f"  Mean: {stat.get('mean', 'N/A')}")
            print(f"  99% CI: {stat.get('ci_99', 'N/A')}")
        
        if 'fuzzy' in report_data:
            fuzzy = report_data['fuzzy']
            print("\nFuzzy Hash:")
            print(f"  Algorithm: {fuzzy.get('algorithm', 'N/A')}")
            print(f"  Pass Rate: {fuzzy.get('pass_rate', 'N/A')}")
            print(f"  Mean Similarity: {fuzzy.get('mean_similarity', 'N/A')}")
        
        if 'provenance' in report_data:
            prov = report_data['provenance']
            print("\nProvenance:")
            print(f"  Merkle Root: {str(prov.get('signed_merkle_root', 'N/A'))[:32]}...")
            print(f"  Verified: {prov.get('verified_inclusion_proof', {}).get('verified', 'N/A')}")
            print(f"  Checks Passed: {len(prov.get('checks_passed', []))}")
    
    print("\n" + "=" * 70)
    print("✅ FULL PIPELINE COMPLETE WITH REPORTING")
    print("=" * 70)
    print("\n📝 Evidence for your paper:")
    print("  • experimental_results/*.json - Detailed metrics")
    print("  • test_results/*.txt - Text reports")
    print("  • test_results/*.md - Markdown reports")
    print("  • All verification metrics with confidence intervals")
    print("  • Complete provenance audit trails")
    print("\nAll tests use open models (GPT-2, DistilGPT-2)")
    print("No authentication tokens required!")

if __name__ == "__main__":
    main()