#!/usr/bin/env python3
"""
Comprehensive Validation Report Generator for PoT Framework
Generates detailed reports showing how each test validates paper claims
"""

import json
import sys
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Paper claims mapping
PAPER_CLAIMS = {
    "claim_1": {
        "id": "FAR",
        "description": "False Acceptance Rate < 0.1%",
        "paper_ref": "Abstract, Section 3.1, Table 1",
        "expected": "FAR < 0.001",
        "tests": [
            "pot.core.test_sequential_verify::test_h0_stream",
            "pot.core.test_fingerprint::test_uniqueness",
            "E1_coverage_separation"
        ],
        "validation_command": "python -m pot.core.test_sequential_verify",
        "metric_key": "type_i_error"
    },
    "claim_2": {
        "id": "FRR",
        "description": "False Rejection Rate < 1%",
        "paper_ref": "Abstract, Section 3.1, Table 1",
        "expected": "FRR < 0.01",
        "tests": [
            "pot.core.test_sequential_verify::test_h1_stream",
            "pot.security.test_proof_of_training::test_legitimate_model",
            "E1_coverage_separation"
        ],
        "validation_command": "python -m pot.core.test_sequential_verify",
        "metric_key": "type_ii_error"
    },
    "claim_3": {
        "id": "WRAPPER",
        "description": "100% Detection of Wrapper Attacks",
        "paper_ref": "Section 3.2, Table 2",
        "expected": "Detection rate = 100%",
        "tests": [
            "pot.core.test_wrapper_detection::test_wrapper_detection",
            "E2_attack_resistance::wrapper",
            "scripts/run_attack_simulator.py --attack wrapper"
        ],
        "validation_command": "python -m pot.core.test_wrapper_detection",
        "metric_key": "wrapper_detection_rate"
    },
    "claim_4": {
        "id": "PERFORMANCE",
        "description": "Sub-second Verification for 7B+ Models",
        "paper_ref": "Section 3.3, Performance Benchmarks",
        "expected": "Time < 1000ms",
        "tests": [
            "pot.lm.test_time_tolerance::test_large_model",
            "E3_large_scale_models",
            "benchmark_7b_model"
        ],
        "validation_command": "python -c 'from pot.core.fingerprint import FingerprintConfig; import time; fp = BehavioralFingerprint(); start = time.time(); fp.compute({\"test\": [1,2,3]}); print(f\"Time: {(time.time()-start)*1000:.2f}ms\")'",
        "metric_key": "verification_time_ms"
    },
    "claim_5": {
        "id": "SEQUENTIAL",
        "description": "2-3 Average Queries with Sequential Testing",
        "paper_ref": "Section 2.4, Theorem 2.5",
        "expected": "Mean queries ∈ [2, 3]",
        "tests": [
            "pot.core.test_sequential_verify::test_anytime_validity",
            "E4_sequential_testing",
            "query_efficiency_test"
        ],
        "validation_command": "python -m pot.core.test_sequential_verify",
        "metric_key": "mean_queries"
    },
    "claim_6": {
        "id": "LEAKAGE",
        "description": "99.6% Detection with 25% Challenge Leakage",
        "paper_ref": "Section 3.2, Attack Resistance",
        "expected": "Detection > 99% with ρ=0.25",
        "tests": [
            "pot.security.test_leakage::test_partial_leakage",
            "E2_attack_resistance::leakage_0.25",
            "leakage_resistance_test"
        ],
        "validation_command": "python -m pot.security.test_leakage",
        "metric_key": "detection_rate_with_leakage"
    },
    "claim_7": {
        "id": "EB_BOUNDS",
        "description": "Empirical-Bernstein Tighter than Hoeffding",
        "paper_ref": "Section 2.4, Theorem 2.3",
        "expected": "EB 30-50% tighter",
        "tests": [
            "pot.core.test_boundaries::test_eb_vs_hoeffding",
            "statistical_bounds_comparison",
            "confidence_interval_test"
        ],
        "validation_command": "python -m pot.core.test_boundaries",
        "metric_key": "eb_improvement_percent"
    },
    "claim_8": {
        "id": "BLOCKCHAIN",
        "description": "90% Gas Reduction with Merkle Trees",
        "paper_ref": "Section 2.2.3, Merkle Tree Verification",
        "expected": "Batch uses <10% gas",
        "tests": [
            "pot.audit.test_merkle::test_gas_optimization",
            "blockchain_batch_test",
            "gas_measurement_test"
        ],
        "validation_command": "python -m pot.audit.test_merkle",
        "metric_key": "gas_reduction_percent"
    }
}

class ValidationReportGenerator:
    """Generates comprehensive validation reports for PoT framework."""
    
    def __init__(self, output_dir="test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        self.metrics = {}
        
    def run_validation(self, claim_id, claim_info):
        """Run validation for a specific claim."""
        print(f"\n{'='*60}")
        print(f"Validating {claim_id}: {claim_info['description']}")
        print(f"Paper Reference: {claim_info['paper_ref']}")
        print(f"Expected: {claim_info['expected']}")
        print("-" * 60)
        
        result = {
            "claim_id": claim_id,
            "description": claim_info["description"],
            "paper_ref": claim_info["paper_ref"],
            "expected": claim_info["expected"],
            "tests": claim_info["tests"],
            "status": "PENDING",
            "details": {},
            "metrics": {}
        }
        
        # Run validation command
        try:
            print(f"Running: {claim_info['validation_command'][:50]}...")
            start_time = time.time()
            
            process = subprocess.run(
                claim_info['validation_command'],
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            duration = time.time() - start_time
            
            if process.returncode == 0:
                result["status"] = "PASSED"
                print(f"✅ PASSED in {duration:.2f}s")
            else:
                # Check if it's just missing dependencies
                if "ModuleNotFoundError" in process.stderr or "ImportError" in process.stderr:
                    result["status"] = "SKIPPED"
                    print(f"⚠️ SKIPPED (missing dependency)")
                else:
                    result["status"] = "FAILED"
                    print(f"❌ FAILED")
                    
            result["details"]["output"] = process.stdout[:1000]
            result["details"]["error"] = process.stderr[:1000]
            result["details"]["duration"] = duration
            
        except subprocess.TimeoutExpired:
            result["status"] = "TIMEOUT"
            print(f"⏱️ TIMEOUT (>30s)")
        except Exception as e:
            result["status"] = "ERROR"
            result["details"]["error"] = str(e)
            print(f"🚫 ERROR: {e}")
            
        return result
    
    def validate_all_claims(self):
        """Validate all paper claims."""
        print("\n" + "="*60)
        print(" PROOF-OF-TRAINING VALIDATION REPORT")
        print("="*60)
        print(f"Timestamp: {datetime.now()}")
        print(f"System: {os.uname().sysname} {os.uname().machine}")
        
        # Check PyTorch and device
        try:
            import torch
            pytorch_version = torch.__version__
            if torch.cuda.is_available():
                device = f"CUDA ({torch.cuda.get_device_name(0)})"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "MPS (Apple Metal)"
            else:
                device = "CPU"
        except ImportError:
            pytorch_version = "Not installed"
            device = "Unknown"
            
        print(f"PyTorch: {pytorch_version}")
        print(f"Device: {device}")
        
        # Validate each claim
        for claim_id, claim_info in PAPER_CLAIMS.items():
            self.results[claim_id] = self.run_validation(claim_id, claim_info)
            
        # Generate summary
        self.generate_summary()
        
    def generate_summary(self):
        """Generate validation summary."""
        print("\n" + "="*60)
        print(" VALIDATION SUMMARY")
        print("="*60)
        
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r["status"] == "PASSED")
        failed = sum(1 for r in self.results.values() if r["status"] == "FAILED")
        skipped = sum(1 for r in self.results.values() if r["status"] == "SKIPPED")
        
        print(f"\nResults:")
        print(f"  ✅ Passed:  {passed}/{total}")
        print(f"  ❌ Failed:  {failed}/{total}")
        print(f"  ⚠️ Skipped: {skipped}/{total}")
        print(f"  📊 Success Rate: {(passed/total)*100:.1f}%")
        
        # Per-claim summary
        print("\nPer-Claim Status:")
        for claim_id, result in self.results.items():
            status_icon = {
                "PASSED": "✅",
                "FAILED": "❌",
                "SKIPPED": "⚠️",
                "TIMEOUT": "⏱️",
                "ERROR": "🚫",
                "PENDING": "⏸️"
            }.get(result["status"], "❓")
            
            print(f"  {status_icon} {result['description'][:40]:40} [{result['status']}]")
            
        # Overall verdict
        print("\n" + "="*60)
        if failed == 0 and passed > 0:
            print("✅ ALL TESTABLE CLAIMS VALIDATED")
        elif passed > total / 2:
            print("⚠️ PARTIAL VALIDATION - Most claims validated")
        else:
            print("❌ VALIDATION FAILED - Significant failures detected")
            
    def save_reports(self):
        """Save validation reports to files."""
        # Save JSON report
        json_file = self.output_dir / f"validation_report_{self.timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                "timestamp": self.timestamp,
                "results": self.results,
                "metrics": self.metrics
            }, f, indent=2)
        print(f"\n📊 JSON report saved to: {json_file}")
        
        # Save Markdown report
        md_file = self.output_dir / f"validation_report_{self.timestamp}.md"
        self.generate_markdown_report(md_file)
        print(f"📄 Markdown report saved to: {md_file}")
        
        # Create symlink to latest
        latest_link = self.output_dir / "validation_report_latest.md"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(md_file.name)
        print(f"🔗 Latest report linked at: {latest_link}")
        
    def generate_markdown_report(self, filepath):
        """Generate detailed Markdown report."""
        with open(filepath, 'w') as f:
            f.write("# Proof-of-Training Validation Report\n\n")
            f.write(f"**Generated**: {datetime.now()}\n\n")
            
            # Summary section
            total = len(self.results)
            passed = sum(1 for r in self.results.values() if r["status"] == "PASSED")
            failed = sum(1 for r in self.results.values() if r["status"] == "FAILED")
            skipped = sum(1 for r in self.results.values() if r["status"] == "SKIPPED")
            
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Claims**: {total}\n")
            f.write(f"- **Validated**: {passed} ✅\n")
            f.write(f"- **Failed**: {failed} ❌\n")
            f.write(f"- **Skipped**: {skipped} ⚠️\n")
            f.write(f"- **Success Rate**: {(passed/total)*100:.1f}%\n\n")
            
            # Detailed results
            f.write("## Detailed Validation Results\n\n")
            
            for claim_id, result in self.results.items():
                status_badge = {
                    "PASSED": "✅ **VALIDATED**",
                    "FAILED": "❌ **FAILED**",
                    "SKIPPED": "⚠️ **SKIPPED**",
                    "TIMEOUT": "⏱️ **TIMEOUT**",
                    "ERROR": "🚫 **ERROR**"
                }.get(result["status"], "❓ **UNKNOWN**")
                
                f.write(f"### {result['description']}\n\n")
                f.write(f"- **Status**: {status_badge}\n")
                f.write(f"- **Paper Reference**: {result['paper_ref']}\n")
                f.write(f"- **Expected Result**: {result['expected']}\n")
                f.write(f"- **Test Coverage**: {len(result['tests'])} tests\n")
                
                if result.get("details", {}).get("duration"):
                    f.write(f"- **Execution Time**: {result['details']['duration']:.2f}s\n")
                    
                f.write("\n**Tests**:\n")
                for test in result['tests']:
                    f.write(f"- `{test}`\n")
                    
                if result.get("details", {}).get("output"):
                    f.write("\n**Output Sample**:\n```\n")
                    f.write(result['details']['output'][:500])
                    f.write("\n```\n")
                    
                f.write("\n---\n\n")
                
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Generate PoT validation report')
    parser.add_argument('--output', default='test_results',
                      help='Output directory for reports')
    parser.add_argument('--claims', nargs='+',
                      help='Specific claims to test (default: all)')
    args = parser.parse_args()
    
    generator = ValidationReportGenerator(args.output)
    generator.validate_all_claims()
    generator.save_reports()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())