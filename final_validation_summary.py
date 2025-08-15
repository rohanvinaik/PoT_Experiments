#!/usr/bin/env python3
"""
Final Validation Summary for PoT Experiments
Tests all major components and generates a comprehensive report.
"""

import sys
import os
import time
import json
import subprocess
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run_component_test(component_name, test_file):
    """Run a component test and return results"""
    print(f"\n{'='*60}")
    print(f"Testing: {component_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, test_file], 
            capture_output=True, 
            text=True, 
            timeout=60
        )
        duration = time.time() - start_time
        
        success = result.returncode == 0
        
        return {
            'component': component_name,
            'success': success,
            'duration': duration,
            'stdout_lines': len(result.stdout.split('\n')),
            'stderr_lines': len(result.stderr.split('\n')),
            'returncode': result.returncode
        }
        
    except subprocess.TimeoutExpired:
        return {
            'component': component_name,
            'success': False,
            'duration': 60.0,
            'error': 'timeout',
            'returncode': -1
        }
    except Exception as e:
        return {
            'component': component_name,
            'success': False,
            'duration': 0.0,
            'error': str(e),
            'returncode': -1
        }

def run_integration_test():
    """Test the integrated ProofOfTraining system"""
    print(f"\n{'='*60}")
    print("Testing: Integrated ProofOfTraining System")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run(
            [sys.executable, 'pot/security/proof_of_training.py'], 
            capture_output=True, 
            text=True, 
            timeout=30
        )
        duration = time.time() - start_time
        
        success = result.returncode == 0
        
        return {
            'component': 'Integrated System',
            'success': success,
            'duration': duration,
            'stdout_lines': len(result.stdout.split('\n')),
            'stderr_lines': len(result.stderr.split('\n')),
            'returncode': result.returncode
        }
        
    except Exception as e:
        return {
            'component': 'Integrated System',
            'success': False,
            'duration': 0.0,
            'error': str(e),
            'returncode': -1
        }

def check_dependencies():
    """Check system dependencies"""
    print("Checking system dependencies...")
    
    deps = {
        'Python': True,
        'NumPy': False,
        'PyTorch': False,
        'Transformers': False,
        'SSDeep': False,
        'TLSH': False
    }
    
    try:
        import numpy
        deps['NumPy'] = True
    except ImportError:
        pass
    
    try:
        import torch
        deps['PyTorch'] = True
    except ImportError:
        pass
        
    try:
        import transformers
        deps['Transformers'] = True
    except ImportError:
        pass
        
    try:
        import ssdeep
        deps['SSDeep'] = True
    except ImportError:
        pass
        
    try:
        import tlsh
        deps['TLSH'] = True
    except ImportError:
        pass
    
    return deps

def main():
    print("="*80)
    print("PROOF-OF-TRAINING FINAL VALIDATION SUMMARY")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check dependencies
    dependencies = check_dependencies()
    print("\nDependency Status:")
    for dep, available in dependencies.items():
        status = "✓" if available else "✗"
        print(f"  {status} {dep}: {'Available' if available else 'Not found'}")
    
    # Component tests to run
    components = [
        ('FuzzyHashVerifier', 'pot/security/test_fuzzy_verifier.py'),
        ('TokenSpaceNormalizer', 'pot/security/test_token_normalizer.py'),
        ('TrainingProvenanceAuditor', 'pot/security/test_provenance_auditor.py')
    ]
    
    # Run component tests
    print(f"\n{'='*80}")
    print("COMPONENT TEST RESULTS")
    print(f"{'='*80}")
    
    results = []
    for component_name, test_file in components:
        if os.path.exists(test_file):
            result = run_component_test(component_name, test_file)
            results.append(result)
            
            status = "✓ PASSED" if result['success'] else "✗ FAILED"
            print(f"{status} {component_name} ({result['duration']:.2f}s)")
        else:
            print(f"✗ SKIPPED {component_name} (test file not found)")
    
    # Run integration test
    integration_result = run_integration_test()
    results.append(integration_result)
    
    status = "✓ PASSED" if integration_result['success'] else "✗ FAILED"
    print(f"{status} Integrated System ({integration_result['duration']:.2f}s)")
    
    # Generate summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    failed_tests = total_tests - passed_tests
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    # System status
    if success_rate == 100:
        print("\n🎉 ALL TESTS PASSED - PoT System is fully operational!")
        print("✅ System ready for production deployment")
    elif success_rate >= 75:
        print(f"\n⚠️  {passed_tests}/{total_tests} tests passed - System mostly functional")
        print("🔧 Some components may need attention")
    else:
        print(f"\n❌ Only {passed_tests}/{total_tests} tests passed - System needs repair")
        print("🚨 Review failed components before deployment")
    
    # Save detailed results
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'dependencies': dependencies,
        'test_results': results,
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': success_rate
        }
    }
    
    with open(f'pot_final_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n📄 Detailed results saved to pot_final_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Create reproducibility artifacts
    artifacts = {
        'git_commit': None,
        'python_version': sys.version,
        'environment': dict(os.environ),
        'test_results': summary_data
    }
    
    try:
        git_result = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
        if git_result.returncode == 0:
            artifacts['git_commit'] = git_result.stdout.strip()
    except:
        pass
    
    with open(f'reproducibility_artifacts_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(artifacts, f, indent=2)
    
    print(f"🔬 Reproducibility artifacts saved")
    
    return 0 if success_rate == 100 else 1

if __name__ == "__main__":
    sys.exit(main())