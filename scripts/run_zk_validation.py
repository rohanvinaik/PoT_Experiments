#!/usr/bin/env python3
"""
ZK System Validation Script

Comprehensive validation of the Zero-Knowledge proof system for external verification.
Generates structured JSON reports for integration with the main pipeline.
"""

import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any, List
import traceback
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import ZK components
try:
    from pot.zk import (
        prove_sgd_step, verify_sgd_step,
        prove_lora_step, verify_lora_step,
        SGDStepStatement, LoRAStepStatement,
        get_zk_metrics_collector
    )
    from pot.zk.diagnostic import ZKDiagnostic
    from pot.zk.version_info import get_system_version
    ZK_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  ZK modules not available: {e}")
    ZK_AVAILABLE = False


class ZKValidator:
    """Validates ZK proof system functionality"""
    
    def __init__(self):
        self.results = {
            'timestamp': time.time(),
            'zk_available': ZK_AVAILABLE,
            'tests': [],
            'metrics': {},
            'summary': {}
        }
        self.metrics_collector = get_zk_metrics_collector() if ZK_AVAILABLE else None
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run comprehensive ZK validation tests"""
        print("🔍 ZK System Validation")
        print("=" * 50)
        
        if not ZK_AVAILABLE:
            self.results['summary'] = {
                'status': 'skipped',
                'reason': 'ZK modules not available',
                'tests_passed': 0,
                'tests_failed': 0,
                'tests_total': 0
            }
            return self.results
        
        # Run test suites
        self.test_basic_functionality()
        self.test_sgd_proving()
        self.test_lora_proving()
        self.test_performance_benchmarks()
        self.test_system_health()
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    def test_basic_functionality(self):
        """Test basic ZK system functionality"""
        test_name = "basic_functionality"
        print(f"\n📋 Testing {test_name}...")
        
        try:
            # Test imports
            assert SGDStepStatement is not None
            assert LoRAStepStatement is not None
            
            # Test statement creation
            sgd_statement = SGDStepStatement(
                model_id="test_model",
                step_number=1,
                epoch=0,
                weights_before_hash="0x" + "0" * 64,
                weights_after_hash="0x" + "1" * 64,
                batch_hash="0x" + "2" * 64,
                learning_rate=0.001,
                batch_size=32
            )
            
            self.record_test_result(test_name, True, "Basic functionality works")
            print("  ✅ Basic functionality test passed")
            
        except Exception as e:
            self.record_test_result(test_name, False, str(e))
            print(f"  ❌ Basic functionality test failed: {e}")
    
    def test_sgd_proving(self):
        """Test SGD proof generation and verification"""
        test_name = "sgd_proving"
        print(f"\n📋 Testing {test_name}...")
        
        try:
            # Create mock data
            weights_before = np.random.randn(100).astype(np.float32)
            weights_after = weights_before - 0.001 * np.random.randn(100).astype(np.float32)
            batch = {
                'inputs': np.random.randn(32, 10).astype(np.float32),
                'targets': np.random.randint(0, 10, 32)
            }
            gradients = np.random.randn(100).astype(np.float32)
            
            # Create statement
            statement = SGDStepStatement(
                model_id="test_sgd",
                step_number=1,
                epoch=0,
                weights_before_hash=self.compute_hash(weights_before),
                weights_after_hash=self.compute_hash(weights_after),
                batch_hash=self.compute_hash(batch['inputs']),
                learning_rate=0.001,
                batch_size=32
            )
            
            # Generate proof
            start_time = time.time()
            proof = prove_sgd_step(
                statement=statement,
                weights_before=weights_before,
                weights_after=weights_after,
                batch=batch,
                gradients=gradients,
                learning_rate=0.001
            )
            proof_time = (time.time() - start_time) * 1000
            
            # Verify proof
            is_valid = verify_sgd_step(statement, proof)
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_proof_generation(
                    proof_type='sgd',
                    duration=proof_time,
                    proof_size=len(str(proof)),
                    success=True
                )
            
            self.record_test_result(
                test_name, 
                is_valid, 
                f"SGD proof {'valid' if is_valid else 'invalid'}, time: {proof_time:.1f}ms"
            )
            
            print(f"  {'✅' if is_valid else '❌'} SGD proving test {'passed' if is_valid else 'failed'}")
            print(f"    Proof generation time: {proof_time:.1f}ms")
            
        except Exception as e:
            self.record_test_result(test_name, False, str(e))
            print(f"  ❌ SGD proving test failed: {e}")
    
    def test_lora_proving(self):
        """Test LoRA proof generation and verification"""
        test_name = "lora_proving"
        print(f"\n📋 Testing {test_name}...")
        
        try:
            # Create mock LoRA data
            base_weights = np.random.randn(1000).astype(np.float32)
            adapter_a = np.random.randn(100, 16).astype(np.float32) * 0.01
            adapter_b = np.random.randn(16, 100).astype(np.float32) * 0.01
            
            # Create statement
            statement = LoRAStepStatement(
                model_id="test_lora",
                step_number=1,
                epoch=0,
                base_weights_hash=self.compute_hash(base_weights),
                adapter_a_hash=self.compute_hash(adapter_a),
                adapter_b_hash=self.compute_hash(adapter_b),
                rank=16,
                scale_factor=32.0
            )
            
            # Generate proof
            start_time = time.time()
            proof = prove_lora_step(
                statement=statement,
                base_weights=base_weights,
                adapter_a=adapter_a,
                adapter_b=adapter_b
            )
            proof_time = (time.time() - start_time) * 1000
            
            # Verify proof
            is_valid = verify_lora_step(statement, proof)
            
            # Record metrics
            if self.metrics_collector:
                self.metrics_collector.record_proof_generation(
                    proof_type='lora',
                    duration=proof_time,
                    proof_size=len(str(proof)),
                    success=True
                )
            
            self.record_test_result(
                test_name,
                is_valid,
                f"LoRA proof {'valid' if is_valid else 'invalid'}, time: {proof_time:.1f}ms"
            )
            
            print(f"  {'✅' if is_valid else '❌'} LoRA proving test {'passed' if is_valid else 'failed'}")
            print(f"    Proof generation time: {proof_time:.1f}ms")
            
        except Exception as e:
            self.record_test_result(test_name, False, str(e))
            print(f"  ❌ LoRA proving test failed: {e}")
    
    def test_performance_benchmarks(self):
        """Test performance metrics and benchmarks"""
        test_name = "performance_benchmarks"
        print(f"\n📋 Testing {test_name}...")
        
        try:
            # Run multiple proof generations
            sgd_times = []
            lora_times = []
            
            print("  Running performance benchmarks...")
            
            for i in range(3):
                # SGD benchmark
                weights = np.random.randn(100).astype(np.float32)
                start = time.time()
                try:
                    _ = prove_sgd_step(
                        SGDStepStatement(
                            model_id=f"bench_{i}",
                            step_number=i,
                            epoch=0,
                            weights_before_hash=self.compute_hash(weights),
                            weights_after_hash=self.compute_hash(weights),
                            batch_hash="0x" + "0" * 64,
                            learning_rate=0.001,
                            batch_size=32
                        ),
                        weights, weights,
                        {'inputs': np.zeros((32, 10)), 'targets': np.zeros(32, dtype=int)},
                        weights, 0.001
                    )
                    sgd_times.append((time.time() - start) * 1000)
                except:
                    pass
                
                # LoRA benchmark
                base = np.random.randn(100).astype(np.float32)
                a = np.random.randn(10, 4).astype(np.float32)
                b = np.random.randn(4, 10).astype(np.float32)
                start = time.time()
                try:
                    _ = prove_lora_step(
                        LoRAStepStatement(
                            model_id=f"bench_{i}",
                            step_number=i,
                            epoch=0,
                            base_weights_hash=self.compute_hash(base),
                            adapter_a_hash=self.compute_hash(a),
                            adapter_b_hash=self.compute_hash(b),
                            rank=4,
                            scale_factor=16.0
                        ),
                        base, a, b
                    )
                    lora_times.append((time.time() - start) * 1000)
                except:
                    pass
            
            # Calculate metrics
            if sgd_times and lora_times:
                avg_sgd = np.mean(sgd_times)
                avg_lora = np.mean(lora_times)
                speedup = avg_sgd / avg_lora if avg_lora > 0 else 0
                
                self.results['metrics']['performance'] = {
                    'sgd_avg_ms': avg_sgd,
                    'lora_avg_ms': avg_lora,
                    'speedup': speedup,
                    'samples': len(sgd_times)
                }
                
                success = speedup > 1.5  # LoRA should be at least 1.5x faster
                self.record_test_result(
                    test_name,
                    success,
                    f"SGD: {avg_sgd:.1f}ms, LoRA: {avg_lora:.1f}ms, Speedup: {speedup:.2f}x"
                )
                
                print(f"  {'✅' if success else '⚠️'} Performance benchmark completed")
                print(f"    Average SGD time: {avg_sgd:.1f}ms")
                print(f"    Average LoRA time: {avg_lora:.1f}ms")
                print(f"    LoRA speedup: {speedup:.2f}x")
            else:
                self.record_test_result(test_name, False, "No timing data collected")
                print("  ⚠️  Performance benchmark incomplete")
            
        except Exception as e:
            self.record_test_result(test_name, False, str(e))
            print(f"  ❌ Performance benchmark failed: {e}")
    
    def test_system_health(self):
        """Test ZK system health and diagnostics"""
        test_name = "system_health"
        print(f"\n📋 Testing {test_name}...")
        
        try:
            # Run diagnostics
            diagnostic = ZKDiagnostic()
            health_report = diagnostic.diagnose_zk_system()
            
            health_score = health_report.get('health_score', 0)
            overall_health = health_report.get('overall_health', 'unknown')
            
            self.results['metrics']['health'] = {
                'score': health_score,
                'status': overall_health,
                'checks_passed': len([t for t in health_report.get('test_results', []) if t.get('status') == 'pass']),
                'checks_total': len(health_report.get('test_results', []))
            }
            
            success = health_score >= 70  # 70+ is considered healthy
            self.record_test_result(
                test_name,
                success,
                f"Health score: {health_score}/100, Status: {overall_health}"
            )
            
            print(f"  {'✅' if success else '⚠️'} System health check completed")
            print(f"    Health score: {health_score}/100")
            print(f"    Overall status: {overall_health}")
            
            # Get version info
            try:
                version_info = get_system_version()
                self.results['metrics']['version'] = {
                    'system_version': version_info.system_version,
                    'python_version': version_info.python_version,
                    'binaries_found': len(version_info.binaries)
                }
            except:
                pass
            
        except Exception as e:
            self.record_test_result(test_name, False, str(e))
            print(f"  ❌ System health check failed: {e}")
    
    def compute_hash(self, data) -> str:
        """Compute hash of data for statements"""
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        elif not isinstance(data, bytes):
            data = str(data).encode()
        return f"0x{hash(data) & ((1 << 256) - 1):064x}"
    
    def record_test_result(self, name: str, success: bool, details: str):
        """Record test result"""
        self.results['tests'].append({
            'name': name,
            'success': success,
            'details': details,
            'timestamp': time.time()
        })
    
    def generate_summary(self):
        """Generate summary of all tests"""
        tests_passed = sum(1 for t in self.results['tests'] if t['success'])
        tests_failed = sum(1 for t in self.results['tests'] if not t['success'])
        tests_total = len(self.results['tests'])
        
        # Get metrics summary
        if self.metrics_collector:
            metrics_report = self.metrics_collector.generate_report()
            self.results['metrics']['collector'] = metrics_report
        
        self.results['summary'] = {
            'status': 'passed' if tests_failed == 0 else 'failed',
            'tests_passed': tests_passed,
            'tests_failed': tests_failed,
            'tests_total': tests_total,
            'success_rate': (tests_passed / tests_total * 100) if tests_total > 0 else 0,
            'zk_functional': ZK_AVAILABLE and tests_passed > 0,
            'performance_validated': 'performance' in self.results['metrics'],
            'health_check_passed': self.results['metrics'].get('health', {}).get('score', 0) >= 70
        }
        
        print(f"\n{'='*50}")
        print("📊 ZK Validation Summary")
        print(f"{'='*50}")
        print(f"Tests Passed: {tests_passed}/{tests_total}")
        print(f"Success Rate: {self.results['summary']['success_rate']:.1f}%")
        print(f"ZK System: {'Functional' if self.results['summary']['zk_functional'] else 'Not Functional'}")
        
        if 'performance' in self.results['metrics']:
            perf = self.results['metrics']['performance']
            print(f"Performance: LoRA {perf['speedup']:.2f}x faster than SGD")
        
        if 'health' in self.results['metrics']:
            health = self.results['metrics']['health']
            print(f"System Health: {health['score']}/100 ({health['status']})")


def main():
    """Main validation entry point"""
    # Get output file from environment or use default
    output_file = os.getenv('ZK_VALIDATION_OUTPUT', 
                           f'experimental_results/zk_validation_{int(time.time())}.json')
    
    print("🚀 Starting ZK System Validation")
    print(f"Output will be saved to: {output_file}")
    
    # Run validation
    validator = ZKValidator()
    results = validator.run_all_tests()
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Return exit code based on success
    if results['summary'].get('status') == 'passed':
        print("\n✅ ZK validation PASSED")
        return 0
    else:
        print("\n❌ ZK validation FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())