#!/usr/bin/env python3
"""
ZK System Diagnostic Module

Comprehensive diagnostic tools for zero-knowledge proof system components,
binary verification, performance analysis, and health monitoring.
"""

import os
import sys
import json
import time
import hashlib
import subprocess
import tempfile
import platform
import shutil
import psutil
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import numpy as np


@dataclass
class DiagnosticResult:
    """Result of a diagnostic test"""
    test_name: str
    status: str  # 'pass', 'fail', 'warning', 'skip'
    message: str
    details: Dict[str, Any]
    duration_ms: float
    timestamp: str


@dataclass
class SystemInfo:
    """System information for diagnostics"""
    platform: str
    architecture: str
    python_version: str
    cpu_count: int
    memory_total_gb: float
    disk_free_gb: float
    rust_version: Optional[str]
    cargo_version: Optional[str]


class ZKDiagnostic:
    """Comprehensive ZK system diagnostic suite"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: List[DiagnosticResult] = []
        self.start_time = time.time()
        
        # Paths
        self.project_root = Path(__file__).parent.parent.parent
        self.prover_dir_debug = self.project_root / "pot/zk/prover_halo2/target/debug"
        self.prover_dir_release = self.project_root / "pot/zk/prover_halo2/target/release"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="zk_diagnostic_"))
        
        # Binary paths
        self.binaries = {
            'sgd_prover_debug': self.prover_dir_debug / "prove_sgd_stdin",
            'sgd_verifier_debug': self.prover_dir_debug / "verify_sgd_stdin",
            'lora_prover_debug': self.prover_dir_debug / "prove_lora_stdin",
            'lora_verifier_debug': self.prover_dir_debug / "verify_lora_stdin",
            'sgd_prover_release': self.prover_dir_release / "prove_sgd_stdin",
            'sgd_verifier_release': self.prover_dir_release / "verify_sgd_stdin",
            'lora_prover_release': self.prover_dir_release / "prove_lora_stdin",
            'lora_verifier_release': self.prover_dir_release / "verify_lora_stdin",
        }
    
    def log(self, message: str):
        """Log message if verbose mode enabled"""
        if self.verbose:
            print(f"[ZK-DIAG] {message}")
    
    def run_test(self, test_name: str, test_func: callable, *args, **kwargs) -> DiagnosticResult:
        """Run a single diagnostic test with timing and error handling"""
        start_time = time.time()
        timestamp = datetime.now(timezone.utc).isoformat()
        
        self.log(f"Running test: {test_name}")
        
        try:
            result = test_func(*args, **kwargs)
            
            if isinstance(result, tuple):
                status, message, details = result
            else:
                status, message, details = 'pass', str(result), {}
                
            duration_ms = (time.time() - start_time) * 1000
            
            diagnostic_result = DiagnosticResult(
                test_name=test_name,
                status=status,
                message=message,
                details=details,
                duration_ms=duration_ms,
                timestamp=timestamp
            )
            
            self.results.append(diagnostic_result)
            return diagnostic_result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            diagnostic_result = DiagnosticResult(
                test_name=test_name,
                status='fail',
                message=f"Test failed with exception: {str(e)}",
                details={'exception': str(e), 'traceback': traceback.format_exc()},
                duration_ms=duration_ms,
                timestamp=timestamp
            )
            
            self.results.append(diagnostic_result)
            return diagnostic_result
    
    def get_system_info(self) -> SystemInfo:
        """Collect system information"""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get Rust version
        rust_version = None
        cargo_version = None
        
        try:
            result = subprocess.run(['rustc', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                rust_version = result.stdout.strip()
        except:
            pass
            
        try:
            result = subprocess.run(['cargo', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                cargo_version = result.stdout.strip()
        except:
            pass
        
        return SystemInfo(
            platform=platform.platform(),
            architecture=platform.machine(),
            python_version=sys.version.split()[0],
            cpu_count=psutil.cpu_count(),
            memory_total_gb=memory.total / (1024**3),
            disk_free_gb=disk.free / (1024**3),
            rust_version=rust_version,
            cargo_version=cargo_version
        )
    
    def check_binaries_exist(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check if ZK binaries exist"""
        results = {}
        
        for name, path in self.binaries.items():
            exists = path.exists()
            executable = exists and os.access(path, os.X_OK)
            size = path.stat().st_size if exists else 0
            
            results[name] = {
                'path': str(path),
                'exists': exists,
                'executable': executable,
                'size_bytes': size,
                'modified': path.stat().st_mtime if exists else None
            }
        
        existing_count = sum(1 for r in results.values() if r['exists'])
        executable_count = sum(1 for r in results.values() if r['executable'])
        
        if existing_count == len(self.binaries):
            if executable_count == len(self.binaries):
                status = 'pass'
                message = f"All {len(self.binaries)} binaries exist and are executable"
            else:
                status = 'warning'
                message = f"{executable_count}/{len(self.binaries)} binaries are executable"
        elif existing_count > 0:
            status = 'warning'
            message = f"{existing_count}/{len(self.binaries)} binaries exist"
        else:
            status = 'fail'
            message = "No ZK binaries found"
        
        return status, message, results
    
    def check_binaries_executable(self) -> Tuple[str, str, Dict[str, Any]]:
        """Test if binaries can be executed"""
        results = {}
        
        for name, path in self.binaries.items():
            if not path.exists():
                results[name] = {
                    'status': 'missing',
                    'message': 'Binary does not exist'
                }
                continue
            
            try:
                # Try to run with --help or similar
                result = subprocess.run([str(path)], 
                                      input=b'', 
                                      capture_output=True, 
                                      timeout=5)
                
                results[name] = {
                    'status': 'executable',
                    'return_code': result.returncode,
                    'stdout_length': len(result.stdout),
                    'stderr_length': len(result.stderr)
                }
            except subprocess.TimeoutExpired:
                results[name] = {
                    'status': 'timeout',
                    'message': 'Binary timed out (may be waiting for input)'
                }
            except Exception as e:
                results[name] = {
                    'status': 'error',
                    'message': str(e)
                }
        
        executable_count = sum(1 for r in results.values() 
                              if r.get('status') in ['executable', 'timeout'])
        
        if executable_count == len(self.binaries):
            status = 'pass'
            message = f"All {len(self.binaries)} binaries are executable"
        elif executable_count > 0:
            status = 'warning'
            message = f"{executable_count}/{len(self.binaries)} binaries are executable"
        else:
            status = 'fail'
            message = "No binaries are executable"
        
        return status, message, results
    
    def get_rust_version(self) -> Tuple[str, str, Dict[str, Any]]:
        """Get Rust toolchain version information"""
        details = {}
        
        try:
            # Check rustc
            result = subprocess.run(['rustc', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                details['rustc'] = result.stdout.strip()
            else:
                details['rustc_error'] = result.stderr.strip()
        except Exception as e:
            details['rustc_error'] = str(e)
        
        try:
            # Check cargo
            result = subprocess.run(['cargo', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                details['cargo'] = result.stdout.strip()
            else:
                details['cargo_error'] = result.stderr.strip()
        except Exception as e:
            details['cargo_error'] = str(e)
        
        try:
            # Check rustup if available
            result = subprocess.run(['rustup', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                details['rustup'] = result.stdout.strip()
        except:
            pass
        
        if 'rustc' in details and 'cargo' in details:
            status = 'pass'
            message = f"Rust toolchain available: {details['rustc'].split()[1]}"
        elif 'rustc' in details or 'cargo' in details:
            status = 'warning'
            message = "Partial Rust toolchain available"
        else:
            status = 'fail'
            message = "Rust toolchain not found"
        
        return status, message, details
    
    def test_proof_generation(self) -> Tuple[str, str, Dict[str, Any]]:
        """Test proof generation with sample data"""
        results = {}
        
        # Test SGD proof generation
        sgd_prover = self.binaries.get('sgd_prover_release') or self.binaries.get('sgd_prover_debug')
        if sgd_prover and sgd_prover.exists():
            try:
                test_input = {
                    "statement": {
                        "W_t_root": "0" * 64,
                        "W_t1_root": "1" * 64,
                        "batch_root": "2" * 64,
                        "hparams_hash": "3" * 32,
                        "step_nonce": 0,
                        "step_number": 1,
                        "epoch": 1
                    },
                    "witness": {
                        "weights_before": [0.1, 0.2, 0.3, 0.4],
                        "weights_after": [0.11, 0.21, 0.31, 0.41],
                        "batch_inputs": [1.0, 2.0, 3.0, 4.0],
                        "batch_targets": [0.5, 1.5],
                        "learning_rate": 0.01
                    }
                }
                
                start_time = time.time()
                result = subprocess.run(
                    [str(sgd_prover)],
                    input=json.dumps(test_input),
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                duration = time.time() - start_time
                
                if result.returncode == 0:
                    proof_size = len(result.stdout.encode()) if result.stdout else 0
                    results['sgd_proof'] = {
                        'status': 'success',
                        'duration_seconds': duration,
                        'proof_size_bytes': proof_size,
                        'return_code': result.returncode
                    }
                else:
                    results['sgd_proof'] = {
                        'status': 'failed',
                        'duration_seconds': duration,
                        'return_code': result.returncode,
                        'stderr': result.stderr[:500]  # Limit error message size
                    }
                
            except subprocess.TimeoutExpired:
                results['sgd_proof'] = {
                    'status': 'timeout',
                    'message': 'SGD proof generation timed out'
                }
            except Exception as e:
                results['sgd_proof'] = {
                    'status': 'error',
                    'message': str(e)
                }
        else:
            results['sgd_proof'] = {
                'status': 'skipped',
                'message': 'SGD prover binary not found'
            }
        
        # Similar test for LoRA (simplified for space)
        lora_prover = self.binaries.get('lora_prover_release') or self.binaries.get('lora_prover_debug')
        if lora_prover and lora_prover.exists():
            results['lora_proof'] = {
                'status': 'skipped',
                'message': 'LoRA proof generation test not implemented yet'
            }
        else:
            results['lora_proof'] = {
                'status': 'skipped',
                'message': 'LoRA prover binary not found'
            }
        
        # Determine overall status
        successful_tests = sum(1 for r in results.values() if r.get('status') == 'success')
        total_tests = len([r for r in results.values() if r.get('status') != 'skipped'])
        
        if total_tests == 0:
            status = 'skip'
            message = "No proof generation tests could be run"
        elif successful_tests == total_tests:
            status = 'pass'
            message = f"All {total_tests} proof generation tests passed"
        elif successful_tests > 0:
            status = 'warning'
            message = f"{successful_tests}/{total_tests} proof generation tests passed"
        else:
            status = 'fail'
            message = f"All {total_tests} proof generation tests failed"
        
        return status, message, results
    
    def test_verification(self) -> Tuple[str, str, Dict[str, Any]]:
        """Test proof verification"""
        return 'skip', 'Verification test not implemented yet', {}
    
    def measure_baseline_performance(self) -> Tuple[str, str, Dict[str, Any]]:
        """Measure baseline performance metrics"""
        results = {}
        
        # Test system performance
        start_time = time.time()
        
        # CPU benchmark - simple computation
        cpu_start = time.time()
        test_array = np.random.randn(1000, 1000).astype(np.float32)
        result_array = np.dot(test_array, test_array.T)
        cpu_duration = time.time() - cpu_start
        
        results['cpu_benchmark'] = {
            'matrix_multiply_1000x1000_seconds': cpu_duration,
            'operations_per_second': (1000 * 1000 * 1000) / cpu_duration
        }
        
        # Memory benchmark
        memory_start = time.time()
        large_array = np.random.randn(10000, 1000).astype(np.float32)
        memory_duration = time.time() - memory_start
        memory_size_mb = large_array.nbytes / (1024 * 1024)
        
        results['memory_benchmark'] = {
            'allocation_10M_floats_seconds': memory_duration,
            'memory_size_mb': memory_size_mb,
            'allocation_rate_mb_per_second': memory_size_mb / memory_duration
        }
        
        # Disk I/O benchmark
        disk_start = time.time()
        test_file = self.temp_dir / "performance_test.bin"
        test_data = np.random.bytes(10 * 1024 * 1024)  # 10MB
        
        with open(test_file, 'wb') as f:
            f.write(test_data)
        
        with open(test_file, 'rb') as f:
            read_data = f.read()
        
        disk_duration = time.time() - disk_start
        
        results['disk_benchmark'] = {
            'write_read_10mb_seconds': disk_duration,
            'throughput_mb_per_second': 20 / disk_duration,  # 10MB write + 10MB read
            'data_integrity': len(read_data) == len(test_data) and read_data == test_data
        }
        
        total_duration = time.time() - start_time
        
        results['summary'] = {
            'total_benchmark_time_seconds': total_duration,
            'system_suitable_for_zk': cpu_duration < 1.0 and memory_duration < 0.1
        }
        
        if results['summary']['system_suitable_for_zk']:
            status = 'pass'
            message = f"System performance suitable for ZK operations (CPU: {cpu_duration:.3f}s)"
        else:
            status = 'warning'
            message = f"System performance may be slow for ZK operations (CPU: {cpu_duration:.3f}s)"
        
        return status, message, results
    
    def check_python_dependencies(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check if required Python packages are available"""
        required_packages = [
            'numpy', 'scipy', 'scikit-learn', 'torch', 'transformers',
            'psutil', 'dataclasses', 'pathlib', 'json', 'hashlib'
        ]
        
        results = {}
        available_count = 0
        
        for package in required_packages:
            try:
                if package == 'torch':
                    import torch
                    results[package] = {
                        'available': True,
                        'version': torch.__version__,
                        'cuda_available': torch.cuda.is_available()
                    }
                elif package == 'transformers':
                    import transformers
                    results[package] = {
                        'available': True,
                        'version': transformers.__version__
                    }
                elif package == 'numpy':
                    import numpy as np
                    results[package] = {
                        'available': True,
                        'version': np.__version__
                    }
                elif package == 'scipy':
                    import scipy
                    results[package] = {
                        'available': True,
                        'version': scipy.__version__
                    }
                elif package == 'scikit-learn':
                    import sklearn
                    results[package] = {
                        'available': True,
                        'version': sklearn.__version__
                    }
                elif package == 'psutil':
                    import psutil
                    results[package] = {
                        'available': True,
                        'version': psutil.__version__
                    }
                else:
                    __import__(package)
                    results[package] = {'available': True}
                
                available_count += 1
                
            except ImportError:
                results[package] = {'available': False}
        
        if available_count == len(required_packages):
            status = 'pass'
            message = f"All {len(required_packages)} required packages available"
        elif available_count >= len(required_packages) * 0.8:
            status = 'warning'
            message = f"{available_count}/{len(required_packages)} required packages available"
        else:
            status = 'fail'
            message = f"Only {available_count}/{len(required_packages)} required packages available"
        
        return status, message, results
    
    def check_zk_module_imports(self) -> Tuple[str, str, Dict[str, Any]]:
        """Check if ZK modules can be imported"""
        zk_modules = [
            'pot.zk.witness',
            'pot.zk.spec', 
            'pot.zk.auto_prover',
            'pot.zk.exceptions',
            'pot.zk.metrics',
            'pot.prototypes.training_provenance_auditor'
        ]
        
        results = {}
        import_count = 0
        
        for module in zk_modules:
            try:
                imported = __import__(module, fromlist=[''])
                results[module] = {
                    'importable': True,
                    'file_path': getattr(imported, '__file__', 'unknown')
                }
                import_count += 1
            except ImportError as e:
                results[module] = {
                    'importable': False,
                    'error': str(e)
                }
        
        if import_count == len(zk_modules):
            status = 'pass'
            message = f"All {len(zk_modules)} ZK modules importable"
        elif import_count > 0:
            status = 'warning'
            message = f"{import_count}/{len(zk_modules)} ZK modules importable"
        else:
            status = 'fail'
            message = "No ZK modules could be imported"
        
        return status, message, results
    
    def diagnose_zk_system(self) -> Dict[str, Any]:
        """Run comprehensive diagnostic of ZK system"""
        self.log("Starting comprehensive ZK system diagnosis...")
        
        # Collect system info
        system_info = self.get_system_info()
        
        # Run all diagnostic tests
        tests = [
            ('Python Dependencies', self.check_python_dependencies),
            ('ZK Module Imports', self.check_zk_module_imports),
            ('Rust Environment', self.get_rust_version),
            ('Binary Existence', self.check_binaries_exist),
            ('Binary Executable', self.check_binaries_executable),
            ('Proof Generation', self.test_proof_generation),
            ('Proof Verification', self.test_verification),
            ('Performance Baseline', self.measure_baseline_performance),
        ]
        
        for test_name, test_func in tests:
            self.run_test(test_name, test_func)
        
        # Generate comprehensive report
        total_duration = time.time() - self.start_time
        
        # Count results by status
        status_counts = {}
        for result in self.results:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
        
        # Calculate overall health score
        total_tests = len(self.results)
        pass_count = status_counts.get('pass', 0)
        warning_count = status_counts.get('warning', 0)
        
        health_score = (pass_count + warning_count * 0.5) / total_tests if total_tests > 0 else 0
        
        # Determine overall status
        if health_score >= 0.9:
            overall_status = 'healthy'
        elif health_score >= 0.7:
            overall_status = 'degraded'
        else:
            overall_status = 'critical'
        
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'diagnostic_version': '1.0.0',
            'system_info': asdict(system_info),
            'overall_status': overall_status,
            'health_score': health_score,
            'summary': {
                'total_tests': total_tests,
                'passed': status_counts.get('pass', 0),
                'warnings': status_counts.get('warning', 0),
                'failures': status_counts.get('fail', 0),
                'skipped': status_counts.get('skip', 0),
                'total_duration_seconds': total_duration
            },
            'test_results': [asdict(result) for result in self.results],
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on diagnostic results"""
        recommendations = []
        
        # Check results and generate recommendations
        for result in self.results:
            if result.test_name == 'Binary Existence' and result.status == 'fail':
                recommendations.append("Build ZK binaries using: cd pot/zk/prover_halo2 && cargo build --release")
            
            elif result.test_name == 'Rust Environment' and result.status == 'fail':
                recommendations.append("Install Rust toolchain from https://rustup.rs/")
            
            elif result.test_name == 'Python Dependencies' and result.status != 'pass':
                recommendations.append("Install missing Python packages using: pip install -r requirements.txt")
            
            elif result.test_name == 'Performance Baseline' and result.status == 'warning':
                recommendations.append("Consider upgrading system hardware for better ZK proof performance")
        
        if not recommendations:
            recommendations.append("All systems operational - no immediate action required")
        
        return recommendations
    
    def __del__(self):
        """Cleanup temporary directory"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)


def diagnose_zk_system(verbose: bool = False) -> Dict[str, Any]:
    """
    Main diagnostic function to run comprehensive ZK system checks
    
    Args:
        verbose: Enable verbose logging
        
    Returns:
        Comprehensive diagnostic report dictionary
    """
    diagnostic = ZKDiagnostic(verbose=verbose)
    return diagnostic.diagnose_zk_system()


def main():
    """Command-line interface for ZK diagnostics"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ZK System Diagnostic Tool')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for diagnostic report (JSON)')
    parser.add_argument('--format', choices=['json', 'human'], default='human',
                       help='Output format')
    
    args = parser.parse_args()
    
    print("🔍 Running ZK System Diagnostics...")
    print("=" * 50)
    
    # Run diagnostics
    report = diagnose_zk_system(verbose=args.verbose)
    
    if args.format == 'json':
        output = json.dumps(report, indent=2, default=str)
    else:
        # Human-readable format
        output = f"""
ZK System Diagnostic Report
===========================

System Status: {report['overall_status'].upper()}
Health Score: {report['health_score']:.1%}
Test Duration: {report['summary']['total_duration_seconds']:.1f}s

System Information:
- Platform: {report['system_info']['platform']}
- Python: {report['system_info']['python_version']}
- CPUs: {report['system_info']['cpu_count']}
- Memory: {report['system_info']['memory_total_gb']:.1f} GB
- Rust: {report['system_info']['rust_version'] or 'Not available'}

Test Results:
- ✅ Passed: {report['summary']['passed']}
- ⚠️  Warnings: {report['summary']['warnings']}
- ❌ Failed: {report['summary']['failures']}
- ⏭️  Skipped: {report['summary']['skipped']}

Recommendations:
"""
        for rec in report['recommendations']:
            output += f"• {rec}\n"
        
        output += "\nFor detailed results, use --format json\n"
    
    if args.output:
        with open(args.output, 'w') as f:
            if args.format == 'json':
                json.dump(report, f, indent=2, default=str)
            else:
                f.write(output)
        print(f"Report saved to: {args.output}")
    else:
        print(output)
    
    # Exit with appropriate code
    if report['overall_status'] == 'healthy':
        sys.exit(0)
    elif report['overall_status'] == 'degraded':
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()