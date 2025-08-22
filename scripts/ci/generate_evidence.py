#!/usr/bin/env python3
"""
CI Evidence Bundle Generation

Generates comprehensive evidence bundles for CI runs, including:
- Test results and coverage reports
- Security scan results
- Performance metrics
- Build artifacts and logs
- Cryptographic verification data
"""

import argparse
import json
import os
import sys
import zipfile
import hashlib
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import tempfile

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.pot.security.audit_trail import AuditTrail
    from src.pot.audit.validation.audit_validator import AuditValidator
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


class CIEvidenceGenerator:
    """Generates comprehensive evidence bundles for CI runs"""
    
    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.evidence_data = {
            'generated_at': datetime.utcnow().isoformat(),
            'generator_version': '1.0.0',
            'evidence_type': 'ci_bundle',
            'metadata': {},
            'components': {}
        }
        self.temp_dir = Path(tempfile.mkdtemp(prefix='ci_evidence_'))
        
    def collect_git_information(self, commit_sha: str, run_id: str) -> Dict[str, Any]:
        """Collect Git repository information"""
        git_info = {
            'commit_sha': commit_sha,
            'run_id': run_id,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            # Get commit details
            result = subprocess.run(
                ['git', 'show', '--format="%H %an %ae %at %s"', '--no-patch', commit_sha],
                capture_output=True, text=True, cwd=Path.cwd()
            )
            if result.returncode == 0:
                commit_line = result.stdout.strip().strip('"')
                parts = commit_line.split(' ', 4)
                if len(parts) >= 5:
                    git_info.update({
                        'author_name': parts[1],
                        'author_email': parts[2],
                        'commit_timestamp': parts[3],
                        'commit_message': parts[4]
                    })
            
            # Get branch name
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True, text=True, cwd=Path.cwd()
            )
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
                
            # Get remote URL
            result = subprocess.run(
                ['git', 'config', '--get', 'remote.origin.url'],
                capture_output=True, text=True, cwd=Path.cwd()
            )
            if result.returncode == 0:
                git_info['remote_url'] = result.stdout.strip()
                
        except Exception as e:
            git_info['collection_error'] = str(e)
            
        return git_info
    
    def collect_test_results(self) -> Dict[str, Any]:
        """Collect test results and coverage data"""
        test_data = {
            'collected_at': datetime.utcnow().isoformat(),
            'test_files': [],
            'coverage_data': None,
            'performance_data': None
        }
        
        # Look for test result files
        test_patterns = [
            'coverage.xml',
            'pytest_results.xml',
            'benchmark_results.json',
            'test_results.json'
        ]
        
        for pattern in test_patterns:
            for test_file in Path.cwd().glob(f'**/{pattern}'):
                if test_file.is_file():
                    try:
                        # Copy to temp directory
                        dest = self.temp_dir / f'tests_{test_file.name}'
                        dest.write_bytes(test_file.read_bytes())
                        
                        test_data['test_files'].append({
                            'name': test_file.name,
                            'path': str(test_file),
                            'size_bytes': test_file.stat().st_size,
                            'copied_to': str(dest)
                        })
                    except Exception as e:
                        test_data['test_files'].append({
                            'name': test_file.name,
                            'path': str(test_file),
                            'error': str(e)
                        })
        
        # Look for experimental results
        results_dir = Path('experimental_results')
        if results_dir.exists():
            try:
                # Copy recent result files
                for result_file in results_dir.glob('*.json'):
                    if result_file.is_file():
                        age_hours = (time.time() - result_file.stat().st_mtime) / 3600
                        if age_hours < 24:  # Only include recent files
                            dest = self.temp_dir / f'results_{result_file.name}'
                            dest.write_bytes(result_file.read_bytes())
                            test_data['test_files'].append({
                                'name': f'experimental_{result_file.name}',
                                'path': str(result_file),
                                'size_bytes': result_file.stat().st_size,
                                'copied_to': str(dest)
                            })
            except Exception as e:
                test_data['experimental_results_error'] = str(e)
        
        return test_data
    
    def collect_security_artifacts(self) -> Dict[str, Any]:
        """Collect security scan results and artifacts"""
        security_data = {
            'collected_at': datetime.utcnow().isoformat(),
            'scan_results': [],
            'audit_trails': []
        }
        
        # Look for security scan results
        security_patterns = [
            'bandit-report.json',
            'safety-report.json',
            'pip-audit-report.json',
            'semgrep.sarif',
            'trivy-results.sarif'
        ]
        
        for pattern in security_patterns:
            for scan_file in Path.cwd().glob(f'**/{pattern}'):
                if scan_file.is_file():
                    try:
                        dest = self.temp_dir / f'security_{scan_file.name}'
                        dest.write_bytes(scan_file.read_bytes())
                        
                        security_data['scan_results'].append({
                            'type': pattern.split('-')[0] if '-' in pattern else pattern.split('.')[0],
                            'file': scan_file.name,
                            'path': str(scan_file),
                            'size_bytes': scan_file.stat().st_size,
                            'copied_to': str(dest)
                        })
                    except Exception as e:
                        security_data['scan_results'].append({
                            'type': pattern,
                            'file': scan_file.name,
                            'error': str(e)
                        })
        
        # Collect audit trails if available
        if AUDIT_AVAILABLE:
            try:
                validator = AuditValidator()
                # Look for audit files
                for audit_file in Path.cwd().glob('**/*audit*.log'):
                    if audit_file.is_file():
                        try:
                            entries = validator.load_audit_trail(audit_file)
                            result = validator.validate(entries)
                            
                            security_data['audit_trails'].append({
                                'file': str(audit_file),
                                'entries_count': len(entries),
                                'validation_result': {
                                    'is_valid': result.is_valid,
                                    'error_count': len(result.errors),
                                    'warning_count': len(result.warnings)
                                }
                            })
                        except Exception as e:
                            security_data['audit_trails'].append({
                                'file': str(audit_file),
                                'validation_error': str(e)
                            })
            except Exception as e:
                security_data['audit_validation_error'] = str(e)
        
        return security_data
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance and resource usage metrics"""
        perf_data = {
            'collected_at': datetime.utcnow().isoformat(),
            'build_metrics': {},
            'benchmark_results': [],
            'resource_usage': {}
        }
        
        try:
            import psutil
            
            # Current system state
            perf_data['resource_usage'] = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage_percent': psutil.disk_usage('/').percent,
                'cpu_count': psutil.cpu_count(),
                'memory_total_gb': psutil.virtual_memory().total / (1024**3)
            }
        except ImportError:
            perf_data['resource_usage']['error'] = 'psutil not available'
        
        # Look for benchmark files
        benchmark_patterns = [
            'benchmark_results.json',
            'performance_*.json',
            'metrics_*.json'
        ]
        
        for pattern in benchmark_patterns:
            for bench_file in Path.cwd().glob(f'**/{pattern}'):
                if bench_file.is_file():
                    try:
                        with open(bench_file, 'r') as f:
                            data = json.load(f)
                        
                        dest = self.temp_dir / f'perf_{bench_file.name}'
                        dest.write_text(json.dumps(data, indent=2))
                        
                        perf_data['benchmark_results'].append({
                            'file': bench_file.name,
                            'path': str(bench_file),
                            'data_summary': {
                                'keys': list(data.keys()) if isinstance(data, dict) else 'non_dict',
                                'size_bytes': bench_file.stat().st_size
                            },
                            'copied_to': str(dest)
                        })
                    except Exception as e:
                        perf_data['benchmark_results'].append({
                            'file': bench_file.name,
                            'error': str(e)
                        })
        
        return perf_data
    
    def collect_build_artifacts(self) -> Dict[str, Any]:
        """Collect build artifacts and logs"""
        build_data = {
            'collected_at': datetime.utcnow().isoformat(),
            'artifacts': [],
            'logs': []
        }
        
        # Look for build artifacts
        artifact_patterns = [
            'target/release/*',  # Rust artifacts
            'target/debug/*',    # Debug Rust artifacts
            'dist/*',           # Python distributions
            '*.whl',           # Python wheels
            '*.egg-info'       # Python egg info
        ]
        
        for pattern in artifact_patterns:
            for artifact in Path.cwd().glob(pattern):
                if artifact.is_file() and artifact.stat().st_size < 100 * 1024 * 1024:  # < 100MB
                    try:
                        dest = self.temp_dir / f'artifact_{artifact.name}'
                        if artifact.suffix in ['.bin', '.exe', '.so', '.dylib']:
                            # Just record metadata for binaries
                            build_data['artifacts'].append({
                                'name': artifact.name,
                                'path': str(artifact),
                                'size_bytes': artifact.stat().st_size,
                                'type': 'binary',
                                'checksum': self._calculate_file_hash(artifact)
                            })
                        else:
                            # Copy text files
                            dest.write_bytes(artifact.read_bytes())
                            build_data['artifacts'].append({
                                'name': artifact.name,
                                'path': str(artifact),
                                'size_bytes': artifact.stat().st_size,
                                'type': 'file',
                                'copied_to': str(dest)
                            })
                    except Exception as e:
                        build_data['artifacts'].append({
                            'name': artifact.name,
                            'error': str(e)
                        })
        
        # Look for recent log files
        log_patterns = ['*.log', 'logs/*.log']
        for pattern in log_patterns:
            for log_file in Path.cwd().glob(pattern):
                if log_file.is_file():
                    age_hours = (time.time() - log_file.stat().st_mtime) / 3600
                    if age_hours < 24:  # Only recent logs
                        try:
                            dest = self.temp_dir / f'log_{log_file.name}'
                            dest.write_bytes(log_file.read_bytes())
                            
                            build_data['logs'].append({
                                'name': log_file.name,
                                'path': str(log_file),
                                'size_bytes': log_file.stat().st_size,
                                'age_hours': age_hours,
                                'copied_to': str(dest)
                            })
                        except Exception as e:
                            build_data['logs'].append({
                                'name': log_file.name,
                                'error': str(e)
                            })
        
        return build_data
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file"""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception:
            return "hash_calculation_failed"
    
    def generate_evidence_bundle(self, commit_sha: str, run_id: str) -> str:
        """Generate complete evidence bundle"""
        print(f"Generating CI evidence bundle for commit {commit_sha[:8]}, run {run_id}")
        
        # Collect all evidence components
        self.evidence_data['metadata'] = self.collect_git_information(commit_sha, run_id)
        self.evidence_data['components']['tests'] = self.collect_test_results()
        self.evidence_data['components']['security'] = self.collect_security_artifacts()
        self.evidence_data['components']['performance'] = self.collect_performance_metrics()
        self.evidence_data['components']['build'] = self.collect_build_artifacts()
        
        # Add bundle metadata
        self.evidence_data['bundle_info'] = {
            'total_files': len(list(self.temp_dir.glob('*'))),
            'temp_directory': str(self.temp_dir),
            'bundle_hash': None  # Will be calculated after creation
        }
        
        # Create evidence metadata file
        metadata_file = self.temp_dir / 'evidence_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.evidence_data, f, indent=2)
        
        # Create ZIP bundle
        print(f"Creating evidence bundle: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(self.output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add all collected files
            for file_path in self.temp_dir.glob('*'):
                if file_path.is_file():
                    zf.write(file_path, file_path.name)
        
        # Calculate final bundle hash
        bundle_hash = self._calculate_file_hash(self.output_path)
        self.evidence_data['bundle_info']['bundle_hash'] = bundle_hash
        
        # Update metadata in bundle
        with open(metadata_file, 'w') as f:
            json.dump(self.evidence_data, f, indent=2)
        
        # Re-create bundle with updated metadata
        with zipfile.ZipFile(self.output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_path in self.temp_dir.glob('*'):
                if file_path.is_file():
                    zf.write(file_path, file_path.name)
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(self.temp_dir)
        
        bundle_size = self.output_path.stat().st_size
        print(f"✅ Evidence bundle created: {self.output_path}")
        print(f"   Size: {bundle_size / (1024*1024):.1f} MB")
        print(f"   Hash: {bundle_hash[:16]}...")
        print(f"   Files: {self.evidence_data['bundle_info']['total_files']}")
        
        return str(self.output_path)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Generate CI evidence bundle'
    )
    parser.add_argument(
        '--commit',
        type=str,
        required=True,
        help='Git commit SHA'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        required=True,
        help='CI run ID'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output evidence bundle path'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    try:
        generator = CIEvidenceGenerator(args.output)
        bundle_path = generator.generate_evidence_bundle(args.commit, args.run_id)
        
        if args.verbose:
            print(f"\nEvidence bundle details:")
            print(f"  Path: {bundle_path}")
            print(f"  Components: {list(generator.evidence_data['components'].keys())}")
            print(f"  Metadata: {generator.evidence_data['metadata']}")
        
        print(f"\n✅ CI evidence generation completed successfully")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Error generating evidence bundle: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()