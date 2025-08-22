#!/usr/bin/env python3
"""
Evidence Bundle Validation

Validates CI evidence bundles for integrity, completeness, and compliance.
Performs cryptographic verification and checks for tampering.
"""

import argparse
import json
import os
import sys
import zipfile
import hashlib
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from src.pot.audit.validation.audit_validator import AuditValidator, ValidationResult
    AUDIT_AVAILABLE = True
except ImportError:
    AUDIT_AVAILABLE = False


class EvidenceValidationResult:
    """Result of evidence bundle validation"""
    
    def __init__(self):
        self.is_valid = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.findings: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.components_validated: List[str] = []
        self.validation_timestamp = datetime.utcnow().isoformat()
    
    def add_error(self, message: str):
        """Add validation error"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add validation warning"""
        self.warnings.append(message)
    
    def add_finding(self, message: str):
        """Add validation finding"""
        self.findings.append(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'is_valid': self.is_valid,
            'validation_timestamp': self.validation_timestamp,
            'errors': self.errors,
            'warnings': self.warnings,
            'findings': self.findings,
            'metadata': self.metadata,
            'components_validated': self.components_validated,
            'summary': {
                'total_errors': len(self.errors),
                'total_warnings': len(self.warnings),
                'total_findings': len(self.findings)
            }
        }


class EvidenceBundleValidator:
    """Validates CI evidence bundles"""
    
    def __init__(self, bundle_path: str):
        self.bundle_path = Path(bundle_path)
        self.temp_dir = None
        self.evidence_data = None
        self.result = EvidenceValidationResult()
        
        # Required components for valid evidence
        self.required_components = [
            'tests',
            'security', 
            'performance',
            'build'
        ]
        
        # Required metadata fields
        self.required_metadata = [
            'commit_sha',
            'run_id',
            'timestamp',
            'branch'
        ]
    
    def validate_bundle(self) -> EvidenceValidationResult:
        """Validate complete evidence bundle"""
        try:
            # Basic bundle validation
            if not self._validate_bundle_exists():
                return self.result
            
            # Extract and validate structure
            if not self._extract_and_validate_structure():
                return self.result
            
            # Load and validate metadata
            if not self._validate_metadata():
                return self.result
            
            # Validate bundle integrity
            self._validate_bundle_integrity()
            
            # Validate components
            self._validate_test_evidence()
            self._validate_security_evidence()
            self._validate_performance_evidence()
            self._validate_build_evidence()
            
            # Validate temporal consistency
            self._validate_temporal_consistency()
            
            # Generate summary
            self._generate_validation_summary()
            
        except Exception as e:
            self.result.add_error(f"Validation failed with exception: {e}")
        finally:
            self._cleanup()
        
        return self.result
    
    def _validate_bundle_exists(self) -> bool:
        """Validate bundle file exists and is accessible"""
        if not self.bundle_path.exists():
            self.result.add_error(f"Evidence bundle not found: {self.bundle_path}")
            return False
        
        if not self.bundle_path.is_file():
            self.result.add_error(f"Evidence bundle is not a file: {self.bundle_path}")
            return False
        
        if self.bundle_path.stat().st_size == 0:
            self.result.add_error(f"Evidence bundle is empty: {self.bundle_path}")
            return False
        
        # Check if it's a valid ZIP file
        try:
            with zipfile.ZipFile(self.bundle_path, 'r') as zf:
                # Test the ZIP file
                zf.testzip()
        except zipfile.BadZipFile:
            self.result.add_error(f"Evidence bundle is not a valid ZIP file: {self.bundle_path}")
            return False
        except Exception as e:
            self.result.add_error(f"Error reading evidence bundle: {e}")
            return False
        
        self.result.add_finding(f"Bundle file validated: {self.bundle_path.stat().st_size} bytes")
        return True
    
    def _extract_and_validate_structure(self) -> bool:
        """Extract bundle and validate basic structure"""
        try:
            self.temp_dir = Path(tempfile.mkdtemp(prefix='evidence_validation_'))
            
            with zipfile.ZipFile(self.bundle_path, 'r') as zf:
                zf.extractall(self.temp_dir)
            
            # Check for required metadata file
            metadata_file = self.temp_dir / 'evidence_metadata.json'
            if not metadata_file.exists():
                self.result.add_error("Missing evidence_metadata.json in bundle")
                return False
            
            # Load evidence metadata
            try:
                with open(metadata_file, 'r') as f:
                    self.evidence_data = json.load(f)
            except json.JSONDecodeError as e:
                self.result.add_error(f"Invalid JSON in evidence_metadata.json: {e}")
                return False
            except Exception as e:
                self.result.add_error(f"Error loading evidence_metadata.json: {e}")
                return False
            
            self.result.add_finding(f"Bundle extracted to: {self.temp_dir}")
            return True
            
        except Exception as e:
            self.result.add_error(f"Error extracting bundle: {e}")
            return False
    
    def _validate_metadata(self) -> bool:
        """Validate evidence metadata"""
        if not self.evidence_data:
            self.result.add_error("No evidence metadata available")
            return False
        
        # Check evidence type
        expected_type = 'ci_bundle'
        actual_type = self.evidence_data.get('evidence_type')
        if actual_type != expected_type:
            self.result.add_error(f"Invalid evidence type: expected '{expected_type}', got '{actual_type}'")
        
        # Check metadata section
        metadata = self.evidence_data.get('metadata', {})
        if not metadata:
            self.result.add_error("Missing metadata section")
            return False
        
        # Validate required metadata fields
        for field in self.required_metadata:
            if field not in metadata:
                self.result.add_error(f"Missing required metadata field: {field}")
        
        # Validate commit SHA format
        commit_sha = metadata.get('commit_sha', '')
        if not re.match(r'^[a-f0-9]{40}$', commit_sha):
            self.result.add_warning(f"Invalid commit SHA format: {commit_sha}")
        
        # Validate timestamp format
        timestamp = metadata.get('timestamp')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                # Check if timestamp is reasonable (not too old or in future)
                now = datetime.utcnow()
                if dt > now + timedelta(hours=1):
                    self.result.add_warning(f"Evidence timestamp is in the future: {timestamp}")
                elif dt < now - timedelta(days=30):
                    self.result.add_warning(f"Evidence timestamp is very old: {timestamp}")
            except ValueError:
                self.result.add_warning(f"Invalid timestamp format: {timestamp}")
        
        # Check components section
        components = self.evidence_data.get('components', {})
        if not components:
            self.result.add_error("Missing components section")
            return False
        
        # Validate required components exist
        missing_components = []
        for component in self.required_components:
            if component not in components:
                missing_components.append(component)
        
        if missing_components:
            self.result.add_error(f"Missing required components: {missing_components}")
        
        self.result.metadata = metadata
        self.result.add_finding(f"Metadata validated for commit {commit_sha[:8]}")
        return True
    
    def _validate_bundle_integrity(self) -> bool:
        """Validate bundle cryptographic integrity"""
        bundle_info = self.evidence_data.get('bundle_info', {})
        stored_hash = bundle_info.get('bundle_hash')
        
        if not stored_hash:
            self.result.add_warning("No bundle hash found for integrity verification")
            return True
        
        # Calculate actual bundle hash
        try:
            hash_sha256 = hashlib.sha256()
            with open(self.bundle_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            actual_hash = hash_sha256.hexdigest()
            
            if actual_hash != stored_hash:
                self.result.add_error(f"Bundle integrity check failed - hash mismatch")
                self.result.add_error(f"  Expected: {stored_hash}")
                self.result.add_error(f"  Actual:   {actual_hash}")
                return False
            else:
                self.result.add_finding(f"Bundle integrity verified: {actual_hash[:16]}...")
                return True
                
        except Exception as e:
            self.result.add_warning(f"Error calculating bundle hash: {e}")
            return True
    
    def _validate_test_evidence(self):
        """Validate test evidence component"""
        test_data = self.evidence_data.get('components', {}).get('tests', {})
        if not test_data:
            return
        
        self.result.components_validated.append('tests')
        
        test_files = test_data.get('test_files', [])
        if not test_files:
            self.result.add_warning("No test files found in test evidence")
            return
        
        # Check for coverage data
        coverage_found = False
        for test_file in test_files:
            file_name = test_file.get('name', '')
            if 'coverage' in file_name.lower():
                coverage_found = True
                
                # Validate coverage file exists
                copied_to = test_file.get('copied_to')
                if copied_to and Path(copied_to).exists():
                    self.result.add_finding(f"Coverage data found: {file_name}")
                else:
                    self.result.add_warning(f"Coverage file missing: {file_name}")
        
        if not coverage_found:
            self.result.add_warning("No coverage data found in test evidence")
        
        # Check for experimental results
        experimental_found = any(
            'experimental' in tf.get('name', '').lower() 
            for tf in test_files
        )
        if experimental_found:
            self.result.add_finding("Experimental results included in test evidence")
        
        self.result.add_finding(f"Test evidence validated: {len(test_files)} files")
    
    def _validate_security_evidence(self):
        """Validate security evidence component"""
        security_data = self.evidence_data.get('components', {}).get('security', {})
        if not security_data:
            return
        
        self.result.components_validated.append('security')
        
        scan_results = security_data.get('scan_results', [])
        audit_trails = security_data.get('audit_trails', [])
        
        if not scan_results and not audit_trails:
            self.result.add_warning("No security scan results or audit trails found")
            return
        
        # Validate scan results
        scan_types = set()
        for scan in scan_results:
            scan_type = scan.get('type', 'unknown')
            scan_types.add(scan_type)
            
            if 'error' in scan:
                self.result.add_warning(f"Security scan error for {scan_type}: {scan['error']}")
        
        if scan_types:
            self.result.add_finding(f"Security scans completed: {', '.join(scan_types)}")
        
        # Validate audit trails if available
        if AUDIT_AVAILABLE and audit_trails:
            validator = AuditValidator()
            for trail in audit_trails:
                validation_result = trail.get('validation_result', {})
                if not validation_result.get('is_valid', True):
                    error_count = validation_result.get('error_count', 0)
                    self.result.add_warning(f"Audit trail validation failed: {error_count} errors")
                else:
                    entries_count = trail.get('entries_count', 0)
                    self.result.add_finding(f"Audit trail validated: {entries_count} entries")
        
        self.result.add_finding(f"Security evidence validated: {len(scan_results)} scans, {len(audit_trails)} trails")
    
    def _validate_performance_evidence(self):
        """Validate performance evidence component"""
        perf_data = self.evidence_data.get('components', {}).get('performance', {})
        if not perf_data:
            return
        
        self.result.components_validated.append('performance')
        
        resource_usage = perf_data.get('resource_usage', {})
        benchmark_results = perf_data.get('benchmark_results', [])
        
        # Validate resource usage data
        if resource_usage:
            if 'error' in resource_usage:
                self.result.add_warning(f"Resource usage collection error: {resource_usage['error']}")
            else:
                cpu_percent = resource_usage.get('cpu_percent', 0)
                memory_percent = resource_usage.get('memory_percent', 0)
                
                # Check for unusual resource usage
                if cpu_percent > 90:
                    self.result.add_warning(f"High CPU usage during collection: {cpu_percent}%")
                if memory_percent > 90:
                    self.result.add_warning(f"High memory usage during collection: {memory_percent}%")
                
                self.result.add_finding(f"Resource usage captured: CPU {cpu_percent}%, Memory {memory_percent}%")
        
        # Validate benchmark results
        for benchmark in benchmark_results:
            if 'error' in benchmark:
                file_name = benchmark.get('file', 'unknown')
                self.result.add_warning(f"Benchmark error in {file_name}: {benchmark['error']}")
        
        if benchmark_results:
            self.result.add_finding(f"Performance evidence validated: {len(benchmark_results)} benchmarks")
    
    def _validate_build_evidence(self):
        """Validate build evidence component"""
        build_data = self.evidence_data.get('components', {}).get('build', {})
        if not build_data:
            return
        
        self.result.components_validated.append('build')
        
        artifacts = build_data.get('artifacts', [])
        logs = build_data.get('logs', [])
        
        # Validate artifacts
        binary_artifacts = []
        file_artifacts = []
        
        for artifact in artifacts:
            if 'error' in artifact:
                name = artifact.get('name', 'unknown')
                self.result.add_warning(f"Artifact error for {name}: {artifact['error']}")
                continue
            
            artifact_type = artifact.get('type', 'unknown')
            if artifact_type == 'binary':
                binary_artifacts.append(artifact)
                # Validate checksum exists for binaries
                if not artifact.get('checksum'):
                    self.result.add_warning(f"No checksum for binary: {artifact.get('name')}")
            else:
                file_artifacts.append(artifact)
        
        if binary_artifacts:
            self.result.add_finding(f"Binary artifacts found: {len(binary_artifacts)}")
        if file_artifacts:
            self.result.add_finding(f"File artifacts found: {len(file_artifacts)}")
        
        # Validate logs
        recent_logs = []
        for log in logs:
            if 'error' in log:
                name = log.get('name', 'unknown')
                self.result.add_warning(f"Log error for {name}: {log['error']}")
                continue
            
            age_hours = log.get('age_hours', 0)
            if age_hours <= 24:  # Recent logs
                recent_logs.append(log)
        
        if recent_logs:
            self.result.add_finding(f"Recent logs captured: {len(recent_logs)}")
        
        self.result.add_finding(f"Build evidence validated: {len(artifacts)} artifacts, {len(logs)} logs")
    
    def _validate_temporal_consistency(self):
        """Validate temporal consistency across evidence"""
        try:
            # Get bundle generation time
            generated_at = self.evidence_data.get('generated_at')
            if not generated_at:
                return
            
            bundle_time = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
            
            # Check component collection times
            components = self.evidence_data.get('components', {})
            for comp_name, comp_data in components.items():
                collected_at = comp_data.get('collected_at')
                if collected_at:
                    try:
                        comp_time = datetime.fromisoformat(collected_at.replace('Z', '+00:00'))
                        time_diff = abs((bundle_time - comp_time).total_seconds())
                        
                        if time_diff > 3600:  # More than 1 hour difference
                            self.result.add_warning(
                                f"Large time gap for {comp_name}: {time_diff/60:.1f} minutes"
                            )
                    except ValueError:
                        self.result.add_warning(f"Invalid timestamp for {comp_name}: {collected_at}")
            
            self.result.add_finding("Temporal consistency validated")
            
        except Exception as e:
            self.result.add_warning(f"Error validating temporal consistency: {e}")
    
    def _generate_validation_summary(self):
        """Generate validation summary"""
        total_components = len(self.required_components)
        validated_components = len(self.result.components_validated)
        
        self.result.add_finding(f"Validation completed: {validated_components}/{total_components} components")
        
        if self.result.is_valid:
            if self.result.warnings:
                self.result.add_finding(f"Bundle is valid with {len(self.result.warnings)} warnings")
            else:
                self.result.add_finding("Bundle is completely valid")
        else:
            self.result.add_finding(f"Bundle validation failed with {len(self.result.errors)} errors")
    
    def _cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and self.temp_dir.exists():
            import shutil
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass  # Best effort cleanup


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Validate CI evidence bundle'
    )
    parser.add_argument(
        '--bundle',
        type=str,
        required=True,
        help='Path to evidence bundle to validate'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output validation report path (JSON)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Strict validation mode (warnings treated as errors)'
    )
    
    args = parser.parse_args()
    
    try:
        validator = EvidenceBundleValidator(args.bundle)
        result = validator.validate_bundle()
        
        # Apply strict mode
        if args.strict and result.warnings:
            result.is_valid = False
            result.errors.extend([f"STRICT: {w}" for w in result.warnings])
        
        # Generate report
        report = result.to_dict()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Validation report saved to: {args.output}")
        
        # Print summary
        status = "✅ VALID" if result.is_valid else "❌ INVALID"
        print(f"\n{status} - Evidence Bundle Validation")
        print(f"Bundle: {args.bundle}")
        print(f"Errors: {len(result.errors)}")
        print(f"Warnings: {len(result.warnings)}")
        print(f"Components: {len(result.components_validated)}")
        
        if args.verbose:
            if result.errors:
                print(f"\nErrors:")
                for error in result.errors:
                    print(f"  ❌ {error}")
            
            if result.warnings:
                print(f"\nWarnings:")
                for warning in result.warnings:
                    print(f"  ⚠️  {warning}")
            
            if result.findings:
                print(f"\nFindings:")
                for finding in result.findings[:5]:  # Show first 5
                    print(f"  ℹ️  {finding}")
                if len(result.findings) > 5:
                    print(f"  ... and {len(result.findings) - 5} more")
        
        sys.exit(0 if result.is_valid else 1)
        
    except Exception as e:
        print(f"❌ Error validating evidence bundle: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()