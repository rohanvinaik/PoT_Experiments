#!/usr/bin/env python3
"""
Result Validator for PoT Framework

Comprehensive validation of experimental results including:
- Data integrity checks
- Statistical comparison with claims
- Consistency validation
- Reconciliation logic for discrepancies
- Strict mode for CI/CD pipelines

Usage:
    from pot.experiments.result_validator import ResultValidator
    
    validator = ResultValidator(strict_mode=False)
    report = validator.validate_directory("path/to/results")
    
    if not report.is_valid:
        print(f"Validation failed: {report.summary}")
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import logging
import warnings
from datetime import datetime
import scipy.stats as stats
import hashlib
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ValidationStatus(Enum):
    """Overall validation status."""
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"
    CRITICAL_FAILURE = "critical_failure"


@dataclass
class ValidationIssue:
    """Container for a single validation issue."""
    severity: ValidationSeverity
    category: str
    message: str
    file_path: Optional[str] = None
    field: Optional[str] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        msg = f"[{self.severity.value.upper()}] {self.category}: {self.message}"
        if self.file_path:
            msg += f" (File: {self.file_path})"
        if self.field:
            msg += f" (Field: {self.field})"
        if self.suggestion:
            msg += f"\n  Suggestion: {self.suggestion}"
        return msg


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    status: ValidationStatus
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_files: int = 0
    valid_files: int = 0
    invalid_files: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    metrics_summary: Dict[str, Any] = field(default_factory=dict)
    reconciliation_suggestions: List[str] = field(default_factory=list)
    confidence_score: float = 1.0
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed."""
        return self.status in [ValidationStatus.PASSED, ValidationStatus.PASSED_WITH_WARNINGS]
    
    @property
    def summary(self) -> str:
        """Generate summary of validation report."""
        summary = []
        summary.append(f"Validation Status: {self.status.value}")
        summary.append(f"Files: {self.valid_files}/{self.total_files} valid")
        
        # Count issues by severity
        severity_counts = {}
        for issue in self.issues:
            severity = issue.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        if severity_counts:
            summary.append("Issues:")
            for severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR, 
                           ValidationSeverity.WARNING, ValidationSeverity.INFO]:
                count = severity_counts.get(severity.value, 0)
                if count > 0:
                    summary.append(f"  {severity.value}: {count}")
        
        summary.append(f"Confidence: {self.confidence_score:.1%}")
        
        return "\n".join(summary)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "status": self.status.value,
            "timestamp": self.timestamp,
            "total_files": self.total_files,
            "valid_files": self.valid_files,
            "invalid_files": self.invalid_files,
            "issues": [
                {
                    "severity": issue.severity.value,
                    "category": issue.category,
                    "message": issue.message,
                    "file_path": issue.file_path,
                    "field": issue.field,
                    "expected": issue.expected,
                    "actual": issue.actual,
                    "suggestion": issue.suggestion
                }
                for issue in self.issues
            ],
            "metrics_summary": self.metrics_summary,
            "reconciliation_suggestions": self.reconciliation_suggestions,
            "confidence_score": self.confidence_score
        }
    
    def to_json(self) -> str:
        """Convert report to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def save(self, file_path: str) -> None:
        """Save report to file."""
        with open(file_path, 'w') as f:
            f.write(self.to_json())
        logger.info(f"Validation report saved to {file_path}")


@dataclass
class ClaimedMetrics:
    """Container for claimed/expected metrics."""
    far: float = 0.01
    frr: float = 0.01
    accuracy: float = 0.99
    avg_queries: float = 10.0
    max_queries: int = 100
    min_accuracy: float = 0.95
    confidence_level: float = 0.95
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClaimedMetrics':
        """Create from dictionary."""
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})
    
    @classmethod
    def from_file(cls, file_path: str) -> 'ClaimedMetrics':
        """Load from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class ResultValidator:
    """Comprehensive result validator for PoT experiments."""
    
    def __init__(self, strict_mode: bool = False, tolerance: float = 0.1):
        """
        Initialize the result validator.
        
        Args:
            strict_mode: If True, fail validation on any claim mismatch
            tolerance: Relative tolerance for metric comparisons (default 10%)
        """
        self.strict_mode = strict_mode
        self.tolerance = tolerance
        self.report = ValidationReport(status=ValidationStatus.PASSED)
        
        # Define validation rules
        self.validation_rules = {
            'confidence_scores': (0.0, 1.0),
            'far': (0.0, 1.0),
            'frr': (0.0, 1.0),
            'accuracy': (0.0, 1.0),
            'precision': (0.0, 1.0),
            'recall': (0.0, 1.0),
            'f1_score': (0.0, 1.0),
            'p_value': (0.0, 1.0),
            'threshold': (0.0, 1.0),
            'stopping_time': (1, float('inf')),
            'queries': (1, 10000),
            'processing_time': (0.0, 3600.0)  # Max 1 hour
        }
        
        logger.info(f"ResultValidator initialized (strict_mode={strict_mode}, tolerance={tolerance:.1%})")
    
    def validate_json_files(self, directory: str) -> ValidationReport:
        """
        Validate all JSON files in a directory for consistency.
        
        Args:
            directory: Path to directory containing JSON files
            
        Returns:
            ValidationReport with detailed findings
        """
        self.report = ValidationReport(status=ValidationStatus.PASSED)
        dir_path = Path(directory)
        
        if not dir_path.exists():
            self._add_issue(
                ValidationSeverity.CRITICAL,
                "Directory",
                f"Directory does not exist: {directory}"
            )
            self.report.status = ValidationStatus.CRITICAL_FAILURE
            return self.report
        
        # Find all JSON files
        json_files = list(dir_path.rglob("*.json"))
        self.report.total_files = len(json_files)
        
        if not json_files:
            self._add_issue(
                ValidationSeverity.WARNING,
                "Directory",
                f"No JSON files found in {directory}"
            )
            return self.report
        
        # Validate each file
        all_data = []
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Validate file structure and content
                if self._validate_json_structure(data, str(file_path)):
                    self.report.valid_files += 1
                    
                    # Collect data for cross-file validation
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
                else:
                    self.report.invalid_files += 1
                    
            except json.JSONDecodeError as e:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    "JSON",
                    f"Invalid JSON format: {e}",
                    file_path=str(file_path)
                )
                self.report.invalid_files += 1
            except Exception as e:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    "File",
                    f"Error reading file: {e}",
                    file_path=str(file_path)
                )
                self.report.invalid_files += 1
        
        # Perform cross-file validation
        if all_data:
            self._validate_data_consistency(all_data)
            self._calculate_metrics_summary(all_data)
        
        # Update overall status
        self._update_status()
        
        logger.info(f"Validated {self.report.total_files} files: "
                   f"{self.report.valid_files} valid, {self.report.invalid_files} invalid")
        
        return self.report
    
    def validate_single_file(self, file_path: str) -> ValidationReport:
        """
        Validate a single JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            ValidationReport for the single file
        """
        self.report = ValidationReport(status=ValidationStatus.PASSED)
        self.report.total_files = 1
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if self._validate_json_structure(data, file_path):
                self.report.valid_files = 1
                
                # Validate data
                all_data = []
                if isinstance(data, list):
                    for i, record in enumerate(data):
                        self._validate_result_record(record, f"{file_path}[{i}]")
                    all_data = data
                else:
                    self._validate_result_record(data, file_path)
                    all_data = [data]
                
                # Perform cross-record validation even for single file
                if all_data:
                    self._validate_data_consistency(all_data)
            else:
                self.report.invalid_files = 1
                
        except Exception as e:
            self._add_issue(
                ValidationSeverity.CRITICAL,
                "File",
                f"Failed to validate file: {e}",
                file_path=file_path
            )
            self.report.invalid_files = 1
            self.report.status = ValidationStatus.CRITICAL_FAILURE
        
        self._update_status()
        return self.report
    
    def compare_claimed_vs_actual(self, claimed_metrics: Union[ClaimedMetrics, Dict[str, Any]], 
                                 actual_results: List[Dict[str, Any]]) -> ValidationReport:
        """
        Statistical comparison of claimed vs actual metrics.
        
        Args:
            claimed_metrics: Expected metrics (ClaimedMetrics or dict)
            actual_results: List of actual result records
            
        Returns:
            ValidationReport with comparison results
        """
        self.report = ValidationReport(status=ValidationStatus.PASSED)
        
        # Convert claimed metrics if needed
        if isinstance(claimed_metrics, dict):
            claimed = ClaimedMetrics.from_dict(claimed_metrics)
        else:
            claimed = claimed_metrics
        
        # Extract actual metrics
        actual = self._extract_metrics(actual_results)
        
        # Compare each metric
        comparisons = [
            ('far', claimed.far, actual.get('far')),
            ('frr', claimed.frr, actual.get('frr')),
            ('accuracy', claimed.accuracy, actual.get('accuracy')),
            ('avg_queries', claimed.avg_queries, actual.get('avg_queries'))
        ]
        
        for metric_name, claimed_value, actual_value in comparisons:
            if actual_value is None:
                self._add_issue(
                    ValidationSeverity.WARNING,
                    "Comparison",
                    f"Missing actual value for {metric_name}",
                    field=metric_name
                )
                continue
            
            # Calculate difference
            abs_diff = abs(actual_value - claimed_value)
            rel_diff = abs_diff / claimed_value if claimed_value != 0 else float('inf')
            
            # Determine severity based on difference
            if rel_diff <= self.tolerance:
                severity = ValidationSeverity.INFO
                message = f"{metric_name}: Within tolerance ({rel_diff:.1%})"
            elif rel_diff <= self.tolerance * 2:
                severity = ValidationSeverity.WARNING
                message = f"{metric_name}: Moderate deviation ({rel_diff:.1%})"
            else:
                severity = ValidationSeverity.ERROR if not self.strict_mode else ValidationSeverity.CRITICAL
                message = f"{metric_name}: Large deviation ({rel_diff:.1%})"
            
            self._add_issue(
                severity,
                "Metric Comparison",
                message,
                field=metric_name,
                expected=claimed_value,
                actual=actual_value,
                suggestion=self._suggest_reconciliation(metric_name, claimed_value, actual_value)
            )
        
        # Perform statistical tests
        self._perform_statistical_tests(actual_results, claimed)
        
        # Update status
        self._update_status()
        
        # Generate reconciliation suggestions
        self._generate_reconciliation_suggestions()
        
        return self.report
    
    def check_data_integrity(self, results: List[Dict[str, Any]]) -> ValidationReport:
        """
        Verify no impossible values in results.
        
        Args:
            results: List of result records to check
            
        Returns:
            ValidationReport with integrity check results
        """
        self.report = ValidationReport(status=ValidationStatus.PASSED)
        
        for i, record in enumerate(results):
            record_id = record.get('experiment_id', f'record_{i}')
            
            # Check confidence scores in [0, 1]
            for field in ['confidence', 'p_value', 'far', 'frr', 'accuracy', 
                         'precision', 'recall', 'f1_score', 'threshold']:
                if field in record:
                    value = record[field]
                    if not isinstance(value, (int, float)):
                        self._add_issue(
                            ValidationSeverity.ERROR,
                            "Data Type",
                            f"{field} must be numeric",
                            field=field,
                            actual=type(value).__name__
                        )
                    elif not 0 <= value <= 1:
                        self._add_issue(
                            ValidationSeverity.ERROR,
                            "Range Check",
                            f"{field} must be in [0, 1]",
                            field=field,
                            actual=value
                        )
            
            # Check FAR + FRR consistency
            if 'far' in record and 'frr' in record:
                far, frr = record['far'], record['frr']
                
                # Check individual ranges (only if both are numeric)
                if isinstance(far, (int, float)) and isinstance(frr, (int, float)) and far + frr > 1.0:
                    self._add_issue(
                        ValidationSeverity.WARNING,
                        "Consistency",
                        f"FAR + FRR > 1.0 in {record_id}",
                        field="far+frr",
                        actual=far + frr,
                        suggestion="Check if FAR and FRR are calculated correctly"
                    )
                
                # Check against accuracy if present
                if 'accuracy' in record:
                    expected_accuracy = 1 - (far + frr) / 2
                    actual_accuracy = record['accuracy']
                    if isinstance(actual_accuracy, (int, float)) and abs(expected_accuracy - actual_accuracy) > 0.01:
                        self._add_issue(
                            ValidationSeverity.WARNING,
                            "Consistency",
                            f"Accuracy inconsistent with FAR/FRR in {record_id}",
                            field="accuracy",
                            expected=expected_accuracy,
                            actual=actual_accuracy
                        )
            
            # Validate stopping times > 0
            for field in ['stopping_time', 'queries', 'n_queries', 'total_queries']:
                if field in record:
                    value = record[field]
                    if not isinstance(value, (int, float)):
                        self._add_issue(
                            ValidationSeverity.ERROR,
                            "Data Type",
                            f"{field} must be numeric",
                            field=field,
                            actual=type(value).__name__
                        )
                    elif value <= 0:
                        self._add_issue(
                            ValidationSeverity.ERROR,
                            "Range Check",
                            f"{field} must be positive",
                            field=field,
                            actual=value
                        )
            
            # Ensure ground truth labels present for verification results
            if 'verified' in record and 'ground_truth' not in record:
                self._add_issue(
                    ValidationSeverity.WARNING,
                    "Missing Field",
                    f"Ground truth label missing for verification result in {record_id}",
                    field="ground_truth",
                    suggestion="Add ground_truth field for proper evaluation"
                )
            
            # Check for NaN or infinite values
            for field, value in record.items():
                if isinstance(value, (int, float)):
                    if np.isnan(value):
                        self._add_issue(
                            ValidationSeverity.ERROR,
                            "Invalid Value",
                            f"NaN value in {field}",
                            field=field
                        )
                    elif np.isinf(value):
                        self._add_issue(
                            ValidationSeverity.ERROR,
                            "Invalid Value",
                            f"Infinite value in {field}",
                            field=field
                        )
        
        self._update_status()
        return self.report
    
    def reconcile_discrepancies(self) -> List[str]:
        """
        Attempt to explain differences and suggest adjustments.
        
        Returns:
            List of reconciliation suggestions
        """
        suggestions = []
        
        # Analyze issues by category
        issue_categories = {}
        for issue in self.report.issues:
            if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                category = issue.category
                if category not in issue_categories:
                    issue_categories[category] = []
                issue_categories[category].append(issue)
        
        # Generate category-specific suggestions
        if "Metric Comparison" in issue_categories:
            metric_issues = issue_categories["Metric Comparison"]
            
            # Check if all metrics are consistently off
            deviations = []
            for issue in metric_issues:
                if issue.expected and issue.actual:
                    deviation = (issue.actual - issue.expected) / issue.expected
                    deviations.append(deviation)
            
            if deviations:
                avg_deviation = np.mean(deviations)
                if abs(avg_deviation) > 0.1:
                    if avg_deviation > 0:
                        suggestions.append(
                            "Metrics are consistently higher than claimed. Consider:\n"
                            "  - Reviewing threshold settings (may be too lenient)\n"
                            "  - Checking if test data is easier than expected\n"
                            "  - Verifying challenge generation parameters"
                        )
                    else:
                        suggestions.append(
                            "Metrics are consistently lower than claimed. Consider:\n"
                            "  - Adjusting model hyperparameters\n"
                            "  - Increasing training iterations\n"
                            "  - Reviewing data preprocessing steps"
                        )
        
        if "Range Check" in issue_categories:
            suggestions.append(
                "Invalid value ranges detected. Potential issues:\n"
                "  - Numerical overflow/underflow in calculations\n"
                "  - Incorrect normalization or scaling\n"
                "  - Missing data validation in preprocessing"
            )
        
        if "Consistency" in issue_categories:
            suggestions.append(
                "Metric consistency issues found. Check:\n"
                "  - FAR/FRR calculation methodology\n"
                "  - Accuracy computation formula\n"
                "  - Threshold application consistency"
            )
        
        # Check for potential bugs based on patterns
        if self._detect_potential_bugs():
            suggestions.append(
                "Potential bugs detected based on result patterns:\n"
                "  - Check for off-by-one errors in indexing\n"
                "  - Verify random seed handling for reproducibility\n"
                "  - Review data loading and batching logic"
            )
        
        self.report.reconciliation_suggestions = suggestions
        return suggestions
    
    def generate_validation_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive validation report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Formatted validation report as string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("POT EXPERIMENT VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Summary section
        report_lines.append("SUMMARY")
        report_lines.append("-" * 30)
        report_lines.append(f"Status: {self.report.status.value.upper()}")
        report_lines.append(f"Timestamp: {self.report.timestamp}")
        report_lines.append(f"Files Validated: {self.report.total_files}")
        report_lines.append(f"Valid Files: {self.report.valid_files}")
        report_lines.append(f"Invalid Files: {self.report.invalid_files}")
        report_lines.append(f"Confidence Score: {self.report.confidence_score:.1%}")
        report_lines.append("")
        
        # Issues by severity
        if self.report.issues:
            report_lines.append("ISSUES FOUND")
            report_lines.append("-" * 30)
            
            # Group by severity
            for severity in [ValidationSeverity.CRITICAL, ValidationSeverity.ERROR,
                           ValidationSeverity.WARNING, ValidationSeverity.INFO]:
                severity_issues = [i for i in self.report.issues if i.severity == severity]
                if severity_issues:
                    report_lines.append(f"\n{severity.value.upper()} ({len(severity_issues)} issues):")
                    for issue in severity_issues[:10]:  # Limit to first 10 per severity
                        report_lines.append(f"  • {issue.message}")
                        if issue.suggestion:
                            report_lines.append(f"    → {issue.suggestion}")
                    if len(severity_issues) > 10:
                        report_lines.append(f"  ... and {len(severity_issues) - 10} more")
        else:
            report_lines.append("✓ No issues found")
        
        report_lines.append("")
        
        # Metrics summary
        if self.report.metrics_summary:
            report_lines.append("METRICS SUMMARY")
            report_lines.append("-" * 30)
            for metric, value in self.report.metrics_summary.items():
                if isinstance(value, float):
                    report_lines.append(f"{metric}: {value:.4f}")
                else:
                    report_lines.append(f"{metric}: {value}")
            report_lines.append("")
        
        # Reconciliation suggestions
        if self.report.reconciliation_suggestions:
            report_lines.append("RECONCILIATION SUGGESTIONS")
            report_lines.append("-" * 30)
            for i, suggestion in enumerate(self.report.reconciliation_suggestions, 1):
                report_lines.append(f"{i}. {suggestion}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS")
        report_lines.append("-" * 30)
        
        if self.report.status == ValidationStatus.PASSED:
            report_lines.append("✓ Results are valid and ready for use")
        elif self.report.status == ValidationStatus.PASSED_WITH_WARNINGS:
            report_lines.append("⚠ Results are acceptable but review warnings")
            report_lines.append("  - Address warning issues before publication")
            report_lines.append("  - Document known limitations")
        elif self.report.status == ValidationStatus.FAILED:
            report_lines.append("✗ Results failed validation")
            report_lines.append("  - Fix critical issues before proceeding")
            report_lines.append("  - Re-run experiments after corrections")
        else:  # CRITICAL_FAILURE
            report_lines.append("✗✗ CRITICAL VALIDATION FAILURE")
            report_lines.append("  - Do not use these results")
            report_lines.append("  - Investigate root cause immediately")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        report_text = "\n".join(report_lines)
        
        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Validation report saved to {output_path}")
        
        return report_text
    
    # Private helper methods
    
    def _add_issue(self, severity: ValidationSeverity, category: str, message: str, **kwargs):
        """Add an issue to the report."""
        issue = ValidationIssue(severity, category, message, **kwargs)
        self.report.issues.append(issue)
        logger.debug(f"Added issue: {issue}")
    
    def _validate_json_structure(self, data: Any, file_path: str) -> bool:
        """Validate JSON structure."""
        if data is None:
            self._add_issue(
                ValidationSeverity.ERROR,
                "Structure",
                "Empty or null data",
                file_path=file_path
            )
            return False
        
        if not isinstance(data, (dict, list)):
            self._add_issue(
                ValidationSeverity.ERROR,
                "Structure",
                f"Invalid data type: {type(data).__name__}",
                file_path=file_path
            )
            return False
        
        if isinstance(data, list) and len(data) == 0:
            self._add_issue(
                ValidationSeverity.WARNING,
                "Structure",
                "Empty list",
                file_path=file_path
            )
        
        return True
    
    def _validate_result_record(self, record: Dict[str, Any], identifier: str) -> bool:
        """Validate a single result record."""
        valid = True
        
        # Check for required fields
        required_fields = []  # Can be customized based on experiment type
        for field in required_fields:
            if field not in record:
                self._add_issue(
                    ValidationSeverity.ERROR,
                    "Missing Field",
                    f"Required field '{field}' missing",
                    file_path=identifier,
                    field=field
                )
                valid = False
        
        # Validate field values based on rules
        for field, value in record.items():
            if field in self.validation_rules:
                min_val, max_val = self.validation_rules[field]
                if isinstance(value, (int, float)):
                    if not min_val <= value <= max_val:
                        self._add_issue(
                            ValidationSeverity.ERROR,
                            "Range Check",
                            f"{field} = {value} outside valid range [{min_val}, {max_val}]",
                            file_path=identifier,
                            field=field,
                            expected=f"[{min_val}, {max_val}]",
                            actual=value
                        )
                        valid = False
        
        return valid
    
    def _validate_data_consistency(self, all_data: List[Dict[str, Any]]) -> None:
        """Validate consistency across multiple records."""
        if not all_data:
            return
        
        # Check for duplicate IDs
        ids = [r.get('experiment_id') for r in all_data if 'experiment_id' in r]
        if ids:
            unique_ids = set(ids)
            if len(ids) != len(unique_ids):
                self._add_issue(
                    ValidationSeverity.WARNING,
                    "Consistency",
                    f"Duplicate experiment IDs found ({len(ids) - len(unique_ids)} duplicates)"
                )
        
        # Check for consistent field sets
        field_sets = [set(r.keys()) for r in all_data]
        if len(set(map(frozenset, field_sets))) > 1:
            self._add_issue(
                ValidationSeverity.INFO,
                "Consistency",
                "Inconsistent field sets across records"
            )
    
    def _calculate_metrics_summary(self, all_data: List[Dict[str, Any]]) -> None:
        """Calculate summary metrics from all data."""
        metrics = {}
        
        # Extract numeric fields
        numeric_fields = ['far', 'frr', 'accuracy', 'precision', 'recall', 
                         'f1_score', 'queries', 'processing_time']
        
        for field in numeric_fields:
            values = [r[field] for r in all_data if field in r and isinstance(r[field], (int, float))]
            if values:
                metrics[f"{field}_mean"] = np.mean(values)
                metrics[f"{field}_std"] = np.std(values)
                metrics[f"{field}_min"] = np.min(values)
                metrics[f"{field}_max"] = np.max(values)
        
        self.report.metrics_summary = metrics
    
    def _extract_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract aggregate metrics from results."""
        metrics = {}
        
        for field in ['far', 'frr', 'accuracy']:
            values = [r[field] for r in results if field in r]
            if values:
                metrics[field] = np.mean(values)
        
        # Calculate average queries
        query_fields = ['queries', 'query_count', 'total_queries', 'n_queries']
        query_values = []
        for r in results:
            for field in query_fields:
                if field in r:
                    query_values.append(r[field])
                    break
        if query_values:
            metrics['avg_queries'] = np.mean(query_values)
        
        return metrics
    
    def _perform_statistical_tests(self, actual_results: List[Dict[str, Any]], 
                                  claimed: ClaimedMetrics) -> None:
        """Perform statistical tests on results."""
        # Extract FAR values for testing
        far_values = [r['far'] for r in actual_results if 'far' in r]
        
        if len(far_values) >= 2:
            # One-sample t-test against claimed FAR
            t_stat, p_value = stats.ttest_1samp(far_values, claimed.far)
            
            if p_value < 0.05:
                self._add_issue(
                    ValidationSeverity.WARNING,
                    "Statistical Test",
                    f"FAR significantly different from claimed (p={p_value:.4f})",
                    field="far",
                    expected=claimed.far,
                    actual=np.mean(far_values)
                )
    
    def _suggest_reconciliation(self, metric: str, claimed: float, actual: float) -> str:
        """Generate reconciliation suggestion for a metric."""
        suggestions = {
            'far': {
                'higher': "FAR higher than expected. Try: decreasing threshold, improving model training, or enhancing feature engineering",
                'lower': "FAR lower than expected. Verify: test set difficulty, threshold calibration, or potential overfitting"
            },
            'frr': {
                'higher': "FRR higher than expected. Try: increasing threshold, adding more positive training examples, or adjusting class weights",
                'lower': "FRR lower than expected. Check: evaluation methodology and ensure proper cross-validation"
            },
            'accuracy': {
                'higher': "Accuracy higher than claimed. Verify: no data leakage, proper test/train split, and realistic test conditions",
                'lower': "Accuracy lower than claimed. Consider: model improvements, hyperparameter tuning, or data quality issues"
            },
            'avg_queries': {
                'higher': "More queries needed than expected. Optimize: early stopping criteria or challenge selection strategy",
                'lower': "Fewer queries than expected. Verify: sequential testing implementation is correct"
            }
        }
        
        if metric in suggestions:
            direction = 'higher' if actual > claimed else 'lower'
            return suggestions[metric].get(direction, "Review implementation and parameters")
        
        return "Review experimental setup and parameters"
    
    def _detect_potential_bugs(self) -> bool:
        """Detect potential bugs based on issue patterns."""
        # Look for suspicious patterns
        error_count = sum(1 for i in self.report.issues if i.severity == ValidationSeverity.ERROR)
        
        # Check for systematic issues
        if error_count > 5:
            return True
        
        # Check for impossible value combinations
        range_errors = [i for i in self.report.issues if i.category == "Range Check"]
        if len(range_errors) > 3:
            return True
        
        return False
    
    def _update_status(self) -> None:
        """Update overall validation status based on issues."""
        critical_count = sum(1 for i in self.report.issues if i.severity == ValidationSeverity.CRITICAL)
        error_count = sum(1 for i in self.report.issues if i.severity == ValidationSeverity.ERROR)
        warning_count = sum(1 for i in self.report.issues if i.severity == ValidationSeverity.WARNING)
        
        if critical_count > 0:
            self.report.status = ValidationStatus.CRITICAL_FAILURE
            self.report.confidence_score = 0.0
        elif error_count > 0:
            self.report.status = ValidationStatus.FAILED
            self.report.confidence_score = max(0.0, 1.0 - error_count * 0.2)
        elif warning_count > 0:
            self.report.status = ValidationStatus.PASSED_WITH_WARNINGS
            self.report.confidence_score = max(0.5, 1.0 - warning_count * 0.05)
        else:
            self.report.status = ValidationStatus.PASSED
            self.report.confidence_score = 1.0
        
        # Apply strict mode
        if self.strict_mode and self.report.status != ValidationStatus.PASSED:
            self.report.status = ValidationStatus.FAILED
    
    def _generate_reconciliation_suggestions(self) -> None:
        """Generate overall reconciliation suggestions."""
        self.report.reconciliation_suggestions = self.reconcile_discrepancies()


def validate_experiment_results(directory: str, claimed_metrics: Optional[Dict[str, Any]] = None,
                               strict_mode: bool = False) -> Tuple[bool, ValidationReport]:
    """
    Convenience function to validate experiment results.
    
    Args:
        directory: Directory containing result files
        claimed_metrics: Optional claimed metrics to compare against
        strict_mode: Whether to use strict validation
        
    Returns:
        Tuple of (is_valid, validation_report)
    """
    validator = ResultValidator(strict_mode=strict_mode)
    
    # Validate files
    report = validator.validate_json_files(directory)
    
    # Check data integrity
    if report.is_valid and report.metrics_summary:
        # Load all data for integrity check
        all_data = []
        for file_path in Path(directory).rglob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            except:
                pass
        
        if all_data:
            validator.check_data_integrity(all_data)
            
            # Compare with claims if provided
            if claimed_metrics:
                validator.compare_claimed_vs_actual(claimed_metrics, all_data)
    
    # Generate report
    print(validator.generate_validation_report())
    
    return report.is_valid, report


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        directory = sys.argv[1]
        strict = "--strict" in sys.argv
        
        # Run validation
        is_valid, report = validate_experiment_results(directory, strict_mode=strict)
        
        # Save report
        report.save(f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Exit with appropriate code
        sys.exit(0 if is_valid else 1)
    else:
        print("Usage: python result_validator.py <directory> [--strict]")
        print("\nExample with sample data:")
        
        # Create sample data for demonstration
        sample_data = [
            {
                "experiment_id": "exp_001",
                "far": 0.008,
                "frr": 0.012,
                "accuracy": 0.99,
                "queries": 8,
                "ground_truth": True,
                "verified": True
            },
            {
                "experiment_id": "exp_002",
                "far": 0.15,  # Invalid: too high
                "frr": -0.01,  # Invalid: negative
                "accuracy": 0.95,
                "queries": 0,  # Invalid: should be positive
            }
        ]
        
        # Save sample data
        with open("sample_validation_data.json", 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        # Run validation
        validator = ResultValidator(strict_mode=False)
        report = validator.validate_single_file("sample_validation_data.json")
        validator.check_data_integrity(sample_data)
        
        # Compare with claims
        claimed = {"far": 0.01, "frr": 0.01, "accuracy": 0.99}
        validator.compare_claimed_vs_actual(claimed, sample_data)
        
        # Generate and print report
        print(validator.generate_validation_report())
        
        # Clean up
        import os
        os.remove("sample_validation_data.json")