"""
PoT Experiments Module

This module provides reproducible experiment runners and validation pipelines
for the Proof-of-Training framework.
"""

from .reproducible_runner import ReproducibleExperimentRunner
from .sequential_decision import SequentialDecisionMaker, SPRTConfig, create_sequential_decision_maker
from .metrics_calculator import MetricsCalculator, PaperClaims, create_metrics_calculator, calculate_all_metrics
from .report_generator import ReportGenerator, ResultMetrics, Discrepancy, create_sample_results
from .result_validator import (
    ResultValidator, ValidationReport, ValidationStatus, ValidationSeverity,
    ValidationIssue, ClaimedMetrics, validate_experiment_results
)

__all__ = [
    'ReproducibleExperimentRunner', 
    'SequentialDecisionMaker', 
    'SPRTConfig', 
    'create_sequential_decision_maker',
    'MetricsCalculator',
    'PaperClaims',
    'create_metrics_calculator',
    'calculate_all_metrics',
    'ReportGenerator',
    'ResultMetrics', 
    'Discrepancy',
    'create_sample_results',
    'ResultValidator',
    'ValidationReport',
    'ValidationStatus',
    'ValidationSeverity',
    'ValidationIssue',
    'ClaimedMetrics',
    'validate_experiment_results'
]