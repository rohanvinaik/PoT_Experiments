"""
PoT Validation Module

End-to-end validation pipeline for Proof-of-Training framework
"""

from .e2e_pipeline import (
    PipelineOrchestrator,
    PipelineConfig,
    PipelineStage,
    StageMetrics,
    TestingMode,
    VerificationMode
)

from .reporting import ReportGenerator

__all__ = [
    'PipelineOrchestrator',
    'PipelineConfig',
    'PipelineStage',
    'StageMetrics',
    'TestingMode',
    'VerificationMode',
    'ReportGenerator'
]

__version__ = '1.0.0'