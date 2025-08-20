"""
Common utilities shared across the PoT framework.
This module provides reusable components to reduce code duplication.
"""

from .model_utils import (
    BaseModelWrapper,
    ModelStateMixin,
    create_mock_model,
    wrap_model,
    get_model_signature
)

from .test_helpers import (
    BaseTestCase,
    create_test_fixture,
    cleanup_test_files,
    generate_test_data
)

from .cli_utils import (
    CLIFormatter,
    add_common_arguments,
    format_results,
    create_progress_bar
)

from .experiment_utils import (
    ExperimentRunner,
    MetricsCollector,
    ResultSerializer,
    run_experiment
)

from .verification_utils import (
    ChallengeGenerator,
    ResponseEvaluator,
    VerificationHelper,
    compute_confidence
)

__all__ = [
    # Model utilities
    'BaseModelWrapper',
    'ModelStateMixin',
    'create_mock_model',
    'wrap_model',
    'get_model_signature',
    
    # Test helpers
    'BaseTestCase',
    'create_test_fixture',
    'cleanup_test_files',
    'generate_test_data',
    
    # CLI utilities
    'CLIFormatter',
    'add_common_arguments',
    'format_results',
    'create_progress_bar',
    
    # Experiment utilities
    'ExperimentRunner',
    'MetricsCollector',
    'ResultSerializer',
    'run_experiment',
    
    # Verification utilities
    'ChallengeGenerator',
    'ResponseEvaluator',
    'VerificationHelper',
    'compute_confidence'
]