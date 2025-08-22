"""
PoT Testing Framework

Provides deterministic test models and validation configurations 
for reliable, reproducible testing of the Proof of Training system.
"""

from .test_models import (
    DeterministicMockModel,
    LinearTestModel, 
    ConsistentHashModel,
    create_test_model
)

from .validation_config import (
    ValidationConfig,
    get_reliable_test_config,
    get_comprehensive_test_config,
    create_test_models_from_config
)

__all__ = [
    'DeterministicMockModel',
    'LinearTestModel',
    'ConsistentHashModel', 
    'create_test_model',
    'ValidationConfig',
    'get_reliable_test_config',
    'get_comprehensive_test_config',
    'create_test_models_from_config'
]