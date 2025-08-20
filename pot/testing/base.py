"""
Base classes and mixins for testing utilities in the PoT framework.
Consolidates duplicate implementations across the codebase.
"""

import unittest
import tempfile
import shutil
import time
import json
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from abc import ABC, abstractmethod


class BaseMockModel:
    """
    Base class for all mock models used in testing.
    Consolidates duplicate forward() and state_dict() implementations.
    """
    
    def __init__(self, model_id: str = "mock_model", output_dim: int = 10):
        """
        Initialize base mock model.
        
        Args:
            model_id: Unique identifier for this model
            output_dim: Dimension of output vector
        """
        self.model_id = model_id
        self.output_dim = output_dim
        self._state = {}
    
    def forward(self, x: Any) -> np.ndarray:
        """
        Standard forward pass returning random output.
        Override this method for custom behavior.
        
        Args:
            x: Input data
            
        Returns:
            Random output vector of specified dimension
        """
        return np.random.randn(self.output_dim)
    
    def __call__(self, x: Any) -> np.ndarray:
        """Make model callable."""
        return self.forward(x)
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return model state dictionary.
        Override this method for custom state.
        
        Returns:
            Dictionary containing model state
        """
        if not self._state:
            # Default state for compatibility
            self._state = {
                'layer': 'weights',
                'model_id': self.model_id,
                'output_dim': self.output_dim
            }
        return self._state
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary."""
        self._state = state_dict


class SimpleForwardModel(BaseMockModel):
    """
    Simple mock model with basic forward() implementation.
    Replaces duplicate implementations in:
    - experimental_results/validation_experiment.py
    - scripts/verify_installation.py
    """
    
    def __init__(self):
        super().__init__(model_id="simple_forward", output_dim=10)
    
    def state_dict(self) -> Dict[str, Any]:
        """Simple state dict with single layer."""
        return {'layer': 'weights'}


class StatefulMockModel(BaseMockModel):
    """
    Mock model with multiple layer state.
    Replaces duplicate implementation in:
    - pot/security/proof_of_training.py
    """
    
    def __init__(self):
        super().__init__(model_id="stateful_mock", output_dim=10)
    
    def state_dict(self) -> Dict[str, Any]:
        """State dict with multiple layers."""
        return {
            'layer1': 'weights',
            'layer2': 'weights'
        }


class StateDictMixin:
    """
    Mixin class providing standard state_dict functionality.
    Use this to add state management to any class.
    """
    
    def __init__(self):
        self._internal_state = {}
    
    def state_dict(self) -> Dict[str, Any]:
        """Return state dictionary."""
        return self._internal_state.copy()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dictionary."""
        self._internal_state = state_dict.copy()
    
    def update_state(self, key: str, value: Any) -> None:
        """Update a single state value."""
        self._internal_state[key] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self._internal_state.get(key, default)


class BaseTestCase(unittest.TestCase):
    """
    Base test case class with common setUp and tearDown methods.
    Consolidates duplicate test setup across test files.
    """
    
    def setUp(self):
        """Common test setup."""
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
        # Initialize common test attributes
        self.start_time = time.time()
        self.test_results = []
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
    def tearDown(self):
        """Common test teardown."""
        # Clean up temporary directory
        if hasattr(self, 'test_dir') and Path(self.test_dir).exists():
            shutil.rmtree(self.test_dir)
        
        # Log test duration
        duration = time.time() - self.start_time
        if hasattr(self, 'test_results'):
            self.test_results.append({
                'test_name': self._testMethodName,
                'duration': duration
            })
    
    def create_temp_file(self, filename: str, content: str = "") -> Path:
        """Create a temporary file for testing."""
        file_path = self.test_path / filename
        file_path.write_text(content)
        return file_path
    
    def create_mock_model(self, model_type: str = "simple") -> BaseMockModel:
        """Create a mock model for testing."""
        if model_type == "simple":
            return SimpleForwardModel()
        elif model_type == "stateful":
            return StatefulMockModel()
        else:
            return BaseMockModel()
    
    def assert_json_equal(self, json1: Any, json2: Any, msg: Optional[str] = None):
        """Assert two JSON-serializable objects are equal."""
        self.assertEqual(
            json.dumps(json1, sort_keys=True),
            json.dumps(json2, sort_keys=True),
            msg
        )


class BaseExperimentRunner(ABC):
    """
    Base class for experiment runners.
    Consolidates duplicate run_experiments() implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config or {}
        self.results = []
        self.metrics = {}
        
    @abstractmethod
    def setup_experiment(self) -> None:
        """Set up experiment environment. Override in subclasses."""
        pass
    
    @abstractmethod
    def run_single_experiment(self, **kwargs) -> Dict[str, Any]:
        """Run a single experiment. Override in subclasses."""
        pass
    
    def run_experiments(self, num_experiments: int = 1, **kwargs) -> List[Dict[str, Any]]:
        """
        Run multiple experiments and collect results.
        
        Args:
            num_experiments: Number of experiments to run
            **kwargs: Additional arguments for experiments
            
        Returns:
            List of experiment results
        """
        self.setup_experiment()
        
        for i in range(num_experiments):
            print(f"Running experiment {i+1}/{num_experiments}")
            
            # Run single experiment
            result = self.run_single_experiment(
                experiment_id=i,
                **kwargs
            )
            
            # Store result
            result['experiment_id'] = i
            result['timestamp'] = time.time()
            self.results.append(result)
            
            # Update metrics
            self._update_metrics(result)
        
        return self.results
    
    def _update_metrics(self, result: Dict[str, Any]) -> None:
        """Update running metrics from experiment result."""
        for key, value in result.items():
            if isinstance(value, (int, float)):
                if key not in self.metrics:
                    self.metrics[key] = []
                self.metrics[key].append(value)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of all experiments.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'num_experiments': len(self.results),
            'metrics': {}
        }
        
        for key, values in self.metrics.items():
            if values:
                summary['metrics'][key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        return summary
    
    def save_results(self, filename: str) -> None:
        """Save experiment results to file."""
        with open(filename, 'w') as f:
            json.dump({
                'config': self.config,
                'results': self.results,
                'summary': self.get_summary()
            }, f, indent=2, default=str)


class SimpleExperimentRunner(BaseExperimentRunner):
    """
    Simple implementation of experiment runner for basic experiments.
    """
    
    def setup_experiment(self) -> None:
        """Basic setup."""
        self.model = BaseMockModel()
        
    def run_single_experiment(self, **kwargs) -> Dict[str, Any]:
        """Run a simple experiment."""
        # Example experiment
        input_data = np.random.randn(100, 10)
        output = self.model.forward(input_data)
        
        return {
            'success': True,
            'output_shape': output.shape,
            'output_mean': float(np.mean(output)),
            'output_std': float(np.std(output))
        }


# Utility functions for common patterns

def create_mock_model(model_type: str = "simple", **kwargs) -> BaseMockModel:
    """
    Factory function to create mock models.
    
    Args:
        model_type: Type of mock model to create
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Mock model instance
    """
    if model_type == "simple":
        return SimpleForwardModel()
    elif model_type == "stateful":
        return StatefulMockModel()
    else:
        return BaseMockModel(**kwargs)


def run_standard_experiment(model: BaseMockModel, 
                           num_trials: int = 10) -> Dict[str, Any]:
    """
    Run a standard experiment with a mock model.
    
    Args:
        model: Mock model to test
        num_trials: Number of trials to run
        
    Returns:
        Experiment results
    """
    results = []
    
    for _ in range(num_trials):
        input_data = np.random.randn(10)
        output = model.forward(input_data)
        results.append(output)
    
    results_array = np.array(results)
    
    return {
        'num_trials': num_trials,
        'mean_output': np.mean(results_array, axis=0).tolist(),
        'std_output': np.std(results_array, axis=0).tolist(),
        'model_state': model.state_dict()
    }