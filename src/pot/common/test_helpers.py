"""
Common test helper utilities shared across the PoT framework.
Provides base test cases, fixtures, and testing utilities.
"""

import unittest
import tempfile
import shutil
import os
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from contextlib import contextmanager
import warnings
import time
from dataclasses import dataclass, field


@dataclass
class TestConfig:
    """Configuration for test execution"""
    test_dir: str = None
    cleanup: bool = True
    verbose: bool = False
    seed: int = 42
    device: str = "cpu"
    tolerance: float = 1e-6
    max_test_time: float = 60.0  # seconds
    
    def __post_init__(self):
        """Set up test directory if not provided"""
        if self.test_dir is None:
            self.test_dir = tempfile.mkdtemp(prefix="pot_test_")


class BaseTestCase(unittest.TestCase):
    """
    Base test case with common setup and teardown.
    Consolidates test infrastructure from multiple test files.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures"""
        cls.test_config = TestConfig()
        cls.test_dir = Path(cls.test_config.test_dir)
        cls.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(cls.test_config.seed)
        torch.manual_seed(cls.test_config.seed)
        
        # Track test timing
        cls.class_start_time = time.time()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test fixtures"""
        if cls.test_config.cleanup and cls.test_dir.exists():
            shutil.rmtree(cls.test_dir)
        
        # Report class test time
        class_time = time.time() - cls.class_start_time
        if cls.test_config.verbose:
            print(f"\n{cls.__name__} completed in {class_time:.2f}s")
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_start_time = time.time()
        self.temp_files = []
        self.temp_dirs = []
        
        # Create test-specific subdirectory
        self.test_subdir = self.test_dir / self._testMethodName
        self.test_subdir.mkdir(exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up temporary files
        for temp_file in self.temp_files:
            if Path(temp_file).exists():
                Path(temp_file).unlink()
        
        # Clean up temporary directories
        for temp_dir in self.temp_dirs:
            if Path(temp_dir).exists():
                shutil.rmtree(temp_dir)
        
        # Check test time
        test_time = time.time() - self.test_start_time
        if test_time > self.test_config.max_test_time:
            warnings.warn(f"{self._testMethodName} took {test_time:.2f}s (max: {self.test_config.max_test_time}s)")
    
    def create_temp_file(self, suffix: str = ".txt", content: Optional[str] = None) -> Path:
        """
        Create a temporary file for testing.
        
        Args:
            suffix: File suffix
            content: Optional content to write
            
        Returns:
            Path to temporary file
        """
        fd, path = tempfile.mkstemp(suffix=suffix, dir=self.test_subdir)
        os.close(fd)
        
        path = Path(path)
        self.temp_files.append(path)
        
        if content is not None:
            path.write_text(content)
        
        return path
    
    def create_temp_dir(self, prefix: str = "test_") -> Path:
        """
        Create a temporary directory for testing.
        
        Args:
            prefix: Directory prefix
            
        Returns:
            Path to temporary directory
        """
        path = Path(tempfile.mkdtemp(prefix=prefix, dir=self.test_subdir))
        self.temp_dirs.append(path)
        return path
    
    def assertTensorEqual(self, tensor1: torch.Tensor, tensor2: torch.Tensor, msg: Optional[str] = None):
        """
        Assert two tensors are equal.
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            msg: Optional message
        """
        self.assertTrue(torch.equal(tensor1, tensor2), msg or f"Tensors not equal")
    
    def assertTensorClose(self, tensor1: torch.Tensor, tensor2: torch.Tensor, 
                         rtol: float = 1e-5, atol: float = 1e-8, msg: Optional[str] = None):
        """
        Assert two tensors are close within tolerance.
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            rtol: Relative tolerance
            atol: Absolute tolerance
            msg: Optional message
        """
        self.assertTrue(torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol),
                       msg or f"Tensors not close (rtol={rtol}, atol={atol})")
    
    def assertArrayEqual(self, arr1: np.ndarray, arr2: np.ndarray, msg: Optional[str] = None):
        """
        Assert two numpy arrays are equal.
        
        Args:
            arr1: First array
            arr2: Second array
            msg: Optional message
        """
        np.testing.assert_array_equal(arr1, arr2, err_msg=msg)
    
    def assertArrayClose(self, arr1: np.ndarray, arr2: np.ndarray,
                        rtol: float = 1e-5, atol: float = 1e-8, msg: Optional[str] = None):
        """
        Assert two numpy arrays are close within tolerance.
        
        Args:
            arr1: First array
            arr2: Second array
            rtol: Relative tolerance
            atol: Absolute tolerance
            msg: Optional message
        """
        np.testing.assert_allclose(arr1, arr2, rtol=rtol, atol=atol, err_msg=msg)
    
    def assertFileExists(self, path: Union[str, Path], msg: Optional[str] = None):
        """Assert a file exists"""
        path = Path(path)
        self.assertTrue(path.exists(), msg or f"File {path} does not exist")
    
    def assertFileContains(self, path: Union[str, Path], text: str, msg: Optional[str] = None):
        """Assert a file contains specific text"""
        path = Path(path)
        content = path.read_text()
        self.assertIn(text, content, msg or f"Text '{text}' not found in {path}")


def create_test_fixture(fixture_type: str, **kwargs) -> Any:
    """
    Create common test fixtures.
    
    Args:
        fixture_type: Type of fixture to create
        **kwargs: Fixture-specific parameters
        
    Returns:
        Test fixture
    """
    if fixture_type == "model":
        from pot.common.model_utils import create_mock_model
        return create_mock_model(**kwargs)
    
    elif fixture_type == "tensor":
        shape = kwargs.get('shape', (10, 10))
        dtype = kwargs.get('dtype', torch.float32)
        device = kwargs.get('device', 'cpu')
        
        if kwargs.get('random', True):
            return torch.randn(shape, dtype=dtype, device=device)
        else:
            return torch.zeros(shape, dtype=dtype, device=device)
    
    elif fixture_type == "array":
        shape = kwargs.get('shape', (10, 10))
        dtype = kwargs.get('dtype', np.float32)
        
        if kwargs.get('random', True):
            return np.random.randn(*shape).astype(dtype)
        else:
            return np.zeros(shape, dtype=dtype)
    
    elif fixture_type == "dataset":
        num_samples = kwargs.get('num_samples', 100)
        input_dim = kwargs.get('input_dim', 10)
        output_dim = kwargs.get('output_dim', 2)
        
        X = np.random.randn(num_samples, input_dim).astype(np.float32)
        y = np.random.randint(0, output_dim, num_samples)
        
        return {'X': X, 'y': y}
    
    elif fixture_type == "config":
        return {
            'model_id': kwargs.get('model_id', 'test_model'),
            'batch_size': kwargs.get('batch_size', 32),
            'learning_rate': kwargs.get('learning_rate', 0.001),
            'epochs': kwargs.get('epochs', 10),
            'device': kwargs.get('device', 'cpu')
        }
    
    else:
        raise ValueError(f"Unknown fixture type: {fixture_type}")


def cleanup_test_files(*paths: Union[str, Path]):
    """
    Clean up test files and directories.
    
    Args:
        *paths: Paths to clean up
    """
    for path in paths:
        path = Path(path)
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)


def generate_test_data(data_type: str, **kwargs) -> Union[np.ndarray, torch.Tensor, Dict]:
    """
    Generate test data of various types.
    
    Args:
        data_type: Type of data to generate
        **kwargs: Data-specific parameters
        
    Returns:
        Generated test data
    """
    if data_type == "embeddings":
        num_samples = kwargs.get('num_samples', 100)
        dim = kwargs.get('dim', 768)
        
        # Generate normalized embeddings
        embeddings = np.random.randn(num_samples, dim).astype(np.float32)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    elif data_type == "time_series":
        length = kwargs.get('length', 1000)
        num_features = kwargs.get('num_features', 1)
        
        # Generate synthetic time series
        t = np.linspace(0, 10, length)
        data = np.zeros((length, num_features))
        
        for i in range(num_features):
            freq = np.random.uniform(0.5, 2.0)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.5, 2.0)
            noise = np.random.randn(length) * 0.1
            
            data[:, i] = amplitude * np.sin(freq * t + phase) + noise
        
        return data
    
    elif data_type == "images":
        num_samples = kwargs.get('num_samples', 10)
        height = kwargs.get('height', 224)
        width = kwargs.get('width', 224)
        channels = kwargs.get('channels', 3)
        
        # Generate random images
        images = np.random.rand(num_samples, channels, height, width).astype(np.float32)
        
        if kwargs.get('as_tensor', False):
            images = torch.from_numpy(images)
        
        return images
    
    elif data_type == "graph":
        num_nodes = kwargs.get('num_nodes', 20)
        num_edges = kwargs.get('num_edges', 40)
        
        # Generate random graph
        edges = []
        for _ in range(num_edges):
            src = np.random.randint(0, num_nodes)
            dst = np.random.randint(0, num_nodes)
            if src != dst:
                edges.append((src, dst))
        
        # Node features
        node_features = np.random.randn(num_nodes, kwargs.get('feature_dim', 10))
        
        return {
            'num_nodes': num_nodes,
            'edges': edges,
            'node_features': node_features
        }
    
    elif data_type == "text":
        num_samples = kwargs.get('num_samples', 10)
        max_length = kwargs.get('max_length', 100)
        vocab_size = kwargs.get('vocab_size', 1000)
        
        # Generate random token sequences
        sequences = []
        for _ in range(num_samples):
            length = np.random.randint(10, max_length)
            sequence = np.random.randint(0, vocab_size, length)
            sequences.append(sequence)
        
        return sequences
    
    else:
        raise ValueError(f"Unknown data type: {data_type}")


@contextmanager
def temporary_env_var(key: str, value: str):
    """
    Context manager for temporarily setting environment variables.
    
    Args:
        key: Environment variable key
        value: Environment variable value
    """
    old_value = os.environ.get(key)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is None:
            del os.environ[key]
        else:
            os.environ[key] = old_value


@contextmanager
def suppress_warnings(category: type = Warning):
    """
    Context manager to suppress warnings during testing.
    
    Args:
        category: Warning category to suppress
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category)
        yield


@contextmanager
def assert_raises_message(exception_class: type, message: str):
    """
    Context manager to assert an exception is raised with specific message.
    
    Args:
        exception_class: Expected exception class
        message: Expected message substring
    """
    try:
        yield
        raise AssertionError(f"Expected {exception_class.__name__} was not raised")
    except exception_class as e:
        if message not in str(e):
            raise AssertionError(f"Expected message '{message}' not found in '{str(e)}'")


class MockProgressBar:
    """Mock progress bar for testing CLI utilities"""
    
    def __init__(self, total: int = 100, desc: str = ""):
        """Initialize mock progress bar"""
        self.total = total
        self.desc = desc
        self.current = 0
        self.updates = []
    
    def update(self, n: int = 1):
        """Update progress"""
        self.current += n
        self.updates.append(n)
    
    def set_description(self, desc: str):
        """Set description"""
        self.desc = desc
    
    def close(self):
        """Close progress bar"""
        pass


class TestTimer:
    """Timer for measuring test execution time"""
    
    def __init__(self):
        """Initialize timer"""
        self.times = {}
        self.current_timer = None
        self.start_time = None
    
    def start(self, name: str):
        """Start a named timer"""
        self.current_timer = name
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop current timer and return elapsed time"""
        if self.current_timer is None:
            raise RuntimeError("No timer started")
        
        elapsed = time.perf_counter() - self.start_time
        
        if self.current_timer not in self.times:
            self.times[self.current_timer] = []
        self.times[self.current_timer].append(elapsed)
        
        self.current_timer = None
        self.start_time = None
        
        return elapsed
    
    @contextmanager
    def measure(self, name: str):
        """Context manager for timing a block"""
        self.start(name)
        try:
            yield
        finally:
            self.stop()
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a named timer"""
        if name not in self.times:
            return {}
        
        times = self.times[name]
        return {
            'count': len(times),
            'total': sum(times),
            'mean': np.mean(times),
            'std': np.std(times),
            'min': min(times),
            'max': max(times)
        }
    
    def report(self) -> str:
        """Generate timing report"""
        lines = ["Test Timing Report:"]
        for name in sorted(self.times.keys()):
            stats = self.get_stats(name)
            lines.append(f"  {name}: {stats['mean']:.3f}s ± {stats['std']:.3f}s (n={stats['count']})")
        return "\n".join(lines)


def compare_files(file1: Path, file2: Path, mode: str = "text") -> bool:
    """
    Compare two files for equality.
    
    Args:
        file1: First file path
        file2: Second file path
        mode: Comparison mode ("text", "binary", "json")
        
    Returns:
        True if files are equal
    """
    if not file1.exists() or not file2.exists():
        return False
    
    if mode == "text":
        return file1.read_text() == file2.read_text()
    
    elif mode == "binary":
        return file1.read_bytes() == file2.read_bytes()
    
    elif mode == "json":
        with open(file1) as f1, open(file2) as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)
        return data1 == data2
    
    else:
        raise ValueError(f"Unknown comparison mode: {mode}")


def save_test_artifact(data: Any, name: str, test_dir: Path, format: str = "auto") -> Path:
    """
    Save test artifact for debugging.
    
    Args:
        data: Data to save
        name: Artifact name
        test_dir: Test directory
        format: Save format ("auto", "json", "pickle", "numpy", "torch")
        
    Returns:
        Path to saved artifact
    """
    test_dir = Path(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "auto":
        if isinstance(data, (dict, list)):
            format = "json"
        elif isinstance(data, np.ndarray):
            format = "numpy"
        elif isinstance(data, torch.Tensor):
            format = "torch"
        else:
            format = "pickle"
    
    if format == "json":
        path = test_dir / f"{name}.json"
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    elif format == "pickle":
        path = test_dir / f"{name}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(data, f)
    
    elif format == "numpy":
        path = test_dir / f"{name}.npy"
        np.save(path, data)
    
    elif format == "torch":
        path = test_dir / f"{name}.pt"
        torch.save(data, path)
    
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return path


def load_test_artifact(path: Path, format: str = "auto") -> Any:
    """
    Load test artifact.
    
    Args:
        path: Path to artifact
        format: Load format
        
    Returns:
        Loaded data
    """
    path = Path(path)
    
    if format == "auto":
        if path.suffix == ".json":
            format = "json"
        elif path.suffix == ".pkl":
            format = "pickle"
        elif path.suffix == ".npy":
            format = "numpy"
        elif path.suffix == ".pt":
            format = "torch"
        else:
            raise ValueError(f"Cannot determine format for {path}")
    
    if format == "json":
        with open(path) as f:
            return json.load(f)
    
    elif format == "pickle":
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    elif format == "numpy":
        return np.load(path)
    
    elif format == "torch":
        return torch.load(path)
    
    else:
        raise ValueError(f"Unknown format: {format}")