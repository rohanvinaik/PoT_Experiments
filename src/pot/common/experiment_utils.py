"""
Common experiment utilities shared across the PoT framework.
Provides experiment runners, metrics collectors, and result serialization.
"""

import json
import pickle
import yaml
import csv
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union, Callable
from datetime import datetime
from dataclasses import dataclass, asdict, field
import numpy as np
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from enum import Enum


logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExperimentConfig:
    """Configuration for experiments"""
    name: str = "experiment"
    description: str = ""
    output_dir: Path = Path("experimental_results")
    save_interval: int = 100
    checkpoint: bool = True
    metrics_to_track: List[str] = field(default_factory=lambda: ["loss", "accuracy"])
    seed: int = 42
    device: str = "cpu"
    max_duration: float = 3600.0  # seconds
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['output_dir'] = str(self.output_dir)
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create from dictionary"""
        if 'output_dir' in data:
            data['output_dir'] = Path(data['output_dir'])
        return cls(**data)


class MetricsCollector:
    """
    Collects and aggregates metrics during experiments.
    Consolidates metric tracking from multiple experiment files.
    """
    
    def __init__(self):
        """Initialize metrics collector"""
        self.metrics = defaultdict(list)
        self.timestamps = defaultdict(list)
        self.metadata = {}
        self.start_time = None
        self.last_log_time = None
    
    def start(self):
        """Start metrics collection"""
        self.start_time = time.time()
        self.last_log_time = self.start_time
    
    def log(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics for current step.
        
        Args:
            metrics: Dictionary of metric values
            step: Optional step number
        """
        current_time = time.time()
        
        for name, value in metrics.items():
            self.metrics[name].append(value)
            self.timestamps[name].append(current_time)
        
        if step is not None:
            self.metrics['step'].append(step)
            self.timestamps['step'].append(current_time)
        
        self.last_log_time = current_time
    
    def log_scalar(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a single scalar metric.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        self.log({name: value}, step)
    
    def log_histogram(self, name: str, values: np.ndarray, step: Optional[int] = None):
        """
        Log histogram data.
        
        Args:
            name: Metric name
            values: Array of values
            step: Optional step number
        """
        stats = {
            f"{name}_mean": np.mean(values),
            f"{name}_std": np.std(values),
            f"{name}_min": np.min(values),
            f"{name}_max": np.max(values),
            f"{name}_median": np.median(values)
        }
        self.log(stats, step)
        
        # Store full histogram data
        if f"{name}_hist" not in self.metrics:
            self.metrics[f"{name}_hist"] = []
        self.metrics[f"{name}_hist"].append(values.tolist())
    
    def get_metric(self, name: str) -> List[float]:
        """Get values for a specific metric"""
        return list(self.metrics.get(name, []))
    
    def get_latest(self, name: str) -> Optional[float]:
        """Get latest value for a metric"""
        values = self.metrics.get(name, [])
        return values[-1] if values else None
    
    def get_average(self, name: str, last_n: Optional[int] = None) -> Optional[float]:
        """
        Get average value for a metric.
        
        Args:
            name: Metric name
            last_n: Average over last n values (None = all)
            
        Returns:
            Average value or None if no data
        """
        values = self.metrics.get(name, [])
        if not values:
            return None
        
        if last_n is not None:
            values = values[-last_n:]
        
        return np.mean(values)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all metrics.
        
        Returns:
            Dictionary of summary statistics
        """
        summary = {}
        
        for name, values in self.metrics.items():
            if name.endswith('_hist'):
                continue  # Skip histogram data
            
            if values:
                summary[name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'last': values[-1]
                }
        
        if self.start_time:
            summary['duration'] = time.time() - self.start_time
        
        return summary
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.timestamps.clear()
        self.metadata.clear()
        self.start_time = None
        self.last_log_time = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'metrics': dict(self.metrics),
            'timestamps': dict(self.timestamps),
            'metadata': self.metadata,
            'start_time': self.start_time,
            'summary': self.get_summary()
        }


class ResultSerializer:
    """
    Serializes experiment results to various formats.
    Consolidates serialization logic from multiple files.
    """
    
    @staticmethod
    def save(data: Any, path: Path, format: str = "auto"):
        """
        Save data to file.
        
        Args:
            data: Data to save
            path: Output path
            format: Format ("auto", "json", "yaml", "pickle", "csv")
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "auto":
            # Determine format from extension
            if path.suffix == ".json":
                format = "json"
            elif path.suffix in [".yaml", ".yml"]:
                format = "yaml"
            elif path.suffix == ".pkl":
                format = "pickle"
            elif path.suffix == ".csv":
                format = "csv"
            else:
                format = "json"  # Default
        
        if format == "json":
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        
        elif format == "yaml":
            with open(path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        elif format == "pickle":
            with open(path, 'wb') as f:
                pickle.dump(data, f)
        
        elif format == "csv":
            ResultSerializer._save_csv(data, path)
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        logger.info(f"Saved results to {path}")
    
    @staticmethod
    def _save_csv(data: Any, path: Path):
        """Save data as CSV"""
        if isinstance(data, dict):
            # Convert dict to list of rows
            if 'metrics' in data:
                # Metrics format
                rows = []
                metrics = data['metrics']
                max_len = max(len(v) for v in metrics.values())
                
                for i in range(max_len):
                    row = {}
                    for key, values in metrics.items():
                        if i < len(values):
                            row[key] = values[i]
                    rows.append(row)
                
                data = rows
            else:
                # Single row
                data = [data]
        
        if data and isinstance(data[0], dict):
            with open(path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)
        else:
            raise ValueError("Cannot save data as CSV")
    
    @staticmethod
    def load(path: Path, format: str = "auto") -> Any:
        """
        Load data from file.
        
        Args:
            path: Input path
            format: Format
            
        Returns:
            Loaded data
        """
        path = Path(path)
        
        if format == "auto":
            if path.suffix == ".json":
                format = "json"
            elif path.suffix in [".yaml", ".yml"]:
                format = "yaml"
            elif path.suffix == ".pkl":
                format = "pickle"
            elif path.suffix == ".csv":
                format = "csv"
            else:
                format = "json"
        
        if format == "json":
            with open(path) as f:
                return json.load(f)
        
        elif format == "yaml":
            with open(path) as f:
                return yaml.safe_load(f)
        
        elif format == "pickle":
            with open(path, 'rb') as f:
                return pickle.load(f)
        
        elif format == "csv":
            return ResultSerializer._load_csv(path)
        
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def _load_csv(path: Path) -> List[Dict[str, Any]]:
        """Load CSV data"""
        with open(path, newline='') as f:
            reader = csv.DictReader(f)
            return list(reader)


class ExperimentRunner(ABC):
    """
    Base class for running experiments.
    Provides common infrastructure for experiment execution.
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config or ExperimentConfig()
        self.metrics = MetricsCollector()
        self.status = ExperimentStatus.PENDING
        self.results = {}
        self.checkpoint_path = None
        
        # Set up output directory
        self.output_dir = Path(self.config.output_dir) / self.config.name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        self._set_seed(self.config.seed)
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        import random
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    @abstractmethod
    def setup(self):
        """Set up experiment (load data, create models, etc.)"""
        pass
    
    @abstractmethod
    def run_iteration(self, iteration: int) -> Dict[str, float]:
        """
        Run a single iteration of the experiment.
        
        Args:
            iteration: Iteration number
            
        Returns:
            Dictionary of metrics for this iteration
        """
        pass
    
    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate experiment results.
        
        Returns:
            Evaluation results
        """
        pass
    
    def run(self, num_iterations: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the complete experiment.
        
        Args:
            num_iterations: Number of iterations to run
            
        Returns:
            Experiment results
        """
        try:
            # Setup
            self.status = ExperimentStatus.RUNNING
            self.metrics.start()
            
            if self.config.verbose:
                logger.info(f"Starting experiment: {self.config.name}")
            
            self.setup()
            
            # Load checkpoint if exists
            if self.config.checkpoint:
                self._load_checkpoint()
            
            # Run iterations
            iteration = 0
            start_time = time.time()
            
            while True:
                # Check stopping conditions
                if num_iterations and iteration >= num_iterations:
                    break
                
                elapsed = time.time() - start_time
                if elapsed > self.config.max_duration:
                    logger.warning(f"Experiment exceeded max duration ({self.config.max_duration}s)")
                    break
                
                # Run iteration
                metrics = self.run_iteration(iteration)
                self.metrics.log(metrics, step=iteration)
                
                # Save checkpoint
                if self.config.checkpoint and iteration % self.config.save_interval == 0:
                    self._save_checkpoint(iteration)
                
                # Log progress
                if self.config.verbose and iteration % 10 == 0:
                    logger.info(f"Iteration {iteration}: {metrics}")
                
                iteration += 1
            
            # Evaluate results
            self.results = self.evaluate()
            self.results['metrics'] = self.metrics.to_dict()
            self.results['config'] = self.config.to_dict()
            self.results['status'] = ExperimentStatus.COMPLETED.value
            
            # Save final results
            self._save_results()
            
            self.status = ExperimentStatus.COMPLETED
            
            if self.config.verbose:
                logger.info(f"Experiment completed: {self.config.name}")
            
            return self.results
            
        except Exception as e:
            self.status = ExperimentStatus.FAILED
            logger.error(f"Experiment failed: {e}")
            
            self.results['error'] = str(e)
            self.results['status'] = ExperimentStatus.FAILED.value
            self._save_results()
            
            raise
    
    def _save_checkpoint(self, iteration: int):
        """Save experiment checkpoint"""
        checkpoint = {
            'iteration': iteration,
            'metrics': self.metrics.to_dict(),
            'status': self.status.value,
            'config': self.config.to_dict()
        }
        
        # Allow subclasses to add custom checkpoint data
        checkpoint.update(self.get_checkpoint_data())
        
        path = self.output_dir / f"checkpoint_{iteration}.pkl"
        with open(path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        self.checkpoint_path = path
        
        # Keep only last few checkpoints
        self._cleanup_old_checkpoints()
    
    def _load_checkpoint(self):
        """Load experiment checkpoint"""
        checkpoints = sorted(self.output_dir.glob("checkpoint_*.pkl"))
        if not checkpoints:
            return
        
        latest = checkpoints[-1]
        with open(latest, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Restore state
        self.metrics = MetricsCollector()
        self.metrics.metrics = defaultdict(list, checkpoint['metrics']['metrics'])
        self.metrics.timestamps = defaultdict(list, checkpoint['metrics']['timestamps'])
        self.metrics.metadata = checkpoint['metrics'].get('metadata', {})
        
        # Allow subclasses to restore custom data
        self.load_checkpoint_data(checkpoint)
        
        logger.info(f"Loaded checkpoint from {latest}")
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """
        Get custom checkpoint data (override in subclasses).
        
        Returns:
            Dictionary of custom checkpoint data
        """
        return {}
    
    def load_checkpoint_data(self, checkpoint: Dict[str, Any]):
        """
        Load custom checkpoint data (override in subclasses).
        
        Args:
            checkpoint: Checkpoint dictionary
        """
        pass
    
    def _cleanup_old_checkpoints(self, keep: int = 3):
        """Keep only the last few checkpoints"""
        checkpoints = sorted(self.output_dir.glob("checkpoint_*.pkl"))
        if len(checkpoints) > keep:
            for checkpoint in checkpoints[:-keep]:
                checkpoint.unlink()
    
    def _save_results(self):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save in multiple formats
        base_path = self.output_dir / f"results_{timestamp}"
        
        # JSON for easy reading
        ResultSerializer.save(self.results, base_path.with_suffix(".json"))
        
        # Pickle for complete data
        ResultSerializer.save(self.results, base_path.with_suffix(".pkl"))
        
        # CSV for metrics
        if 'metrics' in self.results and self.results['metrics']['metrics']:
            ResultSerializer.save(
                self.results['metrics'],
                base_path.with_suffix(".csv"),
                format="csv"
            )
    
    def get_results(self) -> Dict[str, Any]:
        """Get current results"""
        return self.results


def run_experiment(experiment_class: type,
                  config: Optional[ExperimentConfig] = None,
                  **kwargs) -> Dict[str, Any]:
    """
    Convenience function to run an experiment.
    
    Args:
        experiment_class: ExperimentRunner subclass
        config: Experiment configuration
        **kwargs: Additional arguments for the experiment
        
    Returns:
        Experiment results
    """
    if config is None:
        config = ExperimentConfig(**kwargs)
    
    runner = experiment_class(config)
    return runner.run()


class SimpleExperiment(ExperimentRunner):
    """
    Simple experiment runner for basic experiments.
    Can be used directly without subclassing.
    """
    
    def __init__(self,
                 setup_fn: Callable,
                 iteration_fn: Callable,
                 evaluate_fn: Callable,
                 config: Optional[ExperimentConfig] = None):
        """
        Initialize simple experiment.
        
        Args:
            setup_fn: Setup function
            iteration_fn: Iteration function
            evaluate_fn: Evaluation function
            config: Experiment configuration
        """
        super().__init__(config)
        self.setup_fn = setup_fn
        self.iteration_fn = iteration_fn
        self.evaluate_fn = evaluate_fn
        self.state = {}
    
    def setup(self):
        """Set up experiment"""
        self.state = self.setup_fn()
    
    def run_iteration(self, iteration: int) -> Dict[str, float]:
        """Run iteration"""
        return self.iteration_fn(iteration, self.state)
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate results"""
        return self.evaluate_fn(self.state, self.metrics)


def create_experiment_id(name: str, params: Optional[Dict[str, Any]] = None) -> str:
    """
    Create a unique experiment ID.
    
    Args:
        name: Experiment name
        params: Experiment parameters
        
    Returns:
        Unique experiment ID
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if params:
        # Create hash of parameters
        param_str = json.dumps(params, sort_keys=True)
        param_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        return f"{name}_{timestamp}_{param_hash}"
    else:
        return f"{name}_{timestamp}"


def compare_experiments(exp_dir1: Path, exp_dir2: Path) -> Dict[str, Any]:
    """
    Compare results from two experiments.
    
    Args:
        exp_dir1: First experiment directory
        exp_dir2: Second experiment directory
        
    Returns:
        Comparison results
    """
    # Load results
    results1_files = list(Path(exp_dir1).glob("results_*.json"))
    results2_files = list(Path(exp_dir2).glob("results_*.json"))
    
    if not results1_files or not results2_files:
        raise ValueError("Could not find results files")
    
    results1 = ResultSerializer.load(results1_files[-1])
    results2 = ResultSerializer.load(results2_files[-1])
    
    comparison = {
        'experiment1': exp_dir1.name,
        'experiment2': exp_dir2.name,
        'metrics_comparison': {},
        'config_diff': {}
    }
    
    # Compare metrics
    if 'metrics' in results1 and 'metrics' in results2:
        metrics1 = results1['metrics'].get('summary', {})
        metrics2 = results2['metrics'].get('summary', {})
        
        all_metrics = set(metrics1.keys()) | set(metrics2.keys())
        
        for metric in all_metrics:
            if metric in metrics1 and metric in metrics2:
                val1 = metrics1[metric].get('mean', metrics1[metric])
                val2 = metrics2[metric].get('mean', metrics2[metric])
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff = val2 - val1
                    pct_change = (diff / val1 * 100) if val1 != 0 else float('inf')
                    
                    comparison['metrics_comparison'][metric] = {
                        'exp1': val1,
                        'exp2': val2,
                        'difference': diff,
                        'percent_change': pct_change
                    }
    
    # Compare configs
    if 'config' in results1 and 'config' in results2:
        config1 = results1['config']
        config2 = results2['config']
        
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)
            
            if val1 != val2:
                comparison['config_diff'][key] = {
                    'exp1': val1,
                    'exp2': val2
                }
    
    return comparison