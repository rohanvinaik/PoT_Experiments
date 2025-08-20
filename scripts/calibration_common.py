"""
Common calibration utilities shared between calibrate_thresholds.py and test_calibrated_thresholds.py
This module breaks the circular dependency between the two scripts.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass, asdict


@dataclass
class CalibrationConfig:
    """Configuration for calibration process"""
    alpha: float = 0.025
    beta: float = 0.025
    max_iterations: int = 100
    convergence_threshold: float = 0.001
    num_validation_samples: int = 100
    output_file: str = "calibrated_thresholds.json"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationConfig':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class CalibrationResult:
    """Results from calibration process"""
    p0: float
    p1: float
    threshold_alpha: float
    threshold_beta: float
    iterations: int
    converged: bool
    validation_accuracy: float
    false_positive_rate: float
    false_negative_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationResult':
        """Create from dictionary"""
        return cls(**data)


def save_calibration_results(results: CalibrationResult, 
                            config: CalibrationConfig,
                            filepath: Optional[str] = None) -> None:
    """
    Save calibration results to JSON file.
    
    Args:
        results: Calibration results
        config: Calibration configuration
        filepath: Output file path (uses config default if None)
    """
    filepath = filepath or config.output_file
    
    output_data = {
        'config': config.to_dict(),
        'results': results.to_dict(),
        'timestamp': str(np.datetime64('now'))
    }
    
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Calibration results saved to {output_path}")


def load_calibration_results(filepath: str) -> Tuple[CalibrationResult, CalibrationConfig]:
    """
    Load calibration results from JSON file.
    
    Args:
        filepath: Path to calibration results file
        
    Returns:
        Tuple of (CalibrationResult, CalibrationConfig)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    config = CalibrationConfig.from_dict(data['config'])
    results = CalibrationResult.from_dict(data['results'])
    
    return results, config


def compute_decision_thresholds(p0: float, p1: float, 
                               alpha: float, beta: float) -> Tuple[float, float]:
    """
    Compute SPRT decision thresholds.
    
    Args:
        p0: Null hypothesis probability
        p1: Alternative hypothesis probability
        alpha: Type I error rate
        beta: Type II error rate
        
    Returns:
        Tuple of (lower_threshold, upper_threshold)
    """
    A = (1 - beta) / alpha
    B = beta / (1 - alpha)
    
    threshold_alpha = np.log(A)
    threshold_beta = np.log(B)
    
    return threshold_beta, threshold_alpha


def validate_calibration(results: CalibrationResult,
                        test_data: np.ndarray,
                        test_labels: np.ndarray) -> Dict[str, float]:
    """
    Validate calibration results on test data.
    
    Args:
        results: Calibration results
        test_data: Test data array
        test_labels: Test labels (0 or 1)
        
    Returns:
        Dictionary of validation metrics
    """
    # Simple validation logic - can be extended
    predictions = []
    
    for i, data_point in enumerate(test_data):
        # Simulate SPRT decision
        log_likelihood_ratio = np.random.randn()  # Placeholder - replace with actual
        
        if log_likelihood_ratio >= results.threshold_alpha:
            predictions.append(1)  # Accept H1
        elif log_likelihood_ratio <= results.threshold_beta:
            predictions.append(0)  # Accept H0
        else:
            predictions.append(-1)  # Undecided
    
    predictions = np.array(predictions)
    decided_mask = predictions != -1
    
    if np.sum(decided_mask) == 0:
        return {
            'accuracy': 0.0,
            'false_positive_rate': 0.0,
            'false_negative_rate': 0.0,
            'undecided_rate': 1.0
        }
    
    accuracy = np.mean(predictions[decided_mask] == test_labels[decided_mask])
    
    # Calculate error rates
    positive_mask = test_labels == 1
    negative_mask = test_labels == 0
    
    false_positives = np.sum((predictions == 1) & negative_mask)
    false_negatives = np.sum((predictions == 0) & positive_mask)
    
    fpr = false_positives / np.sum(negative_mask) if np.sum(negative_mask) > 0 else 0.0
    fnr = false_negatives / np.sum(positive_mask) if np.sum(positive_mask) > 0 else 0.0
    
    return {
        'accuracy': float(accuracy),
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr),
        'undecided_rate': float(np.mean(predictions == -1))
    }


def generate_synthetic_data(num_samples: int, 
                           p_genuine: float = 0.5,
                           seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for calibration.
    
    Args:
        num_samples: Number of samples to generate
        p_genuine: Probability of genuine samples
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (data, labels)
    """
    if seed is not None:
        np.random.seed(seed)
    
    labels = np.random.binomial(1, p_genuine, num_samples)
    
    # Generate data based on labels
    data = []
    for label in labels:
        if label == 1:
            # Genuine sample - higher values
            value = np.random.normal(0.7, 0.15)
        else:
            # Fake sample - lower values
            value = np.random.normal(0.3, 0.15)
        
        data.append(np.clip(value, 0, 1))
    
    return np.array(data), labels


def calibration_grid_search(data: np.ndarray,
                           labels: np.ndarray,
                           p0_range: Tuple[float, float] = (0.4, 0.6),
                           p1_range: Tuple[float, float] = (0.6, 0.8),
                           grid_size: int = 10) -> CalibrationResult:
    """
    Perform grid search for optimal calibration parameters.
    
    Args:
        data: Training data
        labels: Training labels
        p0_range: Range for p0 search
        p1_range: Range for p1 search
        grid_size: Number of grid points
        
    Returns:
        Best calibration result
    """
    best_result = None
    best_score = -float('inf')
    
    p0_values = np.linspace(p0_range[0], p0_range[1], grid_size)
    p1_values = np.linspace(p1_range[0], p1_range[1], grid_size)
    
    for p0 in p0_values:
        for p1 in p1_values:
            if p1 <= p0:
                continue
            
            # Compute thresholds
            threshold_beta, threshold_alpha = compute_decision_thresholds(
                p0, p1, 0.025, 0.025
            )
            
            # Create result
            result = CalibrationResult(
                p0=p0,
                p1=p1,
                threshold_alpha=threshold_alpha,
                threshold_beta=threshold_beta,
                iterations=1,
                converged=True,
                validation_accuracy=0.0,
                false_positive_rate=0.0,
                false_negative_rate=0.0
            )
            
            # Validate
            metrics = validate_calibration(result, data, labels)
            
            # Update result with metrics
            result.validation_accuracy = metrics['accuracy']
            result.false_positive_rate = metrics['false_positive_rate']
            result.false_negative_rate = metrics['false_negative_rate']
            
            # Score (maximize accuracy, minimize error rates)
            score = metrics['accuracy'] - metrics['false_positive_rate'] - metrics['false_negative_rate']
            
            if score > best_score:
                best_score = score
                best_result = result
    
    return best_result