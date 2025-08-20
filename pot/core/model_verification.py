"""
Model verification utilities for comparing model states.
"""

import numpy as np
from typing import Dict, Any, Union


def verify_model_weights(
    model_before: Dict[str, np.ndarray],
    model_after: Dict[str, np.ndarray],
    threshold: float = 0.01
) -> Dict[str, Any]:
    """
    Verify if two model states are identical within a threshold.
    
    Args:
        model_before: Dictionary of model weights before update
        model_after: Dictionary of model weights after update
        threshold: Maximum allowed difference for weights to be considered identical
        
    Returns:
        Dictionary with verification results
    """
    result = {
        'identical': True,
        'max_difference': 0.0,
        'mean_difference': 0.0,
        'differences_by_layer': {}
    }
    
    # Check if keys match
    if set(model_before.keys()) != set(model_after.keys()):
        result['identical'] = False
        result['error'] = 'Model architectures do not match'
        return result
    
    all_differences = []
    
    for key in model_before.keys():
        before = model_before[key]
        after = model_after[key]
        
        # Check shapes match
        if before.shape != after.shape:
            result['identical'] = False
            result['error'] = f'Shape mismatch in layer {key}'
            return result
        
        # Calculate differences
        diff = np.abs(after - before)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        result['differences_by_layer'][key] = {
            'max': float(max_diff),
            'mean': float(mean_diff)
        }
        
        all_differences.extend(diff.flatten())
        
        # Check if difference exceeds threshold
        if max_diff > threshold:
            result['identical'] = False
    
    # Calculate overall statistics
    if all_differences:
        result['max_difference'] = float(np.max(all_differences))
        result['mean_difference'] = float(np.mean(all_differences))
    
    return result


def compute_model_distance(
    model1: Dict[str, np.ndarray],
    model2: Dict[str, np.ndarray],
    metric: str = 'l2'
) -> float:
    """
    Compute distance between two models.
    
    Args:
        model1: First model weights
        model2: Second model weights
        metric: Distance metric ('l1', 'l2', 'cosine')
        
    Returns:
        Distance value
    """
    if set(model1.keys()) != set(model2.keys()):
        raise ValueError("Model architectures do not match")
    
    if metric == 'l2':
        total_distance = 0.0
        for key in model1.keys():
            diff = model1[key] - model2[key]
            total_distance += np.sum(diff ** 2)
        return np.sqrt(total_distance)
    
    elif metric == 'l1':
        total_distance = 0.0
        for key in model1.keys():
            diff = model1[key] - model2[key]
            total_distance += np.sum(np.abs(diff))
        return total_distance
    
    elif metric == 'cosine':
        # Flatten all weights
        flat1 = np.concatenate([w.flatten() for w in model1.values()])
        flat2 = np.concatenate([w.flatten() for w in model2.values()])
        
        # Compute cosine similarity
        dot_product = np.dot(flat1, flat2)
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)
        
        if norm1 == 0 or norm2 == 0:
            return 1.0  # Maximum distance
        
        cosine_sim = dot_product / (norm1 * norm2)
        return 1.0 - cosine_sim  # Convert to distance
    
    else:
        raise ValueError(f"Unknown metric: {metric}")