"""
Unified experiment runner framework for PoT experiments.
Consolidates duplicate experiment running logic across the codebase.
"""

import sys
import json
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
from abc import ABC, abstractmethod

from pot.testing.base import BaseExperimentRunner, BaseMockModel, SimpleForwardModel, StatefulMockModel


class PoTExperimentRunner(BaseExperimentRunner):
    """
    Specialized experiment runner for Proof-of-Training experiments.
    Consolidates duplicate run_experiments() implementations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize PoT experiment runner."""
        super().__init__(config)
        self.pot_system = None
        self.models = {}
        
    def setup_experiment(self) -> None:
        """Set up PoT experiment environment."""
        from pot.security.proof_of_training import ProofOfTraining
        
        # Initialize PoT system
        pot_config = self.config.get('pot_config', {
            'verification_depth': 'standard',
            'enable_logging': False,
            'confidence_threshold': 0.95
        })
        self.pot_system = ProofOfTraining(pot_config)
        
        # Create test models
        self.models = {
            'simple': SimpleForwardModel(),
            'stateful': StatefulMockModel()
        }
        
    def run_single_experiment(self, 
                             experiment_type: str = "verification",
                             **kwargs) -> Dict[str, Any]:
        """
        Run a single PoT experiment.
        
        Args:
            experiment_type: Type of experiment to run
            **kwargs: Additional experiment parameters
            
        Returns:
            Experiment results
        """
        if experiment_type == "verification":
            return self._run_verification_experiment(**kwargs)
        elif experiment_type == "security":
            return self._run_security_experiment(**kwargs)
        elif experiment_type == "challenge":
            return self._run_challenge_experiment(**kwargs)
        else:
            return self._run_basic_experiment(**kwargs)
    
    def _run_verification_experiment(self, **kwargs) -> Dict[str, Any]:
        """Run verification type experiment."""
        model = self.models.get(kwargs.get('model_type', 'simple'))
        depth = kwargs.get('depth', 'standard')
        
        # Register model
        model_id = self.pot_system.register_model(
            model, 
            "test_architecture",
            1000
        )
        
        # Run verification
        start_time = time.time()
        result = self.pot_system.perform_verification(
            model,
            model_id,
            depth
        )
        duration = time.time() - start_time
        
        return {
            'type': 'verification',
            'depth': depth,
            'verified': result.verified,
            'confidence': result.confidence,
            'duration': duration,
            'model_type': kwargs.get('model_type', 'simple')
        }
    
    def _run_security_experiment(self, **kwargs) -> Dict[str, Any]:
        """Run security level experiment."""
        security_level = kwargs.get('security_level', 'medium')
        model = self.models['simple']
        
        # Configure security settings
        self.pot_system.config['security_level'] = security_level
        
        # Register and verify
        model_id = self.pot_system.register_model(model, "secure_arch", 1000)
        result = self.pot_system.perform_verification(model, model_id, 'comprehensive')
        
        return {
            'type': 'security',
            'security_level': security_level,
            'verified': result.verified,
            'confidence': result.confidence,
            'checks_performed': len(result.details) if hasattr(result, 'details') else 0
        }
    
    def _run_challenge_experiment(self, **kwargs) -> Dict[str, Any]:
        """Run challenge effectiveness experiment."""
        num_challenges = kwargs.get('num_challenges', 10)
        model = self.models['simple']
        
        successes = 0
        total_time = 0
        
        for _ in range(num_challenges):
            start = time.time()
            # Simulate challenge
            output = model.forward(np.random.randn(10))
            success = np.mean(output) < 1.0  # Simple success criterion
            total_time += time.time() - start
            
            if success:
                successes += 1
        
        return {
            'type': 'challenge',
            'num_challenges': num_challenges,
            'success_rate': successes / num_challenges,
            'avg_time': total_time / num_challenges
        }
    
    def _run_basic_experiment(self, **kwargs) -> Dict[str, Any]:
        """Run basic experiment."""
        model = self.models['simple']
        input_data = np.random.randn(100, 10)
        
        outputs = []
        for i in range(10):
            output = model.forward(input_data[i])
            outputs.append(output)
        
        outputs = np.array(outputs)
        
        return {
            'type': 'basic',
            'mean_output': float(np.mean(outputs)),
            'std_output': float(np.std(outputs)),
            'model_state': model.state_dict()
        }


class ValidationExperimentRunner(PoTExperimentRunner):
    """
    Experiment runner for validation experiments.
    Replaces duplicate implementations in validation_experiment.py files.
    """
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run comprehensive validation experiments."""
        results = {
            'timestamp': datetime.now().isoformat(),
            'experiments': []
        }
        
        # Experiment 1: Verification Types
        print("\n=== EXPERIMENT 1: Verification Types Comparison ===")
        exp1_results = self.experiment_verification_types()
        results['experiments'].append(exp1_results)
        
        # Experiment 2: Security Levels
        print("\n=== EXPERIMENT 2: Security Levels Analysis ===")
        exp2_results = self.experiment_security_levels()
        results['experiments'].append(exp2_results)
        
        # Experiment 3: Model Types
        print("\n=== EXPERIMENT 3: Model Type Coverage ===")
        exp3_results = self.experiment_model_types()
        results['experiments'].append(exp3_results)
        
        # Experiment 4: Challenge Effectiveness
        print("\n=== EXPERIMENT 4: Challenge Effectiveness ===")
        exp4_results = self.experiment_challenge_effectiveness()
        results['experiments'].append(exp4_results)
        
        return results
    
    def experiment_verification_types(self) -> Dict[str, Any]:
        """Compare different verification types."""
        depths = ['quick', 'standard', 'comprehensive']
        results = []
        
        for depth in depths:
            result = self.run_single_experiment(
                experiment_type='verification',
                depth=depth
            )
            results.append(result)
            print(f"  {depth}: verified={result['verified']}, "
                  f"confidence={result['confidence']:.2%}, "
                  f"time={result['duration']:.3f}s")
        
        return {
            'name': 'verification_types',
            'results': results
        }
    
    def experiment_security_levels(self) -> Dict[str, Any]:
        """Analyze different security levels."""
        levels = ['low', 'medium', 'high']
        results = []
        
        for level in levels:
            result = self.run_single_experiment(
                experiment_type='security',
                security_level=level
            )
            results.append(result)
            print(f"  {level}: verified={result['verified']}, "
                  f"confidence={result['confidence']:.2%}")
        
        return {
            'name': 'security_levels',
            'results': results
        }
    
    def experiment_model_types(self) -> Dict[str, Any]:
        """Test different model types."""
        model_types = ['simple', 'stateful']
        results = []
        
        for model_type in model_types:
            result = self.run_single_experiment(
                experiment_type='verification',
                model_type=model_type
            )
            results.append(result)
            print(f"  {model_type}: verified={result['verified']}, "
                  f"confidence={result['confidence']:.2%}")
        
        return {
            'name': 'model_types',
            'results': results
        }
    
    def experiment_challenge_effectiveness(self) -> Dict[str, Any]:
        """Test challenge effectiveness."""
        challenge_counts = [5, 10, 20]
        results = []
        
        for count in challenge_counts:
            result = self.run_single_experiment(
                experiment_type='challenge',
                num_challenges=count
            )
            results.append(result)
            print(f"  {count} challenges: success_rate={result['success_rate']:.2%}, "
                  f"avg_time={result['avg_time']:.6f}s")
        
        return {
            'name': 'challenge_effectiveness',
            'results': results
        }


def run_standard_experiments(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run standard set of experiments.
    Replacement for duplicate run_experiments() functions.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Experiment results
    """
    runner = ValidationExperimentRunner(config)
    runner.setup_experiment()
    results = runner.run_comprehensive_validation()
    
    # Save results
    output_dir = Path('experimental_results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'validation_experiment_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    # Run standard experiments when executed directly
    results = run_standard_experiments()
    
    # Print summary
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    
    for exp in results['experiments']:
        print(f"\n{exp['name']}:")
        if 'results' in exp:
            for result in exp['results']:
                if 'verified' in result:
                    print(f"  - verified: {result['verified']}, "
                          f"confidence: {result.get('confidence', 0):.2%}")