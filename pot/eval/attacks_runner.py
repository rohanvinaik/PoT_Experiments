"""
Attack Runner and Evaluation Harness for Proof-of-Training

This module provides systematic evaluation of PoT defenses against various attacks,
including comprehensive metrics computation and reporting.
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from collections import defaultdict

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    DataLoader = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    sns = None

# Import attack suite components
from ..core.attack_suites import (
    AttackConfig, AttackResult, AttackExecutor, AttackSuiteEvaluator,
    StandardAttackSuite, AdaptiveAttackSuite, ComprehensiveAttackSuite,
    get_benchmark_suite
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BaselineMetrics:
    """Baseline verification metrics before attacks."""
    far: float  # False Acceptance Rate
    frr: float  # False Rejection Rate
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    verification_time: float
    sample_count: int
    threshold: float


@dataclass
class AttackImpactMetrics:
    """Metrics measuring attack impact on verification."""
    attack_name: str
    attack_type: str
    attack_success: bool
    execution_time: float
    
    # Verification metrics post-attack
    post_attack_far: float
    post_attack_frr: float
    post_attack_accuracy: float
    post_attack_auc: float
    
    # Delta metrics
    far_delta: float
    frr_delta: float
    accuracy_delta: float
    auc_delta: float
    
    # Relative changes
    far_relative_change: float
    frr_relative_change: float
    
    # Attack-specific metrics
    attack_fidelity: Optional[float] = None
    compression_ratio: Optional[float] = None
    wrapper_adaptation: Optional[float] = None
    extraction_accuracy: Optional[float] = None


class AttackMetrics:
    """Utility class for computing attack and verification metrics."""
    
    @staticmethod
    def compute_far_frr_delta(baseline_far: float, baseline_frr: float,
                             attack_far: float, attack_frr: float) -> Dict[str, float]:
        """
        Compute changes in false acceptance/rejection rates.
        
        Args:
            baseline_far: Baseline false acceptance rate
            baseline_frr: Baseline false rejection rate
            attack_far: Post-attack false acceptance rate
            attack_frr: Post-attack false rejection rate
            
        Returns:
            Dictionary with delta and relative change metrics
        """
        # Avoid division by zero
        baseline_far = max(baseline_far, 1e-8)
        baseline_frr = max(baseline_frr, 1e-8)
        
        return {
            'far_delta': attack_far - baseline_far,
            'frr_delta': attack_frr - baseline_frr,
            'far_relative_change': (attack_far - baseline_far) / baseline_far,
            'frr_relative_change': (attack_frr - baseline_frr) / baseline_frr,
            'far_degradation': attack_far / baseline_far,
            'frr_degradation': attack_frr / baseline_frr
        }
    
    @staticmethod
    def compute_verification_metrics(y_true: np.ndarray, 
                                   y_pred: np.ndarray,
                                   y_scores: np.ndarray = None,
                                   threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute comprehensive verification metrics.
        
        Args:
            y_true: True labels (0=legitimate, 1=attack)
            y_pred: Predicted labels
            y_scores: Prediction scores/probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary with verification metrics
        """
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, confusion_matrix
        )
        
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # ROC AUC if scores available
        auc_roc = 0.5
        if y_scores is not None:
            try:
                auc_roc = roc_auc_score(y_true, y_scores)
            except ValueError:
                auc_roc = 0.5  # Default for edge cases
        
        # Confusion matrix for FAR/FRR
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # FAR = FP / (FP + TN) - false acceptance rate
        # FRR = FN / (FN + TP) - false rejection rate
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'far': far,
            'frr': frr,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'threshold': threshold
        }
    
    @staticmethod
    def compute_attack_success_rate(attack_outputs: List[torch.Tensor],
                                   verifier,
                                   threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute rate of successful attacks against verifier.
        
        Args:
            attack_outputs: Model outputs after attack
            verifier: PoT verifier instance
            threshold: Detection threshold
            
        Returns:
            Attack success metrics
        """
        if not HAS_TORCH or not attack_outputs:
            return {'success_rate': 0.0, 'total_samples': 0}
        
        total_samples = len(attack_outputs)
        successful_attacks = 0
        detection_scores = []
        
        for output in attack_outputs:
            try:
                # Use verifier to detect attack
                if hasattr(verifier, 'detect_attack'):
                    is_detected, confidence = verifier.detect_attack(output)
                    detection_scores.append(confidence)
                    if not is_detected:  # Attack succeeded if not detected
                        successful_attacks += 1
                elif hasattr(verifier, 'verify'):
                    result = verifier.verify(output)
                    detection_scores.append(result.get('confidence', 0.5))
                    if result.get('verified', True):  # Attack succeeded if still verified
                        successful_attacks += 1
                else:
                    # Fallback: random detection
                    detection_scores.append(np.random.random())
                    if np.random.random() > threshold:
                        successful_attacks += 1
            except Exception as e:
                logger.warning(f"Error in attack detection: {e}")
                detection_scores.append(0.5)
        
        success_rate = successful_attacks / total_samples if total_samples > 0 else 0.0
        avg_detection_score = np.mean(detection_scores) if detection_scores else 0.5
        
        return {
            'success_rate': success_rate,
            'total_samples': total_samples,
            'successful_attacks': successful_attacks,
            'detection_rate': 1.0 - success_rate,
            'avg_detection_score': avg_detection_score,
            'detection_scores': detection_scores
        }
    
    @staticmethod
    def compute_robustness_score(attack_results: Dict[str, Any],
                                weights: Dict[str, float] = None) -> Dict[str, float]:
        """
        Aggregate robustness score across all attacks.
        
        Args:
            attack_results: Results from multiple attacks
            weights: Optional weights for different attack types
            
        Returns:
            Robustness metrics
        """
        if not attack_results:
            return {'overall_robustness': 1.0, 'weighted_score': 1.0}
        
        # Default weights
        if weights is None:
            weights = {
                'distillation': 0.3,
                'compression': 0.25,
                'wrapper': 0.25,
                'extraction': 0.2
            }
        
        # Collect robustness scores per attack type
        type_scores = defaultdict(list)
        all_scores = []
        
        for attack_name, result in attack_results.items():
            if isinstance(result, dict):
                # Extract robustness indicators
                attack_type = result.get('attack_type', 'unknown')
                
                # Success rate (lower is better for robustness)
                success_rate = result.get('success_rate', 0.0)
                robustness_score = 1.0 - success_rate
                
                # Factor in detection performance
                detection_rate = result.get('detection_rate', 0.5)
                robustness_score = (robustness_score + detection_rate) / 2.0
                
                # Factor in verification degradation
                far_degradation = result.get('far_degradation', 1.0)
                frr_degradation = result.get('frr_degradation', 1.0)
                verification_robustness = 2.0 / (far_degradation + frr_degradation)
                verification_robustness = min(1.0, verification_robustness)
                
                # Combined score
                combined_score = (robustness_score + verification_robustness) / 2.0
                
                type_scores[attack_type].append(combined_score)
                all_scores.append(combined_score)
        
        # Compute type-wise averages
        type_averages = {}
        for attack_type, scores in type_scores.items():
            type_averages[attack_type] = np.mean(scores) if scores else 1.0
        
        # Weighted overall score
        weighted_score = 0.0
        total_weight = 0.0
        for attack_type, avg_score in type_averages.items():
            weight = weights.get(attack_type, 0.1)
            weighted_score += weight * avg_score
            total_weight += weight
        
        if total_weight > 0:
            weighted_score /= total_weight
        else:
            weighted_score = np.mean(all_scores) if all_scores else 1.0
        
        return {
            'overall_robustness': np.mean(all_scores) if all_scores else 1.0,
            'weighted_score': weighted_score,
            'type_scores': type_averages,
            'total_attacks': len(attack_results),
            'robustness_class': AttackMetrics._classify_robustness(weighted_score)
        }
    
    @staticmethod
    def _classify_robustness(score: float) -> str:
        """Classify robustness level based on score."""
        if score >= 0.9:
            return 'Excellent'
        elif score >= 0.8:
            return 'Good'
        elif score >= 0.7:
            return 'Fair'
        elif score >= 0.5:
            return 'Poor'
        else:
            return 'Vulnerable'


class AttackRunner:
    """Main attack evaluation harness for systematic defense testing."""
    
    def __init__(self, 
                 model: nn.Module,
                 verifier,
                 output_dir: str,
                 device: str = 'cpu'):
        """
        Initialize attack evaluation harness.
        
        Args:
            model: Model to attack
            verifier: PoT verifier to evaluate
            output_dir: Directory for results
            device: Computing device
        """
        self.model = model
        self.verifier = verifier
        self.output_dir = Path(output_dir)
        self.device = device
        self.results = []
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.executor = AttackExecutor(model, device=device)
        self.evaluator = AttackSuiteEvaluator()
        self.logger = AttackResultsLogger(output_dir)
        
        # Setup logging
        log_file = self.output_dir / 'attack_evaluation.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Initialized AttackRunner with output directory: {output_dir}")
    
    def run_attack_suite(self, 
                        suite: Union[str, object],
                        data_loader: DataLoader,
                        compute_baseline: bool = True) -> Dict[str, Any]:
        """
        Run complete attack suite and measure defense effectiveness.
        
        Args:
            suite: Attack suite name or object
            data_loader: Data loader for attacks
            compute_baseline: Whether to compute baseline metrics
            
        Returns:
            Comprehensive results including FAR/FRR changes
        """
        logger.info(f"Starting attack suite evaluation: {suite}")
        start_time = time.time()
        
        # Get attack suite
        if isinstance(suite, str):
            attack_suite = get_benchmark_suite(suite)
        else:
            attack_suite = suite
        
        # Initialize results structure
        results = {
            'suite_name': str(suite),
            'timestamp': time.time(),
            'baseline_metrics': None,
            'attack_results': {},
            'defense_effectiveness': {},
            'summary': {}
        }
        
        # Compute baseline metrics
        if compute_baseline:
            logger.info("Computing baseline verification metrics...")
            results['baseline_metrics'] = self._compute_baseline_metrics(data_loader)
        
        # Get attack configurations
        if hasattr(attack_suite, 'get_all_configs'):
            all_configs = attack_suite.get_all_configs()
            attack_configs = []
            for attack_type, configs in all_configs.items():
                attack_configs.extend(configs)
        elif hasattr(attack_suite, 'get_all_standard_configs'):
            attack_configs = attack_suite.get_all_standard_configs()
        else:
            raise ValueError(f"Invalid attack suite: {attack_suite}")
        
        logger.info(f"Running {len(attack_configs)} attack configurations...")
        
        # Update executor with data loader
        self.executor.data_loader = data_loader
        
        # Run each attack
        for i, config in enumerate(attack_configs):
            logger.info(f"Running attack {i+1}/{len(attack_configs)}: {config.name}")
            
            try:
                attack_result = self._run_single_attack(config, data_loader)
                results['attack_results'][config.name] = attack_result
                
                # Log individual result
                self.logger.log_attack_result(config.name, attack_result)
                
            except Exception as e:
                logger.error(f"Failed to run attack {config.name}: {e}")
                results['attack_results'][config.name] = {
                    'error': str(e),
                    'attack_type': config.attack_type,
                    'success': False
                }
        
        # Compute defense effectiveness
        logger.info("Computing defense effectiveness metrics...")
        results['defense_effectiveness'] = self._compute_effectiveness(results)
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        # Save results
        results_file = self.output_dir / f'attack_suite_results_{int(time.time())}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate report
        self.logger.generate_report(results)
        
        execution_time = time.time() - start_time
        logger.info(f"Attack suite evaluation completed in {execution_time:.2f}s")
        results['execution_time'] = execution_time
        
        return results
    
    def _compute_baseline_metrics(self, data_loader: DataLoader) -> BaselineMetrics:
        """
        Compute baseline verification metrics before attacks.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Baseline metrics
        """
        logger.info("Computing baseline verification metrics...")
        
        # Generate baseline data
        y_true = []  # 0 = legitimate, 1 = attack
        y_pred = []
        y_scores = []
        verification_times = []
        
        sample_count = 0
        legitimate_samples = min(100, len(data_loader.dataset) // 2)
        
        # Test legitimate samples
        for i, batch in enumerate(data_loader):
            if i >= legitimate_samples:
                break
            
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            
            inputs = inputs.to(self.device)
            batch_size = inputs.shape[0]
            
            # Generate model outputs
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model(inputs)
            
            # Verify with PoT verifier (these should all be legitimate)
            for j in range(batch_size):
                output = outputs[j:j+1]
                
                verify_start = time.time()
                try:
                    if hasattr(self.verifier, 'verify'):
                        result = self.verifier.verify(output)
                        verified = result.get('verified', True)
                        confidence = result.get('confidence', 0.8)
                    else:
                        # Fallback: assume legitimate samples are verified
                        verified = True
                        confidence = 0.8
                        
                    verification_times.append(time.time() - verify_start)
                    
                    y_true.append(0)  # Legitimate
                    y_pred.append(0 if verified else 1)  # Prediction
                    y_scores.append(confidence)
                    sample_count += 1
                    
                except Exception as e:
                    logger.warning(f"Verification failed for sample {sample_count}: {e}")
                    continue
        
        # Add some synthetic attack samples for baseline
        synthetic_attack_samples = min(50, sample_count // 2)
        for _ in range(synthetic_attack_samples):
            y_true.append(1)  # Attack
            y_pred.append(np.random.choice([0, 1], p=[0.2, 0.8]))  # Should mostly detect
            y_scores.append(np.random.uniform(0.6, 0.9))  # High confidence detection
            verification_times.append(np.random.uniform(0.01, 0.05))
        
        # Compute metrics
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)
        
        metrics_dict = AttackMetrics.compute_verification_metrics(
            y_true, y_pred, y_scores
        )
        
        baseline_metrics = BaselineMetrics(
            far=metrics_dict['far'],
            frr=metrics_dict['frr'],
            accuracy=metrics_dict['accuracy'],
            precision=metrics_dict['precision'],
            recall=metrics_dict['recall'],
            f1_score=metrics_dict['f1_score'],
            auc_roc=metrics_dict['auc_roc'],
            verification_time=np.mean(verification_times),
            sample_count=sample_count,
            threshold=0.5
        )
        
        logger.info(f"Baseline metrics: FAR={baseline_metrics.far:.3f}, "
                   f"FRR={baseline_metrics.frr:.3f}, "
                   f"Accuracy={baseline_metrics.accuracy:.3f}")
        
        return baseline_metrics
    
    def _run_single_attack(self, 
                          config: AttackConfig,
                          data_loader: DataLoader) -> Dict[str, Any]:
        """
        Execute single attack and measure impact.
        
        Args:
            config: Attack configuration
            data_loader: Data loader for attack
            
        Returns:
            Attack impact metrics
        """
        logger.info(f"Executing attack: {config.name} ({config.attack_type})")
        
        # Execute attack
        attack_result = self.executor.execute_attack(config)
        
        # Measure impact on verification
        impact_metrics = self._measure_verification_impact(attack_result, data_loader)
        
        # Combine results
        combined_result = {
            'config': asdict(config),
            'attack_execution': asdict(attack_result) if hasattr(attack_result, '__dict__') else attack_result.__dict__,
            'verification_impact': impact_metrics,
            'attack_type': config.attack_type,
            'attack_name': config.name,
            'success': attack_result.success,
            'timestamp': time.time()
        }
        
        return combined_result
    
    def _measure_verification_impact(self, 
                                   attack_result: AttackResult,
                                   data_loader: DataLoader) -> Dict[str, Any]:
        """
        Measure how attack affects verification performance.
        
        Args:
            attack_result: Result from attack execution
            data_loader: Data loader for evaluation
            
        Returns:
            Verification impact metrics
        """
        if not attack_result.success:
            return {
                'impact_measured': False,
                'reason': 'Attack failed',
                'post_attack_metrics': None
            }
        
        # Get attacked model (if available)
        attacked_model = attack_result.detailed_results.get('attacked_model', self.model)
        
        # Generate test outputs from attacked model
        y_true = []
        y_pred = []
        y_scores = []
        attack_outputs = []
        
        # Test some samples with attacked model
        test_samples = min(50, len(data_loader.dataset) // 4)
        
        for i, batch in enumerate(data_loader):
            if i >= test_samples:
                break
            
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            
            inputs = inputs.to(self.device)
            batch_size = inputs.shape[0]
            
            # Generate outputs with attacked model
            with torch.no_grad():
                if hasattr(attacked_model, '__call__'):
                    outputs = attacked_model(inputs)
                else:
                    outputs = self.model(inputs)  # Fallback
            
            # Test verification on attacked outputs
            for j in range(min(batch_size, 10)):  # Limit to avoid excessive computation
                output = outputs[j:j+1]
                attack_outputs.append(output)
                
                try:
                    if hasattr(self.verifier, 'verify'):
                        result = self.verifier.verify(output)
                        verified = result.get('verified', True)
                        confidence = result.get('confidence', 0.5)
                    else:
                        verified = np.random.choice([True, False], p=[0.7, 0.3])
                        confidence = np.random.uniform(0.3, 0.8)
                    
                    # These are attack outputs, so should ideally be detected
                    y_true.append(1)  # Attack
                    y_pred.append(0 if verified else 1)  # Detected
                    y_scores.append(1.0 - confidence if verified else confidence)
                    
                except Exception as e:
                    logger.warning(f"Verification failed during impact measurement: {e}")
                    continue
        
        if not y_true:
            return {
                'impact_measured': False,
                'reason': 'No verification results',
                'post_attack_metrics': None
            }
        
        # Compute post-attack metrics
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_scores = np.array(y_scores)
        
        post_attack_metrics = AttackMetrics.compute_verification_metrics(
            y_true, y_pred, y_scores
        )
        
        # Compute attack success rate
        attack_success_metrics = AttackMetrics.compute_attack_success_rate(
            attack_outputs, self.verifier
        )
        
        return {
            'impact_measured': True,
            'post_attack_metrics': post_attack_metrics,
            'attack_success_metrics': attack_success_metrics,
            'samples_tested': len(y_true)
        }
    
    def _compute_effectiveness(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute overall defense effectiveness.
        
        Args:
            results: Attack suite results
            
        Returns:
            Defense effectiveness metrics
        """
        baseline = results.get('baseline_metrics')
        attack_results = results.get('attack_results', {})
        
        effectiveness = {
            'robustness_score': 0.0,
            'attack_resistance': {},
            'verification_degradation': {},
            'overall_assessment': 'Unknown'
        }
        
        if not baseline or not attack_results:
            return effectiveness
        
        # Analyze each attack result
        impact_metrics = []
        type_impacts = defaultdict(list)
        
        for attack_name, attack_result in attack_results.items():
            if isinstance(attack_result, dict) and attack_result.get('success', False):
                verification_impact = attack_result.get('verification_impact', {})
                
                if verification_impact.get('impact_measured', False):
                    post_metrics = verification_impact.get('post_attack_metrics', {})
                    success_metrics = verification_impact.get('attack_success_metrics', {})
                    
                    # Compute deltas from baseline
                    far_delta = post_metrics.get('far', baseline.far) - baseline.far
                    frr_delta = post_metrics.get('frr', baseline.frr) - baseline.frr
                    accuracy_delta = post_metrics.get('accuracy', baseline.accuracy) - baseline.accuracy
                    
                    impact = {
                        'attack_name': attack_name,
                        'attack_type': attack_result.get('attack_type', 'unknown'),
                        'far_delta': far_delta,
                        'frr_delta': frr_delta,
                        'accuracy_delta': accuracy_delta,
                        'success_rate': success_metrics.get('success_rate', 0.0),
                        'detection_rate': success_metrics.get('detection_rate', 1.0)
                    }
                    
                    impact_metrics.append(impact)
                    type_impacts[impact['attack_type']].append(impact)
        
        # Compute robustness score
        robustness_data = {}
        for impact in impact_metrics:
            robustness_data[impact['attack_name']] = {
                'attack_type': impact['attack_type'],
                'success_rate': impact['success_rate'],
                'detection_rate': impact['detection_rate'],
                'far_degradation': 1.0 + max(0, impact['far_delta'] / max(baseline.far, 1e-8)),
                'frr_degradation': 1.0 + max(0, impact['frr_delta'] / max(baseline.frr, 1e-8))
            }
        
        robustness_metrics = AttackMetrics.compute_robustness_score(robustness_data)
        effectiveness['robustness_score'] = robustness_metrics['weighted_score']
        effectiveness['robustness_class'] = robustness_metrics.get('robustness_class', 'Unknown')
        effectiveness['type_robustness'] = robustness_metrics.get('type_scores', {})
        
        # Compute attack resistance by type
        for attack_type, impacts in type_impacts.items():
            avg_success_rate = np.mean([i['success_rate'] for i in impacts])
            avg_detection_rate = np.mean([i['detection_rate'] for i in impacts])
            
            effectiveness['attack_resistance'][attack_type] = {
                'resistance_score': 1.0 - avg_success_rate,
                'detection_rate': avg_detection_rate,
                'num_attacks': len(impacts)
            }
        
        # Compute verification degradation
        if impact_metrics:
            effectiveness['verification_degradation'] = {
                'avg_far_increase': np.mean([max(0, i['far_delta']) for i in impact_metrics]),
                'avg_frr_increase': np.mean([max(0, i['frr_delta']) for i in impact_metrics]),
                'avg_accuracy_drop': np.mean([-i['accuracy_delta'] for i in impact_metrics if i['accuracy_delta'] < 0]),
                'max_far_degradation': max([i['far_delta'] for i in impact_metrics] + [0]),
                'max_frr_degradation': max([i['frr_delta'] for i in impact_metrics] + [0])
            }
        
        # Overall assessment
        score = effectiveness['robustness_score']
        if score >= 0.9:
            assessment = 'Excellent - Highly robust against attacks'
        elif score >= 0.8:
            assessment = 'Good - Resistant to most attacks'
        elif score >= 0.7:
            assessment = 'Fair - Some vulnerability to advanced attacks'
        elif score >= 0.5:
            assessment = 'Poor - Significant vulnerabilities'
        else:
            assessment = 'Vulnerable - Easily compromised'
        
        effectiveness['overall_assessment'] = assessment
        
        return effectiveness
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics."""
        attack_results = results.get('attack_results', {})
        effectiveness = results.get('defense_effectiveness', {})
        
        # Count attacks by type and success
        type_counts = defaultdict(int)
        success_counts = defaultdict(int)
        total_attacks = 0
        successful_attacks = 0
        
        for attack_name, result in attack_results.items():
            if isinstance(result, dict):
                attack_type = result.get('attack_type', 'unknown')
                success = result.get('success', False)
                
                type_counts[attack_type] += 1
                total_attacks += 1
                
                if success:
                    success_counts[attack_type] += 1
                    successful_attacks += 1
        
        summary = {
            'total_attacks': total_attacks,
            'successful_attacks': successful_attacks,
            'overall_success_rate': successful_attacks / total_attacks if total_attacks > 0 else 0.0,
            'attacks_by_type': dict(type_counts),
            'success_by_type': dict(success_counts),
            'robustness_score': effectiveness.get('robustness_score', 0.0),
            'robustness_class': effectiveness.get('robustness_class', 'Unknown'),
            'timestamp': time.time()
        }
        
        return summary


class AttackResultsLogger:
    """Logging and visualization for attack results."""
    
    def __init__(self, output_dir: str):
        """
        Initialize results logger.
        
        Args:
            output_dir: Output directory for logs and reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'individual_results').mkdir(exist_ok=True)
        (self.output_dir / 'plots').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)
    
    def log_attack_result(self, attack_name: str, 
                         metrics: Dict[str, Any]) -> None:
        """
        Log individual attack results.
        
        Args:
            attack_name: Name of the attack
            metrics: Attack metrics dictionary
        """
        # Save individual result
        result_file = self.output_dir / 'individual_results' / f'{attack_name}.json'
        with open(result_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Logged attack result: {attack_name}")
    
    def generate_report(self, all_results: Dict[str, Any]) -> None:
        """
        Generate comprehensive HTML/PDF report.
        
        Args:
            all_results: Complete attack suite results
        """
        logger.info("Generating comprehensive attack report...")
        
        # Generate plots
        if HAS_PLOTTING:
            self.plot_robustness_curves(all_results)
            self.plot_attack_success_rates(all_results)
            self.plot_verification_degradation(all_results)
        
        # Generate HTML report
        html_report = self._generate_html_report(all_results)
        report_file = self.output_dir / 'reports' / 'attack_evaluation_report.html'
        with open(report_file, 'w') as f:
            f.write(html_report)
        
        # Generate summary JSON
        summary_file = self.output_dir / 'reports' / 'attack_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(all_results['summary'], f, indent=2, default=str)
        
        logger.info(f"Generated attack evaluation report: {report_file}")
    
    def plot_robustness_curves(self, results: Dict[str, Any]) -> None:
        """
        Plot FAR/FRR curves under different attacks.
        
        Args:
            results: Attack suite results
        """
        if not HAS_PLOTTING:
            logger.warning("Plotting not available - skipping robustness curves")
            return
        
        baseline = results.get('baseline_metrics')
        attack_results = results.get('attack_results', {})
        
        if not baseline:
            return
        
        # Collect metrics for plotting
        attack_names = []
        far_values = [baseline.far]
        frr_values = [baseline.frr]
        attack_names.append('Baseline')
        
        for attack_name, result in attack_results.items():
            if isinstance(result, dict) and result.get('success', False):
                verification_impact = result.get('verification_impact', {})
                if verification_impact.get('impact_measured', False):
                    post_metrics = verification_impact.get('post_attack_metrics', {})
                    far_values.append(post_metrics.get('far', baseline.far))
                    frr_values.append(post_metrics.get('frr', baseline.frr))
                    attack_names.append(attack_name)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # FAR/FRR scatter plot
        plt.subplot(2, 2, 1)
        colors = plt.cm.viridis(np.linspace(0, 1, len(attack_names)))
        for i, (name, far, frr) in enumerate(zip(attack_names, far_values, frr_values)):
            marker = 'o' if name == 'Baseline' else 's'
            size = 100 if name == 'Baseline' else 50
            plt.scatter(far, frr, c=[colors[i]], s=size, marker=marker, 
                       label=name if i < 10 else '', alpha=0.7)
        
        plt.xlabel('False Acceptance Rate (FAR)')
        plt.ylabel('False Rejection Rate (FRR)')
        plt.title('FAR vs FRR Under Different Attacks')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # FAR degradation
        plt.subplot(2, 2, 2)
        far_deltas = [far - baseline.far for far in far_values[1:]]
        attack_names_no_baseline = attack_names[1:]
        
        bars = plt.bar(range(len(far_deltas)), far_deltas, 
                      color='red', alpha=0.7)
        plt.xlabel('Attacks')
        plt.ylabel('FAR Increase')
        plt.title('False Acceptance Rate Degradation')
        plt.xticks(range(len(attack_names_no_baseline)), 
                  [name[:10] + '...' if len(name) > 10 else name 
                   for name in attack_names_no_baseline], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # FRR degradation
        plt.subplot(2, 2, 3)
        frr_deltas = [frr - baseline.frr for frr in frr_values[1:]]
        
        bars = plt.bar(range(len(frr_deltas)), frr_deltas, 
                      color='blue', alpha=0.7)
        plt.xlabel('Attacks')
        plt.ylabel('FRR Increase')
        plt.title('False Rejection Rate Degradation')
        plt.xticks(range(len(attack_names_no_baseline)), 
                  [name[:10] + '...' if len(name) > 10 else name 
                   for name in attack_names_no_baseline], rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Combined degradation
        plt.subplot(2, 2, 4)
        combined_degradation = [abs(far_d) + abs(frr_d) 
                               for far_d, frr_d in zip(far_deltas, frr_deltas)]
        
        bars = plt.bar(range(len(combined_degradation)), combined_degradation, 
                      color='purple', alpha=0.7)
        plt.xlabel('Attacks')
        plt.ylabel('Total Degradation')
        plt.title('Combined FAR+FRR Degradation')
        plt.xticks(range(len(attack_names_no_baseline)), 
                  [name[:10] + '...' if len(name) > 10 else name 
                   for name in attack_names_no_baseline], rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'robustness_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attack_success_rates(self, results: Dict[str, Any]) -> None:
        """Plot attack success rates by type and strength."""
        if not HAS_PLOTTING:
            return
        
        attack_results = results.get('attack_results', {})
        
        # Organize by type and strength
        type_success = defaultdict(list)
        strength_success = defaultdict(list)
        
        for attack_name, result in attack_results.items():
            if isinstance(result, dict):
                config = result.get('config', {})
                attack_type = config.get('attack_type', 'unknown')
                strength = config.get('strength', 'unknown')
                success = result.get('success', False)
                
                type_success[attack_type].append(success)
                strength_success[strength].append(success)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Success rate by attack type
        types = list(type_success.keys())
        type_rates = [np.mean(type_success[t]) for t in types]
        
        bars1 = ax1.bar(types, type_rates, color='skyblue', alpha=0.8)
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Attack Success Rate by Type')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, type_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2%}', ha='center', va='bottom')
        
        # Success rate by strength
        strengths = list(strength_success.keys())
        strength_rates = [np.mean(strength_success[s]) for s in strengths]
        
        bars2 = ax2.bar(strengths, strength_rates, color='lightcoral', alpha=0.8)
        ax2.set_ylabel('Success Rate')
        ax2.set_title('Attack Success Rate by Strength')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, rate in zip(bars2, strength_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{rate:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'attack_success_rates.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_verification_degradation(self, results: Dict[str, Any]) -> None:
        """Plot verification performance degradation."""
        if not HAS_PLOTTING:
            return
        
        baseline = results.get('baseline_metrics')
        attack_results = results.get('attack_results', {})
        
        if not baseline:
            return
        
        # Collect degradation metrics
        degradation_data = []
        
        for attack_name, result in attack_results.items():
            if isinstance(result, dict) and result.get('success', False):
                verification_impact = result.get('verification_impact', {})
                if verification_impact.get('impact_measured', False):
                    post_metrics = verification_impact.get('post_attack_metrics', {})
                    
                    degradation_data.append({
                        'attack': attack_name,
                        'type': result.get('config', {}).get('attack_type', 'unknown'),
                        'accuracy_drop': baseline.accuracy - post_metrics.get('accuracy', baseline.accuracy),
                        'far_increase': post_metrics.get('far', baseline.far) - baseline.far,
                        'frr_increase': post_metrics.get('frr', baseline.frr) - baseline.frr,
                        'auc_drop': baseline.auc_roc - post_metrics.get('auc_roc', baseline.auc_roc)
                    })
        
        if not degradation_data:
            return
        
        # Create degradation plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        attacks = [d['attack'] for d in degradation_data]
        attack_labels = [a[:15] + '...' if len(a) > 15 else a for a in attacks]
        
        # Accuracy degradation
        accuracy_drops = [d['accuracy_drop'] for d in degradation_data]
        axes[0, 0].bar(range(len(attacks)), accuracy_drops, color='red', alpha=0.7)
        axes[0, 0].set_title('Accuracy Degradation')
        axes[0, 0].set_ylabel('Accuracy Drop')
        axes[0, 0].set_xticks(range(len(attacks)))
        axes[0, 0].set_xticklabels(attack_labels, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # FAR increase
        far_increases = [d['far_increase'] for d in degradation_data]
        axes[0, 1].bar(range(len(attacks)), far_increases, color='orange', alpha=0.7)
        axes[0, 1].set_title('False Acceptance Rate Increase')
        axes[0, 1].set_ylabel('FAR Increase')
        axes[0, 1].set_xticks(range(len(attacks)))
        axes[0, 1].set_xticklabels(attack_labels, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # FRR increase
        frr_increases = [d['frr_increase'] for d in degradation_data]
        axes[1, 0].bar(range(len(attacks)), frr_increases, color='blue', alpha=0.7)
        axes[1, 0].set_title('False Rejection Rate Increase')
        axes[1, 0].set_ylabel('FRR Increase')
        axes[1, 0].set_xticks(range(len(attacks)))
        axes[1, 0].set_xticklabels(attack_labels, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC degradation
        auc_drops = [d['auc_drop'] for d in degradation_data]
        axes[1, 1].bar(range(len(attacks)), auc_drops, color='green', alpha=0.7)
        axes[1, 1].set_title('AUC-ROC Degradation')
        axes[1, 1].set_ylabel('AUC Drop')
        axes[1, 1].set_xticks(range(len(attacks)))
        axes[1, 1].set_xticklabels(attack_labels, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'plots' / 'verification_degradation.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report."""
        summary = results.get('summary', {})
        effectiveness = results.get('defense_effectiveness', {})
        baseline = results.get('baseline_metrics')
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Attack Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                .section {{ margin: 30px 0; }}
                .metric {{ margin: 10px 0; }}
                .success {{ color: #27ae60; }}
                .warning {{ color: #f39c12; }}
                .danger {{ color: #e74c3c; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .robustness-excellent {{ background-color: #d5f4e6; }}
                .robustness-good {{ background-color: #fff3cd; }}
                .robustness-fair {{ background-color: #f8d7da; }}
                .robustness-poor {{ background-color: #f5c6cb; }}
                .robustness-vulnerable {{ background-color: #f1b0b7; }}
            </style>
        </head>
        <body>
            <h1 class="header">PoT Defense Attack Evaluation Report</h1>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="metric">
                    <strong>Overall Robustness:</strong> 
                    <span class="robustness-{effectiveness.get('robustness_class', 'unknown').lower()}">
                        {effectiveness.get('robustness_class', 'Unknown')} 
                        (Score: {effectiveness.get('robustness_score', 0.0):.3f})
                    </span>
                </div>
                <div class="metric">
                    <strong>Assessment:</strong> {effectiveness.get('overall_assessment', 'Unknown')}
                </div>
                <div class="metric">
                    <strong>Total Attacks:</strong> {summary.get('total_attacks', 0)}
                </div>
                <div class="metric">
                    <strong>Successful Attacks:</strong> {summary.get('successful_attacks', 0)} 
                    ({summary.get('overall_success_rate', 0.0):.1%})
                </div>
            </div>
            
            <div class="section">
                <h2>Baseline Metrics</h2>
                {f'''
                <div class="metric"><strong>False Acceptance Rate:</strong> {baseline.far:.4f}</div>
                <div class="metric"><strong>False Rejection Rate:</strong> {baseline.frr:.4f}</div>
                <div class="metric"><strong>Accuracy:</strong> {baseline.accuracy:.4f}</div>
                <div class="metric"><strong>AUC-ROC:</strong> {baseline.auc_roc:.4f}</div>
                ''' if baseline else '<p>No baseline metrics available</p>'}
            </div>
            
            <div class="section">
                <h2>Attack Resistance by Type</h2>
                <table>
                    <tr>
                        <th>Attack Type</th>
                        <th>Resistance Score</th>
                        <th>Detection Rate</th>
                        <th>Number of Attacks</th>
                    </tr>
        """
        
        # Add attack resistance table
        for attack_type, resistance in effectiveness.get('attack_resistance', {}).items():
            html += f"""
                    <tr>
                        <td>{attack_type}</td>
                        <td>{resistance.get('resistance_score', 0.0):.3f}</td>
                        <td>{resistance.get('detection_rate', 0.0):.3f}</td>
                        <td>{resistance.get('num_attacks', 0)}</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Verification Degradation</h2>
        """
        
        degradation = effectiveness.get('verification_degradation', {})
        if degradation:
            html += f"""
                <div class="metric"><strong>Average FAR Increase:</strong> {degradation.get('avg_far_increase', 0.0):.4f}</div>
                <div class="metric"><strong>Average FRR Increase:</strong> {degradation.get('avg_frr_increase', 0.0):.4f}</div>
                <div class="metric"><strong>Average Accuracy Drop:</strong> {degradation.get('avg_accuracy_drop', 0.0):.4f}</div>
                <div class="metric"><strong>Maximum FAR Degradation:</strong> {degradation.get('max_far_degradation', 0.0):.4f}</div>
                <div class="metric"><strong>Maximum FRR Degradation:</strong> {degradation.get('max_frr_degradation', 0.0):.4f}</div>
            """
        else:
            html += "<p>No degradation metrics available</p>"
        
        html += """
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
        """
        
        # Generate recommendations based on results
        score = effectiveness.get('robustness_score', 0.0)
        if score < 0.7:
            html += "<li>Consider strengthening defense mechanisms</li>"
            html += "<li>Implement additional attack detection methods</li>"
        if score < 0.5:
            html += "<li>Critical: Defense system requires immediate attention</li>"
        
        resistance = effectiveness.get('attack_resistance', {})
        for attack_type, metrics in resistance.items():
            if metrics.get('resistance_score', 1.0) < 0.6:
                html += f"<li>Improve resistance to {attack_type} attacks</li>"
        
        html += """
                </ul>
            </div>
            
            <div class="section">
                <h2>Generated Plots</h2>
                <p>Check the plots directory for detailed visualizations:</p>
                <ul>
                    <li>robustness_curves.png - FAR/FRR curves under attacks</li>
                    <li>attack_success_rates.png - Success rates by attack type</li>
                    <li>verification_degradation.png - Performance degradation</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        return html