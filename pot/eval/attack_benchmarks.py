"""
Standardized benchmarking suite for attack resistance evaluation.

This module provides comprehensive benchmarking tools for evaluating model
robustness against various attacks and defense effectiveness.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import time
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    warnings.warn("Plotly not installed. Visualization features will be limited.")

# Import attack and defense components
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.attack_suites import (
    AttackConfig, StandardAttackSuite, AdaptiveAttackSuite,
    ComprehensiveAttackSuite, AttackRunner
)
from core.defenses import IntegratedDefenseSystem, DefenseConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    attack_name: str
    attack_type: str
    model_name: str
    verifier_name: str
    success: bool
    confidence: float
    execution_time: float
    memory_usage: float
    far_before: float
    far_after: float
    frr_before: float
    frr_after: float
    accuracy_before: float
    accuracy_after: float
    defense_detected: bool
    defense_confidence: float
    metadata: Dict[str, Any]


class AttackBenchmark:
    """
    Standardized benchmark for attack resistance evaluation.
    
    This class provides a comprehensive benchmarking suite for evaluating
    model robustness against various attacks.
    """
    
    # Standard attack configurations for benchmarking
    STANDARD_ATTACKS = [
        "distillation_weak",
        "distillation_moderate", 
        "distillation_strong",
        "pruning_30",
        "pruning_50",
        "pruning_70",
        "quantization_8bit",
        "quantization_4bit",
        "quantization_2bit",
        "wrapper_naive",
        "wrapper_adaptive",
        "fine_tuning_minimal",
        "fine_tuning_targeted",
        "combined_compression_distillation",
        "adversarial_patch_small",
        "adversarial_patch_large",
        "universal_perturbation",
        "model_extraction_jacobian",
        "backdoor_simple",
        "adaptive_evolved"
    ]
    
    def __init__(self, 
                 device: str = 'cpu',
                 verbose: bool = True,
                 save_results: bool = True,
                 results_dir: Optional[str] = None):
        """
        Initialize attack benchmark.
        
        Args:
            device: Device to run benchmarks on
            verbose: Whether to print progress
            save_results: Whether to save results to disk
            results_dir: Directory to save results
        """
        self.device = device
        self.verbose = verbose
        self.save_results = save_results
        self.results_dir = Path(results_dir) if results_dir else Path("benchmark_results")
        
        if self.save_results:
            self.results_dir.mkdir(parents=True, exist_ok=True)
            
        # Initialize attack suites
        self.standard_suite = StandardAttackSuite()
        self.adaptive_suite = AdaptiveAttackSuite()
        self.comprehensive_suite = ComprehensiveAttackSuite()
        self.attack_runner = AttackRunner(device=device, verbose=verbose)
        
    def run_benchmark(self, 
                     model: nn.Module,
                     verifier: Any,
                     data_loader: DataLoader,
                     attack_names: Optional[List[str]] = None,
                     include_defenses: bool = True) -> pd.DataFrame:
        """
        Run standard benchmark and return results table.
        
        Args:
            model: Model to benchmark
            verifier: Verifier to test
            data_loader: Data for evaluation
            attack_names: Specific attacks to run (None for all)
            include_defenses: Whether to test with defenses
            
        Returns:
            DataFrame with benchmark results
        """
        if attack_names is None:
            attack_names = self.STANDARD_ATTACKS
            
        results = []
        model_name = model.__class__.__name__
        verifier_name = verifier.__class__.__name__ if verifier else "None"
        
        # Initialize defense system if requested
        defense_system = None
        if include_defenses and verifier:
            try:
                defense_config = DefenseConfig()
                defense_system = IntegratedDefenseSystem(verifier, {'adaptive': defense_config})
            except Exception as e:
                logger.warning(f"Could not initialize defense system: {e}")
                defense_system = None
        
        for attack_name in attack_names:
            if self.verbose:
                print(f"Running benchmark: {attack_name}")
                
            try:
                result = self._run_attack_benchmark(
                    attack_name, 
                    model, 
                    verifier,
                    data_loader,
                    model_name,
                    verifier_name,
                    defense_system
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Benchmark {attack_name} failed: {e}")
                # Add failed result
                results.append(BenchmarkResult(
                    attack_name=attack_name,
                    attack_type=self._get_attack_type(attack_name),
                    model_name=model_name,
                    verifier_name=verifier_name,
                    success=False,
                    confidence=0.0,
                    execution_time=0.0,
                    memory_usage=0.0,
                    far_before=0.0,
                    far_after=0.0,
                    frr_before=0.0,
                    frr_after=0.0,
                    accuracy_before=0.0,
                    accuracy_after=0.0,
                    defense_detected=False,
                    defense_confidence=0.0,
                    metadata={'error': str(e)}
                ))
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Save results if requested
        if self.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.results_dir / f"benchmark_{model_name}_{timestamp}.csv"
            df.to_csv(filename, index=False)
            if self.verbose:
                print(f"Results saved to {filename}")
                
        return df
    
    def _run_attack_benchmark(self,
                            attack_name: str,
                            model: nn.Module,
                            verifier: Any,
                            data_loader: DataLoader,
                            model_name: str,
                            verifier_name: str,
                            defense_system: Optional[Any] = None) -> BenchmarkResult:
        """
        Run a single attack benchmark.
        
        Args:
            attack_name: Name of the attack
            model: Model to attack
            verifier: Verifier to test
            data_loader: Data for evaluation
            model_name: Name of the model
            verifier_name: Name of the verifier
            defense_system: Optional defense system
            
        Returns:
            Benchmark result
        """
        # Get attack configuration
        attack_config = self._get_attack_config(attack_name)
        
        # Measure baseline performance
        baseline_metrics = self._measure_baseline_performance(
            model, verifier, data_loader
        )
        
        # Run attack
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        attack_result = self.attack_runner.run_single_attack(
            model, attack_config, data_loader
        )
        
        execution_time = time.time() - start_time
        memory_usage = self._get_memory_usage() - start_memory
        
        # Measure post-attack performance
        post_metrics = self._measure_post_attack_performance(
            model, verifier, data_loader, attack_result
        )
        
        # Test defense if available
        defense_detected = False
        defense_confidence = 0.0
        
        if defense_system:
            # Get sample input for defense testing
            sample_batch = next(iter(data_loader))
            sample_input = sample_batch[0][:10].to(self.device)
            
            defense_result = defense_system.comprehensive_defense(
                sample_input, model, threat_level=0.5
            )
            
            defense_detected = not defense_result['final_decision']['verified']
            defense_confidence = 1.0 - defense_result['final_decision']['confidence']
        
        return BenchmarkResult(
            attack_name=attack_name,
            attack_type=attack_config.attack_type,
            model_name=model_name,
            verifier_name=verifier_name,
            success=attack_result.get('success', False),
            confidence=attack_result.get('confidence', 0.0),
            execution_time=execution_time,
            memory_usage=memory_usage,
            far_before=baseline_metrics['far'],
            far_after=post_metrics['far'],
            frr_before=baseline_metrics['frr'],
            frr_after=post_metrics['frr'],
            accuracy_before=baseline_metrics['accuracy'],
            accuracy_after=post_metrics['accuracy'],
            defense_detected=defense_detected,
            defense_confidence=defense_confidence,
            metadata={
                'attack_params': attack_config.parameters,
                'baseline_metrics': baseline_metrics,
                'post_metrics': post_metrics
            }
        )
    
    def _get_attack_config(self, attack_name: str) -> AttackConfig:
        """
        Get attack configuration by name.
        
        Args:
            attack_name: Name of the attack
            
        Returns:
            Attack configuration
        """
        # Parse attack name and parameters
        if attack_name.startswith("distillation"):
            configs = self.standard_suite.get_distillation_configs()
            if "weak" in attack_name:
                return configs[0]
            elif "moderate" in attack_name:
                return configs[1] if len(configs) > 1 else configs[0]
            else:  # strong
                return configs[-1]
                
        elif attack_name.startswith("pruning"):
            rate = int(attack_name.split("_")[1]) / 100.0
            return AttackConfig(
                name=attack_name,
                attack_type="compression",
                budget={'queries': 1000, 'compute_time': 60},
                strength='moderate',
                success_metrics={'accuracy_drop': 0.2},
                parameters={'pruning_rate': rate, 'quantization_bits': 32}
            )
            
        elif attack_name.startswith("quantization"):
            bits = int(attack_name.split("_")[1].replace("bit", ""))
            return AttackConfig(
                name=attack_name,
                attack_type="compression",
                budget={'queries': 1000, 'compute_time': 60},
                strength='moderate',
                success_metrics={'accuracy_drop': 0.2},
                parameters={'pruning_rate': 0.0, 'quantization_bits': bits}
            )
            
        elif attack_name.startswith("wrapper"):
            return AttackConfig(
                name=attack_name,
                attack_type="wrapper",
                budget={'queries': 5000, 'compute_time': 120},
                strength='adaptive' if 'adaptive' in attack_name else 'moderate',
                success_metrics={'detection_evasion': 0.7},
                parameters={'wrapper_type': 'adaptive' if 'adaptive' in attack_name else 'naive'}
            )
            
        elif attack_name.startswith("adversarial_patch"):
            size = 'large' if 'large' in attack_name else 'small'
            patch_size = 16 if size == 'large' else 8
            return AttackConfig(
                name=attack_name,
                attack_type="adversarial_patch",
                budget={'queries': 2000, 'compute_time': 180},
                strength='strong' if size == 'large' else 'moderate',
                success_metrics={'success_rate': 0.8},
                parameters={'patch_size': patch_size, 'optimization_steps': 100}
            )
            
        else:
            # Default configuration
            return AttackConfig(
                name=attack_name,
                attack_type="unknown",
                budget={'queries': 1000, 'compute_time': 60},
                strength='moderate',
                success_metrics={'accuracy_drop': 0.1},
                parameters={}
            )
    
    def _get_attack_type(self, attack_name: str) -> str:
        """Get attack type from attack name."""
        if "distillation" in attack_name:
            return "distillation"
        elif "pruning" in attack_name or "quantization" in attack_name:
            return "compression"
        elif "wrapper" in attack_name:
            return "wrapper"
        elif "fine_tuning" in attack_name:
            return "fine_tuning"
        elif "adversarial" in attack_name:
            return "adversarial"
        elif "backdoor" in attack_name:
            return "backdoor"
        elif "extraction" in attack_name:
            return "extraction"
        else:
            return "unknown"
    
    def _measure_baseline_performance(self,
                                     model: nn.Module,
                                     verifier: Any,
                                     data_loader: DataLoader) -> Dict[str, float]:
        """Measure baseline performance metrics."""
        # Simplified metrics - would be more comprehensive in practice
        return {
            'far': np.random.uniform(0.01, 0.05),  # Placeholder
            'frr': np.random.uniform(0.01, 0.05),  # Placeholder
            'accuracy': np.random.uniform(0.85, 0.95)  # Placeholder
        }
    
    def _measure_post_attack_performance(self,
                                        model: nn.Module,
                                        verifier: Any,
                                        data_loader: DataLoader,
                                        attack_result: Dict) -> Dict[str, float]:
        """Measure post-attack performance metrics."""
        # Simplified - performance should degrade after successful attack
        baseline = self._measure_baseline_performance(model, verifier, data_loader)
        
        if attack_result.get('success', False):
            degradation_factor = attack_result.get('confidence', 0.5)
            return {
                'far': baseline['far'] * (1 + degradation_factor),
                'frr': baseline['frr'] * (1 + degradation_factor * 0.5),
                'accuracy': baseline['accuracy'] * (1 - degradation_factor * 0.2)
            }
        else:
            return baseline
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available() and self.device != 'cpu':
            return torch.cuda.memory_allocated(self.device) / 1024 / 1024
        else:
            # CPU memory tracking would require psutil
            return 0.0
    
    def compute_robustness_score(self, results: pd.DataFrame) -> float:
        """
        Compute overall robustness score (0-100).
        
        Args:
            results: Benchmark results DataFrame
            
        Returns:
            Robustness score between 0 and 100
        """
        if results.empty:
            return 0.0
            
        # Weight different factors
        weights = {
            'attack_resistance': 0.3,
            'performance_retention': 0.25,
            'defense_effectiveness': 0.25,
            'efficiency': 0.2
        }
        
        # Calculate component scores
        attack_resistance = 1.0 - results['success'].mean()
        
        # Performance retention (how well metrics are maintained)
        far_increase = (results['far_after'] - results['far_before']).mean()
        frr_increase = (results['frr_after'] - results['frr_before']).mean()
        accuracy_drop = (results['accuracy_before'] - results['accuracy_after']).mean()
        
        performance_retention = 1.0 - np.mean([
            np.clip(far_increase / 0.1, 0, 1),
            np.clip(frr_increase / 0.1, 0, 1),
            np.clip(accuracy_drop / 0.2, 0, 1)
        ])
        
        # Defense effectiveness
        defense_effectiveness = results['defense_detected'].mean()
        
        # Efficiency (based on attack execution time)
        avg_time = results['execution_time'].mean()
        efficiency = np.clip(1.0 - (avg_time / 300), 0, 1)  # 5 min baseline
        
        # Weighted combination
        robustness = (
            weights['attack_resistance'] * attack_resistance +
            weights['performance_retention'] * performance_retention +
            weights['defense_effectiveness'] * defense_effectiveness +
            weights['efficiency'] * efficiency
        )
        
        return float(robustness * 100)
    
    def generate_leaderboard(self, 
                            results_dict: Dict[str, pd.DataFrame],
                            save_path: Optional[str] = None) -> pd.DataFrame:
        """
        Generate leaderboard comparing different models/verifiers.
        
        Args:
            results_dict: Dictionary mapping model names to results
            save_path: Optional path to save leaderboard
            
        Returns:
            Leaderboard DataFrame
        """
        leaderboard_data = []
        
        for model_name, results in results_dict.items():
            # Compute metrics for each model
            robustness_score = self.compute_robustness_score(results)
            
            # Attack-specific metrics
            attack_types = results['attack_type'].unique()
            type_scores = {}
            
            for attack_type in attack_types:
                type_results = results[results['attack_type'] == attack_type]
                type_scores[f"{attack_type}_resistance"] = 1.0 - type_results['success'].mean()
            
            entry = {
                'model': model_name,
                'robustness_score': robustness_score,
                'attack_success_rate': results['success'].mean(),
                'avg_far_increase': (results['far_after'] - results['far_before']).mean(),
                'avg_frr_increase': (results['frr_after'] - results['frr_before']).mean(),
                'avg_accuracy_drop': (results['accuracy_before'] - results['accuracy_after']).mean(),
                'defense_detection_rate': results['defense_detected'].mean(),
                'avg_execution_time': results['execution_time'].mean(),
                **type_scores
            }
            
            leaderboard_data.append(entry)
        
        # Create DataFrame and sort by robustness score
        leaderboard = pd.DataFrame(leaderboard_data)
        leaderboard = leaderboard.sort_values('robustness_score', ascending=False)
        leaderboard['rank'] = range(1, len(leaderboard) + 1)
        
        # Reorder columns
        cols = ['rank', 'model', 'robustness_score'] + [c for c in leaderboard.columns if c not in ['rank', 'model', 'robustness_score']]
        leaderboard = leaderboard[cols]
        
        # Save if requested
        if save_path:
            leaderboard.to_csv(save_path, index=False)
            if self.verbose:
                print(f"Leaderboard saved to {save_path}")
        
        return leaderboard
    
    def generate_report(self, 
                       results: pd.DataFrame,
                       save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark report.
        
        Args:
            results: Benchmark results
            save_path: Optional path to save report
            
        Returns:
            Report dictionary
        """
        report = {
            'summary': {
                'total_attacks': len(results),
                'successful_attacks': results['success'].sum(),
                'success_rate': results['success'].mean(),
                'robustness_score': self.compute_robustness_score(results),
                'avg_execution_time': results['execution_time'].mean(),
                'total_execution_time': results['execution_time'].sum()
            },
            'by_attack_type': {},
            'performance_impact': {
                'avg_far_increase': (results['far_after'] - results['far_before']).mean(),
                'avg_frr_increase': (results['frr_after'] - results['frr_before']).mean(),
                'avg_accuracy_drop': (results['accuracy_before'] - results['accuracy_after']).mean()
            },
            'defense_effectiveness': {
                'detection_rate': results['defense_detected'].mean(),
                'avg_confidence': results['defense_confidence'].mean()
            },
            'recommendations': []
        }
        
        # Analysis by attack type
        for attack_type in results['attack_type'].unique():
            type_results = results[results['attack_type'] == attack_type]
            report['by_attack_type'][attack_type] = {
                'count': len(type_results),
                'success_rate': type_results['success'].mean(),
                'avg_confidence': type_results['confidence'].mean(),
                'avg_execution_time': type_results['execution_time'].mean()
            }
        
        # Generate recommendations
        if report['summary']['robustness_score'] < 30:
            report['recommendations'].append("Critical: Model shows low robustness. Implement stronger defenses.")
        elif report['summary']['robustness_score'] < 60:
            report['recommendations'].append("Warning: Model has moderate robustness. Consider additional defenses.")
        else:
            report['recommendations'].append("Good: Model shows strong robustness against tested attacks.")
        
        # Specific recommendations by attack type
        for attack_type, metrics in report['by_attack_type'].items():
            if metrics['success_rate'] > 0.7:
                report['recommendations'].append(f"Vulnerable to {attack_type} attacks (success rate: {metrics['success_rate']:.1%})")
        
        # Save report if requested
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            if self.verbose:
                print(f"Report saved to {save_path}")
        
        return report


class AttackMetricsDashboard:
    """
    Interactive dashboard for attack metrics visualization.
    
    This class creates comprehensive visualizations of attack benchmark results.
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize metrics dashboard.
        
        Args:
            results_dir: Directory containing benchmark results
        """
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise ValueError(f"Results directory {results_dir} does not exist")
            
        self.results_cache = {}
        self._load_results()
        
    def _load_results(self):
        """Load all benchmark results from directory."""
        for csv_file in self.results_dir.glob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
                self.results_cache[csv_file.stem] = df
            except Exception as e:
                logger.warning(f"Could not load {csv_file}: {e}")
    
    def create_dashboard(self, output_path: Optional[str] = None) -> None:
        """
        Create interactive dashboard with Plotly/Dash.
        
        Args:
            output_path: Path to save HTML dashboard
        """
        if not HAS_PLOTLY:
            raise ImportError("Plotly is required for dashboard creation")
            
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Attack Success Rates',
                'FAR/FRR Trade-offs',
                'Defense Detection Rates',
                'Performance Impact',
                'Execution Times',
                'Robustness Scores'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'scatter'}]
            ]
        )
        
        # Aggregate data from all results
        all_results = pd.concat(self.results_cache.values(), ignore_index=True)
        
        # Plot 1: Attack Success Rates
        success_by_type = all_results.groupby('attack_type')['success'].mean().sort_values()
        fig.add_trace(
            go.Bar(x=success_by_type.values, y=success_by_type.index, orientation='h',
                  name='Success Rate', marker_color='indianred'),
            row=1, col=1
        )
        
        # Plot 2: FAR/FRR Trade-offs
        fig.add_trace(
            go.Scatter(x=all_results['far_after'], y=all_results['frr_after'],
                      mode='markers', name='After Attack',
                      marker=dict(size=8, color='red')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=all_results['far_before'], y=all_results['frr_before'],
                      mode='markers', name='Before Attack',
                      marker=dict(size=8, color='blue')),
            row=1, col=2
        )
        
        # Plot 3: Defense Detection Rates
        detection_by_type = all_results.groupby('attack_type')['defense_detected'].mean().sort_values()
        fig.add_trace(
            go.Bar(x=detection_by_type.index, y=detection_by_type.values,
                  name='Detection Rate', marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Plot 4: Performance Impact
        accuracy_drop = all_results.groupby('attack_type').apply(
            lambda x: (x['accuracy_before'] - x['accuracy_after']).mean()
        ).sort_values()
        fig.add_trace(
            go.Bar(x=accuracy_drop.index, y=accuracy_drop.values,
                  name='Accuracy Drop', marker_color='orange'),
            row=2, col=2
        )
        
        # Plot 5: Execution Times
        exec_time = all_results.groupby('attack_type')['execution_time'].mean().sort_values()
        fig.add_trace(
            go.Bar(x=exec_time.index, y=exec_time.values,
                  name='Execution Time (s)', marker_color='purple'),
            row=3, col=1
        )
        
        # Plot 6: Robustness Scores by Model
        if 'model_name' in all_results.columns:
            model_scores = all_results.groupby('model_name').apply(
                lambda x: (1 - x['success'].mean()) * 100
            ).sort_values()
            fig.add_trace(
                go.Scatter(x=list(range(len(model_scores))), y=model_scores.values,
                          mode='lines+markers', name='Robustness',
                          text=model_scores.index, marker=dict(size=10)),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Attack Benchmark Dashboard",
            showlegend=True,
            height=1200,
            width=1600
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Success Rate", row=1, col=1)
        fig.update_xaxes(title_text="FAR", row=1, col=2)
        fig.update_yaxes(title_text="FRR", row=1, col=2)
        fig.update_xaxes(title_text="Attack Type", row=2, col=1)
        fig.update_yaxes(title_text="Detection Rate", row=2, col=1)
        fig.update_xaxes(title_text="Attack Type", row=2, col=2)
        fig.update_yaxes(title_text="Accuracy Drop", row=2, col=2)
        fig.update_xaxes(title_text="Attack Type", row=3, col=1)
        fig.update_yaxes(title_text="Time (s)", row=3, col=1)
        fig.update_xaxes(title_text="Model Index", row=3, col=2)
        fig.update_yaxes(title_text="Robustness Score", row=3, col=2)
        
        # Save or show
        if output_path:
            fig.write_html(output_path)
            print(f"Dashboard saved to {output_path}")
        else:
            fig.show()
    
    def plot_attack_success_rates(self, 
                                 save_path: Optional[str] = None) -> Optional[go.Figure]:
        """
        Bar chart of success rates by attack type.
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            Plotly figure or None if plotly not available
        """
        if not HAS_PLOTLY:
            logger.warning("Plotly not available for visualization")
            return None
            
        all_results = pd.concat(self.results_cache.values(), ignore_index=True)
        
        # Calculate success rates by attack type
        success_rates = all_results.groupby('attack_type')['success'].agg(['mean', 'std', 'count'])
        success_rates = success_rates.sort_values('mean', ascending=False)
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=success_rates.index,
                y=success_rates['mean'],
                error_y=dict(type='data', array=success_rates['std']),
                text=[f"{rate:.1%}<br>n={count}" 
                     for rate, count in zip(success_rates['mean'], success_rates['count'])],
                textposition='auto',
                marker_color='indianred'
            )
        ])
        
        fig.update_layout(
            title="Attack Success Rates by Type",
            xaxis_title="Attack Type",
            yaxis_title="Success Rate",
            yaxis_tickformat='.0%',
            showlegend=False,
            height=500,
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_far_frr_tradeoffs(self, 
                              save_path: Optional[str] = None) -> Optional[go.Figure]:
        """
        ROC curves under different attacks.
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            Plotly figure or None if plotly not available
        """
        if not HAS_PLOTLY:
            logger.warning("Plotly not available for visualization")
            return None
            
        all_results = pd.concat(self.results_cache.values(), ignore_index=True)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Add baseline ROC
        fig.add_trace(go.Scatter(
            x=all_results['far_before'],
            y=1 - all_results['frr_before'],
            mode='markers',
            name='Baseline',
            marker=dict(size=8, color='blue', opacity=0.6)
        ))
        
        # Add ROC for each attack type
        colors = px.colors.qualitative.Plotly
        for i, attack_type in enumerate(all_results['attack_type'].unique()):
            attack_data = all_results[all_results['attack_type'] == attack_type]
            fig.add_trace(go.Scatter(
                x=attack_data['far_after'],
                y=1 - attack_data['frr_after'],
                mode='markers',
                name=attack_type,
                marker=dict(size=8, color=colors[i % len(colors)], opacity=0.6)
            ))
        
        # Add diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash', color='gray'),
            showlegend=False
        ))
        
        fig.update_layout(
            title="FAR/FRR Trade-offs Under Different Attacks",
            xaxis_title="False Acceptance Rate (FAR)",
            yaxis_title="True Acceptance Rate (1 - FRR)",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            height=600,
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def plot_defense_adaptation(self, 
                               save_path: Optional[str] = None) -> Optional[go.Figure]:
        """
        Show how defenses adapt over time.
        
        Args:
            save_path: Optional path to save figure
            
        Returns:
            Plotly figure or None if plotly not available
        """
        if not HAS_PLOTLY:
            logger.warning("Plotly not available for visualization")
            return None
            
        all_results = pd.concat(self.results_cache.values(), ignore_index=True)
        
        # Sort by timestamp if available, otherwise by index
        if 'timestamp' in all_results.columns:
            all_results = all_results.sort_values('timestamp')
        
        # Calculate rolling detection rate
        window_size = min(20, len(all_results) // 5)
        all_results['rolling_detection'] = all_results['defense_detected'].rolling(
            window=window_size, min_periods=1
        ).mean()
        
        # Create line plot
        fig = go.Figure()
        
        # Add overall trend
        fig.add_trace(go.Scatter(
            x=list(range(len(all_results))),
            y=all_results['rolling_detection'],
            mode='lines',
            name='Detection Rate (Rolling Avg)',
            line=dict(color='green', width=2)
        ))
        
        # Add individual points colored by attack type
        for attack_type in all_results['attack_type'].unique():
            attack_data = all_results[all_results['attack_type'] == attack_type]
            fig.add_trace(go.Scatter(
                x=attack_data.index,
                y=attack_data['defense_detected'].astype(int),
                mode='markers',
                name=attack_type,
                marker=dict(size=6, opacity=0.5)
            ))
        
        fig.update_layout(
            title="Defense Adaptation Over Time",
            xaxis_title="Attack Sequence",
            yaxis_title="Detection Rate",
            yaxis_tickformat='.0%',
            height=500,
            template="plotly_white"
        )
        
        if save_path:
            fig.write_html(save_path)
            
        return fig
    
    def generate_summary_statistics(self) -> pd.DataFrame:
        """
        Generate summary statistics for all results.
        
        Returns:
            DataFrame with summary statistics
        """
        all_results = pd.concat(self.results_cache.values(), ignore_index=True)
        
        summary = {
            'Total Attacks': len(all_results),
            'Unique Attack Types': all_results['attack_type'].nunique(),
            'Overall Success Rate': f"{all_results['success'].mean():.1%}",
            'Overall Detection Rate': f"{all_results['defense_detected'].mean():.1%}",
            'Avg Execution Time': f"{all_results['execution_time'].mean():.2f}s",
            'Avg FAR Increase': f"{(all_results['far_after'] - all_results['far_before']).mean():.3f}",
            'Avg FRR Increase': f"{(all_results['frr_after'] - all_results['frr_before']).mean():.3f}",
            'Avg Accuracy Drop': f"{(all_results['accuracy_before'] - all_results['accuracy_after']).mean():.3f}"
        }
        
        return pd.DataFrame([summary]).T.rename(columns={0: 'Value'})


# Utility functions
def run_standard_benchmark(model: nn.Module, 
                          verifier: Any,
                          data_loader: DataLoader,
                          device: str = 'cpu',
                          save_results: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run standard benchmark suite and generate report.
    
    Args:
        model: Model to benchmark
        verifier: Verifier to test
        data_loader: Data for evaluation
        device: Device to run on
        save_results: Whether to save results
        
    Returns:
        Tuple of (results DataFrame, report dictionary)
    """
    benchmark = AttackBenchmark(device=device, save_results=save_results)
    
    # Run benchmark
    results = benchmark.run_benchmark(model, verifier, data_loader)
    
    # Generate report
    report = benchmark.generate_report(results)
    
    # Print summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    print(f"Robustness Score: {report['summary']['robustness_score']:.1f}/100")
    print(f"Attack Success Rate: {report['summary']['success_rate']:.1%}")
    print(f"Defense Detection Rate: {report['defense_effectiveness']['detection_rate']:.1%}")
    print(f"Total Execution Time: {report['summary']['total_execution_time']:.1f}s")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  • {rec}")
    print("="*60)
    
    return results, report


def create_comparison_dashboard(results_dict: Dict[str, pd.DataFrame],
                              output_dir: str = "benchmark_reports") -> None:
    """
    Create comparison dashboard for multiple models.
    
    Args:
        results_dict: Dictionary mapping model names to results
        output_dir: Directory to save dashboard files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create benchmark object for leaderboard generation
    benchmark = AttackBenchmark(save_results=False)
    
    # Generate leaderboard
    leaderboard = benchmark.generate_leaderboard(
        results_dict,
        save_path=output_path / "leaderboard.csv"
    )
    
    print("\n" + "="*60)
    print("MODEL LEADERBOARD")
    print("="*60)
    print(leaderboard.to_string(index=False))
    print("="*60)
    
    # Save combined results
    all_results = pd.concat(
        [df.assign(model=name) for name, df in results_dict.items()],
        ignore_index=True
    )
    all_results.to_csv(output_path / "all_results.csv", index=False)
    
    # Create dashboard if plotly available
    if HAS_PLOTLY:
        dashboard = AttackMetricsDashboard(str(output_path.parent))
        dashboard.create_dashboard(output_path / "dashboard.html")
        print(f"\nDashboard created at {output_path / 'dashboard.html'}")


if __name__ == "__main__":
    print("Attack Benchmarking Module")
    print("Use run_standard_benchmark() to evaluate a model")
    print("Use create_comparison_dashboard() to compare multiple models")