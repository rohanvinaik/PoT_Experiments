"""
Vision Verification Benchmarking and Evaluation Tools
Provides standardized benchmarks and robustness evaluation for vision verification.
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Import vision verification components
try:
    from .verifier import EnhancedVisionVerifier, VisionVerifierCalibrator
    VERIFIER_AVAILABLE = True
except ImportError:
    VERIFIER_AVAILABLE = False

# Optional plotting imports
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Result of a single benchmark test."""
    model_name: str
    challenge_type: str
    num_challenges: int
    success_rate: float
    avg_inference_time: float
    confidence: float
    verified: bool
    total_time: float
    throughput: float  # challenges per second
    memory_usage: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RobustnessResult:
    """Result of robustness evaluation."""
    test_type: str
    parameter: Union[str, float]
    success_rate: float
    std_dev: float
    num_trials: int
    baseline_success: float
    robustness_score: float  # ratio of perturbed to baseline success


class VisionBenchmark:
    """Standardized benchmark suite for vision verification."""
    
    STANDARD_CHALLENGES = {
        'basic': {
            'num_challenges': 5,
            'types': ['frequency'],
            'difficulty': 'easy',
            'description': 'Basic frequency domain challenges'
        },
        'intermediate': {
            'num_challenges': 10,
            'types': ['frequency', 'texture'],
            'difficulty': 'medium',
            'description': 'Mixed frequency and texture challenges'
        },
        'comprehensive': {
            'num_challenges': 20,
            'types': ['frequency', 'texture', 'natural'],
            'difficulty': 'hard',
            'description': 'Full challenge spectrum evaluation'
        },
        'stress': {
            'num_challenges': 50,
            'types': ['frequency', 'texture', 'natural'],
            'difficulty': 'extreme',
            'description': 'High-volume stress testing'
        }
    }
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize vision benchmark suite.
        
        Args:
            device: Device to run benchmarks on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.results = []
        self.benchmark_history = []
        
        if not VERIFIER_AVAILABLE:
            raise ImportError("Vision verifier components are required for benchmarking")
    
    def run_benchmark(self, 
                     model: nn.Module,
                     model_name: str = None,
                     benchmark_level: str = 'intermediate',
                     calibrate: bool = True,
                     num_calibration_samples: int = 50,
                     warmup_runs: int = 3,
                     measure_memory: bool = True) -> pd.DataFrame:
        """
        Run standardized benchmark on model.
        
        Args:
            model: PyTorch model to benchmark
            model_name: Name for the model (uses class name if None)
            benchmark_level: Level of benchmark to run
            calibrate: Whether to calibrate verifier before testing
            num_calibration_samples: Number of samples for calibration
            warmup_runs: Number of warmup runs for timing
            measure_memory: Whether to measure memory usage
            
        Returns:
            DataFrame with benchmark results
        """
        if benchmark_level not in self.STANDARD_CHALLENGES:
            raise ValueError(f"Unknown benchmark level: {benchmark_level}")
        
        config = self.STANDARD_CHALLENGES[benchmark_level]
        model_name = model_name or model.__class__.__name__
        
        print(f"Running {benchmark_level} benchmark on {model_name}")
        print(f"Description: {config['description']}")
        
        # Create verifier
        verifier_config = {
            'device': str(self.device),
            'verification_method': 'batch'
        }
        verifier = EnhancedVisionVerifier(model, verifier_config, device=str(self.device))
        
        # Calibrate if requested
        if calibrate:
            print(f"Calibrating verifier with {num_calibration_samples} samples...")
            calibrator = VisionVerifierCalibrator(verifier)
            calibrator.calibrate(
                num_samples=num_calibration_samples,
                challenge_types=config['types']
            )
        
        # Warmup runs
        if warmup_runs > 0:
            print(f"Performing {warmup_runs} warmup runs...")
            for _ in range(warmup_runs):
                self._run_single_test(
                    verifier, 
                    config['types'][0], 
                    1, 
                    model_name,
                    measure_memory=False
                )
        
        # Run benchmark tests
        results = []
        
        for challenge_type in config['types']:
            print(f"Testing {challenge_type} challenges...")
            result = self._run_single_test(
                verifier,
                challenge_type,
                config['num_challenges'],
                model_name,
                measure_memory=measure_memory
            )
            results.append(result)
        
        # Convert to DataFrame
        df = pd.DataFrame([asdict(r) for r in results])
        
        # Add summary metrics
        self._add_summary_metrics(df)
        
        # Store results
        self.results.extend(results)
        self.benchmark_history.append({
            'timestamp': time.time(),
            'model_name': model_name,
            'benchmark_level': benchmark_level,
            'results': df.to_dict('records')
        })
        
        print(f"Benchmark completed. Overall success rate: {df['success_rate'].mean():.3f}")
        
        return df
    
    def _run_single_test(self,
                        verifier: EnhancedVisionVerifier,
                        challenge_type: str,
                        num_challenges: int,
                        model_name: str,
                        measure_memory: bool = True) -> BenchmarkResult:
        """Run single benchmark test."""
        
        # Memory measurement
        initial_memory = None
        if measure_memory and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            initial_memory = torch.cuda.memory_allocated(self.device)
        
        start_time = time.time()
        
        # Run verification
        try:
            result = verifier.verify_session(
                num_challenges=num_challenges,
                challenge_types=[challenge_type]
            )
            
            success_rate = result.get('success_rate', 0.0)
            confidence = result.get('confidence', 0.0)
            verified = result.get('verified', False)
            
        except Exception as e:
            print(f"Warning: Verification failed for {challenge_type}: {e}")
            success_rate = 0.0
            confidence = 0.0
            verified = False
        
        total_time = time.time() - start_time
        avg_inference_time = total_time / num_challenges
        throughput = num_challenges / total_time
        
        # Memory usage
        memory_usage = None
        if measure_memory and torch.cuda.is_available() and initial_memory is not None:
            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated(self.device)
            memory_usage = (peak_memory - initial_memory) / 1024**2  # MB
            torch.cuda.reset_peak_memory_stats(self.device)
        
        return BenchmarkResult(
            model_name=model_name,
            challenge_type=challenge_type,
            num_challenges=num_challenges,
            success_rate=success_rate,
            avg_inference_time=avg_inference_time,
            confidence=confidence,
            verified=verified,
            total_time=total_time,
            throughput=throughput,
            memory_usage=memory_usage,
            metadata={
                'device': str(self.device),
                'timestamp': time.time()
            }
        )
    
    def _add_summary_metrics(self, df: pd.DataFrame):
        """Add summary metrics to benchmark DataFrame."""
        df['overall_success'] = df['success_rate'].mean()
        df['avg_confidence'] = df['confidence'].mean()
        df['verification_passed'] = df['verified'].sum() / len(df)
        df['avg_throughput'] = df['throughput'].mean()
        df['total_benchmark_time'] = df['total_time'].sum()
    
    def compare_models(self, 
                      models: Dict[str, nn.Module],
                      benchmark_level: str = 'intermediate',
                      **kwargs) -> pd.DataFrame:
        """
        Compare multiple models on the same benchmark.
        
        Args:
            models: Dictionary mapping model names to models
            benchmark_level: Benchmark level to run
            **kwargs: Additional arguments for run_benchmark
            
        Returns:
            Combined DataFrame with all model results
        """
        print(f"Comparing {len(models)} models on {benchmark_level} benchmark")
        
        all_results = []
        
        for name, model in models.items():
            print(f"\n{'='*50}")
            print(f"Benchmarking {name}")
            print(f"{'='*50}")
            
            df = self.run_benchmark(model, name, benchmark_level, **kwargs)
            all_results.append(df)
        
        # Combine results
        combined_df = pd.concat(all_results, ignore_index=True)
        
        # Add comparison metrics
        self._add_comparison_metrics(combined_df)
        
        return combined_df
    
    def _add_comparison_metrics(self, df: pd.DataFrame):
        """Add comparison metrics across models."""
        # Rank models by overall performance
        model_performance = df.groupby('model_name').agg({
            'success_rate': 'mean',
            'confidence': 'mean',
            'throughput': 'mean',
            'memory_usage': 'mean'
        }).reset_index()
        
        # Calculate composite score (higher is better)
        model_performance['composite_score'] = (
            0.4 * model_performance['success_rate'] +
            0.3 * model_performance['confidence'] +
            0.2 * model_performance['throughput'] / model_performance['throughput'].max() +
            0.1 * (1 - model_performance['memory_usage'] / model_performance['memory_usage'].max())
        )
        
        model_performance['rank'] = model_performance['composite_score'].rank(ascending=False)
        
        # Merge back with main dataframe
        df = df.merge(
            model_performance[['model_name', 'composite_score', 'rank']], 
            on='model_name', 
            how='left'
        )
    
    def generate_report(self, 
                       results: pd.DataFrame, 
                       output_path: str = 'vision_benchmark_report.html',
                       include_plots: bool = True) -> str:
        """
        Generate comprehensive HTML report with visualizations.
        
        Args:
            results: Benchmark results DataFrame
            output_path: Path to save HTML report
            include_plots: Whether to include interactive plots
            
        Returns:
            Path to generated report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if include_plots and PLOTLY_AVAILABLE:
            self._generate_plotly_report(results, output_path)
        elif include_plots and MATPLOTLIB_AVAILABLE:
            self._generate_matplotlib_report(results, output_path)
        else:
            self._generate_text_report(results, output_path)
        
        # Also save detailed CSV
        csv_path = output_path.with_suffix('.csv')
        results.to_csv(csv_path, index=False)
        
        # Save JSON summary
        json_path = output_path.with_suffix('.json')
        summary = self._generate_summary_stats(results)
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Report saved to {output_path}")
        print(f"CSV data saved to {csv_path}")
        print(f"JSON summary saved to {json_path}")
        
        return str(output_path)
    
    def _generate_plotly_report(self, results: pd.DataFrame, output_path: Path):
        """Generate interactive Plotly report."""
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Success Rate by Challenge Type',
                'Inference Time Comparison',
                'Confidence Scores Distribution',
                'Throughput Performance',
                'Memory Usage',
                'Verification Pass Rate'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        models = results['model_name'].unique()
        colors = px.colors.qualitative.Set1
        
        for i, model in enumerate(models):
            model_data = results[results['model_name'] == model]
            color = colors[i % len(colors)]
            
            # Success rate by challenge type
            fig.add_trace(
                go.Bar(
                    x=model_data['challenge_type'],
                    y=model_data['success_rate'],
                    name=f'{model} Success Rate',
                    marker_color=color,
                    showlegend=(i == 0)
                ),
                row=1, col=1
            )
            
            # Inference time
            fig.add_trace(
                go.Box(
                    x=model_data['challenge_type'],
                    y=model_data['avg_inference_time'],
                    name=f'{model} Inference Time',
                    marker_color=color,
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Confidence distribution
            fig.add_trace(
                go.Histogram(
                    x=model_data['confidence'],
                    name=f'{model} Confidence',
                    marker_color=color,
                    showlegend=False,
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Throughput
            fig.add_trace(
                go.Bar(
                    x=model_data['challenge_type'],
                    y=model_data['throughput'],
                    name=f'{model} Throughput',
                    marker_color=color,
                    showlegend=False
                ),
                row=2, col=2
            )
            
            # Memory usage (if available)
            if model_data['memory_usage'].notna().any():
                fig.add_trace(
                    go.Bar(
                        x=model_data['challenge_type'],
                        y=model_data['memory_usage'],
                        name=f'{model} Memory',
                        marker_color=color,
                        showlegend=False
                    ),
                    row=3, col=1
                )
            
            # Verification pass rate
            pass_rate = model_data.groupby('challenge_type')['verified'].mean()
            fig.add_trace(
                go.Bar(
                    x=pass_rate.index,
                    y=pass_rate.values,
                    name=f'{model} Pass Rate',
                    marker_color=color,
                    showlegend=False
                ),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            title='Vision Verification Benchmark Report',
            height=1200,
            showlegend=True
        )
        
        # Add axis labels
        fig.update_xaxes(title_text="Challenge Type", row=3, col=1)
        fig.update_yaxes(title_text="Success Rate", row=1, col=1)
        fig.update_yaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Throughput (challenges/s)", row=2, col=2)
        fig.update_yaxes(title_text="Memory (MB)", row=3, col=1)
        fig.update_yaxes(title_text="Pass Rate", row=3, col=2)
        
        # Save report
        fig.write_html(str(output_path))
    
    def _generate_matplotlib_report(self, results: pd.DataFrame, output_path: Path):
        """Generate static matplotlib report."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Vision Verification Benchmark Report', fontsize=16)
        
        # Success rate by challenge type
        sns.barplot(data=results, x='challenge_type', y='success_rate', 
                   hue='model_name', ax=axes[0, 0])
        axes[0, 0].set_title('Success Rate by Challenge Type')
        axes[0, 0].set_ylabel('Success Rate')
        
        # Inference time
        sns.boxplot(data=results, x='challenge_type', y='avg_inference_time',
                   hue='model_name', ax=axes[0, 1])
        axes[0, 1].set_title('Inference Time Distribution')
        axes[0, 1].set_ylabel('Time (s)')
        
        # Confidence scores
        sns.histplot(data=results, x='confidence', hue='model_name', 
                    alpha=0.7, ax=axes[0, 2])
        axes[0, 2].set_title('Confidence Score Distribution')
        
        # Throughput
        sns.barplot(data=results, x='challenge_type', y='throughput',
                   hue='model_name', ax=axes[1, 0])
        axes[1, 0].set_title('Throughput Performance')
        axes[1, 0].set_ylabel('Challenges/s')
        
        # Memory usage
        if results['memory_usage'].notna().any():
            sns.barplot(data=results, x='challenge_type', y='memory_usage',
                       hue='model_name', ax=axes[1, 1])
            axes[1, 1].set_title('Memory Usage')
            axes[1, 1].set_ylabel('Memory (MB)')
        else:
            axes[1, 1].text(0.5, 0.5, 'Memory data not available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Memory Usage')
        
        # Overall comparison
        summary = results.groupby('model_name').agg({
            'success_rate': 'mean',
            'confidence': 'mean',
            'throughput': 'mean'
        }).reset_index()
        
        x_pos = np.arange(len(summary))
        width = 0.25
        
        axes[1, 2].bar(x_pos - width, summary['success_rate'], width, 
                      label='Success Rate', alpha=0.8)
        axes[1, 2].bar(x_pos, summary['confidence'], width,
                      label='Confidence', alpha=0.8)
        axes[1, 2].bar(x_pos + width, summary['throughput'] / summary['throughput'].max(), 
                      width, label='Throughput (normalized)', alpha=0.8)
        
        axes[1, 2].set_title('Overall Model Comparison')
        axes[1, 2].set_xticks(x_pos)
        axes[1, 2].set_xticklabels(summary['model_name'], rotation=45)
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(str(output_path.with_suffix('.png')), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate HTML wrapper
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Vision Verification Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .summary {{ background-color: #f5f5f5; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                img {{ max-width: 100%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Vision Verification Benchmark Report</h1>
            <div class="summary">
                <h2>Summary Statistics</h2>
                {self._generate_html_summary(results)}
            </div>
            <h2>Visualizations</h2>
            <img src="{output_path.with_suffix('.png').name}" alt="Benchmark Results">
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _generate_text_report(self, results: pd.DataFrame, output_path: Path):
        """Generate simple text-based report."""
        summary = self._generate_summary_stats(results)
        
        report_content = f"""
Vision Verification Benchmark Report
{'='*50}

Summary Statistics:
{json.dumps(summary, indent=2)}

Detailed Results:
{results.to_string()}
        """
        
        with open(output_path.with_suffix('.txt'), 'w') as f:
            f.write(report_content)
    
    def _generate_html_summary(self, results: pd.DataFrame) -> str:
        """Generate HTML summary statistics."""
        summary = self._generate_summary_stats(results)
        
        html = ""
        for key, value in summary.items():
            if isinstance(value, dict):
                html += f"<h3>{key.replace('_', ' ').title()}</h3>"
                for subkey, subvalue in value.items():
                    html += f'<div class="metric"><strong>{subkey}:</strong> {subvalue}</div>'
            else:
                html += f'<div class="metric"><strong>{key}:</strong> {value}</div>'
        
        return html
    
    def _generate_summary_stats(self, results: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics dictionary."""
        return {
            'total_models': results['model_name'].nunique(),
            'total_tests': len(results),
            'challenge_types': list(results['challenge_type'].unique()),
            'overall_metrics': {
                'avg_success_rate': float(results['success_rate'].mean()),
                'avg_confidence': float(results['confidence'].mean()),
                'avg_throughput': float(results['throughput'].mean()),
                'total_challenges': int(results['num_challenges'].sum()),
                'total_time': float(results['total_time'].sum())
            },
            'model_rankings': results.groupby('model_name')['success_rate'].mean().sort_values(ascending=False).to_dict(),
            'best_challenge_type': results.groupby('challenge_type')['success_rate'].mean().idxmax(),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def save_benchmark_state(self, filepath: str):
        """Save benchmark state for later analysis."""
        state = {
            'results': [asdict(r) for r in self.results],
            'history': self.benchmark_history,
            'device': str(self.device),
            'timestamp': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_benchmark_state(self, filepath: str):
        """Load previous benchmark state."""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.results = [BenchmarkResult(**r) for r in state['results']]
        self.benchmark_history = state['history']


class VisionRobustnessEvaluator:
    """Evaluate robustness of vision verification against various perturbations."""
    
    def __init__(self, verifier: EnhancedVisionVerifier, device: str = 'cuda'):
        """
        Initialize robustness evaluator.
        
        Args:
            verifier: Vision verifier to test
            device: Device to run tests on
        """
        self.verifier = verifier
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.baseline_results = {}
        
    def evaluate_noise_robustness(self, 
                                 noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2],
                                 num_trials: int = 20,
                                 challenge_types: List[str] = ['frequency', 'texture']) -> Dict[str, RobustnessResult]:
        """
        Test robustness to additive Gaussian noise.
        
        Args:
            noise_levels: Standard deviations of noise to test
            num_trials: Number of trials per noise level
            challenge_types: Types of challenges to test
            
        Returns:
            Dictionary of robustness results
        """
        print("Evaluating noise robustness...")
        
        results = {}
        
        for challenge_type in challenge_types:
            print(f"Testing {challenge_type} challenges...")
            
            # Get baseline performance
            baseline_success = self._get_baseline_performance(challenge_type, num_trials)
            
            for noise_level in noise_levels:
                print(f"  Noise level: {noise_level}")
                
                success_rates = []
                
                for trial in range(num_trials):
                    try:
                        # Generate clean challenge
                        if challenge_type == 'frequency':
                            challenges = self.verifier.generate_frequency_challenges(1)
                        elif challenge_type == 'texture':
                            challenges = self.verifier.generate_texture_challenges(1)
                        else:
                            continue
                        
                        challenge = challenges[0].to(self.device)
                        
                        # Add noise
                        noise = torch.randn_like(challenge) * noise_level
                        noisy_challenge = torch.clamp(challenge + noise, 0, 1)
                        
                        # Test verification
                        output = self.verifier.run_model(noisy_challenge.unsqueeze(0))
                        success = self.verifier._evaluate_challenge_response(output, challenge_type)
                        success_rates.append(float(success))
                        
                    except Exception as e:
                        print(f"    Trial {trial} failed: {e}")
                        success_rates.append(0.0)
                
                # Calculate robustness metrics
                mean_success = np.mean(success_rates)
                std_success = np.std(success_rates)
                robustness_score = mean_success / baseline_success if baseline_success > 0 else 0.0
                
                result_key = f"{challenge_type}_noise_{noise_level}"
                results[result_key] = RobustnessResult(
                    test_type='additive_noise',
                    parameter=noise_level,
                    success_rate=mean_success,
                    std_dev=std_success,
                    num_trials=num_trials,
                    baseline_success=baseline_success,
                    robustness_score=robustness_score
                )
        
        return results
    
    def evaluate_transformation_robustness(self,
                                         num_trials: int = 20,
                                         challenge_types: List[str] = ['frequency', 'texture']) -> Dict[str, RobustnessResult]:
        """
        Test robustness to common image transformations.
        
        Args:
            num_trials: Number of trials per transformation
            challenge_types: Types of challenges to test
            
        Returns:
            Dictionary of robustness results
        """
        print("Evaluating transformation robustness...")
        
        transformations = {
            'rotation_90': lambda x: torch.rot90(x, 1, [1, 2]),
            'rotation_180': lambda x: torch.rot90(x, 2, [1, 2]),
            'flip_horizontal': lambda x: torch.flip(x, [2]),
            'flip_vertical': lambda x: torch.flip(x, [1]),
            'brightness_increase': lambda x: torch.clamp(x * 1.3, 0, 1),
            'brightness_decrease': lambda x: torch.clamp(x * 0.7, 0, 1),
            'contrast_increase': lambda x: torch.clamp((x - 0.5) * 1.5 + 0.5, 0, 1),
            'contrast_decrease': lambda x: torch.clamp((x - 0.5) * 0.5 + 0.5, 0, 1),
            'gaussian_blur': lambda x: self._gaussian_blur(x, kernel_size=5),
            'sharpen': lambda x: self._sharpen_filter(x)
        }
        
        results = {}
        
        for challenge_type in challenge_types:
            print(f"Testing {challenge_type} challenges...")
            
            # Get baseline performance
            baseline_success = self._get_baseline_performance(challenge_type, num_trials)
            
            for transform_name, transform_fn in transformations.items():
                print(f"  Transformation: {transform_name}")
                
                success_rates = []
                
                for trial in range(num_trials):
                    try:
                        # Generate challenge
                        if challenge_type == 'frequency':
                            challenges = self.verifier.generate_frequency_challenges(1)
                        elif challenge_type == 'texture':
                            challenges = self.verifier.generate_texture_challenges(1)
                        else:
                            continue
                        
                        challenge = challenges[0].to(self.device)
                        
                        # Apply transformation
                        transformed = transform_fn(challenge)
                        
                        # Test verification
                        output = self.verifier.run_model(transformed.unsqueeze(0))
                        success = self.verifier._evaluate_challenge_response(output, challenge_type)
                        success_rates.append(float(success))
                        
                    except Exception as e:
                        print(f"    Trial {trial} failed: {e}")
                        success_rates.append(0.0)
                
                # Calculate robustness metrics
                mean_success = np.mean(success_rates)
                std_success = np.std(success_rates)
                robustness_score = mean_success / baseline_success if baseline_success > 0 else 0.0
                
                result_key = f"{challenge_type}_{transform_name}"
                results[result_key] = RobustnessResult(
                    test_type='transformation',
                    parameter=transform_name,
                    success_rate=mean_success,
                    std_dev=std_success,
                    num_trials=num_trials,
                    baseline_success=baseline_success,
                    robustness_score=robustness_score
                )
        
        return results
    
    def evaluate_adversarial_robustness(self,
                                      epsilon_values: List[float] = [0.01, 0.03, 0.1],
                                      attack_steps: int = 10,
                                      num_trials: int = 10,
                                      challenge_types: List[str] = ['frequency']) -> Dict[str, RobustnessResult]:
        """
        Test robustness to adversarial perturbations using PGD attack.
        
        Args:
            epsilon_values: Maximum perturbation magnitudes
            attack_steps: Number of PGD steps
            num_trials: Number of trials per epsilon
            challenge_types: Types of challenges to test
            
        Returns:
            Dictionary of robustness results
        """
        print("Evaluating adversarial robustness...")
        
        results = {}
        
        for challenge_type in challenge_types:
            print(f"Testing {challenge_type} challenges...")
            
            # Get baseline performance
            baseline_success = self._get_baseline_performance(challenge_type, num_trials)
            
            for epsilon in epsilon_values:
                print(f"  Epsilon: {epsilon}")
                
                success_rates = []
                
                for trial in range(num_trials):
                    try:
                        # Generate challenge
                        if challenge_type == 'frequency':
                            challenges = self.verifier.generate_frequency_challenges(1)
                        elif challenge_type == 'texture':
                            challenges = self.verifier.generate_texture_challenges(1)
                        else:
                            continue
                        
                        challenge = challenges[0].to(self.device)
                        challenge.requires_grad_(True)
                        
                        # Generate adversarial perturbation
                        adversarial_challenge = self._pgd_attack(
                            challenge, epsilon, attack_steps
                        )
                        
                        # Test verification
                        with torch.no_grad():
                            output = self.verifier.run_model(adversarial_challenge.unsqueeze(0))
                            success = self.verifier._evaluate_challenge_response(output, challenge_type)
                            success_rates.append(float(success))
                        
                    except Exception as e:
                        print(f"    Trial {trial} failed: {e}")
                        success_rates.append(0.0)
                
                # Calculate robustness metrics
                mean_success = np.mean(success_rates)
                std_success = np.std(success_rates)
                robustness_score = mean_success / baseline_success if baseline_success > 0 else 0.0
                
                result_key = f"{challenge_type}_adversarial_eps_{epsilon}"
                results[result_key] = RobustnessResult(
                    test_type='adversarial',
                    parameter=epsilon,
                    success_rate=mean_success,
                    std_dev=std_success,
                    num_trials=num_trials,
                    baseline_success=baseline_success,
                    robustness_score=robustness_score
                )
        
        return results
    
    def _get_baseline_performance(self, challenge_type: str, num_trials: int) -> float:
        """Get baseline performance without perturbations."""
        
        cache_key = f"{challenge_type}_{num_trials}"
        if cache_key in self.baseline_results:
            return self.baseline_results[cache_key]
        
        success_rates = []
        
        for trial in range(num_trials):
            try:
                # Generate clean challenge
                if challenge_type == 'frequency':
                    challenges = self.verifier.generate_frequency_challenges(1)
                elif challenge_type == 'texture':
                    challenges = self.verifier.generate_texture_challenges(1)
                else:
                    continue
                
                challenge = challenges[0].to(self.device)
                
                # Test verification
                output = self.verifier.run_model(challenge.unsqueeze(0))
                success = self.verifier._evaluate_challenge_response(output, challenge_type)
                success_rates.append(float(success))
                
            except Exception as e:
                print(f"Baseline trial {trial} failed: {e}")
                success_rates.append(0.0)
        
        baseline = np.mean(success_rates)
        self.baseline_results[cache_key] = baseline
        return baseline
    
    def _gaussian_blur(self, x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
        """Apply Gaussian blur to tensor."""
        
        # Create Gaussian kernel
        kernel_1d = torch.exp(
            -0.5 * (torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2) ** 2 / sigma ** 2
        )
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        kernel = kernel_2d.expand(x.shape[0], 1, kernel_size, kernel_size).to(x.device)
        
        # Apply convolution (blur)
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Pad input
        padding = kernel_size // 2
        x_padded = F.pad(x, (padding, padding, padding, padding), mode='reflect')
        
        # Apply blur channel-wise
        blurred = []
        for c in range(x.shape[1]):
            channel = x_padded[:, c:c+1]
            blurred_channel = F.conv2d(channel, kernel[c:c+1], padding=0)
            blurred.append(blurred_channel)
        
        result = torch.cat(blurred, dim=1)
        
        if squeeze_output:
            result = result.squeeze(0)
        
        return result
    
    def _sharpen_filter(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sharpening filter to tensor."""
        # Sharpening kernel
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(x.device)
        
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Apply sharpening
        sharpened = []
        for c in range(x.shape[1]):
            channel = x[:, c:c+1]
            sharpened_channel = F.conv2d(channel, kernel, padding=1)
            sharpened.append(sharpened_channel)
        
        result = torch.cat(sharpened, dim=1)
        result = torch.clamp(result, 0, 1)
        
        if squeeze_output:
            result = result.squeeze(0)
        
        return result
    
    def _pgd_attack(self, x: torch.Tensor, epsilon: float, num_steps: int) -> torch.Tensor:
        """Generate adversarial example using PGD attack."""
        
        alpha = epsilon / num_steps * 2
        x_adv = x.clone().detach()
        
        for step in range(num_steps):
            x_adv.requires_grad_(True)
            
            # Forward pass
            output = self.verifier.run_model(x_adv.unsqueeze(0))
            
            # Simple loss: try to minimize confidence
            loss = output['logits'].max()
            
            # Backward pass
            loss.backward()
            
            # Update adversarial example
            with torch.no_grad():
                x_adv = x_adv - alpha * x_adv.grad.sign()
                x_adv = torch.clamp(x_adv, x - epsilon, x + epsilon)
                x_adv = torch.clamp(x_adv, 0, 1)
            
            x_adv.grad = None
        
        return x_adv.detach()
    
    def generate_robustness_report(self, 
                                  results: Dict[str, RobustnessResult],
                                  output_path: str = 'robustness_report.html') -> str:
        """Generate robustness evaluation report."""
        
        # Convert results to DataFrame
        df_data = []
        for key, result in results.items():
            data = asdict(result)
            data['test_name'] = key
            df_data.append(data)
        
        df = pd.DataFrame(df_data)
        
        if PLOTLY_AVAILABLE:
            self._generate_robustness_plotly_report(df, output_path)
        else:
            self._generate_robustness_text_report(df, output_path)
        
        # Save CSV
        csv_path = Path(output_path).with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        
        print(f"Robustness report saved to {output_path}")
        return output_path
    
    def _generate_robustness_plotly_report(self, df: pd.DataFrame, output_path: str):
        """Generate Plotly robustness report."""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Robustness Score by Test Type',
                'Success Rate Distribution',
                'Parameter Sensitivity',
                'Baseline vs Perturbed Performance'
            )
        )
        
        # Group by test type
        for test_type in df['test_type'].unique():
            test_data = df[df['test_type'] == test_type]
            
            # Robustness scores
            fig.add_trace(
                go.Bar(
                    x=test_data['test_name'],
                    y=test_data['robustness_score'],
                    name=f'{test_type} Robustness',
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Success rate distribution
            fig.add_trace(
                go.Box(
                    y=test_data['success_rate'],
                    name=f'{test_type} Success Rate',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Parameter sensitivity (for numeric parameters)
        numeric_data = df[pd.to_numeric(df['parameter'], errors='coerce').notna()]
        if not numeric_data.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_numeric(numeric_data['parameter']),
                    y=numeric_data['robustness_score'],
                    mode='lines+markers',
                    name='Parameter Sensitivity',
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Baseline vs perturbed
        fig.add_trace(
            go.Scatter(
                x=df['baseline_success'],
                y=df['success_rate'],
                mode='markers',
                name='Baseline vs Perturbed',
                showlegend=False
            ),
            row=2, col=2
        )
        
        # Add diagonal line for baseline vs perturbed
        max_val = max(df['baseline_success'].max(), df['success_rate'].max())
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode='lines',
                line=dict(dash='dash', color='red'),
                name='Perfect Robustness',
                showlegend=False
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title='Vision Verification Robustness Evaluation',
            height=800
        )
        
        fig.write_html(output_path)
    
    def _generate_robustness_text_report(self, df: pd.DataFrame, output_path: str):
        """Generate text-based robustness report."""
        
        report = f"""
Vision Verification Robustness Evaluation Report
{'='*60}

Summary Statistics:
- Total tests: {len(df)}
- Test types: {', '.join(df['test_type'].unique())}
- Average robustness score: {df['robustness_score'].mean():.3f}
- Best performing test: {df.loc[df['robustness_score'].idxmax(), 'test_name']}
- Worst performing test: {df.loc[df['robustness_score'].idxmin(), 'test_name']}

Detailed Results:
{df.to_string(index=False)}

Test Type Analysis:
"""
        
        for test_type in df['test_type'].unique():
            test_data = df[df['test_type'] == test_type]
            report += f"""
{test_type.upper()} Tests:
- Number of tests: {len(test_data)}
- Average robustness: {test_data['robustness_score'].mean():.3f}
- Average success rate: {test_data['success_rate'].mean():.3f}
- Standard deviation: {test_data['std_dev'].mean():.3f}
"""
        
        with open(Path(output_path).with_suffix('.txt'), 'w') as f:
            f.write(report)


# Export main classes
__all__ = [
    'VisionBenchmark',
    'VisionRobustnessEvaluator', 
    'BenchmarkResult',
    'RobustnessResult'
]