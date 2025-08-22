#!/usr/bin/env python3
"""
Command-line interface for PoT Attack Resistance evaluation.

This module provides CLI commands for running attacks, generating reports,
detecting wrappers, and benchmarking models.
"""

import click
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import sys
import os
from typing import Dict, Any, Optional, List
import warnings

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from pot.core.attack_suites import (
    StandardAttackSuite, 
    AdaptiveAttackSuite,
    ComprehensiveAttackSuite,
    AttackRunner,
    WrapperDetector,
    AttackConfig
)
from pot.core.defenses import IntegratedDefenseSystem, DefenseConfig
from pot.eval.attack_benchmarks import (
    AttackBenchmark,
    AttackMetricsDashboard,
    run_standard_benchmark
)
from pot.vision.attacks import execute_vision_attack
from pot.security.proof_of_training import ProofOfTraining

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def attack_cli(verbose: bool, debug: bool):
    """
    PoT Attack Resistance CLI.
    
    Comprehensive tool for evaluating model robustness against various attacks.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)


@attack_cli.command()
@click.option('--model-path', '-m', required=True, help='Path to model checkpoint')
@click.option('--attack-suite', '-s', default='standard', 
              type=click.Choice(['standard', 'adaptive', 'comprehensive', 'quick', 'vision']),
              help='Attack suite to run')
@click.option('--output-dir', '-o', default='./attack_results', help='Output directory for results')
@click.option('--data-path', '-d', help='Path to test data')
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'cuda', 'mps']),
              help='Device to run attacks on')
@click.option('--batch-size', default=32, help='Batch size for evaluation')
@click.option('--num-samples', default=1000, help='Number of samples to use')
@click.option('--include-defenses', is_flag=True, help='Include defense evaluation')
@click.option('--config', '-c', help='Custom configuration file')
def run_attacks(model_path: str, attack_suite: str, output_dir: str, 
                data_path: Optional[str], device: str, batch_size: int,
                num_samples: int, include_defenses: bool, config: Optional[str]):
    """
    Run attack suite against model.
    
    Examples:
        pot-attack run-attacks -m model.pth -s standard -o results/
        pot-attack run-attacks -m model.pth -s adaptive --include-defenses
    """
    click.echo(f"Running {attack_suite} attack suite against {model_path}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model
        click.echo("Loading model...")
        model = load_model(model_path, device)
        
        # Load or create data
        click.echo("Preparing data...")
        data_loader = prepare_data(data_path, batch_size, num_samples)
        
        # Load configuration if provided
        attack_config = {}
        if config:
            with open(config, 'r') as f:
                attack_config = yaml.safe_load(f)
        
        # Initialize attack suite
        click.echo(f"Initializing {attack_suite} attack suite...")
        suite = get_attack_suite(attack_suite, attack_config)
        
        # Initialize runner
        runner = AttackRunner(device=device, verbose=True)
        
        # Get attack configurations
        if attack_suite == 'standard':
            configs = suite.get_all_configs()
        elif attack_suite == 'adaptive':
            configs = suite.evolve_attacks(model, data_loader)
        elif attack_suite == 'comprehensive':
            configs = suite.get_all_attacks()
        elif attack_suite == 'quick':
            configs = get_quick_configs()
        else:  # vision
            configs = suite.get_vision_specific_configs()
        
        click.echo(f"Running {len(configs)} attacks...")
        
        # Run attacks
        results = []
        with click.progressbar(configs, label='Executing attacks') as bar:
            for config in bar:
                result = runner.run_single_attack(model, config, data_loader)
                results.append(result)
        
        # Include defense evaluation if requested
        if include_defenses:
            click.echo("Evaluating defenses...")
            defense_results = evaluate_defenses(model, results, data_loader, device)
            for i, defense_result in enumerate(defense_results):
                results[i]['defense_evaluation'] = defense_result
        
        # Calculate metrics
        metrics = runner.calculate_metrics(results)
        robustness = runner.calculate_robustness_score(results)
        
        # Save results
        save_results(results, metrics, robustness, output_path)
        
        # Display summary
        click.echo("\n" + "="*60)
        click.echo("ATTACK EVALUATION SUMMARY")
        click.echo("="*60)
        click.echo(f"Total attacks: {metrics['total_attacks']}")
        click.echo(f"Successful attacks: {metrics['successful_attacks']}")
        click.echo(f"Success rate: {metrics['success_rate']:.1%}")
        click.echo(f"Robustness score: {robustness:.1f}/100")
        
        if include_defenses:
            defense_detection = sum(1 for r in results if r.get('defense_evaluation', {}).get('detected', False))
            click.echo(f"Defense detection rate: {defense_detection/len(results):.1%}")
        
        click.echo("="*60)
        click.echo(f"\nResults saved to {output_path}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Attack execution failed")
        sys.exit(1)


@attack_cli.command()
@click.option('--results-dir', '-r', required=True, help='Directory containing attack results')
@click.option('--output-file', '-o', default='report.html', help='Output HTML report file')
@click.option('--include-plots', is_flag=True, help='Include interactive plots')
@click.option('--format', 'report_format', 
              type=click.Choice(['html', 'pdf', 'markdown']), 
              default='html', help='Report format')
def generate_report(results_dir: str, output_file: str, include_plots: bool, report_format: str):
    """
    Generate HTML report from attack results.
    
    Examples:
        pot-attack generate-report -r results/ -o report.html
        pot-attack generate-report -r results/ --include-plots --format pdf
    """
    click.echo(f"Generating {report_format} report from {results_dir}")
    
    try:
        results_path = Path(results_dir)
        
        # Load results
        results = load_results(results_path)
        
        if not results:
            click.echo("No results found in directory", err=True)
            sys.exit(1)
        
        # Create dashboard if plots requested
        if include_plots:
            click.echo("Creating visualizations...")
            dashboard = AttackMetricsDashboard(str(results_path))
            dashboard.create_dashboard(str(results_path / "dashboard.html"))
        
        # Generate report based on format
        if report_format == 'html':
            report_content = generate_html_report(results, results_path)
            output_path = results_path / output_file
        elif report_format == 'pdf':
            click.echo("PDF generation requires additional dependencies")
            report_content = generate_html_report(results, results_path)
            output_path = results_path / output_file.replace('.html', '.pdf')
            # Would use weasyprint or similar for PDF conversion
        else:  # markdown
            report_content = generate_markdown_report(results, results_path)
            output_path = results_path / output_file.replace('.html', '.md')
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        click.echo(f"Report saved to {output_path}")
        
        if include_plots:
            click.echo(f"Interactive dashboard saved to {results_path / 'dashboard.html'}")
        
    except Exception as e:
        click.echo(f"Error generating report: {e}", err=True)
        logger.exception("Report generation failed")
        sys.exit(1)


@attack_cli.command()
@click.option('--model-path', '-m', required=True, help='Path to model to test')
@click.option('--baseline-path', '-b', required=True, help='Path to baseline model')
@click.option('--samples', '-n', default=1000, type=int, help='Number of samples for detection')
@click.option('--output', '-o', help='Output file for detection results')
@click.option('--threshold', '-t', default=0.15, type=float, help='Detection threshold')
@click.option('--methods', '-M', multiple=True, 
              default=['timing', 'ecdf', 'behavioral'],
              help='Detection methods to use')
def detect_wrapper(model_path: str, baseline_path: str, samples: int, 
                  output: Optional[str], threshold: float, methods: List[str]):
    """
    Detect if model is wrapped.
    
    Examples:
        pot-attack detect-wrapper -m model.pth -b baseline.pth
        pot-attack detect-wrapper -m model.pth -b baseline.pth -n 5000 -t 0.2
    """
    click.echo(f"Detecting wrapper for {model_path}")
    click.echo(f"Using baseline: {baseline_path}")
    
    try:
        # Load models
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(model_path, device)
        baseline = load_model(baseline_path, device)
        
        # Generate test samples
        click.echo(f"Generating {samples} test samples...")
        test_data = torch.randn(samples, 3, 224, 224).to(device)
        
        # Collect timing and response data
        click.echo("Collecting baseline statistics...")
        baseline_times, baseline_responses = collect_model_stats(baseline, test_data)
        
        click.echo("Collecting model statistics...")
        model_times, model_responses = collect_model_stats(model, test_data)
        
        # Initialize detector
        detector = WrapperDetector(baseline_times, baseline_responses)
        
        # Run detection
        click.echo("Running wrapper detection...")
        detection_results = {}
        
        for method in methods:
            if method == 'timing':
                result = detector.detect_timing_anomaly(model_times)
            elif method == 'ecdf':
                result = detector.detect_ecdf_anomaly(model_responses)
            elif method == 'behavioral':
                result = detector.detect_behavioral_drift(model_responses)
            else:
                click.echo(f"Unknown method: {method}", err=True)
                continue
            
            detection_results[method] = result
        
        # Combined detection
        combined_result = detector.detect_wrapper(model_times, model_responses)
        detection_results['combined'] = combined_result
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("WRAPPER DETECTION RESULTS")
        click.echo("="*60)
        
        for method, result in detection_results.items():
            is_wrapped = result.get('is_wrapped', False)
            confidence = result.get('confidence', 0.0)
            status = "WRAPPED" if is_wrapped else "CLEAN"
            color = 'red' if is_wrapped else 'green'
            
            click.echo(f"{method.capitalize():15} {status:8} (confidence: {confidence:.3f})")
        
        # Final verdict
        final_wrapped = combined_result['is_wrapped']
        final_confidence = combined_result['confidence']
        
        click.echo("="*60)
        if final_wrapped:
            click.echo(f"VERDICT: Model is likely WRAPPED (confidence: {final_confidence:.3f})", fg='red')
        else:
            click.echo(f"VERDICT: Model appears CLEAN (confidence: {1-final_confidence:.3f})", fg='green')
        
        # Save results if requested
        if output:
            save_detection_results(detection_results, output)
            click.echo(f"\nDetailed results saved to {output}")
        
    except Exception as e:
        click.echo(f"Error in wrapper detection: {e}", err=True)
        logger.exception("Wrapper detection failed")
        sys.exit(1)


@attack_cli.command()
@click.option('--config-file', '-c', required=True, help='Benchmark configuration file')
@click.option('--model-path', '-m', help='Model to benchmark')
@click.option('--output-dir', '-o', default='benchmark_results', help='Output directory')
@click.option('--compare', '-C', multiple=True, help='Additional models to compare')
@click.option('--attacks', '-a', help='Comma-separated list of attacks to run')
@click.option('--device', default='cpu', type=click.Choice(['cpu', 'cuda', 'mps']))
@click.option('--generate-leaderboard', is_flag=True, help='Generate comparison leaderboard')
def benchmark(config_file: str, model_path: Optional[str], output_dir: str,
             compare: List[str], attacks: Optional[str], device: str,
             generate_leaderboard: bool):
    """
    Run standardized attack benchmark.
    
    Examples:
        pot-attack benchmark -c config.yaml -m model.pth
        pot-attack benchmark -c config.yaml -m model1.pth -C model2.pth -C model3.pth --generate-leaderboard
    """
    click.echo(f"Running benchmark with config: {config_file}")
    
    try:
        # Load configuration
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Prepare output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Determine attacks to run
        if attacks:
            attack_list = attacks.split(',')
        elif 'attacks' in config:
            attack_list = config['attacks']
        else:
            attack_list = None  # Use default
        
        # Initialize benchmark
        benchmark_runner = AttackBenchmark(
            device=device,
            verbose=True,
            save_results=True,
            results_dir=str(output_path)
        )
        
        if attack_list:
            benchmark_runner.STANDARD_ATTACKS = attack_list
        
        # Prepare models to benchmark
        models_to_test = {}
        
        if model_path:
            model = load_model(model_path, device)
            model_name = Path(model_path).stem
            models_to_test[model_name] = model
        
        for comp_path in compare:
            model = load_model(comp_path, device)
            model_name = Path(comp_path).stem
            models_to_test[model_name] = model
        
        if not models_to_test:
            click.echo("No models specified for benchmarking", err=True)
            sys.exit(1)
        
        # Prepare data
        data_loader = prepare_benchmark_data(config.get('data', {}))
        
        # Initialize verifier if specified
        verifier = None
        if config.get('verifier', {}).get('enabled', False):
            verifier_config = config['verifier']
            # Initialize appropriate verifier based on config
            click.echo("Initializing verifier...")
            
        # Run benchmarks
        all_results = {}
        
        for model_name, model in models_to_test.items():
            click.echo(f"\nBenchmarking {model_name}...")
            
            results = benchmark_runner.run_benchmark(
                model=model,
                verifier=verifier,
                data_loader=data_loader,
                attack_names=attack_list,
                include_defenses=config.get('defenses', {}).get('enabled', True)
            )
            
            all_results[model_name] = results
            
            # Display summary
            robustness = benchmark_runner.compute_robustness_score(results)
            click.echo(f"  Robustness score: {robustness:.1f}/100")
            click.echo(f"  Success rate: {results['success'].mean():.1%}")
        
        # Generate leaderboard if requested
        if generate_leaderboard and len(all_results) > 1:
            click.echo("\nGenerating leaderboard...")
            leaderboard = benchmark_runner.generate_leaderboard(
                all_results,
                save_path=str(output_path / "leaderboard.csv")
            )
            
            # Display leaderboard
            click.echo("\n" + "="*60)
            click.echo("MODEL LEADERBOARD")
            click.echo("="*60)
            click.echo(leaderboard.to_string(index=False))
        
        # Generate report
        click.echo("\nGenerating benchmark report...")
        for model_name, results in all_results.items():
            report = benchmark_runner.generate_report(
                results,
                save_path=str(output_path / f"{model_name}_report.json")
            )
        
        click.echo(f"\nBenchmark complete. Results saved to {output_path}")
        
    except Exception as e:
        click.echo(f"Error in benchmark: {e}", err=True)
        logger.exception("Benchmark failed")
        sys.exit(1)


@attack_cli.command()
@click.option('--model-path', '-m', required=True, help='Model to evaluate')
@click.option('--reference-path', '-r', required=True, help='Reference model')
@click.option('--profile', '-p', 
              type=click.Choice(['quick', 'standard', 'comprehensive']),
              default='standard', help='Verification profile')
@click.option('--security-level', '-s',
              type=click.Choice(['low', 'medium', 'high']),
              default='medium', help='Security level')
@click.option('--output', '-o', help='Output file for results')
def verify(model_path: str, reference_path: str, profile: str, 
          security_level: str, output: Optional[str]):
    """
    Verify model with PoT and evaluate attack resistance.
    
    Examples:
        pot-attack verify -m model.pth -r reference.pth -p standard
        pot-attack verify -m model.pth -r reference.pth -p comprehensive -s high
    """
    click.echo(f"Verifying {model_path} against {reference_path}")
    click.echo(f"Profile: {profile}, Security: {security_level}")
    
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load models
        model = load_model(model_path, device)
        reference = load_model(reference_path, device)
        
        # Initialize PoT verifier
        pot_config = {
            'model_type': 'vision',  # Could be detected
            'verification_type': 'behavioral',
            'security_level': security_level
        }
        
        pot = ProofOfTraining(pot_config)
        
        # Run verification
        click.echo("Running PoT verification...")
        result = pot.perform_verification(model, Path(model_path).stem, profile)
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("VERIFICATION RESULTS")
        click.echo("="*60)
        click.echo(f"Verified: {result['verified']}")
        click.echo(f"Confidence: {result['confidence']:.3f}")
        click.echo(f"Security level: {result.get('security_level', 'N/A')}")
        
        if 'attack_resistance' in result:
            click.echo(f"Attack resistance: {result['attack_resistance']:.1f}/100")
        
        # Save results if requested
        if output:
            with open(output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            click.echo(f"\nResults saved to {output}")
        
    except Exception as e:
        click.echo(f"Error in verification: {e}", err=True)
        logger.exception("Verification failed")
        sys.exit(1)


@attack_cli.command()
@click.option('--port', '-p', default=8050, help='Port for dashboard server')
@click.option('--results-dir', '-r', default='benchmark_results', 
              help='Directory with benchmark results')
@click.option('--host', '-h', default='127.0.0.1', help='Host to bind to')
@click.option('--debug', is_flag=True, help='Run in debug mode')
def dashboard(port: int, results_dir: str, host: str, debug: bool):
    """
    Launch interactive dashboard for attack metrics.
    
    Examples:
        pot-attack dashboard -r benchmark_results/
        pot-attack dashboard -p 8080 --debug
    """
    click.echo(f"Launching dashboard on {host}:{port}")
    
    try:
        # Check if Dash is available
        try:
            import dash
            import plotly
        except ImportError:
            click.echo("Dash and Plotly required for dashboard. Install with:", err=True)
            click.echo("  pip install dash plotly", err=True)
            sys.exit(1)
        
        # Import dashboard module
        from pot.eval.attack_dashboard import create_dash_app
        
        # Create Dash app
        app = create_dash_app(results_dir)
        
        # Run server
        click.echo(f"Dashboard running at http://{host}:{port}")
        click.echo("Press Ctrl+C to stop")
        
        app.run_server(
            host=host,
            port=port,
            debug=debug
        )
        
    except ImportError:
        # Fallback to static dashboard
        click.echo("Interactive dashboard not available. Generating static version...")
        
        dashboard = AttackMetricsDashboard(results_dir)
        output_file = Path(results_dir) / "dashboard.html"
        dashboard.create_dashboard(str(output_file))
        
        click.echo(f"Static dashboard saved to {output_file}")
        click.echo("Open in your browser to view")
        
    except Exception as e:
        click.echo(f"Error launching dashboard: {e}", err=True)
        logger.exception("Dashboard launch failed")
        sys.exit(1)


# Helper functions

def load_model(model_path: str, device: str) -> nn.Module:
    """Load model from checkpoint."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif 'model' in checkpoint:
        model = checkpoint['model']
    elif 'state_dict' in checkpoint:
        # Need to instantiate model architecture
        # This would need to be customized based on your models
        raise NotImplementedError("Model architecture needed for state_dict loading")
    else:
        model = checkpoint
    
    model.to(device)
    model.eval()
    return model


def prepare_data(data_path: Optional[str], batch_size: int, 
                num_samples: int) -> DataLoader:
    """Prepare data loader for evaluation."""
    if data_path and Path(data_path).exists():
        # Load real data
        # This would need to be customized based on your data format
        pass
    else:
        # Create synthetic data for testing
        data = torch.randn(num_samples, 3, 224, 224)
        labels = torch.randint(0, 10, (num_samples,))
        dataset = TensorDataset(data, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def get_attack_suite(suite_name: str, config: Dict[str, Any]):
    """Get appropriate attack suite."""
    if suite_name == 'standard':
        return StandardAttackSuite()
    elif suite_name == 'adaptive':
        return AdaptiveAttackSuite(
            population_size=config.get('population_size', 20)
        )
    elif suite_name == 'comprehensive':
        return ComprehensiveAttackSuite()
    elif suite_name == 'quick':
        return StandardAttackSuite()  # With limited configs
    else:  # vision
        return StandardAttackSuite()  # With vision configs


def get_quick_configs() -> List[AttackConfig]:
    """Get quick evaluation configs."""
    return [
        AttackConfig(
            name="quick_distillation",
            attack_type="distillation",
            budget={'queries': 100, 'compute_time': 10},
            strength='weak',
            success_metrics={'accuracy_drop': 0.1},
            parameters={'temperature': 3.0, 'epochs': 5}
        ),
        AttackConfig(
            name="quick_compression",
            attack_type="compression",
            budget={'queries': 100, 'compute_time': 10},
            strength='weak',
            success_metrics={'accuracy_drop': 0.2},
            parameters={'pruning_rate': 0.3, 'quantization_bits': 8}
        )
    ]


def evaluate_defenses(model: nn.Module, attack_results: List[Dict],
                     data_loader: DataLoader, device: str) -> List[Dict]:
    """Evaluate defenses against attacks."""
    defense_results = []
    
    # Initialize defense system
    from pot.core.defenses import MockBaseVerifier
    base_verifier = MockBaseVerifier()
    defense_system = IntegratedDefenseSystem(base_verifier)
    
    for attack_result in attack_results:
        # Get sample input
        sample_batch = next(iter(data_loader))
        sample_input = sample_batch[0][:10].to(device)
        
        # Evaluate defense
        defense_result = defense_system.comprehensive_defense(
            sample_input,
            model,
            threat_level=0.5 if attack_result.get('success', False) else 0.2
        )
        
        defense_results.append({
            'detected': not defense_result['final_decision']['verified'],
            'confidence': 1 - defense_result['final_decision']['confidence'],
            'threat_level': defense_result['threat_assessment']['threat_level']
        })
    
    return defense_results


def save_results(results: List[Dict], metrics: Dict, robustness: float,
                output_path: Path):
    """Save attack results to disk."""
    # Save raw results
    with open(output_path / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save metrics
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump({
            'metrics': metrics,
            'robustness_score': robustness,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # Save as CSV for easy analysis
    df = pd.DataFrame(results)
    df.to_csv(output_path / 'results.csv', index=False)


def load_results(results_path: Path) -> Dict[str, Any]:
    """Load results from directory."""
    results = {}
    
    # Load JSON results
    json_files = list(results_path.glob('*.json'))
    for json_file in json_files:
        with open(json_file, 'r') as f:
            results[json_file.stem] = json.load(f)
    
    # Load CSV results
    csv_files = list(results_path.glob('*.csv'))
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        results[csv_file.stem + '_df'] = df
    
    return results


def generate_html_report(results: Dict, results_path: Path) -> str:
    """Generate HTML report."""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Attack Evaluation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            h1 { color: #2c3e50; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .success { color: green; }
            .failure { color: red; }
        </style>
    </head>
    <body>
        <h1>Attack Evaluation Report</h1>
    """
    
    # Add summary
    if 'metrics' in results:
        metrics = results['metrics']
        html += f"""
        <h2>Summary</h2>
        <p>Total Attacks: {metrics.get('total_attacks', 0)}</p>
        <p>Success Rate: {metrics.get('success_rate', 0):.1%}</p>
        <p>Robustness Score: {metrics.get('robustness_score', 0):.1f}/100</p>
        """
    
    # Add results table
    if 'results_df' in results:
        df = results['results_df']
        html += "<h2>Detailed Results</h2>"
        html += df.to_html(classes='results-table', index=False)
    
    html += """
    </body>
    </html>
    """
    
    return html


def generate_markdown_report(results: Dict, results_path: Path) -> str:
    """Generate Markdown report."""
    md = "# Attack Evaluation Report\n\n"
    
    # Add summary
    if 'metrics' in results:
        metrics = results['metrics']
        md += "## Summary\n\n"
        md += f"- Total Attacks: {metrics.get('total_attacks', 0)}\n"
        md += f"- Success Rate: {metrics.get('success_rate', 0):.1%}\n"
        md += f"- Robustness Score: {metrics.get('robustness_score', 0):.1f}/100\n\n"
    
    # Add results table
    if 'results_df' in results:
        df = results['results_df']
        md += "## Detailed Results\n\n"
        md += df.to_markdown(index=False)
    
    return md


def collect_model_stats(model: nn.Module, test_data: torch.Tensor):
    """Collect timing and response statistics."""
    import time
    
    times = []
    responses = []
    
    model.eval()
    with torch.no_grad():
        for i in range(len(test_data)):
            start = time.perf_counter()
            output = model(test_data[i:i+1])
            end = time.perf_counter()
            
            times.append((end - start) * 1000)  # Convert to ms
            responses.append(output.cpu().numpy())
    
    return np.array(times), np.vstack(responses)


def save_detection_results(results: Dict, output_path: str):
    """Save wrapper detection results."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def prepare_benchmark_data(data_config: Dict) -> DataLoader:
    """Prepare data for benchmarking."""
    num_samples = data_config.get('num_samples', 1000)
    batch_size = data_config.get('batch_size', 32)
    
    # Create synthetic data
    data = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(data, labels)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Entry point
def main():
    """Main entry point for CLI."""
    attack_cli()


if __name__ == '__main__':
    main()