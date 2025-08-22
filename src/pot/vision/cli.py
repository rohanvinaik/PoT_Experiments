"""
Vision Verifier Command Line Interface
Provides CLI commands for vision model verification.
"""

import click
import torch
import json
import sys
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

# Import vision components with fallbacks
try:
    from pot.vision.verifier import VisionVerifier, EnhancedVisionVerifier
    from pot.vision.vision_config import VisionVerifierConfig, VisionConfigPresets, load_config
    VISION_AVAILABLE = True
except ImportError as e:
    VISION_AVAILABLE = False
    print(f"Vision components not available: {e}")

# Optional imports
try:
    import torchvision.models as models
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_model(model_name_or_path: str, device: str = "cpu") -> torch.nn.Module:
    """Load model from name or path."""
    if Path(model_name_or_path).exists():
        # Load from file
        try:
            model = torch.load(model_name_or_path, map_location=device)
            click.echo(f"Loaded model from {model_name_or_path}")
            return model
        except Exception as e:
            raise click.ClickException(f"Failed to load model from {model_name_or_path}: {e}")
    
    elif TORCHVISION_AVAILABLE and hasattr(models, model_name_or_path):
        # Load from torchvision
        try:
            model = getattr(models, model_name_or_path)(pretrained=True)
            click.echo(f"Loaded {model_name_or_path} from torchvision")
            return model.to(device)
        except Exception as e:
            raise click.ClickException(f"Failed to load {model_name_or_path} from torchvision: {e}")
    
    else:
        raise click.ClickException(
            f"Model '{model_name_or_path}' not found. "
            f"Provide a valid file path or torchvision model name."
        )


def save_results(results: Dict[str, Any], output_path: str, format: str = "json"):
    """Save verification results."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == "json":
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    elif format.lower() == "yaml" and YAML_AVAILABLE:
        with open(output_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False, indent=2)
    else:
        raise click.ClickException(f"Unsupported output format: {format}")
    
    click.echo(f"Results saved to {output_path}")


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--log-level', default='INFO', help='Logging level')
@click.pass_context
def cli(ctx, verbose: bool, log_level: str):
    """Vision Model Verification CLI."""
    if not VISION_AVAILABLE:
        raise click.ClickException(
            "Vision verification components are not available. "
            "Please install required dependencies."
        )
    
    ctx.ensure_object(dict)
    ctx.obj['verbose'] = verbose
    setup_logging(log_level)


@cli.command()
@click.option('--model', '-m', required=True, help='Model name or path')
@click.option('--config', '-c', help='Configuration file path')
@click.option('--num-challenges', '-n', default=10, help='Number of challenges')
@click.option('--challenge-types', '-t', multiple=True, 
              default=['frequency', 'texture'],
              help='Types of challenges to use')
@click.option('--method', default='sequential', 
              type=click.Choice(['sequential', 'batch']),
              help='Verification method')
@click.option('--device', default='auto', help='Device to use (auto, cpu, cuda)')
@click.option('--output', '-o', default='vision_verification_result.json', 
              help='Output file path')
@click.option('--output-format', default='json',
              type=click.Choice(['json', 'yaml']),
              help='Output format')
@click.option('--preset', type=click.Choice(['quick', 'standard', 'comprehensive', 'research', 'production']),
              help='Use predefined configuration preset')
@click.option('--threshold', default=0.2, type=float, help='Distance threshold')
@click.option('--confidence', default=0.95, type=float, help='Confidence threshold')
@click.option('--seed', default=42, type=int, help='Random seed')
@click.pass_context
def verify(ctx, model: str, config: Optional[str], num_challenges: int, 
           challenge_types: List[str], method: str, device: str, 
           output: str, output_format: str, preset: Optional[str],
           threshold: float, confidence: float, seed: int):
    """Run vision model verification."""
    
    # Set random seed
    torch.manual_seed(seed)
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    click.echo(f"Using device: {device}")
    
    # Load configuration
    if preset:
        if preset == 'quick':
            verify_config = VisionConfigPresets.quick_verification()
        elif preset == 'standard':
            verify_config = VisionConfigPresets.standard_verification()
        elif preset == 'comprehensive':
            verify_config = VisionConfigPresets.comprehensive_verification()
        elif preset == 'research':
            verify_config = VisionConfigPresets.research_verification()
        elif preset == 'production':
            verify_config = VisionConfigPresets.production_verification()
        
        click.echo(f"Using preset configuration: {preset}")
    
    elif config:
        verify_config = load_config(config)
        click.echo(f"Loaded configuration from {config}")
    
    else:
        # Create configuration from CLI arguments
        verify_config = VisionVerifierConfig(
            model_name=model,
            device=device,
            num_challenges=num_challenges,
            challenge_types=list(challenge_types),
            verification_method=method,
            distance_threshold=threshold,
            confidence_threshold=confidence,
            random_seed=seed,
            verbose=ctx.obj['verbose']
        )
    
    # Override device and model name
    verify_config.device = device
    verify_config.model_name = model
    
    try:
        # Load model
        model_instance = load_model(model, device)
        
        # Create verifier
        try:
            verifier = EnhancedVisionVerifier(model_instance, verify_config.to_dict())
            click.echo("Using EnhancedVisionVerifier")
        except:
            # Fallback to basic verifier
            verifier = VisionVerifier(model_instance, verify_config.to_dict())
            click.echo("Using basic VisionVerifier")
        
        # Run verification
        if ctx.obj['verbose']:
            click.echo("Starting verification...")
            
        with click.progressbar(length=num_challenges, 
                              label='Verifying') as bar:
            result = verifier.verify_session(
                num_challenges=num_challenges,
                challenge_types=list(challenge_types),
                progress_callback=lambda: bar.update(1)
            )
        
        # Add configuration to results
        result['configuration'] = verify_config.to_dict()
        result['model_info'] = {
            'name': model,
            'device': device,
            'parameters': sum(p.numel() for p in model_instance.parameters())
        }
        
        # Save results
        save_results(result, output, output_format)
        
        # Print summary
        click.echo("\n" + "="*50)
        click.echo("VERIFICATION RESULTS")
        click.echo("="*50)
        
        status = "✓ PASSED" if result.get('verified', False) else "✗ FAILED"
        click.echo(f"Status: {status}")
        click.echo(f"Confidence: {result.get('confidence', 0):.3f}")
        click.echo(f"Challenges completed: {result.get('num_challenges', 0)}")
        
        if 'early_stopped' in result:
            early_stop = "Yes" if result['early_stopped'] else "No"
            click.echo(f"Early stopping: {early_stop}")
        
        # Print per-challenge results if available
        if 'results' in result and ctx.obj['verbose']:
            click.echo("\nChallenge Details:")
            for i, res in enumerate(result['results']):
                challenge_type = res.get('challenge_type', 'unknown')
                success = "✓" if res.get('success', False) else "✗"
                distance = res.get('distance', 0)
                click.echo(f"  {i+1:2d}. {challenge_type:10s}: {success} (distance: {distance:.4f})")
        
        # Print performance metrics if available
        if 'performance' in result:
            perf = result['performance']
            click.echo(f"\nPerformance:")
            click.echo(f"  Total time: {perf.get('total_time', 0):.2f}s")
            click.echo(f"  Avg time per challenge: {perf.get('avg_time_per_challenge', 0):.2f}s")
        
        click.echo("="*50)
        
        # Exit with appropriate code
        sys.exit(0 if result.get('verified', False) else 1)
        
    except Exception as e:
        raise click.ClickException(f"Verification failed: {e}")


@cli.command()
@click.option('--output', '-o', default='vision_config_template.yaml',
              help='Output configuration file')
@click.option('--format', default='yaml',
              type=click.Choice(['yaml', 'json']),
              help='Configuration format')
@click.option('--preset', type=click.Choice(['quick', 'standard', 'comprehensive', 'research', 'production']),
              help='Use predefined configuration preset')
def create_config(output: str, format: str, preset: Optional[str]):
    """Create a configuration template file."""
    
    if preset:
        if preset == 'quick':
            config = VisionConfigPresets.quick_verification()
        elif preset == 'standard':
            config = VisionConfigPresets.standard_verification()
        elif preset == 'comprehensive':
            config = VisionConfigPresets.comprehensive_verification()
        elif preset == 'research':
            config = VisionConfigPresets.research_verification()
        elif preset == 'production':
            config = VisionConfigPresets.production_verification()
        
        click.echo(f"Creating {preset} configuration template")
    else:
        config = VisionVerifierConfig()
        click.echo("Creating default configuration template")
    
    # Save configuration
    if format == 'yaml':
        if not YAML_AVAILABLE:
            raise click.ClickException("PyYAML is required for YAML format")
        config.save_yaml(output)
    else:
        config.save_json(output)
    
    click.echo(f"Configuration template saved to {output}")


@cli.command()
@click.argument('config_file')
def validate_config(config_file: str):
    """Validate a configuration file."""
    
    try:
        config = load_config(config_file)
        config._validate_config()
        click.echo(f"✓ Configuration file {config_file} is valid")
    except Exception as e:
        click.echo(f"✗ Configuration file {config_file} is invalid: {e}")
        sys.exit(1)


@cli.command()
@click.option('--model', '-m', required=True, help='Model name or path')
@click.option('--device', default='auto', help='Device to use')
def model_info(model: str, device: str):
    """Get information about a model."""
    
    # Determine device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        # Load model
        model_instance = load_model(model, device)
        
        # Get model info
        total_params = sum(p.numel() for p in model_instance.parameters())
        trainable_params = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        
        click.echo(f"Model: {model}")
        click.echo(f"Architecture: {model_instance.__class__.__name__}")
        click.echo(f"Total parameters: {total_params:,}")
        click.echo(f"Trainable parameters: {trainable_params:,}")
        click.echo(f"Device: {device}")
        
        # Test forward pass
        try:
            with torch.no_grad():
                test_input = torch.randn(1, 3, 224, 224).to(device)
                output = model_instance(test_input)
                click.echo(f"Output shape: {output.shape}")
                click.echo("✓ Model forward pass successful")
        except Exception as e:
            click.echo(f"✗ Model forward pass failed: {e}")
        
    except Exception as e:
        raise click.ClickException(f"Failed to load model: {e}")


@cli.command()
def list_presets():
    """List available configuration presets."""
    
    presets = {
        'quick': 'Quick verification with minimal challenges',
        'standard': 'Standard verification configuration',
        'comprehensive': 'Comprehensive verification with all challenge types',
        'research': 'Research-grade verification with extensive probing',
        'production': 'Production-ready verification optimized for speed'
    }
    
    click.echo("Available configuration presets:")
    for name, description in presets.items():
        click.echo(f"  {name:15s}: {description}")


@cli.command()
@click.option('--results-file', '-r', required=True, help='Verification results file')
@click.option('--format', default='table', 
              type=click.Choice(['table', 'json', 'summary']),
              help='Output format')
def show_results(results_file: str, format: str):
    """Display verification results."""
    
    try:
        with open(results_file, 'r') as f:
            if results_file.endswith('.json'):
                results = json.load(f)
            elif results_file.endswith('.yaml') and YAML_AVAILABLE:
                results = yaml.safe_load(f)
            else:
                raise click.ClickException("Unsupported results file format")
        
        if format == 'json':
            click.echo(json.dumps(results, indent=2, default=str))
        
        elif format == 'summary':
            click.echo(f"Verification: {'PASSED' if results.get('verified') else 'FAILED'}")
            click.echo(f"Confidence: {results.get('confidence', 0):.3f}")
            click.echo(f"Challenges: {results.get('num_challenges', 0)}")
            
        elif format == 'table':
            # Table format
            click.echo("Verification Results")
            click.echo("-" * 40)
            click.echo(f"{'Status:':<15} {'✓ PASSED' if results.get('verified') else '✗ FAILED'}")
            click.echo(f"{'Confidence:':<15} {results.get('confidence', 0):.3f}")
            click.echo(f"{'Challenges:':<15} {results.get('num_challenges', 0)}")
            
            if 'results' in results:
                click.echo("\nChallenge Details:")
                click.echo(f"{'#':<3} {'Type':<12} {'Status':<8} {'Distance':<10}")
                click.echo("-" * 35)
                for i, res in enumerate(results['results']):
                    challenge_type = res.get('challenge_type', 'unknown')
                    status = "✓" if res.get('success', False) else "✗"
                    distance = res.get('distance', 0)
                    click.echo(f"{i+1:<3} {challenge_type:<12} {status:<8} {distance:<10.4f}")
        
    except Exception as e:
        raise click.ClickException(f"Failed to read results file: {e}")


if __name__ == '__main__':
    cli()