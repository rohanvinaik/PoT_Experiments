#!/usr/bin/env python3
"""
Command Line Interface for Language Model Verification
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import time

try:
    from .lm_config import LMVerifierConfig, PresetConfigs, load_config
    from .verifier import LMVerifier
except ImportError:
    # Handle standalone execution
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from pot.lm.lm_config import LMVerifierConfig, PresetConfigs, load_config
    from pot.lm.verifier import LMVerifier


@click.group()
def cli():
    """Language Model Verification CLI"""
    pass


@cli.command()
@click.option('--model', '-m', required=True, help='Model name or path')
@click.option('--config', '-c', help='Configuration file or preset name')
@click.option('--num-challenges', '-n', type=int, help='Number of challenges (overrides config)')
@click.option('--method', type=click.Choice(['sequential', 'batch']), help='Verification method (overrides config)')
@click.option('--output', '-o', default='verification_result.json', help='Output file')
@click.option('--threshold', '-t', type=float, help='Distance threshold (overrides config)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--plot', is_flag=True, help='Generate progress plots')
def verify(model: str, config: Optional[str], num_challenges: Optional[int], 
           method: Optional[str], output: str, threshold: Optional[float],
           verbose: bool, plot: bool):
    """Run LM verification."""
    
    if verbose:
        click.echo(f"Starting verification for model: {model}")
    
    try:
        # Load configuration
        if config:
            lm_config = load_config(config)
            if verbose:
                click.echo(f"Loaded config from: {config}")
        else:
            lm_config = LMVerifierConfig()
            if verbose:
                click.echo("Using default configuration")
        
        # Override config with CLI arguments
        if num_challenges is not None:
            lm_config.num_challenges = num_challenges
        if method is not None:
            lm_config.verification_method = method
        if threshold is not None:
            lm_config.distance_threshold = threshold
        if plot:
            lm_config.plot_progress = True
        
        # Validate configuration
        issues = lm_config.validate()
        if issues:
            click.echo("Configuration errors:", err=True)
            for issue in issues:
                click.echo(f"  - {issue}", err=True)
            sys.exit(1)
        
        if verbose:
            click.echo(f"Configuration: {lm_config.num_challenges} challenges, {lm_config.verification_method} method")
        
        # Load model and tokenizer
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            if verbose:
                click.echo(f"Loading model: {model}")
            
            tokenizer = AutoTokenizer.from_pretrained(model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            model_obj = AutoModelForCausalLM.from_pretrained(model)
            
        except Exception as e:
            click.echo(f"Error loading model: {e}", err=True)
            sys.exit(1)
        
        # Create verifier
        verifier = LMVerifier(
            reference_model=model_obj,
            tokenizer=tokenizer,
            config=lm_config
        )
        
        if verbose:
            click.echo("Running verification...")
        
        # Run verification
        start_time = time.time()
        result = verifier.verify_enhanced()
        duration = time.time() - start_time
        
        # Add metadata
        result['metadata'] = {
            'model_name': model,
            'config_source': config or 'default',
            'cli_version': '1.0',
            'timestamp': time.time(),
            'duration_total': duration
        }
        
        # Save results
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        # Print summary
        verified = result.get('verified', False)
        decision = result.get('decision', 'unknown')
        confidence = result.get('confidence', 0.0)
        method_used = result.get('method', 'unknown')
        num_trials = result.get('num_trials', 0)
        
        click.echo()
        click.echo("=" * 50)
        click.echo("VERIFICATION RESULTS")
        click.echo("=" * 50)
        click.echo(f"Model: {model}")
        click.echo(f"Result: {'✓ GENUINE' if verified else '✗ SUSPICIOUS'}")
        click.echo(f"Decision: {decision.upper()}")
        click.echo(f"Confidence: {confidence:.1%}")
        click.echo(f"Method: {method_used}")
        click.echo(f"Challenges: {num_trials}")
        
        if method_used == 'sequential':
            early_stopped = result.get('early_stopped', False)
            click.echo(f"Early stopping: {early_stopped}")
            if 'success_rate' in result:
                click.echo(f"Success rate: {result['success_rate']:.1%}")
        
        click.echo(f"Duration: {duration:.2f}s")
        click.echo(f"Results saved to: {output_path}")
        
        # Exit with appropriate code
        sys.exit(0 if verified else 1)
        
    except Exception as e:
        click.echo(f"Verification failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--preset', type=click.Choice(['quick_test', 'standard_verification', 'comprehensive_verification', 'batch_verification', 'high_security']), help='Preset configuration')
@click.option('--output', '-o', default='config.yaml', help='Output configuration file')
def create_config(preset: Optional[str], output: str):
    """Create a configuration file."""
    
    if preset:
        config = getattr(PresetConfigs, preset)()
        click.echo(f"Created {preset} configuration")
    else:
        config = LMVerifierConfig()
        click.echo("Created default configuration")
    
    # Save configuration
    config.save_to_file(output)
    click.echo(f"Configuration saved to: {output}")


@cli.command()
@click.option('--config', '-c', help='Configuration file to validate')
def validate_config(config: Optional[str]):
    """Validate a configuration file."""
    
    try:
        if config:
            lm_config = LMVerifierConfig.from_file(config)
            click.echo(f"Validating config: {config}")
        else:
            lm_config = LMVerifierConfig()
            click.echo("Validating default configuration")
        
        issues = lm_config.validate()
        
        if not issues:
            click.echo("✓ Configuration is valid")
        else:
            click.echo(f"✗ Configuration has {len(issues)} issues:")
            for issue in issues:
                click.echo(f"  - {issue}")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"Error validating configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
def list_presets():
    """List available configuration presets."""
    
    click.echo("Available Configuration Presets:")
    click.echo("=" * 40)
    
    presets = [
        ('quick_test', 'Quick test for development (5 challenges)'),
        ('standard_verification', 'Standard verification (25 challenges)'),
        ('comprehensive_verification', 'Comprehensive verification (50 challenges)'),
        ('batch_verification', 'Batch verification without early stopping'),
        ('high_security', 'High security verification (100 challenges)')
    ]
    
    for name, description in presets:
        preset_config = getattr(PresetConfigs, name)()
        click.echo(f"{name:25s} - {description}")
        click.echo(f"{'':25s}   {preset_config.num_challenges} challenges, {preset_config.verification_method} method")
        click.echo()


@cli.command()
@click.argument('config_file')
def show_config(config_file: str):
    """Show configuration details."""
    
    try:
        config = LMVerifierConfig.from_file(config_file)
        
        click.echo(f"Configuration: {config_file}")
        click.echo("=" * 40)
        
        # Group settings
        groups = {
            'Model Settings': ['model_name', 'device'],
            'Challenge Settings': ['num_challenges', 'challenge_types', 'difficulty_curve'],
            'Verification Settings': ['verification_method', 'distance_metric', 'distance_threshold'],
            'Sequential Testing': ['sprt_alpha', 'sprt_beta', 'sprt_p0', 'sprt_p1', 'max_trials', 'min_trials'],
            'Fuzzy Matching': ['fuzzy_threshold', 'fuzzy_method'],
            'Hash Settings': ['hash_type', 'hash_size']
        }
        
        config_dict = config.to_dict()
        
        for group_name, keys in groups.items():
            click.echo(f"\n{group_name}:")
            for key in keys:
                if key in config_dict:
                    value = config_dict[key]
                    click.echo(f"  {key}: {value}")
        
    except Exception as e:
        click.echo(f"Error reading configuration: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--model1', '-m1', required=True, help='First model')
@click.option('--model2', '-m2', required=True, help='Second model')
@click.option('--config', '-c', help='Configuration file or preset')
@click.option('--output', '-o', default='comparison_result.json', help='Output file')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def compare(model1: str, model2: str, config: Optional[str], output: str, verbose: bool):
    """Compare two models using the same verification protocol."""
    
    if verbose:
        click.echo(f"Comparing models: {model1} vs {model2}")
    
    try:
        # Load configuration
        if config:
            lm_config = load_config(config)
        else:
            lm_config = LMVerifierConfig()
        
        results = {}
        
        # Verify both models
        for model_name in [model1, model2]:
            if verbose:
                click.echo(f"Verifying {model_name}...")
            
            # Load model
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model_obj = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create verifier
            verifier = LMVerifier(
                reference_model=model_obj,
                tokenizer=tokenizer,
                config=lm_config
            )
            
            # Run verification
            result = verifier.verify_enhanced()
            results[model_name] = result
        
        # Create comparison
        comparison = {
            'model1': model1,
            'model2': model2,
            'results': results,
            'summary': {
                'model1_verified': results[model1].get('verified', False),
                'model2_verified': results[model2].get('verified', False),
                'model1_confidence': results[model1].get('confidence', 0.0),
                'model2_confidence': results[model2].get('confidence', 0.0),
                'better_model': None
            },
            'config': lm_config.to_dict(),
            'timestamp': time.time()
        }
        
        # Determine better model
        conf1 = comparison['summary']['model1_confidence']
        conf2 = comparison['summary']['model2_confidence']
        
        if conf1 > conf2:
            comparison['summary']['better_model'] = model1
        elif conf2 > conf1:
            comparison['summary']['better_model'] = model2
        else:
            comparison['summary']['better_model'] = 'tie'
        
        # Save results
        with open(output, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        # Print summary
        click.echo()
        click.echo("COMPARISON RESULTS")
        click.echo("=" * 40)
        click.echo(f"Model 1: {model1}")
        click.echo(f"  Verified: {'✓' if comparison['summary']['model1_verified'] else '✗'}")
        click.echo(f"  Confidence: {conf1:.1%}")
        click.echo()
        click.echo(f"Model 2: {model2}")
        click.echo(f"  Verified: {'✓' if comparison['summary']['model2_verified'] else '✗'}")
        click.echo(f"  Confidence: {conf2:.1%}")
        click.echo()
        click.echo(f"Better model: {comparison['summary']['better_model']}")
        click.echo(f"Results saved to: {output}")
        
    except Exception as e:
        click.echo(f"Comparison failed: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    cli()