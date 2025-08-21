#!/usr/bin/env python3
"""
Enhanced Statistical Difference Testing CLI

Supports three modes:
1. Quick gate verification
2. Audit grade verification  
3. Calibration for determining γ and δ*
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
from typing import Optional, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pot.core.diff_decision import DiffDecisionConfig, TestingMode
from pot.core.diff_verifier import EnhancedDifferenceVerifier
from pot.core.calibration import ModelCalibrator, load_calibration
from pot.challenges.kdf_prompts import make_prompt_generator
from pot.scoring.diff_scorer import DifferenceScorer, MockScorer
from pot.core.evidence_logger import log_enhanced_diff_test

def setup_logging(verbose: bool):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_model_pairs(pair_strings: List[str]) -> List[Tuple[str, str]]:
    """Parse model pair strings into tuples"""
    pairs = []
    for pair_str in pair_strings:
        parts = pair_str.split(':')
        if len(parts) != 2:
            raise ValueError(f"Invalid pair format: {pair_str}. Use format: model1:model2")
        pairs.append((parts[0], parts[1]))
    return pairs

def run_calibration(args, scorer, prompt_gen, logger):
    """Run calibration mode"""
    logger.info("Starting calibration mode...")
    
    if not args.calibrate_models:
        logger.error("--calibrate-models required for calibration")
        sys.exit(1)
    
    # For testing/demo, use mock models
    if args.use_mock:
        logger.info("Using mock models for calibration demo")
        same_models = [f"mock_model_{i}" for i in range(len(args.calibrate_models))]
        calibrator = ModelCalibrator(
            scorer=MockScorer("same"),
            prompt_generator=prompt_gen,
            n_samples_per_pair=args.n_samples or 50
        )
    else:
        same_models = args.calibrate_models
        calibrator = ModelCalibrator(
            scorer=scorer,
            prompt_generator=prompt_gen,
            n_samples_per_pair=args.n_samples or 50
        )
    
    # Parse near-clone pairs if provided
    near_pairs = None
    if args.calibrate_pairs:
        if args.use_mock:
            # Create mock pairs
            near_pairs = [(f"ref_{i}", f"clone_{i}") for i in range(len(args.calibrate_pairs))]
        else:
            near_pairs = parse_model_pairs(args.calibrate_pairs)
        logger.info(f"Using {len(near_pairs)} near-clone pairs")
    
    # Run calibration
    output_file = Path(args.output_dir) / "calibration.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    result = calibrator.calibrate(
        same_models=same_models,
        near_clone_pairs=near_pairs,
        output_file=str(output_file),
        use_mock=args.use_mock
    )
    
    logger.info(f"Calibration complete. Results saved to {output_file}")
    print(f"\nCalibration successful!")
    print(f"  γ = {result.gamma:.6f}")
    print(f"  δ* = {result.delta_star:.6f}")
    
    return 0

def run_verification(args, scorer, prompt_gen, logger):
    """Run verification mode (quick or audit)"""
    
    # Check required arguments
    if not args.use_mock and (not args.ref_model or not args.cand_model):
        logger.error("Both --ref-model and --cand-model required for verification (or use --use-mock)")
        sys.exit(1)
    
    # Create config based on mode
    if args.mode == "quick":
        config = DiffDecisionConfig(mode=TestingMode.QUICK_GATE)
        logger.info("Using QUICK_GATE mode")
    else:
        config = DiffDecisionConfig(mode=TestingMode.AUDIT_GRADE)
        logger.info("Using AUDIT_GRADE mode")
    
    # Load calibration if provided
    if args.calibration_file:
        try:
            calib = load_calibration(args.calibration_file)
            config.use_calibration = True
            config.gamma = calib.gamma
            config.delta_star = calib.delta_star
            logger.info(f"Loaded calibration: γ={calib.gamma:.6f}, δ*={calib.delta_star:.6f}")
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            sys.exit(1)
    
    # Apply manual overrides
    if args.gamma is not None:
        config.gamma = args.gamma
        logger.info(f"Override γ={args.gamma:.6f}")
    if args.delta_star is not None:
        config.delta_star = args.delta_star
        logger.info(f"Override δ*={args.delta_star:.6f}")
    if args.epsilon_diff is not None:
        config.epsilon_diff = args.epsilon_diff
        logger.info(f"Override ε_diff={args.epsilon_diff:.3f}")
    if args.n_min is not None:
        config.n_min = args.n_min
    if args.n_max is not None:
        config.n_max = args.n_max
    if args.k_positions is not None:
        config.positions_per_prompt = args.k_positions
    
    # Load models or use mocks
    if args.use_mock:
        logger.info("Using mock models for testing")
        from pot.testing.test_models import DeterministicMockModel
        ref_model = DeterministicMockModel("ref_model", seed=42)
        cand_model = DeterministicMockModel("cand_model", seed=43)
        
        # Use mock scorer
        scorer = MockScorer(args.mock_scenario or "same")
    else:
        logger.info(f"Loading models...")
        logger.info(f"  Reference: {args.ref_model}")
        logger.info(f"  Candidate: {args.cand_model}")
        
        try:
            # Attempt to load real models
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            # Determine device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cpu" and torch.backends.mps.is_available():
                device = "mps"
            logger.info(f"Using device: {device}")
            
            # Load models - use float32 for stability with larger models
            # Phi-2, GPT-2-medium and other models need higher precision
            needs_float32 = any(name in args.ref_model.lower() for name in ['phi', 'medium', 'large', 'xl', '7b', '13b'])
            dtype = torch.float32 if (device == "cpu" or needs_float32) else torch.float16
            
            ref_model = AutoModelForCausalLM.from_pretrained(
                args.ref_model,
                torch_dtype=dtype,
                device_map="auto" if device == "cuda" else None
            )
            ref_model.eval()
            if device in ["cpu", "mps"]:
                ref_model = ref_model.to(device)
            
            # Check if candidate model also needs float32
            needs_float32_cand = any(name in args.cand_model.lower() for name in ['phi', 'medium', 'large', 'xl', '7b', '13b'])
            dtype_cand = torch.float32 if (device == "cpu" or needs_float32_cand) else torch.float16
            
            cand_model = AutoModelForCausalLM.from_pretrained(
                args.cand_model,
                torch_dtype=dtype_cand,
                device_map="auto" if device == "cuda" else None
            )
            cand_model.eval()
            if device in ["cpu", "mps"]:
                cand_model = cand_model.to(device)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.ref_model)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            scorer.set_tokenizer(tokenizer)
            
        except ImportError:
            logger.error("Transformers library not available. Use --use-mock for testing.")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            sys.exit(1)
    
    # Create verifier
    verifier = EnhancedDifferenceVerifier(
        score_fn=scorer,
        prompt_generator=prompt_gen,
        config=config,
        verbose=args.verbose
    )
    
    # Run verification
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting {config.mode.value} verification...")
    start_time = time.time()
    
    report = verifier.verify_difference(
        ref_model,
        cand_model,
        output_dir=output_dir,
        save_prompts=args.save_prompts
    )
    
    elapsed = time.time() - start_time
    
    # Display results
    print("\n" + "="*70)
    print(f"{config.mode.value.upper()} VERIFICATION RESULTS")
    print("="*70)
    print(f"Decision: {report['results']['decision']}")
    print(f"Confidence: {config.confidence*100:.0f}%")
    print(f"Samples used: {report['results']['n_used']} (n_eff={report['results']['n_eff']})")
    print(f"Mean difference: {report['results']['mean']:.6f}")
    print(f"CI: [{report['results']['ci'][0]:.6f}, {report['results']['ci'][1]:.6f}]")
    print(f"Half-width: {report['results']['half_width']:.6f}")
    
    if report['results']['decision'] == "SAME":
        print(f"γ (equivalence band): {config.gamma:.6f}")
        print(f"Precision η·γ: {config.eta * config.gamma:.6f}")
    elif report['results']['decision'] == "DIFFERENT":
        print(f"δ* (min effect size): {config.delta_star:.6f}")
        if report['results'].get('rme') is not None:
            print(f"RME: {report['results']['rme']:.3f} (target: {config.epsilon_diff:.3f})")
    
    print(f"\nTime: {elapsed:.2f}s ({report['timing']['scores_per_second']:.1f} scores/sec)")
    
    print("\nInterpretation:")
    print(report['interpretation'])
    
    if report.get('suggestions'):
        print("\nSuggestions:")
        for s in report['suggestions'][:3]:
            print(f"  • {s}")
    
    print("\nNext Steps:")
    for step in report.get('next_steps', [])[:3]:
        print(f"  • {step}")
    
    print("="*70)
    
    # Save summary
    summary_file = output_dir / f"cli_summary_{int(time.time())}.txt"
    with open(summary_file, 'w') as f:
        f.write(f"CLI Verification Summary\n")
        f.write(f"Mode: {config.mode.value}\n")
        f.write(f"Decision: {report['results']['decision']}\n")
        f.write(f"Mean: {report['results']['mean']:.6f}\n")
        f.write(f"CI: [{report['results']['ci'][0]:.6f}, {report['results']['ci'][1]:.6f}]\n")
        f.write(f"Time: {elapsed:.2f}s\n")
    
    print(f"\nResults saved to: {output_dir}")
    
    # Log to evidence system
    log_enhanced_diff_test({
        'statistical_results': {
            'decision': report['results']['decision'],
            'confidence': config.confidence,
            'n_used': report['results']['n_used'],
            'mean_diff': report['results']['mean'],
            'ci_lower': report['results']['ci'][0],
            'ci_upper': report['results']['ci'][1],
            'half_width': report['results']['half_width'],
            'effect_size': abs(report['results']['mean']),
            'rule_fired': f"{report['results']['decision']} criteria met"
        },
        'timing': {
            'total_time': elapsed,
            'scores_per_second': report['timing']['scores_per_second']
        },
        'models': {
            'ref_model': args.ref_model or 'mock_ref',
            'cand_model': args.cand_model or 'mock_cand'
        },
        'success': True,
        'test_type': f"enhanced_diff_{config.mode.value}"
    })
    
    # Return appropriate exit code
    if report['results']['decision'] == "SAME":
        return 0  # Models are equivalent
    elif report['results']['decision'] == "DIFFERENT":
        return 1  # Models are different
    else:
        return 2  # Undecided

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced statistical difference testing with calibration support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibration mode
  %(prog)s --mode calibrate --calibrate-models model1 model2 model3 \\
           --calibrate-pairs ref1:clone1 ref2:clone2 --prf-key deadbeef

  # Quick verification
  %(prog)s --mode quick --ref-model gpt2 --cand-model distilgpt2 \\
           --prf-key deadbeef

  # Audit verification with calibration
  %(prog)s --mode audit --ref-model model_v1 --cand-model model_v2 \\
           --calibration-file calibration.json --prf-key deadbeef
           
  # Testing with mock models
  %(prog)s --mode quick --use-mock --prf-key deadbeef
        """
    )
    
    # Mode selection
    parser.add_argument("--mode", 
                       choices=["quick", "audit", "calibrate"],
                       default="audit",
                       help="Testing mode: quick gate, audit grade, or calibration")
    
    # Models for verification
    parser.add_argument("--ref-model", 
                       help="Reference model path or name")
    parser.add_argument("--cand-model", 
                       help="Candidate model path or name")
    
    # Calibration mode arguments
    parser.add_argument("--calibrate-models", 
                       nargs="+",
                       help="Models for same-model calibration")
    parser.add_argument("--calibrate-pairs", 
                       nargs="+",
                       help="Model pairs for near-clone calibration (format: model1:model2)")
    parser.add_argument("--n-samples",
                       type=int,
                       help="Number of samples per model pair for calibration")
    
    # Use existing calibration
    parser.add_argument("--calibration-file",
                       help="Load calibration from file")
    
    # Manual parameter overrides
    parser.add_argument("--gamma", 
                       type=float,
                       help="Override equivalence band γ")
    parser.add_argument("--delta-star", 
                       type=float,
                       help="Override minimum effect size δ*")
    parser.add_argument("--epsilon-diff", 
                       type=float,
                       help="Override RME target ε_diff")
    
    # Sampling overrides
    parser.add_argument("--n-min", 
                       type=int,
                       help="Override minimum samples")
    parser.add_argument("--n-max", 
                       type=int,
                       help="Override maximum samples")
    parser.add_argument("--k-positions", 
                       type=int,
                       help="Override positions per prompt (K)")
    
    # Testing/demo options
    parser.add_argument("--use-mock",
                       action="store_true",
                       help="Use mock models for testing/demo")
    parser.add_argument("--mock-scenario",
                       choices=["identical", "same", "different", "borderline"],
                       default="same",
                       help="Scenario for mock testing")
    
    # Other parameters
    parser.add_argument("--prf-key", 
                       required=True,
                       help="PRF key for prompt generation (hex string)")
    parser.add_argument("--output-dir", 
                       default="outputs/enhanced_diff",
                       help="Output directory for results")
    parser.add_argument("--save-prompts",
                       action="store_true",
                       help="Save prompts used in verification")
    parser.add_argument("--verbose", 
                       action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Parse PRF key
    try:
        prf_key = bytes.fromhex(args.prf_key)
        if len(prf_key) < 16:
            logger.warning("PRF key should be at least 16 bytes (32 hex chars)")
    except ValueError:
        logger.error("Invalid PRF key format. Please provide a hex string.")
        sys.exit(1)
    
    # Create prompt generator and scorer
    prompt_gen = make_prompt_generator(prf_key, "enhanced:v2")
    scorer = DifferenceScorer(method="delta_ce")
    
    # Route to appropriate mode
    if args.mode == "calibrate":
        return run_calibration(args, scorer, prompt_gen, logger)
    else:
        return run_verification(args, scorer, prompt_gen, logger)

if __name__ == "__main__":
    sys.exit(main())