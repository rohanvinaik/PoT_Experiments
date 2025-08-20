#!/usr/bin/env python3
"""
Configurable Runtime Black-Box Statistical Identity Validation

This version allows full model configuration through:
1. Command-line arguments
2. Environment variables
3. Configuration files
4. Both local and HuggingFace Hub sources
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pot.core.model_loader import UnifiedModelLoader, ModelSource, ModelConfig
from pot.core.diff_decision import (
    DiffDecisionConfig, 
    TestingMode,
    EnhancedSequentialTester
)
from pot.challenges.kdf_prompts import make_prompt_generator
from pot.scoring.diff_scorer import DifferenceScorer, CorrectedDifferenceScorer
from pot.security.provenance_auditor import ProvenanceAuditor
from pot.core.evidence_logger import EvidenceLogger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ValidationConfig:
    """Configuration for validation run"""
    # Model configuration
    model_a: str
    model_b: str
    model_source: str = "auto"  # auto, local, huggingface
    local_model_base: Optional[str] = None
    hf_cache_dir: Optional[str] = None
    use_hf_token: bool = False
    
    # Model loading parameters
    torch_dtype: str = "auto"  # auto, float16, float32, bfloat16
    device_map: Optional[str] = None  # auto, cpu, cuda, mps
    trust_remote_code: bool = False
    
    # Test configuration
    test_mode: str = "adaptive"  # adaptive, quick_gate, audit_grade
    n_queries: int = 12
    positions_per_prompt: int = 32
    
    # Statistical parameters
    confidence: float = 0.95
    gamma: float = 0.02
    delta_star: float = 0.08
    epsilon_diff: float = 0.25
    
    # Output configuration
    output_results: bool = True
    output_dir: str = "experimental_results"
    verbose: bool = False
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ValidationConfig":
        """Create config from command-line arguments"""
        return cls(
            model_a=args.model_a,
            model_b=args.model_b,
            model_source=args.model_source,
            local_model_base=args.local_model_base,
            hf_cache_dir=args.hf_cache_dir,
            use_hf_token=args.use_hf_token,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            trust_remote_code=args.trust_remote_code,
            test_mode=args.test_mode,
            n_queries=args.n_queries,
            positions_per_prompt=args.positions_per_prompt,
            confidence=args.confidence,
            gamma=args.gamma,
            delta_star=args.delta_star,
            epsilon_diff=args.epsilon_diff,
            output_results=args.output_results,
            output_dir=args.output_dir,
            verbose=args.verbose
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> "ValidationConfig":
        """Load config from JSON file"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, json_path: str):
        """Save config to JSON file"""
        with open(json_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)


def run_validation(config: ValidationConfig) -> Dict[str, Any]:
    """
    Run validation with given configuration
    
    Args:
        config: Validation configuration
        
    Returns:
        Dict with validation results
    """
    logger.info("Starting configurable runtime validation")
    logger.info(f"Model A: {config.model_a}")
    logger.info(f"Model B: {config.model_b}")
    logger.info(f"Source: {config.model_source}")
    
    # Initialize model loader
    loader = UnifiedModelLoader(
        local_base=config.local_model_base,
        cache_dir=config.hf_cache_dir,
        default_source=ModelSource[config.model_source.upper()],
        use_auth_token=os.environ.get("HF_TOKEN") if config.use_hf_token else None
    )
    
    # Show available models if verbose
    if config.verbose:
        available = loader.list_available_models()
        logger.info(f"Available local models: {list(available['local'].keys())}")
        logger.info(f"Known HuggingFace models: {len(available['huggingface'])} models")
    
    # Load models
    logger.info("Loading models...")
    try:
        # Create model configs
        model_a_config = ModelConfig(
            name=config.model_a,
            source=ModelSource[config.model_source.upper()],
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code
        )
        
        model_b_config = ModelConfig(
            name=config.model_b,
            source=ModelSource[config.model_source.upper()],
            torch_dtype=config.torch_dtype,
            device_map=config.device_map,
            trust_remote_code=config.trust_remote_code
        )
        
        # Load models
        start_time = time.time()
        model_a, tokenizer_a = loader.load(config.model_a, config=model_a_config)
        load_time_a = time.time() - start_time
        
        start_time = time.time()
        model_b, tokenizer_b = loader.load(config.model_b, config=model_b_config)
        load_time_b = time.time() - start_time
        
        logger.info(f"Models loaded successfully (A: {load_time_a:.2f}s, B: {load_time_b:.2f}s)")
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return {
            "status": "FAILED",
            "error": str(e),
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    # Configure test based on mode
    if config.test_mode == "adaptive":
        test_config = DiffDecisionConfig(
            mode=TestingMode.QUICK_GATE,
            confidence=config.confidence,
            gamma=config.gamma,
            delta_star=config.delta_star,
            epsilon_diff=config.epsilon_diff,
            n_min=10,
            n_max=config.n_queries * 3,
            positions_per_prompt=config.positions_per_prompt
        )
    elif config.test_mode == "quick_gate":
        test_config = DiffDecisionConfig(mode=TestingMode.QUICK_GATE)
    else:  # audit_grade
        test_config = DiffDecisionConfig(mode=TestingMode.AUDIT_GRADE)
    
    # Initialize components
    prompt_gen = make_prompt_generator(
        prf_key=b"configurable_validation",
        namespace="enhanced:v2"
    )
    
    scorer = CorrectedDifferenceScorer()
    tester = EnhancedSequentialTester(test_config)
    audit_logger = ProvenanceAuditor()
    
    # Run test
    logger.info(f"Running {config.test_mode} test with {config.n_queries} queries...")
    
    results = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "config": asdict(config),
        "models": {
            "model_a": config.model_a,
            "model_b": config.model_b,
            "load_time_a": load_time_a,
            "load_time_b": load_time_b
        },
        "test_parameters": {
            "mode": config.test_mode,
            "n_queries": config.n_queries,
            "positions_per_prompt": config.positions_per_prompt,
            "confidence": config.confidence
        }
    }
    
    # Generate prompts
    prompts = []
    for i in range(config.n_queries):
        prompt = prompt_gen()  # The generator is a callable
        prompts.append(prompt)
    
    # Score batches
    start_time = time.time()
    scores = scorer.score_batch(model_a, model_b, prompts, tokenizer_a, k=config.positions_per_prompt)
    inference_time = time.time() - start_time
    
    # Update tester with scores
    for score in scores:
        tester.update(score)
        audit_logger.add_training_event({"type": "score_update", "score": score})
    
    # Get decision
    should_stop, decision_info = tester.should_stop()
    state = tester.get_state()
    
    # Package results
    results["statistical_results"] = {
        "decision": decision_info.get("decision", "UNDECIDED") if decision_info else "UNDECIDED",
        "n_used": state["n"],
        "mean_diff": state["mean"],
        "ci_lower": state["ci"][0],
        "ci_upper": state["ci"][1],
        "half_width": state["half_width"],
        "effect_size": abs(state["mean"]),
        "rule_fired": decision_info.get("rule", "Unknown") if decision_info else "None",
        "confidence": config.confidence
    }
    
    results["timing"] = {
        "t_load_a": load_time_a,
        "t_load_b": load_time_b,
        "t_infer_total": inference_time,
        "t_per_query": inference_time / config.n_queries if config.n_queries > 0 else 0,
        "t_total": load_time_a + load_time_b + inference_time
    }
    
    results["audit"] = {
        "merkle_root": audit_logger.get_merkle_root(),
        "n_events": len(audit_logger.events)
    }
    
    results["status"] = "SUCCESS"
    
    # Log to evidence dashboard
    evidence_logger = EvidenceLogger()
    evidence_logger.log_validation_run({
        "test_type": "configurable_runtime_validation",
        "statistical_results": results["statistical_results"],
        "timing_data": results["timing"],
        "models_tested": {
            "model_a": config.model_a,
            "model_b": config.model_b
        }
    })
    
    # Save results if requested
    if config.output_results:
        os.makedirs(config.output_dir, exist_ok=True)
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_file = f"{config.output_dir}/configurable_validation_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
    
    # Display summary
    print(f"\n📊 Validation Results:")
    print(f"   Decision: {results['statistical_results']['decision']}")
    print(f"   Mean difference: {results['statistical_results']['mean_diff']:.6f}")
    print(f"   Confidence interval: [{results['statistical_results']['ci_lower']:.6f}, {results['statistical_results']['ci_upper']:.6f}]")
    print(f"   Queries used: {results['statistical_results']['n_used']}")
    print(f"   Time per query: {results['timing']['t_per_query']:.3f}s")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Configurable Runtime Black-Box Statistical Identity Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use local models
  %(prog)s --model-a gpt2 --model-b distilgpt2 --model-source local
  
  # Use HuggingFace models
  %(prog)s --model-a gpt2 --model-b distilgpt2 --model-source huggingface
  
  # Use large models with fp16
  %(prog)s --model-a mistral --model-b zephyr --torch-dtype float16
  
  # Load from config file
  %(prog)s --config validation_config.json
  
Environment Variables:
  HF_TOKEN: HuggingFace authentication token for private models
  LOCAL_MODEL_BASE: Default base directory for local models
        """
    )
    
    # Model selection
    parser.add_argument("--model-a", default="gpt2",
                       help="First model to compare")
    parser.add_argument("--model-b", default="distilgpt2",
                       help="Second model to compare")
    parser.add_argument("--model-source", choices=["auto", "local", "huggingface"],
                       default="auto",
                       help="Source for loading models (default: auto)")
    
    # Model loading configuration
    parser.add_argument("--local-model-base",
                       default=os.environ.get("LOCAL_MODEL_BASE", "/Users/rohanvinaik/LLM_Models"),
                       help="Base directory for local models")
    parser.add_argument("--hf-cache-dir",
                       help="Cache directory for HuggingFace models")
    parser.add_argument("--use-hf-token", action="store_true",
                       help="Use HF_TOKEN environment variable for authentication")
    parser.add_argument("--torch-dtype", choices=["auto", "float16", "float32", "bfloat16"],
                       default="auto",
                       help="Torch dtype for model loading")
    parser.add_argument("--device-map",
                       help="Device map for model loading (auto, cpu, cuda, mps)")
    parser.add_argument("--trust-remote-code", action="store_true",
                       help="Trust remote code when loading models")
    
    # Test configuration
    parser.add_argument("--test-mode", choices=["adaptive", "quick_gate", "audit_grade"],
                       default="adaptive",
                       help="Testing mode")
    parser.add_argument("--n-queries", type=int, default=12,
                       help="Number of queries to use")
    parser.add_argument("--positions-per-prompt", type=int, default=32,
                       help="Positions per prompt (K)")
    
    # Statistical parameters
    parser.add_argument("--confidence", type=float, default=0.95,
                       help="Confidence level for testing")
    parser.add_argument("--gamma", type=float, default=0.02,
                       help="Equivalence threshold gamma")
    parser.add_argument("--delta-star", type=float, default=0.08,
                       help="Minimum effect size delta*")
    parser.add_argument("--epsilon-diff", type=float, default=0.25,
                       help="Relative margin of error epsilon_diff")
    
    # Output configuration
    parser.add_argument("--output-results", action="store_true", default=True,
                       help="Save results to file")
    parser.add_argument("--output-dir", default="experimental_results",
                       help="Output directory for results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    # Config file option
    parser.add_argument("--config", help="Load configuration from JSON file")
    parser.add_argument("--save-config", help="Save configuration to JSON file")
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = ValidationConfig.from_json(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = ValidationConfig.from_args(args)
    
    # Save config if requested
    if args.save_config:
        config.to_json(args.save_config)
        logger.info(f"Saved configuration to {args.save_config}")
    
    # Set logging level
    if config.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run validation
    results = run_validation(config)
    
    # Exit with appropriate code
    if results["status"] == "SUCCESS":
        decision = results["statistical_results"]["decision"]
        if decision == "SAME":
            sys.exit(0)
        elif decision == "DIFFERENT":
            sys.exit(1)
        else:  # UNDECIDED
            sys.exit(2)
    else:
        sys.exit(3)


if __name__ == "__main__":
    main()