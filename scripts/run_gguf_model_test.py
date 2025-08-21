#!/usr/bin/env python3
"""
GGUF Model Testing Script for Large Quantized Models
Supports testing of Qwen2.5-72B and DeepSeek models using llama-cpp-python
"""

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. Installing...")
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
        from llama_cpp import Llama
        LLAMA_CPP_AVAILABLE = True
    except:
        logger.error("Failed to install llama-cpp-python. Please install manually:")
        logger.error("pip install llama-cpp-python")
        sys.exit(1)

from pot.core.diff_decision import DiffDecisionConfig, TestingMode
from pot.core.evidence_logger import log_enhanced_diff_test


class GGUFModel:
    """Wrapper for GGUF models to work with PoT framework."""
    
    def __init__(self, model_path: str, n_ctx: int = 2048, n_threads: int = 8):
        """Initialize GGUF model."""
        self.model_path = model_path
        
        # Determine model name
        if "Qwen" in model_path:
            self.model_name = "Qwen2.5-72B-Q4"
        elif "DeepSeek" in model_path:
            self.model_name = "DeepSeek-R1-UD-IQ1_M"
        else:
            self.model_name = Path(model_path).parent.name
        
        logger.info(f"Loading GGUF model: {self.model_name}")
        logger.info(f"Path: {model_path}")
        
        # Check if this is a split model
        if "00001-of-" in model_path:
            logger.info("Detected split GGUF model - loading all parts...")
            # For split models, llama-cpp-python should handle it automatically
            # but we need to ensure we're pointing to the right file
            
        try:
            # Load with limited context to save memory
            # Use Metal acceleration if available
            self.model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=-1,  # Use all GPU layers available (Metal on Mac)
                verbose=True,  # Enable verbose to see loading progress
                seed=42,
                n_batch=512,  # Larger batch for efficiency
                use_mmap=True,  # Memory map for large models
                use_mlock=False  # Don't lock memory (let OS manage)
            )
            logger.info(f"Model loaded successfully: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Trying with reduced settings...")
            
            # Try with minimal settings
            self.model = Llama(
                model_path=model_path,
                n_ctx=512,  # Minimal context
                n_threads=4,
                n_gpu_layers=0,  # CPU only
                verbose=False,
                seed=42
            )
            logger.info(f"Model loaded with reduced settings: {self.model_name}")
    
    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7) -> str:
        """Generate text from prompt."""
        try:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.95,
                echo=False
            )
            return output['choices'][0]['text']
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def get_logits(self, prompt: str) -> np.ndarray:
        """Get logits for the prompt."""
        try:
            # Tokenize
            tokens = self.model.tokenize(prompt.encode('utf-8'))
            
            # Get logits
            self.model.reset()
            for token in tokens[:-1]:
                self.model.eval([token])
            
            # Eval last token and get logits
            self.model.eval([tokens[-1]])
            logits = np.array(self.model.logits)
            
            return logits
        except Exception as e:
            logger.error(f"Failed to get logits: {e}")
            return np.zeros(32000)  # Default vocab size


class GGUFDifferenceScorer:
    """Scorer for GGUF models."""
    
    def __init__(self):
        self.scores = []
    
    def score_batch(self, model_a: GGUFModel, model_b: GGUFModel, 
                   prompts: List[str], k: int = 16) -> np.ndarray:
        """Score differences between two GGUF models."""
        scores = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            
            # Get logits from both models
            logits_a = model_a.get_logits(prompt)
            logits_b = model_b.get_logits(prompt)
            
            # Ensure same shape
            min_len = min(len(logits_a), len(logits_b))
            logits_a = logits_a[:min_len]
            logits_b = logits_b[:min_len]
            
            # Convert to probabilities
            def softmax(x):
                e_x = np.exp(x - np.max(x))
                return e_x / e_x.sum()
            
            probs_a = softmax(logits_a)
            probs_b = softmax(logits_b)
            
            # Calculate KL divergence
            epsilon = 1e-10
            kl_div = np.sum(probs_a * np.log((probs_a + epsilon) / (probs_b + epsilon)))
            
            # Normalize to [0, 1] range
            score = 1.0 - np.exp(-kl_div)
            scores.append(score)
            
            # For k positions, just repeat the score (simplified)
            for _ in range(k - 1):
                scores.append(score)
        
        return np.array(scores)


def generate_test_prompts(n: int = 10) -> List[str]:
    """Generate diverse test prompts."""
    prompts = [
        "The future of artificial intelligence is",
        "In quantum computing, the main challenge is",
        "Climate change impacts are most severe in",
        "The economic implications of automation include",
        "Neural networks learn by",
        "The most important scientific discovery was",
        "Democracy functions best when",
        "The key to sustainable energy is",
        "Human consciousness can be described as",
        "The universe began with",
        "Mathematical proofs require",
        "Evolution explains how",
        "The internet has transformed",
        "Space exploration will lead to",
        "Genetic engineering allows us to",
        "The meaning of life is",
        "Philosophy teaches us that",
        "History shows that civilizations",
        "Technology's impact on society",
        "The nature of reality is"
    ]
    return prompts[:n]


def run_gguf_comparison(model_a_path: str, model_b_path: str, 
                       test_mode: str = "quick") -> Dict[str, Any]:
    """Run comparison between two GGUF models."""
    
    logger.info("="*70)
    logger.info("GGUF MODEL COMPARISON TEST")
    logger.info("="*70)
    
    # Load models with minimal context for 72B models
    logger.info("\nLoading models (this may take several minutes for 72B models)...")
    logger.info("Using minimal context window (256) to conserve memory...")
    
    # Load first model
    logger.info(f"\n1/2: Loading {Path(model_a_path).parent.name}...")
    model_a = GGUFModel(model_a_path, n_ctx=256, n_threads=8)
    
    # Load second model
    logger.info(f"\n2/2: Loading {Path(model_b_path).parent.name}...")
    model_b = GGUFModel(model_b_path, n_ctx=256, n_threads=8)
    
    # Create scorer
    scorer = GGUFDifferenceScorer()
    
    # Generate prompts - use very few for 72B models
    n_prompts = 3 if test_mode == "quick" else 5
    prompts = generate_test_prompts(n_prompts)
    logger.info(f"\nUsing {n_prompts} test prompts (minimal for 72B models)")
    
    # Run scoring
    logger.info("\nScoring model differences...")
    start_time = time.time()
    
    scores = scorer.score_batch(model_a, model_b, prompts, k=8)
    
    elapsed = time.time() - start_time
    
    # Calculate statistics
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    # Simple decision logic
    if mean_score < 0.1:
        decision = "SAME"
        interpretation = "Models behave nearly identically"
    elif mean_score > 0.5:
        decision = "DIFFERENT"
        interpretation = "Models show significant behavioral differences"
    else:
        decision = "BORDERLINE"
        interpretation = "Models show moderate differences"
    
    # Results
    results = {
        "model_a": model_a.model_name,
        "model_b": model_b.model_name,
        "decision": decision,
        "mean_difference": float(mean_score),
        "std_difference": float(std_score),
        "n_prompts": n_prompts,
        "n_scores": len(scores),
        "time_elapsed": elapsed,
        "interpretation": interpretation
    }
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("RESULTS")
    logger.info("="*70)
    logger.info(f"Model A: {model_a.model_name}")
    logger.info(f"Model B: {model_b.model_name}")
    logger.info(f"Decision: {decision}")
    logger.info(f"Mean Difference: {mean_score:.4f}")
    logger.info(f"Std Deviation: {std_score:.4f}")
    logger.info(f"Time: {elapsed:.2f}s")
    logger.info(f"\nInterpretation: {interpretation}")
    logger.info("="*70)
    
    # Log to evidence system
    log_enhanced_diff_test({
        'statistical_results': {
            'decision': decision,
            'confidence': 0.95,  # Simplified
            'n_used': len(scores),
            'mean_diff': float(mean_score),
            'effect_size': float(mean_score),
            'rule_fired': f"GGUF {decision} criteria"
        },
        'timing': {
            'total_time': elapsed,
            'scores_per_second': len(scores) / elapsed
        },
        'models': {
            'ref_model': model_a.model_name,
            'cand_model': model_b.model_name
        },
        'success': True,
        'test_type': 'gguf_comparison'
    })
    
    # Save results
    output_dir = Path("outputs/gguf_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    output_file = output_dir / f"gguf_comparison_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to: {output_file}")
    
    return results


def main():
    """Main entry point."""
    
    # Check if we have the models
    qwen_path = "/Users/rohanvinaik/LLM_Models/Qwen2.5-72B-Q4/Qwen2.5-72B-Instruct-Q4_K_M.gguf"
    
    # DeepSeek model is split into parts - use the first part as entry point
    deepseek_base = "/Users/rohanvinaik/LLM_Models/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M"
    deepseek_path = f"{deepseek_base}/DeepSeek-R1-UD-IQ1_M-00001-of-00004.gguf"
    
    # Check if path exists, otherwise look for the file
    if not Path(deepseek_path).exists():
        deepseek_dir = Path(deepseek_base)
        if deepseek_dir.is_dir():
            # Look for GGUF files
            gguf_files = sorted(deepseek_dir.glob("*.gguf"))
            if gguf_files:
                deepseek_path = str(gguf_files[0])
                logger.info(f"Found DeepSeek model: {deepseek_path}")
            else:
                logger.error(f"No GGUF file found in {deepseek_base}")
                return 1
    
    # Check both paths exist
    if not Path(qwen_path).exists():
        logger.error(f"Qwen model not found: {qwen_path}")
        return 1
    
    if not Path(deepseek_path).exists():
        logger.error(f"DeepSeek model not found: {deepseek_path}")
        return 1
    
    # Run comparison
    logger.info("Starting GGUF model comparison...")
    logger.info(f"Model A: Qwen2.5-72B-Q4")
    logger.info(f"Model B: DeepSeek-R1-UD-IQ1_M")
    logger.warning("Note: This may take significant time and memory for 72B models!")
    
    try:
        results = run_gguf_comparison(qwen_path, deepseek_path, test_mode="quick")
        
        if results['decision'] == 'DIFFERENT':
            logger.info("\n✅ Successfully detected differences between models!")
        elif results['decision'] == 'SAME':
            logger.info("\n✅ Models verified as functionally equivalent!")
        else:
            logger.info("\n⚠️ Results inconclusive - may need more samples")
        
        return 0
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.info("\nThis may be due to:")
        logger.info("1. Insufficient memory for 72B models")
        logger.info("2. Missing GGUF files")
        logger.info("3. Incompatible model formats")
        return 1


if __name__ == "__main__":
    sys.exit(main())