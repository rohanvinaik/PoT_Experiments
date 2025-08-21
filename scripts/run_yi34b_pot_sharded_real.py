#!/usr/bin/env python3
"""
Full PoT pipeline with sharding for Yi-34B models using the REAL PoT framework.
Loads models piece-by-piece to run statistical verification within memory constraints.
"""

import json
import os
import sys
import gc
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import psutil
from dataclasses import dataclass
import hashlib

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the REAL PoT framework components
from transformers import AutoModelForCausalLM, AutoTokenizer
from pot.lm.models import LM
from pot.lm.verifier import LMVerifier, LMVerificationResult
from pot.lm.sequential_tester import SequentialTester, SPRTState
from pot.core.challenge import generate_challenges, ChallengeConfig
from pot.core.kdf_prompt_generator import KDFPromptGenerator
from pot.core.diff_decision import EnhancedSequentialTester, TestingMode

@dataclass
class ShardedPoTConfig:
    """Configuration for sharded PoT verification."""
    max_memory_gb: int = 30
    batch_size: int = 1
    n_challenges: int = 20
    offload_folder: str = "/tmp/yi34b_offload"
    cache_dir: str = "/tmp/yi34b_cache"
    mode: TestingMode = TestingMode.QUICK_GATE
    delta: float = 0.01  # Confidence parameter
    
class ShardedLM(LM):
    """
    Sharded Language Model class that extends the PoT LM class.
    Loads models with aggressive memory management.
    """
    
    def __init__(self, name: str, device: str = "cpu", seed: int = 0, 
                 max_memory_gb: int = 30, offload_folder: str = "/tmp/offload"):
        """Initialize sharded LM with memory constraints."""
        torch.manual_seed(seed)
        
        print(f"📦 Loading sharded model: {name}")
        print(f"  Max memory: {max_memory_gb}GB")
        print(f"  Device: {device}")
        
        # Create offload directory
        os.makedirs(offload_folder, exist_ok=True)
        
        # Load tokenizer (lightweight)
        print("  Loading tokenizer...")
        self.tok = AutoTokenizer.from_pretrained(name, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        
        # Check available memory
        available_gb = psutil.virtual_memory().available / 1e9
        print(f"  Available RAM: {available_gb:.1f}GB")
        
        # Set up device map for automatic sharding
        max_memory = {}
        if torch.cuda.is_available() and device == "cuda":
            # Use GPU if available
            max_memory[0] = f"{min(10, max_memory_gb)}GB"
            max_memory['cpu'] = f"{max_memory_gb}GB"
        else:
            # CPU only
            max_memory['cpu'] = f"{max_memory_gb}GB"
        
        print("  Loading model with automatic sharding...")
        
        try:
            # Load model with aggressive memory management
            self.m = AutoModelForCausalLM.from_pretrained(
                name,
                device_map='auto',
                max_memory=max_memory,
                torch_dtype=torch.float16,  # Use half precision
                low_cpu_mem_usage=True,
                offload_folder=offload_folder,
                offload_state_dict=True
            ).eval()
            
            self.device = device
            
            # Check memory after loading
            used_gb = psutil.virtual_memory().percent
            print(f"  ✅ Model loaded! Memory usage: {used_gb:.1f}%")
            
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
            raise
    
    def unload(self):
        """Unload model and free memory."""
        print("  🗑️  Unloading model...")
        try:
            if hasattr(self, 'm') and self.m is not None:
                # Move to CPU first if on GPU
                if hasattr(self.m, 'cpu'):
                    self.m.cpu()
                
                # Delete model
                del self.m
                self.m = None
            
            # Clear tokenizer
            if hasattr(self, 'tok') and self.tok is not None:
                del self.tok
                self.tok = None
            
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Give OS time to reclaim memory
            time.sleep(2)
            
            print("  ✅ Model unloaded")
            
        except Exception as e:
            print(f"  ⚠️  Error during unload: {e}")

class ShardedLMVerifier(LMVerifier):
    """
    Sharded Language Model Verifier that extends the PoT verifier.
    Handles memory-constrained verification.
    """
    
    def __init__(self, reference_model: ShardedLM, delta: float = 0.01, 
                 use_sequential: bool = True):
        """Initialize sharded verifier."""
        # Call parent constructor
        super().__init__(reference_model, delta, use_sequential, enable_adaptive_challenges=False)
        
        # Override sequential tester for our needs
        if use_sequential:
            self.sequential_tester = SequentialTester(
                alpha=delta,
                beta=delta,
                p0=0.5,  # Null hypothesis (different models)
                p1=0.9,  # Alternative hypothesis (same model)
                max_trials=100,
                min_trials=5
            )
    
    def verify_sharded(self, candidate_model_path: str, prf_key: str, 
                       n_challenges: int = 20, config: ShardedPoTConfig = None) -> Dict[str, Any]:
        """
        Verify candidate model using sharded loading.
        
        Args:
            candidate_model_path: Path to candidate model
            prf_key: PRF key for challenge generation
            n_challenges: Number of challenges to run
            config: Sharded configuration
            
        Returns:
            Verification results
        """
        config = config or ShardedPoTConfig()
        results = {
            'timestamp': datetime.now().isoformat(),
            'reference_model': 'loaded',
            'candidate_model': candidate_model_path,
            'n_challenges': n_challenges,
            'challenges': [],
            'memory_stats': []
        }
        
        # Monitor initial memory
        initial_memory = psutil.virtual_memory().percent
        print(f"\nInitial memory: {initial_memory:.1f}%")
        results['memory_stats'].append({
            'stage': 'initial',
            'percent': initial_memory
        })
        
        # Generate challenges using KDF
        print(f"\n🎯 Generating {n_challenges} challenges...")
        kdf_gen = KDFPromptGenerator(master_key=prf_key, namespace="yi34b")
        challenges = []
        for i in range(n_challenges):
            # Generate deterministic challenge using KDF
            challenge_text = kdf_gen.generate_prompt(i)
            challenges.append(challenge_text)
        
        # Get reference model responses
        print(f"\n{'='*60}")
        print("GENERATING REFERENCE RESPONSES")
        print(f"{'='*60}")
        
        reference_responses = []
        for i, challenge in enumerate(challenges):
            print(f"  Challenge {i+1}/{n_challenges}: ", end='')
            try:
                response = self.reference_model.generate(challenge, max_new_tokens=10)
                # Extract only the generated part
                response = response[len(challenge):].strip()
                reference_responses.append(response)
                print(f"✅ '{response[:20]}...'")
            except Exception as e:
                print(f"❌ Error: {e}")
                reference_responses.append("")
            
            # Check memory
            if (i + 1) % 5 == 0:
                current_mem = psutil.virtual_memory().percent
                if current_mem > 85:
                    print(f"  ⚠️  High memory: {current_mem:.1f}%, stopping")
                    break
        
        mem_after_ref = psutil.virtual_memory().percent
        print(f"Memory after reference: {mem_after_ref:.1f}%")
        results['memory_stats'].append({
            'stage': 'after_reference',
            'percent': mem_after_ref
        })
        
        # Unload reference model to make room for candidate
        print("\n🔄 Unloading reference model to load candidate...")
        self.reference_model.unload()
        
        mem_after_unload = psutil.virtual_memory().percent
        print(f"Memory after unload: {mem_after_unload:.1f}%")
        results['memory_stats'].append({
            'stage': 'reference_unloaded',
            'percent': mem_after_unload
        })
        
        # Wait for memory to stabilize
        time.sleep(3)
        
        # Load candidate model
        print(f"\n{'='*60}")
        print("LOADING CANDIDATE MODEL")
        print(f"{'='*60}")
        
        try:
            candidate_model = ShardedLM(
                candidate_model_path,
                device="cpu",
                seed=0,
                max_memory_gb=config.max_memory_gb,
                offload_folder=config.offload_folder
            )
            
            mem_after_cand = psutil.virtual_memory().percent
            print(f"Memory after loading candidate: {mem_after_cand:.1f}%")
            results['memory_stats'].append({
                'stage': 'candidate_loaded',
                'percent': mem_after_cand
            })
            
            # Generate candidate responses
            print(f"\n{'='*60}")
            print("GENERATING CANDIDATE RESPONSES")
            print(f"{'='*60}")
            
            candidate_responses = []
            n_valid = len(reference_responses)
            
            for i, challenge in enumerate(challenges[:n_valid]):
                print(f"  Challenge {i+1}/{n_valid}: ", end='')
                try:
                    response = candidate_model.generate(challenge, max_new_tokens=10)
                    # Extract only the generated part
                    response = response[len(challenge):].strip()
                    candidate_responses.append(response)
                    print(f"✅ '{response[:20]}...'")
                except Exception as e:
                    print(f"❌ Error: {e}")
                    candidate_responses.append("")
                
                # Check memory
                if (i + 1) % 5 == 0:
                    current_mem = psutil.virtual_memory().percent
                    if current_mem > 85:
                        print(f"  ⚠️  High memory: {current_mem:.1f}%, stopping")
                        break
            
            # Unload candidate model
            candidate_model.unload()
            
            mem_after_unload2 = psutil.virtual_memory().percent
            print(f"Memory after unloading candidate: {mem_after_unload2:.1f}%")
            results['memory_stats'].append({
                'stage': 'candidate_unloaded',
                'percent': mem_after_unload2
            })
            
        except Exception as e:
            print(f"❌ Failed to process candidate model: {e}")
            results['error'] = str(e)
            results['verdict'] = 'FAILED'
            return results
        
        # Statistical verification
        print(f"\n{'='*60}")
        print("STATISTICAL VERIFICATION")
        print(f"{'='*60}")
        
        # Compare responses
        n_compared = min(len(reference_responses), len(candidate_responses))
        differences = []
        
        # Use Enhanced Sequential Tester from PoT framework
        enhanced_tester = EnhancedSequentialTester(mode=config.mode)
        
        for i in range(n_compared):
            ref_resp = reference_responses[i]
            cand_resp = candidate_responses[i]
            
            # Calculate difference
            if ref_resp == cand_resp:
                diff = 0.0
            elif not ref_resp or not cand_resp:
                diff = 1.0
            else:
                # Character-level similarity
                max_len = max(len(ref_resp), len(cand_resp))
                if max_len > 0:
                    common = sum(c1 == c2 for c1, c2 in zip(ref_resp, cand_resp))
                    diff = 1.0 - (common / max_len)
                else:
                    diff = 1.0
            
            differences.append(diff)
            
            # Update sequential tester
            decision, stats = enhanced_tester.update(diff)
            
            results['challenges'].append({
                'index': i,
                'challenge': challenges[i][:30] + '...',
                'ref_response': ref_resp[:30] if ref_resp else "N/A",
                'cand_response': cand_resp[:30] if cand_resp else "N/A",
                'difference': diff,
                'decision': decision,
                'stats': stats
            })
            
            # Check for early stopping
            if decision != 'UNDECIDED':
                print(f"Early stopping at challenge {i+1}: {decision}")
                results['early_stop'] = True
                results['stop_at'] = i + 1
                break
        
        # Get final decision
        final_decision = enhanced_tester.get_decision()
        final_stats = enhanced_tester.get_statistics()
        
        results['verdict'] = final_decision
        results['statistics'] = final_stats
        results['n_compared'] = n_compared
        
        # Summary statistics
        if differences:
            results['summary'] = {
                'mean_difference': float(np.mean(differences)),
                'std_difference': float(np.std(differences)),
                'min_difference': float(np.min(differences)),
                'max_difference': float(np.max(differences)),
                'n_exact_matches': sum(1 for d in differences if d == 0.0)
            }
        
        print(f"\nChallenges compared: {n_compared}")
        if differences:
            print(f"Mean difference: {np.mean(differences):.3f}")
            print(f"Exact matches: {sum(1 for d in differences if d == 0.0)}/{n_compared}")
        print(f"\nVerdict: {final_decision}")
        print(f"Confidence: {final_stats.get('confidence', 0):.1%}")
        
        # Memory summary
        peak_memory = max(m['percent'] for m in results['memory_stats'])
        final_memory = psutil.virtual_memory().percent
        
        print(f"\n{'='*60}")
        print("MEMORY SUMMARY")
        print(f"{'='*60}")
        print(f"Initial: {initial_memory:.1f}%")
        print(f"Peak: {peak_memory:.1f}%")
        print(f"Final: {final_memory:.1f}%")
        print(f"Max increase: {peak_memory - initial_memory:.1f}%")
        
        results['memory_summary'] = {
            'initial': initial_memory,
            'peak': peak_memory,
            'final': final_memory,
            'max_increase': peak_memory - initial_memory
        }
        
        return results

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sharded PoT verification using REAL framework')
    parser.add_argument('--ref-model', default='/Users/rohanvinaik/LLM_Models/yi-34b',
                      help='Path to reference model')
    parser.add_argument('--cand-model', default='/Users/rohanvinaik/LLM_Models/yi-34b-chat',
                      help='Path to candidate model')
    parser.add_argument('--prf-key', default='deadbeefcafebabe1234567890abcdef',
                      help='PRF key for challenge generation')
    parser.add_argument('--n-challenges', type=int, default=10,
                      help='Number of challenges')
    parser.add_argument('--max-memory', type=int, default=30,
                      help='Max memory per model in GB')
    parser.add_argument('--mode', choices=['quick', 'audit'], default='quick',
                      help='Verification mode')
    parser.add_argument('--output', default='experimental_results/yi34b_pot_sharded_real.json',
                      help='Output file for results')
    
    args = parser.parse_args()
    
    # Check if models exist
    if not os.path.exists(args.ref_model):
        print(f"❌ Reference model not found: {args.ref_model}")
        sys.exit(1)
    if not os.path.exists(args.cand_model):
        print(f"❌ Candidate model not found: {args.cand_model}")
        sys.exit(1)
    
    # Configure
    config = ShardedPoTConfig(
        max_memory_gb=args.max_memory,
        n_challenges=args.n_challenges,
        mode=TestingMode.AUDIT_GRADE if args.mode == 'audit' else TestingMode.QUICK_GATE
    )
    
    # Create offload directories
    os.makedirs(config.offload_folder, exist_ok=True)
    os.makedirs(config.cache_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"SHARDED POT VERIFICATION USING REAL FRAMEWORK")
    print(f"{'='*70}")
    print(f"Reference: {args.ref_model}")
    print(f"Candidate: {args.cand_model}")
    print(f"Mode: {args.mode}")
    print(f"Challenges: {args.n_challenges}")
    print(f"Memory limit: {args.max_memory}GB")
    print(f"{'='*70}")
    
    try:
        # Load reference model with sharding
        print("\n📦 Loading reference model...")
        reference_model = ShardedLM(
            args.ref_model,
            device="cpu",
            seed=0,
            max_memory_gb=config.max_memory_gb,
            offload_folder=config.offload_folder
        )
        
        # Create verifier
        verifier = ShardedLMVerifier(reference_model, delta=config.delta, use_sequential=True)
        
        # Run verification
        results = verifier.verify_sharded(
            args.cand_model,
            args.prf_key,
            n_challenges=config.n_challenges,
            config=config
        )
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_path}")
        
        # Cleanup
        if hasattr(reference_model, 'unload'):
            reference_model.unload()
        
        # Clean up offload directories
        import shutil
        for dir_path in [config.offload_folder, config.cache_dir]:
            if os.path.exists(dir_path):
                try:
                    shutil.rmtree(dir_path)
                    print(f"  Cleaned: {dir_path}")
                except:
                    pass
        
        # Return appropriate exit code
        if results.get('verdict') == 'SAME':
            return 0
        elif results.get('verdict') == 'DIFFERENT':
            return 1
        else:
            return 2
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 3

if __name__ == '__main__':
    sys.exit(main())