#!/usr/bin/env python3
"""
Full PoT pipeline with sharding for Yi-34B models.
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

# Conditional imports with fallbacks
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
    from transformers import BitsAndBytesConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️  Transformers not available, using mock mode")

try:
    from pot.core.diff_decision import EnhancedSequentialTester, TestingMode
    from pot.core.kdf import KDF
    HAS_POT = True
except ImportError:
    HAS_POT = False
    print("⚠️  PoT framework not available, using simplified version")
    
    class TestingMode:
        QUICK_GATE = 'quick'
        AUDIT_GRADE = 'audit'
    
    class KDF:
        @staticmethod
        def generate(prf_key: str, info: str, length: int = 32) -> bytes:
            return hashlib.sha256(f"{prf_key}{info}".encode()).digest()[:length]

@dataclass
class ShardedPoTConfig:
    """Configuration for sharded PoT verification."""
    max_memory_gb: int = 25
    batch_size: int = 1
    n_challenges: int = 20  # Reduced for memory
    use_8bit: bool = True
    use_4bit: bool = False  # Even more aggressive quantization
    offload_to_cpu: bool = True
    max_new_tokens: int = 10  # Short responses to save memory
    temperature: float = 0.0  # Deterministic
    mode: TestingMode = TestingMode.QUICK_GATE
    cache_dir: Optional[str] = "/tmp/yi34b_cache"
    
class ShardedModelLoader:
    """Loads large models with aggressive memory management."""
    
    def __init__(self, config: ShardedPoTConfig):
        self.config = config
        self.current_model = None
        self.current_model_path = None
        self.tokenizer = None
        
    def load_model_sharded(self, model_path: str) -> bool:
        """Load model with maximum memory efficiency."""
        print(f"\n📦 Loading model (sharded): {model_path}")
        
        # Clean up any existing model
        if self.current_model is not None:
            self.unload_model()
        
        # Check available memory
        available_gb = psutil.virtual_memory().available / 1e9
        print(f"  Available RAM: {available_gb:.1f}GB")
        
        if available_gb < 15:  # Need at least 15GB free
            print("  ❌ Insufficient memory for sharded loading")
            return False
        
        try:
            # Load tokenizer (lightweight)
            print("  Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                padding_side='left'
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Prepare quantization config
            quantization_config = None
            if self.config.use_4bit:
                print("  Using 4-bit quantization (most aggressive)")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif self.config.use_8bit:
                print("  Using 8-bit quantization")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
            
            # Load model with device_map for automatic sharding
            print("  Loading model shards with device_map='auto'...")
            
            # Set memory limits
            max_memory = {}
            if torch.cuda.is_available():
                # Reserve some GPU memory if available
                max_memory[0] = f"{min(10, self.config.max_memory_gb)}GB"
            max_memory['cpu'] = f"{self.config.max_memory_gb}GB"
            
            self.current_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map='auto',
                max_memory=max_memory,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                offload_folder=self.config.cache_dir,
                offload_state_dict=True  # Offload parts to disk
            )
            
            self.current_model_path = model_path
            
            # Check memory after loading
            used_gb = psutil.virtual_memory().percent
            print(f"  ✅ Model loaded! Memory usage: {used_gb:.1f}%")
            
            # Set to eval mode
            self.current_model.eval()
            
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to load model: {e}")
            self.unload_model()
            return False
    
    def unload_model(self):
        """Completely unload model and free memory."""
        if self.current_model is not None:
            print("  🗑️  Unloading model...")
            try:
                # Move to CPU first if on GPU
                if hasattr(self.current_model, 'cpu'):
                    self.current_model.cpu()
                
                # Delete model
                del self.current_model
                self.current_model = None
                self.current_model_path = None
                
                # Clear tokenizer
                if self.tokenizer is not None:
                    del self.tokenizer
                    self.tokenizer = None
                
                # Force garbage collection
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Give OS time to reclaim memory
                time.sleep(2)
                
                print("  ✅ Model unloaded")
                
            except Exception as e:
                print(f"  ⚠️  Error during unload: {e}")
    
    def generate_response(self, prompt: str) -> Optional[str]:
        """Generate response with current model."""
        if self.current_model is None or self.tokenizer is None:
            return None
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to same device as model
            device = next(self.current_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate with minimal memory usage
            with torch.no_grad():
                outputs = self.current_model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Return only the new tokens
            return response[len(prompt):].strip()
            
        except Exception as e:
            print(f"  ⚠️  Generation error: {e}")
            return None

class ShardedPoTVerifier:
    """Runs PoT verification with sharded model loading."""
    
    def __init__(self, config: ShardedPoTConfig = None):
        self.config = config or ShardedPoTConfig()
        self.loader = ShardedModelLoader(self.config)
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'max_memory_gb': self.config.max_memory_gb,
                'n_challenges': self.config.n_challenges,
                'use_8bit': self.config.use_8bit,
                'use_4bit': self.config.use_4bit,
                'mode': str(self.config.mode)
            },
            'challenges': [],
            'memory_stats': []
        }
        
        # Create cache directory
        if self.config.cache_dir:
            os.makedirs(self.config.cache_dir, exist_ok=True)
    
    def generate_challenges(self, prf_key: str, n: int = None) -> List[str]:
        """Generate deterministic challenges."""
        n = n or self.config.n_challenges
        challenges = []
        
        for i in range(n):
            # Generate deterministic challenge
            seed = KDF.generate(prf_key, f"challenge_{i}", 32)
            challenge_text = f"Complete this sequence: {seed.hex()[:8]}"
            challenges.append(challenge_text)
        
        return challenges
    
    def run_verification(self, model1_path: str, model2_path: str, prf_key: str) -> Dict[str, Any]:
        """Run sharded PoT verification."""
        print(f"\n{'='*70}")
        print(f"SHARDED POT VERIFICATION FOR YI-34B MODELS")
        print(f"{'='*70}")
        print(f"Model 1: {model1_path}")
        print(f"Model 2: {model2_path}")
        print(f"Mode: {self.config.mode}")
        print(f"Challenges: {self.config.n_challenges}")
        print(f"Memory limit: {self.config.max_memory_gb}GB")
        print(f"{'='*70}")
        
        # Monitor initial memory
        initial_memory = psutil.virtual_memory().percent
        print(f"\nInitial memory: {initial_memory:.1f}%")
        self.results['memory_stats'].append({
            'stage': 'initial',
            'percent': initial_memory
        })
        
        # Generate challenges
        print(f"\n🎯 Generating {self.config.n_challenges} challenges...")
        challenges = self.generate_challenges(prf_key)
        
        # Process each model separately to avoid loading both
        model1_responses = []
        model2_responses = []
        
        # Process Model 1
        print(f"\n{'='*60}")
        print("PROCESSING MODEL 1")
        print(f"{'='*60}")
        
        if self.loader.load_model_sharded(model1_path):
            mem_after_load1 = psutil.virtual_memory().percent
            print(f"Memory after loading Model 1: {mem_after_load1:.1f}%")
            self.results['memory_stats'].append({
                'stage': 'model1_loaded',
                'percent': mem_after_load1
            })
            
            print(f"\nGenerating responses for Model 1...")
            for i, challenge in enumerate(challenges):
                print(f"  Challenge {i+1}/{len(challenges)}: ", end='')
                response = self.loader.generate_response(challenge)
                
                if response is not None:
                    model1_responses.append(response)
                    print(f"✅ '{response[:20]}...'")
                else:
                    model1_responses.append("")
                    print("❌ Failed")
                
                # Check memory periodically
                if (i + 1) % 5 == 0:
                    current_mem = psutil.virtual_memory().percent
                    if current_mem > 80:
                        print(f"  ⚠️  High memory: {current_mem:.1f}%, stopping")
                        break
            
            # Unload Model 1
            self.loader.unload_model()
            
            mem_after_unload1 = psutil.virtual_memory().percent
            print(f"Memory after unloading Model 1: {mem_after_unload1:.1f}%")
            self.results['memory_stats'].append({
                'stage': 'model1_unloaded',
                'percent': mem_after_unload1
            })
            
            # Wait for memory to stabilize
            time.sleep(3)
        
        # Process Model 2
        print(f"\n{'='*60}")
        print("PROCESSING MODEL 2")
        print(f"{'='*60}")
        
        if self.loader.load_model_sharded(model2_path):
            mem_after_load2 = psutil.virtual_memory().percent
            print(f"Memory after loading Model 2: {mem_after_load2:.1f}%")
            self.results['memory_stats'].append({
                'stage': 'model2_loaded',
                'percent': mem_after_load2
            })
            
            print(f"\nGenerating responses for Model 2...")
            for i, challenge in enumerate(challenges[:len(model1_responses)]):
                print(f"  Challenge {i+1}/{len(model1_responses)}: ", end='')
                response = self.loader.generate_response(challenge)
                
                if response is not None:
                    model2_responses.append(response)
                    print(f"✅ '{response[:20]}...'")
                else:
                    model2_responses.append("")
                    print("❌ Failed")
                
                # Check memory periodically
                if (i + 1) % 5 == 0:
                    current_mem = psutil.virtual_memory().percent
                    if current_mem > 80:
                        print(f"  ⚠️  High memory: {current_mem:.1f}%, stopping")
                        break
            
            # Unload Model 2
            self.loader.unload_model()
            
            mem_after_unload2 = psutil.virtual_memory().percent
            print(f"Memory after unloading Model 2: {mem_after_unload2:.1f}%")
            self.results['memory_stats'].append({
                'stage': 'model2_unloaded',
                'percent': mem_after_unload2
            })
        
        # Compare responses
        print(f"\n{'='*60}")
        print("STATISTICAL VERIFICATION")
        print(f"{'='*60}")
        
        if model1_responses and model2_responses:
            n_valid = min(len(model1_responses), len(model2_responses))
            
            # Calculate differences
            differences = []
            for i in range(n_valid):
                r1 = model1_responses[i]
                r2 = model2_responses[i]
                
                # Simple difference metric
                if r1 == r2:
                    diff = 0.0
                elif not r1 or not r2:
                    diff = 1.0
                else:
                    # Character-level similarity
                    max_len = max(len(r1), len(r2))
                    if max_len > 0:
                        common = sum(c1 == c2 for c1, c2 in zip(r1, r2))
                        diff = 1.0 - (common / max_len)
                    else:
                        diff = 1.0
                
                differences.append(diff)
                
                self.results['challenges'].append({
                    'index': i,
                    'challenge': challenges[i][:50] + '...',
                    'response1': r1[:50] if r1 else "N/A",
                    'response2': r2[:50] if r2 else "N/A",
                    'difference': diff
                })
            
            # Statistical analysis
            mean_diff = np.mean(differences)
            std_diff = np.std(differences)
            
            print(f"Challenges compared: {n_valid}")
            print(f"Mean difference: {mean_diff:.3f}")
            print(f"Std deviation: {std_diff:.3f}")
            
            # Decision
            if mean_diff < 0.1:
                verdict = "LIKELY_SAME"
                confidence = 0.95
            elif mean_diff > 0.5:
                verdict = "LIKELY_DIFFERENT"
                confidence = 0.95
            else:
                verdict = "UNDECIDED"
                confidence = 0.5
            
            self.results['verdict'] = verdict
            self.results['confidence'] = confidence
            self.results['statistics'] = {
                'n_challenges': n_valid,
                'mean_difference': float(mean_diff),
                'std_deviation': float(std_diff),
                'min_difference': float(np.min(differences)),
                'max_difference': float(np.max(differences))
            }
            
            print(f"\nVerdict: {verdict} (confidence: {confidence:.1%})")
        else:
            self.results['verdict'] = "FAILED"
            self.results['error'] = "Could not generate responses"
            print("❌ Verification failed: Could not generate responses")
        
        # Final memory status
        final_memory = psutil.virtual_memory().percent
        peak_memory = max(m['percent'] for m in self.results['memory_stats'])
        
        print(f"\n{'='*60}")
        print("MEMORY SUMMARY")
        print(f"{'='*60}")
        print(f"Initial: {initial_memory:.1f}%")
        print(f"Peak: {peak_memory:.1f}%")
        print(f"Final: {final_memory:.1f}%")
        print(f"Max increase: {peak_memory - initial_memory:.1f}%")
        
        self.results['memory_summary'] = {
            'initial': initial_memory,
            'peak': peak_memory,
            'final': final_memory,
            'max_increase': peak_memory - initial_memory
        }
        
        return self.results
    
    def cleanup(self):
        """Clean up resources."""
        self.loader.unload_model()
        
        # Clean cache directory
        if self.config.cache_dir and os.path.exists(self.config.cache_dir):
            try:
                import shutil
                shutil.rmtree(self.config.cache_dir)
                print(f"  Cleaned cache: {self.config.cache_dir}")
            except:
                pass

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sharded PoT verification for Yi-34B models')
    parser.add_argument('--model1', default='/Users/rohanvinaik/LLM_Models/yi-34b',
                      help='Path to first model')
    parser.add_argument('--model2', default='/Users/rohanvinaik/LLM_Models/yi-34b-chat',
                      help='Path to second model')
    parser.add_argument('--prf-key', default='deadbeefcafebabe1234567890abcdef',
                      help='PRF key for challenge generation')
    parser.add_argument('--n-challenges', type=int, default=10,
                      help='Number of challenges')
    parser.add_argument('--max-memory', type=int, default=25,
                      help='Max memory per model in GB')
    parser.add_argument('--use-4bit', action='store_true',
                      help='Use 4-bit quantization (most aggressive)')
    parser.add_argument('--mode', choices=['quick', 'audit'], default='quick',
                      help='Verification mode')
    parser.add_argument('--output', default='experimental_results/yi34b_pot_sharded.json',
                      help='Output file for results')
    
    args = parser.parse_args()
    
    # Check if models exist
    if not os.path.exists(args.model1):
        print(f"❌ Model 1 not found: {args.model1}")
        sys.exit(1)
    if not os.path.exists(args.model2):
        print(f"❌ Model 2 not found: {args.model2}")
        sys.exit(1)
    
    # Check transformers
    if not HAS_TRANSFORMERS:
        print("❌ Transformers library required. Install with: pip install transformers")
        sys.exit(1)
    
    # Configure
    config = ShardedPoTConfig(
        max_memory_gb=args.max_memory,
        n_challenges=args.n_challenges,
        use_4bit=args.use_4bit,
        use_8bit=not args.use_4bit,  # Use 8-bit if not 4-bit
        mode=TestingMode.AUDIT_GRADE if args.mode == 'audit' else TestingMode.QUICK_GATE
    )
    
    # Run verification
    verifier = ShardedPoTVerifier(config)
    
    try:
        results = verifier.run_verification(args.model1, args.model2, args.prf_key)
        
        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_path}")
        
        # Return appropriate exit code
        if results.get('verdict') == 'LIKELY_SAME':
            return 0
        elif results.get('verdict') == 'LIKELY_DIFFERENT':
            return 1
        else:
            return 2
            
    finally:
        # Always cleanup
        verifier.cleanup()

if __name__ == '__main__':
    sys.exit(main())