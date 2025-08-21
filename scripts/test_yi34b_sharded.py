#!/usr/bin/env python3
"""
Sharded verification for Yi-34B models.
Loads and verifies models piece by piece to work within memory constraints.
"""

import json
import os
import sys
import gc
import time
import hashlib
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import psutil
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@dataclass
class ShardConfig:
    """Configuration for sharded model loading."""
    max_memory_gb: int = 30  # Max memory per shard
    batch_size: int = 1
    use_8bit: bool = True
    use_cpu_offload: bool = True
    verify_layers: List[int] = None  # Specific layers to verify
    
class ShardedModelVerifier:
    """Verifies large models by loading shards sequentially."""
    
    def __init__(self, config: ShardConfig = None):
        self.config = config or ShardConfig()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'shard_results': []
        }
        
    def load_single_shard(self, model_path: str, shard_file: str) -> Optional[Dict]:
        """Load a single model shard and extract info."""
        print(f"Loading shard: {shard_file}")
        
        shard_path = os.path.join(model_path, shard_file)
        shard_size = os.path.getsize(shard_path) / 1e9
        
        # Check if we have enough memory
        available_ram = psutil.virtual_memory().available / 1e9
        if available_ram < shard_size * 2:  # Need 2x for safety
            print(f"  ⚠️  Insufficient RAM: {available_ram:.1f}GB available, need {shard_size*2:.1f}GB")
            return None
            
        try:
            # Load just the shard
            print(f"  Loading {shard_size:.2f}GB shard...")
            if shard_file.endswith('.safetensors'):
                from safetensors import safe_open
                tensors = {}
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensors[key] = f.get_tensor(key)
            else:
                tensors = torch.load(shard_path, map_location='cpu')
            
            # Extract layer information
            layer_info = {}
            for key, tensor in tensors.items():
                if 'layer' in key.lower():
                    # Extract layer number
                    parts = key.split('.')
                    for part in parts:
                        if part.isdigit():
                            layer_num = int(part)
                            if layer_num not in layer_info:
                                layer_info[layer_num] = []
                            layer_info[layer_num].append({
                                'name': key,
                                'shape': list(tensor.shape),
                                'dtype': str(tensor.dtype),
                                'size_mb': tensor.numel() * tensor.element_size() / 1e6
                            })
                            break
            
            # Calculate shard statistics
            total_params = sum(t.numel() for t in tensors.values())
            total_size_gb = sum(t.numel() * t.element_size() for t in tensors.values()) / 1e9
            
            result = {
                'shard': shard_file,
                'layers': list(layer_info.keys()),
                'n_layers': len(layer_info),
                'n_params': total_params,
                'size_gb': total_size_gb,
                'tensors': len(tensors)
            }
            
            print(f"  ✅ Loaded: {len(tensors)} tensors, {total_params/1e6:.1f}M params, {total_size_gb:.2f}GB")
            print(f"  Layers: {sorted(layer_info.keys())[:10]}{'...' if len(layer_info) > 10 else ''}")
            
            # Clean up
            del tensors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return result
            
        except Exception as e:
            print(f"  ❌ Error loading shard: {e}")
            return None
    
    def verify_layer_wise(self, model1_path: str, model2_path: str) -> Dict[str, Any]:
        """Verify models layer by layer."""
        print(f"\n{'='*60}")
        print("LAYER-WISE VERIFICATION")
        print(f"{'='*60}")
        
        result = {
            'test': 'layer_wise',
            'model1': model1_path,
            'model2': model2_path,
            'layers_compared': []
        }
        
        try:
            # Get model shards
            shards1 = self.get_model_shards(model1_path)
            shards2 = self.get_model_shards(model2_path)
            
            print(f"\nModel 1: {len(shards1)} shards")
            print(f"Model 2: {len(shards2)} shards")
            
            # Process each shard pair
            max_shards = min(3, min(len(shards1), len(shards2)))  # Limit to 3 for memory
            
            for i in range(max_shards):
                print(f"\n--- Comparing shard {i+1}/{max_shards} ---")
                
                # Check memory before loading
                mem_before = psutil.virtual_memory().percent
                print(f"Memory before: {mem_before:.1f}%")
                
                if mem_before > 70:
                    print("⚠️  Memory too high, stopping shard comparison")
                    break
                
                # Load shards one at a time
                print(f"\nLoading from Model 1...")
                shard1_info = self.load_single_shard(model1_path, shards1[i])
                
                print(f"\nLoading from Model 2...")
                shard2_info = self.load_single_shard(model2_path, shards2[i])
                
                if shard1_info and shard2_info:
                    # Compare shard info
                    comparison = {
                        'shard_index': i,
                        'model1_layers': shard1_info['n_layers'],
                        'model2_layers': shard2_info['n_layers'],
                        'model1_params': shard1_info['n_params'],
                        'model2_params': shard2_info['n_params'],
                        'param_diff': abs(shard1_info['n_params'] - shard2_info['n_params']),
                        'layers_match': set(shard1_info['layers']) == set(shard2_info['layers'])
                    }
                    result['layers_compared'].append(comparison)
                    
                    print(f"\n  Comparison results:")
                    print(f"    Layers match: {'✅' if comparison['layers_match'] else '❌'}")
                    print(f"    Param difference: {comparison['param_diff']:,}")
                
                # Force cleanup
                gc.collect()
                time.sleep(1)  # Give system time to reclaim memory
                
                mem_after = psutil.virtual_memory().percent
                print(f"Memory after: {mem_after:.1f}%")
            
            result['success'] = True
            result['shards_compared'] = len(result['layers_compared'])
            
        except Exception as e:
            print(f"❌ Error: {e}")
            result['success'] = False
            result['error'] = str(e)
        
        return result
    
    def get_model_shards(self, model_path: str) -> List[str]:
        """Get list of model shard files."""
        shards = []
        
        # Check for PyTorch shards
        shards.extend(sorted([f for f in os.listdir(model_path) 
                            if f.startswith('pytorch_model-') and f.endswith('.bin')]))
        
        # Check for SafeTensors shards
        shards.extend(sorted([f for f in os.listdir(model_path)
                            if f.startswith('model-') and f.endswith('.safetensors')]))
        
        return shards
    
    def verify_embedding_layers(self, model1_path: str, model2_path: str) -> Dict[str, Any]:
        """Verify just embedding layers (usually much smaller)."""
        print(f"\n{'='*60}")
        print("EMBEDDING LAYER VERIFICATION")
        print(f"{'='*60}")
        
        result = {
            'test': 'embeddings',
            'model1': model1_path,
            'model2': model2_path
        }
        
        try:
            # Load model configs
            with open(os.path.join(model1_path, 'config.json'), 'r') as f:
                config1 = json.load(f)
            with open(os.path.join(model2_path, 'config.json'), 'r') as f:
                config2 = json.load(f)
            
            vocab_size1 = config1.get('vocab_size', 0)
            vocab_size2 = config2.get('vocab_size', 0)
            hidden_size1 = config1.get('hidden_size', 0)
            hidden_size2 = config2.get('hidden_size', 0)
            
            print(f"Model 1 embeddings: {vocab_size1} x {hidden_size1}")
            print(f"Model 2 embeddings: {vocab_size2} x {hidden_size2}")
            
            # Estimate embedding size
            embed_size1_gb = (vocab_size1 * hidden_size1 * 2) / 1e9  # 2 bytes for fp16
            embed_size2_gb = (vocab_size2 * hidden_size2 * 2) / 1e9
            
            print(f"Embedding sizes: {embed_size1_gb:.2f}GB vs {embed_size2_gb:.2f}GB")
            
            result['vocab_match'] = (vocab_size1 == vocab_size2)
            result['hidden_match'] = (hidden_size1 == hidden_size2)
            result['embed_size1_gb'] = embed_size1_gb
            result['embed_size2_gb'] = embed_size2_gb
            
            # Try to load just first shard (usually contains embeddings)
            shards1 = self.get_model_shards(model1_path)
            shards2 = self.get_model_shards(model2_path)
            
            if shards1 and shards2:
                print(f"\nAttempting to load embedding weights from first shard...")
                
                # Check memory
                available_ram = psutil.virtual_memory().available / 1e9
                required_ram = max(embed_size1_gb, embed_size2_gb) * 3  # 3x for safety
                
                if available_ram > required_ram:
                    print(f"  Memory check: {available_ram:.1f}GB available, {required_ram:.1f}GB required")
                    
                    # Load first shards (usually contain embeddings)
                    shard1_info = self.load_single_shard(model1_path, shards1[0])
                    shard2_info = self.load_single_shard(model2_path, shards2[0])
                    
                    if shard1_info and shard2_info:
                        result['first_shard_compared'] = True
                else:
                    print(f"  ⚠️  Insufficient memory for embeddings: {available_ram:.1f}GB < {required_ram:.1f}GB")
                    result['first_shard_compared'] = False
            
            result['success'] = True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            result['success'] = False
            result['error'] = str(e)
        
        return result
    
    def verify_with_sampling(self, model1_path: str, model2_path: str) -> Dict[str, Any]:
        """Verify models by sampling random layers."""
        print(f"\n{'='*60}")
        print("SAMPLING-BASED VERIFICATION")  
        print(f"{'='*60}")
        
        result = {
            'test': 'sampling',
            'model1': model1_path,
            'model2': model2_path,
            'samples': []
        }
        
        try:
            # Load index files to understand structure
            index1_path = os.path.join(model1_path, 'pytorch_model.bin.index.json')
            index2_path = os.path.join(model2_path, 'model.safetensors.index.json')
            
            weight_map1 = {}
            weight_map2 = {}
            
            if os.path.exists(index1_path):
                with open(index1_path, 'r') as f:
                    index1 = json.load(f)
                    weight_map1 = index1.get('weight_map', {})
            
            if os.path.exists(index2_path):
                with open(index2_path, 'r') as f:
                    index2 = json.load(f)
                    weight_map2 = index2.get('weight_map', {})
            
            print(f"Model 1 has {len(weight_map1)} weights across shards")
            print(f"Model 2 has {len(weight_map2)} weights across shards")
            
            # Find common layer patterns
            common_patterns = ['embed', 'layer.0.', 'layer.30.', 'layer.59.', 'lm_head']
            
            for pattern in common_patterns:
                print(f"\nChecking pattern: {pattern}")
                
                keys1 = [k for k in weight_map1.keys() if pattern in k][:5]
                keys2 = [k for k in weight_map2.keys() if pattern in k][:5]
                
                if keys1 and keys2:
                    print(f"  Model 1: Found {len(keys1)} matching weights")
                    print(f"  Model 2: Found {len(keys2)} matching weights")
                    
                    # Compare weight names
                    common_structure = len(set(k.split('.')[-1] for k in keys1) & 
                                         set(k.split('.')[-1] for k in keys2)) > 0
                    
                    result['samples'].append({
                        'pattern': pattern,
                        'model1_matches': len(keys1),
                        'model2_matches': len(keys2),
                        'structure_match': common_structure
                    })
                    
                    print(f"  Structure match: {'✅' if common_structure else '❌'}")
            
            result['success'] = True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            result['success'] = False
            result['error'] = str(e)
        
        return result
    
    def run_all_tests(self, model1_path: str, model2_path: str) -> Dict[str, Any]:
        """Run all sharded verification tests."""
        print(f"\n{'='*70}")
        print(f"SHARDED VERIFICATION FOR YI-34B MODELS")
        print(f"{'='*70}")
        print(f"Model 1: {model1_path}")
        print(f"Model 2: {model2_path}")
        print(f"Memory limit: {self.config.max_memory_gb}GB per operation")
        print(f"{'='*70}")
        
        # Monitor memory
        initial_memory = psutil.virtual_memory().percent
        print(f"\nInitial memory usage: {initial_memory:.1f}%")
        
        # Run tests in order of increasing memory requirement
        tests = [
            ('Sampling Verification', self.verify_with_sampling),
            ('Embedding Verification', self.verify_embedding_layers),
            ('Layer-wise Verification', self.verify_layer_wise),
        ]
        
        for test_name, test_func in tests:
            print(f"\n{'='*70}")
            print(f"Running: {test_name}")
            print(f"{'='*70}")
            
            # Check memory before test
            current_memory = psutil.virtual_memory().percent
            if current_memory > 75:
                print(f"⚠️  Skipping {test_name}: Memory at {current_memory:.1f}%")
                continue
            
            result = test_func(model1_path, model2_path)
            self.results['tests'].append(result)
            
            # Force cleanup after each test
            gc.collect()
            time.sleep(2)
            
            current_memory = psutil.virtual_memory().percent
            print(f"\nMemory after {test_name}: {current_memory:.1f}%")
        
        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        
        successful = sum(1 for t in self.results['tests'] if t.get('success', False))
        total = len(self.results['tests'])
        
        print(f"Tests completed: {successful}/{total}")
        
        # Analyze results
        sampling_match = any(
            all(s.get('structure_match', False) for s in t.get('samples', []))
            for t in self.results['tests'] if t.get('test') == 'sampling'
        )
        
        embedding_match = any(
            t.get('vocab_match', False) and t.get('hidden_match', False)
            for t in self.results['tests'] if t.get('test') == 'embeddings'
        )
        
        layers_match = any(
            all(l.get('layers_match', False) for l in t.get('layers_compared', []))
            for t in self.results['tests'] if t.get('test') == 'layer_wise'
        )
        
        if sampling_match and embedding_match:
            verdict = "LIKELY_SAME_ARCHITECTURE"
            explanation = "Structure and embeddings match"
        elif sampling_match:
            verdict = "SIMILAR_ARCHITECTURE"
            explanation = "Weight structure matches but embeddings differ"
        else:
            verdict = "DIFFERENT_OR_UNKNOWN"
            explanation = "Could not verify enough to determine similarity"
        
        self.results['verdict'] = verdict
        self.results['explanation'] = explanation
        
        print(f"\nVerdict: {verdict}")
        print(f"Explanation: {explanation}")
        
        # Memory status
        final_memory = psutil.virtual_memory().percent
        print(f"\nFinal memory usage: {final_memory:.1f}%")
        print(f"Peak memory increase: {max(final_memory - initial_memory, 0):.1f}%")
        
        return self.results

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sharded verification for Yi-34B models')
    parser.add_argument('--model1', default='/Users/rohanvinaik/LLM_Models/yi-34b',
                      help='Path to first model')
    parser.add_argument('--model2', default='/Users/rohanvinaik/LLM_Models/yi-34b-chat',
                      help='Path to second model')
    parser.add_argument('--max-memory', type=int, default=30,
                      help='Max memory per operation in GB')
    parser.add_argument('--output', default='experimental_results/yi34b_sharded_verification.json',
                      help='Output file for results')
    
    args = parser.parse_args()
    
    # Check models exist
    if not os.path.exists(args.model1):
        print(f"❌ Model 1 not found: {args.model1}")
        sys.exit(1)
    if not os.path.exists(args.model2):
        print(f"❌ Model 2 not found: {args.model2}")
        sys.exit(1)
    
    # Configure sharded loading
    config = ShardConfig(max_memory_gb=args.max_memory)
    
    # Run verification
    verifier = ShardedModelVerifier(config)
    results = verifier.run_all_tests(args.model1, args.model2)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    return 0 if results.get('verdict') != 'DIFFERENT_OR_UNKNOWN' else 1

if __name__ == '__main__':
    sys.exit(main())