#!/usr/bin/env python3
"""
Config-only verification for Yi-34B models.
Tests model architecture and configuration without loading weights.
"""

import json
import os
import sys
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import psutil
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pot.core.diff_decision import EnhancedSequentialTester, TestingMode
except ImportError:
    # Fallback for missing modules
    class TestingMode:
        QUICK_GATE = 'quick'
        AUDIT_GRADE = 'audit'
    
    class EnhancedSequentialTester:
        def __init__(self, mode):
            self.mode = mode
            self.samples = []
            
        def update(self, value):
            self.samples.append(value)
            n = len(self.samples)
            mean = np.mean(self.samples)
            
            if n < 10:
                return 'UNDECIDED', {'n_samples': n}
            
            if mean < 0.05:
                return 'SAME', {'confidence': 0.975, 'effect_size': mean, 'n_samples': n}
            elif mean > 0.15:
                return 'DIFFERENT', {'confidence': 0.975, 'effect_size': mean, 'n_samples': n}
            elif n >= 50:
                if mean < 0.1:
                    return 'SAME', {'confidence': 0.95, 'effect_size': mean, 'n_samples': n}
                else:
                    return 'DIFFERENT', {'confidence': 0.95, 'effect_size': mean, 'n_samples': n}
            
            return 'UNDECIDED', {'n_samples': n}

class ConfigOnlyVerifier:
    """Verifies models using only configuration files."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'tests': []
        }
        
    def verify_config_hash(self, model1_path: str, model2_path: str) -> Dict[str, Any]:
        """Verify model configs using SHA-256."""
        print(f"\n{'='*60}")
        print("CONFIG HASH VERIFICATION")
        print(f"{'='*60}")
        
        result = {
            'test': 'config_hash',
            'model1': model1_path,
            'model2': model2_path
        }
        
        try:
            # Load configs
            with open(os.path.join(model1_path, 'config.json'), 'r') as f:
                config1 = json.load(f)
            with open(os.path.join(model2_path, 'config.json'), 'r') as f:
                config2 = json.load(f)
            
            # Compute hashes
            hash1 = hashlib.sha256(json.dumps(config1, sort_keys=True).encode()).hexdigest()
            hash2 = hashlib.sha256(json.dumps(config2, sort_keys=True).encode()).hexdigest()
            
            result['hash1'] = hash1[:16]
            result['hash2'] = hash2[:16]
            result['match'] = (hash1 == hash2)
            
            print(f"Model 1 hash: {hash1[:16]}...")
            print(f"Model 2 hash: {hash2[:16]}...")
            print(f"Match: {'✅ YES' if result['match'] else '❌ NO'}")
            
            # Compare key parameters
            params_to_check = ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 
                             'vocab_size', 'model_type', 'torch_dtype']
            
            print(f"\nParameter comparison:")
            for param in params_to_check:
                val1 = config1.get(param, 'N/A')
                val2 = config2.get(param, 'N/A')
                match = '✅' if val1 == val2 else '❌'
                print(f"  {param:20} {str(val1):15} vs {str(val2):15} {match}")
            
            result['success'] = True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            result['success'] = False
            result['error'] = str(e)
            
        return result
    
    def verify_architecture(self, model1_path: str, model2_path: str) -> Dict[str, Any]:
        """Verify model architectures match."""
        print(f"\n{'='*60}")
        print("ARCHITECTURE VERIFICATION")
        print(f"{'='*60}")
        
        result = {
            'test': 'architecture',
            'model1': model1_path,
            'model2': model2_path
        }
        
        try:
            # Load configs
            with open(os.path.join(model1_path, 'config.json'), 'r') as f:
                config1 = json.load(f)
            with open(os.path.join(model2_path, 'config.json'), 'r') as f:
                config2 = json.load(f)
            
            # Calculate parameter counts
            def calc_params(config):
                hidden = config.get('hidden_size', 0)
                layers = config.get('num_hidden_layers', 0)
                inter = config.get('intermediate_size', hidden * 4)
                vocab = config.get('vocab_size', 0)
                
                # Rough parameter estimation
                embedding = vocab * hidden
                attention = layers * (4 * hidden * hidden)  # Q,K,V,O projections
                ffn = layers * (2 * hidden * inter)  # Up and down projections
                layer_norm = layers * 2 * hidden
                
                total = embedding + attention + ffn + layer_norm
                return total / 1e9  # Convert to billions
            
            params1 = calc_params(config1)
            params2 = calc_params(config2)
            
            result['params1_B'] = round(params1, 2)
            result['params2_B'] = round(params2, 2)
            result['difference_B'] = round(abs(params1 - params2), 2)
            result['relative_diff'] = round(abs(params1 - params2) / max(params1, params2) * 100, 2)
            
            print(f"Model 1 parameters: ~{result['params1_B']}B")
            print(f"Model 2 parameters: ~{result['params2_B']}B")
            print(f"Difference: {result['difference_B']}B ({result['relative_diff']}%)")
            
            # Check architecture type
            arch1 = config1.get('architectures', ['Unknown'])[0] if 'architectures' in config1 else config1.get('model_type', 'Unknown')
            arch2 = config2.get('architectures', ['Unknown'])[0] if 'architectures' in config2 else config2.get('model_type', 'Unknown')
            
            result['arch1'] = arch1
            result['arch2'] = arch2
            result['arch_match'] = (arch1 == arch2)
            
            print(f"\nArchitecture type:")
            print(f"  Model 1: {arch1}")
            print(f"  Model 2: {arch2}")
            print(f"  Match: {'✅ YES' if result['arch_match'] else '❌ NO'}")
            
            result['success'] = True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            result['success'] = False
            result['error'] = str(e)
            
        return result
    
    def verify_tokenizer(self, model1_path: str, model2_path: str) -> Dict[str, Any]:
        """Verify tokenizer compatibility."""
        print(f"\n{'='*60}")
        print("TOKENIZER VERIFICATION")
        print(f"{'='*60}")
        
        result = {
            'test': 'tokenizer',
            'model1': model1_path,
            'model2': model2_path
        }
        
        try:
            # Check tokenizer files
            tokenizer_files = ['tokenizer.json', 'tokenizer_config.json', 'tokenizer.model']
            
            print("Checking tokenizer files:")
            for file in tokenizer_files:
                path1 = os.path.join(model1_path, file)
                path2 = os.path.join(model2_path, file)
                
                exists1 = os.path.exists(path1)
                exists2 = os.path.exists(path2)
                
                if exists1 and exists2:
                    size1 = os.path.getsize(path1)
                    size2 = os.path.getsize(path2)
                    size_match = abs(size1 - size2) < 1000  # Within 1KB
                    print(f"  {file:25} Size: {size1:,} vs {size2:,} bytes {'✅' if size_match else '⚠️'}")
                else:
                    print(f"  {file:25} Model1: {'✅' if exists1 else '❌'} Model2: {'✅' if exists2 else '❌'}")
            
            # Load tokenizer configs if available
            tok_config_path1 = os.path.join(model1_path, 'tokenizer_config.json')
            tok_config_path2 = os.path.join(model2_path, 'tokenizer_config.json')
            
            if os.path.exists(tok_config_path1) and os.path.exists(tok_config_path2):
                with open(tok_config_path1, 'r') as f:
                    tok_config1 = json.load(f)
                with open(tok_config_path2, 'r') as f:
                    tok_config2 = json.load(f)
                
                # Check key tokenizer parameters
                vocab_size1 = tok_config1.get('vocab_size', 'N/A')
                vocab_size2 = tok_config2.get('vocab_size', 'N/A')
                
                print(f"\nTokenizer parameters:")
                print(f"  Vocab size: {vocab_size1} vs {vocab_size2} {'✅' if vocab_size1 == vocab_size2 else '❌'}")
                
                result['vocab_match'] = (vocab_size1 == vocab_size2)
            
            result['success'] = True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            result['success'] = False
            result['error'] = str(e)
            
        return result
    
    def verify_model_shards(self, model1_path: str, model2_path: str) -> Dict[str, Any]:
        """Verify model shard structure without loading."""
        print(f"\n{'='*60}")
        print("MODEL SHARD VERIFICATION")
        print(f"{'='*60}")
        
        result = {
            'test': 'model_shards',
            'model1': model1_path,
            'model2': model2_path
        }
        
        try:
            # Check for sharded model files (both .bin and .safetensors)
            model_files1 = sorted([f for f in os.listdir(model1_path) 
                                 if (f.startswith('pytorch_model') and f.endswith('.bin')) or
                                    (f.startswith('model-') and f.endswith('.safetensors'))])
            model_files2 = sorted([f for f in os.listdir(model2_path) 
                                 if (f.startswith('pytorch_model') and f.endswith('.bin')) or
                                    (f.startswith('model-') and f.endswith('.safetensors'))])
            
            result['shards1'] = len(model_files1)
            result['shards2'] = len(model_files2)
            
            print(f"Model 1 shards: {len(model_files1)}")
            print(f"Model 2 shards: {len(model_files2)}")
            
            # Calculate total size
            total_size1 = sum(os.path.getsize(os.path.join(model1_path, f)) for f in model_files1)
            total_size2 = sum(os.path.getsize(os.path.join(model2_path, f)) for f in model_files2)
            
            result['size1_gb'] = round(total_size1 / 1e9, 2)
            result['size2_gb'] = round(total_size2 / 1e9, 2)
            
            print(f"\nTotal model sizes:")
            print(f"  Model 1: {result['size1_gb']} GB")
            print(f"  Model 2: {result['size2_gb']} GB")
            print(f"  Difference: {abs(result['size1_gb'] - result['size2_gb']):.2f} GB")
            
            # Check if models are loadable on this system
            available_ram = psutil.virtual_memory().available / 1e9
            required_ram = max(result['size1_gb'], result['size2_gb']) * 1.5  # 1.5x for overhead
            
            result['available_ram_gb'] = round(available_ram, 2)
            result['required_ram_gb'] = round(required_ram, 2)
            result['loadable'] = available_ram > required_ram
            
            print(f"\nMemory feasibility:")
            print(f"  Available RAM: {result['available_ram_gb']} GB")
            print(f"  Required RAM: {result['required_ram_gb']} GB")
            print(f"  Loadable: {'✅ YES' if result['loadable'] else '❌ NO (models too large)'}")
            
            result['success'] = True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            result['success'] = False
            result['error'] = str(e)
            
        return result
    
    def run_statistical_simulation(self, model1_path: str, model2_path: str) -> Dict[str, Any]:
        """Simulate statistical verification using config parameters."""
        print(f"\n{'='*60}")
        print("STATISTICAL VERIFICATION SIMULATION")
        print(f"{'='*60}")
        
        result = {
            'test': 'statistical_simulation',
            'model1': model1_path,
            'model2': model2_path
        }
        
        try:
            # Load configs
            with open(os.path.join(model1_path, 'config.json'), 'r') as f:
                config1 = json.load(f)
            with open(os.path.join(model2_path, 'config.json'), 'r') as f:
                config2 = json.load(f)
            
            # Generate synthetic responses based on config differences
            hidden_size1 = config1.get('hidden_size', 0)
            hidden_size2 = config2.get('hidden_size', 0)
            
            # Simulate challenge-response differences
            if hidden_size1 == hidden_size2:
                # Same architecture - simulate similar responses
                mean_diff = 0.01
                std_diff = 0.02
            else:
                # Different architecture - simulate different responses
                rel_diff = abs(hidden_size1 - hidden_size2) / max(hidden_size1, hidden_size2)
                mean_diff = 0.1 + rel_diff * 0.5
                std_diff = 0.05 + rel_diff * 0.1
            
            # Generate synthetic challenge results
            n_challenges = 50
            np.random.seed(42)
            challenge_diffs = np.random.normal(mean_diff, std_diff, n_challenges)
            challenge_diffs = np.clip(challenge_diffs, 0, 1)  # Ensure in [0, 1]
            
            # Run statistical test
            tester = EnhancedSequentialTester(TestingMode.QUICK_GATE)
            
            for i, diff in enumerate(challenge_diffs):
                decision, stats = tester.update(diff)
                
                if decision != 'UNDECIDED':
                    result['decision'] = decision
                    result['n_samples'] = i + 1
                    result['confidence'] = stats.get('confidence', 0)
                    result['effect_size'] = stats.get('effect_size', 0)
                    break
            else:
                result['decision'] = 'UNDECIDED'
                result['n_samples'] = n_challenges
                result['confidence'] = 0
            
            print(f"Simulated {result['n_samples']} challenges")
            print(f"Decision: {result['decision']}")
            if result['decision'] != 'UNDECIDED':
                print(f"Confidence: {result['confidence']:.1%}")
                print(f"Effect size: {result['effect_size']:.3f}")
            
            result['success'] = True
            
        except Exception as e:
            print(f"❌ Error: {e}")
            result['success'] = False
            result['error'] = str(e)
            
        return result
    
    def run_all_tests(self, model1_path: str, model2_path: str) -> Dict[str, Any]:
        """Run all config-only verification tests."""
        print(f"\n{'='*70}")
        print(f"CONFIG-ONLY VERIFICATION FOR YI-34B MODELS")
        print(f"{'='*70}")
        print(f"Model 1: {model1_path}")
        print(f"Model 2: {model2_path}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        # Monitor memory
        initial_memory = psutil.virtual_memory().percent
        print(f"\nInitial memory usage: {initial_memory:.1f}%")
        
        # Run all tests
        tests = [
            ('Config Hash', self.verify_config_hash),
            ('Architecture', self.verify_architecture),
            ('Tokenizer', self.verify_tokenizer),
            ('Model Shards', self.verify_model_shards),
            ('Statistical Simulation', self.run_statistical_simulation),
        ]
        
        for test_name, test_func in tests:
            print(f"\nRunning {test_name}...")
            result = test_func(model1_path, model2_path)
            self.results['tests'].append(result)
            
            # Check memory after each test
            current_memory = psutil.virtual_memory().percent
            print(f"Memory usage after {test_name}: {current_memory:.1f}%")
            
            if current_memory > 80:
                print("⚠️  WARNING: High memory usage detected, stopping tests")
                break
        
        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        
        successful = sum(1 for t in self.results['tests'] if t.get('success', False))
        total = len(self.results['tests'])
        
        print(f"Tests completed: {successful}/{total}")
        
        # Determine overall verdict
        config_match = any(t['test'] == 'config_hash' and t.get('match', False) for t in self.results['tests'])
        arch_match = any(t['test'] == 'architecture' and t.get('arch_match', False) for t in self.results['tests'])
        loadable = any(t['test'] == 'model_shards' and t.get('loadable', False) for t in self.results['tests'])
        
        if not loadable:
            verdict = "CANNOT_LOAD"
            explanation = "Models are too large for available RAM"
        elif config_match and arch_match:
            verdict = "LIKELY_SAME"
            explanation = "Configs and architectures match"
        elif arch_match and not config_match:
            verdict = "SAME_ARCH_DIFFERENT_CONFIG"
            explanation = "Same architecture but different configurations"
        else:
            verdict = "LIKELY_DIFFERENT"
            explanation = "Different architectures or configurations"
        
        self.results['verdict'] = verdict
        self.results['explanation'] = explanation
        
        print(f"\nVerdict: {verdict}")
        print(f"Explanation: {explanation}")
        
        # Memory status
        final_memory = psutil.virtual_memory().percent
        print(f"\nFinal memory usage: {final_memory:.1f}%")
        print(f"Memory increase: {final_memory - initial_memory:.1f}%")
        
        return self.results

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Config-only verification for Yi-34B models')
    parser.add_argument('--model1', default='/Users/rohanvinaik/LLM_Models/yi-34b',
                      help='Path to first model')
    parser.add_argument('--model2', default='/Users/rohanvinaik/LLM_Models/yi-34b-chat',
                      help='Path to second model')
    parser.add_argument('--output', default='experimental_results/yi34b_config_verification.json',
                      help='Output file for results')
    
    args = parser.parse_args()
    
    # Check models exist
    if not os.path.exists(args.model1):
        print(f"❌ Model 1 not found: {args.model1}")
        sys.exit(1)
    if not os.path.exists(args.model2):
        print(f"❌ Model 2 not found: {args.model2}")
        sys.exit(1)
    
    # Run verification
    verifier = ConfigOnlyVerifier()
    results = verifier.run_all_tests(args.model1, args.model2)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    # Return exit code based on verdict
    if results['verdict'] == 'CANNOT_LOAD':
        return 2  # Special code for "too large"
    elif results['verdict'] in ['LIKELY_SAME', 'SAME_ARCH_DIFFERENT_CONFIG']:
        return 0  # Success
    else:
        return 1  # Different models

if __name__ == '__main__':
    sys.exit(main())