#!/usr/bin/env python3
"""
Fingerprint-based verification for Yi-34B models.
Creates lightweight fingerprints without loading full weights.
"""

import json
import os
import sys
import hashlib
import struct
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import psutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ModelFingerprinter:
    """Creates lightweight fingerprints of large models."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'fingerprints': {}
        }
    
    def fingerprint_shard(self, shard_path: str, sample_rate: float = 0.001) -> Dict[str, Any]:
        """Create fingerprint of a model shard by sampling."""
        print(f"  Fingerprinting: {os.path.basename(shard_path)}")
        
        file_size = os.path.getsize(shard_path)
        
        # Read file metadata
        fingerprint = {
            'file': os.path.basename(shard_path),
            'size_bytes': file_size,
            'size_gb': file_size / 1e9
        }
        
        # Sample file at regular intervals
        sample_size = 1024  # Read 1KB samples
        n_samples = max(100, int(file_size * sample_rate / sample_size))
        stride = file_size // n_samples
        
        hasher = hashlib.sha256()
        samples = []
        
        try:
            with open(shard_path, 'rb') as f:
                for i in range(n_samples):
                    f.seek(i * stride)
                    sample = f.read(sample_size)
                    hasher.update(sample)
                    
                    # Extract some float values if possible
                    if len(sample) >= 4:
                        try:
                            # Try to interpret as float32
                            float_val = struct.unpack('f', sample[:4])[0]
                            if not np.isnan(float_val) and abs(float_val) < 1e10:
                                samples.append(float_val)
                        except:
                            pass
            
            fingerprint['hash'] = hasher.hexdigest()[:16]
            fingerprint['n_samples'] = n_samples
            
            if samples:
                fingerprint['sample_stats'] = {
                    'mean': float(np.mean(samples)),
                    'std': float(np.std(samples)),
                    'min': float(np.min(samples)),
                    'max': float(np.max(samples)),
                    'n_valid': len(samples)
                }
            
            print(f"    ✅ Hash: {fingerprint['hash']}, Size: {fingerprint['size_gb']:.2f}GB")
            
        except Exception as e:
            print(f"    ❌ Error: {e}")
            fingerprint['error'] = str(e)
        
        return fingerprint
    
    def fingerprint_model(self, model_path: str) -> Dict[str, Any]:
        """Create complete fingerprint of a model."""
        print(f"\nFingerprinting model: {model_path}")
        
        model_fingerprint = {
            'path': model_path,
            'shards': []
        }
        
        # Get all model files
        model_files = []
        
        # PyTorch files
        model_files.extend(sorted([f for f in os.listdir(model_path)
                                  if f.startswith('pytorch_model') and f.endswith('.bin')]))
        
        # SafeTensors files  
        model_files.extend(sorted([f for f in os.listdir(model_path)
                                  if f.startswith('model-') and f.endswith('.safetensors')]))
        
        print(f"Found {len(model_files)} model shards")
        
        # Fingerprint each shard
        total_size = 0
        for shard_file in model_files:
            shard_path = os.path.join(model_path, shard_file)
            fingerprint = self.fingerprint_shard(shard_path)
            model_fingerprint['shards'].append(fingerprint)
            total_size += fingerprint['size_gb']
        
        model_fingerprint['total_size_gb'] = total_size
        model_fingerprint['n_shards'] = len(model_files)
        
        # Create combined fingerprint
        combined_hash = hashlib.sha256()
        for shard in model_fingerprint['shards']:
            if 'hash' in shard:
                combined_hash.update(shard['hash'].encode())
        
        model_fingerprint['combined_hash'] = combined_hash.hexdigest()[:16]
        
        print(f"Total size: {total_size:.2f}GB, Combined hash: {model_fingerprint['combined_hash']}")
        
        return model_fingerprint
    
    def compare_fingerprints(self, fp1: Dict, fp2: Dict) -> Dict[str, Any]:
        """Compare two model fingerprints."""
        print(f"\n{'='*60}")
        print("FINGERPRINT COMPARISON")
        print(f"{'='*60}")
        
        comparison = {
            'model1_size_gb': fp1['total_size_gb'],
            'model2_size_gb': fp2['total_size_gb'],
            'size_diff_gb': abs(fp1['total_size_gb'] - fp2['total_size_gb']),
            'model1_shards': fp1['n_shards'],
            'model2_shards': fp2['n_shards'],
            'hash_match': fp1['combined_hash'] == fp2['combined_hash']
        }
        
        print(f"Model 1: {fp1['total_size_gb']:.2f}GB in {fp1['n_shards']} shards")
        print(f"Model 2: {fp2['total_size_gb']:.2f}GB in {fp2['n_shards']} shards")
        print(f"Size difference: {comparison['size_diff_gb']:.2f}GB")
        print(f"Hash match: {'✅ YES' if comparison['hash_match'] else '❌ NO'}")
        
        # Compare individual shards
        shard_matches = 0
        max_shards = min(fp1['n_shards'], fp2['n_shards'])
        
        for i in range(max_shards):
            if i < len(fp1['shards']) and i < len(fp2['shards']):
                shard1 = fp1['shards'][i]
                shard2 = fp2['shards'][i]
                
                if 'hash' in shard1 and 'hash' in shard2:
                    if shard1['hash'] == shard2['hash']:
                        shard_matches += 1
        
        comparison['shard_matches'] = shard_matches
        comparison['match_ratio'] = shard_matches / max_shards if max_shards > 0 else 0
        
        print(f"\nShard comparison:")
        print(f"  Matching shards: {shard_matches}/{max_shards}")
        print(f"  Match ratio: {comparison['match_ratio']:.1%}")
        
        # Statistical comparison if available
        if any('sample_stats' in s for s in fp1['shards']) and \
           any('sample_stats' in s for s in fp2['shards']):
            
            stats1 = [s.get('sample_stats', {}) for s in fp1['shards'] if 'sample_stats' in s]
            stats2 = [s.get('sample_stats', {}) for s in fp2['shards'] if 'sample_stats' in s]
            
            if stats1 and stats2:
                mean1 = np.mean([s['mean'] for s in stats1 if 'mean' in s])
                mean2 = np.mean([s['mean'] for s in stats2 if 'mean' in s])
                std1 = np.mean([s['std'] for s in stats1 if 'std' in s])
                std2 = np.mean([s['std'] for s in stats2 if 'std' in s])
                
                comparison['stats'] = {
                    'mean_diff': float(abs(mean1 - mean2)),
                    'std_diff': float(abs(std1 - std2)),
                    'similar_distribution': bool(abs(mean1 - mean2) < 0.1 and abs(std1 - std2) < 0.1)
                }
                
                print(f"\nStatistical comparison:")
                print(f"  Mean difference: {comparison['stats']['mean_diff']:.6f}")
                print(f"  Std difference: {comparison['stats']['std_diff']:.6f}")
                print(f"  Similar distribution: {'✅ YES' if comparison['stats']['similar_distribution'] else '❌ NO'}")
        
        return comparison
    
    def run_verification(self, model1_path: str, model2_path: str) -> Dict[str, Any]:
        """Run complete fingerprint verification."""
        print(f"\n{'='*70}")
        print(f"FINGERPRINT VERIFICATION FOR YI-34B MODELS")
        print(f"{'='*70}")
        print(f"Model 1: {model1_path}")
        print(f"Model 2: {model2_path}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")
        
        # Monitor memory
        initial_memory = psutil.virtual_memory().percent
        print(f"\nInitial memory usage: {initial_memory:.1f}%")
        
        # Create fingerprints
        print("\n--- Creating Model 1 Fingerprint ---")
        fp1 = self.fingerprint_model(model1_path)
        self.results['fingerprints']['model1'] = fp1
        
        mem_after_fp1 = psutil.virtual_memory().percent
        print(f"Memory after Model 1: {mem_after_fp1:.1f}%")
        
        print("\n--- Creating Model 2 Fingerprint ---")
        fp2 = self.fingerprint_model(model2_path)
        self.results['fingerprints']['model2'] = fp2
        
        mem_after_fp2 = psutil.virtual_memory().percent
        print(f"Memory after Model 2: {mem_after_fp2:.1f}%")
        
        # Compare fingerprints
        comparison = self.compare_fingerprints(fp1, fp2)
        self.results['comparison'] = comparison
        
        # Determine verdict
        if comparison['hash_match']:
            verdict = "IDENTICAL"
            explanation = "Models have identical fingerprints"
        elif comparison['match_ratio'] > 0.8:
            verdict = "VERY_SIMILAR"
            explanation = f"{comparison['match_ratio']:.0%} of shards match"
        elif comparison.get('stats', {}).get('similar_distribution', False):
            verdict = "SIMILAR_DISTRIBUTION"
            explanation = "Different weights but similar statistical distribution"
        elif abs(fp1['total_size_gb'] - fp2['total_size_gb']) < 1.0:
            verdict = "SAME_SIZE_DIFFERENT_WEIGHTS"
            explanation = "Same architecture size but different weights"
        else:
            verdict = "DIFFERENT"
            explanation = "Models appear to be different"
        
        self.results['verdict'] = verdict
        self.results['explanation'] = explanation
        
        # Summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Verdict: {verdict}")
        print(f"Explanation: {explanation}")
        
        # Memory status
        final_memory = psutil.virtual_memory().percent
        print(f"\nFinal memory usage: {final_memory:.1f}%")
        print(f"Memory increase: {final_memory - initial_memory:.1f}%")
        print(f"Peak memory: {max(mem_after_fp1, mem_after_fp2, final_memory):.1f}%")
        
        self.results['memory'] = {
            'initial': initial_memory,
            'peak': max(mem_after_fp1, mem_after_fp2, final_memory),
            'final': final_memory,
            'increase': final_memory - initial_memory
        }
        
        return self.results

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fingerprint verification for Yi-34B models')
    parser.add_argument('--model1', default='/Users/rohanvinaik/LLM_Models/yi-34b',
                      help='Path to first model')
    parser.add_argument('--model2', default='/Users/rohanvinaik/LLM_Models/yi-34b-chat',
                      help='Path to second model')
    parser.add_argument('--output', default='experimental_results/yi34b_fingerprint_verification.json',
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
    fingerprinter = ModelFingerprinter()
    results = fingerprinter.run_verification(args.model1, args.model2)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_path}")
    
    return 0 if results['verdict'] in ['IDENTICAL', 'VERY_SIMILAR'] else 1

if __name__ == '__main__':
    sys.exit(main())